"""Trusted public evaluator for discrete Wigner-transform candidates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from ._protocol import VALIDATION_CASES, benchmark_case


PROTOCOL_ID = "wigner-dwv-v1"
DEFAULT_SIZE = 2048
DEFAULT_REPEATS = 5
DEFAULT_TIMEOUT_SECONDS = 120.0
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10


def _candidate_path() -> Path:
    return Path(__file__).with_name("candidate.py")


def _candidate_sha256() -> str:
    return hashlib.sha256(_candidate_path().read_bytes()).hexdigest()


def _git_commit() -> str:
    injected = os.environ.get("AUTORESEARCH_GIT_COMMIT")
    if injected:
        return injected
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _failure(label: str, status: str, error: str, **details: object) -> dict[str, object]:
    return {
        "protocol": PROTOCOL_ID,
        "label": label,
        "status": status,
        "score": None,
        "candidate_sha256": _candidate_sha256(),
        "git_commit": _git_commit(),
        "error": error,
        **details,
    }


def _run_worker(
    implementation: str,
    output_dir: Path,
    size: int,
    repeats: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    command = [
        sys.executable,
        "-m",
        "autoresearch._worker",
        "--implementation",
        implementation,
        "--output-dir",
        str(output_dir),
        "--size",
        str(size),
        "--repeats",
        str(repeats),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise TimeoutError(f"{implementation} exceeded {timeout_seconds:.1f} seconds")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{implementation} emitted no result; stderr: {stderr[-2000:]}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{implementation} emitted invalid JSON: {lines[-1][-2000:]}") from exc
    if process.returncode != 0 or payload.get("status") != "ok":
        message = payload.get("error") or stderr[-2000:] or f"exit code {process.returncode}"
        raise RuntimeError(f"{implementation} failed: {message}")
    return payload


def _compare_outputs(candidate_dir: Path, reference_dir: Path, size: int) -> dict[str, float | str]:
    worst_case = ""
    max_absolute_error = 0.0
    max_relative_error = 0.0
    cases = (*VALIDATION_CASES, benchmark_case(size))

    for case in cases:
        candidate = np.load(candidate_dir / f"{case.name}.npy", allow_pickle=False)
        reference = np.load(reference_dir / f"{case.name}.npy", allow_pickle=False)
        if candidate.shape != reference.shape:
            raise ValueError(f"{case.name}: shape {candidate.shape} != {reference.shape}")
        difference = np.abs(candidate - reference)
        absolute_error = float(np.max(difference, initial=0.0))
        scale = max(float(np.max(np.abs(reference), initial=0.0)), ABSOLUTE_TOLERANCE)
        relative_error = absolute_error / scale
        if absolute_error >= max_absolute_error:
            worst_case = case.name
            max_absolute_error = absolute_error
        max_relative_error = max(max_relative_error, relative_error)
        if not np.allclose(
            candidate,
            reference,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"{case.name}: max_abs_error={absolute_error:.6g}, "
                f"max_rel_error={relative_error:.6g}"
            )

    return {
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "worst_case": worst_case,
    }


def evaluate(label: str, size: int, repeats: int, timeout_seconds: float) -> dict[str, object]:
    if size < 8 or size % 2:
        return _failure(label, "invalid", "size must be an even integer of at least 8")
    if repeats < 1:
        return _failure(label, "invalid", "repeats must be at least 1")

    with tempfile.TemporaryDirectory(prefix="wigner-autoresearch-") as temporary:
        root = Path(temporary)
        candidate_dir = root / "candidate"
        reference_dir = root / "reference"
        oracle_dir = root / "oracle"
        try:
            # Run untrusted code first, terminate its process group, then create the oracle outputs.
            candidate = _run_worker("candidate", candidate_dir, size, repeats, timeout_seconds)
            _run_worker("oracle", oracle_dir, size, 1, timeout_seconds)
            reference = _run_worker("reference", reference_dir, size, repeats, timeout_seconds)
            _compare_outputs(reference_dir, oracle_dir, size)
            correctness = _compare_outputs(candidate_dir, oracle_dir, size)
        except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
            return _failure(label, "failed", str(exc), size=size, repeats=repeats)

    candidate_seconds = float(candidate["median_seconds"])
    reference_seconds = float(reference["median_seconds"])
    candidate_rss = float(candidate["peak_rss_mb"])
    reference_rss = float(reference["peak_rss_mb"])
    time_speedup = reference_seconds / candidate_seconds
    memory_improvement = reference_rss / candidate_rss
    score = time_speedup**0.75 * memory_improvement**0.25

    return {
        "protocol": PROTOCOL_ID,
        "label": label,
        "status": "pass",
        "score": score,
        "candidate_sha256": _candidate_sha256(),
        "git_commit": _git_commit(),
        "size": size,
        "repeats": repeats,
        "candidate_median_seconds": candidate_seconds,
        "reference_median_seconds": reference_seconds,
        "time_speedup": time_speedup,
        "candidate_peak_rss_mb": candidate_rss,
        "reference_peak_rss_mb": reference_rss,
        "memory_improvement": memory_improvement,
        **correctness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    result = evaluate(args.label, args.size, args.repeats, args.timeout)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
