"""Isolated worker used by the trusted evaluator.

The worker writes NumPy outputs into a temporary directory and emits one JSON
record. Candidate stdout and stderr are captured so the protocol remains
machine-readable.
"""

import argparse
import contextlib
import gc
import importlib
import io
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback
from collections.abc import Callable

import numpy as np

from ._protocol import VALIDATION_CASES, benchmark_case, expected_shape, make_signals


Transform = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def _load_transform(implementation: str) -> Transform:
    module_names = {
        "candidate": "autoresearch.candidate",
        "reference": "autoresearch._reference",
        "oracle": "sigproc",
    }
    module_name = module_names[implementation]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        module = importlib.import_module(module_name)
    transform = getattr(module, "dwv", None)
    if not callable(transform):
        raise TypeError(f"{module_name}.dwv must be callable")
    return transform


def _run_transform(transform: Transform, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = np.asarray(transform(signal1.copy(), signal2.copy()))
    return result


def _validate_result(result: np.ndarray, size: int) -> None:
    if result.shape != expected_shape(size):
        raise ValueError(f"expected shape {expected_shape(size)}, got {result.shape}")
    if result.dtype == object or not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"expected a numeric array, got {result.dtype}")
    if not np.isfinite(result).all():
        raise ValueError("candidate returned non-finite values")


def run(implementation: str, output_dir: Path, size: int, repeats: int) -> dict[str, object]:
    transform = _load_transform(implementation)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in VALIDATION_CASES:
        signal1, signal2 = make_signals(case)
        result = _run_transform(transform, signal1, signal2)
        _validate_result(result, case.size)
        np.save(output_dir / f"{case.name}.npy", result, allow_pickle=False)

    case = benchmark_case(size)
    signal1, signal2 = make_signals(case)
    timings: list[float] = []
    for repeat in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = _run_transform(transform, signal1, signal2)
        timings.append(time.perf_counter() - started)
        _validate_result(result, size)
        if repeat == 0:
            np.save(output_dir / f"{case.name}.npy", result, allow_pickle=False)
        del result

    return {
        "implementation": implementation,
        "median_seconds": statistics.median(timings),
        "peak_rss_mb": _peak_rss_mb(),
        "repeats": repeats,
        "timings_seconds": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=("candidate", "reference", "oracle"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()

    try:
        payload = {"status": "ok", **run(args.implementation, args.output_dir, args.size, args.repeats)}
    except Exception as exc:  # The parent turns this structured failure into a failed experiment.
        payload = {
            "status": "error",
            "implementation": args.implementation,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
        print(json.dumps(payload, sort_keys=True))
        return 1

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
