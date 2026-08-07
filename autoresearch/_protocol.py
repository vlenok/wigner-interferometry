"""Public correctness cases and deterministic benchmark inputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    size: int
    seed: int
    complex_inputs: bool
    same_inputs: bool


VALIDATION_CASES = (
    Case("real_self_8", 8, 104729, False, True),
    Case("real_cross_16", 16, 130363, False, False),
    Case("complex_cross_32", 32, 155921, True, False),
    Case("real_self_64", 64, 196613, False, True),
)


def benchmark_case(size: int) -> Case:
    if size < 8 or size % 2:
        raise ValueError("benchmark size must be an even integer of at least 8")
    return Case(f"benchmark_real_self_{size}", size, 262147, False, True)


def make_signals(case: Case) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    signal1 = rng.normal(size=case.size)
    if case.complex_inputs:
        signal1 = signal1 + 1j * rng.normal(size=case.size)

    if case.same_inputs:
        signal2 = signal1.copy()
    else:
        signal2 = rng.normal(size=case.size)
        if case.complex_inputs:
            signal2 = signal2 + 1j * rng.normal(size=case.size)

    return signal1, signal2


def expected_shape(size: int) -> tuple[int, int]:
    return size, 2 * (size - 1)
