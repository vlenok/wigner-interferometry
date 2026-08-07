"""Frozen NumPy-only performance baseline equivalent to ``sigproc.dwv``."""

import numpy as np


def dwv(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    n = signal1.shape[0] - 1
    signal2_conj = np.conjugate(signal2)
    doubled_signal1 = np.stack((signal1, signal1)).T.flatten()
    doubled_signal2 = np.stack((signal2_conj, signal2_conj)).T.flatten()
    midpoint = signal1.shape[0] // 2

    first_half = [
        doubled_signal1[0 : index * 4 + 1]
        * np.flip(doubled_signal2[0 : index * 4 + 1])
        for index in range(midpoint)
    ]
    first_half = [item[:-1] for item in first_half]
    first_half = [
        np.pad(
            item,
            int(0.5 * (2 * n - item.shape[0])),
            "constant",
            constant_values=0.0,
        )
        for item in first_half
    ]
    first_half = [np.roll(item, int(0.5 * item.shape[0])) for item in first_half]

    second_half = [
        doubled_signal1[3 + index * 4 : (midpoint - 1) * 4 + 2]
        * np.flip(doubled_signal2[3 + index * 4 : (midpoint - 1) * 4 + 2])
        for index in range(midpoint)
    ]
    second_half = [item[:-1] for item in second_half]
    second_half = [
        np.pad(
            item,
            int(0.5 * (2 * n - item.shape[0])),
            "constant",
            constant_values=0.0,
        )
        for item in second_half
    ]
    second_half = [np.roll(item, int(0.5 * item.shape[0])) for item in second_half]

    wigner_plane = np.array(first_half + second_half)
    return np.fft.fft(wigner_plane, axis=1)
