"""Agent-editable discrete Wigner transform candidate."""

import numpy as np


def dwv(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """Return the cross discrete Wigner transform used by the notebooks."""
    size = signal1.shape[0]
    width = 2 * (size - 1)
    midpoint = size // 2

    doubled_signal1 = np.repeat(signal1, 2)
    doubled_signal2 = np.repeat(np.conjugate(signal2), 2)
    plane = np.zeros(
        (size, width),
        dtype=np.result_type(doubled_signal1, doubled_signal2),
    )

    # The reference pads each row symmetrically and rolls it by half a row.
    # After that roll, every nonzero row is split evenly across the two edges.
    for row in range(1, midpoint):
        half_width = 2 * row
        source_stop = 2 * half_width
        np.multiply(
            doubled_signal1[:half_width],
            doubled_signal2[source_stop:half_width:-1],
            out=plane[row, -half_width:],
        )
        np.multiply(
            doubled_signal1[half_width:source_stop],
            doubled_signal2[half_width:0:-1],
            out=plane[row, :half_width],
        )

    for offset in range(midpoint - 1):
        row = midpoint + offset
        half_width = size - 3 - 2 * offset
        source_start = 3 + 4 * offset
        reversed_signal2 = doubled_signal2[width - 1 : source_start : -1]
        np.multiply(
            doubled_signal1[source_start : source_start + half_width],
            reversed_signal2[:half_width],
            out=plane[row, -half_width:],
        )
        np.multiply(
            doubled_signal1[source_start + half_width : width - 1],
            reversed_signal2[half_width:],
            out=plane[row, :half_width],
        )

    return np.fft.fft(plane, axis=1)
