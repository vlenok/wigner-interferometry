"""Render the resolution concept figure for the paper.

The correlation curves reproduce the circular-source theory used by
``plot-frontfig.ipynb``. The image panels use one SHADOW-TD ray-traced,
optically thin thick-disk model containing direct emission, a lensing ring,
and a narrow photon ring. They differ only by the width of a Gaussian
point-spread function. The relative PSF width represents the conservative
twofold theoretical resolution improvement; it is not an absolute telescope
forecast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy import ndimage
from scipy.special import jv, struve

SPEED_OF_LIGHT = 3.0e8
SOURCE_RADIUS = np.radians(0.25)
W0 = 0.0
W = 2.0 * np.pi * 2.0e9
H = 2.0 * np.pi * 8.5e9

COLORS = {
    "background": "#ffffff",
    "ink": "#202124",
    "muted": "#6f7478",
    "classical": "#8faac7",
    "wigner": "#b22222",
    "wigner_fill": "#d98b83",
}

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "shadow-td-fig11-r2c3.npz"
)

TEXT = {
    "x": "Interferometer baseline (m)",
    "y": "Normalized correlation",
    "classical": "Classical",
    "wigner": "Wigner",
}


def _m_n(frequency: float, baselines: np.ndarray) -> np.ndarray:
    normalization = (
        4.0 * np.pi * SPEED_OF_LIGHT**2 / (frequency * baselines**2)
    )
    z = 2.0 * frequency * baselines * SOURCE_RADIUS / SPEED_OF_LIGHT
    auxiliary = z * jv(0, z) + 0.5 * np.pi * z * (
        jv(1, z) * struve(0, z) - jv(0, z) * struve(1, z)
    )
    return normalization * (-0.25 * z * jv(1, z) + 0.25 * z * auxiliary)


def _m(
    a: float,
    b: float,
    c: float,
    d: float,
    baselines: np.ndarray,
) -> np.ndarray:
    frequencies = [
        abs(W - a) + sign_d * d + sign_c * c + sign_b * b
        for sign_d in (1.0, -1.0)
        for sign_c in (1.0, -1.0)
        for sign_b in (1.0, -1.0)
    ]
    frequencies += [
        abs(W + a) + sign_d * d + sign_c * c + sign_b * b
        for sign_d in (1.0, -1.0)
        for sign_c in (1.0, -1.0)
        for sign_b in (1.0, -1.0)
    ]
    return sum(-0.25 * np.pi * _m_n(item, baselines) / 8.0 for item in frequencies)


def _self_interference(baselines: np.ndarray) -> np.ndarray:
    a = _m(0.0, H, W, W0, baselines)
    b1 = _m(H, 0.0, W, W0, baselines)
    b3 = _m(W0, 0.0, H, W, baselines)
    c1 = _m(W - H, 0.0, 0.0, W0, baselines)
    c2 = _m(W + H, 0.0, 0.0, W0, baselines)
    c3 = _m(W0 - H, 0.0, 0.0, W, baselines)
    c4 = _m(W0 + H, 0.0, 0.0, W, baselines)
    c5 = _m(W0 - W, 0.0, 0.0, H, baselines)
    c6 = _m(W0 + W, 0.0, 0.0, H, baselines)
    d1 = _m(W0 - W - H, 0.0, 0.0, 0.0, baselines)
    d2 = _m(W0 - W + H, 0.0, 0.0, 0.0, baselines)
    d3 = _m(W0 + W - H, 0.0, 0.0, 0.0, baselines)
    d4 = _m(W0 + W + H, 0.0, 0.0, 0.0, baselines)
    return (
        a
        - b1
        + 2.0 * b3
        - 0.5 * (c1 + c2 + c3 + c4 - c5 - c6)
        - 0.25 * (d1 + d2 + d3 + d4)
    )


def _f(a: float) -> float:
    return -0.5 * np.pi * (abs(W - a) + abs(W + a))


def _cross_interference(baselines: np.ndarray) -> np.ndarray:
    frequency_factor = (
        _f(0.0)
        - _f(H)
        + 2.0 * _f(W0)
        - 0.5
        * (
            _f(W - H)
            + _f(W + H)
            + _f(W0 - H)
            + _f(W0 + H)
            - _f(W0 - W)
            - _f(W0 + W)
        )
        - 0.25
        * (
            _f(W0 - W - H)
            + _f(W0 - W + H)
            + _f(W0 + W - H)
            + _f(W0 + W + H)
        )
    )
    v = 2.0 * (W / SPEED_OF_LIGHT) * SOURCE_RADIUS * baselines
    source_transform = np.pi * SOURCE_RADIUS**2 * 2.0 * jv(1, v) / v
    return frequency_factor * source_transform


def correlation_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baselines = np.linspace(0.001, 30.0, 300)
    wavelength = SPEED_OF_LIGHT / 2.0e9
    classical_argument = 2.0 * np.pi * SOURCE_RADIUS * baselines / wavelength
    classical = 2.0 * jv(1, classical_argument) / classical_argument

    self_interference = _self_interference(baselines)
    cross_interference = _cross_interference(baselines)
    wigner = self_interference + cross_interference
    wigner /= wigner[0]
    return baselines, classical, wigner


def black_hole_model(model_path: Path = MODEL_PATH) -> tuple[np.ndarray, float]:
    """Load linear intensity from the committed SHADOW-TD source model."""
    with np.load(model_path, allow_pickle=False) as archive:
        image = np.asarray(archive["intensity"], dtype=np.float64)

    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"expected a square 2D model image, got {image.shape}")
    if not np.isfinite(image).all() or float(image.max()) <= 0.0:
        raise ValueError("SHADOW-TD model contains invalid or empty intensity")

    image /= image.max()
    pixel_scale = 2.6 / image.shape[0]
    return image, pixel_scale


def _image_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "black-hole",
        ["#020203", "#21050b", "#6e1214", "#d05120", "#ffb547", "#fff0bd"],
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )


def render(output_dir: Path) -> tuple[Path, Path]:
    labels = TEXT
    baselines, classical, wigner = correlation_curves()
    factor = 2.0
    display_factor = f"{factor:g}"

    ideal, pixel_scale = black_hole_model()
    current_sigma = 0.19
    expected_sigma = current_sigma / factor
    current = ndimage.gaussian_filter(ideal, current_sigma / pixel_scale, mode="constant")
    expected = ndimage.gaussian_filter(ideal, expected_sigma / pixel_scale, mode="constant")
    # Apply the PSF to linear intensity. Both panels then use the same linear
    # display mapping so faint outer-disk emission cannot mask the rings.
    current_display = current
    expected_display = expected
    common_max = max(float(current_display.max()), float(expected_display.max()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "svg.fonttype": "none",
            "svg.hashsalt": "wigner-interferometry",
        }
    )
    figure = plt.figure(figsize=(13.5, 4.8), facecolor=COLORS["background"])
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.36, 1.36, 1.0, 1.0),
        left=0.06,
        right=0.99,
        top=0.89,
        bottom=0.16,
        wspace=0.25,
    )

    curve_axis = figure.add_subplot(grid[0, :2])
    _panel_label(curve_axis, "A")
    curve_axis.set_facecolor(COLORS["background"])
    curve_axis.axhline(0.0, color=COLORS["ink"], linewidth=0.75, alpha=0.55)
    curve_axis.plot(
        baselines,
        classical,
        color=COLORS["classical"],
        linewidth=3.0,
        label=labels["classical"],
    )
    curve_axis.plot(
        baselines,
        wigner,
        color=COLORS["wigner"],
        linewidth=3.0,
        label=labels["wigner"],
    )
    curve_axis.set_xlim(0.0, 30.0)
    curve_axis.set_ylim(-0.2, 1.07)
    curve_axis.set_xticks((0, 5, 10, 15, 20, 25, 30))
    curve_axis.set_yticks((-0.2, 0.0, 0.5, 1.0))
    curve_axis.set_xlabel(labels["x"], fontsize=10)
    curve_axis.set_ylabel(labels["y"], fontsize=10)
    curve_axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    curve_axis.spines[["top", "right"]].set_visible(False)
    curve_axis.spines[["left", "bottom"]].set_color(COLORS["muted"])
    curve_axis.tick_params(length=4, width=0.8)

    image_cmap = _image_colormap()
    image_extent = (-1.3, 1.3, -1.3, 1.3)
    for column, image, title, sigma, panel in (
        (2, current_display, "Classical (θ)", current_sigma, "B"),
        (
            3,
            expected_display,
            f"Wigner (θ/{display_factor})",
            expected_sigma,
            "C",
        ),
    ):
        image_axis = figure.add_subplot(grid[0, column])
        _panel_label(image_axis, panel)
        image_axis.set_facecolor("#020203")
        image_axis.imshow(
            image,
            origin="lower",
            extent=image_extent,
            cmap=image_cmap,
            vmin=0.0,
            vmax=common_max,
            interpolation="bicubic",
        )
        image_axis.set_title(title, fontsize=10, pad=7)
        fwhm = 2.355 * sigma
        image_axis.add_patch(
            Circle(
                (-0.91, -0.91),
                radius=fwhm / 2.0,
                facecolor="none",
                edgecolor="white",
                linewidth=1.2,
                alpha=0.95,
            )
        )
        image_axis.set_xlim(-1.3, 1.3)
        image_axis.set_ylim(-1.3, 1.3)
        image_axis.set_xticks(())
        image_axis.set_yticks(())
        for spine in image_axis.spines.values():
            spine.set_color("#3a3b3d")
            spine.set_linewidth(0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "resolution-concept"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(png_path, dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(
        svg_path,
        facecolor=figure.get_facecolor(),
        metadata={"Date": None},
    )
    plt.close(figure)
    # Matplotlib writes spaces before newlines inside SVG path data. Normalize
    # the generated text so repository whitespace checks stay useful.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    png_path, svg_path = render(args.output_dir)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
