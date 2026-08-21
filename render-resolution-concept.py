"""Render an annotated resolution concept figure for the grant application.

The correlation curves reproduce the circular-source theory used by
``plot-frontfig.ipynb``. The image panels deliberately use one synthetic
black-hole model and differ only by the width of a Gaussian point-spread
function. Their relative PSF width is derived from the half-maximum locations
of the two plotted responses; it is not an absolute telescope forecast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage
from scipy.special import jv, struve


SPEED_OF_LIGHT = 3.0e8
SOURCE_RADIUS = np.radians(0.25)
W0 = 0.0
W = 2.0 * np.pi * 2.0e9
H = 2.0 * np.pi * 8.5e9

COLORS = {
    "background": "#f7f5f1",
    "ink": "#202124",
    "muted": "#6f7478",
    "grid": "#d8d4cd",
    "classical": "#8faac7",
    "wigner": "#b22222",
    "wigner_fill": "#d98b83",
}

TEXT = {
    "en": {
        "title": "Wigner correlation: a narrower response reveals finer structure",
        "x": "Interferometer baseline (m)",
        "y": "Normalized correlation",
        "classical": "Classical correlation",
        "wigner": "Wigner correlation",
        "width": "{factor}× narrower\nat half maximum",
        "images": "Same black-hole model  •  Gaussian blur only",
        "current": "Current resolution",
        "expected": "Expected resolution",
        "psf_current": "Classical • Gaussian PSF: θ",
        "psf_expected": "Wigner • Gaussian PSF: θ/{factor}",
        "note": (
            "Illustrative mapping: the relative factor comes from the half-maximum widths "
            "of the plotted responses; absolute angular scales require an instrument model."
        ),
    },
    "ru": {
        "title": "Корреляция Вигнера: более узкий отклик выявляет более мелкие детали",
        "x": "База интерферометра (м)",
        "y": "Нормированная корреляция",
        "classical": "Классическая корреляция",
        "wigner": "Корреляция Вигнера",
        "width": "отклик уже в {factor} раза\nна половине максимума",
        "images": "Одна модель чёрной дыры  •  различается только гауссово размытие",
        "current": "Текущее разрешение",
        "expected": "Ожидаемое разрешение",
        "psf_current": "Классика • Гауссова ФРТ: θ",
        "psf_expected": "Вигнер • Гауссова ФРТ: θ/{factor}",
        "note": (
            "Иллюстративное отображение: относительный коэффициент взят из ширины "
            "кривых на половине максимума; абсолютный угловой масштаб требует модели инструмента."
        ),
    },
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


def half_maximum_x(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.flatnonzero(y <= 0.5)[0])
    return float(np.interp(0.5, y[index - 1 : index + 1][::-1], x[index - 1 : index + 1][::-1]))


def black_hole_model(size: int = 720) -> tuple[np.ndarray, float]:
    axis = np.linspace(-1.3, 1.3, size)
    x, y = np.meshgrid(axis, axis)

    angle = np.radians(-16.0)
    rotated_x = np.cos(angle) * x - np.sin(angle) * y
    rotated_y = np.sin(angle) * x + np.cos(angle) * y
    elliptical_radius = np.sqrt(rotated_x**2 + (rotated_y / 0.78) ** 2)
    azimuth = np.arctan2(rotated_y / 0.78, rotated_x)

    ring = np.exp(-0.5 * ((elliptical_radius - 0.64) / 0.075) ** 2)
    doppler = 0.18 + 0.82 * (0.5 + 0.5 * np.cos(azimuth + 0.8)) ** 1.7
    outer_glow = 0.14 * np.exp(-0.5 * ((elliptical_radius - 0.70) / 0.22) ** 2)
    photon_arc = 0.35 * np.exp(-0.5 * ((elliptical_radius - 0.51) / 0.035) ** 2)
    photon_arc *= np.exp(-0.5 * ((azimuth + 0.65) / 1.0) ** 2)
    image = (ring * doppler + outer_glow * doppler + photon_arc) ** 0.72

    shadow = 1.0 - np.exp(-((elliptical_radius / 0.34) ** 8))
    image *= shadow
    image /= image.max()
    return image, float(axis[1] - axis[0])


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


def render(language: str, output_dir: Path) -> tuple[Path, Path]:
    labels = TEXT[language]
    baselines, classical, wigner = correlation_curves()
    wigner_half = half_maximum_x(baselines, wigner)
    classical_half = half_maximum_x(baselines, classical)
    factor = classical_half / wigner_half
    display_factor = f"{factor:.1f}".replace(".", ",") if language == "ru" else f"{factor:.1f}"

    ideal, pixel_scale = black_hole_model()
    current_sigma = 0.19
    expected_sigma = current_sigma / factor
    current = ndimage.gaussian_filter(ideal, current_sigma / pixel_scale, mode="constant")
    expected = ndimage.gaussian_filter(ideal, expected_sigma / pixel_scale, mode="constant")
    common_max = max(float(current.max()), float(expected.max()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(14.8, 6.2), facecolor=COLORS["background"])
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.36, 1.36, 1.0, 1.0),
        height_ratios=(0.13, 1.0),
        left=0.055,
        right=0.985,
        top=0.91,
        bottom=0.17,
        wspace=0.23,
        hspace=0.03,
    )

    figure.suptitle(
        labels["title"],
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
    )

    curve_axis = figure.add_subplot(grid[1, :2])
    _panel_label(curve_axis, "A")
    curve_axis.set_facecolor(COLORS["background"])
    curve_axis.axhline(0.0, color=COLORS["ink"], linewidth=0.75, alpha=0.55)
    curve_axis.axhline(0.5, color=COLORS["grid"], linewidth=0.8, linestyle=(0, (3, 3)))
    curve_axis.fill_between(
        baselines,
        wigner,
        0.0,
        where=wigner >= 0.0,
        color=COLORS["wigner_fill"],
        alpha=0.18,
        linewidth=0,
    )
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
    curve_axis.vlines(
        [wigner_half, classical_half],
        0.0,
        0.5,
        colors=[COLORS["wigner"], COLORS["classical"]],
        linestyles=(0, (3, 3)),
        linewidth=1.1,
    )
    curve_axis.annotate(
        "",
        xy=(classical_half, 0.57),
        xytext=(wigner_half, 0.57),
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["ink"],
            "linewidth": 1.1,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    curve_axis.text(
        (wigner_half + classical_half) / 2.0,
        0.64,
        labels["width"].format(factor=display_factor),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )
    curve_axis.text(
        21.0,
        float(np.interp(21.0, baselines, classical)) + 0.06,
        labels["classical"],
        color=COLORS["classical"],
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    curve_axis.text(
        8.6,
        float(np.interp(8.6, baselines, wigner)) - 0.12,
        labels["wigner"],
        color=COLORS["wigner"],
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    curve_axis.set_xlim(0.0, 30.0)
    curve_axis.set_ylim(-0.2, 1.07)
    curve_axis.set_xticks((0, 5, 10, 15, 20, 25, 30))
    curve_axis.set_yticks((-0.2, 0.0, 0.5, 1.0))
    curve_axis.set_xlabel(labels["x"], fontsize=11, fontweight="bold")
    curve_axis.set_ylabel(labels["y"], fontsize=11, fontweight="bold")
    curve_axis.spines[["top", "right"]].set_visible(False)
    curve_axis.spines[["left", "bottom"]].set_color(COLORS["muted"])
    curve_axis.tick_params(length=4, width=0.8)

    heading_axis = figure.add_subplot(grid[0, 2:])
    heading_axis.axis("off")
    heading_axis.text(
        0.5,
        0.5,
        labels["images"],
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    image_cmap = _image_colormap()
    image_extent = (-1.3, 1.3, -1.3, 1.3)
    for column, image, title, psf_label, sigma, panel in (
        (2, current, labels["current"], labels["psf_current"], current_sigma, "B"),
        (
            3,
            expected,
            labels["expected"],
            labels["psf_expected"].format(factor=display_factor),
            expected_sigma,
            "C",
        ),
    ):
        image_axis = figure.add_subplot(grid[1, column])
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
        image_axis.text(
            0.04,
            0.94,
            panel,
            transform=image_axis.transAxes,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )
        image_axis.set_title(title, fontsize=11, fontweight="bold", pad=9)
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
        image_axis.text(
            0.5,
            0.04,
            psf_label,
            transform=image_axis.transAxes,
            color="white",
            fontsize=8.5,
            ha="center",
            va="bottom",
        )
        image_axis.set_xlim(-1.3, 1.3)
        image_axis.set_ylim(-1.3, 1.3)
        image_axis.set_xticks(())
        image_axis.set_yticks(())
        for spine in image_axis.spines.values():
            spine.set_color("#3a3b3d")
            spine.set_linewidth(0.8)

    figure.text(
        0.055,
        0.055,
        labels["note"],
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=COLORS["muted"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"resolution-concept-{language}"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(png_path, dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(TEXT), default="en")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    png_path, svg_path = render(args.language, args.output_dir)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
