# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "matplotlib>=3.10,<3.11",
#   "numba>=0.61,<0.62",
#   "numpy>=2.2,<2.4",
#   "pandas>=2.3,<2.4",
#   "scipy>=1.16,<1.17",
#   "tqdm>=4.67,<5",
# ]
# ///
"""Generate the ray-traced source model used by the resolution illustration.

By default this downloads a pinned SHADOW-TD revision, runs its geodesic and
flux calculations with the parameters of Figure 11, row 2, column 3 in Wang
(2025), and rasterizes the resulting linear intensity:

    uv run --script generate-shadow-td-model.py

The upstream calculation takes several minutes and creates a large temporary
working directory. To rasterize an already-computed upstream flux archive:

    uv run --script generate-shadow-td-model.py --flux-npz path/to/flux.npz
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SHADOW_TD_COMMIT = "d77a2bd51514ec87d9cbbc295761c9d25cb36c82"
SHADOW_TD_ARCHIVE = (
    "https://github.com/ziliang-wang0/SHADOW-TD/archive/"
    f"{SHADOW_TD_COMMIT}.zip"
)

FIGURE_PARAMETERS: dict[str, float | str] = {
    "kappa_ff": 0.5,
    "kappa_K": 0.1,
    "r_in": 6.0,
    "psi0_deg": 30.0,
    "theta0_deg": 50.0,
    "optical_regime": "thin",
    "shadow_xmax": 17.0,
    "shadow_ymax": 17.0,
}


def save_npz_deterministic(output: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an ``np.load``-compatible archive without variable timestamps."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, buffer.getvalue())


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> Path:
    """Extract a GitHub source archive after rejecting path traversal."""
    destination_resolved = destination.resolve()
    members = archive.infolist()
    for member in members:
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination_resolved):
            raise ValueError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination, members=members)

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one source directory, found {len(roots)}")
    return roots[0]


def _download_source(destination: Path) -> Path:
    archive_path = destination / "shadow-td.zip"
    print(f"Downloading SHADOW-TD revision {SHADOW_TD_COMMIT}...")
    urllib.request.urlretrieve(SHADOW_TD_ARCHIVE, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        return _safe_extract(archive, destination / "source")


def _run_shadow_td(source: Path) -> Path:
    config_path = source / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(FIGURE_PARAMETERS)
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    for script in (
        source / "scripts" / "step1_all_geodesic.py",
        source / "scripts" / "step2_theta0_psi0_v2.py",
    ):
        subprocess.run([sys.executable, str(script)], cwd=source, check=True)

    output = source / "output" / (
        "flux_rmax=50.0_optical_thin_psi0=30.0_rin=6.0_"
        "theta0=50.0_kappaff=0.500_kappaK=0.100.npz"
    )
    if not output.is_file():
        raise RuntimeError(f"SHADOW-TD did not create the expected output: {output}")
    return output


def rasterize_flux(
    flux_path: Path,
    *,
    grid_size: int,
    extent_m: float,
    max_distance_m: float,
) -> np.ndarray:
    """Map the upstream polar samples to a regular Cartesian intensity grid."""
    if grid_size < 2:
        raise ValueError("grid size must be at least 2")
    if extent_m <= 0.0 or max_distance_m <= 0.0:
        raise ValueError("extent and maximum interpolation distance must be positive")

    with np.load(flux_path, allow_pickle=False) as archive:
        b = np.asarray(archive["b"], dtype=np.float64)
        alpha = np.asarray(archive["alpha"], dtype=np.float64)
        flux = np.asarray(archive["F"], dtype=np.float64)

    x = b * np.cos(alpha)
    y = b * np.sin(alpha)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(flux)
        & (x >= -extent_m)
        & (x <= extent_m)
        & (y >= -extent_m)
        & (y <= extent_m)
    )
    if not np.any(valid):
        raise ValueError("the SHADOW-TD flux archive contains no valid samples")

    coordinates = np.linspace(-extent_m, extent_m, grid_size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    source_points = np.column_stack((x[valid], y[valid]))

    distances, indices = cKDTree(source_points).query(
        query_points,
        distance_upper_bound=max_distance_m,
        workers=-1,
    )
    raster = np.zeros(query_points.shape[0], dtype=np.float64)
    matched = np.isfinite(distances)
    raster[matched] = flux[valid][indices[matched]]
    raster = raster.reshape(grid_size, grid_size)
    raster = np.clip(raster, 0.0, None)
    peak = float(raster.max())
    if peak <= 0.0:
        raise ValueError("rasterized SHADOW-TD model has zero intensity")
    return (raster / peak).astype(np.float32)


def generate(
    output: Path,
    *,
    flux_npz: Path | None,
    grid_size: int,
    extent_m: float,
    max_distance_m: float,
) -> None:
    """Generate the compact, deterministic source archive."""
    if flux_npz is None:
        with tempfile.TemporaryDirectory(prefix="shadow-td-") as temporary:
            source = _download_source(Path(temporary))
            flux_path = _run_shadow_td(source)
            image = rasterize_flux(
                flux_path,
                grid_size=grid_size,
                extent_m=extent_m,
                max_distance_m=max_distance_m,
            )
    else:
        image = rasterize_flux(
            flux_npz,
            grid_size=grid_size,
            extent_m=extent_m,
            max_distance_m=max_distance_m,
        )

    save_npz_deterministic(
        output,
        {
            "intensity": image,
            "extent_m": np.array(extent_m, dtype=np.float64),
            "r_in_m": np.array(FIGURE_PARAMETERS["r_in"], dtype=np.float64),
            "psi0_deg": np.array(
                FIGURE_PARAMETERS["psi0_deg"],
                dtype=np.float64,
            ),
            "theta0_deg": np.array(
                FIGURE_PARAMETERS["theta0_deg"],
                dtype=np.float64,
            ),
            "kappa_ff": np.array(
                FIGURE_PARAMETERS["kappa_ff"],
                dtype=np.float64,
            ),
            "kappa_K": np.array(
                FIGURE_PARAMETERS["kappa_K"],
                dtype=np.float64,
            ),
            "optical_regime": np.array(FIGURE_PARAMETERS["optical_regime"]),
            "shadow_td_commit": np.array(SHADOW_TD_COMMIT),
        },
    )
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/shadow-td-fig11-r2c3.npz"),
    )
    parser.add_argument(
        "--flux-npz",
        type=Path,
        help="Rasterize an existing SHADOW-TD flux archive instead of rerunning it",
    )
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--extent-m", type=float, default=17.0)
    parser.add_argument("--max-distance-m", type=float, default=0.05)
    args = parser.parse_args()
    generate(
        args.output,
        flux_npz=args.flux_npz,
        grid_size=args.grid_size,
        extent_m=args.extent_m,
        max_distance_m=args.max_distance_m,
    )


if __name__ == "__main__":
    main()
