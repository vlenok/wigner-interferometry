# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "bam @ git+https://github.com/danielpalumbo/BAM.git@389412b75d86e9649e9afb6a309fac1b93474001",
# ]
# ///
"""Generate a KerrBAM source image with separable photon subrings.

Run this optional source-model step from the repository root:

    uv run --script generate-kerrbam-model.py

The compressed archive preserves linear intensity for the direct n=0 image,
the n=1 photon ring, and their sum. These arrays can be used as inputs for
interferogram simulations; display transforms belong in plotting code.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import ehtim as eh
import numpy as np
from bam.inference.jfuncs import double_power_law_jfunc
from bam.inference.kerrbam import KerrBam

BAM_COMMIT = "389412b75d86e9649e9afb6a309fac1b93474001"


def save_npz_deterministic(output: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an ``np.load``-compatible archive without variable timestamps."""
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


def generate(
    output: Path,
    *,
    spin: float,
    inclination_deg: float,
    emission_radius_m: float,
    fov_uas: float,
    npix: int,
    adaptive_factor: int,
) -> None:
    """Generate and save the direct and n=1 subimages."""
    if not -1.0 < spin < 1.0:
        raise ValueError("spin must lie strictly between -1 and 1")
    if not 0.0 <= inclination_deg <= 90.0:
        raise ValueError("inclination must lie between 0 and 90 degrees")

    mass_to_distance_uas = 3.8
    model = KerrBam(
        fov_uas * eh.RADPERUAS,
        npix,
        double_power_law_jfunc,
        ["radius", "inner_power", "outer_power"],
        [emission_radius_m, 5.0, 5.0],
        mass_to_distance_uas,
        spin,
        np.radians(inclination_deg),
        1.0,
        PA=0.0,
        chi=-np.pi / 2.0,
        eta=None,
        nmax=1,
        beta=0.5,
        iota=np.pi / 3.0,
        spec=1.0,
        alpha_zeta=1.0,
        polflux=True,
        adap_fac=adaptive_factor,
        compute_P=False,
        interp_order=1,
    )
    model.make_rotated_image(n="all")

    side = npix * adaptive_factor
    direct = model.ivecs[0].reshape(side, side)
    photon_ring = model.ivecs[1].reshape(side, side)
    direct = np.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0)
    photon_ring = np.nan_to_num(
        photon_ring,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    direct[direct < 0.0] = 0.0
    photon_ring[photon_ring < 0.0] = 0.0
    total = direct + photon_ring
    scale = float(total.max())
    if scale <= 0.0:
        raise RuntimeError("KerrBAM returned an empty image")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(
        output,
        {
            "total": (total / scale).astype(np.float32),
            "direct_n0": (direct / scale).astype(np.float32),
            "photon_ring_n1": (photon_ring / scale).astype(np.float32),
            "spin": np.array(spin, dtype=np.float64),
            "inclination_deg": np.array(inclination_deg, dtype=np.float64),
            "emission_radius_m": np.array(emission_radius_m, dtype=np.float64),
            "fov_uas": np.array(fov_uas, dtype=np.float64),
            "mass_to_distance_uas": np.array(
                mass_to_distance_uas,
                dtype=np.float64,
            ),
            "npix": np.array(npix, dtype=np.int64),
            "adaptive_factor": np.array(adaptive_factor, dtype=np.int64),
            "bam_commit": np.array(BAM_COMMIT),
        },
    )
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/kerrbam-a-plus0.8-inc54.npz"),
    )
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--inclination-deg", type=float, default=54.0)
    parser.add_argument("--emission-radius-m", type=float, default=4.5)
    parser.add_argument("--fov-uas", type=float, default=80.0)
    parser.add_argument("--npix", type=int, default=128)
    parser.add_argument("--adaptive-factor", type=int, default=4)
    args = parser.parse_args()
    generate(
        args.output,
        spin=args.spin,
        inclination_deg=args.inclination_deg,
        emission_radius_m=args.emission_radius_m,
        fov_uas=args.fov_uas,
        npix=args.npix,
        adaptive_factor=args.adaptive_factor,
    )


if __name__ == "__main__":
    main()
