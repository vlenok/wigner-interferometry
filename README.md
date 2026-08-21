# Wigner interferometry

Proof-of-principle simulations for the "Wigner interferometry" paper.

![Wigner correlation for a circular source](figures/front-fig.svg)

---

> **Wigner interferometry**
>
> Vladimir Lenok
>
> <https://arxiv.org/abs/xxxx>
>
> **Abstract:** TBD

---

## Repository contents

The notebooks reproduce the numerical simulations, analytical correlation functions, and figures used by the paper:

- `compute-rect.ipynb` computes the results for a rectangular source.
- `compute-circ.ipynb` computes the results for a circular source.
- `plot-paper.ipynb` generates Figure 3 of the paper.
- `plot-frontfig.ipynb` generates the image shown at the top of this README.
- `sigproc.py` provides the filter and Wigner-transform functions used by the compute notebooks.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A working LaTeX and `dvipng` toolchain containing `amsmath`, `amssymb`, and the Latin Modern fonts (required only by `plot-paper.ipynb`)

The repository pins Python 3.10.15 in `.python-version`. If it is not already installed, `uv` downloads it automatically.

## Setup

Clone the repository, enter its directory, and create the locked environment:

```bash
git clone https://github.com/vlenok/wigner-interferometry.git
cd wigner-interferometry
uv sync --locked
```

Then start JupyterLab in that environment:

```bash
uv run jupyter lab
```

## Reproducing the results

Run every cell in the notebooks in this order:

1. `compute-rect.ipynb`
2. `compute-circ.ipynb`
3. `plot-paper.ipynb`
4. `plot-frontfig.ipynb`

The first two notebooks are the computationally intensive part of the workflow. They use fixed random seeds and write their numerical results to `data/`. The plotting notebooks read those files and create:

- `figures/comb-fig.pdf` from `plot-paper.ipynb`
- `figures/front-fig.svg` and `figures/front-fig.pdf` from `plot-frontfig.ipynb`

All notebooks assume that Jupyter was started from the repository root. Running a plotting notebook before both compute notebooks finish will fail because the required files in `data/` do not exist yet.

For reference, a verification run on Linux with an AMD EPYC 7402P took 16 minutes 31 seconds for `compute-rect.ipynb` and 11 minutes 49 seconds for `compute-circ.ipynb`. Each notebook used one CPU core and peaked at approximately 18 GB of RAM. Actual resource use and runtime will vary by machine.

### Non-interactive execution

The entire workflow can also be run sequentially from the terminal. Executed copies of the notebooks are written to `executed-notebooks/`, leaving the source notebooks unchanged:

```bash
mkdir -p executed-notebooks
(
    set -e
    for notebook in \
        compute-rect.ipynb \
        compute-circ.ipynb \
        plot-paper.ipynb \
        plot-frontfig.ipynb
    do
        uv run jupyter nbconvert \
            --to notebook \
            --execute \
            --ExecutePreprocessor.timeout=-1 \
            --output-dir executed-notebooks \
            "$notebook"
    done
)
```

## Alternative setup without `uv`

The same project can be installed with standard Python tooling:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
jupyter lab
```

This installs compatible dependency versions, but does not use the exact versions recorded in `uv.lock`.

## Resolution concept illustration

Render the English grant illustration from the committed linear-intensity
source model:

```bash
uv run python render-resolution-concept.py
```

This writes `figures/resolution-concept-en.png` and
`figures/resolution-concept-en.svg`. Panels B and C use exactly the same source
image and differ only in the width of the Gaussian point-spread function.

The source image was ray-traced with
[`SHADOW-TD`](https://github.com/ziliang-wang0/SHADOW-TD) at a pinned upstream
revision. It reproduces the optically thin model in Figure 11, row 2, column 3
of Wang (2025): inner disk radius `r_in = 6 M`, disk half-opening angle
`psi_0 = 30 degrees`, observer inclination `theta_0 = 50 degrees`,
`kappa_ff = 0.5`, and `kappa_K = 0.1`. The broad bright structure is the
lensing ring; the photon ring is its narrow inner boundary.

Regenerate the committed linear-intensity archive with:

```bash
uv run --script generate-shadow-td-model.py
```

This upstream calculation takes several minutes; the rendering command itself
uses the compact committed archive. The model follows
[Wang (2025)](https://doi.org/10.1103/fhqj-wgcm). SHADOW-TD uses a Schwarzschild
black hole and a parameterized thick accretion disk; it is a physical
ray-tracing model, but not a Kerr or GRMHD simulation. The Gaussian width ratio
in the illustration comes from the half-maximum widths of the plotted
correlation responses; it is not an absolute telescope forecast.

## Citation

For citations, please use the following:

```bibtex
@ARTICLE{xxx,
    author = {Lenok, Vladimir},
    title  = {Wigner interferometry},
}
```
