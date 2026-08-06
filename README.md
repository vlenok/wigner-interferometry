# Wigner interferometry
Proof-of-principle simulations for the "Wigner interferometry" paper.

![](figures/front-fig.svg)

___

> **Wigner interferometry**<br>
> Vladimir Lenok<br>
> <a href="https://arxiv.org/abs/xxxx" target="_blank">https://arxiv.org/abs/xxxx </a> <br>
>
>**Abstract:** TBD


___

## Project Structure
Jupiter notebooks for numerical simulations and computing corresponding analytical Wigner correlation functions:
- `compute-rect.ipynb` - for a rectangular source
- `compute-circ.ipynb` - for a circular source

Plotting scripts:
- `plot-paper.ipynb` - reproduces the plot on Figure 3 from the paper
- `plot-frontfig.ipynb` - reproduces the front image of this repository

Service script:
- `sigproc.py` - contains service functions for filters and computations of the Wigner transform

## Running sequence
To reproduce results of the paper:
- run entire notebooks `compute-rect.ipynb` and `compute-circ.ipynb` to compute the Wigner correlations numerically and analytically (the results will be stored in a `./data` directory)
- run `plot-paper.ipynb` to reproduce the main plot of the paper
- run `plot-frontfig.ipynb` to reproduce the front image of this repository

## Requirements
- `python >= 3.10.15`
- `numpy >= 1.26.4`
- `scipy >= 1.15.3`
- `matplotlib >= 3.10.0`
- `jupyter`

## Citation
For citations, please use the following.

```
@ARTICLE{xxx,
    author    = {Lenok, Vladimir},
    title     = {# Wigner interferometry},
}
```
