# Direct Wigner-plane and real-FFT experiment

This report records the first bounded run of the autoresearch environment on
2026-08-07. The result is an experimental candidate; `sigproc.dwv` and the paper
notebooks remain unchanged.

## Result

The retained candidate combines two changes:

1. It writes the centered autocorrelation rows directly into a single
   zero-initialized Wigner plane. This removes the reference implementation's
   lists of padded and rolled row allocations.
2. For the paper's `float64` inputs, it computes the nonredundant real FFT into
   the final output array and reconstructs the negative-frequency half using
   Hermitian symmetry. Other real and complex dtypes retain the full FFT path.

All measurements used one CPU in the repository's no-network, read-only Docker
sandbox. Times are medians within each evaluator invocation and are specific to
this machine.

| Candidate | Size | Candidate time | Reference time | Candidate RSS | Reference RSS | Composite score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct plane, median of 3 runs | 2048 | 0.221 s | 0.370 s | 360 MB | 451 MB | 1.56 |
| Retained candidate, run 1 | 2048 | 0.150 s | 0.358 s | 232 MB | 451 MB | 2.27 |
| Retained candidate, run 2 | 2048 | 0.138 s | 0.358 s | 232 MB | 450 MB | 2.42 |
| Retained candidate, promotion run | 4096 | 0.323 s | 0.805 s | 832 MB | 1700 MB | 2.37 |

At size 4096, the retained candidate was 2.50 times faster and used 51% of the
reference peak memory. Its candidate hash was
`f09a078d281fccc619572bfd27af12293b28bf4c997c2dbe154192bcf716a0d3` at commit
`f253503`.

## Correctness evidence

- Every sandbox run passed the frozen public oracle at `rtol=1e-10` and
  `atol=1e-10`.
- The retained 2048-point runs had maximum absolute error `1.71e-13`.
- The 4096-point promotion run had maximum absolute error `1.92e-13`.
- An additional 64 unscored comparisons covered self and cross inputs, sizes
  8 through 512, and `float32`, `float64`, `complex64`, and `complex128`. Their
  worst absolute error was `5.72e-14`.
- The first unrestricted real-FFT attempt failed the strict contract for a
  size-10 `float32` cross case (`2.98e-08` absolute error). Restricting the
  specialization to `float64` fixed it; other dtypes use the full FFT.

## Reproduce

```bash
docker build -f autoresearch/Dockerfile -t wigner-autoresearch:local .
./autoresearch/sandbox.sh \
    --label reproduce-optimized \
    --size 2048 \
    --repeats 5
```

Before promoting this implementation into `sigproc.py`, independently review
the index derivation, run externally held-out cases, and execute the full notebook
workflow. These measurements establish an implementation improvement, not a new
Wigner-transform algorithm or a scientific novelty claim.
