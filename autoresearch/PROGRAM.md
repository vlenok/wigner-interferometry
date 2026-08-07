# Agent research program

You are optimizing the discrete cross-Wigner transform used by the Wigner interferometry notebooks. Work experimentally: form a hypothesis, implement it, measure it, and preserve enough evidence for another researcher to audit the result.

## Contract

- You may edit only `autoresearch/candidate.py`.
- Treat `sigproc.py`, `autoresearch/contract.json`, `_reference.py`, `_protocol.py`, `_worker.py`, `evaluate.py`, `pyproject.toml`, and `uv.lock` as immutable.
- Do not add dependencies, change the correctness tolerance, special-case public seeds or sizes, inspect another process, or bypass the evaluator.
- Preserve `dwv(signal1, signal2)` for even one-dimensional real or complex NumPy arrays of equal length.
- Correctness is a hard gate. The score exists only after every public output agrees with `sigproc.dwv` within the declared tolerance.
- The score is paired to the reference on the same machine. Higher is better: `time_speedup^0.75 * memory_improvement^0.25`.
- Public fixtures are an iteration signal, not proof. A merge candidate needs human review and held-out cases that were unavailable during search.

## Durable state

Create `autoresearch/scratchpad/` and maintain:

- `THREAD.md`: current best commit/hash, score distribution, active hypothesis, and next actions.
- `ideas.md`: source, claimed mechanism, predicted effect, and disposition for every idea.
- `failures.md`: correctness failures, crashes, and negative results so they are not repeated.

`results.jsonl` and `scratchpad/` are deliberately untracked. The evaluator records a candidate hash and git commit in every result; never transcribe measurements by hand when the structured record is available.

## Setup and baseline

1. Read `README.md`, `sigproc.py`, `autoresearch/README.md`, `contract.json`, and this file.
2. Create a fresh branch or worktree for this research run. Multiple agents must use separate worktrees and result logs.
3. Build the trusted image once, before modifying `candidate.py`:

   ```bash
   docker build -f autoresearch/Dockerfile -t wigner-autoresearch:local .
   ```

4. Warm the container/image cache once, then run the baseline at least three times:

   ```bash
   ./autoresearch/sandbox.sh --label warmup --size 2048 --repeats 5
   ./autoresearch/sandbox.sh --label baseline --size 2048 --repeats 5
   ```

   Do not include the warm-up result in the baseline distribution.

5. Record the median score and the observed score spread in `scratchpad/THREAD.md`. Initial warm local measurements put the identical-code score near 0.98-1.01, so require at least a 5% confirmed improvement unless a new machine-specific noise study justifies another threshold.

## Bounded experiment loop

Run at most 10 experiments or two hours before a human checkpoint.

1. Review the best candidate, negative results, and relevant literature. For multi-day work, refresh upstream sources periodically.
2. Write one falsifiable hypothesis in `ideas.md`. Connect it to a source or mark it explicitly as speculation.
3. Restore `candidate.py` from the best known commit before branching into a new idea. Change one conceptual factor at a time when practical.
4. Commit the candidate with an `experiment:` prefix before measuring it so the result has immutable lineage.
5. Run the sandboxed evaluator. A crash or correctness failure is a negative result, not an invitation to weaken the contract.
6. If the first result looks better, repeat it at least twice. Keep it only when the improvement exceeds the baseline noise floor and neither time nor memory regresses materially.
7. After stacking improvements, spend roughly one run in twenty on leave-one-out pruning. Prefer the simpler candidate when measurements are indistinguishable.
8. Update the scratchpad with the source, hypothesis, commit, candidate hash, all measurements, conclusion, and remaining uncertainty.

Never push, merge, claim novelty, or modify the paper without explicit human approval. Stop early if the evaluator, sandbox, or reference is suspect.
