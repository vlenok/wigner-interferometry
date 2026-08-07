# Experimental autoresearch environment

This directory is a model-agnostic environment for bounded algorithm research. Its first target is `sigproc.dwv`, because the paper's full notebooks showed a concrete bottleneck: each 4096-sample computation peaked near 18 GB of RAM in the reproducibility run.

The environment is intentionally narrower than an autonomous paper-writing system. An agent edits one candidate implementation; a fixed evaluator checks it against the repository implementation and measures it against a paired reference run. Humans retain control over the research question, hidden validation, scientific interpretation, and merging.

## Architecture

- `candidate.py`: the only agent-editable implementation.
- `contract.json`: machine-readable editable surface, correctness contract, and score.
- `_reference.py`: frozen NumPy-only performance baseline, checked against `sigproc.dwv` on every run.
- `_protocol.py`: deterministic public real, cross, and complex input cases.
- `_worker.py`: subprocess protocol for candidate and reference execution.
- `evaluate.py`: trusted correctness gate and paired scorer.
- `PROGRAM.md`: operating instructions for a coding agent.
- `Dockerfile` and `sandbox.sh`: no-network, read-only, CPU/memory/PID-limited execution of candidate code.
- `results.jsonl` and `scratchpad/`: untracked measurements and durable research state.

The score follows the paired-measurement idea used by MLXFast:

```text
time_speedup       = reference_seconds / candidate_seconds
memory_improvement = reference_peak_rss / candidate_peak_rss
score              = time_speedup^0.75 * memory_improvement^0.25
```

Higher is better, but no score is emitted unless the candidate and the isolated performance baseline both agree with the canonical `sigproc.dwv` oracle on every public case. Raw scores are machine-specific and should never be compared across different hardware.

## Trusted local smoke test

Running generated code directly on the host is appropriate only while the candidate is still trusted:

```bash
uv sync --locked
uv run python -m unittest autoresearch.test_evaluate
uv run python -m autoresearch.evaluate \
    --label baseline-local \
    --size 2048 \
    --repeats 5
```

## Sandboxed agent run

Build the evaluator image before giving an agent write access to `candidate.py`:

```bash
docker build -f autoresearch/Dockerfile -t wigner-autoresearch:local .
./autoresearch/sandbox.sh --label baseline --size 2048 --repeats 5
```

The runner disables networking, mounts only `candidate.py` from the working tree, makes the container filesystem read-only, drops Linux capabilities, and limits the run to one CPU, 4 GB RAM, 64 processes, and the evaluator timeout. Each JSON result is appended to `autoresearch/results.jsonl` on the host.

Point a coding agent at `autoresearch/PROGRAM.md` after the baseline noise floor is known. Use a separate branch/worktree and results file for every concurrent agent.

## What this can and cannot establish

This environment can support reproducible search for implementation improvements and alternative algorithms with executable invariants. It cannot establish scientific novelty, physical validity beyond the frozen oracle, or generalization outside the tested domain.

The public evaluator is visible to the agent and can be gamed. A serious result therefore needs:

1. held-out signals, sizes, and seeds evaluated outside the agent's workspace;
2. repeated paired measurements and a declared noise floor;
3. property checks relevant to the Wigner transform, not only element-wise agreement;
4. leave-one-out simplification and independent code review;
5. literature review and explicit provenance for any novelty claim;
6. full notebook execution before replacing `sigproc.dwv`.

## Design context

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) demonstrates the small editable surface, fixed budget, scalar metric, and experiment log.
- [MLXFast Challenge](https://github.com/Layr-Labs/mlxfast-challenge) separates a trusted harness from a sandboxed worker, treats correctness as a hard gate, uses hidden cases, and pairs candidate timing with a same-machine baseline.
- [Prime Intellect's autonomous nanoGPT study](https://www.primeintellect.ai/auto-nanogpt) shows the value of durable scratchpads, statistical noise floors, experiment lineage, periodic source refresh, and pruning; it also reports that agents mostly recombined known work and failed to improve the novelty-gated track.
- [AIDE](https://github.com/WecoAI/aideml) and [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) explore tree search and broader ideation, while warning that open-ended systems have lower success rates and that generated code requires sandboxing.
- Research-agent benchmarks such as [MLE-bench](https://github.com/openai/mle-bench) and [CORE-Bench](https://arxiv.org/abs/2409.11363) reinforce repeated evaluation and computational reproducibility before discovery claims.

Relevant algorithm literature includes efficient Wigner-Ville implementations based on sparse autocorrelation structure, pruned FFTs, real transforms, and ambiguity-domain formulations. Those are research directions, not pre-approved changes: the repository's exact discrete convention remains the oracle until the paper author validates an alternative formulation.
