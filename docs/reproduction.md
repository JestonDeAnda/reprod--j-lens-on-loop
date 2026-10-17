# Reproduction protocol

This repository targets the methods and principal structural claims of
[Wang & Reid (2026)](https://arxiv.org/abs/2609.01924v1). The paper does not
link an official implementation, so the harness extends Anthropic's released
[`jacobian-lens`](https://github.com/anthropics/jacobian-lens) code at pinned
commit `581d3986`.

## What is reproduced

The implementation provides the prerequisite for every paper experiment:
an exact virtual residual stack for recurrent models. Each firing of a shared
block gets a unique hook point while the original model performs the forward
pass. The upstream Jacobian estimator and lens format remain unchanged.

The first quantitative milestone is the transport/readout result:

1. Fit one lens per Ouro loop end and at Huginn recurrences 4/8/12 plus the
   coda end.
2. Verify Ouro's within-loop transport and cross-loop failure.
3. Verify Huginn's approximately two-recurrence transport horizon.
4. Extend Huginn targets to every recurrence end and check that the apparent
   depth trend flattens under equal source-to-target distance.

The eleven causal families require substantial GPU time and per-trial
artefacts. They are a second milestone, not part of the smoke test.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
pytest
```

Model weights require accepting the relevant Hugging Face terms and enough
local disk. `trust_remote_code=True` is required for both recurrent models.

## Mandatory preflight

Run this before a costly fit:

```bash
python scripts/fit_family.py huginn \
  --output artifacts/huginn --validate-only

python scripts/fit_family.py ouro \
  --output artifacts/ouro --validate-only
```

The preflight fails unless:

- every virtual hook fires exactly once;
- two forwards are bit-exact;
- every virtual source has a non-zero path to the target;
- distinct firings do not collapse to one captured activation.

These checks catch the paper's three silent failures. In particular, Huginn
must receive `num_steps=(0, 16)`: a scalar `16` puts all recurrences inside its
`no_grad` phase and silently produces zero recurrent Jacobians.

## Paper-scale fitting

The exact corpus settings are defaults:

```bash
python scripts/fit_family.py ouro \
  --output artifacts/ouro-paper --full-sources

python scripts/fit_family.py huginn \
  --output artifacts/huginn-paper --full-sources
```

Each target is checkpointed independently and resumes after interruption.
For a cheaper pilot, omit `--full-sources`; this retains only the two-core-
iteration window nearest each target. The pilot can validate plumbing and the
local transport horizon but cannot support full-depth claims. `--source-window N`
shrinks that window further to the N sources nearest the target, which is only
a plumbing check.

To complete the 16-target Huginn tiling:

```bash
python scripts/fit_family.py huginn \
  --output artifacts/huginn-tiling --full-sources \
  --targets 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

## Expected resource envelope

The paper reports bf16 fitting on one to three NVIDIA B200 GPUs:

- Ouro: roughly 1 hour for the final-target family plus 1.6 hours for loop
  targets;
- Huginn: roughly 7.3 hours for its primary fit, plus each gap-fill target;
- Qwen baseline: roughly 3 hours for all layers and 2 hours for the band-entry
  lens.

These are reference numbers, not scheduling guarantees. Full-source Huginn
also uses several GB of CPU memory for fp32 running Jacobian sums.

## Interpretation guardrails

- Compare architectures only at matched source-to-target distance.
- A low matrix norm, poor top-1 agreement, or high KL does not by itself make
  a long-range lens unusable; test a semantic direction and a causal edit.
- Never infer a depth trend from unevenly spaced lens targets.
- Jacobian lenses are blind to content that does not influence the current
  target, so a null result is not proof that the model stores no such content.
- Record raw per-trial outputs. Aggregate-only files cannot independently
  verify percentages or confidence intervals.
