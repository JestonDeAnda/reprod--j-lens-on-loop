# Jacobian Lens on Looped Transformers

Reproduction harness for
[Looped Transformers under the Jacobian Lens: Does the Global Workspace
Survive Recurrence?](https://arxiv.org/abs/2609.01924v1).

The paper applies mean input-output Jacobian lenses to two weight-tied
transformers. This project contributes the missing virtual-unrolling layer
that presents recurrent block firings as an ordinary deep stack:

- **Ouro-2.6B:** every physical layer firing is indexed by
  `loop × physical_depth + layer`;
- **Huginn-0125:** the virtual stack is
  `prelude + recurrence × core + coda`.

It uses Anthropic's
[`jacobian-lens`](https://github.com/anthropics/jacobian-lens) as a pinned
dependency for fitting and readout.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
pytest
```

Validate the adapter before allocating an expensive job:

```bash
python scripts/fit_family.py huginn \
  --output artifacts/huginn --validate-only
```

Fit the paper's primary lens family:

```bash
python scripts/fit_family.py ouro \
  --output artifacts/ouro --full-sources

python scripts/fit_family.py huginn \
  --output artifacts/huginn --full-sources
```

See [the reproduction protocol](docs/reproduction.md) for scope, exact
settings, validation criteria, resource expectations, and interpretation
guardrails.

## Status

- [x] Paper protocol extracted and documented
- [x] Ouro virtual-unrolling adapter
- [x] Huginn virtual-unrolling adapter
- [x] Determinism, hook-routing, and gradient-flow preflight
- [x] Resumable family fitting on WikiText-103
- [ ] Paper-scale lens artefacts
- [ ] Transport/readout figures
- [ ] Eleven causal experiment families

Unchecked items require model weights and GPU compute. No paper result is
claimed reproduced until its raw artefact and verification output are saved.

## Attribution

The fitting code is supplied by the Apache-2.0-licensed
`anthropics/jacobian-lens` dependency. This repository's adapter and
experiment orchestration are released under the MIT license.
