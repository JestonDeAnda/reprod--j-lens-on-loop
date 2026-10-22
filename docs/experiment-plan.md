# Experiment plan after lens fitting

This document records the approved plan for completing the experiments in
Wang & Reid (2026), arXiv:2609.01924. No reported paper number is considered
reproduced unless it can be recomputed from a saved raw artefact.

## Pinned upstream

The fitting and lens serialization code comes from Anthropic's public
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
repository at commit
[`581d398613e5602a5af361e1c34d3a92ea82ba8e`](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e).
The dependency is also pinned to this SHA in `pyproject.toml`.

Clone the exact source with:

```bash
git clone https://github.com/anthropics/jacobian-lens.git
git -C jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
```

The existing DSW checkout is:

```text
/mnt/nas-save/wangjunqi/deps/jacobian-lens
```

The same NAS checkout appears inside a DLC worker as:

```text
/mnt/nas-save/nas-save/wangjunqi/deps/jacobian-lens
```

The upstream project supplies the estimator, lens format, readout operations,
and original workspace prompt assets. This repository owns the recurrent-model
adapters, experiment configuration, orchestration, and verification.

## Stage 0: protocol and data lock

Inventory the upstream prompt sets and score functions for all eleven causal
families. Record model/tokenizer revisions, input hashes, token-position rules,
intervention windows, strengths, sample counts, and random seeds under
`configs/experiments/`. Missing paper-specific assets must be identified rather
than silently reconstructed and presented as exact.

Acceptance:

- every experiment has a versioned config and input hash;
- every reported metric has a named raw field from which it is computed;
- no result exists only as an aggregate.

## Stage 1: lens completion and structural analysis

Finish the 4 Ouro, 4 Huginn primary, and 16 Huginn tiling lenses. Extend the
transport analyser to produce raw JSON for:

1. distance-controlled Jacobian norms, top-1 agreement, and KL;
2. Ouro within-loop and cross-loop semantic transport;
3. matched-offset matrix cosine and singular-subspace overlap;
4. mean-Jacobian factorisation through iteration boundaries;
5. Huginn's input-reinjection path;
6. Ouro checkpoint logit-lens and early-exit statistics;
7. Huginn's sliding readout window under the complete tiling.

Comparisons across models or targets must match source-to-target distance.

## Stage 2: shared causal engine

Implement one intervention harness rather than eleven independent programs.
It must support:

- activation recording at virtual layers and selected token positions;
- token directions `normalize(J.T @ unembedding[token])`;
- additive injection, direction ablation, and clamp-transfer;
- single-direction and top-k pursuit-subspace ablation;
- unrelated-token, random-subspace, and no-intervention controls;
- baseline/intervention paired generation;
- per-trial JSONL with prompt ID, condition, model and lens revisions, layers,
  positions, direction tokens, strength, seed, outputs, ranks, and scores;
- Wilson intervals and paired comparisons derived in a separate verifier.

Configured intervention interfaces:

- Ouro: loop-relative bands, with single-loop and all-remaining-loop conditions;
- Huginn: validated lens windows, with previous, last, and all-window conditions;
- Qwen baseline: the paper's L24–48 band;
- Nanbeige: exploratory extension only, never pooled with the paper reproduction.

## Stage 3: causal gates

Run three high-value families first on small pilots, then at paper scale:

1. flexible generalisation;
2. argument/answer/control direction ablation, followed by top-10 subspace
   ablation;
3. verbal report and introspection.

These test the headline interface claims: Ouro writes must survive later
re-derivation, Huginn's last valid window should match all valid windows, and
verbalising existing versus newly injected content should separate.

Do not start the remaining suite if baseline task performance, random controls,
or deterministic replay fail their configured gates.

## Stage 4: remaining causal families

Add and run:

1. directed modulation (focus/control/suppress);
2. top-down summoning;
3. two-hop probe swap;
4. language selectivity;
5. line-count selectivity;
6. ignition;
7. capacity;
8. dual task.

All families must retain per-trial records even where the paper released only
condition-level aggregates.

## Stage 5: demonstration dumps and concept inventory

Generate the eight-prompt, 128-token greedy-completion dumps for every virtual
layer and position. Save both J-lens and logit-lens top-8 probability fields.
Implement non-negative gradient pursuit with:

```text
k = 8, 16, 25, 50
coefficient threshold = 0, 0.01, 0.05 times ||h||
```

Derive occupancy, explained variance, probability-field calibration, and the
inventory-based capacity and subspace-ablation results from these dumps.

## Stage 6: baselines and robustness

After the looped-model suite is validated:

- fit and test the exact Qwen3.6-27B baseline checkpoint used by the paper;
- repeat the Ouro pipeline on Ouro-2.6B-Thinking;
- use Nanbeige4.2-3B as a separately labelled architecture extension.

The exact Qwen model identifier and availability are a blocking provenance
check; no nearby Qwen checkpoint may be substituted without changing the claim.

## DLC execution policy

Use Priority 1 and one GPU per job unless separately approved. CPU jobs handle
artefact verification and statistics. GPU pilots precede paper-scale runs.
Each submitted job uses the required `fsNNNN-wangjunqi-stage-content-sequence`
name and records its PAI JobId, code commit, resource shape, expected and actual
runtime, status, and durable artefact paths.

Suggested execution order:

1. complete and verify current Huginn fits;
2. run structural analysis on completed Ouro and Huginn artefacts;
3. implement the shared causal engine and CPU toy tests;
4. run the three causal gates on Ouro, then Huginn;
5. complete the remaining eight families;
6. run occupancy/inventory experiments;
7. add Qwen, Thinking, and Nanbeige comparisons.
