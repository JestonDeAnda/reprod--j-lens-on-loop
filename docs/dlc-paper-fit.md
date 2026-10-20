# Paper-scale DLC fit

Launched 2026-10-20 after the L20D plumbing preflight in `docs/dlc-run.md`.
This is the paper's WikiText-1000 / full-sources lens family, not the smoke fit.

## Code pin

| Field | Value |
| --- | --- |
| Commit | `49f0f65` (`Add paper-scale DLC fit stages and transport analysis`) |
| Launcher | `scripts/dlc_paper_fit.sh` |
| Shared run name | `paper-fit-20261020T034458Z` |
| NAS output | `/mnt/nas-save/wangjunqi/j-lens-on-loop/runs/paper-fit-20261020T034458Z` |
| DLC view of same path | `/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/runs/paper-fit-20261020T034458Z` |
| Prompt cache | `/mnt/nas-save/wangjunqi/j-lens-on-loop/cache/wikitext-1000x128.json` |

## Jobs

Same image / quota family as the successful preflight `dlc1au579n35631v`
(`nemo:26.06`, `quota746juyotckp`, 1×GPU / 16 CPU / 64Gi / 16Gi shm).
`JobMaxRunningTimeMinutes=43200`. Workers observed so far are NVIDIA L20D.

| Stage | Job id | Targets | Status |
| --- | --- | --- | --- |
| ouro | `dlcgake1793hluil` | loops 1–4, full sources, 1000 prompts | Running on L20D |
| huginn-primary | `dlcgkjzezie0uuq8` | r=4/8/12/16, full sources | Running on L20D |
| huginn-tiling | `dlcgkjzez9j7x6fi` | r=1…16, full sources | Running on L20D |

### Progress snapshot (2026-10-20 ~15:08 UTC / 23:08 CST)

Completed `*.lens.pt` so far:

- `ouro-paper/loop-1.lens.pt`
- `huginn-primary/recurrence-4.lens.pt`
- `huginn-tiling/recurrence-{1,2,3}.lens.pt`

Active checkpoints (still writing):

- `ouro-paper/loop-2.checkpoint.pt`
- `huginn-primary/recurrence-8.checkpoint.pt`
- `huginn-tiling/recurrence-4.checkpoint.pt`

The first DSW monitor agent exited after Cursor API reconnect failures; jobs kept
running independently. A new `jlens-paper` tmux monitor was restarted.

Command template:

```bash
STAGE=<stage> RUN_NAME=paper-fit-20261020T034458Z \
  bash /mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/scripts/dlc_paper_fit.sh
```

## Completion criteria

- `ouro-paper/loop-{1..4}.lens.pt`
- `huginn-primary/recurrence-{4,8,12,16}.lens.pt`
- `huginn-tiling/recurrence-{1..16}.lens.pt`
- Transport reports from `scripts/analyze_transport.py`
- This file updated with final status, wall times, and artefact sizes

## Not yet claimed

Eleven causal experiment families are still a later milestone. Fitting lenses
is the paper's first quantitative gate; causal results will not be asserted
from smoke norms or incomplete artefacts.

## Monitor

DSW tmux session: `jlens-paper`  
Agent log: `/root/jlens-paper-agent.log`  
Status scratch: `/tmp/paper-fit-status.json`
