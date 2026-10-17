# DLC preflight

One NVIDIA L20D job checked the real ByteDance/Ouro-2.6B and tomg-group-umd/huginn-0125 checkpoints and fit one short prompt with two source layers. That is a plumbing check. It is not the paper's multi-hour B200 fit, and that fit was not submitted.

DSW unit tests and `ruff check src tests scripts` passed on the GPU-less instance before the successful job (11 tests). The worker does not see the DSW filesystem or its Python environment. Code and weights were read from NAS.

## Job that passed

| Field | Value |
| --- | --- |
| Job id | `dlc1au579n35631v` |
| Status | Succeeded (`JobSucceeded`), duration 475s |
| Created | 2026-10-17T04:27:21Z |
| Display name | `fs00150-wangjunqi-jlens-preflight` |
| Quota | `quota746juyotckp` (taken from a recent workspace-270 job; not hardcoded) |
| Image | `sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06` |
| Resource | `ResourceConfig`: 1 GPU, 16 CPU, 64Gi memory, 16Gi shared memory. No ECS spec. |
| GPU that ran | NVIDIA L20D, 275040 MiB, driver 580.105.08, CUDA 13.2 |
| Cap | `JobMaxRunningTimeMinutes` 120 |
| Region / endpoint | `cn-zhongwei`, `pai-dlc.cn-zhongwei.aliyuncs.com` |

Log, from the DSW mount:

`/mnt/nas-save/wangjunqi/j-lens-on-loop/runs/dlc-smoke-20261017T042755Z/smoke.log`

The same directory on the worker is `/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/runs/dlc-smoke-20261017T042755Z`. DSW mounts the NFS export `/nas-save` at `/mnt/nas-save`. A DLC data source for the NAS URI mounts the filesystem root, so that export is one level lower.

### Validate-only

Both forwards were bit-exact (`max_repeat_error` 0), every virtual layer fired once, adjacent firings differed, and every source had a nonzero gradient into the final virtual layer.

| Model | Virtual layers | Result |
| --- | --- | --- |
| Ouro-2.6B (48 layers × `total_ut_steps=4`) | 192 | pass |
| huginn-0125, harness recurrence 16 (prelude 2 + 16×4 + coda 2) | 68 | pass |

Ouro attention was eager. Huginn kept its own math-SDPA path with flash and memory-efficient kernels disabled, which is what made the two forwards match.

### One-prompt fit

`--source-window 2`, `--max-seq-len 32`, `--dim-batch 16`, one local sentence instead of WikiText. `max_d_mean=nan` is the pinned jacobian-lens statistic for the first prompt: there is no previous mean to compare. The Jacobians themselves were written.

| Run | Target | seq_len | n_valid | max\|\|J\|\|/sqrt(d) | Artefacts |
| --- | --- | --- | --- | --- | --- |
| Ouro loop 1 | L47 | 30 | 13 | 1.267 | `ouro-smoke/loop-1.lens.pt` (16,779,141 bytes), `loop-1.checkpoint.pt` (33,556,589 bytes) |
| Huginn recurrence 4 | L17 | 31 | 14 | 1.120 | `huginn-smoke/recurrence-4.lens.pt` (111,515,637 bytes), `recurrence-4.checkpoint.pt` (223,029,469 bytes) |

`metadata.json` sits next to each pair. `*.lens.pt` and `*.checkpoint.pt` stay on NAS; they are gitignored. These norms are one prompt and two adjacent sources. They are not paper-scale measurements.

Weights used, already on NAS (not copied onto the DSW container disk):

- Ouro: `/mnt/nas-save/wangjunqi/ouro/models/Ouro-2.6B`
- Huginn: `/mnt/nas-save/wangjunqi/j-lens-on-loop/models/huginn-0125` (downloaded by `dlc1of8nwrfycpnt`)
- jacobian-lens pin `581d398613e5602a5af361e1c34d3a92ea82ba8e`: `/mnt/nas-save/wangjunqi/deps/jacobian-lens`

## Earlier jobs

All four used the same image, quota, and 1×GPU / 16 CPU / 64Gi resource config.

| Job | Created | Duration | Exit | Why it stopped |
| --- | --- | --- | --- | --- |
| `dlc1lmqrvl1n0ty7` | 2026-10-17T03:46:13Z | 37s | 127 | Worker command was `bash /mnt/nas-save/wangjunqi/j-lens-on-loop/scripts/dlc_preflight.sh`. That path is the DSW view and does not exist in the worker. |
| `dlct2ijnve2c3f1u` | 2026-10-17T03:47:53Z | 32s | 127 | Same path. `ls /mnt/nas-save` showed the filesystem root, including a child directory `nas-save`. The repo lives under that child. |
| `dlc1of8nwrfycpnt` | 2026-10-17T03:49:46Z | 1297s | 1 | Nested path worked and Huginn weights finished downloading. Ouro `from_pretrained` died in `OuroModel.__init__`: the NAS `config.json` has bos/eos and no `pad_token_id`, and transformers 5.8 raises `AttributeError` on the missing attribute. Log: `runs/dlc-smoke-20261017T035019Z/smoke.log`. |
| `dlcdjf47f98m18w9` | 2026-10-17T04:16:49Z | 301s | 1 | Ouro validate passed (192 layers, `max_repeat_error` 0). Huginn `post_init` then crashed because `_tied_weights_keys = ["lm_head.weight"]` is the transformers 4 list form; 5.8 calls `.keys()` on it. Log: `runs/dlc-smoke-20261017T041724Z/smoke.log`. |

The loader now sets a missing `pad_token_id` to `None` (so `nn.Embedding` does not zero a real token row) and rewrites a legacy tied-key list to `{target: input-embedding weight}` before `post_init`. Those are commits `a201b3a` and `73477a1`.

## Run the same check again

From this DSW:

```bash
bash /root/.claude/skills/launching-dlc-jobs/scripts/run-dlc-dummy.sh \
  --name fs00150-wangjunqi-jlens-preflight \
  --gpu 1 --cpu 16 --memory 64Gi --shm 16Gi \
  --timeout 120 --poll 30 \
  --command 'bash /mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/scripts/dlc_preflight.sh'
```

The script installs the NAS checkout, skips the Huginn download when `models/huginn-0125/config.json` is already present, runs both validate-only passes, then the one-prompt fits. Copy any newer `src/` and `scripts/` onto `/mnt/nas-save/wangjunqi/j-lens-on-loop` first. The worker installs that tree, not GitHub.

Inside a worker that already has the mounts, the two validate commands are:

```bash
export HF_HOME=/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/hf-home
export HF_ENDPOINT=https://hf-mirror.com
cd /mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop
python3 scripts/fit_family.py ouro \
  --model /mnt/nas-save/nas-save/wangjunqi/ouro/models/Ouro-2.6B \
  --output "$PWD/runs/manual-ouro" --validate-only
python3 scripts/fit_family.py huginn \
  --model /mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/models/huginn-0125 \
  --output "$PWD/runs/manual-huginn" --validate-only
```

The paper-scale commands in `docs/reproduction.md` (`--full-sources`, default 1000 WikiText prompts, and the 16-target Huginn tiling) were not submitted. The paper's own envelope is hours on one to three B200s. This quota accepted one L20D with a 120 minute cap; that is not a budget for that run.
