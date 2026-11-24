# Qwen3.8-27B with GCG on this branch

Branch `qwen3.8-support`, based on `anna/forecasting` (621d8cb). Ported from Jonas' `jonas` branch, where
Qwen3.8-27B ran with GCG, claudini and PAIR. Scope here: **GCG only, thinking off**.

**Status: CPU-tested only. Nothing on this branch has run on a GPU yet.** Do the first-GPU checklist at
the end before launching a sweep.

## What changed and why

| Change | Why |
|---|---|
| `pyproject.toml`, `pixi.lock`, `requirements.txt`: transformers 4.57.1 -> **5.10.4**, peft 0.14.0 -> 0.19.1, huggingface-hub 0.x -> >=1.5,<2, add `protobuf`, `flash-linear-attention>=0.6,<1`, conda `gcc==15.2.0` + `ld_impl_linux-64==2.45.1`; drop `jailbreakbench` and `litellm` | Qwen3.8 is the `qwen3_5` architecture (Gated DeltaNet + full attention), which needs transformers 5.x. `jailbreakbench` pins transformers<5. `litellm==1.25.2` is unused and blocks every re-solve. fla gives the fused GatedDeltaNet kernels; without it transformers falls back to a slow, memory-hungry torch path. Triton (pulled in by fla) JIT-compiles a C stub at model load, so a compiler must be in the env at runtime. |
| `adversariallm/dataset/jbb_behaviors.py`: load `JailbreakBench/JBB-Behaviors` with `datasets` | Replaces the `jailbreakbench` package. Checked: same 100 Goals/Targets in the same order as `jbb.read_dataset()`. |
| `lm_utils/generation.py`: `HybridCache` -> `StaticCache` | `HybridCache` no longer exists in transformers 5 (import error for every model). Only the Gemma-2/3 cache path uses it. |
| `lm_utils/generation.py`: `is_hybrid_linear_attention_model`, `prefix_cache_allowed`, `DynamicCache(config=...)` for hybrid models, `reorder_cache` for linear-attention layers; `LAYER_TYPE_CACHE_MAPPING` import fallback | Without these, cached generation on Qwen3.5/3.8 crashes ("has_previous_state can only be called on LinearAttention layers"). The import fallback keeps it working if you later bump to transformers 5.15 (symbol renamed there). Non-hybrid models take the same code path as before. |
| `attacks/gcg.py` (and `gcg_refusal.py`, `beast.py`): prefix cache only if `prefix_cache_allowed(...)` | These attacks reuse a prefix cache and reset it with `crop()`. `crop()` is a no-op for linear-attention state, so on Qwen the cache would silently keep the previous candidate's suffix and **every candidate loss would be wrong**. Hybrid models now run without a prefix cache (logged as a warning). Correct, only slower. Other models are unchanged. |
| `lm_utils/batching.py`: halve the batch on "Triton Error [CUDA]: out of memory" and on `canUse32BitIndexMath` | fla's Triton kernels report OOM as a plain `RuntimeError`, which used to escape the retry. Qwen's causal conv1d hits the 32-bit index limit at large batch x length. All other `RuntimeError`s are still raised. |
| `lm_utils/tokenization.py`: `get_chat_template_kwargs`; `tokenize_chats` passes them to `apply_chat_template` | **The key Qwen3.8 fix.** Its template adds a system message "Reasoning effort is set to xhigh. Please think carefully ..." to *every* render unless `enable_thinking=False` is passed, also for `add_generation_prompt=False` (which is how `prepare_conversation` renders). Without the kwargs, GCG would optimise against a prompt that asks for reasoning, then answers in non-thinking form. Default: thinking off for templates that have an `enable_thinking` switch, no kwargs otherwise (so other models render exactly as before). |
| `lm_utils/tokenization.py`: Llama-2 `</s><s> [INST]` space | transformers 5 tokenizes Llama-2 turn boundaries differently; this keeps the historical tokens. Only relevant for multi-turn Llama-2. |
| `io_utils/model_loading.py`: attach `chat_template_kwargs` from the model config to the tokenizer | Feeds the above. |
| `conf/models/models.yaml`: `Qwen/Qwen3.8-27B` with `chat_template_kwargs: {enable_thinking: false}` | Model key. |
| `tests/test_qwen38_gcg.py` | CPU tests, see below. |

### What the model sees (verified on CPU)
GCG input for prompt P, suffix S, target T:
```
<|im_start|>user\n{P}{S}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{T}<|im_end|>\n
```
- No system message, no BOS (Qwen has no BOS token, so the BOS handling in `tokenize_chats` never applies).
- Everything before `{T}` is token-identical to `apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)`, i.e. the official non-thinking generation prompt. The empty `<think></think>` block is
  part of that official prompt, not something we insert.
- Target starts right after that header; only `<|im_end|>\n` follows it.
- Stop ids: `<|im_end|>` (248046, tokenizer eos) and `<|endoftext|>` (248044, from `generation_config.json`).
  Pad = `<|endoftext|>`.
- GCG's default init `"x x x ..."` merges with prompts ending in a letter ("nitazenex"), so GCG uses its existing
  fallback `" x x x ..."` (20 tokens).
- Cross-check: for all 220 sampled steps of Jonas' 20 GCG reference runs on Qwen3.8 (see below), this branch's
  `prepare_conversation` reproduces the logged `model_input_tokens` exactly.

## GCG: differences between this branch and `jonas`, and what was done

| Topic | Anna's GCG (kept) | jonas | Handling |
|---|---|---|---|
| Prefix cache | always (except Gemma) | off for hybrid models | **ported** (required for correctness) |
| Generation input | `prepare_conversation` re-tokenises the decoded suffix | same | unchanged. Safe because `filter_ids: true` keeps only suffixes that round-trip; do not set `filter_ids: false` for Qwen. (This is GCG; the claudini retokenisation bug does not apply.) |
| `filter_ids` round trip | `filter_suffix` | same | unchanged; tested on Qwen: ordinary tokens kept, embedding padding rows and a trailing space (the Qwen template `trim`s message content) rejected |
| Embedding scale | Gemma `embed_scale` only | + Muse-Glimmer `embed_norm` | Qwen has neither; not ported |
| `allow_non_ascii` | false | false | unchanged |
| Disallowed tokens | `get_disallowed_ids` | same | unchanged, see "Open decision" below |
| `sampling_schedule` | `generation_config.sampling_schedule` | `attacks.gcg.sampling_schedule` | kept Anna's |
| Resume | `resume_from`, no metadata | + `generation_time`, `resume_metadata`, one-item guard, `weights_only=False` | kept Anna's (not Qwen-specific) |

### Open decision: added tokens in the GCG candidate pool
Qwen3.8 registers `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, FIM
and repo tokens (248058-248069) as **non-special** added tokens. `get_disallowed_ids(allow_special=False)` only
drops tokens that decode to an empty string with `skip_special_tokens=True`, so these stay allowed. They are plain
ASCII and round-trip through the tokenizer, so they are text-reachable and pass `filter_suffix`.
GCG uses them a lot: in Jonas' 20 reference runs, the final 20-token suffixes over 10,000 logged steps contained
`<|file_sep|>` 2213x, `<|fim_suffix|>` 1369x, `<|fim_middle|>` 1188x, `<tool_response>` 654x, `</think>` 490x, ...
This branch keeps that behaviour, so results stay comparable to those runs. claudini on `jonas` (>= 0.0.3) forbids
all added tokens and the 243 embedding padding rows instead. If you want that for GCG too, it is a deliberate
protocol change: add it as a config flag and bump `attacks.gcg.version`.
The 243 padding rows (embedding 248320 rows vs 248077 tokens) are not disallowed either, but `filter_ids: true`
rejects them (they decode to "").

## Environment

Pinned in `pyproject.toml` / `pixi.lock` (re-solved 2026-11-24; `pixi install --frozen` works):

| package | version |
|---|---|
| python | 3.10.16 |
| torch | 2.7.1+cu128 (unchanged; triton 3.3.1) |
| transformers | **5.10.4** |
| flash-linear-attention / fla-core | 0.6.0 |
| peft | 0.19.1 |
| huggingface-hub | 1.33.0 (resolved from >=1.5,<2) |
| safetensors | 0.7.0 |
| protobuf | 7.36.2 |
| gcc (conda) | 15.2.0 |

This matches the env of Jonas' Qwen3.8 claudini/PAIR sweeps (transformers 5.10.4, torch 2.7.1+cu128, fla 0.6.0)
except huggingface-hub 1.26 -> 1.33 and no torchvision/wandb.

- **Do not upgrade to torch 2.11 / cu130.** It does not run on the DAML nodes' CUDA 12.8 driver.
- **Do not upgrade to transformers 5.15** without measuring. 5.15.1 changes Qwen3.x numerics: it always uses the
  torch `Qwen3_5RMSNormGated` instead of fla's fused norm, and looks up `recurrent_gated_delta_rule`, which fla 0.6.0
  does not export, so decode falls back to torch (about 2x slower generation, mean KL ~1e-4, GCG/claudini
  trajectories diverge from step 1). Llama/Gemma/judges are bit-identical across the two. 5.15 would also need
  `safetensors>=0.8`. Treat a series that spans both versions as two series.
- Note: Jonas' 20 **GCG reference runs** (below) were made with transformers **5.15.1**; his claudini/PAIR sweeps with
  5.10.4. Compare GCG curves from this branch to them statistically, not run-by-run.

## How to run GCG on Qwen3.8-27B

- Model key: `model=Qwen/Qwen3.8-27B`. Weights are in the shared cache `/ceph/ssd/shared/hf_models`
  (52 GB, refs/main = `37561816`; the older snapshot `1d4bf0f2` has the same weight/config/tokenizer blobs and an
  equivalent chat template).
- GPU: one **H100 or H200 (80 GB+)**. 27B in bf16 is ~52 GB of weights plus activations for 512 candidates.
  **Never A100** (the default launcher `a100h100` includes `gpu_a100`). Ask for enough host RAM (the H100 launcher
  default of 16 GB is too small for comfort; Jonas used 96 GB).
- Always `umask 002` and `HF_HUB_OFFLINE=1` (the model is cached; offline avoids Hub metadata calls, lock files and
  silent re-downloads).
- Expect ~1.5 min of Triton JIT on the first step, then roughly 8-16 s per GCG step on H100/H200 (Jonas' reference:
  500 steps + 11 x 64 samples in 70-140 min).

Example (one behaviour, 500 steps, 64 samples at step 0 and every 50 steps):
```bash
umask 002
export HF_HUB_OFFLINE=1
SCHEDULE=$(python3 -c "print('[' + ','.join(str(64 if (i == 0 or (i + 1) % 50 == 0) else 0) for i in range(500)) + ']')")
pixi run --frozen python run_attacks.py -m \
  hydra/launcher=h100 hydra.launcher.partition=gpu_h100,gpu_h200 hydra.launcher.mem_gb=96 hydra.launcher.timeout_min=240 \
  model=Qwen/Qwen3.8-27B dataset=guidedbench "datasets.guidedbench.idx=[14]" \
  attack=gcg attacks.gcg.num_steps=500 \
  generation_config.temperature=1.0 generation_config.max_new_tokens=256 \
  "generation_config.sampling_schedule=$SCHEDULE"
```
On this branch the schedule is `generation_config.sampling_schedule` (it reaches `attacks.gcg.generation_config`;
checked with `--cfg job --resolve`), not `attacks.gcg.sampling_schedule` as on `jonas`. GuidedBench is read from
`${root_dir}/../GuidedBench/Dataset/GuidedBench.csv` here.

Reference runs from `jonas` (GCG defaults, 500 steps, seed 0, 20 GuidedBench behaviours, 64 samples at T=1.0 at
step 0 and every 50 steps, strong_reject): run list
`/ceph/ssd/staff/dornb/llm-quick-check/claudini_investigation/gcg_reference/gcg_asis_Qwen_Qwen3.8-27B_runs.txt`,
curves `gcg_asis_Qwen_Qwen3.8-27B_curves.json` in the same directory. Loss goes from 1.1-3.4 at step 0 to 0.1-1.6 at
step 499; 14,076 of 14,080 sampled completions are non-empty.

## Pitfalls (symptom -> fix)

- **transformers 5.15 drift**: Qwen losses/samples differ from a 5.10.4 run from step 1, generation ~2x slower ->
  stay on 5.10.4 (pinned); record the version if you ever mix.
- **"The fast path is not available because one of the required library is not installed. Falling back to torch
  implementation."** -> harmless: `causal-conv1d` is deliberately not installed (sdist only, needs nvcc); the torch
  conv fallback is cheap. fla still provides the GatedDeltaNet kernels.
- **"Failed to find C compiler"** at model load -> the env lacks `gcc`; use the pinned env (`pixi install --frozen`),
  do not run with a bare system python.
- **Missing / unreadable shard** (`OSError` / "does not appear to have a file named model-000XX-of-00018.safetensors",
  or an empty snapshot dir) -> transient: someone is downloading or updating the shared cache (refs/main can point to
  a commit whose snapshot is still being filled). Use `HF_HUB_OFFLINE=1`, check that the snapshot has all 18 shards,
  and resubmit. Never redirect `HF_HUB_CACHE` to a personal dir as a workaround.
- **`PermissionError` on `/ceph/ssd/shared/hf_models/.locks/...lock`** -> another user downloaded with umask 022 and
  left 0644 lock files. `HF_HUB_OFFLINE=1` avoids locks entirely. Stale locks can be removed without admin rights
  (the lock dirs are group-writable): `find /ceph/ssd/shared/hf_models -path '*/.locks/*' -type f ! -writable -delete`
  (only when nothing is downloading). Always `umask 002` before any download.
- **"Prefix caching disabled: ... linear-attention layers"** warning -> expected for Qwen, not an error.
- **Silent empty completions**: a run can finish with `run.json` but empty `model_completions`; always check them
  (checklist below), not only the exit status.
- **First step slow** (~90 s) -> Triton JIT, expected.

## Available on `jonas` but deliberately not ported
- Full `reasoning` config block (`lm_utils/reasoning.py`: reasoning/answer split, `model_completions_raw`,
  reasoning token budgets). Not needed with thinking off; here `chat_template_kwargs` does the same for Qwen3.8.
- `prepare_conversation` rewrite that cuts every part from the generation prompt (`add_generation_prompt=True`).
  For Qwen3.8 single-turn GCG it gives exactly the same tokens as this branch (checked against the reference runs).
  It matters for Gemma-4, gpt-oss and other reasoning models, and changes their behaviour, so it was left out.
- `tokenize_chats` Gemma-4 BOS fix: Qwen has no BOS, irrelevant here. Needed if you add Gemma-4.
- Muse-Glimmer `embed_norm` in GCG, claudini 0.0.4-0.0.7 fixes (generate from ids, round-trip filter, Gemma
  `embed_scale`), PAIR changes and perf knobs (`hybrid_left_pad_prefill`, fla recurrent decode), Helmholtz/B300
  deployment, sticky batch-size memo, JSON-constrained generation fix for transformers 5 (`lm_utils/filters.py`;
  affects PAIR/JSON-filtered generation, not GCG), shared-cache lock fallback in the model loader.

## Not tested
- No GPU run on this branch: model load, GCG optimisation, generation and judging with transformers 5.10.4 are
  untested *here* (they worked on `jonas` with the same library versions).
- Other attacks and models under transformers 5 on this branch (only CPU tokenizer tests, see below). Gemma-2/3
  generation (now `StaticCache`) and anything using peft are untested.
- `run_judges.py` / judgezoo under transformers 5 on this branch.

## Tests run (CPU)
- `HF_HUB_OFFLINE=1 pixi run pytest tests/test_qwen38_gcg.py` (needs the cached Qwen3.8 tokenizer/config): template
  vs native non-thinking prompt, target position, no reasoning system message / single empty think block, stop ids,
  `filter_suffix` round trip, padding rows, hybrid detection, prefix-cache guard, batch halving.
- `tests/test_lm_utils.py` subset (`-k "ground_truth and (Qwen2.5 or Llama-2 or Meta-Llama-3.1 or Mistral-7B-Instruct-v0.3 or gemma-3-1b) or basic or invalid"`): same result on `anna/forecasting` + transformers 4.57.1 and on this branch (18 passed, 4 failed: the Llama-3.1 ground truth, which was already failing before this branch because of its date fixture).

## First GPU use: checklist
1. `pixi install --frozen`; `pixi run python -c "import transformers, fla; print(transformers.__version__)"` -> 5.10.4.
2. Smoke run on one H100/H200: `model=Qwen/Qwen3.8-27B dataset=guidedbench "datasets.guidedbench.idx=[14]"
   attack=gcg attacks.gcg.num_steps=5` with sampling at step 0 and 4 (e.g. 8 samples, `max_new_tokens=128`).
3. In the log: the "Prefix caching disabled" warning appears; no "Failed to find C compiler"; no traceback in
   `.submitit/*_log.err`.
4. In `run.json`:
   - every sampled step has non-empty `model_completions`, and they are plain answers: no `<think>`, `</think>`, or
     "Reasoning effort" text, not all identical, `time_taken` > 0;
   - `model_input_tokens` ends with `248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271`
     (`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`) and contains no system turn;
   - `loss` at step 0 is in the range of the reference runs (about 1.1-3.4 for GuidedBench) and does not increase
     over the 5 steps; a loss near 0 or > 10 at step 0 means a broken prompt.
5. For idx 14 the reference run's step-0 loss was 2.55 (transformers 5.15.1, so expect close, not identical).
6. Then a 500-step run on 1-2 behaviours and compare its loss curve with the reference curves before a sweep.
