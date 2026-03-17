# MoE Expert Staging Buffer for Reduced VRAM

## Problem

When using `--cpu-moe`, llama.cpp keeps expert weights on CPU and selectively copies only the activated experts to GPU via DMA. However, the GPU-side copy tensor is still allocated at full size `[n_embd, n_ff, n_expert]`, wasting VRAM. For a model like Step-3.5-Flash with 12,096 experts, the GPU allocates space for all 12,096 experts even though only 8 are ever used per token.

## Solution

A new `--moe-staging N` flag that allocates a GPU staging buffer sized for only N experts (typically `n_expert_used`, the top-k count). At runtime, selected experts are packed contiguously into slots 0..k-1 and the expert IDs are remapped accordingly.

## Usage

```bash
./llama-cli -m mixtral.gguf --cpu-moe --moe-staging 2 -ngl 999 -p "Hello" -n 32
```

- `--moe-staging 2` — allocate GPU space for only 2 expert slots (Mixtral's top-k) instead of all 8
- VRAM per MoE weight tensor reduced by 4x for Mixtral (2/8), or ~1500x for Step-3.5-Flash (8/12096)
- Output is identical with and without staging (correctness-preserving)
- Env var: `LLAMA_ARG_MOE_STAGING=N`

## Files Changed

### `ggml/include/ggml-backend.h`
- New public API: `ggml_backend_sched_set_moe_staging(sched, cap)`

### `ggml/src/ggml-backend.cpp`
- Added `moe_staging_cap` field to `ggml_backend_sched` struct
- **`split_graph`**: When creating GPU copy tensors for cross-backend MoE weights, detects staging mode and creates tensor with `ne[2] = moe_staging_cap` instead of `n_expert`
  - Detection: src is a weight tensor, 3D (`ne[2] > 1`), used as `src[0]` of `MUL_MAT_ID`
  - Preserves original strides for `ne[0]` and `ne[1]` (handles quantized types)
- **`compute_splits`**: In staging mode, packs selected experts contiguously and remaps IDs
  - Builds remap table: `absolute_expert_id → staging_slot`
  - Copies each used expert to its staging slot with MMQ padding
  - Writes remapped IDs to the GPU ids tensor copy
  - Falls back to original sparse-copy logic when staging is disabled
- Added `ggml_backend_sched_set_moe_staging()` implementation

### `include/llama.h`
- Added `int32_t moe_staging` to `llama_context_params` (default: 0 = disabled)

### `src/llama-cparams.h`
- Added `int32_t moe_staging` to internal `llama_cparams`

### `src/llama-context.cpp`
- Wires `params.moe_staging` → `cparams.moe_staging`
- Calls `ggml_backend_sched_set_moe_staging()` after scheduler creation
- Also sets it in the pipeline-parallel retry path
- Added default value (0) in `llama_context_default_params()`

### `common/common.h`
- Added `int32_t moe_staging = 0` to `common_params`

### `common/common.cpp`
- Wires `params.moe_staging` → `cparams.moe_staging` in `common_context_params_to_llama()`

### `common/arg.cpp`
- Added `--moe-staging N` CLI argument with `LLAMA_ARG_MOE_STAGING` env var

## How It Works

### Graph Time (`split_graph`)
When the scheduler creates GPU copy tensors for MoE weight tensors that cross backends:
```
// Instead of: ne[2] = n_expert (e.g., 8 for Mixtral, 12096 for Step-3.5-Flash)
// Creates:    ne[2] = moe_staging_cap (e.g., 2 for Mixtral top-k)
```

### Runtime (`compute_splits`)
```
1. Read expert IDs from the ids tensor (existing logic)
2. Build bitset of used experts (existing logic)
3. NEW: Build remap table (absolute_id → slot)
4. NEW: Pack experts contiguously into staging buffer
   - Expert 37 → slot 0, Expert 142 → slot 1, etc.
5. NEW: Rewrite ids tensor on GPU with remapped indices
   - 37 → 0, 142 → 1, etc.
6. mul_mat_id iterates over ne[2]=k experts, accesses correct data
```

## Verification

```bash
# Build
cmake -B build && cmake --build build

# Test (compare output with and without --moe-staging)
./build/bin/llama-cli -m mixtral.gguf --cpu-moe -ngl 999 -p "Hello" -n 32
./build/bin/llama-cli -m mixtral.gguf --cpu-moe --moe-staging 2 -ngl 999 -p "Hello" -n 32

# Check VRAM reduction
GGML_SCHED_DEBUG=1 ./build/bin/llama-cli -m mixtral.gguf --cpu-moe --moe-staging 2 -ngl 999 -p "Hello" -n 32
```

## Limitations / Future Work

- `moe_staging_cap` must be >= the number of unique experts activated in a single batch. For single-token inference this equals `n_expert_used`. For batched inference with many tokens, the unique count could exceed `n_expert_used` — if it exceeds `moe_staging_cap`, an assertion fires. A future improvement could auto-size or fall back gracefully.
- Currently requires explicit `--moe-staging N` flag. Could auto-enable when `--cpu-moe` is used by reading `n_expert_used` from model metadata.
