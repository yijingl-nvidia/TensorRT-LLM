<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Fused MLA KV Cache Guide

This note explains how the Python attention metadata and C++ KV cache manager
buffers become the paged KV-cache interface consumed by attention kernels. The
target path is GLM-5 FP8 MLA serving with DSA sparse attention, batch size 1, and
MTP enabled.

## Ownership

The DeepSeekV3/GLM model modules do not allocate the KV cache backing memory in
their constructors. The executor/resource-manager path owns it.

- `_torch/pyexecutor/_util.py` selects the concrete cache manager class. For DSA
  sparse attention, `get_sparse_attn_kv_cache_manager()` returns
  `DSACacheManager`.
- `DSACacheManager` derives from `KVCacheManager`.
- `KVCacheManager.__init__()` creates `KVCacheManagerCpp`, calls
  `allocate_pools()`, and exposes pool metadata through:
  - `kv_cache_pool_pointers`
  - `kv_cache_pool_mapping`
  - `host_kv_cache_block_offsets`
- Each serving step, `resource_manager.prepare_resources()` asks the C++ manager
  to add sequences/tokens and refresh pages.
- `TrtllmAttentionMetadata.prepare()` calls
  `kv_cache_manager.copy_batch_block_offsets(...)` to copy the active request
  block table into `metadata.kv_cache_block_offsets`.

The model forward receives only `attn_metadata`. Attention kernels reach the
cache through that metadata.

## Python Interface Tensors

These are the important tensors to pass from Python into a fused C++/CUDA op.

### `metadata.kv_cache_manager.kv_cache_pool_pointers`

CPU tensor containing raw KV-cache pool addresses.

- Usual dtype: `torch.int64` from Python's point of view, representing raw
  pointer values.
- Normal shape: `[num_pools, 2]`.
- The `2` axis is `[primary_pool_ptr, secondary_pool_ptr]`.
- For cache formats with separate block-scale pools, the shape can be
  `[num_pools, 2, 2]`, where the last axis is `[data_ptr, scale_ptr]`.

For GLM-5 FP8 MLA, the data pool stores compressed latent KV, not expanded K/V.
The per-token cache vector size is:

```text
kv_lora_rank + qk_rope_head_dim
```

For the config being traced:

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
cached vector size = 576 elements
```

### `metadata.kv_cache_manager.kv_cache_pool_mapping`

CPU `torch.int32` tensor with shape `[num_local_layers, 2]`.

For each local layer:

```text
kv_cache_pool_mapping[layer_idx, 0] = pool_index
kv_cache_pool_mapping[layer_idx, 1] = layer_index_within_pool
```

The THOP helper uses the current `layer_idx` to choose the backing pool and to
compute the byte offset for this layer inside the pool.

### `metadata.kv_cache_block_offsets`

CUDA `torch.int32` tensor with shape:

```text
[num_attention_op_pools, max_num_sequences, 2, max_blocks_per_seq]
```

This is the request block table for the current forward step. Although it is an
`int32` tensor, each element is interpreted as `KVCacheIndex`, not as a plain
byte offset.

The `2` axis is the historical K/V table axis. For MLA `SELFKONLY`, the useful
compressed latent cache is accessed through the K side by convention.

For batch size 1 with MTP, `max_num_sequences` is still the engine capacity. The
runtime slice consumed by kernels is determined by `attn_metadata.num_seqs`,
`attn_metadata.num_contexts`, and `attn_metadata.num_generations`.

## `attentionOp.cpp` Conversion

`cpp/tensorrt_llm/thop/attentionOp.cpp` does not call the C++ KV cache manager.
It consumes the three tensors above and converts them into a small C++ struct
that CUDA kernels can use.

The central helper is:

```cpp
tensorrt_llm::torch_ext::buildPagedKvCacheBuffers(...)
```

The helper performs these steps.

1. Select the pool for the current layer:

```cpp
int32_t poolIndex = host_kv_cache_pool_mapping[layer_idx, 0];
int32_t layerIdxInCachePool = host_kv_cache_pool_mapping[layer_idx, 1];
```

2. Select the phase-local block table:

```cpp
auto* blockOffsets = kv_cache_block_offsets[poolIndex, seq_offset].data_ptr();
```

`seq_offset` is `0` for context and `num_contexts` for generation. After this
slice, sequence index `0` in the kernel means the first sequence in that phase.

3. Compute the layer byte offset inside the selected pool:

```cpp
blockSize = tokens_per_block * kv_head_num * size_per_head;
bytesPerBlock = blockSize * cacheElemBits / 8;
kvFactor = is_mla_enable ? 1 : 2;
intraPoolOffset = layerIdxInCachePool * kvFactor * bytesPerBlock;
```

For GLM-5 FP8 MLA:

```text
is_mla_enable = true
kvFactor = 1
kv_head_num = 1
size_per_head = kv_lora_rank + qk_rope_head_dim = 576
cacheElemBits = 8
bytesPerToken = 576
bytesPerBlock = tokens_per_block * 576
```

4. Build layer-local pool pointers:

```cpp
primaryPoolPtr = host_kv_cache_pool_pointers[poolIndex, 0] + intraPoolOffset;
secondaryPoolPtr = host_kv_cache_pool_pointers[poolIndex, 1] + intraPoolOffset;
```

5. Construct `KVBlockArray`:

```cpp
KVBlockArray(
    batchSize,
    maxBlocksPerSeq,
    tokensPerBlock,
    sizePerToken,
    cyclicAttentionWindowSize,
    maxCyclicAttentionWindowSize,
    sinkTokenLen,
    canUseOneMoreBlock,
    primaryPoolPtr,
    secondaryPoolPtr,
    blockOffsets);
```

Your fused C++ launcher should either call `buildPagedKvCacheBuffers()` directly
or reproduce this setup exactly.

## Kernel Address Calculation

`KVBlockArray` is defined in `cpp/tensorrt_llm/kernels/kvCacheUtils.h`. It is
passed by value into CUDA kernels and contains:

```cpp
void* mPrimaryPoolPtr;
void* mSecondaryPoolPtr;
KVCacheIndex const* data;
int32_t mMaxBlocksPerSeq;
int32_t mTokensPerBlock;
int32_t mBytesPerBlock;
```

The standard kernel access pattern is:

```cpp
int tokenKVIdx = kvCacheBuffer.getKVTokenIdx(tokenIdx);
auto* block = reinterpret_cast<TCache*>(
    kvCacheBuffer.getKBlockPtr(seqIdx, tokenKVIdx));
int local = kvCacheBuffer.getKVLocalIdx(
    tokenKVIdx, headIdx, dimsPerHead, channelIdx);
TCache value = block[local];
```

Internally, `getKBlockPtr()` resolves a paged address like this:

```cpp
row = data + seqIdx * maxBlocksPerSeq * 2 + K_IDX * maxBlocksPerSeq;
page = row[tokenIdx / tokensPerBlock];
pool = page.isPrimary() ? mPrimaryPoolPtr : mSecondaryPoolPtr;
block = pool + page.get() * mBytesPerBlock;
```

`KVCacheIndex` is an `int32` wrapper. Its high bit marks the secondary pool; the
remaining bits are the physical block index inside that pool.

```cpp
page.isPrimary()
page.get()
```

Use these helpers instead of treating the block table as plain block IDs.

## MLA Convention

For standard attention, K and V use `getKBlockPtr()` and `getVBlockPtr()`. For
MLA compressed KV, the cached latent vector is stored and loaded through the K
side:

```cpp
kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache)
```

Existing MLA examples:

- `cpp/tensorrt_llm/kernels/mlaKernels.cu` reads cached compressed KV with
  `getKBlockPtr()` in `loadPagedKVCacheForMLAKernel`.
- `cpp/tensorrt_llm/kernels/mlaKernels.cu` writes appended latent cache with
  `getKBlockPtr()` in `applyMLARopeAppendPagedKVAssignQKernel`.

## Minimal Fused Launcher Pattern

In a new THOP/C++ wrapper, accept the same tensors Python already has:

```cpp
torch::Tensor const& kv_cache_block_offsets,
torch::Tensor const& host_kv_cache_pool_pointers,
torch::Tensor const& host_kv_cache_pool_mapping,
int64_t layer_idx,
int64_t tokens_per_block,
int64_t attention_window_size,
int64_t quant_mode
```

Then build the cache view:

```cpp
auto kvCache = tensorrt_llm::torch_ext::buildPagedKvCacheBuffers(
    std::optional(kv_cache_block_offsets),
    std::optional(host_kv_cache_pool_pointers),
    std::optional(host_kv_cache_pool_mapping),
    kvCacheQuantMode,
    layer_idx,
    batch_beam,
    tokens_per_block,
    1,      // kv_head_num for MLA
    576,    // kv_lora_rank + qk_rope_head_dim for GLM-5
    attention_window_size,
    attention_window_size,
    0,      // sink_token_length
    1,      // beam_width
    seq_offset,
    true,   // is_mla_enable
    elem_size).kvCacheBuffer;
```

Pass `kvCache` into your CUDA kernel and use `getKBlockPtr()` plus
`getKVLocalIdx()` for all cached latent KV reads/writes.

This preserves compatibility with:

- paged cache allocation,
- block reuse,
- secondary/host-offloaded blocks,
- layer-to-pool mapping,
- context/generation phase slicing,
- CUDA graph padding.
