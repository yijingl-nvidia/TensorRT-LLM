/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * reference: https://github.com/deepseek-ai/FlashMLA
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params
{
    using index_t = int64_t;

    int b;               // Batch size, how many requests/sequences
    int seqlen_q;        // How many tokens to compute queries for on each request/sequence. This is the number of rows
                         // for the query matrix.
    int d;               // Q/K head dimension.
    int d_v;             // V/O head dimension.
    int h;               // Number of heads per token on this kernel.
    int h_h_k_ratio;     // Number of query heads per KV head; `kv_head = query_head / h_h_k_ratio`.
    int ngroups;         // Number of consecutive Q rows that share one causal token position.
    bool is_causal;      // Whether to apply the causal attention mask.
    float scale_softmax; // Multiplicative scale applied to QK scores before softmax.
    float scale_softmax_log2;                    // `scale_softmax` converted to base-2 exponent units.
    int* __restrict__ cu_seqlens_k;              // [b] int32, KV sequence length for each request/sequence.

    void* __restrict__ q_ptr;                    // Q tensor base pointer, logically [b, seqlen_q, h, d].
    void* __restrict__ k_ptr;                    // Paged K cache base pointer.
    void* __restrict__ v_ptr;                    // Paged V cache base pointer.
    void* __restrict__ o_ptr;                    // Output tensor base pointer, logically [b, seqlen_q, h, d_v].
    void* __restrict__ softmax_lse_ptr;          // Final softmax LSE tensor, logically [b, h, seqlen_q].

    float* __restrict__ descale_q_ptr = nullptr; // FP8 Q dequant scale pointer, or null for non-FP8 Q.
    float* __restrict__ descale_k_ptr = nullptr; // FP8 K/V dequant scale pointer, or null for non-FP8 K/V.

    index_t q_batch_stride;                      // Q stride for advancing one index in b, in elements.
    index_t k_batch_stride;                      // Stride between consecutive K cache pages/blocks in elements.
    index_t v_batch_stride;                      // Stride between consecutive V cache pages/blocks in elements.
    index_t o_batch_stride;                      // O stride for advancing one index in b, in elements.
    index_t q_row_stride;                        // Stride between consecutive query rows in Q elements.
    index_t k_row_stride;                        // Stride between consecutive K rows within a cache block in elements.
    index_t v_row_stride;                        // Stride between consecutive V rows within a cache block in elements.
    index_t o_row_stride;                        // Stride between consecutive output rows in O elements.
    index_t q_head_stride;                       // Stride between consecutive query heads in Q elements.
    index_t k_head_stride;                       // Stride between consecutive K heads in the cache in elements.
    index_t v_head_stride;                       // Stride between consecutive V heads in the cache in elements.
    index_t o_head_stride;                       // Stride between consecutive output heads in O elements.

    int* __restrict__ block_table;    // [b, block_table_batch_stride] int32, maps logical KV blocks to cache blocks.
    index_t block_table_batch_stride; // `block_table` stride for advancing one index in b.
    int page_block_size;              // Number of KV tokens per paged cache block.

    // [num_sm_parts, TileSchedulerMetaDataSize=8] int32, KV split scheduler metadata.
    // [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]
    // Each CTA processes a part of the (batch and sequence dim) flattened sequence:
    // - begin_idx: starting request ID
    // - begin_seqlen: starting token on the starting sequence
    // - end_idx: end request ID
    // - end_seqlen: ending token on the ending sequence
    // - begin_n_split_idx: output split idx to write the starting request's partial o to
    int* __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts; // Number of SM partitions used for split-KV scheduling. This number of CTAs split all sequences
                      // connected together (flatten batch and sequence dimension), into small sequence parts for
                      // parallelism (one part per CTA).
    int* __restrict__ num_splits_ptr;        // [b + 1] int32, prefix sum of split counts over b.

    void* __restrict__ softmax_lseaccum_ptr; // Partial softmax LSE scratch tensor for split-KV.
    void* __restrict__ oaccum_ptr;           // Partial output scratch tensor for split-KV.
};

// [batch_begin_idx, begin_seqlen, batch_end_idx, end_seqlen, begin_n_split_idx, _, _, _]
static constexpr int TileSchedulerMetaDataSize = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params& params, cudaStream_t stream);

struct Mla_metadata_params
{
    int* __restrict__ seqlens_k_ptr;
    int* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void get_mla_metadata_func(Mla_metadata_params& params, cudaStream_t stream);
