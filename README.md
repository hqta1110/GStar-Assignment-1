# FlashAttention Assignment Report

This report documents the implementation of FlashAttention and its variants across seven problems. Later problem builds upon earlier work, reusing the same online softmax and blockwise strategy, while introducing new masking or memory management techniques.

## Common Pattern

All problem follow the same block-streaming and online-softmax calculation. The main difference between each problem is the way I handle indices, offsets and masks

1. **Calculate per-program inddices**
    * `q_block_idx = tl.program_id(0)` picks the query block (row tile).
    * `batch_head_idx = tl.program_id(1)` encodes batch and head; use this to calculate `batch_idx`, `q_head_idx`, `kv_head_idx` for problem_5 onwards.
2. **Initialize value**
    * Init the value for m_i, l_i, acc
3. **Load Q block**
    * `q_offsets = q_block_idx * BLOCK_M + arange(BLOCK_M)`.
    * Build `q_ptrs` using `batch_idx`, `q_head_idx` (or `head_idx` for problem 5 onwards), and `q_offsets` -> `tl.load(q_ptrs, mask=q_offsets < SEQ_LEN)` to get `q_block`
4. **Stream K/V block**
    * For each `start_n` (block start) compute `k_offsets = start_n + arange(BLOCK_N)`.
    * Build `k_ptrs` and `v_ptrs` with `batch_idx`, `head_idx` and `k_offsets` for K/V. Load `k_block` and `v_block`. (Head index for K/V may be `head_idx` or a mapped `kv_head_idx` depending on problem).
5. **Compute scores and mask**
   * `scores = dot(q_block, k_block) * scale`.
   * Construct a `valid_mask` (depends on causal check, window, sink, and sequence end).
   * Apply mask via `scores = where(valid_mask, scores, -inf)` (i.e., clamp invalid positions to a large negative). This keeps the online-softmax update identical across variants.

6. **Online softmax update**
   * `s_max = max(scores, axis=1)` -> `m_new = max(m_i, s_max)`
   * Rescale accumulator: `acc *= e^(m_i - m_new)`, `l_i *= e^(m_i - m_new)` (implemented with `tl.exp2`)
   * `prob = e^(scores - m_new[:,None])` masked appropriately.
   * `acc += dot(prob, v_block)` ; `l_i += sum(prob, axis=1)` ; `m_i = m_new`.
     This online formulation lets you stream K/V blocks without storing full score matrices. (See Problem 3).&#x20;

7. **Finalize**
   * `acc /= (l_i[:,None] + eps)` and `store` back to output `O` via pointer built from `q_offsets` and `q_head_idx` (or `head_idx`).

**Key note:** the **online-softmax code never changes** — only the **pointer arithmetic** (which K/V blocks to load) and the **mask** (which scores are allowed) change between problems.

## Technical implementation

Below I show how solution is different between problems.

### A. Head mapping (Grouped-Query Attention - Problems 5, 6, 7)

**What changes:** which head index is used when forming `k_ptrs` and `v_ptrs`.

* **Baseline (Problem 3):** use `head_idx` for K/V pointers (one K/V per Q head)

* **GQA (Problem 5):** compute `q_per_kv_heads = N_Q_HEADS // N_KV_HEADS` then `kv_head_idx = q_head_idx // q_per_kv_heads`. Use `kv_head_idx` in `k_ptrs`/`v_ptrs`. This is the only change required to reuse K/V across groups - everything else (including Q loads and O stores) keeps using the original `q_head_idx`. The code that implements this mapping sits at the top of the kernel -> all of the other parts remain unchanged compare to problem_4

* **SWA & Sink variants (Problems 6 & 7):** reuse the same mapping (`kv_head_idx`) and feed it to pointer arithmetic for K/V loads.
---

### B. Offsets & pointer arithmetic (all problems)

**Common recipe:**

* `q_offsets = q_block_idx * BLOCK_M + arange(BLOCK_M)`
* `k_offsets = start_n + arange(BLOCK_N)`
* `q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:,None] * q_stride_s + arange(HEAD_DIM)[None,:])`
* `k_ptrs` / `v_ptrs` similar but with head index for K/V (either `head_idx` or `kv_head_idx`), and with `k_offsets` instead of `q_offsets`.

**Notes**

* All implementations use `mask=` in `tl.load` to avoid out-of-range reads (e.g., `mask=k_offsets < SEQ_LEN`).
* The pointer arithmetic is the unifying low-level trick: by controlling `start_n` ranges and the head index used, every attention variant is implemented without copying or reshaping K/V in memory. This is visible in all student kernels.
---

### C. Masking strategies (Problems 4, 5, 6, 7)

**Core approach (all masked variants):** build a boolean `valid_mask` and apply `scores = where(valid_mask, scores, -INF)`. This lets the same online-softmax update run unchanged.

* **Non-masked baseline (Problem 3):** no mask; all `k_offsets` are valid until sequence-end.

* **Causal (Problem 4):** In diagonal blocks, build `causal_mask = (q_offsets[:,None] >= k_offsets[None,:])`. Combine with `valid_cols = (k_offsets < SEQ_LEN)` to produce `valid_mask = causal_mask & valid_cols[None,:]`. Apply with `where`. This prevents “future” keys from contributing when query index <= key index. Problem 4’s kernel has a two-phase loop: off-diagonal (no mask) and diagonal (apply causal mask).

* **GQA + causal (Problem 5):** reuses the causal-mask recipe from Problem 4 but uses `kv_head_idx` for K/V pointer computation; the masking logic for diagonal tiles is the same as Problem 4.

* **Sliding window (Problem 6):** combine a **window mask** with causal condition:

  * `window_mask = (q_offsets[:,None] - k_offsets[None,:]) < WINDOW_SIZE`
  * `causal_mask = (q_offsets[:,None] >= k_offsets[None,:])`
  * `valid_mask = window_mask & causal_mask & (k_offsets < SEQ_LEN)`
    This enforces both locality (only recent keys within `WINDOW_SIZE`) and causality — implemented for both off-diagonal (windowed range) and diagonal blocks. The code computes a `window_start` and *only iterates key blocks within the window*, further reducing loads.

* **Sink + SWA + GQA (Problem 7):** three-phase approach:

  1. **Sink phase (phase 0):** first `SINK_SIZE` tokens are processed separately. For sink blocks use `sink_mask = k_offsets < SINK_SIZE` combined with causal mask; these tokens are globally visible to all queries. Processing sinks first ensures they are in the running accumulator early.
  2. **Window phase (phase 1):** only process key blocks in the sliding window `[window_start, q_block_idx * BLOCK_M)`. Apply `window_mask & causal_mask & non_sink_mask` where `non_sink_mask` excludes sink keys in this phase.
  3. **Diagonal phase (phase 2):** process the diagonal block(s) with the same triple-mask and finalize.

---

### D. Window / phase selection (Problems 6 & 7)

**Window start computation:** two equivalent responsibilities:

* Determine which key blocks to iterate (avoid scanning entire history),
* Ensure sink blocks are only processed in the sink phase (Problem 7).

**Problem 6 (SWA)**: the code computes a `window_start` (`tl.maximum(0, q_block_idx - WINDOW_SIZE + 1)`) and iterates `start_n` from `window_start` to just before the query block. Also applies `window_mask` on the per-element level.

**Problem 7 (SWA + Sink)**: computes `window_start = max(SINK_SIZE, q_block_idx * BLOCK_M - WINDOW_SIZE + 1)` so sink tokens (first `SINK_SIZE`) are not reprocessed in the window phase. 


## 4. Key takeaways

1. **Pointer arithmetic is the main lever.** Changing `k_ptrs`/`v_ptrs` (head index and `k_offsets`) implements different attention patterns without layout change.
2. **Mask-by-clamping is low-cost and composable.** Use boolean masks combined with `tl.where(..., -INF)` so the same online-softmax update is reused for all topologies.
3. **GQA is trivial to add.** Compute `kv_head_idx = q_head_idx // (N_Q_HEADS // N_KV_HEADS)` and use it for K/V loads — no tensor reshaping required.


