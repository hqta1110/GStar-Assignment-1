# FlashAttention Assignment Report

This report documents the implementation of FlashAttention and its variants across seven problems. Later problems build upon earlier work, reusing the same online softmax and blockwise strategy, while introducing new masking or memory management techniques.

---

## Common Pattern

All problems (3–7) follow the same block-streaming and online-softmax calculation. The main differences are in how indices, offsets, and masks are handled.

1. **Calculate per-program indices**

   * `q_block_idx = tl.program_id(0)` picks the query block (row tile).
   * `batch_head_idx = tl.program_id(1)` encodes batch and head.
   * From this, calculate `batch_idx`, `q_head_idx`, and (from Problem 5 onwards) `kv_head_idx`.

2. **Initialize values**

   * Initialize `m_i`, `l_i`, `acc` for the online softmax.

3. **Load Q block**

   * `q_offsets = q_block_idx * BLOCK_M + arange(BLOCK_M)`.
   * Build `q_ptrs` from `batch_idx`, `q_head_idx` (or `head_idx`), and `q_offsets`.
   * Load `q_block` with masking against sequence length.

4. **Stream K/V blocks**

   * For each `start_n`, compute `k_offsets = start_n + arange(BLOCK_N)`.
   * Build `k_ptrs` / `v_ptrs` with `batch_idx` and either `head_idx` or `kv_head_idx`.
   * Load `k_block`, `v_block`.

5. **Compute scores and mask**

   * `scores = dot(q_block, k_block) * scale`.
   * Construct `valid_mask` (causal, window, sink, sequence end).
   * Apply via `scores = where(valid_mask, scores, -INF)`.

6. **Online softmax update**

   * `s_max = max(scores)` → `m_new = max(m_i, s_max)`.
   * Rescale: `acc *= exp2(m_i - m_new)`, `l_i *= exp2(m_i - m_new)`.
   * `prob = exp2(scores - m_new[:,None])` masked.
   * Update: `acc += dot(prob, v_block)` ; `l_i += sum(prob)` ; `m_i = m_new`.

7. **Finalize**

   * Normalize `acc /= (l_i[:,None] + eps)`.
   * Store back to `O` using `q_offsets` and `q_head_idx` (or `head_idx`).

**Key note:** the **online-softmax core never changes** - only **pointer arithmetic** (which K/V blocks are loaded) and **masking rules** change between problems.

---

## Technical Implementation

### A. Head mapping (GQA - Problems 5, 6, 7)

* **Problem 3:** use `head_idx` directly for K/V.
* **Problem 5:** introduce GQA. Compute
  `q_per_kv_heads = N_Q_HEADS // N_KV_HEADS`
  `kv_head_idx = q_head_idx // q_per_kv_heads`
  Use `kv_head_idx` for K/V pointers, while Q and O still use `q_head_idx`.
* **Problems 6 & 7:** same GQA mapping, reused.

---

### B. Offsets & pointer arithmetic (all problems)

* `q_offsets = q_block_idx * BLOCK_M + arange(BLOCK_M)`
* `k_offsets = start_n + arange(BLOCK_N)`
* `q_ptrs` and `k_ptrs/v_ptrs` use batch/head stride plus offsets.

**Notes**

* All kernels mask out-of-range loads with `mask=` in `tl.load`.
* By adjusting only `start_n` ranges and head indices, every variant is implemented without reshaping tensors.

---

### C. Masking strategies (Problems 4-7)

* **Problem 3:** no mask.
* **Problem 4 (causal):**
  In diagonal blocks, apply `causal_mask = (q_offsets >= k_offsets)` combined with sequence mask. Kernel split into off-diagonal (no mask) + diagonal (causal mask).
* **Problem 5 (GQA + causal):** reuse Problem 4 causal mask, but with `kv_head_idx` for K/V.
* **Problem 6 (SWA + GQA):**
  Add windowing:
  `window_mask = (q_offsets - k_offsets) < WINDOW_SIZE`
  `causal_mask = (q_offsets >= k_offsets)`
  Combined with sequence mask. Enforces locality + causality.
* **Problem 7 (Sink + SWA + GQA):**
  Adds a **sink phase** before window/diagonal:

  1. **Phase 0 (sink):** first `SINK_SIZE` tokens processed separately with `sink_mask = (k_offsets < SINK_SIZE)` ∧ causal. Guarantees global visibility of sinks.
  2. **Phase 1 (window):** sliding window `[window_start, q_block_idx * BLOCK_M)` with `window_mask ∧ causal_mask ∧ non_sink_mask`.
  3. **Phase 2 (diagonal):** same masks applied on the current block.
     Sink-first ordering ensures all queries include sink tokens early in the online softmax.

---

### D. Window / phase selection (Problems 6 & 7)

* **Problem 6 (SWA):**
  Compute `window_start = max(0, q_block_idx - WINDOW_SIZE + 1)`.
  Loop from `window_start` up to (not including) the current query block.
  Then process the diagonal separately. Both loops use elementwise `window_mask`.
* **Problem 7 (SWA + Sink):**
  Compute `window_start = max(SINK_SIZE, q_block_idx * BLOCK_M - WINDOW_SIZE + 1)` so sink tokens are excluded from the window phase.
  Then run phases in order: **sink → window → diagonal**.

---

## Key Takeaways

1. **Pointer arithmetic is the main lever.** Changing `k_ptrs`/`v_ptrs` (head index and offsets) implements new variants without tensor reshaping.
2. **Mask-by-clamping is low-cost and composable.** Boolean masks with `where(..., -INF)` allow reusing the same online softmax logic.
3. **GQA is trivial to add.** A single mapping from `q_head_idx` to `kv_head_idx` suffices.
4. **Sink-first ordering is the true novelty of Problem 7.** It ensures global tokens are always included early in accumulation, stabilizing the online softmax.

---
