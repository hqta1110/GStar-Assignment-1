# FlashAttention Assignment Report

This report documents the implementation of FlashAttention and its variants across seven problems. Each problem builds upon earlier work, reusing the same online softmax and blockwise strategy, while introducing new masking or memory management techniques.

---

## Problem 1 — PyTorch FlashAttention (Reference Implementation)

- Implemented the **forward pass** of FlashAttention in pure PyTorch.  
- Queries (`Q`), Keys (`K`), and Values (`V`) are divided into **blocks**.  
- For each query block:
  - Stream over all key blocks.  
  - Compute attention scores:

    ```python
    scores = (Q_block @ K_block.T) / math.sqrt(d)
    ```

  - Apply masking:
    - **Causal mask**:  
      ```python
      scores = scores.masked_fill(q_idx[:, None] < k_idx[None, :], float('-inf'))
      ```
    - **Bounds mask** for short sequences.  
  - Update accumulators (`m_i`, `l_i`, `acc`) using the online softmax method.  

- Final output:  
  ```python
  O = acc / l_i[:, None]
  ```
* Serves as the **reference baseline** for all Triton kernels.

---

## Problem 2 — Introduction to Triton

* Learned **Triton’s programming model**: block-level execution, pointers, and masks.

* Practiced **pointer arithmetic** for batched tensors:

  ```python
  offsets = pid * BLOCK + tl.arange(0, BLOCK)
  ptrs = base_ptr + offsets * stride
  data = tl.load(ptrs, mask=offsets < n_elements, other=0.0)
  ```

* No attention yet, but established how to work with **offsets, masks, and memory loading** in Triton.

---

## Problem 3 — FlashAttention in Triton (Non-Causal)

* Implemented FlashAttention forward pass in Triton.

* Core loop over key blocks:

  ```python
  for start_n in range(0, N_CTX, BLOCK_N):
      k_ptrs = K_ptr + ...
      v_ptrs = V_ptr + ...
      K_block = tl.load(k_ptrs, mask=mask_k)
      V_block = tl.load(v_ptrs, mask=mask_v)

      scores = tl.dot(Q_block, K_block)
      ...
      acc = acc * exp_scale[:, None] + tl.dot(p, V_block)
      l_i = l_i * exp_scale + tl.sum(p, axis=1)
  ```

* Key features:

  * **Pointer construction** for Q/K/V using strides.
  * **Bounds masks** applied during `tl.load`.
  * Online softmax rescaling (`m_i`, `l_i`, `acc`) same as Problem 1.

---

## Problem 4 — FlashAttention with Causal Masking

* Extended Problem 3 with **causal masking**.

* Implemented **two-phase strategy**:

  * **Off-diagonal blocks**: no causal check.
  * **Diagonal block**: apply causal mask:

    ```python
    causal_mask = q_offsets[:, None] >= k_offsets[None, :]
    scores = tl.where(causal_mask, scores, float('-inf'))
    ```

* Reduced branching overhead while maintaining correctness.

* Pointer arithmetic and online softmax are unchanged from Problem 3.

---

## Problem 5 — Grouped-Query Attention (GQA)

* Built on Problems 3–4.

* Introduced **head sharing** for K/V:

  ```python
  q_per_kv = n_q_heads // n_kv_heads
  kv_head_idx = q_head_idx // q_per_kv
  ```

* Modified **pointer arithmetic** for K/V:

  ```python
  k_ptrs = K_ptr + batch * k_stride_b + kv_head_idx * k_stride_h + ...
  v_ptrs = V_ptr + batch * v_stride_b + kv_head_idx * v_stride_h + ...
  ```

* Only K/V head indexing changed.

* Online softmax and masking identical to earlier problems.

---

## Problem 6 — Sliding Window Attention (SWA)

* Extended to **local attention** with a fixed window.

* Adjusted **iteration range** of key blocks:

  ```python
  window_start = max(0, q_block_start - WINDOW_SIZE)
  for start_n in range(window_start, q_block_start + 1, BLOCK_N):
      ...
  ```

* Added **window mask**:

  ```python
  window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
  scores = tl.where(window_mask, scores, float('-inf'))
  ```

* Works with causal masking and GQA.

* Rest of kernel logic unchanged.

---

## Problem 7 — Attention Sinks (with GQA + SWA)

* Combined **GQA**, **SWA**, and **attention sinks**.

* Multi-phase design:

  1. **Sink phase**: process sink tokens (`k_offsets < SINK_SIZE`).
  2. **Windowed off-diagonal phase**: process non-sink tokens inside the window.
  3. **Diagonal phase**: handle overlap, exclude sinks with `non_sink_mask`.

* Example masking:

  ```python
  sink_mask = k_offsets < SINK_SIZE
  non_sink_mask = k_offsets >= SINK_SIZE
  combined_mask = window_mask & causal_mask & non_sink_mask
  scores = tl.where(combined_mask, scores, float('-inf'))
  ```

* Reuses:

  * GQA mapping from Problem 5.
  * Sliding window logic from Problem 6.

* Main addition: **sink mask + separate sink phase**.

---

## Summary

* **Problem 1**: PyTorch baseline implementation.
* **Problem 2**: Triton basics (offsets, pointers, masks).
* **Problem 3**: Triton FlashAttention (non-causal).
* **Problem 4**: Added causal masking with two-phase design.
* **Problem 5**: Introduced GQA (pointer remapping for K/V).
* **Problem 6**: Implemented sliding window attention.
* **Problem 7**: Integrated GQA + SWA + attention sinks.

Across all problems, the **core design** remains consistent:

* Blockwise pointer arithmetic for memory efficiency.
* Online softmax with rescaling for numerical stability.
* Mask composition (bounds, causal, window, sink) applied flexibly.

```
