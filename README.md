# FlashAttention Assignment Report

This report documents the implementation of FlashAttention and its variants across seven problems. Each problem builds upon earlier work, reusing the same online softmax and blockwise strategy, while introducing new masking or memory management techniques.

---

## Problem 1 — PyTorch FlashAttention (Reference Implementation)

- Implemented the **forward pass** of FlashAttention in pure PyTorch.  
- Queries (`Q`), Keys (`K`), and Values (`V`) are divided into **blocks**.  
- For each query block:
  - Stream over all key blocks.  
  - Compute attention scores `S_ij = Q_block @ K_block^T / sqrt(d)`.  
  - Apply masking if required:  
    - **Causal mask** ensures tokens cannot attend to future tokens.  
    - **Bounds mask** handles sequences shorter than the block size.  
  - Update accumulators:
    - `m_i`: running maximum of logits (for numerical stability).  
    - `l_i`: running sum of exponentials.  
    - `acc`: running weighted sum of values.  
- Final output `O` is obtained as `acc / l_i`.  
- This served as the **reference baseline** for later Triton kernels.

---

## Problem 2 — Introduction to Triton

- Learned Triton’s programming model:  
  - **Block-level parallelism**: each kernel instance processes a query block.  
  - **Pointer arithmetic**: construct memory pointers with tensor strides (`stride_b`, `stride_h`, `stride_s`).  
  - **Vectorized loading**: use `tl.load` with masks to safely fetch data into SRAM.  
- Implemented simple examples to practice **offset calculation** and **masking**.  
- No attention logic yet, but built the foundation for memory-efficient kernels in later problems.

---

## Problem 3 — FlashAttention in Triton (Non-Causal)

- First full Triton implementation of FlashAttention forward pass.  
- Used the **same online softmax method** as in Problem 1 (`m_i`, `l_i`, `acc`).  
- Key elements:
  - **Pointer construction**: queries, keys, and values are loaded into SRAM via block offsets.  
  - **Bounds masks**: prevent out-of-bounds reads when sequence length is not divisible by block size.  
  - **Accumulation loop**: iterate over key blocks, compute `Q_block @ K_block^T`, rescale accumulators, and update with new contributions.  
- Produces correct outputs for the non-causal case, establishing the Triton baseline.

---

## Problem 4 — FlashAttention with Causal Masking

- Extended Problem 3 to support **causal masking**.  
- Introduced a **two-phase strategy**:
  1. **Off-diagonal blocks** (where all keys precede the current query block): no masking needed.  
  2. **Diagonal block** (where queries and keys overlap): apply per-element causal mask `q_idx >= k_idx`.  
- This optimization avoids unnecessary masking in most blocks and improves efficiency.  
- Other logic (online softmax, pointer arithmetic, accumulation) reused from Problem 3.

---

## Problem 5 — Grouped-Query Attention (GQA)

- Built on the kernel from Problems 3–4.  
- Implemented **head-sharing** between queries and fewer key/value heads:  
  - Mapping: `kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)`.  
- Key modifications:
  - **Pointer arithmetic** for K and V changed to use `kv_head_idx` instead of `q_head_idx`.  
  - Q still uses its original head index.  
- This reduces the memory and compute cost of storing/processing K and V, while keeping the rest of the kernel unchanged.  
- Masking and online softmax logic are identical to earlier problems.

---

## Problem 6 — Sliding Window Attention (SWA)

- Extended FlashAttention to support **local attention** with a fixed window size.  
- Modifications compared to Problem 3–5:
  - **Restricted iteration range**: for each query block, only process key blocks within the sliding window.  
  - **Window mask**: ensure only tokens inside the window contribute to attention scores.  
- Combined naturally with causal masking when needed.  
- This shows how the blockwise pointer/masking framework can be adapted for locality constraints.

---

## Problem 7 — Attention Sinks (with GQA + SWA)

- Final problem combined **GQA**, **sliding window attention**, and **attention sinks**.  
- Design divided into multiple phases:
  1. **Sink phase**: process initial sink tokens (globally visible to all queries).  
  2. **Windowed off-diagonal phase**: process non-sink tokens within the sliding window.  
  3. **Diagonal phase**: handle overlapping query/key blocks, applying both causal and window masks, and excluding sink tokens with a **non-sink mask**.  
- Reused:
  - GQA head mapping from Problem 5.  
  - Sliding window iteration and masks from Problem 6.  
- The main new addition is the **sink mask** and handling sinks as a separate phase.  
- This problem demonstrated how multiple advanced constraints can be layered on top of the same core FlashAttention framework.

---

## Summary

- **Problem 1** provided a PyTorch baseline for FlashAttention.  
- **Problem 2** introduced Triton basics.  
- **Problem 3** implemented FlashAttention in Triton (non-causal).  
- **Problem 4** added causal masking with a two-phase diagonal/off-diagonal approach.  
- **Problem 5** introduced Grouped-Query Attention by modifying key/value head pointers.  
- **Problem 6** extended the kernel with sliding window constraints.  
- **Problem 7** integrated GQA, SWA, and sink tokens into a single unified kernel.  

Across all problems, the **core design** of blockwise pointer arithmetic, accumulator rescaling, and online softmax remained consistent, while masking and pointer mapping strategies were adapted to introduce each new variant.
