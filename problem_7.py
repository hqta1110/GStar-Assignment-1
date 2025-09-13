import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # convert to base-2 exponent domain for exp2 use
    qk_scale = softmax_scale * 1.4426950408889634  # log2(e)

    # --- STUDENT IMPLEMENTATION: Combine GQA, SWA, and Sink logic ---
    # Calculate sliding window bounds; ensure we don't re-process sink tokens.
    window_start = max(SINK_SIZE, q_block_idx * BLOCK_M - WINDOW_SIZE + 1)
    
    # --- Phase 0: Sink blocks (always attend to first SINK_SIZE tokens) ---
    sink_blocks = triton.cdiv(SINK_SIZE, BLOCK_N)
    for sink_block_idx in range(sink_blocks):
        start_n = sink_block_idx * BLOCK_N
        k_offsets = start_n + tl.arange(0, BLOCK_N)
            
        # Load K as [HEAD_DIM, BLOCK_N]
        k_ptrs = (
            K_ptr
            + batch_idx * k_stride_b
            + kv_head_idx * k_stride_h
            + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        )
        # Load V as [BLOCK_N, HEAD_DIM]
        v_ptrs = (
            V_ptr
            + batch_idx * v_stride_b
            + kv_head_idx * v_stride_h
            + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        )

        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
        scores = tl.dot(q_block, k_block) * qk_scale
        
        # For sink tokens, only apply causal mask and sink size limit
        sink_mask = k_offsets[None, :] < SINK_SIZE
        causal_mask = (q_offsets[:, None] >= k_offsets[None, :])
        scores = tl.where(sink_mask & causal_mask, scores, -1e20)
        
        s_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, s_max)
        
        exp_scale = tl.exp2(m_i - m_new)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        prob = tl.exp2(scores - m_new[:, None])
        acc += tl.dot(prob, v_block)
        l_i += tl.sum(prob, axis=1)
        
        m_i = m_new
    first_block = (window_start // BLOCK_N) * BLOCK_N
    # --- Phase 1: Off-Diagonal Blocks (within the sliding window) ---
    for start_n in range(first_block, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        
        # Load K as [HEAD_DIM, BLOCK_N]
        k_ptrs = (
            K_ptr
            + batch_idx * k_stride_b
            + kv_head_idx * k_stride_h
            + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        )
        # Load V as [BLOCK_N, HEAD_DIM]
        v_ptrs = (
            V_ptr
            + batch_idx * v_stride_b
            + kv_head_idx * v_stride_h
            + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        )

        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
        scores = tl.dot(q_block, k_block) * qk_scale
        
        # Apply sliding window mask, causal mask, and exclude sink tokens (already handled)
        window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        causal_mask = (q_offsets[:, None] >= k_offsets[None, :])
        non_sink_mask = k_offsets[None, :] >= SINK_SIZE
        scores = tl.where(window_mask & causal_mask & non_sink_mask, scores, -1e20)
        
        s_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, s_max)
        
        exp_scale = tl.exp2(m_i - m_new)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        prob = tl.exp2(scores - m_new[:, None])
        acc += tl.dot(prob, v_block)
        l_i += tl.sum(prob, axis=1)
        
        m_i = m_new

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        
        # Load K as [HEAD_DIM, BLOCK_N]
        k_ptrs = (
            K_ptr
            + batch_idx * k_stride_b
            + kv_head_idx * k_stride_h
            + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        )
        # Load V as [BLOCK_N, HEAD_DIM]
        v_ptrs = (
            V_ptr
            + batch_idx * v_stride_b
            + kv_head_idx * v_stride_h
            + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        )

        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
        scores = tl.dot(q_block, k_block) * qk_scale
        
        # Apply sliding window mask, causal mask, and exclude sink tokens
        window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        causal_mask = (q_offsets[:, None] >= k_offsets[None, :])
        non_sink_mask = k_offsets[None, :] >= SINK_SIZE
        scores = tl.where(window_mask & causal_mask & non_sink_mask, scores, -1e20)
        
        s_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, s_max)
        
        exp_scale = tl.exp2(m_i - m_new)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        prob = tl.exp2(scores - m_new[:, None])
        acc += tl.dot(prob, v_block)
        l_i += tl.sum(prob, axis=1)
        
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
