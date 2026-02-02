"""
Paged Attention Golden Implementation

Implements the online softmax algorithm for paged attention computation.
Used as reference for validation of the simulation results.
"""

import struct
import numpy as np

# Output tensor names
__outputs__ = ["out"]

# Tensor order matching orchestration function parameter order
# Args layout: [7 ptrs, 7 sizes, count] = 15 args
TENSOR_ORDER = ["query", "key_cache", "value_cache", "block_table", "context_lens", "out", "config"]

# Comparison tolerances
RTOL = 1e-3
ATOL = 1e-3

# Test parameter sets
PARAMS_LIST = [
    {
        "batch": 2,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 64,
        "block_num": 4,
        "context_len": 256,
    },
]


def generate_inputs(params: dict) -> dict:
    """
    Generate input and output tensors for paged attention.

    Args:
        params: Parameter dictionary from PARAMS_LIST

    Returns:
        Dictionary containing all tensors (inputs + outputs)
    """
    batch = params.get("batch", 2)
    num_heads = params.get("num_heads", 16)
    kv_head_num = params.get("kv_head_num", 1)
    head_dim = params.get("head_dim", 128)
    block_size = params.get("block_size", 64)
    block_num = params.get("block_num", 4)
    context_len = params.get("context_len", 256)
    seed = params.get("seed", 42)

    np.random.seed(seed)

    total_blocks = batch * block_num
    scale_value = 1.0 / np.sqrt(head_dim)

    # Convert scale_value float to uint32 bit representation
    scale_bits = struct.unpack('I', struct.pack('f', scale_value))[0]

    # Block table: sequential mapping
    block_table = np.zeros(batch * block_num, dtype=np.int32)
    for b in range(batch):
        for bn in range(block_num):
            block_table[b * block_num + bn] = b * block_num + bn

    # Context lengths
    context_lens = np.full(batch, context_len, dtype=np.int32)

    # Config tensor: scalar parameters for orchestration
    # [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits]
    config = np.array(
        [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits],
        dtype=np.int64,
    )

    return {
        "query": np.random.randn(batch * num_heads * head_dim).astype(np.float32) * 0.1,
        "key_cache": np.random.randn(total_blocks * block_size * kv_head_num * head_dim).astype(np.float32) * 0.1,
        "value_cache": np.random.randn(total_blocks * block_size * kv_head_num * head_dim).astype(np.float32) * 0.1,
        "block_table": block_table,
        "context_lens": context_lens,
        "out": np.zeros(batch * num_heads * head_dim, dtype=np.float32),
        "config": config,
    }


def compute_golden(tensors: dict, params: dict) -> None:
    """
    Compute expected paged attention output in-place.

    Uses online softmax algorithm to incrementally compute attention
    across KV cache blocks.

    Args:
        tensors: Dictionary containing all tensors (inputs and outputs)
        params: Parameter dictionary
    """
    batch = params.get("batch", 2)
    num_heads = params.get("num_heads", 16)
    kv_head_num = params.get("kv_head_num", 1)
    head_dim = params.get("head_dim", 128)
    block_size = params.get("block_size", 64)
    block_num = params.get("block_num", 4)

    heads_per_kv = num_heads // kv_head_num
    scale_value = 1.0 / np.sqrt(head_dim)

    query = tensors["query"].reshape(batch, num_heads, head_dim)
    key_cache = tensors["key_cache"].reshape(-1, block_size, kv_head_num, head_dim)
    value_cache = tensors["value_cache"].reshape(-1, block_size, kv_head_num, head_dim)
    block_table = tensors["block_table"].reshape(batch, block_num)
    context_lens = tensors["context_lens"]

    out = np.zeros((batch, num_heads, head_dim), dtype=np.float32)

    for b_idx in range(batch):
        cur_seq = context_lens[b_idx]
        bn_this_batch = (cur_seq + block_size - 1) // block_size

        for h_idx in range(num_heads):
            kv_h_idx = h_idx // heads_per_kv
            qi = query[b_idx, h_idx, :]

            mi = -np.inf
            li = 0.0
            oi = np.zeros(head_dim, dtype=np.float32)

            for bn in range(bn_this_batch):
                cur_block_idx = block_table[b_idx, bn]
                kj = key_cache[cur_block_idx, :, kv_h_idx, :]
                vj = value_cache[cur_block_idx, :, kv_h_idx, :]

                valid_tokens = cur_seq - bn * block_size if bn == bn_this_batch - 1 else block_size

                sij = qi @ kj.T
                sij_scale = sij * scale_value

                if valid_tokens < block_size:
                    sij_scale[valid_tokens:] = -np.inf

                mij = np.max(sij_scale)
                pij = np.exp(sij_scale - mij)
                lij = np.sum(pij)
                oi_new = pij @ vj

                if bn == 0:
                    mi, li, oi = mij, lij, oi_new
                else:
                    mi_new = max(mi, mij)
                    alpha = np.exp(mi - mi_new)
                    beta = np.exp(mij - mi_new)
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new

            oi = oi / li
            out[b_idx, h_idx, :] = oi

    tensors["out"][:] = out.flatten()


# For standalone testing
if __name__ == "__main__":
    params = PARAMS_LIST[0]
    tensors = generate_inputs(params)
    
    print("=== Paged Attention Golden Test ===")
    print(f"batch={params['batch']}, num_heads={params['num_heads']}, head_dim={params['head_dim']}")
    print(f"block_size={params['block_size']}, block_num={params['block_num']}")
    
    compute_golden(tensors, params)
    
    out = tensors["out"].reshape(params["batch"], params["num_heads"], params["head_dim"])
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}")
    print("Golden test passed!")
