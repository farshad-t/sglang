import torch
import time
import math
# Use the common_ops kernel as requested
import sgl_kernel

def benchmark_decode_attention():
    """
    Measures the performance of the sgl-kernel decode_attention_cpu function.
    """
    print("--- Preparing CPU Decode Attention Benchmark ---")

    # --- 1. Define Benchmark Parameters ---
    # These parameters simulate a batch of requests during the decoding phase.
    batch_size = 32
    num_heads = 40          # Number of query heads
    num_kv_heads = 8        # Number of key/value heads (for Grouped Query Attention)
    head_dim = 128
    max_seq_len_in_batch = 1024 # The longest sequence in the current batch
    dtype = torch.float32
    device = "cpu"
    
    # Parameters for the new function signature
    sm_scale = 1.0 / math.sqrt(head_dim)
    logit_cap = 0.0  # 0.0 means no cap

    # --- 2. Create Dummy Input Tensors on the CPU ---
    # In decode, the query is always for a single new token.
    q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)
    o = torch.empty_like(q)

    # Simulate the KV cache data structures used by SGLang
    max_total_len = batch_size * max_seq_len_in_batch
    k_buffer = torch.randn((max_total_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v_buffer = torch.randn((max_total_len, num_kv_heads, head_dim), dtype=dtype, device=device)

    # Create additional tensors required by the new function signature
    key = torch.empty((batch_size, max_seq_len_in_batch, num_kv_heads, head_dim), dtype=dtype, device=device)
    value = torch.empty((batch_size, max_seq_len_in_batch, num_kv_heads, head_dim), dtype=dtype, device=device)
    loc = torch.arange(0, max_total_len, dtype=torch.int32, device=device).view(batch_size, max_seq_len_in_batch)
    attn_logits = torch.empty((batch_size, num_heads, max_seq_len_in_batch), dtype=dtype, device=device)
    
    # For this benchmark, we assume all sequences have the same length
    kv_indptr = torch.arange(0, (batch_size + 1) * max_seq_len_in_batch, max_seq_len_in_batch, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, batch_size * max_seq_len_in_batch, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch_size,), max_seq_len_in_batch, dtype=torch.int32, device=device)


    print(f"Parameters: Batch Size={batch_size}, Heads={num_heads}, KV Heads={num_kv_heads}, Seq Len={max_seq_len_in_batch}\n")

    # --- 3. Run Warmup Iterations ---
    # This helps to get more stable measurements by warming up system caches.
    num_warmup = 10
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        sgl_kernel.common_ops.decode_attention_cpu(
            q, k_buffer, v_buffer, o,
            key, value, loc, attn_logits,
            kv_indptr, kv_indices, seq_lens,
            sm_scale, logit_cap
        )

    # --- 4. Run the Benchmark ---
    num_runs = 50
    print(f"Running benchmark for {num_runs} iterations...")
    
    start_time = time.perf_counter()

    for _ in range(num_runs):
        sgl_kernel.common_ops.decode_attention_cpu(
            q, k_buffer, v_buffer, o,
            key, value, loc, attn_logits,
            kv_indptr, kv_indices, seq_lens,
            sm_scale, logit_cap
        )

    end_time = time.perf_counter()

    # --- 5. Report the Results ---
    elapsed_time_ms = (end_time - start_time) * 1000
    avg_time_ms = elapsed_time_ms / num_runs

    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_runs} runs: {elapsed_time_ms:.3f} ms")
    print(f"Average execution time: {avg_time_ms:.3f} ms per call")

if __name__ == "__main__":
    benchmark_decode_attention()

