#!/usr/bin/env python3
"""
Analyze prefill phase expert activation patterns with K-means bucketing - log-only input version.

This script analyzes prefill phase expert activation patterns from SGLang expert distribution recordings
using only a log file as input. It automatically extracts all required metadata from the log file.

Usage:
    python analyze_prefill_from_log.py <log_file>

Example:
    python analyze_prefill_from_log.py bench-throughput-logs/run.log

The script will:
1. Extract .pt file path, batch_size, input_len, output_len from log file
2. Load expert distribution records from the .pt file
3. Separate prefill vs decode phases based on token counts
4. Generate histogram of expert activations for prefill phase
5. Apply K-means clustering to group experts into 5 balanced buckets
6. Save bucketed results with batch_size differentiation
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def parse_log_file(log_file_path: Path) -> Dict:
    """
    Parse log file to extract all required metadata.
    
    Returns dict with keys: pt_file_path, batch_size, input_len, output_len,
                           main_tokens, warmup_tokens
    """
    info = {}
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Extract pt file path
    pt_match = re.search(r'Write expert distribution to (.+\.pt)', content)
    if pt_match:
        pt_path_str = pt_match.group(1).strip()
        pt_path = Path(pt_path_str)
        
        # Make absolute path: pt file path is relative to where benchmark was run
        # Log file structure: <benchmark_dir>/<log_name>.log
        # PT file path in log: <benchmark_dir>/expert_logs/<file>.pt
        # So we resolve relative to log file's parent directory's parent
        if not pt_path.is_absolute():
            # Check if pt path starts with the log directory name
            log_dir_name = log_file_path.parent.name
            if pt_path.parts[0] == log_dir_name:
                # PT path is relative to grandparent of log file
                pt_path = (log_file_path.parent.parent / pt_path_str).resolve()
            else:
                # PT path is relative to parent of log file  
                pt_path = (log_file_path.parent / pt_path_str).resolve()
        
        # If path doesn't exist, try remapping /code/ docker path to host path
        if not pt_path.exists() and '/code/' in pt_path_str:
            host_base = Path('/data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints')
            pt_path = (host_base / pt_path_str.replace('/code/', '', 1)).resolve()
        info['pt_file_path'] = pt_path
    else:
        raise ValueError(f"Could not find .pt file path in log: {log_file_path}")

    # Extract dtype and quantization
    dtype_match = re.search(r"dtype='([^']*)'", content)
    if dtype_match:
        info['dtype'] = dtype_match.group(1)
    
    quantization_match = re.search(r"quantization='([^']*)'", content)
    if quantization_match:
        quant = quantization_match.group(1)
        info['quantization'] = quant if quant else None
    else:
        info['quantization'] = None
    
    # Override dtype based on quantization (w8a8 uses int8 compute)
    if info.get('quantization') and 'w8a8' in info['quantization']:
        info['dtype'] = 'int8'
    
    # --- bench_one_batch fallback: extract from load weight line and decode lines ---
    if 'dtype' not in info:
        # Extract from "Load weight end. ... quant=fp8" or "quant=None"
        quant_match = re.search(r'Load weight end\..*quant=(\w+)', content)
        if quant_match:
            quant_val = quant_match.group(1)
            if quant_val == 'None' or quant_val == 'none':
                info['dtype'] = 'bfloat16'
                info['quantization'] = None
            else:
                info['dtype'] = quant_val
                info['quantization'] = quant_val
        else:
            info['dtype'] = 'bfloat16'
    
    # Extract batch_size and input/output lengths from token counts
    # The log contains "#Input tokens: X #Output tokens: Y" entries
    # First entry is main benchmark, last entry is warmup
    token_matches = list(re.finditer(r'#Input tokens:\s+(\d+)\s+#Output tokens:\s+(\d+)', content))
    
    if len(token_matches) >= 2:
        # Main benchmark is first match
        main_input = int(token_matches[0].group(1))
        main_output = int(token_matches[0].group(2))
        
        # Warmup is last match
        warmup_input = int(token_matches[-1].group(1))
        warmup_output = int(token_matches[-1].group(2))
        
        info['main_tokens'] = {'input': main_input, 'output': main_output}
        info['warmup_tokens'] = {'input': warmup_input, 'output': warmup_output}
        
        # Calculate batch_size, input_len, output_len from main tokens
        # main_input = batch_size * input_len
        # main_output = batch_size * output_len
        # We can extract from filename or try to infer
        
        # Try to extract from filename: ..._input_len-1024_output_len-8192_conc-3-...
        filename_match = re.search(r'input_len-(\d+)_output_len-(\d+)_conc-(\d+)', log_file_path.name)
        if filename_match:
            input_len = int(filename_match.group(1))
            output_len = int(filename_match.group(2))
            batch_size = int(filename_match.group(3))
            
            info['input_len'] = input_len
            info['output_len'] = output_len
            info['batch_size'] = batch_size
        else:
            # Fallback: try batch_size from successful requests
            batch_match = re.search(r'Successful requests:\s+(\d+)', content)
            if batch_match:
                batch_size = int(batch_match.group(1))
                info['batch_size'] = batch_size
                info['input_len'] = main_input // batch_size
                info['output_len'] = main_output // batch_size
    else:
        # Fallback to old method
        batch_match = re.search(r'Successful requests:\s+(\d+)', content)
        if batch_match:
            info['batch_size'] = int(batch_match.group(1))
    
    # --- bench_one_batch fallback: extract batch_size from decode lines ---
    if 'batch_size' not in info:
        # "Decode 0. Batch size: 22, latency: ..."
        decode_bs_match = re.search(r'Decode \d+\. Batch size: (\d+)', content)
        if decode_bs_match:
            info['batch_size'] = int(decode_bs_match.group(1))
    
    # --- bench_one_batch fallback: extract input/output from throughput/latency ---
    if 'input_len' not in info and 'batch_size' in info:
        batch_size = info['batch_size']
        
        # Try from Prefill/Total throughput lines
        prefill_match_full = list(re.finditer(
            r'Prefill\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content))
        total_match = re.findall(
            r'Total\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content)
        
        if prefill_match_full and total_match:
            # Last entries are from the benchmark run (first is warmup)
            last_prefill = prefill_match_full[-1]
            prefill_latency = float(last_prefill.group(1))
            prefill_throughput = float(last_prefill.group(2))
            prefill_tokens = int(round(prefill_throughput * prefill_latency))
            
            total_latency = float(total_match[-1][0])
            total_throughput = float(total_match[-1][1])
            total_tokens = int(round(total_throughput * total_latency))
            
            input_len = prefill_tokens // batch_size
            output_len = (total_tokens // batch_size) - input_len
            info['input_len'] = input_len
            info['output_len'] = output_len
            info['main_tokens'] = {'input': prefill_tokens, 'output': total_tokens - prefill_tokens}
    
    return info


def simple_kmeans(data: np.ndarray, k: int, max_iters: int = 100, seed: int = 42) -> np.ndarray:
    """
    Simple K-means clustering implementation.

    Args:
        data: 1D array of values to cluster
        k: Number of clusters
        max_iters: Maximum number of iterations
        seed: Random seed for reproducibility

    Returns:
        cluster_labels: Array of cluster assignments (0 to k-1)
    """
    rng = np.random.RandomState(seed)
    # Initialize centroids using k-means++
    centroids = []
    centroids.append(data[rng.randint(len(data))])
    
    for _ in range(1, k):
        # Calculate distances to nearest centroid
        distances = np.min([np.abs(data - c) for c in centroids], axis=0)
        # Probability proportional to squared distance
        probs = distances ** 2
        probs = probs.astype(np.float64)
        probs /= probs.sum()
        # Select next centroid
        centroids.append(data[rng.choice(len(data), p=probs)])
    
    centroids = np.array(centroids)
    
    # K-means iterations
    for _ in range(max_iters):
        # Assign points to nearest centroid
        labels = np.argmin(np.abs(data[:, None] - centroids), axis=1)
        
        # Update centroids
        new_centroids = np.array([data[labels == i].mean() if (labels == i).any() else centroids[i] 
                                   for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels


def compute_weighted_cv(data: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Compute compute-weighted coefficient of variation.
    
    Args:
        data: Array of activation counts
        labels: Cluster assignments
        k: Number of clusters
    
    Returns:
        Weighted CV value (lower is better)
    """
    cluster_totals = []
    for i in range(k):
        cluster_mask = (labels == i)
        if cluster_mask.any():
            cluster_totals.append(data[cluster_mask].sum())
        else:
            cluster_totals.append(0)
    
    cluster_totals = np.array(cluster_totals)
    mean_total = cluster_totals.mean()
    std_total = cluster_totals.std()
    
    if mean_total == 0:
        return float('inf')
    
    cv = std_total / mean_total
    return cv


def analyze_single_prefill_bucketed(
    pt_file_path: Path,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_buckets: int = 5,
    dtype: str = None,
    quantization: str = None
) -> pd.DataFrame:
    """
    Analyze prefill phase from expert distribution records with K-means bucketing.
    
    Args:
        pt_file_path: Path to .pt file containing expert distribution records
        batch_size: Number of concurrent requests
        input_len: Input sequence length per request
        output_len: Output sequence length per request
        num_buckets: Number of buckets to create (default 5)
        dtype: Data type (e.g., 'bfloat16')
        quantization: Quantization method (e.g., 'w8a8_int8')
    
    Returns:
        DataFrame with bucketed results containing: Layer, Bucket, Expert_IDs, Avg_Activations, Batch_Size, TopK, Dtype, Quantization
    """
    # Load records
    print(f"\nLoading expert distribution records from: {pt_file_path}")
    data = torch.load(pt_file_path)
    
    # Handle different formats: dict or list
    if isinstance(data, dict):
        records = data['records']
    else:
        records = data
    
    # Determine prefill threshold
    expected_prefill_tokens = batch_size * input_len
    prefill_threshold = expected_prefill_tokens * 0.5
    
    print(f"Prefill threshold: {prefill_threshold} tokens (batch_size={batch_size}, input_len={input_len})")
    
    # Separate prefill vs decode by num_tokens dimension
    prefill_records = []
    decode_records = []
    
    for record in records:
        # Extract tensor from dict format or use directly
        if isinstance(record, dict):
            tensor = record['topk_ids_of_layer']
            # Use forward_mode if available (1=prefill, 2=decode), otherwise use threshold
            if 'forward_mode' in record:
                is_prefill = (record['forward_mode'] == 1)
            else:
                num_tokens = tensor.shape[1]
                is_prefill = (num_tokens > prefill_threshold)
        else:
            tensor = record
            num_tokens = tensor.shape[1]
            is_prefill = (num_tokens > prefill_threshold)
        
        if is_prefill:
            prefill_records.append(tensor)
        else:
            decode_records.append(tensor)
    
    print(f"Found {len(prefill_records)} prefill records, {len(decode_records)} decode records")
    
    if not prefill_records:
        raise ValueError("No prefill records found!")
    
    # Combine all prefill records
    combined_prefill = torch.cat(prefill_records, dim=1)  # Shape: [num_layers, total_prefill_tokens, topk_buf]

    num_layers = combined_prefill.shape[0]
    num_tokens_prefill = combined_prefill.shape[1]
    topk_buf = combined_prefill.shape[2]

    # Detect actual topk by checking for -1 padding in buffer
    # (buffer may be larger than actual topk, padded with -1)
    sample = combined_prefill[0, 0]  # first token of first layer
    valid_mask = sample != -1
    topk = int(valid_mask.sum().item())
    if topk < topk_buf:
        print(f"  Detected padding: buffer width={topk_buf}, actual TopK={topk}")
        combined_prefill = combined_prefill[:, :, :topk]

    print(f"Combined prefill shape: {combined_prefill.shape}")
    print(f"  Layers: {num_layers}, Prefill tokens: {num_tokens_prefill}, TopK: {topk}")

    # Process each layer
    all_results = []
    expected_total_activations = num_tokens_prefill * topk
    validation_errors = 0

    for layer_idx in range(num_layers):
        layer_experts = combined_prefill[layer_idx]  # Shape: [num_tokens_prefill, topk]

        # Flatten and count expert activations (exclude any remaining -1 padding)
        flat = layer_experts.flatten().tolist()
        expert_counts = Counter(x for x in flat if x != -1)
        
        # Get all expert IDs and their activation counts
        expert_ids = np.array(sorted(expert_counts.keys()))
        activation_counts = np.array([expert_counts[eid] for eid in expert_ids])
        
        if len(expert_ids) == 0:
            continue
        
        # Validate: total activations must equal num_tokens_prefill * topk
        actual_total = int(activation_counts.sum())
        if actual_total != expected_total_activations:
            print(f"  WARNING Layer {layer_idx}: total activations {actual_total} != expected {expected_total_activations}")
            validation_errors += 1
        
        # Try three bucketing approaches
        best_method = None
        best_cv = float('inf')
        best_labels = None
        
        # Method 1: K-means on absolute activation counts
        if len(expert_ids) >= num_buckets:
            labels_abs = simple_kmeans(activation_counts, num_buckets)
            cv_abs = compute_weighted_cv(activation_counts, labels_abs, num_buckets)
            
            if cv_abs < best_cv:
                best_cv = cv_abs
                best_method = "kmeans_abs"
                best_labels = labels_abs
        
        # Method 2: K-means on log-space activation counts
        if len(expert_ids) >= num_buckets:
            log_counts = np.log1p(activation_counts)
            labels_log = simple_kmeans(log_counts, num_buckets)
            cv_log = compute_weighted_cv(activation_counts, labels_log, num_buckets)
            
            if cv_log < best_cv:
                best_cv = cv_log
                best_method = "kmeans_log"
                best_labels = labels_log
        
        # Method 3: Quantile-based bucketing
        bucket_size = len(expert_ids) // num_buckets
        labels_quantile = np.zeros(len(expert_ids), dtype=int)
        sorted_indices = np.argsort(activation_counts)
        
        for i, idx in enumerate(sorted_indices):
            bucket_id = min(i // bucket_size, num_buckets - 1)
            labels_quantile[idx] = bucket_id
        
        cv_quantile = compute_weighted_cv(activation_counts, labels_quantile, num_buckets)
        
        if cv_quantile < best_cv:
            best_cv = cv_quantile
            best_method = "quantile"
            best_labels = labels_quantile
        
        # If no method worked (too few experts), assign all to bucket 0
        if best_labels is None:
            best_labels = np.zeros(len(expert_ids), dtype=int)
            best_method = "single_bucket"
        
        # Group experts by bucket, then enforce conservation law:
        # Σ(Avg_Activations × Num_Experts) == num_tokens_prefill × topk
        target = expected_total_activations
        bucket_list = []

        for bucket_id in range(num_buckets):
            bucket_mask = (best_labels == bucket_id)
            if not bucket_mask.any():
                continue
            n = int(bucket_mask.sum())
            avg = round(float(activation_counts[bucket_mask].mean()))
            bucket_list.append({'bucket_id': bucket_id, 'num_experts': n, 'avg': avg})

        # Enforce conservation: Σ(Avg × N) >= target, with minimum overshoot.
        # We guarantee the model is at least as expensive as reality.
        actual = sum(b['avg'] * b['num_experts'] for b in bucket_list)
        delta = target - actual  # positive = under-count, negative = over-count

        if delta > 0:
            # Under-count: add tokens. Pick bucket where +1 to avg
            # causes the smallest overshoot (i.e., bucket with largest N
            # that doesn't overshoot by more than N, or smallest N that
            # brings us to or above target with minimum excess).
            best_bucket = min(bucket_list,
                             key=lambda b: (b['num_experts'] - delta % b['num_experts']) % b['num_experts'])
            n = best_bucket['num_experts']
            best_bucket['avg'] += (delta + n - 1) // n  # ceiling division
        elif delta < 0:
            # Over-count: we're already >= target, leave as-is (model is
            # more expensive). But if overshoot is large, try to reduce it.
            # Find bucket where -1 to avg still keeps us >= target.
            for b in sorted(bucket_list, key=lambda b: b['num_experts']):
                if actual - b['num_experts'] >= target:
                    b['avg'] -= 1
                    actual -= b['num_experts']
                    if actual <= target + b['num_experts']:
                        break

        csv_total = sum(b['avg'] * b['num_experts'] for b in bucket_list)
        # Round Σ up to ceil(csv_total / K) * K so downstream reshape [-1, K, D] works.
        # Split 1 expert off a bucket and add the pad to that single expert.
        remainder = csv_total % topk
        if remainder != 0:
            pad = topk - remainder
            donor = max(bucket_list, key=lambda b: b['num_experts'])
            donor['num_experts'] -= 1
            bucket_list.append({'bucket_id': donor['bucket_id'], 'num_experts': 1, 'avg': donor['avg'] + pad})
            csv_total += pad
        overshoot = csv_total - target
        assert csv_total >= target, f"Layer {layer_idx}: conservation violated {csv_total} < {target}"
        assert csv_total % topk == 0, f"Layer {layer_idx}: Σ(N×A)={csv_total} not divisible by K={topk}"

        for b in bucket_list:
            all_results.append({
                'Layer': layer_idx,
                'Bucket': b['bucket_id'],
                'Num_Experts': b['num_experts'],
                'Avg_Activations': b['avg'],
                'Batch_Size': batch_size,
                'TopK': topk,
                'Dtype': dtype,
                'Quantization': quantization if quantization else ''
            })

        if overshoot > max(b['num_experts'] for b in bucket_list):
            print(f"  WARNING Layer {layer_idx}: overshoot={overshoot} (target={target}, actual={csv_total})")
            validation_errors += 1

    if validation_errors == 0:
        print(f"\n  VALIDATION PASSED: all {num_layers} layers satisfy Σ(N×A) >= target")
    else:
        print(f"\n  VALIDATION FAILED: {validation_errors} layer(s) with excessive overshoot")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prefill phase expert activation patterns from log file(s) with K-means bucketing"
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="Path(s) to log file(s) containing expert distribution run metadata"
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=5,
        help="Number of buckets for K-means clustering (default: 5)"
    )
    parser.add_argument(
        "--return-dataframe",
        action="store_true",
        help="Return DataFrame instead of saving (for programmatic use)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for combined CSV (default: first log file directory/statistics)"
    )
    
    args = parser.parse_args()
    
    # Process files and combine
    print(f"Processing {len(args.log_files)} log file(s)...\n")
    all_dfs = []
    
    for log_file in args.log_files:
        print(f"\n{'='*80}")
        print(f"Processing: {log_file.name}")
        print(f"{'='*80}")
        
        try:
            log_info = parse_log_file(log_file)
            
            print(f"  batch_size={log_info.get('batch_size')}, "
                  f"input_len={log_info.get('input_len')}, "
                  f"output_len={log_info.get('output_len')}")
            
            if 'batch_size' not in log_info or 'input_len' not in log_info:
                print(f"  ✗ Skipping: missing required metadata")
                continue
            
            pt_file_path = log_info['pt_file_path']
            if not pt_file_path.exists():
                print(f"  ✗ Skipping: .pt file not found")
                continue
            
            df = analyze_single_prefill_bucketed(
                pt_file_path=pt_file_path,
                batch_size=log_info['batch_size'],
                input_len=log_info['input_len'],
                output_len=log_info['output_len'],
                num_buckets=args.num_buckets,
                dtype=log_info.get('dtype'),
                quantization=log_info.get('quantization')
            )
            all_dfs.append(df)
            print(f"  ✓ Processed {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    if not all_dfs:
        print("\nError: No files were successfully processed")
        sys.exit(1)
    
    if args.return_dataframe:
        if len(all_dfs) == 1:
            return all_dfs[0]
        return pd.concat(all_dfs, ignore_index=True)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.log_files[0].parent / "statistics"
    
    output_dir.mkdir(exist_ok=True)
    parent_tag = args.log_files[0].parent.name
    output_file = output_dir / f"{parent_tag}_prefill_{args.num_buckets}buckets.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"\n\n{'='*80}")
    print(f"✓ Saved prefill distribution to: {output_file}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Batch sizes: {sorted(combined_df['Batch_Size'].unique())}")
    print(f"  Format: Layer, Bucket, Num_Experts, Avg_Activations, Batch_Size, TopK, Dtype, Quantization")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
