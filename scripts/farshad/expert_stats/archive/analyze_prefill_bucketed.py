#!/usr/bin/env python3
"""
Analyze expert activation patterns for PREFILL phase and generate bucketed output.
Directly produces 5-bucket CSV for ArchBench modeling (skips intermediate histogram).
"""
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def simple_kmeans(data, k, max_iter=100):
    """Simple K-means implementation without sklearn."""
    # Initialize centers using quantiles
    centers = np.percentile(data, np.linspace(0, 100, k+2)[1:-1])
    
    for _ in range(max_iter):
        # Assign to nearest center
        distances = np.abs(data[:, np.newaxis] - centers)
        labels = np.argmin(distances, axis=1)
        
        # Update centers
        new_centers = np.array([data[labels == i].mean() if np.any(labels == i) else centers[i] 
                                for i in range(k)])
        
        # Check convergence
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers


def compute_weighted_cv(buckets):
    """Calculate compute-weighted coefficient of variation."""
    total_compute = sum(b['mean'] * b['num_experts'] for b in buckets)
    weighted_cv = sum((b['std']/b['mean']*100) * (b['mean'] * b['num_experts']) / total_compute 
                     for b in buckets)
    return weighted_cv


def analyze_single_prefill_bucketed(pt_file_path, num_buckets=5):
    """
    Analyze prefill phase from a single .pt file and generate bucketed output.
    
    Args:
        pt_file_path: Path to expert distribution .pt file
        num_buckets: Number of buckets for K-means clustering
    
    Returns:
        DataFrame with bucketed prefill analysis
    """
    pt_file_path = Path(pt_file_path)
    log_dir = pt_file_path.parent
    parent_dir = log_dir.parent
    
    print(f"\n{'='*80}")
    print(f"Analyzing PREFILL phase: {pt_file_path.name}")
    print(f"{'='*80}\n")
    
    # First, try to parse log file to get batch_size and input_len
    batch_size_from_log = None
    input_len_from_log = None
    output_len_from_log = None
    
    log_files = list(parent_dir.glob('*.log'))
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Look for command-line args patterns
            import re
            
            # Find --random-input-len 1024 (from command line)
            input_match = re.search(r'--random-input-len[\s=](\d+)', content)
            if input_match:
                input_len_from_log = int(input_match.group(1))
            
            # Find --random-output-len 8192 (from command line)
            output_match = re.search(r'--random-output-len[\s=](\d+)', content)
            if output_match:
                output_len_from_log = int(output_match.group(1))
            
            # Find batch_size from benchmark results: "Successful requests: 22"
            batch_match = re.search(r'Successful requests:\s+(\d+)', content)
            if batch_match:
                batch_size_from_log = int(batch_match.group(1))
            
            if batch_size_from_log and input_len_from_log:
                print(f"Parsed from log ({log_file.name}):")
                print(f"  batch_size={batch_size_from_log}, input_len={input_len_from_log}, output_len={output_len_from_log}")
                break
    
    if batch_size_from_log and input_len_from_log:
        # Prefill threshold: 50% of expected prefill size
        prefill_threshold = batch_size_from_log * input_len_from_log * 0.5
    else:
        print(f"Warning: Could not parse batch_size/input_len from log files")
        prefill_threshold = None
    
    print(f"Loading expert distribution data from: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')
    
    records = data['records']
    print(f"Processing {len(records)} records...")
    
    # Collect all record sizes first
    record_sizes = [record['topk_ids_of_layer'].shape[1] for record in records]
    
    # Auto-detect threshold if not available from log
    if prefill_threshold is None:
        # Use bimodal distribution detection
        sorted_sizes = sorted(record_sizes)
        gaps = [sorted_sizes[i+1] - sorted_sizes[i] for i in range(len(sorted_sizes)-1)]
        max_gap_idx = gaps.index(max(gaps))
        prefill_threshold = (sorted_sizes[max_gap_idx] + sorted_sizes[max_gap_idx+1]) / 2
        print(f"Auto-detected prefill threshold: {prefill_threshold:.0f} tokens")
    else:
        print(f"Using log-based prefill threshold: {prefill_threshold:.0f} tokens")
    
    # Separate prefill and decode records
    prefill_topk_ids = []
    prefill_sizes = []
    decode_batch_sizes = []
    
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens_in_record = topk_ids_of_layer.shape[1]
        
        if num_tokens_in_record > prefill_threshold:
            prefill_topk_ids.append(topk_ids_of_layer)
            prefill_sizes.append(num_tokens_in_record)
        else:
            decode_batch_sizes.append(num_tokens_in_record)
    
    print(f"  Prefill records: {len(prefill_topk_ids)}")
    print(f"  Decode records: {len(decode_batch_sizes)}")
    
    if not prefill_topk_ids:
        print("\nERROR: No prefill records found!")
        return None
    
    # Find the largest record (main prefill batch)
    largest_idx = prefill_sizes.index(max(prefill_sizes))
    largest_size = prefill_sizes[largest_idx]
    
    print(f"\nUsing main prefill batch: Record {largest_idx} with {largest_size} tokens")
    
    # Infer batch size from decode records (most reliable)
    if decode_batch_sizes:
        from collections import Counter
        batch_size_counts = Counter(decode_batch_sizes)
        BATCH_SIZE = batch_size_counts.most_common(1)[0][0]
        print(f"Batch size (inferred from decode records): {BATCH_SIZE}")
    else:
        # Fallback: assume input_len=1024, batch_size = total_prefill_tokens / input_len
        INPUT_LEN = 1024
        BATCH_SIZE = largest_size // INPUT_LEN
        print(f"Batch size (inferred from prefill tokens / {INPUT_LEN}): {BATCH_SIZE}")
    
    prefill_data = prefill_topk_ids[largest_idx]
    num_layers, num_tokens, topk = prefill_data.shape
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}, Prefill tokens: {num_tokens}, TopK: {topk}, Batch: {BATCH_SIZE}")
    
    # ========================================================================
    # STEP 1: Generate per-layer histograms
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Analyzing Expert Activation Patterns")
    print(f"{'='*80}")
    
    layer_histograms = []
    
    print(f"Processing layers: ", end='', flush=True)
    
    for layer_idx in range(num_layers):
        if layer_idx % 10 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx+1}/{num_layers}...", end=' ', flush=True)
        
        # Get all expert IDs across all prefill tokens
        layer_data = prefill_data[layer_idx]  # [num_tokens, topk]
        all_expert_ids = layer_data.flatten().tolist()
        
        # Count occurrences of each expert
        expert_counts = defaultdict(int)
        for expert_id in all_expert_ids:
            expert_counts[expert_id] += 1
        
        # Create histogram: how many experts are activated N times
        activation_histogram = defaultdict(int)
        for expert_id, count in expert_counts.items():
            activation_histogram[count] += 1
        
        layer_histograms.append({
            'layer': layer_idx,
            'histogram': activation_histogram,
            'total_tokens': num_tokens,
            'topk': topk
        })
    
    print(" Done!")
    
    # ========================================================================
    # STEP 2: Apply K-means bucketing across all layers
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: K-Means Bucketing (k={num_buckets})")
    print(f"{'='*80}")
    
    # Collect all activation counts weighted by number of experts
    activation_counts = []
    for layer_info in layer_histograms:
        for act_count, num_experts in layer_info['histogram'].items():
            activation_counts.extend([act_count] * num_experts)
    
    activation_counts = np.array(activation_counts)
    
    print(f"\nDistribution statistics:")
    print(f"  Total expert instances: {len(activation_counts):,}")
    print(f"  Range: {activation_counts.min():.0f} - {activation_counts.max():.0f} activations")
    print(f"  Mean: {activation_counts.mean():.1f} ± {activation_counts.std():.1f}")
    
    # Try three bucketing approaches and choose best
    approaches = []
    
    # Approach 1: K-means in absolute space
    labels_abs, centers_abs = simple_kmeans(activation_counts, num_buckets)
    buckets_abs = []
    for i, cluster_idx in enumerate(np.argsort(centers_abs)):
        cluster_mask = labels_abs == cluster_idx
        cluster_data = activation_counts[cluster_mask]
        buckets_abs.append({
            'bucket_id': i,
            'min': cluster_data.min(),
            'max': cluster_data.max(),
            'mean': cluster_data.mean(),
            'std': cluster_data.std(),
            'num_experts': len(cluster_data)
        })
    approaches.append(('kmeans_abs', buckets_abs))
    
    # Approach 2: K-means in log-space
    log_activations = np.log(activation_counts)
    labels_log, centers_log = simple_kmeans(log_activations, num_buckets)
    buckets_log = []
    for i, cluster_idx in enumerate(np.argsort(centers_log)):
        cluster_mask = labels_log == cluster_idx
        cluster_data = activation_counts[cluster_mask]
        buckets_log.append({
            'bucket_id': i,
            'min': cluster_data.min(),
            'max': cluster_data.max(),
            'mean': cluster_data.mean(),
            'std': cluster_data.std(),
            'num_experts': len(cluster_data)
        })
    approaches.append(('kmeans_log', buckets_log))
    
    # Approach 3: Quantile-based
    quantile_boundaries = np.percentile(activation_counts, np.linspace(0, 100, num_buckets+1))
    buckets_quantile = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            bucket_mask = (activation_counts >= quantile_boundaries[i]) & (activation_counts <= quantile_boundaries[i+1])
        else:
            bucket_mask = (activation_counts >= quantile_boundaries[i]) & (activation_counts < quantile_boundaries[i+1])
        
        bucket_data = activation_counts[bucket_mask]
        if len(bucket_data) > 0:
            buckets_quantile.append({
                'bucket_id': i,
                'min': bucket_data.min(),
                'max': bucket_data.max(),
                'mean': bucket_data.mean(),
                'std': bucket_data.std(),
                'num_experts': len(bucket_data)
            })
    approaches.append(('quantile', buckets_quantile))
    
    # Choose best approach based on compute-weighted CV
    best_name, best_buckets = min(approaches, key=lambda x: compute_weighted_cv(x[1]))
    best_cv = compute_weighted_cv(best_buckets)
    
    print(f"\n✓ Selected: {best_name} (Compute-weighted CV: {best_cv:.1f}%)")
    print(f"\nBucket summary:")
    for bucket in best_buckets:
        print(f"  Bucket {bucket['bucket_id']}: [{bucket['min']:.0f}-{bucket['max']:.0f}] "
              f"activations, {bucket['num_experts']:,} experts, "
              f"mean={bucket['mean']:.1f}±{bucket['std']:.1f}")
    
    # ========================================================================
    # STEP 3: Apply bucketing to each layer
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: Applying Bucketing Per Layer")
    print(f"{'='*80}")
    
    rebucketed_data = []
    
    for layer_info in layer_histograms:
        layer_idx = layer_info['layer']
        histogram = layer_info['histogram']
        total_tokens = layer_info['total_tokens']
        topk = layer_info['topk']
        
        # For each bucket, aggregate experts
        for bucket in best_buckets:
            bucket_id = bucket['bucket_id']
            bucket_min = bucket['min']
            bucket_max = bucket['max']
            
            # Find experts in this bucket for this layer
            total_experts = 0
            total_activations = 0
            
            for act_count, num_experts in histogram.items():
                if bucket_min <= act_count <= bucket_max:
                    total_experts += num_experts
                    total_activations += act_count * num_experts
            
            if total_experts == 0:
                continue
            
            avg_activations = total_activations / total_experts
            
            rebucketed_data.append({
                'Layer': layer_idx,
                'Bucket_ID': bucket_id,
                'Bucket_Range': f"{bucket_min:.0f}-{bucket_max:.0f}",
                'Avg_Activations_Per_Expert': round(avg_activations),
                'Num_Experts': total_experts,
                'Total_Activations': total_activations,
                'Total_Prefill_Tokens': total_tokens,
                'Batch_Size': BATCH_SIZE,
                'TopK': topk
            })
    
    rebucketed_df = pd.DataFrame(rebucketed_data)
    
    print(f"\nReturning DataFrame with {len(rebucketed_df)} rows (Batch_Size={BATCH_SIZE})")
    return rebucketed_df, BATCH_SIZE, best_cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze prefill expert distribution and generate bucketed output for ArchBench"
    )
    parser.add_argument("log_dir", help="Directory containing expert logs and .pt files")
    parser.add_argument("--num-buckets", type=int, default=5, 
                       help="Number of buckets (default: 5)")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    pt_files = list(log_dir.glob("expert_distribution_recorder_*.pt"))
    
    if not pt_files:
        print(f"ERROR: No expert distribution files found in {log_dir}")
        exit(1)
    
    print(f"Found {len(pt_files)} expert distribution file(s):")
    for i, f in enumerate(pt_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
    
    # Collect DataFrames from all files
    all_dfs = []
    batch_info = []
    
    for pt_file in pt_files:
        df, batch_size, cv = analyze_single_prefill_bucketed(pt_file, args.num_buckets)
        if df is not None:
            all_dfs.append(df)
            batch_info.append((batch_size, len(df), cv))
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_dir = log_dir.parent / "statistics"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "expert_activation_histogram_prefill_5buckets.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("COMBINED RESULTS")
        print(f"{'='*80}")
        print(f"\n✓ Saved combined prefill analysis from {len(pt_files)} batch sizes to: {output_file}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Format: Layer, Bucket_ID, Bucket_Range, Avg_Activations_Per_Expert, Num_Experts, Total_Activations, Total_Prefill_Tokens, Batch_Size, TopK")
        print(f"\nBatch sizes processed:")
        for batch_size, num_rows, cv in sorted(batch_info):
            print(f"  Batch_Size={batch_size}: {num_rows} rows, CV={cv:.1f}%")
        
        # Show sample
        print(f"\nSample output (Layer 0, first batch size):")
        first_batch = sorted(combined_df['Batch_Size'].unique())[0]
        sample = combined_df[(combined_df['Layer'] == 0) & (combined_df['Batch_Size'] == first_batch)]
        print(sample.to_string(index=False))

