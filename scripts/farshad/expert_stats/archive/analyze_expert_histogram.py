#!/usr/bin/env python3
"""
Analyze expert token mappings using histogram-based bucketing.
For each decode step, creates a histogram showing how many experts are activated X times.
Then averages histograms across all decode steps.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd


def analyze_expert_histogram(pt_file_path, return_dataframe=False):
    """
    Analyze expert distribution using per-decode-step histograms.
    
    Args:
        pt_file_path: Path to the .pt file containing expert distribution data
        return_dataframe: If True, return DataFrame instead of saving to CSV
    
    Returns:
        DataFrame if return_dataframe=True, else None
    """
    pt_file_path = Path(pt_file_path)
    log_dir = pt_file_path.parent.parent
    output_dir = log_dir / "statistics"
    output_dir.mkdir(exist_ok=True)
    
    # Try to read actual token counts and batch size from benchmark log
    log_files = list(log_dir.glob('*.log'))
    output_tokens_generated = None
    batch_size = None
    if log_files:
        for log_file in log_files:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
                # Find the benchmark result section
                benchmark_result_idx = None
                for i, line in enumerate(lines):
                    if '====== Offline Throughput Benchmark Result =======' in line:
                        benchmark_result_idx = i
                        break
                
                if benchmark_result_idx:
                    # Find the LARGEST output token count (main benchmark, not warmup)
                    max_output_tokens = 0
                    for line in lines[:benchmark_result_idx]:
                        if '#Output tokens:' in line:
                            try:
                                tokens = int(line.split(':')[1].strip())
                                if tokens > max_output_tokens:
                                    max_output_tokens = tokens
                            except:
                                pass
                    if max_output_tokens >= 1000:
                        output_tokens_generated = max_output_tokens
                
                # Find batch size from benchmark result section
                if benchmark_result_idx:
                    for line in lines[benchmark_result_idx:]:
                        if 'Successful requests:' in line:
                            try:
                                batch_size = int(line.split(':')[1].strip())
                                break
                            except:
                                pass
                
                if output_tokens_generated and batch_size:
                    break
            if output_tokens_generated and batch_size:
                break
    
    print(f"Loading expert distribution data from: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')
    
    print(f"\nData keys: {data.keys()}")
    
    records = data['records']
    print(f"\nProcessing {len(records)} records...")
    
    # Separate prefill and decode records
    prefill_topk_ids = []
    decode_topk_ids = []
    
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens_in_record = topk_ids_of_layer.shape[1]
        
        if num_tokens_in_record > 100:
            prefill_topk_ids.append(topk_ids_of_layer)
        else:
            decode_topk_ids.append(topk_ids_of_layer)
    
    print(f"  Prefill records: {len(prefill_topk_ids)}")
    print(f"  Decode records: {len(decode_topk_ids)}")
    
    if decode_topk_ids:
        decode_data = torch.cat(decode_topk_ids, dim=1)
        print(f"  Decode tokens: {decode_data.shape[1]}")
    else:
        print("\nERROR: No decode records found!")
        return None
    
    # Use DECODE data for analysis
    all_topk_ids = decode_data
    num_layers, num_tokens, topk = all_topk_ids.shape
    
    if output_tokens_generated and batch_size:
        BATCH_SIZE = batch_size
        print(f"\nBatch size (from log): {BATCH_SIZE}")
        print(f"Output tokens generated: {output_tokens_generated}")
    else:
        print("\nWARNING: Could not read batch size from log, assuming 23")
        BATCH_SIZE = 23
    
    print(f"\nModel Configuration:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total decode tokens: {num_tokens}")
    print(f"  Top-K experts per token: {topk}")
    print(f"  Batch size (tokens per decode step): {BATCH_SIZE}")
    
    # Reshape data into decode steps
    num_decode_steps = num_tokens // BATCH_SIZE
    remaining_tokens = num_tokens % BATCH_SIZE
    
    if remaining_tokens > 0:
        print(f"\nNote: Discarding {remaining_tokens} incomplete tokens at end")
        all_topk_ids = all_topk_ids[:, :num_decode_steps * BATCH_SIZE, :]
        num_tokens = num_decode_steps * BATCH_SIZE
    
    print(f"  Number of decode steps: {num_decode_steps}")
    
    # Reshape to group tokens into decode steps: [num_layers, num_decode_steps, batch_size, topk]
    reshaped_data = all_topk_ids.reshape(num_layers, num_decode_steps, BATCH_SIZE, topk)
    
    print("\n" + "="*80)
    print("HISTOGRAM-BASED ANALYSIS")
    print("="*80)
    print(f"\nFor each decode step ({BATCH_SIZE} tokens), create histogram of expert activations")
    print(f"Then average across all {num_decode_steps} decode steps\n")
    
    # Prepare data for CSV output
    histogram_data = []
    
    print(f"Processing layers: ", end='', flush=True)
    
    for layer_idx in range(num_layers):
        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx+1}/{num_layers}...", end=' ', flush=True)
        
        # For this layer, analyze each decode step separately
        layer_histograms = []  # List of histograms, one per decode step
        
        for step_idx in range(num_decode_steps):
            # Get expert IDs for this decode step (batch_size tokens)
            step_expert_ids = reshaped_data[layer_idx, step_idx]  # Shape: [batch_size, topk]
            
            # Count activations per expert in this decode step
            expert_activation_count = defaultdict(int)
            for token_idx in range(BATCH_SIZE):
                for expert_id in step_expert_ids[token_idx].numpy():
                    expert_activation_count[int(expert_id)] += 1
            
            # Create histogram: bucket[k] = number of experts activated exactly k times
            histogram = defaultdict(int)
            for expert_id, activation_count in expert_activation_count.items():
                histogram[activation_count] += 1
            
            # Verify correctness: sum(k × count[k]) should equal batch_size × topk
            total_activations = sum(k * count for k, count in histogram.items())
            expected_activations = BATCH_SIZE * topk
            if total_activations != expected_activations:
                print(f"\nWARNING: Layer {layer_idx}, step {step_idx}: total={total_activations}, expected={expected_activations}")
            
            layer_histograms.append(histogram)
        
        # Average histograms across all decode steps
        # Find all activation counts that appear
        all_activation_counts = set()
        for hist in layer_histograms:
            all_activation_counts.update(hist.keys())
        
        # Calculate average number of experts for each activation count
        avg_histogram = {}
        for activation_count in sorted(all_activation_counts):
            expert_counts = [hist.get(activation_count, 0) for hist in layer_histograms]
            avg_num_experts = np.mean(expert_counts)
            avg_histogram[activation_count] = avg_num_experts
        
        # Round averages starting from buckets with MOST experts (most important patterns)
        # Stop when we reach the target total (batch_size × topk)
        target_total = BATCH_SIZE * topk
        rounded_histogram = {}
        current_total = 0
        
        # Sort by avg_num_experts descending (process most frequent patterns first)
        sorted_by_frequency = sorted(avg_histogram.items(), key=lambda x: x[1], reverse=True)
        
        for activation_count, avg_num_experts in sorted_by_frequency:
            rounded_experts = round(avg_num_experts)
            
            if rounded_experts > 0:  # Only include if at least 1 expert
                contribution = activation_count * rounded_experts
                if current_total + contribution <= target_total:
                    rounded_histogram[activation_count] = rounded_experts
                    current_total += contribution
                elif current_total < target_total:
                    # Partial fit: add what we can to reach exactly target_total
                    remaining = target_total - current_total
                    if remaining >= activation_count:
                        final_experts = remaining // activation_count
                        if final_experts > 0:
                            rounded_histogram[activation_count] = final_experts
                            current_total += activation_count * final_experts
        
        # Final adjustment: ensure we reach exactly target_total
        current_total = sum(k * v for k, v in rounded_histogram.items())
        
        while current_total < target_total:
            remaining = target_total - current_total
            # Add 1 expert to the bucket with smallest activation count that fits
            added = False
            for act_count in sorted(rounded_histogram.keys()):
                if act_count <= remaining:
                    rounded_histogram[act_count] += 1
                    current_total += act_count
                    added = True
                    break
            
            # If can't add to existing buckets, try unused activation counts
            if not added:
                for act_count in sorted(avg_histogram.keys()):
                    if act_count not in rounded_histogram and act_count <= remaining:
                        rounded_histogram[act_count] = 1
                        current_total += act_count
                        added = True
                        break
            
            if not added:
                # Can't reach target exactly, stop
                break
        
        final_total = sum(k * v for k, v in rounded_histogram.items())
        
        final_total = sum(k * v for k, v in rounded_histogram.items())
        
        # Print summary for first/last few layers
        if layer_idx < 3 or layer_idx >= num_layers - 2:
            print(f"\n\nLayer {layer_idx:2d}:")
            print(f"  Histogram (rounded integers, averaged across {num_decode_steps} decode steps):")
            for activation_count in sorted(rounded_histogram.keys()):
                print(f"    {activation_count:2d} activations/expert: {rounded_histogram[activation_count]:3d} experts")
            
            # Verification
            total_experts = sum(rounded_histogram.values())
            print(f"  Total unique experts: {total_experts}")
            print(f"  Total activations: {final_total} (expected: {target_total})")
            if final_total == target_total:
                print(f"  ✓ EXACT match")
            else:
                print(f"  ✗ ERROR: {final_total - target_total}")
        
        # Save to CSV
        for activation_count in sorted(rounded_histogram.keys()):
            histogram_data.append({
                'Layer': layer_idx,
                'Activations_Per_Expert': activation_count,
                'Num_Experts': rounded_histogram[activation_count],
                'Batch_Size': BATCH_SIZE,
                'TopK': topk
            })
    
    print("\n\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    # Create DataFrame
    df = pd.DataFrame(histogram_data)
    
    if return_dataframe:
        print(f"\nReturning DataFrame with {len(df)} rows (Batch_Size={BATCH_SIZE})")
        return df
    
    # Save to CSV
    output_file = output_dir / 'expert_activation_histogram.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nHistogram analysis saved to: {output_file}")
    print(f"  Format: Layer, Activations_Per_Expert, Num_Experts, Batch_Size")
    print(f"  Total rows: {len(histogram_data)}")
    print(f"  Example: 'Layer 0, 3 activations/expert, 15 experts, Batch_Size 22'")
    print(f"  Meaning: 15 experts are activated exactly 3 times per decode step")
    print(f"  Note: Rounded to integers, sum(activations × experts) = {BATCH_SIZE} × {topk} = {BATCH_SIZE * topk}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Analyze expert distribution using histograms')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Directory containing expert distribution .pt files')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    pt_files = list(log_dir.glob('expert_distribution_recorder_*.pt'))
    
    if not pt_files:
        print(f"No expert distribution files found in {log_dir}")
        return
    
    print(f"Found {len(pt_files)} expert distribution file(s):")
    for i, pt_file in enumerate(pt_files, 1):
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {pt_file.name} ({size_mb:.1f} MB)")
    
    # Collect DataFrames from all files
    all_dfs = []
    
    for pt_file in pt_files:
        print("\n" + "="*80)
        print(f"Analyzing: {pt_file.name}")
        print("="*80 + "\n")
        
        df = analyze_expert_histogram(pt_file, return_dataframe=True)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_dir = log_dir.parent / "statistics"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'expert_activation_histogram.csv'
        combined_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("COMBINED RESULTS")
        print("="*80)
        print(f"\n✓ Saved combined histogram from {len(pt_files)} batch sizes to: {output_file}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Format: Layer, Activations_Per_Expert, Num_Experts, Batch_Size, TopK")
        print(f"\nBatch sizes processed:")
        for batch_size in sorted(combined_df['Batch_Size'].unique()):
            count = len(combined_df[combined_df['Batch_Size'] == batch_size])
            print(f"  Batch_Size={batch_size}: {count} rows")


if __name__ == "__main__":
    main()
