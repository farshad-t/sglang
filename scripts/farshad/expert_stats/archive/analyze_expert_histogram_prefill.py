#!/usr/bin/env python3
"""
Analyze expert activation histograms for PREFILL phase only.
Similar to decode analysis but for prefill records.
"""
import torch
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_prefill_experts(log_dir):
    log_dir = Path(log_dir)
    
    # Find expert distribution files
    pt_files = list(log_dir.glob("expert_distribution_recorder_*.pt"))
    
    if not pt_files:
        print(f"ERROR: No expert distribution files found in {log_dir}")
        return None
    
    print(f"Found {len(pt_files)} expert distribution file(s):")
    for i, f in enumerate(pt_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
    
    # Use first file
    pt_file_path = pt_files[0]
    
    print(f"\n{'='*80}")
    print(f"Analyzing PREFILL phase: {pt_file_path.name}")
    print(f"{'='*80}\n")
    
    # Find corresponding log file to get batch size
    log_files = list(log_dir.glob("*.log"))
    batch_size = None
    output_tokens_generated = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Total output tokens:' in line:
                    try:
                        tokens = int(line.split(':')[1].strip())
                        if tokens >= 1000 and (output_tokens_generated is None or tokens > output_tokens_generated):
                            output_tokens_generated = tokens
                    except:
                        pass
                
                if 'Successful requests:' in line:
                    if output_tokens_generated:
                        try:
                            batch_size = int(line.split(':')[1].strip())
                            break
                        except:
                            pass
            
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
    prefill_sizes = []
    
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens_in_record = topk_ids_of_layer.shape[1]
        
        if num_tokens_in_record > 100:
            prefill_topk_ids.append(topk_ids_of_layer)
            prefill_sizes.append(num_tokens_in_record)
        else:
            decode_topk_ids.append(topk_ids_of_layer)
    
    print(f"  Prefill records: {len(prefill_topk_ids)}")
    print(f"  Decode records: {len(decode_topk_ids)}")
    
    if prefill_topk_ids:
        print(f"\n  Prefill record sizes:")
        for i, size in enumerate(prefill_sizes):
            print(f"    Record {i}: {size} tokens")
        
        total_prefill_tokens = sum(prefill_sizes)
        print(f"  Total prefill tokens: {total_prefill_tokens}")
    else:
        print("\nERROR: No prefill records found!")
        return None
    
    # Find the main prefill batch (batch_size × input_len)
    # This should be close to 23 × 1024 = 23,552 tokens
    print(f"\n{'='*80}")
    print("PREFILL RECORD SELECTION")
    print(f"{'='*80}")
    print(f"\nYou have {len(prefill_topk_ids)} prefill records.")
    print(f"Expected main prefill: batch_size × input_len ≈ 23 × 1024 = 23,552 tokens")
    
    # Find the largest record (should be the main prefill batch)
    largest_idx = prefill_sizes.index(max(prefill_sizes))
    largest_size = prefill_sizes[largest_idx]
    
    print(f"\nLargest prefill record: Record {largest_idx} with {largest_size} tokens")
    print(f"Using ONLY this record (main prefill batch) for analysis")
    
    prefill_data = prefill_topk_ids[largest_idx]
    
    all_topk_ids = prefill_data
    num_layers, num_tokens, topk = all_topk_ids.shape
    
    if batch_size:
        BATCH_SIZE = batch_size
        print(f"\nBatch size (from log): {BATCH_SIZE}")
    else:
        print(f"\nWARNING: Could not read batch size from log")
        print(f"Using batch size = {len(prefill_topk_ids)} (number of prefill records)")
        BATCH_SIZE = len(prefill_topk_ids)
    
    print(f"\nModel Configuration:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total prefill tokens: {num_tokens}")
    print(f"  Top-K experts per token: {topk}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # For prefill, we have two options:
    # Option 1: Treat all prefill tokens together (like one big step)
    # Option 2: Divide into chunks matching batch size
    
    # Let's try Option 1 first: analyze all prefill tokens as one "step"
    print(f"\n{'='*80}")
    print("PREFILL HISTOGRAM ANALYSIS")
    print(f"{'='*80}")
    print(f"\nAnalyzing main prefill batch: {num_tokens} prefill tokens")
    print(f"(Computing histogram across entire prefill batch)\n")
    
    # Prepare data for CSV output
    histogram_data = []
    
    print(f"Processing layers: ", end='', flush=True)
    
    for layer_idx in range(num_layers):
        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx+1}/{num_layers}...", end=' ', flush=True)
        
        # For this layer, get all expert IDs across all prefill tokens
        layer_data = all_topk_ids[layer_idx]  # [num_tokens, topk]
        
        # Flatten to get all expert IDs
        all_expert_ids = layer_data.flatten().tolist()
        
        # Count occurrences of each expert
        expert_counts = defaultdict(int)
        for expert_id in all_expert_ids:
            expert_counts[expert_id] += 1
        
        # Create histogram: how many experts are activated N times
        activation_histogram = defaultdict(int)
        for expert_id, count in expert_counts.items():
            activation_histogram[count] += 1
        
        # Verify total activations
        total_activations = sum(act_count * num_experts for act_count, num_experts in activation_histogram.items())
        expected_activations = num_tokens * topk
        
        if layer_idx < 3 or layer_idx >= num_layers - 3:
            print(f"\n\nLayer {layer_idx:2d}:")
            print(f"  Histogram (main prefill batch: {num_tokens} tokens):")
            for act_count in sorted(activation_histogram.keys()):
                num_experts = activation_histogram[act_count]
                print(f"    {act_count:4d} activations/expert: {num_experts:3d} experts")
            
            print(f"  Total unique experts: {len(expert_counts)}")
            print(f"  Total activations: {total_activations} (expected: {expected_activations})")
            
            if total_activations == expected_activations:
                print(f"  ✓ EXACT match")
            else:
                print(f"  ✗ ERROR: {total_activations - expected_activations:+d}")
        
        # Save to CSV data
        for act_count in sorted(activation_histogram.keys()):
            num_experts = activation_histogram[act_count]
            histogram_data.append({
                'Layer': layer_idx,
                'Activations_Per_Expert': act_count,
                'Num_Experts': num_experts,
                'Total_Prefill_Tokens': num_tokens,
                'TopK': topk
            })
    
    print(f"\n\n{'='*80}")
    print("SAVING OUTPUT")
    print(f"{'='*80}\n")
    
    # Save to CSV
    import csv
    # Save to parent directory's statistics folder
    parent_dir = log_dir.parent
    output_dir = parent_dir / "statistics"
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / "expert_activation_histogram_prefill.csv"
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Layer', 'Activations_Per_Expert', 'Num_Experts', 
                                                'Total_Prefill_Tokens', 'TopK'])
        writer.writeheader()
        writer.writerows(histogram_data)
    
    print(f"Prefill histogram analysis saved to: {output_csv}")
    print(f"  Format: Layer, Activations_Per_Expert, Num_Experts")
    print(f"  Total rows: {len(histogram_data)}")
    print(f"  Analyzed: Main prefill batch (Record {largest_idx}: {num_tokens} tokens)")
    print(f"  Example: 'Layer 0, 45 activations/expert, 3 experts'")
    print(f"  Meaning: 3 experts are activated exactly 45 times across the prefill batch")
    print(f"  Note: sum(activations × experts) = {num_tokens} × {topk} = {num_tokens * topk}")
    
    return histogram_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze expert activation histograms for prefill phase")
    parser.add_argument("--log-dir", required=True, help="Directory containing expert logs and .pt files")
    args = parser.parse_args()
    
    analyze_prefill_experts(args.log_dir)
