#!/usr/bin/env python3
"""
Analyze decode phase expert distribution from SGLang benchmark log file.
Takes log file as input, extracts all metadata, and loads corresponding .pt file.
"""
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re


def parse_log_file(log_file_path):
    """
    Extract all relevant information from log file.
    Supports both bench_offline_throughput and bench_one_batch log formats.
    
    Returns:
        dict with keys: batch_size, input_len, output_len, pt_file_path, 
                       warmup_info, prefill_info
    """
    log_file_path = Path(log_file_path)
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    info = {}
    
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
    elif token_matches:
        # Single token match
        info['main_tokens'] = {'input': int(token_matches[0].group(1)), 'output': int(token_matches[0].group(2))}
    
    # --- bench_one_batch fallback: extract batch_size from decode lines ---
    if 'batch_size' not in info:
        # "Decode 0. Batch size: 22, latency: ..."
        decode_bs_match = re.search(r'Decode \d+\. Batch size: (\d+)', content)
        if decode_bs_match:
            info['batch_size'] = int(decode_bs_match.group(1))
    
    # --- bench_one_batch fallback: extract input/output from directory name ---
    # Directory pattern: qwen3.5-35B-A3B-fp8_tp1_bs22 (parent dir of benchmark.log)
    if 'input_len' not in info:
        # Try parent directory name pattern: ..._tp<N>_bs<N>
        dir_name = log_file_path.parent.name
        dir_bs_match = re.search(r'_bs(\d+)', dir_name)
        if dir_bs_match and 'batch_size' not in info:
            info['batch_size'] = int(dir_bs_match.group(1))
        
        # Count decode steps from log to infer output_len
        # bench_one_batch logs: "Benchmark ..." section has decode steps
        # The benchmark section prefill throughput tells us total prefill tokens
        prefill_matches = list(re.finditer(r'Prefill\. latency:.*throughput:\s+([\d.]+) token/s', content))
        if prefill_matches and 'batch_size' in info:
            # Last prefill is the benchmark run (first is warmup)
            # Use Total line to compute: Total tokens = batch_size * (input_len + output_len)
            total_match = re.findall(r'Total\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content)
            if total_match and len(total_match) >= 2:
                # Second Total is the benchmark run
                total_throughput = float(total_match[-1][1])
                total_latency = float(total_match[-1][0])
                total_tokens = int(round(total_throughput * total_latency))
                batch_size = info['batch_size']
                # total_tokens = batch_size * (input_len + output_len)
                # We need input_len - try from prefill latency
                prefill_latency = None
                prefill_throughput = None
                # Find benchmark prefill (the last one)
                for m in prefill_matches:
                    prefill_throughput = float(m.group(1))
                prefill_match_full = list(re.finditer(r'Prefill\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content))
                if prefill_match_full:
                    last_prefill = prefill_match_full[-1]
                    prefill_latency = float(last_prefill.group(1))
                    prefill_throughput = float(last_prefill.group(2))
                    prefill_tokens = int(round(prefill_throughput * prefill_latency))
                    input_len = prefill_tokens // batch_size
                    output_len = (total_tokens // batch_size) - input_len
                    info['input_len'] = input_len
                    info['output_len'] = output_len
                    info['main_tokens'] = {'input': prefill_tokens, 'output': total_tokens - prefill_tokens}
    
    # Fallback: try batch_size from successful requests (old method)
    if 'batch_size' not in info:
        batch_match = re.search(r'Successful requests:\s+(\d+)', content)
        if batch_match:
            info['batch_size'] = int(batch_match.group(1))
    
    # Extract warmup info (input/output tokens before main benchmark)
    if 'warmup_tokens' not in info:
        warmup_tokens = []
        for match in re.finditer(r'#Input tokens:\s+(\d+)\s+#Output tokens:\s+(\d+)', content):
            inp = int(match.group(1))
            out = int(match.group(2))
            warmup_tokens.append({'input': inp, 'output': out})
        
        # The last entries before benchmark results are warmup, earlier ones are main
        if len(warmup_tokens) > 1:
            info['warmup_tokens'] = warmup_tokens[-1]  # Small warmup at end
            info['main_tokens'] = warmup_tokens[0]      # Main benchmark at start
        elif warmup_tokens:
            info['main_tokens'] = warmup_tokens[0]
    
    # Extract prefill info from main tokens
    if 'main_tokens' in info and 'batch_size' in info and 'input_len' in info:
        expected_prefill_tokens = info['batch_size'] * info['input_len']
        actual_input_tokens = info['main_tokens']['input']
        info['prefill_tokens'] = actual_input_tokens
        info['prefill_match'] = (actual_input_tokens == expected_prefill_tokens)
    
    return info


def analyze_decode_histogram(log_file_path, return_dataframe=False):
    """
    Analyze expert distribution from log file.
    
    Args:
        log_file_path: Path to SGLang benchmark log file
        return_dataframe: If True, return DataFrame instead of saving
    
    Returns:
        DataFrame if return_dataframe=True, else None
    """
    log_file_path = Path(log_file_path)
    
    print(f"\n{'='*80}")
    print(f"Analyzing log file: {log_file_path.name}")
    print(f"{'='*80}\n")
    
    # Parse log file to get all metadata
    info = parse_log_file(log_file_path)
    
    print("Parsed from log file:")
    print(f"  batch_size: {info.get('batch_size', 'NOT FOUND')}")
    print(f"  input_len: {info.get('input_len', 'NOT FOUND')}")
    print(f"  output_len: {info.get('output_len', 'NOT FOUND')}")
    print(f"  pt_file: {info['pt_file_path'].name}")
    
    if 'prefill_tokens' in info:
        print(f"  prefill_tokens: {info['prefill_tokens']} (expected: {info['batch_size'] * info['input_len']})")
        print(f"  prefill_match: {'✓' if info['prefill_match'] else '✗'}")
    
    if 'warmup_tokens' in info:
        print(f"  warmup: {info['warmup_tokens']['input']} input, {info['warmup_tokens']['output']} output")
    
    # Load .pt file
    pt_file_path = info['pt_file_path']
    if not pt_file_path.exists():
        raise FileNotFoundError(f".pt file not found: {pt_file_path}")
    
    print(f"\nLoading expert distribution data from: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')
    
    records = data['records']
    print(f"Processing {len(records)} records...")
    
    # Calculate prefill threshold
    if 'batch_size' in info and 'input_len' in info:
        prefill_threshold = info['batch_size'] * info['input_len'] * 0.5
        print(f"Using log-based prefill threshold: {prefill_threshold:.0f} tokens")
    else:
        # Fallback: auto-detect
        record_sizes = [record['topk_ids_of_layer'].shape[1] for record in records]
        sorted_sizes = sorted(record_sizes)
        gaps = [sorted_sizes[i+1] - sorted_sizes[i] for i in range(len(sorted_sizes)-1)]
        max_gap_idx = gaps.index(max(gaps))
        prefill_threshold = (sorted_sizes[max_gap_idx] + sorted_sizes[max_gap_idx+1]) / 2
        print(f"Using auto-detected prefill threshold: {prefill_threshold:.0f} tokens")
    
    # Separate prefill and decode records
    prefill_topk_ids = []
    decode_topk_ids = []
    decode_batch_sizes = []
    
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens_in_record = topk_ids_of_layer.shape[1]
        
        # Use forward_mode if available (1=prefill, 2=decode), otherwise use threshold
        if 'forward_mode' in record:
            is_prefill = (record['forward_mode'] == 1)
        else:
            is_prefill = (num_tokens_in_record > prefill_threshold)
        
        if is_prefill:
            prefill_topk_ids.append(topk_ids_of_layer)
        else:
            decode_topk_ids.append(topk_ids_of_layer)
            decode_batch_sizes.append(num_tokens_in_record)
    
    print(f"  Prefill records: {len(prefill_topk_ids)}")
    print(f"  Decode records: {len(decode_topk_ids)}")
    
    if not decode_topk_ids:
        print("\nERROR: No decode records found!")
        return None
    
    decode_data = torch.cat(decode_topk_ids, dim=1)
    print(f"  Decode tokens: {decode_data.shape[1]}")
    
    # Verify batch size from data
    batch_size_counts = Counter(decode_batch_sizes)
    BATCH_SIZE = batch_size_counts.most_common(1)[0][0]
    
    if 'batch_size' in info and BATCH_SIZE != info['batch_size']:
        print(f"\nWARNING: Batch size mismatch!")
        print(f"  From log: {info['batch_size']}")
        print(f"  From data: {BATCH_SIZE}")
        print(f"  Using data-inferred: {BATCH_SIZE}")
    else:
        print(f"\nBatch size (inferred from decode records): {BATCH_SIZE}")
    
    # Analyze decode phase
    all_topk_ids = decode_data
    num_layers, num_tokens, topk = all_topk_ids.shape
    
    print(f"\nModel Configuration:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total decode tokens: {num_tokens}")
    print(f"  Top-K experts per token: {topk}")
    print(f"  Batch size (tokens per decode step): {BATCH_SIZE}")
    
    # Reshape into decode steps
    num_decode_steps = num_tokens // BATCH_SIZE
    remaining_tokens = num_tokens % BATCH_SIZE
    
    if remaining_tokens > 0:
        print(f"  Note: {remaining_tokens} remaining tokens (incomplete step), will be excluded")
        all_topk_ids = all_topk_ids[:, :num_tokens - remaining_tokens, :]
        num_tokens = num_tokens - remaining_tokens
        num_decode_steps = num_tokens // BATCH_SIZE
    
    print(f"  Number of decode steps: {num_decode_steps}")
    
    # Reshape: [num_layers, num_decode_steps, BATCH_SIZE, topk]
    reshaped = all_topk_ids.view(num_layers, num_decode_steps, BATCH_SIZE, topk)
    
    # Create histogram per layer
    print(f"\n{'='*80}")
    print("HISTOGRAM-BASED ANALYSIS")
    print(f"{'='*80}\n")
    print("For each decode step, create histogram of expert activations")
    print(f"Then average across all {num_decode_steps} decode steps\n")
    
    histogram_data = []
    
    print(f"Processing layers: ", end='', flush=True)
    
    for layer_idx in range(num_layers):
        if layer_idx % 5 == 0 or layer_idx < 3 or layer_idx >= num_layers - 3:
            print(f"{layer_idx+1}/{num_layers}... ", end='', flush=True)
        
        layer_histograms = []
        
        # For each decode step, create histogram
        for step_idx in range(num_decode_steps):
            step_data = reshaped[layer_idx, step_idx]  # [BATCH_SIZE, topk]
            all_expert_ids = step_data.flatten().tolist()
            
            expert_counts = defaultdict(int)
            for expert_id in all_expert_ids:
                expert_counts[expert_id] += 1
            
            activation_histogram = defaultdict(int)
            for expert_id, count in expert_counts.items():
                activation_histogram[count] += 1
            
            layer_histograms.append(activation_histogram)
        
        # Average histograms
        all_activation_counts = set()
        for hist in layer_histograms:
            all_activation_counts.update(hist.keys())
        
        averaged_histogram = {}
        for act_count in all_activation_counts:
            counts = [hist.get(act_count, 0) for hist in layer_histograms]
            averaged_histogram[act_count] = np.mean(counts)
        
        # Round to integers with residual correction to ensure:
        #   sum(activations_per_expert * num_experts) == batch_size * top_k
        #
        # Problem: Averaging across decode steps produces fractional expert counts.
        #   Naive round() causes the total activations to drift from the exact target.
        #
        # Solution — 3-step greedy fill + residual correction:
        #   Step 1: Greedy fill — round each bucket, accept only if within budget.
        #   Step 2: Partial fit — if a rounded bucket overshoots, take as many
        #           experts as fit in the remaining budget.
        #   Step 3: Residual adjustment — if still under target, add +1 expert
        #           to the smallest bucket that fits, repeating until exact.
        
        target_total = BATCH_SIZE * topk
        rounded_histogram = {}
        current_total = 0
        
        # Step 1 & 2: Greedy fill with partial-fit fallback.
        # Process most frequent patterns first (largest avg_num_experts)
        # so the most representative buckets are filled before budget runs out.
        sorted_by_frequency = sorted(averaged_histogram.items(), key=lambda x: x[1], reverse=True)
        
        for activation_count, avg_num_experts in sorted_by_frequency:
            rounded_experts = round(avg_num_experts)
            
            if rounded_experts > 0:  # Only include if at least 1 expert
                contribution = activation_count * rounded_experts
                # Step 1: Full fit — add entire rounded bucket if within budget
                if current_total + contribution <= target_total:
                    rounded_histogram[activation_count] = rounded_experts
                    current_total += contribution
                # Step 2: Partial fit — bucket overshoots, take what fits
                elif current_total < target_total:
                    remaining = target_total - current_total
                    if remaining >= activation_count:
                        final_experts = remaining // activation_count
                        if final_experts > 0:
                            rounded_histogram[activation_count] = final_experts
                            current_total += activation_count * final_experts
        
        # Step 3: Residual adjustment — fill any gap left by rounding.
        # Add +1 expert to the smallest existing bucket whose activation count
        # fits in the remaining budget. If no existing bucket works, try
        # buckets from the original averaged histogram that were rounded to 0.
        current_total = sum(k * v for k, v in rounded_histogram.items())
        
        while current_total < target_total:
            remaining = target_total - current_total
            added = False
            # Try existing buckets (smallest activation count first)
            for act_count in sorted(rounded_histogram.keys()):
                if act_count <= remaining:
                    rounded_histogram[act_count] += 1
                    current_total += act_count
                    added = True
                    break
            
            # Try previously-unused buckets (rounded to 0 earlier)
            if not added:
                for act_count in sorted(averaged_histogram.keys()):
                    if act_count not in rounded_histogram and act_count <= remaining:
                        rounded_histogram[act_count] = 1
                        current_total += act_count
                        added = True
                        break
            
            if not added:
                break  # Can't reach target exactly
        
        final_total = sum(k * v for k, v in rounded_histogram.items())
        
        if layer_idx < 3 or layer_idx >= num_layers - 3:
            print(f"\n\nLayer {layer_idx:2d}:")
            print(f"  Histogram (rounded integers, averaged across {num_decode_steps} decode steps):")
            for activation_count in sorted(rounded_histogram.keys()):
                num_experts = rounded_histogram[activation_count]
                print(f"    {activation_count:2d} activations/expert: {num_experts:3d} experts")
            
            total_experts = sum(rounded_histogram.values())
            print(f"  Total unique experts: {total_experts}")
            print(f"  Total activations: {final_total} (expected: {target_total})")
            
            if final_total == target_total:
                print(f"  ✓ EXACT match")
            else:
                print(f"  ✗ ERROR: {final_total - target_total}")
        
        # Save to histogram data
        for activation_count in sorted(rounded_histogram.keys()):
            num_experts = rounded_histogram[activation_count]
            histogram_data.append({
                'Layer': layer_idx,
                'Activations_Per_Expert': activation_count,
                'Num_Experts': num_experts,
                'Batch_Size': BATCH_SIZE,
                'TopK': topk,
                'Dtype': info.get('dtype'),
                'Quantization': info.get('quantization') if info.get('quantization') else ''
            })
    
    print("\n")
    
    # Create DataFrame
    df = pd.DataFrame(histogram_data)
    
    if return_dataframe:
        print(f"Returning DataFrame with {len(df)} rows (Batch_Size={BATCH_SIZE})")
        return df
    
    # Save to CSV
    output_dir = log_file_path.parent / "statistics"
    output_dir.mkdir(exist_ok=True)
    parent_tag = log_file_path.parent.name
    output_file = output_dir / f'{parent_tag}_decode.csv'
    df.to_csv(output_file, index=False)

    print(f"✓ Saved decode distribution to: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Format: Layer, Activations_Per_Expert, Num_Experts, Batch_Size, TopK")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze decode expert distribution from SGLang benchmark log file(s)'
    )
    parser.add_argument('log_files', nargs='+', help='Path(s) to SGLang benchmark log file(s)')
    parser.add_argument('--return-dataframe', action='store_true',
                       help='Return DataFrame instead of saving (for multi-file processing)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for combined CSV (default: first log file directory/statistics)')
    
    args = parser.parse_args()
    
    # Process files and combine
    print(f"Processing {len(args.log_files)} log file(s)...\n")
    all_dfs = []
    
    for log_file in args.log_files:
        print(f"\n{'='*80}")
        print(f"Processing: {Path(log_file).name}")
        print(f"{'='*80}")
        df = analyze_decode_histogram(log_file, return_dataframe=True)
        all_dfs.append(df)
    
    if args.return_dataframe:
        if len(all_dfs) == 1:
            return all_dfs[0]
        return pd.concat(all_dfs, ignore_index=True)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.log_files[0]).parent / "statistics"
    
    output_dir.mkdir(exist_ok=True)
    parent_tag = Path(args.log_files[0]).parent.name
    output_file = output_dir / f'{parent_tag}_decode.csv'
    combined_df.to_csv(output_file, index=False)

    print(f"\n\n{'='*80}")
    print(f"✓ Saved decode distribution to: {output_file}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Batch sizes: {sorted(combined_df['Batch_Size'].unique())}")
    print(f"  Format: Layer, Activations_Per_Expert, Num_Experts, Batch_Size, TopK, Dtype, Quantization")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
