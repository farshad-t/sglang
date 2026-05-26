#!/usr/bin/env python3
"""
Analyze expert distribution averaged across ALL layers (single aggregate per variant).

Produces two output files per variant:
  - *_decode_all_layers.csv: One histogram averaged across all layers
  - *_prefill_all_layers_5buckets.csv: One K-means bucketing averaged across all layers

Usage:
    python analyze_all_layers_averaged.py <log_file> [--num-buckets 5] [--output-dir DIR]

Example:
    python analyze_all_layers_averaged.py benchmark.log --output-dir results/
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_log_file(log_file_path):
    """Parse log file to extract metadata. Same as existing scripts."""
    log_file_path = Path(log_file_path)
    info = {}

    with open(log_file_path, 'r') as f:
        content = f.read()

    pt_match = re.search(r'Write expert distribution to (.+\.pt)', content)
    if pt_match:
        pt_path_str = pt_match.group(1).strip()
        pt_path = Path(pt_path_str)
        if not pt_path.is_absolute():
            log_dir_name = log_file_path.parent.name
            if pt_path.parts[0] == log_dir_name:
                pt_path = (log_file_path.parent.parent / pt_path_str).resolve()
            else:
                pt_path = (log_file_path.parent / pt_path_str).resolve()
        # If path doesn't exist, try remapping /code/ docker path to host path
        if not pt_path.exists() and '/code/' in pt_path_str:
            host_base = Path('/data/farshad/github/vllm-llama-endpoints/closed/Intel/code/endpoints')
            pt_path = (host_base / pt_path_str.replace('/code/', '', 1)).resolve()
        info['pt_file_path'] = pt_path
    else:
        raise ValueError(f"Could not find .pt file path in log: {log_file_path}")

    # dtype/quantization
    if 'dtype' not in info:
        quant_match = re.search(r'Load weight end\..*quant=(\w+)', content)
        if quant_match:
            quant_val = quant_match.group(1)
            if quant_val in ('None', 'none'):
                info['dtype'] = 'bfloat16'
                info['quantization'] = None
            else:
                info['dtype'] = quant_val
                info['quantization'] = quant_val
        else:
            dtype_match = re.search(r"dtype='([^']*)'", content)
            if dtype_match:
                info['dtype'] = dtype_match.group(1)
            else:
                info['dtype'] = 'bfloat16'
            quantization_match = re.search(r"quantization='([^']*)'", content)
            if quantization_match:
                quant = quantization_match.group(1)
                info['quantization'] = quant if quant else None
            else:
                info['quantization'] = None

    if info.get('quantization') and 'w8a8' in info['quantization']:
        info['dtype'] = 'int8'

    # batch_size from decode lines
    decode_bs_match = re.search(r'Decode \d+\. Batch size: (\d+)', content)
    if decode_bs_match:
        info['batch_size'] = int(decode_bs_match.group(1))

    # input/output from throughput lines
    if 'batch_size' in info:
        batch_size = info['batch_size']
        prefill_match_full = list(re.finditer(
            r'Prefill\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content))
        total_match = re.findall(
            r'Total\. latency:\s+([\d.]+) s, throughput:\s+([\d.]+) token/s', content)
        if prefill_match_full and total_match:
            last_prefill = prefill_match_full[-1]
            prefill_latency = float(last_prefill.group(1))
            prefill_throughput = float(last_prefill.group(2))
            prefill_tokens = int(round(prefill_throughput * prefill_latency))
            total_latency = float(total_match[-1][0])
            total_throughput = float(total_match[-1][1])
            total_tokens = int(round(total_throughput * total_latency))
            info['input_len'] = prefill_tokens // batch_size
            info['output_len'] = (total_tokens // batch_size) - info['input_len']

    # Fallback from filename
    if 'input_len' not in info:
        filename_match = re.search(r'input_len-(\d+)_output_len-(\d+)_conc-(\d+)', log_file_path.name)
        if filename_match:
            info['input_len'] = int(filename_match.group(1))
            info['output_len'] = int(filename_match.group(2))
            if 'batch_size' not in info:
                info['batch_size'] = int(filename_match.group(3))

    # Fallback batch_size
    if 'batch_size' not in info:
        batch_match = re.search(r'Successful requests:\s+(\d+)', content)
        if batch_match:
            info['batch_size'] = int(batch_match.group(1))
        dir_bs_match = re.search(r'_bs(\d+)', log_file_path.parent.name)
        if dir_bs_match:
            info['batch_size'] = int(dir_bs_match.group(1))

    return info


def simple_kmeans(data, k, max_iters=100, seed=42):
    """Simple K-means clustering."""
    if len(data) <= k:
        return np.arange(len(data))

    rng = np.random.RandomState(seed)
    centroids = [data[rng.randint(len(data))]]
    for _ in range(1, k):
        distances = np.min([np.abs(data - c) for c in centroids], axis=0)
        probs = (distances ** 2).astype(np.float64)
        probs /= probs.sum()
        centroids.append(data[rng.choice(len(data), p=probs)])
    centroids = np.array(centroids)

    for _ in range(max_iters):
        labels = np.argmin(np.abs(data[:, None] - centroids), axis=1)
        new_centroids = np.array([
            data[labels == i].mean() if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def analyze_all_layers_decode(records, info):
    """
    Analyze decode phase: average histogram across ALL layers.
    Returns a DataFrame with one aggregated histogram.
    """
    batch_size = info['batch_size']
    input_len = info.get('input_len', 1024)
    prefill_threshold = batch_size * input_len * 0.5

    decode_topk_ids = []
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens = topk_ids_of_layer.shape[1]
        if 'forward_mode' in record:
            is_prefill = (record['forward_mode'] == 1)
        else:
            is_prefill = (num_tokens > prefill_threshold)
        if not is_prefill:
            decode_topk_ids.append(topk_ids_of_layer)

    if not decode_topk_ids:
        print("ERROR: No decode records found!")
        return pd.DataFrame()

    decode_data = torch.cat(decode_topk_ids, dim=1)
    num_layers, num_tokens, topk_buf = decode_data.shape

    # Detect actual topk by checking for -1 padding
    sample = decode_data[0, 0]
    valid_mask = sample != -1
    topk = int(valid_mask.sum().item())
    if topk < topk_buf:
        decode_data = decode_data[:, :, :topk]

    # Infer batch size from decode record sizes
    decode_batch_sizes = [r['topk_ids_of_layer'].shape[1] for r in records
                          if r['topk_ids_of_layer'].shape[1] <= prefill_threshold]
    if decode_batch_sizes:
        BATCH_SIZE = Counter(decode_batch_sizes).most_common(1)[0][0]
    else:
        BATCH_SIZE = batch_size

    num_decode_steps = num_tokens // BATCH_SIZE
    remaining = num_tokens % BATCH_SIZE
    if remaining > 0:
        decode_data = decode_data[:, :num_tokens - remaining, :]
        num_tokens = num_tokens - remaining
        num_decode_steps = num_tokens // BATCH_SIZE

    print(f"  Decode: {num_layers} layers, {num_tokens} tokens, {num_decode_steps} steps, BS={BATCH_SIZE}, TopK={topk}")

    reshaped = decode_data.view(num_layers, num_decode_steps, BATCH_SIZE, topk)

    # Collect histograms from ALL layers and ALL steps, then average
    all_histograms = []
    total_steps = num_layers * num_decode_steps

    for layer_idx in range(num_layers):
        for step_idx in range(num_decode_steps):
            step_data = reshaped[layer_idx, step_idx].flatten().tolist()
            expert_counts = defaultdict(int)
            for eid in step_data:
                expert_counts[eid] += 1
            activation_histogram = defaultdict(int)
            for eid, count in expert_counts.items():
                activation_histogram[count] += 1
            all_histograms.append(activation_histogram)

    # Average across all (layers × steps)
    all_activation_counts = set()
    for hist in all_histograms:
        all_activation_counts.update(hist.keys())

    averaged_histogram = {}
    for act_count in all_activation_counts:
        counts = [hist.get(act_count, 0) for hist in all_histograms]
        averaged_histogram[act_count] = np.mean(counts)

    # Round with residual correction (same logic as per-layer script)
    target_total = BATCH_SIZE * topk
    rounded_histogram = {}
    current_total = 0

    sorted_by_frequency = sorted(averaged_histogram.items(), key=lambda x: x[1], reverse=True)
    for activation_count, avg_num_experts in sorted_by_frequency:
        rounded_experts = round(avg_num_experts)
        if rounded_experts > 0:
            contribution = activation_count * rounded_experts
            if current_total + contribution <= target_total:
                rounded_histogram[activation_count] = rounded_experts
                current_total += contribution
            elif current_total < target_total:
                remaining_budget = target_total - current_total
                if remaining_budget >= activation_count:
                    final_experts = remaining_budget // activation_count
                    if final_experts > 0:
                        rounded_histogram[activation_count] = final_experts
                        current_total += activation_count * final_experts

    current_total = sum(k * v for k, v in rounded_histogram.items())
    while current_total < target_total:
        remaining_budget = target_total - current_total
        added = False
        for act_count in sorted(rounded_histogram.keys()):
            if act_count <= remaining_budget:
                rounded_histogram[act_count] += 1
                current_total += act_count
                added = True
                break
        if not added:
            for act_count in sorted(averaged_histogram.keys()):
                if act_count not in rounded_histogram and act_count <= remaining_budget:
                    rounded_histogram[act_count] = 1
                    current_total += act_count
                    added = True
                    break
        if not added:
            break

    final_total = sum(k * v for k, v in rounded_histogram.items())
    print(f"  Decode all-layers histogram: {len(rounded_histogram)} bins, total_activations={final_total} (expected={target_total})")

    rows = []
    for act_count in sorted(rounded_histogram.keys()):
        rows.append({
            'Activations_Per_Expert': act_count,
            'Num_Experts': rounded_histogram[act_count],
            'Batch_Size': BATCH_SIZE,
            'TopK': topk,
            'Num_Layers': num_layers,
            'Num_Decode_Steps': num_decode_steps,
            'Dtype': info.get('dtype'),
            'Quantization': info.get('quantization') if info.get('quantization') else ''
        })

    return pd.DataFrame(rows)


def analyze_all_layers_prefill(records, info, num_buckets=5):
    """
    Analyze prefill phase: aggregate activations across ALL layers, then bucket once.
    Returns a DataFrame with one set of buckets.
    """
    batch_size = info['batch_size']
    input_len = info.get('input_len', 1024)
    prefill_threshold = batch_size * input_len * 0.5

    prefill_records = []
    for record in records:
        topk_ids_of_layer = record['topk_ids_of_layer']
        num_tokens = topk_ids_of_layer.shape[1]
        if 'forward_mode' in record:
            is_prefill = (record['forward_mode'] == 1)
        else:
            is_prefill = (num_tokens > prefill_threshold)
        if is_prefill:
            prefill_records.append(topk_ids_of_layer)

    if not prefill_records:
        print("ERROR: No prefill records found!")
        return pd.DataFrame()

    combined = torch.cat(prefill_records, dim=1)
    num_layers, num_tokens_prefill, topk_buf = combined.shape

    # Detect actual topk by checking for -1 padding
    sample = combined[0, 0]
    valid_mask = sample != -1
    topk = int(valid_mask.sum().item())
    if topk < topk_buf:
        print(f"  Detected padding: buffer width={topk_buf}, actual TopK={topk}")
        combined = combined[:, :, :topk]

    print(f"  Prefill: {num_layers} layers, {num_tokens_prefill} tokens, TopK={topk}")

    # Count activations per expert averaged across all layers
    # For each expert ID, sum activations across all layers, then divide by num_layers
    all_layer_counts = defaultdict(float)
    for layer_idx in range(num_layers):
        layer_data = combined[layer_idx].flatten().tolist()
        layer_counter = Counter(x for x in layer_data if x != -1)
        for eid, count in layer_counter.items():
            all_layer_counts[eid] += count

    # Average per layer
    for eid in all_layer_counts:
        all_layer_counts[eid] /= num_layers

    expert_ids = np.array(sorted(all_layer_counts.keys()))
    activation_counts = np.array([all_layer_counts[eid] for eid in expert_ids])

    print(f"  Unique experts: {len(expert_ids)}, avg activations/expert: {activation_counts.mean():.1f}")

    # K-means bucketing (try multiple methods, pick best)
    best_cv = float('inf')
    best_labels = None

    if len(expert_ids) >= num_buckets:
        # Method 1: absolute
        labels1 = simple_kmeans(activation_counts, num_buckets)
        cv1 = _compute_cv(activation_counts, labels1, num_buckets)
        if cv1 < best_cv:
            best_cv, best_labels = cv1, labels1

        # Method 2: log-space
        labels2 = simple_kmeans(np.log1p(activation_counts), num_buckets)
        cv2 = _compute_cv(activation_counts, labels2, num_buckets)
        if cv2 < best_cv:
            best_cv, best_labels = cv2, labels2

        # Method 3: quantile
        bucket_size = len(expert_ids) // num_buckets
        labels3 = np.zeros(len(expert_ids), dtype=int)
        sorted_indices = np.argsort(activation_counts)
        for i, idx in enumerate(sorted_indices):
            labels3[idx] = min(i // bucket_size, num_buckets - 1)
        cv3 = _compute_cv(activation_counts, labels3, num_buckets)
        if cv3 < best_cv:
            best_cv, best_labels = cv3, labels3

    if best_labels is None:
        best_labels = np.zeros(len(expert_ids), dtype=int)

    # Build bucket rows
    bucket_rows = []
    for bucket_id in range(num_buckets):
        mask = (best_labels == bucket_id)
        if not mask.any():
            continue
        bucket_experts = expert_ids[mask]
        bucket_activations = activation_counts[mask]
        bucket_rows.append({
            'bucket_id': bucket_id,
            'num_experts': len(bucket_experts),
            'avg_activations': round(bucket_activations.mean()),
            'min_activations': round(bucket_activations.min()),
            'max_activations': round(bucket_activations.max()),
        })

    # Enforce conservation: Σ(Avg_Activations × Num_Experts) >= target,
    # with minimum overshoot. Model is always at least as expensive as reality.
    target = num_tokens_prefill * topk
    actual = sum(r['avg_activations'] * r['num_experts'] for r in bucket_rows)
    delta = target - actual  # positive = under-count

    if delta > 0:
        best_bucket = min(bucket_rows,
                         key=lambda b: (b['num_experts'] - delta % b['num_experts']) % b['num_experts'])
        n = best_bucket['num_experts']
        best_bucket['avg_activations'] += (delta + n - 1) // n
    elif delta < 0:
        for r in sorted(bucket_rows, key=lambda b: b['num_experts']):
            if actual - r['num_experts'] >= target:
                r['avg_activations'] -= 1
                actual -= r['num_experts']
                if actual <= target + r['num_experts']:
                    break

    csv_total = sum(r['avg_activations'] * r['num_experts'] for r in bucket_rows)
    # Round Σ up to ceil(csv_total / K) * K so downstream reshape [-1, K, D] works.
    # Split 1 expert off a bucket and add the pad to that single expert.
    remainder = csv_total % topk
    if remainder != 0:
        pad = topk - remainder
        donor = max(bucket_rows, key=lambda b: b['num_experts'])
        donor['num_experts'] -= 1
        bucket_rows.append({
            'bucket_id': donor['bucket_id'],
            'num_experts': 1,
            'avg_activations': donor['avg_activations'] + pad,
            'min_activations': donor['min_activations'],
            'max_activations': donor['max_activations'],
        })
        csv_total += pad
    overshoot = csv_total - target
    assert csv_total >= target, f"Prefill all-layers: conservation violated {csv_total} < {target}"
    assert csv_total % topk == 0, f"Prefill all-layers: Σ(N×A)={csv_total} not divisible by K={topk}"

    rows = []
    for r in bucket_rows:
        rows.append({
            'Bucket': r['bucket_id'],
            'Num_Experts': r['num_experts'],
            'Avg_Activations': r['avg_activations'],
            'Min_Activations': r['min_activations'],
            'Max_Activations': r['max_activations'],
            'Batch_Size': batch_size,
            'TopK': topk,
            'Num_Layers': num_layers,
            'Num_Prefill_Tokens': num_tokens_prefill,
            'Dtype': info.get('dtype'),
            'Quantization': info.get('quantization') if info.get('quantization') else ''
        })

    print(f"  Prefill all-layers: {len(rows)} buckets, CV={best_cv:.4f}, "
          f"Σ(N×A)={csv_total} >= target={target}, overshoot={overshoot}")
    return pd.DataFrame(rows)


def _compute_cv(data, labels, k):
    """Weighted coefficient of variation across clusters."""
    totals = []
    for i in range(k):
        mask = (labels == i)
        totals.append(data[mask].sum() if mask.any() else 0)
    totals = np.array(totals)
    mean = totals.mean()
    if mean == 0:
        return float('inf')
    return totals.std() / mean


def main():
    parser = argparse.ArgumentParser(
        description='Analyze expert distribution averaged across ALL layers (one aggregate per variant)'
    )
    parser.add_argument('log_files', nargs='+', type=Path,
                        help='Path(s) to SGLang benchmark log file(s)')
    parser.add_argument('--num-buckets', type=int, default=5,
                        help='Number of buckets for prefill K-means (default: 5)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for CSVs (default: log file parent/statistics)')

    args = parser.parse_args()

    for log_file in args.log_files:
        print(f"\n{'='*80}")
        print(f"Processing: {log_file}")
        print(f"{'='*80}")

        try:
            info = parse_log_file(log_file)
            print(f"  batch_size={info.get('batch_size')}, input_len={info.get('input_len')}, "
                  f"output_len={info.get('output_len')}, dtype={info.get('dtype')}")

            if 'batch_size' not in info:
                print("  ERROR: Could not determine batch_size. Skipping.")
                continue

            pt_path = info['pt_file_path']
            if not pt_path.exists():
                print(f"  ERROR: .pt file not found: {pt_path}")
                continue

            data = torch.load(pt_path, map_location='cpu')
            records = data['records'] if isinstance(data, dict) else data

            # Determine output directory
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = log_file.parent / "statistics"
            output_dir.mkdir(exist_ok=True)

            parent_tag = log_file.parent.name

            # Decode analysis
            df_decode = analyze_all_layers_decode(records, info)
            if not df_decode.empty:
                out_decode = output_dir / f"{parent_tag}_decode_all_layers.csv"
                df_decode.to_csv(out_decode, index=False)
                print(f"  Saved: {out_decode}")

            # Prefill analysis
            df_prefill = analyze_all_layers_prefill(records, info, num_buckets=args.num_buckets)
            if not df_prefill.empty:
                out_prefill = output_dir / f"{parent_tag}_prefill_all_layers_{args.num_buckets}buckets.csv"
                df_prefill.to_csv(out_prefill, index=False)
                print(f"  Saved: {out_prefill}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
