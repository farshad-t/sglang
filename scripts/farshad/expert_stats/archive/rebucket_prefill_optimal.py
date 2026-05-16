#!/usr/bin/env python3
"""
Rebucket prefill expert activations into 5 optimal buckets.
Goal: Minimize variance within each bucket (group similar activation counts together).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def simple_kmeans(data, k, max_iter=100):
    """
    Simple K-means implementation without sklearn.
    """
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

def analyze_and_rebucket(csv_path, num_buckets=5):
    """
    Find optimal bucketing that minimizes within-bucket variance.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Activation range: {df['Activations_Per_Expert'].min()} to {df['Activations_Per_Expert'].max()}")
    
    # Get all activation counts (weighted by number of experts)
    activation_counts = []
    for _, row in df.iterrows():
        activation_counts.extend([row['Activations_Per_Expert']] * row['Num_Experts'])
    
    activation_counts = np.array(activation_counts)
    print(f"\nTotal expert instances: {len(activation_counts)}")
    print(f"Mean: {activation_counts.mean():.1f}, Std: {activation_counts.std():.1f}")
    print(f"Median: {np.median(activation_counts):.1f}")
    
    # Distribution analysis
    print(f"\nDistribution percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(activation_counts, p):.0f}")
    
    # ========================================================================
    # Approach 1: K-Means Clustering (optimal for minimizing variance)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"APPROACH 1: K-Means Clustering (k={num_buckets})")
    print(f"{'='*80}")
    
    # Fit K-means
    labels, centers = simple_kmeans(activation_counts, num_buckets)
    
    # Sort clusters by center value
    sorted_indices = np.argsort(centers)
    
    # Create bucket boundaries
    kmeans_buckets = []
    for i, cluster_idx in enumerate(sorted_indices):
        cluster_mask = labels == cluster_idx
        cluster_data = activation_counts[cluster_mask]
        
        bucket_min = cluster_data.min()
        bucket_max = cluster_data.max()
        bucket_mean = cluster_data.mean()
        bucket_median = np.median(cluster_data)
        bucket_std = cluster_data.std()
        num_experts = len(cluster_data)
        
        kmeans_buckets.append({
            'bucket_id': i,
            'min': bucket_min,
            'max': bucket_max,
            'mean': bucket_mean,
            'median': bucket_median,
            'std': bucket_std,
            'num_experts': num_experts,
            'center': centers[cluster_idx]
        })
        
        print(f"\nBucket {i}: [{bucket_min:.0f} - {bucket_max:.0f}] activations")
        print(f"  Experts: {num_experts:,}")
        print(f"  Mean: {bucket_mean:.1f} ± {bucket_std:.1f}")
        print(f"  Median: {bucket_median:.0f}")
        print(f"  Coefficient of Variation: {bucket_std/bucket_mean*100:.1f}%")
    
    # ========================================================================
    # Approach 2: Log-Space K-Means (optimal for minimizing CV)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"APPROACH 2: Log-Space K-Means (optimizes for relative differences, k={num_buckets})")
    print(f"{'='*80}")
    
    # Transform to log-space
    log_activations = np.log(activation_counts)
    
    # Fit K-means in log-space
    log_labels, log_centers = simple_kmeans(log_activations, num_buckets)
    
    # Sort clusters by center value
    sorted_indices_log = np.argsort(log_centers)
    
    # Create bucket boundaries
    log_kmeans_buckets = []
    for i, cluster_idx in enumerate(sorted_indices_log):
        cluster_mask = log_labels == cluster_idx
        cluster_data = activation_counts[cluster_mask]
        
        bucket_min = cluster_data.min()
        bucket_max = cluster_data.max()
        bucket_mean = cluster_data.mean()
        bucket_median = np.median(cluster_data)
        bucket_std = cluster_data.std()
        num_experts = len(cluster_data)
        
        log_kmeans_buckets.append({
            'bucket_id': i,
            'min': bucket_min,
            'max': bucket_max,
            'mean': bucket_mean,
            'median': bucket_median,
            'std': bucket_std,
            'num_experts': num_experts
        })
        
        print(f"\nBucket {i}: [{bucket_min:.0f} - {bucket_max:.0f}] activations")
        print(f"  Experts: {num_experts:,}")
        print(f"  Mean: {bucket_mean:.1f} ± {bucket_std:.1f}")
        print(f"  Median: {bucket_median:.0f}")
        print(f"  Coefficient of Variation: {bucket_std/bucket_mean*100:.1f}%")
    
    # ========================================================================
    # Approach 3: Quantile-based (equal number of experts per bucket)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"APPROACH 3: Quantile-based (equal experts per bucket)")
    print(f"{'='*80}")
    
    quantile_boundaries = np.percentile(activation_counts, np.linspace(0, 100, num_buckets+1))
    
    quantile_buckets = []
    for i in range(num_buckets):
        if i == num_buckets - 1:
            # Last bucket includes max
            bucket_mask = (activation_counts >= quantile_boundaries[i]) & (activation_counts <= quantile_boundaries[i+1])
        else:
            bucket_mask = (activation_counts >= quantile_boundaries[i]) & (activation_counts < quantile_boundaries[i+1])
        
        bucket_data = activation_counts[bucket_mask]
        
        if len(bucket_data) == 0:
            continue
        
        bucket_min = bucket_data.min()
        bucket_max = bucket_data.max()
        bucket_mean = bucket_data.mean()
        bucket_median = np.median(bucket_data)
        bucket_std = bucket_data.std()
        num_experts = len(bucket_data)
        
        quantile_buckets.append({
            'bucket_id': i,
            'min': bucket_min,
            'max': bucket_max,
            'mean': bucket_mean,
            'median': bucket_median,
            'std': bucket_std,
            'num_experts': num_experts
        })
        
        print(f"\nBucket {i}: [{bucket_min:.0f} - {bucket_max:.0f}] activations")
        print(f"  Experts: {num_experts:,}")
        print(f"  Mean: {bucket_mean:.1f} ± {bucket_std:.1f}")
        print(f"  Median: {bucket_median:.0f}")
        print(f"  Coefficient of Variation: {bucket_std/bucket_mean*100:.1f}%")
    
    # ========================================================================
    # Comparison: Which approach has lower CV (better for weight sharing)?
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    
    # Calculate standard metrics
    kmeans_total_variance = sum(b['std']**2 * b['num_experts'] for b in kmeans_buckets)
    kmeans_avg_cv = np.mean([b['std']/b['mean']*100 for b in kmeans_buckets])
    
    log_kmeans_total_variance = sum(b['std']**2 * b['num_experts'] for b in log_kmeans_buckets)
    log_kmeans_avg_cv = np.mean([b['std']/b['mean']*100 for b in log_kmeans_buckets])
    
    quantile_total_variance = sum(b['std']**2 * b['num_experts'] for b in quantile_buckets)
    quantile_avg_cv = np.mean([b['std']/b['mean']*100 for b in quantile_buckets])
    
    # Calculate COMPUTE-WEIGHTED CV (weight by total activations = compute volume)
    def compute_weighted_cv(buckets):
        total_compute = sum(b['mean'] * b['num_experts'] for b in buckets)
        weighted_cv = sum((b['std']/b['mean']*100) * (b['mean'] * b['num_experts']) / total_compute 
                         for b in buckets)
        return weighted_cv
    
    kmeans_weighted_cv = compute_weighted_cv(kmeans_buckets)
    log_kmeans_weighted_cv = compute_weighted_cv(log_kmeans_buckets)
    quantile_weighted_cv = compute_weighted_cv(quantile_buckets)
    
    print(f"\nApproach 1 - K-Means (absolute space):")
    print(f"  Total within-bucket variance: {kmeans_total_variance:,.0f}")
    print(f"  Average Coefficient of Variation: {kmeans_avg_cv:.1f}%")
    print(f"  Compute-Weighted CV (weight by total activations): {kmeans_weighted_cv:.1f}%")
    
    print(f"\nApproach 2 - K-Means (log-space):")
    print(f"  Total within-bucket variance: {log_kmeans_total_variance:,.0f}")
    print(f"  Average Coefficient of Variation: {log_kmeans_avg_cv:.1f}%")
    print(f"  Compute-Weighted CV (weight by total activations): {log_kmeans_weighted_cv:.1f}%")
    
    print(f"\nApproach 3 - Quantile:")
    print(f"  Total within-bucket variance: {quantile_total_variance:,.0f}")
    print(f"  Average Coefficient of Variation: {quantile_avg_cv:.1f}%")
    print(f"  Compute-Weighted CV (weight by total activations): {quantile_weighted_cv:.1f}%")
    
    # Choose best approach based on compute-weighted CV (most important for performance modeling)
    all_approaches = [
        ('kmeans', kmeans_buckets, kmeans_weighted_cv, kmeans_avg_cv),
        ('log_kmeans', log_kmeans_buckets, log_kmeans_weighted_cv, log_kmeans_avg_cv),
        ('quantile', quantile_buckets, quantile_weighted_cv, quantile_avg_cv)
    ]
    
    best_approach, best_buckets, best_weighted_cv, best_avg_cv = min(all_approaches, key=lambda x: x[2])
    
    print(f"\n✓ Best approach: {best_approach}")
    print(f"  → Compute-weighted CV: {best_weighted_cv:.1f}% (optimizes for high-compute buckets)")
    print(f"  → Simple average CV: {best_avg_cv:.1f}%")
    
    # ========================================================================
    # Apply bucketing to original CSV
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"REBUCKETING DATA (using {best_approach})")
    print(f"{'='*80}")
    
    rebucketed_data = []
    
    for layer in df['Layer'].unique():
        layer_df = df[df['Layer'] == layer]
        
        # For each bucket, aggregate experts
        for bucket_info in best_buckets:
            bucket_id = bucket_info['bucket_id']
            bucket_min = bucket_info['min']
            bucket_max = bucket_info['max']
            
            # Find experts in this bucket
            bucket_mask = (layer_df['Activations_Per_Expert'] >= bucket_min) & \
                          (layer_df['Activations_Per_Expert'] <= bucket_max)
            bucket_experts = layer_df[bucket_mask]
            
            if len(bucket_experts) == 0:
                continue
            
            # Sum up experts and compute average activations
            total_experts = bucket_experts['Num_Experts'].sum()
            total_activations = (bucket_experts['Activations_Per_Expert'] * bucket_experts['Num_Experts']).sum()
            avg_activations = total_activations / total_experts
            
            rebucketed_data.append({
                'Layer': layer,
                'Bucket_ID': bucket_id,
                'Bucket_Range': f"{bucket_min:.0f}-{bucket_max:.0f}",
                'Avg_Activations_Per_Expert': round(avg_activations),
                'Num_Experts': total_experts,
                'Total_Activations': total_activations,
                'Total_Prefill_Tokens': layer_df['Total_Prefill_Tokens'].iloc[0],
                'TopK': layer_df['TopK'].iloc[0]
            })
    
    rebucketed_df = pd.DataFrame(rebucketed_data)
    
    # Save
    output_csv = csv_path.parent / "expert_activation_histogram_prefill_5buckets.csv"
    rebucketed_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved rebucketed data to: {output_csv}")
    print(f"  Original rows: {len(df)}")
    print(f"  Rebucketed rows: {len(rebucketed_df)}")
    print(f"  Reduction: {(1-len(rebucketed_df)/len(df))*100:.1f}%")
    
    # Show sample
    print(f"\nSample of rebucketed data (Layer 0):")
    print(rebucketed_df[rebucketed_df['Layer'] == 0].to_string(index=False))
    
    # Verification per layer
    print(f"\nVerification (check if total activations match):")
    for layer in range(min(3, df['Layer'].max()+1)):
        original_total = (df[df['Layer'] == layer]['Activations_Per_Expert'] * \
                          df[df['Layer'] == layer]['Num_Experts']).sum()
        rebucketed_total = rebucketed_df[rebucketed_df['Layer'] == layer]['Total_Activations'].sum()
        
        if abs(original_total - rebucketed_total) < 1:
            status = "✓"
        else:
            status = "✗"
        
        print(f"  Layer {layer}: Original={original_total:.0f}, Rebucketed={rebucketed_total:.0f} {status}")
    
    return rebucketed_df, best_buckets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebucket prefill data into 5 optimal buckets")
    parser.add_argument("--csv", required=True, help="Path to expert_activation_histogram_prefill.csv")
    parser.add_argument("--num-buckets", type=int, default=5, help="Number of buckets (default: 5)")
    args = parser.parse_args()
    
    analyze_and_rebucket(args.csv, args.num_buckets)
