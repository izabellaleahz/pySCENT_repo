#!/usr/bin/env python3
"""
Comprehensive Benchmark: R vs GPU Implementation of SCENT
Tests performance and accuracy across different input sizes
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import time
import subprocess
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def convert_to_native_types(obj):
    """Convert NumPy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def generate_test_peak_info(
    original_file: str,
    output_file: str,
    n_pairs: int,
    seed: int = 42
) -> None:
    """Generate test peak_info file with specified number of pairs"""
    print(f"  Generating {output_file} with {n_pairs} pairs...")
    
    # Read original file
    df = pd.read_csv(original_file, sep='\t', header=None, names=['gene', 'peak'])
    
    # Sample n_pairs randomly with seed for reproducibility
    np.random.seed(seed)
    if n_pairs < len(df):
        sampled = df.sample(n=n_pairs, random_state=seed)
    else:
        sampled = df
    
    # Save to file
    sampled.to_csv(output_file, sep='\t', header=False, index=False)
    print(f"  ✓ Created {output_file}")


def run_implementation(
    implementation: str,
    dataset: str,
    processed_dir: str,
    peak_info_file: str,
    output_dir: str,
    chunk_id: str,
    ncores: int = 4,
    device: str = 'cuda',
    seed: int = 42
) -> Tuple[bool, float, str]:
    """
    Run R or GPU implementation
    
    Returns:
        (success, elapsed_time, stdout)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if implementation == 'R':
        # Try to find Rscript
        import shutil
        rscript_path = shutil.which('Rscript')
        if rscript_path is None:
            print("  ERROR: Rscript not found in PATH")
            print("  Please install R or load R module")
            return False, 0.0, "Rscript not found"
        
        cmd = [
            rscript_path, 'run_scent.R',
            '--dataset', dataset,
            '--processed_dir', processed_dir,
            '--peak_info', peak_info_file,
            '--output_dir', output_dir,
            '--chunk_id', chunk_id,
            '--ncores', str(ncores)
        ]
    elif implementation == 'GPU':
        cmd = [
            'python3', 'scent_gpu/run_scent_gpu.py',
            '--dataset', dataset,
            '--processed_dir', processed_dir,
            '--peak_info', peak_info_file,
            '--output_dir', output_dir,
            '--chunk_id', chunk_id,
            '--device', device,
            '--seed', str(seed)
        ]
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    print(f"  Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd='/home/izabellaz/bio_stats_project'
        )
        elapsed = time.time() - start_time
        return True, elapsed, result.stdout
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"  ERROR: {e.stderr[-500:]}")
        return False, elapsed, e.stderr


def load_results(results_dir: str, dataset: str, chunk_id: str) -> pd.DataFrame:
    """Load SCENT results from directory"""
    results_dir = Path(results_dir)
    pattern = f"{dataset}_chunk_{chunk_id}_complete.tsv.gz"
    results_file = results_dir / pattern
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    results = pd.read_csv(results_file, sep='\t', compression='gzip')
    return results


def compare_outputs(
    r_results: pd.DataFrame,
    gpu_results: pd.DataFrame,
    pair_count: int
) -> Dict:
    """
    Compare R and GPU outputs for accuracy
    
    Returns detailed comparison metrics
    """
    print(f"  Comparing outputs...")
    
    # Merge on gene-peak
    merged = r_results.merge(
        gpu_results,
        on=['gene', 'peak'],
        suffixes=('_r', '_gpu'),
        how='inner'
    )
    
    if len(merged) == 0:
        print("  ERROR: No common gene-peak pairs found!")
        return {'valid': False, 'error': 'No common pairs'}
    
    metrics = {
        'valid': True,
        'n_pairs': pair_count,
        'n_common': len(merged),
        'n_r_only': len(r_results) - len(merged),
        'n_gpu_only': len(gpu_results) - len(merged),
    }
    
    # Compare beta coefficients
    beta_diff = np.abs(merged['beta_r'] - merged['beta_gpu'])
    metrics['beta_mae'] = beta_diff.mean()
    metrics['beta_max_diff'] = beta_diff.max()
    metrics['beta_corr'] = np.corrcoef(merged['beta_r'], merged['beta_gpu'])[0,1]
    metrics['beta_rmse'] = np.sqrt(np.mean(beta_diff**2))
    
    # Compare standard errors
    if 'se_r' in merged.columns and 'se_gpu' in merged.columns:
        se_diff = np.abs(merged['se_r'] - merged['se_gpu'])
        metrics['se_mae'] = se_diff.mean()
        metrics['se_max_diff'] = se_diff.max()
    
    # Compare p-values
    if 'p_r' in merged.columns and 'p_gpu' in merged.columns:
        p_diff = np.abs(merged['p_r'] - merged['p_gpu'])
        metrics['p_mae'] = p_diff.mean()
        metrics['p_max_diff'] = p_diff.max()
    
    # Compare bootstrap p-values
    if 'boot_basic_p_r' in merged.columns and 'boot_basic_p_gpu' in merged.columns:
        boot_diff = np.abs(merged['boot_basic_p_r'] - merged['boot_basic_p_gpu'])
        metrics['boot_p_mae'] = boot_diff.mean()
        metrics['boot_p_max_diff'] = boot_diff.max()
        metrics['boot_p_corr'] = np.corrcoef(
            merged['boot_basic_p_r'],
            merged['boot_basic_p_gpu']
        )[0,1]
    
    # Compare FDR
    if 'FDR_r' in merged.columns and 'FDR_gpu' in merged.columns:
        fdr_diff = np.abs(merged['FDR_r'] - merged['FDR_gpu'])
        metrics['fdr_mae'] = fdr_diff.mean()
        metrics['fdr_max_diff'] = fdr_diff.max()
        
        # Agreement on significant hits
        fdr_threshold = 0.10
        r_sig = merged['FDR_r'] <= fdr_threshold
        gpu_sig = merged['FDR_gpu'] <= fdr_threshold
        metrics['n_r_sig'] = r_sig.sum()
        metrics['n_gpu_sig'] = gpu_sig.sum()
        metrics['n_both_sig'] = (r_sig & gpu_sig).sum()
        metrics['n_either_sig'] = (r_sig | gpu_sig).sum()
        
        # Jaccard similarity for significant hits
        if metrics['n_either_sig'] > 0:
            metrics['sig_jaccard'] = metrics['n_both_sig'] / metrics['n_either_sig']
        else:
            metrics['sig_jaccard'] = 1.0
    
    # Overall accuracy assessment
    # Consider outputs "matching" if beta correlation > 0.99
    metrics['outputs_match'] = metrics['beta_corr'] > 0.99
    
    print(f"  ✓ Comparison complete:")
    print(f"    Beta MAE: {metrics['beta_mae']:.6f}, Corr: {metrics['beta_corr']:.6f}")
    if 'boot_p_corr' in metrics:
        print(f"    Boot P Corr: {metrics['boot_p_corr']:.6f}")
    if 'sig_jaccard' in metrics:
        print(f"    Significant hits Jaccard: {metrics['sig_jaccard']:.4f}")
    
    return metrics


def run_benchmark_case(
    n_pairs: int,
    dataset: str,
    processed_dir: str,
    original_peak_info: str,
    output_base: str,
    device: str = 'cuda',
    seed: int = 42,
    ncores: int = 4
) -> Dict:
    """
    Run a single benchmark case with n_pairs
    
    Returns dictionary with timing and accuracy results
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {n_pairs} pairs")
    print(f"{'='*80}")
    
    # Generate test peak_info
    test_peak_info = f"{output_base}/peak_info_{n_pairs}pairs.tsv"
    generate_test_peak_info(original_peak_info, test_peak_info, n_pairs, seed)
    
    chunk_id = f"bench_{n_pairs}"
    
    # Setup output directories
    output_dir_r = f"{output_base}/R/{dataset}"
    output_dir_gpu = f"{output_base}/GPU/{dataset}"
    
    results = {
        'n_pairs': n_pairs,
        'dataset': dataset,
        'device': device,
        'seed': seed,
    }
    
    # Run R implementation
    print(f"\n[1/2] Running R implementation...")
    r_success, r_time, r_stdout = run_implementation(
        'R', dataset, processed_dir, test_peak_info,
        output_dir_r, chunk_id, ncores=ncores
    )
    results['r_success'] = r_success
    results['r_time'] = r_time
    
    if not r_success:
        print(f"  ✗ R implementation failed!")
        return results
    
    print(f"  ✓ R completed in {r_time:.2f}s")
    
    # Run GPU implementation
    print(f"\n[2/2] Running GPU implementation...")
    gpu_success, gpu_time, gpu_stdout = run_implementation(
        'GPU', dataset, processed_dir, test_peak_info,
        output_dir_gpu, chunk_id, device=device, seed=seed
    )
    results['gpu_success'] = gpu_success
    results['gpu_time'] = gpu_time
    
    if not gpu_success:
        print(f"  ✗ GPU implementation failed!")
        return results
    
    print(f"  ✓ GPU completed in {gpu_time:.2f}s")
    
    # Calculate speedup
    speedup = r_time / gpu_time
    results['speedup'] = speedup
    print(f"\n  Speedup: {speedup:.2f}x (R: {r_time:.2f}s → GPU: {gpu_time:.2f}s)")
    
    # Compare outputs
    print(f"\n[3/3] Comparing outputs...")
    try:
        r_results = load_results(output_dir_r, dataset, chunk_id)
        gpu_results = load_results(output_dir_gpu, dataset, chunk_id)
        
        comparison = compare_outputs(r_results, gpu_results, n_pairs)
        results.update(comparison)
        
    except Exception as e:
        print(f"  ERROR comparing outputs: {e}")
        import traceback
        traceback.print_exc()
        results['valid'] = False
        results['error'] = str(e)
    
    print(f"\n{'='*80}")
    if results.get('outputs_match', False):
        print(f"✓ BENCHMARK PASSED: {n_pairs} pairs")
    else:
        print(f"⚠ BENCHMARK WARNING: {n_pairs} pairs - outputs differ")
    print(f"{'='*80}")
    
    return results


def plot_results(all_results: List[Dict], output_dir: str):
    """Generate comprehensive plots comparing R vs GPU"""
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    
    # Filter to successful runs
    df_success = df[df['r_success'] & df['gpu_success']].copy()
    
    if len(df_success) == 0:
        print("  ERROR: No successful runs to plot!")
        return
    
    print(f"  Plotting {len(df_success)} successful benchmarks...")
    
    # ========================================================================
    # Figure 1: Execution Time Comparison
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('R vs GPU Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Absolute execution time
    ax = axes[0, 0]
    x = np.arange(len(df_success))
    width = 0.35
    ax.bar(x - width/2, df_success['r_time'], width, label='R', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, df_success['gpu_time'], width, label='GPU', alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Number of Gene-Peak Pairs')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Absolute Execution Time')
    ax.set_xticks(x)
    ax.set_xticklabels(df_success['n_pairs'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log scale time
    ax = axes[0, 1]
    ax.plot(df_success['n_pairs'], df_success['r_time'], 
            'o-', label='R', linewidth=2, markersize=8, color='#3498db')
    ax.plot(df_success['n_pairs'], df_success['gpu_time'], 
            's-', label='GPU', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Number of Gene-Peak Pairs')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time (Log-Log Scale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Speedup
    ax = axes[1, 0]
    colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in df_success['speedup']]
    ax.bar(x, df_success['speedup'], color=colors, alpha=0.8)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='1x (no speedup)')
    ax.set_xlabel('Number of Gene-Peak Pairs')
    ax.set_ylabel('Speedup (R time / GPU time)')
    ax.set_title('GPU Speedup Factor')
    ax.set_xticks(x)
    ax.set_xticklabels(df_success['n_pairs'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for i, (xi, yi) in enumerate(zip(x, df_success['speedup'])):
        ax.text(xi, yi + 0.05, f'{yi:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Time per pair
    ax = axes[1, 1]
    df_success['r_time_per_pair'] = df_success['r_time'] / df_success['n_pairs']
    df_success['gpu_time_per_pair'] = df_success['gpu_time'] / df_success['n_pairs']
    
    ax.plot(df_success['n_pairs'], df_success['r_time_per_pair'] * 1000,
            'o-', label='R', linewidth=2, markersize=8, color='#3498db')
    ax.plot(df_success['n_pairs'], df_success['gpu_time_per_pair'] * 1000,
            's-', label='GPU', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Number of Gene-Peak Pairs')
    ax.set_ylabel('Time per Pair (milliseconds)')
    ax.set_title('Computational Cost per Gene-Peak Pair')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_path / 'benchmark_performance.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {plot_file}")
    plt.close()
    
    # ========================================================================
    # Figure 2: Output Accuracy Comparison
    # ========================================================================
    df_valid = df_success[df_success['valid']].copy()
    
    if len(df_valid) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('R vs GPU Output Accuracy Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Beta coefficient correlation
        ax = axes[0, 0]
        ax.plot(df_valid['n_pairs'], df_valid['beta_corr'],
                'o-', linewidth=2, markersize=10, color='#9b59b6')
        ax.axhline(y=0.99, color='red', linestyle='--', linewidth=1, 
                   label='0.99 (excellent agreement)')
        ax.set_xlabel('Number of Gene-Peak Pairs')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Beta Coefficient Correlation (R vs GPU)')
        ax.set_ylim([0.98, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Beta MAE
        ax = axes[0, 1]
        ax.plot(df_valid['n_pairs'], df_valid['beta_mae'],
                's-', linewidth=2, markersize=10, color='#e67e22')
        ax.set_xlabel('Number of Gene-Peak Pairs')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Beta Coefficient MAE (R vs GPU)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Bootstrap p-value correlation
        if 'boot_p_corr' in df_valid.columns:
            ax = axes[1, 0]
            ax.plot(df_valid['n_pairs'], df_valid['boot_p_corr'],
                    'o-', linewidth=2, markersize=10, color='#1abc9c')
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1,
                       label='0.95 (good agreement)')
            ax.set_xlabel('Number of Gene-Peak Pairs')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title('Bootstrap P-value Correlation (R vs GPU)')
            ax.set_ylim([0.9, 1.0])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Significant hits agreement
        if 'sig_jaccard' in df_valid.columns:
            ax = axes[1, 1]
            ax.plot(df_valid['n_pairs'], df_valid['sig_jaccard'],
                    's-', linewidth=2, markersize=10, color='#f39c12')
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1,
                       label='0.95 (high agreement)')
            ax.set_xlabel('Number of Gene-Peak Pairs')
            ax.set_ylabel('Jaccard Similarity')
            ax.set_title('Agreement on Significant Hits (FDR ≤ 0.10)')
            ax.set_ylim([0.8, 1.0])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_path / 'benchmark_accuracy.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {plot_file}")
        plt.close()
    
    # ========================================================================
    # Figure 3: Summary Table
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary table data
    table_data = []
    for _, row in df_success.iterrows():
        table_data.append([
            f"{row['n_pairs']:,}",
            f"{row['r_time']:.2f}s",
            f"{row['gpu_time']:.2f}s",
            f"{row['speedup']:.2f}x",
            f"{row.get('beta_corr', np.nan):.6f}",
            f"{row.get('boot_p_corr', np.nan):.6f}",
            f"{row.get('sig_jaccard', np.nan):.4f}",
            "✓" if row.get('outputs_match', False) else "⚠"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Pairs', 'R Time', 'GPU Time', 'Speedup', 
                   'Beta Corr', 'Boot P Corr', 'Sig Jaccard', 'Match'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Benchmark Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plot_file = output_path / 'benchmark_summary_table.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {plot_file}")
    plt.close()
    
    print(f"\n{'='*80}")
    print("✓ All plots generated successfully")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark: R vs GPU SCENT implementation'
    )
    parser.add_argument('--dataset', type=str, default='arthritis-tissue',
                      help='Dataset name')
    parser.add_argument('--processed_dir', type=str, 
                      default='/home/izabellaz/bio_stats_project/data/processed',
                      help='Processed data directory')
    parser.add_argument('--original_peak_info', type=str,
                      default=None,
                      help='Original peak_info file for sampling')
    parser.add_argument('--output_dir', type=str,
                      default='/home/izabellaz/bio_stats_project/benchmark_results',
                      help='Output directory for benchmark results')
    parser.add_argument('--sizes', nargs='+', type=int,
                      default=[10, 100, 1000, 10000],
                      help='Number of pairs to test')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--ncores', type=int, default=4,
                      help='Number of cores for R')
    
    args = parser.parse_args()
    
    # Auto-detect original peak_info if not provided
    if args.original_peak_info is None:
        peak_info_path = Path(args.processed_dir) / args.dataset / 'peak_lists'
        peak_info_files = list(peak_info_path.glob('peak_info_chunk_*.tsv'))
        if len(peak_info_files) == 0:
            print(f"ERROR: No peak_info files found in {peak_info_path}")
            return 1
        args.original_peak_info = str(sorted(peak_info_files)[0])
        print(f"Using peak_info: {args.original_peak_info}")
    
    # Print configuration
    print("="*80)
    print("BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Peak info: {args.original_peak_info}")
    print(f"Output dir: {args.output_dir}")
    print(f"Test sizes: {args.sizes}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"R cores: {args.ncores}")
    print("="*80)
    
    # Run benchmarks
    all_results = []
    
    for n_pairs in args.sizes:
        try:
            result = run_benchmark_case(
                n_pairs=n_pairs,
                dataset=args.dataset,
                processed_dir=args.processed_dir,
                original_peak_info=args.original_peak_info,
                output_base=args.output_dir,
                device=args.device,
                seed=args.seed,
                ncores=args.ncores
            )
            all_results.append(result)
            
            # Save intermediate results
            results_file = Path(args.output_dir) / 'benchmark_results.json'
            with open(results_file, 'w') as f:
                json.dump(convert_to_native_types(all_results), f, indent=2)
            
        except Exception as e:
            print(f"ERROR in benchmark case {n_pairs}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final results
    results_file = Path(args.output_dir) / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_to_native_types(all_results), f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    if len(all_results) > 0:
        plot_results(all_results, args.output_dir)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("\n⚠ No benchmark results!")
        return 1
    
    # Check if columns exist before filtering
    if 'r_success' in df.columns and 'gpu_success' in df.columns:
        df_success = df[df['r_success'] & df['gpu_success']]
    else:
        print("\n⚠ Benchmark runs did not complete successfully")
        return 1
    
    if len(df_success) > 0:
        print(f"\nSuccessful benchmarks: {len(df_success)}/{len(df)}")
        print(f"\nSpeedup statistics:")
        print(f"  Mean: {df_success['speedup'].mean():.2f}x")
        print(f"  Median: {df_success['speedup'].median():.2f}x")
        print(f"  Min: {df_success['speedup'].min():.2f}x")
        print(f"  Max: {df_success['speedup'].max():.2f}x")
        
        df_valid = df_success[df_success['valid']]
        if len(df_valid) > 0:
            print(f"\nAccuracy statistics:")
            print(f"  Beta correlation: {df_valid['beta_corr'].mean():.6f} (mean)")
            print(f"  Beta MAE: {df_valid['beta_mae'].mean():.6f} (mean)")
            if 'boot_p_corr' in df_valid.columns:
                print(f"  Boot P correlation: {df_valid['boot_p_corr'].mean():.6f} (mean)")
            if 'sig_jaccard' in df_valid.columns:
                print(f"  Significant hits Jaccard: {df_valid['sig_jaccard'].mean():.4f} (mean)")
            
            all_match = df_valid['outputs_match'].all()
            print(f"\n  All outputs match: {'✓ YES' if all_match else '⚠ NO'}")
    else:
        print("\n⚠ No successful benchmarks!")
    
    print("\n" + "="*80)
    print("✓ BENCHMARK COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

