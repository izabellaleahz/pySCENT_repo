#!/usr/bin/env python3
"""
Test GPU vs R implementation on a mini dataset (10 pairs)
Verifies that both produce the same results and clarifies FDR calculation scope
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import json
from pathlib import Path

def create_mini_peak_info(input_file, output_file, n_pairs=10, seed=42):
    """Create a mini peak info file with n_pairs random pairs"""
    print(f"Creating mini peak info file with {n_pairs} pairs...")
    
    # Read full peak info
    df = pd.read_csv(input_file, sep='\t')
    print(f"  Original file: {len(df)} pairs")
    
    # Sample n_pairs
    np.random.seed(seed)
    sample_idx = np.random.choice(len(df), size=min(n_pairs, len(df)), replace=False)
    mini_df = df.iloc[sample_idx].copy()
    
    # Save
    mini_df.to_csv(output_file, sep='\t', index=False)
    print(f"  Mini file saved: {len(mini_df)} pairs → {output_file}")
    
    return mini_df


def run_gpu_version(dataset, processed_dir, peak_info, output_dir, celltype=None):
    """Run GPU version"""
    print("\n" + "="*80)
    print("RUNNING GPU VERSION")
    print("="*80)
    
    cmd = [
        'python', 'scent_gpu/run_scent_gpu.py',
        '--dataset', dataset,
        '--processed_dir', processed_dir,
        '--peak_info', peak_info,
        '--output_dir', output_dir,
        '--chunk_id', 'mini_test',
        '--fdr_threshold', '0.1',
        '--device', 'cuda',
        '--seed', '42'
    ]
    
    if celltype:
        cmd.extend(['--celltypes', celltype])
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR running GPU version:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Load results
    result_file = f"{output_dir}/{dataset}_chunk_mini_test_complete.tsv.gz"
    if not Path(result_file).exists():
        print(f"ERROR: Result file not found: {result_file}")
        return None
    
    df = pd.read_csv(result_file, sep='\t', compression='gzip')
    print(f"\n✓ GPU results loaded: {len(df)} rows")
    
    return df


def run_r_version(dataset, processed_dir, peak_info, output_dir, celltype=None):
    """Run R version"""
    print("\n" + "="*80)
    print("RUNNING R VERSION")
    print("="*80)
    
    cmd = [
        'Rscript', 'run_scent.R',
        '--dataset', dataset,
        '--processed_dir', processed_dir,
        '--peak_info', peak_info,
        '--output_dir', output_dir,
        '--chunk_id', 'mini_test_r',
        '--fdr_threshold', '0.1',
        '--ncores', '4'
    ]
    
    if celltype:
        cmd.extend(['--celltypes', celltype])
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR running R version:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Load results
    result_file = f"{output_dir}/{dataset}_chunk_mini_test_r_complete.tsv.gz"
    if not Path(result_file).exists():
        print(f"ERROR: Result file not found: {result_file}")
        return None
    
    df = pd.read_csv(result_file, sep='\t', compression='gzip')
    print(f"\n✓ R results loaded: {len(df)} rows")
    
    return df


def compare_results(gpu_df, r_df, tolerance=1e-3):
    """Compare GPU and R results"""
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80)
    
    # Create pair IDs for matching
    gpu_df['pair_id'] = gpu_df['gene'] + '|' + gpu_df['peak']
    r_df['pair_id'] = r_df['gene'] + '|' + r_df['peak']
    
    # Find shared pairs
    gpu_pairs = set(gpu_df['pair_id'])
    r_pairs = set(r_df['pair_id'])
    
    shared_pairs = gpu_pairs & r_pairs
    gpu_only = gpu_pairs - r_pairs
    r_only = r_pairs - gpu_pairs
    
    print(f"\nPair coverage:")
    print(f"  GPU pairs: {len(gpu_pairs)}")
    print(f"  R pairs: {len(r_pairs)}")
    print(f"  Shared: {len(shared_pairs)}")
    print(f"  GPU only: {len(gpu_only)}")
    print(f"  R only: {len(r_only)}")
    
    if len(shared_pairs) == 0:
        print("\n⚠ WARNING: No shared pairs! Cannot compare.")
        return
    
    # Merge on shared pairs
    gpu_shared = gpu_df[gpu_df['pair_id'].isin(shared_pairs)].sort_values('pair_id').reset_index(drop=True)
    r_shared = r_df[r_df['pair_id'].isin(shared_pairs)].sort_values('pair_id').reset_index(drop=True)
    
    comparison = pd.DataFrame({
        'pair_id': gpu_shared['pair_id'],
        'gene': gpu_shared['gene'],
        'peak': gpu_shared['peak'],
        'gpu_beta': gpu_shared['beta'],
        'r_beta': r_shared['beta'],
        'gpu_p': gpu_shared['boot_basic_p'],
        'r_p': r_shared['boot_basic_p'],
        'gpu_FDR': gpu_shared['FDR'],
        'r_FDR': r_shared['FDR']
    })
    
    # Calculate differences
    comparison['beta_diff'] = comparison['gpu_beta'] - comparison['r_beta']
    comparison['beta_abs_diff'] = comparison['beta_diff'].abs()
    comparison['p_diff'] = comparison['gpu_p'] - comparison['r_p']
    comparison['p_abs_diff'] = comparison['p_diff'].abs()
    comparison['FDR_diff'] = comparison['gpu_FDR'] - comparison['r_FDR']
    comparison['FDR_abs_diff'] = comparison['FDR_diff'].abs()
    
    print(f"\n{'='*80}")
    print("COEFFICIENT COMPARISON (beta)")
    print(f"{'='*80}")
    print(f"  Mean absolute difference: {comparison['beta_abs_diff'].mean():.6f}")
    print(f"  Max absolute difference: {comparison['beta_abs_diff'].max():.6f}")
    print(f"  Correlation: {comparison['gpu_beta'].corr(comparison['r_beta']):.6f}")
    
    matches_beta = (comparison['beta_abs_diff'] < tolerance).sum()
    print(f"  Matches (|diff| < {tolerance}): {matches_beta}/{len(comparison)} ({100*matches_beta/len(comparison):.1f}%)")
    
    print(f"\n{'='*80}")
    print("BOOTSTRAP P-VALUE COMPARISON")
    print(f"{'='*80}")
    print(f"  Mean absolute difference: {comparison['p_abs_diff'].mean():.6f}")
    print(f"  Max absolute difference: {comparison['p_abs_diff'].max():.6f}")
    print(f"  Correlation: {comparison['gpu_p'].corr(comparison['r_p']):.6f}")
    
    matches_p = (comparison['p_abs_diff'] < 0.1).sum()
    print(f"  Matches (|diff| < 0.1): {matches_p}/{len(comparison)} ({100*matches_p/len(comparison):.1f}%)")
    
    print(f"\n{'='*80}")
    print("FDR COMPARISON")
    print(f"{'='*80}")
    print(f"  Mean absolute difference: {comparison['FDR_abs_diff'].mean():.6f}")
    print(f"  Max absolute difference: {comparison['FDR_abs_diff'].max():.6f}")
    print(f"  Correlation: {comparison['gpu_FDR'].corr(comparison['r_FDR']):.6f}")
    
    matches_fdr = (comparison['FDR_abs_diff'] < 0.01).sum()
    print(f"  Matches (|diff| < 0.01): {matches_fdr}/{len(comparison)} ({100*matches_fdr/len(comparison):.1f}%)")
    
    # Show examples
    print(f"\n{'='*80}")
    print("TOP 5 PAIRS (sorted by GPU beta)")
    print(f"{'='*80}")
    print(comparison[['gene', 'peak', 'gpu_beta', 'r_beta', 'beta_diff', 'gpu_p', 'r_p', 'gpu_FDR', 'r_FDR']].head().to_string(index=False))
    
    # Show largest differences
    print(f"\n{'='*80}")
    print("TOP 5 LARGEST BETA DIFFERENCES")
    print(f"{'='*80}")
    largest_diff = comparison.nlargest(5, 'beta_abs_diff')
    print(largest_diff[['gene', 'peak', 'gpu_beta', 'r_beta', 'beta_diff', 'gpu_p', 'r_p']].to_string(index=False))
    
    # Overall verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    beta_ok = comparison['beta_abs_diff'].mean() < tolerance
    p_ok = comparison['p_abs_diff'].mean() < 0.1
    fdr_ok = comparison['FDR_abs_diff'].mean() < 0.01
    
    if beta_ok and p_ok and fdr_ok:
        print("✓✓✓ EXCELLENT: GPU and R produce nearly identical results!")
    elif beta_ok and p_ok:
        print("✓✓ GOOD: GPU and R coefficients and p-values match closely.")
        print("   ⚠ FDR differences may be due to different number of tests or rounding.")
    elif beta_ok:
        print("✓ ACCEPTABLE: GPU and R coefficients match closely.")
        print("   ⚠ Bootstrap p-values differ (expected due to random sampling).")
    else:
        print("✗ POOR: GPU and R results differ significantly!")
        print("   Check implementation differences.")
    
    return comparison


def investigate_fdr_scope(results_file):
    """
    Investigate whether FDR is calculated per cell type or globally
    by checking if we can reverse-engineer the FDR calculation
    """
    print("\n" + "="*80)
    print("INVESTIGATING FDR SCOPE")
    print("="*80)
    print("Question: Is FDR calculated per cell type or globally across all cell types?")
    print("="*80)
    
    df = pd.read_csv(results_file, sep='\t', compression='gzip')
    
    if 'celltype' not in df.columns:
        print("⚠ No celltype column found. Cannot determine scope.")
        return
    
    celltypes = df['celltype'].unique()
    print(f"\nCell types in results: {', '.join(celltypes)} ({len(celltypes)} total)")
    
    # Test 1: Calculate per-celltype FDR and compare
    print(f"\n{'='*80}")
    print("TEST 1: Recalculate FDR per cell type")
    print(f"{'='*80}")
    
    from scipy.stats import false_discovery_control
    
    df['FDR_percelltype'] = np.nan
    
    for ct in celltypes:
        ct_mask = df['celltype'] == ct
        ct_pvals = df.loc[ct_mask, 'boot_basic_p'].values
        
        # Handle NaN
        ct_pvals = np.where(np.isnan(ct_pvals), 1.0, ct_pvals)
        ct_pvals = np.clip(ct_pvals, 0.0, 1.0)
        
        # Calculate FDR for this cell type only
        fdr_percelltype = false_discovery_control(ct_pvals, method='bh')
        df.loc[ct_mask, 'FDR_percelltype'] = fdr_percelltype
        
        print(f"  {ct}: {ct_mask.sum()} tests")
    
    # Test 2: Calculate global FDR and compare
    print(f"\n{'='*80}")
    print("TEST 2: Recalculate FDR globally (all cell types combined)")
    print(f"{'='*80}")
    
    all_pvals = df['boot_basic_p'].values.copy()
    all_pvals = np.where(np.isnan(all_pvals), 1.0, all_pvals)
    all_pvals = np.clip(all_pvals, 0.0, 1.0)
    
    fdr_global = false_discovery_control(all_pvals, method='bh')
    df['FDR_global'] = fdr_global
    
    print(f"  Total tests: {len(df)}")
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    # Check which matches better
    diff_percelltype = (df['FDR'] - df['FDR_percelltype']).abs().mean()
    diff_global = (df['FDR'] - df['FDR_global']).abs().mean()
    
    print(f"  Mean |diff| with per-celltype FDR: {diff_percelltype:.6e}")
    print(f"  Mean |diff| with global FDR: {diff_global:.6e}")
    
    if diff_global < 1e-10:
        print(f"\n  ✓✓✓ FDR is calculated GLOBALLY across all cell types!")
        print(f"      (difference is essentially zero: {diff_global:.6e})")
    elif diff_percelltype < 1e-10:
        print(f"\n  ✓✓✓ FDR is calculated PER CELL TYPE!")
        print(f"      (difference is essentially zero: {diff_percelltype:.6e})")
    elif diff_global < diff_percelltype:
        print(f"\n  ✓ FDR appears to be calculated GLOBALLY")
        print(f"    (global is {diff_percelltype/diff_global:.1f}x closer)")
    else:
        print(f"\n  ✓ FDR appears to be calculated PER CELL TYPE")
        print(f"    (per-celltype is {diff_global/diff_percelltype:.1f}x closer)")
    
    # Show examples
    print(f"\n{'='*80}")
    print("EXAMPLES (first 5 rows)")
    print(f"{'='*80}")
    print(df[['gene', 'peak', 'celltype', 'boot_basic_p', 'FDR', 'FDR_percelltype', 'FDR_global']].head(10).to_string(index=False))
    
    # Save detailed comparison
    output_file = results_file.replace('_complete.tsv.gz', '_fdr_scope_analysis.tsv.gz')
    df.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"\n✓ Saved detailed comparison: {output_file}")
    
    return df


def main():
    # Configuration
    dataset = 'arthritis-tissue'
    processed_dir = 'data/processed'
    full_peak_info = 'data/processed/arthritis-tissue/peak_lists/peak_info_combined.tsv'
    
    # Create directories
    mini_dir = Path('test_mini')
    mini_dir.mkdir(exist_ok=True)
    
    mini_peak_info = mini_dir / 'peak_info_mini_10.tsv'
    output_dir_gpu = mini_dir / 'gpu_results'
    output_dir_r = mini_dir / 'r_results'
    
    output_dir_gpu.mkdir(exist_ok=True)
    output_dir_r.mkdir(exist_ok=True)
    
    # Step 1: Create mini peak info
    print("\n" + "="*80)
    print("STEP 1: CREATE MINI DATASET")
    print("="*80)
    
    if not mini_peak_info.exists():
        mini_df = create_mini_peak_info(full_peak_info, mini_peak_info, n_pairs=10, seed=42)
    else:
        print(f"Using existing mini peak info: {mini_peak_info}")
        mini_df = pd.read_csv(mini_peak_info, sep='\t')
    
    print(f"\nMini peak info:")
    print(mini_df)
    
    # Step 2: Run GPU version
    gpu_results = run_gpu_version(
        dataset=dataset,
        processed_dir=processed_dir,
        peak_info=str(mini_peak_info),
        output_dir=str(output_dir_gpu),
        celltype='Endothelial'  # Single cell type for simplicity
    )
    
    if gpu_results is None:
        print("ERROR: GPU version failed")
        return 1
    
    # Step 3: Run R version
    r_results = run_r_version(
        dataset=dataset,
        processed_dir=processed_dir,
        peak_info=str(mini_peak_info),
        output_dir=str(output_dir_r),
        celltype='Endothelial'
    )
    
    if r_results is None:
        print("ERROR: R version failed")
        return 1
    
    # Step 4: Compare results
    comparison = compare_results(gpu_results, r_results, tolerance=1e-3)
    
    # Step 5: Investigate FDR scope
    gpu_complete_file = output_dir_gpu / f"{dataset}_chunk_mini_test_complete.tsv.gz"
    investigate_fdr_scope(gpu_complete_file)
    
    print("\n" + "="*80)
    print("✓ MINI TEST COMPLETE")
    print("="*80)
    print(f"Results saved in: {mini_dir}")
    print(f"\nTo investigate FDR scope on the full endothelial dataset:")
    print(f"  python -c \"from test_gpu_vs_r_mini import investigate_fdr_scope; investigate_fdr_scope('results/SCENT_GPU_full_run/arthritis_endothelial/arthritis_endothelial_full_combined_all_results.tsv.gz')\"")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

