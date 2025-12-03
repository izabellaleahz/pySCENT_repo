#!/usr/bin/env python3
"""
Test and Validation Suite for GPU-Accelerated SCENT
Compare GPU results against R implementation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time
import subprocess


def load_results(results_dir: str, chunk_id: str = None) -> pd.DataFrame:
    """Load SCENT results from directory"""
    results_dir = Path(results_dir)
    
    # Find results files
    if chunk_id:
        pattern = f"*chunk_{chunk_id}_complete.tsv.gz"
    else:
        pattern = "*complete.tsv.gz"
    
    results_files = list(results_dir.glob(pattern))
    
    if len(results_files) == 0:
        raise FileNotFoundError(f"No results files found matching {pattern} in {results_dir}")
    
    if len(results_files) > 1:
        print(f"WARNING: Multiple results files found, using {results_files[0]}")
    
    results = pd.read_csv(results_files[0], sep='\t', compression='gzip')
    return results


def compare_results(
    r_results: pd.DataFrame,
    gpu_results: pd.DataFrame,
    tolerance_beta: float = 1e-4,
    tolerance_p: float = 1e-3,
    tolerance_boot_p: float = 0.1
) -> dict:
    """
    Compare R and GPU results for validation
    
    Returns:
        Dictionary with comparison metrics and pass/fail status
    """
    print("\n" + "="*80)
    print("COMPARING R vs GPU RESULTS")
    print("="*80)
    
    # Check dimensions
    print(f"\nDimensions:")
    print(f"  R results: {len(r_results)} pairs")
    print(f"  GPU results: {len(gpu_results)} pairs")
    
    # Merge on gene-peak
    merged = r_results.merge(
        gpu_results,
        on=['gene', 'peak'],
        suffixes=('_r', '_gpu'),
        how='inner'
    )
    
    print(f"  Common pairs: {len(merged)}")
    
    if len(merged) == 0:
        print("ERROR: No common gene-peak pairs found!")
        return {'passed': False, 'error': 'No common pairs'}
    
    # Compare coefficients (beta)
    print(f"\nBeta coefficients:")
    beta_diff = np.abs(merged['beta_r'] - merged['beta_gpu'])
    print(f"  Mean absolute difference: {beta_diff.mean():.6f}")
    print(f"  Max absolute difference: {beta_diff.max():.6f}")
    print(f"  Correlation: {np.corrcoef(merged['beta_r'], merged['beta_gpu'])[0,1]:.6f}")
    beta_pass = beta_diff.max() < tolerance_beta
    print(f"  PASS: {beta_pass} (tolerance: {tolerance_beta})")
    
    # Compare standard errors
    if 'se_r' in merged.columns and 'se_gpu' in merged.columns:
        print(f"\nStandard errors:")
        se_diff = np.abs(merged['se_r'] - merged['se_gpu'])
        print(f"  Mean absolute difference: {se_diff.mean():.6f}")
        print(f"  Max absolute difference: {se_diff.max():.6f}")
        se_pass = se_diff.max() < tolerance_beta
        print(f"  PASS: {se_pass} (tolerance: {tolerance_beta})")
    else:
        se_pass = None
    
    # Compare p-values
    if 'p_r' in merged.columns and 'p_gpu' in merged.columns:
        print(f"\nP-values:")
        p_diff = np.abs(merged['p_r'] - merged['p_gpu'])
        print(f"  Mean absolute difference: {p_diff.mean():.6f}")
        print(f"  Max absolute difference: {p_diff.max():.6f}")
        p_pass = p_diff.max() < tolerance_p
        print(f"  PASS: {p_pass} (tolerance: {tolerance_p})")
    else:
        p_pass = None
    
    # Compare bootstrap p-values (allow more variability due to sampling)
    print(f"\nBootstrap P-values:")
    boot_diff = np.abs(merged['boot_basic_p_r'] - merged['boot_basic_p_gpu'])
    print(f"  Mean absolute difference: {boot_diff.mean():.6f}")
    print(f"  Max absolute difference: {boot_diff.max():.6f}")
    print(f"  Correlation: {np.corrcoef(merged['boot_basic_p_r'], merged['boot_basic_p_gpu'])[0,1]:.6f}")
    boot_pass = boot_diff.mean() < tolerance_boot_p  # Use mean for bootstrap
    print(f"  PASS: {boot_pass} (mean tolerance: {tolerance_boot_p})")
    
    # Compare FDR if present
    if 'FDR_r' in merged.columns and 'FDR_gpu' in merged.columns:
        print(f"\nFDR:")
        fdr_diff = np.abs(merged['FDR_r'] - merged['FDR_gpu'])
        print(f"  Mean absolute difference: {fdr_diff.mean():.6f}")
        print(f"  Max absolute difference: {fdr_diff.max():.6f}")
    
    # Overall pass/fail
    checks = [beta_pass, boot_pass]
    if se_pass is not None:
        checks.append(se_pass)
    if p_pass is not None:
        checks.append(p_pass)
    
    all_passed = all(checks)
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print(f"{'='*80}")
    
    return {
        'passed': all_passed,
        'n_common_pairs': len(merged),
        'beta_max_diff': beta_diff.max(),
        'beta_corr': np.corrcoef(merged['beta_r'], merged['beta_gpu'])[0,1],
        'boot_p_mean_diff': boot_diff.mean(),
        'boot_p_max_diff': boot_diff.max(),
        'boot_p_corr': np.corrcoef(merged['boot_basic_p_r'], merged['boot_basic_p_gpu'])[0,1],
    }


def run_test_case(
    dataset: str,
    processed_dir: str,
    peak_info_file: str,
    output_dir_r: str,
    output_dir_gpu: str,
    chunk_id: str,
    device: str = 'cuda',
    run_r: bool = True,
    run_gpu: bool = True
) -> dict:
    """
    Run a test case: execute R and GPU implementations and compare
    
    Returns:
        Dictionary with results and timing information
    """
    results = {}
    
    # Run R implementation
    if run_r:
        print("\n" + "="*80)
        print("RUNNING R IMPLEMENTATION")
        print("="*80)
        
        r_cmd = [
            'Rscript', 'run_scent.R',
            '--dataset', dataset,
            '--processed_dir', processed_dir,
            '--peak_info', peak_info_file,
            '--output_dir', output_dir_r,
            '--chunk_id', chunk_id,
            '--ncores', '4'
        ]
        
        print(f"Command: {' '.join(r_cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(r_cmd, capture_output=True, text=True, check=True)
            r_time = time.time() - start_time
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(f"\n✓ R completed in {r_time:.1f} seconds")
            results['r_time'] = r_time
            results['r_success'] = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR running R: {e.stderr}")
            results['r_success'] = False
            return results
    
    # Run GPU implementation
    if run_gpu:
        print("\n" + "="*80)
        print("RUNNING GPU IMPLEMENTATION")
        print("="*80)
        
        gpu_cmd = [
            'python', 'scent_gpu/run_scent_gpu.py',
            '--dataset', dataset,
            '--processed_dir', processed_dir,
            '--peak_info', peak_info_file,
            '--output_dir', output_dir_gpu,
            '--chunk_id', chunk_id,
            '--device', device
        ]
        
        print(f"Command: {' '.join(gpu_cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, check=True)
            gpu_time = time.time() - start_time
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(f"\n✓ GPU completed in {gpu_time:.1f} seconds")
            results['gpu_time'] = gpu_time
            results['gpu_success'] = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR running GPU: {e.stderr}")
            results['gpu_success'] = False
            return results
    
    # Compare results
    if run_r and run_gpu:
        print("\n" + "="*80)
        print("LOADING RESULTS FOR COMPARISON")
        print("="*80)
        
        try:
            r_results = load_results(output_dir_r, chunk_id)
            gpu_results = load_results(output_dir_gpu, chunk_id)
            
            comparison = compare_results(r_results, gpu_results)
            results.update(comparison)
            
            # Calculate speedup
            if 'r_time' in results and 'gpu_time' in results:
                speedup = results['r_time'] / results['gpu_time']
                print(f"\nSpeedup: {speedup:.2f}x")
                results['speedup'] = speedup
                
        except Exception as e:
            print(f"ERROR comparing results: {e}")
            import traceback
            traceback.print_exc()
            results['passed'] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test GPU-Accelerated SCENT')
    parser.add_argument('--dataset', type=str, default='arthritis-tissue',
                      help='Dataset name')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Processed data directory')
    parser.add_argument('--peak_info', type=str, default=None,
                      help='Peak info file (auto-generated if not provided)')
    parser.add_argument('--chunk_id', type=str, default='test',
                      help='Chunk ID for test')
    parser.add_argument('--n_pairs', type=int, default=10,
                      help='Number of pairs to test (10, 100, or 10000)')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'])
    parser.add_argument('--skip_r', action='store_true',
                      help='Skip R implementation (only run GPU)')
    parser.add_argument('--skip_gpu', action='store_true',
                      help='Skip GPU implementation (only run R)')
    
    args = parser.parse_args()
    
    # Generate test peak_info if not provided
    if args.peak_info is None:
        original_peak_info = f"{args.processed_dir}/{args.dataset}/peak_lists/peak_info_chunk_0001.tsv"
        test_peak_info = f"test_{args.n_pairs}pairs.tsv"
        
        print(f"Generating test peak_info with {args.n_pairs} pairs...")
        subprocess.run(f"head -{args.n_pairs} {original_peak_info} > {test_peak_info}", 
                      shell=True, check=True)
        args.peak_info = test_peak_info
    
    # Run test
    output_dir_r = f"results/SCENT_test_R/{args.dataset}"
    output_dir_gpu = f"results/SCENT_test_GPU/{args.dataset}"
    
    Path(output_dir_r).mkdir(parents=True, exist_ok=True)
    Path(output_dir_gpu).mkdir(parents=True, exist_ok=True)
    
    results = run_test_case(
        dataset=args.dataset,
        processed_dir=args.processed_dir,
        peak_info_file=args.peak_info,
        output_dir_r=output_dir_r,
        output_dir_gpu=output_dir_gpu,
        chunk_id=args.chunk_id,
        device=args.device,
        run_r=not args.skip_r,
        run_gpu=not args.skip_gpu
    )
    
    # Save results
    results_file = f"test_results_{args.n_pairs}pairs.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Exit code
    if results.get('passed', False):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

