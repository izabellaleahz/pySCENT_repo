"""
Profiling and Validation Harness for GPU-Accelerated SCENT
Provides timing, profiling, and validation against R results
"""

import torch
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("Warning: torch.profiler not available. Install PyTorch >= 1.8.1 for profiling.")


def profile_scent_stage(
    func,
    *args,
    stage_name: str = "stage",
    device: str = 'cuda',
    **kwargs
) -> Tuple[any, Dict]:
    """
    Profile a single stage of the SCENT pipeline
    
    Args:
        func: Function to profile
        *args: Positional arguments for func
        stage_name: Name of the stage for profiling
        device: Device to profile ('cuda' or 'cpu')
        **kwargs: Keyword arguments for func
    
    Returns:
        result: Result from func
        timing_info: Dictionary with timing information
    """
    timing_info = {}
    
    if PROFILER_AVAILABLE and device == 'cuda':
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    elif PROFILER_AVAILABLE:
        activities = [ProfilerActivity.CPU]
    else:
        activities = None
    
    # Warmup
    if device == 'cuda':
        torch.cuda.synchronize()
    _ = func(*args, **kwargs)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time the function
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    if PROFILER_AVAILABLE and activities:
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function(stage_name):
                result = func(*args, **kwargs)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Extract timing info
        timing_info['wall_time'] = end_time - start_time
        timing_info['profiler'] = prof
        
        # Get CUDA timing if available
        if device == 'cuda':
            events = prof.key_averages()
            cuda_time = sum([e.cuda_time_total for e in events if e.cuda_time_total > 0])
            timing_info['cuda_time_ms'] = cuda_time / 1000.0  # Convert to seconds
    else:
        result = func(*args, **kwargs)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        timing_info['wall_time'] = end_time - start_time
    
    return result, timing_info


def validate_against_r(
    gpu_results: pd.DataFrame,
    r_results_file: str,
    tolerance: Dict[str, float] = None
) -> Dict:
    """
    Validate GPU results against R SCENT results
    
    Args:
        gpu_results: DataFrame with GPU results (columns: gene, peak, beta, se, z, p, boot_basic_p)
        r_results_file: Path to R results file (TSV or JSON)
        tolerance: Dictionary with tolerance for each metric (default: beta=0.01, se=0.01, p=0.01)
    
    Returns:
        validation_dict: Dictionary with validation results
    """
    if tolerance is None:
        tolerance = {
            'beta': 0.01,
            'se': 0.01,
            'z': 0.01,
            'p': 0.01,
            'boot_basic_p': 0.01
        }
    
    # Load R results
    r_file = Path(r_results_file)
    if r_file.suffix == '.json':
        with open(r_file, 'r') as f:
            r_data = json.load(f)
        r_results = pd.DataFrame(r_data)
    else:
        r_results = pd.read_csv(r_file, sep='\t')
    
    # Merge on gene and peak
    merged = pd.merge(
        gpu_results,
        r_results,
        on=['gene', 'peak'],
        suffixes=('_gpu', '_r'),
        how='inner'
    )
    
    if len(merged) == 0:
        return {
            'n_matched': 0,
            'error': 'No matching gene-peak pairs found'
        }
    
    validation = {
        'n_matched': len(merged),
        'n_gpu_only': len(gpu_results) - len(merged),
        'n_r_only': len(r_results) - len(merged)
    }
    
    # Compare each metric
    metrics = ['beta', 'se', 'z', 'p', 'boot_basic_p']
    for metric in metrics:
        gpu_col = f'{metric}_gpu' if f'{metric}_gpu' in merged.columns else metric
        r_col = f'{metric}_r' if f'{metric}_r' in merged.columns else metric
        
        if gpu_col not in merged.columns or r_col not in merged.columns:
            validation[f'{metric}_comparison'] = 'columns_not_found'
            continue
        
        gpu_vals = merged[gpu_col].values
        r_vals = merged[r_col].values
        
        # Remove NaN values for comparison
        valid_mask = ~(np.isnan(gpu_vals) | np.isnan(r_vals))
        if valid_mask.sum() == 0:
            validation[f'{metric}_comparison'] = 'no_valid_values'
            continue
        
        gpu_vals_clean = gpu_vals[valid_mask]
        r_vals_clean = r_vals[valid_mask]
        
        # Calculate differences
        abs_diff = np.abs(gpu_vals_clean - r_vals_clean)
        rel_diff = abs_diff / (np.abs(r_vals_clean) + 1e-10)
        
        tol = tolerance.get(metric, 0.01)
        within_tol = (abs_diff <= tol) | (rel_diff <= tol)
        
        validation[f'{metric}_mean_abs_diff'] = float(abs_diff.mean())
        validation[f'{metric}_max_abs_diff'] = float(abs_diff.max())
        validation[f'{metric}_mean_rel_diff'] = float(rel_diff.mean())
        validation[f'{metric}_within_tolerance'] = int(within_tol.sum())
        validation[f'{metric}_total'] = int(valid_mask.sum())
        validation[f'{metric}_pct_within_tol'] = float(within_tol.sum() / valid_mask.sum() * 100)
    
    return validation


def profile_scent_pipeline(
    data: Dict,
    celltype: str,
    peak_info_subset: pd.DataFrame,
    device: str = 'cuda',
    n_pairs: int = 10
) -> Dict:
    """
    Profile the full SCENT pipeline for a subset of pairs
    
    Args:
        data: Data dictionary
        celltype: Cell type to analyze
        peak_info_subset: Subset of peak_info to test
        device: Device to use
        n_pairs: Number of pairs to profile
    
    Returns:
        profile_dict: Dictionary with profiling results
    """
    from .scent_gpu import scent_algorithm_gpu
    
    # Limit to n_pairs
    peak_info_test = peak_info_subset.head(n_pairs).copy()
    
    profile_results = {
        'n_pairs': len(peak_info_test),
        'celltype': celltype,
        'device': device
    }
    
    # Profile full pipeline
    start_time = time.time()
    if device == 'cuda':
        torch.cuda.synchronize()
    
    results, timing_info = profile_scent_stage(
        scent_algorithm_gpu,
        data,
        celltype,
        peak_info=peak_info_test,
        device=device,
        verbose=False,
        stage_name='full_pipeline'
    )
    
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    profile_results['total_time'] = end_time - start_time
    profile_results['time_per_pair'] = profile_results['total_time'] / len(peak_info_test)
    profile_results['timing_info'] = timing_info
    
    if PROFILER_AVAILABLE and 'profiler' in timing_info:
        prof = timing_info['profiler']
        profile_results['profiler_export'] = prof.export_chrome_trace(f'/tmp/scent_profile_{celltype}.json')
    
    return profile_results


def print_validation_summary(validation: Dict):
    """Print a formatted summary of validation results"""
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80)
    print(f"Matched pairs: {validation.get('n_matched', 0)}")
    print(f"GPU-only pairs: {validation.get('n_gpu_only', 0)}")
    print(f"R-only pairs: {validation.get('n_r_only', 0)}")
    
    metrics = ['beta', 'se', 'z', 'p', 'boot_basic_p']
    for metric in metrics:
        comp_key = f'{metric}_comparison'
        if comp_key in validation:
            if validation[comp_key] != 'columns_not_found':
                print(f"\n{metric.upper()}:")
                print(f"  Mean absolute difference: {validation.get(f'{metric}_mean_abs_diff', 'N/A'):.6f}")
                print(f"  Max absolute difference: {validation.get(f'{metric}_max_abs_diff', 'N/A'):.6f}")
                print(f"  Mean relative difference: {validation.get(f'{metric}_mean_rel_diff', 'N/A'):.6f}")
                print(f"  Within tolerance: {validation.get(f'{metric}_within_tolerance', 0)}/{validation.get(f'{metric}_total', 0)} ({validation.get(f'{metric}_pct_within_tol', 0):.1f}%)")
    
    print("="*80 + "\n")


def print_profiling_summary(profile_results: Dict):
    """Print a formatted summary of profiling results"""
    print("\n" + "="*80)
    print("Profiling Summary")
    print("="*80)
    print(f"Cell type: {profile_results.get('celltype', 'N/A')}")
    print(f"Device: {profile_results.get('device', 'N/A')}")
    print(f"Number of pairs: {profile_results.get('n_pairs', 0)}")
    print(f"Total time: {profile_results.get('total_time', 0):.2f} seconds")
    print(f"Time per pair: {profile_results.get('time_per_pair', 0):.3f} seconds")
    
    timing_info = profile_results.get('timing_info', {})
    if 'cuda_time_ms' in timing_info:
        print(f"CUDA time: {timing_info['cuda_time_ms']:.2f} seconds")
    
    print("="*80 + "\n")

