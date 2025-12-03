"""
Main GPU-Accelerated SCENT Algorithm
Coordinates Poisson GLM fitting and adaptive bootstrapping
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import OrderedDict
import time

from .poisson_gpu import fit_single_pair
from .bootstrap_gpu import adaptive_bootstrap_gpu
from .data_loader import load_scent_data, extract_gene_peak_data


def scent_algorithm_gpu(
    data: Dict,
    celltype: str,
    celltype_col: str = 'ct',
    covariates: Optional[List[str]] = None,
    peak_info: Optional[pd.DataFrame] = None,
    min_expr_frac: float = 0.05,
    device: str = 'cuda',
    batch_size: int = 1,
    seed: int = 42,
    verbose: bool = True,
    regression_type: str = 'poisson',
    bootstrap_counts: Optional[list] = None,
    bootstrap_thresholds: Optional[list] = None
) -> pd.DataFrame:
    """
    Run SCENT algorithm on GPU for a specific cell type
    
    Args:
        data: Dictionary with rna, atac, metadata, gene_names, peak_names
        celltype: Cell type to analyze
        celltype_col: Column name for cell type in metadata
        covariates: List of covariate column names
        peak_info: DataFrame with gene-peak pairs (uses data['peak_info'] if None)
        min_expr_frac: Minimum expression fraction (default 0.05 = 5%)
        device: 'cuda' or 'cpu'
        batch_size: Number of pairs to process in parallel (currently 1 for stability)
        seed: Random seed for bootstrapping
        verbose: Print progress
        regression_type: 'poisson' or 'negbin' for Poisson or Negative Binomial regression
    
    Returns:
        DataFrame with columns: gene, peak, beta, se, z, p, boot_basic_p, celltype
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running SCENT GPU for celltype: {celltype}")
        print(f"{'='*80}")
    
    # Use peak_info from data if not provided
    if peak_info is None:
        peak_info = data['peak_info']
    
    if peak_info is None or len(peak_info) == 0:
        raise ValueError("No peak_info provided")
    
    # Default covariates if not specified
    if covariates is None:
        covariates = []
        for cov in ['log_nUMI', 'percent_mito', 'donor_num']:
            if cov in data['metadata'].columns:
                covariates.append(cov)
    
    if verbose:
        print(f"Cell type: {celltype}")
        print(f"Covariates: {', '.join(covariates)}")
        print(f"Gene-peak pairs to test: {len(peak_info)}")
        print(f"Device: {device}")
        print(f"Regression type: {regression_type}")
    
    # Get data dimensions and build dict lookups for O(1) access
    gene_names = data['gene_names']
    peak_names = data['peak_names']
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    peak_name_to_idx = {name: idx for idx, name in enumerate(peak_names)}
    
    # Ensure matrices reside on the requested device
    if not torch.is_tensor(data['rna']) or not torch.is_tensor(data['atac']):
        raise ValueError("RNA/ATAC matrices must be torch tensors; load with load_to_gpu=True for GPU execution.")
    
    if data['rna'].device.type != device:
        data['rna'] = data['rna'].to(device)
    if data['atac'].device.type != device:
        data['atac'] = data['atac'].to(device)
    
    # Filter metadata to cell type and precompute cell-type-specific data
    celltype_mask = data['metadata'][celltype_col] == celltype
    n_cells_ct = int(celltype_mask.sum())
    
    if verbose:
        print(f"Cells in {celltype}: {n_cells_ct}")
    
    # Precompute cell-type-specific tensors on GPU once
    celltype_mask_tensor = torch.tensor(
        celltype_mask.values, dtype=torch.bool, device=device
    )
    
    # Precompute covariate matrix for this cell type (once, not per pair)
    if len(covariates) > 0:
        cov_data = data['metadata'].loc[celltype_mask, covariates].values
        cov_matrix_base = torch.as_tensor(cov_data, device=device, dtype=torch.float32)
    else:
        cov_matrix_base = torch.empty((n_cells_ct, 0), device=device)
    
    # Prepare results storage
    results = []
    
    # Lightweight LRU caches for repeated row accesses within this cell type
    gene_cache: OrderedDict = OrderedDict()
    peak_cache: OrderedDict = OrderedDict()
    GENE_CACHE_MAX = 1024
    PEAK_CACHE_MAX = 1024

    # Process each gene-peak pair
    start_time = time.time()
    
    iterator = tqdm(peak_info.iterrows(), total=len(peak_info), desc="Processing pairs") if verbose else peak_info.iterrows()
    
    # Normalize peak names in lookup dict to handle colon vs dash format
    def normalize_peak_name(p):
        if isinstance(p, str) and ':' in p:
            return p.replace(':', '-')
        return p
    
    # Create normalized peak lookup (try both formats)
    peak_name_to_idx_normalized = {}
    for name, idx in peak_name_to_idx.items():
        normalized = normalize_peak_name(name)
        peak_name_to_idx_normalized[normalized] = idx
        peak_name_to_idx_normalized[name] = idx  # Also keep original
    
    for pair_idx, (idx, row) in enumerate(iterator):
        gene = row['gene']
        peak = row['peak']
        
        # Get indices using dict lookup (O(1) instead of O(N))
        gene_idx = gene_name_to_idx.get(gene)
        # Try normalized peak name first, then original
        peak_idx = peak_name_to_idx_normalized.get(peak) or peak_name_to_idx.get(peak)
        
        if gene_idx is None or peak_idx is None:
            continue
        
        # Extract data for this pair (using precomputed cell type mask)
        pair_data = extract_gene_peak_data(
            data['rna'], data['atac'],
            celltype_mask_tensor, cov_matrix_base,
            gene_idx, peak_idx,
            min_expr_frac,
            gene_cache=gene_cache,
            peak_cache=peak_cache,
            gene_cache_max=GENE_CACHE_MAX,
            peak_cache_max=PEAK_CACHE_MAX
        )
        
        if pair_data is None:
            # Filtered out due to low expression
            continue
        
        gene_expr, peak_acc, cov_matrix = pair_data
        
        # DEBUG: Check that we have cells
        if gene_expr.shape[0] == 0:
            if verbose:
                print(f"  Warning: {gene}-{peak} passed filter but has 0 cells!")
            continue
        
        # Fit initial GLM (Poisson or Negative Binomial)
        try:
            if regression_type == 'negbin':
                from .negative_binomial_gpu import fit_single_pair_negbin
                glm_results = fit_single_pair_negbin(gene_expr, peak_acc, cov_matrix, device)
            else:
                from .poisson_gpu import fit_single_pair
                glm_results = fit_single_pair(gene_expr, peak_acc, cov_matrix, device)
        except Exception as e:
            if verbose:
                print(f"  Warning: GLM failed for {gene}-{peak}: {e}")
            continue
        
        # Create generator for this pair (seed = base_seed + pair_idx)
        if seed is not None:
            pair_generator = torch.Generator(device=device)
            pair_generator.manual_seed(seed + pair_idx)
        else:
            pair_generator = None
        
        # Run adaptive bootstrap
        try:
            boot_pval, n_bootstrap = adaptive_bootstrap_gpu(
                gene_expr, peak_acc, cov_matrix,
                glm_results['beta'],
                device=device,
                seed=None if pair_generator is not None else seed,
                generator=pair_generator,
                regression_type=regression_type,
                bootstrap_counts=bootstrap_counts,
                thresholds=bootstrap_thresholds
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Bootstrap failed for {gene}-{peak}: {e}")
            continue
        
        # Store results
        results.append({
            'gene': gene,
            'peak': peak,
            'beta': glm_results['beta'],
            'se': glm_results['se'],
            'z': glm_results['z'],
            'p': glm_results['p'],
            'boot_basic_p': boot_pval,
            'celltype': celltype
        })
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nCompleted {len(results)} tests in {elapsed:.1f} seconds")
        if len(results) > 0:
            print(f"Average time per pair: {elapsed/len(results):.2f} seconds")
        else:
            print("Average time per pair: N/A (no pairs passed filters)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def scent_multi_celltype_gpu(
    data: Dict,
    celltypes: List[str],
    celltype_col: str = 'ct',
    covariates: Optional[List[str]] = None,
    device: str = 'cuda',
    verbose: bool = True,
    regression_type: str = 'poisson',
    bootstrap_counts: Optional[list] = None,
    bootstrap_thresholds: Optional[list] = None,
    save_incremental: bool = False,
    incremental_output_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    fdr_threshold: float = 0.10
) -> pd.DataFrame:
    """
    Run SCENT for multiple cell types and combine results
    
    Args:
        data: Data dictionary from load_scent_data
        celltypes: List of cell types to analyze
        celltype_col: Column name for cell type
        covariates: List of covariate names
        device: 'cuda' or 'cpu'
        verbose: Print progress
        save_incremental: If True, save results after each cell type completes
        incremental_output_dir: Directory to save incremental results
        dataset_name: Dataset name for incremental file naming
        fdr_threshold: FDR threshold for incremental hits
    
    Returns:
        Combined DataFrame with results for all cell types
    """
    all_results = []
    
    for ct in celltypes:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing cell type: {ct}")
            print(f"{'='*80}")
        
        ct_results = scent_algorithm_gpu(
            data, ct, celltype_col, covariates,
            device=device, verbose=verbose,
            regression_type=regression_type,
            bootstrap_counts=bootstrap_counts,
            bootstrap_thresholds=bootstrap_thresholds
        )
        
        if len(ct_results) > 0:
            all_results.append(ct_results)
            
            # Save incrementally if requested
            if save_incremental and incremental_output_dir is not None:
                from pathlib import Path
                import gzip
                
                output_dir = Path(incremental_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Replace spaces in cell type name for filename
                ct_safe = ct.replace(' ', '_').replace('/', '-')
                prefix = f'{dataset_name}_{ct_safe}' if dataset_name else ct_safe
                
                # Save all results WITHOUT FDR (FDR will be calculated globally at the end)
                # This matches R SCENT behavior: FDR across all cell types, not per-celltype
                all_file = output_dir / f'{prefix}_all_results_incremental.tsv.gz'
                with gzip.open(all_file, 'wt') as f:
                    ct_results.to_csv(f, sep='\t', index=False)
                
                if verbose:
                    print(f"  ✓ Incremental save: {len(ct_results)} pairs (FDR will be calculated globally)")
    
    # Combine all results
    if len(all_results) == 0:
        return pd.DataFrame()
    
    combined = pd.concat(all_results, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Total results: {len(combined)} tests across {len(celltypes)} cell types")
        print(f"{'='*80}")
    
    return combined


def calculate_fdr(results: pd.DataFrame, p_col: str = 'boot_basic_p') -> pd.DataFrame:
    """
    Calculate FDR from p-values using Benjamini-Hochberg procedure
    
    Args:
        results: DataFrame with p-values
        p_col: Column name for p-values (default: boot_basic_p)
    
    Returns:
        DataFrame with added FDR column
    """
    from scipy.stats import false_discovery_control
    
    if len(results) == 0:
        results['FDR'] = []
        return results
    
    # Get p-values and handle invalid values
    import numpy as np
    p_values = results[p_col].values.copy()
    
    # Replace NaN with 1.0 (conservative)
    p_values = np.where(np.isnan(p_values), 1.0, p_values)
    
    # Clip to [0, 1] range (handle any values outside)
    p_values = np.clip(p_values, 0.0, 1.0)
    
    # Calculate FDR
    fdr = false_discovery_control(p_values, method='bh')
    results['FDR'] = fdr
    
    return results


def save_scent_results(
    results: pd.DataFrame,
    output_prefix: str,
    fdr_threshold: float = 0.10,
    verbose: bool = True
):
    """
    Save SCENT results in the same format as R implementation
    
    Args:
        results: DataFrame with SCENT results
        output_prefix: Prefix for output files (e.g., 'results/SCENT/dataset_chunk_001')
        fdr_threshold: FDR threshold for hits (default 0.10)
        verbose: Print summary
    """
    import gzip
    import json
    from pathlib import Path
    
    # Ensure FDR is calculated
    if 'FDR' not in results.columns:
        results = calculate_fdr(results)
    
    # Separate hits and non-hits
    hits = results[results['FDR'] <= fdr_threshold].copy()
    nonhits = results[results['FDR'] > fdr_threshold].copy()
    
    if verbose:
        print(f"\nSaving results:")
        print(f"  Total tests: {len(results)}")
        print(f"  Significant hits (FDR <= {fdr_threshold}): {len(hits)}")
        print(f"  Non-hits: {len(nonhits)}")
    
    # Ensure output directory exists
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    # Save hits
    hits_file = f"{output_prefix}_hits.tsv.gz"
    with gzip.open(hits_file, 'wt') as f:
        hits.to_csv(f, sep='\t', index=False)
    if verbose:
        print(f"  Saved: {hits_file}")
    
    # Save non-hits
    nonhits_file = f"{output_prefix}_nonhits.tsv.gz"
    with gzip.open(nonhits_file, 'wt') as f:
        nonhits.to_csv(f, sep='\t', index=False)
    if verbose:
        print(f"  Saved: {nonhits_file}")
    
    # Save complete results
    complete_file = f"{output_prefix}_complete.tsv.gz"
    with gzip.open(complete_file, 'wt') as f:
        results.to_csv(f, sep='\t', index=False)
    if verbose:
        print(f"  Saved: {complete_file}")
    
    # Save summary
    summary = {
        'n_tests': len(results),
        'n_hits': len(hits),
        'n_nonhits': len(nonhits),
        'fdr_threshold': fdr_threshold
    }
    
    summary_file = f"{output_prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"  Saved: {summary_file}")
