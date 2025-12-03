#!/usr/bin/env python3
"""
GPU-Accelerated SCENT CLI
Drop-in replacement for run_scent.R with identical interface
"""

import argparse
import sys
import json
from pathlib import Path
import time
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scent_gpu.data_loader import load_scent_data
from scent_gpu.scent_gpu import scent_algorithm_gpu, scent_multi_celltype_gpu, calculate_fdr, save_scent_results


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated SCENT for multimodal single-cell data'
    )
    
    # Match run_scent.R arguments exactly
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., arthritis-tissue)')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory with processed data')
    parser.add_argument('--peak_info', type=str, required=True,
                      help='Peak info file (gene-peak pairs)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for results')
    parser.add_argument('--fdr_threshold', type=float, default=0.10,
                      help='FDR threshold (default 0.10 per paper)')
    parser.add_argument('--chunk_id', type=str, default='all',
                      help='Chunk ID for parallel processing')
    parser.add_argument('--ncores', type=int, default=1,
                      help='Number of cores (not used for GPU, kept for compatibility)')
    parser.add_argument('--celltypes', nargs='+', default=None,
                      help='Optional list of cell types to run (defaults to all present)')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--bootstrap-counts', nargs='+', type=int, default=None,
                      help='Bootstrap counts for adaptive bootstrapping (e.g., 100 500 2500 5000)')
    parser.add_argument('--bootstrap-thresholds', nargs='+', type=float, default=None,
                      help='P-value thresholds for adaptive bootstrapping (e.g., 0.1 0.05 0.01)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("="*80)
    print(f"RUNNING SCENT GPU: {args.dataset}")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Peak info: {args.peak_info}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chunk ID: {args.chunk_id}")
    print(f"FDR threshold: {args.fdr_threshold}")
    print(f"Device: {args.device}")
    
    start_time = time.time()
    
    # Load data
    print("\nLoading data...")
    try:
        data = load_scent_data(
            dataset=args.dataset,
            processed_dir=args.processed_dir,
            peak_info_file=args.peak_info,
            device=args.device,
            load_to_gpu=(args.device == 'cuda')
        )
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    print(f"Data loaded successfully:")
    print(f"  RNA: {len(data['gene_names'])} genes x {len(data['cell_names'])} cells")
    print(f"  ATAC: {len(data['peak_names'])} peaks x {len(data['cell_names'])} cells")
    print(f"  Gene-peak pairs: {len(data['peak_info'])}")
    
    # Determine cell type column
    celltype_col = None
    for col in ['ct', 'cell_type', 'celltype', 'CellType']:
        if col in data['metadata'].columns:
            celltype_col = col
            break
    
    if celltype_col is None:
        print("WARNING: No cell type column found, using all cells as one group")
        data['metadata']['celltype'] = 'all'
        celltype_col = 'celltype'
    
    # Get cell types
    available_celltypes = data['metadata'][celltype_col].unique().tolist()
    print(f"\nAvailable cell types: {', '.join(available_celltypes)}")
    
    # Determine which cell types to run
    if args.celltypes:
        target_celltypes = [ct for ct in args.celltypes if ct in available_celltypes]
        if len(target_celltypes) == 0:
            print("ERROR: Specified cell types not found in metadata")
            sys.exit(1)
    else:
        target_celltypes = available_celltypes
    
    print(f"Target cell types: {', '.join(target_celltypes)}")
    
    # Determine covariates
    covariates = []
    for cov in ['log_nUMI', 'percent_mito', 'donor_num']:
        if cov in data['metadata'].columns:
            covariates.append(cov)
    
    print(f"Covariates: {', '.join(covariates)}")
    
    # Run SCENT
    print("\n" + "="*80)
    print("RUNNING SCENT")
    print("="*80)
    
    try:
        # Set bootstrap parameters
        bootstrap_counts = args.bootstrap_counts
        bootstrap_thresholds = args.bootstrap_thresholds
        
        if bootstrap_counts:
            print(f"Bootstrap counts: {bootstrap_counts}")
        if bootstrap_thresholds:
            print(f"Bootstrap thresholds: {bootstrap_thresholds}")
        
        if len(target_celltypes) == 1:
            results = scent_algorithm_gpu(
                data=data,
                celltype=target_celltypes[0],
                celltype_col=celltype_col,
                covariates=covariates,
                device=args.device,
                seed=args.seed,
                verbose=True,
                bootstrap_counts=bootstrap_counts,
                bootstrap_thresholds=bootstrap_thresholds
            )
        else:
            results = scent_multi_celltype_gpu(
                data=data,
                celltypes=target_celltypes,
                celltype_col=celltype_col,
                covariates=covariates,
                device=args.device,
                verbose=True,
                bootstrap_counts=bootstrap_counts,
                bootstrap_thresholds=bootstrap_thresholds
            )
    except Exception as e:
        print(f"ERROR running SCENT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if len(results) == 0:
        print("WARNING: No results generated")
        sys.exit(1)
    
    # Calculate FDR
    print("\nCalculating FDR...")
    results = calculate_fdr(results)
    
    print(f"\nFDR summary:")
    print(f"  Min FDR: {results['FDR'].min():.6f}")
    print(f"  Median FDR: {results['FDR'].median():.6f}")
    print(f"  Max FDR: {results['FDR'].max():.6f}")
    print(f"  Pairs with FDR <= {args.fdr_threshold}: {(results['FDR'] <= args.fdr_threshold).sum()}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_prefix = Path(args.output_dir) / f"{args.dataset}_chunk_{args.chunk_id}"
    
    try:
        save_scent_results(
            results=results,
            output_prefix=str(output_prefix),
            fdr_threshold=args.fdr_threshold,
            verbose=True
        )
    except Exception as e:
        print(f"ERROR saving results: {e}")
        sys.exit(1)
    
    # Save extended summary with metadata
    summary_file = f"{output_prefix}_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    summary.update({
        'dataset': args.dataset,
        'chunk_id': args.chunk_id,
        'n_cells': len(data['cell_names']),
        'n_genes': len(data['gene_names']),
        'n_peaks': len(data['peak_names']),
        'covariates': ','.join(covariates),
        'celltype_col': celltype_col,
        'celltypes': ','.join(target_celltypes),
        'device': args.device,
        'execution_time_seconds': time.time() - start_time
    })
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ SCENT GPU completed successfully in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

