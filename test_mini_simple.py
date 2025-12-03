#!/usr/bin/env python3
"""
Simple mini test for pySCENT - runs 10 gene-peak pairs
"""

import sys
import os
from pathlib import Path

# Add pySCENT to path
sys.path.insert(0, str(Path(__file__).parent / 'pySCENT'))

def test_mini():
    """Run mini test with 10 pairs"""
    print("="*80)
    print("PYSCENT MINI TEST (10 pairs)")
    print("="*80)

    from pySCENT import scent_multi_celltype_gpu
    import pandas as pd
    import anndata as ad
    import torch
    from scipy import sparse as sp
    from pySCENT.data_loader import sparse_to_torch_coo

    # Paths to data
    rna_file = "/home/izabellaz/bio_stats_project/data/processed_qc/arthritis/arthritis_RNA_qc_endothelial_only.h5ad"
    atac_file = "/home/izabellaz/bio_stats_project/data/processed_qc/arthritis/arthritis_ATAC_endothelial_only.h5ad"
    peak_info_file = str(Path(__file__).parent / "examples/mini_test/peak_info_mini_10.tsv")

    print(f"RNA: {rna_file}")
    print(f"ATAC: {atac_file}")
    print(f"Peak info: {peak_info_file}")

    # Check files exist
    for f in [rna_file, atac_file, peak_info_file]:
        if not Path(f).exists():
            print(f"❌ Missing: {f}")
            return False
    print("✓ All files exist")

    # Load peak info
    peak_info = pd.read_csv(peak_info_file, sep='\t', header=None, names=['gene', 'peak'])
    print(f"✓ Loaded {len(peak_info)} gene-peak pairs")

    # Load data
    print("\nLoading data...")
    rna_ad = ad.read_h5ad(rna_file)
    atac_ad = ad.read_h5ad(atac_file)

    # Convert to CSR if needed
    rna_matrix = rna_ad.X
    atac_matrix = atac_ad.X
    if not sp.issparse(rna_matrix):
        rna_matrix = sp.csr_matrix(rna_matrix)
    if not sp.issparse(atac_matrix):
        atac_matrix = sp.csr_matrix(atac_matrix)

    # Convert to dense tensors and transfer to GPU
    print("Converting to dense and transferring to GPU...")
    import torch
    rna_gpu = torch.tensor(rna_matrix.toarray(), dtype=torch.float32, device='cuda')
    atac_gpu = torch.tensor(atac_matrix.toarray(), dtype=torch.float32, device='cuda')

    # Build data dict
    data = {
        'rna': rna_gpu,
        'atac': atac_gpu,
        'metadata': rna_ad.obs,
        'peak_info': peak_info,
        'gene_names': rna_ad.var_names.tolist(),
        'peak_names': atac_ad.var_names.tolist(),
        'cell_names': rna_ad.obs_names.tolist()
    }

    print(f"✓ Data loaded: {rna_matrix.shape[0]} cells, {rna_matrix.shape[1]} genes, {atac_matrix.shape[1]} peaks")

    # Run SCENT with reduced bootstraps for speed
    print("\n" + "="*80)
    print("RUNNING SCENT GPU")
    print("="*80)

    try:
        results = scent_multi_celltype_gpu(
            data=data,
            celltypes=None,  # Auto-detect cell types
            device='cuda',
            verbose=True,
            bootstrap_counts=[50, 100],  # Reduced for testing
            bootstrap_thresholds=[0.1]
        )

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total tests: {len(results)}")

        if len(results) > 0 and 'FDR' in results.columns:
            n_sig = (results['FDR'] <= 0.10).sum()
            print(f"Significant hits (FDR ≤ 0.10): {n_sig}/{len(results)}")
            print("\nFirst 5 results:")
            print(results.head())
            print("\n✅ TEST PASSED!")
            return True
        else:
            print("❌ No FDR column in results")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_mini()
    sys.exit(0 if success else 1)
