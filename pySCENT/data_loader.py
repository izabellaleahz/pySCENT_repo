"""
Data Loader for GPU-Accelerated SCENT
Load sparse matrices from RDS/h5ad and transfer to GPU
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse as sp
from scipy.io import mmread, mmwrite
import subprocess
import tempfile
import json
from typing import Dict, Tuple, Optional


def load_rds_matrix(rds_file: str) -> sp.csr_matrix:
    """Load sparse matrix from RDS file using R"""
    # Find Rscript in conda environment or system
    import shutil
    rscript_path = shutil.which('Rscript')
    
    if rscript_path is None:
        # Try conda environment path
        import os
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            potential_path = os.path.join(conda_prefix, 'bin', 'Rscript')
            if os.path.exists(potential_path):
                rscript_path = potential_path
    
    if rscript_path is None:
        raise RuntimeError(
            "Rscript not found. Please ensure R is installed and accessible.\n"
            "If using conda: conda activate bio_stats\n"
            "Or install: conda install -c conda-forge r-base"
        )
    
    # Use R to convert RDS to CSV temporarily
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    
    r_script = f"""
    library(Matrix)
    mat <- readRDS('{rds_file}')
    if (!inherits(mat, 'dgCMatrix')) {{
        mat <- as(mat, 'dgCMatrix')
    }}
    writeMM(mat, '{tmp_path}.mtx')
    write.table(rownames(mat), '{tmp_path}.rows', row.names=FALSE, col.names=FALSE, quote=FALSE)
    write.table(colnames(mat), '{tmp_path}.cols', row.names=FALSE, col.names=FALSE, quote=FALSE)
    """
    
    try:
        subprocess.run([rscript_path, '-e', r_script], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"R script failed: {e.stderr}")
    
    # Load from Market Matrix format
    try:
        mat = mmread(f'{tmp_path}.mtx').tocsr()
        rows = pd.read_csv(f'{tmp_path}.rows', header=None).squeeze().tolist()
        cols = pd.read_csv(f'{tmp_path}.cols', header=None).squeeze().tolist()
    finally:
        # Clean up temp files
        for ext in ['.mtx', '.rows', '.cols']:
            temp_file = Path(f'{tmp_path}{ext}')
            if temp_file.exists():
                temp_file.unlink()
    
    return mat, rows, cols


def sparse_to_torch_coo(scipy_sparse: sp.csr_matrix, device: str = 'cuda') -> torch.sparse.FloatTensor:
    """Convert scipy sparse matrix to PyTorch sparse COO tensor"""
    coo = scipy_sparse.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return sparse_tensor.to(device)


def load_scent_data(
    dataset: str,
    processed_dir: str = 'data/processed',
    peak_info_file: Optional[str] = None,
    device: str = 'cuda',
    load_to_gpu: bool = True
) -> Dict:
    """
    Load SCENT data (RNA, ATAC, metadata, peak_info) and optionally transfer to GPU
    
    Args:
        dataset: Dataset name (e.g., 'arthritis-tissue')
        processed_dir: Directory containing processed data
        peak_info_file: Path to peak_info TSV file
        device: 'cuda' or 'cpu'
        load_to_gpu: Whether to transfer matrices to GPU immediately
    
    Returns:
        Dictionary with:
            - rna: RNA sparse matrix (scipy or torch sparse)
            - atac: ATAC sparse matrix (scipy or torch sparse)
            - metadata: pandas DataFrame
            - peak_info: pandas DataFrame with gene-peak pairs
            - gene_names: list of gene names
            - peak_names: list of peak names
            - cell_names: list of cell names
    """
    print(f"Loading data for {dataset}...")
    dataset_dir = Path(processed_dir) / dataset
    
    # Try h5ad files first (GPU SCENT), fall back to RDS if needed
    # Check multiple naming patterns
    rna_h5ad = None
    for pattern in [f"{dataset}_RNA_qc.h5ad", f"{dataset}_RNA.h5ad", "RNA_matrix.h5ad"]:
        candidate = dataset_dir / pattern
        if candidate.exists():
            rna_h5ad = candidate
            break
    rna_rds = dataset_dir / "RNA_matrix.rds"
    
    if rna_h5ad and rna_h5ad.exists():
        # Load from h5ad (GPU SCENT format) - load fully into memory
        import anndata as ad
        print(f"  Loading RNA from {rna_h5ad}...")
        rna_adata = ad.read_h5ad(rna_h5ad)  # No backed='r' - load fully into memory
        # Transpose: genes x cells (anndata stores cells x genes)
        rna_mat = rna_adata.X.T.tocsr() if hasattr(rna_adata.X, 'T') else sp.csr_matrix(rna_adata.X).T
        gene_names = list(rna_adata.var_names)
        rna_cells = list(rna_adata.obs_names)
    elif rna_rds.exists():
        # Fall back to RDS (R SCENT format)
        print(f"  Loading RNA from {rna_rds}...")
        rna_mat, gene_names, rna_cells = load_rds_matrix(str(rna_rds))
    else:
        raise FileNotFoundError(f"RNA file not found. Tried: {rna_h5ad} and {rna_rds}")
    print(f"  RNA: {rna_mat.shape[0]} genes x {rna_mat.shape[1]} cells")
    
    # Load ATAC matrix - check multiple naming patterns
    atac_h5ad = None
    for pattern in [f"{dataset}_ATAC_qc.h5ad", f"{dataset}_ATAC.h5ad", "ATAC_matrix.h5ad"]:
        candidate = dataset_dir / pattern
        if candidate.exists():
            atac_h5ad = candidate
            break
    atac_rds = dataset_dir / "ATAC_matrix.rds"
    
    if atac_h5ad and atac_h5ad.exists():
        # Load from h5ad (GPU SCENT format) - load fully into memory
        import anndata as ad
        print(f"  Loading ATAC from {atac_h5ad}...")
        atac_adata = ad.read_h5ad(atac_h5ad)  # No backed='r' - load fully into memory
        # Transpose: peaks x cells (anndata stores cells x peaks)
        atac_mat = atac_adata.X.T.tocsr() if hasattr(atac_adata.X, 'T') else sp.csr_matrix(atac_adata.X).T
        peak_names = list(atac_adata.var_names)
        atac_cells = list(atac_adata.obs_names)
    elif atac_rds.exists():
        # Fall back to RDS (R SCENT format)
        print(f"  Loading ATAC from {atac_rds}...")
        atac_mat, peak_names, atac_cells = load_rds_matrix(str(atac_rds))
    else:
        raise FileNotFoundError(f"ATAC file not found. Tried: {atac_h5ad} and {atac_rds}")
    print(f"  ATAC: {atac_mat.shape[0]} peaks x {atac_mat.shape[1]} cells")
    
    # Load metadata - try from h5ad first, then TSV
    metadata_file = dataset_dir / f"{dataset}_metadata.tsv"
    if rna_h5ad and rna_h5ad.exists() and 'rna_adata' in locals():
        # Use metadata from RNA h5ad
        print(f"  Loading metadata from RNA h5ad...")
        metadata = rna_adata.obs.copy()
        if 'cell' not in metadata.columns:
            metadata['cell'] = metadata.index
        print(f"  Metadata: {len(metadata)} cells")
    elif metadata_file.exists():
        # Fall back to TSV file
        print(f"  Loading metadata from {metadata_file}...")
        metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
        if 'cell' not in metadata.columns:
            metadata['cell'] = metadata.index
        print(f"  Metadata: {len(metadata)} cells")
    else:
        raise FileNotFoundError(f"Metadata not found. Tried: {metadata_file} and RNA h5ad")
    
    # Find common cells
    common_cells = list(set(rna_cells) & set(atac_cells) & set(metadata.index))
    print(f"  Common cells: {len(common_cells)}")
    
    # Subset to common cells
    rna_idx = [rna_cells.index(c) for c in common_cells]
    atac_idx = [atac_cells.index(c) for c in common_cells]
    
    rna_mat = rna_mat[:, rna_idx]
    atac_mat = atac_mat[:, atac_idx]
    metadata = metadata.loc[common_cells]
    
    # Load peak_info
    if peak_info_file and Path(peak_info_file).exists():
        print(f"  Loading peak_info from {peak_info_file}...")
        peak_info = pd.read_csv(peak_info_file, sep='\t', header=None, names=['gene', 'peak'])
        
        # Filter to genes and peaks present in matrices
        peak_info = peak_info[
            peak_info['gene'].isin(gene_names) &
            peak_info['peak'].isin(peak_names)
        ]
        print(f"  Peak info: {len(peak_info)} gene-peak pairs")
    else:
        if peak_info_file:
            print(f"  WARNING: Peak info file not found: {peak_info_file}")
            print(f"  Will generate peak_info from all gene-peak pairs within 500kb...")
        # Generate peak_info from all possible pairs (will be filtered later)
        peak_info = None
    
    data = {
        'gene_names': gene_names,
        'peak_names': peak_names,
        'cell_names': common_cells,
        'metadata': metadata,
        'peak_info': peak_info
    }
    
    # Transfer to GPU if requested
    if load_to_gpu and device == 'cuda':
        print(f"  Transferring matrices to {device}...")
        data['rna'] = sparse_to_torch_coo(rna_mat, device)
        data['atac'] = sparse_to_torch_coo(atac_mat, device)
        print(f"  GPU transfer complete")
    else:
        # Keep as scipy sparse
        data['rna'] = rna_mat
        data['atac'] = atac_mat
    
    return data


def extract_gene_peak_data(
    rna: torch.Tensor,
    atac: torch.Tensor,
    celltype_mask: torch.Tensor,
    covariate_tensor: torch.Tensor,
    gene_idx: int,
    peak_idx: int,
    min_expr_frac: float = 0.05,
    gene_cache: Optional[dict] = None,
    peak_cache: Optional[dict] = None,
    gene_cache_max: Optional[int] = None,
    peak_cache_max: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Extract and prepare data for a single gene-peak pair on GPU using precomputed masks/tensors.
    
    Returns:
        Tuple of (gene_expr, peak_acc, covariate_matrix) or None if filters not met
    """
    # Extract gene expression and peak accessibility as dense tensors on device
    if gene_cache is not None and gene_idx in gene_cache:
        gene_expr = gene_cache.pop(gene_idx)
        gene_cache[gene_idx] = gene_expr  # mark as recently used
    else:
        gene_expr = rna[:, gene_idx].to_dense().squeeze()[celltype_mask]
        if gene_cache is not None:
            gene_cache[gene_idx] = gene_expr
            if gene_cache_max is not None and len(gene_cache) > gene_cache_max:
                gene_cache.popitem(last=False)

    if peak_cache is not None and peak_idx in peak_cache:
        peak_acc = peak_cache.pop(peak_idx)
        peak_cache[peak_idx] = peak_acc
    else:
        peak_acc = atac[:, peak_idx].to_dense().squeeze()[celltype_mask]
        peak_acc = (peak_acc > 0).float()
        if peak_cache is not None:
            peak_cache[peak_idx] = peak_acc
            if peak_cache_max is not None and len(peak_cache) > peak_cache_max:
                peak_cache.popitem(last=False)
    
    # Check expression thresholds (>5% of cells)
    nonzero_gene = (gene_expr > 0).float().mean()
    nonzero_peak = peak_acc.mean()
    
    if nonzero_gene < min_expr_frac or nonzero_peak < min_expr_frac:
        return None
    
    return gene_expr, peak_acc, covariate_tensor


def extract_gene_peak_data_precomputed(
    rna: torch.sparse.FloatTensor,
    atac: torch.sparse.FloatTensor,
    gene_idx: int,
    peak_idx: int,
    celltype_mask_tensor: torch.Tensor,
    cov_matrix_base: torch.Tensor,
    min_expr_frac: float = 0.05,
    device: str = 'cuda'
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Backwards-compatible wrapper that delegates to extract_gene_peak_data with precomputed tensors.
    """
    return extract_gene_peak_data(
        rna,
        atac,
        celltype_mask_tensor,
        cov_matrix_base,
        gene_idx,
        peak_idx,
        min_expr_frac
    )


def prepare_batch_data(
    data: Dict,
    peak_info_batch: pd.DataFrame,
    celltype_mask: torch.Tensor,
    covariate_tensor: torch.Tensor,
    gene_to_idx: Dict[str, int],
    peak_to_idx: Dict[str, int],
    min_expr_frac: float = 0.05
) -> list:
    """
    Prepare data for a batch of gene-peak pairs
    
    Returns:
        List of dictionaries, each containing data for one gene-peak pair
    """
    batch_data = []
    
    for idx, row in peak_info_batch.iterrows():
        gene = row['gene']
        peak = row['peak']
        
        gene_idx = gene_to_idx.get(gene)
        peak_idx = peak_to_idx.get(peak)
        
        if gene_idx is None or peak_idx is None:
            continue
        
        pair_data = extract_gene_peak_data(
            data['rna'], data['atac'],
            celltype_mask, covariate_tensor,
            gene_idx, peak_idx,
            min_expr_frac
        )
        
        if pair_data is not None:
            gene_expr, peak_acc, cov_matrix = pair_data
            batch_data.append({
                'gene': gene,
                'peak': peak,
                'gene_expr': gene_expr,
                'peak_acc': peak_acc,
                'covariates': cov_matrix
            })
    
    return batch_data
