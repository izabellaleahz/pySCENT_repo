"""
GPU-Accelerated Bootstrap Engine for SCENT
Implements adaptive bootstrapping with early stopping
"""

import torch
from typing import Optional, Tuple
# Import moved to function to avoid circular imports


def interp_pval_gpu(q: torch.Tensor) -> torch.Tensor:
    """
    Interpolate p-value from bootstrap quantiles (GPU vectorized)
    
    Args:
        q: Bootstrap quantiles centered at null (theta = 0), shape (n_bootstrap,)
    
    Returns:
        Two-sided p-value (scalar)
    """
    R = q.shape[0]
    tstar = torch.sort(q)[0]
    
    # Find where 0 falls in sorted quantiles
    zero = torch.searchsorted(tstar, torch.tensor(0.0, device=q.device))
    
    # At or beyond extreme values
    if zero == 0 or zero == R:
        return torch.tensor(2.0 / R, device=q.device)
    
    # Two-sided p-value
    pval = 2.0 * torch.min(
        torch.tensor(zero.float() / R, device=q.device),
        torch.tensor((R - zero).float() / R, device=q.device)
    )
    
    return pval


def basic_p_gpu(obs: torch.Tensor, boot: torch.Tensor, null: float = 0.0) -> torch.Tensor:
    """
    Calculate bootstrap p-value using the "basic" method
    
    Args:
        obs: Observed statistic (coefficient)
        boot: Bootstrap statistics, shape (n_bootstrap,)
        null: Null hypothesis value (default 0)
    
    Returns:
        P-value (scalar)
    """
    q = 2 * obs - boot - null
    return interp_pval_gpu(q)


def bootstrap_single_pair_gpu(
    gene_expr: torch.Tensor,
    peak_acc: torch.Tensor,
    covariates: torch.Tensor,
    n_bootstrap: int,
    device: str = 'cuda',
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 2048,
    regression_type: str = 'poisson'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform bootstrap resampling for a single gene-peak pair using batched IRLS
    
    Args:
        gene_expr: Gene expression counts, shape (n_cells,)
        peak_acc: Peak accessibility (binarized), shape (n_cells,)
        covariates: Covariate matrix, shape (n_cells, n_covariates)
        n_bootstrap: Number of bootstrap iterations
        device: 'cuda' or 'cpu'
        seed: Optional seed to create a generator when one is not provided
        generator: torch.Generator for RNG (ensures proper seeding and state carryover)
        batch_size: Number of bootstraps to process in parallel
        regression_type: 'poisson' or 'negbin' for Poisson or Negative Binomial regression
    
    Returns:
        boot_coefs: Bootstrap coefficients for peak, shape (n_bootstrap,)
        boot_vars: Bootstrap variances for peak, shape (n_bootstrap,)
    """
    if regression_type == 'negbin':
        from .negative_binomial_gpu import negative_binomial_glm_batch_gpu
        glm_batch_fn = negative_binomial_glm_batch_gpu
    else:
        from .poisson_gpu import poisson_glm_batch_gpu
        glm_batch_fn = poisson_glm_batch_gpu

    n_cells = gene_expr.shape[0]

    # If no generator is provided, create one when a seed is supplied
    if generator is None and seed is not None:
        generator = torch.Generator(device=device if device == 'cuda' else 'cpu')
        generator.manual_seed(seed)
    
    # Construct design matrix: [intercept, peak, covariates]
    # Must match fit_single_pair() - R's glm() includes intercept
    intercept = torch.ones((n_cells, 1), device=device, dtype=gene_expr.dtype)
    X = torch.cat([intercept, peak_acc.unsqueeze(1), covariates], dim=1)
    
    # Pre-allocate results
    boot_coefs = torch.zeros(n_bootstrap, device=device, dtype=gene_expr.dtype)
    boot_vars = torch.zeros(n_bootstrap, device=device, dtype=gene_expr.dtype)
    
    # Process bootstraps in batches
    n_batches = (n_bootstrap + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_bootstrap)
        current_batch_size = end_idx - start_idx
        
        # Generate bootstrap indices for this batch
        boot_indices = torch.randint(
            0, n_cells, (current_batch_size, n_cells),
            device=device,
            generator=generator
        )
        
        # Build batched X and y
        # X_batch: (current_batch_size, n_cells, n_features)
        # y_batch: (current_batch_size, n_cells)
        X_batch = X[boot_indices]  # Advanced indexing
        y_batch = gene_expr[boot_indices]
        
        # Fit GLMs in batch
        try:
            if regression_type == 'negbin':
                beta_batch, se_batch, _, _ = glm_batch_fn(
                    y_batch, X_batch, device=device, max_iter=30
                )
            else:
                beta_batch, se_batch, _, _ = glm_batch_fn(
                    y_batch, X_batch, device=device, max_iter=15
                )
            # Extract peak coefficient (index 1) and variance
            boot_coefs[start_idx:end_idx] = beta_batch[:, 1]
            boot_vars[start_idx:end_idx] = se_batch[:, 1] ** 2
        except Exception:
            # If fitting fails, use NaN
            boot_coefs[start_idx:end_idx] = torch.nan
            boot_vars[start_idx:end_idx] = torch.nan
    
    return boot_coefs, boot_vars


def adaptive_bootstrap_gpu(
    gene_expr: torch.Tensor,
    peak_acc: torch.Tensor,
    covariates: torch.Tensor,
    obs_coef: float,
    device: str = 'cuda',
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    regression_type: str = 'poisson',
    bootstrap_counts: Optional[list] = None,
    thresholds: Optional[list] = None
) -> Tuple[float, int]:
    """
    Adaptive bootstrap following SCENT's strategy:
    - Start with 100 bootstraps
    - If p < 0.1, continue to 500
    - If p < 0.05, continue to 2500
    - If p < 0.01, continue to 25000
    - If p < 0.001, continue to 50000
    
    Args:
        gene_expr: Gene expression counts
        peak_acc: Peak accessibility
        covariates: Covariate matrix
        obs_coef: Observed coefficient from original fit
        device: 'cuda' or 'cpu'
        seed: Optional base seed (creates a generator if one is not provided)
        generator: torch.Generator for RNG (ensures proper seeding, no reseeding per stage)
        regression_type: 'poisson' or 'negbin' for Poisson or Negative Binomial regression
        bootstrap_counts: Custom list of bootstrap counts [default: [100, 500, 2500, 25000, 50000]]
        thresholds: Custom list of p-value thresholds [default: [0.1, 0.05, 0.01, 0.001]]
    
    Returns:
        final_pval: Final bootstrap p-value
        n_bootstrap_used: Total number of bootstraps performed
    """
    # Adaptive thresholds and bootstrap counts (default to SCENT paper values)
    if thresholds is None:
        thresholds = [0.1, 0.05, 0.01, 0.001]
    if bootstrap_counts is None:
        bootstrap_counts = [100, 500, 2500, 25000, 50000]
    
    obs_coef_tensor = torch.tensor(obs_coef, device=device)
    all_boot_coefs = []
    n_previous = 0

    # Build a generator once so stages draw new, non-overlapping resamples
    if generator is None and seed is not None:
        generator = torch.Generator(device=device if device == 'cuda' else 'cpu')
        generator.manual_seed(seed)
    
    for i, (threshold, n_boot) in enumerate(zip(thresholds + [0], bootstrap_counts)):
        # Calculate how many more bootstraps to run
        n_new = n_boot - n_previous
        
        # Run additional bootstraps (generator ensures new resamples, not duplicates)
        boot_coefs, _ = bootstrap_single_pair_gpu(
            gene_expr,
            peak_acc,
            covariates,
            n_new,
            device=device,
            generator=generator,
            regression_type=regression_type
        )
        
        # Remove NaN values
        boot_coefs = boot_coefs[~torch.isnan(boot_coefs)]
        if boot_coefs.numel() == 0:
            n_previous = n_boot
            continue

        all_boot_coefs.append(boot_coefs)
        
        # Combine all bootstraps so far
        combined_boots = torch.cat(all_boot_coefs)
        
        # Calculate p-value
        pval = basic_p_gpu(obs_coef_tensor, combined_boots)
        
        # Early stopping: if p-value above threshold, stop
        if i < len(thresholds) and pval >= threshold:
            return pval.item(), combined_boots.shape[0]
        
        n_previous = n_boot
    
    if all_boot_coefs:
        combined_boots = torch.cat(all_boot_coefs)
        return pval.item(), combined_boots.shape[0]
    
    return float('nan'), 0


def adaptive_bootstrap_batch_gpu(
    batch_data: list,
    obs_coefs: list,
    device: str = 'cuda',
    seed: int = None
) -> Tuple[list, list]:
    """
    Run adaptive bootstrap for a batch of gene-peak pairs in parallel
    
    Note: This processes pairs sequentially but uses GPU for each pair's bootstraps.
    True batch parallelization would require more complex memory management.
    
    Args:
        batch_data: List of dictionaries with gene_expr, peak_acc, covariates
        obs_coefs: List of observed coefficients
        device: 'cuda' or 'cpu'
        seed: Random seed (base seed, each pair gets its own generator)
    
    Returns:
        boot_pvals: List of bootstrap p-values
        n_bootstraps: List of bootstrap counts used
    """
    boot_pvals = []
    n_bootstraps = []
    
    for i, (data, obs_coef) in enumerate(zip(batch_data, obs_coefs)):
        # Create a separate generator for each pair to avoid correlation
        if seed is not None:
            pair_generator = torch.Generator(device=device)
            pair_generator.manual_seed(seed + i)
        else:
            pair_generator = None
        
        pval, n_boot = adaptive_bootstrap_gpu(
            data['gene_expr'],
            data['peak_acc'],
            data['covariates'],
            obs_coef,
            device=device,
            generator=pair_generator
        )
        
        boot_pvals.append(pval)
        n_bootstraps.append(n_boot)
    
    return boot_pvals, n_bootstraps


def bootstrap_parallel_gpu(
    batch_data: list,
    obs_coefs: list,
    n_bootstrap: int = 100,
    device: str = 'cuda',
    seed: int = None
) -> Tuple[list, list]:
    """
    Simplified bootstrap: run fixed number of bootstraps for all pairs
    (For testing and benchmarking without adaptive strategy)
    
    Args:
        batch_data: List of dictionaries with gene_expr, peak_acc, covariates
        obs_coefs: List of observed coefficients  
        n_bootstrap: Number of bootstrap iterations
        device: 'cuda' or 'cpu'
        seed: Random seed (base seed, each pair gets its own generator)
    
    Returns:
        boot_pvals: List of bootstrap p-values
        boot_coefs: List of bootstrap coefficient arrays
    """
    boot_pvals = []
    boot_coefs_list = []
    
    for i, (data, obs_coef) in enumerate(zip(batch_data, obs_coefs)):
        # Create a separate generator for each pair
        if seed is not None:
            pair_generator = torch.Generator(device=device)
            pair_generator.manual_seed(seed + i)
        else:
            pair_generator = None
        
        boot_coefs, _ = bootstrap_single_pair_gpu(
            data['gene_expr'],
            data['peak_acc'],
            data['covariates'],
            n_bootstrap,
            device=device,
            generator=pair_generator
        )
        
        # Remove NaNs
        boot_coefs = boot_coefs[~torch.isnan(boot_coefs)]
        
        # Calculate p-value
        obs_coef_tensor = torch.tensor(obs_coef, device=device)
        pval = basic_p_gpu(obs_coef_tensor, boot_coefs)
        
        boot_pvals.append(pval.item())
        boot_coefs_list.append(boot_coefs.cpu().numpy())
    
    return boot_pvals, boot_coefs_list
