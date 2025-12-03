"""
GPU-Accelerated Negative Binomial GLM using PyTorch
Implements Iteratively Reweighted Least Squares (IRLS) for Negative Binomial regression
"""

import torch
from typing import Tuple, Optional


def negative_binomial_glm_gpu(
    y: torch.Tensor,
    X: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
    device: str = 'cuda',
    theta_init: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit Negative Binomial GLM using IRLS (Iteratively Reweighted Least Squares) on GPU
    
    Uses the NB2 parameterization: Var(Y) = mu + mu^2/theta
    where theta is the dispersion parameter (larger = less overdispersion)
    
    Args:
        y: Response variable (gene expression counts), shape (n_samples,)
        X: Design matrix (peak + covariates), shape (n_samples, n_features)
        max_iter: Maximum iterations for IRLS
        tol: Convergence tolerance
        device: 'cuda' or 'cpu'
        theta_init: Initial value for dispersion parameter (if None, estimated from data)
    
    Returns:
        beta: Coefficient estimates, shape (n_features,)
        se: Standard errors, shape (n_features,)
        z: Z-scores, shape (n_features,)
        p: P-values, shape (n_features,)
    """
    n_samples, n_features = X.shape
    
    # Ensure tensors are on the correct device
    y = y.to(device)
    X = X.to(device)
    
    # Initialize coefficients with log-link GLM
    mu_init = y.mean() + 0.5
    beta = torch.zeros(n_features, device=device)
    beta[0] = torch.log(mu_init)  # Intercept should be X[:, 0] = 1
    
    # Initialize dispersion parameter (theta)
    # Start with a reasonable value (theta = 1 means high overdispersion)
    # Typical values range from 0.1 to 10
    if theta_init is None:
        # Estimate initial theta from mean and variance
        mu_sample = y.mean()
        var_sample = y.var()
        if var_sample > mu_sample:
            # Overdispersed: estimate theta from variance
            theta_val = mu_sample**2 / (var_sample - mu_sample + 1e-6)
            theta = torch.clamp(torch.tensor(theta_val, device=device, dtype=torch.float32), min=0.1, max=100.0)
        else:
            # Underdispersed or Poisson-like: use large theta
            theta = torch.tensor(10.0, device=device)
    else:
        theta = torch.tensor(theta_init, device=device)
    
    # IRLS iterations with dispersion estimation
    for iteration in range(max_iter):
        # Linear predictor: eta = X @ beta
        eta = X @ beta
        
        # Mean: mu = exp(eta)
        eta = torch.clamp(eta, min=-20, max=20)
        mu = torch.exp(eta)
        
        # Add small constant to avoid division by zero
        mu = torch.clamp(mu, min=1e-8)
        
        # Negative Binomial variance: Var(Y) = mu + mu^2/theta
        # Working weights: w = mu / (1 + mu/theta)
        # This is the inverse of the variance function
        w = mu / (1.0 + mu / theta)
        
        # Working response: z = eta + (y - mu) / mu
        z = eta + (y - mu) / mu
        
        # Weighted least squares: beta_new = (X^T W X)^{-1} X^T W z
        # where W = diag(w)
        
        # X^T W X
        XtWX = X.t() @ (w.unsqueeze(1) * X)
        
        # X^T W z
        XtWz = X.t() @ (w * z)
        
        # Solve: XtWX @ beta_new = XtWz
        # Add small ridge for numerical stability
        ridge = 1e-6 * torch.eye(n_features, device=device)
        try:
            beta_new = torch.linalg.solve(XtWX + ridge, XtWz)
        except RuntimeError:
            # If singular, use pseudoinverse
            beta_new = torch.linalg.lstsq(XtWX + ridge, XtWz).solution
        
        # Update dispersion parameter (theta) using method of moments
        # This is a simplified approach - could use MLE but this is faster
        eta_new = X @ beta_new
        eta_new = torch.clamp(eta_new, min=-20, max=20)
        mu_new = torch.exp(eta_new)
        mu_new = torch.clamp(mu_new, min=1e-8)
        
        # Estimate theta from residuals
        residuals = (y - mu_new)**2
        var_est = residuals.mean()
        mu_mean = mu_new.mean()
        
        if var_est > mu_mean:
            theta_new = mu_mean**2 / (var_est - mu_mean + 1e-6)
            theta_new = torch.clamp(theta_new, min=0.1, max=100.0)
            # Use exponential moving average for stability
            theta = 0.7 * theta + 0.3 * theta_new
        else:
            # Keep current theta if underdispersed
            pass
        
        # Check convergence
        delta = torch.abs(beta_new - beta).max()
        beta = beta_new
        
        if delta < tol:
            break
    
    # Compute standard errors from Fisher information
    # Var(beta) = (X^T W X)^{-1}
    eta = X @ beta
    eta = torch.clamp(eta, min=-20, max=20)
    mu = torch.exp(eta)
    mu = torch.clamp(mu, min=1e-8)
    w = mu / (1.0 + mu / theta)
    
    XtWX = X.t() @ (w.unsqueeze(1) * X)
    ridge = 1e-6 * torch.eye(n_features, device=device)
    
    try:
        cov_matrix = torch.linalg.inv(XtWX + ridge)
    except RuntimeError:
        cov_matrix = torch.linalg.pinv(XtWX + ridge)
    
    # Standard errors are sqrt of diagonal
    se = torch.sqrt(torch.diag(cov_matrix))
    
    # Z-scores
    z = beta / (se + 1e-10)
    
    # P-values (two-tailed) - computed on GPU using torch.special.ndtr
    z_abs = torch.abs(z)
    p = 2 * (1 - torch.special.ndtr(z_abs))
    
    return beta, se, z, p


def negative_binomial_glm_batch_gpu(
    y_batch: torch.Tensor,
    X_batch: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
    device: str = 'cuda',
    theta_init: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit multiple Negative Binomial GLMs in parallel on GPU
    
    Args:
        y_batch: Batch of response variables, shape (batch_size, n_samples)
        X_batch: Batch of design matrices, shape (batch_size, n_samples, n_features)
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance
        device: 'cuda' or 'cpu'
        theta_init: Initial value for dispersion parameter
    
    Returns:
        beta_batch: Coefficients, shape (batch_size, n_features)
        se_batch: Standard errors, shape (batch_size, n_features)
        z_batch: Z-scores, shape (batch_size, n_features)
        p_batch: P-values, shape (batch_size, n_features)
    """
    batch_size, n_samples, n_features = X_batch.shape
    
    y_batch = y_batch.to(device)
    X_batch = X_batch.to(device)
    
    # Initialize coefficients
    mu_init = y_batch.mean(dim=1, keepdim=True) + 0.5
    beta_batch = torch.zeros(batch_size, n_features, device=device)
    beta_batch[:, 0] = torch.log(mu_init).squeeze()
    
    # Initialize dispersion parameter
    if theta_init is None:
        mu_sample = y_batch.mean(dim=1)
        var_sample = y_batch.var(dim=1)
        theta_batch = torch.ones(batch_size, device=device) * 1.0
        for i in range(batch_size):
            if var_sample[i] > mu_sample[i]:
                theta_val = mu_sample[i]**2 / (var_sample[i] - mu_sample[i] + 1e-6)
                theta_batch[i] = torch.clamp(torch.tensor(theta_val, device=device, dtype=torch.float32), min=0.1, max=100.0)
    else:
        theta_batch = torch.ones(batch_size, device=device) * theta_init
    
    # IRLS iterations
    for iteration in range(max_iter):
        # Linear predictor: eta = X @ beta (batched)
        eta = torch.bmm(X_batch, beta_batch.unsqueeze(2)).squeeze(2)
        
        # Mean: mu = exp(eta)
        eta = torch.clamp(eta, min=-20, max=20)
        mu = torch.exp(eta)
        mu = torch.clamp(mu, min=1e-8)
        
        # Working weights: w = mu / (1 + mu/theta)
        theta_expanded = theta_batch.unsqueeze(1).expand(-1, n_samples)
        w = mu / (1.0 + mu / theta_expanded)
        
        # Working response
        z = eta + (y_batch - mu) / mu
        
        # Weighted least squares (batched)
        w_expanded = w.unsqueeze(2)  # (batch_size, n_samples, 1)
        XtWX = torch.bmm(X_batch.transpose(1, 2), w_expanded * X_batch)
        
        XtWz = torch.bmm(X_batch.transpose(1, 2), (w * z).unsqueeze(2)).squeeze(2)
        
        # Solve with ridge regularization
        ridge = 1e-6 * torch.eye(n_features, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        try:
            beta_new = torch.linalg.solve(XtWX + ridge, XtWz.unsqueeze(2)).squeeze(2)
        except RuntimeError:
            beta_new = torch.linalg.lstsq(XtWX + ridge, XtWz.unsqueeze(2)).solution.squeeze(2)
        
        # Update dispersion parameter
        eta_new = torch.bmm(X_batch, beta_new.unsqueeze(2)).squeeze(2)
        eta_new = torch.clamp(eta_new, min=-20, max=20)
        mu_new = torch.exp(eta_new)
        mu_new = torch.clamp(mu_new, min=1e-8)
        
        # Estimate theta from residuals
        residuals = (y_batch - mu_new)**2
        var_est = residuals.mean(dim=1)
        mu_mean = mu_new.mean(dim=1)
        
        for i in range(batch_size):
            if var_est[i] > mu_mean[i]:
                theta_new = mu_mean[i]**2 / (var_est[i] - mu_mean[i] + 1e-6)
                theta_new = torch.clamp(theta_new, min=0.1, max=100.0)
                theta_batch[i] = 0.7 * theta_batch[i] + 0.3 * theta_new
        
        # Check convergence
        delta = torch.abs(beta_new - beta_batch).max()
        beta_batch = beta_new
        
        if delta < tol:
            break
    
    # Compute standard errors
    eta = torch.bmm(X_batch, beta_batch.unsqueeze(2)).squeeze(2)
    eta = torch.clamp(eta, min=-20, max=20)
    mu = torch.exp(eta)
    mu = torch.clamp(mu, min=1e-8)
    theta_expanded = theta_batch.unsqueeze(1).expand(-1, n_samples)
    w = mu / (1.0 + mu / theta_expanded)
    
    w_expanded = w.unsqueeze(2)
    XtWX = torch.bmm(X_batch.transpose(1, 2), w_expanded * X_batch)
    ridge = 1e-6 * torch.eye(n_features, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    try:
        cov_matrices = torch.linalg.inv(XtWX + ridge)
    except RuntimeError:
        cov_matrices = torch.linalg.pinv(XtWX + ridge)
    
    # Standard errors from diagonal of covariance matrices
    se_batch = torch.sqrt(torch.diagonal(cov_matrices, dim1=1, dim2=2))
    
    # Z-scores
    z_batch = beta_batch / (se_batch + 1e-10)
    
    # P-values (two-tailed)
    z_abs = torch.abs(z_batch)
    p_batch = 2 * (1 - torch.special.ndtr(z_abs))
    
    return beta_batch, se_batch, z_batch, p_batch


def fit_single_pair_negbin(
    gene_expr: torch.Tensor,
    peak_acc: torch.Tensor,
    covariates: torch.Tensor,
    device: str = 'cuda'
) -> dict:
    """
    Fit Negative Binomial GLM for a single gene-peak pair
    
    Args:
        gene_expr: Gene expression counts, shape (n_cells,)
        peak_acc: Peak accessibility (binarized), shape (n_cells,)
        covariates: Covariate matrix, shape (n_cells, n_covariates)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with beta, se, z, p for the peak (atac) coefficient
    """
    # Construct design matrix: [intercept, peak, covariates]
    n_cells = gene_expr.shape[0]
    intercept = torch.ones((n_cells, 1), device=device)
    X = torch.cat([intercept, peak_acc.unsqueeze(1), covariates], dim=1)
    
    # Fit GLM
    beta, se, z, p = negative_binomial_glm_gpu(gene_expr, X, device=device)
    
    # Return coefficient for peak (index 1, since intercept is index 0)
    return {
        'beta': beta[1].item(),
        'se': se[1].item(),
        'z': z[1].item(),
        'p': p[1].item()
    }

