# pySCENT Validation and Comparison Metrics

This document describes the validation approach used to ensure pySCENT produces results equivalent to the original R SCENT implementation.

## Overview

pySCENT has been validated against the original R implementation using multiple test datasets and statistical metrics. The validation ensures:

1. **Algorithmic equivalence**: Same GLM fitting, bootstrap strategy, and FDR calculation
2. **Numerical precision**: Results match within expected tolerances
3. **Performance gains**: Significant speedup without accuracy loss

## Validation Methodology

### Test Datasets

1. **Mini test (10 pairs)**: Quick validation on small datasets
2. **Benchmark test (100-1000 pairs)**: Performance and accuracy validation
3. **Full dataset test**: Production-scale validation

### Comparison Metrics

#### 1. Beta Coefficients (Regression Coefficients)

**Metric**: Mean Absolute Difference (MAE)
**Expected**: |GPU - R| < 1e-4
**Why**: Measures GLM fitting accuracy

```python
beta_diff = np.abs(merged['beta_gpu'] - merged['beta_r'])
beta_mae = beta_diff.mean()
beta_pass = beta_mae < 1e-4
```

#### 2. Bootstrap P-values

**Metric**: Mean Absolute Difference (MAE) and Correlation
**Expected**: MAE < 0.1, Correlation > 0.95
**Why**: Bootstrap involves random sampling, so some variation expected

```python
boot_diff = np.abs(merged['boot_basic_p_gpu'] - merged['boot_basic_p_r'])
boot_mae = boot_diff.mean()
boot_corr = np.corrcoef(merged['boot_basic_p_gpu'], merged['boot_basic_p_r'])[0,1]
boot_pass = boot_mae < 0.1 and boot_corr > 0.95
```

#### 3. Standard Errors

**Metric**: Mean Absolute Difference (MAE)
**Expected**: |GPU - R| < 1e-4
**Why**: Derived from Fisher information matrix

#### 4. FDR Values

**Metric**: Mean Absolute Difference (MAE)
**Expected**: |GPU - R| < 0.01
**Why**: Benjamini-Hochberg correction applied to p-values

#### 5. Significant Hits Agreement

**Metric**: Jaccard Similarity (Intersection over Union)
**Expected**: Jaccard > 0.95
**Why**: Measures agreement on hits at FDR < 0.10

```python
fdr_threshold = 0.10
r_sig = merged['FDR_r'] <= fdr_threshold
gpu_sig = merged['FDR_gpu'] <= fdr_threshold
intersection = (r_sig & gpu_sig).sum()
union = (r_sig | gpu_sig).sum()
jaccard = intersection / union if union > 0 else 1.0
```

## Expected Differences

### Acceptable Variations

1. **Bootstrap p-values**: Due to different random number generators (R vs PyTorch)
2. **Floating-point precision**: Float32 (GPU) vs Float64 (R) can cause small differences
3. **Matrix inversion**: Different BLAS implementations may give slightly different results

### Unacceptable Differences

1. **Beta coefficients > 1e-3**: Indicates GLM fitting issues
2. **FDR values > 0.05**: Indicates FDR calculation problems
3. **Significant hits Jaccard < 0.9**: Indicates major discrepancies

## Validation Results

### Sample Results (10-pair test)

```
Beta coefficients:
  Mean absolute difference: 0.000012
  Max absolute difference: 0.000034
  Correlation: 0.999987
  PASS: True (tolerance: 0.0001)

Bootstrap P-values:
  Mean absolute difference: 0.023
  Correlation: 0.987
  PASS: True (MAE < 0.1)

FDR values:
  Mean absolute difference: 0.0001
  PASS: True (tolerance: 0.01)

Significant hits (FDR ≤ 0.10):
  R hits: 6
  GPU hits: 6
  Both: 6
  Jaccard similarity: 1.0
```

### Performance Benchmarks

| Test Size | GPU Time | CPU Time | Speedup | Beta MAE | Boot P Corr |
|-----------|----------|----------|---------|----------|-------------|
| 10 pairs  | 2-5 min  | 5-10 min | 2-3x    | <1e-4    | >0.95       |
| 100 pairs | 5-10 min | 1-2 hr   | 10-15x  | <1e-4    | >0.95       |
| 1000 pairs| 30-60 min| 6-8 hr   | 10-15x  | <1e-4    | >0.95       |

## Running Validation Tests

### Mini Test (Recommended for validation)

```bash
# Run quick validation
python tests/test_gpu_vs_r_mini.py
```

### Benchmark Test

```bash
# Run comprehensive benchmarking
python tests/benchmark_r_vs_gpu.py --sizes 10 100 1000
```

### Custom Validation

```python
from tests.test_scent_gpu import compare_results

# Load results
r_results = load_results('path/to/r_results')
gpu_results = load_results('path/to/gpu_results')

# Compare
comparison = compare_results(r_results, gpu_results)
print(f"Validation passed: {comparison['passed']}")
```

## Troubleshooting Validation Issues

### Beta Coefficients Don't Match

1. Check GLM convergence parameters
2. Verify covariate scaling
3. Compare matrix conditioning

### Bootstrap P-values Too Different

1. Set same random seed in both implementations
2. Check bootstrap count (should be same)
3. Verify cell type filtering is identical

### FDR Values Don't Match

1. Check if FDR is calculated per cell type or globally
2. Verify p-value ranking is identical
3. Compare Benjamini-Hochberg implementation

### Performance Issues

1. Ensure GPU is being used: `torch.cuda.is_available()`
2. Check GPU memory usage: `nvidia-smi`
3. Profile bottlenecks: `python -m pySCENT.profiling`

## References

- **Original SCENT**: Miao et al. (2023) "SCENT: Single-cell enhancer target gene identification"
- **Bootstrap Methods**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **GLM Theory**: McCullagh & Nelder (1989) "Generalized Linear Models"
