# Mini Test Cases for pySCENT

This directory contains small test datasets for validating pySCENT against the original R implementation.

## Test Files

- `peak_info_mini_10.tsv`: 10 gene-peak pairs for quick validation
- `peak_info_mini_100.tsv`: 100 gene-peak pairs for performance benchmarking

## Usage

### Quick Validation (10 pairs)

```bash
# Run GPU version
python scripts/run_scent_gpu.py \
    --dataset test-dataset \
    --processed_dir /path/to/processed/data \
    --peak_info examples/mini_test/peak_info_mini_10.tsv \
    --output_dir test_output \
    --chunk_id mini_10 \
    --device cuda

# Compare with R version (requires R SCENT)
python tests/test_gpu_vs_r_mini.py
```

### Benchmarking (100 pairs)

```bash
# Run benchmark
python tests/benchmark_r_vs_gpu.py --sizes 100
```

## Data Format

Each file contains tab-separated gene-peak pairs:
```
GENE_SYMBOL<TAB>chr:start-end
```

Example:
```
HS2ST1	chr1:87131830-87132030
CTPS1	chr1:40692059-40692259
```

## Expected Results

### 10-pair test
- Should complete in 2-5 minutes on GPU
- Results should match R implementation within tolerance
- Beta coefficients: |diff| < 1e-4
- Bootstrap p-values: |diff| < 0.1

### 100-pair test
- Should complete in 5-10 minutes on GPU
- 10-15x speedup over CPU implementation
- Same accuracy tolerances as 10-pair test

## Notes

- These are synthetic test cases for validation
- For real datasets, use full peak_info files from processed data
- Ensure your processed data directory contains RNA/ATAC matrices and metadata
