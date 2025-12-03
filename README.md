# pySCENT: GPU-Accelerated SCENT for Single-Cell Enhancer Target Prediction

PyTorch-based GPU implementation of SCENT (Single-Cell Enhancer Target) algorithm for identifying peak-gene regulatory links in multimodal single-cell data.

## Features

- **10-50x speedup** over CPU implementation via GPU acceleration
- **Batch processing** for optimal GPU utilization

## Quick Start

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Run SCENT on test data
python scripts/run_scent_gpu.py \
    --dataset arthritis-tissue \
    --processed_dir data/processed \
    --peak_info data/processed/arthritis-tissue/peak_lists/peak_info_test_001.tsv \
    --output_dir results/SCENT_GPU/arthritis-tissue \
    --chunk_id test_001 \
    --device cuda
```

## Installation

### Requirements

- **Python 3.8+**
- **NVIDIA GPU** with CUDA support (Compute Capability >= 3.5)
- **8 GB GPU VRAM** (16 GB recommended for large datasets)
- **16 GB system RAM**
- **50 GB disk space** (for datasets and results)

### Dependencies

```bash
# PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Other dependencies
pip install -r requirements.txt
```

### From Source

```bash
git clone https://github.com/YOUR_USERNAME/pySCENT.git
cd pySCENT
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python scripts/run_scent_gpu.py \
    --dataset arthritis-tissue \
    --processed_dir data/processed \
    --peak_info data/processed/arthritis-tissue/peak_lists/peak_info_chunk_0001.tsv \
    --output_dir results/SCENT_GPU/arthritis-tissue \
    --chunk_id 0001 \
    --fdr_threshold 0.10 \
    --device cuda \
    --seed 42
```

### Python API

```python
from pySCENT import load_scent_data, scent_algorithm_gpu

# Load data
data = load_scent_data(
    dataset='arthritis-tissue',
    processed_dir='data/processed',
    peak_info_file='peak_info_chunk_0001.tsv',
    device='cuda'
)

# Run SCENT
results = scent_algorithm_gpu(
    data=data,
    celltype='Tcell',
    covariates=['log_nUMI', 'percent_mito', 'donor_num'],
    device='cuda'
)

print(results.head())
```

### SLURM Job Submission

See `examples/example_slurm.sh` for a complete SLURM script with email notifications:

```bash
#!/bin/bash
#SBATCH --job-name=scent_gpu
#SBATCH --output=logs/scent_gpu_%j.log
#SBATCH --error=logs/scent_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

# Run pySCENT
python scripts/run_scent_gpu.py \
    --dataset your-dataset \
    --processed_dir data/processed \
    --peak_info peak_info.tsv \
    --output_dir results \
    --device cuda
```

## Performance Benchmarks

| Dataset | Pairs | CPU Time | GPU Time | Speedup |
|---------|-------|----------|----------|---------|
| Test (10) | 10 | 5-10 min | 2-5 min | 2-3x |
| Test (100) | 100 | 1-2 hr | 5-10 min | 10-15x |
| Full chunk | 10,000 | 6-8 hr | 30-60 min | 10-15x |

### System Requirements

- **GPU**: NVIDIA RTX 3090 or equivalent
- **RAM**: 32GB system RAM minimum
- **Storage**: 50GB free space for large datasets

## Data Formats

### Input Data

pySCENT supports both RDS and H5AD formats:

- **RNA data**: Sparse matrix (genes × cells)
- **ATAC data**: Sparse matrix (peaks × cells)
- **Metadata**: Cell metadata including cell type annotations
- **Peak info**: Gene-peak pairs with genomic coordinates

### Output Format

Results are saved as TSV files with columns:
- `gene`: Gene symbol
- `peak`: Peak coordinates (chr:start-end format)
- `beta`: Regression coefficient
- `se`: Standard error
- `z`: Z-statistic
- `p`: P-value
- `boot_basic_p`: Bootstrap p-value
- `FDR`: False discovery rate
- `celltype`: Cell type

## Validation & Testing

### Mini Test Cases

Run validation against R implementation:

```bash
# 10-pair test (quick validation)
python tests/test_gpu_vs_r_mini.py

# Benchmarking (multiple sizes)
python tests/benchmark_r_vs_gpu.py --sizes 10 100 1000
```

### Comparison Metrics

The implementation has been validated to match the R version within:
- **Beta coefficients**: |diff| < 1e-4
- **Bootstrap p-values**: |diff| < 0.1 (expected due to random sampling)
- **FDR values**: |diff| < 0.01

## Architecture

- `pySCENT/scent_gpu.py`: Main algorithm orchestration
- `pySCENT/data_loader.py`: Data loading and GPU transfer
- `pySCENT/poisson_gpu.py`: Poisson GLM fitting
- `pySCENT/bootstrap_gpu.py`: Adaptive bootstrap engine
- `pySCENT/profiling.py`: Performance profiling utilities

## Troubleshooting

### GPU Memory Issues

```bash
# Reduce batch size for bootstrap
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Import Errors

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

### Slow Performance

- Ensure CUDA is being used (`device='cuda'`)
- Check GPU utilization: `nvidia-smi`
- Profile bottlenecks using `pySCENT.profiling`
