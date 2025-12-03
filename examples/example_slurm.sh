#!/bin/bash
#SBATCH --job-name=pyscent_gpu
#SBATCH --output=logs/pyscent_gpu_%j.log
#SBATCH --error=logs/pyscent_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# pySCENT GPU SLURM Job Script
# Example script for running pySCENT on a GPU cluster
#
# Customize the following variables:
# - EMAIL: Your email address for notifications
# - DATASET: Name of your dataset
# - PROCESSED_DIR: Path to processed data directory
# - PEAK_INFO: Path to peak_info file
# - OUTPUT_DIR: Output directory for results
# - CONDA_ENV: Name of conda environment

# ==============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ==============================================================================

# Email for job notifications (change this!)
EMAIL="your.email@example.com"

# Dataset configuration
DATASET="your-dataset"
PROCESSED_DIR="/path/to/processed/data"
PEAK_INFO="${PROCESSED_DIR}/${DATASET}/peak_lists/peak_info_chunk_001.tsv"
OUTPUT_DIR="/path/to/output/results/SCENT_GPU/${DATASET}"

# Environment configuration
CONDA_ENV="pyscent_env"  # Change to your conda environment name

# Job configuration
FDR_THRESHOLD=0.10
SEED=42

# ==============================================================================
# JOB SETUP
# ==============================================================================

set -e  # Exit on any error

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print job information
echo "================================================================================"
echo "PYSCENT GPU JOB"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPU: $SLURM_GPUS"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Peak info: $PEAK_INFO"
echo "  Output dir: $OUTPUT_DIR"
echo "  FDR threshold: $FDR_THRESHOLD"
echo "  Random seed: $SEED"
echo "================================================================================"
echo ""

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."

# Load required modules (uncomment and modify as needed)
# module load cuda/11.8
# module load anaconda3

# Activate conda environment
# Try multiple common conda paths
if command -v conda &> /dev/null; then
    echo "Found conda in PATH"
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda installation"
    exit 1
fi

# Activate environment
echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV'"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Verify GPU availability
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found"
fi

# Verify PyTorch CUDA
echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""

# ==============================================================================
# RUN PYSCENT
# ==============================================================================

echo "Starting pySCENT..."
echo "Command: python scripts/run_scent_gpu.py \\"
echo "    --dataset $DATASET \\"
echo "    --processed_dir $PROCESSED_DIR \\"
echo "    --peak_info $PEAK_INFO \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --chunk_id ${SLURM_ARRAY_TASK_ID:-001} \\"
echo "    --fdr_threshold $FDR_THRESHOLD \\"
echo "    --device cuda \\"
echo "    --seed $SEED"
echo ""

# Record start time
START_TIME=$(date +%s)

/usr/bin/time -v python scripts/run_scent_gpu.py \
    --dataset "$DATASET" \
    --processed_dir "$PROCESSED_DIR" \
    --peak_info "$PEAK_INFO" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-001}" \
    --fdr_threshold "$FDR_THRESHOLD" \
    --device cuda \
    --seed "$SEED"

EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PYSCENT COMPLETED SUCCESSFULLY"
    echo "Job completed in $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
else
    echo "❌ PYSCENT FAILED with exit code $EXIT_CODE"
    echo "Check the error log: logs/pyscent_gpu_${SLURM_JOB_ID}.err"
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Finished: $(date)"
echo "================================================================================"

# ==============================================================================
# CLEANUP (optional)
# ==============================================================================

# Deactivate conda environment
conda deactivate

exit $EXIT_CODE
