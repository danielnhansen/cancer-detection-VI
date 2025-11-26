#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="breast-cancer-train"
#SBATCH --time=08:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"
# Use Slurm placeholders: %j=jobid, %x=jobname, %A=array jobid, %a=array index
#SBATCH --output=slurm_outputs/%j.txt
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --mail-user=danielnh@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1

pip install uv --user

# Sync the environment
uv sync

# Make sure PyTorch has CUDA enabled
python - <<'PY'
import torch
print("✅ PyTorch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("✅ GPU name(s):", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
else:
    print("⚠️  CUDA is NOT available. Check that you're using a GPU-enabled PyTorch build.")
PY

# Run the Python script
uv run train.py --config ./train_config.yaml  

echo "Job finished"