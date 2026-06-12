#!/bin/bash
#SBATCH --job-name=gpu-burn
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load nvhpc
cd "${SLURM_SUBMIT_DIR}"

# These can be overridden when submitting with:
# sbatch --export=ALL,PRECISION=fp16,DURATION=600,MEMORY_PERCENT=90 gpu_burn.slurm
PRECISION="${PRECISION:-fp32}"
DURATION="${DURATION:-600}"
MEMORY_PERCENT="${MEMORY_PERCENT:-90}"

make

echo "Running on ${SLURMD_NODENAME} with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
srun ./gpu_burn --precision "${PRECISION}" --memory "${MEMORY_PERCENT}" "${DURATION}"
