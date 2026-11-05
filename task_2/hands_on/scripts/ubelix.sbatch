#!/usr/bin/env bash
# Submit from task_2/hands_on/ after creating an environment on UBELIX.
# Override the GPU type/account on the sbatch command line when appropriate.
#SBATCH --job-name=world-model-interview
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
: "${WM_PYTHON:?Set WM_PYTHON to the absolute Python path in your UBELIX environment}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/wm-matplotlib-${SLURM_JOB_ID}"
mkdir -p outputs
nvidia-smi --query-gpu=name,memory.total --format=csv
"$WM_PYTHON" -m scripts.make_data
"$WM_PYTHON" -m scripts.make_data --output data/heldout_pong.npz --episodes 32 --seed 90000
"$WM_PYTHON" -m world_model.trainer --device cuda --output outputs/run
"$WM_PYTHON" -m scripts.diagnose --device cuda --checkpoint outputs/run/model.pt --output outputs/diagnostics
