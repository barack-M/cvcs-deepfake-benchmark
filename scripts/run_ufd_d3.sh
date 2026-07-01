#!/bin/bash
#SBATCH --job-name=ufd_d3_eval
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/job_ufd_d3_%j.out
#SBATCH --account=cvcs2026

# Scarica eventuali moduli CUDA precaricati e carica la versione desiderata
module unload cuda
module load cuda/12.6.3

# Carica l'ambiente Anaconda globale di sistema (necessario per ereditare i pacchetti base)
module load anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base

# Attiva il virtual environment
source /homes/mbaracchi/cvcs2026/venv/bin/activate

# Esegui la valutazione ottimizzata sul dataset D3
python scripts/evaluate_ufd_d3.py \
    --manifest /work/cvcs2026/deep_pixels/datasets/D3/manifest.csv \
    --parquet_dir /work/cvcs2026/deep_pixels/datasets/D3/data \
    --output /work/cvcs2026/deep_pixels/results/ufd-d3.csv \
    --batch_size 64
