#!/bin/bash
#SBATCH --job-name=ufd_tsne
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/job_tsne_%j.out
#SBATCH --account=cvcs2026

module unload cuda
module load cuda/12.6.3
module load anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base
source /homes/mbaracchi/cvcs2026/venv/bin/activate

echo "=== AVVIO GENERAZIONE GRAFICO t-SNE (UFD su D3) ==="
python scripts/generate_tsne_ufd.py \
    --manifest /work/cvcs2026/deep_pixels/datasets/D3/manifest.csv \
    --parquet_dir /work/cvcs2026/deep_pixels/datasets/D3/data \
    --subset_size 300 \
    --output_plot /work/cvcs2026/deep_pixels/results/tsne-ufd-d3.png
