#!/bin/bash
#SBATCH --job-name=ufd_robustness
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_robustness_%j.out
#SBATCH --account=cvcs2026

module unload cuda
module load cuda/12.6.3
module load anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base
source /homes/mbaracchi/cvcs2026/venv/bin/activate

echo "=== AVVIO VALUTAZIONE DI ROBUSTEZZA (JPEG + RESIZE) SU OPENFAKE ==="
python scripts/evaluate_robustness.py \
    --manifest /work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv \
    --parquet_dir /work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/ \
    --subset_size 200 \
    --output_report /work/cvcs2026/deep_pixels/results/robustness_report.md
