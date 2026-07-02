#!/bin/bash
#SBATCH --job-name=aide_genimage
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_aide_genimage_%j.out
#SBATCH --account=cvcs2026

set -e

module unload cuda
module load cuda/12.6.3
module load anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base
source /homes/mbaracchi/cvcs2026/venv/bin/activate

echo "=== INIZIO VALUTAZIONE AIDE-GENIMAGE SU TUTTI I DATASET ==="

echo "--- Run 1/3: AIDE-GenImage su GAN ---"
python scripts/evaluate_aide.py --dataset gan --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 16

echo "--- Run 2/3: AIDE-GenImage su D3 ---"
python scripts/evaluate_aide.py --dataset d3 --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 16

echo "--- Run 3/3: AIDE-GenImage su OpenFake ---"
python scripts/evaluate_aide.py --dataset openfake --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 16

echo "=== VALUTAZIONE AIDE-GENIMAGE COMPLETATA ==="
