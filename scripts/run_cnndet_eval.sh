#!/bin/bash
#SBATCH --job-name=cnndet_eval
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/job_cnndet_eval_%j.out
#SBATCH --account=cvcs2026

module unload cuda
module load cuda/12.6.3
module load anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base
source /homes/mbaracchi/cvcs2026/venv/bin/activate

echo "=== INIZIO VALUTAZIONE CNNDETECTION SU TUTTI I DATASET ==="

echo "--- Run 1/3: CNNDetection su GAN ---"
python scripts/evaluate_cnndet.py --dataset gan --checkpoint /work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth --batch_size 64

echo "--- Run 2/3: CNNDetection su D3 ---"
python scripts/evaluate_cnndet.py --dataset d3 --checkpoint /work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth --batch_size 64

echo "--- Run 3/3: CNNDetection su OpenFake ---"
python scripts/evaluate_cnndet.py --dataset openfake --checkpoint /work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth --batch_size 64

echo "=== VALUTAZIONE CNNDETECTION COMPLETATA ==="
