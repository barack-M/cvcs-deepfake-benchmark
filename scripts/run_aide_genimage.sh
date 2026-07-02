#!/bin/bash
#SBATCH --job-name=aide_genimage
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

CHUNK_IDX=$1
NUM_CHUNKS=4

if [ -n "$CHUNK_IDX" ]; then
    echo "=== MODALITÀ PARALLELA ATTIVA: ELABORAZIONE CHUNK $CHUNK_IDX OF OPENFAKE ==="
    python scripts/evaluate_aide.py --dataset openfake --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 32 --num_workers 6 --chunk_idx $CHUNK_IDX --num_chunks $NUM_CHUNKS
else
    echo "=== MODALITÀ SEQUENZIALE STANDARD ==="
    echo "--- Run 1/3: AIDE-GenImage su GAN ---"
    python scripts/evaluate_aide.py --dataset gan --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 32 --num_workers 6

    echo "--- Run 2/3: AIDE-GenImage su D3 ---"
    python scripts/evaluate_aide.py --dataset d3 --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 32 --num_workers 6

    echo "--- Run 3/3: AIDE-GenImage su OpenFake ---"
    python scripts/evaluate_aide.py --dataset openfake --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth --tag aide_genimage --batch_size 32 --num_workers 6
fi

echo "=== VALUTAZIONE AIDE-GENIMAGE COMPLETATA ==="
