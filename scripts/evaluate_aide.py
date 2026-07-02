# -*- coding: utf-8 -*-
"""
Script unificato di inferenza AIDE (Yan et al., ICLR 2025) su GAN, D3 e OpenFake.
Produce un CSV compatibile con il benchmark ed aggregate_results.py.

Utilizzo:
    python scripts/evaluate_aide.py \
        --dataset [gan|d3|openfake] \
        --checkpoint /work/cvcs2026/deep_pixels/weights/aide/aide_progan.pth \
        --tag aide_progan
"""
import os
# Disabilitiamo il multi-threading interno delle librerie matematiche ed I/O per evitare conflitti e deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import io
import argparse
from pathlib import Path
import torch.multiprocessing as mp

try:
    # Usiamo spawn al posto di fork per evitare deadlock con PyArrow nei worker
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import pyarrow as pa
# Limitiamo pyarrow a 1 thread ma lasciamo che PyTorch usi i thread di sistema
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm

# === Percorsi progetto ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIDE_ROOT = PROJECT_ROOT / "AIDE"

# Aggiungiamo AIDE e il progetto al sys.path
sys.path.insert(0, str(AIDE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models.AIDE import AIDE as build_aide_model
from data.dct import DCT_base_Rec_Module
from src.data.dataset import UnifiedDeepfakeDataset

# === Preprocessing di AIDE ===
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

transform_normalize = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dct_module = DCT_base_Rec_Module()

def preprocess_aide(img_pil):
    """
    Replica il preprocessing di AIDE:
    1. ToTensor
    2. DCT → 4 view (minmin, maxmax, minmin1, maxmax1)
    3. Normalize tutte le 5 view (4 DCT + originale)
    4. Stack → [5, C, H, W]
    """
    img_tensor = transform_to_tensor(img_pil)  # [C, H, W]
    try:
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_tensor)
    except Exception as e:
        # Fallback se l'immagine è troppo piccola per la decomposizione DCT 32x32
        x_minmin = x_maxmax = x_minmin1 = x_maxmax1 = img_tensor

    x_0 = transform_normalize(img_tensor)
    x_minmin = transform_normalize(x_minmin)
    x_maxmax = transform_normalize(x_maxmax)
    x_minmin1 = transform_normalize(x_minmin1)
    x_maxmax1 = transform_normalize(x_maxmax1)

    return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)


def main():
    parser = argparse.ArgumentParser(description="Valutazione unificata AIDE su dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["gan", "d3", "openfake"],
                        help="Nome del dataset da valutare (gan, d3, openfake)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Percorso al checkpoint AIDE (es. aide_progan.pth)")
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag identificativo della run (es. aide_progan, aide_genimage)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per l'inferenza (AIDE è pesante)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # === Controlla se l'inferenza è già stata completata ===
    output_path = Path("/work/cvcs2026/deep_pixels/results") / f"{args.tag}-{args.dataset}.csv"
    if output_path.exists():
        print(f"Risultati già presenti in {output_path}. Salto l'inferenza.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # === 1. Mappatura Dataset e Path ===
    if args.dataset == "gan":
        manifest_path = "/work/cvcs2026/deep_pixels/datasets/GAN/manifest.csv"
        openfake_dir = None
        d3_dir = None
    elif args.dataset == "d3":
        manifest_path = "/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv"
        openfake_dir = None
        d3_dir = "/work/cvcs2026/deep_pixels/datasets/D3/data"
    elif args.dataset == "openfake":
        manifest_path = "/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv"
        openfake_dir = "/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/"
        d3_dir = None

    # === 2. Caricamento del modello AIDE ===
    print(f"Caricamento del modello AIDE dal checkpoint: {args.checkpoint}")
    model = build_aide_model(resnet_path=None, convnext_path="laion2b_s34b_b82k_augreg_soup")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # === 3. Inizializzazione Dataloader ===
    print(f"Caricamento del dataset {args.dataset.upper()}...")
    dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        openfake_parquet_dir=openfake_dir,
        d3_parquet_dir=d3_dir,
        transform=preprocess_aide
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Totale immagini: {len(dataset)}")

    # === 4. Inferenza ===
    results = []
    print("Inizio inferenza su GPU...")
    with torch.no_grad():
        for i, (imgs, labels, generators, datasets) in enumerate(tqdm(dataloader, desc="Inference batches")):
            imgs = imgs.to(device)
            logits = model(imgs)  # [B, 2]
            probs = softmax(logits.cpu().numpy(), axis=1)[:, 1]  # probabilità classe fake

            start_idx = i * args.batch_size
            for idx_offset, prob in enumerate(probs):
                global_idx = start_idx + idx_offset
                results.append({
                    "sample_id": global_idx,
                    "ground_truth": int(labels[idx_offset].item()),
                    "generator": generators[idx_offset],
                    f"{args.tag}_score": float(prob.item())
                })

    # === 5. Salvataggio Risultati ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"Salvataggio completato! Predizioni salvate in: {output_path}")

    # Calcolo metriche globali indicative
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = df_results["ground_truth"].values
    y_scores = df_results[f"{args.tag}_score"].values
    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(f"AUROC Globale: {auroc:.4f} | AP Globale: {ap:.4f}")


if __name__ == "__main__":
    main()
