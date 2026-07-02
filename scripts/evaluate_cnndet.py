# -*- coding: utf-8 -*-
"""
Script unificato di inferenza CNNDetection (Wang et al., CVPR 2020) su GAN, D3 e OpenFake.
Produce un CSV compatibile con il benchmark ed aggregate_results.py.

Utilizzo:
    python scripts/evaluate_cnndet.py \
        --dataset [gan|d3|openfake] \
        --checkpoint /work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth
"""
import os
# Disabilitiamo il multi-threading interno delle librerie matematiche ed I/O per evitare conflitti e deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# === Percorsi progetto ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CNNDET_ROOT = PROJECT_ROOT / "CNNDetection"

# Aggiungiamo CNNDetection e il progetto al sys.path
sys.path.insert(0, str(CNNDET_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from networks.resnet import resnet50
from src.data.dataset import UnifiedDeepfakeDataset

# === Preprocessing ufficiale CNNDetection (scale_and_crop) ===
try:
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = Image.BILINEAR

# Replica le impostazioni di default: loadSize=256, cropSize=224
cnndet_transform = transforms.Compose([
    transforms.Resize(256, interpolation=BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser(description="Valutazione unificata CNNDetection su dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["gan", "d3", "openfake"],
                        help="Nome del dataset da valutare (gan, d3, openfake)")
    parser.add_argument("--checkpoint", type=str,
                        default="/work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth",
                        help="Percorso al checkpoint di CNNDetection")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per l'inferenza")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # === Controlla se l'inferenza è già stata completata ===
    output_path = Path("/work/cvcs2026/deep_pixels/results") / f"cnndet-{args.dataset}.csv"
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

    # === 2. Caricamento del modello CNNDetection ===
    print(f"Caricamento del modello CNNDetection da: {args.checkpoint}")
    model = resnet50(num_classes=1)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    
    # Riconciliamo le chiavi se carichiamo un checkpoint salvato dal loro script di train
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    model.eval()
    print("Modello CNNDetection caricato con successo.")

    # === 3. Inizializzazione Dataloader ===
    print(f"Caricamento del dataset {args.dataset.upper()}...")
    dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        openfake_parquet_dir=openfake_dir,
        d3_parquet_dir=d3_dir,
        transform=cnndet_transform
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
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()

            start_idx = i * args.batch_size
            for idx_offset, prob in enumerate(probs):
                global_idx = start_idx + idx_offset
                results.append({
                    "sample_id": global_idx,
                    "ground_truth": int(labels[idx_offset].item()),
                    "generator": generators[idx_offset],
                    "cnndet_score": float(prob.item())
                })

    # === 5. Salvataggio Risultati ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"Salvataggio completato! Predizioni salvate in: {output_path}")

    # Calcolo metriche globali indicative
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = df_results["ground_truth"].values
    y_scores = df_results["cnndet_score"].values
    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(f"AUROC Globale: {auroc:.4f} | AP Globale: {ap:.4f}")


if __name__ == "__main__":
    main()
