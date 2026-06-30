# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Aggiungiamo la cartella del progetto e di UFD al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "UniversalFakeDetect"))

from src.data.dataset import UnifiedDeepfakeDataset

# Definizione delle trasformazioni CLIP (Standard per UFD)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

clip_transform = Compose([
    Resize(224, interpolation=BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def main():
    parser = argparse.ArgumentParser(description="Valutazione del modello UniversalFakeDetect (UFD)")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/GAN/manifest.csv",
                        help="Percorso al file manifest.csv")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Numero di thread del dataloader")
    parser.add_argument("--output", type=str, default="/work/cvcs2026/deep_pixels/results/ufd-gan.csv",
                        help="File CSV dove salvare le predizioni")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # 1. Caricamento Modello UFD
    print("Inizializzazione del modello UFD (CLIP:ViT-L/14)...")
    from models import get_model
    model = get_model("CLIP:ViT-L/14")
    
    weights_path = Path("/work/cvcs2026/deep_pixels/weights/ufd/fc_weights.pth")
    print(f"Caricamento pesi classificatore lineare da: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("Modello UFD pronto per la valutazione.")

    # 2. Inizializzazione Dataset e DataLoader
    print(f"Caricamento manifest da: {args.manifest}")
    dataset = UnifiedDeepfakeDataset(
        manifest_path=args.manifest,
        transform=clip_transform
    )
    
    # Per il dataset GAN (CNN_synth) non ci sono file parquet, quindi il dataloader 
    # esegue solo letture di immagini fisiche da disco, risultando estremamente efficiente.
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Ciclo di Inferenza
    results = []
    print("Inizio inferenza su GPU...")
    with torch.no_grad():
        for i, (imgs, labels, generators, datasets) in enumerate(tqdm(dataloader, desc="Inference batches")):
            imgs = imgs.to(device)
            outputs = model(imgs)
            # UFD restituisce logits, applichiamo la sigmoid per ottenere le probabilità
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Calcoliamo gli indici globali del batch rispetto al manifest
            start_idx = i * args.batch_size
            for idx_offset, prob in enumerate(probs):
                global_idx = start_idx + idx_offset
                results.append({
                    "sample_id": global_idx,
                    "ground_truth": int(labels[idx_offset].item()),
                    "generator": generators[idx_offset],
                    "ufd_score": float(prob.item())
                })

    # 4. Salvataggio risultati
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"Salvataggio completato! Predizioni salvate in: {output_path}")

if __name__ == "__main__":
    main()
