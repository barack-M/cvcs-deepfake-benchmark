# -*- coding: utf-8 -*-
import os
import sys
import argparse
import io
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Aggiungiamo la cartella del progetto e di UFD al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "UniversalFakeDetect"))

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

# Funzione per caricare le immagini in modo sicuro
def load_image_safely(raw_data):
    if isinstance(raw_data, Image.Image):
        img = raw_data
    elif isinstance(raw_data, dict):
        bytes_data = raw_data.get("bytes")
        img = Image.open(io.BytesIO(bytes_data))
    else:
        img = Image.open(io.BytesIO(raw_data))
        
    # Evitiamo il warning "Palette images with Transparency" convertendo prima in RGBA
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA").convert("RGB")
    return img.convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Valutazione ottimizzata di UFD sul dataset OpenFake")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv",
                        help="Percorso al manifest.csv di OpenFake")
    parser.add_argument("--parquet_dir", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core",
                        help="Cartella dei file Parquet di OpenFake")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch per la GPU")
    parser.add_argument("--output", type=str, default="/work/cvcs2026/deep_pixels/results/ufd-openfake.csv",
                        help="File CSV finale dei risultati")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # Caricamento manifest
    print(f"Caricamento manifest da: {args.manifest}")
    df_manifest = pd.read_csv(args.manifest)

    # 1. INIZIALIZZAZIONE MODELLO UFD
    print("Inizializzazione del modello UFD (CLIP:ViT-L/14)...")
    from models import get_model
    model = get_model("CLIP:ViT-L/14")
    
    weights_path = Path("/work/cvcs2026/deep_pixels/weights/ufd/fc_weights.pth")
    print(f"Caricamento pesi classificatore lineare da: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("Modello UFD caricato con successo.")

    # Dizionario per memorizzare i risultati indicizzati
    scores_dict = {}  # index_parquet -> score

    # 2. PASSAGGIO: VALUTAZIONE SEQUENZIALE DEL PARQUET DI OPENFAKE
    print(f"\n--- Fase 1: Inferenza sul dataset Parquet di OpenFake (Caricamento sequenziale) ---")
    import datasets
    
    parquet_files = os.path.join(args.parquet_dir, "*.parquet")
    print(f"Caricamento file Parquet da: {parquet_files}")
    parquet_dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
    
    total_rows = len(parquet_dataset)
    print(f"Righe totali del Parquet da scansionare: {total_rows}")
    
    pbar = tqdm(total=total_rows, desc="Reading Parquet rows")
    
    row_idx = 0
    total_processed = 0
    with torch.no_grad():
        for batch in parquet_dataset.iter(batch_size=args.batch_size):
            batch_len = len(batch["image"])
            
            batch_tensors = []
            batch_indices = []
            
            for i in range(batch_len):
                global_row_index = row_idx + i
                raw_data = batch["image"][i]
                
                # Caricamento sicuro dell'immagine
                try:
                    img = load_image_safely(raw_data)
                    img_tensor = clip_transform(img)
                    batch_tensors.append(img_tensor)
                    batch_indices.append(global_row_index)
                except Exception as e:
                    print(f"Errore di caricamento alla riga {global_row_index}: {e}. Salto l'immagine.")
            
            # Inferenza sulla GPU per il batch
            if batch_tensors:
                stacked_tensors = torch.stack(batch_tensors).to(device)
                outputs = model(stacked_tensors)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # Memorizziamo i punteggi nel dizionario temporaneo
                for g_idx, prob in zip(batch_indices, probs):
                    scores_dict[g_idx] = float(prob.item())
                    total_processed += 1
            
            row_idx += batch_len
            pbar.update(batch_len)
            
            if row_idx % 2000 == 0 or row_idx == total_rows:
                print(f"[Progresso OpenFake] Elaborate {row_idx}/{total_rows} righe Parquet (immagini elaborate con successo: {total_processed})...")
            
    pbar.close()

    # 3. PASSAGGIO: ALLINEAMENTO E COSTRUZIONE DEL CSV FINALE
    print("\n--- Fase 2: Allineamento dei risultati con il manifest ---")
    results = []
    
    # Iteriamo sul manifest originale riga per riga per garantire lo stesso ordine
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Aligning rows"):
        label = int(row["label"])
        generator = row["generator"]
        parquet_idx = int(row["index"])
        
        # Recuperiamo lo score se presente, altrimenti default a 0.0 (es. se un'immagine era corrotta)
        score = scores_dict.get(parquet_idx, 0.0)
            
        results.append({
            "sample_id": idx,
            "ground_truth": label,
            "generator": generator,
            "ufd_score": score
        })

    # Salvataggio file finale
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nSalvataggio completato! Predizioni salvate in: {output_path}")

if __name__ == "__main__":
    main()
