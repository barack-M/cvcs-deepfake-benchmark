# -*- coding: utf-8 -*-
import os
import sys
import argparse
import io
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

# Funzione per caricare le immagini in modo sicuro, gestendo sia byte, percorsi stringa che oggetti PIL.
# Converte correttamente le palette trasparenti per evitare i warning di Pillow.
def load_image_safely(raw_data):
    if isinstance(raw_data, (str, Path)):
        img = Image.open(raw_data)
    elif isinstance(raw_data, Image.Image):
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

# Dataset di supporto per le sole immagini su disco (Reali di D3)
class SimpleDiskDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = load_image_safely(path)
        if self.transform:
            img = self.transform(img)
        return img, path

def main():
    parser = argparse.ArgumentParser(description="Valutazione ottimizzata di UFD sul dataset D3")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv",
                        help="Percorso al manifest.csv di D3")
    parser.add_argument("--parquet_dir", type=str, default="/work/cvcs2026/deep_pixels/datasets/D3/data",
                        help="Cartella dei file Parquet di D3")
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch per la GPU")
    parser.add_argument("--output", type=str, default="/work/cvcs2026/deep_pixels/results/ufd-d3.csv",
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

    # Dizionari per memorizzare i risultati prima dell'allineamento finale
    real_scores = {}  # path -> score
    fake_scores = {}  # (index, generator) -> score

    # 2. PASSAGGIO 1: VALUTAZIONE DELLE IMMAGINI REALI (SU DISCO)
    df_reals = df_manifest[df_manifest["label"] == 0]
    real_paths = df_reals["path"].tolist()
    print(f"\n--- Fase 1: Inferenza su {len(real_paths)} immagini reali da disco ---")
    
    real_dataset = SimpleDiskDataset(real_paths, transform=clip_transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    total_reals_processed = 0
    with torch.no_grad():
        for i, (imgs, paths) in enumerate(tqdm(real_loader, desc="Real images batches")):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            for path, prob in zip(paths, probs):
                real_scores[path] = float(prob.item())
            
            total_reals_processed += len(paths)
            if total_reals_processed % 500 == 0 or total_reals_processed == len(real_paths):
                print(f"[Progresso Reali] Elaborate {total_reals_processed}/{len(real_paths)} immagini reali...")

    # 3. PASSAGGIO 2: VALUTAZIONE DELLE IMMAGINI FAKE (DA PARQUET CON ACCESSO SEQUENZIALE)
    print(f"\n--- Fase 2: Inferenza su immagini fake da file Parquet (Caricamento sequenziale) ---")
    import datasets
    
    # Carichiamo il dataset Parquet in modalità lazy/streaming per velocizzare l'I/O
    parquet_files = os.path.join(args.parquet_dir, "*.parquet")
    print(f"Caricamento file Parquet da: {parquet_files}")
    parquet_dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
    
    # Mappatura colonne del Parquet rispetto ai nomi dei generatori del manifest
    gen_cols = {
        "deepfloyd": "image_gen0",
        "sd14": "image_gen1",
        "sd21": "image_gen2",
        "sdxl": "image_gen3"
    }

    # Definiamo la dimensione del batch di lettura delle righe dal parquet
    # Ogni riga ha 4 immagini generate, quindi leggere 32 righe produce 128 immagini da elaborare
    parquet_read_batch_size = max(1, args.batch_size // 4)
    total_rows = len(parquet_dataset)
    
    print(f"Righe totali del Parquet da scansionare: {total_rows}")
    print(f"Dimensione del batch di lettura: {parquet_read_batch_size} righe (equivalente a {parquet_read_batch_size * 4} immagini per batch)")
    
    pbar = tqdm(total=total_rows, desc="Reading Parquet rows")
    
    row_idx = 0
    total_fakes_processed = 0
    with torch.no_grad():
        for batch in parquet_dataset.iter(batch_size=parquet_read_batch_size):
            batch_len = len(batch["image_gen0"])
            
            # Prepariamo la lista dei tensor per questo batch
            batch_tensors = []
            batch_metadata = []  # conterrà le tuple (index, generator_name)
            
            for i in range(batch_len):
                global_row_index = row_idx + i
                
                # Per ciascuna delle 4 colonne di generatori presenti nella riga
                for gen_name, col_name in gen_cols.items():
                    raw_data = batch[col_name][i]
                    
                    # Caricamento sicuro senza dipendere dal tipo di dato grezzo
                    img = load_image_safely(raw_data)
                    img_tensor = clip_transform(img)
                    
                    batch_tensors.append(img_tensor)
                    batch_metadata.append((global_row_index, gen_name))
            
            # Se abbiamo accumulato immagini, le passiamo alla GPU
            if batch_tensors:
                stacked_tensors = torch.stack(batch_tensors).to(device)
                
                # Elaboriamo in sotto-batch se la dimensione supera args.batch_size
                num_imgs = len(batch_tensors)
                probs_list = []
                for sub_idx in range(0, num_imgs, args.batch_size):
                    sub_tensors = stacked_tensors[sub_idx:sub_idx+args.batch_size]
                    outputs = model(sub_tensors)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    probs_list.extend(probs)
                
                # Salviamo i risultati nel dizionario
                for (g_idx, gen_name), prob in zip(batch_metadata, probs_list):
                    fake_scores[(g_idx, gen_name)] = float(prob.item())
                    total_fakes_processed += 1
            
            row_idx += batch_len
            pbar.update(batch_len)
            
            if row_idx % 400 == 0 or row_idx == total_rows:
                print(f"[Progresso Fake] Scansionate {row_idx}/{total_rows} righe Parquet (immagini elaborate: {total_fakes_processed})...")
            
    pbar.close()

    # 4. PASSAGGIO 3: ALLINEAMENTO E COSTRUZIONE DEL CSV FINALE
    print("\n--- Fase 3: Allineamento dei risultati con il manifest ---")
    results = []
    
    # Iteriamo sul manifest originale riga per riga per garantire lo stesso ordine
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Aligning rows"):
        label = int(row["label"])
        generator = row["generator"]
        
        score = None
        if label == 0:
            path = row["path"]
            score = real_scores.get(path, 0.0)
        else:
            parquet_idx = int(row["index"])
            score = fake_scores.get((parquet_idx, generator), 0.0)
            
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
