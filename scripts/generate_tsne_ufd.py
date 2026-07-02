# -*- coding: utf-8 -*-
import os
import sys
import io
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Aggiungiamo la cartella del progetto al sys.path
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
        
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA").convert("RGB")
    return img.convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Generazione t-SNE degli embedding CLIP per UFD su D3")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv",
                        help="Percorso al manifest.csv di D3")
    parser.add_argument("--parquet_dir", type=str, default="/work/cvcs2026/deep_pixels/datasets/D3/data",
                        help="Cartella dei file Parquet di D3")
    parser.add_argument("--subset_size", type=int, default=500,
                        help="Numero di immagini da usare per ciascun generatore/reale")
    parser.add_argument("--output_plot", type=str, default="/work/cvcs2026/deep_pixels/results/tsne-ufd-d3.png",
                        help="Percorso dove salvare l'immagine del grafico t-SNE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # Caricamento manifest
    df_manifest = pd.read_csv(args.manifest)

    # 1. CARICAMENTO MODELLO CLIP (Backbone di UFD)
    print("Caricamento del modello CLIP ViT-L/14...")
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()
    print("Modello CLIP pronto.")

    # 2. SELEZIONE DEI SUBSET DI IMMAGINI DAL MANIFEST
    # Selezioniamo subset_size immagini reali
    df_reals = df_manifest[df_manifest["label"] == 0].sample(n=args.subset_size, random_state=42)
    
    # Selezioniamo subset_size immagini fake per ciascun generatore
    fake_generators = [g for g in df_manifest["generator"].unique() if g != "real"]
    df_fakes_list = []
    for gen in fake_generators:
        df_gen = df_manifest[(df_manifest["generator"] == gen) & (df_manifest["label"] == 1)]
        df_fakes_list.append(df_gen.sample(n=min(len(df_gen), args.subset_size), random_state=42))
    
    df_selected = pd.concat([df_reals] + df_fakes_list)
    print(f"Totale immagini selezionate per il t-SNE: {len(df_selected)}")

    # 3. ESTRAZIONE DEGLI EMBEDDING
    print("\n--- Estrazione degli embedding ad alta dimensione da CLIP ---")
    embeddings = []
    labels = []
    generators = []

    # Carichiamo il dataset Parquet in modalità streaming per i fake
    import datasets
    parquet_files = os.path.join(args.parquet_dir, "*.parquet")
    parquet_dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")

    gen_cols = {
        "deepfloyd": "image_gen0",
        "sd14": "image_gen1",
        "sd21": "image_gen2",
        "sdxl": "image_gen3"
    }

    # Per velocizzare, estraiamo gli embedding riga per riga per le immagini selezionate
    with torch.no_grad():
        for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Extracting embeddings"):
            label = int(row["label"])
            generator = row["generator"]
            
            # Carica l'immagine corretta (da disco o da Parquet)
            if label == 0:
                img_path = row["path"]
                img = load_image_safely(img_path)
            else:
                parquet_idx = int(row["index"])
                col_name = gen_cols[generator]
                # Carichiamo la riga specifica del Parquet
                raw_data = parquet_dataset[parquet_idx][col_name]
                img = load_image_safely(raw_data)

            # Trasforma e calcola l'embedding CLIP (768 dimensioni)
            img_tensor = clip_transform(img).unsqueeze(0).to(device)
            # Encode image estrae le feature dal penultimo layer di CLIP
            feat = model.encode_image(img_tensor)
            # Normalizzazione del vettore di feature
            feat /= feat.norm(dim=-1, keepdim=True)
            
            embeddings.append(feat.cpu().numpy().flatten())
            labels.append(label)
            generators.append(generator)

    embeddings = np.array(embeddings)
    print(f"Matrice degli embedding estratta con successo! Dimensione: {embeddings.shape}")

    # 4. CALCOLO DEL t-SNE
    print("\nCalcolo del t-SNE (riduzione a 2 dimensioni)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE completato.")

    # 5. GENERAZIONE DEL GRAFICO
    print(f"\nGenerazione del grafico e salvataggio in: {args.output_plot}")
    plt.figure(figsize=(10, 8), dpi=150)

    # Definiamo colori e marcatori specifici per ciascun tipo
    color_map = {
        "real": "#2ca02c",        # Verde
        "deepfloyd": "#d62728",    # Rosso
        "sd14": "#1f77b4",         # Blu
        "sd21": "#ff7f0e",         # Arancione
        "sdxl": "#9467bd"          # Viola
    }
    
    label_map = {
        "real": "Real (LAION)",
        "deepfloyd": "Fake (DeepFloyd)",
        "sd14": "Fake (Stable Diffusion 1.4)",
        "sd21": "Fake (Stable Diffusion 2.1)",
        "sdxl": "Fake (Stable Diffusion XL)"
    }

    # Tracciamo i punti sul piano 2D suddivisi per generatore
    for gen in sorted(color_map.keys()):
        indices = [i for i, g in enumerate(generators) if g == gen]
        if not indices:
            continue
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            c=color_map[gen],
            label=label_map[gen],
            alpha=0.6,
            edgecolors="none",
            s=20
        )

    plt.title("Visualizzazione t-SNE dello Spazio dei Caratteri CLIP (UFD) su D3", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Dimensione t-SNE 1", fontsize=10)
    plt.ylabel("Dimensione t-SNE 2", fontsize=10)
    plt.legend(loc="best", fontsize=9, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Salva il grafico
    output_path = Path(args.output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Grafico salvato con successo!")

if __name__ == "__main__":
    main()
