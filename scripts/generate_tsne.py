# -*- coding: utf-8 -*-
"""
Script unificato per la generazione di grafici t-SNE / UMAP degli embedding CLIP.
Aiuta a visualizzare la separabilità dei reali rispetto alle diverse famiglie di fake.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Aggiungiamo la cartella del progetto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "UniversalFakeDetect"))

from src.data.dataset import UnifiedDeepfakeDataset

# Definizione delle trasformazioni CLIP
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
    parser = argparse.ArgumentParser(description="Generazione t-SNE / UMAP degli embedding CLIP per UFD")
    parser.add_argument("--dataset", type=str, required=True, choices=["gan", "d3", "openfake"],
                        help="Dataset su cui calcolare il t-SNE")
    parser.add_argument("--subset_size", type=int, default=300,
                        help="Numero massimo di campioni per generatore/reale")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"],
                        help="Algoritmo di riduzione dimensionale (tsne, umap)")
    parser.add_argument("--output_plot", type=str, default=None,
                        help="Percorso dove salvare l'immagine del plot (default in results/)")
    args = parser.parse_args()

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

    if args.output_plot is None:
        args.output_plot = str((Path(__file__).resolve().parent.parent / "results") / f"{args.method}-clip-{args.dataset}.png")

    # === 2. Caricamento del dataset ===
    print(f"Caricamento del dataset {args.dataset.upper()}...")
    dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        openfake_parquet_dir=openfake_dir,
        d3_parquet_dir=d3_dir,
        transform=clip_transform
    )
    df = dataset.df

    # === 3. Selezione del subset bilanciato ===
    # Campioni reali (label=0)
    df_reals = df[df["label"] == 0]
    sampled_reals_idx = df_reals.sample(n=min(len(df_reals), args.subset_size * 2), random_state=42).index.tolist()
    
    # Campioni fake (label=1) disaggregati per generatore
    fake_generators = sorted([g for g in df["generator"].unique() if g not in ["imagenet", "docci", "real", "real_laion"]])
    
    selected_indices = sampled_reals_idx.copy()
    for gen in fake_generators:
        df_gen = df[(df["generator"] == gen) & (df["label"] == 1)]
        sampled_fake_idx = df_gen.sample(n=min(len(df_gen), args.subset_size), random_state=42).index.tolist()
        selected_indices.extend(sampled_fake_idx)
        
    print(f"Totale campioni selezionati: {len(selected_indices)} (Reali: {len(sampled_reals_idx)}, Fake: {len(selected_indices) - len(sampled_reals_idx)})")

    # === 4. Caricamento modello CLIP Visual ===
    print("Caricamento del modello CLIP ViT-L/14...")
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()
    print("Modello CLIP caricato.")

    # === 5. Estrazione degli embedding con DataLoader ===
    print("Estrazione degli embedding CLIP con DataLoader in parallelo...")
    from torch.utils.data import Subset, DataLoader
    
    subset_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    embeddings = []
    labels = []
    generators = []
    
    with torch.no_grad():
        for img_tensors, lbls, gens, _ in tqdm(dataloader, desc="Extracting CLIP features"):
            img_tensors = img_tensors.to(device)
            feats = model.encode_image(img_tensors)
            feats /= feats.norm(dim=-1, keepdim=True)
            
            embeddings.append(feats.cpu().numpy())
            labels.extend(lbls.tolist())
            generators.extend(gens)
            
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Matrice degli embedding estratta con successo! Dimensione: {embeddings.shape}")

    # === 6. Riduzione dimensionale ===
    print(f"\nEsecuzione della riduzione dimensionale con {args.method.upper()}...")
    if args.method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif args.method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("[WARNING] Libreria 'umap-learn' non installata. Ripiego su t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            args.method = "tsne"
            
    print("Riduzione dimensionale completata.")

    # === 7. Generazione e salvataggio del plot ===
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Palette colori qualitativa (fino a 20 colori)
    cmap = plt.cm.get_cmap("tab20", len(fake_generators) + 2)
    
    # Colore unico per i reali
    color_real = "#2ca02c" # Verde foresta
    
    # Tracciamento dei reali (raggruppando imagenet e docci)
    real_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    plt.scatter(
        embeddings_2d[real_indices, 0],
        embeddings_2d[real_indices, 1],
        c=color_real,
        label="Real Images",
        alpha=0.6,
        edgecolors="none",
        s=15
    )
    
    # Tracciamento di ciascun generatore fake
    for i, gen in enumerate(fake_generators):
        indices = [idx for idx, g in enumerate(generators) if g == gen and labels[idx] == 1]
        if not indices:
            continue
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            c=[cmap(i)],
            label=f"Fake ({gen})",
            alpha=0.6,
            edgecolors="none",
            s=15
        )
        
    plt.title(f"Visualizzazione {args.method.upper()} dello Spazio CLIP su {args.dataset.upper()}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"Dimensione {args.method.upper()} 1", fontsize=10)
    plt.ylabel(f"Dimensione {args.method.upper()} 2", fontsize=10)
    
    # Posiziona la legenda all'esterno del grafico per non coprire i punti
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    output_path = Path(args.output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Grafico salvato con successo in: {output_path}")

if __name__ == "__main__":
    main()
