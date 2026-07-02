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
from sklearn.metrics import roc_auc_score

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

# Funzione per caricare le immagini in modo sicuro
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

# Funzione per applicare la compressione JPEG in-memory
def compress_jpeg_in_memory(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# Funzione per applicare il ridimensionamento (downsampling)
def resize_in_memory(img, size=128):
    # Ridimensioniamo a bassa risoluzione (bilineare) per perdere dettagli
    img_down = img.resize((size, size), Image.BILINEAR)
    return img_down

def main():
    parser = argparse.ArgumentParser(description="Valutazione robustezza UFD sotto alterazioni (JPEG + Resize) su OpenFake")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv",
                        help="Percorso al manifest.csv di OpenFake")
    parser.add_argument("--parquet_dir", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/",
                        help="Cartella dei file Parquet di OpenFake")
    parser.add_argument("--subset_size", type=int, default=200,
                        help="Numero di immagini da usare per ciascun generatore/reale")
    parser.add_argument("--output_report", type=str, default="/work/cvcs2026/deep_pixels/results/robustness_report.md",
                        help="Percorso per salvare il report markdown della robustezza")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # Caricamento manifest
    df_manifest = pd.read_csv(args.manifest)

    # 1. CARICAMENTO MODELLO UFD
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

    # 2. SELEZIONE DEI SUBSET DI IMMAGINI DAL MANIFEST (200 per sorgente reale e 200 per generatore fake)
    # Reali: imagenet e docci
    df_reals_imagenet = df_manifest[(df_manifest["label"] == 0) & (df_manifest["generator"] == "imagenet")].sample(n=args.subset_size, random_state=42)
    df_reals_docci = df_manifest[(df_manifest["label"] == 0) & (df_manifest["generator"] == "docci")].sample(n=args.subset_size, random_state=42)
    df_reals = pd.concat([df_reals_imagenet, df_reals_docci])
    
    # Fake generators
    fake_generators = [g for g in df_manifest["generator"].unique() if g not in ["imagenet", "docci"]]
    df_fakes_list = []
    for gen in fake_generators:
        df_gen = df_manifest[(df_manifest["generator"] == gen) & (df_manifest["label"] == 1)]
        df_fakes_list.append(df_gen.sample(n=min(len(df_gen), args.subset_size), random_state=42))
    
    df_selected = pd.concat([df_reals] + df_fakes_list)
    total_imgs = len(df_selected)
    print(f"Immagini selezionate per il test di robustezza: {total_imgs}")

    # Carichiamo il dataset Parquet in modalità streaming
    import datasets
    parquet_files = os.path.join(args.parquet_dir, "*.parquet")
    parquet_dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")

    # 3. CICLO DI INFERENZA CON ALTERAZIONI (Stesse immagini valutate 4 volte)
    pipeline_modes = ["Clean", "JPEG_70", "JPEG_50", "Resize_128"]
    predictions = {m: [] for m in pipeline_modes}
    metadata = []  # conterrà tuple (label, generator)

    print("\nAvvio inferenza con alterazioni in-memory su GPU...")
    with torch.no_grad():
        for idx, row in tqdm(df_selected.iterrows(), total=total_imgs, desc="Processing images"):
            label = int(row["label"])
            generator = row["generator"]
            parquet_idx = int(row["index"])
            
            # Leggiamo l'immagine dal Parquet (sia reali che fake sono memorizzate lì in OpenFake)
            raw_data = parquet_dataset[parquet_idx]["image"]
            img_orig = load_image_safely(raw_data)

            metadata.append((label, generator))

            # Valutiamo ciascun tipo di alterazione sullo stesso file
            for mode in pipeline_modes:
                if mode == "Clean":
                    img_processed = img_orig
                elif mode == "JPEG_70":
                    img_processed = compress_jpeg_in_memory(img_orig, quality=70)
                elif mode == "JPEG_50":
                    img_processed = compress_jpeg_in_memory(img_orig, quality=50)
                elif mode == "Resize_128":
                    img_processed = resize_in_memory(img_orig, size=128)

                img_tensor = clip_transform(img_processed).unsqueeze(0).to(device)
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
                predictions[mode].append(prob)

    # 4. CALCOLO E CONFRONTO METRICHE
    print("\n--- Analisi dei risultati per livello di alterazione ---")
    
    # Identifichiamo gli indici corrispondenti a tutti i reali (imagenet + docci) nel nostro subset
    real_indices = [i for i, (lbl, _) in enumerate(metadata) if lbl == 0]
    
    comparison_rows = []
    for gen in sorted(fake_generators):
        fake_indices = [i for i, (lbl, g) in enumerate(metadata) if lbl == 1 and g == gen]
        paired_indices = fake_indices + real_indices
        
        y_true_paired = [metadata[i][0] for i in paired_indices]
        
        row_entry = {"Generator": gen}
        for mode in pipeline_modes:
            y_scores_paired = [predictions[mode][i] for i in paired_indices]
            auroc = roc_auc_score(y_true_paired, y_scores_paired)
            row_entry[f"AUROC ({mode})"] = auroc
            
        comparison_rows.append(row_entry)

    # Aggiungiamo anche la riga Overall (tutto il subset)
    y_true_all = [m[0] for m in metadata]
    row_overall = {"Generator": "**OVERALL**"}
    for mode in pipeline_modes:
        y_scores_all = predictions[mode]
        auroc_all = roc_auc_score(y_true_all, y_scores_all)
        row_overall[f"AUROC ({mode})"] = auroc_all
    comparison_rows.append(row_overall)

    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison.to_markdown(index=False, floatfmt=".4f"))

    # 5. SALVATAGGIO REPORT
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("# REPORT METRICHE DI ROBUSTEZZA (JPEG + RESIZE) SU OPENFAKE\n\n")
        f_out.write("> **Nota:** Questo file valuta la robustezza di **UFD** simulando le alterazioni in-memory dello stesso subset bilanciato di OpenFake.\n")
        f_out.write("> JPEG_70/50 simula la compressione social, Resize_128 simula il ridimensionamento a bassa risoluzione (128x128).\n\n")
        f_out.write(df_comparison.to_markdown(index=False, floatfmt=".4f"))
        f_out.write("\n")
        
    print(f"\nReport robustezza salvato correttamente in: {output_path}")

if __name__ == "__main__":
    main()
