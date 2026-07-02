# -*- coding: utf-8 -*-
"""
Script unificato per la valutazione della robustezza dei detector (UFD, CNNDet, AIDE)
sotto perturbazioni (JPEG compression, Resize) su un subset bilanciato di OpenFake.
"""
import os
import sys
import io
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# Aggiungiamo i percorsi delle repository esterne al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "UniversalFakeDetect"))
sys.path.insert(0, str(PROJECT_ROOT / "CNNDetection"))
sys.path.insert(0, str(PROJECT_ROOT / "AIDE"))

from src.data.dataset import UnifiedDeepfakeDataset

# ==========================================
# 1. TRASFORMAZIONI E PREPROCESSING DETECTOR
# ==========================================

# A. Trasformazioni CLIP per UFD
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

clip_transform = Compose([
    Resize(224, interpolation=BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# B. Trasformazioni per CNNDetection
cnndet_transform = Compose([
    Resize(256, interpolation=BILINEAR),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# C. Preprocessing per AIDE (richiede stack a 5 view)
from data.dct import DCT_base_Rec_Module
dct_module = DCT_base_Rec_Module()

transform_normalize_aide = Compose([
    Resize([256, 256]),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_aide(img_pil):
    img_tensor = ToTensor()(img_pil)  # [C, H, W]
    try:
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_tensor)
    except Exception:
        x_minmin = x_maxmax = x_minmin1 = x_maxmax1 = img_tensor

    x_0 = transform_normalize_aide(img_tensor)
    x_minmin = transform_normalize_aide(x_minmin)
    x_maxmax = transform_normalize_aide(x_maxmax)
    x_minmin1 = transform_normalize_aide(x_minmin1)
    x_maxmax1 = transform_normalize_aide(x_maxmax1)

    return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)

# ==========================================
# 2. FUNZIONI PER PERTURBAZIONI IN-MEMORY
# ==========================================

def compress_jpeg_in_memory(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    # Carichiamo immediatamente per evitare lazy evaluation issues
    img_jpg = Image.open(buffer)
    img_jpg.load()
    return img_jpg

def resize_in_memory(img, size=128):
    # Ridimensioniamo a bassa risoluzione (bilineare) e poi torniamo alla dimensione originale
    # per forzare la perdita di dettagli ad alta frequenza
    w, h = img.size
    img_down = img.resize((size, size), Image.BILINEAR)
    img_up = img_down.resize((w, h), Image.BILINEAR)
    return img_up

class RobustnessDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        try:
            img_pil, label, generator, _ = self.base_dataset[global_idx]
            
            # Applichiamo perturbazioni in-memory
            img_clean = img_pil
            img_j70 = compress_jpeg_in_memory(img_pil, 70)
            img_j50 = compress_jpeg_in_memory(img_pil, 50)
            img_r128 = resize_in_memory(img_pil, 128)
            
            # Trasformiamo
            t_clean = self.transform(img_clean)
            t_j70 = self.transform(img_j70)
            t_j50 = self.transform(img_j50)
            t_r128 = self.transform(img_r128)
            
            return t_clean, t_j70, t_j50, t_r128, label, generator
        except Exception as e:
            # Fallback in caso di errore di caricamento
            # Ritorniamo tensori dummy di dimensione compatibile
            # AIDE usa 5 viste da [3, 256, 256], gli altri usano [3, 224, 224]
            if hasattr(self.transform, "transforms"): # CNNDet/UFD
                dummy = torch.zeros(3, 224, 224)
            else:
                dummy = torch.zeros(5, 3, 256, 256)
            return dummy, dummy, dummy, dummy, 0, "unknown"

# ==========================================
# 3. LOGICA DI CARICAMENTO MODELLI
# ==========================================

def load_detector(detector_name, device):
    if detector_name == "ufd":
        from models import get_model
        model = get_model("CLIP:ViT-L/14")
        weights_path = "/work/cvcs2026/deep_pixels/weights/ufd/fc_weights.pth"
        print(f"Caricamento pesi UFD da: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.fc.load_state_dict(state_dict)
        transform = clip_transform
        
    elif detector_name == "cnndet":
        from networks.resnet import resnet50
        model = resnet50(num_classes=1)
        weights_path = "/work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth"
        print(f"Caricamento pesi CNNDet da: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
        transform = cnndet_transform
        
    elif detector_name in ["aide_progan", "aide_genimage"]:
        from models.AIDE import AIDE as build_aide_model
        model = build_aide_model(resnet_path=None, convnext_path="laion2b_s34b_b82k_augreg_soup")
        chk_name = "aide_progan.pth" if detector_name == "aide_progan" else "aide_genimage.pth"
        weights_path = f"/work/cvcs2026/deep_pixels/weights/aide/{chk_name}"
        print(f"Caricamento pesi AIDE ({detector_name}) da: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        transform = preprocess_aide
        
    else:
        raise ValueError(f"Detector non supportato: {detector_name}")
        
    model = model.to(device)
    model.eval()
    return model, transform

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Valutazione robustezza detector sotto alterazioni (JPEG + Resize)")
    parser.add_argument("--detector", type=str, required=True, choices=["ufd", "cnndet", "aide_progan", "aide_genimage"],
                        help="Rilevatore da valutare")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv",
                        help="Manifest.csv di OpenFake")
    parser.add_argument("--parquet_dir", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/",
                        help="Cartella dei file Parquet di OpenFake")
    parser.add_argument("--subset_size", type=int, default=200,
                        help="Numero di immagini da usare per ciascun generatore/reale (default 200 per classe)")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "results"),
                        help="Directory di output per report e CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # Caricamento manifest
    df_manifest = pd.read_csv(args.manifest)

    # Carichiamo il modello del detector
    model, detector_transform = load_detector(args.detector, device)

    # Selezioniamo il subset bilanciato in modo identico usando un seed fisso (random_state=42)
    # Reali: imagenet e docci (200 campioni ciascuno -> total 400 reali)
    df_reals_imagenet = df_manifest[(df_manifest["label"] == 0) & (df_manifest["generator"] == "imagenet")].sample(n=args.subset_size, random_state=42)
    df_reals_docci = df_manifest[(df_manifest["label"] == 0) & (df_manifest["generator"] == "docci")].sample(n=args.subset_size, random_state=42)
    df_reals = pd.concat([df_reals_imagenet, df_reals_docci])
    
    # Fake generators: 200 campioni ciascuno
    fake_generators = sorted([g for g in df_manifest["generator"].unique() if g not in ["imagenet", "docci"]])
    df_fakes_list = []
    for gen in fake_generators:
        df_gen = df_manifest[(df_manifest["generator"] == gen) & (df_manifest["label"] == 1)]
        df_fakes_list.append(df_gen.sample(n=min(len(df_gen), args.subset_size), random_state=42))
        
    df_selected = pd.concat([df_reals] + df_fakes_list)
    total_imgs = len(df_selected)
    print(f"Totale immagini selezionate per il test di robustezza: {total_imgs}")

    # Inizializziamo il dataset con transform=None per recuperare le immagini PIL grezze
    dataset = UnifiedDeepfakeDataset(
        manifest_path=args.manifest,
        openfake_parquet_dir=args.parquet_dir,
        d3_parquet_dir=None,
        transform=None
    )

    from torch.utils.data import DataLoader
    
    robust_dataset = RobustnessDataset(dataset, df_selected.index.tolist(), detector_transform)
    batch_size = 16 if args.detector not in ["ufd", "cnndet"] else 64
    
    dataloader = DataLoader(
        robust_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    pipeline_modes = ["Clean", "JPEG_70", "JPEG_50", "Resize_128"]
    predictions = {m: [] for m in pipeline_modes}
    metadata = []  # conterrà tuple (label, generator)

    print(f"\nAvvio inferenza parallela con alterazioni in-memory (Batch Size: {batch_size})...")
    with torch.no_grad():
        for t_clean, t_j70, t_j50, t_r128, labels_batch, gens_batch in tqdm(dataloader, desc="Evaluating robustness"):
            t_clean = t_clean.to(device)
            t_j70 = t_j70.to(device)
            t_j50 = t_j50.to(device)
            t_r128 = t_r128.to(device)
            
            # Forward pass per ciascuna versione alterata del batch
            out_clean = model(t_clean)
            out_j70 = model(t_j70)
            out_j50 = model(t_j50)
            out_r128 = model(t_r128)
            
            # Calcolo probabilità in PyTorch su GPU
            if args.detector in ["ufd", "cnndet"]:
                p_clean = torch.sigmoid(out_clean).cpu().numpy().flatten()
                p_j70 = torch.sigmoid(out_j70).cpu().numpy().flatten()
                p_j50 = torch.sigmoid(out_j50).cpu().numpy().flatten()
                p_r128 = torch.sigmoid(out_r128).cpu().numpy().flatten()
            else:
                p_clean = torch.softmax(out_clean, dim=1)[:, 1].cpu().numpy().flatten()
                p_j70 = torch.softmax(out_j70, dim=1)[:, 1].cpu().numpy().flatten()
                p_j50 = torch.softmax(out_j50, dim=1)[:, 1].cpu().numpy().flatten()
                p_r128 = torch.softmax(out_r128, dim=1)[:, 1].cpu().numpy().flatten()
                
            predictions["Clean"].extend(p_clean.tolist())
            predictions["JPEG_70"].extend(p_j70.tolist())
            predictions["JPEG_50"].extend(p_j50.tolist())
            predictions["Resize_128"].extend(p_r128.tolist())
            
            for l, g in zip(labels_batch.tolist(), gens_batch):
                metadata.append((l, g))

    # === Calcolo Metriche disaggregate ===
    print("\n--- Calcolo metriche per livello di alterazione ---")
    real_indices = [i for i, (lbl, _) in enumerate(metadata) if lbl == 0]
    
    comparison_rows = []
    for gen in fake_generators:
        fake_indices = [i for i, (lbl, g) in enumerate(metadata) if lbl == 1 and g == gen]
        paired_indices = fake_indices + real_indices
        
        y_true_paired = [metadata[i][0] for i in paired_indices]
        
        row_entry = {
            "Detector": args.detector.upper(),
            "Generator": gen
        }
        for mode in pipeline_modes:
            y_scores_paired = [predictions[mode][i] for i in paired_indices]
            auroc = roc_auc_score(y_true_paired, y_scores_paired)
            row_entry[f"AUROC_{mode}"] = auroc
            
        comparison_rows.append(row_entry)

    # Riga complessiva (Overall)
    y_true_all = [m[0] for m in metadata]
    row_overall = {
        "Detector": args.detector.upper(),
        "Generator": "OVERALL"
    }
    for mode in pipeline_modes:
        y_scores_all = predictions[mode]
        auroc_all = roc_auc_score(y_true_all, y_scores_all)
        row_overall[f"AUROC_{mode}"] = auroc_all
    comparison_rows.append(row_overall)

    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison.to_markdown(index=False, floatfmt=".4f"))

    # === Salvataggio Risultati ===
    out_path_dir = Path(args.output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Salvataggio report Markdown specifico del detector
    report_path = out_path_dir / f"robustness_report_{args.detector}.md"
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# REPORT ROBUSTEZZA DI {args.detector.upper()} SU OPENFAKE\n\n")
        f_out.write(f"> **Nota:** Questo report valuta la stabilità del rilevatore **{args.detector.upper()}** sotto perturbazioni a runtime.\n")
        f_out.write(f"> Valutazione eseguita su un subset di {total_imgs} immagini bilanciate (seed=42).\n\n")
        f_out.write(df_comparison.to_markdown(index=False, floatfmt=".4f"))
        f_out.write("\n")
    print(f"\nReport salvato in: {report_path}")

    # 2. Append/Salvataggio su file CSV cumulativo di robustezza
    csv_path = out_path_dir / "robustness_results.csv"
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # Rimuoviamo righe vecchie per lo stesso detector per evitare duplicati
        df_old = df_old[df_old["Detector"] != args.detector.upper()]
        df_cumulative = pd.concat([df_old, df_comparison], ignore_index=True)
    else:
        df_cumulative = df_comparison
        
    df_cumulative.to_csv(csv_path, index=False)
    print(f"Risultati cumulativi di robustezza aggiornati in: {csv_path}")

if __name__ == "__main__":
    main()
