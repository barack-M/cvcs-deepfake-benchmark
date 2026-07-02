# -*- coding: utf-8 -*-
import time
import torch
from torch.utils.data import DataLoader
from src.data.dataset import UnifiedDeepfakeDataset
from scripts.evaluate_aide import preprocess_aide

def main():
    print("Inizializzazione dataset OpenFake per profiling...")
    dataset = UnifiedDeepfakeDataset(
        manifest_path="/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv",
        openfake_parquet_dir="/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/",
        d3_parquet_dir=None,
        transform=None
    )
    
    print("\n1. Profiling caricamento immagini grezze (I/O Parquet)...")
    t0 = time.time()
    for i in range(32):
        img, label, gen, ds = dataset[i]
    t1 = time.time()
    print(f"Tempo caricamento 32 immagini grezze: {t1 - t0:.4f} s ({(t1-t0)/32:.4f} s/img)")
    
    print("\n2. Profiling preprocessing AIDE (DCT + Normalize)...")
    t0 = time.time()
    for i in range(32):
        img, _, _, _ = dataset[i]
        _ = preprocess_aide(img)
    t1 = time.time()
    print(f"Tempo preprocessing di 32 immagini: {t1 - t0:.4f} s ({(t1-t0)/32:.4f} s/img)")
    
    print("\n3. Profiling forward pass AIDE su GPU...")
    from models.AIDE import AIDE as build_aide_model
    model = build_aide_model(resnet_path=None, convnext_path="laion2b_s34b_b82k_augreg_soup")
    checkpoint = torch.load("/work/cvcs2026/deep_pixels/weights/aide/aide_progan.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepariamo un batch fittizio
    batch_tensor = torch.randn(32, 5, 3, 256, 256).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(batch_tensor)
        
    t0 = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch_tensor)
    t1 = time.time()
    print(f"Tempo forward pass di 5 batch da 32 immagini: {t1 - t0:.4f} s ({(t1-t0)/5:.4f} s/batch)")

if __name__ == "__main__":
    main()
