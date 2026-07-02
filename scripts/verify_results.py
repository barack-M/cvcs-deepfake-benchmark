# -*- coding: utf-8 -*-
"""
Script per il controllo automatico di integrità e coerenza (Sanity Check)
dei file CSV dei risultati di inferenza.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Configurazione dei percorsi dei manifest
MANIFESTS = {
    "gan": "/work/cvcs2026/deep_pixels/datasets/GAN/manifest.csv",
    "d3": "/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv",
    "openfake": "/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv"
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def verify_file(file_path):
    name = file_path.stem
    if "-" not in name:
        return None
    
    parts = name.split("-")
    detector = parts[0].upper()
    dataset_name = parts[1].lower()
    
    if dataset_name not in MANIFESTS:
        print(f"[!] File {file_path.name} ignorato (dataset sconosciuto: {dataset_name})")
        return None
        
    print(f"\n==================================================")
    print(f"VERIFICA: {file_path.name} (Detector: {detector}, Dataset: {dataset_name.upper()})")
    print(f"==================================================")
    
    # 1. Caricamento del manifest
    manifest_path = MANIFESTS[dataset_name]
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest non trovato a: {manifest_path}")
        return False
        
    df_manifest = pd.read_csv(manifest_path)
    len_manifest = len(df_manifest)
    
    # 2. Caricamento dei risultati
    try:
        df_results = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Impossibile leggere il file CSV: {e}")
        return False
        
    len_results = len(df_results)
    
    # 3. Controllo completezza (righe)
    if len_results != len_manifest:
        print(f"[FAIL] ❌ Numero di righe non corrisponde! Risultati: {len_results}, Manifest: {len_manifest}")
        return False
    else:
        print(f"[OK]   ✅ Numero di righe coerente ({len_results})")
        
    # 4. Rilevamento colonna score
    score_col = None
    for col in df_results.columns:
        if "score" in col:
            score_col = col
            break
            
    if not score_col:
        print(f"[ERROR] Nessuna colonna contenente 'score' trouvata!")
        return False
    print(f"[OK]   ✅ Colonna dei punteggi rilevata: {score_col}")
    
    # 5. Verifica allineamento e coerenza riga per riga
    mismatches_gt = (df_results["ground_truth"] != df_manifest["label"]).sum()
    
    # Per il generator, facciamo attenzione a possibili differenze di maiuscole/minuscole o spazi
    mismatches_gen = (df_results["generator"].astype(str).str.strip().str.lower() != 
                      df_manifest["generator"].astype(str).str.strip().str.lower()).sum()
                      
    if mismatches_gt > 0:
        print(f"[FAIL] ❌ Rilevati {mismatches_gt} mismatch sulla colonna ground_truth!")
        return False
    else:
        print(f"[OK]   ✅ Allineamento ground_truth perfetto con il manifest")
        
    if mismatches_gen > 0:
        print(f"[FAIL] ❌ Rilevati {mismatches_gen} mismatch sulla colonna generator!")
        # Stampiamo i primi 5 mismatch per debug
        idx_mismatch = df_results[df_results["generator"].astype(str).str.strip().str.lower() != 
                                  df_manifest["generator"].astype(str).str.strip().str.lower()].index
        print("Primi 5 mismatch generator:")
        for idx in idx_mismatch[:5]:
            print(f"  Riga {idx}: Risultati='{df_results.loc[idx, 'generator']}', Manifest='{df_manifest.loc[idx, 'generator']}'")
        return False
    else:
        print(f"[OK]   ✅ Allineamento generator perfetto con il manifest")
        
    # 6. Verifica valori di score (NaN, range [0, 1])
    scores = df_results[score_col].values
    nan_count = np.isnan(scores).sum()
    inf_count = np.isinf(scores).sum()
    
    if nan_count > 0:
        print(f"[FAIL] ❌ Rilevati {nan_count} valori NaN nella colonna score!")
        return False
    if inf_count > 0:
        print(f"[FAIL] ❌ Rilevati {inf_count} valori Inf nella colonna score!")
        return False
        
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if min_score < 0.0 or max_score > 1.0:
        print(f"[WARNING] ⚠️ Valori score fuori dall'intervallo standard [0, 1]! Min: {min_score:.6f}, Max: {max_score:.6f}")
    else:
        print(f"[OK]   ✅ Valori score nel range [0, 1] (Min: {min_score:.6f}, Max: {max_score:.6f})")
        
    # 7. Analisi statistica disaggregata per classe (Real vs Fake)
    df_real = df_results[df_results["ground_truth"] == 0]
    df_fake = df_results[df_results["ground_truth"] == 1]
    
    scores_real = df_real[score_col].values
    scores_fake = df_fake[score_col].values
    
    print(f"\n--- STATISTICHE DEGLI SCORE ---")
    print(f"Reali (Real - GT=0) [Totale: {len(df_real)}]:")
    print(f"  Media:   {np.mean(scores_real):.4f}")
    print(f"  Mediana: {np.median(scores_real):.4f}")
    print(f"  Std Dev: {np.std(scores_real):.4f}")
    print(f"  Min/Max: {np.min(scores_real):.4f} / {np.max(scores_real):.4f}")
    print(f"  Falsi Positivi (Score >= 0.5): {(scores_real >= 0.5).sum()} ({(scores_real >= 0.5).mean()*100:.2f}%)")
    
    print(f"Fake (Fake - GT=1) [Totale: {len(df_fake)}]:")
    print(f"  Media:   {np.mean(scores_fake):.4f}")
    print(f"  Mediana: {np.median(scores_fake):.4f}")
    print(f"  Std Dev: {np.std(scores_fake):.4f}")
    print(f"  Min/Max: {np.min(scores_fake):.4f} / {np.max(scores_fake):.4f}")
    print(f"  Veri Positivi (Score >= 0.5): {(scores_fake >= 0.5).sum()} ({(scores_fake >= 0.5).mean()*100:.2f}%)")
    
    # 8. Controllo discriminazione
    diff_mean = np.mean(scores_fake) - np.mean(scores_real)
    print(f"Differenza delle medie (Fake - Real): {diff_mean:.4f}")
    if diff_mean <= 0.05:
        print(f"[WARNING] ⚠️ La differenza tra le medie è molto bassa ({diff_mean:.4f}). Il modello potrebbe non discriminare!")
    else:
        print(f"[OK]   ✅ La differenza tra le medie è positiva ({diff_mean:.4f})")
        
    return True

def main():
    if not RESULTS_DIR.exists():
        print(f"La cartella {RESULTS_DIR} non esiste.")
        return
        
    csv_files = sorted(list(RESULTS_DIR.glob("*.csv")))
    
    evaluation_files = []
    for f in csv_files:
        # Consideriamo solo i file nel formato detector-dataset.csv
        # escludendo summary_report.csv, ecc.
        if "-" in f.name and not f.name.startswith("summary_report") and not f.name.startswith("benchmark_speed"):
            evaluation_files.append(f)
            
    if not evaluation_files:
        print("Nessun file di valutazione trovato da verificare.")
        return
        
    print(f"Trovati {len(evaluation_files)} file di valutazione.")
    
    results = {}
    for f in evaluation_files:
        success = verify_file(f)
        results[f.name] = "SUPERATO ✅" if success else "FALLITO ❌"
        
    print("\n" + "=" * 50)
    print("RIEPILOGO STATO SANITY CHECK")
    print("=" * 50)
    for name, status in results.items():
        print(f"{name:<35}: {status}")
    print("=" * 50)

if __name__ == "__main__":
    main()
