# -*- coding: utf-8 -*-
"""
Script per l'analisi qualitativa multi-detector dei failure modes.
Combina i risultati di tutti i detector su un dataset, identifica campioni
discordanti d'interesse e salva una lista di candidati per l'estrazione delle immagini.
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Configurazione dei manifest
MANIFESTS = {
    "gan": "/work/cvcs2026/deep_pixels/datasets/GAN/manifest.csv",
    "d3": "/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv",
    "openfake": "/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv"
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def main():
    parser = argparse.ArgumentParser(description="Analisi qualitativa dei failure modes su tutti i detector")
    parser.add_argument("--dataset", type=str, default="openfake", choices=["gan", "d3", "openfake"],
                        help="Dataset su cui effettuare l'analisi dei failure modes")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Numero massimo di esempi da stampare a terminale per categoria")
    args = parser.parse_args()

    # 1. Caricamento del manifest
    manifest_path = Path(MANIFESTS[args.dataset])
    if not manifest_path.exists():
        print(f"Errore: Manifest del dataset non trovato in {manifest_path}")
        return
    df_manifest = pd.read_csv(manifest_path)
    
    # 2. Ricerca e caricamento dei CSV dei detector per questo dataset
    csv_files = list(RESULTS_DIR.glob(f"*-{args.dataset}.csv"))
    
    # Escludiamo eventuali file temporanei o report compilati
    evaluation_files = [f for f in csv_files if not f.name.startswith("summary_report")]
    
    if not evaluation_files:
        print(f"Nessun file di predizioni trovato per il dataset {args.dataset.upper()} in {RESULTS_DIR}")
        return
        
    print(f"Trovati {len(evaluation_files)} file di predizione per {args.dataset.upper()}:")
    for f in evaluation_files:
        print(f"  - {f.name}")

    # 3. Join progressiva di tutti i detector su sample_id
    df_merged = None
    
    for file_path in sorted(evaluation_files):
        df_det = pd.read_csv(file_path)
        
        # Rileva colonna score
        score_col = None
        for col in df_det.columns:
            if "score" in col:
                score_col = col
                break
                
        if not score_col:
            continue
            
        # Rinominiamo la colonna score per identificare il detector
        df_det = df_det[["sample_id", "ground_truth", "generator", score_col]]
        
        if df_merged is None:
            df_merged = df_det
        else:
            # Join su sample_id, ground_truth, generator
            df_merged = pd.merge(df_merged, df_det, on=["sample_id", "ground_truth", "generator"], how="inner")

    if df_merged is None:
        print("Errore nella fusione dei dati.")
        return
        
    print(f"Fusione completata. Righe totali allineate: {len(df_merged)}")

    # Allineiamo gli indici con il manifest per recuperare le coordinate dei file (path o parquet index)
    df_merged = df_merged.rename(columns={"sample_id": "manifest_row_idx"})
    df_final = pd.merge(df_merged, df_manifest, left_on="manifest_row_idx", right_index=True, suffixes=("", "_manifest"))
    
    # Rileviamo le colonne score disponibili
    score_cols = [col for col in df_final.columns if "score" in col]
    print(f"Colonne score caricate: {score_cols}")

    # === CATEGORIA 1: CoDE corretto ed UFD errato (se entrambi disponibili) ===
    # CoDE è addestrato su diffusion, UFD su ProGAN (CLIP visual space).
    # Cerchiamo diffusion fake (Flux, SDXL, ecc.) viste correttamente da CoDE ma non da UFD.
    if "code_score" in df_final.columns and "ufd_score" in df_final.columns:
        print("\n" + "=" * 80)
        print("CATEGORIA 1: CoDE Corretto (score > 0.8), UFD Errato (score < 0.2) su FAKE")
        print("=" * 80)
        cond_fake = (df_final["ground_truth"] == 1) & (df_final["code_score"] > 0.8) & (df_final["ufd_score"] < 0.2)
        df_c1 = df_final[cond_fake]
        print(f"Trovati {len(df_c1)} campioni.")
        if len(df_c1) > 0:
            show_cols = ["manifest_row_idx", "generator", "code_score", "ufd_score"]
            if "index" in df_final.columns: show_cols.append("index")
            print(df_c1[show_cols].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
            df_c1.to_csv(RESULTS_DIR / f"failure_c1_{args.dataset}.csv", index=False)
            
    # === CATEGORIA 2: AIDE-GenImage corretto ed AIDE-ProGAN errato (se entrambi disponibili) ===
    # Mostra l'impatto dell'aggiornamento dei dati di training (GenImage multi-generatore vs ProGAN antico)
    # a parità di architettura multi-esperto spettrale.
    if "aide_genimage_score" in df_final.columns and "aide_progan_score" in df_final.columns:
        print("\n" + "=" * 80)
        print("CATEGORIA 2: AIDE-GenImage Corretto (score > 0.8), AIDE-ProGAN Errato (score < 0.2) su FAKE")
        print("=" * 80)
        cond_fake = (df_final["ground_truth"] == 1) & (df_final["aide_genimage_score"] > 0.8) & (df_final["aide_progan_score"] < 0.2)
        df_c2 = df_final[cond_fake]
        print(f"Trovati {len(df_c2)} campioni.")
        if len(df_c2) > 0:
            show_cols = ["manifest_row_idx", "generator", "aide_genimage_score", "aide_progan_score"]
            if "index" in df_final.columns: show_cols.append("index")
            print(df_c2[show_cols].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
            df_c2.to_csv(RESULTS_DIR / f"failure_c2_{args.dataset}.csv", index=False)

    # === CATEGORIA 3: Fallimento unanime (TUTTI i detector sbagliano) ===
    # Fake visti come reali da tutti (score < 0.15 per tutti i detector)
    # o Reali visti come fake da tutti (score > 0.85 per tutti i detector)
    print("\n" + "=" * 80)
    print("CATEGORIA 3: Fallimento Unanime (Tutti i detector sbagliano con alta confidenza)")
    print("=" * 80)
    
    # Costruiamo la condizione di fallimento su tutti i detector caricati
    cond_all_fake = (df_final["ground_truth"] == 1)
    cond_all_real = (df_final["ground_truth"] == 0)
    
    for col in score_cols:
        cond_all_fake &= (df_final[col] < 0.20)
        cond_all_real &= (df_final[col] > 0.80)
        
    df_c3_fake = df_final[cond_all_fake]
    df_c3_real = df_final[cond_all_real]
    df_c3 = pd.concat([df_c3_fake, df_c3_real])
    
    print(f"Trovati {len(df_c3)} campioni (Fake non rilevati: {len(df_c3_fake)}, Reali visti come Fake: {len(df_c3_real)}).")
    if len(df_c3) > 0:
        show_cols = ["manifest_row_idx", "ground_truth", "generator"] + score_cols
        if "index" in df_final.columns: show_cols.append("index")
        print(df_c3[show_cols].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
        df_c3.to_csv(RESULTS_DIR / f"failure_c3_{args.dataset}.csv", index=False)

    # Salva il file globale di predizioni allineate
    merged_output = RESULTS_DIR / f"merged_predictions_{args.dataset}.csv"
    df_final.to_csv(merged_output, index=False)
    print(f"\nTabella globale delle predizioni allineate salvata in: {merged_output}")
    
    # Salva una lista unificata di candidati
    candidates = []
    if 'df_c1' in locals() and len(df_c1) > 0:
        df_c1_sub = df_c1.head(5).copy()
        df_c1_sub["category"] = "code_right_ufd_wrong"
        candidates.append(df_c1_sub)
    if 'df_c2' in locals() and len(df_c2) > 0:
        df_c2_sub = df_c2.head(5).copy()
        df_c2_sub["category"] = "aide_genimage_right_progan_wrong"
        candidates.append(df_c2_sub)
    if len(df_c3) > 0:
        df_c3_sub = df_c3.head(10).copy()
        df_c3_sub["category"] = "unanimous_failure"
        candidates.append(df_c3_sub)
        
    if candidates:
        df_candidates = pd.concat(candidates, ignore_index=True)
        candidates_path = RESULTS_DIR / f"failure_candidates_{args.dataset}.csv"
        df_candidates.to_csv(candidates_path, index=False)
        print(f"Lista candidati all'estrazione salvata in: {candidates_path}")

if __name__ == "__main__":
    main()
