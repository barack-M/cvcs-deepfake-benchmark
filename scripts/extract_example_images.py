# -*- coding: utf-8 -*-
"""
Script per estrarre fisicamente le immagini PNG corrispondenti ai candidati
dei failure modes e strutturarle in una directory per il download e la scrittura del report.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import UnifiedDeepfakeDataset

# Configurazione dei manifest e directory
MANIFESTS = {
    "d3": "/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv",
    "openfake": "/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv"
}

PARQUET_DIRS = {
    "d3": "/work/cvcs2026/deep_pixels/datasets/D3/data",
    "openfake": "/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/"
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def get_failure_description(category, row, score_cols):
    """
    Ritorna una descrizione dettagliata del perché il campione è un fallimento
    e i punteggi dei vari detector.
    """
    scores_str = ", ".join([f"{col.replace('_score', '').upper()}={row[col]:.3f}" for col in score_cols if col in row])
    
    label_str = "FAKE" if row["ground_truth"] == 1 else "REALE"
    gen_str = row["generator"]
    
    if category == "code_right_ufd_wrong":
        desc = (f"Immagine {label_str} generata da {gen_str.upper()} identificata CORRETTAMENTE da CoDE "
                f"ma che INGANNATE UFD (CLIP visual). Probabile assenza di artefatti spettrali storici, "
                f"rilevata tramite pattern di diffusione locali da CoDE.")
    elif category == "aide_genimage_right_progan_wrong":
        desc = (f"Immagine {label_str} generata da {gen_str.upper()} identificata CORRETTAMENTE da AIDE-GenImage "
                f"ma che INGANNA AIDE-ProGAN. Evidenzia il guadagno di generalizzazione cross-generator "
                f"ottenuto addestrando su dati multi-generatore moderni (GenImage) rispetto a ProGAN antico.")
    elif category == "unanimous_failure":
        if row["ground_truth"] == 1:
            desc = (f"Immagine FAKE generata da {gen_str.upper()} che INGANNA TUTTI i detector (unanimous failure). "
                    f"Rappresenta una generazione fotorealistica ottimale sia a livello semantico che di frequenza.")
        else:
            desc = (f"Immagine REALE ({gen_str.upper()}) che TUTTI i detector classificano erroneamente come FAKE. "
                    f"Presenta anomalie visive, texture insolite o compressioni che simulano artefatti da generazione.")
    else:
        desc = f"Campione {label_str} da {gen_str.upper()} ({category})."
        
    return f"{desc} [Punteggi: {scores_str}]"

def main():
    parser = argparse.ArgumentParser(description="Estrae le immagini di esempio per i failure modes")
    parser.add_argument("--dataset", type=str, default="openfake", choices=["d3", "openfake"],
                        help="Dataset da cui estrarre (d3, openfake)")
    args = parser.parse_args()

    candidates_file = RESULTS_DIR / f"failure_candidates_{args.dataset}.csv"
    if not candidates_file.exists():
        print(f"Errore: File dei candidati non trovato in {candidates_file}")
        print("Esegui prima: python scripts/find_failure_modes.py --dataset " + args.dataset)
        return
        
    df_candidates = pd.read_csv(candidates_file)
    print(f"Caricati {len(df_candidates)} candidati all'estrazione per {args.dataset.upper()}.")

    # Inizializzazione del dataset per estrarre le immagini PIL grezze (transform=None)
    manifest_path = MANIFESTS[args.dataset]
    openfake_dir = PARQUET_DIRS["openfake"] if args.dataset == "openfake" else None
    d3_dir = PARQUET_DIRS["d3"] if args.dataset == "d3" else None
    
    dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        openfake_parquet_dir=openfake_dir,
        d3_parquet_dir=d3_dir,
        transform=None
    )

    output_root = RESULTS_DIR / "failure_examples" / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    
    score_cols = [col for col in df_candidates.columns if "score" in col]

    extracted_rows = []
    
    for idx, row in df_candidates.iterrows():
        manifest_idx = int(row["manifest_row_idx"])
        category = row["category"]
        generator = row["generator"]
        label = int(row["ground_truth"])
        
        category_dir = output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome file immagine di output
        img_name = f"img_{manifest_idx}_{generator}_{'fake' if label==1 else 'real'}.png"
        img_path = category_dir / img_name
        
        print(f"Estrazione [{category}] -> {img_name}...")
        
        try:
            # Estraiamo l'immagine dal dataset
            img_pil, _, _, _ = dataset[manifest_idx]
            
            # Salvataggio PNG
            img_pil.save(img_path, format="PNG")
            
            # Generazione descrizione
            description = get_failure_description(category, row, score_cols)
            
            row_data = {
                "filename": f"failure_examples/{args.dataset}/{category}/{img_name}",
                "manifest_row_idx": manifest_idx,
                "category": category,
                "ground_truth": label,
                "generator": generator,
                "description": description
            }
            for col in score_cols:
                row_data[col] = row[col]
                
            extracted_rows.append(row_data)
            
        except Exception as e:
            print(f"  ❌ Errore durante l'estrazione della riga {manifest_idx}: {e}")

    # Scrittura del file index.csv all'interno della cartella failure_examples
    if extracted_rows:
        df_index = pd.DataFrame(extracted_rows)
        index_path = output_root / "index.csv"
        df_index.to_csv(index_path, index=False)
        print("\n" + "=" * 70)
        print(f"ESTRAZIONE COMPLETATA PER {args.dataset.upper()}!")
        print("=" * 70)
        print(f"Immagini salvate in:       {output_root}/")
        print(f"Tabella index di riepilogo: {index_path}")
        print("=" * 70)
        
        # Stampiamo l'indice
        for i, r in df_index.iterrows():
            print(f"\n[{i+1}] File: {r['filename']}")
            print(f"    Descrizione: {r['description']}")
        print("=" * 70)

if __name__ == "__main__":
    main()
