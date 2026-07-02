# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Analisi qualitativa dei failure modes: CoDE vs UFD")
    parser.add_argument("--code_csv", type=str, default="/work/cvcs2026/deep_pixels/results/code-openfake.csv",
                        help="Percorso al file CSV dei risultati di CoDE")
    parser.add_argument("--ufd_csv", type=str, default="/work/cvcs2026/deep_pixels/results/ufd-openfake.csv",
                        help="Percorso al file CSV dei risultati di UFD")
    parser.add_argument("--manifest", type=str, default="/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv",
                        help="Percorso al manifest.csv di OpenFake")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Numero di esempi da stampare per ciascuna categoria")
    args = parser.parse_args()

    # Verifica esistenza file
    code_path = Path(args.code_csv)
    ufd_path = Path(args.ufd_csv)
    manifest_path = Path(args.manifest)

    if not code_path.exists() or not ufd_path.exists():
        print("Errore: I file dei risultati per CoDE o UFD non sono presenti.")
        print(f"Verifica presenza di:\n - {code_path}\n - {ufd_path}")
        return
        
    if not manifest_path.exists():
        print(f"Errore: Il manifest di OpenFake non esiste a: {manifest_path}")
        return

    print("Caricamento dei file...")
    df_code = pd.read_csv(code_path)
    df_ufd = pd.read_csv(ufd_path)
    df_manifest = pd.read_csv(manifest_path)

    # Allineiamo i nomi delle colonne dei punteggi
    # CoDE: code_score, UFD: ufd_score
    score_col_code = "code_score"
    score_col_ufd = "ufd_score"

    # Join dei due set di predizioni su sample_id (l'indice di riga del manifest)
    print("Esecuzione della join dei dati...")
    df_merged = pd.merge(df_code, df_ufd, on=["sample_id", "ground_truth", "generator"])

    # Eseguiamo la join con il manifest originale per recuperare il vero indice del parquet
    df_merged = df_merged.rename(columns={"sample_id": "manifest_row_idx"})
    df_final = pd.merge(df_merged, df_manifest, left_on="manifest_row_idx", right_index=True)
    # A questo punto df_final ha le colonne: 
    # [manifest_row_idx, ground_truth, generator, code_score, ufd_score, index (riga parquet), label, dataset]

    print("\n" + "=" * 80)
    print("ANALISI DEI FAILURE MODES PAIRWISE (CoDE vs UFD) SU OPENFAKE")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. CoDE ha ragione, UFD ha torto
    # -------------------------------------------------------------
    print("\n>>> CATEGORIA A: CoDE Corretto (Confidente), UFD Errato (Ingannato)")
    # Per fakes: CoDE > 0.8, UFD < 0.2
    # Per reals: CoDE < 0.2, UFD > 0.8
    cond_a_fake = (df_final["ground_truth"] == 1) & (df_final[score_col_code] > 0.8) & (df_final[score_col_ufd] < 0.2)
    cond_a_real = (df_final["ground_truth"] == 0) & (df_final[score_col_code] < 0.2) & (df_final[score_col_ufd] > 0.8)
    df_a = df_final[cond_a_fake | cond_a_real]
    
    print(f"Trovati {len(df_a)} campioni in questa categoria.")
    if len(df_a) > 0:
        cols_to_show = ["index", "ground_truth", "generator", score_col_code, score_col_ufd]
        print(df_a[cols_to_show].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
    else:
        print("Nessun esempio trovato con soglie stringenti.")

    # -------------------------------------------------------------
    # 2. UFD ha ragione, CoDE ha torto
    # -------------------------------------------------------------
    print("\n>>> CATEGORIA B: UFD Corretto (Confidente), CoDE Errato (Ingannato)")
    # Per fakes: UFD > 0.8, CoDE < 0.2
    # Per reals: UFD < 0.2, CoDE > 0.8
    cond_b_fake = (df_final["ground_truth"] == 1) & (df_final[score_col_ufd] > 0.8) & (df_final[score_col_code] < 0.2)
    cond_b_real = (df_final["ground_truth"] == 0) & (df_final[score_col_ufd] < 0.2) & (df_final[score_col_code] > 0.8)
    df_b = df_final[cond_b_fake | cond_b_real]
    
    print(f"Trovati {len(df_b)} campioni in questa categoria.")
    if len(df_b) > 0:
        cols_to_show = ["index", "ground_truth", "generator", score_col_code, score_col_ufd]
        print(df_b[cols_to_show].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
    else:
        print("Nessun esempio trovato con soglie stringenti.")

    # -------------------------------------------------------------
    # 3. Fallimento Unanime (Entrambi ingannati)
    # -------------------------------------------------------------
    print("\n>>> CATEGORIA C: Fallimento Unanime (Entrambi i detector sbagliano con alta confidenza)")
    # Per fakes: CoDE < 0.2 e UFD < 0.2 (vedono fake come reali)
    # Per reals: CoDE > 0.8 e UFD > 0.8 (vedono reali come fake)
    cond_c_fake = (df_final["ground_truth"] == 1) & (df_final[score_col_code] < 0.2) & (df_final[score_col_ufd] < 0.2)
    cond_c_real = (df_final["ground_truth"] == 0) & (df_final[score_col_code] > 0.8) & (df_final[score_col_ufd] > 0.8)
    df_c = df_final[cond_c_fake | cond_c_real]
    
    print(f"Trovati {len(df_c)} campioni in questa categoria.")
    if len(df_c) > 0:
        cols_to_show = ["index", "ground_truth", "generator", score_col_code, score_col_ufd]
        print(df_c[cols_to_show].head(args.num_examples).to_markdown(index=False, floatfmt=".4f"))
    else:
        print("Nessun esempio trovato con soglie stringenti.")

    print("\n" + "=" * 80)
    print("Suggerimento: Usa l'indice del Parquet ('index') sopra riportato per estrarre e visualizzare")
    print("l'immagine corrispondente all'interno del file Parquet di Openfake per il vostro PDF.")
    print("=" * 80)

if __name__ == "__main__":
    main()
