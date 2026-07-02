# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def calculate_eer(y_true, y_scores):
    """
    Calcola l'Equal Error Rate (EER) e la soglia ottimale.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    optimal_threshold = thresholds[idx]
    return eer, optimal_threshold

def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if not results_dir.exists():
        print(f"Errore: La cartella dei risultati {results_dir} non esiste.")
        return

    csv_files = list(results_dir.glob("*.csv"))
    
    # Filtriamo i file di riepilogo generati per non includerli nell'analisi
    evaluation_files = []
    for f in csv_files:
        # Consideriamo solo i file nel formato "detector-dataset.csv" 
        # (esclude i file di benchmark_speed o i vecchi file di test)
        if "-" in f.name and not f.name.startswith("summary_report"):
            evaluation_files.append(f)

    if not evaluation_files:
        print(f"Nessun file di valutazione nel formato 'detector-dataset.csv' trovato in {results_dir}.")
        print("Suggerimento: Assicurati di rinominare eventuali file estranei per seguire questo pattern (es. code-openfake.csv).")
        return

    print("=" * 70)
    print(f"AVVIO AGGREGAZIONE RISULTATI (Trovati {len(evaluation_files)} file di predizione)")
    print("=" * 70)

    summary_rows = []
    all_breakdown_rows = []
    breakdown_tables = {}  # Mappa (dataset, detector) -> markdown string

    for file_path in sorted(evaluation_files):
        # Il nome del file è detector-dataset.csv (es. ufd-gan.csv -> detector='ufd', dataset='gan')
        name_without_ext = file_path.stem
        parts = name_without_ext.split("-")
        
        detector = parts[0].upper()
        dataset = parts[1].upper()

        print(f"Elaborazione: Detector={detector} | Dataset={dataset} ({file_path.name})...")
        
        try:
            df = pd.read_csv(file_path)
            
            # Rileva colonna score
            score_col = None
            for col in df.columns:
                if "score" in col:
                    score_col = col
                    break
                    
            if not score_col:
                print(f"  -> Salto: Nessuna colonna score trovata in {file_path.name}")
                continue
                
            # Rilevamento e rimozione di valori NaN/Inf per evitare crash di scikit-learn
            if df["ground_truth"].isnull().any() or df[score_col].isnull().any():
                print(f"  -> [WARNING] Rilevati valori NaN in {file_path.name}. Rimozione delle righe incomplete...")
                df = df.dropna(subset=["ground_truth", score_col])
                
            y_true = df["ground_truth"].values
            y_scores = df[score_col].values
            
            if len(np.unique(y_true)) < 2:
                print(f"  -> [WARNING] Meno di 2 classi presenti in {file_path.name} dopo la pulizia. Salto.")
                continue

            # 1. Calcolo metriche globali
            auroc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            acc_05 = accuracy_score(y_true, y_scores >= 0.5)
            eer, opt_th = calculate_eer(y_true, y_scores)
            acc_opt = accuracy_score(y_true, y_scores >= opt_th)

            summary_rows.append({
                "Detector": detector,
                "Dataset": dataset,
                "AUROC": auroc,
                "AP": ap,
                "Acc (th=0.5)": acc_05,
                "EER": eer,
                "Opt Threshold": opt_th,
                "Acc (Opt th)": acc_opt
            })
            
            # 2. Calcolo breakdown per generatore
            real_gens = set(df[df["ground_truth"] == 0]["generator"].unique())
            fake_gens = set(df[df["ground_truth"] == 1]["generator"].unique())
            is_shared_real = len(real_gens.intersection(fake_gens)) == 0
            
            breakdown_data = []
            
            if is_shared_real:
                df_reals = df[df["ground_truth"] == 0]
                for gen in sorted(fake_gens):
                    df_fakes = df[(df["generator"] == gen) & (df["ground_truth"] == 1)]
                    df_gen = pd.concat([df_fakes, df_reals])
                    
                    y_true_gen = df_gen["ground_truth"].values
                    y_scores_gen = df_gen[score_col].values
                    
                    b_auroc = roc_auc_score(y_true_gen, y_scores_gen)
                    b_ap = average_precision_score(y_true_gen, y_scores_gen)
                    b_eer, b_opt_t = calculate_eer(y_true_gen, y_scores_gen)
                    b_acc_05 = accuracy_score(y_true_gen, y_scores_gen >= 0.5)
                    
                    breakdown_data.append({
                        "Generator": gen,
                        "Samples (F+R)": f"{len(df_fakes)}+{len(df_reals)}",
                        "AUROC": b_auroc,
                        "AP": b_ap,
                        "Acc (th=0.5)": b_acc_05,
                        "EER": b_eer
                    })
                    all_breakdown_rows.append({
                        "Detector": detector,
                        "Dataset": dataset,
                        "Generator": gen,
                        "Samples": f"{len(df_fakes)}+{len(df_reals)}",
                        "AUROC": b_auroc,
                        "AP": b_ap,
                        "Acc (th=0.5)": b_acc_05,
                        "EER": b_eer
                    })
            else:
                generators = df["generator"].unique()
                for gen in sorted(generators):
                    df_gen = df[df["generator"] == gen]
                    y_true_gen = df_gen["ground_truth"].values
                    y_scores_gen = df_gen[score_col].values
                    
                    if len(np.unique(y_true_gen)) < 2:
                        b_auroc, b_ap, b_eer = np.nan, np.nan, np.nan
                    else:
                        b_auroc = roc_auc_score(y_true_gen, y_scores_gen)
                        b_ap = average_precision_score(y_true_gen, y_scores_gen)
                        b_eer, _ = calculate_eer(y_true_gen, y_scores_gen)
                        
                    b_acc_05 = accuracy_score(y_true_gen, y_scores_gen >= 0.5)
                    
                    breakdown_data.append({
                        "Generator": gen,
                        "Samples (F+R)": f"{len(df_gen)}",
                        "AUROC": b_auroc,
                        "AP": b_ap,
                        "Acc (th=0.5)": b_acc_05,
                        "EER": b_eer
                    })
                    all_breakdown_rows.append({
                        "Detector": detector,
                        "Dataset": dataset,
                        "Generator": gen,
                        "Samples": f"{len(df_gen)}",
                        "AUROC": b_auroc,
                        "AP": b_ap,
                        "Acc (th=0.5)": b_acc_05,
                        "EER": b_eer
                    })
                    
            df_breakdown = pd.DataFrame(breakdown_data)
            breakdown_tables[(dataset, detector)] = df_breakdown.to_markdown(index=False, floatfmt=".4f")
            
        except Exception as e:
            print(f"  -> Errore nell'elaborazione di {file_path.name}: {e}")

    if not summary_rows:
        print("Nessun dato di riepilogo generato.")
        return

    # Creazione DataFrame complessivo ordinato per Detector e poi per Dataset
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by=["Detector", "Dataset"])

    # Output file
    md_output_path = results_dir / "summary_report.md"
    csv_output_path = results_dir / "summary_report.csv"

    # Genera report Markdown completo
    with open(md_output_path, "w", encoding="utf-8") as f_out:
        f_out.write("# RIEPILOGO GENERALE METRICHE BENCHMARK\n\n")
        f_out.write("> **Nota:** Questo file è autogenerato dallo script `aggregate_results.py`.\n")
        f_out.write("> Raccoglie e calcola le metriche per tutti i file di inferenza in `/work`.\n\n")
        
        f_out.write("## 1. Metriche Globali (Sintesi)\n\n")
        f_out.write(df_summary.to_markdown(index=False, floatfmt=".4f"))
        f_out.write("\n\n")
        
        f_out.write("## 2. Breakdown per Generatore\n\n")
        for (dataset, detector), md_table in sorted(breakdown_tables.items()):
            f_out.write(f"### Detector: **{detector}** | Dataset: **{dataset}**\n\n")
            f_out.write(md_table)
            f_out.write("\n\n")

    # Salva CSV
    df_summary.to_csv(csv_output_path, index=False)

    # Salva CSV di breakdown completo
    if all_breakdown_rows:
        df_all_breakdowns = pd.DataFrame(all_breakdown_rows)
        breakdown_output_path = results_dir / "breakdown_per_generator.csv"
        df_all_breakdowns.to_csv(breakdown_output_path, index=False)
        print(f"Tabella breakdown per generatore salvata in: {breakdown_output_path}")

    print("\n" + "=" * 70)
    print("RIEPILOGO GENERALE METRICHE COMPILATO:")
    print("=" * 70)
    print(df_summary.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 70)
    print(f"Report Markdown salvato in: {md_output_path}")
    print(f"Foglio dati CSV salvato in:   {csv_output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
