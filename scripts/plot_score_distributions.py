# -*- coding: utf-8 -*-
"""
Script per tracciare la distribuzione dei punteggi dei detector (Reali vs Fake)
per monitorare visivamente la calibrazione e la separabilità.
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    optimal_threshold = thresholds[idx]
    return eer, optimal_threshold

def plot_distribution(file_path, output_dir):
    name = file_path.stem
    if "-" not in name:
        return
    parts = name.split("-")
    detector = parts[0].upper()
    dataset = parts[1].upper()
    
    df = pd.read_csv(file_path)
    
    # Rilevamento colonna score
    score_col = None
    for col in df.columns:
        if "score" in col:
            score_col = col
            break
            
    if not score_col:
        return
        
    # Drop NaN
    df = df.dropna(subset=["ground_truth", score_col])
    
    y_true = df["ground_truth"].values
    y_scores = df[score_col].values
    
    if len(np.unique(y_true)) < 2:
        return
        
    auroc = roc_auc_score(y_true, y_scores)
    eer, opt_th = calculate_eer(y_true, y_scores)
    
    # Separazione per classe
    scores_real = df[df["ground_truth"] == 0][score_col].values
    scores_fake = df[df["ground_truth"] == 1][score_col].values
    
    plt.figure(figsize=(7, 5), dpi=150)
    
    # Istogramma reali (Verde)
    plt.hist(scores_real, bins=50, alpha=0.5, label='Reali (Label=0)', color='#2ca02c', density=True)
    # Istogramma fake (Rosso)
    plt.hist(scores_fake, bins=50, alpha=0.5, label='Fake (Label=1)', color='#d62728', density=True)
    
    # Soglia fissa 0.5
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Soglia standard (0.5)')
    # Soglia ottimale EER
    plt.axvline(opt_th, color='#1f77b4', linestyle='-', linewidth=2, label=f'Soglia EER ({opt_th:.3f})')
    
    plt.title(f"Distribuzione Score: {detector} su {dataset}\n(AUROC: {auroc:.4f} | EER: {eer:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel("Score (Probabilità Fake)")
    plt.ylabel("Densità di Frequenza")
    plt.xlim(-0.02, 1.02)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    out_file = output_dir / f"score_dist-{parts[0]}-{parts[1]}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Grafico salvato: {out_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Genera grafici delle distribuzioni dei punteggi reali vs fake")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                        help="Cartella di output per i grafici")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(list(RESULTS_DIR.glob("*.csv")))
    evaluation_files = []
    for f in csv_files:
        if "-" in f.name and not f.name.startswith("summary_report") and not f.name.startswith("benchmark_speed"):
            evaluation_files.append(f)
            
    if not evaluation_files:
        print("Nessun file di predizioni trovato.")
        return
        
    print(f"Generazione grafici per {len(evaluation_files)} file...")
    for f in evaluation_files:
        try:
            plot_distribution(f, out_dir)
        except Exception as e:
            print(f"Errore su {f.name}: {e}")
            
if __name__ == "__main__":
    main()
