# -*- coding: utf-8 -*-
"""
Script per la generazione di grafici e tabelle visuali "publication-ready" per il report PDF.
Genera heatmap delle metriche globali, breakdown disaggregati e grafici di robustezza.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "report_figures"

def plot_global_heatmap(df_summary, out_dir):
    """
    Genera una heatmap dell'AUROC globale per ciascun detector su ciascun dataset.
    """
    try:
        pivot_df = df_summary.pivot(index="Detector", columns="Dataset", values="AUROC")
        # Ordiniamo in modo specifico per coerenza
        detectors_order = ["CNNDET", "UFD", "CODE", "AIDE_PROGAN", "AIDE_GENIMAGE"]
        datasets_order = ["GAN", "D3", "OPENFAKE"]
        
        # Filtriamo solo quelli disponibili
        detectors_order = [d for d in detectors_order if d in pivot_df.index]
        datasets_order = [d for d in datasets_order if d in pivot_df.columns]
        
        if not detectors_order or not datasets_order:
            return
            
        pivot_df = pivot_df.loc[detectors_order, datasets_order]
        
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        im = ax.imshow(pivot_df.values, cmap="YlGnBu", vmin=0.3, vmax=1.0)
        
        # Annotazioni numeriche
        for i in range(len(detectors_order)):
            for j in range(len(datasets_order)):
                val = pivot_df.values[i, j]
                text_color = "white" if val > 0.65 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontweight="bold")
                
        # Etichette assi
        ax.set_xticks(np.arange(len(datasets_order)))
        ax.set_yticks(np.arange(len(detectors_order)))
        ax.set_xticklabels([d.upper() for d in datasets_order], fontweight="bold")
        ax.set_yticklabels([d.replace("_", "-") for d in detectors_order], fontweight="bold")
        
        plt.title("Confronto AUROC Globale dei Detector", fontsize=12, fontweight="bold", pad=15)
        fig.colorbar(im, ax=ax, label="AUROC Score")
        
        plt.tight_layout()
        plt.savefig(out_dir / "fig_global_auroc_heatmap.png", bbox_inches="tight")
        plt.savefig(out_dir / "fig_global_auroc_heatmap.pdf", bbox_inches="tight")
        plt.close()
        print("Grafico globale salvato.")
    except Exception as e:
        print(f"Errore generazione heatmap globale: {e}")

def plot_breakdown_heatmap(df_breakdown, dataset_name, out_dir):
    """
    Genera una heatmap dei risultati disaggregati per generatore per un dataset specifico.
    """
    df_ds = df_breakdown[df_breakdown["Dataset"].str.upper() == dataset_name.upper()]
    if df_ds.empty:
        return
        
    try:
        # Pivot: righe = Generatore, colonne = Detector, valore = AUROC
        pivot_df = df_ds.pivot(index="Generator", columns="Detector", values="AUROC")
        
        # Ordiniamo colonne (detector)
        cols_order = ["CNNDET", "UFD", "CODE", "AIDE_PROGAN", "AIDE_GENIMAGE"]
        cols_order = [c for c in cols_order if c in pivot_df.columns]
        pivot_df = pivot_df[cols_order]
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(pivot_df.values, cmap="RdYlGn", vmin=0.3, vmax=1.0)
        
        for i in range(len(pivot_df.index)):
            for j in range(len(cols_order)):
                val = pivot_df.values[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray")
                else:
                    text_color = "white" if (val > 0.75 or val < 0.45) else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold")
                    
        ax.set_xticks(np.arange(len(cols_order)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels([c.replace("_", "-") for c in cols_order], rotation=15, fontweight="bold")
        ax.set_yticklabels(pivot_df.index, fontweight="bold")
        
        plt.title(f"AUROC per Generatore su {dataset_name.upper()}", fontsize=12, fontweight="bold", pad=15)
        fig.colorbar(im, ax=ax, label="AUROC")
        
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_breakdown_heatmap_{dataset_name.lower()}.png", bbox_inches="tight")
        plt.savefig(out_dir / f"fig_breakdown_heatmap_{dataset_name.lower()}.pdf", bbox_inches="tight")
        plt.close()
        print(f"Heatmap breakdown per {dataset_name} salvata.")
    except Exception as e:
        print(f"Errore generazione heatmap breakdown {dataset_name}: {e}")

def plot_robustness_chart(df_robustness, out_dir):
    """
    Genera un grafico a linee della degradazione della robustezza (Clean -> JPEG_70 -> JPEG_50 -> Resize_128).
    """
    df_overall = df_robustness[df_robustness["Generator"] == "OVERALL"]
    if df_overall.empty:
        return
        
    try:
        modes = ["Clean", "JPEG_70", "JPEG_50", "Resize_128"]
        
        plt.figure(figsize=(7, 5), dpi=150)
        
        styles = {
            "CNNDET": ("-o", "#d62728"),
            "UFD": ("-s", "#2ca02c"),
            "AIDE_PROGAN": ("--^", "#ff7f0e"),
            "AIDE_GENIMAGE": ("-^", "#1f77b4")
        }
        
        for i, row in df_overall.iterrows():
            det = row["Detector"]
            style = styles.get(det, ("-x", "gray"))
            
            y_vals = [row[f"AUROC_{m}"] for m in modes]
            plt.plot(modes, y_vals, style[0], label=det.replace("_", "-"), color=style[1], linewidth=2.5, markersize=8)
            
        plt.title("Sensibilità dei Detector ad Alterazioni Social/Risoluzione (OpenFake)", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Tipo di Perturbazione", fontsize=10)
        plt.ylabel("AUROC Globale", fontsize=10)
        plt.ylim(0.35, 1.02)
        plt.legend(loc="lower left", frameon=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.3)
        
        plt.savefig(out_dir / "fig_robustness_degradation.png", bbox_inches="tight")
        plt.savefig(out_dir / "fig_robustness_degradation.pdf", bbox_inches="tight")
        plt.close()
        print("Grafico degradazione robustezza salvato.")
    except Exception as e:
        print(f"Errore generazione grafico robustezza: {e}")

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Caricamento summary
    summary_path = RESULTS_DIR / "summary_report.csv"
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        plot_global_heatmap(df_sum, FIG_DIR)
    else:
        print("[WARNING] summary_report.csv non trovato. Salto heatmap globale.")
        
    # 2. Caricamento breakdown
    breakdown_path = RESULTS_DIR / "breakdown_per_generator.csv"
    if breakdown_path.exists():
        df_breakdown = pd.read_csv(breakdown_path)
        plot_breakdown_heatmap(df_breakdown, "gan", FIG_DIR)
        plot_breakdown_heatmap(df_breakdown, "d3", FIG_DIR)
        plot_breakdown_heatmap(df_breakdown, "openfake", FIG_DIR)
    else:
        print("[WARNING] breakdown_per_generator.csv non trovato. Salto heatmap breakdown.")
        
    # 3. Caricamento robustezza
    robustness_path = RESULTS_DIR / "robustness_results.csv"
    if robustness_path.exists():
        df_robustness = pd.read_csv(robustness_path)
        plot_robustness_chart(df_robustness, FIG_DIR)
    else:
        print("[WARNING] robustness_results.csv non trovato. Salto grafico robustezza.")

if __name__ == "__main__":
    main()
