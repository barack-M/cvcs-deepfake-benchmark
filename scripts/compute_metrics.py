# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def calculate_eer(y_true, y_scores):
    """
    Calcola l'Equal Error Rate (EER) e la soglia ottimale corrispondente.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Troviamo l'indice in cui la differenza tra FPR e FNR è minima
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    optimal_threshold = thresholds[idx]
    return eer, optimal_threshold

def main():
    parser = argparse.ArgumentParser(description="Calcolo delle metriche di valutazione per Deepfake Detection")
    parser.add_argument("--input", type=str, required=True,
                        help="Percorso al file CSV dei punteggi (es. ufd-gan.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Errore: Il file {input_path} non esiste.")
        return

    df = pd.read_csv(input_path)
    
    # Rileviamo automaticamente quale detector è stato valutato
    score_col = None
    for col in df.columns:
        if "score" in col:
            score_col = col
            break
            
    if not score_col:
        print("Errore: Nessuna colonna di punteggi trovata (es. 'ufd_score', 'code_score').")
        return

    print("=" * 60)
    print(f"ANALISI METRICHE PER: {input_path.name}")
    print(f"Colonna score rilevata: {score_col}")
    print("=" * 60)

    y_true = df["ground_truth"].values
    y_scores = df[score_col].values

    # 1. Metriche Generali (Overall)
    overall_auroc = roc_auc_score(y_true, y_scores)
    overall_ap = average_precision_score(y_true, y_scores)
    overall_acc_05 = accuracy_score(y_true, y_scores >= 0.5)
    
    # Calcolo EER e soglia ottimale
    overall_eer, opt_thresh = calculate_eer(y_true, y_scores)
    overall_acc_opt = accuracy_score(y_true, y_scores >= opt_thresh)

    print("\n--- METRICHE GENERALI ---")
    print(f"AUROC:             {overall_auroc:.4f}")
    print(f"Average Precision: {overall_ap:.4f}")
    print(f"Accuracy (th=0.5): {overall_acc_05:.4f}")
    print(f"Equal Error Rate:  {overall_eer:.4f} (th_ottimale={opt_thresh:.4f})")
    print(f"Accuracy (th_ott): {overall_acc_opt:.4f}")

    # 2. Breakdown per Generatore (Specifico per dataset)
    print("\n--- BREAKDOWN PER GENERATORE ---")
    
    # Rileviamo se i reali sono condivisi sotto nomi separati (es. 'imagenet', 'docci', 'real')
    # o se sono distribuiti all'interno di ciascun generatore condividendo lo stesso nome (es. 'progan')
    real_gens = set(df[df["ground_truth"] == 0]["generator"].unique())
    fake_gens = set(df[df["ground_truth"] == 1]["generator"].unique())
    
    # Se l'intersezione è vuota, significa che le classi reali e fake non condividono i nomi dei generatori
    # (reali = {'imagenet', 'docci'}; fake = {'flux.2-klein-9b', 'veo-3', ...})
    is_shared_real = len(real_gens.intersection(fake_gens)) == 0
    
    breakdown_data = []
    
    if is_shared_real:
        # D3 / OpenFake: Accoppiamo le fake di ogni generatore con TUTTI i reali del dataset
        df_reals = df[df["ground_truth"] == 0]
        
        for gen in sorted(fake_gens):
            df_fakes = df[(df["generator"] == gen) & (df["ground_truth"] == 1)]
            # Uniamo le fake di questo generatore con tutti i reali
            df_gen = pd.concat([df_fakes, df_reals])
            
            y_true_gen = df_gen["ground_truth"].values
            y_scores_gen = df_gen[score_col].values
            
            auroc = roc_auc_score(y_true_gen, y_scores_gen)
            ap = average_precision_score(y_true_gen, y_scores_gen)
            eer, opt_t = calculate_eer(y_true_gen, y_scores_gen)
            acc_05 = accuracy_score(y_true_gen, y_scores_gen >= 0.5)
            
            breakdown_data.append({
                "Generator": gen,
                "Samples (F+R)": f"{len(df_fakes)}+{len(df_reals)}",
                "AUROC": auroc,
                "AP": ap,
                "Acc (th=0.5)": acc_05,
                "EER": eer
            })
    else:
        # GAN: I reali sono già distribuiti all'interno di ciascun generatore condividendo il nome
        generators = df["generator"].unique()
        for gen in sorted(generators):
            df_gen = df[df["generator"] == gen]
            y_true_gen = df_gen["ground_truth"].values
            y_scores_gen = df_gen[score_col].values
            
            if len(np.unique(y_true_gen)) < 2:
                auroc, ap, eer = np.nan, np.nan, np.nan
            else:
                auroc = roc_auc_score(y_true_gen, y_scores_gen)
                ap = average_precision_score(y_true_gen, y_scores_gen)
                eer, _ = calculate_eer(y_true_gen, y_scores_gen)
                
            acc_05 = accuracy_score(y_true_gen, y_scores_gen >= 0.5)
            
            breakdown_data.append({
                "Generator": gen,
                "Samples (F+R)": f"{len(df_gen)}",
                "AUROC": auroc,
                "AP": ap,
                "Acc (th=0.5)": acc_05,
                "EER": eer
            })
        
    df_breakdown = pd.DataFrame(breakdown_data)
    
    # Stampiamo a terminale una tabella formattata
    print(df_breakdown.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 60)

if __name__ == "__main__":
    main()
