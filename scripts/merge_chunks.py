# -*- coding: utf-8 -*-
"""
Script per unire i file CSV dei chunk di inferenza parallela in un unico file finale.
Ordina i record per sample_id per ripristinare la sequenza originale e rimuove i chunk.
"""
import argparse
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def merge_chunks(tag, dataset, num_chunks):
    print(f"Fusione dei {num_chunks} chunk per {tag} su {dataset.upper()}...")
    chunks = []
    for i in range(num_chunks):
        chunk_path = RESULTS_DIR / f"{tag}-{dataset}-chunk{i}.csv"
        if not chunk_path.exists():
            print(f"[ERROR] Chunk {i} non trovato a: {chunk_path}")
            return False
        df_chunk = pd.read_csv(chunk_path)
        chunks.append(df_chunk)
        
    df_merged = pd.concat(chunks, ignore_index=True)
    
    # Ordiniamo per sample_id per garantire che l'ordine corrisponda al manifest
    df_merged = df_merged.sort_values(by="sample_id")
    
    output_path = RESULTS_DIR / f"{tag}-{dataset}.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] Fusione completata con successo! Salvato in: {output_path}")
    
    # Rimozione dei file chunk per pulizia
    for i in range(num_chunks):
        chunk_path = RESULTS_DIR / f"{tag}-{dataset}-chunk{i}.csv"
        try:
            chunk_path.unlink()
        except Exception as e:
            print(f"[WARNING] Impossibile rimuovere il chunk {chunk_path.name}: {e}")
    print("File dei chunk intermedi rimossi per pulizia del workspace.")
    return True
    
def main():
    parser = argparse.ArgumentParser(description="Unisce i chunk CSV di inferenza in un unico file di risultati")
    parser.add_argument("--tag", type=str, required=True, help="Tag del modello (es. aide_progan, aide_genimage)")
    parser.add_argument("--dataset", type=str, default="openfake", help="Nome del dataset (es. openfake)")
    parser.add_argument("--num_chunks", type=int, default=4, help="Numero di chunk totali")
    args = parser.parse_args()
    
    merge_chunks(args.tag, args.dataset, args.num_chunks)
    
if __name__ == "__main__":
    main()
