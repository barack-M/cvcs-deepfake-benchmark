# -*- coding: utf-8 -*-
import os
import csv
import shutil
import urllib.request
import io
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from PIL import Image

D3_DIR = Path("/work/cvcs2026/deep_pixels/datasets/D3")
IMAGES_DIR = D3_DIR / "images" / "real"
MANIFEST_CSV = D3_DIR / "manifest.csv"

GENERATORS = ["deepfloyd", "sd14", "sd21", "sdxl"]
TARGET_SUCCESSES = 3000

def download_and_validate(row_idx, url):
    """
    Scarica l'immagine, verifica che sia apribile da PIL e che abbia dimensioni >= 100x100.
    Restituisce i dettagli solo in caso di successo assoluto.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            content_type = response.headers.get('Content-Type', '')
            raw_bytes = response.read()
            
            # 1. Verifica integrità con PIL in memoria
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()  # Verifica che il file non sia corrotto
                
            # Riapriamo per leggere la dimensione (verify chiude il file)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                w, h = img.size
                fmt = img.format
                
                # 2. Verifica dimensioni minime (evita pixel 1x1 o icone minuscole)
                if w < 100 or h < 100:
                    return row_idx, None, False, f"Dimensioni troppo piccole: {w}x{h}"
                
            # Determiniamo l'estensione adatta
            ext = mimetypes.guess_extension(content_type)
            if not ext or ext == '.jpe':
                ext = '.jpg'
            
            file_name = f"{row_idx:04d}{ext}"
            file_path = IMAGES_DIR / file_name
            
            # 3. Salvataggio binario puro dei byte originari
            with open(file_path, "wb") as f:
                f.write(raw_bytes)
                
            return row_idx, str(file_path.resolve()), True, None
            
    except Exception as e:
        return row_idx, None, False, str(e)

def main():
    print(f"Svuotamento preventivo della directory delle immagini reali: {IMAGES_DIR}")
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(glob_glob_parquet())
    if not parquet_files:
        print(f"Errore: Nessun file parquet in {D3_DIR}/data/")
        return

    print("Caricamento URL dal dataset D3...")
    df = pd.concat([pd.read_parquet(f, columns=["url"]) for f in parquet_files], ignore_index=True)
    
    # Prepariamo i task per tutti i 4800 URL possibili
    tasks = [(idx, url) for idx, url in enumerate(df["url"])]
    print(f"Righe totali caricate: {len(tasks)}")
    print(f"Inizio download con validazione integrata (50 worker). Target: {TARGET_SUCCESSES} immagini sane...")

    downloaded_paths = {}
    completed = 0
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(download_and_validate, idx, url): (idx, url) for idx, url in tasks}
        
        for future in as_completed(futures):
            idx, path, success, err = future.result()
            completed += 1
            
            if success:
                # Salviamo solo se non abbiamo ancora raggiunto il target
                if len(downloaded_paths) < TARGET_SUCCESSES:
                    downloaded_paths[idx] = path
                else:
                    # Raggiunto il target, eliminiamo l'eventuale file appena scaricato per eccesso
                    try:
                        Path(path).unlink()
                    except Exception:
                        pass
            
            if completed % 300 == 0:
                print(f"  Elaborati: {completed}/{len(tasks)} | Successi validi: {len(downloaded_paths)}")
                
            # Se abbiamo raggiunto esattamente il target, possiamo interrompere
            if len(downloaded_paths) == TARGET_SUCCESSES:
                print(f"\nTarget di {TARGET_SUCCESSES} immagini reali sane raggiunto!")
                break

    print(f"\nDownload completato. Immagini sane salvate: {len(downloaded_paths)}")

    if len(downloaded_paths) < TARGET_SUCCESSES:
        print(f"ATTENZIONE: Non è stato possibile raggiungere il target. Trovati solo {len(downloaded_paths)} successi.")
    
    # Selezioniamo solo i successi effettivi per il manifest
    final_reals = downloaded_paths
    
    # Rimuoviamo fisicamente dal disco qualsiasi file extra che i thread in background
    # hanno continuato a scaricare mentre l'esecutore si spegneva.
    print("Rimozione dei file extra scaricati in background...")
    valid_paths = set(final_reals.values())
    removed_extra = 0
    for f in os.listdir(IMAGES_DIR):
        file_path = IMAGES_DIR / f
        if str(file_path.resolve()) not in valid_paths:
            try:
                file_path.unlink()
                removed_extra += 1
            except Exception:
                pass
    print(f"  Rimosse {removed_extra} immagini extra orfane.")
    
    print("\nGenerazione del manifest.csv bilanciato per D3...")
    manifest_rows = []
    
    # 1. Aggiungiamo le immagini REALI scaricate (label = 0)
    for idx, path in sorted(final_reals.items()):
        manifest_rows.append({
            "path": path,
            "index": "",
            "label": 0,
            "generator": "real",
            "dataset": "d3"
        })
        
    # 2. Aggiungiamo le immagini FAKE corrispondenti (label = 1)
    for idx in sorted(final_reals.keys()):
        for gen in GENERATORS:
            manifest_rows.append({
                "path": "",
                "index": idx,
                "label": 1,
                "generator": gen,
                "dataset": "d3"
            })
            
    # Scrittura del CSV finale
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "index", "label", "generator", "dataset"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(f"Manifest D3 unificato scritto: {MANIFEST_CSV} ({len(manifest_rows)} righe)")
    print(f"  - Reali (su disco): {len(final_reals)}")
    print(f"  - Fake (nei Parquet): {len(final_reals) * len(GENERATORS)} (3000 per ciascuno dei 4 generatori)")

def glob_glob_parquet():
    import glob
    return glob.glob(str(D3_DIR / "data" / "validation-*.parquet"))

if __name__ == "__main__":
    main()
