"""
build_subset.py
===============
Estrae un subset RIPRODUCIBILE di immagini fake da D3 (validation) e le salva
come file JPEG su disco, producendo anche un manifest CSV.

Struttura output:
  /work/cvcs2026/deep_pixels/datasets/D3/images/
      deepfloyd/0000.jpg ... 1999.jpg
      sd14/0000.jpg      ... 1999.jpg
      sd21/0000.jpg      ... 1999.jpg
      sdxl/0000.jpg      ... 1999.jpg
  /work/cvcs2026/deep_pixels/datasets/D3/manifest.csv
      path, label, generator, source

Uso:
    python scripts/build_subset.py [--n 2000] [--seed 42]
"""
import argparse
import csv
import io
import glob
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from PIL import Image

D3_DIR = Path("/work/cvcs2026/deep_pixels/datasets/D3")
IMG_DIR = D3_DIR / "images"

GENERATORS = {
    "image_gen0": "deepfloyd",
    "image_gen1": "sd14",
    "image_gen2": "sd21",
    "image_gen3": "sdxl",
}


def extract_image(raw) -> Image.Image:
    """Converte il campo immagine parquet (bytes o dict) in PIL Image."""
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def main(n: int, seed: int):
    parquet_files = sorted(glob.glob(str(D3_DIR / "data" / "validation-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Nessun parquet in {D3_DIR}/data/")

    print(f"Shard trovati: {len(parquet_files)}")
    print(f"Campione per generatore: {n}  |  seed: {seed}")

    # Legge solo le colonne immagine (evita di caricare tutto in RAM)
    cols = list(GENERATORS.keys())
    df = pd.concat(
        [pd.read_parquet(f, columns=cols) for f in parquet_files],
        ignore_index=True,
    )
    print(f"Righe totali caricate: {len(df)}")

    df_sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"Campionate: {len(df_sample)} righe")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for col, gen_name in GENERATORS.items():
        gen_dir = IMG_DIR / gen_name
        gen_dir.mkdir(exist_ok=True)
        print(f"  Estraggo {n} immagini per '{gen_name}' ...", end="", flush=True)
        for idx, raw in enumerate(df_sample[col]):
            img = extract_image(raw)
            out_path = gen_dir / f"{idx:04d}.jpg"
            img.save(out_path, "JPEG", quality=95)
            manifest_rows.append({
                "path": str(out_path),
                "label": "fake",
                "generator": gen_name,
                "source": "D3",
            })
        print(f" fatto ({n} file)")

    manifest_path = D3_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest scritto: {manifest_path}  ({len(manifest_rows)} righe)")
    print(f"Immagini totali D3 fake: {len(manifest_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Immagini per generatore")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.n, args.seed)
