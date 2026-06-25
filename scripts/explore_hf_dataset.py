"""
explore_hf_dataset.py
=====================
Ispeziona la STRUTTURA di un dataset HuggingFace, in due modalità:

  --api   (CONSIGLIATA) interroga il dataset-viewer di HuggingFace: schema +
          distribuzione dei valori delle colonne categoriche, ISTANTANEO, NON
          scarica nulla. Ideale per dataset con immagini grandi (es. OpenFake).

  default modalità STREAMING: scarica i primi N esempi per ispezionare le colonne.
          Va bene per dataset con immagini piccole (es. D3), ma può essere lento se
          le immagini sono grandi (deve scaricare il primo shard parquet).

Scopo: scoprire quali colonne ha il dataset e, soprattutto, DOVE sta la label
del generatore (es. "stable-diffusion-xl", "FLUX", ...) e la label real/fake.

Esempi d'uso (sul server):
    # ispezione istantanea via API (consigliata)
    python scripts/explore_hf_dataset.py --repo ComplexDataLab/OpenFake --config core --split test --api
    # streaming (per dataset leggeri)
    python scripts/explore_hf_dataset.py --repo elsaEU/ELSA_D3 --split train

Se il dataset è "gated" e dà errore di autenticazione:
    huggingface-cli login        # incolla un token da https://huggingface.co/settings/tokens
"""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset

VIEWER = "https://datasets-server.huggingface.co"


def _get_json(endpoint, qs):
    with urllib.request.urlopen(f"{VIEWER}/{endpoint}?{qs}") as r:
        return json.load(r)


def distribution_from_parquet(repo, config, split, columns, max_files=8, row_groups_per_file=1):
    """Legge SOLO le colonne indicate, e solo i primi row-group, da alcuni file parquet,
    usando range-request HTTP (pyarrow + fsspec) → non scarica le immagini né i file interi.
    NB: è un CAMPIONE (primi row-group di alcuni shard), non il conteggio totale dello split."""
    qs = urllib.parse.urlencode({"dataset": repo, "config": config or "default", "split": split})
    info = _get_json("parquet", qs)
    files = [f["url"] for f in info.get("parquet_files", []) if f["split"] == split]
    if not files:
        print("Nessun file parquet trovato per questo split.")
        return
    print(f"Campiono le colonne {columns}: primo row-group di {min(max_files, len(files))} "
          f"shard su {len(files)} (range-request, niente immagini)...\n")
    frames = []
    for url in files[:max_files]:
        with fsspec.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(min(row_groups_per_file, pf.num_row_groups)):
                frames.append(pf.read_row_group(rg, columns=columns).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    print(f"Righe campionate: {len(df)}\n")
    for c in columns:
        vc = df[c].value_counts()
        print(f"[{c}] {len(vc)} valori distinti (nel campione):")
        for val, cnt in vc.items():
            print(f"      {cnt:7d}  {val}")
        print()


def explore_via_api(repo, config, split):
    """Interroga il dataset-viewer di HF: schema + frequenze delle colonne categoriche.
    Istantaneo, non scarica i dati. Per split enormi le statistiche possono mancare (404):
    in tal caso ripiega sulla lettura colonnare dei parquet."""
    qs = urllib.parse.urlencode({"dataset": repo, "config": config or "default", "split": split})

    # 1) schema dalle prime righe
    first = _get_json("first-rows", qs)
    columns = [feat["name"] for feat in first.get("features", [])]
    print("COLONNE (dallo schema del dataset-viewer):")
    for feat in first.get("features", []):
        ftype = feat["type"].get("dtype") or feat["type"].get("_type", "?")
        print(f"  - {feat['name']:20s} {ftype}")
    print()

    # 2) statistiche: frequenze dei valori per le colonne categoriche
    try:
        stats = _get_json("statistics", qs)
    except urllib.error.HTTPError as e:
        print(f"Statistiche non disponibili (HTTP {e.code}) — split troppo grande.")
        print("Ripiego: leggo le colonne categoriche direttamente dai parquet.\n")
        cat_cols = [c for c in ("label", "model", "type") if c in columns]
        distribution_from_parquet(repo, config, split, cat_cols)
        return

    n = stats.get("num_examples")
    print(f"Statistiche su {n} esempi (campione del dataset-viewer):\n")
    for col in stats.get("statistics", []):
        name = col["column_name"]
        cstats = col.get("column_statistics", {})
        freqs = cstats.get("frequencies")
        if freqs:
            print(f"[{name}] {len(freqs)} valori distinti:")
            for val, cnt in sorted(freqs.items(), key=lambda kv: -kv[1]):
                print(f"      {cnt:7d}  {val}")
            print()


def preview(value, maxlen=90):
    """Rappresentazione compatta di un valore (le immagini PIL non si stampano per intero)."""
    if hasattr(value, "size") and hasattr(value, "mode"):      # immagine PIL
        return f"<PIL image mode={value.mode} size={value.size}>"
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    s = repr(value)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def is_image(value):
    return hasattr(value, "size") and hasattr(value, "mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="es. elsaEU/ELSA_D3 o ComplexDataLab/OpenFake")
    ap.add_argument("--split", default="train", help="train / validation / test")
    ap.add_argument("--config", default=None, help="configurazione dataset, es. 'core' o 'reddit' per OpenFake")
    ap.add_argument("--n", type=int, default=300, help="quanti esempi usare per le statistiche (solo streaming)")
    ap.add_argument("--api", action="store_true",
                    help="usa il dataset-viewer di HF (istantaneo, non scarica): CONSIGLIATO")
    args = ap.parse_args()

    config_str = f" config={args.config}" if args.config else ""

    if args.api:
        print(f"\n=== {args.repo} — split={args.split}{config_str} (API dataset-viewer) ===\n")
        explore_via_api(args.repo, args.config, args.split)
        return

    print(f"\n=== {args.repo} — split={args.split}{config_str} (streaming) ===\n")
    ds = load_dataset(args.repo, args.config, split=args.split, streaming=True)
    it = iter(ds)
    first = next(it)


    # 1) Colonne del primo esempio: nome, tipo, anteprima
    print("COLONNE del primo esempio:")
    for k, v in first.items():
        print(f"  - {k:28s} {type(v).__name__:12s} {preview(v)}")
    print()

    # 2) Raccogli N esempi per studiare i campi (quali sono categorici?)
    examples = [first]
    for _ in range(args.n - 1):
        try:
            examples.append(next(it))
        except StopIteration:
            break
    print(f"Analizzo {len(examples)} esempi per trovare i campi categorici "
          f"(candidati: generatore, label real/fake)\n")

    # 3) Per ogni colonna NON-immagine con tipi semplici, mostra la distribuzione dei valori.
    #    Un campo con pochi valori distinti è probabilmente il 'generatore' o la 'label'.
    for k in first.keys():
        if is_image(first[k]):
            print(f"[{k}] -> immagine (saltato)\n")
            continue
        vals = [e.get(k) for e in examples]
        if not all(isinstance(v, (str, int, bool, float, type(None))) for v in vals):
            print(f"[{k}] -> tipo complesso (dict/list/bytes), ispezionare a mano\n")
            continue
        counter = Counter(vals)
        if len(counter) <= 30:
            print(f"[{k}] CAMPO CATEGORICO — {len(counter)} valori distinti:")
            for val, cnt in counter.most_common():
                print(f"      {cnt:4d}  {preview(val)}")
            print()
        else:
            print(f"[{k}] {len(counter)} valori distinti su {len(vals)} "
                  f"(probabile testo/id libero, non categorico)\n")

if __name__ == "__main__":
    main()
