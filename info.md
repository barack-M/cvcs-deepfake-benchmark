# INFO — Progetto CVCS Deepfake Detection

> **Documento condiviso del gruppo** (Matteo + Enrico). Qui raccogliamo TUTTO sul progetto: idea,
> cose da fare, decisioni prese e da prendere, dataset, protocollo, detector, metriche.
>
> 👉 Per il funzionamento del **SERVER** (percorsi, quote, comandi, SLURM, cosa NON fare) vedi **`info_server.md`**.

**Progetto:** *Deepfake Detection in the Wild: Cross-Generator Generalization Benchmarking*  
**Corso:** Computer Vision and Cognitive Systems (CVCS) 2025/2026, UNIMORE — prof. Lorenzo Baraldi  
**Gruppo:** Deep Pixels — Matteo Baracchi + Enrico Ranieri  
**Tutor:** Silvia Cappelletti (silvia.cappelletti@unimore.it), Tobia Poppi (tobia.poppi@unimore.it)  
**Repo:** https://github.com/barack-M/cvcs-deepfake-benchmark

---

## 0. L'idea in una frase

Un detector di deepfake addestrato sui generatori di *ieri* funziona ancora sui generatori di *oggi*? 
Prendiamo detector **pre-addestrati**, li valutiamo **cross-generator** (su famiglie di generatori mai viste in training) e analizziamo **quando, perché e quanto** degradano le loro prestazioni.

L'asse temporale e tecnologico dell'analisi copre: **GAN-based** $\rightarrow$ **Early Diffusion (SD1.4, SD2.1, DeepFloyd)** $\rightarrow$ **Late Diffusion (SDXL)** $\rightarrow$ **Modern Open/Proprietary (FLUX, Midjourney, Sora, Veo)**.

---

## 1. WORKFLOW — come lavoriamo

### Dove sta cosa (vale per entrambi; `<username>` = mbaracchi / eranieri)
* **Codice** $\rightarrow$ nella propria HOME: `/homes/<username>/cvcs2026/cvcs-deepfake-benchmark` (clone git)
* **Python** $\rightarrow$ venv `/homes/<username>/cvcs2026/venv` (Python 3.10.12). Attivare sempre con:
  `source /homes/<username>/cvcs2026/venv/bin/activate`
* **Dati, pesi, feature, risultati** $\rightarrow$ cartella condivisa `/work/cvcs2026/deep_pixels/` (BeeGFS):
  * `datasets/` $\rightarrow$ percorsi e manifest (D3, OpenFake, GAN)
  * `weights/` $\rightarrow$ i file `.pth` pre-addestrati dei detector
  * `results/` $\rightarrow$ score grezzi per immagine (detector × dataset) in CSV per calcolare le metriche

### Linee guida per il codice
* **Percorsi ASSOLUTI** nei file di configurazione e negli script per evitare conflitti.
* **Nessuna duplicazione di file immagine su BeeGFS:** I dataset in Parquet (OpenFake, D3 fakes) rimangono in Parquet per non esaurire la quota di inode e lo spazio su `/work` (che ha solo ~260 GB liberi per tutto il corso).
* **Niente file binari o dati su git:** Tutti i dataset e i manifest grandi sono esclusi tramite `.gitignore` e risiedono solo su `/work`.

---

## 2. TODO — cose da fare

Legenda: `[ ]` da fare · `[/]` in corso · `[x]` fatto

### Setup
* [x] Repo GitHub creata e clonata (locale + home sul server)
* [x] Cartelle condivise create su `/work`: `datasets/`, `weights/`, `results/`
* [x] PyCharm remote (SSH interpreter su venv + Deployment) e configurazione venv
* [ ] Clonare i repository ufficiali dei detector (CoDE, UniversalFakeDetect) in home

### Dati (Completati e Uniformati ✅)
* [x] Esplorazione e analisi schemi di D3, GAN (ForenSynths) e OpenFake
* [x] **GAN (ForenSynths):** Estrazione selettiva delle PNG native (6.000 file totali, no JPEG compression) + scrittura `manifest.csv`.
* [x] **OpenFake:** Scaricato l'intero split test (68 GB Parquet) + allineato `manifest.csv` basato su indici (91.398 righe).
* [x] **D3:** Scritto e lanciato lo screening delle URL reali. Identificati 3.500 prompt-matched funzionanti.
* [x] **D3:** Download binario puro (lossless) delle 3.000 reali sane (eliminando file corrotti o 1x1px) + allineato `manifest.csv` misto (disco/parquet, 15.000 righe).

### Detector
* [ ] CoDE: Estendere la bozza di inferenza (Enrico) a un dataset intero usando i manifest
* [ ] UniversalFakeDetect: Configurare la pipeline per estrarre le feature CLIP ed eseguire il classificatore lineare
* [ ] Decidere il terzo detector (es. Effort da DeepfakeBench) e scaricarne i pesi
* [ ] Reperire e organizzare i pesi pre-addestrati dei 3 detector in `/work/cvcs2026/deep_pixels/weights/`

### Evaluation & Analisi
* [ ] Pipeline di inferenza: (detector, dataset_manifest) $\rightarrow$ predizioni in `results/`
* [ ] Calcolo metriche per-generatore (AUROC, AP, Accuracy con soglia fissa/EER)
* [ ] Generazione t-SNE degli embedding delle feature per studiare lo spazio di separabilità
* [ ] Analisi dei failure mode (sensibilità a compressione JPEG, contenuto dell'immagine, etc.)

---

## 3. MAPPA E STRUTTURA DEI DATASET

Per garantire l'assenza di bias legati alla ricompressione o alla pipeline di salvataggio (es. evitare che il detector impari a distinguere la qualità JPEG di un'immagine rispetto a un'altra), tutti i dataset sono memorizzati nei loro formati nativi (PNG o Parquet compressi). 

I manifest sono pronti e formattati in modo coerente.

```
/work/cvcs2026/deep_pixels/datasets/
├── D3/
│   ├── data/                           # 11 file Parquet con le immagini generate
│   ├── images/real/                    # 3.000 immagini reali scaricate da LAION
│   ├── live_validation_indices.csv     # Indici di backup URL vivi
│   └── manifest.csv                    # Manifest D3 (15.000 righe)
├── GAN/
│   ├── CNN_synth_testset/              # Sotto-cartelle (cyclegan, progan, stylegan)
│   └── manifest.csv                    # Manifest GAN (6.000 righe)
└── OpenFake/
    ├── test_set/core/                  # 13 file Parquet (65 GB) dello split test
    ├── manifest_openfake.csv           # File originale di Enrico
    └── manifest.csv                    # Manifest OpenFake (91.398 righe)
```

---

### 3.1 Dataset D3 (ELSA_D3) — paired validation set (15.000 righe)

* **Metodologia:** D3 non ha immagini reali embedded. Poiché i nodi di calcolo GPU non hanno accesso a Internet, abbiamo scaricato preventivamente le immagini reali. Per evitare la ricompressione e garantire la riproducibilità, abbiamo testato l'integrità dei file con PIL e scaricato solo le prime **3.000 immagini reali sane (risoluzione $\ge$ 100x100px)** in modalità binaria pura (`wb`). Le corrispettive fake (SD1.4, SD2.1, SDXL, DeepFloyd) sono caricate direttamente dai Parquet a runtime.
* **Percorso Dati Reali:** `/work/cvcs2026/deep_pixels/datasets/D3/images/real/`
* **Percorso Parquet Fake:** `/work/cvcs2026/deep_pixels/datasets/D3/data/`
* **Manifest:** `/work/cvcs2026/deep_pixels/datasets/D3/manifest.csv`
  * **Intestazioni:** `path,index,label,generator,dataset`
  * **Reali (label=0):** `path` è il percorso assoluto sul disco, `index` è vuoto (es. `/work/.../real/0000.png,,0,real,d3`).
  * **Fake (label=1):** `path` è vuoto, `index` è l'indice di riga del Parquet (es. `,4786,1,sdxl,d3`).

---

### 3.2 Dataset GAN (ForenSynths) — bilanciato su disco (6.000 righe)

* **Metodologia:** Per non sforare lo spazio su disco, abbiamo estratto selettivamente solo le immagini per i generatori `progan`, `stylegan` e `cyclegan` (1.000 reali e 1.000 fake per ciascuno) dal file ZIP originale. I file sono stati estratti nel loro formato **PNG nativo** (nessun re-encoding JPEG o compressione lossy).
* **Percorso Immagini:** `/work/cvcs2026/deep_pixels/datasets/GAN/CNN_synth_testset/`
* **Manifest:** `/work/cvcs2026/deep_pixels/datasets/GAN/manifest.csv`
  * **Intestazioni:** `path,label,generator,dataset` (allineato alle colonne comuni, `index` non presente).
  * **Righe (es.):** `/work/cvcs2026/deep_pixels/datasets/GAN/CNN_synth_testset/cyclegan/apple/0_real/n07740461_10011.png,0,cyclegan,forensynth`

---

### 3.3 Dataset OpenFake — basato su Parquet (91.398 righe)

* **Metodologia:** Il test set completo di OpenFake core pesa 64 GB. Per risparmiare inode e spazio, le immagini non sono estratte su disco. Il manifest associa ogni riga all'indice sequenziale del dataset Parquet.
* **Percorso Parquet:** `/work/cvcs2026/deep_pixels/datasets/OpenFake/test_set/core/`
* **Manifest:** `/work/cvcs2026/deep_pixels/datasets/OpenFake/manifest.csv`
  * **Intestazioni:** `index,label,generator,dataset`
  * **Righe (es.):**
    * Reale: `91382,0,imagenet,openfake`
    * Fake: `91383,1,gpt-image-1.5,openfake`

---

## 4. GUIDA ALL'IMPLEMENTAZIONE DEL DATALOADER (PYTORCH)

Questa sezione fornisce un template di riferimento in Python per caricare i tre dataset in modo unificato all'interno della pipeline di valutazione. Gestisce in modo trasparente il caricamento da file fisici (GAN e reali D3) e da file Parquet (OpenFake e fake D3).

### Codice di riferimento per `dataset_loader.py`

```python
# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import pyarrow.parquet as pq

class UnifiedDeepfakeDataset(Dataset):
    def __init__(self, manifest_path, openfake_parquet_dir=None, d3_parquet_dir=None, transform=None):
        """
        Dataloader unificato per la valutazione dei detector di deepfake.
        """
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        
        # Inizializziamo i lettori Parquet solo se necessario
        self.openfake_parquet_dir = openfake_parquet_dir
        self.d3_parquet_dir = d3_parquet_dir
        
        self.openfake_dataset = None
        self.d3_dataset = None

    def _get_openfake_dataset(self):
        # Caricamento lazy per evitare di sovraccaricare la memoria dei thread secondari
        import datasets
        if self.openfake_dataset is None:
            # Carica la directory contenente i 13 file parquet
            self.openfake_dataset = datasets.load_dataset(
                "parquet", 
                data_files=os.path.join(self.openfake_parquet_dir, "*.parquet"), 
                split="train"  # I parquet di test_set/core sono etichettati come train di default da HF se non diversamente specificato
            )
        return self.openfake_dataset

    def _get_d3_dataset(self):
        import datasets
        if self.d3_dataset is None:
            self.d3_dataset = datasets.load_dataset(
                "parquet", 
                data_files=os.path.join(self.d3_parquet_dir, "*.parquet"), 
                split="train"
            )
        return self.d3_dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset = row["dataset"]
        label = int(row["label"])
        generator = row["generator"]
        
        # 1. CARICAMENTO IMMAGINE
        img = None
        
        if dataset == "forensynth":
            # GAN: Caricamento standard da file PNG
            img = Image.open(row["path"]).convert("RGB")
            
        elif dataset == "d3":
            if label == 0:
                # D3 Reale: Caricamento da file immagine (LAION)
                img = Image.open(row["path"]).convert("RGB")
            else:
                # D3 Fake: Caricamento lazy dal Parquet locale
                d3_ds = self._get_d3_dataset()
                parquet_idx = int(row["index"])
                
                # Mappatura delle colonne dei generatori nel Parquet D3
                # gen0 = deepfloyd | gen1 = sd14 | gen2 = sd21 | gen3 = sdxl
                gen_col_map = {"deepfloyd": "image_gen0", "sd14": "image_gen1", "sd21": "image_gen2", "sdxl": "image_gen3"}
                col_name = gen_col_map[generator]
                
                # Estraiamo i byte grezzi
                raw_data = d3_ds[parquet_idx][col_name]
                if isinstance(raw_data, dict):
                    raw_data = raw_data.get("bytes")
                
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                
        elif dataset == "openfake":
            # OpenFake (reals e fakes): Caricamento dal dataset Parquet
            openfake_ds = self._get_openfake_dataset()
            parquet_idx = int(row["index"])
            
            raw_data = openfake_ds[parquet_idx]["image"]
            if isinstance(raw_data, dict):
                raw_data = raw_data.get("bytes")
                
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # 2. APPLICAZIONE TRASFORMAZIONI (Preprocessing)
        if self.transform:
            img = self.transform(img)
            
        return img, label, generator, dataset
```

---

## 5. LOG DELLE DECISIONI & PROGRESSI

* **2026-06-25 — Setup:** Repo creata. Scelta detector (CoDE, UFD, Effort). PyCharm remoto e venv configurati.
* **2026-06-29 — Download D3:** Scaricati gli 11 parquet di validation split per evitare data leakage su CoDE.
* **2026-06-29 — Primo D3 subset (Superato):** `build_subset.py` ha estratto 19.200 file JPEG. Approccio superato per via del domain bias introdotto dalla ricompressione lossy JPEG.
* **2026-06-30 — Download GAN:** Scaricato `CNN_synth_testset.zip` via HF. Scompattati i generatori target in formato PNG originale (no JPEG bias). Spazio totale: **628 MB** (6.000 file).
* **2026-06-30 — OpenFake Manifest:** Allineato il manifest del test set basato su indici Parquet. Spazio totale: **64 GB** (13 file parquet).
* **2026-06-30 — Ripristino D3 Lossless:** 
  * Eliminate le vecchie JPEG.
  * Lanciato `check_d3_urls.py`: identificati 3.515 URL vivi escludendo i file non conformi (corrotti, 1x1 pixel).
  * Lanciato `download_d3_reals_secure.py`: scaricate in binario puro esattamente **3.000 immagini reali** ed eliminati i file orfani generati in background dai thread.
  * Generato `manifest.csv` D3 bilanciato a 15.000 righe totali. Spazio totale: **5.2 GB**.
