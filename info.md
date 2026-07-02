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

### Setup & Dati
* [x] Repo GitHub creata e clonata (locale + home sul server)
* [x] Cartelle condivise create su `/work`: `datasets/`, `weights/`, `results/`
* [x] PyCharm remote (SSH interpreter su venv + Deployment) e configurazione venv
* [x] **GAN (ForenSynths):** Estrazione selettiva delle PNG native + scrittura `manifest.csv` (6.000 righe)
* [x] **OpenFake:** Scaricato lo split test in Parquet (68 GB) + scrittura `manifest.csv` (91.398 righe)
* [x] **D3:** Screening delle URL reali, identificati 3.500 prompt-matched vivi, download lossless di 3.000 reali, e generazione `manifest.csv` bilanciato (15.000 righe)
* [x] Implementare la classe dataloader unificata `UnifiedDeepfakeDataset` in `src/data/dataset.py`

### Integrazione e Valutazione dei Detector
* [x] **UniversalFakeDetect (UFD):** Clonazione, configurazione pesi `fc_weights.pth`, unificazione in `evaluate_ufd.py` ed esecuzione su GAN, D3 e OpenFake via SLURM
* [x] **CNNDetection:** Clonazione, download dei pesi ufficiali `blur_jpg_prob0.5.pth`, unificazione in `evaluate_cnndet.py` ed esecuzione su GAN, D3 e OpenFake via SLURM
* [x] **AIDE:** Clonazione, download dei due pesi per l'ablation study (`aide_progan.pth` e `aide_genimage.pth`), unificazione in `evaluate_aide.py` ed esecuzione su GAN, D3 e OpenFake via SLURM (in corso)
* [x] Risoluzione del bug di caricamento delle immagini PIL da Parquet e del deadlock PyArrow/PyTorch tramite configurazione `spawn` e limitazione dei thread

### Analisi e Reportistica
* [ ] Eseguire `aggregate_results.py` per compilare le tabelle finali una volta terminato AIDE
* [ ] Generazione t-SNE / UMAP degli embedding dei modelli su un subset bilanciato
* [ ] Esecuzione dei test di robustezza JPEG/Resize
* [ ] Analisi qualitativa dei failure mode ed estrazione delle immagini rappresentative

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

## 3.5. DETECTOR — DETTAGLI TECNICI E ARCHITETTURE

Per questo benchmark confrontiamo le capacità di generalizzazione cross-generator di rilevatori universali rappresentativi basati su filosofie diverse:

### A. UniversalFakeDetect (UFD)
* **Autori & Pubblicazione:** University of Wisconsin-Madison (Ojha et al.), CVPR 2023.
* **Architettura:** Backbone pre-addestrato **CLIP (ViT-L/14)** (con pesi congelati) accoppiato a una testa di classificazione lineare (**Linear Classifier**) addestrata ad hoc.
* **Allenamento:** Il classificatore lineare (`fc_weights.pth`) è stato addestrato esclusivamente sulle feature estratte dalle immagini reali e fake generate da **ProGAN** (20 modelli differenti).
* **Meccanismo:** Sfrutta lo spazio semantico latente di CLIP, già robusto e pre-addestrato su miliardi di coppie immagine-testo dal web. Rileva impronte digitali geometriche e anomalie locali ad alta frequenza tipiche delle reti convoluzionali (GAN). Generalizza egregiamente all'interno della famiglia delle GAN, ma si trova fuori calibrazione e viene ingannato sistematicamente dalle diffusion moderne che producono immagini molto "lisce" nel dominio delle frequenze (es. su OpenFake l'AUROC generale scende a 0.41).

### B. CoDE (Contrastive Deepfake Embeddings)
* **Autori & Pubblicazione:** AImageLab, UNIMORE (Ranieri et al.), ECCV 2024.
* **Architettura:** Rete neurale basata su **ViT-Tiny** (Vision Transformer Tiny), ottimizzata per bilanciare alta accuratezza e bassa impronta di memoria/alto throughput per uso pratico.
* **Allenamento:** Addestrato sul massivo dataset proprietario **D3** (9.2 milioni di immagini generate da 4 modelli a diffusione: Stable Diffusion 1.4, 2.1, SDXL, e DeepFloyd).
* **Meccanismo:** Utilizza una funzione di perdita contrastiva (Contrastive Loss) abbinata a una modellizzazione esplicita della **similarità globale-locale** (global-local similarities). A differenza di CLIP, si concentra sui dettagli strutturali intrinseci e sulle incongruenze spettrali delle generazioni text-to-image a diffusione. Mostra una robustezza generale migliore sui modelli a diffusione (AUROC 0.62 su OpenFake), ma subisce forti inversioni prestazionali di fronte a generatori fotorealistici di ultimissima generazione non visti in fase di training (come Flux.2).

### C. CNNDetection (Wang 2020)
* **Autori & Pubblicazione:** Adobe Research & UC Berkeley (Wang et al.), CVPR 2020.
* **Architettura:** Rete convoluzionale standard **ResNet-50** modificata (rimozione del primo downsampling/maxpooling per conservare informazioni spettrali ad alta frequenza).
* **Allenamento:** Addestrata esclusivamente su immagini reali e fake generate da **ProGAN** (20 categorie differenti) applicando forti augmentazioni di sfocatura e compressione JPEG (`blur_jpg_limit4.pth`).
* **Meccanismo:** Rileva le impronte digitali geometriche e spettrali (griglie di campionamento) lasciate dal generatore ProGAN ad alta frequenza. Serve come baseline CNN classica del benchmark. Il confronto diretto con UFD (anch'esso addestrato solo su ProGAN ma basato su CLIP) ci consente di isolare il ruolo del tipo di backbone e del pre-training semantico a parità di dati di addestramento.

### D. AIDE (AI-generated Image DEtector)
* **Autori & Pubblicazione:** Yan et al. (ICLR 2025). Paper: "A Sanity Check for AI-generated Image Detection".
* **Architettura:** Rete ibrida multi-esperto composta da:
  - **Frequency Expert:** Due ResNet-50 custom che ricevono in input 30 filtri SRM (Steganalysis Rich Model) high-pass al posto dei 3 canali RGB. Processa 4 view DCT della stessa immagine (le 2 patch con frequenza minima e massima) e ne media le feature (2048-dim).
  - **Semantic Expert:** ConvNeXt-XXL pre-addestrato via `open_clip` (parametri congelati), con avg-pool + proiezione lineare a 256 dimensioni.
  - **Fusione:** Concatenazione (2048 + 256 = 2304 dim) → MLP → 2 classi (real/fake).
* **Input al modello:** Tensore di shape `[B, 5, C, H, W]` — 5 view della stessa immagine (4 view DCT + 1 originale normalizzata).
* **Score di output:** `softmax(logits)[:, 1]` → probabilità della classe "fake".
* **Checkpoint utilizzati per il benchmark (analisi a matrice architettura × dati):**
  1. **AIDE-ProGAN** (`aide_progan.pth`): Addestrato su ProGAN (stesso training set antico di CNNDetection e UFD). Utilizzato per confrontare l'efficacia del design delle reti a parità di dati di addestramento (**asse architettura**).
  2. **AIDE-GenImage** (`aide_genimage.pth`): Addestrato su GenImage (multi-generatore moderno). Utilizzato per valutare l'impatto dell'aggiornamento dei dati di training a parità di architettura (**asse dati**).

---

## 3.6. DOWNLOAD DEI PESI E SETUP DEI DETECTOR AGGIUNTIVI

Per consentire l'esecuzione di CNNDetection e AIDE sul cluster, documentiamo qui le istruzioni esatte per reperire i pesi pre-addestrati e configurare i file:

### A. CNNDetection (Wang 2020)
* **Download dei pesi tramite script ufficiale:**
  Spostati nella cartella di CNNDetection ed esegui lo script ufficiale fornito dagli autori per scaricare i pesi via Dropbox:
  ```bash
  cd CNNDetection
  bash weights/download_weights.sh
  ```
* **Spostamento dei pesi su /work:**
  Copia i pesi scaricati nella cartella dei pesi di `/work` per l'inferenza del benchmark:
  ```bash
  mkdir -p /work/cvcs2026/deep_pixels/weights/cnndet
  cp weights/blur_jpg_prob0.5.pth /work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth
  ```
* **Esecuzione inferenza sul benchmark:**
  ```bash
  # Valutazione di UFD (su tutti i 3 dataset)
  sbatch scripts/run_ufd_eval.sh

  # Valutazione di CNNDetection (su tutti i 3 dataset)
  sbatch scripts/run_cnndet_eval.sh

  # Valutazione di AIDE (in parallelo su 2 GPU per ProGAN e GenImage)
  sbatch scripts/run_aide_progan.sh
  sbatch scripts/run_aide_genimage.sh
  ```
* **Caricamento in PyTorch:**
  Il modello viene caricato istanziando una ResNet-50 standard di `torchvision` con `num_classes=1` e caricando lo state dict:
  ```python
  from torchvision.models import resnet50
  model = resnet50(num_classes=1)
  state_dict = torch.load("/work/cvcs2026/deep_pixels/weights/cnndet/blur_jpg_prob0.5.pth", map_location="cpu")
  model.load_state_dict(state_dict)
  ```

### B. AIDE (Yan et al. - ICLR 2025)
* **Download repository:**
  Clona la repository ufficiale del modello sul server:
  ```bash
  git clone https://github.com/shilinyan99/AIDE.git
  ```
  *(La repo viene clonata all'interno del progetto ed è già inserita in `.gitignore` per evitare di tracciarla).*
* **Download dei pesi pre-addestrati tramite gdown:**
  I checkpoint ufficiali degli autori sono ospitati su Google Drive. Scaricali sul server (dal login node, che ha accesso a internet) usando `gdown`:
  ```bash
  pip install gdown
  mkdir -p /work/cvcs2026/deep_pixels/weights/aide
  
  # Scarica i checkpoint nella cartella temporanea
  gdown --folder https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4 -O /tmp/aide_weights
  
  # Copia e rinomina i due checkpoint di interesse
  cp /tmp/aide_weights/progan_train.pth /work/cvcs2026/deep_pixels/weights/aide/aide_progan.pth
  cp /tmp/aide_weights/GenImage_train.pth /work/cvcs2026/deep_pixels/weights/aide/aide_genimage.pth
  ```
* **PRE-CACHING DI CONVNEXT (IMPORTANTE PER NODI GPU SENZA INTERNET):**
  Dato che i nodi GPU del cluster non hanno accesso a internet e AIDE scarica dinamicamente il modello ConvNeXt-XXL di OpenCLIP (~3.5 GB), **devi lanciare questo comando sul login node (con internet abilitato) prima di inviare i job SLURM**:
  ```bash
  python -c "import open_clip; open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')"
  ```
  Questo scaricherà i pesi nella cache globale condivisa (NFS) accessibile anche dai nodi offline.
* **Dipendenze aggiuntive:**
  AIDE richiede `open_clip`, `kornia`, e `timm`. Installali nel venv:
  ```bash
  pip install open-clip-torch kornia timm
  ```

---

## 4. GUIDA ALL'IMPLEMENTAZIONE DEL DATALOADER (PYTORCH)

La classe dataset unificata è stata implementata e si trova nel file sorgente **[`src/data/dataset.py`](file:///Users/barack/Desktop/Computer_Vision/Lab/code/cvcs-deepfake-benchmark/src/data/dataset.py)**. 

### Utilizzo del DataLoader
Il dataloader gestisce in modo trasparente il caricamento da file fisici (GAN e reali D3) e da file Parquet (OpenFake e fake D3). 
* **Input `idx`:** Il parametro `idx` in `__getitem__(self, idx)` corrisponde all'indice di riga del file **`manifest.csv` unificato** (da `0` a `N-1`).
* **Funzionamento:** A runtime, il dataset legge la riga `idx` del manifest, rileva la sorgente (`forensynth`, `d3`, o `openfake`), recupera il percorso dell'immagine (se fisica) o l'indice di riga originario del file Parquet (se tabellare), carica l'immagine convertendola in RGB. Gestisce in modo robusto le immagini a palette indicizzate con trasparenza per evitare errori. Restituisce infine la tupla: `(img, label, generator, dataset)`.

---

## 5. PIANO DI ANALISI QUALITATIVA E QUANTITATIVA

Il benchmark mira a valutare la generalizzazione cross-generator lungo assi complementari:

### 5.1. Analisi Quantitativa (Metriche Globali e per Generatore)
I risultati dell'inferenza vengono aggregati tramite `scripts/aggregate_results.py`. Per ogni modello valutato su ciascuno dei tre dataset (GAN, D3, OpenFake), vengono calcolate le seguenti metriche:
- **AUROC (Area Under the ROC Curve):** Misura la capacità di discriminazione complessiva indipendente dalla soglia.
- **AP (Average Precision):** Indicatore robusto per lo sbilanciamento delle classi.
- **Accuracy (con soglia fissa a 0.5):** Valutazione dell'accuratezza in scenari reali standard.
- **Accuracy (con soglia ottimale EER):** Accuratezza calcolata alla soglia ottimale in cui il tasso di falsi positivi (FPR) equivale al tasso di falsi negativi (FNR).
- **Breakdown disaggregato per singolo generatore:** Le metriche vengono scorporate e calcolate separatamente per ciascun generatore (es. SD1.4, SD2.1, SDXL, DeepFloyd, Flux, Midjourney, ecc.) sfruttando la colonna `generator` all'interno dei manifest.

### 5.2. Visualizzazione degli Spazi Latenti (t-SNE / UMAP)
Per analizzare e interpretare graficamente lo spazio delle caratteristiche appreso dai vari modelli, estraiamo gli embedding (le feature penultime) su un **subset bilanciato** del dataset (es. 1.000 reali e 1.000 fake divise equamente tra generatori storici e moderni).
- Gli embedding vengono proiettati in 2D tramite **t-SNE** o **UMAP**.
- Consente di verificare visivamente se il detector ha creato un confine di decisione netto e se le diffusion moderne si mescolano interamente con le immagini reali nello spazio del modello (indicando un fallimento di generalizzazione semantica/spettrale).

### 5.3. Test di Robustezza (JPEG & Resize)
Valutiamo la sensibilità dei detector alle manipolazioni post-processing tipiche del trasferimento su social media. Su un subset bilanciato di **OpenFake** (1.000 reali, 1.000 fake), applichiamo in-memory a runtime le seguenti trasformazioni:
1. **Clean (Originale):** Baseline di controllo non degradata.
2. **JPEG Compression (Pillow):** Compressione a qualità $Q=70$ (standard social) e $Q=50$ (forte degradazione).
3. **Resize (Downsampling):** Ridimensionamento bilineare a $128 \times 128$ pixel e successivo ri-upscaling alla risoluzione nativa del modello ($224 \times 224$).
- Il confronto diretto delle metriche (AUROC/AP) tra le versioni "Clean" e "Transformed" sullo stesso identico subset di immagini consente di misurare la stabilità dei detector.

### 5.4. Analisi Qualitativa dei Failure Modes (Casi Discordanti)
Per interpretare meglio i numeri quantitativi, conduciamo un'analisi visiva identificando immagini in cui i detector mostrano comportamenti contrastanti o fallimenti sistematici:
- **Pairwise Discordant Cases (es. CoDE corretto vs UFD errato):** Individuazione di immagini fake generate da modelli moderni (es. Flux o SDXL) che eludono UFD (che le classifica come reali con confidenza estrema per via dell'assenza di rumore ad alta frequenza tipico di ProGAN) ma che vengono correttamente smascherate da CoDE.
- **UFD corretto vs CoDE errato:** Immagini in cui le caratteristiche semantiche globali di CLIP (UFD) catturano dettagli ignorati dai pattern di diffusione locali (CoDE).
- **Fallimenti Unanimi (Confidenza Estrema):** Analisi visiva dei fake "perfetti" che ingannano tutti i detector contemporaneamente, per comprendere quali strutture visive o semantiche siano le più critiche.

---

### 5.5. MATRICE DEI DETECTOR (RIASSUNTO METADATI PER LA RIPRODUCIBILITÀ)

| Detector | Dati di Training Originari | Dataset di Test | Generatori Valutati | Protocollo di Preprocessing |
|---|---|---|---|---|
| **UFD** (Ojha 2023) | Reali + Fake ProGAN (20 classi) | GAN, D3, OpenFake | Tutti i generatori dei 3 dataset | CLIP standard: Resize 224, CenterCrop 224, Normalizzazione CLIP. |
| **CoDE** (Ranieri 2024) | Reali + Fake D3 (SD1.4, SD2.1, SDXL, DeepFloyd) | GAN, D3, OpenFake | Tutti i generatori dei 3 dataset | ViT standard: Resize 224, CenterCrop 224, Normalizzazione ImageNet. |
| **CNNDetection** (Wang 2020) | Reali + Fake ProGAN (20 classi con sfocatura/JPEG 50%) | GAN, D3, OpenFake | Tutti i generatori dei 3 dataset | CNN standard: Resize 256, CenterCrop 224, Normalizzazione ImageNet. |
| **AIDE-ProGAN** (Yan 2025) | Reali + Fake ProGAN | GAN, D3, OpenFake | Tutti i generatori dei 3 dataset | Multi-Expert: Decomposizione DCT in-memory → 5-view tensor `[5, C, 256, 256]` (4 view SRM + 1 originale normalizzata). |
| **AIDE-GenImage** (Yan 2025) | Reali + Fake GenImage (Multi-generatore moderno) | GAN, D3, OpenFake | Tutti i generatori dei 3 dataset | Multi-Expert: Preprocessing identico ad AIDE-ProGAN. |

---

## 6. LOG DELLE DECISIONI & PROGRESSI

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
* **2026-06-30 — Setup UniversalFakeDetect (UFD):**
  * Creato `src/data/dataset.py` per isolare il caricamento dati.
  * Aggiunte le dipendenze di CLIP (`ftfy`, `regex`, `tqdm`, `git+CLIP`) in `requirements.txt`.
  * Clonato UFD in `UniversalFakeDetect/` (ignorato in `.gitignore`).
  * Scaricati i pesi `fc_weights.pth` su `/work/.../weights/ufd/`.
  * Eseguito con successo `scripts/test_ufd_setup.py` (inferenza mock OK).
* **2026-07-01 — Integrazione AIDE e CNNDetection:**
  * Creato lo script unificato `scripts/evaluate_cnndet.py` per CNNDetection ed eseguiti con successo i test su GAN e D3.
  * Creato lo script unificato `scripts/evaluate_aide.py` per AIDE configurato con il metodo di multiprocessing `spawn` per prevenire i deadlock tra PyTorch DataLoader e PyArrow I/O durante la lettura dei file Parquet.
  * Configurato l'ablation study di AIDE (ProGAN vs GenImage) e scaricati i pesi corrispondenti (~6.8 GB totali) archiviandoli sotto `/work/.../weights/aide/`.
* **2026-07-02 — Esecuzione Benchmark:**
  * Completata con successo la valutazione di **UFD** su tutti e 3 i dataset (GAN, D3, OpenFake) e di **CNNDetection** su tutti e 3 i dataset.
  * Riorganizzati e unificati tutti i file di script per UFD, lasciando il repository estremamente pulito ed ordinato.
  * Avviate in parallelo le esecuzioni di **AIDE-ProGAN** e **AIDE-GenImage** su OpenFake sfruttando la pre-elaborazione parallelizzata a 4 worker.


