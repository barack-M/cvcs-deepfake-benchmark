# INFO — Progetto CVCS Deepfake Detection

> File di lavoro personale. Qui dentro raccogliamo TUTTO sul PROGETTO:
> cose da fare, decisioni prese, dataset, protocollo, detector, metriche.
> Aggiornare man mano. Quando una cosa è "decisa e stabile", spostarla nel README ufficiale.
>
> 👉 Per tutto ciò che riguarda il **funzionamento del SERVER** (percorsi, quote, comandi, SLURM,
> cosa NON fare) vedi il file separato **`info_server.md`**.

**Progetto:** *Deepfake Detection in the Wild: Cross-Generator Generalization Benchmarking*
**Corso:** Computer Vision and Cognitive Systems (CVCS) 2025/2026, UNIMORE — prof. Lorenzo Baraldi
**Gruppo:** Deep Pixels — Matteo Baracchi + Enrico Ranieri
**Tutor di riferimento:** Silvia Cappelletti (silvia.cappelletti@unimore.it), Tobia Poppi (tobia.poppi@unimore.it)
**Repo GitHub:** https://github.com/barack-M/cvcs-deepfake-benchmark

---

## 0. L'idea in una frase

Un detector di deepfake addestrato sui generatori di *oggi* funziona ancora sui generatori di *domani*?
Prendiamo detector pre-addestrati, li testiamo **cross-generator** (su famiglie di generatori mai viste in training)
e analizziamo **quando, perché e quanto** falliscono. Non proponiamo un metodo nuovo: produciamo
un'**analisi rigorosa e riproducibile** dei punti di forza e dei *failure mode*.

Insight chiave già emerso: **CoDE si ferma a Stable Diffusion XL** — tutto ciò che è più recente
(FLUX, Midjourney v6/v7, DALL-E 3, ...) tende a sfuggirgli. Documentare e spiegare *perché* è il cuore del progetto.

---

## 1. WORKFLOW — come lavoriamo

### Dove sta cosa
- **Codice + venv** → nella propria HOME sul server: `/homes/<username>/cvcs2026/...` (NFS gestisce bene tanti file piccoli)
- **Dataset, pesi, checkpoint, feature, risultati** → `/work/cvcs2026/deep_pixels/...` (BeeGFS, alta banda)
  - `datasets/` → i dati (file parquet di D3, subset OpenFake, DF40)
  - `weights/` → i `.pth` pre-addestrati dei detector
  - (creeremo) `features/`, `results/` per output ed embedding
- **Regola filesystem:** usare sempre PATH ASSOLUTI (niente `$HOME`/`$WORK` su questo cluster).
- **BeeGFS:** evitare milioni di file piccoli → tenere i dati come parquet/archivi/LMDB, non immagini sparse.

### Ciclo di sviluppo normale (con PyCharm)
1. Scrivo/modifico un file in PyCharm **in locale**
2. Al salvataggio, PyCharm Deployment **carica automaticamente** il file sul server via SFTP
3. Premo Run in PyCharm → esegue il codice sul server usando il venv remoto
4. I dati NON passano da git: stanno già su `/work`, il codice li legge via path assoluto

**Git è separato da PyCharm Deployment** — sono due sistemi indipendenti:
- PyCharm Deployment: sincronizza i file **in tempo reale** tra locale e server (non usa git)
- Git: serve per condividere il codice con **Enrico** e per avere la storia del progetto
- Un `git push` da locale mette il codice su GitHub, ma **NON aggiorna il server automaticamente**
- Per aggiornare il server via git (es. dopo che Enrico ha pushato): `git pull` da terminale sul server

**Ciclo con Enrico:**
- Lavoro in locale → PyCharm carica → eseguo sul server (tutto automatico)
- Quando voglio condividere con Enrico: `git add`, `git commit`, `git push` da locale
- Enrico fa `git pull` sul suo server per ricevere le mie modifiche

> Il repo contiene **solo codice e script**. Dati/pesi/output sono esclusi dal `.gitignore`.

### Struttura del codice (verso cui andiamo)
```
src/        # codice RIUTILIZZABILE importato dagli script (data/, models/, metrics/, utils/)
scripts/    # entry-point "sottili" che si LANCIANO, uno per compito
configs/    # config per esperimento (path, generatori, seed)
slurm/      # job script .sh per il cluster
requirements.txt   # librerie (su git; torch escluso, dipende da CUDA)
```
Principio: logica condivisa (es. "leggi parquet", "calcola metriche") sta in `src/`, MAI copiata negli
script. Gli script in `scripts/` leggono argomenti → chiamano `src/` → salvano output.
File esistenti: `scripts/explore_hf_dataset.py` (ispeziona un dataset HF in streaming, parametrico per repo).

### PyCharm remote (configurato)
- **SSH Interpreter**: usa il Python del venv sul server (`/homes/mbaracchi/cvcs2026/.../venv/bin/python`)
- **Deployment (SFTP)**: auto-upload a ogni salvataggio verso `/homes/mbaracchi/cvcs2026/cvcs-deepfake-benchmark`
- **Terminale integrato**: View → Tool Windows → Terminal → apre una SSH shell sul server (per installare librerie, creare cartelle, ecc. senza uscire da PyCharm)

**Python:** il cluster ha anaconda installato da admin con tutti i pacchetti base (datasets, pandas, ecc.).
Nessun venv necessario. Pacchetti extra: `pip install --user nome`.
PyCharm interpreter: punta al Python di sistema (trovare path con `which python` sul server).
NON usare il tab "Python Packages" di PyCharm per i venv remoti — funziona male.

### Server: accesso, SLURM, quota, comandi
→ Tutto in **`info_server.md`**. Promemoria essenziali:
- ⚠️ **`/work` del corso al 92%** (4 TB condivisi tra tutti i gruppi) → verificare spazio prima di scaricare.
- Codice leggero (ispezione parquet, test su poche immagini) → ok sul login node.
- Inferenza GPU su dataset interi → serve **SLURM** (nodo di calcolo).
- Niente `sudo`/`apt`; `module load git`; venv in `/homes`, dati in `/work`.

---

## 2. TODO — lista cose da fare (aggiornare nel tempo)

Legenda: [ ] da fare · [~] in corso · [x] fatto

### Setup
- [x] Creare repo GitHub e clonarla in locale
- [x] Chiarire con tutor accesso ai dataset e organizzazione filesystem
- [x] Clonare repo nelle home di Matteo ed Enrico sul server
- [ ] Creare venv in home + installare requisiti base (torch, torchvision, huggingface_hub, datasets, scikit-learn, matplotlib, pandas, pyarrow)
- [ ] Creare cartelle su /work: `datasets/`, `weights/`, `features/`, `results/`
- [ ] Clonare DeepfakeBench e/o CoDE nelle home (valutare quale framework usare — vedi §6)
- [x] Configurare PyCharm remote (SSH interpreter + deployment)

### Dati
- [ ] Scaricare 1 file parquet di D3 e ISPEZIONARLO (che colonne ha? c'è la label del generatore?)
- [ ] Scaricare subset D3 train (100k–200k record, con label generatore) per metriche per-generatore
- [ ] (Eventuale) scaricare D3 external_test per eval GLOBALE (no label per-generatore)
- [ ] Ispezionare OpenFake in streaming (`ex.keys()`: che campi? generatore? real/fake?)
- [ ] Costruire subset OpenFake test (OOD target), bilanciato per generatore e real/fake
- [ ] (Eventuale) selezionare 4–8 metodi DF40 via gdown
- [ ] Fissare e annotare il SEED di campionamento; salvare le liste di file/indici dei subset

### Detector
- [ ] Far girare CoDE in inferenza su un dataset intero (Enrico ha già bozza su poche immagini)
- [ ] Decidere uso di Xception: checkpoint pre-addestrato as-is vs re-train su D3 (vedi §5)
- [ ] Reperire pesi UniversalFakeDetect / CLIP-based
- [ ] Raccogliere i `.pth` in /work/.../weights e annotare provenienza/training set di ciascuno

### Evaluation & Analisi
- [ ] Definire pipeline: dato (detector, dataset-subset) → score per immagine → metriche
- [ ] Calcolare AUROC, Average Precision, Accuracy per ogni coppia detector × generatore
- [ ] Tabella riassuntiva per-generatore (righe = detector, colonne = generatori)
- [ ] t-SNE degli spazi delle feature
- [ ] Analisi failure mode: per architettura del generatore, per contenuto, sotto compressione social
- [ ] (Eventuale) test robustezza a compressione (es. ricompressione JPEG stile WhatsApp)

### Report
- [ ] Scrivere il report con protocollo, tabelle, grafici e analisi dei failure mode
- [ ] Dichiarare nel report TUTTI i subset usati (numerosità) e il seed (riproducibilità)

---

## 3. DATASET

### Principio guida (dal tutor)
> Scaricare/processare più dati possibile compatibilmente con spazio e tempi. Quando troppo grandi,
> costruire **subset** e lavorare su quelli, poi **dichiarare nel report quanti esempi** sono stati usati.
> Per campionamenti random: **fissare un SEED e riportarlo** → esperimento riproducibile.

D3 completo è ~2,6 TB → **impossibile** scaricarlo tutto (quota /work limitata). Si lavora a subset.

### 3.1 D3 (ELSA_D3) — distribuzione "in-distribution" / source
- **Cos'è:** dataset large-scale diffusion-focused di AImageLab UNIMORE (legato a ELSA EU). Coppie real/fake;
  le fake generate da modelli di diffusione. È il dataset su cui CoDE è stato sviluppato.
- **Accesso:** NON montabile in sola lettura sul cluster. Si scarica da HuggingFace.
  - Train/reference: https://huggingface.co/datasets/elsaEU/ELSA_D3 → cartella `data/` con file **parquet**
  - Test ufficiale: https://huggingface.co/datasets/elsaEU/ELSA_D3_external_test
- **Formato:** parquet (un file impacchetta molte immagini + metadati → ottimo per BeeGFS).
- **IMPORTANTE — label dei generatori:**
  - `ELSA_D3_external_test`: 12 generatori (inclusi i 4 di train/val) **MA senza label del generatore** nella
    versione pubblica → utilizzabile **solo per eval GLOBALE** (real vs fake aggregato), NON per-generatore.
  - Per **metriche per-generatore** su D3: ritagliare un subset **dal TRAIN** (`elsaEU/ELSA_D3`), dove il
    generatore È indicato. Bastano **5k–20k coppie per generatore** (la tesi della tutor usava ~483k, a noi
    serve soprattutto un protocollo chiaro e riproducibile).
- **Quantità consigliata:** 100k–200k record per train/reference, scaricando solo alcuni parquet.
- **Comando esempio (dal tutor):**
  ```bash
  mkdir -p /work/cvcs2026/deep_pixels/datasets/D3
  huggingface-cli download elsaEU/ELSA_D3 \
    --repo-type dataset \
    --include "data/train-00000-of-*.parquet" \
    --include "data/train-00001-of-*.parquet" \
    --include "data/validation-*.parquet" \
    --local-dir /work/cvcs2026/deep_pixels/datasets/D3
  ```
  > TODO: prima scaricare UN solo parquet e ispezionare le colonne (capire dove sta la label del generatore).

### 3.2 OpenFake — distribuzione OUT-OF-DISTRIBUTION moderna / target  ⭐ DATASET PRINCIPALE
- **Cos'è:** dataset 2025 (arXiv 2509.09495) con generatori recenti/proprietari → **target OOD** ideale.
- **Accesso:** https://huggingface.co/datasets/ComplexDataLab/OpenFake — **3.44 TB**, 2.46M righe → SOLO streaming/subset.
- **Schema (verificato via dataset-viewer API):** `image`, `label` (real/fake), `model` (generatore),
  `type` (base/image/video/real), `prompt`, `release_date`. → Ha TUTTO: label binaria + label generatore. ✅
- **Config / split:** config `core` (train 2.31M / validation 59K / test 91K) e config `reddit` (test 36K in-the-wild).
- **`core/test` (il nostro target OOD):** label bilanciate ~50/50 real/fake. Reali da `imagenet`/`docci`.
  Generatori fake 2025-2026 (MAI visti dai detector pre-addestrati):
  - immagini: flux.2-klein, z-image-turbo, gpt-image-1.5/2, nano-banana-pro, midjourney-7, ideogram-2.0,
    recraft-v3, seedream-v5, illustrious (finetune SDXL), ...
  - VIDEO (frame estratti): sora-2, veo-3, wan-video-2.5 → valutare se escluderli o analizzarli a parte.
- **⚠️ Nota bias reali:** le reali sono imagenet/docci → un detector potrebbe distinguere per *stile dataset*
  invece che per artefatti di generazione. Da discutere nel report.
- **Come ispezionare (istantaneo, no download):**
  `python scripts/explore_hf_dataset.py --repo ComplexDataLab/OpenFake --config core --split test --api`
- **Modalità consigliata: STREAMING** (non scaricare tutto):
  ```python
  from datasets import load_dataset
  SPLIT = "test"   # oppure "train" / "validation"
  ds = load_dataset("ComplexDataLab/OpenFake", split=SPLIT, streaming=True)
  for ex in ds:
      print(ex.keys())   # ispezionare: label, generatore, immagine
      break
  # iterare e salvare solo gli esempi che servono
  ```
- **Quanto usarne:**
  - Se OpenFake = source/training/reference → 100k–200k esempi dallo split train.
  - Se OpenFake = solo target OOD (nostro caso più probabile) → split **test**, idealmente completo;
    se troppo grande, **subset fisso** 50k–100k esempi, **bilanciato per generatore e real/fake**.

### 3.3 DF40 — opzionale
- **Cos'è:** dataset NeurIPS 2024, 40 tecniche di deepfake (face swap, reenactment, entire-face-synthesis,
  face editing). Già **supportato da DeepfakeBench**.
- **Accesso:** https://github.com/YZY-stack/DF40 (immagini su Google Drive per metodo).
- **Strategia (dal tutor):** NON scaricare tutto. Selezionare ~4–8 metodi fake: 2 face-swap, 2 reenactment,
  2 entire-face-synthesis, 1–2 editing. ~1k–5k immagini fake per metodo + pari numero di real.
  Download per-metodo: `gdown --folder <link_cartella_drive_del_metodo>`.
- **Nota:** DF40 è face-centrico (deepfake di volti), mentre D3/OpenFake sono immagini sintetiche full-frame.
  Da valutare se inserirlo o tenerlo come estensione.

### Subset & riproducibilità (regola trasversale)
- Ogni subset: **seed fisso**, salvare la **lista di file/indici** selezionati (così è ricostruibile).
- Bilanciare real/fake e, dove possibile, numerosità per-generatore.
- Annotare qui la numerosità finale di ogni subset usato.

---

## 4. PROTOCOLLO sperimentale (deve essere CHIARO e RIPRODUCIBILE)

> Bozza — da raffinare. L'obiettivo è che chiunque, leggendo questa sezione, possa rifare gli esperimenti.

### Domanda di ricerca
Quanto degrada la detection quando si passa da generatori "visti" (GAN/early diffusion) a generatori
moderni/proprietari mai visti in training? Quali detector generalizzano meglio e perché? Dove falliscono?

### Setup cross-generator — DESIGN (allineato a tutor + testo progetto)
**Insight:** i nostri detector sono GIÀ pre-addestrati altrove (CoDE→D3, Xception→FF++, UniversalFakeDetect→GAN).
Non li addestriamo noi. La tutor ha indicato: se OpenFake è SOLO target OOD → usare SOLO lo split **test**
(idealmente completo; se troppo grande, subset fisso 50k–100k, bilanciato per generatore e real/fake).

| Ruolo | Fonte | Generatori |
|---|---|---|
| In-distribution / "vecchi" | **D3** (training di CoDE) | DeepFloyd, SD1.4, SD2.1, SDXL |
| OOD / "nuovi" | **OpenFake `core/test`** | flux.2, gpt-image-2, midjourney-7, z-image, ... |

- OpenFake `core/test` HA le label di generatore (colonna `model`) → analisi per-generatore OK lì.
- D3 dà l'estremo "vecchio" + il soffitto in-distribution di CoDE.
- Per ogni detector → **AUROC/AP per-generatore** → la curva mostra dove ogni detector crolla (vecchi→nuovi).
- (NB giro precedente: avevo proposto di usare train+test di OpenFake come unico pool; corretto: tutor dice solo test.)

### Raggruppamento generatori in FAMIGLIE/ERE (backbone dell'analisi)
100+ generatori singoli sono troppo granulari → raggruppare per famiglia/era:
- **GAN-based:** D3 = solo diffusion (confermato). OpenFake = nel campione letto solo diffusion+proprietari,
  niente GAN classici (verificare con grep completo della colonna `model` per sicurezza). DF40 HA metodi GAN
  (StyleGAN in Entire-Face-Synthesis e Face-Editing) MA nel dominio FACCE → diverso da immagini full-frame.
  → Per una famiglia GAN coerente (full-frame): fonte pulita = **ForenSynths / test set di UniversalFakeDetect**
  (ProGAN, StyleGAN, BigGAN su immagini generiche) — è anche il terreno di casa di UniversalFakeDetect. → DECIDERE.
- **Early diffusion:** SD 1.x, SD 2.x
- **Late diffusion / SDXL era:** SDXL e finetune, SD 3.5, Playground
- **Modern open:** FLUX family, z-image, HiDream, Qwen-Image, Chroma
- **Proprietary:** DALL-E 3, Imagen, Midjourney, GPT Image, Ideogram, nano-banana, Grok, Seedream
- **Video (a parte):** sora-2, veo-3, wan-video → frame estratti, caso diverso → valutare se escludere

### ⚠️ Caveat metodologici emersi dai dati
- **Sorgenti "real" diverse tra split:** train real = laion/pexels; test real = imagenet/docci. Se si mischiano,
  la classe "real" è eterogenea → un detector potrebbe sfruttare lo *stile del dataset* invece degli artefatti.
  Scegliere una sorgente real coerente o documentare la cosa.
- **Colonna `type` sporca:** valori con casing incoerente (base/Base, finetune/Fine-tune, lora/LoRA) → normalizzare.
- **D3 ruolo residuo:** OpenFake ha già sd-1.5/2.1/sdxl, quindi D3 forse non serve. Resta utile SOLO se vogliamo
  mostrare il soffitto in-distribution di CoDE sugli esatti 4 generatori del suo training. → DECIDERE se includerlo.

### ⚠️ D3 ha ruoli DIVERSI per detector diverso (data-leakage)
- **CoDE è addestrato su D3** → testarlo su D3 misura la performance *in-distribution* (soffitto), NON la
  generalizzazione. Testarlo sulle STESSE immagini di train = **data-leakage** (numeri gonfiati).
  → Per CoDE usare lo split **validation** di D3 (non visto in training) per le metriche per-generatore "vecchi".
  → Il risultato interessante per CoDE è il **crollo su OpenFake** (generatori veri non-visti).
- **Xception / UniversalFakeDetect NON sono addestrati su D3** → per loro D3 è già cross-domain/non-visto,
  nessun leakage: D3 è un test legittimo.
- **D3 external_test (senza label generatore):** dà solo un numero GLOBALE aggregato → utile come
  sanity-check, NON per l'analisi per-generatore. Bassa priorità.
- DECISIONE: usare lo split di D3 che porta la label del generatore (train vs validation → verificare con
  `scripts/explore_hf_dataset.py`); per CoDE preferire validation per ridurre il leakage.

### Cosa misuriamo
Per ogni **detector** e per ogni **generatore** (o famiglia): AUROC, Average Precision, Accuracy.
Più una vista **globale** (su tutto il test aggregato) e una **per-generatore** (per vedere dove crolla).

### Output attesi
1. Tabella riassuntiva: righe = detector, colonne = generatori/famiglie, celle = metrica.
2. Breakdown per-generatore (bar chart o heatmap).
3. t-SNE degli embedding (per vedere se real/fake sono separabili nello spazio delle feature, e come
   si posizionano i generatori non visti).
4. Analisi failure mode: per architettura del generatore, per tipo di contenuto, sotto compressione social.

### Riproducibilità — checklist
- [ ] Seed fissato e riportato
- [ ] Liste dei subset salvate (file/indici)
- [ ] Per ogni detector: documentato su quali dati è stato (pre-)addestrato e su quali è testato
- [ ] Criterio/soglia per l'Accuracy esplicitato (vedi §6)
- [ ] Versioni di dataset/modelli annotate

---

## 5. DETECTOR

Scegliamo 3 detector di tipologia/generazione diversa (selezione confermata sensata dal tutor).
Regola d'oro: per OGNI modello, avere sempre chiaro **su quali dati è (pre-)addestrato** e **su quali è testato**.

### 5.1 CoDE — modello principale (obbligatorio)
- **Cos'è:** *Contrasting Deepfakes Diffusion via Contrastive Learning and Global-Local Similarities*
  (Baraldi et al., ECCV 2024). Approccio a **embedding contrastivi** sviluppato da AImageLab UNIMORE.
- **Perché:** è il modello del laboratorio del corso; è il riferimento sul quale verte l'analisi.
- **Addestrato su:** D3 (diffusion, fino a ~SDXL). → Ci aspettiamo buona performance in-distribution e
  **crollo sui generatori moderni** (FLUX, Midjourney v7, ...). Questo è il fenomeno da documentare.
- **Codice:** https://github.com/aimagelab/CoDE — pesi pre-addestrati pubblici.
- Enrico ha già una bozza di inferenza su poche immagini → estendere a dataset interi.

### 5.2 Xception — baseline classica
- **Cos'è:** CNN standard, modello "storico" della deepfake detection. Presente in DeepfakeBench (categoria Naive).
- **Perché:** mostra come un detector classico/standard si comporta nel cross-generator (probabile crollo netto).
- **DECISIONE DA PRENDERE (dal tutor):**
  - (A) usare il **checkpoint DeepfakeBench pre-addestrato su face-deepfake** → baseline fortemente
    *out-of-domain* (è addestrata su volti, non su immagini sintetiche full-frame) — interessante ma molto OOD.
  - (B) **ri-allenare/fine-tunare** Xception sul nostro source (es. D3) → CNN baseline più *comparabile*
    a CoDE/CLIP, nel protocollo "train su D3 → test su OpenFake".
  - Propensione attuale: (A) per il messaggio "modello classico che crolla", MA da dichiarare bene.
    Valutare se (B) è fattibile nei tempi. → DECISIONE: ____

### 5.3 UniversalFakeDetect / CLIP-based — il "generalizzatore"
- **Cos'è:** *Towards Universal Fake Image Detectors that Generalize Across Generative Models*
  (Ojha et al., CVPR 2023). Usa feature CLIP + classificatore (nearest-neighbor / linear) → noto per
  buona **generalizzazione cross-model**.
- **Perché:** controparte "moderna" che dovrebbe reggere meglio sui generatori nuovi, ma fallire in modi
  *specifici* → materiale ricco per l'analisi dei failure mode.
- **Addestrato su:** feature CLIP (pre-training su larga scala) + testa addestrata su fake GAN/diffusion.
- DeepfakeBench include un detector `CLIP`; verificare se corrisponde o se usare il repo originale di Ojha.

---

## 6. DEEPFAKEBENCH + METRICHE

### Cos'è DeepfakeBench
- Framework unificato (NeurIPS 2023 D&B): https://github.com/SCLBD/DeepfakeBench
- Standardizza data loading, detector e **metriche** (AUROC, AP, Accuracy, EER, ...).
- Vantaggio: ci risparmia di riscrivere dataloader e calcolo metriche → ci concentriamo su esperimento + analisi.
- **ATTENZIONE / da verificare:** DeepfakeBench nasce per **video di face-deepfake**, mentre D3/OpenFake sono
  **immagini sintetiche full-frame**. Probabile serva adattamento (dataloader custom, niente face-crop/landmark).
  → DECISIONE APERTA: usare DeepfakeBench *integralmente*, oppure usarne solo la parte di metriche/analisi e
    appoggiarci al repo nativo di CoDE per l'inferenza? Da decidere dopo aver ispezionato entrambi.
- **Da documentare nel report:** quale **soglia/criterio** DeepfakeBench usa per calcolare l'**Accuracy**
  (es. soglia 0.5 fissa, oppure soglia ottimale su EER, ...). È cruciale per non scrivere numeri ambigui.

### Le metriche (cosa sono e perché le usiamo)
- **AUROC (Area Under ROC Curve):** probabilità che il modello dia score più alto a una fake che a una real
  presa a caso. **Indipendente dalla soglia** → ideale per confrontare detector e generatori. Range 0.5 (caso)–1.0 (perfetto).
- **Average Precision (AP, area sotto precision-recall):** sensibile soprattutto alla classe positiva (fake);
  più informativa di AUROC quando le classi sono **sbilanciate**. Anch'essa **soglia-indipendente**.
- **Accuracy:** % di classificazioni corrette. **Dipende dalla soglia** → va sempre detto QUALE soglia.
  Utile come numero "intuitivo" ma meno robusto delle prime due nel cross-generator.

Perché queste tre: AUROC + AP danno una misura solida e soglia-indipendente del potere discriminante;
l'Accuracy aggiunge la lettura pratica "quante ne becca", a patto di dichiarare la soglia.

---

## 7. DECISIONI APERTE (da risolvere)
- [ ] DeepfakeBench integrale vs solo-metriche + repo CoDE nativo (immagini full-frame vs video face)
- [ ] Xception: pre-addestrato as-is (A) vs re-train su D3 (B)
- [ ] Terza famiglia GAN-based: la includiamo? da dove?
- [ ] OpenFake: ci serve solo come target OOD (probabile) o anche come source?
- [ ] Numerosità finale dei subset (D3 per-generatore, OpenFake test)
- [ ] Test di compressione social: lo facciamo? (ricompressione JPEG / pipeline tipo WhatsApp)

## 8. LOG decisioni & progressi (append in fondo, con data)
- 2026-06-25 — Setup iniziale: repo creata, info.md creato. Tutor ha confermato accesso dati (HF) e
  organizzazione filesystem. Selezione detector (CoDE + Xception + UniversalFakeDetect) validata.
- 2026-06-25 — PyCharm SSH interpreter + Deployment configurati. /work al 92% → chiedere quota prima di scaricare.
  Script per download dati: da creare nel repo (così sono salvati e riproducibili). Cartelle /work: da creare via terminale.
- 2026-06-25 — Creato `info_server.md` (riferimento completo sul cluster: percorsi, quote, SLURM, comandi, cose da NON fare).
- 2026-06-25 — Niente venv: anaconda di sistema ha già i pacchetti. Esplorati D3 (solo fake, 4 generatori, no reali
  embedded) e OpenFake (ha label+model+real/fake, 3.44TB → solo streaming/API). Aggiunta modalità `--api` allo
  script di esplorazione (dataset-viewer + lettura colonnare parquet via range-request).
- 2026-06-25 — SVOLTA DESIGN: OpenFake è il pool principale; asse = colonna `model` (generatore), non lo split.
  Detector già pre-addestrati altrove → serve solo un pool etichettato multi-generatore. Famiglie di generatori
  definite. Aperti: GAN-family assente (serve fonte extra?), real-source mismatch, D3 da includere o no. → ALLINEARE CON ENRICO.
