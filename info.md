# INFO — Progetto CVCS Deepfake Detection

> **Documento condiviso del gruppo** (Matteo + Enrico). Qui raccogliamo TUTTO sul progetto: idea,
> cose da fare, decisioni prese e da prendere, dataset, protocollo, detector, metriche.
> Aggiornare man mano: ogni passo fatto va spuntato nella §2 e annotato nel LOG (§9).
>
> 👉 Per il funzionamento del **SERVER** (percorsi, quote, comandi, SLURM, cosa NON fare) vedi **`info_server.md`**.

**Progetto:** *Deepfake Detection in the Wild: Cross-Generator Generalization Benchmarking*
**Corso:** Computer Vision and Cognitive Systems (CVCS) 2025/2026, UNIMORE — prof. Lorenzo Baraldi
**Gruppo:** Deep Pixels — Matteo Baracchi + Enrico Ranieri
**Tutor:** Silvia Cappelletti (silvia.cappelletti@unimore.it), Tobia Poppi (tobia.poppi@unimore.it)
**Repo:** https://github.com/barack-M/cvcs-deepfake-benchmark

---

## 0. L'idea in una frase

Un detector di deepfake addestrato sui generatori di *oggi* funziona ancora sui generatori di *domani*?
Prendiamo detector **pre-addestrati**, li testiamo **cross-generator** (su famiglie di generatori mai viste
in training) e analizziamo **quando, perché e quanto** falliscono. Non proponiamo un metodo nuovo: produciamo
un'**analisi rigorosa e riproducibile** dei punti di forza e dei *failure mode*.

Insight guida: ci aspettiamo che **CoDE regga fino a ~SDXL** e crolli sui generatori recenti (FLUX, Midjourney v7,
GPT-Image, ...). Documentare e spiegare *dove e perché* ogni detector crolla è il cuore del progetto.

---

## 1. WORKFLOW — come lavoriamo

### Dove sta cosa (vale per entrambi; `<username>` = mbaracchi / eranieri)
- **Codice** → nella propria HOME: `/homes/<username>/cvcs2026/cvcs-deepfake-benchmark` (clone git)
- **Python** → venv `/homes/<username>/cvcs2026/venv` creato con `python3 -m venv --system-site-packages`
  (versione di sistema: **Python 3.10.12**; eredita i pacchetti di sistema, evita di esaurire la quota inode).
  Attivare sempre con `source /homes/<username>/cvcs2026/venv/bin/activate`. Dettagli in `info_server.md` §5.
- **Dati, pesi, feature, risultati** → cartella condivisa `/work/cvcs2026/deep_pixels/` (BeeGFS):
  - `datasets/` → parquet/subset (D3, OpenFake, ...)
  - `weights/` → i `.pth` pre-addestrati dei detector
  - `features/` → embedding estratti (per t-SNE e per detector tipo UniversalFakeDetect)
  - `results/` → score per immagine (per modello × generatore) → da qui calcoliamo le metriche
- **Regole filesystem:** path SEMPRE assoluti; su BeeGFS niente milioni di file piccoli (tenere parquet/archivi).

### Ciclo di sviluppo
- Si scrive il codice in **locale** (PyCharm) → si esegue sul **server**.
- Chi usa **PyCharm Deployment**: al salvataggio il file viene caricato sul server in automatico (SFTP);
  Run/Debug girano sul server tramite SSH interpreter. (NB: l'esecuzione PyCharm avviene sul login node.)
- Condivisione tra noi via **git**: `git add/commit/push` da locale → l'altro fa `git pull` sul server.
  Git e PyCharm-Deployment sono indipendenti: il push NON aggiorna il server dell'altro, serve il pull.
- I **dati non passano da git** (stanno su `/work`, esclusi dal `.gitignore`); il codice li legge via path assoluto.

### Esecuzione pesante
- Codice leggero (ispezione dati, prove su poche immagini) → ok sul login node.
- Inferenza dei detector su molte immagini (GPU) → **SLURM** (vedi `info_server.md`).

### Struttura del codice (verso cui andiamo)
```
src/        # codice RIUTILIZZABILE importato dagli script (data/, models/, metrics/, utils/)
scripts/    # entry-point "sottili" che si LANCIANO, uno per compito
configs/    # config per esperimento (path, generatori, seed)
slurm/      # job script .sh per il cluster
requirements.txt   # librerie (su git; torch escluso → dipende dalla CUDA del cluster)
```
Principio: la logica condivisa (es. "leggi parquet", "calcola metriche") sta in `src/`, MAI copiata negli script.
Gli script in `scripts/` leggono argomenti → chiamano `src/` → salvano output.
**File già presenti:** `scripts/explore_hf_dataset.py` — ispeziona un dataset HF (modalità `--api` istantanea via
dataset-viewer, oppure streaming). Usato per esplorare D3 e OpenFake.

---

## 2. TODO — cose da fare

Legenda: [ ] da fare · [~] in corso · [x] fatto

### Setup
- [x] Repo GitHub creata e clonata (locale + home di entrambi sul server)
- [x] Chiarito con tutor: accesso dataset (HuggingFace) e organizzazione filesystem
- [x] Cartelle su /work: `datasets/`, `weights/`, `features/`, `results/`
- [x] PyCharm remote (SSH interpreter su venv + Deployment)
- [x] Venv creato (`python3 -m venv --system-site-packages /homes/<username>/cvcs2026/venv`) — Python 3.10.12
- [x] Pacchetti installati nel venv: `huggingface_hub datasets pyarrow fsspec`
- [ ] Clonare CoDE e/o DeepfakeBench nelle home (dipende dalla decisione D5, §8)

### Dati
- [x] Esplorato D3 (train/validation/external_test) — struttura nota (vedi §3.1)
- [x] Esplorato OpenFake (core train/test) — struttura e generatori noti (vedi §3.2)
- [x] D3 validation scaricato: 11 shard, 4.83 GB, 4.800 righe totali (= 4.800 img/generatore)
      → `/work/cvcs2026/deep_pixels/datasets/D3/data/validation-*.parquet`
- [x] `build_subset.py` scritto: estrae immagini fake da D3, **n=4800 (intero validation), seed=42**
      → output: `D3/images/{deepfloyd,sd14,sd21,sdxl}/` + `D3/manifest.csv`
- [x] Lancio di `build_subset.py` (estratte 19.200 JPEG fake D3 e allineato il manifest)
- [x] Scaricare il test set GAN di UniversalFakeDetect/ForenSynths (CNN_synth_testset) in `datasets/`
      → Eseguito download parallelo ed estrazione selettiva (progan, stylegan, cyclegan, max 1000 img/classe, 6.000 img totali) + generato manifest allineato.
- [x] OpenFake `core/test`: scaricato l'intero test set (68 GB in Parquet) + generato manifest allineato basato su indici (91.398 righe).
- [ ] Per le fake di D3 usare reali da OpenFake (D3 non ha reali embedded)
- [x] Salvare la lista di file/indici dei subset + annotare numerosità nel report (Fatto in `info.md` e nei manifest)

### Detector  (decisi: CoDE + UniversalFakeDetect; terzo IN DECISIONE → Effort?, vedi §5.3/§8)i: CoDE + UniversalFakeDetect; terzo IN DECISIONE → Effort?, vedi §5.3/§8)
- [ ] CoDE: estendere la bozza di inferenza (Enrico) a un dataset intero
- [ ] Decidere il terzo detector (Effort?) e verificarne pesi pubblici + dataset di training
- [ ] Reperire pesi pre-addestrati dei 3 detector → in `/work/.../weights`
- [ ] Per ogni detector annotare su quali dati è (pre-)addestrato

### Evaluation & Analisi
- [ ] Pipeline: (detector, subset) → score per immagine → file in `results/`
- [ ] AUROC, Average Precision, Accuracy per ogni coppia detector × generatore
- [ ] Tabella riassuntiva per-generatore (righe = detector, colonne = generatori/famiglie)
- [ ] t-SNE degli spazi delle feature
- [ ] Analisi failure mode: per architettura del generatore, per contenuto, sotto compressione social
- [ ] (Eventuale) test robustezza a compressione JPEG (stile social)

### Report
- [ ] Report con protocollo, tabelle, grafici, analisi failure mode
- [ ] Dichiarare numerosità di tutti i subset e il seed (riproducibilità)

---

## 3. DATASET

### Principio guida (dal tutor)
> Usare più dati possibile compatibilmente con spazio/tempi. Se troppo grandi → **subset**, dichiarando nel
> report **quanti esempi** si sono usati. Per campionamenti random: **fissare un SEED e riportarlo**.

### 3.1 D3 (ELSA_D3) — riferimento "in-distribution" / generatori vecchi  *(struttura VERIFICATA)*
- **Cos'è:** dataset diffusion-focused di AImageLab UNIMORE (ELSA EU). È il dataset su cui è stato costruito CoDE.
- **Accesso:** da HuggingFace (NON montato sul cluster). ~2,6 TB → solo subset.
  - `elsaEU/ELSA_D3` (split `train`, `validation`)
  - `elsaEU/ELSA_D3_external_test` (split `train`, `test_set_transf`)
- **Struttura REALE (verificata):** ogni riga = 1 prompt + URL di **un'immagine reale (NON embedded)** +
  **4 immagini FAKE** generate dagli stessi 4 generatori FISSI per tutto il dataset:
  | campo | generatore | famiglia |
  |---|---|---|
  | gen0 | DeepFloyd/IF-II-L-v1.0 | pixel diffusion (256px) |
  | gen1 | CompVis/stable-diffusion-v1-4 | SD 1.4 |
  | gen2 | stabilityai/stable-diffusion-2-1-base | SD 2.1 |
  | gen3 | stabilityai/stable-diffusion-xl-base-1.0 | SDXL |
  Colonne immagine: `image_gen0..3` (PIL, embedded). Label generatore = nota e pulita. ✅
- **⚠️ PROBLEMA: niente immagini REALI embedded.** Le reali sono solo `url` (link esterni). Per fare detection
  real-vs-fake servono le reali → o si scaricano dagli URL (rischioso: link morti, lento) o si usa un corpus
  reale esterno. → vedi decisione **D2** (§8).
- **`ELSA_D3_external_test`:** ha SOLO colonne `image` + `id` → **nessuna label** (né real/fake né generatore).
  Inutile per la nostra analisi. Scartato.
- **Ispezione:** `python scripts/explore_hf_dataset.py --repo elsaEU/ELSA_D3 --split train`

### 3.2 OpenFake — target OUT-OF-DISTRIBUTION / generatori nuovi  *(struttura VERIFICATA)* ⭐
- **Cos'è:** dataset 2025 (arXiv 2509.09495), immagini sintetiche full-frame con generatori recenti/proprietari.
- **Accesso:** `ComplexDataLab/OpenFake` su HuggingFace — **3.44 TB**, 2.46M righe → SOLO streaming/subset.
- **Schema (verificato):** `image`, `label` (real/fake), `model` (generatore ✅), `type`
  (base/image/video/real/finetune/lora), `prompt`, `release_date`. → Ha TUTTO ciò che serve.
- **Config/split:** `core` (train 2.31M / validation 59K / test 91K) e `reddit` (test 36K "in-the-wild").
- **`core/test` (nostro target OOD):** ~50/50 real/fake. **Reali da `imagenet`/`docci`.** Generatori fake 2025-26:
  flux.2-klein, z-image-turbo, gpt-image-1.5/2, nano-banana-pro, midjourney-7, ideogram-2.0, recraft-v3,
  seedream-v5, illustrious (finetune SDXL), e VIDEO (sora-2, veo-3, wan-video) → frame estratti.
- **`core/train`:** 100+ generatori, vecchi E nuovi mischiati (sd-1.5/2.1, sdxl, flux.1, sd-3.5, dalle-3,
  imagen-4, midjourney-6, gpt-image-1, ...). **Reali da `laion`/`pexels`.**
- **Indicazione tutor:** se OpenFake è SOLO target OOD → usare **solo lo split `test`** (idealmente completo;
  se troppo grande, subset fisso **50k–100k**, bilanciato per generatore e real/fake).
- **Ispezione (istantanea):** `python scripts/explore_hf_dataset.py --repo ComplexDataLab/OpenFake --config core --split test --api`

### 3.3 Test set di UniversalFakeDetect (ForenSynths) — famiglia GAN-based  ✅ DECISO (risolve D1)
- **Cos'è:** il set di valutazione di Ojha et al. (CVPR 2023), costruito su **ForenSynths** (Wang 2020).
  Immagini full-frame (oggetti/scene da LSUN/ImageNet) **con le reali appaiate incluse**, da molti generatori.
- **Generatori GAN che usiamo:** ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN.
  (Contiene anche early-diffusion — LDM, GLIDE, ADM/guided, DALL-E mini — usabili o ignorabili. EVITARE i
  sottoinsiemi di VOLTI tipo "deepfake/whichfaceisreal" per restare full-frame.)
- **Vantaggi:** (1) completa le 3 famiglie del prof (GAN→diffusion→proprietary); (2) **porta le proprie reali**
  → niente problema reali-mancanti; (3) **leggero** (qualche GB, non TB) → nessun problema di quota.
- **⚠️ In-distribution per UniversalFakeDetect e CNNDetection** (addestrati su ProGAN) → su questo set vanno
  benissimo (atteso, da dichiarare). Per **CoDE** (diffusion-trained) i GAN sono NON visti → atteso crollo.
  → confronto perfetto: stesso set fa brillare un detector e crollare un altro a seconda del training.
- **Dove:** repo UniversalFakeDetect (Ojha) / CNNDetection (Wang). Scaricato in `/work/cvcs2026/deep_pixels/datasets/GAN/` tramite lo script `scripts/download_gan_dataset.py`, che scarica lo zip da Hugging Face (`sywang/CNNDetection`). Per evitare di sforare la quota disco (~19 GB scompattato), lo script esegue un'**estrazione selettiva**: estrae solo le immagini per i generatori target (`progan`, `stylegan`, `cyclegan`) limitandosi a un massimo di 1000 immagini reali (`0_real`) e 1000 immagini fake (`1_fake`) per generatore. Questo riduce la dimensione finale a circa **600 MB (6.000 file totali)** ed evita il blocco degli inode. Crea automaticamente il manifest CSV (`manifest.csv`).

### Subset & riproducibilità (regola trasversale)
- Ogni subset: **seed fisso**, salvare la **lista di file/indici** selezionati, bilanciare real/fake e per-generatore.
- Annotare qui la numerosità finale di ogni subset usato (da compilare quando li costruiamo).

---

## 4. PROTOCOLLO sperimentale (CHIARO e RIPRODUCIBILE)

### Domanda di ricerca
Quanto degrada la detection passando da generatori "vecchi" (early/late diffusion) a generatori
moderni/proprietari mai visti? Quali detector generalizzano meglio e perché? Dove falliscono?

### Costruzione del POOL di valutazione (DECISO) — 3 famiglie di generatori
Pool etichettato real/fake con generatore noto, da 3 fonti (una per famiglia):
- **GAN-based** → test set UniversalFakeDetect/ForenSynths: FAKE (ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN,
  GauGAN) + le sue REALI appaiate (LSUN/ImageNet).
- **Early/late diffusion** → D3: FAKE (DeepFloyd, SD1.4, SD2.1, SDXL, da `model_gen0..3`). Reali: vedi sotto.
- **Recent proprietary + modern open** → OpenFake `core/test`: FAKE (flux.2, gpt-image-2, midjourney-7, ...)
  + REALI (imagenet/docci).
- Per le FAKE di D3 (che non hanno reali proprie) → usare reali da OpenFake. Bilanciare ~50/50 per famiglia.

Tutte e 3 le fonti hanno la label di generatore → **analisi per-generatore** OK.
Asse del racconto: **GAN → early diffusion → late/SDXL → modern open → proprietary** (vecchi → nuovi).

**⚠️ Caveat da dichiarare nel report:**
- **Sorgenti real eterogenee per famiglia** (GAN→LSUN/ImageNet, D3→OpenFake reali, proprietary→imagenet/docci):
  OK per AUROC per-generatore (ogni generatore vs un real coerente); per un numero GLOBALE aggregato la classe
  "real" è eterogenea → gestire/dichiarare.
- **Risoluzione:** fake D3 piccole (256–640px) vs OpenFake ~1024px → rischio "piccolo=fake". Mitigato dal resize
  a 224px in input, ma da segnalare.
- **Contenuto:** le 3 fonti nascono da distribuzioni di contenuto diverse.
- **Leakage di dominio:** il set GAN è in-distribution per UniversalFakeDetect/CNNDetection → loro vanno bene lì
  per costruzione; per CoDE i GAN sono non visti. Dichiararlo (è il punto, non un bug).
- **VIDEO esclusi:** generatori video di OpenFake (sora-2, veo-3, wan) fuori dal pool principale.

### Raggruppamento generatori in FAMIGLIE/ERE (backbone dell'analisi)
100+ generatori singoli sono troppo granulari → raggruppare:
- **GAN-based:** ProGAN, StyleGAN(2), BigGAN, CycleGAN, StarGAN, GauGAN (da UniversalFakeDetect/ForenSynths)
- **Early diffusion:** SD 1.x, SD 2.x, DeepFloyd
- **Late diffusion / SDXL era:** SDXL e finetune, SD 3.5, Playground
- **Modern open:** FLUX family, z-image, HiDream, Qwen-Image, Chroma
- **Proprietary:** DALL-E 3, Imagen, Midjourney, GPT Image, Ideogram, nano-banana, Grok, Seedream
- **Video (a parte):** sora-2, veo-3, wan-video → frame estratti, caso diverso (decisione D3)

### Attenzione al DATA-LEAKAGE (ruoli diversi di D3 per detector diverso)
- **CoDE è addestrato su D3** → su D3 misura la performance *in-distribution* (soffitto), NON la generalizzazione;
  testarlo sulle stesse immagini di training = leakage. Usare lo split **validation** di D3 per ridurlo.
  Il risultato interessante per CoDE è il **crollo su OpenFake** (generatori veri non visti).
- **UniversalFakeDetect (e il 3° detector) NON sono addestrati su D3** → per loro D3 è già non-visto, nessun leakage.
- Regola d'oro: per OGNI detector, dichiarare sempre su quali dati è (pre-)addestrato e su quali è testato.

### Caveat metodologici emersi dai dati
- **Sorgenti "real" diverse:** D3 reali = URL esterni; OpenFake train reali = laion/pexels; OpenFake test
  reali = imagenet/docci. Mischiarle rende la classe "real" eterogenea → un detector potrebbe sfruttare lo
  *stile del dataset* invece degli artefatti. Scegliere una sorgente real coerente o documentarlo (decisione D6).
- **Colonna `type` di OpenFake sporca:** casing incoerente (base/Base, finetune/Fine-tune, lora/LoRA) → normalizzare.

### Cosa misuriamo
Per ogni **detector** × **generatore** (o famiglia): AUROC, Average Precision, Accuracy.
Più una vista **globale** (test aggregato) e una **per-generatore** (per vedere dove crolla).

### Output attesi
1. Tabella riassuntiva: righe = detector, colonne = generatori/famiglie.
2. Breakdown per-generatore (bar chart / heatmap).
3. t-SNE degli embedding (separabilità real/fake; posizione dei generatori non visti).
4. Analisi failure mode (architettura del generatore, contenuto, compressione social).

### Riproducibilità — checklist
- [ ] Seed fissato e riportato · [ ] Liste subset salvate · [ ] Per ogni detector: train-set vs test-set documentati
- [ ] Soglia/criterio dell'Accuracy esplicitato (§6) · [ ] Versioni di dataset/modelli annotate

---

## 5. DETECTOR

### ⚠️ DUE ASSI DA NON CONFONDERE
- **Asse A — famiglie di GENERATORI = i DATI di test** (ciò che intende il prof con "GAN-based, early diffusion,
  recent proprietary"): da quale generatore vengono le fake che diamo in pasto. Si copre coi DATASET (§3/§4).
- **Asse B — i DETECTOR e su cosa sono ADDESTRATI:** il background di ciascun modello, che spiega dove generalizza
  e dove crolla. Si documenta per ogni detector qui sotto.
- "recent proprietary" è una famiglia di GENERATORI (OpenFake), NON richiede un detector "proprietario".

### 5.1 CoDE — modello principale (obbligatorio) — DECISO
- *Contrasting Deepfakes Diffusion...* (Baraldi et al., ECCV 2024). **Embedding contrastivi**, di AImageLab UNIMORE.
- **Perché:** è il modello del laboratorio del corso, riferimento dell'analisi.
- **Addestrato su:** D3 (diffusion, fino a ~SDXL) → atteso buon in-distribution e **crollo sui generatori moderni**.
- **Codice + pesi:** https://github.com/aimagelab/CoDE — Enrico ha già una bozza di inferenza su poche immagini.

### 5.2 UniversalFakeDetect (Ojha et al., CVPR 2023) — il "generalizzatore" — DECISO
- Feature **CLIP** (frozen) + classificatore nearest-neighbor/linear → buona **generalizzazione cross-model**.
- **Perché:** suggerito dal prof; dovrebbe reggere meglio sui nuovi ma fallire in modi *specifici*.
- **Addestrato su:** fake GAN (ProGAN) + feature CLIP. NB: il training su GAN è una *scelta di design* del paper
  (un GAN + CLIP generalizza), NON un segno di "vecchiaia": il metodo è del 2023.

### 5.3 Terzo detector — IN DECISIONE (vedi D-detector in §8)
Slot del prof: *"any recent detector available on DeepfakeBench"*. Opzioni:
- **Effort** (Yan et al., ICML 2025, su DeepfakeBench) — CLIP-based con decomposizione a sottospazi ortogonali,
  progettato per AIGC **generale (non volti)** → **in-dominio** e **recente**. ✅ fedele al prof.
  Differenza da Xception: stesso framework (DeepfakeBench) ma Xception è addestrato su VOLTI (out-of-domain,
  confermato tutor), Effort lavora su immagini full-frame. ⚠️ MA è CLIP-based come UniversalFakeDetect → poca
  diversità architetturale (avremmo 1 contrastive + 2 CLIP). Verificare: pesi pubblici + dataset di training.
- **CNNDetection** (Wang 2020) — ResNet-50 CNN pura full-frame, ProGAN. Darebbe diversità (CNN puro) ma è
  vecchio e non è "recente da DeepfakeBench". Scartato (utente: troppo vecchio).
- **Xception** — scartato: addestrato su VOLTI → out-of-domain.

> Lineup in via di definizione: CoDE (contrastive, diffusion-trained) + UniversalFakeDetect (CLIP, GAN-trained)
> + [Effort?] (CLIP, recente). DeepfakeBench SOLO per metriche/utility; inferenza coi repo nativi.

---

## 6. DEEPFAKEBENCH + METRICHE

### DeepfakeBench
- Framework unificato (NeurIPS 2023): https://github.com/SCLBD/DeepfakeBench — standardizza dataloader,
  detector e metriche (AUROC, AP, Accuracy, EER, ...). Ci farebbe risparmiare codice.
- **⚠️ Nato per video di face-deepfake**, mentre noi abbiamo immagini full-frame → probabile bisogno di
  adattamento (dataloader custom, niente face-crop/landmark). → decisione **D5** (§8).
- Da documentare: **quale soglia** usa per l'Accuracy (0.5 fissa? ottimale su EER?).

### Le metriche
- **AUROC:** prob. che il modello dia score più alto a una fake che a una real casuale. **Soglia-indipendente**,
  ideale per confrontare. Range 0.5 (caso) – 1.0 (perfetto).
- **Average Precision (AP):** area sotto precision-recall, sensibile alla classe fake; utile se classi sbilanciate.
  Anch'essa soglia-indipendente.
- **Accuracy:** % corrette. **Dipende dalla soglia** → dire sempre quale. Numero intuitivo ma meno robusto.
- Usiamo tutte e tre: AUROC+AP per il potere discriminante (robuste), Accuracy per la lettura pratica.

---

## 7. STATO ATTUALE (a colpo d'occhio)
- ✅ Infrastruttura pronta (server, venv Python 3.10.12, repo, PyCharm, cartelle /work).
- ✅ Esplorazione dati completa: sappiamo com'è fatto D3 e OpenFake.
- ✅ D3 validation scaricato (4.83 GB, 11 shard, 4.800 img/generatore).
- ✅ `build_subset.py` scritto e lanciato (n=4800, seed=42 → 19.200 JPEG fake D3).
- ⏭️ Prossimo: scaricare OpenFake `core/test` + ForenSynths, poi pesi detector, poi inferenza.

---

## 8. DECISIONI

### ✅ Risolte
- **D2 — D3 incluso (OBBLIGATORIO da consegna prof).** Reali mancanti risolte: fake da D3 + fake da OpenFake +
  reali da OpenFake (sorgente real unica e coerente). → risolve anche D6.
- **D3 — Generatori VIDEO:** ESCLUSI dal pool principale (frame, caso diverso). Eventuale curiosità a parte.
- **D4 — Xception scartato:** addestrato su volti (FF++) → out-of-domain (confermato tutor). Il terzo detector
  sarà full-frame/in-dominio (vedi D-det sotto, ancora aperta).
- **D5 — DeepfakeBench:** SOLO metriche/utility; inferenza con repo nativi dei detector.
- **D6 — Sorgente real:** per famiglia (GAN porta le sue reali; D3 usa reali OpenFake; proprietary reali OpenFake)
  → coerente per AUROC per-generatore; eterogenea solo nel numero globale aggregato (da dichiarare).
- **D1 — Famiglia GAN-based:** ✅ INCLUSA via test set UniversalFakeDetect/ForenSynths (ProGAN/StyleGAN/BigGAN/
  CycleGAN/StarGAN/GauGAN). Full-frame, porta le sue reali, leggero. Copre le 3 famiglie del prof.

### ⏳ Ancora aperte
- **D-det — Terzo detector:** Effort (recente, in-dominio, su DeepfakeBench) vs altro. Propensione: Effort,
  ma è CLIP-based come UniversalFakeDetect (poca diversità architetturale). Verificare pesi+training di Effort.
  CNNDetection scartato (troppo vecchio), Xception scartato (volti). (vedi §5.3)
- **D7 — Numerosità subset + seed:** ✅ PARZIALMENTE RISOLTO:
  - **D3:** n=4.800/generatore (intero validation split, no campionamento), **seed=42**.
    Motivazione: validation = no leakage per CoDE; 4.800 ≈ 5k (fascia tutor: 5k–20k).
  - **OpenFake test:** ⏳ da decidere (tutor: 50k–100k totali, bilanciato per generatore e real/fake).
- **D8 — Test di compressione social:** lo facciamo? Il prof lo cita esplicitamente ("Under social-media
  compression?") → punto a favore. Ricompressione JPEG stile WhatsApp sul pool, e si rimisurano le metriche.

---

## 9. LOG decisioni & progressi (append in fondo, con data)
- 2026-06-25 — Setup: repo creata; tutor conferma accesso dati (HF) e organizzazione filesystem; selezione
  detector (CoDE + Xception + UniversalFakeDetect) validata.
- 2026-06-25 — PyCharm SSH interpreter + Deployment configurati; creato `info_server.md`.
- 2026-06-25 — Niente venv: anaconda di sistema ha i pacchetti. Quota /work del corso al ~92% (4 TB condivisi).
- 2026-06-25 — Esplorati D3 (4 generatori fissi diffusion, NO reali embedded, external_test senza label) e
  OpenFake (label+model+real/fake; 3.44TB; test = generatori 2025-26, reali imagenet/docci). Aggiunta modalità
  `--api` allo script di esplorazione (dataset-viewer + lettura colonnare parquet via range-request).
- 2026-06-25 — Design allineato a tutor: OpenFake `core/test` = target OOD; D3 = ancora in-distribution "vecchi".
  Verificato: GAN assenti in D3/OpenFake (solo diffusion); DF40 ha GAN ma di volti; fonte GAN full-frame =
  ForenSynths/UniversalFakeDetect. Aperte le decisioni D1–D8 → da prendere insieme.
- 2026-06-29 — Ambiente Python risolto: `python3` di sistema (3.10.12) + venv con `--system-site-packages`
  in `/homes/mbaracchi/cvcs2026/venv`. Nota: `python` = Python 2 sul server, usare sempre `python3` per creare
  il venv; poi all'interno del venv attivato `python` funziona correttamente.
- 2026-06-29 — D3 validation scaricato: tutti e 11 gli shard, 4.83 GB, 34 secondi a 80–100 MB/s.
  Righe totali: **4.800** (437 per shard × 11). Ogni riga = 4 img fake (gen0..3). Path:
  `/work/cvcs2026/deep_pixels/datasets/D3/data/validation-*.parquet`.
- 2026-06-29 — Decisione D7 (D3): **n=4.800/generatore, seed=42** (intero validation split, nessun
  campionamento). Usare validation evita data-leakage con CoDE (addestrato su D3 train).
  Script `scripts/build_subset.py` scritto e lanciato → estrae 19.200 JPEG in `D3/images/` + `D3/manifest.csv`.
