# INFO SERVER — AImageLab-HPC (cluster CVCS)

> File di riferimento sul funzionamento del cluster. Lo leggiamo sia io (Matteo) sia Claude.
> Serve a NON sbagliare comandi e a sapere sempre dove sta cosa. Personale (nel `.gitignore`).
> Fonti: quick-start CVCS, doc "File Systems and Data Management", doc moduli/SLURM del portale ColdFront.

---
## 1. ACCESSO

```bash
ssh mbaracchi@ailb-login-02.ing.unimore.it     # login node principale
ssh mbaracchi@ailb-login-03.ing.unimore.it     # login node alternativo
```
- Username AImageLab-HPC: **mbaracchi** (diverso dallo username UNIMORE!)
- Password: quella UNIMORE
- Da fuori rete UNIMORE: serve `ssh-copy-id` la prima volta (la password resta comunque richiesta)
- ⚠️ Le GPU di `ailb-login-02` NON sono compatibili con CUDA 12.x (usare CUDA 11.8 lì, o meglio: girare su nodi di calcolo via SLURM)

---

## 2. FILESYSTEM — dove sta cosa

| Path | Tipo | Quota | Backup | A cosa serve |
|---|---|---|---|---|
| `/homes/mbaracchi` | NFS | 100 GB | ❌ No | Codice, script, **venv Python**, config (.bashrc, .ssh) |
| `/work/cvcs2026/deep_pixels` | BeeGFS | condivisa col corso | ❌ No | **Dataset, pesi, checkpoint, feature, output** |
| `/dres/<project>` | BeeGFS | su richiesta | ❌ No | Storage a lungo termine (montato sui login node) — non lo usiamo per ora |
| `/tmp` (sul nodo) | tmpfs | varia | ❌ No | File temporanei dentro un job (sparisce a fine job) |

**REGOLE D'ORO:**
- **NESSUN backup, da nessuna parte.** Se cancelli, è perso per sempre.
- `/work` viene **cancellato immediatamente** alla scadenza del progetto → esportare ciò che serve.
- Usare sempre **PATH ASSOLUTI**. Su questo cluster NON esistono `$HOME`, `$WORK`, `$DRES`.
- **Codice/venv → `/homes`** (NFS gestisce bene tanti file piccoli).
- **Dati pesanti → `/work`** (BeeGFS, alta banda per file grandi).

### I nostri percorsi
- Codice (home): `/homes/mbaracchi/cvcs2026/cvcs-deepfake-benchmark`
- Python: anaconda di sistema (già installato da admin, ha tutti i pacchetti base — nessun venv)
- Dati condivisi (work): `/work/cvcs2026/deep_pixels`
  - `datasets/` → parquet D3, subset OpenFake, DF40
  - `weights/` → file `.pth` pre-addestrati
  - `features/`, `results/` → output (da creare)

---

## 3. QUOTA — ⚠️ DA TENERE D'OCCHIO

```
Filesystem  User/Project   Usage (GB)    Quota (GB)    %
/homes      mbaracchi      14.67         100.00        14.67%   ← ok
/work       cvcs2026       3774.60       4096.00       92.15%   ← QUASI PIENO (condiviso da TUTTI i gruppi!)
```
- La quota `/work` è del corso **intero** (4 TB), spartita tra tutti i gruppi in `/work/cvcs2026/`.
- Al 92% → **pochissimo spazio libero**. Prima di scaricare dati: verificare e, se serve, scrivere ai tutor.

**Comando per controllare la quota (veloce, leggi i contatori):**
```bash
squota
```
> ⚠️ NON usare `du -sh /work/...` per controllare lo spazio: traversa tutto il filesystem e
> sovraccarica BeeGFS per tutti. Usare `squota`. (`du` su una nostra piccola sottocartella è tollerabile.)

---

## 4. MODULI — caricare software (niente sudo/apt!)

Sul cluster NON sei root: il software si carica con `module`, non si installa con `apt`.

```bash
module avail                       # elenca tutti i moduli disponibili
module load git                    # carica git (NON è attivo di default!)
module load python/3.11.11-gcc-11.4.0   # python (3.11 è il default, di solito già attivo)
module list                        # mostra i moduli caricati ora
module purge                       # scarica tutti i moduli
module help <pacchetto>            # documentazione di un modulo
```

**Software disponibile (per noi rilevante):**
- Python 3.9 / 3.10 / 3.11 (3.11 default)
- CUDA 12.6.3 (default) e 11.8.0 — cuDNN annessi
- PyTorch 2.7.0 e 2.8.0 (build CUDA-specifiche) — disponibili come modulo
- NumPy, FAISS, Singularity, ecc.

> Nota: possiamo usare PyTorch dal modulo, oppure installarlo nel nostro venv con pip. Per coerenza/riproducibilità
> conviene il venv (così le versioni sono nostre e dichiarate nel report). Da decidere.

---

## 5. AMBIENTE PYTHON

Il cluster ha **anaconda già installato da admin** con la maggior parte dei pacchetti necessari
(datasets, pandas, pyarrow, pillow, huggingface_hub, numpy, ...). **Non serve un venv.**

```bash
# verificare che il Python di sistema abbia i pacchetti
python -c "import datasets, pandas, pyarrow, PIL, huggingface_hub; print('tutto ok')"

# trovare il path esatto del Python (serve per PyCharm)
which python

# installare un pacchetto aggiuntivo non presente nel sistema (va in ~/.local)
pip install --user nome-pacchetto

# verificare cosa è installato
pip list | grep -i nome
```

Path anaconda di sistema (aggiornare se cambia):
`/homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-.../bin/python`
(trovare il path esatto con `which python` dal login node)

**Nota torch/CUDA:** quando servirà per l'inferenza GPU, si usa il modulo PyTorch del cluster
(`module load pytorch/...`) oppure `pip install --user torch` con la build CUDA giusta.
Non installare torch nel sistema manualmente — usare il modulo.

---

## 6. SLURM — eseguire sui nodi di calcolo (specie con GPU)

**Concetto:** il login node è condiviso e **non va usato per calcolo pesante / GPU**. Per girare modelli
si chiede un nodo di calcolo a SLURM (lo scheduler), che mette in coda e assegna risorse.

### Quando serve SLURM
| Attività | Dove |
|---|---|
| Scrivere codice, git, ispezionare un parquet, scaricare file piccoli | Login node (ok) |
| Inferenza detector su migliaia di immagini (GPU) | **SLURM** |
| Jupyter Lab | **SLURM** (vedi quick-start corso) |

### Comandi base
```bash
sbatch script.sh        # sottometti un job (batch)
squeue --me             # i miei job in coda/in esecuzione (guarda colonna NODELIST = nodo assegnato)
scancel <job_id>        # cancella un job
sinfo                   # stato dei nodi/partition
sinfo -o "%n %G" -p all_usr_prod    # vedere la config gres (GPU/tmpfs) per nodo
```

### Esempio di job script (da quick-start, partition seriale CPU)
```bash
#!/bin/bash
#SBATCH --job-name=cvcs_job
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/homes/mbaracchi/cvcs2026/job_%j.out
#SBATCH --account=cvcs2026

source /homes/mbaracchi/cvcs2026/cvcs-deepfake-benchmark/venv/bin/activate
python mio_script.py
```
Sottometti con: `sbatch script.sh` · monitora con: `squeue --me` · log in: `job_<jobid>.out`

### Chiedere una GPU  ⚠️ DA VERIFICARE
- L'esempio del corso usa `--partition=all_serial`, che è **seriale/CPU** (niente GPU).
- Per avere una GPU serve una partition GPU + la richiesta gres. Nella doc compare la partition
  `all_usr_prod` (con `sinfo ... -p all_usr_prod`), probabile partition GPU. Schema tipico:
  ```bash
  #SBATCH --partition=all_usr_prod        # <-- DA CONFERMARE col tutor/doc
  #SBATCH --gres=gpu:1                     # 1 GPU
  #SBATCH --account=cvcs2026
  ```
- tmpfs locale veloce per I/O intensivo: `#SBATCH --gres=gpu:1,tmpfs:50G`
- **TODO: confermare con tutor il nome esatto della partition GPU e i limiti (tempo/GPU) per l'account cvcs2026.**

### Pattern I/O consigliato (dataset grossi)
Copia i dati su `/tmp` del nodo a inizio job, lavora lì, copia i risultati su `/work` a fine job:
```bash
cp /work/cvcs2026/deep_pixels/datasets/D3/file.parquet /tmp/
python train.py --data /tmp/file.parquet --out /tmp/out
cp -r /tmp/out /work/cvcs2026/deep_pixels/results/
```

---

## 7. TRASFERIMENTO FILE

- **File piccoli** → da/verso login node con `scp` / `rsync` / `sftp`.
- **File/dataset grandi** → usare il **data mover** dedicato: `ailb-data.ing.unimore.it`
  (nessun limite di CPU time; accetta SOLO trasferimenti, niente shell interattiva).

```bash
# upload grande dataset (da locale al cluster)
rsync -avP /local/dataset/ mbaracchi@ailb-data.ing.unimore.it:/work/cvcs2026/deep_pixels/datasets/
# download risultati (dal cluster a locale)
rsync -avP mbaracchi@ailb-data.ing.unimore.it:/work/cvcs2026/deep_pixels/results/ /local/results/
```
> Per scaricare i dataset da HuggingFace direttamente sul server NON serve il data mover:
> si usa `huggingface-cli download` / `datasets` da dentro un job o dal login node (file → vanno su /work).

---

## 8. BeeGFS — buone pratiche su /work

BeeGFS è ottimo per **file grandi** e letture sequenziali; **pessimo con milioni di file piccoli**.
- ✅ Tenere i dataset come **parquet / archivi (tar) / LMDB** e leggerli in streaming.
- ❌ Evitare di scompattare milioni di immagini sciolte su `/work` (rallenta `ls`/`find`/avvio job per TUTTI).
- ❌ Evitare scritture piccole e frequenti in loop (bufferizzare, scrivere a blocchi).
- Se proprio servono tanti file: sottocartelle da ≤ 10.000 file ciascuna.
- Venv/pacchetti pip → in `/homes`, MAI in `/work`.

---

## 9. ⚠️ COMANDI / COSE DA NON FARE SUL SERVER

| ❌ NON fare | Perché | ✅ Invece |
|---|---|---|
| `sudo ...` / `apt install ...` | Non sei root, dà errore | `module load <pkg>` o `pip install` nel venv |
| Girare training/inferenza GPU sul **login node** | È condiviso, ti bloccano | Job SLURM su nodo di calcolo |
| `du -sh /work/cvcs2026` o `find /work` ampi | Sovraccarica BeeGFS per tutti | `squota` per le quote; `find` con `-maxdepth` |
| `ls -lR /work/...` ricorsivo enorme | Stessa cosa (metadata storm) | scope ristretto, `-maxdepth` |
| Scompattare un dataset in milioni di file su `/work` | Degrada BeeGFS | tenere parquet/tar/LMDB, leggere in streaming |
| Creare un venv inutile su `/homes` | Riempie la quota con file piccoli; anaconda ha già tutto | Usare Python di sistema; `pip install --user` per extra |
| Lanciare Jupyter sul login node | Non permesso | Job SLURM (vedi quick-start) |
| Affidarsi al "backup" | NON esiste backup | tenere copia di ciò che è prezioso |
| Riempire `/work` senza controllare | È al 92%, condiviso | `squota` prima, scaricare subset |
| Sessione lunga interattiva sul login per download enormi | Limite CPU time | data mover `ailb-data` o job SLURM |

---

## 10. CHEAT SHEET veloce

```bash
# navigazione
pwd                      # dove sono
ls -la                   # contenuto cartella
cd /homes/mbaracchi      # spostarsi (path assoluto!)

# git (ricorda: module load git all'inizio)
module load git
git status
git pull
git push

# python (nessun venv — anaconda di sistema ha già tutto)
python -c "import datasets; print('ok')"   # verifica
pip install --user <pkg>                    # per pacchetti extra non presenti

# slurm
sbatch script.sh
squeue --me
scancel <jobid>

# disco
squota                   # quota (NON du su /work intero)
```

---

## DA VERIFICARE / TODO server
- [x] Nessun venv necessario: il cluster ha anaconda con tutti i pacchetti base già installati
- [ ] Trovare il path esatto del Python con `which python` e aggiornarlo in §5
- [ ] Nome esatto della partition GPU per l'account `cvcs2026` (all_usr_prod?) e limiti (tempo/numero GPU)
- [ ] Esiste una sotto-quota per `deep_pixels` o si pesca dai 4 TB comuni del corso? (chiedere ai tutor)
- [ ] Conviene PyTorch da modulo o da pip nel venv? (riproducibilità vs comodità)
