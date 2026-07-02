# INFO SERVER — AImageLab-HPC (cluster CVCS)

> File di riferimento sul funzionamento del cluster. Lo leggiamo sia io (Matteo) sia Claude.
> Serve a NON sbagliare comandi e a sapere sempre dove sta cosa. Personale (nel `.gitignore`).
> Fonti: quick-start CVCS, doc "File Systems and Data Management", doc moduli/SLURM del portale ColdFront.

---
## 1. ACCESSO

```bash
ssh mbaracchi@ailb-login-02.ing.unimore.it
```
```bash
cd /homes/mbaracchi/cvcs2026/cvcs-deepfake-benchmark
```
```bash
source /homes/mbaracchi/cvcs2026/venv/bin/activate
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
- Python: venv `/homes/mbaracchi/cvcs2026/venv` creato con `--system-site-packages` (eredita i pacchetti
  dell'anaconda di sistema, ci installiamo sopra solo il nostro — vedi §5)
- Dati condivisi (work): `/work/cvcs2026/deep_pixels`
  - `datasets/` → parquet D3, subset OpenFake, DF40
  - `weights/` → file `.pth` pre-addestrati
  - `features/`, `results/` → output (da creare)

### ⚠️ Permessi di condivisione (ACL) — access vs default
Un `setfacl -m u:<partner>:rwx <dir>` dà accesso SOLO a quella cartella e NON viene ereditato dalle
sottocartelle create dopo (il partner le vede ma non può scriverci). Per la condivisione completa servono
DUE comandi (li lancia il proprietario, una volta):
```bash
setfacl -R -m  u:eranieri:rwx /work/cvcs2026/deep_pixels   # accesso a tutto ciò che ESISTE già (ricorsivo)
setfacl -R -m d:u:eranieri:rwx /work/cvcs2026/deep_pixels   # DEFAULT ACL → le cartelle FUTURE ereditano
```
Verifica con `getfacl <dir>`: devono comparire sia `user:eranieri:rwx` sia `default:user:eranieri:rwx`.

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
module list                        # mostra i moduli caricati ora
module help <pacchetto>            # documentazione di un modulo
```

> ⚠️ **NON lanciare `module purge`!** Rimuove anche l'anaconda caricata di default → perdi `conda`/`python 3`
> e torni al Python 2 di sistema. Se è successo: apri un terminale NUOVO (al login `(base)` si riattiva), oppure
> `module avail 2>&1 | grep -i conda` e ricarica il modulo anaconda. Recupero diretto (path noto):
> `source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-*/etc/profile.d/conda.sh && conda activate base`

> ⚠️ **GIT sul login node:** `module load git` può fallire ("Unable to locate a modulefile") e `git` non è
> in PATH. **NON serve git sul server.** Facciamo tutto il versioning dal **Mac in locale** (PyCharm Git o
> terminale del Mac); PyCharm Deployment sincronizza i file sul server. Il server è solo copia di lavoro.
> SSH interpreter (esegue codice) e git (versioning) sono cose indipendenti.

**Software disponibile (per noi rilevante):**
- Python 3.9 / 3.10 / 3.11 (3.11 default)
- CUDA 12.6.3 (default) e 11.8.0 — cuDNN annessi
- PyTorch 2.7.0 e 2.8.0 (build CUDA-specifiche) — disponibili come modulo
- NumPy, FAISS, Singularity, ecc.

> Nota: possiamo usare PyTorch dal modulo, oppure installarlo nel nostro venv con pip. Per coerenza/riproducibilità
> conviene il venv (così le versioni sono nostre e dichiarate nel report). Da decidere.

---

## 5. AMBIENTE PYTHON — venv con `--system-site-packages`

**Decisione:** usiamo un venv (come da quick-start del corso) per riproducibilità e per pinnare torch,
MA creato con `--system-site-packages` così EREDITA i pacchetti pesanti dell'anaconda di sistema
(datasets, pandas, numpy, pyarrow, pillow, huggingface_hub) invece di riscaricarli.

**Perché `--system-site-packages`:** installare da zero scipy/sklearn/pandas/... nel venv aveva dato
`Disk quota exceeded` — NON per i GB (eravamo al 14%), ma per il **numero di file (inode)** della home NFS.
Ereditando i pacchetti base, il venv resta leggero → niente problema di quota. Ci installiamo SOLO il
nostro (torch & co.).

> ⚠️ **CONTROLLA SEMPRE `(base)` nel prompt** prima di creare/usare il venv. Se manca `(base)`, `python`
> è il Python 2 di sistema. Recupero:
> 1. `module avail 2>&1 | grep -iE 'anaconda|conda'` → se c'è un modulo, `module load <nome>`;
> 2. altrimenti: `source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-*/etc/profile.d/conda.sh && conda activate base`
> 3. **per renderlo automatico ai login futuri:** una volta che `conda` funziona, lancia `conda init bash`
>    (scrive l'attivazione in `~/.bashrc` → `(base)` parte da solo nei prossimi terminali).

```bash
# crea il venv (FUORI dal repo, posizione della doc) — con (base) attivo
python -m venv --system-site-packages /homes/mbaracchi/cvcs2026/venv

# attiva (ogni sessione)
source /homes/mbaracchi/cvcs2026/venv/bin/activate     # prompt → (venv)

pip install --upgrade pip
python -c "import datasets, pandas, pyarrow, PIL, huggingface_hub; print('ok')"   # base visibile

# installare nostri pacchetti specifici (es. torch quando servirà), leggeri grazie a --no-cache-dir:
pip install --no-cache-dir <pkg>

squota          # controllare che la quota file non esploda
```
- Venv: `/homes/mbaracchi/cvcs2026/venv` (FUORI dalla repo → non c'entra col .gitignore).
- PyCharm SSH interpreter → puntare a `/homes/mbaracchi/cvcs2026/venv/bin/python`.
- `requirements.txt` (su git) = ciò che installiamo NOI sopra la base (per ora ~niente; torch poi).

**Nota torch/CUDA:** Installare nel venv con `pip install --no-cache-dir torch torchvision`. 
* **Perché via pip nel venv e non via moduli?** I moduli di PyTorch forniti dall'amministratore (Spack) sono compilati per i nodi di calcolo GPU e **non sono visibili o caricabili sul login node** (`ailb-login-02`). Installarlo localmente nel venv con `pip` garantisce che il codice si compili ed esegua ovunque.
* **⚠️ Warning CUDA sul login node:** Quando si esegue `torch.cuda.is_available()` sul login node, esso restituirà `False` e mostrerà un avviso inerente i driver NVIDIA obsoleti (`found version 11040`). Questo comportamento è **normale**: il login node non ha driver grafici attivi. Quando lo script girerà su nodo di calcolo (via SLURM), CUDA funzionerà a pieno regime.

---

## 6. SLURM — eseguire sui nodi di calcolo (specie con GPU)


**Concetto:** il login node è condiviso e **non va usato per calcolo pesante / GPU**. Per girare modelli
si chiede un nodo di calcolo a SLURM (lo scheduler), che mette in coda e assegna risorse.

### Quando serve SLURM
| Attività | Dove |
|---|---|
| Ispezionare un parquet, scaricare file piccoli, comandi leggeri | Login node (ok) |
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
| `sudo ...` / `apt install ...` | Non sei root, dà errore | `module load <pkg>` o `pip install --user` |
| `module purge` | Rimuove l'anaconda di default → perdi conda/python3, torni a Python 2 | terminale nuovo, o ricarica il modulo anaconda |
| Girare training/inferenza GPU sul **login node** | È condiviso, ti bloccano | Job SLURM su nodo di calcolo |
| `du -sh /work/cvcs2026` o `find /work` ampi | Sovraccarica BeeGFS per tutti | `squota` per le quote; `find` con `-maxdepth` |
| `ls -lR /work/...` ricorsivo enorme | Stessa cosa (metadata storm) | scope ristretto, `-maxdepth` |
| Scompattare un dataset in milioni di file su `/work` | Degrada BeeGFS | tenere parquet/tar/LMDB, leggere in streaming |
| Creare un venv SENZA `--system-site-packages` | Reinstalla migliaia di file → `Disk quota exceeded` (inode) | venv con `--system-site-packages` (eredita la base) |
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

# git → si fa dal MAC in locale (PyCharm Git o terminale del Mac), NON sul server.
# Il server riceve i file via PyCharm Deployment, non gli serve git.

# python: venv con --system-site-packages (vedi §5)
source /homes/mbaracchi/cvcs2026/venv/bin/activate
python -c "import datasets; print('ok')"
pip install --no-cache-dir <pkg>

# slurm
sbatch script.sh
squeue --me
scancel <jobid>

# disco
squota                   # quota (NON du su /work intero)
```

---

## DA VERIFICARE / TODO server
- [x] Nessun venv necessario: il cluster ha anaconda con tutti i pacchetti base già installati (Smarcato: ereditati via `--system-site-packages`).
- [x] Trovare il path esatto del Python (Smarcato: `/homes/mbaracchi/cvcs2026/venv/bin/python` e python3.10 come base venv).
- [ ] Nome esatto della partition GPU per l'account `cvcs2026` (all_usr_prod?) e limiti (tempo/numero GPU)
- [ ] Esiste una sotto-quota per `deep_pixels` o si pesca dai 4 TB comuni del corso? (chiedere ai tutor)
- [x] Conviene PyTorch da modulo o da pip nel venv? (Smarcato: Usato `pip` nel venv per compatibilità con il login node, poiché i moduli non sono disponibili lì).
