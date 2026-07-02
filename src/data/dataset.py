# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class DirectParquetReader:
    def __init__(self, parquet_dir):
        import glob
        import pyarrow.parquet as pq
        self.parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.parquet_files:
            raise FileNotFoundError(f"Nessun file parquet trovato in {parquet_dir}")
            
        self.row_counts = []
        self.cumulative_rows = [0]
        
        total_rows = 0
        for f in self.parquet_files:
            pf = pq.ParquetFile(f)
            num_rows = pf.metadata.num_rows
            self.row_counts.append(num_rows)
            total_rows += num_rows
            self.cumulative_rows.append(total_rows)
            
        # Cache per il Row Group attivo
        self.current_file_idx = -1
        self.current_rg_idx = -1
        self.current_column = None
        self.current_rg_table = None
        self.current_rg_start_row = -1
        self.current_rg_end_row = -1
        
    def read_row(self, global_idx, column_name):
        import pyarrow.parquet as pq
        import bisect
        
        # 1. Trova il file parquet corrispondente all'indice globale
        file_idx = bisect.bisect_right(self.cumulative_rows, global_idx) - 1
        if file_idx < 0 or file_idx >= len(self.parquet_files):
            raise IndexError(f"Indice globale {global_idx} fuori intervallo per il dataset.")
            
        local_idx = global_idx - self.cumulative_rows[file_idx]
        
        # 2. Controlla se il row group ed il canale richiesto sono già in cache
        # (senza aprire il file se l'indice ricade nel range caricato)
        if (file_idx == self.current_file_idx and 
            column_name == self.current_column and 
            self.current_rg_table is not None and 
            self.current_rg_start_row <= local_idx < self.current_rg_end_row):
            
            rg_local_idx = local_idx - self.current_rg_start_row
            cell_value = self.current_rg_table.column(column_name)[rg_local_idx].as_py()
            return cell_value
            
        # 3. Cache MISS: apriamo il file e cerchiamo il row group
        file_path = self.parquet_files[file_idx]
        pf = pq.ParquetFile(file_path)
        
        rg_idx = -1
        current_rows = 0
        rg_start_row = -1
        rg_end_row = -1
        rg_local_idx = -1
        
        for i in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(i)
            num_rows = rg_meta.num_rows
            if current_rows + num_rows > local_idx:
                rg_idx = i
                rg_start_row = current_rows
                rg_end_row = current_rows + num_rows
                rg_local_idx = local_idx - current_rows
                break
            current_rows += num_rows
            
        if rg_idx == -1:
            raise IndexError(f"Indice locale {local_idx} fuori intervallo nel file {file_path}.")
            
        # Leggiamo solo la colonna di interesse per questo row group in memoria
        table = pf.read_row_group(rg_idx, columns=[column_name])
        
        # Aggiorniamo la cache
        self.current_file_idx = file_idx
        self.current_rg_idx = rg_idx
        self.current_column = column_name
        self.current_rg_table = table
        self.current_rg_start_row = rg_start_row
        self.current_rg_end_row = rg_end_row
        
        # Estrarre i byte o il dizionario dell'immagine
        cell_value = table.column(column_name)[rg_local_idx].as_py()
        return cell_value


class UnifiedDeepfakeDataset(Dataset):
    def __init__(self, manifest_path, openfake_parquet_dir=None, d3_parquet_dir=None, transform=None):
        """
        Dataloader unificato per la valutazione dei detector di deepfake.
        Legge il manifest.csv ed estrae in modo trasparente le immagini da file fisici o da Parquet.
        
        Args:
            manifest_path (str/Path): Percorso al manifest.csv del dataset.
            openfake_parquet_dir (str/Path): Directory contenente i file parquet di OpenFake.
            d3_parquet_dir (str/Path): Directory contenente i file parquet di D3.
            transform (callable): Trasformazioni torchvision da applicare all'immagine.
        """
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        
        self.openfake_parquet_dir = openfake_parquet_dir
        self.d3_parquet_dir = d3_parquet_dir
        
        # Lettori diretti PyArrow (inizializzati lazily nei thread del dataloader)
        self.openfake_reader = None
        self.d3_reader = None

    def _get_openfake_reader(self):
        if self.openfake_reader is None:
            if not self.openfake_parquet_dir:
                raise ValueError("Errore: openfake_parquet_dir è richiesto per caricare dati da OpenFake.")
            self.openfake_reader = DirectParquetReader(self.openfake_parquet_dir)
        return self.openfake_reader

    def _get_d3_reader(self):
        if self.d3_reader is None:
            if not self.d3_parquet_dir:
                raise ValueError("Errore: d3_parquet_dir è richiesto per caricare le fake di D3.")
            self.d3_reader = DirectParquetReader(self.d3_parquet_dir)
        return self.d3_reader

    def _load_image_from_raw(self, raw_data):
        if isinstance(raw_data, Image.Image):
            return raw_data.convert("RGB")
        elif isinstance(raw_data, dict) and "bytes" in raw_data:
            if raw_data["bytes"] is not None:
                return Image.open(io.BytesIO(raw_data["bytes"])).convert("RGB")
            elif raw_data["path"] is not None:
                return Image.open(raw_data["path"]).convert("RGB")
            else:
                raise ValueError("Errore: dati immagine vuoti (sia bytes che path sono None).")
        elif isinstance(raw_data, bytes):
            return Image.open(io.BytesIO(raw_data)).convert("RGB")
        else:
            return Image.open(io.BytesIO(raw_data)).convert("RGB")

    def _get_single_item(self, row):
        dataset = row["dataset"]
        label = int(row["label"])
        generator = row["generator"]
        
        if dataset == "forensynth":
            img = Image.open(row["path"]).convert("RGB")
        elif dataset == "d3":
            if label == 0:
                img = Image.open(row["path"]).convert("RGB")
            else:
                reader = self._get_d3_reader()
                parquet_idx = int(row["index"])
                gen_col_map = {
                    "deepfloyd": "image_gen0", 
                    "sd14": "image_gen1", 
                    "sd21": "image_gen2", 
                    "sdxl": "image_gen3"
                }
                col_name = gen_col_map[generator]
                raw_data = reader.read_row(parquet_idx, col_name)
                img = self._load_image_from_raw(raw_data)
        elif dataset == "openfake":
            reader = self._get_openfake_reader()
            parquet_idx = int(row["index"])
            raw_data = reader.read_row(parquet_idx, "image")
            img = self._load_image_from_raw(raw_data)
        else:
            raise ValueError(f"Dataset sconosciuto: {dataset}")
            
        return img, label, generator, dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        row = self.df.iloc[idx]
        img, label, generator, dataset = self._get_single_item(row)
        if self.transform:
            img = self.transform(img)
        return img, label, generator, dataset
