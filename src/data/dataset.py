# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class UnifiedDeepfakeDataset(Dataset):
    def __init__(self, manifest_path, openfake_parquet_dir=None, d3_parquet_dir=None, transform=None):
        """
        Dataloader unificato per la valutazione dei detector di deepfake.
        Legge il manifest.csv ed estrae in modo trasparente le immagini da file fisici o da Parquet.
        
        Args:
            manifest_path (str/Path): Percorso al manifest.csv del dataset.
            openfake_parquet_dir (str/Path): Directory contenente i file parquet di OpenFake (richiesto per OpenFake).
            d3_parquet_dir (str/Path): Directory contenente i file parquet di D3 (richiesto per fakes di D3).
            transform (callable): Trasformazioni torchvision da applicare all'immagine.
        """
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        
        self.openfake_parquet_dir = openfake_parquet_dir
        self.d3_parquet_dir = d3_parquet_dir
        
        # Lettura Lazy (inizializzata al primo caricamento nei thread del dataloader)
        self.openfake_dataset = None
        self.d3_dataset = None

    def _get_openfake_dataset(self):
        import datasets
        if self.openfake_dataset is None:
            if not self.openfake_parquet_dir:
                raise ValueError("Errore: openfake_parquet_dir è richiesto per caricare dati da OpenFake.")
            self.openfake_dataset = datasets.load_dataset(
                "parquet", 
                data_files=os.path.join(self.openfake_parquet_dir, "*.parquet"), 
                split="train"
            )
        return self.openfake_dataset

    def _get_d3_dataset(self):
        import datasets
        if self.d3_dataset is None:
            if not self.d3_parquet_dir:
                raise ValueError("Errore: d3_parquet_dir è richiesto per caricare le fake di D3.")
            self.d3_dataset = datasets.load_dataset(
                "parquet", 
                data_files=os.path.join(self.d3_parquet_dir, "*.parquet"), 
                split="train"
            )
        return self.d3_dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # idx rappresenta l'indice di riga del file manifest.csv (da 0 a N-1)
        row = self.df.iloc[idx]
        dataset = row["dataset"]
        label = int(row["label"])
        generator = row["generator"]
        
        # 1. CARICAMENTO IMMAGINE DALLA SORGENTE CORRETTA
        img = None
        
        if dataset == "forensynth":
            # GAN: Caricamento standard da file PNG nativo
            img = Image.open(row["path"]).convert("RGB")
            
        elif dataset == "d3":
            if label == 0:
                # D3 Reale: Caricamento da file immagine scaricata da LAION
                img = Image.open(row["path"]).convert("RGB")
            else:
                # D3 Fake: Caricamento dal Parquet originale
                d3_ds = self._get_d3_dataset()
                parquet_idx = int(row["index"])
                
                # Mappatura colonne: gen0=deepfloyd | gen1=sd14 | gen2=sd21 | gen3=sdxl
                gen_col_map = {
                    "deepfloyd": "image_gen0", 
                    "sd14": "image_gen1", 
                    "sd21": "image_gen2", 
                    "sdxl": "image_gen3"
                }
                col_name = gen_col_map[generator]
                
                raw_data = d3_ds[parquet_idx][col_name]
                if isinstance(raw_data, dict):
                    raw_data = raw_data.get("bytes")
                
                img = Image.open(io.BytesIO(raw_data)).convert("RGB")
                
        elif dataset == "openfake":
            # OpenFake (reals e fakes): Caricamento dal dataset Parquet
            openfake_ds = self._get_openfake_dataset()
            parquet_idx = int(row["index"])
            
            raw_data = openfake_ds[parquet_idx]["image"]
            if isinstance(raw_data, dict):
                raw_data = raw_data.get("bytes")
                
            img = Image.open(io.BytesIO(raw_data)).convert("RGB")

        # 2. APPLICAZIONE DEL PREPROCESSING
        if self.transform:
            img = self.transform(img)
            
        # Restituiamo anche i metadati per consentire analisi disaggregate per generatore/dataset
        return img, label, generator, dataset
