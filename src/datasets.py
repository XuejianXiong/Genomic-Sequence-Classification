
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer
import pytorch_lightning as pl


# =========================
# Functions
# =========================
def seq_to_kmers(seq: str, k: int = 6) -> str:
    return " ".join(
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
    )


def get_data_for_model(data_type: str, datafile: str) -> pd.DataFrame:
    
    # synthetic data: [sequence,label]
    # or
    # dhs data: [dhs_id chr start end DHS_width summit numsamples
    # total_signal component proportion sequence K562_ENCLB843GMH
    # hESCT0_ENCLB449ZZZ HepG2_ENCLB029COU GM12878_ENCLB441ZZZ TAG     
    # additional_replicates_with_peak other_samples_with_peak_not_considering_reps]

    if data_type == "synthetic":
        df = pd.read_csv(datafile)
        # remove any missing tags or sequences
        df_sub = df.dropna(subset=["sequence", "label"])

        label_map = {
            "non-enhancer": 0,
            "enhancer": 1,
        }

        df_sub["label"] = df_sub["label"].map(label_map)
        df_sub["id"] = [f"seq_{i}" for i in range(len(df_sub))]

    elif data_type == "dhs":
        df = pd.read_csv(datafile, sep="\t")
        df_sub = df[["dhs_id", "sequence", "chr", "TAG"]].copy()
        df_sub.rename(columns={"dhs_id": "id"}, inplace=True)
    
        # remove any missing tags or sequences
        df_sub = df_sub.dropna(subset=["sequence", "TAG"])

        # sanity check
        print(df_sub["TAG"].value_counts())
    
        label_map = {
            "K562_ENCLB843GMH": 0,
            "hESCT0_ENCLB449ZZZ": 1,
            "HepG2_ENCLB029COU": 2,
            "GM12878_ENCLB441ZZZ": 3,
        }

        df_sub["label"] = df_sub["TAG"].map(label_map)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # revert the keys and items in label_map
    label_map_trans = {v: k for k, v in label_map.items()}
    
    # remove any missing labels
    df_sub = df_sub.dropna(subset=["label"])
    # make sure the labels are integers 
    df_sub["label"] = df_sub["label"].astype(int)

    return(df_sub, label_map_trans)


# =========================
# Classes
# =========================
# Dataset class
class DNASequenceDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer_name: str,
        data_type: str,
        max_len: int = 256,
        kmer_size: int = 6,
        precompute_kmers: bool = True,
    ):
        super().__init__()

        # load + preprocess
        self.df, self.label_map_trans = get_data_for_model(data_type, csv_file)

        self.ids = self.df["id"].tolist()
        self.sequences = self.df["sequence"].tolist()
        self.labels = self.df["label"].astype(int).values

        self.max_len = max_len
        self.k = kmer_size
        self.precompute_kmers = precompute_kmers

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # -----------------------------
        # optional speed optimization
        # -----------------------------
        if self.precompute_kmers:
            self.kmers = [seq_to_kmers(seq, self.k) for seq in self.sequences]
        else:
            self.kmers = None

    def __len__(self):
        return len(self.labels)

    def _encode(self, kmers: str):
        encoding = self.tokenizer(
            kmers,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index):
        seq_kmers = (
            self.kmers[index]
            if self.kmers is not None
            else seq_to_kmers(self.sequences[index], self.k)
        )

        encoding = self._encode(seq_kmers)

        return {
            "id": self.ids[index],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }
    

# DataModule (Lightning standard)
class DNADataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file: str,
        tokenizer_name: str,
        data_type: str,
        batch_size: int = 32,
        split_mode: str = "random",  # "random" or "chromosome"
        train_chroms=None,
        val_chroms=None,
        test_chroms=None,
    ):
        super().__init__()

        self.csv_file = csv_file
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.data_type = data_type
        self.split_mode = split_mode

        self.train_chroms = train_chroms
        self.val_chroms = val_chroms
        self.test_chroms = test_chroms

    # -------------------------
    # MAIN SETUP
    # -------------------------
    def setup(self, stage=None):

        dataset = DNASequenceDataset(
            csv_file=self.csv_file,
            tokenizer_name=self.tokenizer_name,
            data_type=self.data_type,
            max_len=256
        )

        df = dataset.df
        labels = dataset.labels
        
        self.num_classes = len(dataset.label_map_trans)
        self.class_names = [dataset.label_map_trans[i] for i in range(self.num_classes)]
        
        indices = np.arange(len(dataset))

        # =========================================================
        # OPTION 1: RANDOM SPLIT (baseline only, NOT genomics-safe)
        # =========================================================
        if self.split_mode == "random":

            train_ind, test_ind = train_test_split(
                indices,
                test_size=0.2,
                stratify=labels,
                random_state=42
            )

            train_labels = labels[train_ind]

            train_ind, val_ind = train_test_split(
                train_ind,
                test_size=0.2,
                stratify=train_labels,
                random_state=42
            )

        # =========================================================
        # OPTION 2: CHROMOSOME SPLIT
        # =========================================================
        elif self.split_mode == "chromosome":

            if "chr" not in df.columns:
                raise ValueError("Chromosome split requires 'chr' column in dataset")

            chrom = df["chr"].values

            if self.train_chroms is None or self.val_chroms is None or self.test_chroms is None:
                raise ValueError(
                    "Please provide train_chroms, val_chroms, test_chroms"
                )

            train_ind = np.where(np.isin(chrom, self.train_chroms))[0]
            val_ind   = np.where(np.isin(chrom, self.val_chroms))[0]
            test_ind  = np.where(np.isin(chrom, self.test_chroms))[0]

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        # -------------------------
        # STORE SUBSETS
        # -------------------------
        self.train_data = Subset(dataset, train_ind)
        self.val_data = Subset(dataset, val_ind)
        self.test_data = Subset(dataset, test_ind)

        # -------------------------
        # DEBUG PRINTS
        # -------------------------
        print("\nSplit Summary:")
        print("Train:", pd.Series(labels[train_ind]).value_counts().to_dict())
        print("Val:", pd.Series(labels[val_ind]).value_counts().to_dict())
        print("Test:", pd.Series(labels[test_ind]).value_counts().to_dict())

    # -------------------------
    # DATALOADERS
    # -------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )