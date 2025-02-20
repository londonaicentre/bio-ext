import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import pytorch_lightning as pl
from multiclass_NLP.NLPDataset import NLPDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import mlflow

pd.set_option("future.no_silent_downcasting", True)


class NLPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        batch_size=8,
        max_token_len=256,
        num_workers=2,
        random_state=1,
    ):
        """Initialize the DataModule. Set batch size, data path, tokenizer,
        maximum token length, label columns, sample status and number of workers.

        Args:
            data_path (str): Path to the data file.
            tokenizer: Tokenizer to be used.
            batch_size (int, optional): Size of the data batches. Default is 8.
            max_token_len (int, optional): Maximum length of tokens. Default is 256.
            num_workers (int, optional): Number of workers. Default is 0.
            sample (bool, optional): If True, use WeightedRandomSampler. Default is False.
        """

        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.random_state = random_state
        self.df = None
        # self.train_df = None
        # self.val_df = None
        # self.label_columns = None
        self.train_dataset = None
        self.val_dataset = None
        # self.df = load_data(self.data_path)

    def setup(self, stage=None):
        """
        Set up the data module. Parse the data and create train and validation datasets.
        """
        print("Running DataModule setup")

        raw_dataset = load_dataset("json", data_files=self.data_path, split="train")
        single_label_dataset = raw_dataset.map(select_label)
        cleaned_dataset = single_label_dataset.remove_columns(
            ["id", "source_id", "Comments", "label"]
        )

        # label_enum = {k: j for j, k in enumerate(set(cleaned_dataset["labels"]))}
        # num_labels = len(label_enum)
        # cleaned_dataset["labels"] = cleaned_dataset["labels"].apply(
        #     lambda x: [1.0 if label_enum[x] == i else 0.0 for i in range(num_labels)]
        # )

        # print("after encoding")
        # print(cleaned_dataset[0])

        encoded_dataset = cleaned_dataset.class_encode_column("labels")
        class_label_feature = encoded_dataset.features["labels"]
        # casted_dataset = encoded_dataset.cast_column("labels", class_label_feature)
        # print(encoded_dataset.features)

        self.num_classes = encoded_dataset.features["labels"].num_classes

        tokenized_dataset = encoded_dataset.map(
            lambda x: tokenize_function(x, self.tokenizer, self.max_token_len),
            batched=True,
        )
        tokenized_dataset.set_format("torch")

        dataset_dict = tokenized_dataset.train_test_split(
            test_size=0.2,
            # random_state=self.random_state,
            # stratify=self.df["labels"],
        )
        # print(tokenized_dataset.features)
        self.df = tokenized_dataset.to_pandas()

        # if self.fold_indices:
        #     train_indices, val_indices = self.fold_indices
        #     self.train_df = self.df.iloc[train_indices]
        #     self.val_df = self.df.iloc[val_indices]
        # else:
        #     self.default_train_val_split()

        self.train_dataset = dataset_dict["train"]
        self.val_dataset = dataset_dict["test"]

    def train_dataloader(self):
        """
        Create and return a data loader for the training data.

        Returns:
            DataLoader: Data loader for the training data.
        """
        print("self.train_dataset")
        # print(self.train_dataset.to_pandas().head())
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Create and return a data loader for the validation data.

        Returns:
            DataLoader: Data loader for the validation data.
        """
        print("val_dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

    def steps_per_epoch(self):
        """
        Calculate and return the number of steps per epoch based on the batch size.

        Returns:
            int: Number of steps per epoch.
        """
        # print(len(self.train_dataset))
        # if self.train_dataset is None or len(self.train_dataset) == 0:
        #     self.parse_df_data()
        #     self.default_train_val_split()
        return len(self.train_dataset) // self.batch_size

    def dataset_stats(self, dataset) -> dict:
        """
        Calculate and return a dictionary of dataset statistics including label distribution and number of samples.

        Args:
            dataset (pandas.DataFrame): The dataset to analyze.

        Returns:
            dict: Dictionary of dataset statistics.
        """
        stats = {}
        stats["n_samples"] = len(dataset)
        label_counts = dataset.loc["ClassLabel"].value_counts()
        label_counts_dict = label_counts.to_dict()
        stats["label_counts"] = label_counts_dict
        return stats


def select_label(ex):
    ex["labels"] = ex["label"][0]
    return ex


def tokenize_function(examples, tokenizer, max_token_len):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        # max_length=512,
        max_length=max_token_len,
        # return_token_type_ids=False,
        # return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
