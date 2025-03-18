import os
import zipfile
from datetime import datetime
from datasets import load_dataset, Dataset
from typing import Optional
from enum import Enum, auto
from transformers import AutoTokenizer
from bioext.doccano_utils import DoccanoSession
from dataclasses import dataclass

# from dotenv import load_dotenv ## let's keep this outside of utils
# # Load credentials from env file
# load_dotenv()


### *** add data source types
class DataSource(Enum):
    """
    Enum for supported data sources
    """
    DOCCANO = auto()
    EMPTY = auto()

### *** add NLP task types
class TaskType(Enum):
    """
    Enum for supported NLP task types
    """
    MULTICLASS = "multiclass" # where num_labels = 2, this is binary classification
    MULTILABEL = "multilabel"
    NER = "NER"

### *** Added configuraiton object for modularity
@dataclass
class DataHandlerConfig:
    ## data source config
    doc_project_id: int = 1 ### *** renamed to be more explicit and consistent with doccano_utils
    source: DataSource = DataSource.DOCCANO

    ## task config
    task: TaskType = TaskType.MULTICLASS
    num_labels: int = 2 ## *** set to 2 for default binary task

    ## tokenizer config
    model_name: str = "bert-base-uncased"
    padding: str = "max_length"
    max_length: int = 512
    truncation: bool = True
    return_tensors: str = "pt"
    add_special_tokens: bool = True
    return_token_type_ids: bool = False
    return_attention_mask: bool = True

    ## test/train split
    test_size: float = 0.2

class DataHandler:
    """
    Pipeline from raw annotated data to training-ready datasets
    """
    def __init__(self, tokenizer=None, config: Optional[DataHandlerConfig] = None):
        self.config = config or DataHandlerConfig()

        # create tokenizer from model_name or argument
        if tokenizer is None:
            if not self.config.model_name:
                raise ValueError("config.model_name or a custom tokenizer must be provided")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.tokenizer = tokenizer

        ### *** initialise empty result attributes
        self.raw_dataset = None
        self.project_type = None
        self.tokenized_dataset = None
        self.train_dataset = None
        self.test_dataset = None

        # *** with typing and some error handling
        if self.config.source == DataSource.DOCCANO:
            if self.config.doc_project_id is None:
                raise ValueError("doc_project_id must be provided when source is DOCCANO")
            self.raw_dataset, self.project_type = self.load_data_from_doccano()
        elif self.config.source == DataSource.EMPTY:
            self.raw_dataset = Dataset.from_dict({"text": [], "label": []})
            self.project_type = None
        else:
            raise ValueError(f"Unsupported data source: {self.config.source}")

        self.tokenized_dataset = self.preprocess_dataset()
        self.train_dataset, self.test_dataset = self.split_dataset()


    def load_data_from_doccano(self):
        """
        Load data from a Doccano project
        """
        doc_session = DoccanoSession()
        doc_project_id = doc_session.client.find_project_by_id(self.config.doc_project_id)

        zip_file = doc_session.client.download(
            project_id=self.config.doc_project_id,
            format="JSONL",
            only_approved=False,
            dir_name="data",
        )

        # print(type(zip_file))  # pathlib.PosixPath
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_file)

        datestamp = datetime.now().astimezone().strftime("%Y%m%d")
        unzipped_file = "data/doccanoadmin.jsonl" ### *** my exports are called doccanoadmin?
        dataset = load_dataset("json", data_files=unzipped_file, split="train")

        os.rename(unzipped_file, f"data/{doc_project_id.name}_{datestamp}.jsonl")

        return dataset, doc_project_id.project_type

    def tokenize_function(self, examples):
        """
        Preprocessing function that tokenizes and truncates each example
        """
        return self.tokenizer(
            examples["text"],
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_token_type_ids=self.config.return_token_type_ids,
            return_attention_mask=self.config.return_attention_mask,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens,
        )

    def preprocess_dataset(self):
        """
        Preprocess a loaded dataset by normalising labels and tokenising text.

        Returns:
            Dataset: a tokenized HF dataset.
        """
        labeled_dataset = self.raw_dataset.select_columns(["text", "label"]) ### *** the columns to remove could change

        if self.config.task == TaskType.NER:
            pass # TO DO: handle multiple spans, convert from char indexing to token indexing
        elif self.config.task == TaskType.MULTILABEL:
            pass # TO DO
        elif self.config.task == TaskType.MULTICLASS:

            def _extract_label(example):
                """
                For multi-class only, extract the first label from Doccano
                """
                example["label"] = example["label"][0]
                return example

            labeled_dataset = labeled_dataset.map(_extract_label)
            # labels expected to be an integer!
            # use a LabelEncoder or similar after export from Doccano
            encoded_dataset = labeled_dataset.class_encode_column("label")
            print(encoded_dataset.features)

            tokenized_dataset = encoded_dataset.map(
                lambda x: self.tokenize_function(x),
                batched=True,
            )
            tokenized_dataset.set_format("torch")

            return tokenized_dataset
        else:
            print("Task type must be set")
            return None

    def split_dataset(self):
        """
        Create a "train" and "test" split from a HF dataset.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing train and test datasets.
        """
        # train test split shuffles by default
        dataset_dict = self.tokenized_dataset.train_test_split(test_size=self.config.test_size)
        # print(type(dataset_dict["train"]))  # <class 'datasets.arrow_dataset.Dataset'>
        return dataset_dict["train"], dataset_dict["test"]

