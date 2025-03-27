import os
import zipfile
from datetime import datetime
from datasets import load_dataset, Dataset
from enum import Enum, auto
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import Dataset
from bioext.doccano_utils import DoccanoSession
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

"""
hfpipeline.py
This module abstracts Hugging Face Transformers NLP tasks into an easy to implement pipeline
Supports end-to-end workflow from data loading and preprocessing to model training and evaluation.

Key components:
- Data loading from annotation platforms (currently supports Doccano)
- Text tokenization and preprocessing for various NLP tasks
- Dataset splitting and preparation
- Model training with appropriate configurations
- Performance evaluation with task-specific metrics

Supported task types:
- Multi-class classification: Each text belongs to exactly one class
- Multi-label classification: Each text can belong to multiple classes
- TO DO: Named Entity Recognition
- TO DO: Question Answering

Usage example:
    ```
    # Imports
    from bioext.hfpipeline import GlobalConfig, DataHandler, HFSequenceClassificationTrainer
    from bioext.hfpipeline import DataSource, TaskType

    # Configure the pipeline
    config = GlobalConfig(
    doc_project_id=42,
    source=DataSource.DOCCANO,
    task=TaskType.MULTICLASS,
    num_labels=3,
    model_name="distilbert-base-uncased"
    )

    # Load and preprocess data
    data_handler = DataHandler(config=config)

    # Set up and run training
    trainer = HFSequenceClassificationTrainer(config=config, tokenizer=data_handler.tokenizer)
    trainer.setup_trainer(data_handler.train_dataset, data_handler.test_dataset)
    metrics = trainer.train()
    ```
"""

class DataSource(Enum):
    """
    Enum for supported data sources
    """
    DOCCANO = auto()
    EMPTY = auto()

class TaskType(Enum):
    """
    Enum for supported NLP task types with corresponding HF problem_type
    """
    MULTICLASS = "single_label_classification"  # incl. binary classification
    MULTILABEL = "multi_label_classification"
    NER = "token_classification"  # TO DO: will need special handling

@dataclass
class GlobalConfig:
    """
    Global configuration for data handling and model training, including:
    - Data source (which project, source type)
    - Task-specific (task type, number of labels)
    - Model selection (which pre-trained model to use)
    - Tokenization parameters (padding, truncation, etc.)
    - Dataset preparation (test/train split ratio)
    - Training hyperparameters (batch size, learning rate, etc.)
    """
    # Data source
    doc_project_id: int = 1
    source: DataSource = DataSource.DOCCANO

    # Task type
    task: TaskType = TaskType.MULTICLASS
    num_labels: int = 2  # number of unique labels that can be assigned

    # Model
    model_name: str = "bert-base-uncased"

    # Tokenizer
    padding: str = "max_length"
    max_length: int = 512
    truncation: bool = True
    return_tensors: str = "pt"
    add_special_tokens: bool = True
    return_token_type_ids: bool = False
    return_attention_mask: bool = True

    # Dataset split
    test_size: float = 0.2

    # Training
    output_dir: str = "outputs"
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optimizer_type: str = "adamw_torch"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    fp16: bool = False  # Whether to use mixed precision training
    threshold: float = 0.5  # For multi-label classification

    @property
    def problem_type(self) -> str:
        """
        Get the HuggingFace problem_type corresponding to the task
        """
        return self.task.value


"""
DataHandler
Manages complete data pipeline from raw annotated data to training-ready datasets.

Handles data import:
- Connects to annotation sources (currently Doccano)
- Downloads and extracts annotated data
- Converts to HuggingFace Dataset format

Preprocessing:
- Implements task-specific label processing:
    - For MULTICLASS: Extract first label, encode to integers, rename to 'labels'
    - For MULTILABEL: Convert label lists to multi-hot vectors in 'labels'
    - (TODO) NER: Process character spans to token classification format
- Performs text tokenization with appropriate model tokenizer
- Handles dataset formatting for HuggingFace Trainer compatibility

Dataset Management:
- Splits data into training and test sets
- Provides access to raw, processed, and split datasets
"""

class DataHandler:
    """
    Pipeline from raw annotated data to training-ready datasets
    """
    def __init__(self, tokenizer=None, config: Optional[GlobalConfig] = None):
        self.config = config or GlobalConfig()

        # create tokenizer from model_name or argument
        if tokenizer is None:
            if not self.config.model_name:
                raise ValueError("config.model_name or a custom tokenizer must be provided")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.tokenizer = tokenizer

        self.raw_dataset = None
        self.project_type = None
        self.tokenized_dataset = None
        self.train_dataset = None
        self.test_dataset = None

        # load data
        if self.config.source == DataSource.DOCCANO:
            if self.config.doc_project_id is None:
                raise ValueError("doc_project_id must be provided when source is DOCCANO")
            self.raw_dataset, self.project_type = self.load_data_from_doccano()
        elif self.config.source == DataSource.EMPTY:
            self.raw_dataset = Dataset.from_dict({"text": [], "label": []})
            self.project_type = None
        else:
            raise ValueError(f"Unsupported data source: {self.config.source}")

        # tokenise and split into train/test
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

        # we can read the name of the file from inside the zip
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            jsonl_file = next((f for f in zip_ref.namelist() if f.endswith('.jsonl')), None)
            if not jsonl_file:
                raise FileNotFoundError("No JSONL file found in the downloaded archive")
            zip_ref.extractall("data/")

        os.remove(zip_file)

        unzipped_file = os.path.join("data", jsonl_file)
        dataset = load_dataset("json", data_files=unzipped_file, split="train")

        datestamp = datetime.now().astimezone().strftime("%Y%m%d")
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
        labeled_dataset = self.raw_dataset.select_columns(["text", "label"])

        if self.config.task == TaskType.NER:
            pass # TO DO: handle multiple spans, convert from char indexing to token indexing

        elif self.config.task == TaskType.MULTILABEL:
            all_labels = set()
            for example in labeled_dataset:
                all_labels.update(example["label"])
            all_labels = sorted(list(all_labels))

            # label to index mapping
            self.label2id = {label: i for i, label in enumerate(all_labels)}
            self.id2label = {i: label for i, label in enumerate(all_labels)}

            # make sure number of labels match
            assert self.config.num_labels == len(all_labels), (
                f"Config specifies {self.config.num_labels} labels, but found {len(all_labels)}."
                f"Please update your configuration to match."
            )

            def _convert_to_multilabel(example):
                """
                Convert label list to multi-hot encoding
                """
                # initialise 0.0 (float) vector of label length
                labels = [0.0] * self.config.num_labels

                # one hot encode each present label
                for label in example["label"]:
                    if label in self.label2id:
                        labels[self.label2id[label]] = 1.0

                example["labels"] = labels
                return example

            # apply multilabel encoding
            encoded_dataset = labeled_dataset.map(_convert_to_multilabel)

            # remove 'label' column
            if "label" in encoded_dataset.column_names:
                encoded_dataset = encoded_dataset.remove_columns(["label"])

            # tokenise
            tokenized_dataset = encoded_dataset.map(
                lambda x: self.tokenize_function(x),
                batched=True,
            )
            tokenized_dataset.set_format("torch")

            return tokenized_dataset

        elif self.config.task == TaskType.MULTICLASS:
            def _extract_label(example):
                """
                For multi-class only, extract the first label from Doccano
                """
                example["label"] = example["label"][0]
                return example

            labeled_dataset = labeled_dataset.map(_extract_label)
            encoded_dataset = labeled_dataset.class_encode_column("label")
            print(encoded_dataset.features)

            # rename column to "labels" for HF Trainer
            encoded_dataset = encoded_dataset.rename_column("label", "labels")

            tokenized_dataset = encoded_dataset.map(
                lambda x: self.tokenize_function(x),
                batched=True,
            )
            tokenized_dataset.set_format("torch")

            return tokenized_dataset

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


"""
HFSequenceClassifier
Sets up a Huggingface trainer class with a task, base model, and training configurations

Multi-Class Classification:
- Loss and activation baked into problem_type = "single_label_classification"
- Each sequence of text belongs to exactly one class
- Two classes or more
- Typically uses Cross-Entropy Loss (nn.CrossEntropyLoss)
- Uses softmax activation (built into model) to standardise across probas
- Predictions should use argmax, i.e. select highest probability class
- Example format:
    ```
    dataset = {
        'text': ["This movie is great", "The service was terrible", "The experience was average"],
        'label': [0, 1, 2]  # single integer label per example
    }
    ```
- Metrics: precision, recall, f1, accuracy (using predicted class)

Multi-Label Classification:
- Loss and activation baked into problem_type = "multi_label_classification"
- Each example can belong to multiple classes simultaneously
- Typically Binary Cross-Entropy (nn.BCEWithLogitsLoss) > average of indepdent loss per prediction
- Uses sigmoid activation (1/(1+e^-x)) to get independent probabilities per label
- Predictions should use a threshold (e.g. 0.5) to determine which classes are positive
- Example format:
    ```
    dataset = {
        'text': ["Amazing thriller with lots of action", "Great comedy with mediocre sci-fi elements"],
        'labels': [[1, 0, 0], [1, 0, 1]]  # binary vector per example
    }
    ```
- Metrics: micro-averaged f1, macro F1, precision, recall, Jaccard index, accuracy (comparing binary vectors)
"""

class HFSequenceClassificationTrainer:
    """
    Trainer specifically for sequence classification tasks.
    Supports both multi-class and multi-label classification.
    """

    def __init__(self, config: Optional[GlobalConfig] = None, tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        Initialize the trainer with global config
        """
        self.config = config or GlobalConfig()
        self.device = self._choose_device()

        # reuse tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            model_max_length=self.config.max_length,
        )

        # init model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type=self.config.problem_type,
        )
        self.model.to(self.device)
        self.trainer = None

    def _choose_device(self) -> torch.device:
        """
        Choose the best available device for training: CUDA, MPS, or CPU (in order of preference).
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for classification evaluation.
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if self.config.task == TaskType.MULTILABEL:
            # Multi-label classification: apply sigmoid and threshold
            predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid
            predictions = (predictions > self.config.threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="micro", zero_division=0
            )
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                labels, predictions, average="macro", zero_division=0
            )
            metrics_dict = {
                "accuracy": accuracy,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        else:
            # Multi-class classification: apply argmax
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                labels, predictions, average="weighted", zero_division=0
            )
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                labels, predictions, average="macro", zero_division=0
            )

            # Confusion matrix
            if self.config.num_labels <= 10:
                conf_matrix = confusion_matrix(labels, predictions)
                conf_matrix_dict = {
                    f"cm_{i}_{j}": float(conf_matrix[i][j])
                    for i in range(conf_matrix.shape[0])
                    for j in range(conf_matrix.shape[1])
                }
            else:
                conf_matrix_dict = {}

            metrics_dict = {
                "accuracy": accuracy,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                **conf_matrix_dict,
            }

        return metrics_dict

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> Trainer:
        """
        Set up the Hugging Face Trainer with pre-tokenized datasets.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            evaluation_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=f"eval_{self.config.metric_for_best_model}",
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        return self.trainer

    def train(self) -> Dict[str, float]:
        """
        Run training and return metrics.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")

        train_result = self.trainer.train()
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        return train_result.metrics

    # Accessors
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get the trainer's tokenizer
        """
        return self.tokenizer

    def get_model(self) -> PreTrainedModel:
        """
        Get the trainer's model
        """
        return self.model