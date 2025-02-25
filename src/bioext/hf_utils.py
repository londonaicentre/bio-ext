import os
from typing import Dict, List, Optional
from dataclasses import dataclass
# from enum import Enum, auto
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    # AutoModelForTokenClassification,
    # AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedModel
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix
)

"""
hf_utils.py
Sets up a Huggingface trainer class with a task, base model, and training configurations

Multi-Class Classification:
* Loss and activation baked into problem_type = "single_label_classification"
* Each sequence of text belongs to exactly one class
* Two classes or more
* Typically uses Cross-Entropy Loss (nn.CrossEntropyLoss)
* Uses softmax activation (built into model) to standardise across probas
* Predictions should use argmax, i.e. select highest probability class
* Example format:
    ```
    dataset = {
        'text': ["This movie is great", "The service was terrible", "The experience was average"],
        'label': [0, 1, 2]  # single integer label per example
    }
    ```
* Metrics: precision, recall, f1, accuracy (using predicted class)

Multi-Label Classification:
* Loss and activation baked into problem_type = "multi_label_classification"
* Each example can belong to multiple classes simultaneously
* Typically Binary Cross-Entropy (nn.BCEWithLogitsLoss) > average of indepdent loss per prediction
* Uses sigmoid activation (1/(1+e^-x)) to get independent probabilities per label
* Predictions should use a threshold (e.g. 0.5) to determine which classes are positive
* Example format:
    ```
    dataset = {
        'text': ["Amazing thriller with lots of action", "Great comedy with mediocre sci-fi elements"],
        'labels': [[1, 0, 0], [1, 0, 1]]  # binary vector per example
    }
    ```
* Metrics: micro-averaged f1, macro F1, precision, recall, Jaccard index, accuracy (comparing binary vectors)
"""

# class TaskType(Enum):
#     """
#     Enum for supported NLP task types
#     """
#     SEQUENCE_CLASSIFICATION = auto()
#     TOKEN_CLASSIFICATION = auto()
#     QUESTION_ANSWERING = auto()

@dataclass
class HFTrainingConfig:
    """
    Dataclass holding configuration for a HF transformers TrainingArguments class
    See parameters here: https://huggingface.co/docs/transformers/en/main_classes/trainer
    """
    # task_type: TaskType  # enum ['SEQUENCE_CLASSIFICATION', 'TOKEN_CLASSIFICATION', 'QUESTION_ANSWERING']
    model_name: str  # HF Hub name, e.g. 'bert-base-uncased'
    num_labels: int = 2 # multi-class = number of classes; multi-label = number of possible labels (each 1 vs 0)
    tokenizer_name: Optional[str] = None  # If None uses model_name as default tokenizer
    problem_type: str = "single_label_classification"  # "single_label_classification" = multi-class; or "multi_label_classification"
    threshold: float = 0.5
    output_dir: str = "outputs"
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optimizer_type: str = "adamw_torch"
    max_seq_length: int = 512
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True # where metric_for_best_model is not a loss
    fp16: bool = False # default train sin 32bit

class HFSequenceClassificationTrainer:
    """
    Trainer specifically for sequence classification tasks
    Includes multi-class and multi-label functionality
    """

    def __init__(self, config: HFTrainingConfig):
        self.config = config
        self.device = self._choose_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            use_fast=True,
            model_max_length=config.max_seq_length
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            problem_type=config.problem_type
        )
        self.model.to(self.device)
        self.trainer = None

    def _choose_device(self):
        """
        Chooses the best available device for training: CUDA, MPS, or CPU (in order of preference)

        Returns:
            torch.device: Selected device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def tokenize_data(self, dataset: Dataset, text_column: str, label_column: str) -> Dataset:
        """
        Apply tokenisation to dataset
        Dataset is a HF class optimised for NLP tasks

        Args:
            dataset:
                The input dataset
            text_column:
                Name of the text column
            label_column:
                Name of the label column
        """
        # check columns
        if text_column not in dataset.column_names:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in dataset.column_names:
            raise ValueError(f"Label column '{label_column}' not found in dataset")

        def tokenize_example(examples):
            # tokenisation
            result = self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length
            )

            # copy labels to training dataframe
            result["labels"] = examples[label_column]
            return result

        tokenized_dataset = dataset.map(tokenize_example, batched=True)

        return tokenized_dataset

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for classification evaluation task
        Note that EvalPrediction is a default HF data structure, produced silently by Trainer
        This contains the output of Trainer, with predictions (logits) and label_ids (ground truth labels)
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if self.config.problem_type == "multi_label_classification":
            # multi-label classification
            predictions = 1 / (1 + np.exp(-predictions))
            predictions = (predictions > self.config.threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='micro', zero_division=0
            )
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                labels, predictions, average='macro', zero_division=0
            )
            metrics_dict = {
                'accuracy': accuracy,
                'micro_precision': precision,
                'micro_recall': recall,
                'micro_f1': f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
            }

            return metrics_dict

        else:
            # binary/multi-class classification
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)

            # weighted metrics for imbalanced classes)
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )

            #  macro metrics / unweighted average across classes
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                labels, predictions, average='macro', zero_division=0
            )

            # get confusion matrix
            if self.config.num_labels <= 10:
                conf_matrix = confusion_matrix(labels, predictions)
                conf_matrix_dict = {
                    f"cm_{i}_{j}": float(conf_matrix[i][j])
                    for i in range(conf_matrix.shape[0])
                    for j in range(conf_matrix.shape[1])
                }
            else:
                conf_matrix_dict = {}

            return {
                'accuracy': accuracy,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                **conf_matrix_dict
            }

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Trainer:
        """
        Set up the Trainer with datasets
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
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model=("eval_" + self.config.metric_for_best_model), # trainer automatically prefixes metrics with eval_
            greater_is_better=self.config.greater_is_better,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            #tokenizer=self.tokenizer, # deprecated
            compute_metrics=self.compute_metrics # callback per eval strategy (default by epoch)
        )

        return self.trainer

    def train(self) -> Dict[str, float]:
        """
        Run training and return metrics
        Placeholder before pipeline and MLFlow class
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")

        train_result = self.trainer.train()
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        return train_result.metrics # training statistics

    ### accessors

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get the active tokeniser

        Returns:
            PreTrainedtokenizer: The initialised tokenizer
        """
        return self.tokenizer

    def get_model(self) -> PreTrainedModel:
        """
        Get the active model

        Returns:
            PreTrainedModel: The initialised model
        """
        return self.model