from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
from transformers import (
    Autotokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    PreTrainedtokenizer,
    PreTrainedModel
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class TaskType(Enum):
    """
    Enum for supported NLP task types
    """
    SEQUENCE_CLASSIFICATION = auto()
    TOKEN_CLASSIFICATION = auto()
    QUESTION_ANSWERING = auto()

@dataclass
class HFTrainingConfig:
    """
    Dataclass holding configuration for a HF transformers TrainingArguments class
    See parameters here: https://huggingface.co/docs/transformers/en/main_classes/trainer
    """
    task_type: TaskType  # enum ['SEQUENCE_CLASSIFICATION', 'TOKEN_CLASSIFICATION', 'QUESTION_ANSWERING']
    model_name: str  # HF Hub name, e.g. 'bert-base-uncased'
    tokenizer_name: Optional[str] = None  # If None uses model_name as default tokenizer
    output_dir: str = "outputs"
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optimizer_type: str = "adamw_torch"
    max_seq_length: int = 512
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True # where metric_for_best_model is not a loss
    fp16: bool = False # default train sin 32bit
