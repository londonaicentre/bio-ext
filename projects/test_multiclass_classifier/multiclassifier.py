import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch import cuda
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import mlflow
from mlflow.models import infer_signature

# TODO: refactor from pandas to hugginface datasets
# TODO: Consider class weighted splitting
# TODO: tidy up re: infer signature handlign missing data
# TODO: tidy up MPS / CUDA etc

# Declare device
# device = "cuda" if cuda.is_available() else "cpu"

# Declare constants TODO: should refactor in config file
DATA_PATH = "data/breast_brca_labelled.json"
FRAC = 1
RANDOM_STATE = 42
TRAIN_TEST_SPLITPERC = 0.8
CHOSEN_MODEL = "bert-base-uncased"
BATCH_SIZE = 8
SAVELOCATION = "mclassout"


# sense check NEED TO GO TO MLFLOW
# df["label"].value_counts()
# df["category"].value_counts()
# cols = ds["train"].column_names


def load_process_pretokeniser(path):
    """read the data, generate labelsdict, id2label,label2id

    Args:
        path (_type_, optional): _description_. Defaults to DATA_PATH.
    """
    df = pd.read_json(path)
    df["label"] = df["label"].astype("category")
    labels = list(df["label"].unique())
    num_labels = len(labels)
    labelsdict = {key: value for key, value in enumerate(labels)}
    id2label = labelsdict
    label2id = {value: idx for idx, value in id2label.items()}
    df["category"] = df["label"].apply(lambda x: label2id[x])
    df["category"] = df["category"].astype("int")
    df.columns = ["text", "category", "labels"]
    print(f"preprocessing complete")
    return df, labels, num_labels, id2label, label2id


def create_hfds(data, frac, random_state, splitfrac):
    """_summary_

    Args:
        data (_type_): _description_
        frac (_type_, optional): _description_. Defaults to FRAC.
        random_state (_type_, optional): _description_. Defaults to RANDOM_STATE.
        splitfrac (_type_, optional): _description_. Defaults to TRAIN_TEST_SPLITPERC.

    Returns:
        _type_: _description_
    """
    print(
        f"creating training and testing data by shuffling the {frac} with a random state of {random_state}"
    )
    data_shuffled = data.sample(frac=frac, random_state=random_state).reset_index(
        drop=True
    )
    split_index = int(splitfrac * len(data_shuffled))
    train_df = data_shuffled.iloc[:split_index]
    test_df = data_shuffled.iloc[split_index:]

    ds = DatasetDict()
    train = Dataset.from_pandas(train_df)
    val = Dataset.from_pandas(test_df)
    ds["train"] = train
    ds["validation"] = val
    print(f"training and testing data generated as Huggingface format")
    return ds


def prepare_model_and_tokeniser(model_name, numberoflabels, l2i, i2l):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=numberoflabels, label2id=l2i, id2label=i2l
    )
    return tokenizer, model


def tokenize_encode(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(predictions), dim=-1)
    labels = torch.from_numpy(labels)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    conf_matrix = confusion_matrix(labels, predictions)

    conf_matrix_dict = {
        f"cm_{i}_{j}": conf_matrix[i][j]
        for i in range(conf_matrix.shape[0])
        for j in range(conf_matrix.shape[1])
    }

    return {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        **conf_matrix_dict,
    }


def train_model(
    model,
    training_dataset,
    evaluation_dataset,
    metrics=compute_metrics,
):

    trainingargs = TrainingArguments(
        output_dir="mclassout",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        use_cpu=True,
    )

    trainer = Trainer(
        model,
        args=trainingargs,
        train_dataset=training_dataset,
        eval_dataset=evaluation_dataset,
        compute_metrics=metrics,
    )
    trainer.train()
    return trainer


def main():
    experiment_name = "bert_multiclass-singlelabel-classification"
    mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print("ML Flow experiement set up ")

    df, labels, num_labels, id2label, label2id = load_process_pretokeniser(
        path=DATA_PATH
    )
    print("Data is read, labels dict generated.")

    ds = create_hfds(
        data=df, frac=FRAC, random_state=RANDOM_STATE, splitfrac=TRAIN_TEST_SPLITPERC
    )
    print("Hugginface dataset generated")

    with mlflow.start_run() as run:

        tokenizer, model = prepare_model_and_tokeniser(
            model_name=CHOSEN_MODEL,
            numberoflabels=num_labels,
            l2i=label2id,
            i2l=id2label,
        )
        ds_enc = ds.map(
            tokenize_encode,
            fn_kwargs={"tokenizer": tokenizer},  # NOTE this!
            batched=True,
            remove_columns=["category", "text"],
        )
        trainer = train_model(
            model=model,
            training_dataset=ds_enc["train"],
            evaluation_dataset=ds_enc["validation"],
            metrics=compute_metrics,
        )

        print("Prepare sample")
        sample_input = ds_enc["train"][0]

        # Log model
        print("Log training parameters")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer,
            },
            artifact_path="bert_model",
            task="text-classification",  # NOTE:DO not change this!
            signature=infer_signature(sample_input, torch.tensor([[0.1, 0.9]])),
        )
        trainer.model = trainer.model.cpu()

        # model.to(torch.device("mps"))

        trainermetrics = trainer.evaluate()
        mlflow.log_metrics(trainermetrics)


if __name__ == "__main__":
    main()
