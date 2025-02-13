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

import mlflow
from mlflow.models import infer_signature

device = "cuda" if cuda.is_available() else "cpu"
# TODO: refactor from pandas to hugginface datasets
# TODO: Consider class weighted splitting

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


def tokenize_ds(ds, tokenizer):
    def tokenize_encode(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    ds_enc = ds.map(tokenize_encode, batched=True, remove_columns=["category", "text"])
    return ds_enc


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(predictions), dim=-1)
    labels = torch.from_numpy(labels)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}


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
        no_cuda=True,
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
        ds_enc = tokenize_ds(ds=ds, tokenizer=tokenizer)

        trainer = train_model(
            model=model,
            training_dataset=ds_enc["train"],
            evaluation_dataset=ds_enc["validation"],
            metrics=compute_metrics,
        )

        trainermetrics = trainer.evaluate()
        mlflow.log_metrics(trainermetrics)


if __name__ == "__main__":
    main()
