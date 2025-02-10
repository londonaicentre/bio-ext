import mlflow
from mlflow.models import infer_signature
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from bioext.doccano_utils import DoccanoSession, save_labelled_docs
import numpy as np
from sklearn.metrics import accuracy_score

from dotenv import load_dotenv

# Load credentials from env file
load_dotenv()


def prepare_model_and_tokenizer(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    return model, tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )


def load_data():
    # Initialise connection to Doccano
    doc_session = DoccanoSession()
    project_id = 1

    # labelled_samples = save_labelled_docs(doc_session, project_id)
    # dataset = Dataset.from_generator(labelled_samples)

    # zipfile = doc_session.client.download(
    #     project_id=project_id,
    #     format="JSONL",
    #     only_approved=False,
    #     dir_name="data",
    # )
    # need to unzip
    unzipped_file = "data/admin.jsonl"

    return load_dataset("json", data_files=unzipped_file)


def select_label(ex):
    ex["labels"] = ex["label"][0]
    return ex


def load_and_prepare_data(unzipped_file, tokenizer):
    raw_dataset = load_dataset("json", data_files=unzipped_file, split="train")
    single_label_dataset = raw_dataset.map(select_label)
    cleaned_dataset = single_label_dataset.remove_columns(
        ["id", "source_id", "Comments", "label"]
    )
    encoded_dataset = cleaned_dataset.class_encode_column("labels")
    tokenized_dataset = encoded_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    tokenized_dataset.set_format("torch")
    dataset_dict = tokenized_dataset.train_test_split(test_size=0.2)
    # print(type(dataset_dict["train"]))  # <class 'datasets.arrow_dataset.Dataset'>
    return dataset_dict["train"], dataset_dict["test"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def train_model(model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    print("actual training begins")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def main():
    # load model
    model_name = "bert-base-uncased"
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    print("Prepared model and tokenizer")

    # load data
    unzipped_file = "data/admin.jsonl"  # "data/breast_brca_binary_labelled.json"
    # labels expected to be an integer!
    # use a LabelEncoder or similar at export from Doccano
    train_dataset, eval_dataset = load_and_prepare_data(unzipped_file, tokenizer)
    print("Loaded, preprocessed and tokenised data")

    experiment_name = "brca-binary-classification"
    mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print("found exp")
    mlflow.set_experiment(experiment_name)
    print("Set up Experiment on MLflow")

    # start mlflow run
    with mlflow.start_run() as run:
        # train
        print("Train model")
        trainer = train_model(
            model,
            train_dataset,
            eval_dataset,
        )

        # log model to mlflow
        print("Log training params")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer,
            },
            artifact_path="bert_model",
            task="text-classification",
            signature=infer_signature(
                train_dataset[0],
                np.array(
                    [[0.1, 0.9]],
                ),
            ),
        )

        # log metrics to mlflow
        print("Evaluation metrics")
        metrics = trainer.evaluate()
        print(metrics)
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
