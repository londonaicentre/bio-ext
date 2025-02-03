import mlflow
from mlflow.models import infer_signature
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score


def load_and_prepare_data():
    dataset = load_dataset("imdb", split="train[:1000]")
    # print(dataset.to_pandas().head())
    dataset_dict = dataset.train_test_split(test_size=0.2)
    return dataset_dict["train"], dataset_dict["test"]


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
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def train_model(model, tokenizer, train_dataset, eval_dataset):
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    eval_tokenized = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer, train_tokenized


def main():
    experiment_name = "bert-binary-classification"
    mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print("Set up Experiment on MLflow")

    # load data
    train_dataset, eval_dataset = load_and_prepare_data()
    print("Loaded and prepared data")

    # start mlflow run
    with mlflow.start_run() as run:
        # load model
        model, tokenizer = prepare_model_and_tokenizer()
        print("Prepared model and tokenizer")

        # train
        print("Train model")
        trainer, train_tokenized = train_model(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
        )

        # sample input for model signature
        print("Prepare sample")
        sample_input = tokenizer(
            train_dataset["text"][0],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
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
            signature=infer_signature(sample_input, np.array([[0.1, 0.9]])),
        )

        # log metrics to mlflow
        print("Evaluation metrics")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
