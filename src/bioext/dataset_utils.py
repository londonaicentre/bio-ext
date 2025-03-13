from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
)
from bioext.doccano_utils import DoccanoSession, save_labelled_docs
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load credentials from env file
load_dotenv()

# Declare constants TODO: should refactor in config file
DATA_PATH = "data/breast_brca_labelled.json"
FRAC = 1
RANDOM_STATE = 42
TRAIN_TEST_SPLITPERC = 0.8


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


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True,
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


def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load data
    unzipped_file = "data/binary_BRCA.jsonl"  # "data/multiclass_BRCA.json"
    # labels expected to be an integer!
    # use a LabelEncoder or similar at export from Doccano
    train_dataset, eval_dataset = load_and_prepare_data(unzipped_file, tokenizer)
    print("Loaded, preprocessed and tokenised data")

    df, labels, num_labels, id2label, label2id = load_process_pretokeniser(
        path=DATA_PATH
    )
    print("Data is read, labels dict generated.")

    ds = create_hfds(
        data=df, frac=FRAC, random_state=RANDOM_STATE, splitfrac=TRAIN_TEST_SPLITPERC
    )
    print("Hugginface dataset generated")
