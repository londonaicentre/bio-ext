import os
import zipfile
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from bioext.doccano_utils import DoccanoSession
from dotenv import load_dotenv

# Load credentials from env file
load_dotenv()


class DataHandler:
    def __init__(
        self,
        source,
        doc_project: int,
        tokenizer,
        task="binary",
        num_labels: int = 1,
        padding: str = "max_length",
        max_length: int = 512,
        test_size: int = 0.2,
    ):
        self.tokenizer = tokenizer
        if source == "Doccano":
            self.raw_dataset, self.project_type = self.load_data_Doccano(doc_project)
        else:
            self.raw_dataset = Dataset()
        self.tokenized_dataset = self.preprocess_dataset(
            task, num_labels, padding, max_length
        )
        self.train_dataset, self.test_dataset = self.split_dataset(test_size)

    def load_data_Doccano(self, project_id: int = 1):
        """Load data from a Doccano project into a HF Dataset
        assuming authentication details are available as env variables

        Args:
            project_id (int, optional): Doccano project ID. Defaults to 1.

        Returns:
            Dataset: HF dataset
        """
        # Initialise connection to Doccano
        doc_session = DoccanoSession()
        doc_project = doc_session.client.find_project_by_id(project_id)

        zip_file = doc_session.client.download(
            project_id=project_id,
            format="JSONL",
            only_approved=False,
            dir_name="data",
        )

        # print(type(zip_file))  # pathlib.PosixPath
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_file)

        datestamp = datetime.now().astimezone().strftime("%Y%m%d")
        unzipped_file = "data/admin.jsonl"
        ds = load_dataset("json", data_files=unzipped_file, split="train")

        os.rename(unzipped_file, f"data/{doc_project.name}_{datestamp}.jsonl")

        return ds, doc_project.project_type

    def tokenize_function(self, examples, padding="max_length", max_length=512):
        return self.tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

    def select_single_label(self, ex):
        """Doccano exports labels always as a list, containing different type of elements based on the project_type.

        Args:
            ex (_type_): _description_

        Returns:
            _type_: _description_
        """
        ex["labels"] = ex["label"][0]
        return ex

    def preprocess_dataset(
        self,
        # raw_dataset,
        # tokenizer,
        task="binary",
        num_labels: int = 1,
        padding: str = "max_length",
        max_length: int = 512,
    ):
        """Preprocess a loaded dataset by normalising labels and tokenising text.

        Args:
            raw_dataset (Dataset): HF dataset object.
            tokenizer (Tokenizer): HF tokeniser.
            task (str, optional): Type of task, may be "binary", "multiclass", "multilabel" or "NER". Defaults to "binary".
            num_labels (int, optional): Number of labels if multiple. Defaults to 1.
            padding (str, optional): Padding strategy. Defaults to "max_length".
            max_length (int, optional): Number of max token length. Defaults to 512.

        Returns:
            Dataset: a tokenized HF dataset.
        """
        raw_dataset = self.raw_dataset.remove_columns(["id", "source_id", "Comments"])

        if task == "NER":
            labeled_dataset = raw_dataset
            # TODO: handle multiple spans, convert from char indexing to token indexing
        elif task == "multilabel":
            labeled_dataset = raw_dataset
        else:
            labeled_dataset = raw_dataset.map(self.select_single_label)
            cleaned_dataset = labeled_dataset.remove_columns(["label"])
            # labels expected to be an integer!
            # use a LabelEncoder or similar after export from Doccano
            encoded_dataset = cleaned_dataset.class_encode_column("labels")
            print(encoded_dataset.features)
            tokenized_dataset = encoded_dataset.map(
                lambda x: self.tokenize_function(x, padding, max_length),
                batched=True,
            )
            tokenized_dataset.set_format("torch")

            return tokenized_dataset

    def split_dataset(self, test_size=0.2):
        """Create a "train" and "test" split from a HF dataset.

        Args:
            tokenized_dataset (Dataset): Dataset object to split
            test_size (float, optional): Test to train sample ratio. Defaults to 0.2.

        Returns:
            DataDict: A HF dictionary of Dataset for "train" and "test".
        """
        dataset_dict = self.tokenized_dataset.train_test_split(test_size=test_size)
        # print(type(dataset_dict["train"]))  # <class 'datasets.arrow_dataset.Dataset'>
        return dataset_dict["train"], dataset_dict["test"]


def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datahandler = DataHandler(
        source="Doccano", doc_project=1, tokenizer=tokenizer, test_size=0.2
    )

    print("Loaded, tokenised and split the dataset from Doccano project")
    print(len(datahandler.train_dataset))
    print(len(datahandler.test_dataset))

    print(datahandler.train_dataset[0].keys())
    print(datahandler.test_dataset[10]["labels"])


if __name__ == "__main__":
    main()
