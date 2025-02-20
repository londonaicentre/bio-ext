import json
import pandas as pd
import mlflow


def load_data(file_path):
    """Load data from file or stream from Doccano

    Args:
        file_path (str): Path to labelled data file.

    Returns:
        pd.DataFrame: dataframe with columns `text` and `label`.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path).drop_duplicates()
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data_dict = json.load(f)
            df = pd.DataFrame.from_dict(data_dict)
    else:
        df = None
    return df


def data_preprocessing(df: pd.DataFrame):
    """
    Parse the data. Create label dictionary and add 'label' and 'data_type' columns.
    Split the data into train and validation dataframes. Store the label columns in a pickle file.
    """

    print("parse_df_data")
    label_columns = df["ClassLabel"].unique()
    label_dict = {}
    for index, possible_label in enumerate(label_columns):
        label_dict[possible_label] = index
        df[possible_label] = 0
        df.loc[df["ClassLabel"] == possible_label, [possible_label]] = 1

    df["label"] = df.ClassLabel.replace(label_dict)
    # result.infer_objects(copy=False)
    num_labels = len(label_dict)
    num_classes = len(list(set(df.label)))

    # with open("label_columns.data", "wb") as filehandle:
    #     pickle.dump(label_columns, filehandle)

    class_counts = df["ClassLabel"].value_counts()
    for class_label, count in class_counts.items():
        mlflow.log_param(f"class_{class_label}_count", count)
