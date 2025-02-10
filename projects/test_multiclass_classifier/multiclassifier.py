
import pandas as pd
import transformers
import torch
import datasets
from pathlib import Path
from datasets import Dataset, load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel,BertModel,BertForSequenceClassification,
                          TrainingArguments,Trainer)

from transformers.modeling_outputs import SequenceClassifierOutput

# TODO : this is need to be checked.
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# LOAD DATA AND ONE HOT ENCODE 
## task1 label2index ; index2label gen 

DATA_PATH = "data/breast_brca_labelled.json"
df = pd.read_json(DATA_PATH)

## one-hot encode, conver to FLOAT and concat as list.
df["label"] = df["label"].astype("category")
labels = list(df["label"].unique())

labelsdict = {key:value for key,value in enumerate(labels)}
id2label = labelsdict
label2id = {value:idx for idx,value in id2label.items()}

df["category"] = df["label"].apply(lambda x: label2id[x])

# sense check
df["label"].value_counts()
df["category"].value_counts()
df["category"] = df["category"].astype("int")

#one_hot = pd.get_dummies(df["label"])
#df = df.drop("label",axis=1)
#df = df.join(one_hot)
#df[labels] = df[labels].astype(float)

df.columns = ["text","category","labels"]

# TRAIN TEST SPLIT
## TODO : consider class weighted splitting.
df_shuffled = df.sample(frac = 1, random_state= 42).reset_index(drop=True)
split_index = int(0.8 * len(df_shuffled))
train_df = df_shuffled.iloc[:split_index]
test_df = df_shuffled.iloc[split_index:]

# GENERATE TRANSFORMERS DATASET
## NOTE: do the column names list concat in transformers as its faster 
train = Dataset.from_pandas(train_df)
val = Dataset.from_pandas(test_df)
ds = DatasetDict()
ds['train'] = train
ds['validation'] = val

## NOTE: remove 
cols = ds["train"].column_names
# ds = ds.map(lambda x: {"labels":[x[c] for c in cols if c != "text"]})

#id2label = {idx:label for idx,label in enumerate(labels)}
#label2id = {label:idx for idx,label in enumerate(labels)}
# ESTABLISH MODEL AND TOKENIZER
model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)

# note the examples["text"] has to match with the ones from ds 
def tokenize_encode(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
        )

#cols.remove("labels")
ds_enc = ds.map(tokenize_encode,batched=True,remove_columns=["category","text"])

num_labels = len(labels)
# now we are tokenised 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=5,label2id=label2id,id2label=id2label)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(predictions),dim = -1)
    labels = torch.from_numpy(labels)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

batch_size = 8

args = TrainingArguments(
    output_dir="mclassout",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    use_cpu=True,
    no_cuda=True
)

trainer = Trainer(
    model,
    args,
    train_dataset=ds_enc["train"],
    eval_dataset=ds_enc["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

trainer.train()
trainer.evaluate()
trainer.save_model("model/")

def predictor(text,model):
    #x = [text]
    encodings = tokenize_encode(text)
    print("tokenised")
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.nn.functional.softmax(outputs.logits,dim=-1)
        predicted_label = torch.argmax(predictions,dim = -1)
    
    return id2label[predicted_label.item()]


text2 = "this is just gibberish is now detected in our samples."
x = {}
x["text"] = text2
tokenize_encode(x)
prediction = predictor(text = x, model = model)


# metrics
# accuracy[global measure - correct/all], precision and recall for each class
# confusion matrix
# macro averaging (equal weight to all class) thats why you average
# micro averaging (equal weight to each instance ) is where you
# F1 score 
# Mattheus Correlation Coefficient. correlation between predicted and actuall. 
# Cohens kappa 


# TODO
# unknown tokens
# tokenizer.convert_tokens_to_ids(["VUS"])
# this code returns a 100

example = ds_enc["train"][0]
tokenizer.decode(example["input_ids"])
# problem is brca got split


