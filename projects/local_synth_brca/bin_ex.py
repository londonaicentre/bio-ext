# imdb_hf_03_use.py
# use tuned HF model for IMDB sentiment analysis accuracy
# zipped raw data at:
# https://ai.stanford.edu/~amaas/data/sentiment/

import numpy as np  # not used
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from transformers import logging  # to suppress warnings

device = torch.device("cpu")


def main():
    # 0. get ready
    print("\nBegin use IMDB HF model demo ")
    logging.set_verbosity_error()  # suppress wordy warnings
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. load pretrained model
    print("\nLoading untuned DistilBERT model ")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )
    model.to(device)
    print("Done ")

    # 2. load tuned model wts and biases
    print("\nLoading tuned model wts and biases ")
    # model.load_state_dict(torch.load(".\\Models\\imdb_state.pt"))
    model.eval()
    print("Done ")

    # 3. set up input review
    review_text = ["This was a GREAT waste of my time."]
    print("\nreview_text = ")
    print(review_text)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    review_tokenized = tokenizer(review_text, truncation=True, padding=True)

    print("\nreview_tokenized = ")
    print(review_tokenized)
    # {'input_ids': [[101, 2023, 2001, 1037, 2307, 5949,
    #    1997, 2026, 2051, 1012, 102]],
    #  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

    input_ids = review_tokenized["input_ids"]
    print("\nTokens: ")
    for id in input_ids[0]:
        tok = tokenizer.decode(id)
        print("%6d %s " % (id, tok))

    input_ids = torch.tensor(input_ids).to(device)
    mask = torch.tensor(review_tokenized["attention_mask"]).to(device)
    dummy_label = torch.tensor([0]).to(device)

    # 4. feed review to model, fetch result
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=mask, labels=dummy_label)
    print("\noutputs = ")
    print(outputs)
    # SequenceClassifierOutput(
    # loss=tensor(0.1055),
    # logits=tensor([[ 0.9256, -1.2700]]),
    # hidden_states=None,
    # attentions=None)

    # 5. interpret result
    logits = outputs[1]
    print("\nlogits = ")
    print(logits)

    pred_class = torch.argmax(logits, dim=1)
    print("\npred_class = ")
    print(pred_class)

    print("\nEnd demo ")


if __name__ == "__main__":
    main()
