# Example BERT model training on BRCA synth data as a MULTI CLASS SINGLE LABEL task

#https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

# Feature
1. write a function that check if mlflow is functional or not. if not, create an experiment

2. confidence interval from the ml models. bootstrap etc
3. how do we save the model and then reuse it. ideally with a pipeline. ideaelly with idx2label and label2idx stuff 
https://discuss.huggingface.co/t/predicting-on-new-text-with-fine-tuned-multi-label-model/13046/3
there is a way to save this label2id stuff in config.json from the model.

https://github.com/Dirkster99/PyNotes/blob/master/Transformers/LocalModelUsage_Finetuning/66_Transformer_4_Language_Classification_MultiClass.ipynb

https://github.com/Dirkster99/PyNotes/blob/master/Transformers/LocalModelUsage_Finetuning/30%20MultiClass%20Classification%20in%2010%20Minutes%20with%20BERT-TensorFlow-SoftMax-LocalModel.ipynb

https://colab.research.google.com/drive/1X7l8pM6t4VLqxQVJ23ssIxmrsc4Kpc5q?usp=sharing

https://colab.research.google.com/drive/1X7l8pM6t4VLqxQVJ23ssIxmrsc4Kpc5q?usp=sharing#scrollTo=zI10-X_YowwH

https://discuss.huggingface.co/t/fine-tune-for-multiclass-or-multilabel-multiclass/4035/40

https://huggingface.co/docs/evaluate/en/installation
evaluation stuff
# Learnings

# Improving model performance

For text classification with BERT, you can perform model diagnostics through several techniques:

Confusion Matrix Visualization


Shows per-class prediction errors
Reveals misclassification patterns
Can highlight class imbalance issues


https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb

multi label classification but with detailed torch stuff
Loss and Accuracy Curves


Plot training/validation loss and accuracy over epochs
Identify overfitting (divergence between train/validation curves)
Detect learning dynamics


Attention Visualization


Inspect which tokens contribute most to classification
Understand model's feature importance
Identify potential bias or limited learning


Prediction Probability Distributions


Histogram of model's confidence scores
Detect uncertain predictions
Reveal regions of low model reliability


Embedding Space Analysis


t-SNE or UMAP projections of text embeddings
Visualize class separability
Identify potential clustering or separation issues


Residual Analysis


Compare predictions against ground truth
Manually inspect misclassified instances
Understand model's systematic errors

I recommend using libraries like:

scikit-learn (confusion matrix)
matplotlib/seaborn (curves, distributions)
transformers' built-in visualization tools
tools like Weight & Biases for comprehensive tracking

Would you like me to elaborate on any specific diagnostic technique?


es, absolutely. These diagnostic techniques are crucial for:

Identifying model weaknesses
Understanding misclassification patterns
Guiding targeted improvements
Preventing overfitting

Key priorities:

Confusion matrix
Error analysis
Attention visualization

Most impactful for performance tuning. Highly recommended.

here some about hyperparameter tuning https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=NboJ7kDOIrJq

https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html