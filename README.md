# Text mining (classification) using scikit learn 
This repo contain simple usage of scikit learn text mining(classification), It predict the label of given preprocessed text, using the pre-trained model

This repo is only used by writer as reference, just basic usage of scikit learn text mining capabilities.

## Step contained
This repo contain the original text, 
script to prepare it as unigram(like CountVectorizer result), doing feature selection with chi2,
And model training using (choose-able) random forest, svm , or naive bayess, also prediction with trained model.

## Data
The data is a sentence, each line represent as a sentence

## Dependencies
### Install runtime
Python 2.7.*

### Install packages
```shell
pip install -U scikit-learn
```

```shell
pip install pandas
```

```shell
pip install scipy
```
