from __future__ import print_function
import numpy as np
import pandas
import csv
import os
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


TRAINING_SPLIT = .2
VALIDATION_SPLIT = .2
PRETRAIN_LABEL_COUNT_MAX = 320

all_features = []
# Dictionary for our label encoding
labels_index = {}
# Array of labels in numeric encoding
all_labels = []

all_data = pandas.read_csv('./training_data.csv', usecols=['Country', 'FormalEducation', "PronounceGIF"])
for _, in all_data.iterows():
    if sample["PronounceGIF"] not in labels_index:
        labels_index[sample["PronounceGIF"]] = len(labels_index)
        labels_count[sample["PronounceGIF"]] = 0
    all_features.append([sample['Country'], sample['FormalEducation']])
    all_labels.append(labels_index[sample["PronounceGIF"]])
    labels_count[sample["PronounceGIF"]] += 1
  # if we don't have too much pre-training data for the current label, add it to pretraining data
    if labels_count[sample['PronounceGIF']] < PRETRAIN_LABEL_COUNT_MAX:
        pretrain_texts.append([sample['Country'], sample['FormalEducation']])
        pretrain_labels.append(labels_index[sample["PronounceGIF"]])

# invert dictionary to look up labels later
labels_lookup = {v: k for k, v in labels_index.items()}
print('Loaded %s training samples with %s categories.' % (len(all_texts), len(labels_index)))
