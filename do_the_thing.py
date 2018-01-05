from __future__ import print_function
import numpy as np
import pandas
import csv
import os
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
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
pretrain_features = []
labels_index = {}
all_labels = []
pretrain_labels = []
labels_count = {}

all_data = pandas.read_csv('./training_data.csv', usecols=['Country', 'FormalEducation', 'PronounceGIF'])
for _, sample in all_data.iterrows():
    if sample["PronounceGIF"] not in labels_index:
        labels_index[sample["PronounceGIF"]] = len(labels_index)
        labels_count[sample["PronounceGIF"]] = 0
    # all_features.append([sample['Country'], sample['FormalEducation']])
    wordsForFeature = sample['Country'] + sample['FormalEducation']
    all_features.append(kpt.one_hot(wordsForFeature, 300, lower=True))
    all_labels.append(labels_index[sample["PronounceGIF"]])
    labels_count[sample["PronounceGIF"]] += 1
    if labels_count[sample['PronounceGIF']] < PRETRAIN_LABEL_COUNT_MAX:
        wordsForFeature = sample['Country'] + sample['FormalEducation']
        pretrain_features.append(kpt.one_hot(wordsForFeature, 300, lower=True))
        pretrain_labels.append(labels_index[sample["PronounceGIF"]])
all_labels = to_categorical(np.asarray(all_labels))
pretrain_labels = to_categorical(np.asarray(pretrain_labels))

# invert dictionary to look up labels later
labels_lookup = {v: k for k, v in labels_index.items()}
print('Loaded %s training samples with %s categories.' % (len(all_features), len(labels_index)))

test_data = pandas.read_csv('./test_data.csv', usecols=['Id', 'Country', 'FormalEducation', "PronounceGIF"])
test_features = []
test_ids = []
for _, sample in test_data.iterrows():
    # test_features.append([sample['Country'], sample['FormalEducation']])
    wordsForFeature = sample['Country'] + sample['FormalEducation']
    all_features.append(kpt.one_hot(wordsForFeature, 300, lower=True))
    test_ids.append(sample['Id'])
