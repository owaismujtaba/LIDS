from data_utils import drop_constant_features, drop_meaningless_cols
from data_utils import get_file_names, load_files, min_max_scaler
from info_gain import info_gain
from scipy.stats import entropy
import warnings
import pandas as pd
import numpy as np

import pdb


def clean_dataset(PATH, nrows):
    """
    Cleans the dataset according to the author
    1. Deleting records having NULL values
    2. Deleting constant Features
    3. Deletion of Duplicate records
    4. Deletion of meaning less columns
    """

    filenames = get_file_names(PATH)
    dataset = load_files(PATH, filenames, nrows)
    dataset.dropna(axis=0, inplace=True)
    meaning_less_cols = ['Unnamed: 0', 'Flow ID', ' Timestamp', ' Source IP',
                         'SimillarHTTP', ' Source Port', ' Destination IP', ' Destination Port']

    dataset = drop_meaningless_cols(dataset, meaning_less_cols)
    dataset = drop_constant_features(dataset)

    dataset = random_sampling(dataset)
    dataset.drop_duplicates(inplace=True)

    for column in dataset.columns:
        dataset[column] = dataset[column].replace([np.inf, -np.inf], dataset[column].median())

    dataset = drop_less_information_gain_features(dataset)

    dataset = min_max_scaler(dataset)

    return dataset


def random_sampling(dataset):
    dataset1 = dataset.copy()
    n_samples = min(dataset1[' Label'].value_counts().values)

    dataset1 = dataset1.groupby(' Label', group_keys=False).apply(lambda x: x.sample(n_samples))

    dataset1 = dataset1.reset_index()
    dataset1.drop('index', axis=1, inplace=True)

    print("Dataset, shape", dataset1.shape)

    return dataset1


def drop_less_information_gain_features(dataset, threshold=0.2):
    """
    Measures the reduction in entropy after the split  
    Args:
        dataset: the dataset
        threshold: threshold parameter
    Returns
        dataset: modified dataset without less info gain features
    """
    print("*******************************Deleting less info_gain features*************************")

    ig_cols = []
    features = list(set(dataset.columns) - set([' Label']))
    for col in features:
        info_gain_value = info_gain.info_gain(dataset[col], dataset[' Label'])
        if info_gain_value > threshold:
            ig_cols.append(col)
    ig_cols.append(' Label')

    for i in range(len(ig_cols)):
        print("{}. {}".format(i + 1, ig_cols[i]))

    print("*******************************Less info_gain features Deleted*************************")

    dataset = dataset[ig_cols]
    print("Dataset Shape", dataset.shape)

    return dataset
