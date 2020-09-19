"""
Module DeepD.data is for normalization and data clipping as we described in Methods section
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter


def random_partition(train_df, test_df, n_genes=978, annotation_col='nc_label', seed=0, validation_ratio=0.3):
    train, test = normalize_train_and_test(train_df, test_df, annot_col=annotation_col, n_genes=n_genes)
    print("[Preprocessing] Creating validation set...")
    np.random.seed(seed)
    n_experiment_samples = train['value'].shape[0]
    valid_pos = np.random.choice(range(n_experiment_samples), int(validation_ratio * n_experiment_samples),
                                 replace=False)
    train_pos = list(set(range(n_experiment_samples)) - set(valid_pos))
    valid = {'value': train['value'][valid_pos], 'full_label': train['full_label'].iloc[valid_pos],
             'class_annot': train['class_annot'][valid_pos]}
    train = {'value': train['value'][train_pos], 'full_label': train['full_label'].iloc[train_pos],
             'class_annot': train['class_annot'][train_pos]}
    return train, valid, test, valid_pos


def normalize_train_and_test(train_df, test_df, annot_col='class label', n_genes=978):
    """
    :param train_df: (pandas.DataFrame) training data from disk
    :param test_df: (pandas.DataFrame) test data frame from disk
    :param annot_col: (str) which column to be used as classification label
    :param n_genes: (int) the length of feature vectors (e.g. for L1000 genes: 978)
    :return: train, test (dict, dict): processed dataset instances, each has:
            'value': (pandas.DataFrame) data matrix
            'full_label': (pandas.DataFrame) all the metadata information including classification annotations
            'class_annot': (numpy.ndarray) classification annotations
    """
    train = preprocess_df(train_df, annot_col=annot_col, task='train_set', n_genes=n_genes)
    test = preprocess_df(test_df, annot_col=annot_col, task='test_set', normalize_on=train_df, n_genes=n_genes)
    return train, test


def preprocess_df(df, annot_col='class label', task='untitled_set', normalize_on=None, n_genes=978):
    print("[Preprocessing] Processing dataset {}...".format(task.upper()))
    assert df.shape[1] > n_genes  # L1000 genes
    data = df.values[:, -n_genes:].astype('float')
    labels = df.iloc[:, :-n_genes]
    label_to_classify = df[annot_col].values.flatten().astype('int')

    if data.max() == 1 and data.min() == -1:
        print("[Preprocessing] Input data {} detected has a range of (-1, 1), skipping data preprocessing..."
              .format(task.upper()))
    elif normalize_on is not None:
        print("[Preprocessing] Scalers are detected for input data {} skipping data preprocessing..."
              .format(task.upper()))
        data = rescale_and_clip(data, scale_on=normalize_on.iloc[:, -n_genes:])
    else:
        data = rescale_and_clip(data)

    return {'value': data, 'full_label': labels, 'class_annot': label_to_classify}


def assert_scale(x):
    return x.mean() < 1e-10 and x.std() < 1e-10


def rescale_and_clip(data, scale_on=None):
    if np.all([assert_scale(row) for row in data]) and np.all([assert_scale(col) for col in data.transpose()]):
        print("[Preprocessing] Input data detected has mean=0 and sd=1, skipping data scaling...")
    else:
        print("[Preprocessing] Scaling...")
        if scale_on is None:
            scale_on = data
        scaler = StandardScaler()  # rescale each gene
        scaler.fit(scale_on)
        data = scaler.transform(data)
        scaler = StandardScaler()  # rescale each cell
        data = np.transpose(scaler.fit_transform(np.transpose(data)))
    print("[Preprocessing] Clipping...")
    clipping_thre = 5.  # cutting |outliers| > 5 sigma (assuming Gaussian)
    data = np.clip(data, -clipping_thre, clipping_thre)/clipping_thre
    print("[Preprocessing] Dataset is ready for training DeepD...")
    return data
