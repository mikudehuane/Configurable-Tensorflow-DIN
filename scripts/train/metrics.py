# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 11:06 上午
# @Author  : islander
# @File    : metrics.py
# @Software: PyCharm

"""
This file implement some customized metrics operating on np.ndarray

All metrics follows the following sign

metric(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any])
  y_true: true label of each sample (given by class id), with shape (batch_size,)
  y_pred: predicted probabilities of each class, y_pred[i][j]=the probability that sample i belongs to class j,
    with shape (batch_size, num_classes)
  meta_dict: other data required to compute this metric, e.g., clients for gauc
"""

from typing import Dict, Tuple, List, Any

import sklearn.metrics
import numpy as np


# given data is emtpy
EMPTY_DATA = -1.
# when calculating AUC, only one label is given
AUC_ONE_CLASS = -2.
# some of the metrics, e.g., false_prop, is only for binary classifier
ONLY_ALLOW_BINARY = -3.
# required meta data, e.g., client_offsets not given
MISSING_META_DATA = -4.
# when calculating GAUC, all clients have no positive samples
GAUC_NO_VALID_CLIENT = -5.
# value registered to be got directly is not given
VALUE_NOT_GIVEN = -6.


def squeeze_labels(y_true):
    if len(y_true.shape) == 2:
        y_true = y_true.squeeze(-1)
    return y_true


def auc(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    if len(y_pred) == 0:
        return EMPTY_DATA

    y_true = squeeze_labels(y_true)
    # one hot
    y_true_hot = np.eye(y_pred.shape[1], dtype=np.int32)[y_true]
    try:
        auc_ = sklearn.metrics.roc_auc_score(y_true_hot, y_pred)
    except ValueError as e:  # only one class in y_true
        first_value = y_true[0]
        for value in y_true:
            if value != first_value:
                raise e
        # tf.logging.warning("when calculating auc, only one class in y_true")
        auc_ = AUC_ONE_CLASS
    return auc_


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    if len(y_pred) == 0:
        return EMPTY_DATA

    y_true = squeeze_labels(y_true)
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


def false_prop(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    if len(y_pred) == 0:
        return EMPTY_DATA

    y_true = squeeze_labels(y_true)
    # ratio of false samples
    if y_pred.shape[1] != 2:
        return ONLY_ALLOW_BINARY

    return 1 - y_true.mean()


def neg_log_loss(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    if len(y_pred) == 0:
        return EMPTY_DATA

    y_true = squeeze_labels(y_true)
    # negative log loss (e.g., for cross entropy)
    eps = 1e-7
    true_probs = y_pred[range(y_pred.shape[0]), y_true]
    log_probs = np.log(true_probs + eps)
    return - log_probs.mean()


def num_samples(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    y_true = squeeze_labels(y_true)
    return len(y_true)


def max_true_prob(y_true: np.ndarray, y_pred: np.ndarray, meta_dict: Dict[str, Any] = None):
    if len(y_pred) == 0:
        return EMPTY_DATA

    y_true = squeeze_labels(y_true)

    if y_pred.shape[1] != 2:
        return ONLY_ALLOW_BINARY

    return np.max(y_pred[:, 1])
