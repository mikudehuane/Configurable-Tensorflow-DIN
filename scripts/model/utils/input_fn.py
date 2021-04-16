# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 11:31 下午
# @Author  : islander
# @File    : input_fn.py
# @Software: PyCharm
from typing import List

from . import constant
import tensorflow as tf
from pprint import pprint
import numpy as np


def omit_batch_size(shape):
    shape = tf.TensorShape([None]).concatenate(shape[1:])
    return shape


def get_batch(data_iter):
    try:
        feat_dict, mask_dict, labels = data_iter.__next__()
    except StopIteration:
        return None
    features = dict()
    for feat_name, feature in feat_dict.items():
        new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.FEAT, feat_name])
        features[new_feat_name] = feature
    for seq_name, mask in mask_dict.items():
        new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, seq_name])
        features[new_feat_name] = mask

    # squeeze verbose dimension of labels
    if len(labels.shape) != 1:
        if not get_batch.warned:
            tf.logging.warning(f'expect labels to be flattened arrays, but got shape {labels.shape}, force flattening')
            get_batch.warned = True
        if labels.shape[1] != 1:
            raise RuntimeError(f'one sample should be associated with only one label, '
                               f'but got labels with shape {labels.shape}')
        else:
            labels = labels.reshape([-1])
    return features, labels


get_batch.warned = False


class EmptyDatasetError(Exception):
    pass


class DataInputFn(object):
    """A function-like class to be fed as input_fn

    Args:
        data_gen: uncalled generator
        data_gen_kwargs: dict of args to be fed to the call of data_gen, e.g. dict(mode='test')
    """

    def __init__(self, data_gen, data_gen_kwargs):
        self._data_gen = data_gen
        self._data_gen_kwargs = data_gen_kwargs

    def __call__(self, require='dataset'):
        """wrap the input function

        Args:
            require:
              - dataset: return the dataset used for estimator
              - generator: return the data generator

        Returns:
            replies on <require> arg
        """
        tf.logging.debug('input_fn called')
        data_iter = self._data_gen(**self._data_gen_kwargs)

        first_batch = get_batch(data_iter=data_iter)
        if first_batch is None:
            raise EmptyDatasetError('the first batch generated should not be None, '
                                    'but DataInputFn._get_batch accepted StopIteration (maybe an emtpy train/test set)')

        first_features, first_labels = first_batch
        feature_shapes = {key: omit_batch_size(value.shape) for key, value in first_features.items()}
        output_shapes = (feature_shapes, omit_batch_size(first_labels.shape))
        feature_dtypes = {key: value.dtype for key, value in first_features.items()}
        output_dtypes = (feature_dtypes, first_labels.dtype)

        def data_gen_wrapper():
            yield first_batch
            while True:
                batch = get_batch(data_iter=data_iter)
                if batch is None:
                    return
                else:
                    yield batch

        if require == 'dataset':
            return tf.data.Dataset.from_generator(data_gen_wrapper,
                                                  output_types=output_dtypes, output_shapes=output_shapes)
        elif require == 'generator':
            return data_gen_wrapper()
        else:
            raise ValueError('require is expected to be one of (dataset, generator), but got %s' % require)
