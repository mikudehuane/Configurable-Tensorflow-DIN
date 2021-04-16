# -*- coding: utf-8 -*-
# @Time    : 2021/1/4 3:51 下午
# @Author  : islander
# @File    : parse_empty_inputs.py
# @Software: PyCharm
from typing import Dict, Tuple, List, Set, Any

import tensorflow as tf
from . import constant
from . import widgets


def get_full_input_config(input_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """get the input config including masks

    Args:
        input_config: input configuration

    Returns:
        dict mapping input name to config
    """
    full_input_config = dict()
    seq_names = set()

    for feat_name, config in input_config.items():
        new_key = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.FEAT, feat_name])
        full_input_config[new_key] = config

        if constant.InputConfigKeys.SEQ_NAME in config:
            seq_name = config[constant.InputConfigKeys.SEQ_NAME]
            if seq_name not in seq_names:
                seq_names.add(seq_name)
                mask_shape = config[constant.InputConfigKeys.SHAPE][:1]
                new_key = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, seq_name])
                full_input_config[new_key] = {
                    constant.InputConfigKeys.CATEGORY: constant.InputCategory.MASK,
                    constant.InputConfigKeys.SHAPE: mask_shape
                }

    return full_input_config


def get_full_feat_names(input_config: Dict[str, Dict[str, object]], ordered=True) -> List[str]:
    """get all feat names including masks from input_config

    Args:
        input_config: input configuration
        ordered: whether the returned list is sorted

    Returns:
        list of feat_names, consistent with the keys of features passed to Din.__call__
    """
    full_feat_names = []

    for feat_name, config in input_config.items():
        new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.FEAT, feat_name])
        full_feat_names.append(new_feat_name)

    seq_names = get_seq_names(input_config)
    for seq_name in seq_names:
        new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, seq_name])
        full_feat_names.append(new_feat_name)

    if ordered:
        full_feat_names = sorted(full_feat_names)

    return full_feat_names


def get_seq_names(input_config: Dict[str, Dict[str, object]]) -> Set[str]:
    """get all seq names from input_config

    Args:
        input_config: input configuration

    Returns:
        set of seq_names
    """
    seq_names = set()
    for feat_name, config in input_config.items():
        category = config[constant.InputConfigKeys.CATEGORY]

        if widgets.is_category(category, constant.InputCategoryPlace.SEQ):
            seq_name = str(config[constant.InputConfigKeys.SEQ_NAME])
            seq_names.add(seq_name)
    return seq_names


def get_inputs_ph(input_config: Dict[str, Dict[str, object]],
                  batch_size=None, create_scope=True) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """create a placeholder for inputs

    Args:
        batch_size: batch_size of the place holders
        input_config: input configuration dict
        create_scope: whether to create a scope out from the tensors, useful for export model

    Returns:
        a tuple of (features, labels), following the same rule with the returned value of input_fn
    """
    def _get_inputs():
        features = dict()
        mask_dict = dict()

        for feat_name, config in input_config.items():
            category = config[constant.InputConfigKeys.CATEGORY]
            shape = config[constant.InputConfigKeys.SHAPE]
            shape = [batch_size] + list(shape)  # batch size
            dtype = tf.int32 if widgets.is_category(category, constant.InputCategoryType.EMB) else tf.float32

            new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.FEAT, feat_name])
            features[new_feat_name] = tf.placeholder(shape=shape, dtype=dtype, name=new_feat_name)

            if widgets.is_category(category, constant.InputCategoryPlace.SEQ):
                seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                mask_dict[seq_name] = {constant.InputConfigKeys.SHAPE: shape[:-1]}

        for seq_name, config in mask_dict.items():
            shape = config[constant.InputConfigKeys.SHAPE]
            new_feat_name = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, seq_name])
            features[new_feat_name] = tf.placeholder(shape=shape, dtype=tf.int32, name=new_feat_name)

        labels = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='labels')
        return features, labels

    if create_scope:
        with tf.name_scope('inputs_placeholder'):
            return _get_inputs()
    else:
        return _get_inputs()
