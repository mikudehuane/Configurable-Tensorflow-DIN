# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 9:48 下午
# @Author  : islander
# @File    : widgets.py
# @Software: PyCharm

import tensorflow as tf
from .constant import *


def get_ordered_dict_values(dict_input):
    inputs = dict_input.items()
    inputs = sorted(inputs, key=lambda x: x[0])
    return [input_[1] for input_ in inputs]


def concat_emb_val(embs, vals, name):
    """concat embbeddings and hard coded vals

    Args:
        embs: embeddings looked-up from layers (or None)
        vals: hard coded features (or None)
        name: name of the operation

    Returns:
        concat tensor
    """
    embs = get_ordered_dict_values(embs)
    vals = get_ordered_dict_values(vals)

    emb_list = []
    with tf.name_scope(name):
        if embs:
            cated_emb = tf.concat(embs, axis=-2, name='emb_cat')
            cated_emb_shape = cated_emb.get_shape().as_list()
            last_dim = cated_emb_shape[-1] * cated_emb_shape[-2]
            cated_emb_shape[-2:] = [last_dim]
            cated_emb_shape[0] = -1
            cated_emb = tf.reshape(cated_emb, cated_emb_shape, name='emb_reshape')
            emb_list.append(cated_emb)
        if vals:
            emb_list.extend(vals)

        if emb_list:
            return tf.concat(emb_list, axis=-1, name='concat_emb_with_val')
        else:
            raise RuntimeError(f'either embs or vals should be not None, but got both Nones in {name}')


def replace_out_of_ranges(value, target_range, name, replace_value=0):
    """take mask the value, fill invalid locations with pad

    Args:
        value: tensor to be masked
        target_range: tuple of (range_min, range_max), indicating range [range_min, range_max)
        name: operation name
        replace_value: out-of-range values will be replcaed with this
    """
    with tf.name_scope(name):
        range_min, range_max = target_range
        with tf.name_scope('get_out_of_range_places'):
            replaces = tf.math.logical_or(value < range_min, value >= range_max)  # whether to replace
        with tf.name_scope('create_pad_tensor'):
            paddings = tf.ones_like(value, name='padding_ones', dtype=value.dtype) * replace_value
        value = tf.where(replaces, paddings, value, name='apply_replace')
    return value


def get_all_constants(constant_class):
    return [v for k, v in vars(constant_class).items() if not k.startswith('_') and not callable(v)]


def is_category(category, super_category):
    """check whether category is one from super_category

    Args:
        category: one from Din._TYPES_OF_INPUT
        super_category: Din.VAL, Din.EMB, Din.VEC, Din.TGT, Din.SEQ

    Returns:
        True/False

    Raises:
        ValueError: when unknown super_category or category given.
    """
    if category not in get_all_constants(InputCategory):
        raise ValueError(f"category should be one of InputCategory, but got {category}")

    if super_category in get_all_constants(InputCategoryPlace):
        return category.endswith(super_category)
    elif super_category in get_all_constants(InputCategoryType):
        return category.startswith(super_category)
    elif super_category == InputCategory.MASK:
        return category == InputCategory.MASK
    else:
        raise ValueError(f'super_category should be from InputCategory* or MASK, but got {super_category}')


def check_config(input_config, shared_emb_config):
    """check whether the configuration for __init__ is valid

    Raises:
        ValueError: when the configurations is not valid
            - category key missing
            - category not valid
            - Din.EMB_SHAPE missed for embedding input config
            - CONFIG_SEPARATOR appears in input_config keys, it is used to separate fields

    """
    for feat_name, config in input_config.items():
        # check for feat_name
        if FEATURE_SEPARATOR in feat_name:
            raise ValueError(f'separator {FEATURE_SEPARATOR} should not appear in input_config keys, '
                             f'but got {feat_name}')

        # check for Din.CATEGORY
        if InputConfigKeys.CATEGORY not in config:
            raise ValueError(f'all configs should have {InputConfigKeys.CATEGORY} key, '
                             f'but config for {feat_name} does not')
        category = config[InputConfigKeys.CATEGORY]
        if category not in get_all_constants(InputCategory):
            raise ValueError(f'category should be one from {InputCategory}, '
                             f'but got {category}')

        # check for Din.SHAPE
        if InputConfigKeys.SHAPE not in config:
            raise ValueError(f'all configs should have {InputConfigKeys.SHAPE} key, '
                             f'but config for {feat_name} does not')

        # check for Din.EMB_SHAPE
        if is_category(category, InputCategoryType.EMB):
            if InputConfigKeys.EMB_SHAPE not in config:
                raise ValueError(f'category from {InputCategoryType.EMB} should have {InputConfigKeys.EMB_SHAPE} key, '
                                 f'but config for {feat_name} does not')

        # check for Din.SEQ_NAME
        if is_category(category, InputCategoryPlace.SEQ):
            if InputConfigKeys.SEQ_NAME not in config:
                raise ValueError(f'category from {InputCategoryPlace.SEQ} should have {InputConfigKeys.SEQ_NAME} key',
                                 f'but config for {feat_name} does not')

    for emb_name, feat_names in shared_emb_config.items():
        emb_shape = input_config[feat_names[0]][InputConfigKeys.EMB_SHAPE]
        for feat_name in feat_names[1:]:
            # feat_names in shared_emb_config is allowed to be not contained in input_config
            if feat_name in input_config:
                _emb_shape = input_config[feat_name][InputConfigKeys.EMB_SHAPE]
                if _emb_shape != emb_shape:
                    raise ValueError(f'inputs sharing the same embedding should '
                                     f'have the same {InputConfigKeys.EMB_SHAPE}, '
                                     f'but got {emb_shape} for {feat_names[0]} and {_emb_shape} for {feat_name}')
