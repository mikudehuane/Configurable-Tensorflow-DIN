# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 10:38 上午
# @Author  : islander
# @File    : constants.py
# @Software: PyCharm


class InputConfigKeys(object):  # useful keys in each input_config
    CATEGORY = 'category'
    SHAPE = 'shape'
    EMB_SHAPE = 'emb_shape'
    SEQ_NAME = 'seq_name'
    EMB_PROCESS = 'emb_process'
    DEFAULT_VAL = 'default_val'


class InputCategoryPlace(object):  # attribute of categories, indicating where it is in the network
    VEC = 'vec'
    SEQ = 'seq'
    TGT = 'tgt'


class InputCategoryType(object):  # attribute of categories, indicating how it is processed
    VAL = 'val'
    EMB = 'emb'


class InputCategory(object):  # values of categories
    VAL_VEC = '_'.join([InputCategoryType.VAL, InputCategoryPlace.VEC])
    VAL_SEQ = '_'.join([InputCategoryType.VAL, InputCategoryPlace.SEQ])
    VAL_TGT = '_'.join([InputCategoryType.VAL, InputCategoryPlace.TGT])
    EMB_VEC = '_'.join([InputCategoryType.EMB, InputCategoryPlace.VEC])
    EMB_SEQ = '_'.join([InputCategoryType.EMB, InputCategoryPlace.SEQ])
    EMB_TGT = '_'.join([InputCategoryType.EMB, InputCategoryPlace.TGT])
    MASK = 'mask'


class FeaturePrefix(object):  # Din separate configures with prefix (e.g., feat and mask) with input_config keys
    FEAT = 'feat'
    MASK = 'mask'


FEATURE_SEPARATOR = '/'

