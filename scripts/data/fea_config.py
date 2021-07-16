# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 下午2:23
# @Author  : islander
# @File    : fea_config.py
# @Software: PyCharm
from collections import OrderedDict

EMB_DIM = 18
DEFAULT_SEQ_LEN = 50


"""Features Configuration

To switch to another dataset and model structure, we only need to rewrite the configuration, following the rules:
    'shape': shape of the input tensor without the batch dim, 
        e.g., shape = (2,), then data with shape (batch_size, 2) will be fed into the model
    'category': 'val_vec'/'val_seq'/'val_tgt'/'emb_vec'/'emb_seq'/'emb_tgt'
        val_: continuous input values directly fed into the model
        emb_: discrete input values which will be passed into an embedding layer
        _vec: values fed into the MLP of DIN directly, e.g., user profile
        _seq: sequence values that will be attentioned, e.g., history click sequence
        _tgt: target value that will be directly fed into the MLP of DIN, and attention with the _seq inputs, 
            e.g., target commodity
    'emb_shape': shape of the embedding table (emb_num, emb_dim), for inputs with category emb_*,
        emb_num: number of items
        emb_dim: dimension of each embedding
    'seq_name': sequence name for inputs with category *_seq
        this key is used to support multiple sequences, e.g., purchasing history and browsing history,
        each sequence corresponds a mask input named 'mask/${seq_name}', 
        and will be coupled with an independent attention layer
    'default_val': when filling placeholder values (e.g., for sequence masking), the value to be filled, by default is 0
"""
_FEA_CONFIG = {
    'user_id': {
        'shape': (1,),
        'category': 'emb_vec',
        'emb_shape': (49023, EMB_DIM),
    },
    'good_id': {
        'shape': (1,),
        'category': 'emb_tgt',
        'emb_shape': (143534, EMB_DIM)
    },
    'category_id': {
        'shape': (1,),
        'category': 'emb_tgt',
        'emb_shape': (4815, EMB_DIM)
    },
    'his_good_id': {
        'shape': (DEFAULT_SEQ_LEN, 1),
        'category': 'emb_seq',
        'emb_shape': (143534, EMB_DIM),
        'seq_name': 'his_click'
    },
    'his_category_id': {
        'shape': (DEFAULT_SEQ_LEN, 1),
        'category': 'emb_seq',
        'emb_shape': (4815, EMB_DIM),
        'seq_name': 'his_click'
    }
}


# data file columns should follow this order: label,*_fea_config_order
_fea_config_order = ('user_id', 'good_id', 'category_id', 'his_good_id', 'his_category_id')

# although python>=3.7 dict is ordered by default, to compat lower python version, use an OrderDict object here
FEA_CONFIG = OrderedDict()
for key in _fea_config_order:
    FEA_CONFIG[key] = _FEA_CONFIG[key]


"""Configuration of which inputs share the embedding

format: embedding_name -> (input1, input2, ...)
"""
SHARED_EMB_CONFIG = {
    'good_emb': ('good_id', 'his_good_id'),
    'category_emb': ('category_id', 'his_category_id')
}

if not isinstance(FEA_CONFIG, OrderedDict):
    raise AssertionError('FEA_CONFIG not an OrderedDict object')
