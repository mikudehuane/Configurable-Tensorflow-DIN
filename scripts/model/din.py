# -*- coding: utf-8 -*-
# @Time    : 2020/11/24 17:51
# @Author  : islander
# @File    : model.py
from typing import Dict, List, Union

import tensorflow as tf

from . import layers
from .utils import *
from . import utils
import logging
from .model_frame import ModelFrame

_logger = logging.getLogger('model.din')


class Din(ModelFrame):
    """Deep Interest Net wrapper

    The output of this model is expected to indicate probabilities (e.g., after a softmax)

    Args:
        input_config: the embedding configuration dict
        shared_emb_config: specify shared embedding layers, format: emb_name -> tup of feat_names
            feat_names is allowed to be not contained in input_config
        attention_layers: list of layers.AirLayer
            list attention layer operating seq, (each seq associated with one attention layer)
        forward_net: layers.AirLayer
            forward net producing score (CTR is the softmaxed score)
        required_feat_names: see ModelFrame

    Keyword Args:
        batch_norm: one of: bn (batchnorm), ln (layernorm), None (do not norm)
        include_his_sum: whether to feed a naive history embedding sum into forward net, *besides* the attention output
        epsilon: arithmetic robustness
        use_seq: whether feed a specific seq info into forward net
            this is given as a dict mapping seq_name to bool
        use_vec: whether feed vec info into forward net (True is overrided if no VEC category appears in input_config)
        use_moving_statistics: whether to use moving statistics for batchnorm in test phase,
            note that this actually degrade the performance in practice
            can be True or False or 'always' (use in train and eval mode)

    Raises:
        ValueError:
            - when the number of attention layers is not the number of seqs
            - when the given attention layers or forward_net is not subclass of layers.AirLayer
            - when the given configuration violate rules, e.g, different shape for a shared embedding
    """

    def __init__(self, input_config: Dict, shared_emb_config=None, attention_layers=None, forward_net=None,
                 required_feat_names=None,
                 **kwargs):
        super().__init__(required_feat_names=required_feat_names, input_config=input_config, **kwargs)

        # kwargs
        self._batch_norm = kwargs.pop('batch_norm', 'bn')
        self._include_his_sum = kwargs.pop('include_his_sum', True)
        self._use_moving_statistics = kwargs.pop('use_moving_statistics', True)
        # will be override later in config
        self._use_vec = False
        self._use_seq = False

        # declare index member variables
        # check whether the input_config and shared_emb_config is valid
        utils.check_config(input_config=input_config, shared_emb_config=shared_emb_config)
        # map feature name to embedding layer name (for shared embedding config)
        # feat_name -> emb_layer_name (feat_name if not in shared_emb_config else emb_name)
        self._emb_dict = dict()
        # map embedding name to whether it is trainable
        # emb_name -> bool
        self._emb_trainable_dict = dict()
        # ordered list of seq_names
        # [seq_name_0, seq_name_1, ...]
        self._seq_names = set()

        # parse input_config (input configurations)
        for feat_name, config in input_config.items():
            # judge whether to use vec or seq, according to whether there exists such features
            input_category = config[constant.InputConfigKeys.CATEGORY]
            if utils.is_category(input_category, constant.InputCategoryPlace.VEC):
                self._use_vec = True
            if utils.is_category(input_category, constant.InputCategoryPlace.SEQ):
                self._use_seq = True

            # fill _emb_dict with feat_name
            if utils.is_category(input_category, constant.InputCategoryType.EMB):
                self._emb_dict[feat_name] = feat_name

            # for sequence inputs (val_seq or emb_seq), generate mask config
            if constant.InputConfigKeys.SEQ_NAME in config:
                seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                self._seq_names.add(seq_name)
        self._seq_names = sorted(list(self._seq_names))
        self._use_seq = {seq_name: self._use_seq for seq_name in self._seq_names}

        # replace shared embeddings in _emb_dict with emb_name
        if shared_emb_config is not None:
            for emb_name, feat_names in shared_emb_config.items():
                for feat_name in feat_names:
                    if feat_name in self._emb_dict:  # feat_names is allowed to be not contained
                        self._emb_dict[feat_name] = emb_name

        # config trainable
        for feat_name, emb_name in self._emb_dict.items():
            self._emb_trainable_dict[emb_name] = True
        self._bn_before_forward_net_trainable = True

        # override use_vec and use_seq from keyed args
        if self._use_vec:
            self._use_vec = kwargs.pop('use_vec', True)
        _use_seq = kwargs.pop('use_seq', self._use_seq)
        for seq_name, is_using in self._use_seq.items():
            if is_using:
                self._use_seq[seq_name] = _use_seq[seq_name]

        # create default layers
        self._attention_layers, self._forward_net = self.__get_net(attention_layers, forward_net)

        if len(kwargs) != 0:
            raise ValueError(f"Unrecognized kwargs: {kwargs}")

    def _get_emb_name2feat_name(self):  # reverse self._emb_dict, which is feat_name -> emb_name
        emb_name2feat_name = dict()
        for feat_name, emb_name in self._emb_dict.items():
            emb_name2feat_name[emb_name] = feat_name
        return emb_name2feat_name

    def forward(self, features, mode):
        """forward pass and produce logits

        Raises:
            RuntimeError: when config parsing failed
        """
        _INPUT_PARSE_FAIL_MSG = 'failure when parsing inputs for Din'

        # separate inputs
        separated_features = {
            constant.InputCategory.EMB_VEC: dict(),
            constant.InputCategory.EMB_SEQ: {seq_name: dict() for seq_name in self._seq_names},
            constant.InputCategory.EMB_TGT: dict(),
            constant.InputCategory.VAL_VEC: dict(),
            constant.InputCategory.VAL_SEQ: {seq_name: dict() for seq_name in self._seq_names},
            constant.InputCategory.VAL_TGT: dict(),
            constant.InputCategory.MASK: dict(),
        }

        with tf.variable_scope('embedding', reuse=False) as embedding_scope:  # create embedding variables, reuse=False
            # parse inputs
            for feat_name in self._input_config:
                feature = features[feat_name]
                config = self._input_config[feat_name]
                category = config[constant.InputConfigKeys.CATEGORY]
                if utils.is_category(category, constant.InputCategory.MASK):
                    separated_features[category][feat_name] = feature
                elif utils.is_category(category, constant.InputCategoryPlace.SEQ):
                    seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                    separated_features[category][seq_name][feat_name] = feature
                else:
                    separated_features[category][feat_name] = feature
            # get variables
            emb_name2emb_variable = dict()
            for emb_name, _feat_name in self._get_emb_name2feat_name().items():
                feat_name = constant.FEATURE_SEPARATOR.join((constant.FeaturePrefix.FEAT, _feat_name))
                emb_shape = self._input_config[feat_name][constant.InputConfigKeys.EMB_SHAPE]
                trainable = self._emb_trainable_dict[emb_name]
                emb_var = tf.get_variable(emb_name, shape=emb_shape, dtype=tf.float32, trainable=trainable)
                emb_name2emb_variable[emb_name] = emb_var

            # fetch embeddings
            emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_VEC], emb_name2emb_variable)
            separated_features[constant.InputCategory.EMB_VEC] = emb_feat
            emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_TGT], emb_name2emb_variable)
            separated_features[constant.InputCategory.EMB_TGT] = emb_feat
            for seq_name in self._seq_names:
                emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_SEQ][seq_name],
                                                 emb_name2emb_variable)
                separated_features[constant.InputCategory.EMB_SEQ][seq_name] = emb_feat

            # concat tgt and vec
            if self._use_vec:
                vec_cat = concat_emb_val(separated_features[constant.InputCategory.EMB_VEC],
                                         separated_features[constant.InputCategory.VAL_VEC], name='concat_vec')
            if self._use_seq:
                seq_cat = [concat_emb_val(separated_features[constant.InputCategory.EMB_SEQ][seq_name],
                                          separated_features[constant.InputCategory.VAL_SEQ][seq_name],
                                          name=f'concat_seq_{seq_name}')
                           for seq_name in self._seq_names]
            tgt_cat = concat_emb_val(separated_features[constant.InputCategory.EMB_TGT],
                                     separated_features[constant.InputCategory.VAL_TGT], name='concat_tgt')

        # apply attention and sum pool, to prepare inputs for forward_net
        forward_net_inps = []
        if self._use_vec:
            forward_net_inps.append(vec_cat)
        if self._use_seq:
            ordered_masks = get_ordered_dict_values(separated_features[constant.InputCategory.MASK])
            for seq_name, seq, mask in zip(self._seq_names, seq_cat, ordered_masks):
                if self._use_seq[seq_name]:
                    att = self._attention_layers[seq_name]
                    att_seq = att({
                        constant.InputCategoryPlace.TGT: tgt_cat,
                        constant.InputCategoryPlace.SEQ: seq,
                        constant.InputCategory.MASK: mask,
                    }, mode=mode, name=f'attention_{seq_name}')
                    forward_net_inps.append(att_seq)
                    if self._include_his_sum:
                        sum_seq = tf.reduce_sum(seq, axis=1, name=f'sum_{seq_name}')
                        forward_net_inps.append(sum_seq)
        forward_net_inps.append(tgt_cat)
        # prepare forward net input
        forward_net_inps = tf.concat(forward_net_inps, axis=-1, name='concat_for_forward_net')

        # forward net op
        if self._batch_norm is not None:
            if self._batch_norm == 'bn':
                bn = tf.layers.BatchNormalization(name='forward_net_bn', trainable=self._bn_before_forward_net_trainable)
                if self._use_moving_statistics is True:
                    training = (mode == tf.estimator.ModeKeys.TRAIN)
                elif self._use_moving_statistics is False:
                    training = True
                elif self._use_moving_statistics == 'always':
                    training = False
                else:
                    raise ValueError('unrecognized _use_moving_statistics attribute: {}'.format(
                        self._use_moving_statistics)
                    )
                forward_net_inps = bn(forward_net_inps, training=training)
            elif self._batch_norm == 'ln':
                bn = layers.LayerNormalization()
                forward_net_inps = bn(forward_net_inps, name='forward_net_bn', mode=mode)
            else:
                raise ValueError('batch_norm should be None or bn or ln, but got %s' % self._batch_norm)
        if self._use_moving_statistics == 'always':
            _mode = tf.estimator.ModeKeys.EVAL  # always use moving statistics, mode as train
        else:
            _mode = mode if self._use_moving_statistics else tf.estimator.ModeKeys.TRAIN
        logits = self._forward_net(forward_net_inps, name='forward_net', mode=_mode)
        return logits

    def freeze_embeddings(self, emb_names=None):
        """freeze the embeddings specified in layer_names

        Args:
            emb_names:
                - str: freeze the embedding layer
                - list: freeze embedding layers specified
                - None: freeze all embedding layers
        """
        if emb_names is None:
            emb_names = self._emb_trainable_dict.keys()
        if isinstance(emb_names, str):
            emb_names = [emb_names]

        for name in emb_names:
            self._emb_trainable_dict[name] = False

    def freeze_bn(self):
        """freeze the batchnorm layer before forward_net
        """
        self._bn_before_forward_net_trainable = False

    def freeze_forward_net(self, layer_names=None):
        """freeze the forward_net layers specified in layer_names

        keys: dense_l{idx}, activation_l{idx}
        """
        self._forward_net.freeze(layer_names)

    def freeze_attention(self, seq_names=None, layer_names=None):
        """freeze the attention layers specified by seq_name and layer_names

        - when seq_names is None, freeze all attention layers with layer_names
        - when layer_names is None, freeze all layers in the attention layer corresponding to seq_name
        """
        if seq_names is None:
            seq_names = self._seq_names
        for seq_name in seq_names:
            layer = self._attention_layers[seq_name]
            layer.freeze(layer_names)

    def freeze_all(self):
        """freeze all layers
        """
        self.freeze_bn()
        self.freeze_attention()
        self.freeze_embeddings()
        self.freeze_forward_net()

    def __get_embeddings(self, feat_dict, emb_name2emb_variable):
        """get the corresponding embedding of the coming feat_dict, embeddings will be processed according to self._input_config

        Args:
            feat_dict: feat_name -> feat_values
            emb_name2emb_variable: the embedding dictionary

        Returns:
            the fetched embeddings
        """
        emb_feats = dict()
        for feat_name in feat_dict:
            _feat_name = feat_name.split(constant.FEATURE_SEPARATOR)[-1]
            emb_indices = feat_dict[feat_name]
            emb_name = self._emb_dict[_feat_name]
            # fetch values from feature config
            config = self._input_config[feat_name]
            emb_shape = config[constant.InputConfigKeys.EMB_SHAPE]
            default_val = config.get(constant.InputConfigKeys.DEFAULT_VAL, 0)
            emb_process = config.get(constant.InputConfigKeys.EMB_PROCESS, 'concat')

            # padded values may exceed the boundaries, force those to the configed default_val
            emb_indices = utils.replace_out_of_ranges(
                value=emb_indices, target_range=(0, emb_shape[0]), name='remove_invalid_emb_id', replace_value=default_val)
            embeddings = emb_name2emb_variable[emb_name]  # the embedding table for the feature
            if emb_process == 'concat':
                emb_feat = tf.nn.embedding_lookup(embeddings, emb_indices, name=f'lookup_{_feat_name}')
            else:
                assert emb_process == 'mean_skip_padding'
                # note: if no valid idx in the group given, the embedding will be reduced to zero
                emb_mask = tf.equal(emb_indices, default_val, name='get_non-padding_mask')  # convert to bool tensor
                weights = tf.where(emb_mask, tf.zeros_like(emb_mask, dtype=tf.float32),
                                   tf.ones_like(emb_mask, dtype=tf.float32), name='convert_mask2weight')  # this is to weight the embeddings shape=(*, d)
                weights_sum = tf.reduce_sum(weights, axis=-1, name='sum_weights', keepdims=True) + 1e-7  # the number of valid entries for each embedding group shape=(*, 1)
                weights = weights / weights_sum  # get the real weights (sum = 1)
                weights = tf.expand_dims(weights, -1)  # expand to the same len(shape) with emb_feat
                emb_feat = tf.nn.embedding_lookup(embeddings, emb_indices, name=f'lookup_{_feat_name}')
                emb_feat = emb_feat * weights  # apply reduce
                emb_feat = tf.reduce_sum(emb_feat, axis=-2, keepdims=True)  # reduce on the embedding group dim
            emb_feats[feat_name] = emb_feat
        return emb_feats

    def __get_net(self, attention_layers, forward_net):  # create default layers and apply check
        # create default attention layers
        if attention_layers is None:  # construct default attention
            attention_layers = {seq_name: layers.DinAttention() for seq_name in self._seq_names}
        else:  # check the number of attention layers match the number of sequences
            if len(attention_layers) != len(self._seq_names):
                raise ValueError(f"the number of attention layers should match the number of sequences, "
                                 f"but got {len(attention_layers)} layers and {len(self._seq_names)} sequences")

        # create default forward net
        if forward_net is None:
            forward_net = layers.MLP(layer_dims=[200, 80, 2], activations=layers.Dice())

        # check for layer superclass
        for seq_name, layer in attention_layers.items():
            if not isinstance(layer, layers.AirLayer):
                raise ValueError(f"attention_layers should subcalss layers.AirLayer, "
                                 f"but the {seq_name} attention layer is {type(layer)}")
        if not isinstance(forward_net, layers.AirLayer):
            raise ValueError(f"forward_net should subclass layers.AirLayer, but find {type(forward_net)}")
        return attention_layers, forward_net
