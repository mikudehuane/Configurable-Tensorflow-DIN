# -*- coding: utf-8 -*-
# @Time    : 2020/11/24 17:51
# @Author  : islander
# @File    : model.py
from pprint import pprint
from typing import Dict, List

import tensorflow as tf

from . import layers
from .utils import *
from . import utils


class NetNotBuiltError(Exception):
    pass


class Din(object):
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
        required_feat_names: keys in features passed to the build_net func, by default, [].
            for each registered feat, the model will map the input to a named input via identity.

    Attributes:
        tensor_name_dict: maps keys in features to the tensorflow name in the computation graph,
                additionally, the following tensor names will also be added:
                    - probs: computed probabilities
                    - labels: true labels
                    - loss: loss in session
                note that this attribute will be updated in-place
        features_ph: placeholder for the input features, if multiple graphs built, consistent with the current graph
        labels_ph: placeholder for the labels, if multiple graphs built, consistent with the current graph
        outputs: placeholder for the outputs, if multiple graphs built, consistent with the current graph
        session: running session for the model, if multiple graphs built, consistent with the current graph
        saver: checkpoint saver for the graph
        current_graph: key of the current_graph

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
                 required_feat_names: List[str] = None,
                 **kwargs):
        super().__init__()

        # kwargs
        self._batch_norm = kwargs.pop('batch_norm', 'bn')
        self._include_his_sum = kwargs.pop('include_his_sum', True)
        self._use_moving_statistics = kwargs.pop('use_moving_statistics', True)
        self._epsilon = kwargs.pop('epsilon', 1e-7)
        # will be override later in config
        self._use_vec = False
        self._use_seq = False

        if required_feat_names is None:
            required_feat_names = []
        self._required_feat_names = required_feat_names

        # generate tensor_name_dict
        self.tensor_name_dict = {
            'probs': 'predict/probabilities',
            'labels': 'labels',
            'loss': 'compute_loss/loss',
        }
        for feat_name in self._required_feat_names:
            self.tensor_name_dict[feat_name] = feat_name

        # check whether the input_config and shared_emb_config is valid
        utils.check_config(input_config=input_config, shared_emb_config=shared_emb_config)

        # declare index member variables
        self._input_config_raw = input_config
        # map feature name to configurations
        # feat_name -> config
        self._input_config = utils.get_full_input_config(input_config=input_config)
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

        self._current_graph = None
        self._features_phs = {}
        self._labels_phs = {}
        self._outputss = {}
        self._sessions = {}
        self._graphs = {}
        self._savers = {}

    @property
    def features_ph(self):
        return self._features_phs[self._current_graph]

    @property
    def labels_ph(self):
        return self._labels_phs[self._current_graph]

    @property
    def outputs(self):
        return self._outputss[self._current_graph]

    @property
    def session(self) -> tf.Session:
        return self._sessions[self._current_graph]

    @property
    def saver(self) -> tf.train.Saver:
        return self._savers[self._current_graph]

    @property
    def graph(self) -> tf.Graph:
        return self._graphs[self._current_graph]

    @property
    def current_graph(self):
        return self._current_graph

    def switch_graph(self, key):
        """switch the current graph

        Args:
            key: key for the new graph, consistent with the one provided for Din.build_graph_
        """
        if not self.has_graph(key=key):
            raise NetNotBuiltError('attempting to switch to graph {}, but has built only the following graphs: {}'.format(
                key, self._graphs.keys()
            ))
        self._current_graph = key

    def has_graph(self, key):
        return key in self._graphs

    def load_from_to(self, from_key, to_key):
        """encapsulation of load_from, load values from from_key to to_key
        """
        tmp = self.current_graph
        self.switch_graph(key=to_key)
        self.load_from(from_key)
        self.switch_graph(tmp)  # to avoid change current graph

    def load_from(self, key):
        """load the variables from another graph given by key

        Args:
            key: source graph key
        """
        source_sess = self._sessions[key]

        def _get_var_name(variable):
            # vars are indicated by name, discard the ':\d' on the right
            return variable.name.rsplit(':', 1)[0]

        # get values from the source session
        def _get_val_var(sess):
            with sess.graph.as_default():
                # var_name -> (var_value, var_dtype)
                return {_get_var_name(_var): (sess.run(_var), _var.dtype) for _var in tf.global_variables()}
        source_val_var = _get_val_var(source_sess)

        # apply values to the current session
        def _load_values(sess, values):
            with sess.graph.as_default():
                for variable in tf.global_variables():
                    val, dtype = values[_get_var_name(variable)]
                    variable.load(val, sess)
        _load_values(self.session, source_val_var)

    def _get_emb_name2feat_name(self):
        emb_name2feat_name = dict()
        for feat_name, emb_name in self._emb_dict.items():
            emb_name2feat_name[emb_name] = feat_name
        return emb_name2feat_name

    def build_graph_(self, key, *,
                     mode, device='gpu', optimizer=None, seed=None):
        """build the graph in self, and initializing pars with random values

        Args:
            mode: tf.estimator.ModeKeys.TRAIN or EVAL
            device: 'cpu' or 'gpu'
            key: indicator for the current graph, when switch mode, will use the key
                suggested to be set to 'train' and 'eval'
            optimizer: optimizer for the model, None means do not build the optimization part
            seed: random seed for graph initialization
        """
        graph = tf.Graph()
        with graph.as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)
            with tf.device(device):
                # generate input placeholders
                features_ph, labels_ph = utils.get_inputs_ph(input_config=self._input_config_raw, batch_size=None)
                outputs = self.build_graph(features=features_ph, labels=labels_ph, mode=mode,
                                           params={'optimizer': optimizer})
                session_config = tf.ConfigProto(allow_soft_placement=True,
                                                gpu_options=tf.GPUOptions(allow_growth=True))
                session = tf.Session(config=session_config)
                session.run(tf.global_variables_initializer())
                saver = tf.train.Saver(max_to_keep=None)

        self._features_phs[key] = features_ph
        self._labels_phs[key] = labels_ph
        self._outputss[key] = outputs
        self._sessions[key] = session
        self._graphs[key] = graph
        self._savers[key] = saver

        if self._current_graph is None:  # first graph, set current graph to this graph
            self._current_graph = key

    def build_graph(self, features: Dict[str, tf.Tensor], labels: tf.Tensor, mode, params=None):
        """build the training graph (including optimization)

        Args: the same with model_fn

        Returns:
            a dict of result tensors
        """
        ret = dict()

        # name the inputs
        tf.identity(labels, name=self.tensor_name_dict['labels'])
        for feat_name in self._required_feat_names:
            feature = features[feat_name]
            tf.identity(feature, name=self.tensor_name_dict[feat_name])

        ret["labels"] = labels
        ret["logits"] = self.forward(features=features, mode=mode)

        with tf.name_scope('predict'):
            ret["probs"] = tf.nn.softmax(ret["logits"], name="probabilities")
            ret["classes"] = tf.argmax(input=ret["logits"], axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        with tf.name_scope('compute_loss'):
            # tf.losses.sparse_softmax_cross_entropy produces strange results only in odps, seems like a bug?
            labels_one_hot = tf.one_hot(ret["labels"], depth=ret["logits"].get_shape()[-1], name='one_hot')
            neg_log_probs = - tf.log(ret["probs"] + self._epsilon, name='log_probabilities')
            loss_sum = tf.reduce_sum(labels_one_hot * neg_log_probs, axis=1, name='true_label_neg_log_prob')
            ret["loss"] = tf.reduce_mean(loss_sum, axis=0, name='loss')

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.name_scope('optimize'):
                optimizer = params.pop('optimizer')
                optimizer: tf.train.Optimizer
                ret['gradient'] = tf.gradients(ret["loss"], tf.trainable_variables())
                ret['gradient'], _ = tf.clip_by_global_norm(ret['gradient'], 5)
                ret['gradient'] = list(zip(ret['gradient'], tf.trainable_variables()))
                train_op = optimizer.apply_gradients(
                    ret['gradient'], global_step=tf.train.get_or_create_global_step()
                )
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                ret["train_op"] = tf.group([train_op, update_ops])

        return ret

    def model_fn(self, features, labels, mode, params):
        """model_fn for estimator
        """
        outputs = self.build_graph(features=features, labels=labels, mode=mode, params=params)

        predictions = {
            "classes": outputs['classes'],
            "probs": outputs['probs'],
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=outputs['loss'], train_op=outputs['train_op'])

        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.name_scope('evaluate'):
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(labels=outputs['labels'], predictions=predictions['classes']),
                }
                if outputs['probs'].shape[1] == 2:  # only log binary aucs
                    # use tf auc is calculated on multi-classes, but it diverges from binary auc, so have to transform
                    true_probs = outputs['probs'][:, 1]
                    num_thresholds = 10000
                    eval_metric_ops.update({
                        "auc": tf.metrics.auc(labels=outputs['labels'], predictions=true_probs, name='auc',
                                              num_thresholds=num_thresholds, summation_method='careful_interpolation'),
                        # lower bound of auc approximation
                        "auc_min": tf.metrics.auc(labels=outputs['labels'], predictions=true_probs,
                                                  num_thresholds=num_thresholds,
                                                  summation_method='minoring', name='auc_min'),
                        # upper bound of auc approximation
                        "auc_max": tf.metrics.auc(labels=outputs['labels'], predictions=true_probs,
                                                  num_thresholds=num_thresholds,
                                                  summation_method='majoring', name='auc_max'),
                    })
            return tf.estimator.EstimatorSpec(mode=mode, loss=outputs['loss'], eval_metric_ops=eval_metric_ops)

        raise ValueError(f'mode should be tf.estimator.ModeKeys, but got {mode}')

    def forward(self, features, mode):
        """forward pass and produce logits

        Raises:
            RuntimeError: when config parsing failed
        """
        tf.train.get_or_create_global_step()

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
        emb_feats = dict()
        for feat_name in feat_dict:
            _feat_name = feat_name.split(constant.FEATURE_SEPARATOR)[-1]
            emb_indices = feat_dict[feat_name]
            emb_name = self._emb_dict[_feat_name]
            emb_shape = self._input_config[feat_name][constant.InputConfigKeys.EMB_SHAPE]

            emb_indices = utils.replace_out_of_ranges(
                value=emb_indices, target_range=(0, emb_shape[0]), name='remove_invalid_emb_id', replace_value=0
            )
            embeddings = emb_name2emb_variable[emb_name]
            emb_feat = tf.nn.embedding_lookup(embeddings, emb_indices, name=f'lookup_{_feat_name}')
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
