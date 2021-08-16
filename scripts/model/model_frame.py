# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午12:02
# @Author  : islander
# @File    : ModelFrame.py
# @Software: PyCharm
from abc import abstractmethod
from typing import Dict
from . import utils

import tensorflow as tf


class NetNotBuiltError(Exception):
    pass


class ModelFrame(object):
    """framework for constructing the model

    Args:
        required_feat_names: keys in features passed to the build_net func, by default, [].
            - if given as a List, for each registered feat, the model will map the input to a named input via identity.
            - if given as a Dict, map the key to the feat corresponding to the value
        input_config: input configuration, reserve as member var, for generating input placeholder

    Keyword Args:
        epsilon: arithmetic robustness
    """
    def __init__(self, input_config, required_feat_names=None, **kwargs):
        self._input_config_raw = input_config

        self._epsilon = kwargs.pop('epsilon', 1e-7)

        self._current_graph = None
        self._features_phs = {}
        self._labels_phs = {}
        self._outputss = {}
        self._sessions = {}
        self._graphs = {}
        self._savers = {}

        if required_feat_names is None:
            required_feat_names = []
        self._required_feat_names = required_feat_names
        if not isinstance(self._required_feat_names, Dict):  # 列表转为 identity map
            self._required_feat_names = {val: val for val in self._required_feat_names}

        # generate tensor_name_dict, this is for mapping tensor names to get tensor by name
        self.tensor_name_dict = {
            'probs': 'predict/probabilities',
            'labels': 'labels',
            'loss': 'compute_loss/loss',
        }
        for semantic_feat_name, feat_name in self._required_feat_names.items():
            self.tensor_name_dict[semantic_feat_name] = feat_name

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

    def set_lr(self, new_lr):
        """change the learning rate of the current graph

        Args:
            new_lr: new learning rate

        Notes:
            this method will attempt to find the variable named learning_rate in self.graph,
            so when building graph, the optimizer should be passed with a function which create lr as a variable inside
        """
        with self.graph.as_default():
            with tf.variable_scope('', reuse=True):
                learning_rate = tf.get_variable('learning_rate')
                learning_rate.load(new_lr, self.session)

    @abstractmethod
    def forward(self, features, mode):
        pass

    def build_graph(self, features, labels, mode, params=None):
        """build the training graph (including optimization)

        Args: the same with model_fn

        Returns:
            a dict of result tensors
        """
        ret = dict()

        # name the inputs
        tf.identity(labels, name=self.tensor_name_dict['labels'])
        for semantic_feat_name, feat_name in self._required_feat_names.items():
            feature = features[feat_name]
            tf.identity(feature, name=self.tensor_name_dict[semantic_feat_name])

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

    def build_graph_(self, key, *,
                     mode, device='gpu', optimizer=None, seed=None):
        """build the graph in self, and initializing pars with random values

        Args:
            mode: tf.estimator.ModeKeys.TRAIN or EVAL
            device: 'cpu' or 'gpu'
            key: indicator for the current graph, when switch mode, will use the key
                suggested to be set to 'train' and 'eval'
            optimizer: optimizer for the model, can be the following
                - tf.train.Optimizer object
                - None (means do not build the optimization part)
                - Callable object returning a tf.train.Optimizer (this is useful to create lr as a variable)
            seed: random seed for graph initialization
        """
        graph = tf.Graph()
        with graph.as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)
            with tf.device(device):
                if callable(optimizer):
                    optimizer = optimizer()

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
