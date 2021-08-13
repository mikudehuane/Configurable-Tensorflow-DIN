# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午12:02
# @Author  : islander
# @File    : ModelFrame.py
# @Software: PyCharm

import tensorflow as tf


class NetNotBuiltError(Exception):
    pass


class ModelFrame():
    """构建模型的代码框架，定义了构建模型的通用代码，让 tf 计算图对象放在模型对象内，从而可以类似 pytorch 调用

    """
    def __init__(self):
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
