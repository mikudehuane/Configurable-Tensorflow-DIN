# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 2:28 下午
# @Author  : islander
# @File    : activation.py
# @Software: PyCharm


import tensorflow as tf
from .interface import AirActivation


class Dice(AirActivation):
    """Dice activation layer

    Args:
        axis: the axis that are not involved in mean/std computation
        epsilon: arithmetic robustness
        momentum: use moving mean and test during training for test, the moving statistics are calculated by
            last * momentum + (1 - momentum) * now

    Examples:
        layer = model.layers.Dice()
        inputs = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
        output = layer(inputs, name='dice', trainable=True, mode=tf.estimator.ModeKeys.TRAIN)
    """
    def __init__(self, **kwargs):
        self._axis = kwargs.pop('axis', -1)
        self._epsilon = kwargs.pop('epsilon', 1e-9)
        self._momentum = kwargs.pop('momentum', 0.99)

    def __call__(self, inputs, name, trainable=True, mode=tf.estimator.ModeKeys.TRAIN):
        input_shape = inputs.get_shape().as_list()

        with tf.variable_scope(name, reuse=False):
            bn_layer = tf.layers.BatchNormalization(axis=self._axis, momentum=self._momentum, epsilon=self._epsilon,
                                                    center=False, scale=False, name='norm_inputs')
            alphas = tf.get_variable(
                'alpha', shape=[int(input_shape[self._axis])], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0), trainable=trainable,
            )

            x_normed = bn_layer(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))

            x_p = tf.sigmoid(x_normed, 'compute_prob')
            with tf.name_scope('activation'):
                activated = alphas * (1.0 - x_p) * inputs + x_p * inputs

            return activated


class PReLU(AirActivation):
    """PReLU activation layer

        Args:
            axis: the axis that are not involved in mean/std computation

        Examples:
            layer = model.layers.PReLU()
            inputs = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
            output = layer(inputs, name='prelu', trainable=True, mode=tf.estimator.ModeKeys.TRAIN)
        """

    def __init__(self, **kwargs):
        self._axis = kwargs.pop('axis', -1)

    def __call__(self, inputs, name, trainable=True, mode=tf.estimator.ModeKeys.TRAIN):
        input_shape = inputs.get_shape().as_list()
        with tf.variable_scope(name, reuse=False):
            alphas = tf.get_variable(
                'alpha', shape=[int(input_shape[self._axis])], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0), trainable=trainable,
            )

            with tf.name_scope('compute_prob'):
                ones = tf.ones_like(inputs, name='one_holder')
                zeros = tf.zeros_like(inputs, name='zero_holder')
                x_p = tf.where(inputs > 0, ones, zeros, 'threshold')

            with tf.name_scope('activation'):
                activated = alphas * (1.0 - x_p) * inputs + x_p * inputs

            return activated
