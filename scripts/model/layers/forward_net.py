# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 2:27 下午
# @Author  : islander
# @File    : mlp.py
# @Software: PyCharm

import tensorflow as tf
from .interface import AirLayer, AirActivation


class LayerNormalization(AirLayer):
    """layer normalization layer, normalize on axis 1

    Similar to batchnorm, trainable scaling and bias variable for each component

    Keyword Args:
        epsilon: arithmetic robustness
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._epsilon = kwargs.pop('epsilon', 1e-3)
        self.layer_name2trainable['gamma'] = True
        self.layer_name2trainable['beta'] = True

    def __call__(self, inputs, name, mode=tf.estimator.ModeKeys.TRAIN):
        with tf.variable_scope(name):
            with tf.name_scope('norm_inputs'):
                mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
                bias = inputs - mean
                variance = tf.reduce_mean(bias ** 2, axis=1, keepdims=True)
                inputs: tf.Tensor = bias / tf.sqrt(variance + self._epsilon)
            scale_dim = inputs.get_shape().as_list()[-1]
            gamma = tf.get_variable('gamma', shape=(scale_dim,), dtype=tf.float32,
                                    trainable=self.layer_name2trainable['gamma'])
            beta = tf.get_variable('beta', shape=(scale_dim,), dtype=tf.float32,
                                   trainable=self.layer_name2trainable['beta'])
            inputs = inputs * gamma + beta
            return inputs


class MLP(AirLayer):
    """customized MLP layer

    Args:
        layer_dims: output dimensions of each dense layer
        activations: can be one of the followings, last layer is not affected
            - a layers.AirActivation subclass
            - a str specify activation (corresponding to tf.layers.Layer), see MLP.allowed_activations
            - None for no activation
            - a list of above types

    Keyword Args:
        last_layer_activation: layers.AirActivation/str/None
            specify the activation for the last layer, default as None (no activation)

    Examples:
        net = model.layers.MLP(layer_dims=[4, 5, 3], activations=Dice())
        inputs = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
        net.freeze(['dense1', 'activation2'])
        output = net(inputs, name='mlp', mode=tf.estimator.ModeKeys.TRAIN)

    Raises:
        ValueError:
            - if the number of activations given is wrong
            - if specified activation is not recognized
    """

    allowed_activations = {
        'sigmoid': tf.sigmoid,
        'relu': tf.nn.relu,
    }

    def __init__(self, layer_dims, activations='sigmoid', **kwargs):
        super().__init__()

        last_layer_activation = kwargs.pop('last_layer_activation', None)

        # handle non-list activations
        num_layers = len(layer_dims)
        if not isinstance(activations, list):
            activations = [activations for _ in range(num_layers - 1)]
            activations.append(last_layer_activation)
        else:
            if len(activations) != num_layers - 1:
                raise ValueError('the number of activations given should be the number of layers minus one, '
                                 f'but got {len(activations)} activations with {num_layers} layers')

        for activation in activations:
            if activation is not None and not isinstance(activation, AirActivation) and not isinstance(activation, str):
                raise ValueError(f'activations should be one from (None, AirActivation, str), '
                                 f'but got type {type(activation)}')
            if isinstance(activation, str) and activation not in MLP.allowed_activations:
                raise ValueError(f'activations given as str can only be one of {MLP.allowed_activations.keys()} '
                                 f'but got {activation}')

        self._activations = activations
        self._layer_dims = layer_dims

        for layer_idx in range(num_layers):
            dense_name = f'dense{layer_idx}'
            activation_name = f'activation{layer_idx}'
            self.layer_name2trainable[dense_name] = True
            self.layer_name2trainable[activation_name] = True

        if len(kwargs) != 0:
            raise ValueError(f'Unrecognized kwargs: {kwargs}')

    def __call__(self, inputs, name, mode=tf.estimator.ModeKeys.TRAIN):
        with tf.variable_scope(name, reuse=False):
            for idx, (activation, dim) in enumerate(zip(self._activations, self._layer_dims)):
                dense_name = f'dense{idx}'
                activation_name = f'activation{idx}'
                dense_trainable = self.layer_name2trainable[dense_name]
                activation_trainable = self.layer_name2trainable[activation_name]

                dense_layer = tf.layers.Dense(dim, name=dense_name, trainable=dense_trainable)
                inputs = dense_layer(inputs)

                if activation is not None:
                    if isinstance(activation, AirActivation):
                        inputs = activation(inputs, name=activation_name, trainable=activation_trainable, mode=mode)
                    elif isinstance(activation, str):
                        activation = MLP.allowed_activations[activation]
                        inputs = activation(inputs, name=activation_name)
                    else:
                        raise RuntimeError(f'activation should be None/AirActivation/str, but got {type(activation)}, '
                                           f'this is a bug in forward_net, please fix it')
            return inputs
