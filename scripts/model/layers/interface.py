# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 4:34 下午
# @Author  : islander
# @File    : attention.py
# @Software: PyCharm

import tensorflow as tf
from abc import ABC, abstractmethod


class AirLayer(ABC):
    """interface for a customized layer

    Examples:
        def __init__(...):
            super().__init__()
            # associate hyper-parameters with members
            # fill layer_name2trainable with all layers that are possible to be trainable, and map to proper defaults

        def __call__(self, inputs, name, mode=tf.estimator.ModeKeys.TRAIN),
            with tf.name_scope(name):
                # create local variables for tf.Variable
                # create layers, use self.layer_name2trainable as parameters for trainable
                # implement forward pass

    Attributes:
        layer_name2trainable: map layer names to bool, if trainable, assign True

    Notes:
        regard this class as a pure wrapper of parameters:
            - do not create any tensor outside __call__ or associate tensors with member variables,
            - otherwise, estimator will raise error: not the same graph
    """

    def __init__(self):
        self.layer_name2trainable = dict()

    @abstractmethod
    def __call__(self, inputs, name, mode=tf.estimator.ModeKeys.TRAIN):
        """call the net

        Args:
            inputs: keyed inputs or Tensor
            name: name of this operation
            mode: graph execution mode, affect batchnorm, etc.

        Returns:
            layer output
        """
        pass

    def freeze(self, layer_names=None):
        """freeze the parameters of the layer with name layer_name

        Args:
            layer_names: the layer to be frozen (key in self.layers), None means freeze all layers
                can be a list of layer names

        Raises:
            ValueError: if layer_names not match keys in layers
        """
        if layer_names is None:
            layer_names = self.layer_name2trainable.keys()
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for name in layer_names:
            if name not in self.layer_name2trainable:
                raise ValueError(f'layer_names provided for freeze should appear in layers, '
                                 f'but got {name} not in {self.layer_name2trainable.keys()}')
            self.freeze_by_name(name)

    def freeze_by_name(self, name):
        """freeze a layer by name, can be overrided by subclasses, for example, to recursively freeze sub layers

        Args:
            name: a key in layer_name2trainable
        """
        self.layer_name2trainable[name] = False


class AirActivation(ABC):
    """interface for your customized activation function

    Args:
        name: name of the dice layer

    Examples:
        class Dice(AirActivation):
            def __init__(self, name, trainable=True, **kwargs):
                super().__init__(name=name, trainable=trainable)
                # other initializations
            def __call__(self, inputs, trainable, mode=tf.estimator.ModeKeys.TRAIN):
                # define tensors
                # implement forward pass

    Notes:
        do not create any tensor outside __call__ or associate any tensor with members
    """

    @abstractmethod
    def __call__(self, inputs, name, trainable, mode=tf.estimator.ModeKeys.TRAIN):
        pass
