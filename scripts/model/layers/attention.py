# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 2:27 下午
# @Author  : islander
# @File    : attention.py
# @Software: PyCharm


import tensorflow as tf
from .interface import AirLayer
from .forward_net import MLP
from ..utils import constant


class DinAttention(AirLayer):
    """Din attention layer, produce attentioned sequence

    Args:
        forward_net: after concat input tensor, they will be fed into this net,
            by default a MLP with hidden_dim 80-40, sigmoid activation

    Examples:
        net = model.layers.DinAttention()
        net.freeze_by_name('forward_net/dense0')

        tgts = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
        seqs = tf.constant([
            [[1., 2.], [4., 5.], [1., 2.], [4., 5.]],
            [[1., 2.], [4., 5.], [0., 0.], [0., 0.]],
        ], dtype=tf.float32)
        mask = tf.constant([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=tf.int32)

        output = net({constants.Din.TGT: tgts, constants.Din.SEQ: seqs, constants.Din.MASK: mask},
                     name='attention', mode=tf.estimator.ModeKeys.TRAIN)
    """

    def __init__(self, forward_net=None):
        super().__init__()
        if forward_net is None:
            # noinspection PyTypeChecker
            forward_net = MLP(layer_dims=[80, 40, 1], activations='sigmoid', last_layer_activation=None)
        forward_net: AirLayer
        self._forward_net = forward_net

        # update trainable according to forward_net, so previous freezing to forward_net is valid
        for layer_name, trainable in forward_net.layer_name2trainable.items():
            new_layer_name = '/'.join(['forward_net', layer_name])
            self.layer_name2trainable[new_layer_name] = trainable

    def __call__(self, inputs, name, mode=tf.estimator.ModeKeys.TRAIN):
        """implement forward pass

        Args:
            inputs: a dict with the following keys
                Din.TGT: embeddings of the target commodities [batch_size, dim]
                Din.SEQ: embeddings of the user history information [batch_size, his_len, dim1track]
                Din.MASK: 1 means a valid ut_track [batch_size, his_len]
            name: operation name
            mode: execution mode

        Returns:
            attentioned sequence
        """
        tgt = inputs[constant.InputCategoryPlace.TGT]
        seq = inputs[constant.InputCategoryPlace.SEQ]
        mask = inputs[constant.InputCategory.MASK]
        seq_shape = seq.get_shape().as_list()
        batch_size = seq_shape[0]
        his_len = seq_shape[1]

        with tf.variable_scope(name):
            with tf.name_scope('broadcast_target_to_sequence'):
                # repeat tgts 'his_len' times in the second dimension (to be consistent with seq)
                tgts = tf.tile(tgt, [1, his_len], name='tile')  # [batch_size, his_len*dim1track]
                # [batch_size, his_len, dim]
                tgts = tf.reshape(tgts, [-1, his_len, tgt.shape[-1]], name='reshape')

            # concat tgts and seq
            # DIEN official implementation also concats tgts-seq and tgts*seq
            # but this is unfeasible since dim != dim1track
            # [batch_size, his_len, dim+dim1track]
            attention_input = tf.concat([tgts, seq], axis=-1, name='concat_target_and_sequence')

            # compute model output
            weights = self._forward_net(attention_input, name='compute_weight_logits', mode=mode)

            with tf.name_scope('apply_mask2weights'):
                mask = tf.equal(mask, 1, name='convert2bool')  # convert to bool tensor
                mask = tf.expand_dims(mask, 1, name='broadcast_mask')  # [batch_size, 1, his_len]
                weights = tf.reshape(weights, [-1, 1, his_len], name='broadcast_weights')  # [batch_size, 1, his_len]
                with tf.name_scope('pad_weights'):
                    paddings = tf.ones_like(weights, name='pad_one') * (-2 ** 32 + 1)
                weights = tf.where(mask, weights, paddings, name='apply_mask')  # [batch_size, 1, his_len]

            weights = tf.nn.softmax(weights, name='softmax_weights')  # [batch_size, 1, his_len]

            with tf.name_scope('apply_weights'):
                # Weighted sum
                # [batch_size, 1, his_len] * [batch_size, his_len, dim1track] -> [batch_size, 1, dim1track]
                output = tf.matmul(weights, seq, name='apply_weight2seq')
                output = tf.squeeze(output, axis=1, name='squeeze_seq')

        return output

    def freeze_by_name(self, name):
        super().freeze_by_name(name)

        # also update trainable variables for mlp
        split_names = name.split('/')
        field = split_names[0]
        assert field == 'forward_net'
        content = '/'.join(split_names[1:])
        self._forward_net.freeze_by_name(content)
