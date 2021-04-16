# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 5:15 下午
# @Author  : islander
# @File    : evaluate.py
# @Software: PyCharm


import tensorflow as tf
from typing import Dict
import numpy as np


def evaluate_by_net(net, input_fn, **kwargs):
    """encapsulate evaluate
    """
    ret = evaluate(
        graph=net.graph, sess=net.session,
        fea_ph=net.features_ph, label_ph=net.labels_ph, outputs=net.outputs,
        input_fn=input_fn, **kwargs
    )
    return ret


def evaluate(graph: tf.Graph, sess: tf.Session,
             fea_ph: Dict[str, tf.Tensor], label_ph: tf.Tensor, outputs: Dict[str, tf.Tensor],
             input_fn, *,
             test_steps=None, capture_feat_names=()) -> Dict[str, np.ndarray]:
    """evaluate according to input_fn

    Args:
        sess: the session to be run
        test_steps: the number of batches to be tested, None means test the whole set
        graph: the evaluation graph
        input_fn: generator to produce inputs
        fea_ph: feature input placeholder of the computation graph
        label_ph: label input placeholder of the computation graph
        outputs: dict of output nodes in the computation graph
        capture_feat_names: keys in features, which will be added into return values

    Returns: dict with the following content
        - outputs.keys() -> outputs.values()
        - capture_feat_names -> features[feat_name]
    """
    with graph.as_default():
        whole_outputs = dict()  # put return values
        for step, (features, labels) in enumerate(input_fn()):
            if test_steps is not None and step >= test_steps:
                break

            # associate placeholder with data
            feed_dict = {fea_ph[feat_name]: features[feat_name] for feat_name in fea_ph}
            feed_dict[label_ph] = labels
            # do computation
            outputs_ = sess.run(outputs, feed_dict=feed_dict)
            for feat_name in capture_feat_names:
                outputs_[feat_name] = features[feat_name]

            # fill outputs into return values
            for key, value in outputs_.items():
                if key not in whole_outputs:
                    whole_outputs[key] = []
                try:
                    value = list(value)
                except TypeError:  # not list type, e.g., loss, skip
                    continue
                whole_outputs[key].extend(value)

        # empty testset, construct placeholder outputs and return
        if len(whole_outputs) == 0:
            for name, tensor in outputs.items():
                shape = tensor.shape.as_list()
                if len(shape) != 0:
                    shape[0] = 0
                    tensor = np.zeros(shape)
                    whole_outputs[name] = tensor
            for feat_name in capture_feat_names:
                whole_outputs[feat_name] = np.zeros(shape=(0, 1))  # arbitrarily chosen dim[1]

        for key in whole_outputs:
            whole_outputs[key] = np.array(whole_outputs[key])

    return whole_outputs
