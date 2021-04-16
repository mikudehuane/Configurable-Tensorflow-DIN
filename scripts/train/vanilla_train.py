# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 5:17 下午
# @Author  : islander
# @File    : vanilla_train.py
# @Software: PyCharm


import tensorflow as tf
from typing import Dict


# wrapper of train, replace ph and sess with model.Din
def train_by_net(net, **kwargs):
    return train(sess=net.session,
                 fea_ph=net.features_ph, label_ph=net.labels_ph, outputs=net.outputs,
                 **kwargs)


def train(
        sess: tf.Session,
        fea_ph: Dict[str, tf.Tensor], label_ph: tf.Tensor, outputs: Dict[str, tf.Tensor],
        input_fn,
        train_steps,
        verbose=False
):
    """train model by input_fn, without any log or checkpoint

    Args:
        sess: the session to be run
        input_fn: generator to produce inputs
        fea_ph: feature input placeholder of the computation graph
        label_ph: label input placeholder of the computation graph
        outputs: dict of output nodes in the computation graph
        train_steps: maximum training steps
        verbose: whether to print progress

    Returns:
        True if input_fn has data else False
    """
    steps_count = 0

    for features, labels in input_fn(reset=False):
        # associate placeholder with data
        feed_dict = {fea_ph[feat_name]: features[feat_name] for feat_name in fea_ph}
        feed_dict[label_ph] = labels
        # do real computation
        _ = sess.run(outputs, feed_dict=feed_dict)

        steps_count += 1
        if verbose:
            if steps_count % 100 == 0:
                print('{} steps passed'.format(steps_count))
        if steps_count >= train_steps:
            break

    if steps_count == 0:
        return False
    else:
        return True
