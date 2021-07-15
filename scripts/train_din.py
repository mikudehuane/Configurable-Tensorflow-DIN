# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 3:23 下午
# @Author  : islander
# @File    : train_vanilla_loop.py
# @Software: PyCharm


def main():
    # entry test script, absolute import project modules
    import tensorflow as tf
    # noinspection PyUnresolvedReferences
    import data
    # noinspection PyUnresolvedReferences
    import model
    # noinspection PyUnresolvedReferences
    import common
    # noinspection PyUnresolvedReferences
    import train

    metric_dict = {
        'auc': train.metrics.auc,
        'accuracy': train.metrics.accuracy,
        'false_prop': train.metrics.false_prop,
        'loss': train.metrics.neg_log_loss,
        'num_samples': train.metrics.num_samples,
        'max_true_prob': train.metrics.max_true_prob,
    }

    # input object
    def _get_data_gen(source_fp, shuffle):
        _reader = data.DataReader(source_fp=source_fp)
        if shuffle:
            _reader = data.ShuffleReader(_reader)
        _gen = data.DataGenerator(reader=_reader, batch_size=1024)
        return _gen

    gen_train = _get_data_gen(common.train_data_fp, shuffle=True)
    gen_test = _get_data_gen(common.test_data_fp, shuffle=False)

    # get the model object
    net = model.Din(
        input_config=data.FEA_CONFIG,
        shared_emb_config=data.SHARED_EMB_CONFIG,
        use_moving_statistics=True,
    )

    net.build_graph_(key='train', mode=tf.estimator.ModeKeys.TRAIN,
                     optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0), seed=0)
    net.build_graph_(key='eval', mode=tf.estimator.ModeKeys.EVAL)

    evaluate_every = 2000
    while True:
        # switch to training graph
        net.switch_graph('train')

        is_model_trained = train.vanilla_train.train_by_net(
            net=net, input_fn=gen_train,
            train_steps=evaluate_every, verbose=True
        )
        if not is_model_trained:
            break

        # get training global step
        with net.graph.as_default():
            global_step = net.session.run(tf.train.get_global_step())

        net.switch_graph('eval')
        net.load_from('train')

        model_pred_outputs = train.evaluate.evaluate_by_net(
            net=net,
            input_fn=gen_test
        )
        probs = model_pred_outputs['probs']
        labels = model_pred_outputs['labels']

        msg = ['step:{}'.format(global_step)]
        for metric_name, metric_func in metric_dict.items():
            metric_value = metric_func(labels, probs)
            msg.append('{}:{}'.format(metric_name, metric_value))
        msg = ' '.join(msg)
        print(msg)


if __name__ == '__main__':
    main()
