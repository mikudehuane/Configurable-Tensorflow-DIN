# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 下午2:00
# @Author  : islander
# @File    : generate_samples.py
# @Software: PyCharm

"""A tool script for generating test samples
"""

from ..scripts import data
import random
import numpy as np


def generate_sample(
        fp, fea_config, *,
        num_samples=1000
):
    """generate random samples according to fea_config

    Args:
        fp: target file path
        fea_config: feature configuration
        num_samples: number of samples to generate
    """

    def _generate_line():
        row_content = []

        # generate label
        label = random.randint(0, 1)
        row_content.append(label)

        rand_shapes = dict()  # seq_name -> shape[0], shape[0] is the same for features in the same sequence

        for fea_name, config in fea_config.items():
            category = config['category']
            shape = config['shape']

            if category.endswith('_seq'):  # sequence
                if config['seq_name'] not in rand_shapes:
                    # random length, history length can exceed shape
                    rand_shapes[config['seq_name']] = random.randint(0, shape[0] * 2)
                rand_shape = (rand_shapes[config['seq_name']], *shape[1:])

                if category.startswith('emb_'):
                    feat = np.random.randint(low=0, high=config['emb_shape'][0], size=rand_shape).reshape(-1)
                else:
                    feat = np.random.random(size=rand_shape).reshape(-1)
            else:  # single value
                if category.startswith('emb_'):
                    feat = np.random.randint(low=0, high=config['emb_shape'][0], size=shape).reshape(-1)
                else:
                    feat = np.random.random(size=shape).reshape(-1)

            feat = ' '.join([str(x) for x in feat])
            row_content.append(feat)

        row_content = ','.join([str(x) for x in row_content])
        return row_content

    with open(fp, 'w') as f:
        for _ in range(num_samples):
            f.write(_generate_line() + '\n')


def main():
    random.seed(0)
    np.random.seed(0)

    generate_sample(fp='train.csv', fea_config=data.FEA_CONFIG, num_samples=10000)
    generate_sample(fp='test.csv', fea_config=data.FEA_CONFIG, num_samples=500)


if __name__ == '__main__':
    main()
