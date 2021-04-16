# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 4:19 下午
# @Author  : islander
# @File    : config.py
# @Software: PyCharm

# this file config variables that are likely to be referenced among all scripts


import config
import os.path as osp

# directory where the data are
data_fd = osp.join(config.project_fd, 'data')
train_data_fp = osp.join(data_fd, 'train.csv')
test_data_fp = osp.join(data_fd, 'test.csv')

# directory to save the running log, checkpoints, etc.
log_fd = osp.join(config.project_fd, 'log')
