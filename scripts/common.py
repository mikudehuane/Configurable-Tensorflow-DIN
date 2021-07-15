# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 4:19 下午
# @Author  : islander
# @File    : config.py
# @Software: PyCharm

import os.path as osp

project_fd = osp.normpath(osp.join(__file__, '..', '..'))  # 项目根目录

# directory where the data are
data_fd = osp.join(project_fd, 'data')
train_data_fp = osp.join(data_fd, 'train.csv')
test_data_fp = osp.join(data_fd, 'test.csv')

# directory to save the running log, checkpoints, etc.
log_fd = osp.join(project_fd, 'log')
