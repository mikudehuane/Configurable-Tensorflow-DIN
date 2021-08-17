# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 2:33 下午
# @Author  : islander
# @File    : __init__.py
# @Software: PyCharm

"""
Notes:
    use forward to define a customized training loop:
        - to freeze a layer in the middle of a run, please define another graph by "with tf.Graph()",
"""

from . import layers
from .din import Din
from .utils import constant
from . import utils
from .model_frame import ModelFrame
