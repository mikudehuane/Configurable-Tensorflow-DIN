# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 下午3:00
# @Author  : islander
# @File    : _add_scripts_path1.py
# @Software: PyCharm


import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'scripts')
add_path(lib_path)
