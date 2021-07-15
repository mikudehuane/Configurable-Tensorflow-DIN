# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 下午2:23
# @Author  : islander
# @File    : dataset.py
# @Software: PyCharm


import random

import numpy as np
from tensorflow.python.platform import gfile
import logging

from .fea_config import FEA_CONFIG


_logger = logging.getLogger('data.dataset')


class DataReader(object):
    """data reader

    Args:
        source_fp: source data file path
        config: feature configuration

    Notes:
        max user id: 49022
        number of user: 49022
        max good id: 143533
        number of good: 143533
        max category id: 4814
        number of categories: 4814
    """
    def __init__(self, source_fp, *, config=FEA_CONFIG):
        self.source_fp = source_fp
        self.fea_config = config

        self._split_column = ','
        self._split_his = ' '

        if gfile.Exists(source_fp):
            self._reader = gfile.GFile(source_fp, 'r')
        else:
            self._reader = None

    def read(self, size=1):
        if self.is_no_data_left():
            raise EOFError('Reader reaches EOF.')

        ret = []
        for _ in range(size):
            raw_row = self._reader.readline()
            if not raw_row:
                break

            raw_row = raw_row.strip().split(self._split_column)

            # click
            row = [int(raw_row[0])]

            for config, entry in zip(self.fea_config.values(), raw_row[1:]):
                category = config['category']
                if category.endswith('_seq'):
                    if entry == '':  # no sequence
                        entry = []
                    else:
                        entry = [int(x) for x in entry.split(self._split_his)]
                else:
                    entry = int(entry)
                row.append(entry)

            ret.append(row)

        return ret

    def seek(self, offset=0):
        assert offset == 0
        self._reader.seek(0)

    def close(self):
        self._reader.close()

    def is_no_data_left(self):
        if self._reader is None:
            return True
        else:
            return self._reader.tell() == self._reader.size()


class ShuffleReader(object):
    """encapsulate DataReader, shuffle data in memory

    Args:
        reader: encapsulated reader
        cache_size: cache size for shuffling, the larger，the more random data are shuffled
    """
    # reader status
    STATE_INIT = 'init'  # init status, no cache filled
    STATE_FULL = 'full'  # cache filled, data not all read
    STATE_NO_NEW_DATA = 'no_new_data'  # data has been read
    STATE_EOF = 'eof'  # data has been read, and cache is empty

    def read(self, size: int = 1):
        """shuffle and read data

        status transformation
            INIT data_sufficient-> FULL data_insufficient-> NO_NEW_DATA empty_cache-> EOF
                 data_insufficient-> NO_NEW_DATA empty_cache-> EOF
                 emtpy_file -> EOF
        """

        def _read_data_and_switch_state(_size):  # read data and switch status
            _records = self._reader.read(_size)
            if self._reader.is_no_data_left():  # data read
                self._switch_to_state_no_new_data()
            else:
                self._state = ShuffleReader.STATE_FULL
            return _records

        if self._state == ShuffleReader.STATE_EOF:
            raise EOFError('Reader reaches EOF.')

        if self._state == ShuffleReader.STATE_INIT:  # init status fill cache
            try:
                records = _read_data_and_switch_state(self._cache_size)
            except EOFError as e:  # empty data reraise
                # only here is possible to get EOFError
                # when file is all read, switch to EOF
                self._state = ShuffleReader.STATE_EOF
                raise e
            self._cache.extend(records)  # fill cache
            _logger.debug(str(self._cache[0]))

        ret = []

        if self._state == ShuffleReader.STATE_FULL:  # new data come, get from cache, and fill cache with new one
            records = _read_data_and_switch_state(size)
            while records:  # has new data to read
                index = random.randint(0, self._cache_size - 1)
                # read to ret
                ret.append(self._cache[index])
                # fill cache
                self._cache[index] = records.pop()

        if self._state == ShuffleReader.STATE_NO_NEW_DATA:
            while self._cache:
                if len(ret) == size:  # data is sufficient
                    break
                else:
                    ret.append(self._cache.pop())
            if not self._cache:
                self._state = ShuffleReader.STATE_EOF

        return ret

    def is_no_data_left(self) -> bool:
        return self._state == ShuffleReader.STATE_EOF

    def _switch_to_state_no_new_data(self):
        """switch to STATUS_NO_NEW_DATA，and shuffle cache
        """
        self._state = ShuffleReader.STATE_NO_NEW_DATA
        random.shuffle(self._cache)

    def seek(self, offset: int) -> None:
        self._reset()
        self._reader.seek(offset)

    def _reset(self):
        """reset cache
        """
        self._state = ShuffleReader.STATE_INIT
        self._cache = []

    def __init__(self, reader, cache_size=1000):
        self._reader = reader
        self._cache_size = cache_size

        # declaration
        self._state = None
        self._cache = None

        self._reset()


class DataGenerator(object):
    """data generator, concat batch

    Args:
        reader: data reader
        batch_size: batch size
    """

    def __init__(self, reader, *, config=FEA_CONFIG, batch_size=1024):
        self._reader = reader
        self._batch_size = batch_size
        self._config = config

    def __call__(self, reset=True):
        """return a generator to generate samples

        reset: whether to reset self._reader to head
        """
        if reset:
            if self._reader is not None:  # reset reader
                self._reader.seek(0)

        while True:
            try:
                batch = self._reader.read(size=self._batch_size)
            except EOFError:
                return

            features, labels = self._get_batch(batch)
            yield features, labels

    def _get_batch(self, batch):  # process a raw list of data returned from DataReader
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError('batch size should not be zero')

        labels = np.zeros((batch_size,), dtype=np.int32)
        features = dict()
        masks = dict()
        for feat_name, config in self._config.items():
            shape = (batch_size, *config['shape'])
            dtype = np.int32 if config['category'].startswith('emb_') else np.float32
            features[feat_name] = np.zeros(shape=shape, dtype=dtype)

            if config['category'].endswith('_seq'):
                seq_name = config['seq_name']
                if seq_name not in masks:
                    masks[seq_name] = np.zeros(shape=(batch_size, config['shape'][0]), dtype=np.int32)

        mask_write = {seq_name: False for seq_name in masks}

        for index, row in enumerate(batch):
            labels[index] = row[0]

            for (feat_name, config), entry in zip(self._config.items(), row[1:]):
                category = config['category']
                shape = config['shape']
                entry = np.array(entry)
                entry = entry.reshape(-1, *shape[1:])  # first dimension (seq_len for seq unknown)
                if category.endswith('_seq'):
                    his_len = entry.shape[0]
                    if his_len > shape[0]:  # sequence too long
                        entry = entry[:shape[0]]
                    features[feat_name][index, :his_len] = entry

                    seq_name = config['seq_name']
                    if not mask_write[seq_name]:
                        masks[seq_name][index, :his_len] = 1

                else:
                    features[feat_name][index, :] = entry

        complete_features = dict()
        for name, value in features.items():
            complete_features['feat/' + name] = value
        for name, value in masks.items():
            complete_features['mask/' + name] = value

        return complete_features, labels
