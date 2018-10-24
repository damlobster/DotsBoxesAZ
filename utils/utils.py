import asyncio
import multiprocessing as mp
import copy
import numpy as np
import os
import pickle
import torch
from torch.utils import data
import pandas as pd


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)
        
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self.__setitem__(key, value)

class DictWithDefault(dict):
    def __init__(self, lmbda):
        super(DictWithDefault, self).__init__()
        self.lmbda = lmbda

    def __missing__(self, key):
        res = self[key] = self.lmbda(key)
        return res

class PickleDataset(data.Dataset):
    def __init__(self, data_directory, file=None, from_idx=0, to_idx=int(1e12)):
        self.data = []
        files = sorted(os.listdir(data_directory), reverse=True)
        for fn in files:
            if file is None or file == fn:
                with open(data_directory+fn, "rb") as f:
                    self.data.extend([sample for sample in pickle.load(f)])
                print("selfplay samples loaded from: " + fn)
                if len(self.data) > to_idx:
                    break
        self.data = self.data[from_idx:to_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        board, visits, value = self.data[index]

        # Load data and get label
        return board, visits, np.asarray([value], dtype=np.float32)


def _reshape(X, features_shape):            
    if features_shape is None:
        return X
    else:
        from operator import mul
        from functools import reduce
        assert reduce(mul, features_shape) == X.shape[1], "Cannot reshape features to: {}".format(features_shape)
        return X.reshape((-1, *features_shape))

class HDFStoreDataset(data.Dataset):
    def __init__(self, hdf_file, key, train, features_shape=None, where=None, max_samples=500000):
        super(HDFStoreDataset, self).__init__()
        with pd.HDFStore(hdf_file, "r") as hdf_store:
            df = hdf_store.select(key, where)
            df.sort_index(ascending=False, inplace=True)
            df = df[:max_samples][df.training == (1 if train else -1)]
            cols = df.columns
            self.features = _reshape(df[list(c for c in cols if c.startswith("x_"))].values.astype(np.float32), features_shape)
            self.policy = df[list(c for c in cols if c.startswith("pi_"))].values.astype(np.float32)
            self.value = df.z.values.astype(np.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        board = self.features[index]
        policy = self.policy[index]
        value = self.value[index]
        return board, policy, np.asarray([value])

def write_to_hdf(hdf_file, key, dataframe):
    with pd.HDFStore(hdf_file, mode="a") as store:
        store.append(key, dataframe, format="table")

def get_cuda_devices_list():
    return list(map(lambda id: "cuda:"+str(id), range(torch.cuda.device_count())))
