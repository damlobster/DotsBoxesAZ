import numpy as np
import torch
from torch.utils import data
import pandas as pd
import math


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

    def merge(self, merge_dct):
        for k, v in merge_dct.items():
            if (k in self and isinstance(self[k], dict)
                    and isinstance(merge_dct[k], dict)):
                self[k].merge(merge_dct[k])
            else:
                self[k] = merge_dct[k]

class DictWithDefault(dict):
    def __init__(self, lmbda):
        super(DictWithDefault, self).__init__()
        self.lmbda = lmbda

    def __missing__(self, key):
        res = self[key] = self.lmbda(key)
        return res


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
            #self.value = df.z.values.astype(np.float32)
            self.value = 0.5 * (df.z.values.astype(np.float32) + df.q_value.values.astype(np.float32))

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

def read_from_hdf(hdf_file, key, where):
    return pd.read_hdf(hdf_file, key, where=where)

def get_cuda_devices_list():
    return list(map(lambda id: "cuda:"+str(id), range(torch.cuda.device_count())))


def elo_rating(elo0, elo1, winner, K=30):
    def elo_winning_prob(rating1, rating2):
        return 1.0 / (1 + 1.0 * math.pow(10, (rating1 - rating2) / 400))

    p1 = elo_winning_prob(elo0, elo1)
    p0 = elo_winning_prob(elo1, elo0)

    if winner == 0:
        elo0 = elo0 + K * (1 - p0)
        elo1 = elo1 + K * (0 - p1)
    else:
        elo0 = elo0 + K * (0 - p0)
        elo1 = elo1 + K * (1 - p1)

    return elo0, elo1

def elo_rating2(elo0, elo1, n0, n1, K=30):
    def elo_winning_prob(rating1, rating2):
        return 1.0 / (1 + 1.0 * math.pow(10, (rating1 - rating2) / 400))
    p1 = elo_winning_prob(elo0, elo1)
    p0 = 1 - p1
    elo0 = elo0 + K * (n0*p1 - n1*p0)
    elo1 = elo1 + K * (n1*p0 - n0*p1)

    return elo0, elo1
