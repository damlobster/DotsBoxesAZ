import asyncio
import copy
import numpy as np
import os
import pickle
import torch
from torch.utils import data


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
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
            if file is not None and file == fn:
                with open(data_directory+fn, "rb") as f:
                    self.data.extend([sample for sample in pickle.load(f)])
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


class AsyncBatchedProxy():
    def __init__(self, func, batch_builder, batch_size):
        super(AsyncBatchedProxy, self).__init__()
        self.func = func
        self.batch_size = batch_size
        self.batch_builder = batch_builder
        self.queue = asyncio.Queue(maxsize=batch_size + batch_size//3)
        self.future_res = {}

    async def __call__(self, argument, callback):
        await self.queue.put((argument, callback))

    async def batch_runner(self):
        batch, cbs = [], []
        arg = True
        while arg is not None:
            arg, cb = await self.queue.get()
            if arg is not None:
                batch.append(arg)
                cbs.append(cb)
            if len(batch) >= self.batch_size or (not arg and batch):
                results = await self.func(self.batch_builder(batch))
                for cb, p, v in zip(cbs, *results):
                    cb((p, v))

                batch, cbs = [], []
