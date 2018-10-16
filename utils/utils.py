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


class HDFStoreDataset(data.Dataset):
    def __init__(self, hdf_file, features_shape=None, where=None):
        def reshape(X):            
            if features_shape is None:
                return X
            else:
                from operator import mul
                from functools import reduce
                assert reduce(mul, features_shape) == X.shape[1], "Cannot reshape features to: {}".format(features_shape)
                return X.reshape((-1, *features_shape))

        super(HDFStoreDataset, self).__init__()
        with pd.HDFStore(hdf_file, "r") as hdf_store:
            self.features = reshape(hdf_store.select("data/features", where).values.astype(np.float32))
            self.policy = hdf_store.select("data/policy", where).values
            self.value = hdf_store.select("data/value", where).values.astype(np.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        board = self.features[index]
        policy = self.policy[index]
        value = self.value[index]
        return board, visits, np.asarray([value])


def default_batch_builder(states_batch):
    return np.concatenate(tuple(gs.get_features() for gs in states_batch))

"""class AsyncBatchedProxy():
    def __init__(self, func, batch_size, batch_builder=default_batch_builder,
                 max_queue_size=None, loop=asyncio.get_event_loop()):
        super(AsyncBatchedProxy, self).__init__()
        self.func = func
        self.batch_size = batch_size
        self.batch_builder = batch_builder
        self.loop = loop
        self.queue = asyncio.Queue(maxsize = max_queue_size if max_queue_size else batch_size + 2)
        self.not_launched = True

    async def __call__(self, argument, no_batch=False):
        if no_batch:
            return await self.func(self.batch_builder(argument)[np.newaxis,:])

        if self.not_launched:
            self.loop.create_task(self.batch_runner())

        fut = asyncio.Future()
        await self.queue.put((argument, fut))
        res = await fut
        return res

    async def batch_runner(self):
        batch, futs = [], []
        arg = True
        while arg is not None:
            arg, fut = await self.queue.get()
            if arg is not None:
                batch.append(arg)
                futs.append(fut)
            if len(batch) >= self.batch_size or (not arg and batch):
                results = await self.func(self.batch_builder(batch))
                for fut, p, v in zip(futs, *results):
                    fut.set_result((p, v))

                batch, futs = [], []


class PipedProxy():
    class CallerProxy():
        def __init__(self, pipe):
            self.pipe = pipe

        async def __call__(self, *args, **kwargs):
            if self.pipe.closed:
                raise ConnectionError("Pipe already closed!")
            call = functools.partial(child.send, (args, kwargs))
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, call)
            return await loop.run_in_executor(None, child.recv)
            

    def __init__(self, func, batch_size=8):
        self.func = func
        self.batch_size = batch_size
        self.parent, self.child = mp.connection.Pipe()
        self.is_running = False

    def get_caller_proxy(self):
        async def caller_proxy(*args, **kwargs):



def create_pipe_proxy(func):
    parent, child = Pipe()

    async def caller(*args, **kwargs):
        if child.closed:
            raise ConnectionError("Pipe already closed!")
        call = functools.partial(child.send, (args, kwargs))
        await asyncio.get_event_loop().run_in_executor(None, call)
        return await asyncio.get_event_loop().run_in_executor(None, child.recv)
    caller.close = lambda: child.close()

    async def callee():
        callee._closed = False
        while not callee._closed:
            args, kwargs = await asyncio.get_event_loop().run_in_executor(None, parent.recv)
            res = await func(*args, **kwargs)
            await asyncio.get_event_loop().run_in_executor(None, parent.send, res)
        parent.close()

    def c():
        callee._closed = True
    callee.close = c

    return callee, caller
"""
