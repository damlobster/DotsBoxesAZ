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


class PickleDataset(data.Dataset):
  def __init__(self, data_directory, size_limit=int(1e7)):
    self.data = []
    files = sorted(os.listdir(data_directory), reverse=True)
    for fn in files:
      with open(data_directory+fn, "rb") as f:
        self.data.extend([sample for chunk in pickle.load(f) for sample in chunk])
      if len(self.data)>size_limit:
        self.data = self.data[:size_limit]
        break

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    board, visits, value = self.data[index]

    # Load data and get label
    return board, visits, np.asarray([value], dtype=np.float32)
