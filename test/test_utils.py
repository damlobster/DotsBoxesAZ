import asyncio
import unittest
from functools import partial

import torch

from utils.utils import SymetriesGenerator
from utils.proxies import AsyncBatchedProxy

class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    # def test_async_batcher(self):
    #     array = list(range(10))

    #     async def nn(batch):
    #         await asyncio.sleep(0.1)
    #         #print("batched call: ", batch)
    #         return map(lambda v: v**2, batch)

    #     async def callback(i, v):
    #         #print("Callback called with value={}".format(v))
    #         array[i] = v

    #     batched_nn = AsyncBatchedProxy(nn, lambda vs: vs, 3)

    #     async def producer():
    #         for i in array:
    #             await batched_nn(i, partial(callback, i))
    #         await batched_nn(None, callback)

    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(asyncio.gather(batched_nn.batch_runner(), producer()))
    #     loop.close()

    #     assert array == list(map(lambda i: i**2, range(10)))


    def test_symetries_generator(self):
        t = torch.arange(0,1*3*3*3, dtype=torch.float32).reshape(1, 3, 3, 3)
        sg = SymetriesGenerator()
        f, p, v = sg(t, torch.arange(0,-1*2*3*3, -1, dtype=torch.float32).reshape(1,18), torch.arange(0,1))
        print(f)
        print(p)
        print(v)
