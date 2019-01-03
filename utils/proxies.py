import logging
logger = logging.getLogger(__name__)

import asyncio
from async_timeout import timeout
import multiprocessing as mp
import numpy as np
import os
import sys
import time
from functools import partial
import pylru


def default_batch_builder(states_batch):
    return np.concatenate(tuple(gs[0].get_features() for gs in states_batch))[np.newaxis, :]

class AsyncBatchedProxy():
    def __init__(self, func, batch_size, timeout=None, batch_builder=None, max_queue_size=None,
                 cache_size=400000, cache_hash=lambda args: args[0].get_hash()):
        super(AsyncBatchedProxy, self).__init__()
        self.func = func
        self.with_cache = cache_size > 0
        if self.with_cache:
            self.cache = pylru.lrucache(cache_size)
            self.cache.cache_hash = cache_hash
        self.batch_size = batch_size
        self.timeout = timeout
        self.time_first_in = None
        self.batch_builder = batch_builder
        self.queue = asyncio.Queue(
            maxsize=max_queue_size if max_queue_size else 2*batch_size)

    async def __call__(self, *args):
        if self.with_cache:
            arg_hash = self.cache.cache_hash(args)
            if arg_hash in self.cache:
                return self.cache[arg_hash]
        fut = asyncio.Future()
        await self.queue.put((time.time(), args, fut))
        res = await fut
        if self.with_cache:
            self.cache[arg_hash] = res
        return res

    async def run(self):
        try:
            ts, args, futs = [], [], []
            while True:
                try:
                    async with timeout(self.timeout) as t_out:
                        t, a, f = await self.queue.get()
                    ts.append(t)
                    args.append(a)
                    futs.append(f)
                except asyncio.TimeoutError as e:
                    pass

                call_func = time.time() - ts[0] > self.timeout if len(futs)>0 else False
                if call_func:
                    n = min(len(ts), self.batch_size)
                    logger.debug("batch size=%d", n)
                    results = await self.func(self.batch_builder(*args[:n]))
                    ps, vs = results
                    for i, fut in enumerate(futs[:n]):
                        try:
                            fut.set_result((ps[i], vs[i]))
                        except:
                            print("Exception!")
                            raise

                    ts, args, futs = ts[n:], args[n:], futs[n:]

        except asyncio.CancelledError:
            return # silently exit the worker
