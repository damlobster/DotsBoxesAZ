import pylru
from functools import partial
import time
import sys
import os
import numpy as np
import multiprocessing as mp
from async_timeout import timeout
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


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

                call_func = time.time() - \
                    ts[0] > self.timeout if len(futs) > 0 else False
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
            return  # silently exit the worker


class PipeProxyClient:
    def __init__(self, id, conn):
        super().__init__()
        self.id = id
        self.conn = conn
        self.pending = {}
        self.counter = 0
        self.loop = None

    def set_asyncio_loop(self, loop):
        self.loop = loop

    async def __call__(self, *args):
        self.counter = (self.counter + 1) % 5196
        reqn = self.counter
        self.conn.send(self.id, reqn, args)
        fut = asyncio.Future(loop=self.loop)
        self.pending[reqn] = fut
        return await fut

    async def run(self):
        try:
            while True:
                reqn, result = await self.loop.run_in_executor(None, self.conn.recv())
                self.pending[reqn].set_result(result)
                del self.pending[reqn]
        except Exception as e:
            return  # silently exit the worker


class PipeProxyServer:
    def __init__(self, func, batch_size=64, timeout=0.1, cache_size=int(1e6), cache_hash=lambda args: args[0].get_hash()):
        super().__init__()
        self.func = func
        self.id_gen = 0
        self.connections = {}
        self.buffer = []
        self.batch_size = batch_size
        self.timeout = timeout
        self.with_cache = cache_size > 0
        if self.with_cache:
            self.cache = pylru.lrucache(cache_size)
            self.cache.cache_hash = cache_hash

        self.last_req_t = time.time()

    def create_client(self):
        self.id_gen += 1
        id = self.id_gen
        me, you = mp.Pipe(True)
        self.connections[id] = me
        return PipeProxyClient(id, you)

    def handle_requests(self):
        ready = mp.connection.wait(self.connections.values(), 0)
        if len(ready) > 0:
            for pipe in ready:
                try:

                    client_id, seq, args = pipe.recv()
                    if self.with_cache:
                        arg_hash = self.cache.cache_hash(args)
                        if arg_hash in self.cache:
                            self.connections[id].send(
                                seq, self.cache[arg_hash])
                    else:
                        self.buffer.append((client_id, seq, args))

                except EOFError as e:
                    raise e

            buffer_full = len(self.buffer) >= self.batch_size
            if buffer_full or (len(self.buffer) > 0 and self.last_req_t + self.timeout <= time.time()):
                self.last_req_t = time.time()
                client_ids, seqs, args = zip(self.buffer)
                self.buffer = []

                results = self.func(args)

                for i, id in enumerate(client_ids):
                    self.connections[id].send(seqs[i], results[i])
