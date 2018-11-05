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


"""
def pipe_proxy_init():
    child, parent = mp.connection.Pipe()
    pipe_proxy.child_pipe = child
    pipe_proxy.parent_pipe = parent
    asyncio.set_event_loop(asyncio.new_event_loop())

def _pipe_proxy_get_parent(ignored):
    import time
    time.sleep(0.01)
    return pipe_proxy.parent_pipe

async def pipe_proxy(*args):
    print("ppc 1", flush=True) #!!!!
    pipe = pipe_proxy.child_pipe
    if pipe.closed:
        raise ConnectionError("Pipe already closed for pid: %d", pid)
    pipe.send(args)
    print("ppc 2", flush=True) #!!!!
    await asyncio.get_event_loop().run_in_executor(None, pipe.poll, None)
    print("ppc 3", flush=True) #!!!!
    return pipe.recv()

class PipedProxy():
    def __init__(self, func, loop, mp_pool):
        self.func = func
        self.my_pipes = set(mp_pool.map(_pipe_proxy_get_parent, range(mp_pool._processes)))
        print(self.my_pipes, flush=True)
        self.loop = loop

    async def run(self):
        def send(pipe, fut):
            print("ppr send", flush=True) #!!!!    
            pipe.send(fut.result())
        print("ppr 1", flush=True) #!!!!
        try:
            while self.my_pipes:
                ready_list = await self.loop.run_in_executor(None, mp.connection.wait, self.my_pipes)
                print("ppr 2", flush=True) #!!!!
                for pipe in ready_list:
                    try:
                        print("ppr 3", flush=True) #!!!!
                        args = pipe.recv()
                        fut = asyncio.ensure_future(self.func(*args), loop=self.loop)
                        print("ppr 4", flush=True) #!!!!
                        fut.add_done_callback(partial(send, pipe))
                    except EOFError as e:
                        self.my_pipes.remove(pipe)
        except asyncio.CancelledError:
            return #silently exit the worker
"""