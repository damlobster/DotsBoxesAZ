import asyncio
from async_timeout import timeout
import multiprocessing as mp
import numpy as np
import os
import sys
import time
from functools import partial


def default_batch_builder(states_batch):
    return np.concatenate(tuple(gs[0].get_features() for gs in states_batch))[np.newaxis, :]

class AsyncBatchedProxy():
    def __init__(self, func, batch_size, timeout=None, batch_builder=None,
                 max_queue_size=None, loop=asyncio.get_event_loop()):
        super(AsyncBatchedProxy, self).__init__()
        self.func = func
        self.cache = {}
        self.batch_size = batch_size
        self.timeout = timeout
        self.time_first_in = None
        self.batch_builder = batch_builder
        self.loop = loop
        self.queue = asyncio.Queue(
            maxsize=max_queue_size if max_queue_size else 2*batch_size)

    async def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]

        fut = asyncio.Future()
        await self.queue.put((time.time(), args, fut))
        res = await fut
        self.cache[args] = res
        return res

    async def run(self):
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
                print("run batch:", n, len(ts), flush=True)
                results = await self.func(self.batch_builder(*args[:n]))
                for fut, res in zip(futs[:n], zip(*results)):
                    fut.set_result(res)

                ts, args, futs = ts[n:], args[n:], futs[n:]


def pipe_proxy_init():
    child, parent = mp.connection.Pipe()
    pipe_proxy.child_pipe = child
    pipe_proxy.parent_pipe = parent

def _pipe_proxy_get_parent(ignored):
    return pipe_proxy.parent_pipe

async def pipe_proxy(*args):
    pipe = pipe_proxy.child_pipe
    if pipe.closed:
        raise ConnectionError("Pipe already closed for pid: %d", pid)
    pipe.send(args)
    await asyncio.get_event_loop().run_in_executor(None, pipe.poll, None)
    return pipe.recv()

class PipedProxy():
    def __init__(self, func, loop, mp_pool):
        self.func = func
        self.my_pipes = set(mp_pool.map(_pipe_proxy_get_parent, range(mp_pool._processes)))
        self.loop = loop

    async def run(self):
        def send(pipe, fut):
            pipe.send(fut.result())
        try:
            while self.my_pipes:
                ready_list = await self.loop.run_in_executor(None, mp.connection.wait, self.my_pipes)
                for pipe in ready_list:
                    try:
                        args = pipe.recv()
                        fut = asyncio.ensure_future(self.func(*args), loop=self.loop)
                        fut.add_done_callback(partial(send, pipe))
                    except EOFError as e:
                        self.my_pipes.remove(pipe)
        except Exception:
            print("Exception!", flush=True)
            raise
