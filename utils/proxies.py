import asyncio
from async_timeout import timeout
import multiprocessing as mp
import os
import time


def default_batch_builder(states_batch):
    return np.concatenate(tuple(gs.get_features() for gs in states_batch))[np.newaxis, :]

class AsyncBatchedProxy():
    def __init__(self, func, batch_size, timeout=None, batch_builder=default_batch_builder,
                 max_queue_size=None, loop=asyncio.get_event_loop()):
        super(AsyncBatchedProxy, self).__init__()
        self.func = func
        self.batch_size = batch_size
        self.timeout = timeout
        self.time_first_in = None
        self.batch_builder = batch_builder
        self.loop = loop
        self.queue = asyncio.Queue(
            maxsize=max_queue_size if max_queue_size else batch_size + 2)

    async def __call__(self, argument):
        fut = asyncio.Future()
        await self.queue.put((time.time(), argument, fut))
        res = await fut
        return res

    async def run(self):
        ts, args, futs = [], [], []
        while True:
            timeout_occured = False
            try:
                async with timeout(self.timeout) as t_out:
                    t, a, f = await self.queue.get()
                    ts.append(t)
                    args.append(a)
                    futs.append(f)
            except asyncio.TimeoutError as e:
                timeout_occured = True
            
            call_func = time.time() - ts[0] > self.timeout if ts else False
            if call_func:
                n = min(len(args), self.batch_size)
                results = await self.func(self.batch_builder(args[:n]))
                for fut, res in zip(futs[:n], results):
                    fut.set_result(res)

                ts, args, futs = ts[:n], args[:n], futs[:n]


_PipedProxyChild_child_pipe = None
_PipedProxyChild_parent_pipe = None
def pipe_proxy_init():
    global _PipedProxyChild_child_pipe
    global _PipedProxyChild_parent_pipe
    _PipedProxyChild_child_pipe, _PipedProxyChild_parent_pipe = mp.connection.Pipe()

def _pipe_proxy_get_parent(ignored):
    global _PipedProxyChild_parent_pipe
    pipe = _PipedProxyChild_parent_pipe
    _PipedProxyChild_parent_pipe = None
    return pipe

async def pipe_proxy(*args):
    pipe = _PipedProxyChild_child_pipe
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
        try:
            while self.my_pipes:
                ready_list = await self.loop.run_in_executor(None, mp.connection.wait, self.my_pipes)
                for pipe in ready_list:
                    try:
                        args = pipe.recv()
                        res = await self.func(*args)
                        pipe.send(res)
                    except EOFError as e:
                        self.my_pipes.remove(pipe)
        except Exception:
            raise
