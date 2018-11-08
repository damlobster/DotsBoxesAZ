import logging
logging.basicConfig(level=logging.INFO)
import math 
from functools import partial
import asyncio

from mcts import create_root_uct_node, UCT_search
import torch.multiprocessing as mp
from utils import utils
from utils.proxies import AsyncBatchedProxy

import configuration as cfg


class AZPlayer(mp.Process):
    def __init__(self, params, time_limit, requ_queue: mp.Queue, resp_queue: mp.Queue):
        super(AZPlayer, self).__init__()
        self.params = params
        self.models = {}
        self.time_limit = self.time_limit
        self.requ_queue = requ_queue
        self.resp_queue = resp_queue

    def _load_model(self, generation):
        if generation in self.models:
            return self.models[generation]
        
        model = self.params.nn.model_class(self.params)
        model.load_parameters(generation)
        self.models[generation] = model
        return model

    def run(self):
        from mcts import UCT_search, create_root_uct_node
        from nn import NeuralNetWrapper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        nn_wrapper = NeuralNetWrapper(None, self.params)

        nn = AsyncBatchedProxy(nn_wrapper, batch_size=params.nn_batch_size,
                                    timeout=params.nn_batch_timeout,
                                    batch_builder=params.nn_batch_builder)
        nn_task = asyncio.ensure_future(nn.run(), loop=loop)

        mcts_cfg = self.params.self_play.mcts
        while True:
            req = self.requ_queue.get()
            if req is None:
                # we received a poison pill
                break

            game_uuid, generation, game_state = req
            nn_wrapper.set_model(self._load_model(generation))

            node = create_root_uct_node(game_state)
            fut = asyncio.ensure_future(UCT_search(node, None, nn, mcts_cfg.mcts_cpuct, 
                                mcts_cfg.max_async_searches, (0.0, 0.0), self.time_limit))
            loop.run_until_complete(fut)
            
            policy = fut.result()
            if policy.sum() > 0:
                # we greedly return the best action
                self.resp_queue.put((game_uuid, policy.argmax()))
            else:
                self.resp_queue.put((game_uuid, None))

        nn_task.cancel()
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()