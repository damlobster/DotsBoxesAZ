import unittest
import asyncio
import numpy as np
import pandas as pd
from mcts import UCT_search, create_root_uct_node, print_mcts_tree
from .unittest_utils import async_test
import configuration
from nn import NeuralNetWrapper
from utils.proxies import AsyncBatchedProxy

NUM_SEARCHES = 800
MAX_SEARCHES = 800
MCTS_TREE_LEVELS = 3
DIRICHLET = (0.0, 0.0)
CPUCT = (1.25, 19652)
SAMPLE_IDS = None #[5,6,7]


class MCTSTestCase(unittest.TestCase):

    @async_test
    async def test_mcts_mock_nn(self, loop):
        print("test_mcts_mock_nn ...")

        async def nn(state):
            pi = np.array(list(map(int, state.get_valid_moves())))
            pi = pi/pi.sum()
            return (pi, state.get_result() or 0)

        await _run_tests(nn, NUM_SEARCHES, MAX_SEARCHES)

    @async_test
    async def test_mcts_nn(self, loop):
        nn_generation = 15
        params = configuration.params
        params.rewrite_str("_exp_", "resnet20_1229")
        params.self_play.pytorch_devices = "cpu"
        sp_params = params.self_play

        params.nn.model_parameters = {
            "resnet": {
                "pad_layer0": True,
                "in_channels": 3,
                "nb_channels": 128,
                "kernel_size": 3,
                "nb_blocks": 10,
                "n_groups": 4
            },
            "policy_head": {
                "in_channels": 128,
                "inner_channels": 16,
                "fc_in": 256,
                "nb_actions": 32
            },
            "value_head": {
                "in_channels": 128,
                "inner_channels": 1,
                "fc_in": 16,
                "fc_inner": 12
            }
        }

        print(params.nn.model_parameters)

        model = params.nn.model_class(params)
        model.load_parameters(
            nn_generation, to_device=params.self_play.pytorch_devices)

        nn_wrapper = NeuralNetWrapper(model, params)
        nnet = AsyncBatchedProxy(nn_wrapper, batch_size=sp_params.nn_batch_size,
                                 timeout=sp_params.nn_batch_timeout,
                                 batch_builder=sp_params.nn_batch_builder,
                                 cache_size=400000)

        fut = asyncio.ensure_future(nnet.run(), loop=loop)

        await _run_tests(nnet, num_searches=NUM_SEARCHES,
                         max_search=MAX_SEARCHES)

        fut.cancel()
        await fut


async def _run_tests(nn, num_searches=800, max_search=int(1e5)):
    from dots_boxes.dots_boxes_game import BoxesState
    from test.nn_unittests import load_boards_samples

    df = load_boards_samples()

    success = True
    for i, sample in df.iterrows():
        if SAMPLE_IDS is not None and i not in SAMPLE_IDS:
            continue

        print("Sample id =", i)
        print(sample.game)
        root_node = create_root_uct_node(sample.game)

        print(await nn(sample.game))

        searches = 0
        ok = False
        while searches < max_search:
            policy = await UCT_search(root_node, num_searches, nn, max_pending_evals=1, dirichlet=DIRICHLET, cpuct=CPUCT)
            searches += num_searches
            if np.argsort(policy)[-1] in sample.next_moves:
                print("------ OK ------")
                ok = True
                break

        success = success and ok
        if not ok:
            print("------ FAILED ------")

        print(f"Nb searches= {searches}")
        print(f"Expected moves: {sample.next_moves}")
        print("\n")
        print("priors=", get_maxs(root_node.child_priors))
        print("policy=", get_maxs(policy))
        print("ucb=", get_maxs(root_node.children_ucb_score()))
        print("values=", get_maxs(root_node.child_total_value))

        print("stats=", root_node.parent.get_tree_stats())
        print("\n")

        print_mcts_tree(root_node, MCTS_TREE_LEVELS)

    assert success, "Some mcts searches doesn't returned the expected move!"


def get_maxs(arr):
    _min = max(arr) == 0
    s = []
    for idx in np.argsort(arr)[::1 if _min else -1][:3]:
        s.append(f"{idx}->{arr[idx]:.4f}")
    return "; ".join(s)
