import unittest
import asyncio
import numpy as np
import pandas as pd
from mcts import UCT_search, create_root_uct_node, print_mcts_tree
from .unittest_utils import async_test


class MCTSTestCase(unittest.TestCase):

    # def test_benchmark(self):
    #     print()
    #     from dots_boxes.dots_boxes_game import BoxesState
    #     import time
    #     import resource

    #     async def nn(state):
    #         await asyncio.sleep(0.1)
    #         return (np.random.uniform(0, 1, state.get_actions_size()), 0)

    #     num_reads = 20
    #     root_node = create_root_uct_node(BoxesState())
    #     tick = time.time()

    #     loop = asyncio.get_event_loop()
    #     policy = loop.run_until_complete(asyncio.ensure_future(UCT_search(root_node, num_reads, nn, max_pending_evals=10)))
    #     loop.close()
        
    #     print(policy.astype(int))
    #     tock = time.time()
    #     print("Took %s sec to run %s times" % (tock - tick, num_reads))
    #     print("Consumed %.2fMB memory" %
    #           (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))

    #     print("MCTS tree *********")
    #     print_mcts_tree(root_node)

    #     self.assertEqual(sum(policy), num_reads)

    def test_mcts(self):
        print()
        from dots_boxes.dots_boxes_game import BoxesState
        from test.nn_unittests import load_boards_samples

        async def nn(state):
            pi = np.array(list(map(int, state.get_valid_moves())))
            pi = pi/pi.sum()
            return (pi, state.get_result() or 0)

        max_search = 100000

        df = load_boards_samples()

        loop = asyncio.get_event_loop()

        for i, sample in df.iterrows():
            # if i != 0:
            #     continue
            print(sample.game)
            root_node = create_root_uct_node(sample.game)

            searches = 0
            while searches < max_search:
                policy = loop.run_until_complete(asyncio.ensure_future(UCT_search(root_node, 10000, nn, max_pending_evals=1)))
                searches += 10000
                if np.argsort(policy)[-1] in sample.next_moves:
                    print("------ OK ------")
                    break
            print(f"Nb searches= {searches}")
            print(f"Expected moves: {sample.next_moves}")
            for idx in np.argsort(policy)[::-1][:3]:
                print(f"{idx}->{policy[idx]}", end="; ")
            

            print("\n")
            # print_mcts_tree(root_node)
            print("ucb=", root_node.children_ucb_score().round(3))
            print("values=", root_node.child_total_value)

            print("stats=", root_node.parent.get_tree_stats())
            print("\n")

        loop.close()