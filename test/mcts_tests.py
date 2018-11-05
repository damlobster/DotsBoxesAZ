import unittest
import asyncio
import numpy as np
from mcts import UCT_search, create_root_uct_node, print_mcts_tree
from .unittest_utils import async_test


class MCTSTestCase(unittest.TestCase):

    def test_benchmark(self):
        print()
        from dots_boxes.dots_boxes_game import BoxesState
        import time
        import resource

        async def nn(state):
            await asyncio.sleep(0.1)
            return (np.random.uniform(0, 1, state.get_actions_size()), 0)

        num_reads = 20
        root_node = create_root_uct_node(BoxesState())
        tick = time.time()

        loop = asyncio.get_event_loop()
        policy = loop.run_until_complete(asyncio.ensure_future(UCT_search(root_node, num_reads, nn, max_pending_evals=10)))
        loop.close()
        
        print(policy.astype(int))
        tock = time.time()
        print("Took %s sec to run %s times" % (tock - tick, num_reads))
        print("Consumed %.2fMB memory" %
              (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))

        print("MCTS tree *********")
        print_mcts_tree(root_node)

        self.assertEqual(sum(policy), num_reads)
