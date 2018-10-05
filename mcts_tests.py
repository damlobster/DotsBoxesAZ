from mcts import UCT_search, create_root_uct_node
import unittest
import numpy as np


class MCTSTestCase(unittest.TestCase):

    def test_benchmark(self):
        from dots_boxes.dots_boxes_game import BoxesState
        import time
        import resource
        num_reads = 1000
        root_node = create_root_uct_node(BoxesState())
        tick = time.time()
        policy = UCT_search(root_node, num_reads,
                            lambda state: (np.ones(state.get_actions_size()), 0))
        print(policy.astype(int))
        tock = time.time()
        print("Took %s sec to run %s times" % (tock - tick, num_reads))
        print("Consumed %.2fMB memory" %
              (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))
