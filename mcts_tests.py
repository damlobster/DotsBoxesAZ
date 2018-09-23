from mcts import UCT_search
import unittest
import numpy as np

class MCTSTestCase(unittest.TestCase):

  def test_benchmark(self):
    from dots_boxes.dots_boxes_game import BoxesState
    import time
    import resource
    num_reads = 11
    tick = time.time()
    policy = UCT_search(BoxesState((1, 1)), num_reads,
                      lambda state: (np.ones(state.get_actions_size()), 0))
    print(policy.astype(int))
    tock = time.time()
    print("Took %s sec to run %s times" % (tock - tick, num_reads))
    print("Consumed %.2fMB memory" %
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))
