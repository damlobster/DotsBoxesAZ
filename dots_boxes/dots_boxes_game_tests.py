import unittest
import random
import copy
from dots_boxes.dots_boxes_game import BoxesState, moves_to_string


class BoxesStateTest(unittest.TestCase):
    def setUp(self):
        BoxesState.init_static_fields(dims=(3, 3))

    def test_hash(self):
        state = BoxesState()
        state1 = state.play(0)

        self.assertEqual(state, copy.deepcopy(state))
        self.assertNotEqual(state, state1)

    def test_repr(self):
        state = BoxesState()
        for m in state.get_valid_moves(as_indices=True)[:10]:
            state.play_(m)

        for m in state.get_valid_moves(as_indices=True)[-10:]:
            state.play_(m)

        result = """------------------------------
Player = 0
Next player = 1
Boxes to close = [1.5, 2.5]
Result = None
+---+---+---+
        |   |   
+---+---+---+
|   |   |   |   
+---+---+---+
|   |   |   |   
+---+   +   +
"""
        self.assertEqual(str(state), result)

    def test_moves_to_string(self):
        g = BoxesState()
        moves = g.get_valid_moves(True)
        s = moves_to_string(moves[:10] + moves[-10:])

        result = """------------------------------
Player = 0
Next player = 1
Boxes to close = [1.5, 2.5]
Result = None
+---+---+---+
        | 1 |   
+---+---+---+
| 0 | 0 | 0 |   
+---+---+---+
| 1 |   |   |   
+---+   +   +
"""
        self.assertEqual(s, result)
