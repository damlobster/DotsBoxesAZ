from dots_boxes.dots_boxes_game import BoxesState
import sys
sys.path.append('..')
from project.default_config import DefaultConfig

class DotsBoxesConfig(DefaultConfig):
  GAME_NEW_STATE = lambda _: BoxesState((3,3))
  MCTS_NUM_READ = 500 
