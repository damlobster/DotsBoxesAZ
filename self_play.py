import numpy as np
import random
import mcts

class SelfPlay(object):

  def __init__(self, nn, config):
    self.nn = nn
    self.conf = config

  def play_game(self, game_state):
    alpha = self.conf.SELFPLAY_DIRICHLET_ALPHA
    coeff = self.conf.SELFPLAY_DIRICHLET_COEFF

    moves_sequence = []
    
    root_node = mcts.create_root_uct_node(game_state)
    while not root_node.is_terminal:
      tau = 1.0
      visit_counts = mcts.UCT_search(root_node, self.conf.MCTS_NUM_READ, self.nn)
      visit_counts = visit_counts ** (1/tau)

      probs = visit_counts / visit_counts.sum()
      valid_actions = root_node.game_state.get_valid_moves()
      valid_actions[valid_actions == 0] = 1e-60
      noise = np.random.dirichlet(valid_actions*alpha, 1).ravel()
      probs = (1-coeff)*probs + coeff*noise
      
      move = np.argmax(probs)

      next_node = root_node.children[move]
      root_node.children = None
      moves_sequence.append(root_node)
      root_node = next_node

    moves_sequence.append(root_node)

    result = root_node.game_state.get_result()
    features = []
    policies = []
    values = []
    moves = []
    for node in reversed(moves_sequence):
      if node.move is not None:
        moves.append(node.move)
      features.append(node.game_state.get_features())
      policies.append(node.child_number_visits / node.child_number_visits.sum())
      values.append(result)
      if node.game_state.player != node.game_state.next_player:
        result = result * -1

    return features, policies, values, list(reversed(moves))

def main():
  from dots_boxes import configuration
  from dots_boxes.dots_boxes_game import BoxesState
  conf = configuration.DotsBoxesConfig()
  sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), conf)
  game_state = conf.GAME_NEW_STATE()
  features, policies, values, moves = sp.play_game(game_state)

  dim = game_state.board.shape[1:]
  for i in range(1, len(moves)+1):
    print(BoxesState.moves_to_string(moves[:i], (dim[0]-1, dim[1]-1)))
  

if __name__ == '__main__':
  main()
