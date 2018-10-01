import numpy as np
import random
import utils
import mcts

params = utils.DotDict({
    "num_games": 1000,
    "reuse_mcts_tree": True,
    "noise": (1.0, 0.25),  # alpha, coeff
    "mcts_num_read": 500,
    "temperature": 1.0 #1e-50,
})

class SelfPlay(object):

  def __init__(self, nn, params):
    self.nn = nn
    self.played_games = []
    self.params = params

  def play_game(self, game_state):
    params = self.params
    alpha, coeff = params.noise
    
    moves_sequence = []
    root_node = mcts.create_root_uct_node(game_state)
    while not root_node.is_terminal:
      visit_counts = mcts.UCT_search(root_node, params.mcts_num_read, self.nn)
      probs = (visit_counts/visit_counts.max()) ** (1/params.temperature)
      
      probs = probs / probs.sum()
      valid_actions = root_node.game_state.get_valid_moves()
      valid_actions[valid_actions == 0] = 1e-60
      noise = np.random.dirichlet(valid_actions*alpha, 1).ravel()
      probs = (1-coeff)*probs + coeff*noise

      move = np.argmax(np.random.multinomial(1, probs / probs.sum(), 1))

      moves_sequence.append(root_node)
      next_node = None
      if params.reuse_mcts_tree:
        next_node = root_node.children[move]
        root_node.children = None
      else:
        next_node = mcts.create_root_uct_node(root_node.children[move].game_state)
      root_node = next_node

    moves_sequence.append(root_node)
    self.played_games.append((moves_sequence, root_node.game_state.get_result())) 

  def play_games(self, game_state, n_iters, show_progress=False):
    for _ in range(n_iters):
      if show_progress:
        print(".", end="", flush=True)
      self.play_game(game_state)

  def get_training_data(self):
    features = []
    policies = []
    values = []
    for moves_seq, result in self.played_games:
      for node in reversed(moves_seq[:-1]):
        features.append(node.game_state.get_features())
        policies.append(node.child_number_visits / node.child_number_visits.sum())
        values.append(result)
        if node.game_state.player != node.game_state.next_player:
          result = result * -1

    return list(zip(features, policies, values))

  def get_games_moves(self):
    moves = []
    visit_counts = []
    for moves_seq, _ in self.played_games:
      for node in moves_seq[1:]:
        moves.append(node.move)
        visit_counts.append(node.child_number_visits)
    vc = np.asarray(visit_counts, dtype=float)
    return moves, vc


def test():
  from dots_boxes.dots_boxes_game import BoxesState, moves_to_string
  sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), params)
  game_state = BoxesState((3,3))
  sp.play_game(game_state)
  moves, visit_counts = sp.get_games_moves()

  for i in range(1, len(moves)+1):
    print(moves_to_string(moves[:i], visit_counts[i-1]))

def worker(n_games, j):
  print(str(j))
  from dots_boxes.dots_boxes_game import BoxesState
  game_state = BoxesState((3,3))
  sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), params)
  sp.play_games(game_state, n_games, show_progress=True)
  return sp.get_training_data()

def generate_games():
  import pickle
  import multiprocessing as mp
  with mp.Pool(mp.cpu_count()) as pool:
    for i in range(1):
      results = [pool.apply_async(worker, args=(100,j)) for j in range(20)]
      data = [p.get() for p in results]
      with open("./data/selfplay{}.pkl".format(i), "wb") as f:
        pickle.dump(list(data), f)


if __name__ == '__main__':
  #test()
  generate_games()