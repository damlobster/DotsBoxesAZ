class DefaultConfig():

  def __repr__(self):
    members = dir(self)
    strings = []
    strings.append(str(self.__class__.__name__))
    for m in members:
      if not m.startswith("__"):
        strings.append(m + " = " + str(self.__getattribute__(m)))
    return "\n".join(strings)

  MCTS_NUM_READ = 100

  SELFPLAY_NUM_GAME = 5000
  SELFPLAY_REUSE_MCTS_TREE = True
  SELFPLAY_DIRICHLET_ALPHA = 1.0
  SELFPLAY_DIRICHLET_COEFF = 0.25
