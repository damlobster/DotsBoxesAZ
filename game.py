class GameState():
    __slots__ = []

    @staticmethod
    def init_static_fields(*args, **kwargs):
        """This methods initialize the size the of the Dots and Boxes board
        
        Arguments:
            dims {tuple} -- the number of boxes for each dimensions: (horizontally, vertically). 
        """
        pass

    def get_actions_size(self):
        pass

    def get_valid_moves(self, as_indices=False):
        """Return the valid moves indices from the current position.
        
        Keyword Arguments:
            as_indices {bool} -- if true, a boolean array is returned in place of the indices (default: {False})
        
        Returns:
            The valid actions
        """
        pass

    def get_result(self):
        """Score the game from the perspective of the current player.
        
        Returns:
            [int] -- -1 for a loss, 0 for draw and 1 for a win
        """
        pass

    def play_(self, move):
        """Play the move, this method should modify the current instance.
        
        Arguments:
            move {int} -- the move to play
        
        Raises:
            ValueError -- in case of an invalid the move
        
        Returns:
            object -- any value useful for the game logic 
        """
        pass

    def play(self, move):
        """Play the move, this method should clone the game state first.
        
        Arguments:
            move {int} -- the move to play
        
        Returns:
            self
        """
        pass
        
    def get_features(self):
        """Build the features for the neural net evaluation.
        """
        pass

    def get_hash(self):
        """Get the hash of the board position. This method should return a unique value for each game state and should not recompute the hash.
        """
        pass

    def __hash__(self):
        return self.get_hash().__hash__()

    def __eq__(self, other):
        return self.get_hash() == other.get_hash()
