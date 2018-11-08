class GameState():
    __slots__ = []

    @staticmethod
    def init_static_fields(*args, **kwargs):
        pass

    def get_actions_size(self):
        pass

    def get_valid_moves(self, as_indices=False):
        pass

    def get_result(self):
        pass

    def play_(self, move):
        pass

    def play(self, move):
        pass

    def move_human_to_index(human):
        pass

    def move_index_to_human(index):
        pass
        
    def get_features(self):
        pass

    def get_hash(self):
        pass
        
    def __hash__(self):
        pass
