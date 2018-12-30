import pandas as pd
from dots_boxes.dots_boxes_game import BoxesState

def load_boards_samples():
    df = pd.read_csv("test/test_boards.csv", comment="#", sep=";", index_col="id")
    df = df.applymap(lambda s: list(map(int, s.split(","))) if isinstance(s, str) else s)

    games = []
    for idx, sample in df.iterrows():
        g = BoxesState()
        for m in sample.moves:
            g.play_(int(m))

        games.append(g)

    df["game"] = games
    return df