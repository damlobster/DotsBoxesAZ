import os
import sys
sys.path.append("..")

import asyncio
import numpy as np
import pandas as pd
from dots_boxes.dots_boxes_game import BoxesState
from mcts import UCT_search, create_root_uct_node, print_mcts_tree
import configuration
from nn import NeuralNetWrapper
from utils.proxies import AsyncBatchedProxy
from make_tikz_board import game_to_tikz

PARAMS = configuration.simple
RUN = "simple_0105_2"
FOLDER = "simple"

SAVE_TIKZ_GEN = [1, 20, 60]
NUM_SEARCHES = 800
MAX_SEARCHES = 800
DIRICHLET = (0.0, 0.0)
CPUCT = (1.25, 19652)

params = PARAMS
params.rewrite_str("data/", f"../data/")
params.rewrite_str("_exp_", RUN)
params.self_play.pytorch_devices = "cuda:0"


SAMPLES = pd.read_csv("../test/test_boards.csv", comment="#", sep=";", index_col="id")
SAMPLES = SAMPLES.applymap(lambda s: list(map(int, s.split(" "))) if isinstance(s, str) else s)
games = []
for idx, sample in SAMPLES.iterrows():
    g = BoxesState()
    for m in sample.moves:
        g.play_(int(m))
    games.append(g)
SAMPLES["game"] = games


async def test_mcts_nn(loop, generation):
    sp_params = params.self_play

    model = params.nn.model_class(params)
    model.load_parameters(generation, to_device=params.self_play.pytorch_devices)

    nn_wrapper = NeuralNetWrapper(model, params)
    nnet = AsyncBatchedProxy(nn_wrapper, batch_size=48,
                                timeout=sp_params.nn_batch_timeout,
                                batch_builder=sp_params.nn_batch_builder,
                                cache_size=400000)

    fut = asyncio.ensure_future(nnet.run(), loop=loop)

    nn_corrects = 0
    mcts_corrects = 0
    for i, sample in SAMPLES.iterrows():
        print(f"Generate tikz for sample {i}")
        p, v = await nnet(sample.game)
        nn_corrects += p.argmax() in sample.next_moves
        if generation in SAVE_TIKZ_GEN and i > 0:
            tikz = game_to_tikz(sample.moves, sample.next_moves, p)
            with open(f"positions_samples/{FOLDER}/gen{generation}_nn_{i}.tikz", "w") as f:
                f.write(tikz)

        root_node = create_root_uct_node(sample.game)
        p = await UCT_search(root_node, NUM_SEARCHES, nnet, max_pending_evals=64, dirichlet=DIRICHLET, cpuct=CPUCT)
        p = p / (p.sum() or 1.0)
        mcts_corrects += p.argmax() in sample.next_moves
        if generation in SAVE_TIKZ_GEN and i > 0:
            tikz = game_to_tikz(sample.moves, sample.next_moves, p)
            with open(f"positions_samples/{FOLDER}/gen{generation}_mcts_{i}.tikz", "w") as f:
                f.write(tikz)

    with open(f"positions_samples/{FOLDER}/accuracy.txt", "a") as f:
        f.write(f"{generation},{nn_corrects/SAMPLES.shape[0]},{mcts_corrects/SAMPLES.shape[0]}\n")

    fut.cancel()
    await fut

TABLE_TMPLT = """
\\begin{center}
    \\begin{tabular}{ |c|c|c|c|c|c|c| }
        \\hline
     
        & \\multicolumn{3}{|c|}{NN} & \\multicolumn{3}{|c|}{MCTS} \\\\
        \\hline
     
        Generation & 1 & 20 & 60 & 1 & 20 & 60  \\\\
        \\hline

%s

        \\multicolumn{7}{|p{\\columnwidth}|}{
            The bold black edge shows the last move played. The colored edge(s) the best moves to play next. The numbers on the edges show the policies returned by the network or the MCTS search. 
        } \\\\
        \\hline
    \\end{tabular}
\\end{center}
"""

ROW_TMPLT = """
        Sample _i_ & \\input{figures/unittests/simple/gen1_nn__n_.tikz} & \\input{figures/unittests/simple/gen20_nn__n_.tikz} & \\input{figures/unittests/simple/gen60_nn__n_.tikz} & 
            \\input{figures/unittests/simple/gen1_mcts__n_.tikz} & \\input{figures/unittests/simple/gen20_mcts__n_.tikz} & \\input{figures/unittests/simple/gen60_mcts__n_.tikz} \\\\ 
        \\hline
"""

def create_table():
    rows = []
    for idx in SAMPLES.index.values:
        if idx > 0:
            rows.append(ROW_TMPLT.replace("_i_", str(idx)).replace("_n_", str(idx)))

    with open(f"positions_samples/{FOLDER}_table.tex", "w") as f:
        f.write(TABLE_TMPLT % ("\n".join(rows),))

if not os.path.exists(f"positions_samples/{FOLDER}/"):
    os.makedirs(f"positions_samples/{FOLDER}/")

with open(f"positions_samples/{FOLDER}/accuracy.txt", "w") as f:
    f.write(f"generation,nn_ratio_correct,mcts_ratio_corrects\n")

loop = asyncio.new_event_loop()

for gen in range(61):
    print("Process generation:", gen)
    loop.run_until_complete(test_mcts_nn(loop, gen))

create_table()