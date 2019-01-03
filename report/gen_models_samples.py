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

SAVE_TIKZ_GEN = [1, 20, 60]
RUN = "resnet20_1230"
NUM_SEARCHES = 800
MAX_SEARCHES = 800
DIRICHLET = (0.0, 0.0)
CPUCT = (1.25, 19652)

params = configuration.resnet
params.rewrite_str("data/", "../data/")
params.self_play.pytorch_devices = "cpu"

async def test_mcts_nn(loop, generation):
    sp_params = params.self_play

    model = params.nn.model_class(params)
    model.load_parameters(generation, to_device=params.self_play.pytorch_devices)

    nn_wrapper = NeuralNetWrapper(model, params)
    nnet = AsyncBatchedProxy(nn_wrapper, batch_size=sp_params.nn_batch_size,
                                timeout=sp_params.nn_batch_timeout,
                                batch_builder=sp_params.nn_batch_builder,
                                cache_size=400000)

    fut = asyncio.ensure_future(nnet.run(), loop=loop)

    df = pd.read_csv("../test/test_boards.csv", comment="#", sep=";", index_col="id")
    df = df.applymap(lambda s: list(map(int, s.split(" "))) if isinstance(s, str) else s)
    games = []
    for idx, sample in df.iterrows():
        g = BoxesState()
        for m in sample.moves:
            g.play_(int(m))
        games.append(g)
    df["game"] = games

    nn_corrects = 0
    mcts_corrects = 0
    for i, sample in df.iterrows():
        print(f"Generate tikz for sample {i}")
        p, v = await nnet(sample.game)
        nn_corrects += p.argmax() in sample.next_moves
        if generation in SAVE_TIKZ_GEN:
            tikz = game_to_tikz(sample.moves, sample.next_moves, p)
            with open(f"positions_samples/{RUN}/gen{generation}_nn_{i}.tikz", "w") as f:
                f.write(tikz)

        root_node = create_root_uct_node(sample.game)
        p = await UCT_search(root_node, NUM_SEARCHES, nnet, max_pending_evals=1, dirichlet=DIRICHLET, cpuct=CPUCT)
        p = p / (p.sum() or 1.0)
        mcts_corrects += p.argmax() in sample.next_moves
        if generation in SAVE_TIKZ_GEN:
            tikz = game_to_tikz(sample.moves, sample.next_moves, p)
            with open(f"positions_samples/{RUN}/gen{generation}_mcts_{i}.tikz", "w") as f:
                f.write(tikz)

    with open(f"positions_samples/{RUN}/accuracy.txt", "a") as f:
        f.write(f"{generation},{nn_corrects/df.shape[0]},{mcts_corrects/df.shape[0]}\n")

    fut.cancel()
    await fut

if not os.path.exists(f"positions_samples/{RUN}/"):
    os.makedirs(f"positions_samples/{RUN}/")

with open(f"positions_samples/{RUN}/accuracy.txt", "w") as f:
    f.write(f"generation,nn_ratio_correct,mcts_ratio_corrects\n")

loop = asyncio.new_event_loop()
for gen in range(61):
    print("Process generation:", gen)
    loop.run_until_complete(test_mcts_nn(loop, gen))
