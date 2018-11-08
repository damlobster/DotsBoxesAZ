#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxesagent_az.py
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random
from functools import partial

from dots_boxes.dots_boxes_game import BoxesState
import configuration
from mcts import UCT_search, create_root_uct_node

logger = logging.getLogger(__name__)
games = {}
agentclass = None


def _fut_cb(fut):
    if fut.exception():
        print(fut.exception(), flush=True)
        raise fut.exception()


class DotsAndBoxesAgent:
    """Example Dots and Boxes agent implementation base class.
    It returns a random next move.

    A DotsAndBoxesAgent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game

    This class does not necessarily use the best data structures for the
    approach you want to use.
    """
    def __init__(self, player, timelimit, params, generation):
        """Create Dots and Boxes agent.

        :param player: Player number, 1 or 2
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        :param timelimit: Maximum time allowed to send a next action.
        :param nn: the neural net model
        :param nn_chkpts_filename: the pickled state_dict of the model
        """
        self.player = {player}
        self.timelimit = timelimit
        self.params = params
        self.params.game.init()
        self.BSIZE = self.state.FEATURES_SHAPE[1] * self.state.FEATURES_SHAPE[2]
        self.state = BoxesState()
        nn = params.nn.model_class(params)
        nn.load_parameters(generation)
        nn_wrapper = NeuralNetWrapper(self.nn, params)
        nnet = AsyncBatchedProxy(nn_wrapper, batch_size=sp_params.nn_batch_size,
                                    timeout=sp_params.nn_batch_timeout,
                                    batch_builder=sp_params.nn_batch_builder,
                                    cache_size=0)
        self.nn = nnet
        self.tasks = []
        fut = asyncio.ensure_future(_env_.nnet.run(), loop=loop)
        fut.add_done_callback(_fut_cb)  # re-raise exception if occured in nnet
        self.tasks.append(fut)

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)

    def register_action(self, row, column, orientation, player):
        """Register action played in game.

        :param row:
        :param columns:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        move = 0 if orientation == "h" else self.BSIZE
        move += row * self.state.FEATURES_SHAPE[2] + column
        self.state.play_(move)

    def next_action(self):
        """Return the next action this agent wants to perform.

        In this example, the function implements a random move. Replace this
        function with your own approach.

        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))

        move = UCT_search(create_root_uct_node(self.state), None, self.nn, \
            self.params.self_play.mcts.mcts_cpuct, \
            self.params.self_play.mcts.max_async_searches, (0.0, 0.0))


        o = "h" if move < self.BSIZE else "v"
        r = (move % self.BSIZE) / self.state.FEATURES_SHAPE[2]
        c = (move % self.BSIZE) % self.state.FEATURES_SHAPE[2]
        return r, c, o

    def end_game(self):
        self.state = self.params.game.clazz()


## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    game = None
    # msg = await websocket.recv()
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None
            if msg["type"] == "start":
                # Initialize game
                if msg["game"] in games:
                    games[msg["game"]].add_player(msg["player"])
                else:
                    nb_rows, nb_cols = msg["grid"]
                    games[msg["game"]] = agentclass(msg["player"],
                                                    nb_rows,
                                                    nb_cols,
                                                    msg["timelimit"])
                if msg["player"] == 1:
                    # Start the game
                    nm = games[game].next_action()
                    print('nm = {}'.format(nm))
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    r, c, o = nm
                    answer = {
                        'type': 'action',
                        'location': [r, c],
                        'orientation': o
                    }
                else:
                    # Wait for the opponent
                    answer = None

            elif msg["type"] == "action":
                # An action has been played
                r, c = msg["location"]
                o = msg["orientation"]
                games[game].register_action(r, c, o, msg["player"])
                if msg["nextplayer"] in games[game].player:
                    # Compute your move
                    nm = games[game].next_action()
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    nr, nc, no = nm
                    answer = {
                        'type': 'action',
                        'location': [nr, nc],
                        'orientation': no
                    }
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                print(answer)
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass
    parser = argparse.ArgumentParser(description='Start agent to play Dots and Boxes')
    parser.add_argument('params', default=configuration.params, type=str, help='eg. "configuration.resnet20')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    params = eval(args.params)

    agentclass = partial(DotsAndBoxesAgent, params=params)
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
