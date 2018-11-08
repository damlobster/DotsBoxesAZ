#!/usr/bin/env python3
# encoding: utf-8

#./dotsandboxesagent_az.py "configuration.resnet20" "resnet-0811-0020" 10.0.0.11 8081
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
import multiprocessing as mp

import sys
sys.path.append("..")
import configuration
from mcts import UCT_search, create_root_uct_node
from players import AZPlayer

logger = logging.getLogger(__name__)

games = {}
Game = None
azplayer = None
queues = (None, None)

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
    def __init__(self, time_limit, game_uuid):
        """Create Dots and Boxes agent.

        :param player: Player number, 1 or 2
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        :param timelimit: Maximum time allowed to send a next action.
        :param nn: the neural net model
        :param nn_chkpts_filename: the pickled state_dict of the model
        """
        logger.debug("Init agent" + str((time_limit, game_uuid)))
        self.player = set()
        self.time_limit = time_limit
        self.BSIZE = Game.FEATURES_SHAPE[1] * Game.FEATURES_SHAPE[2]
        self.state = Game()
        self.req_queue, self.res_queue = queues
        self.game_uuid = game_uuid
        self.generations = [None, 0, 0]

    def add_player(self, player, generation):
        """Use the same agent for multiple players."""
        self.player.add(player)
        self.generations[player] = generation

    def register_action(self, row, column, orientation):
        """Register action played in game.

        :param row:
        :param columns:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        move = 0 if orientation == "h" else self.BSIZE
        move += row * Game.FEATURES_SHAPE[2] + column
        if self.state.get_valid_moves()[move]:
            logger.debug("BEFORE move: %s", self.state) 
            self.state.play_(move)
            logger.debug("AFTER move: %s", self.state)
        else:
            logger.debug("Move already played, ignore !")

    async def next_action(self, player):
        """Return the next action this agent wants to perform.

        In this example, the function implements a random move. Replace this
        function with your own approach.

        :return: (row, column, orientation)
        """
        logger.debug("Computing next move (uuid={}, player={})"\
                .format(self.game_uuid, player))

        if player != self.state.next_player+1:
            return None

        try:
            self.req_queue.put((self.game_uuid, self.state, self.generations[player], self.time_limit))
            loop = asyncio.get_event_loop()
            game_uuid, move = await loop.run_in_executor(None, self.res_queue.get)
            if game_uuid != self.game_uuid:
                print("Games UUID don't match, received=", game_uuid, "self=", self.game_uuid, flush=True)

            logger.debug("move mcts="+str(move))
            if move is None:
                return -1, -1, "not_my_turn"

            o = "h" if move < self.BSIZE else "v"
            r = (move % self.BSIZE) // Game.FEATURES_SHAPE[2]
            c = (move % self.BSIZE) % Game.FEATURES_SHAPE[2]
            logger.debug("move web: r=%d, c=%d, o=%s", r, c, o)
            return int(r), int(c), o
        except Exception as err:
            logger.exception("Error occured in next_action()")

    def end_game(self):
        self.state = Game()


## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    generation = int(path[1:]) if path != "/" else 0
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            uuid = msg["game"]
            answer = None
            if msg["type"] == "start":
                # Initialize game
                if uuid in games:
                    logger.debug(f"Add player {msg['player']} with gen {generation}")
                    games[uuid].add_player(msg["player"], generation)
                else:
                    logger.debug("Init new game")
                    nb_rows, nb_cols = msg["grid"]
                    games[uuid] = DotsAndBoxesAgent(msg["timelimit"], uuid)
                    games[uuid].add_player(msg["player"], generation)

                if msg["player"] == 1:
                    # Start the game
                    try:
                        nm = await games[uuid].next_action(msg["player"])
                    except Exception as e:
                        logger.exception(e)
                    if nm is None:
                        # Game over
                        if uuid in games: del games[uuid]
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
                games[uuid].register_action(r, c, o)
                if msg["nextplayer"] in games[uuid].player:
                    # Compute your move
                    nm = await games[uuid].next_action(msg["nextplayer"])
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        if uuid in games: del games[uuid]
                        continue
                    nr, nc, no = nm
                    if no != "not_my_turn":
                        answer = {
                            'type': 'action',
                            'location': [nr, nc],
                            'orientation': no
                        }
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                if uuid in games: del games[uuid]
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        print(err, flush=True)
        logger.exception("Connection closed")
    except Exception as ex:
        print(ex, flush=True)
        logger.exception("Other exception!")

    logger.info("Exit handler")


def start_server(ip, port):
    server = websockets.serve(handler, ip, port)
    print(f"Running on ws://{ip}:{port}")
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()
    asyncio.get_event_loop().set_debug()


## COMMAND LINE INTERFACE

def main(argv=None):
    global azplayer
    global queues
    global Game

    parser = argparse.ArgumentParser(description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('params', default=configuration.params, type=str, help='eg. "configuration.resnet20')
    parser.add_argument('exp', type=str, help='eg. "resnet-0811-0900')
    parser.add_argument('ip', metavar='IP', type=str, help='IP to use for server')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))

    params = eval(args.params)
    Game = params.game.clazz
    params.game.init()
    params.rewrite_str("_exp_", args.exp)
    params.rewrite_str("data/", "../data/")

    queues = (mp.Queue(), mp.Queue())
    azplayer = AZPlayer(params, 1, *queues)
    azplayer.start()

    start_server(args.ip, args.port)


if __name__ == "__main__":
    sys.exit(main())
