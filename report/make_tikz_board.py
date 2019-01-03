import sys
sys.path.append("..")
import numpy as np
from dots_boxes.dots_boxes_game import BoxesState
from tikz import Path, Picture
import argparse


def dot(row, col):
    return f"{row}{col}"


class BoxesGameTikz:
    def __init__(self, rows, columns, bw=False, style="x=.75pt,y=.75pt,yscale=-1.75,xscale=1.75,every node/.style={inner sep=0,outer sep=0}"):
        self.rows = rows
        self.columns = columns
        self.bw = bw
        self.pic = Picture(style=style)

        with self.pic.path(style="black, fill=black") as draw:
            for row in range(rows+1):
                for col in range(columns+1):
                    draw.at(10j*row + 10 * col)\
                        .node(name=dot(row, col))\
                        .circle(0.5)

    def color(self, to_play):
        style = ""
        if not self.bw and to_play is not None:
            style += "red," if to_play else "blue,"
        return style

    def draw_moves(self, *moves):
        for m in moves:
            self.draw_move(m)

    def draw_move(self, move, to_play=None, label=None, style=""):
        if move < 16:
            row = move // (self.rows + 1)
            col = move % (self.columns + 1)
            at = dot(row, col)
            to = dot(row, col + 1)
        else:
            move = move - 16
            row = move // (self.rows + 1)
            col = move % (self.columns + 1)
            at = dot(row, col)
            to = dot((row + 1), col)

        with self.pic.path(style="draw,"+self.color(to_play)+style) as draw:
            edge = draw.at(at).line_to(to)
            if label:
                edge.node(text="\\tiny" + str(label),
                            style="fill=white,midway,sloped")

    def fill_boxes(self, player, *boxes):
        for r, c in boxes:
            with self.pic.path("draw," + self.color(player)) as draw:
                draw.at(r * 10j + 5j + c * 10 +
                        5).node("A" if player else "B")

    def make(self):
        return self.pic.make()

def game_to_tikz(moves, next_move, probs=[], bw=False, dims=(3, 3)):
    BoxesState.init_static_fields((dims,))
    bs = BoxesState()
    bs.to_play = 1
    tikz = BoxesGameTikz(*dims, bw)
    for m in moves:
        tikz.draw_move(m, style="" if m!=moves[-1] else "line width=1.0")
        closed = bs.play_(m)
        if closed:
            tikz.fill_boxes(bs.just_played, *closed)
    probs = np.asarray(probs)
    max_probs = probs.argsort()#[:-5:-1]
    ps = ""
    if max_probs is not None and max_probs.any():
        probs = probs.round(2)
        for i in max_probs:
            if probs[i] > 0.05 or i in next_move:
                p = f"{probs[i]:.2f}".lstrip('0').rstrip('0')
                if probs[i]==0:
                    p = '.0'
                ps += f"{i}->{p};"
                tikz.draw_move(i, bs.to_play if i in next_move else None, p, "" if i in next_move else "gray")
    else:
        tikz.draw_move(next_move, bs.to_play, "$\\times$")

    s = str(bs)
    s += f"probs={ps}\nnext_move={next_move}"
    s = "\n".join(map(lambda line: "% "+line, s.split("\n")))
    return s + "\n" + tikz.make()

def multicols(buffer, cols, spacing="5pt"):
    result = []
    for b in buffer:
        result.append("""\\begin{subfigure}[b]{%.2f\\textwidth}
\\centering
%s  
\\caption{}
%%\\label{}
\\end{subfigure}""" % (1/cols, b)) #spacing, 
    return "\n".join(result) #\\resizebox{\columnwidth}{!}{\hspace*{%s}%s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tikz figure')
    parser.add_argument('moves', type=int, nargs='+', help='moves')
    parser.add_argument('-f', '--from', dest='_from', type=int, default=0, help='Start print from this move')
    parser.add_argument('-b', '--bw', dest='bw', type=bool, default=False, help='Black&White')
    parser.add_argument('-c', '--cols', dest='cols', type=int, default=0, help='Number of columns')
    parser.add_argument('-s', '--spacing', dest='spacing', type=str, default=0, help='The spacing between columns')
    args = parser.parse_args()

    result = "nothing"
    if args.cols > 0:
        buffer = []
        for i in range(args._from, len(args.moves)):
            buffer.append(game_to_tikz(args.moves[:i], args.moves[i], bw=args.bw))
        result = multicols(buffer, args.cols, args.spacing)
    else:
        result = game_to_tikz(args.moves[:i-1], args.moves[i], bw=args.bw)

    print(result)
