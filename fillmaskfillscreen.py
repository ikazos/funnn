import curses
from curses import textpad
import math
import textwrap
import time
import random

import distilbert_fillmask
import nltk
import numpy as np
from scipy.special import softmax

def main(stdscr):
    # Clear screen
    stdscr.clear()

    WIDTH = curses.COLS
    HEIGHT = curses.LINES

    def makebox(prefix, masked="", suffix="", width=20):
        sentence = " ".join([prefix, masked, suffix])
        lines = textwrap.wrap(sentence, width=width)

        words_on_lines = [] # (word, offset, is masked word?)
        all_len = 0
        prefix_found = False
        for line in lines:
            words_on_line = []
            offset = 0

            for word in line.split():
                if (not prefix_found) and (all_len >= len(prefix)):
                    prefix_found = True
                    words_on_line.append((word, offset, True))
                else:
                    words_on_line.append((word, offset, False))
                    all_len += len(word) + 1

                offset += len(word) + 1

            words_on_lines.append(words_on_line)

        width = max(map(len, lines)) + 1
        height = len(lines) + 1

        return (words_on_lines, width, height)

    def drawbox(box, yx=None):
        (words_on_lines, w, h) = box

        if yx is not None:
            y, x = yx
        else:
            y = math.floor(random.random() * (HEIGHT - h))
            x = math.floor(random.random() * (WIDTH - w))

        for yy in range(y, y + h):
            stdscr.addstr(yy, x, " " * w)

        textpad.rectangle(stdscr, y, x, y + h, x + w)

        for k, words_on_line in enumerate(words_on_lines):
            for word, offset, is_masked in words_on_line:
                attr = curses.A_BOLD if is_masked else curses.A_NORMAL
                stdscr.addstr(y + 1 + k, x + 1 + offset, word, attr)

        if yx is None:
            return (y, x)

    def bresenham(x1, y1, x2, y2):
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0 or slope > 1:
            raise Exception(f"slope {slope} not in range [0, 1].")

        # https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html
        route = []
        dx = x2 - x1
        dy = y2 - y1
        y = y1
        eps = 0

        for x in range(x1, x2 + 1): # +1 for inclusive end
            route.append((y, x))
            eps += dy
            if eps * 2 >= dx:
                y += 1
                route.append((y, x)) # not pixel perfect anymore!
                eps -= dx

        return route

    def getdir(yx0, yx1):
        y0, x0 = yx0
        y1, x1 = yx1
        if x0 == x1 and y0 == y1 - 1:
            return "down"
        elif x0 == x1 and y0 == y1 + 1:
            return "up"
        elif y0 == y1 and x0 == x1 - 1:
            return "right"
        elif y0 == y1 and x0 == x1 + 1:
            return "left"
        else:
            raise Exception(f"Bad direction from yx = ({yx0}) to yx = ({yx1})")

    def getacs(yx0, yx1, yx2):
        indir = getdir(yx0, yx1)
        outdir = getdir(yx1, yx2)
        dirs = (indir, outdir)

        if dirs in (("down", "down"), ("up", "up")):
            return curses.ACS_VLINE
        elif dirs in (("left", "left"), ("right", "right")):
            return curses.ACS_HLINE
        elif dirs in (("right", "up"), ("down", "left")):
            return curses.ACS_LRCORNER
        elif dirs in (("right", "down"), ("up", "left")):
            return curses.ACS_URCORNER
        elif dirs in (("left", "up"), ("down", "right")):
            return curses.ACS_LLCORNER
        elif dirs in (("left", "down"), ("up", "right")):
            return curses.ACS_ULCORNER
        else:
            raise Exception(f"Bad dirs: {dirs}")

    def drawline(yx1, yx2):
        y1, x1 = yx1
        y2, x2 = yx2

        slope = (y2 - y1) / math.copysign(max(0.001, abs(x2 - x1)), x2 - x1)

        flipx = x2 < x1
        flipy = y2 < y1
        swapxy = abs(slope) > 1

        if flipx:
            x1 *= -1
            x2 *= -1

        if flipy:
            y1 *= -1
            y2 *= -1

        if swapxy:
            t1, t2 = x1, x2
            x1, x2 = y1, y2
            y1, y2 = t1, t2

        points = bresenham(x1, y1, x2, y2)

        yx0 = (
            points[0][0] - (points[1][0] - points[0][0]),
            points[0][1] - (points[1][1] - points[0][1])
        )

        yx3 = (
            points[-1][0] - (points[-2][0] - points[-1][0]),
            points[-1][1] - (points[-2][1] - points[-1][1])
        )

        points = [ yx0 ] + points + [ yx3 ]

        def unwind(yx, swapxy, flipy, flipx):
            y, x = yx
            if swapxy:
                t = y
                y = x
                x = t

            if flipy:
                y *= -1

            if flipx:
                x *= -1

            return (y, x)

        for k, yx in enumerate(points):
            if k == 0 or k == len(points) - 1:
                continue

            yx = unwind(yx, swapxy, flipy, flipx)

            prev_yx = points[k - 1] if k > 0 else yx0
            next_yx = points[k + 1] if k < len(points) - 1 else yx3

            prev_yx = unwind(prev_yx, swapxy, flipy, flipx)
            next_yx = unwind(next_yx, swapxy, flipy, flipx)

            acs = getacs(prev_yx, yx, next_yx)

            y, x = yx
            stdscr.addch(y, x, acs)

    def testbox():
        box = makebox("This is a", "test", "sentence! Don't take it too seriously.")
        drawbox(box)

    def testline():
        cy = HEIGHT // 2
        cx = WIDTH // 2

        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)

        drawline((cy + 5, cx + 4), (cy + 5, cx + 5), (cy + 10, cx + 10), (cy + 10, cx + 11))
        #drawline((cy + 5, cx - 5), (cy + 10, cx - 10))
        #drawline((cy - 5, cx + 5), (cy - 10, cx + 10))
        #drawline((cy - 5, cx - 5), (cy - 10, cx - 10))

        stdscr.attron(curses.color_pair(1))
        #drawline((cy + 3, cx + 5), (cy + 6, cx + 10))
        #drawline((cy + 3, cx - 5), (cy + 6, cx - 10))
        #drawline((cy - 3, cx + 5), (cy - 6, cx + 10))
        #drawline((cy - 3, cx - 5), (cy - 6, cx - 10))
        stdscr.attroff(curses.color_pair(1))

        stdscr.attron(curses.color_pair(2))
        #drawline((cy + 5, cx + 3), (cy + 10, cx + 6))
        #drawline((cy + 5, cx - 3), (cy + 10, cx - 6))
        #drawline((cy - 5, cx + 3), (cy - 10, cx + 6))
        #drawline((cy - 5, cx - 3), (cy - 10, cx - 6))
        stdscr.attroff(curses.color_pair(2))

        #drawline((cy, cx + 5), (cy, cx + 10))
        #drawline((cy, cx - 5), (cy, cx - 10))
        #drawline((cy + 5, cx), (cy + 10, cx))
        #drawline((cy - 5, cx), (cy - 10, cx))

    #testbox()
    #testline()

    sentence = "The boy kicked the horse."

    box = makebox(sentence)
    boxw = box[1]
    boxh = box[2]
    box_yx = drawbox(box)
    box_cyx = (box_yx[0] + boxh // 2, box_yx[1] + boxw // 2)

    while stdscr.getkey() != 'q':
        words = nltk.word_tokenize(sentence)
        k = math.floor(random.random() * len(words))

        prefix = " ".join(words[:k])
        suffix = " ".join(words[k+1:])

        masked_sentence = " ".join([prefix, "[MASK]", suffix])
        res = distilbert_fillmask.unmask_sentences([masked_sentence])
        candidates = [ obj["token_str"] for obj in res ]
        scores = np.array([ obj["score"] for obj in res ])

        pvals = softmax(scores)
        onehot = np.random.multinomial(1, pvals, size=1)
        masked = candidates[np.argmax(onehot)]

        newbox = makebox(prefix, masked, suffix)
        newboxw = newbox[1]
        newboxh = newbox[2]
        newboxy = math.floor(random.random() * (HEIGHT - newboxh))
        newboxx = math.floor(random.random() * (WIDTH - newboxw))
        newbox_cyx = (newboxy + newboxh // 2, newboxx + newboxw // 2)
        newbox_yx = (newboxy, newboxx)

        sentence = " ".join([prefix, masked, suffix])
        drawline(box_cyx, newbox_cyx)
        drawbox(box, yx=box_yx)
        drawbox(newbox, yx=newbox_yx)

        box = newbox
        box_yx = newbox_yx
        box_cyx = newbox_cyx

curses.wrapper(main)