import matplotlib.pyplot as plt


def border(ax, left=False, right=False, top=False, bottom=False):
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)


# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

def add_scalebar(ax, pos, size, direction="vertical", width=3):
    if direction == "vertical":
        obj = ax.vlines([pos[0]], pos[1], pos[1] + size, linewidth=width, color="Black")
    else:
        obj = ax.hlines([pos[1]], pos[0], pos[0] + size, linewidth=width, color="Black")
    return obj
