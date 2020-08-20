# Overall statistics and figures for comparisons of >2 closely-related images

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from imtools.reports import *

def generate_table(data, fn):
    """Generate a comparison table between several images.
    Takes a dict 'data' of Image objects keyed by names, and an operation 'fn' which
    takes two image objects and returns a list of 4 results (see e.g. 'stats.mses')
    """
    names = list(data.keys())
    nnames = len(names)

    table = np.zeros((nnames,nnames,4))
    for code1 in names:
        for code2 in names:
            new_vec = fn(data[code1], data[code2])
            # Don't record NaNs
            for i in range(len(new_vec)):
                if np.isfinite(new_vec[i]):
                    table[names.index(code1), names.index(code2)][i] = new_vec[i]

    return table

def table_color_plot(table, names, cmap='RdBu_r', figsize=(14,4)):
    names = list(names)
    nnames = len(names)
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    for i,st in enumerate(["I", "Q", "U", "V"]):
        vmax = np.max(np.abs(table))
        vmin = -vmax
        ax[i].pcolormesh(table[:,:,i], vmax=vmax, vmin=vmin, cmap=cmap)
        ax[i].set_xticks(np.arange(nnames)+0.5)
        ax[i].set_yticks(np.arange(nnames)+0.5)
        ax[i].set_xticklabels(names, rotation=75)
        ax[i].set_yticklabels(names)
        ax[i].set_title("Stokes {}".format(st))
        ax[i].set_aspect('equal')
    fig.tight_layout()
    return fig, ax

def print_table(table, names, color=False, cmap='RdBu_r', figsize=(6, 10)):
    names = list(names)
    # Initialize the colormap
    colormap = matplotlib.cm.get_cmap(cmap)
    norm = np.max(np.abs(table))

    fig, ax = plt.subplots(4, 1, figsize=figsize)
    for i,st in enumerate(["I", "Q", "U", "V"]):
        tb = table[:,:,i]
        # Generate text and background colors to fill table
        text = [ [] for _ in range(tb.shape[0]) ]
        colors = [ [] for _ in range(tb.shape[0]) ]
        for ii in range(tb.shape[0]):
            for jj in range(tb.shape[1]):
                text[ii].append("{:.1f}%".format(100*tb[ii, jj]))
                colors[ii].append(colormap(tb[ii, jj] / norm / 2 + 0.5))

        ax[i].set_axis_off()
        tbl = ax[i].table(cellText=text, cellColours=colors, rowLabels=names, colLabels=names,
                          loc='center')
        tbl.scale(1, 1.5)
        ax[i].set_title("Stokes {}".format(st))

    return fig, ax

def all_compares_code(data, name, abs=True):
    if abs:
        compare_list = [data[n] - data[name] for n in data.keys() if n != name]
    else:
        compare_list = [data[n].rel_diff(data[name], clip=(-1,1)) for n in data.keys() if n != name]
    fig, ax = plot_stokes_line(compare_list, relative=True)
    fig.suptitle("Comparisons with "+name)
    return fig, ax