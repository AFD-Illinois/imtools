__license__ = """
 File: comparison.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__doc__ = \
"""Tools related to comparing sets (>2 images) of similar polarized images.
Probably not broadly useful outside the context of comparing polarized GRRT schemes.
"""

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from imtools.parallel import map_parallel
from imtools.figures import plot_stokes_rows
from imtools.plots import _colorbar

def generate_table(data, fn, n_returns, symmetric=False):
    """Generate a comparison table between several images.

    :param data: a dictionary of Image objects keyed by names
    :param fn: a function which takes two Image objects and returns
               a list of n_returns results (see e.g. 'stats.mses')
    :param n_returns: length of the list of stats returned by running fn

    :returns table: a 2D array of each comparator output, indexed by the keys() list
                   in each direction (i.e. table[i,i] == 0 for most comparison metrics)
    """
    names = list(data.keys())
    nnames = len(names)

    # Parallel, beta2 is expensive
    def run_fn(codes, fn=fn, data=data):
            return np.nan_to_num(fn(data[codes[0]], data[codes[1]]), nan=1)

    if symmetric:
        all_combos = list(itertools.combinations(names, 2))
    else:
        all_combos = list(itertools.product(names, names))

    vals = map_parallel(run_fn, all_combos)

    table = np.zeros((nnames, nnames, n_returns))
    for i, codes in enumerate(all_combos):
        table[names.index(codes[0]), names.index(codes[1]), :] = vals[i][:n_returns]
        if symmetric:
            table[names.index(codes[1]), names.index(codes[0]), :] = -vals[i][:n_returns]

    return table

def table_color_plot(table, names, cmap='RdBu_r', n_tables=4, figsize=(14,4),
                    labels=("Stokes I", "Stokes Q", "Stokes U", "Stokes V"),
                    clabels=("%", "%", "%", "%"),
                    is_percent=(True, True, True, True), polar=False, vmax=None, shrink_text_by=2.5,
                    upper_tri_only=True):
    """Plot a 2D array indexed by the list 'names' on each axis.
    Usually for plotting output of generate_table for comparisons
    """
    names = list(names)
    nnames = len(names)
    if n_tables == 4:
        fig, ax = plt.subplots(2, 2, figsize=(8, 3.5))
        ax = ax.flatten()
    elif n_tables == 6:
        fig, ax = plt.subplots(2, 3, figsize=(10, 3.5))
        ax = ax.flatten()
    else:
        fig, ax = plt.subplots(1, n_tables, figsize=figsize)

    if isinstance(cmap, str):
        cmap = (cmap,)*n_tables

    for i_table,label in enumerate(labels[:n_tables]):
        if vmax is None:
            lvmax = np.max(np.abs(table[:, :, i_table] * (1,100)[is_percent[i_table]]))
        else:
            lvmax = vmax[i_table]

        if cmap[i_table] == 'RdBu_r' or cmap[i_table] == 'bwr':
            lvmin = -lvmax
        else:
            lvmin = 0

        row_labels = names
        col_labels = names
        if n_tables == 4:
            if i_table == 1 or i_table == 3:
                row_labels = [""]*len(names)
            if i_table == 2 or i_table == 3:
                col_labels = [""]*len(names)
        elif n_tables == 6:
            # Quash rows except left side
            if i_table not in (0, 3):
                row_labels = [""]*len(names)
            # Quash cols on bottom row
            if i_table not in (0, 1, 2):
                col_labels = [""]*len(names)
        elif i_table != 0: # For plots arranged in a line
                row_labels = [""]*len(names)

        if upper_tri_only:
            # Zero table elements below and on the diagonal
            for tab_i in range(table.shape[0]):
                for tab_j in range(table.shape[1]):
                    if tab_i >= tab_j:
                        table[tab_i, tab_j, :] = 0
            # Take out first column and last row of labels and table
            row_labels = row_labels[:-1]
            col_labels = col_labels[1:]
            table_vals = table[:-1, 1:, i_table]*(1,100)[is_percent[i_table]]
        else:
            table_vals = table[:, :, i_table]*(1,100)[is_percent[i_table]]

        im, cbar = _heatmap(table_vals, row_labels, col_labels, ax=ax[i_table],
                            cmap=cmap[i_table], cbarlabel=clabels[i_table], vmax=lvmax, vmin=lvmin, aspect='auto')
        _annotate_heatmap(im, valfmt=("{x:.2g}", "{x:.2f}%")[is_percent[i_table]], threshold=lvmax/2,
                          shrink_text_by=shrink_text_by)
        ax[i_table].set_title(label, pad=20)

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    return fig

def table_interleave_plot(table, names, cmap='RdBu_r', vmax=None, n_stokes=4, figsize=(8,5), cbarlabel=""):
    """Plot a 2D array indexed by the list 'names' on each axis.
    Usually for plotting output of generate_table for comparisons
    """
    names = list(names)
    nnames = len(names)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmax is None:
        vmax = np.max(np.abs(100 * table))
    if cmap == 'RdBu_r' or cmap == 'bwr':
        vmin = -vmax
    else:
        vmin = 0

    table2 = np.zeros((table.shape[0], table.shape[1]*n_stokes))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            for s in range(n_stokes):
                table2[i,n_stokes*j+s] = table[i,j,s]

    inames = [(names[i//n_stokes]+" " if i % n_stokes == 0 else "") + (["I","Q","U","V"][i%n_stokes]) for i in range(nnames*n_stokes)]
    im, cbar = _heatmap(table2*100, names, inames, ax=ax, cmap=cmap, cbar_kw={'size': "2%", 'pad': 0.02}, cbarlabel=cbarlabel, vmax=vmax, vmin=vmin)
    for i in range(1, nnames):
        ax.axvline(i * n_stokes - 0.5, ymin=0, ymax=nnames, color='k')
        ax.axhline(i - 0.5, xmin=0, xmax=nnames * n_stokes, color='k')
    _annotate_heatmap(im, valfmt="{x:.1f}%", threshold=vmax/2)
    fig.tight_layout()
    return fig

def table_interleave_plot_rotated(table, names, cmap='RdBu_r', vmax=None, n_stokes=4, figsize=(8,5), cbarlabel=""):
    """Plot a 2D array indexed by the list 'names' on each axis.
    Usually for plotting output of generate_table for comparisons
    """
    names = list(names)
    nnames = len(names)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmax is None:
        vmax = np.max(np.abs(100 * table))
    if cmap == 'RdBu_r' or cmap == 'bwr':
        vmin = -vmax
    else:
        vmin = 0

    table2 = np.zeros((table.shape[0]*n_stokes, table.shape[1]))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            for s in range(n_stokes):
                table2[n_stokes*i+s,j] = table[i,j,s]

    inames = [(names[i//n_stokes]+" " if i % n_stokes == 0 else "") + (["I","Q","U","V"][i%n_stokes]) for i in range(nnames*n_stokes)]
    im, cbar = _heatmap(table2*100, inames, names, ax=ax, cmap=cmap, cbar_kw={'size': "2%", 'pad': 0.02}, cbarlabel=cbarlabel, vmax=vmax, vmin=vmin)
    for i in range(1, nnames):
        ax.axhline(i * n_stokes - 0.5, xmin=0, xmax=nnames, color='k')
        ax.axvline(i - 0.5, ymin=0, ymax=nnames * n_stokes, color='k')
    _annotate_heatmap(im, valfmt="{x:.1f}%", threshold=vmax/2)
    return fig

def print_table(table, names, color=False, cmap='RdBu_r', n_stokes=4, figsize=(6, 10)):
    """Make a table from a 2D array indexed by the list 'names' on each axis.
    Usually for showing output of generate_table for comparisons
    """
    names = list(names)
    # Initialize the colormap
    colormap = matplotlib.cm.get_cmap(cmap)
    norm = np.max(np.abs(table))

    fig, ax = plt.subplots(n_stokes, 1, figsize=figsize)
    for i,st in enumerate(["I", "Q", "U", "V"][:n_stokes]):
        tb = table[:,:,i]
        # Generate text and background colors to fill table
        text = [ [] for _ in range(tb.shape[0]) ]
        colors = [ [] for _ in range(tb.shape[0]) ]
        lnames = [name + " " + st for name in names]
        for ii in range(tb.shape[0]):
            for jj in range(tb.shape[1]):
                text[ii].append("{:.1f}%".format(100*tb[ii, jj]))
                colors[ii].append(colormap(tb[ii, jj] / norm / 2 + 0.5))

        ax[i].set_axis_off()
        tbl = ax[i].table(cellText=text, cellColours=colors, rowLabels=lnames, colLabels=lnames,
                          loc='center')
        tbl.scale(1, 1.5)
        ax[i].set_ylabel("Stokes {}".format(st))

    return fig

def compare_all_with(data, name, polar=False, compare_abs=True, **kwargs):
    """Compare each other element in 'data' with the named element
    compare_abs specifies an absolute difference, otherwise the relative difference is taken
    relative diff is clipped to [-1,1]
    """
    if polar:
        if compare_abs:
            compare_list = []
            for n in data.keys():
                if n != name:
                    new_img = data[n] - data[name]
                    new_img.Q = np.sqrt(data[n].Q**2 + data[n].U**2) - np.sqrt(data[name].Q**2 + data[name].U**2)
                    new_img.U = data[n].evpa() - data[name].evpa()
                    compare_list.append(new_img)
        else:
            compare_list = []
            for n in data.keys():
                if n != name:
                    new_img = data[n] - data[name]
                    new_img.Q = np.clip((np.sqrt(data[n].Q**2 + data[n].U**2) - np.sqrt(data[name].Q**2 + data[name].U**2)) / \
                                        np.sqrt(data[name].Q**2 + data[name].U**2), -1, 1)
                    new_img.U = data[n].evpa() - data[name].evpa()
                    compare_list.append(new_img)
    else:
        if compare_abs:
            compare_list = [data[n] - data[name] for n in data.keys() if n != name]
        else:
            compare_list = [data[n].rel_diff(data[name], clip=(-1,1)) for n in data.keys() if n != name]

    fig = plot_stokes_rows(compare_list, relative=True, **kwargs)
    fig.suptitle(["Relative ", "Absolute "][int(compare_abs)] + " comparisons with " + name)
    return fig


# Following stolen from matplotlib docs at
# https://matplotlib.org/3.2.2/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

def _heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbarlabel != "":
        cbar = _colorbar(im, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, shrink_text_by=1, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is None:
        threshold = data.max()/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    old_size = plt.rcParams['font.size']
    plt.rc('font', size=old_size - shrink_text_by)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(abs(data[i, j]) > threshold)])
            if data[i,j] != 0.0:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    
    plt.rc('font', size=old_size)

    return texts