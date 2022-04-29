#!/usr/bin/env python3

"""Simple plotting script that explicitly apes ipole's plot.py and plot_pol.py
"""

import sys
import click

from imtools.io import read_image
from imtools.figures import *

@click.command()
@click.argument('fnames', nargs=-1, type=click.Path(exists=True))
# Plot types
@click.option('-s', '--simple', is_flag=True, help="Simpler 4-pane plot")
@click.option('-u', '--unpol', is_flag=True, help="1-pane plot")
# General plotting options
@click.option('--fig_x', default=8.0, help="Figure width in inches.")
@click.option('--fig_y', default=8.0, help="Figure height in inches.")
@click.option('--fig_dpi', default=100, help="DPI of resulting figure.")
# EVPA tick overlay
@click.option('-o', '--overlay', is_flag=True, help="Overlay EVPA ticks")
@click.option('-n', '--n_ticks', default=20, help="Number of EVPA angle ticks to plot")
def plot(fnames, simple, unpol, fig_x, fig_y, fig_dpi, overlay, n_ticks):
    """USAGE: plot_pol.py [options] fname [fname fname]

    Plots some files in ipole HDF5 format.  By default, plots polarized image
    statistics.  You can plot the Stokes parameters directly with '-s', or
    just the unpolarized image with '-u'.
    Other plotting parameters are controlled by options below:
    """

    for fname in fnames:
        # Only plot .h5 files if we've listed e.g. folders, etc
        if fname[-3:] != ".h5": continue
        print("plotting {0:s}".format(fname))

        image = read_image(fname)

        # create plots
        if simple:
            fig = plot_pol(image, figsize=(fig_x, fig_y))
        elif unpol:
            fig = plot_unpol(image, figsize=(fig_x, fig_y))
        else:
            fig = plot_stokes_square(image, figsize=(fig_x, fig_y))
        
        if overlay:
            plot_evpa_ticks(fig.get_axes(0), image, n_evpa=n_ticks)
        # TODO whatever options
        
        fig.savefig(fname.replace(".h5",".png"))



if __name__ == "__main__":
    plot()