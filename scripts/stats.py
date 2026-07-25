#!/usr/bin/env python3

"""Simple plotting script that explicitly apes ipole's plot.py and plot_pol.py
"""

import sys
import click

from imtools.io import read_image
from imtools.figures import *

@click.command()
@click.argument('fnames', nargs=-1, type=click.Path(exists=True))
@click.option('-b', '--blur', default=20., help="Blur kernel for image-averaged quantities, muas")
@click.option('-u', '--unpol', is_flag=True, help="only look at unpolarized image")
def plot(fnames, blur, unpol):
    """USAGE: stats.py [options] fname [fname fname]

    Print basic image-integrated statistics.
    """

    for fname in fnames:
        # Only plot .h5 files if we've listed e.g. folders, etc
        if fname[-3:] != ".h5": continue
        print("plotting {0:s}".format(fname))

        image = read_image(fname, only_unpolarized=unpol)
        print(f"{fname}:")
        print(f"Flux (unpol transport): {image.flux_unpol()}")
        if not unpol:
            print(f"Flux (pol transport): {image.flux()}")

            print(f"Integrated LP fraction: {image.lpfrac_int()}")
            print(f"Integrated CP fraction: {image.cpfrac_int()}")
            print(f"Integrated EVPA: {image.evpa_int()}")

            print(f"Resolved LP fraction: {image.lpfrac_av(blur=blur)}")
            # TODO when eht-imaging is compatible again
            #print(f"PWP beta2: {image.beta(blur=blur)}")

        print(f"Avg optical depth: {image.tau_av()}")
        if not unpol:
            print(f"Avg Faraday depth: {image.tauF_av()}")



if __name__ == "__main__":
    plot()