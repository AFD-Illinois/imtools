#!/usr/bin/env python3

# Usage: plot_pol.py [-s] file.h5 [other_file.h5 ...]
# Mimic the classic plot_pol.py in ipole
# This is also available as a function, reports.plot_pol,
# to call from your own code

import sys

from imtools.io import read_image
from imtools.figures import plot_pol, plot_stokes_square

if __name__ == "__main__":

    if sys.argv[1] == "-s":
        for fname in sys.argv[2:]:
            if fname[-3:] != ".h5": continue
            print("plotting {0:s}".format(fname))

            image = read_image(fname)

            # create plots
            plot_pol(image).savefig(fname.replace(".h5",".png"))

    for fname in sys.argv[1:]:
        if fname[-3:] != ".h5": continue
        print("plotting {0:s}".format(fname))

        image = read_image(fname)

        # create plots
        plot_stokes_square(image).savefig(fname.replace(".h5",".png"))

