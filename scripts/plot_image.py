# Mimic the classic plot_pol.py in ipole
# This is now an automatic report, since it's useful

from imtools.io import read_image
from imtools.reports import generate_plot_pol

import sys

if __name__ == "__main__":

  for fname in sys.argv[1:]:

    if fname[-3:] != ".h5": continue
    print("plotting {0:s}".format(fname))

    image = read_image(sys.argv[1])

    # create plots
    generate_plot_pol(image, fname.replace(".h5",".png")) #TODO current folder instead of original?

