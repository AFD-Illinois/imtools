# Mimic the classic plot_pol.py in ipole

from imtools.image import Image
from imtools.io import read_image
from imtools.plots import plot_I, plot_lpfrac, plot_cpfrac, plot_evpa_rainbow, plot_evpa_ticks

import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

  for fname in sys.argv[1:]:

    if fname[-3:] != ".h5": continue
    print("plotting {0:s}".format(fname))

    image = read_image(sys.argv[1])

    # create plots
    plt.close('all')
    plt.figure(figsize=(8,8))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    # total intensity
    plot_I(ax1, image)
    # quiver on intensity
    plot_evpa_ticks(ax1, image)

    # linear polarization fraction
    plot_lpfrac(ax2, image)

    # circular polarization fraction
    plot_cpfrac(ax4, image)

    # evpa
    plot_evpa_rainbow(ax3, image)

    # print image-average quantities to command line
    print("Flux [Jy]:    {0:g} ({1:g} unpol)".format(image.Itot*image.scale, np.sum(image.unpol)*image.scale))
    print("I,Q,U,V [Jy]: {0:g} {1:g} {2:g} {3:g}".format(image.Itot*image.scale,
                                                        image.Qtot*image.scale,
                                                        image.Utot*image.scale,
                                                        image.Vtot*image.scale))
    print("LP [%]:       {0:g}".format(100.*image.lpfrac_int()))
    print("CP [%]:       {0:g}".format(100.*image.cpfrac_int()))
    print("EVPA [deg]:   {0:g}".format(image.evpa_int()))

    # saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname.replace(".h5",".png")) #TODO current folder instead of original?

