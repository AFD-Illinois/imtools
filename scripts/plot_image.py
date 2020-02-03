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

    plot_lpfrac(ax2, image)

    # circular polarization fraction
    plot_cpfrac(ax4, image)

    # evpa
    plot_evpa_rainbow(ax3, image)

    # command line output
    # print("Flux [Jy]:    {0:g} {1:g}".format(I.sum()*scale, unpol.sum()*scale))
    # print("I,Q,U,V [Jy]: {0:g} {1:g} {2:g} {3:g}".format(I.sum()*scale,Q.sum()*scale,U.sum()*scale,V.sum()*scale))
    # print("LP [%]:       {0:g}".format(100.*np.sqrt(Q.sum()**2+U.sum()**2)/I.sum()))
    # print("CP [%]:       {0:g}".format(100.*V.sum()/I.sum()))
    # evpatot = 180./3.14159*0.5*np.arctan2(U.sum(),Q.sum())
    # if evpa_0 == "W":
    #   evpatot += 90. 
    #   if evpatot > 90.:
    #     evpatot -= 180
    # if EVPA_CONV == "NofW":
    #   evpatot += 90.
    #   if evpatot > 90.:
    #     evpatot -= 180
    # print("EVPA [deg]:   {0:g}".format(evpatot))

    # saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname.replace(".h5",".png"))

