#!/usr/bin/env python3
"""
  Compare two images and quantify their differences
"""

import sys
import numpy as np
import click

from imtools.io import read_image
from imtools.stats import mse, mses, ssim, ssims, dssim, dssims

# Suppress runtime math warnings -- images have some zeros and that's okay
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# TODO blur option, ssim & dssim stats, image-integrated, etc.
@click.command()
@click.argument('imagepath1', nargs=1)
@click.argument('imagepath2', nargs=1)
@click.option('--divide-by-scale', 'rescale_2', is_flag=True, help="Rescale the second image from Jy/px to intensity")
def compare(imagepath1, imagepath2, rescale_2):

    print("Comparing {} vs {}".format(imagepath1, imagepath2))

    # load
    image1 = read_image(imagepath1)
    image2 = read_image(imagepath2)
    unpol1 = image1.unpol
    I1 = image1.I
    Q1 = image1.Q
    U1 = image1.U
    V1 = image1.V

    if rescale_2:
        image2 /= image1.scale

    unpol2 = image2.unpol
    I2 = image2.I
    Q2 = image2.Q
    U2 = image2.U
    V2 = image2.V

    # command line output
    print("Images\t\tMSE\t\tDSSIM\t\tabs flux diff\trel flux diff")
    if unpol1 is not None and unpol2 is not None:
        mseu = mse(image1.unpol, image2.unpol); dssimu = dssim(unpol1, unpol2);
        diffu = unpol2.sum() - unpol1.sum(); rdiffu = (unpol2.sum() - unpol1.sum())/unpol1.sum()
        print("Unpolarized:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mseu, dssimu, diffu, rdiffu))
    mse_all = mses(image1, image2)
    dssim_all = dssims(image1, image2)
    mseI = mse(I1, I2); dssimI = dssim(I1, I2); diffI = I2.sum() - I1.sum(); rdiffI = (I2.sum() - I1.sum())/I1.sum()
    print("Stokes I:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mseI, dssimI, diffI, rdiffI))
    print("Stokes I:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mse_all[0], dssim_all[0], diffI, rdiffI))
    mseQ = mse(Q1, Q2); dssimQ = dssim(Q1, Q2); diffQ = Q2.sum() - Q1.sum(); rdiffQ = (Q2.sum() - Q1.sum())/Q1.sum()
    print("Stokes Q:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mseQ, dssimQ, diffQ, rdiffQ))
    mseU = mse(U1, U2); dssimU = dssim(U1, U2); diffU = U2.sum() - U1.sum(); rdiffU = (U2.sum() - U1.sum())/U1.sum()
    print("Stokes U:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mseU, dssimU, diffU, rdiffU))
    mseV = mse(V1, V2); dssimV = dssim(V1, V2); diffV = V2.sum() - V1.sum(); rdiffV = (V2.sum() - V1.sum())/V1.sum()
    print("Stokes V:\t{:.7g}\t{:.7g}\t{:.7g}\t{:.7g}".format(mseV, dssimV, diffV, rdiffV))

    lp1 = 100.*np.sqrt(Q1.sum()**2+U1.sum()**2)/I1.sum()
    lp2 = 100.*np.sqrt(Q2.sum()**2+U2.sum()**2)/I2.sum()
    print("LP [%]: {:g} {:g} diff: {:g}".format(lp1, lp2, lp2 - lp1))
    cp1 = 100.*V1.sum()/I1.sum()
    cp2 = 100.*V2.sum()/I2.sum()
    print("CP [%]: {:g} {:g} diff: {:g}".format(cp1, cp2, cp2 - cp1))
    evpatot1 = 180./3.14159*0.5*np.arctan2(U1.sum(),Q1.sum())
    evpatot2 = 180./3.14159*0.5*np.arctan2(U2.sum(),Q2.sum())
    print("EVPA [deg]: {:g} {:g} diff: {:g}".format(evpatot1, evpatot2, evpatot2 - evpatot1))

    # Return code for automated testing.  Adjust stringency to taste
    if mseI > 0.005 or mseQ > 0.01 or mseU > 0.01 or mseV > 0.03:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    compare()
