# Useful figures to generate of images

import matplotlib.pyplot as plt

from imtools.io import read_image
from imtools.plots import *

def generate_comparison(ax, code_dict, code1name, code2name, include_diff=True, scale=False, vmax=1.0e-3):
    dcode1 = code_dict[code1name][:,:,0]
    dcode2 = code_dict[code2name][:,:,0]
    if scale:
        scalefac = np.mean(dcode2)/np.mean(dcode1)
    else:
        scalefac = 1
    
    params = {'cmap':'jet', 'clabel':True, 'vmin':0, 'vmax':vmax}
    plot_image(ax[0], dcode1*scalefac, **params)
    ax[0].set_title(code1name)
    plot_image(ax[1], dcode2, **params)
    ax[1].set_title(code2name)

    if include_diff:
        plot_image(ax[2], dcode1*scalefac - dcode2, cmap='RdBu_r', clabel=True)
        #plot_image(ax[2], np.abs(dcode1*scalefac - dcode2), cmap='jet', clabel=True)
        ax[2].set_title("Difference")

        plot_image(ax[3], np.clip((dcode1*scalefac - dcode2)/dcode2,-1,1), cmap='RdBu_r', clabel=True)
        ax[3].set_title("Relative Difference")

    print("Ftot {}: {}".format(code1name, np.sum(dcode1)))
    print("Ftot {}: {}".format(code2name, np.sum(dcode2)))
    print("{} - {} MSE in I is {}".format(code1name, code2name, mse(dcode1*scalefac, dcode2)))

def generate_plot_pol(image, outfname, figsize=(8,8), print_stats=True):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    # Total intensity
    plot_I(ax1, image)
    # Quiver on intensity
    plot_evpa_ticks(ax1, image)

    # Linear polarization fraction
    plot_lpfrac(ax2, image)

    # circular polarization fraction
    plot_cpfrac(ax4, image)

    # evpa
    plot_evpa_rainbow(ax3, image)

    if print_stats:
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
    plt.savefig(outfname)

def generate_collage(library, outfname, nimg, mad_spins=('-0.94', '-0.5', '0', '0.5', '0.94'),
                    sane_spins=('-0.94', '-0.5', '0', '0.5', '0.94'), figsize=(20,10), blur=0):
    plt.figure(figsize=figsize)
    plt.suptitle("Polarization Snapshots")
    width = len(mad_spins) + len(sane_spins)
    wid_mad = len(mad_spins)

    for nflux, flux in enumerate(['MAD', 'SANE']):
        for nspin, spin in enumerate( (mad_spins, sane_spins)[nflux] ):
            for nrhigh, rhigh in enumerate(['1', '10', '20', '40', '80', '160']):
                if nrhigh == 0:
                    plt.title(flux + ", a = " + spin)
                if nflux == 0 and nspin == 0:
                    plt.ylabel("Rhigh = " + rhigh)

                imgname = library.get_fname(flux, spin, rhigh, nimg)
                if imgname is None:
                    continue
                image = read_image(imgname)
                if image is None:
                    continue

                # Blur
                if blur > 0:
                    image = image.blurred(blur)

                # total intensity
                ax = plt.subplot(6, width, width * nrhigh + wid_mad * nflux + nspin + 1)
                plot_I(ax, image, zoom=2, clean=True)
                plot_evpa_ticks(ax, image, only_ring=True)

    plt.subplots_adjust(top=0.90, left=0.05, right=0.95, bottom=0.05, hspace=0, wspace=0)
    plt.savefig(outfname, dpi=200)