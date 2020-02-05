# Useful figures to generate of images

import matplotlib.pyplot as plt

from imtools.library import ImageSet
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

def generate_plot_pol(image, outfname, figsize=(8,8), print_stats=True, scaled=True):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    # Total intensity
    plot_I(ax1, image, xlabel=False)
    # Quiver on intensity
    plot_evpa_ticks(ax1, image, n_evpa=30, scaled=scaled)

    # Linear polarization fraction
    plot_lpfrac(ax2, image, xlabel=False, ylabel=False)

    # evpa
    plot_evpa_rainbow(ax3, image)

    # circular polarization fraction
    plot_cpfrac(ax4, image, ylabel=False)

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

def generate_collage(library, outfname, nimg, ignore_time=True, mad_spins=ImageSet.canon_spins,
                    sane_spins=ImageSet.canon_spins, figsize=(15.7,10), zoom=2, blur=0, average=False):
    """Generate a figure with a collage of all models at a particular snapshot, or averaged
    """
    plt.figure(figsize=figsize)
    plt.suptitle("Polarization Snapshots")
    width = len(mad_spins) + len(sane_spins)
    wid_mad = len(mad_spins)
    height = len(library.canon_rhighs)

    for nflux, flux in enumerate(library.canon_fluxes):
        for nspin, spin in enumerate( (mad_spins, sane_spins)[nflux] ):
            for nrhigh, rhigh in enumerate(library.canon_rhighs):

                # Get the image. Note averaging is only to demo how silly it is
                if average:
                    image = library.average_image(flux,spin,rhigh)
                else:
                    if ignore_time:
                        imgname = library.get_all_fnames(flux,spin,rhigh)[nimg]
                        image = read_image(imgname)
                    else:
                        image = library.get_image(flux, spin, rhigh, nimg)

                    if image is None:
                        continue

                # Blur
                if blur > 0:
                    image = image.blurred(blur)

                # total intensity
                ax = plt.subplot(height, width, width * nrhigh + wid_mad * nflux + nspin + 1)

                plot_I(ax, image, zoom=zoom, clean=True)
                plot_evpa_ticks(ax, image, only_ring=True)

                # Label the border plots
                if nrhigh == 0:
                    ax.set_title(flux + ", a = " + spin)
                if nflux == 0 and nspin == 0:
                    ax.set_ylabel("Rhigh = " + rhigh)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(outfname, dpi=200)

def generate_lcs(library, outfname, figsize=(15.5,10)):
    pass