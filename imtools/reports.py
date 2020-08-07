# Useful figures to generate of images

import numpy as np
import matplotlib.pyplot as plt

from imtools.library import ImageSet
from imtools.io import read_image
from imtools.plots import *
import imtools.stats as stats
from imtools.grey_plot import plot_I_greyscale

def generate_comparison_unpol(axes, image1, image2, name1, name2, include_diff=True, scale=False, vmax=1.0e-3):
    if scale:
        scalefac = np.mean(image1.I)/np.mean(image2.I)
    else:
        scalefac = 1
    
    params = {'cmap':'jet', 'clabel':True, 'vmin':0, 'vmax':vmax}
    plot_var(axes[0], image1.I * scalefac, image1, **params)
    axes[0].set_title(name1)
    plot_var(axes[1], image2.I, image2, **params)
    axes[1].set_title(name2)

    if include_diff:
        plot_var(axes[2], image1.I*scalefac - image2.I, image1, cmap='RdBu_r', clabel=True)
        #plot_image(axes[2], np.abs(dcode1*scalefac - dcode2), cmap='jet', clabel=True)
        axes[2].set_title("Difference")

        plot_var(axes[3], np.clip((image1.I*scalefac - image2.I)/image2.I,-1,1), image1, cmap='RdBu_r', clabel=True)
        axes[3].set_title("Relative Difference")

    print("Ftot {}: {}".format(name1, image1.flux))
    print("Ftot {}: {}".format(name2, image2.flux))
    print("{} - {} MSE in I is {}".format(name1, name2, stats.mse(image1.I*scalefac, image2.I)))

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
        print("Flux [Jy]:    {0:g} ({1:g} unpol)".format(image.Itot()*image.scale, np.sum(image.unpol)*image.scale))
        print("I,Q,U,V [Jy]: {0:g} {1:g} {2:g} {3:g}".format(image.Itot()*image.scale,
                                                            image.Qtot()*image.scale,
                                                            image.Utot()*image.scale,
                                                            image.Vtot()*image.scale))
        print("LP [%]:       {0:g}".format(100.*image.lpfrac_int()))
        print("CP [%]:       {0:g}".format(100.*image.cpfrac_int()))
        print("EVPA [deg]:   {0:g}".format(image.evpa_int()))

    # saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outfname)

def generate_collage(library, outfname, nimg, greyscale="none", rotated=False, show_spin=False, mad_spins=ImageSet.canon_spins,
                    sane_spins=ImageSet.canon_spins, rhighs=ImageSet.canon_rhighs, figsize=(16,9), zoom=2, blur=0, average=False,
                    title="", evpa=True, n_evpa=20, duplicate=False, scaled=False, verbose=False):
    """Generate a figure with a collage of all models at a particular snapshot, or averaged
    @param library: 
    @param outfname: Save to this output, respects file endings
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    width = len(mad_spins) + len(sane_spins)
    wid_mad = len(mad_spins)

    if duplicate:
        rhighs = rhighs + rhighs
    height = len(rhighs)

    for nflux, flux in enumerate(library.canon_fluxes):
        for nspin, spin in enumerate( (mad_spins, sane_spins)[nflux] ):
            for nrhigh, rhigh in enumerate(rhighs):

                # Get the image. Note averaging is only to demo how silly it is
                if average:
                    image = library.average_image(flux, spin, rhigh)
                else:
                    image = library.get_image(flux, spin, rhigh, nimg, verbose=verbose)
                    if image is None:
                        continue

                # Blur, unless we're building the top half of a contrast "duplicate" figure
                if blur > 0 and not (duplicate and nrhigh < len(rhighs) / 2):
                    image = image.blurred(blur)

                # Rotate by 90 degrees so that bright spot ends up on (or usually somewhere near) the bottom
                if rotated:
                    # Positive spin -> 163 degrees -> forward jet is South so rot90 CCW
                    # Negative spin -> 17 degrees -> forward jet is north, CW
                    if float(spin) >= 0:
                        image.rot90(-1)
                        arrow_lim = (-8, 0)
                    else:
                        image.rot90(1)
                        arrow_lim = (8, 0)

                # total intensity
                ax = plt.subplot(height, width, width * nrhigh + wid_mad * nflux + nspin + 1)

                if greyscale == "full" or (greyscale == "half" and not (duplicate and nrhigh < len(rhighs) / 2)):
                    _, qv = plot_I_greyscale(ax, image)
                    #plot_evpa_rainbow(ax, image, zoom=zoom, clean=True)
                else:
                    plot_I(ax, image, zoom=zoom, clean=True)
                    if evpa:
                        plot_evpa_ticks(ax, image, only_ring=True, n_evpa=n_evpa, scaled=scaled)
                
                if show_spin and float(spin) != 0.0:
                    ax.arrow(0, 0, *arrow_lim, color='black', head_width=5)

                # Label the border plots
                if nrhigh == 0:
                    ax.set_title(flux + ", a = " + spin, fontsize=10)
                if nflux == 0 and nspin == 0:
                    ax.set_ylabel(r"$R_{\mathrm{high}} = $" + rhigh)

    # TODO this needs to be more adaptable
    if greyscale == "half" or greyscale == "full":
        plt.subplots_adjust(right=0.90, hspace=0, wspace=0) # TODO make sure there's no gaps despite aspect='equal'
        cax = fig.add_axes([0.91, 0.13, 0.01, 0.745]) # Left, Bottom, 
        # Just use the last quiver, they're standard if we're doing this
        cbar = plt.colorbar(qv, cax=cax)
        cbar.set_label(r'Fractional Polarization')
        #cbar.ax.tick_params(labelsize=30)
    else:
        plt.subplots_adjust(bottom=0.09, top=0.91, hspace=0, wspace=0) # TODO make sure there's no gaps despite aspect='equal'
    plt.savefig(outfname, dpi=400)

def generate_lcs(library, outfname, fn, label, mad_spins=ImageSet.canon_spins, sane_spins=ImageSet.canon_spins, figsize=(15.5,10)):
    plt.figure(figsize=figsize)
    plt.suptitle( label + " vs t" )
    height = len(library.canon_rhighs)
    min_t = 1e7
    max_t = 0

    for nflux, flux in enumerate(library.canon_fluxes):
        for nrhigh, rhigh in enumerate(library.canon_rhighs):
            # TODO gridspec rather than re-visiting
            plt.subplot(height, 2, 2 * nrhigh + nflux + 1)

            for spin in (mad_spins, sane_spins)[nflux]:
                # Get values of the fn for all the images
                t, lc = library.run_lc(flux, spin, rhigh, fn)

                plt.plot(t, lc, label="a = " + spin)

                if t[0] < min_t:
                    min_t = t[0]
                if t[-1] > max_t:
                    max_t = t[-1]
            
            # Parameters common to all spins
            plt.tick_params(axis='both', which='both', bottom=False, left=False, top=False, right=False,
                            labelbottom=False, labelleft=False)

            plt.xlim(min_t-10, max_t+10)
            #plt.ylim([lim[0] * 1.1, lim[1] * 1.1])
            plt.grid()

            if nrhigh == 0:
                plt.title(flux)
                plt.legend(loc='upper right')
            elif nrhigh == len(library.canon_rhighs) - 1:
                plt.xlabel("Time in M")
                plt.tick_params(axis='x', bottom=True, labelbottom=True)
            if nflux == 0:
                plt.ylabel(r"$R_{\mathrm{high}} = $" + rhigh)
                plt.tick_params(axis='y', left=True, labelleft=True)
            else:
                plt.tick_params(axis='y', right=True, labelright=True)


    plt.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.05, hspace=0, wspace=0.06)
    plt.savefig(outfname, dpi=200)