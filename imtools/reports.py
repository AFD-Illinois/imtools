"""
 File: reports.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import matplotlib.pyplot as plt

from imtools.library import ImageSet
from imtools.io import read_image
from imtools.plots import *
import imtools.stats as stats
from imtools.grey_plot import plot_I_greyscale

"""
Reports take an image(s) or image set(s) and return a figure object for any further
modification or saving. (You can modify the individual axes by running fig.get_axes())

Some reports also optionally print statistics to console.
"""

def compare_unpol(image1, image2, scale_image=False, same_colorscale=False, print_stats=True, figsize=(20,5), **kwargs):
    if scale_image:
        scalefac = np.mean(image1.I)/np.mean(image2.I)
        clabel1 = "Scaled Intensity"
    else:
        scalefac = 1
        clabel1 = "Intensity"
    
    fig, ax = plt.subplots(1, 4, figsize=figsize)

    if same_colorscale:
        vmax = max(np.max(image1.I * scalefac), np.max(image2.I))
    else:
        vmax = None

    plot_var(ax[0], image1.I * scalefac, image1, cmap='jet', vmax=vmax, clabel=clabel1, **kwargs)
    ax[0].set_title(image1.name)

    plot_var(ax[1], image2.I, image2, cmap='jet', vmax=vmax, clabel="Intensity", **kwargs)
    ax[1].set_title(image2.name)

    plot_var(ax[2], image1.I*scalefac - image2.I, image1, cmap='bwr', symmetric=True, **kwargs)
    ax[2].set_title("Difference")

    plot_var(ax[3], np.clip((image1.I*scalefac - image2.I)/image2.I,-1,1), image1, cmap='bwr', symmetric=True)
    ax[3].set_title("Relative Difference")

    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.05, top=0.95, wspace=0.5)

    if print_stats:
        print("Ftot {}: {}".format(image1.name, image1.flux()))
        print("Ftot {}: {}".format(image2.name, image2.flux()))
        print("{} - {} MSE in I is {}".format(image1.name, image2.name, stats.mse(image1.I*scalefac, image2.I)))

    return fig

def compare(image1, image2, figsize=(12,6)):
    fig, ax = plt.subplots(2, 4, figsize=figsize)

    diff = image2 - image1
    fig.suptitle("Difference in raw Stokes {} vs {}".format(image1.name, image2.name))
    plot_all_stokes(ax[0,:], diff, relative=True)
    plot_all_stokes(ax[1,:], image1.rel_diff(image2, clip=(-1,1)), relative=True)

    return fig



def plot_pol(image, figsize=(8,8), print_stats=True, scaled=True):
    """Mimics the plot_pol.py script in ipole/scripts"""
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # Total intensity
    plot_I(ax[0,0], image, xlabel=False)
    # Quiver on intensity
    plot_evpa_ticks(ax[0,0], image, n_evpa=30, scaled=scaled)

    # Linear polarization fraction
    plot_lpfrac(ax[0,1], image, xlabel=False, ylabel=False)

    # evpa
    plot_evpa_rainbow(ax[1,0], image)

    # circular polarization fraction
    plot_cpfrac(ax[1,1], image, ylabel=False)

    if print_stats:
        # print image-average quantities to command line
        print("Flux [Jy]:    {0:g} ({1:g} unpol)".format(image.flux(), image.flux_unpol()))
        print("I,Q,U,V [Jy]: {0:g} {1:g} {2:g} {3:g}".format(image.Itot(), image.Qtot(),
                                                            image.Utot(), image.Vtot()))
        print("LP [%]:       {0:g}".format(100.*image.lpfrac_int()))
        print("CP [%]:       {0:g}".format(100.*image.cpfrac_int()))
        print("EVPA [deg]:   {0:g}".format(image.evpa_int()))

    return fig

def plot_unpol(image, figsize=(8,8), print_stats=True, scaled=True, **kwargs):
    """Mimics the plot.py script in ipole/scripts"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Total intensity
    plot_I(ax, image, **kwargs)

    if print_stats:
        # Print just total flux
        print("Flux [Jy]:    {0:g}".format(image.flux_unpol()))

    return fig

def plot_stokes_square(image, figsize=(8,10), **kwargs):
    """Plot a 4-pane image showing each Stokes parameter.
    Defaults to large 
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    plot_all_stokes(ax.flatten(), image, layout="square", **kwargs)
    #plt.subplots_adjust(hspace=0, wspace=0)
    #plt.tight_layout()
    return fig


def plot_stokes_rows(images, figsize=(12,12), units="Jy", n_stokes=4, consistent_scale=True, key_image=0, **kwargs):
    """Plot a series of horizontal 4-pane Stokes images, for comparing each parameter down columns
    Places both colorbars on the right side
    """
    fig, ax = plt.subplots(len(images), n_stokes, figsize=figsize)

    if consistent_scale:
        # Set consistent vmax.  Doesn't matter which image we key from
        key_im = images[key_image]
        vmax = [np.max(np.abs(key_im.I * key_im.scale_flux(units))),
                np.max(np.abs(key_im.Q * key_im.scale_flux(units))),
                np.max(np.abs(key_im.U * key_im.scale_flux(units))),
                np.max(np.abs(key_im.V * key_im.scale_flux(units)))]
    else:
        vmax = None

    if 'polar' in kwargs and kwargs['polar']:
        titles = ["Stokes I", "LP Emission", "EVPA", "Stokes V"]
    else:
        titles = ["Stokes I", "Stokes Q", "Stokes U", "Stokes V"]

    if len(images) == 1:
        # Plot each code's Stokes parameters
        meshes = plot_all_stokes(ax, images[0], clean=True, vmax=vmax, n_stokes=n_stokes, **kwargs)
        ax[0].set_ylabel(images[0].get_name())
        # Finalize figure: names
        for i in range(n_stokes):
            ax[i].set_title(titles[i])
    else:
        for i,image in enumerate(images):
            # Plot each code's Stokes parameters
            meshes = plot_all_stokes(ax[i,:], image, clean=True, vmax=vmax, n_stokes=n_stokes, **kwargs)
            ax[i,0].set_ylabel(image.get_name())

        # Finalize figure: names
        for i in range(n_stokes):
            ax[0,i].set_title(titles[i])


    # colorbars
    caxes = subplots_adjust_square(fig, n_stokes, len(images), col_cbars=True)
    for i in range(n_stokes):
        cbar = plt.colorbar(meshes[i], cax=fig.add_axes(caxes[i]), orientation='horizontal')
        cbar.formatter.set_powerlimits((0, 0))

    return fig

def get_snapshots_at(nimg, spins=None, mad_spins=ImageSet.canon_spins, sane_spins=ImageSet.canon_spins,
                     rhighs=ImageSet.canon_rhighs, rotated=True):
    return

def collage(library, nimg, greyscale="none", evpa_rainbows=False, rotated=False, show_spin=False, mad_spins=ImageSet.canon_spins,
                    sane_spins=ImageSet.canon_spins, rhighs=ImageSet.canon_rhighs, figsize=(16,9), zoom=2, blur=0, vmax=None, average=False,
                    title="", evpa=True, n_evpa=20, duplicate=False, scaled=False, compress_scale=False, verbose=False):
    """Generate a figure with a collage of all models at a particular snapshot, or averaged
    @param library:
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
                    my_n_evpa = n_evpa
                    did_blur = True
                else:
                    my_n_evpa = n_evpa + 5
                    did_blur = False

                # Rotate by 90 degrees so that bright spot ends up on (or somewhere near...) the South
                if rotated:
                    # Positive spin -> 163 degrees -> forward jet is North so rot90 CW
                    # Negative spin -> 17 degrees -> forward jet is South, CCW
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
                elif evpa_rainbows:
                    plot_evpa_rainbow(ax, image, clean=True)
                else:
                    plot_I(ax, image, zoom=zoom, clean=True, vmax=vmax)
                    if evpa:
                        plot_evpa_ticks(ax, image, emission_cutoff=(1.0 + (not did_blur)*1.6), n_evpa=my_n_evpa, scaled=scaled, compress_scale=compress_scale)
                
                
                if show_spin and float(spin) != 0.0:
                    ax.arrow(0, 0, *arrow_lim, color='black', head_width=5)

                # Label the border plots
                if nrhigh == 0:
                    ax.set_title(flux + ", a = " + spin, fontsize=10)
                if nflux == 0 and nspin == 0:
                    ax.set_ylabel(r"$R_{\mathrm{high}} = $" + rhigh)

    if greyscale == "half" or greyscale == "full":
        cax = subplots_adjust_square(fig, width, height, global_cbar=True)
        # Just use the last quiver, they're standard if we're doing this
        cbar = plt.colorbar(qv, cax=cax)
        cbar.set_label(r'Fractional Polarization')
    else:
        subplots_adjust_square(fig, width, height)

    return fig

def lightcurves(library, fn, label, mad_spins=ImageSet.canon_spins, sane_spins=ImageSet.canon_spins,
                rhighs=ImageSet.canon_rhighs, figsize=(15.5,10)):
    height = len(rhighs)
    min_t = 1e7
    max_t = 0

    fig, ax = plt.subplots((height, 2), figsize=figsize)
    plt.suptitle(label + " vs t")

    for nflux, flux in enumerate(ImageSet.canon_fluxes):
        for nrhigh, rhigh in enumerate(rhighs):
            axis = ax[nrhigh, nflux]

            for spin in (mad_spins, sane_spins)[nflux]:
                # Get values of the fn for all the images
                t, lc = library.run_lc(flux, spin, rhigh, fn)

                axis.plot(t, lc, label="a = " + spin)

                if t[0] < min_t:
                    min_t = t[0]
                if t[-1] > max_t:
                    max_t = t[-1]
            
            # Parameters common to all spins
            axis.tick_params(axis='both', which='both', bottom=False, left=False, top=False, right=False,
                            labelbottom=False, labelleft=False)

            axis.set_xlim(min_t-10, max_t+10)
            #plt.ylim([lim[0] * 1.1, lim[1] * 1.1])
            axis.set_grid(True)

            if nrhigh == 0:
                plt.title(flux)
                plt.legend(loc='upper right')
            elif nrhigh == len(rhighs) - 1:
                plt.xlabel("Time in M")
                plt.tick_params(axis='x', bottom=True, labelbottom=True)
            if nflux == 0:
                plt.ylabel(r"$R_{\mathrm{high}} = $" + rhigh)
                plt.tick_params(axis='y', left=True, labelleft=True)
            else:
                plt.tick_params(axis='y', right=True, labelright=True)

    plt.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.05, hspace=0, wspace=0.06)
    return fig
