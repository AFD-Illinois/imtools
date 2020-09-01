# plots.py

import numpy as np

"""
Different plots that can be made of images, or combinations of images.
These are the functions which use an existing axis, for more complete
functions which return a whole figure see reports.py
"""

def plot_var(ax, var, image, fov_units="muas", xlabel=True, ylabel=True, add_cbar=True, clabel=None, zoom=1, clean=False, **kwargs):
    """General function for plotting a set of pixels, given some options about how"""
    extent_og = image.extent(fov_units)

    X, Y = np.meshgrid(np.linspace(extent_og[0], extent_og[1], image.nx+1),
                        np.linspace(extent_og[2], extent_og[3], image.ny+1))
    mesh = ax.pcolormesh(X, Y, var, **kwargs)

    # Colorbar
    if add_cbar and not clean:
        cbar = _colorbar(mesh)
        if clabel is not None:
            cbar.set_label(clabel)

        if ('vmax' not in kwargs or kwargs['vmax'] is None) or kwargs['vmax'] < 1:
            cbar.formatter.set_powerlimits((0, 0))

        cbar.update_ticks()

    # Cosmetics
    ax.grid(False)
    if clean:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
    else:
        ax.set_yticks(ax.get_xticks())

        if xlabel:
            if fov_units == "muas":
                ax.set_xlabel(r"x ($\mu as$)")
            else:
                ax.set_xlabel("x ({})".format(fov_units))
        else:
            ax.set_xticklabels([])

        if ylabel:
            if fov_units == "muas":
                ax.set_ylabel(r"y ($\mu as$)")
            else:
                ax.set_ylabel("y ({})".format(fov_units))
        else:
            ax.set_yticklabels([])

    # Set window and aspect last to ensure they survive shenanigans
    ax.set_aspect('equal')
    window = [extent_og[0]/zoom, extent_og[1]/zoom, extent_og[2]/zoom, extent_og[3]/zoom]
    ax.axis(window)
    #print("Window:", window)

    return mesh

def plot_I(ax, image, units="Jy", cmap='afmhot', add_title=True, clean=False, tag="", **kwargs):
    """Plot Stokes I.  Separate implementation for afmhot colors and black background"""
    mesh = plot_var(ax, image.I * image.scale_flux(units), image, clean=clean, cmap=cmap, **kwargs)

    if add_title and not clean:
        if tag is not None:
            ax.set_title("{} Stokes I".format(tag))
        else:
            ax.set_title("{} Stokes I".format(image.get_name()))

    return mesh

def plot_one_stokes(ax, image, num, units="Jy", cmap='RdBu_r', vmax=None, add_title=True, clean=False, tag=None, **kwargs):
    """Plot the Stokes parameter indicated by num, where I==0, Q==1, etc."""
    var = image.get_stokes(num)
    # Take vmax if speccd, otherwise use a symmetric scale around the largest abs()
    if vmax is None:
        max_abs = max(np.abs(np.max(var)), np.abs(np.min(var)))
        max_abs = min(max_abs, 1e3) # Clip to stay remotely reasonable
    else:
        max_abs = vmax

    mesh = plot_var(ax, image.get_stokes(num) * image.scale_flux(units), image, clean=clean, cmap=cmap,
                    vmin=-max_abs, vmax=max_abs, **kwargs)

    if add_title and not clean:
        if tag is not None:
            ax.set_title("{} Stokes {}".format(tag, ["I", "Q", "U", "V"][num]))
        else:
            ax.set_title("{} Stokes {}".format(image.get_name(), ["I", "Q", "U", "V"][num]))

    return mesh

def plot_Q(ax, image, **kwargs):
    """Plot Stokes Q"""
    return plot_one_stokes(ax, image, 1, **kwargs)
def plot_U(ax, image, **kwargs):
    """Plot Stokes U"""
    return plot_one_stokes(ax, image, 2, **kwargs)
def plot_V(ax, image, **kwargs):
    """Plot Stokes V"""
    return plot_one_stokes(ax, image, 3, **kwargs)

def plot_all_stokes(axes, image, relative=False, vmax=None, units="Jy", n_stokes=4, layout="none", **kwargs):
    """Plot all raw Stokes parameters on a set of 4 axes.
    If vmax is given as a list, use the respective elements as vmax for I,Q,U,V
    """
    ax = axes.flatten()

    try:
        vmax[0]
    except TypeError:
        vmax = [vmax, vmax, vmax, vmax]

    if layout == "line":
        xlabel_flags = [True, True, True, True]
        ylabel_flags = [True, False, False, False]
    elif layout == "square":
        xlabel_flags = [False, False, True, True]
        ylabel_flags = [True, False, True, False]
    else:
        xlabel_flags = [True, True, True, True]
        ylabel_flags = [True, True, True, True]

    meshes = []
    for i in range(n_stokes):
        if units == "cgs":
            clabel = "cgs"
        elif "Jy" in units:
            clabel = "Jy/px"
        else:
            clabel = None
        if i == 0 and not relative:
            meshes.append( plot_I(ax[i], image, clabel=clabel, vmax=vmax[i],
                                  xlabel=xlabel_flags[i], ylabel=ylabel_flags[i], **kwargs) )
        else:
            meshes.append( plot_one_stokes(ax[i], image, i, clabel=clabel, vmax=vmax[i],
                                           xlabel=xlabel_flags[i], ylabel=ylabel_flags[i], **kwargs) )

    return meshes

def plot_lpfrac(ax, image, **kwargs):
    """Plot the percentage of emission which is linearly polarized.
    """
    ax.set_facecolor('black')
    plot_var(ax, 100*image.lpfrac(mask_zero=True), image, cmap='jet', vmin=0., vmax=100., **kwargs)
    ax.set_title("LP [%]")

def plot_cpfrac(ax, image, **kwargs):
    """Plot the percentage of emission which is circularly polarized.  Signed.
    (as in ipole's bundled plot_pol.py)
    """
    cpfrac = image.cpfrac(mask_zero=True)
    vext = max(np.abs(np.nanmin(cpfrac)),np.abs(np.nanmax(cpfrac)))
    vext = max(vext, 1.)
    if np.isnan(vext): vext = 10.

    ax.set_facecolor('black')
    plot_var(ax, 100*image.cpfrac(mask_zero=True), image, cmap='jet', vmin=0., vmax=100., **kwargs)
    ax.set_title("CP [%]")

def plot_evpa_rainbow(ax, image, evpa_conv="EofN", clean=False, **kwargs):
    """EVPA rainbow plot -- color pixels by angle without regard to polarized emission fraction
    (as in ipole's bundled plot_pol.py)
    """
    ax.set_facecolor('black')
    plot_var(ax, image.evpa(evpa_conv, mask_zero=True), image, cmap='hsv', vmin=-90., vmax=90., clean=clean, **kwargs)
    if not clean:
        ax.set_title("EVPA [deg]")

def plot_evpa_ticks(ax, image, n_evpa=20, scaled=False, emission_cutoff=0.0, fov_units="muas", 
                    custom_wid=1.0, prominent=False, compress_scale=False, **kwargs):
    """Superimpose EVPA as a quiver plot, either scaled by the polarization fraction or not.
    """
    im_evpa = image.downsampled(image.nx // n_evpa)
    evpa = np.pi*im_evpa.evpa(evpa_conv="NofW")/180.
    if scaled:
        # Scaled to polarization fraction
        amp = np.sqrt(im_evpa.Q ** 2 + im_evpa.U ** 2)
        if compress_scale:
            scal = np.max(amp)/compress_scale # TODO consistent scaling option for animations
            amp[amp > scal] = scal
        else:
            scal = np.max(amp)
        vx = amp * np.cos(evpa) / scal
        vy = amp * np.sin(evpa) / scal
    else:
        # Normalized (evpa only)
        vx = np.cos(evpa)
        vy = np.sin(evpa)

    # Currently cuts on I.  (option to) cut on polarized emission?
    slc = im_evpa.ring_mask(cut=emission_cutoff)

    extent_og = im_evpa.extent(fov_units)
    Xlocs, xstep = np.linspace(extent_og[0], extent_og[1], im_evpa.nx, endpoint=False, retstep=True)
    Ylocs, ystep = np.linspace(extent_og[2], extent_og[3], im_evpa.ny, endpoint=False, retstep=True)
    X, Y = np.meshgrid(Xlocs + 0.5*xstep, Ylocs + 0.5*ystep)

    # TODO set number of ticks to be constant with plot size?
    # TODO make scaled look good too

    xlim = ax.get_xlim()
    xdim = xlim[1] - xlim[0]
    if scaled:
        ax.quiver(X[slc], Y[slc], vx[slc], vy[slc], headwidth=1, headlength=.01,
                    width=.01*xdim, units='x', color='k', pivot='mid', scale=0.7*n_evpa/xdim, **kwargs)
        ax.quiver(X[slc], Y[slc], vx[slc], vy[slc], headwidth=1, headlength=.01,
                        width=.005*xdim, units='x', color='w', pivot='mid', scale=0.8*n_evpa/xdim, **kwargs)
    if prominent:
        # Big white ticks
        ax.quiver(X[slc], Y[slc], vx[slc], vy[slc],
                        headwidth=1, headlength=0,
                        width=.01*xdim, units='x', color='w', pivot='mid', scale=0.8*n_evpa/xdim, **kwargs)
    else:
        # Black ticks surrounding white
        ax.quiver(X[slc], Y[slc], vx[slc], vy[slc],
                    headwidth=1, headlength=0,
                    width=custom_wid*0.022*xdim, units='x', color='k', pivot='mid', scale=0.7*n_evpa/xdim, **kwargs)
        ax.quiver(X[slc], Y[slc], vx[slc], vy[slc],
                        headwidth=1, headlength=0,
                        width=custom_wid*0.01*xdim, units='x', color='w', pivot='mid', scale=0.8*n_evpa/xdim, **kwargs)

def subplots_adjust_square(fig, nx, ny, bottom=0.05, top=0.95, left=0.05, right=0.95, global_cbar=False, col_cbars=False):
    """Adjust the height of a grid with aspect='equal' so that there is no space between rows or columns.
    If global_cbar=True, adjust for the extra width and return a suitable set of axes
    """

    # If we should add room for a cbar...
    if global_cbar:
        cbar_wid = 0.08
        cbar_bottom = bottom + 0.03
        cbar_top = top - 0.03
        cbar_height = cbar_top - cbar_bottom
        # ...make room by adjusting "right"
        right -= cbar_wid
        # and return an axes
        cax_loc = [right+0.01, cbar_bottom, 0.1*cbar_wid, cbar_height] # Left, bottom, width, height
    elif col_cbars:
        # A colorbar at the bottom, for each of the columns. Crazy?
        cax_loc = []
        cbar_space = 0.06
        cbar_thick = 0.01
        cbar_bottom = bottom + cbar_space - cbar_thick
        plotw = (right - left) / nx
        for i in range(nx):
            cax_loc.append([left + i*plotw, cbar_bottom, plotw, cbar_thick])
        bottom = bottom + cbar_space
    else:
        cax_loc = None

    # Now adjust for square plots
    w, _ = fig.get_size_inches()
    w_of_plots = w * (right - left) # Width of plots
    h_of_plots = ny * w_of_plots/nx # Height of plots to ensure square
    h =  h_of_plots / (top - bottom) # Total height to ensure correct h_of_plots
    fig.set_size_inches(w, h)

    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right, hspace=0, wspace=0)
    return cax_loc

# Local support functions
def _colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)