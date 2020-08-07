# Different plots that can be made of images, or combinations of images
# These are the functions which use an existing axis, for functions that generate a figure see reports.py

import numpy as np

from imtools.image import Image

def plot_var(ax, var, image, fov_units="muas", xlabel=True, ylabel=True, add_cbar=True, clabel=None, zoom=1, clean=False, **kwargs):
    """General function for plotting a set of pixels, given some options about how"""
    extent_og = image.extent(fov_units)

    X, Y = np.meshgrid(np.linspace(extent_og[0], extent_og[1], image.nx+1),
                        np.linspace(extent_og[2], extent_og[3], image.ny+1))
    im = ax.pcolormesh(X, Y, var, **kwargs)

    # Window and aspect
    ax.axis([extent_og[0]/zoom, extent_og[1]/zoom, extent_og[2]/zoom, extent_og[3]/zoom])
    ax.set_aspect('equal')

    # Colorbar
    if add_cbar and not clean:
        cbar = _colorbar(im)
        if clabel is not None:
            cbar.set_label(clabel)

        if 'vmax' not in kwargs or kwargs['vmax'] < 1:
            cbar.formatter.set_powerlimits((0, 0))

        cbar.update_ticks()

    # Cosmetics
    if clean:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
    else:
        ax.grid(False)
        ax.set_xticks(ax.get_yticks())

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

    return im

# TODO TODO Return meshes, mesh/quivers from these
def plot_I(ax, image, units="cgs", cmap='afmhot', clean=False, tag="", **kwargs):
    # TODO consistent scaling option for movies
    plot_var(ax, image.I * image.scale_flux(units), image, clean=clean, cmap=cmap, **kwargs) #TODO sensible max
    if not clean:
        ax.set_title(tag+"Stokes I [{}]".format(units))

def plot_Q(ax, image, **kwargs):
    return plot_one_stokes(ax, image, 1, **kwargs)
def plot_U(ax, image, **kwargs):
    return plot_one_stokes(ax, image, 2, **kwargs)
def plot_V(ax, image, **kwargs):
    return plot_one_stokes(ax, image, 3, **kwargs)

def plot_one_stokes(ax, image, num, units="cgs", cmap='RdBu_r', clean=False, tag="", **kwargs):
    var = image.get_stokes(num)
    # Take vmax if speccd
    if 'vmax' not in kwargs:
        max_abs = min(max(np.abs(np.max(var)), np.abs(np.min(var))),1e3)
    else:
        max_abs = kwargs['vmax']
    # Ensure no double kwargs
    kwargs.pop('vmax', None)
    kwargs.pop('vmin', None)
    plot_var(ax, image.get_stokes(num) * image.scale_flux(units), image, clean=clean, cmap=cmap,
                vmin=-max_abs, vmax=max_abs, **kwargs)
    if not clean:
        ax.set_title("{} Stokes {} [{}]".format(tag, ["I", "Q", "U", "V"][num], units))

def plot_all_stokes(axes, image, relative=False, units="cgs", n_stokes=4, **kwargs):
    """Plot the raw Stokes parameters on a set of 4 axes"""
    ax = axes.flatten()

    for i in range(n_stokes):
        if i == n_stokes - 1 and units == "cgs":
            clabel = "cgs" # TODO
        elif i == n_stokes - 1 and units == "Jy":
            clabel = "Jy/px"
        else:
            clabel = None
        if i == 0 and not relative:
            plot_I(ax[i], image, clabel=clabel, **kwargs)
        else:
            plot_one_stokes(ax[i], image, i, clabel=clabel, **kwargs)

def plot_lpfrac(ax, image, **kwargs):
    ax.set_facecolor('black')
    plot_var(ax, 100*image.lpfrac(mask_zero=True), image, cmap='jet', vmin=0., vmax=100., **kwargs)
    ax.set_title("LP [%]")

def plot_cpfrac(ax, image, **kwargs):
    cpfrac = image.cpfrac(mask_zero=True)
    vext = max(np.abs(np.nanmin(cpfrac)),np.abs(np.nanmax(cpfrac)))
    vext = max(vext, 1.)
    if np.isnan(vext): vext = 10.

    ax.set_facecolor('black')
    plot_var(ax, 100*image.cpfrac(mask_zero=True), image, cmap='jet', vmin=0., vmax=100., **kwargs)
    ax.set_title("CP [%]")

def plot_evpa_rainbow(ax, image, evpa_conv="EofN", clean=False, **kwargs):
    ax.set_facecolor('black')
    plot_var(ax, image.evpa(evpa_conv, mask_zero=True), image, cmap='hsv', vmin=-90., vmax=90., clean=clean, **kwargs)
    if not clean:
        ax.set_title("EVPA [deg]")

def plot_evpa_ticks(ax, image, n_evpa=20, scaled=False, only_ring=False, fov_units="muas", 
                    custom_wid=1.0, prominent=False, **kwargs):
    """Superimpose EVPA as a quiver plot, either scaled or (TODO) colored by polarization fraction.
    """
    im_evpa = image.downsampled(image.nx // n_evpa)
    evpa = np.pi*im_evpa.evpa(evpa_conv="NofW")/180.
    if scaled:
        # Scaled to polarization fraction
        amp = np.sqrt(im_evpa.Q ** 2 + im_evpa.U ** 2)
        scal = np.max(amp) # TODO consistent scaling option for animations
        vx = amp * np.cos(evpa) / scal
        vy = amp * np.sin(evpa) / scal
    else:
        # Normalized (evpa only)
        vx = np.cos(evpa)
        vy = np.sin(evpa)

    if only_ring:
        slc = im_evpa.ring_mask()
    else:
        slc = (slice(None), slice(None))

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

# Local support functions
def _colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)