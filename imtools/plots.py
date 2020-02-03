# Different plots that can be made of images, or combinations of images
# Functions with their own figures begin with generate_,
# those that take an axis on which to plot begin plot_

import numpy as np

from imtools.image import Image

def plot_var(ax, var, image, scale=1, fov_units="muas", xlabel=True, ylabel=True, clabel=None, **kwargs):
    """General function for plotting a set of pixels, given some options about how"""
    im = ax.imshow(var*scale, origin='lower', extent=image.extent(fov_units), **kwargs)

    cbar = _colorbar(im)
    if clabel is not None:
        cbar.set_label(clabel)

    if 'vmax' not in kwargs or kwargs['vmax'] < 1:
        cbar.formatter.set_powerlimits((0, 0))

    cbar.update_ticks()

    ax.set_aspect('equal')
    ax.grid(False)
    if xlabel:
        if fov_units == "muas":
            ax.set_xlabel(r"x ($\mu as$)")
        else:
            ax.set_xlabel("x ({})".format(fov_units))
    if ylabel:
        if fov_units == "muas":
            ax.set_xlabel(r"y ($\mu as$)")
        else:
            ax.set_ylabel("y ({})".format(fov_units))

    return im

def plot_I(ax, image, units="cgs", fov_units="muas", **kwargs):
    plot_var(ax, image.I, image, cmap='afmhot', vmin=0., vmax=1.e-4, **kwargs) #TODO sensible max
    ax.set_title("Stokes I [{}]".format(units))

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


def plot_evpa_rainbow(ax, image, evpa_conv="EofN", **kwargs):
    ax.set_facecolor('black')
    plot_var(ax, image.evpa(evpa_conv, mask_zero=True), image, cmap='hsv', vmin=-90., vmax=90., **kwargs)
    ax.set_title("EVPA [deg]")

def plot_evpa_ticks(ax, image, n_evpa=32, scaled=False, only_ring=False, **kwargs):
    """Superimpose EVPA as a quiver plot, either scaled or (TODO) colored by polarization fraction.
    """
    # TODO ticks are generally too long
    evpa = image.evpa()
    if scaled:
        # Scaled to polarization fraction
        amp = np.sqrt(image.Q ** 2 + image.U ** 2)
        scal = np.max(amp) # TODO consistent scaling option for animations
        vx = amp * np.cos(evpa) / scal
        vy = amp * np.sin(evpa) / scal
    else:
        # Normalized (evpa only)
        vx = np.cos(evpa)
        vy = np.sin(evpa)

    skipx = int(image.nx / n_evpa)
    skipy = int(image.ny / (n_evpa * (image.ny / image.nx)))

    if only_ring:
        slc = np.where(image.ring_mask(skipx, skipy))
    else:
        slc = (slice(None), slice(None))

    i, j = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], image.nx), np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], image.nx))

    # TODO the plot_var niceties
    ax.quiver(i[::skipx, ::skipy][slc], j[::skipx, ::skipy][slc], vx[::skipx, ::skipy][slc],
               vy[::skipx, ::skipy][slc], headwidth=1, headlength=1, **kwargs)


# Local support functions
def _colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)