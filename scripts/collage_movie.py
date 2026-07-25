#!/usr/bin/env python3

import os
from itertools import product
from functools import partial
import click
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import h5py

# Big white serif font
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.monospace": ["DejaVu Sans Mono"],
    "text.color": 'white',
    "axes.labelcolor": 'white',
    "xtick.labeltop": True,
    "xtick.labelbottom": True
    #"text.usetex": True
})

import imtools as it

import hallmark as hm

def do_plot(snapshot, flux, spin, rhigh, inc, kwargs, annotations):
    # Time from the first imaged snapshot
    frame = snapshot - 3000

    annotations['note_inc'] = False
    annotations['note_spin'] = False
    annotations['note_rhigh'] = False

    if kwargs['build']:
        sep_build = kwargs['sep_build']
        is_in = False
        if frame <= sep_build and (flux, spin, rhigh, inc) == ('M', '+0.94', 160, 90):
            is_in = True
        elif frame > sep_build and frame <= 2*sep_build and (flux, spin, rhigh) == ('M', '+0.94', 160):
            matplotlib.rcParams["font.size"] = 16
            annotations['note_inc'] = (frame > sep_build + kwargs['note_delay'])
            is_in = True
        elif frame > 2*sep_build and frame <= 3*sep_build and (flux, rhigh) == ('M', 160):
            matplotlib.rcParams["font.size"] = 16
            annotations['note_spin'] = ((frame > 2*sep_build + kwargs['note_delay']) and inc == 10)
            is_in = True
        elif frame > 3*sep_build and frame <= 4*sep_build and flux == 'M':
            matplotlib.rcParams["font.size"] = 24
            annotations['note_rhigh'] = ((frame > 3*sep_build + kwargs['note_delay']) and inc == 50 and spin == '+0.94')
            is_in = True
        elif frame > 4*sep_build:
            is_in = True
    else:
        is_in = True

    if kwargs['cuts']:
        # Select this model in the table
        flux_map = {'S': 0, 'M': 1}
        pt = kwargs['pt']
        pt_sel = pt.loc[(pt['flux'] == flux_map[flux]) &
                        (pt['spin'] == float(spin)) &
                        (pt['inc'] == inc) &
                        (pt['rhigh'] == rhigh)]

        # Take models out starting where we left off
        # Cut order:
        # EHT
        # 230GHz Size: 230GHz_size
        # "Shape"
        # * Null Location: Null_loc
        # * Mring asymmetry: Mring_f1
        # * Mring Diameter: Mring_d
        # * Mring Width: Mring_w
        # NON-EHT
        # "Spectrum"
        # * 86GHz flux: 86GHz_flux
        # * 2um flux: 2um_flux
        # * X-ray flux: Xray_flux
        # 86GHz Size: 86GHz_size
        frame -= kwargs['pause']
        if kwargs['build']:
            # Start after the end of the build anim
            frame -= 5*kwargs['sep_build']
        sep_cut = kwargs['sep_cuts']
        is_out = False
        if frame > 0 and pt_sel['230GHz_size'].iloc[0] != 0:
            is_out = True
        elif frame > sep_cut and (pt_sel['Null_loc'].iloc[0] != 0
                    or pt_sel['Mring_f1'].iloc[0] != 0
                        or pt_sel['Mring_d'].iloc[0] != 0
                        or pt_sel['Mring_w'].iloc[0] != 0):
            is_out = True
        elif frame > 2*sep_cut and (pt_sel['86GHz_flux'].iloc[0] != 0
                        or pt_sel['2um_flux'].iloc[0] != 0
                        or pt_sel['Xray_flux'].iloc[0] != 0):
            is_out = True
        elif frame > 3*sep_cut and (pt_sel['86GHz_size'].iloc[0] != 0):
            is_out = True
    else:
        is_out = False

    if is_in and not is_out:
        return True
    else:
        return False

def get_path(pf, snapshot, flux, spin, rhigh, inc):
        # Select our image. Note multiple args are OR'd together,
        # so we must use separate calls
        sel = pf(flux=flux)
        sel = sel(spin=spin)
        sel = sel(Rhigh=rhigh)
        sel = sel(inc=inc)
        sel = sel(win=(snapshot // 1000))
        # Get the image path
        return sel.path.iloc[0]

# Process a frame
def plot_frame(snapshot, kwargs={}):
    fig, _ = plt.subplots(kwargs['nplotsy'], kwargs['nplotsx'], figsize=(kwargs['nplotsx'],kwargs['nplotsy']))
    ax = fig.get_axes()

    for axis in ax:
        axis.set_facecolor('k')

    # Select snapshot here if imaging for individual files
    pf = kwargs['pf']
    params = kwargs['params']
    vmaxes = kwargs['vmaxes']

    i = 0
    # For all models, in the order we choose...
    for flux, spin, rhigh, inc in product(params['flux'], ('-0.94', '-0.5', '0', '+0.5', '+0.94'), params['Rhigh'], params['inc']):
        # Ignore "duplicate" inclinations
        if inc > 90:
            continue

        # Bare plots
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        # Leave black boxes for cut or not-introduced plots
        annotations = {}
        if do_plot(snapshot, flux, spin, rhigh, inc, kwargs, annotations):
            # Get the image path
            path = get_path(pf, snapshot, flux, spin, rhigh, inc)
            # Set vmax based on cached values
            if flux == 'M':
                vmax = vmaxes[path] * 0.1
            else:
                vmax = vmaxes[path] * 0.3
            # Plot image
            image = it.io.read_from_cache(path, snapshot % 1000)
            it.plot_unpolarized(ax[i], image, zoom=2, clean=True, vmax=vmax, cmap='afmhot_us')

            # Annotate if we need
            if kwargs['annotate']:
                if 'note_inc' in annotations and annotations['note_inc']:
                    ax[i].set_title(r"$i = {}^\circ$".format(inc))
                    ax[i].set_zorder(100)
                if 'note_spin' in annotations and annotations['note_spin']:
                    ax[i].set_ylabel(r"$a = {}$".format(spin), rotation=0, ha="right", zorder=100)
                    ax[i].set_zorder(100)
                if 'note_rhigh' in annotations and annotations['note_rhigh']:
                    ax[i].set_xlabel(r"$R_{\mathrm{high}} = %d$" % rhigh, zorder=100)
                    ax[i].set_zorder(100)

        # Advance regardless, leaving bare boxes where we do not plot
        i += 1
    plt.subplots_adjust(top=1.0, left=0.0, right=1.0, bottom=0.0, hspace=0, wspace=0.0)
    plt.savefig(kwargs['out_dir']+"/frame_{}.png".format(snapshot), dpi=kwargs['fig_dpi'])
    plt.close()


@click.command()
@click.argument('src_fmt', nargs=1, default="/fd1/eht/SgrA/cache/Illinois_thermal_w{win:d}/{flux}a{spin}_i{inc:d}/summ_Rh{Rhigh:d}_230GHz.h5")
# Individual images: "/bd6/eht/Illinois_SgrA_v3check/230GHz/{flux}a{spin}_w{win:d}/img_s{snapshot:d}_Rh{Rhigh:d}_i{inc:d}.h5" REQUIRES TWEAKING! ALSO V SLOW
# Plot types
@click.option('-b', '--build', is_flag=True, help="Build up the grid one axis at a time")
@click.option('-c', '--cuts', is_flag=True, help="Apply Paper V constraints as cuts")
@click.option('-d', '--out_dir', default="frames_collage", help="Directory for output")
# Timings
@click.option('--sep_build', default=240, help="Timing in frames between each stage when building")
@click.option('--sep_cuts', default=120, help="Timing in frames between each stage when cutting")
@click.option('--pause', default=240, help="Timing in frames of the pause between build & cut")
@click.option('--note_delay', default=0, help="Delay before annotations are added to a new build phase, to accommodate zoom")
# General plotting options
@click.option('--nplotsx', default=20, help="Number of columns of plots")
@click.option('--nplotsy', default=10, help="Number of rows of plots")
@click.option('--fig_dpi', default=200, help="DPI of resulting figure")
@click.option('-a', '--annotate', is_flag=True, help="Annotate the image during each build step")
@click.option('-n', '--nframes', default=3000, help="Number of total frames to render")
def collage_movie(src_fmt, **kwargs):
    """Generate frames corresponding to a movie of a whole set of models, optionally introduced or cut over the
    runtime.  Models can be introduced in some order and cut out according to a table of models passing certain
    constraints.

    Usage: collage_movie.py [src_fmt] [-bc] [options]
    """
    # Find input models using hallmark `ParaFrame`
    # This finds all values of replacements in src_fmt which will generate valid filenames
    pf = hm.ParaFrame(src_fmt)

    # Automatically determine parameters, and turn `params` into
    # a dict mapping parameters to lists of their unique values
    params = list(pf.keys())
    params.remove('path')
    params = {p:np.unique(pf[p]) for p in params}

    # Cache maximum values for each pane so we can normalize independently
    # Note this only supports reading caches! There's no easy way to do this for files.
    try:
        with open('vmaxes.p', 'rb') as handle:
            vmaxes = pickle.load(handle)
    except IOError:
        # Get all the vmaxes
        vmaxes = {}
        for flux, spin, rhigh, inc in product(params['flux'], ('-0.94', '-0.5', '0', '+0.5', '+0.94'), params['Rhigh'], params['inc']):
            path = get_path(pf, 3000, flux, spin, rhigh, inc)
            with h5py.File(path, 'r') as cachef:
                vmax = np.max(cachef['unpol'])
                vmaxes[path] = vmax
                vmaxes[get_path(pf, 4000, flux, spin, rhigh, inc)] = vmax
                vmaxes[get_path(pf, 5000, flux, spin, rhigh, inc)] = vmax

        # Save them for next time
        with open('vmaxes.p', 'wb') as handle:
            pickle.dump(vmaxes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Stash things in the 'kwargs' dict for plot_frame calls laster
    kwargs['pf'] = pf
    kwargs['params'] = params
    kwargs['vmaxes'] = vmaxes

    # Read the table of cuts
    if kwargs['cuts']:
        with open('Illinois_Pass_Table.dat') as inf:
            header = inf.readline()[1:].split() + ['lp']
            kwargs['pt'] = pandas.read_table('Illinois_Pass_Table.dat', comment='#', names=header)

    if not os.path.exists(kwargs['out_dir']):
        os.mkdir(kwargs['out_dir'])

    worker = partial(plot_frame, kwargs=kwargs)
    it.parallel.map_parallel(worker, range(3000, 3000+kwargs['nframes']))

if __name__ == "__main__":
    collage_movie()
