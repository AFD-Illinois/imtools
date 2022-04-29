"""

$ python plspec.py path/to/spectrum/files

makes a nice plot from hdf5 file.

"""

import units

import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Just constants
cgs = units.get_cgs()


def plot_spectrum(nu, nuLnu, fname, figsize=(8,8), split_spectrum=True):
    # plot
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if split_spectrum:
        ax.step(nu, nuLnu.sum(axis=0), "k", label="total")
        ax.step(nu, nuLnu[0,:], label="(synch) base")
        ax.step(nu, nuLnu[1,:], label="(synch) once")
        ax.step(nu, nuLnu[2,:], label="(synch) twice")
        ax.step(nu, nuLnu[3,:], label="(synch) > twice")
        ax.step(nu, nuLnu[4,:], label="(brems) base")
        ax.step(nu, nuLnu[5,:], label="(brems) once")
        ax.step(nu, nuLnu[6,:], label="(brems) twice")
        ax.step(nu, nuLnu[7,:], label="(brems) > twice")
    else:
        ax.step(nu, nuLnu, "k", label="total")

    # formatting
    nuLnu_max = nuLnu.max()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1.e8, 1.e24])
    ax.set_ylim([1.e-10 * nuLnu_max, 1.e1 * nuLnu_max])
    ax.set_xlabel(r"$\nu$ (Hz)", fontsize=16)
    ax.set_ylabel()
    ax.legend()
    ax.set_grid(True)

    return fig


def read_spectra(fname_glob, split_spectrum=True):
    """Read a spectrum, summing from one or more files.
    """
    fname_list = np.sort(glob.glob(fname_glob))
    # TODO detect split spectrum in file, sum when split_spectrum=False
    nuLnu_total = 0
    for fname in fname_list:
        with h5py.File(fname, "r") as fp:
            # load data
            if "githash" in fp.attrs.keys():
                nu = np.power(10.,fp["output"]["lnu"]) * cgs['ME'] * cgs['CL']**2 / cgs['HPL']
                nuLnu = np.array(fp["output"]["nuLnu"]) * cgs['LSOLAR']
                if split_spectrum:
                    nuLnu = nuLnu[:,:,-1]
                else:
                    nuLnu = nuLnu[:,-1]
            else:
                nu = np.power(10.,fp["ebin"]) * cgs['ME'] * cgs['CL']**2 / cgs['HPL']
                nuLnu = fp["nuLnu"][:,0] * cgs['LSOLAR']

        nuLnu_total += nuLnu

    return nu, nuLnu_total
