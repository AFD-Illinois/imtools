# Polarization modes analysis
# Stolen nearly verbatim from George Wong, and before that Daniel Palumbo
# Copyright status: ?????

import h5py
import numpy as np

from ehtim_compat import to_eht_im
import ehtim as eh

from imtools.image import Image


def pmodes(image, ms=(2,), r_min=0, r_max=25, norm_in_int=False, norm_with_StokesI=True):
    """Return beta_m coefficients for m in ms within extent r_min/r_max."""
    im = to_eht_im(image)

    if type(im) == eh.image.Image:
        npix = im.xdim
        iarr = im.ivec.reshape(npix, npix)
        qarr = im.qvec.reshape(npix, npix)
        uarr = im.uvec.reshape(npix, npix)
        varr = im.vvec.reshape(npix, npix)
        fov_muas = im.fovx()/eh.RADPERUAS

    elif type(im) == Image:
        DX = im['camera']['dx']
        dsource = im['dsource']
        lunit = im['units']['L_unit']
        scale = im['scale']

        fov_muas = DX / dsource * lunit * 2.06265e11
        npix = im.I.shape[0]
        iarr = im.I
        qarr = im.Q
        uarr = im.U
        varr = im.V

    parr = qarr + 1j*uarr
    normparr = np.abs(parr)
    marr = parr/iarr
    phatarr = parr/normparr
    area = (r_max*r_max - r_min*r_min) * np.pi
    pxi = (np.arange(npix)-0.01)/npix-0.5
    pxj = np.arange(npix)/npix-0.5
    mui = pxi*fov_muas
    muj = pxj*fov_muas
    MUI, MUJ = np.meshgrid(mui, muj)
    MUDISTS = np.sqrt(np.power(MUI, 2.)+np.power(MUJ, 2.))

    # get angles measured East of North
    PXI, PXJ = np.meshgrid(pxi, pxj)
    angles = np.arctan2(-PXJ, PXI) - np.pi/2.
    angles[angles < 0.] += 2.*np.pi

    # get flux in annulus
    tf = iarr[(MUDISTS <= r_max) & (MUDISTS >= r_min)].sum()

    # get total polarized flux in annulus
    pf = normparr[(MUDISTS <= r_max) & (MUDISTS >= r_min)].sum()

    # get number of pixels in annulus
    npix = iarr[(MUDISTS <= r_max) & (MUDISTS >= r_min)].size

    # get number of pixels in annulus with flux >= some % of the peak flux
    ann_iarr = iarr[(MUDISTS <= r_max) & (MUDISTS >= r_min)]
    peak = np.max(ann_iarr)
    #num_above5 = ann_iarr[ann_iarr > .05 * peak].size
    #num_above10 = ann_iarr[ann_iarr > .1 * peak].size

    # compute betas
    betas = []
    for m in ms:
        qbasis = np.cos(-angles*m)
        ubasis = np.sin(-angles*m)
        pbasis = qbasis + 1.j*ubasis
        if norm_in_int:
            if norm_with_StokesI:
                prod = marr * pbasis
            else:
                prod = phatarr * pbasis
            coeff = prod[(MUDISTS <= r_max) & (MUDISTS >= r_min)].sum()
            coeff /= npix
        else:
            prod = parr * pbasis
            coeff = prod[(MUDISTS <= r_max) & (MUDISTS >= r_min)].sum()
            if norm_with_StokesI:
                coeff /= tf
            else:
                coeff /= pf
        betas.append(coeff)

    # Find some way to keep tf, pf, npix, num_above5, num_above10
    if len(betas) == 1:
        return betas[0]
    else:
        return betas
