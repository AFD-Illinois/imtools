import os
import uuid

import h5py
import numpy as np

import ehtim as eh
from ehtim.features import rex
from ehtim.io.save import save_im_fits

from .ehtim_compat import to_eht_im
from . import io

def get_rex_profile(im, blur=20, verbose=False):
    """Wrapper for the eht-imaging ring-extractor, "rex".
    Returns a ring 'Profile' object with the centered image and
    ring parameters.
    """

    if isinstance(im, str):
        imname = im
        im = io.read_image(imname)
    else:
        try:
            imname = im.properties['fname']
        except KeyError:
            imname = ""

    using_tmp = False
    if not '.h5' in imname or '.fits' in imname:
        # Rex always wants a filename so it can do bad string things to it.
        # So if we only have an image, we oblige by writing an image to /tmp,
        # the least-worst place to do so.
        # The other arg is "postprocdir," which is unused
        # We ensure we fail if it's written to for some reason
        imname = "/tmp/"+str(uuid.uuid4())+".fits"
        save_im_fits(to_eht_im(im), imname)
        using_tmp = True

    pp = rex.FindProfileSingle(imname, "/", blur=blur)

    if using_tmp:
        os.remove(imname)

    if verbose:
        im_center = (pp.x0, pp.y0)
        diam = pp.RingSize1[0]
        width = pp.RingWidth[0]
        print("{} rex center: {} diam: {} width: {}".format(im.name, im_center, diam, width))

    return pp

def rex_and_pmodes(im, blur=20, ms=2, width_coeff=2, **kwargs):
    """Return the PWP beta coefficient m of the image.
    This uses the ring extractor 'rex' from eht-imaging to find the ring center & width,
    then calculates the inner product with a set of basis functions representing patterns in EVPA
    vs angle, with m=2 being the rotationally symmetric mode.

    :param im: path to an image file readable by ehtim
    :param blur: blur to be applied by this function in muas
    :param ms: coefficient or list of coefficients to calculate. You probably want m=2
    :param width_coeff: proportion of ring width considered to the rex value
    :returns a complex number representing the mode, with abs(p) ~ polarization degree in the mode,
                and angle(p) representing average EVPA angle vs the mode
    """
    # Take an image object or filename
    if isinstance(im, str):
        im = io.read_image(im)

    pp = get_rex_profile(im, blur)
    diam = pp.RingSize1[0]
    width = pp.RingWidth[0]
    # Also translate width/coeff to rmin/max
    minr = (diam - width_coeff*width) / 2
    maxr = (diam + width_coeff*width) / 2 
    return pmodes(pp.im_center, ms, r_min=minr, r_max=maxr, **kwargs)

def pmodes_over(im, pp, blur=20, ms=2, width_coeff=2, **kwargs):
    """Return PWP beta_m coefficients of an image, given a particular ring profile.
    ONLY used for sharing a ring profile between multiple images, otherwise see rex_and_pmodes
    """
    # Take all the non-image properties from the centering profile pp
    im_center = (pp.x0, pp.y0)
    diam = pp.RingSize1[0]
    width = pp.RingWidth[0]
    # Center the image to our spec, the way ehtim does
    eim = to_eht_im(im.blurred(blur))
    deltay = -(eim.fovy()/2. - im_center[1] * eh.RADPERUAS) / eim.psize
    deltax = (eim.fovx()/2. - im_center[0] * eh.RADPERUAS) / eim.psize
    im_centered = eim.shift([int(np.round(deltay)), int(np.round(deltax))])
    # Also translate width/coeff to rmin/max
    minr = (diam - width_coeff*width) / 2
    maxr = (diam + width_coeff*width) / 2 
    return pmodes(im_centered, ms, r_min=minr, r_max=maxr, **kwargs)

def pmodes(im, ms, r_min, r_max, norm_in_int=False, norm_with_StokesI=True, return_product=False):
    """Return PWP beta_m coefficients over the given region of a centered image.
    
    :param im: image object from either ehtim or imtools
    :param ms: list of coefficients m to calculate, or single coefficient number e.g. 2
    :param r_min, r_max: radii within which to consider linearly polarized emission
    :param norm_in_int: normalize the sum *before* integrating, rather than after
    :param norm_with_StokesI: normalize to *all* emission, rather than just total polarized emission
    :param return_product: instead of summing to find the coefficient value, return the integrand P*exp(i m phi)
    """

    # Accept single coefficients
    if not (isinstance(ms, list) or isinstance(ms, tuple)):
        ms = (ms,)

    # Load image data.
    if type(im) == eh.image.Image:
        # ehtim image
        fov_muas = im.fovx() / eh.RADPERUAS
        npix = im.xdim
        iarr = im.ivec.reshape(npix, npix)
        qarr = im.qvec.reshape(npix, npix)
        uarr = im.uvec.reshape(npix, npix)
    else:
        # Native imtools image
        fov_muas = im.fov_muas_x
        npix = im.nx
        iarr = im.I * im.scale
        qarr = im.Q * im.scale
        uarr = im.U * im.scale

    parr = qarr + 1j*uarr
    normparr = np.abs(parr)
    marr = parr/iarr
    phatarr = parr/normparr

    # Get distances from center in px
    pxi = (np.arange(npix)-0.01)/npix-0.5
    pxj = np.arange(npix)/npix-0.5
    # Get distances from center in muas
    idist, jdist = np.meshgrid(pxi*fov_muas, pxj*fov_muas)
    dist = np.sqrt(idist**2 + jdist**2)

    # get angles measured East of North
    PXI,PXJ = np.meshgrid(pxi,pxj)
    angles = np.arctan2(-PXJ,PXI) - np.pi/2.
    angles[angles < 0.] += 2.*np.pi

    # Annulus cut
    annulus = (dist <= r_max) & (dist >= r_min)

    # Get properties over just the annulus
    Iann = iarr[annulus].sum()
    Pann = normparr[annulus].sum()
    npix = iarr[annulus].size

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
            coeff = prod[annulus].sum()
            coeff /= npix
        else:
            prod = parr * pbasis
            if return_product:
                return prod
            coeff = prod[annulus].sum()
            if norm_with_StokesI:
                #print("npix: {} Pann: {}".format(npix, Pann))
                #print("beta_2 integral: {} total annulus emission: {}".format(coeff, Iann))
                coeff /= Iann
            else:
                coeff /= Pann
            betas.append(coeff)

    if len(betas) == 1:
        return betas[0]
    else:
        return betas
