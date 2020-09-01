
import numpy as np
import ehtim
import ehtim.const_def as ehc

def from_eht_im(im):
    # TODO fill properties struct...
    pass
    #return Image(self.properties.copy(), I, Q, U, V, tau=tau, tauF=tauF, unpol=unpol, init_type="multi_arrays_stokes")

def to_eht_im(image):
    """Convert an imtools image to ehtim Image object

    "Don't reinvent the wheel.  Reinvent the sled and strap wheels on it afterward"
    """
    dsource = image.properties['dsource']          # distance to source in cm
    jyscale = image.properties['scale']            # convert cgs intensity -> Jy flux density
    rf = image.properties['freqcgs']               # in cgs
    tunit = image.properties['units']['T_unit']    # in seconds
    lunit = image.properties['units']['L_unit']    # in cm
    DX = image.properties['camera']['dx']          # in GM/c^2
    nx = image.properties['camera']['nx']          # width in pixels
    time = image.properties['t'] * tunit / 3600.       # time in hours

    # This works, I'm pretty sure ¯\_(ツ)_/¯
    poldat = np.rot90(image.get_raw().transpose(1, 2, 0), 2)

    # Make a guess at the source based on distance and optionally fall back on mass
    src = ehc.SOURCE_DEFAULT
    if dsource > 4.e25 and dsource < 6.2e25:
        src = "M87"
    elif dsource > 2.45e22 and dsource < 2.6e22:
        src = "SgrA"

    # Fill in information according to the source
    ra = ehc.RA_DEFAULT
    dec = ehc.DEC_DEFAULT
    if src == "SgrA":
        ra = 17.76112247
        dec = -28.992189444
    elif src == "M87":
        ra = 187.70593075
        dec = 12.391123306

    # Process image to set proper dimensions
    fovmuas = DX / dsource * lunit * 2.06265e11
    psize_x = ehc.RADPERUAS * fovmuas / nx

    Iim = poldat[:, :, 0] * jyscale
    # Make sure we give ehtim an "East of North" image.
    if image.evpa_0 == "N":
        Qim = poldat[:, :, 1] * jyscale
        Uim = poldat[:, :, 2] * jyscale
    else:
        Qim = -poldat[:, :, 1] * jyscale
        Uim = -poldat[:, :, 2] * jyscale
    Vim = poldat[:, :, 3] * jyscale

    outim = ehtim.image.Image(Iim, psize_x, ra, dec, rf=rf, source=src,
                              polrep='stokes', pol_prim='I', time=time)
    outim.add_qu(Qim, Uim)
    outim.add_v(Vim)

    return outim