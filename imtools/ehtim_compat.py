__license__ = """
 File: ehtim_compat.py
 
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

have_ehtim = False
try:
    import ehtim
    import ehtim.const_def as ehc
except ModuleNotFoundError:
    print("Couldn't import ehtim.  Compatibility & some plots disabled.")
    have_ehtim = True

"""Conversions between "Image" objects in this library and "Image" objects in
`eht-imaging <https://github.com/achael/eht-imaging>`_.
Not all elements of each object are supported, YMMV.
"""

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
