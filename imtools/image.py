"""
 File: image.py
 
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

# image.py

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import copy

from scipy.ndimage import gaussian_filter

import imtools.stats as stats
from imtools.units import cgs

# TODO list:
# Standard way of representing a GRMHD model, camera, and e- parameters, instead of a dict
# All as one "model" or separate?

def _power_of_two(target):
    """Finds the next greatest power of two
    """
    cur = 1
    if target > 1:
        for i in range(0, int(target)):
            if (cur >= target):
                return cur
            else:
                cur *= 2
    else:
        return 1

def _visibilities_from_image(imarr, fft_pad_factor):
    xdim = imarr.shape[0]
    ydim = imarr.shape[1]
    # Padded image size
    npad = fft_pad_factor * np.max((xdim, ydim))
    npad = _power_of_two(npad)

    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    imarr = imarr.T
    imarr = np.pad(imarr, ((padvalx1, padvalx2), (padvaly1, padvaly2)),
                       'constant', constant_values=0.0)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

class Image(object):
    """Images are intended for polarized theory "truth" data in the image plane only,
    represented as Stokes parameters and with square pixels (i.e. FOVx/FOVy == nx/ny).
    """
    data_members = ["I", "Q", "U", "V", "tau", "tauF", "unpol"]

    def __init__(self, properties, array1, array2=None, array3=None, array4=None, tau=None, tauF=None, unpol=None,
                    init_type="one_array_forward"):
        """Initialize an Image object from data.

        @param: properties: dict of image properties.
        This has no standard format on account of there's no standard image format.
        Generally this library expects the ipole metadata to be present here,
        see ipole image format doc on Illinois wiki:
        https://github.com/AFD-Illinois/docs/wiki/Image-Format

        @param array1, array2, array3, array4: Stokes parameters in CGS. Contents interpreted based on init_type.
            one_array_forward: array of Stokes in index-first form i,j,S.
            one_array_backward: array in Stokes-first form S,i,j.
            multi_arrays_stokes: four arrays size i,j representing I,Q,U,V.
            multi_arrays_rl: NOT IMPLEMENTED -- initialize from right- and left-circular components

            In all cases, i, j, should be ordered such that when plotted wiht meshgrid or imshow(origin='lower'),
            the resulting image "looks correct" with North pointing in the +y direction

        @param init_type: see above
        @param tau: if not none, set optical depth member tau to this array
        @param tauF: if not none, set Faraday depth member tauF to this array
        @param unpol: if not none, add a version of stokes I computed with unpolarized transport.
        Note that unpolarized images are better off setting just stokes I --
        this member is for comparisons and sanity checks
        """
        # Leave the possibility for initializing ehtim-style RL images
        if init_type == "one_array_forward":
            self.I = array1[:,:,0]
            self.Q = array1[:,:,1]
            self.U = array1[:,:,2]
            self.V = array1[:,:,3]
        elif init_type == "one_array_backward":
            self.I = array1[0,:,:]
            self.Q = array1[1,:,:]
            self.U = array1[2,:,:]
            self.V = array1[3,:,:]
        elif init_type == "multi_arrays_stokes":
            # TODO can we get away with not deep-copying in this case?
            self.I = array1[:,:]
            self.Q = array2[:,:]
            self.U = array3[:,:]
            self.V = array4[:,:]
        # This makes sure the members exist
        # We check if they are None before using
        self.tau = tau
        self.tauF = tauF
        self.unpol = unpol
        self.parse_properties(properties)


    def parse_properties(self, properties):
        self.properties = copy.deepcopy(properties)

        # Load camera parameters if we can
        if 'camera' in properties:
            # Size.  Could be obtained from np arrays but good to check header
            self.nx = properties['camera']['nx']
            self.ny = properties['camera']['ny']

            # FOV in M. Named historically...
            self.Dx = properties['camera']['dx']
            self.Dy = properties['camera']['dy']

            self.dsource = properties['dsource']
            self.lunit = properties['units']['L_unit']
            self.MBH = self.lunit*cgs['CL']**2/cgs['GNEWT']/cgs['MSOLAR']

            # Magic number is muas per radian
            # TODO multiplication order?
            self.fov_muas_x = self.Dx / self.dsource * self.lunit * 2.06265e11
            self.fov_muas_y = self.Dy / self.dsource * self.lunit * 2.06265e11

            self.scale = properties['scale']
            self.t = properties['t']
        else:
            # Just guess size and have done
            # TODO probably quite dangerous when reading dat files from at least ipole...
            self.nx = self.I.shape[0]
            self.ny = self.I.shape[1]
            self.scale = 1

        if 'evpa_0' in properties:
            self.evpa_0 = properties['evpa_0']
        else:
            # Guess that people probably follow theory convention
            print("Warning: guessing EVPA 0 point, EVPA may be off by 90deg")
            self.evpa_0 = "W"

        if 'name' in self.properties:
            self.name = self.properties['name']
        else:
            self.name = ""

    # Per-pixel derived quanties: return a new array of size i,j
    def lpfrac(self, mask_zero=False):
        lpfrac = np.sqrt(self.Q**2 + self.U**2) / self.I
        if mask_zero: lpfrac[self.zero_mask()] = np.nan
        return lpfrac
    def cpfrac(self, mask_zero=False):
        cpfrac = self.V / self.I
        if mask_zero: cpfrac[self.zero_mask()] = np.nan
        return cpfrac
    def evpa(self, evpa_conv="EofN", mask_zero=False):
        """Return the per-pixel EVPA in degrees in *EAST OF NORTH* convention by default"""
        evpa = (180./np.pi)*0.5*np.arctan2(self.U, self.Q)
        if self.evpa_0 == "W":
            evpa += 90.
            evpa[evpa > 90.] -= 180.
        if evpa_conv == "NofW":
            evpa -= 90.
            evpa[evpa < -90.] += 180.
        if mask_zero:
            evpa[self.zero_mask()] = np.nan
        return evpa
    
    # Integrated a.k.a zero-baseline quantities
    def lpfrac_int(self):
        """Integrated (zero-baseline) linear polarization fraction. NOT percentage"""
        return np.sqrt(self.Qtot()**2 + self.Utot()**2) / self.Itot()
    def cpfrac_int(self):
        """Integrated (zero-baseline) circular polarization fraction. NOT percentage"""
        return self.Vtot() / self.Itot()
    def evpa_int(self):
        """Integrated (zero-baseline) EVPA"""
        return (180./np.pi)*0.5*np.arctan2(self.Utot(), self.Qtot())

    # Average lpfrac with given blur (NOT zero-baseline, taken per-px and averaged)
    def lpfrac_av(self, blur=20, mask_zero=False):
        """Average linear polarization fraction per-pixel across the image.  Heavily blur dependent
        """
        # TODO Jason *must* massage this, right?
        return np.mean(self.blurred(blur).lpfrac(mask_zero))
    
    def tauF_av(self):
        """Average full Faraday rotation angle across the image"""
        return np.mean(self.tauF)
    def tau_av(self):
        """Average optical depth across the image"""
        return np.mean(self.tau)

    # Basic quantities.  Note for multithreading we use e.g.
    # Image.get_t(particular_image)
    # So we can't always use members
    def get_name(self):
        return self.name
    def get_t(self):
        return self.t
    def N(self):
        return self.I.shape[0]*self.I.shape[1]

    def Itot(self):
        """Total Stokes I flux in Jy"""
        return np.sum(self.I) * self.scale
    def Qtot(self):
        """Total Stokes Q flux in Jy"""
        return np.sum(self.Q) * self.scale
    def Utot(self):
        """Total Stokes U flux in Jy"""
        return np.sum(self.U) * self.scale
    def Vtot(self):
        """Total Stokes V flux in Jy"""
        return np.sum(self.V) * self.scale
    def flux(self):
        """Total Stokes I flux in Jy"""
        return self.Itot()
    def flux_unpol(self):
        """Total flux in Jy of unpolarized image"""
        if self.unpol is not None:
            return np.sum(self.unpol) * self.scale
        else:
            return 0

    def get_raw(self):
        """Array [s,i,j] of all Stokes parameters"""
        return np.array([self.I, self.Q, self.U, self.V])

    # Programmatic Stokes access 0,1,2,3
    # Returns None on invalid input
    def get_stokes(self, n):
        """Get image [i,j] of Stokes parameter #n where 0==I"""
        if n == 0: return self.I
        if n == 1: return self.Q
        if n == 2: return self.U
        if n == 3: return self.V

    # Comparison quantities
    # SEE STATS for full descriptions
    def mse(self, var1, var2):
        """MSE of one var vs another.
        Follows the usual convention of dividing by the sum(var) of *this* image
        """
        return stats.mse(var1, var2)
    def mses(self, other):
        """MSE for each variable with another image"""
        return stats.mses(self, other)
    def ssim(self, var1, var2):
        """Structural similarity SSIM as defined in Gold et al eq 10-ish"""
        return stats.ssim(var1, var2)
    def ssims(self, other):
        """SSIM for each variable with another image"""
        return stats.ssims(self, other)
    def dssim(self, var1, var2):
        """Structural dissimilarity DSSIM is 1/SSIM - 1"""
        return stats.dssim(var1, var2)
    def dssims(self, other):
        """DSSIM for each variable with another image"""
        return stats.dssims(self, other)
    def zncc(self, var1, var2):
        """Image Zero-Normalized Cross-Correlation"""
        return stats.zncc(var1, var2)
    def znccs(self, other):
        """ZNCC for each variable with another image"""
        return stats.znccs(self, other)
    def ncc(self, var1, var2):
        """Image Normalized Cross-Correlation: ZNCC without subtracting the mean"""
        return stats.ncc(var1, var2)
    def nccs(self, other):
        """NCC for each variable with another image"""
        return stats.nccs(self, other)

    # Plotting convenience quantities
    def zero_mask(self):
        """Get a mask of just the zones where I is vanishingly small compared to the main image.
        Set them to nan to avoid printing BS lpfrac/cpfrac when dividing by ~<=0
        """
        #Imaskval = np.abs(self.I.min()) * 100.
        Imaskval = np.nanmax(self.I) / self.I.shape[0]**5
        return np.abs(self.I) < Imaskval
    
    def ring_mask(self, var=None, cut=1.0, add_std=False):
        """Get a mask of just the ring, i.e. zones with I greater than some portion of mean(I),
        Or optionally mean(I) + stddev
        """
        if var is None: # Clumsy option for custom cuts, e.g. polarized emission (useful?)
            var = self.I
        if add_std:
            return var > np.mean(var) + np.std(var)
        else:
            return var > cut * np.mean(var)

    def extent(self, fov_units):
        """Window corresponding to full image"""
        if fov_units == "muas":
            return [ -self.fov_muas_x/2, self.fov_muas_x/2, -self.fov_muas_y/2, self.fov_muas_y/2 ]
        elif fov_units == "M":
            return [ -self.Dx/2, self.Dx/2, -self.Dy/2, self.Dy/2 ]
        else:
            print("! unrecognized units for FOV {}. quitting.".format(fov_units))

    def scale_flux(self, units):
        if units == "cgs":
            return 1
        elif units == "Jy/px" or units == "Jy":
            return self.scale

    # Volatile Operations
    def rot90(self, rot):
        """Rotate this image in-place 90 degrees CCW 'rot' times"""
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                self.__dict__[m] = np.rot90(self.__dict__[m], rot)
        self.Q *= (-1)**(rot % 2)
        self.U *= (-1)**(rot % 2)

    # Nonvolatile Operations
    # TODO generalize applying a single-argument op, like the infixes below
    def rel_diff(self, other, clip=None):
        """Return the image representing pixel-wise relative difference between this image and another.
        That is, (im2-im1)/im1 for each pixel.  Optionally clipped 
        """
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None and other.__dict__[m] is not None:
                if clip is not None:
                    new_vars[m] = np.clip((other.__dict__[m] - self.__dict__[m]) / self.__dict__[m], clip[0], clip[1])
                else:
                    new_vars[m] = (other.__dict__[m] - self.__dict__[m]) / self.__dict__[m]
            else:
                new_vars[m] = None

        return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    def rotated90(self, rot):
        """Return an image rotated by 90 degrees CCW 'rot' times"""
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                new_vars[m] = np.rot90(self.__dict__[m], rot)
            else:
                new_vars[m] = None
        new_vars['Q'] *= (-1)**(rot % 2)
        new_vars['U'] *= (-1)**(rot % 2)

        return Image(self.properties, new_vars['I'], -new_vars['Q'], -new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    def blurred(self, fwhm=20):
        """Return a version of this image blurred by a circular gaussian of 
        """
        fwhm_px = fwhm / self.fov_muas_x * self.nx
        sigma = (fwhm_px / (2 * np.sqrt(2 * np.log(2))))
        I = gaussian_filter(self.I, sigma=sigma)
        Q = gaussian_filter(self.Q, sigma=sigma)
        U = gaussian_filter(self.U, sigma=sigma)
        V = gaussian_filter(self.V, sigma=sigma)
        # TODO not sure these mean anything
        if self.tau is not None:
            tau = gaussian_filter(self.tau, sigma=sigma)
        else:
            tau = None
        if self.tauF is not None:
            tauF = gaussian_filter(self.tauF, sigma=sigma)
        else:
            tauF = None
        if self.unpol is not None:
            unpol = gaussian_filter(self.unpol, sigma=sigma)
        else:
            unpol = None
        # TODO does this change any properties?
        return Image(self.properties.copy(), I, Q, U, V, tau=tau, tauF=tauF, unpol=unpol, init_type="multi_arrays_stokes")
    
    def downsampled(self, skip=2):
        I = self.I[::skip, ::skip]
        Q = self.Q[::skip, ::skip]
        U = self.U[::skip, ::skip]
        V = self.V[::skip, ::skip]
        if self.tau is not None:
            tau = self.tau[::skip, ::skip]
        else:
            tau = None
        if self.tauF is not None:
            tauF = self.tauF[::skip, ::skip]
        else:
            tauF = None
        if self.unpol is not None:
            unpol = self.unpol[::skip, ::skip]
        else:
            unpol = None
        props = copy.deepcopy(self.properties)
        props['camera']['nx'] = I.shape[0]
        props['camera']['ny'] = I.shape[1]
        return Image(props, I, Q, U, V, tau=tau, tauF=tauF, unpol=unpol, init_type="multi_arrays_stokes")
    
    def visibilities(self, pad_x=10):
        """Get the complex visibilities corresponding to the image
        @param Number of times by which to pad FFT to preserve resolution in desired area.
               Image will be padded with zeros to a total of N1*pad_x by N2*pad_x pixels.
        @return "Image" with complex visibilities of each Stokes parameter
        """
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                new_vars[m] = _visibilities_from_image(self.__dict__[m], pad_x)
            else:
                new_vars[m] = None
        return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    def visibility_amplitudes(self, pad_x=10, crop_x=10):
        """Get the complex visibilities corresponding to the image
        @param pad_x Number of times by which to pad FFT to preserve resolution in desired area.
               Image will be padded with zeros to a total of N1*pad_x by N2*pad_x pixels.
        @param re_crop Crop image back to N1xN2 pixels in the center of the FFT'd version
        @return "Image" with complex visibilities of each Stokes parameter
        """
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                new_vars[m] = np.abs(_visibilities_from_image(self.__dict__[m], pad_x))
                if crop_x < pad_x:
                    center = [s//2 for s in new_vars[m].shape]
                    window = [s*crop_x//2 for s in self.__dict__[m].shape]
                    new_vars[m] = new_vars[m][center[0]-window[0]:center[0]+window[0],center[1]-window[1]:center[1]+window[1]]
            else:
                new_vars[m] = None
        return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    def visibility_phases(self, pad_x=4, crop_x=10):
        """Get the complex visibilities corresponding to the image
        @param pad_x Number of times by which to pad FFT to preserve resolution in desired area.
               Image will be padded with zeros to a total of N1*pad_x by N2*pad_x pixels.
        @param re_crop Crop image back to N1xN2 pixels in the center of the FFT'd version
        @return "Image" with complex visibilities of each Stokes parameter
        """
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                new_vars[m] = np.angle(_visibilities_from_image(self.__dict__[m], pad_x))
                if crop_x < pad_x:
                    center = [s//2 for s in new_vars[m].shape]
                    window = [s*crop_x//2 for s in self.__dict__[m].shape]
                    new_vars[m] = new_vars[m][center[0]-window[0]:center[0]+window[0],center[1]-window[1]:center[1]+window[1]]
            else:
                new_vars[m] = None
        return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    # Operators
    def _do_op(self, other, op):
        """Perform a math operation by iterating over relevant members and doing it with numpy
        TODO update any cache members we add
        """
        if type(other) == Image:
            new_vars = {}
            for m in Image.data_members:
                if self.__dict__[m] is not None and other.__dict__[m] is not None:
                    new_vars[m] = getattr(self.__dict__[m], op)(other.__dict__[m])
                else:
                    new_vars[m] = None
            return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                        tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                        init_type="multi_arrays_stokes")
        else:
            new_vars = {}
            for m in Image.data_members:
                if self.__dict__[m] is not None:
                    new_vars[m] = getattr(self.__dict__[m], op)(other)
                else:
                    new_vars[m] = None
            return Image(self.properties, new_vars['I'], new_vars['Q'], new_vars['U'], new_vars['V'],
                        tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                        init_type="multi_arrays_stokes")

    def _do_iop(self, other, op):
        """Perform a math operation by iterating over relevant members and doing it with numpy
        TODO update any cache members we add
        """
        if type(other) == Image:
            for m in Image.data_members:
                if self.__dict__[m] is not None and other.__dict__[m] is not None:
                    getattr(self.__dict__[m], op)(other.__dict__[m])
            return self
        else:
            for m in Image.data_members:
                if self.__dict__[m] is not None:
                    getattr(self.__dict__[m], op)(other)
            return self

    def __sub__(self, other):
        """Difference images or subtract a constant"""
        return self._do_op(other, "__sub__")
    def __add__(self, other):
        """Add image values or a constant factor"""
        return self._do_op(other, "__add__")
    def __mul__(self, other):
        """Multiply images (why?) or rescale by factor"""
        return self._do_op(other, "__mul__")
    def __truediv__(self, other):
        """Elementwise division or divide by factor"""
        return self._do_op(other, "__truediv__")
    def __isub__(self, other):
        """Difference images or subtract a constant"""
        return self._do_iop(other, "__isub__")
    def __iadd__(self, other):
        """Add image values or a constant factor"""
        return self._do_iop(other, "__iadd__")
    def __imul__(self, other):
        """Multiply images (why?) or rescale by factor"""
        return self._do_iop(other, "__imul__")
    def __itruediv__(self, other):
        """Elementwise division or divide by factor"""
        return self._do_iop(other, "__itruediv__")