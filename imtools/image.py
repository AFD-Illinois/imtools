
import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

from scipy.ndimage import gaussian_filter

import imtools.stats as stats

# TODO list:
# Standard way of representing a GRMHD model, camera, and e- parameters, instead of a dict
# All as one "model" or separate?

# General container for polarized (Stokes I,Q,U,V) theory (truth) images
# Not big or fancy like ehtim :)
# This also includes lots of derived quantities
class Image(object):
    data_members = ["I", "Q", "U", "V", "tau", "tauF", "unpol"]

    def __init__(self, properties, array1, array2=None, array3=None, array4=None, tau=None, tauF=None, unpol=None,
                    init_type="one_array_forward"):
        """Initialize an Image object from data.  Resulting "object" doesn't really have member functions, more like a C struct
        Object provides a standard format for functions that manipulate images

        @param: properties: dict of image properties.  No standard format on account of there's no standard format.
        Generally model and lib expect ipole stuff to be present, see ipole image format doc on Illinois wiki:
        https://github.com/AFD-Illinois/docs/wiki/Image-Format

        @param array1, array2, array3, array4: Stokes parameters in CGS. Contents interpreted based on init_type.
            one_array_forward: array of Stokes in index-first form i,j,S.
            one_array_backward: array in Stokes-first form S,i,j.
            multi_arrays_stokes: four arrays size i,j representing I,Q,U,V.
            multi_arrays_rl: NOT IMPLEMENTED

            In all cases, i, j, should "look correct" with North pointing upward and looking as observed from Earth,
            when plotted either with meshgrid, or imshow(origin='lower')

        @param init_type: see above
        @param tau: if not none, set optical depth member tau to this array
        @param tauF: if not none, set Faraday depth member tauF to this array
        @param unpol: if not none, add a version of stokes I with unpolarized transport
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

            # Magic number is muas per radian
            # TODO multiplication order?
            self.fov_muas_x = self.Dx / self.dsource * self.lunit * 2.06265e11
            self.fov_muas_y = self.Dy / self.dsource * self.lunit * 2.06265e11

            self.scale = properties['scale']
            self.t = properties['t']
        else:
            # Just guess size and have done
            self.nx = self.I.shape[0]
            self.ny = self.I.shape[1]
            self.scale = 1

        if 'evpa_0' in properties:
            self.evpa_0 = properties['evpa_0']
        else:
            # Guess that people probably follow theory convention
            print("Warning: guessing EVPA 0 point, EVPA may be off by 90deg")
            self.evpa_0 = "W"
    
    # Per-pixel transformations: return new image
    def blurred(self, fwhm=20):
        # Gaussian blur, fwhm in muas
        # Assumes square pixels because what monster wouldn't use those
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
        return np.sqrt(self.Qtot()**2 + self.Utot()**2) / self.Itot()
    def cpfrac_int(self):
        return self.Vtot() / self.Itot()
    def evpa_int(self):
        return (180./np.pi)*0.5*np.arctan2(self.Utot(), self.Qtot())

    # Average lpfrac with given blur (NOT zero-baseline, taken per-px and averaged)
    def lpfrac_av(self, blur=20, mask_zero=False):
        # TODO Jason *must* massage this, right?
        return np.mean(self.blurred(blur).lpfrac(mask_zero))
    
    def tauF_av(self):
        return np.mean(self.tauF)
    def tau_av(self):
        return np.mean(self.tau)

    # Basic quantities.  Note for multithreading we use e.g.
    # Image.get_t(particular_image)
    # So we can't always use members
    def get_t(self):
        return self.t
    def fluxtot(self):
        return np.sum(self.I) * self.scale
    def Itot(self):
        return np.sum(self.I)
    def Qtot(self):
        return np.sum(self.Q)
    def Utot(self):
        return np.sum(self.U)
    def Vtot(self):
        return np.sum(self.V)
    def N(self):
        return self.I.shape[0]*self.I.shape[1]
    
    # Return raw array, for stuff that expects just an ndarray
    def get_raw(self):
        return np.array([self.I, self.Q, self.U, self.V])

    # Programmatic Stokes access 0,1,2,3
    # Returns None on invalid input
    def get_stokes(self, n):
        if n == 0: return self.I
        if n == 1: return self.Q
        if n == 2: return self.U
        if n == 3: return self.V

    # Comparison quantities
    def mse(self, var1, var2):
        """MSE of one var vs animage2.
        Follows GRRT paper definition in dividing by the sum(var) of *this* image
        """
        return stats.mse(var1, var2)

    def mses(self, other):
        """MSE of this image vs another.
        Follows GRRT paper definition in dividing by the sum(var) of *this* image
        """
        return stats.mses(self, other)

    def ssim(self, var1, var2):
        """SSIM as defined in Gold et al eq 10-ish"""
        return stats.ssim(var1, var2)

    def ssims(self, other):
        """SSIM for each variable with another image"""
        return stats.ssims(self, other)

    def dssim(self, var1, var2):
        """Image dissimilarity DSSIM is 1/SSIM - 1"""
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
        """Image Normalized Cross-Correlation: no subtracted mean"""
        return stats.ncc(var1, var2)

    def nccs(self, other):
        """NCC for each variable with another image"""
        return stats.nccs(self, other)
    # TODO ad rels

    # Plotting convenience quantities
    def zero_mask(self):
        """Get a mask of just the zones where I is vanishingly small compared to the main image.
        Set them to nan to avoid printing BS lpfrac/cpfrac when dividing by ~<=0
        """
        #Imaskval = np.abs(self.I.min()) * 100.
        Imaskval = np.nanmax(self.I) / self.I.shape[0]**5
        return np.abs(self.I) < Imaskval
    
    def ring_mask(self, high=False):
        if high:
            return self.I > np.mean(self.I) + np.std(self.I)
        else:
            return self.I > np.mean(self.I)

    def extent(self, fov_units):
        if fov_units == "muas":
            return [ -self.fov_muas_x/2, self.fov_muas_x/2, -self.fov_muas_y/2, self.fov_muas_y/2 ]
        elif fov_units == "M":
            return [ -self.Dx/2, self.Dx/2, -self.Dy/2, self.Dy/2 ]
        else:
            print("! unrecognized units for FOV {}. quitting.".format(fov_units))

    def scale_flux(self, units):
        if units == "cgs":
            return 1
        elif units == "Jy/px":
            return self.scale

    # Nonvolatile Operations
    def rel_diff(self, other, clip=None):
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

    # TODO come back and do Q,U correctly
    def rot90(self, rot):
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                self.__dict__[m] = np.rot90(self.__dict__[m], rot)
        self.Q *= -1
        self.U *= -1

    def rotated(self, rot):
        new_vars = {}
        for m in Image.data_members:
            if self.__dict__[m] is not None:
                new_vars[m] = np.rot90(self.__dict__[m], rot)
            else:
                new_vars[m] = None

        return Image(self.properties, new_vars['I'], -new_vars['Q'], -new_vars['U'], new_vars['V'],
                    tau=new_vars['tau'], tauF=new_vars['tauF'], unpol=new_vars['unpol'],
                    init_type="multi_arrays_stokes")

    # Operators
    def do_op(self, other, op):
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

    def do_iop(self, other, op):
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
        return self.do_op(other, "__sub__")
    def __add__(self, other):
        """Add image values or a constant factor"""
        return self.do_op(other, "__add__")
    def __mul__(self, other):
        """Multiply images (why?) or rescale by factor"""
        return self.do_op(other, "__mul__")
    def __truediv__(self, other):
        """Elementwise division or divide by factor"""
        return self.do_op(other, "__truediv__")
    def __isub__(self, other):
        """Difference images or subtract a constant"""
        return self.do_iop(other, "__isub__")
    def __iadd__(self, other):
        """Add image values or a constant factor"""
        return self.do_iop(other, "__iadd__")
    def __imul__(self, other):
        """Multiply images (why?) or rescale by factor"""
        return self.do_iop(other, "__imul__")
    def __itruediv__(self, other):
        """Elementwise division or divide by factor"""
        return self.do_iop(other, "__itruediv__")