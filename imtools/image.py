
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# General container for polarized (Stokes I,Q,U,V) theory (truth) images
# Not big or fancy like ehtim :)
# This also includes lots of derived quantities
class Image(object):
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
        if tau is not None:
            self.tau = tau
        if tauF is not None:
            self.tauF = tauF
        if unpol is not None:
            self.unpol = unpol
        
        self.properties = properties

        # Size.  Could be obtained from np arrays but good to check header
        self.nx = properties['camera']['nx']
        self.ny = properties['camera']['ny']

        # FOV in M. Named historically...
        self.Dx = properties['camera']['dx']
        self.Dy = properties['camera']['dy']

        self.dsource = properties['dsource']
        self.lunit = properties['units']['L_unit']
        # Magic number is for 
        self.fov_muas_x = self.Dx / self.dsource * self.lunit * 2.06265e11
        self.fov_muas_y = self.Dy / self.dsource * self.lunit * 2.06265e11

        self.scale = properties['scale']

        if 'evpa_0' in properties:
            self.evpa_0 = properties['evpa_0']
        else:
            # Before this was
            print("Warning: guessing EVPA 0 point, EVPA may be off by 90deg")
            self.evpa_0 = "W"
        
        # Pre-calculate some sums
        self.Itot = np.sum(self.I)
        self.Qtot = np.sum(self.Q)
        self.Utot = np.sum(self.U)
        self.Vtot = np.sum(self.V)
    
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
        props = self.properties.copy()
        props['camera']['nx'] = I.shape[0]
        props['camera']['ny'] = I.shape[1]
        return Image(self.properties, I, Q, U, V, tau=tau, tauF=tauF, unpol=unpol, init_type="multi_arrays_stokes")

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
            # TODO check this...
            evpa -= 90.
            evpa[evpa < -90.] += 180.
        if mask_zero:
            evpa[self.zero_mask()] = np.nan
        return evpa
    
    # Integrated a.k.a zero-baseline quantities
    def lpfrac_int(self):
        return np.sqrt(self.Qtot**2 + self.Utot**2) / self.Itot
    def cpfrac_int(self):
        return self.Vtot / self.Itot
    def evpa_int(self):
        return (180./np.pi)*0.5*np.arctan2(self.Utot, self.Qtot)

    # Average lpfrac with given blur (NOT zero-baseline, taken per-px and averaged)
    def lpfrac_av(self, blur=20, mask_zero=False):
        # TODO Jason *must* massage this, right?
        return np.mean(self.blurred(blur).lpfrac(mask_zero))
    
    def tauF_av(self):
        return np.mean(self.tauF)
    def tau_av(self):
        return np.mean(self.tau)
    
    # Comparison quantities
    def mse(self, var1, var2):
        """MSE of one var vs another.
        Follows GRRT paper definition in dividing by the sum(var) of *this* image
        """
        # Avoid dividing by 0
        norm = np.sum(np.abs(var1)**2)
        if norm == 0: norm = 1
        return np.sum(np.abs(var1 - var2)**2) / norm
    def mses(self, other):
        """MSE of this image vs another.
        Follows GRRT paper definition in dividing by the sum(var) of *this* image
        """
        mses = []
        mses.append(self.mse(self.I, other.I))
        mses.append(self.mse(self.Q, other.Q))
        mses.append(self.mse(self.U, other.U))
        mses.append(self.mse(self.V, other.V))
        return mses

    def ssim(self, imageI, imageK):
        """SSIM as defined in Gold et al eq 10-ish"""
        N = imageI.shape[0] * imageI.shape[1]
        mu_I = np.sum(imageI) / N
        mu_K = np.sum(imageK) / N
        sig2_I = np.sum((imageI - mu_I)**2) / (N-1)
        sig2_K = np.sum((imageK - mu_K)**2) / (N-1)
        sig_IK = np.sum((imageI - mu_I)*(imageK - mu_K)) / (N-1)
        return (2*mu_I*mu_K) / (mu_I**2 + mu_K**2) * (2*sig_IK) / (sig2_I + sig2_K)
    def ssims(self, other):
        """SSIM for each variable with another image"""
        ssims = []
        ssims.append(self.ssim(self.I, other.I))
        ssims.append(self.ssim(self.Q, other.Q))
        ssims.append(self.ssim(self.U, other.U))
        ssims.append(self.ssim(self.V, other.V))
        return ssims

    def dssim(self, imageI, imageK):
        tssim = self.ssim(imageI, imageK)
        if np.isnan(tssim):
            return 0.0
        else:
            return 1/np.abs(tssim) - 1
    def dssims(self, other):
        """DSSIM for each variable with another image"""
        dssims = []
        dssims.append(self.dssim(self.I, other.I))
        dssims.append(self.dssim(self.Q, other.Q))
        dssims.append(self.dssim(self.U, other.U))
        dssims.append(self.dssim(self.V, other.V))
        return dssims

    # Plotting convenience quantities
    def zero_mask(self):
        """Get a mask of just the zones where I is vanishingly small compared to the main image.
        Set them to nan to avoid printing BS lpfrac/cpfrac when dividing by ~<=0
        """
        #Imaskval = np.abs(self.I.min()) * 100.
        Imaskval = np.nanmax(self.I) / self.I.shape[0]**5
        return np.abs(self.I) < Imaskval
    
    def ring_mask(self):
        return self.I > np.mean(self.I) + np.std(self.I)

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
    
    # Operators
    def __iadd__(self, other):
        self.I += other.I
        self.Q += other.Q
        self.U += other.U
        self.V += other.V
        # TODO if not none
        self.tau += other.tau
        self.tauF += other.tauF
        self.unpol += other.unpol
        return self
    
    def __itruediv__(self, other):
        """Divide image quantities. Only defined for scalars, only needed for averages"""
        self.I /= other
        self.Q /= other
        self.U /= other
        self.V /= other
        # TODO if not none
        self.tau /= other
        self.tauF /= other
        self.unpol /= other
        return self