import os
import sys
import itertools

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py

# Plotting functions
# Note that most of these functions rotate the resulting image automatically
# That is, their X,Y convention is 90 degrees from imshow's
def plot_image(ax, image, clabel=True, **kwargs):
    im = ax.pcolormesh(image, **kwargs)
    cbar = plt.colorbar(im, ax=ax)
    if clabel:
        cbar.set_label("Flux/px (Jy)")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_aspect('equal')
    ax.grid(False)

def plot_raw_stokes(ax, stokes, tag="", relative=False, **kwargs):
    """Plot the raw Stokes parameters on a set of 4 axes"""
    ax = ax.flatten()
    for i in range(4):
        var = stokes[:,:,i]
        max_abs = min(max(np.abs(np.max(var)), np.abs(np.min(var))),1e3)
        on_right = (i == 3)
        if i == 0 and not relative:
            plot_image(ax[i], var, cmap='afmhot', clabel=on_right, **kwargs)
        else:
            plot_image(ax[i], var, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, clabel=on_right, **kwargs)
        ax[i].set_title(tag + " Stokes " + ["I", "Q", "U", "V"][i])

def plot_quiver(ax, stokes, tag="", n_evpa=32, scaled=False, only_ring=False, **kwargs):
    """Plot Stokes I with overlaid quiver plot of linear polarization.
    Optionally scaled or (TODO) colored by total polarization or (TODO) polarization fraction,
    or overlaid only where emission is greatest
    """
    plot_image(ax, stokes[:,:,0], **kwargs)
    quiver_evpa(ax, stokes, n_evpa=n_evpa, scaled=scaled, only_ring=only_ring)

    # Assemble an okay default title
    plt.title(tag + (" scaled " if scaled else " constant ") + "quiver plot") # TODO note ring-only?

def quiver_evpa(ax, stokes, n_evpa=32, scaled=False, only_ring=False):
    """Superimpose EVPA as a quiver plot, either scaled or (TODO) colored by polarization fraction.
    """
    if stokes.shape[0] < n_evpa:
        return
    
    # TODO quiver is too long...
    Is, Qs, Us, Vs = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
    evpa = 0.5 * np.arctan2(Us, Qs)

    if scaled:
        # Scaled to polarization fraction
        amp = np.sqrt(Qs ** 2 + Us ** 2)
        scal = np.max(amp) # TODO consistent scaling option for animations
        vx = np.sqrt(Qs**2 + Us**2)*np.cos(evpa)/scal
        vy = np.sqrt(Qs**2 + Us**2)*np.sin(evpa)/scal
    else:
        # Normalized (evpa only)
        vx = np.cos(evpa)
        vy = np.sin(evpa)

    skip = int(Is.shape[0] / n_evpa)
    if only_ring:
        slc = np.where(Is[::skip, ::skip] > np.mean(Is) + np.std(Is))
    else:
        slc = (slice(None), slice(None))

    i, j = np.meshgrid(range(Is.shape[0]),range(Is.shape[1]))

    ax.quiver(i[::skip, ::skip][slc], j[::skip, ::skip][slc], vx[::skip, ::skip][slc],
               vy[::skip, ::skip][slc], headwidth=1, headlength=1)

def blur(im, fwhm=20):
    # 20uas FWHM Gaussian blur. Assume 1px/uas
    return gaussian_filter(im, sigma=(fwhm / (2 * np.sqrt(2 * np.log(2)))))

# Statistics functions
def mse(image1, image2):
    """MSE of image1 vs image2.  As in GRRT paper definition of said, divides by image1 sum"""
    # Avoid dividing by 0
    norm = np.sum(np.abs(image1)**2)
    if norm == 0: norm = 1

    return np.sum(np.abs(image1 - image2)**2) / norm

def ssim(imageI, imageK):
    """SSIM as defined in Gold et al eq 10-ish"""
    N = imageI.shape[0] * imageI.shape[1]
    mu_I = np.sum(imageI) / N
    mu_K = np.sum(imageK) / N
    sig2_I = np.sum((imageI - mu_I)**2) / (N-1)
    sig2_K = np.sum((imageK - mu_K)**2) / (N-1)
    sig_IK = np.sum((imageI - mu_I)*(imageK - mu_K)) / (N-1)
    return (2*mu_I*mu_K) / (mu_I**2 + mu_K**2) * (2*sig_IK) / (sig2_I + sig2_K)

def dssim(imageI, imageK):
    tssim = ssim(imageI, imageK)
    if np.isnan(tssim):
        return 0.0
    else:
        return 1/np.abs(tssim) - 1

def generate_comparison(ax, code_dict, code1name, code2name, include_diff=True, scale=False, vmax=1.0e-3):
    dcode1 = code_dict[code1name][:,:,0]
    dcode2 = code_dict[code2name][:,:,0]
    if scale:
        scalefac = np.mean(dcode2)/np.mean(dcode1)
    else:
        scalefac = 1
    
    params = {'cmap':'jet', 'clabel':True, 'vmin':0, 'vmax':vmax}
    plot_image(ax[0], dcode1*scalefac, **params)
    ax[0].set_title(code1name)
    plot_image(ax[1], dcode2, **params)
    ax[1].set_title(code2name)

    if include_diff:
        plot_image(ax[2], dcode1*scalefac - dcode2, cmap='RdBu_r', clabel=True)
        #plot_image(ax[2], np.abs(dcode1*scalefac - dcode2), cmap='jet', clabel=True)
        ax[2].set_title("Difference")

        plot_image(ax[3], np.clip((dcode1*scalefac - dcode2)/dcode2,-1,1), cmap='RdBu_r', clabel=True)
        ax[3].set_title("Relative Difference")

    #print("Ftot {}: {}".format(code1name, np.sum(dcode1)))
    #print("Ftot {}: {}".format(code2name, np.sum(dcode2)))
    print("{} - {} MSE in I is {}".format(code1name, code2name, mse(dcode1*scalefac, dcode2)))
    
# Data functions
def load_data_ipole(fname, n_stokes=4):
    file_ipole = h5py.File(fname, 'r')
    scale = file_ipole['header/scale'][()] # Convert to Jy
    print("Importing ipole data using CGS->Jy scale {}".format(scale))
    if n_stokes == 1:
        data_ipole = file_ipole['unpol'][:,:].transpose(1,0)*scale
        data_ipole = data_ipole[:,:,None]
    else:
        data_ipole = file_ipole['pol'][:,:,:n_stokes].transpose(1,0,2)*scale
    file_ipole.close()
    return np.nan_to_num(data_ipole)

# Data/input functions
def load_data(prob, variant="", codes=("ipole", "grtrans"), basedir="..", imname="image.h5", load_stokes=True):
    """Load each code's data for one problem (/variant)
    returns dict with code names, each element of which is an ndarray, [Npx,Npx,4],
    containing each Stokes flux through each pixel in Jy
    """
    data = {}
    test_dir = os.path.expanduser(basedir)
    if load_stokes:
        n_stokes = 4
    else:
        n_stokes = 1
    
    for code in codes:
        if code == "ipole":
            fname_ipole = os.path.join(test_dir,prob,code,"test_thin_disk.dat")
            _,_,_,I,Q,U,V = np.loadtxt(fname_ipole, unpack=True)
            ImRes = int(np.sqrt(len(I)))
            data[code] = np.array([I,-Q,-U,V]).reshape(n_stokes,ImRes,ImRes).transpose(2,1,0)
        
        elif "ipole" in code:
            ### CODE: IPOLE ###
            fname_ipole = os.path.join(test_dir,prob,code,imname.replace(".h5", variant+".h5"))
            #print("Loading {}".format(fname_ipole))
            data[code] = load_data_ipole(fname_ipole, n_stokes)

        elif "grtrans" in code:
            ### CODE: GRTRANS ###
            try:
                # TODO fits output like grtrans usually does when not coaxed
                file_grtrans = h5py.File(os.path.join(test_dir,prob,code,variant,imname), 'r')
                # Grtrans output is in the form [stokes, px_num, freq].  We don't care about the last one and want to split the second one
                # Then we want stokes index last
                # python wrapper for grtrans ensures this is already in Jy, also note grtrans will only output n_stokes of full matrix
                ImRes = int(np.sqrt(len(file_grtrans['ivals'][:,:,0].flatten())/n_stokes))
                data[code] = file_grtrans['ivals'][:,:,0].reshape(n_stokes,ImRes,ImRes).transpose(2,1,0)
                file_grtrans.close()
                # Correct Q,U convention
                if n_stokes > 1:
                    data[code][:,:,1] *= -1
                if n_stokes > 2:
                    data[code][:,:,2] *= -1
            except OSError as e:
                print("Not loading grtrans: {}".format(e))
        
        elif ("odyssey" in code) or ("raptor" in code):
            fil = os.path.join(test_dir,prob,code,"test3"+variant+"_output.txt")
            i0, j0, I, Q, U, V = np.loadtxt(fil, unpack=True)
            ImRes = int(np.sqrt(len(i0)))
            if "odyssey" in code:
                data[code] = np.array([I,Q,U,V]).reshape(4,ImRes,ImRes).transpose(1,2,0)
            else:
                data[code] = np.array([I,Q,U,V]).reshape(4,ImRes,ImRes).transpose(2,1,0)
            
        else:
            print("Not loading code: ",code)

    return data