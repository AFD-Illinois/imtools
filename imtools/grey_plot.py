# MMOS, June 25 2019
# maps a gray Stokes I map with colorful polarization ticks on the top, color codes fractional linear polarization
#style options: 'basic' plots tick size proportional to Stokes I, no color information
#                'vlbi' plots tick size proportional to P, color proportional to fractional pol m

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import ehtim as eh
import imtools.ehtim_compat as compat
from imtools.plots import _colorbar

def plot_I_greyscale(ax, im, style='basic', cfun='gray', title="",
                     rot_rm=0., beam_size=20., bar=False, scal=12e-6, zoom=1.0):

    im_loc = compat.to_eht_im(im)

    # limits
    Imax=max(im_loc.imvec)

    ax = plt.gca()
    pixel = im_loc.psize/eh.RADPERUAS #uas
    FOVx = pixel*im_loc.xdim
    FOVy = pixel*im_loc.ydim

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-FOVy/2, FOVy/2, pixel),
                    slice(-FOVx/2, FOVx/2, pixel)]

    # to brightness temperature
    #TBfactor = 3.254e13/(im_loc.rf**2 * im_loc.psize**2)/1e9 

    #total intensity in gray scale and dont show I below 0.01*Imax    
    imarr = im_loc.imvec.reshape(im_loc.ydim, im_loc.xdim)
    icut=0.01
    imarr_m = np.ma.masked_where(imarr < icut * Imax, imarr)   

    # stokes I, masked and normalized   
    mesh = ax.pcolormesh(-x, -y, imarr_m/Imax, cmap='gray_r', vmin=0, vmax=1.)#*1.5)

    # map of fractional polarization, this isn't rescaled by TBfactor
    mfrac=(np.sqrt(im_loc.qvec**2+im_loc.uvec**2)/im_loc.imvec).reshape(im_loc.xdim, im_loc.ydim)
    #mask pol in regions where I<0.1*Imax
    pcut=0.1
    mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac)   

    # length of the tick ~ to Stokes sqrt(Q2+U2)
    amp = (np.sqrt(im_loc.qvec**2+im_loc.uvec**2))
    scal = max(amp)

    vx = (-np.sin(np.angle(im_loc.qvec + 1j * im_loc.uvec) / 2) * amp / scal).reshape(im_loc.ydim, im_loc.xdim)
    vy = ( np.cos(np.angle(im_loc.qvec + 1j * im_loc.uvec) / 2) * amp / scal).reshape(im_loc.ydim, im_loc.xdim)

    skip=10
    qv = ax.quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
               mfrac_m[::skip,::skip],
               headlength=0,
               headwidth = 1,
               pivot='mid',
               width=0.008,
               cmap='rainbow',
               linewidth=1,
               scale=16)

    qv.set_clim(0.,0.3)

    if bar == 1:
        cbar = _colorbar(qv)
        cbar.set_label(r'Fractional Polarization m')
        cbar.ax.tick_params(labelsize=30)

    empty_string_labels = ['']
    ax.set_xticklabels(empty_string_labels)
    ax.set_yticklabels(empty_string_labels)
    ax.tick_params(right=False, top=False, left=False, bottom=False)

    ax.set_aspect('equal')

    pic_fovx = FOVx/zoom
    pic_fovy = FOVy/zoom
    ax.set_xlim(-pic_fovx/2., pic_fovx/2.)
    ax.set_ylim(-pic_fovy/2., pic_fovy/2.) 

#     beam_radius=beam_size/2.    
#     circ = ax.Circle((-45.,-45.),beam_radius,facecolor='None', edgecolor='k',linewidth=3)
#     ax.add_patch(circ)

    #scale bar
    scale=False 
    if scale == True:
        scale_len = 40.
        startx = pic_fov/2. 
        endx = pic_fov/2. - scale_len 
        starty = -50.
        endy = -50.
        ax.plot([startx, endx],
                 [starty, endy],
                 color="k", lw=3) # plot a line
        ax.text(x=(startx+endx)/2.0, y=starty-5.,
                 s= str(int(scale_len)) + r" $\mu$as", color="k",
                 ha="center", va="center",fontsize=50)
    
    # Return the quiver plot so we can mess with it if we want
    return mesh, qv

def plot_polimg_jose(im, style='basic', cfun='gray', title="",
                     PDF=None,rot_rm=0.,beam_size=20.,bar=0,scal=12e-6):

    fontname='Latin Modern Roman'
    font = font_manager.FontProperties(family=fontname, style='normal', size=40.)
    plt.rc('font', family=fontname)

    im_loc = compat.to_eht_im(im)

    # limits
    Imax=max(im_loc.imvec)

    if bar==0:
        plt.figure(1,(13,13))
    if bar==1:
        plt.figure(1,(15,13))

       
    ax = plt.gca()
    pixel = im_loc.psize/eh.RADPERUAS #uas
    FOV = pixel*im_loc.xdim

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                    slice(-FOV/2, FOV/2, pixel)]

    # to brightness temperature
    #TBfactor = 3.254e13/(im_loc.rf**2 * im_loc.psize**2)/1e9 

    #total intensity in gray scale and dont show I below 0.01*Imax    
    imarr = im_loc.imvec.reshape(im_loc.ydim, im_loc.xdim)
    icut=0.01
    imarr_m = np.ma.masked_where(imarr < icut * Imax, imarr)   

    # stokes I, masked and normalized   
    plt.pcolormesh(-x, -y, imarr_m/Imax, cmap='gray_r', vmin=0, vmax=1.)#*1.5)

    # map of fractional polarization, this isn't rescaled by TBfactor
    mfrac=(np.sqrt(im_loc.qvec**2+im_loc.uvec**2)/im_loc.imvec).reshape(im_loc.xdim, im_loc.ydim)
    #mask pol in regions where I<0.1*Imax
    pcut=0.1
    mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac)   

    # length of the tick ~ to Stokes sqrt(Q2+U2)
    amp = (np.sqrt(im_loc.qvec**2+im_loc.uvec**2))
    scal = max(amp)

    vx = (-np.sin(np.angle(im_loc.qvec + 1j * im_loc.uvec) / 2) * amp / scal).reshape(im_loc.ydim, im_loc.xdim)
    vy = ( np.cos(np.angle(im_loc.qvec + 1j * im_loc.uvec) / 2) * amp / scal).reshape(im_loc.ydim, im_loc.xdim)

    skip=6
    plt.quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
               mfrac_m[::skip,::skip],
               headlength=0,
               headwidth = 1,
               pivot='mid',
               width=0.008,
               cmap='rainbow',
               linewidth=1,
               scale=16)

    plt.clim(0.,0.3)

    if bar == 1:
        cbar = plt.colorbar(pad=0.0,shrink=0.79) 
        cbar.set_label(r'Fractional Polarization m', fontname=fontname, fontsize=40.,)
        cbar.ax.tick_params(labelsize=30) 


    empty_string_labels = ['']
    ax.set_xticklabels(empty_string_labels)
    ax.set_yticklabels(empty_string_labels)
    ax.tick_params(right=False, top=False, left=False, bottom=False)

    ax.set_aspect('equal')

    pic_fov=120.
    ax.set_xlim(pic_fov/2.,-pic_fov/2.)
    ax.set_ylim(-pic_fov/2.,pic_fov/2.) 

#     beam_radius=beam_size/2.    
#     circ = plt.Circle((-45.,-45.),beam_radius,facecolor='None', edgecolor='k',linewidth=3)
#     ax.add_patch(circ)

    #scale bar
    scale=False 
    if scale == True:
        scale_len = 40.
        startx = pic_fov/2. 
        endx = pic_fov/2. - scale_len 
        starty = -50.
        endy = -50.
        plt.plot([startx, endx],
                 [starty, endy],
                 color="k", lw=3) # plot a line
        plt.text(x=(startx+endx)/2.0, y=starty-5.,
                 s= str(int(scale_len)) + r" $\mu$as", color="k",
                 ha="center", va="center",fontsize=50)

    plt.tight_layout()

    plt.text(45,45,title,fontsize=30)

    if PDF != None:
        plt.savefig(PDF)
    plt.show()