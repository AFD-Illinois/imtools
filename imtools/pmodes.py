import h5py
import numpy as np
import ehtim as eh

def get_rex(im, uniq, tag, datadir_rex="rexdata/", rerun_rex=False):
    """Get rex-centered image and stats from image+tag.
    :param datadir_rex: folder where profiles and rex-centered images should go
    :param rerun_rex: if true, ignore previously-computed profiles
    """
    # create data directories
    if not os.path.exists(datadir_rex):
        os.makedirs(datadir_rex)

    # Run eht-imaging's rex algorithm on an image
    pp = rex.FindProfileSingle(im, out_dir=datadir_rex,
            out=uniq, save_files=True, rerun=rerun_rex, tag=tag)
    imcent = eh.image.load_image(datadir_rex+uniq+tag+"_cent.fits")
    with open(datadir_rex+uniq+tag+".txt",'r') as dfp:
        rexdata = dfp.readlines()
    diam = float(rexdata[2].split(' ')[1])
    width = float(rexdata[10].split(' ')[1])
    return imcent, diam, width

def get_pmodes(im, ms, diam=20., width=20., width_coeff=2.,
                norm_with_StokesI=True, norm_in_int=False):
    """Run pmodes and return beta coefficients."""
    minr = (diam - width_coeff*width)/2.
    maxr = (diam + width_coeff*width)/2. 
    out = pm.pmodes(im, ms, r_min=minr, r_max=maxr,
            norm_in_int=norm_in_int, 
            norm_with_StokesI=norm_with_StokesI)
  return out

def process(fname, bluruas=0, ms=None, width_coeff=2., 
        norm_with_StokesI=True, norm_in_int=False):
  """Run the full pmodes pipeline on an individual image (specified by file path)."""

    # get defaults
    if ms is None: 
        ms = np.array([2]) # The important mode

    # get unique tags
    uniq = get_unique_name(fname)
    tag = "_blur{0:d}uas".format(bluruas)

    # load image, run rex, and run pmodes
    im = load_image(fname, bluruas=bluruas)

    imcent, diam, width = get_rex(im, uniq, tag)

    return get_pmodes(imcent, ms, diam=diam, width=width, 
                    width_coeff=width_coeff, norm_with_StokesI=norm_with_StokesI,
                    norm_in_int=norm_in_int)  

    return pmodes_out

def pmodes(im, ms, r_min=0, r_max=25, norm_in_int = False, norm_with_StokesI = True):
  """Return beta_m coefficients for m in ms within extent r_min/r_max."""

    if type(im) == eh.image.Image:
        npix = im.xdim
        iarr = im.ivec.reshape(npix, npix)
        qarr = im.qvec.reshape(npix, npix)
        uarr = im.uvec.reshape(npix, npix)
        varr = im.vvec.reshape(npix, npix)
        fov_muas = im.fovx()/eh.RADPERUAS

    else if type(im) == str:
        hfp = h5py.File(im,'r')
        DX = hfp['header']['camera']['dx'][()]
        dsource = hfp['header']['dsource'][()]
        lunit = hfp['header']['units']['L_unit'][()]
        scale = hfp['header']['scale'][()]
        pol = np.flip(np.copy(hfp['pol']).transpose((1,0,2)),axis=0) * scale
        hfp.close()
        fov_muas = DX / dsource * lunit * 2.06265e11
        npix = pol.shape[0]
        iarr = pol[:,:,0]
        qarr = pol[:,:,1]
        uarr = pol[:,:,2]
        varr = pol[:,:,3]

    else:
        DX = im.Dx
        dsource = im.dsource
        lunit = im.lunit
        scale = im.scale
        fov_muas = im.fov_muas_x
        #pol = np.flip(np.copy(hfp['pol']).transpose((1,0,2)),axis=0) * scale
        npix = im.nx
        iarr = im.I * scale
        qarr = im.Q * scale
        uarr = im.U * scale
        varr = im.V * scale

    parr = qarr + 1j*uarr
    normparr = np.abs(parr)
    marr = parr/iarr
    phatarr = parr/normparr
    pxi = (np.arange(npix)-0.01)/npix-0.5
    pxj = np.arange(npix)/npix-0.5
    mui = pxi*fov_muas
    muj = pxj*fov_muas
    MUI,MUJ = np.meshgrid(mui,muj)
    MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))

    # get angles measured East of North
    PXI,PXJ = np.meshgrid(pxi,pxj)
    angles = np.arctan2(-PXJ,PXI) - np.pi/2.
    angles[angles<0.] += 2.*np.pi

    # get flux in annulus
    tf = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

    # get total polarized flux in annulus
    pf = normparr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

    #get number of pixels in annulus
    npix = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].size

    #get number of pixels in annulus with flux >= some % of the peak flux
    ann_iarr = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ]
    peak = np.max(ann_iarr)
    num_above5 = ann_iarr[ann_iarr > .05* peak].size
    num_above10 = ann_iarr[ann_iarr > .1* peak].size

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
            coeff = prod[ (MUDISTS <= r_max) & (MUDISTS >= r_min) ].sum()
            coeff /= npix
        else:
            prod = parr * pbasis
            coeff = prod[ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()
            if norm_with_StokesI:
                coeff /= tf
            else:
                coeff /= pf
            betas.append(coeff)

    if len(betas) == 1:
        return betas[0]
    else:
        return betas
