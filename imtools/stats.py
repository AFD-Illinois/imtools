# Tools for calculating derived quantities (image2 images or reductions) based on an image.
# Should all be in the form def q(image) to be accessible as image.

import numpy as np

# Comparison quantities
def mse(var1, var2):
    """MSE between images.
    Follows GRRT paper definition in dividing by the sum(var) of *first* image
    """
    norm = np.sum(np.abs(var1)**2)
    # Avoid dividing by 0
    if norm == 0:
        norm = 1
    return np.sum(np.abs(var1 - var2)**2) / norm

def mses(image1, image2):
    """MSE of this image vs animage2.
    Follows GRRT paper definition in dividing by the sum(var) of *first* image
    """
    mses = []
    mses.append(mse(image1.I, image2.I))
    mses.append(mse(image1.Q, image2.Q))
    mses.append(mse(image1.U, image2.U))
    mses.append(mse(image1.V, image2.V))
    return mses

def ssim(var1, var2):
    """Image similarity SSIM as defined in Gold et al eq 14"""
    N = var1.shape[0] * var1.shape[1]
    mu_1 = np.mean(var1)
    mu_2 = np.mean(var2)
    sig2_1 = np.sum((var1 - mu_1)**2) / (N-1)
    sig2_2 = np.sum((var2 - mu_2)**2) / (N-1)
    sig_12 = np.sum((var1 - mu_1)*(var2 - mu_2)) / (N-1)
    return (2*mu_1*mu_2) / (mu_1**2 + mu_2**2) * (2*sig_12) / (sig2_1 + sig2_2)

def ssims(image1, image2):
    """SSIM for each variable"""
    ssims = []
    ssims.append(ssim(image1.I, image2.I))
    ssims.append(ssim(image1.Q, image2.Q))
    ssims.append(ssim(image1.U, image2.U))
    ssims.append(ssim(image1.V, image2.V))
    return ssims

def dssim(var1, var2):
    """Image dissimilarity DSSIM is 1/|SSIM| - 1"""
    tssim = ssim(var1, var2)
    if np.isnan(tssim):
        return 0.0
    else:
        return 1/np.abs(tssim) - 1

def dssims(image1, image2):
    """DSSIM for each variable"""
    dssims = []
    dssims.append(dssim(image1.I, image2.I))
    dssims.append(dssim(image1.Q, image2.Q))
    dssims.append(dssim(image1.U, image2.U))
    dssims.append(dssim(image1.V, image2.V))
    return dssims

def zncc(var1, var2):
    """Zero-Normalized Cross-Correlation ZNCC
    (normalized correlation of deviation from mean)
    """
    N = var1.shape[0] * var1.shape[1]
    sigma_1 = np.std(var1)
    sigma_2 = np.std(var2)
    mu_1 = np.mean(var1)
    mu_2 = np.mean(var2)
    return 1/N * np.sum(1/(sigma_1 * sigma_2) * (var1 - mu_1)*(var2 - mu_2))

def znccs(image1, image2):
    """ZNCC for each variable"""
    znccs = []
    znccs.append(zncc(image1.I, image2.I))
    znccs.append(zncc(image1.Q, image2.Q))
    znccs.append(zncc(image1.U, image2.U))
    znccs.append(zncc(image1.V, image2.V))
    return znccs

def ncc(var1, var2):
    """Normalized Cross-Correlation NCC
    (like ZNCC without mean subtraction)
    """
    N = var1.shape[0] * var1.shape[1]
    sigma_1 = np.std(var1)
    sigma_2 = np.std(var2)
    return 1/N * np.sum(1/(sigma_1 * sigma_2) * var1*var2)

def nccs(image1, image2):
    """NCC for each variable"""
    nccs = []
    nccs.append(ncc(image1.I, image2.I))
    nccs.append(ncc(image1.Q, image2.Q))
    nccs.append(ncc(image1.U, image2.U))
    nccs.append(ncc(image1.V, image2.V))
    return nccs

def rel_integrated(var1, var2):
    return np.sum(var2) / np.sum(var1) - 1

def rels_integrated(image1, image2):
    """NCC for each variable"""
    rels = []
    rels.append(rel_integrated(image1.I, image2.I))
    rels.append(rel_integrated(image1.Q, image2.Q))
    rels.append(rel_integrated(image1.U, image2.U))
    rels.append(rel_integrated(image1.V, image2.V))
    return rels