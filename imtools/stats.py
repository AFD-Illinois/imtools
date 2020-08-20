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
    return np.array([mse(image1.I, image2.I),
                    mse(image1.Q, image2.Q),
                    mse(image1.U, image2.U),
                    mse(image1.V, image2.V)])

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
    return np.array([ssim(image1.I, image2.I),
                    ssim(image1.Q, image2.Q),
                    ssim(image1.U, image2.U),
                    ssim(image1.V, image2.V)])

def dssim(var1, var2):
    """Image dissimilarity DSSIM is 1/|SSIM| - 1"""
    tssim = ssim(var1, var2)
    if np.isnan(tssim):
        return 0.0
    else:
        return 1/np.abs(tssim) - 1

def dssims(image1, image2):
    """DSSIM for each variable"""
    return np.array([dssim(image1.I, image2.I),
                    dssim(image1.Q, image2.Q),
                    dssim(image1.U, image2.U),
                    dssim(image1.V, image2.V)])

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
    return np.array([zncc(image1.I, image2.I),
                    zncc(image1.Q, image2.Q),
                    zncc(image1.U, image2.U),
                    zncc(image1.V, image2.V)])

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
    return np.array([ncc(image1.I, image2.I),
                    ncc(image1.Q, image2.Q),
                    ncc(image1.U, image2.U),
                    ncc(image1.V, image2.V)])

def rel_integrated(var1, var2):
    return np.sum(var2) / np.sum(var1) - 1

def rels_integrated(image1, image2):
    """NCC for each variable"""
    return np.array([rel_integrated(image1.I, image2.I),
                    rel_integrated(image1.Q, image2.Q),
                    rel_integrated(image1.U, image2.U),
                    rel_integrated(image1.V, image2.V)])