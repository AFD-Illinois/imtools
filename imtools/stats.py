"""
 File: stats.py
 
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

"""
Tools for calculating derived quantities (image2 images or reductions) based on an image.
Should all be in the form def q(image) to be accessible as image.
"""

# Comparison quantities
def mle(var1, var2):
    """Mean "Linear" error (normalized L1 norm)
    Follows GRRT paper definition in dividing by the sum(var) of *first* image
    """
    norm = np.sum(np.abs(var1))
    # Avoid dividing by 0
    if norm == 0:
        norm = 1
    return np.sum(np.abs(var1 - var2)) / norm

def mles(image1, image2):
    """MSE of this image vs animage2.
    Follows GRRT paper definition in dividing by the sum(var) of *first* image
    """
    return np.array([mle(image1.I * image1.scale, image2.I * image2.scale),
                    mle(image1.Q * image1.scale, image2.Q * image2.scale),
                    mle(image1.U * image1.scale, image2.U * image2.scale),
                    mle(image1.V * image1.scale, image2.V * image2.scale)])

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
    return np.array([mse(image1.I * image1.scale, image2.I * image2.scale),
                    mse(image1.Q * image1.scale, image2.Q * image2.scale),
                    mse(image1.U * image1.scale, image2.U * image2.scale),
                    mse(image1.V * image1.scale, image2.V * image2.scale)])

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
    return np.array([ssim(image1.I * image1.scale, image2.I * image2.scale),
                    ssim(image1.Q * image1.scale, image2.Q * image2.scale),
                    ssim(image1.U * image1.scale, image2.U * image2.scale),
                    ssim(image1.V * image1.scale, image2.V * image2.scale)])

def dssim(var1, var2):
    """Image dissimilarity DSSIM is 1/|SSIM| - 1"""
    tssim = ssim(var1, var2)
    if np.isnan(tssim):
        return 0.0
    else:
        return 1/np.abs(tssim) - 1

def dssims(image1, image2):
    """DSSIM for each variable"""
    return np.array([dssim(image1.I * image1.scale, image2.I * image2.scale),
                    dssim(image1.Q * image1.scale, image2.Q * image2.scale),
                    dssim(image1.U * image1.scale, image2.U * image2.scale),
                    dssim(image1.V * image1.scale, image2.V * image2.scale)])

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
    return np.array([zncc(image1.I * image1.scale, image2.I * image2.scale),
                    zncc(image1.Q * image1.scale, image2.Q * image2.scale),
                    zncc(image1.U * image1.scale, image2.U * image2.scale),
                    zncc(image1.V * image1.scale, image2.V * image2.scale)])

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
    return np.array([ncc(image1.I * image1.scale, image2.I * image2.scale),
                    ncc(image1.Q * image1.scale, image2.Q * image2.scale),
                    ncc(image1.U * image1.scale, image2.U * image2.scale),
                    ncc(image1.V * image1.scale, image2.V * image2.scale)])

def rel_integrated(var1, var2):
    """Relative error of summed value in a variable"""
    return np.sum(var2) / np.sum(var1) - 1

def rels_integrated(image1, image2):
    """Integrated relative errors for each variable"""
    return np.array([rel_integrated(image1.I * image1.scale, image2.I * image2.scale),
                    rel_integrated(image1.Q * image1.scale, image2.Q * image2.scale),
                    rel_integrated(image1.U * image1.scale, image2.U * image2.scale),
                    rel_integrated(image1.V * image1.scale, image2.V * image2.scale)])

def polar_rels_integrated(image1, image2):
    return np.array([rel_integrated(image1.I * image1.scale, image2.I * image2.scale),
                    image2.lpfrac_int() / image1.lpfrac_int() - 1,
                    image2.evpa_int() - image1.evpa_int(),
                    image2.cpfrac_int() / image1.cpfrac_int() - 1])

def polar_abs_integrated(image1, image2):
    return np.array([image2.flux() - image1.flux(),
                    image2.lpfrac_int() - image1.lpfrac_int(),
                    image2.evpa_int() - image1.evpa_int(),
                    image2.cpfrac_int() - image1.cpfrac_int()])