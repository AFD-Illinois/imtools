"""
 File: paperv_vals.py

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
"""
Contains GRMHD and munit fit information for the various runs performed for
  the 2017 EHT sequence papers, i.e., "The Library". Note that the values are
  for the runs used in the canonical analysis, and therefore only a subset of
  the resolutions are reported.

  The canonical values are taken to be:

    GRMHD:
      Ma+0.94_384x192x192_IHARM
      Ma+0.5_384x192x192_IHARM
      Ma0.94_384x192x192_IHARM
      Ma-0.5_384x192x192_IHARM
      Ma-0.94_384x192x192_IHARM
      Sa+0.94_288x128x128_IHARM
      Sa+0.5_288x128x128_IHARM
      Sa0_288x128x128_IHARM
      Sa-0.5_288x128x128_IHARM
      Sa-0.94_288x128x128_IHARM

    GRRT:
      Mbh = 6.2e9 Msun
      Ftot_{230GHz,compact} = 0.5 Jy
      DM87 = 16.9 Mpc
      inc = 163/17 degrees for positive/negative spin

  last modified: gnw 2019.11.17
"""

GRMHD = {
    'MAD': {
        '0.94' :
            { 'res':"384x192x192", 'dmin':1000, 'dmax':2000},
        '0.5'  :
            { 'res':"384x192x192", 'dmin':1000, 'dmax':2000},
        '0.0'     :
            { 'res':"384x192x192", 'dmin':1000, 'dmax':2000},
        '-0.5'  :
            { 'res':"384x192x192", 'dmin':1000, 'dmax':1800},
        '-0.94' :
            { 'res':"384x192x192", 'dmin':1400, 'dmax':2000},
    },
    'SANE': {
        '0.94' :
            { 'res':"288x128x128", 'dmin':600,  'dmax':1200},
        '0.5'  :
            { 'res':"288x128x128", 'dmin':600,  'dmax':1100},
        '0.0'     :
            { 'res':"288x128x128", 'dmin':1000, 'dmax':2000},
        '-0.5'  :
            { 'res':"288x128x128", 'dmin':1000, 'dmax':1600},
        '-0.94' :
            { 'res':"288x128x128", 'dmin':1200, 'dmax':1800},
    }
}

# TODO add different flux values and angles
GRRT = {
    'MAD': {
        '0.94' :
            [
            { '163' : [ 3.94933e+24, 6.08504e+24, 7.48745e+24, 9.50325e+24, 1.25564e+25, 1.75529e+25 ], }
            ],
        '0.5' :
            [
            { '163' : [ 5.61134e+24, 9.76006e+24, 1.23868e+25, 1.59542e+25, 2.10276e+25, 2.87962e+25 ], }
            ],
        '0.0' :
            [
            { '163' : [ 7.45201e+24, 1.23416e+25, 1.52312e+25, 1.90889e+25, 2.46114e+25, 3.31910e+25 ], }
            ],
        '-0.5' :
            [
            { '17'  : [ 1.03238e+25, 1.59578e+25, 1.93870e+25, 2.42680e+25, 3.16368e+25, 4.39173e+25 ], }
            ],
        '-0.94' :
            [
            { '17'  : [ 1.39753e+25, 2.08374e+25, 2.52678e+25, 3.19200e+25, 4.25303e+25, 6.07155e+25 ], }
            ],
    },
    'SANE': {
        '0.94' :
            [
            { '163' : [ 1.24732e+26, 7.95250e+26, 1.82523e+27, 3.38593e+27, 5.03176e+27, 6.9915e+27 ], }
            ],
        '0.5' :
            [
            { '163' : [ 6.46716e+26, 4.81158e+27, 1.03092e+28, 2.02097e+28, 3.20183e+28, 4.26347e+28 ], }
            ],
        '0.0' :
            [
            { '163' : [ 1.38083e+28, 2.70363e+29, 4.95542e+29, 5.65676e+29, 6.37130e+29, 7.5794e+29 ], }
            ],
        '-0.5' :
            [
            { '17'  : [ 1.21342e+28, 8.99534e+28, 1.04354e+29, 1.21020e+29, 1.47946e+29, 1.94533e+29 ], }
            ],
        '-0.94' :
            [
            { '17'  : [ 1.59892e+28, 6.83760e+28, 8.03224e+28, 9.71287e+28, 1.22955e+29, 1.65703e+29 ], }
            ]
    }
}
