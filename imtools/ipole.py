__license__ = """
 File: ipole.py
 
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

import sys
import functools
import subprocess
import numpy as np
from scipy.optimize import minimize_scalar

"""Methods for running & interacting with ipole.  Original credit George Wong
"""

defaults = {
    'rcam': 1000.0,
    'phicam': 0.0,
    'rotcam': 0.0,
    'nx': 320,
    'ny': 320,
    'freqcgs': 2.3e+11,
    'trat_small': 1.0,
    'qu_conv': 0,
    'xoff': 0.0,
    'yoff': 0.0,
    'sigma_cut': 1.e20
}

defaults_m87 = {
    'thetacam': 17.0,
    'dsource': 1.69e+07,
    'fovx_dsource': 160.0,
    'fovy_dsource': 160.0,
    'MBH': 6.2e+09
}

# Remember must take thetacam as arg
defaults_sgra = {
    'dsource': 8127,
    'fovx_dsource': 200.0,
    'fovy_dsource': 200.0,
    'MBH': 4.14e+06
}

def write_pars(args, filename):
    # TODO general header comment
    with open(filename, 'w') as file:
        for item in args.keys():
            file.write(f"{item} {args[item]}\n")

def fit_munit_m87(exe, dump_name, target=0.5, trat_large=20, res=80, **kwargs):
    args = {**defaults, **defaults_m87}
    args['nx'] = res
    args['ny'] = res
    args['trat_large'] = trat_large
    return fit_munit(args, dump_name, target, exe, **kwargs)

def fit_munit_sgra(exe, dump_name, target=0.5, trat_large=20, res=80, **kwargs):
    args = {**defaults, **defaults_sgra}
    args['nx'] = res
    args['ny'] = res
    args['trat_large'] = trat_large
    return fit_munit(args, dump_name, target, exe, **kwargs)

def fit_munit(args, dump_name, target, exe="ipole", bounds=(1e20, 1e30), **kwargs):
    args['dump'] = dump_name
    def f(log_munit):
        args['M_unit'] = np.exp(log_munit)
        results = run_ipole(args, exe, pol=False, output=False, **kwargs)
        if 'pol' in kwargs and kwargs['pol']:
            res = abs(results['Ftot_pol'] - target)
        else:
            res = abs(results['Ftot_unpol'] - target)
        if 'verbose' in kwargs and kwargs['verbose'] > 0:
            print(f"Residual: {res}")
        return res
    # Now minimize f
    try:
        res = minimize_scalar(f, bracket=(1, np.log(bounds[0]), np.log(bounds[1])), tol=1e-2)
    except ValueError as e:
        print("Error in Brent fit:",e,"\nTrying loose bounded...")
        res = minimize_scalar(f, bounds=(1, np.log(1e200)), tol=1e-2)
    if 'verbose' in kwargs and kwargs['verbose'] >= 0:
        print(f"Fitted munit in {res.nit} steps")
    return np.exp(res.x)

def run_ipole_m87(args, exe, dump_name, outfile="image.h5", res=320, **kwargs):
        all_args = {**defaults, **defaults_m87, **args}
        all_args['dump'] = dump_name
        all_args['outfile'] = outfile
        all_args['nx'] = res
        all_args['ny'] = res
        parfile = outfile.replace(".h5", ".par")
        write_pars(all_args, parfile)
        return run_ipole({}, exe, parfile=parfile)

def run_ipole(args, exe="ipole", output=True, pol=True, parfile=None, verbose=0):
    """Runs ipole with config as specified by args."""

    cmd = [exe]
    if parfile is not None:
        cmd += ["-par", parfile]
    cmd += [f"--{key}={args[key]}" for key in args]

    if not output: cmd += ["-quench"]
    if not pol: cmd += ["-unpol"]

    if verbose > 1:
        print(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = [ z for y in [ str(x)[2:-1].split("\\n") for x in proc.communicate() ] for z in y ]

    if verbose > 2:
        print(output)

    results = {}
    for line in output:
        if verbose > 1: print(line)
        if "Ftot" in line:
            proc = line.replace('(','').replace(')','').split()
            results['Ftot_pol'] = float(proc[3])
            results['Ftot_unpol'] = float(proc[5])
    
    if (not 'Ftot_pol' in results) and (not 'Ftot_unpol' in results):
        print("\n".join(output), file=sys.stderr)
        raise ArgumentException("Bad call to ipole while fitting!")

    if verbose > 0:
        print(f"Munit: {args['M_unit']} Flux: {results['Ftot_unpol']}")

    return results

