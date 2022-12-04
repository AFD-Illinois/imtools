#!/usr/bin/env python3

# Usage: parameter-from-image.py /path/to/image.h5 > parameters.par
# Make a parameter file (minus the filename) from a finished image

# TODO:
# Support execution parameters (e.g. "add_ppm")
# Support tracing
# Support taking filenames on command line

import sys
import h5py

f = h5py.File(sys.argv[1], "r")

cgs = {
    'CL': 2.99792458e10,
    'GNEWT': 6.6742e-8,
    'PC': 3.085678e18,
    'MSOLAR': 1.989e33
}

print("rcam {}".format(f['/header/camera/rcam'][()]))
print("thetacam {}".format(f['/header/camera/thetacam'][()]))
print("phicam {}".format(f['/header/camera/phicam'][()]))
print("rotcam {}".format(f['/header/camera/rotcam'][()]))

print("nx {}".format(f['/header/camera/nx'][()]))
print("ny {}".format(f['/header/camera/ny'][()]))

print("dsource {:g}".format(f['/header/dsource'][()]/cgs['PC']))

print("fovx_dsource {}".format(f['/header/camera/fovx_dsource'][()]))
print("fovy_dsource {}".format(f['/header/camera/fovy_dsource'][()]))

print("freqcgs {:g}".format(f['/header/freqcgs'][()]))

print("MBH {:g}".format(f['/header/units/L_unit'][()]*cgs['CL']**2/cgs['GNEWT']/cgs['MSOLAR']))

print("M_unit {}".format(f['/header/units/M_unit'][()]))

if f['/header/electrons/type'][()] == 1:
    print("tp_over_te {}".format(f['/header/electrons/tp_over_te'][()]))
elif f['/header/electrons/type'][()] == 2:
    print("trat_small {}".format(f['/header/electrons/rlow'][()]))
    print("trat_large {}".format(f['/header/electrons/rhigh'][()]))

# TODO counterjet seems like an important thing to record...
#print("counterjet {}".format())

if f['/header/evpa_0'][()] == b'N':
    print("qu_conv {}".format(0))
elif f['/header/evpa_0'][()] == b'W':
    print("qu_conv {}".format(1))
else:
    print("Unrecognized Q,U convention! Not specifying in parameter file!", file=sys.stderr)

print("xoff {}".format(f['/header/camera/xoff'][()]))
print("yoff {}".format(f['/header/camera/yoff'][()]))

try:
	print("reverse_field {}".format(f['/header/field_config'][()]))
except KeyError:
	pass

# TODO replicate non-physical parameters optionally?
#add_ppm 0
#quench_output 0
#only_unpolarized 0
#trace parameters
