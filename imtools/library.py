"""
Reference for the locations of organized images based on their properties
"""

import os
import h5py
import re
import pickle
import numpy as np
import glob

from imtools.io import read_image

MAX_N_IMAGES = 4000

class ImageSet(object):

    def __init__(self, basedir):
        self.basedir = basedir.rstrip("/")
        cachefile = "names"+self.basedir.replace("/","_")+".p"
        if os.path.exists(cachefile):
            with open(cachefile, "rb") as cf:
                self.names = pickle.load(cf)
        else:
            self.names = {}
            reg_4dnum = re.compile(r"_\d\d\d\d_")
            for fname in glob.iglob(os.path.join(basedir, "**", "image_*.h5"), recursive=True):
                # Skip even building the expensive Image object.  Just HDF5
                try:
                    im_h5 = h5py.File(fname, "r")
                    if '/fluid_header/geom/mks/a' in im_h5:
                        a = im_h5['/fluid_header/geom/mks/a'][()]
                    elif '/fluid_header/geom/mmks/a' in im_h5:
                        a = im_h5['/fluid_header/geom/mmks/a'][()]

                    if '/header/electrons/trat_large' in im_h5:
                        rhigh = int(im_h5['/header/electrons/trat_large'][()])
                    elif '/header/electrons/rhigh' in im_h5:
                        rhigh = int(im_h5['/header/electrons/rhigh'][()])

                    # File names are *so* not reliable for this, but the 5M library cadence is forever
                    nimg = int(np.floor(im_h5['/header/t'][()] / 5.0))

                    im_h5.close()
                except RuntimeError as e:
                    print("Failed to read ", fname.replace(self.basedir,"").lstrip("/"))

                # This only uses filename, not included in images
                if "SANE" in fname or "Sa+" in fname or "Sa-" in fname or "Sa0" in fname:
                    flux = "SANE"
                elif "MAD" in fname or "Ma+" in fname or "Ma-" in fname or "Ma0" in fname:
                    flux = "MAD"

                # Insert a new 
                key = flux + "/" + "{:.2}".format(a) + "/" + "{}".format(rhigh)
                if not key in self.names:
                    self.names[key] = np.zeros(MAX_N_IMAGES, dtype="S1024")
                
                self.names[key][nimg] = fname.replace(self.basedir, "").lstrip("/")

            with open(cachefile, "wb") as cf:
                pickle.dump(self.names, cf)

    def get_fname(self, flux, spin, rhigh, nimg):
        key = flux + "/" + spin + "/" + rhigh
        try:
            fpath = self.names[key][nimg].decode('utf-8').lstrip("/")
            if fpath == '':
                print("Image does not exist: model {} #{}".format(key, nimg))
                path = None
            else:
                path = os.path.join(self.basedir, fpath)
        except KeyError:
            print("Model not found: {}".format(key, nimg))
            path = None
        return path

    def get_all_fnames(self, flux, spin, rhigh):
        key = flux + "/" + spin + "/" + rhigh
        paths = []
        try:
            for nimg in range(MAX_N_IMAGES):
                fpath = self.names[key][nimg].decode('utf-8').lstrip("/")
                if fpath != '':
                    paths.append(os.path.join(self.basedir, fpath))
        except KeyError:
            print("Model not found: {}".format(key, nimg))
            paths = []
        return paths
