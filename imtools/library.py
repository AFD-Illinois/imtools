"""
Reference for the locations of organized images based on their properties
"""

import os
import h5py
import re
import pickle
import numpy as np
import glob

from imtools.image import Image
from imtools.io import read_image
from imtools.parallel import map_parallel, iter_parallel

MAX_N_IMAGES = 4000
N_PROCS = 15


class ImageSet(object):
    """An ImageSet represents a folder containing images from a set of 60 models, sharing
    all of the same imaging choices and differing only in underlying physics.
    Thus the only parameters it needs to distinguish models are:
    BH flux level MAD/SANE
    BH spin
    Electron model, usually parameterized solely by Rhigh

    If you have more images, see Library
    """
    canon_fluxes = ("MAD", "SANE")
    canon_spins = ("-0.94", "-0.5", "0.0", "0.5", "0.94")
    canon_rhighs = ("1", "10", "20", "40", "80", "160")

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
    
    def get_fnames_structured(self, flux, spin, rhigh):
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
    
    # These should all be backend-independent as they manipulate images or collections
    # TODO time-based gets: get image closest to physical time, etc
    def get_image(self, flux, spin, rhigh, nimg):
        imgname = self.get_fname(flux, spin, rhigh, nimg)
        if imgname is None:
            return None
        return read_image(imgname)

    # Parallel operations
    def average_image(self, flux, spin, rhigh, nprocs=N_PROCS):
        """Return the "average" image of a run, summing all Stokes of all images and dividing
        This is not good for polarized images: the average will not be representative
        """
        def merge(n, other, output):
            output += other

        imgnames = self.get_all_fnames(flux, spin, rhigh)
        image = read_image(imgnames[0])
        iter_parallel(read_image, merge, imgnames[1:], image, nprocs=nprocs)
        image /= len(imgnames)

        return image
    
    def run_on(self, flux, spin, rhigh, fn, nprocs=N_PROCS):
        """Apply a function to every existing image in a model
        @return a list of results
        """
        read_and_fn = lambda imname: fn(read_image(imname)) 
        return map_parallel(read_and_fn, self.get_all_fnames(flux, spin, rhigh), nprocs)

    def run_lc(self, flux, spin, rhigh, fn, nprocs=N_PROCS):
        """Apply a function to every image in a model, and also return the simulation time of the image
        @return a list of tuples (t, fn(image))
        """
        fn_and_t = lambda image: (image.t, fn(image))
        read_and_fn_and_t = lambda imname: fn_and_t(read_image(imname))
        zipped = map_parallel(read_and_fn_and_t, self.get_all_fnames(flux, spin, rhigh), nprocs)
        # return lists of t and fn(images) as the 2 elements of a list, rather than a single list of tuples
        return list(zip(*zipped))

    # Option to use only the serial versions
    # def average_image(self, flux, spin, rhigh):
    #     return self.average_image_serial(flux, spin, rhigh)

    # def run_on(self, flux, spin, rhigh, fn):
    #     return self.run_on_serial(flux, spin, rhigh, fn)

    # def run_lc(self, flux, spin, rhigh, fn):
    #     return self.run_lc_serial(flux, spin, rhigh, fn)

    # Serial versions
    def average_image_serial(self, flux, spin, rhigh):
        imgnames = self.get_all_fnames(flux, spin, rhigh)
        image = read_image(imgnames[0])
        for i in range(1,len(imgnames)):
            image += read_image(imgnames[i])
        image /= len(imgnames)
        return image
    
    def run_on_serial(self, flux, spin, rhigh, fn):
        results = []
        for imname in self.get_all_fnames(flux, spin, rhigh):
            results.append(fn(read_image(imname)))
        return results
    
    def run_lc_serial(self, flux, spin, rhigh, fn):
        results = []
        ts = []
        for imname in self.get_all_fnames(flux, spin, rhigh):
            image = read_image(imname)
            results.append(fn(image))
            ts.append(image.t)
            del image
        return (ts, results)


class Library(object):
    """A Library is any collection of images.  It keeps track of all parameters with a Model object.
    Images are stored in a database indexed by Model and dump number
    """