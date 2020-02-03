
import numpy as np
import h5py
import matplotlib.pyplot as plt

from imtools.model import Model

# General container for polarized (Stokes I,Q,U,V) theory (truth) images
# Not big or fancy like ehtim :)
# Split this into IImage for intensity if you must
class Image(object):

    def __init__(self, file, model=None):
        if isinstance(file, str):
            self.manage_file = True
            if file[-3:] == ".h5":
                self.fname = file
                file = h5py.File(file, "r")
            elif file[-4:] == ".dat":
                self.fname = file
                file_dat = np.loadtxt(file).T
                if file_dat.shape[0] == 7:
                    imres = np.sqrt(file_dat.shape[1])
                    file = {'unpol':file_dat[2].reshape(imres,imres),
                            'pol': file_dat[3:7].reshape(4,imres,imres)}
                elif file_dat.shape[0] == 6:
                    imres = np.sqrt(file_dat.shape[1])
                    file = {'unpol':file_dat[2].reshape(imres,imres),
                            'pol': file_dat[2:6].reshape(4,imres,imres)}
        elif isinstance(file, h5py.File):
            self.manage_file = False
            self.fname = file.filename
        elif isinstance(file, _io.TextIOWrapper):
            self.manage_file = False
            self.fname = file.
        else:
            raise ValueError("This is not an image!")

        self.data = file['pol'][:4]
        self.I = file['unpol'][()]
        if model is None:
            if 'header' in file:
                self.model = Model(file['header'])
            else:
                self.model = Model(self.fname)
                print("WARNING: Guessing model: {}".format(self.model))
        else:
            if isinstance(model, str):
                self.model = Model(model)
            else:
                self.model = model

    def plot(self, ax, clabel=True, **kwargs):
        im = ax.pcolormesh(self., **kwargs)
        cbar = plt.colorbar(im, ax=ax)
        if clabel:
            cbar.set_label("Flux/px (Jy)")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        ax.set_aspect('equal')
        ax.grid(False)

    def __del__(self):
        if self.manage_file