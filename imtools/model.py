
from enum import Enum
import h5py

default_model = {

}

class BField(Enum):
    MAD = 0
    SANE = 1
    INSANE = 2
    SEMIMAD = 3

class Model(object):
    """
    Defines a single imaging model: a set of physical parameters which change an image
    These include the following imaging parameters:
    * Fast/slow light approximation
    * Camera angle (thetacam and phicam in degrees)
    * FOV from earth (in muas)
    * Resolution (int for square images or tuple for (barely supported) rectangular)
    * Average flux
    *
    As well as the following physical model parameters:
    * B field flux (MAD/SANE)
    * Spin
    * Electron energy distribution (Rhigh=X or native)

    Also, these functions only support "new" images, i.e. those produced by ipole
    after Summer 2019
    """
    param_names = ['thetacam', 'fov', 'nx']

    def __init__(self, input, library=None):
        self.library = library

        if isinstance(input, h5py.Group):
            if 'header' in input:
                input = input['header']
            self.thetacam = input['camera/thetacam']
            self.fov = input['camera/fov']
            self.nx = input['nx']
            self.ny = input['ny']

            self.avg_flux = None # TODO SUPPORT THIS

        elif isinstance(input, dict):


        elif isinstance(input, str):




    def folder_relative(self):
        return "{}/{}/{}/{}/{}"

    def __eq__(self, other):


    def __repr__(self):
        return "{} model with {} electrons"