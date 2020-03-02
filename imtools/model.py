
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
    Defines a full collection of image and model parameters.
    This includes MAD/SANE, resolution, and spin of the original GRMHD run,
    as well as all parameters used to run the image, i.e. anything and everything
    recorded in header/ of HDF5 images produced by ipole.
    """

    