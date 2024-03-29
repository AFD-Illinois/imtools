__license__ = """
 File: io.py
 
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
import h5py

from imtools.image import Image

__doc__ = \
"""Read all or parts of image files.  Supports reading data from a few types of files,
but full parameters and Image objects only from the ``ipole`` HDF5 format.
"""

def read(fname, **kwargs):
    """Read a file.  In future will try to intelligently guess image/movie/etc
    """
    return read_image(fname, **kwargs)


def read_image_parameters(fname, load_fluid_header=False):
    """Read just the parameters dictionary from an ``ipole`` image file.
    Useful if reading many images run with the same parameters.
    """
    infile = h5py.File(fname, "r")
    header = hdf5_to_dict(infile['header'])
    if load_fluid_header:
        header.update(hdf5_to_dict(infile['fluid_header']))
    for key in infile.keys():
        if key not in ['header', 'fluid_header', 'pol', 'unpol', 'tau']:
            header[key] = infile[key][()]
    infile.close()
    return header

def read_image_array(fname):
    """Sometimes you just need some numbers fast.
    Only supports ipole HDF5 format!
    """
    infile = h5py.File(fname, "r")
    pol = infile['pol'][:4]
    infile.close()
    return pol

# TODO mutable default arg is probably v bad
def read_image(fname, name=None, load_fluid_header=False, parameters={}, format_hint="ipole", units="cgs", only_unpolarized=False):
    """Read image from the given path or file object.

    :param fname: name (preferably) of file.  Limited support for hdf5 file objects if that's useful for performance
    :param name: Optionally add a name as an identifier in plots/text output
    :param load_fluid_header: Whether to load all the GRMHD parameters in ipole HDF5 images
    :param parameters: Anything that should be added to the Image parameters
    :param format_hint:
        | Resolve ambiguous text file image formats. Currently used for:
        | * "odyssey" to use odyssey format for 8-column files: ``alpha, beta, I, Q, U, V, unpol, tau``
        | * "ipole" to use ipole format for 8-column files: ``i, j, unpol, I, Q, U, V, tau``
    :param units: whether to renormalize
    :param only_unpolarized: even if the image has polarization data, load just the unpolarized version into stokes I instead.
    :returns: standard :class:`imtools.image.Image` object read from file

    Caveats of this function:

        * ``imtools`` expects the array to consist of CGS intensity values -- some formats require renormalization after reading.
        * Filetype guessing is best-effort especially for text files. Parameters even more so. Use the format_hint.
        * non-square images are supported only in the ipole HDF5 format.
    """
    if isinstance(fname, str):
        manage_file = True
        ftype = None
        if fname[-3:] == ".h5":
            try:
                infile = h5py.File(fname, "r")
            except IOError:
                print("Couldn't read file: ", fname)
                return None
            if 'header' in infile:
                ftype = "ipole_h5"
            else:
                ftype = "grtrans_h5"
        elif fname[-4:] == ".dat":
            try:
                infile = np.loadtxt(fname).T
                manage_file = False
            except OSError:
                print("Couldn't read file: ", fname)
                return None
            if infile.shape[0] == 8:
                if format_hint == "odyssey":
                    ftype = "odyssey_dat_8"
                else:
                    ftype = "ipole_dat_8"
            if infile.shape[0] == 7:
                ftype = "ipole_dat_7"
            elif infile.shape[0] == 6:
                if format_hint == "odyssey":
                    ftype = "odyssey_dat_6"
                else:
                    ftype = "ipole_dat_6"
            elif infile.shape[0] == 3:
                ftype = "ibothros_dat_3"
        elif fname[-4:] == ".npy":
            infile = np.load(fname)
            manage_file = False # We're done with the file now
            ftype = "grtrans_npy"
    elif isinstance(fname, h5py.File):
        manage_file = False
        infile = fname
        fname = infile.filename
        if 'header' in infile:
            ftype = "ipole_h5"
        else:
            ftype = "grtrans_h5"

    if ftype is None:
        print("Unknown file type: ", fname)
        return None

    # Default optional parameters to None/empty
    only_use_unpol = only_unpolarized
    unpol_data = None
    tauF = None
    tau = None
    header = {}

    if ftype == "ipole_h5":
        try:
            pol_data = infile['pol'][:,:,:4].transpose(1,0,2)
            unpol_data = infile['unpol'][()].T
            tauF = infile['pol'][:,:,4].T
            tau = infile['tau'][()].T
            header = hdf5_to_dict(infile['header'])
            if load_fluid_header:
                header.update(hdf5_to_dict(infile['fluid_header']))
            for key in infile.keys():
                if key not in ['header', 'fluid_header', 'pol', 'unpol', 'tau']:
                    header[key] = infile[key][()]
        except KeyError:
            print("Warning: unable to open object in file ", fname)
            return None
    elif ftype == "grtrans_h5":
        # Grtrans output is in the form [stokes, px_num, freq].
        # We don't care about the last one and want to split the second one,
        # then we want stokes index last
        # The python wrapper for grtrans ensures this is already in Jy,
        # also note grtrans will only output n_stokes of full matrix
        try:
            ImRes = int(np.sqrt(infile['ivals'].shape[1]))
            pol_data = infile['ivals'][:,:,0].reshape(4,ImRes,ImRes).transpose(2,1,0)
            # Correct the Q,U convention
            pol_data[:,:,1] *= -1
            pol_data[:,:,2] *= -1
        except KeyError:
            print("Warning: unable to open object in file ", fname)
            return None
    elif ftype == "grtrans_npy":
        # Numpy files from Jason are corrected and image-only, so:
        pol_data = infile
    elif ftype == "odyssey_dat_8":
        # Odyssey 8-column: i, j, alpha, beta, I, Q, U, V
        # Odyssey 8-column format was: alpha, beta, I, Q, U, V, unpol, tau
        imres = int(np.sqrt(infile.shape[1]))
        pol_data = infile[4:8].reshape(4,imres,imres).transpose(2,1,0)
        #unpol_data = infile[6].reshape(imres,imres)
        #tau = infile[7].reshape(imres,imres)
        header = parse_name(fname)
    elif ftype == "ipole_dat_8":
        # ipole 8-column: i, j, unpol, I, Q, U, V, tauF
        imres = int(np.sqrt(infile.shape[1]))
        unpol_data = infile[2].reshape(imres,imres).T
        pol_data = infile[3:7].reshape(4,imres,imres).transpose(2,1,0)
        tauF = infile[7].reshape(imres,imres)
        header = parse_name(fname)
    elif ftype == "ipole_dat_7":
        # ipole 7-column: i, j, unpol, I, Q, U, V
        imres = int(np.sqrt(infile.shape[1]))
        unpol_data = infile[2].reshape(imres,imres).T
        pol_data = infile[3:7].reshape(4,imres,imres).transpose(2,1,0)
        header = parse_name(fname)
    elif ftype == "ipole_dat_6":
        # Common 6-column: i, j, I, Q, U, V
        imres = int(np.sqrt(infile.shape[1]))
        pol_data = infile[2:6].reshape(4,imres,imres).transpose(2,1,0)
        header = parse_name(fname)
    elif ftype == "odyssey_dat_6":
        # Odyssey/BHOSS 6-column: alpha, beta, I, Q, U, V
        imres = int(np.sqrt(infile.shape[1]))
        pol_data = infile[2:6].reshape(4,imres,imres).transpose(1,2,0)
        header = parse_name(fname)
    elif ftype == "ibothros_dat_3":
        imres = int(np.sqrt(infile.shape[1]))
        unpol_data = infile[2].reshape(imres,imres).T
        only_use_unpol = True # forced to use unpolarized data
        header = parse_name(fname)

    # Carry around some useful things we picked up
    header['fname'] = fname
    if name is not None:
        header['name'] = name

    if manage_file:
        infile.close()
    
    if only_use_unpol:
        nx = unpol_data.shape[0]
        ny = unpol_data.shape[1]
        pol_data = np.zeros((nx,ny,4))
        pol_data[:,:,0] = unpol_data

    return Image({**header, **parameters}, pol_data, tau=tau, tauF=tauF, unpol=unpol_data)

def read_from_cache(fname, number, name=None, load_fluid_header=False, parameters={}):
    """Read an image from an HDF5 "cache" or "summary" created with summary.py
    There are other things in those summaries, but the image is the important part
    """
    if isinstance(fname, str):
        manage_file = True
        infile = h5py.File(fname, "r")
    elif isinstance(fname, h5py.File):
        manage_file = False
        infile = fname
        fname = infile.filename

    unpol_data = infile['unpol'][number, :, :].T
    header = hdf5_to_dict(infile['header']) #TODO read once/cache?
    if load_fluid_header:
        header.update(hdf5_to_dict(infile['fluid_header']))

    # Carry around some useful things we picked up
    header['fname'] = fname
    if name is not None:
        header['name'] = name

    if manage_file:
        infile.close()

    # Copy to Stokes I, leave others blank
    nx = unpol_data.shape[0]
    ny = unpol_data.shape[1]
    pol_data = np.zeros((nx,ny,4))
    pol_data[:,:,0] = unpol_data

    return Image({**header, **parameters}, pol_data, unpol=unpol_data)

def parse_name(fname):
    """This function could be used to parse images without metadata,
    but with a consistent naming scheme. Currently it does nothing.
    """
    return {}

def hdf5_to_dict(h5grp):
    """Recursively load group contents into nested dictionaries."""
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r")
        do_close = True

    ans = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py._hl.group.Group):
            # Call recursively
            ans[key] = hdf5_to_dict(h5grp[key])
        elif isinstance(item, h5py._hl.dataset.Dataset):
            # Otherwise read the dataset
            ans[key] = item[()]

    if do_close:
        h5grp.close()

    # This runs the un-bytes-ing too much, but somehow not enough
    decode_all(ans)
    return ans

def decode_all(bytes_dict):
    """Recursively un-bytes some HDF5 bytestrings for Python3 compatibility."""
    for key in bytes_dict:
        # Decode bytes/numpy bytes
        if isinstance(bytes_dict[key], (bytes, np.bytes_)):
            bytes_dict[key] = bytes_dict[key].decode('UTF-8')
        # Split ndarray of bytes into list of strings
        elif isinstance(bytes_dict[key], np.ndarray):
            if bytes_dict[key].dtype.kind == 'S':
                bytes_dict[key] = [el.decode('UTF-8') for el in bytes_dict[key]]
        # Recurse for any subfolders
        elif isinstance(bytes_dict[key], dict):
            decode_all(bytes_dict[key])

    return bytes_dict
