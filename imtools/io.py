# Reading files

import numpy as np
import h5py

from imtools.image import Image


def read_image_array(fname):
    """Sometimes you just need some numbers"""
    infile = h5py.File(fname)
    pol = infile['pol'][:4]
    return pol

# TODO mutable default arg is probably v bad
def read_image(fname, parameters={}, load_fluid_header=False):
    """Read image from the given path or file object.
    @param fname: name (preferably) of file.  Can be hdf5 file object
    @param parameters: Anything that should be added to the Image parameters

    @return standard Image object
    """
    if isinstance(fname, str):
        manage_file = True
        ftype = None
        if fname[-3:] == ".h5":
            try:
                infile = h5py.File(fname, "r")
            except IOError:
                print("Warning: cannot read ", fname)
                return None
            ftype = "ipole_h5"
        elif fname[-4:] == ".dat":
            infile = np.loadtxt(fname).T
            if infile.shape[0] == 7:
                ftype = "ipole_dat_7"
            elif infile.shape[0] == 6:
                ftype = "ipole_dat_6"
    elif isinstance(fname, h5py.File):
        # Please don't hand us files, but we will try to interpret if you do
        manage_file = False
        infile = fname
        fname = infile.filename
        ftype = "ipole_h5"

    if ftype is None:
        print("Warning: unknown file {}".format(fname))
        return None

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
    elif ftype == "ipole_dat_7":
        imres = np.sqrt(infile.shape[1])
        pol_data = infile[3:7].reshape(4,imres,imres)
        unpol_data = infile[2].reshape(imres,imres)
        tauF = infile[7].reshape(imres,imres)
        tau = None
        header = parse_name(fname)
    elif ftype == "ipole_dat_6":
        imres = np.sqrt(infile.shape[1])
        pol_data = infile[2:6].reshape(4,imres,imres)
        unpol_data = None
        tauF = infile[6].reshape(imres,imres)
        tau = None
        header = parse_name(fname)

    header['fname'] = fname # We probably want to carry this around, just in case

    if manage_file:
        infile.close()

    return Image({**header, **parameters}, pol_data, tau=tau, tauF=tauF, unpol=unpol_data)


def hdf5_to_dict(h5grp):
    """Recursively load group contents into nested dictionaries"""
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
    """Recursively un-bytes some HDF5 bytestrings for Python3 compatibility"""
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

def parse_name(fname):
    # TODO
    return {}