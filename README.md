# imtools
Tools for creating and analyzing (many) simulated images of black holes.
Organize and compute overall statistics by model, and keep track of camera parameters in a consistent way.

# Dependencies
The usual: `numpy`, `scipy`, `matplotlib`, `h5py`.  Install with
```bash
$ python setup.py develop
```
of if you won't want to modify the library itself, you can install it more permanently with
```bash
$ pip install .
```

If [eht-imaging](https://github.com/achael/eht-imaging) is installed, there is limited support for converting imtools `Image` objects for use in eht-imaging and vice versa.

# Use
Check the `scripts` directory for basic usage of the `Image` object, likely the primary useful part of this library.  Example notebooks, including use of the  `ImageSet` library object, will be added later.

The comparison utilities exist mostly to support [this](https://github.com/afd-illinois/polarized-grrt-comparison) (private to EHT members until publication).
