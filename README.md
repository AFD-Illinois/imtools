# imtools
Tools for creating and analyzing (many) simulated images of black holes.
Useful both for individual image computations, and management of large image sets, e.g. computing overall statistics by model, or plotting results from a set of different models together as a "collage."

# Dependencies
`imtools` requires only `numpy`, `scipy`, `matplotlib`, and `h5py`.  Install it anywhere with
```bash
$ pip install -e .
```

If [eht-imaging](https://github.com/achael/eht-imaging) is installed to the same environment, there is limited support for converting imtools `Image` objects for use in eht-imaging and vice versa.

# Use
Basic documentation for this library can be found [here](https://iimtools.readthedocs.io/en/latest/).

Check the `scripts` directory for basic usage of the `Image` object and `figures` plotting functions, likely the primary useful parts of this library.  Example notebooks, including use of the  `ImageSet` library object, will be added later.

The comparison utilities exist mostly to support [this](https://github.com/afd-illinois/polarized-grrt-comparison) (private to EHT members until publication).
