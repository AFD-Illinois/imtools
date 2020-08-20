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

# Use
Say you have some images.  An entire library of images of different black hole accretion flow models, taken at different times.
Maybe you want to explore overall properties of these images, or compare them.
Have I got some Python for you.

See the `notebooks` folder for some example Jupyter notebooks using imtools for various operations.  Note that library-based
examples require the images in question, and take a long time to run the first time, as the images are read once and sorted
into a cache mapping models -> image file paths.  They use the paths from BH by default, modify if running locally.