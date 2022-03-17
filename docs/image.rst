
.. _image:

The Image Object
================

The :class:`imtools.image.Image` object forms the basis of most operations in ``imtools``.  It includes a number of member functions representing both in-place operations and operations which return a new & modified form of the image.

Images consist of numpy arrays corresponding to at least the measured cgs intensity (Stokes I), and potentially measurements of the linear and circular polarization (specifically in the form of the other Stokes parameters Q, U, V).  As auxiliary variables they can potentially also include the optical depth and total Faraday rotation angle.  It is rare to need to construct an Image object directly, usually they are returned by functions which read existing image files.

Several common operations are supported either in-place on an image or as a new return value; these include blurring, integer downsampling, and most mathematical operations and comparison functions, vs other images or constants.  In addition, several per-pixel and integrated statistics such as linear and circular polarization fractions, EVPA, and plotting hint operations can be queried.

.. autoclass:: imtools.image.Image
   :special-members:
   :members: