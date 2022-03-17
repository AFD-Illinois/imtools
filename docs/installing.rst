Installing imtools
==================

The preferred method of obtaining and installing ``imtools`` is to run simply:
::

    $ git clone https://github.com/AFD-Illinois/imtools.git
    $ cd imtools/
    $ pip3 install -e . 

(If ``pip3`` is not found, just use ``pip``, which *usually* points to the same thing.)

This will install the package and all dependencies (which are just the usual scipy stack and HDF5 in any case).  So far, no special accommodation has been made (or found necessary) for Anaconda environments vs native Python installations.