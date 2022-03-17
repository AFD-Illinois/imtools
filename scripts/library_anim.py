#!/usr/bin/env python3

import os
import sys
from imtools.parallel import map_parallel
from imtools.image import Image
from imtools.library import ImageSet
from imtools.figures import *

lib = ImageSet(sys.argv[1])
blur = int(sys.argv[2])
outdir = "frames_collage_blur{}".format(blur)

os.makedirs(outdir, exist_ok=True)

def make_frame(n):
    frame = collage(lib, n, rotated=True, zoom=1.65, blur=blur, n_evpa=20, evpa_scale="emission", compress_scale=2, figsize=(10.8,19.2))
    frame.savefig(outdir+"/frame_{:03}.png".format(n), dpi=100)

map_parallel(make_frame, list(range(200)))