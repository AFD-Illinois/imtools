#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import imtools as it

im1 = it.read(sys.argv[1], name="Polarized")
im2 = it.read(sys.argv[1], only_unpolarized=True, name="Unpolarized")
fig = it.compare_unpol(im1, im2, same_colorscale=True)
fig.savefig(sys.argv[1][:-3]+"_unpol_comp.png")
plt.close(fig)
