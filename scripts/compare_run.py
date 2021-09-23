import os
import sys
import itertools

from importlib import reload
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imtools.io as imio
import imtools.plots as ip
import imtools.stats as stats

ot = open("comp_table.csv", "w")
i = 0

for file in sys.argv[1:]:
    im1 = imio.read(file)
    im2 = imio.read(file.replace("grtrans","ipole"))
    if im1 == None or im2 == None:
        continue

    ot.write("{},".format(i))
    ot.write("{:.5},{:.5},{:.5},{:.5},".format(im1.Itot(), im1.Qtot(), im1.Utot(), im1.Vtot()))
    ot.write("{:.5},{:.5},{:.5},".format(im1.lpfrac_int(), im1.cpfrac_int(), im1.evpa_int()))
    ot.write("{:.5},{:.5},{:.5},{:.5},".format(im2.Itot(), im2.Qtot(), im2.Utot(), im2.Vtot()))
    ot.write("{:.5},{:.5},{:.5},".format(im2.lpfrac_int(), im2.cpfrac_int(), im2.evpa_int()))
    ot.write("{:.5},{:.5},{:.5},{:.5},".format(*stats.polar_abs_integrated(im1, im2)))
    ot.write("{:.5},{:.5},{:.5},{:.5}".format(*stats.mses(im1, im2)))
    ot.write("{:.5},{:.5},{:.5},{:.5}".format(*stats.mses(im1.blurred(20), im2.blurred(20))))

    ot.write("\n")
    i += 1

