#!/usr/bin/env python3

import os
import pathlib
import shutil
import sys
import glob
import numpy as np
import h5py

import pyHARM.ana.units as units
import pyHARM.coordinates as coords

cgs = units.get_cgs()

org_dir = os.path.join("staging_3", "H5S")

if not os.path.exists(org_dir):
    os.mkdir(org_dir)

# Our database of munits as "tags" for different runs
munit_flux_database = {}

# Munits which correspond to known "good" runs and should take precedence on collision
known_munits = []
with open("known_munits.txt", "r") as fil:
    for line in fil:
        known_munits.extend([float(chunk) for chunk in line.split()])

# Round & stringify them for comparisons
known_munits = ["{:.6g}".format(mu) for mu in known_munits]
print("Loaded {} known M_unit values".format(len(known_munits)))

# TODO worth making one-step on staging_1?

for fpath in (glob.glob(os.path.join("staging_2", "*", "*")) +
    glob.glob(os.path.join("GRMHD", "*", "*", "*", "*", "*")) +
    glob.glob(os.path.join("RadGRMHD", "*", "*", "*", "*"))):
    files = glob.glob(os.path.join(fpath,"image_*.h5"))
    if len(files) == 0:
        print("Found no files in {}".format(fpath))
        continue
    else:
        print("Processing {}".format(fpath))

    for fil in files:
        infile = h5py.File(fil, "r")
        fname = fil.split("/")[-1]

        to_update = {}

        # This is inefficient but necessary for one-step operation
        # Find the average flux of files with our same Munit+frequency
        ftots_u = []
        ftots_p = []
        munit_float = infile['/header/units/M_unit'][()]
        munit = "{:.6g}".format(munit_float)
        freq = "{}GHz".format(int(infile['/header/freqcgs'][()]/1e9))
        if munit+freq in munit_flux_database:
            ftot_float, nrun = munit_flux_database[munit+freq]
        else:
            for fil_comp in files:
                inf = h5py.File(fil_comp, "r")
                their_munit = "{:.6g}".format(inf['/header/units/M_unit'][()])
                their_freq = "{}GHz".format(int(inf['/header/freqcgs'][()]/1e9))
                if munit == their_munit and freq == their_freq:
                    ftots_u.append(inf['/Ftot_unpol'][()])
                    ftots_p.append(inf['/Ftot'][()])
                inf.close()
            ftots_u = np.array(ftots_u)
            ftots_p = np.array(ftots_p)
            ftot_u_float = np.mean(ftots_u)
            ftot_p_float = np.mean(ftots_p)
            # Either we round normally and maybe "miss" things
            ftot_float = ftot_u_float
            # Or we be very lenient with rounding...
#             if 0.44 <= round(ftot_u_float, 2) <= 0.56 or 0.44 <= round(ftot_p_float, 2) <= 0.56:
#                 ftot_float = 0.5
#             elif 0.75 <= round(ftot_u_float, 2) <= 0.85 or 0.75 <= round(ftot_p_float, 2) <= 0.85:
#                 ftot_float = 0.8
#             else:
#                 ftot_float = ftot_u_float
            nrun = len(ftots_u)
            munit_flux_database[munit+freq] = (ftot_float, nrun)
            print("Found run: w/Munit {} and {} files, avg flux {:.1}Jy".format(munit, nrun, ftot_float))
        ftot = "{:.1f}Jy".format(ftot_float)

        # Read enough header data to figure out where to move it
        npx = infile['/header/camera/nx'][()]
        res = "{}px".format(npx)
        
        # Get FOV the *right* way.
        try:
            fovmuas = infile['/header/camera/dx'][()] / infile['/header/dsource'][()] * infile['/header/units/L_unit'][()] * 2.06265e11
        except ValueError as e:
            print("Not guessing FOV! {}".format(fil))
            continue

        if np.isnan(fovmuas):
            print("Not continuing: FOV is NaN! {}".format(fil))
            continue

        nfov = int(fovmuas)
        if nfov == 159:
            nfov = 160
        fov = "{}muas".format(nfov)

        fast = "fast"
        
        mbh = "{:.2g}".format(infile['/header/units/L_unit'][()] * cgs['CL']**2 / cgs['GNEWT'] / cgs['MSOLAR'])
        if mbh not in ["3.5e+09", "6.2e+09", "7.2e+09"]:
            mbh_f = mbh
            mbh = "sweep"

        if '/header/dsource' in infile:
            dsource = "{:.3g}pc".format(infile['/header/dsource'][()]/cgs['PC'])
        else:
            dsource = "16.9e+09pc"
            to_update['/header/dsource'] = 16.9e6*cgs['PC']

        if 'header/electrons/type' in infile:
            e_type = infile['header/electrons/type'][()]
            if e_type == 1:
                if "H10" in fpath:
                    model = "native_H10"
                elif "R17" in fpath:
                    model = "native_R17"
                else:
                    model = "native"
            elif e_type == 2:
                model = "m_{}_{}_{}".format(1, int(infile['header/electrons/rlow'][()]), int(infile['header/electrons/rhigh'][()]))
            else:
                model = "f_{}".format(int(infile['header/electrons/tp_over_te'][()]))
        else:
            # Get from last underscore to file extension, which is where rhi gets appended
            rhi_or_munit = float(".".join(fil.split("_")[-1].split(".")[:-1]))
            if rhi_or_munit > 200:
                print("Guessing fixed Tp/Te = 3")
                model = "f_3"
                to_update['/header/electrons/type'] = 0
                to_update['/header/electrons/tp_over_te'] = 3.0
            else:
                print("Not guessing electron model! {}".format(fil))
                continue

        # Haaaack
        if "INSANE" in fpath:
            flux = "INSANE"
        elif "semiMAD" in fpath:
            flux = "semiMAD"
        elif "MAD" in fpath or "Ma" in fpath:
            flux = "MAD"
        elif "SANE" in fpath or "Sa" in fpath:
            flux = "SANE"
        else:
            print("Could not guess flux for {}".format(fpath))

        if '/header/mhd' not in infile:
            to_update['/header/mhd'] = None
        if '/header/mhd/b_flux_type' not in infile:
            to_update['/header/mhd/b_flux_type'] = flux

        spin = None
        if '/fluid_header/geom' in infile:
            for geom in ['mks', 'mks3', 'mmks', 'fmks']:
                if geom in infile['/fluid_header/geom']:
                    spin = "a{:.2}".format(infile['/fluid_header/geom/'+geom+'/a'][()])
        else:
            for chunk in fil.split("_"):
                if "a+" in chunk or "a-" in chunk or "a0" in chunk:
                    spin = chunk.replace("+","")
                    # Populate a very basic geometry folder
                    # Everything imaged early on was iharm3d & therefore MMKS
                    to_update['/fluid_header/geom'] = None
                    to_update['/fluid_header/metric'] = "MMKS"
                    to_update['/fluid_header/geom/mmks'] = None
                    to_update['/fluid_header/geom/mmks/a'] = float(spin.replace("a",""))

        if spin is None:
            print("Not guessing spin! {}".format(fpath))
            continue

        if '/fluid_header/n1' in infile:
            n1 = infile['/fluid_header/n1'][()]
            n2 = infile['/fluid_header/n2'][()]
            n3 = infile['/fluid_header/n3'][()]
        else:
            n1 = None
            chunks = fil.split("_")
            for i,chunk in enumerate(chunks):
                if chunk == "MAD":
                    try:
                        n1 = int(chunks[i+1])
                    except:
                        pass
                elif chunk == "SANE":
                    try:
                        n1 = int(chunks[i+1])
                    except:
                        pass
            if n1 is None:
                print("Can't find grid size!")
                continue
            else:
                n2 = {'192': 96, '288': 128, '384': 192, '448': 224}[str(n1)]
                n3 = n2
            to_update['/fluid_header'] = None
            to_update['/fluid_header/n1'] = n1
            to_update['/fluid_header/n2'] = n2
            to_update['/fluid_header/n3'] = n3


        if '/header/camera/thetacam' in infile:
            inc_d = int(infile['/header/camera/thetacam'][()])
        else:
            # Get theta from x2cam.  It sucks.
            if '/fluid_header/geom/mks' in infile:
                if '/fluid_header/geom/mks/hslope' in infile:
                    hslope = infile['/fluid_header/geom/mks/hslope'][()]
                else:
                    hslope = 0.3
                cd = coords.MKS({'a': infile['/fluid_header/geom/mks/a'][()],
                                 'hslope': hslope})
                metric_name = "mks"
            if '/fluid_header/geom/mks3' in infile:
                print("When this comes up, I'll implement it")
                metric_name = "mks3"
                continue
            else:
                # TODO is this even any different out at such large r?
                # Note pyHARM FMKS == ipole MMKS, sadly.
                if '/fluid_header/geom/mmks/a' in infile:
                    a = infile['/fluid_header/geom/mmks/a'][()]
                else:
                    a = float(spin.replace("a", ""))
                if '/fluid_header/geom/mmks/r_out' in infile:
                    rout = infile['/fluid_header/geom/mmks/r_out'][()]
                elif '/fluid_header/geom/mmks/Rout' in infile:
                    rout = infile['/fluid_header/geom/mmks/Rout'][()]
                else:
                    if flux == "MAD":
                        rout = 1000.0
                    elif flux == "SANE":
                        rout = 50.0
                    else:
                        print("Can't guess r_out! Using a default")
                        rout = 50.0


                if '/fluid_header/geom/mmks/hslope' in infile:
                    hslope = infile['/fluid_header/geom/mmks/hslope'][()]
                    poly_xt = infile['/fluid_header/geom/mmks/poly_xt'][()]
                    poly_alpha = infile['/fluid_header/geom/mmks/poly_alpha'][()]
                    mks_smooth = infile['/fluid_header/geom/mmks/mks_smooth'][()]
                else:
                    hslope = 0.3
                    poly_xt = 0.82
                    poly_alpha = 14.0
                    mks_smooth = 0.5
                cd = coords.FMKS({'a': a,
                                  'hslope': hslope,
                                  'poly_xt': poly_xt,
                                  'poly_alpha': poly_alpha,
                                  'mks_smooth': mks_smooth,
                                  'n1tot': n1,
                                  'r_out': rout})
                metric_name = "mmks"

            # If we had to derive thetacam, write it and all its friends
            xcam = infile['/header/camera/x'][()]
            inc_d = int(cd.th(xcam)*180/np.pi)
            to_update['/header/camera/thetacam'] = float(inc_d)
            # Round
            to_update['/header/camera/rcam'] = float(int(cd.r(xcam)))
            to_update['/header/camera/phicam'] = float(int(cd.phi(xcam)))

        if inc_d in [12, 17, 22, 158, 163, 168]:
            # Hide opposing angles, i.e. redefine inc in files to be w.r.t BH spin
            if "-" in spin:
                inc = "{}".format(180 - inc_d)
            else:
                inc = "{}".format(inc_d)
        else:
            inc_f = inc
            inc = "sweep"


        # Preserve run tags, but use the above n1,2,3 to make sure there are no more "x96x96" fiascos
        tag = None
        for code in ["bhlight", "KORAL", "BHAC", "IHARM"]:
            if code in fpath:
                loc = fpath.find(code)
                tag = "{}x{}x{}_{}".format(n1, n2, n3, fpath[loc:loc+15].split("/")[0])
                write_code = code
        # If that didn't work,
        if tag is None:
            # Special-case the hell out of some KORAL runs,
            if n1 == 302 or n1 == 194:
                tag = "{}x{}x{}_KORAL".format(n1, n2, n3)
            else:
                # Or just guess IHARM
                tag = "{}x{}x{}_IHARM".format(n1, n2, n3)
                if "gamma53" in fpath:
                    tag += "_gamma53"
                write_code = "IHARM"

        to_update['/header/mhd/code_name'] = write_code
        
        t = infile['/header/t'][()]
        if t == 0.0:
            t = float(int(fname.split("_")[2]))

        infile.close()
        
        prop_list_f = prop_list = [fast,freq,res,fov,mbh,dsource,ftot,inc,model,flux,spin,tag]
        
        # If a directory is "sweep" we should disambiguate its files.
        if mbh == "sweep":
            prop_list_f[4] = mbh_f
        if inc == "sweep":
            prop_list_f[7] = inc_f
        if nrun < 50:
            prop_list[6] = "sweep"
            prop_list_f[6] = "{:.3f}Jy".format(ftot_float)
        
        new_fname = "_".join(["image"] + prop_list_f + ["{:09.2f}.h5".format(t)])
        

        moveto = os.path.join(org_dir, *prop_list, new_fname)
        #print("Moving file {} to {}".format(fil, moveto))

        pathlib.Path(os.path.dirname(moveto)).mkdir(parents=True, exist_ok=True)
        # If there are name collisions,
        # Decide on the fly which file should take precedence, and delete or copy accordingly
        if os.path.exists(moveto):
            ex_file = h5py.File(moveto, "r")
            ex_munit = "{:.6g}".format(ex_file['/header/units/M_unit'][()])
            ex_file.close()
            if munit in known_munits:
                if ex_munit in known_munits:
                    print("!!! RUN COLLISION of KNOWN MUNITS: {} and {} !!! Choosing the former arbitrarily!".format(munit, ex_munit))
                    known_munits.append(munit)
                # We're official, you're not
                os.remove(moveto)
                shutil.copy(fil, moveto)
            else:
                if ex_munit in known_munits:
                    # Official one is in place, we're the impostor
                    continue
                else:
                    print("!!! RUN COLLISION of UNKNOWN MUNITS: {} and {} !!! Choosing the former arbitrarily!".format(munit, ex_munit))
                    known_munits.append(munit)
                    os.remove(moveto)
                    shutil.copy(fil, moveto)
        else:
            shutil.copy(fil, moveto)

        # Then update our new copy!
        outfile = h5py.File(moveto, "r+")

        # Write all the myriad header updates we listed above
        for key in to_update:
            if key not in outfile:
                if to_update[key] is None:
                    outfile.create_group(key)
                elif isinstance(to_update[key], str):
                    outfile[key] = to_update[key].encode("ascii", "ignore")
                elif isinstance(to_update[key], list) and isinstance(to_update[key][0], str):
                    outfile[key] = [n.encode("ascii", "ignore") for n in to_update[key]]
                else:
                    outfile[key] = to_update[key]
            else:
                if to_update[key] is None:
                    pass
                elif isinstance(to_update[key], str):
                    outfile[key][()] = to_update[key].encode("ascii", "ignore")
                elif isinstance(to_update[key], list) and isinstance(to_update[key][0], str):
                    outfile[key][()] = [n.encode("ascii", "ignore") for n in to_update[key]]
                else:
                    outfile[key][()] = to_update[key]

        # Rotate the polarization convention and note that in the finished file
        if '/header/evpa_0' not in outfile or outfile['/header/evpa_0'][()] == b"W":
            outfile['/pol'][:,:,1] *= -1
            outfile['/pol'][:,:,2] *= -1
            outfile['/header/evpa_0'] = b"N"

        outfile.close()
