#!/usr/bin/python3

# Based on code provided by UCSD in their dartFISH data submission to HuBMAP.

# import immac, re, shutil, warnings, sys
import re

# Kian: added 201011
import shutil
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from os import chdir, getcwd, listdir, makedirs, path, remove, walk
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

# Kian: added 201011
from code_lib import TwoDimensionalAligner_2 as myAligner
from code_lib import tifffile as tiff

# <http://www.lfd.uci.edu/~gohlke/code/tifffile.py> # Kian: added 201011


def listdirectories(directory=".", pattern="*"):  # Kian: added 201011
    """
    listdirectories returns a list of all directories in path
    """
    directories = []
    for i in next(walk(directory))[1]:
        # 		if match(pattern, i):
        directories.append(i)
    directories.sort()
    return directories


def mip_gauss_tiled(rnd, fov, dir_root, dir_output="./MIP_gauss", sigma=0.7, channel_int="ch00"):
    """
    Modified from Matt Cai's MIP.py
    Maximum intensity projection along z-axis
    """
    # get current directory and change to working directory
    current_dir = getcwd()

    dir_parent = path.join(dir_root)
    chdir(dir_parent + "/" + rnd)

    # get all files for position for channel
    image_names = [
        f
        for f in listdir(".")
        if re.match(r".*_s" + "{:02d}".format(fov) + r".*_" + channel_int + r"\.tif", f)
    ]

    # put images of correct z_range in list of array
    nImages = len(image_names)
    image_list = [None] * nImages
    for i in range(len(image_names)):
        image_list[i] = ndimage.gaussian_filter(plt.imread(image_names[i]), sigma=sigma)

    # change directory back to original
    chdir(current_dir)

    image_stack = np.dstack(image_list)

    max_array = np.amax(image_stack, axis=2)

    # PIL unable to save uint16 tif file
    # Need to use alternative (like libtiff)

    # Make directories if necessary
    if not dir_output.endswith("/"):
        dir_output = "{0}/".format(dir_output)
    # mkreqdir(dir_output)
    if not path.isdir(dir_output + "FOV{:03d}".format(fov)):
        makedirs(dir_output + "FOV{:03d}".format(fov))

    # Change to output dir and save MIP file
    chdir(dir_output)
    tiff.imsave(
        "FOV{:03d}".format(fov)
        + "/MIP_"
        + rnd
        + "_FOV{:03d}".format(fov)
        + "_"
        + channel_int
        + ".tif",
        max_array,
    )

    # change directory back to original
    chdir(current_dir)


def cli(
    dir_data_raw,
    dir_output,
    dir_output_aligned,
    rnd_list,
    n_fovs,
    sigma,
    cycle_reference_index,
    channel_DIC_reference,
    channel_DIC,
    cycle_other,
    channel_DIC_other_vals,
    skip_projection,
    skip_align,
):

    currentTime = datetime.now()
    reportFile = path.join(dir_output, currentTime.strftime("%Y-%d-%m_%H:%M_SITKAlignment.log"))

    # Which cycle to align to
    cycle_reference = rnd_list[cycle_reference_index]
    channel_DIC_other = {}
    for i in range(len(channel_DIC_other_vals)):
        channel_DIC_other[cycle_other[i]] = channel_DIC_other_vals[i]
    t0 = time()
    # MIP
    if skip_projection:
        # for filename in glob(path.join(dir_data_raw,"**/*.*"), recursive=True):
        # 	shutil.copy(filename, dir_output)
        shutil.copytree(dir_data_raw, dir_output)
        sys.stdout = open(reportFile, "w")  # redirecting the stdout to the log file
        sys.stderr = open(
            reportFile, "w"
        )  # need to put this in a weird spot due to folder creation collision
        print("skipping image projection")
    else:
        if not path.isdir(dir_output):
            makedirs(dir_output)
        sys.stdout = open(reportFile, "w")  # redirecting the stdout to the log file
        sys.stderr = open(
            reportFile, "w"
        )  # need to put this in a weird spot due to folder creation collision
        for rnd in rnd_list:
            if "DRAQ5" in rnd or "anchor" in rnd:
                channel_list = [0, 1]
            else:
                channel_list = [0, 1, 2, 3]
            for channel in channel_list:
                print("Generating MIPs for " + rnd + " channel {0} ...".format(channel))
                for fov in range(n_fovs):
                    mip_gauss_tiled(
                        rnd,
                        fov,
                        dir_data_raw,
                        dir_output,
                        sigma=sigma,
                        channel_int="ch0{0}".format(channel),
                    )
                print("Done\n")

    t1 = time()

    print("Elapsed time ", t1 - t0)

    position_list = listdirectories(path.join(dir_output))
    if skip_align:
        print("skipping alignment")
        # for filename in glob(path.join(dir_output,"**/*.*"), recursive=True):
        # 	shutil.copy(filename, dir_output_aligned)
        shutil.copytree(dir_output, dir_output_aligned)
    else:
        # Align
        if not path.isdir(dir_output_aligned):
            makedirs(dir_output_aligned)
        for position in position_list:
            # 	position = 'Position{:03d}'.format(posi)
            # 	cycle_list = ["dc3","dc4","DRAQ5"]
            for rnd in rnd_list:
                print(
                    datetime.now().strftime(
                        "%Y-%d-%m_%H:%M:%S: "
                        + str(position)
                        + ", cycle "
                        + rnd
                        + " started to align"
                    )
                )
                aligner = myAligner.TwoDimensionalAligner(
                    destinationImagesFolder=path.join(dir_output, position),
                    originImagesFolder=path.join(dir_output, position),
                    originMatchingChannel=channel_DIC
                    if rnd not in cycle_other
                    else channel_DIC_other[rnd],
                    destinationMatchingChannel=channel_DIC_reference,
                    imagesPosition=position,
                    destinationCycle=cycle_reference,
                    originCycle=rnd,
                    resultDirectory=path.join(dir_output_aligned, position),
                    MaximumNumberOfIterations=400,
                )
            for file in [file for file in listdir() if file.startswith("IterationInfo.0")]:
                if path.isfile(
                    path.join(dir_output_aligned, position, "MetaData", file)
                ):  # removing a file with the same name already exists, remove it.
                    remove(path.join(dir_output_aligned, position, "MetaData", file))
                shutil.move(src=file, dst=path.join(dir_output_aligned, position, "MetaData"))

    t2 = time()
    print("Elapsed time ", t2 - t1)

    print("Total elapsed time ", t2 - t0)
    sys.stdout = sys.__stdout__  # restoring the stdout pipe to normal
    sys.stderr = sys.__stderr__  # restoring the stdout pipe to normal


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--fov-count", type=int)
    p.add_argument("--round-list", nargs="+")
    p.add_argument("--sigma", type=float)

    p.add_argument("--cycle-ref-ind", type=int)
    p.add_argument("--channel-dic-reference", type=str)
    p.add_argument("--channel-dic", type=str)
    p.add_argument("--cycle-other", nargs="+")
    p.add_argument("--channel-dic-other", nargs="+")
    p.add_argument("--skip-projection", dest="skip_projection", action="store_true")
    p.add_argument("--skip-align", dest="skip_align", action="store_true")

    args = p.parse_args()

    # out = OutputGrabber()
    # out.start()
    cli(
        args.raw_dir,
        "1_Projected",
        "2_Registered",
        args.round_list,
        args.fov_count,
        args.sigma,
        args.cycle_ref_ind,
        args.channel_dic_reference,
        args.channel_dic,
        args.cycle_other,
        args.channel_dic_other,
        args.skip_projection,
        args.skip_align,
    )
    # out.end()
