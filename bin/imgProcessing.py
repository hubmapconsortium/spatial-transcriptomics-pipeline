#!/usr/bin/env python

import functools
import hashlib
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from functools import partial, partialmethod
from os import cpu_count, makedirs, path
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd
import skimage
import starfish
import tifffile as tiff
import xarray as xr
from scipy import ndimage
from skimage import exposure, morphology, registration, restoration
from skimage.morphology import ball, dilation, disk, opening
from skimage.registration import phase_cross_correlation
from starfish import BinaryMaskCollection, Experiment, ImageStack
from starfish.experiment.builder import write_experiment_json
from starfish.types import Levels
from tqdm import tqdm


def saveImg(loc: str, prefix: str, img: ImageStack):
    # save the individual slices of an image in the same format starfish does
    for r in range(img.num_rounds):
        for c in range(img.num_chs):
            for z in range(img.num_zplanes):
                tiff.imsave(
                    "{}/{}-c{}-r{}-z{}.tiff".format(loc, prefix, c, r, z), img._data[r, c, z, :, :]
                )


def saveExp(source_dir: str, save_dir: str, exp: Experiment = None):
    # go through and save all images, if an experiment is provided
    if exp:
        for fov in exp.keys():
            for view in exp[fov].image_types:
                img = exp[fov].get_image(view)
                prefix = f"{view}-{fov}"
                saveImg(save_dir, prefix, img)

    # copy the non-tiff files to the new directory
    cp_files = [x for x in os.listdir(source_dir) if x[-5:] != ".tiff" and x[-4:] != ".log"]
    for file in cp_files:
        if "fov" in file:
            # if file contains images, we need to update sha's
            data = json.load(open(str(source_dir) + "/" + file))
            for i in range(len(data["tiles"])):
                abspath = str(save_dir) + "/" + data["tiles"][i]["file"]
                with open(os.fspath(abspath), "rb") as fh:
                    hsh = hashlib.sha256(fh.read()).hexdigest()
                data["tiles"][i]["sha256"] = hsh
                print(f"\tupdated hash for {data['tiles'][i]['file']}")
            with open(str(save_dir) + "/" + file, "w") as f:
                json.dump(data, f)
            print(f"saved {file} with modified hashes")
        else:
            # we can just copy the rest of the files
            shutil.copyfile(f"{source_dir}/{file}", f"{save_dir}/{file}")
            print(f"copied {file}")


def register_primary(img, reg_img, chs_per_reg):
    """
    Register primary images using provided registration images.
    chs_per_reg is number of primary image channels
    associated with each registration image.
    """
    # Calculate registration shifts from registration images
    shifts = {}
    # Reference is set arbitrarily to first round/channel
    reference = reg_img.xarray.data[0, 0]
    for r in range(reg_img.num_rounds):
        for ch in range(reg_img.num_chs):
            shift, error, diffphase = phase_cross_correlation(
                reference, reg_img.xarray.data[r, ch], upsample_factor=100
            )
            shifts[(r, ch)] = shift

    # Create transformation matrices
    shape = img.raw_shape
    tforms = {}
    for (r, ch) in shifts:
        tform = np.diag([1.0] * 4)
        # Start from 1 because we don't want to shift in the z direction (if there is one)
        for i in range(1, 3):
            tform[i, 3] = shifts[(r, ch)][i]
        tforms[(r, ch)] = tform

    # Register primary images
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            img.xarray.data[r, ch] = ndimage.affine_transform(
                img.xarray.data[r, ch],
                np.linalg.inv(tforms[(r, ch // chs_per_reg)]),
                output_shape=shape[2:],
            )
    return img


def subtract_background(img, background, reg_img=None, back_reg_ch=None):
    """
    Subtract real background image from primary images. Will register to same reference as primary images were
    aligned to if reg_img and back_reg_ch are provided.
    """

    # If a background registration channel is given register background images to primary images before subtracting.
    if reg_img and back_reg_ch:

        # Calculate registration shift for backround image
        reference = reg_img.xarray.data[0, 0]
        shift, error, diffphase = phase_cross_correlation(
            reference, background[back_reg_ch], upsample_factor=100
        )
        shape = img.raw_shape

        # Create transformation matrices
        tform = np.diag([1.0] * 4)
        # Start from 1
        for i in range(1, 3):
            tform[i, 3] = shift[i]

        # Register background images
        for ch in range(background.shape[0] - 1):
            background[ch] = ndimage.affine_transform(
                background[ch], np.linalg.inv(tform), output_shape=shape[2:]
            )

    # Subtract background images from primary
    num_chs = background.shape[0]
    if back_reg_ch is not None:
        num_chs -= 1
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            for z in range(img.num_zplanes):
                img.xarray.data[r, ch, z] -= background[ch % num_chs, z] / (2**16)
            img.xarray.data[r, ch][img.xarray.data[r, ch] < 0] = 0
    return img


def morph_open(rchs, image, size):
    """
    Multiprocessing helper function to run morphological openings in parallel.
    """
    morphed = []
    for r, ch in rchs:
        background = np.zeros_like(image[r, ch])
        for z in range(image.shape[2]):
            background[z] = cv2.morphologyEx(image[r, ch, z], cv2.MORPH_OPEN, disk(size))
        morphed.append(background)
    return morphed


def subtract_background_estimate(img, num_threads):
    """
    Estimate background using large morphological opening (radis = 100px) and subtract from image.
    """
    # Chunk round/channel combinations for parallel run
    rchs = [(r, ch) for r in range(img.num_rounds) for ch in range(img.num_chs)]

    # Calculates index ranges to chunk data by
    ranges = [0]
    for i in range(1, num_threads + 1):
        ranges.append(int((len(rchs) / num_threads) * i))
    chunked_rchs = [rchs[ranges[i] : ranges[i + 1]] for i in range(len(ranges[:-1]))]

    # Run morph open in parallel
    size = 100
    with ProcessPoolExecutor() as pool:
        part = partial(morph_open, image=img.xarray.data, size=size)
        poolMap = pool.map(part, [rch_chunk for rch_chunk in chunked_rchs])
        results = [x for x in poolMap]

    # Replace values in img
    for i in range(len(chunked_rchs)):
        for j in range(len(chunked_rchs[i])):
            r, ch = chunked_rchs[i][j]
            img.xarray.data[r, ch] = img.xarray.data[r, ch] - results[i][j]
            img.xarray.data[r, ch][img.xarray.data[r, ch] < 0] = 0
    return img


def rolling_ball(img, rolling_rad=3, num_threads=1):
    """
    Peform rolling ball background subtraction.
    """
    # Have to convert to integer values first as otherwise the resulting images are blank
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            for z in range(img.num_zplanes):
                data = np.rint(img.xarray.data[r, ch, z] * 2**16)
                background = restoration.rolling_ball(
                    data, radius=rolling_rad, num_threads=num_threads
                )
                data -= background
                data /= 2**16
                img.xarray.data[r, ch, z] = deepcopy(data)
    return img


def match_hist_2_min(img):
    """
    Calculate the lowest average intensity image in stack and use as reference to match histograms for all
    other rounds/channels.
    """
    # Calculate image means to find min
    meds = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            meds[(r, ch)] = np.mean(img.xarray.data[r, ch])
    min_rch = sorted(meds.items(), key=lambda item: item[1])[0][0]

    # Use min image as reference for histogram matching (need to convert to ints or it takes a VERY long time)
    reference = np.rint(img.xarray.data[min_rch[0], min_rch[1]] * 2**16)
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            data = np.rint(img.xarray.data[r, ch] * 2**16)
            matched = exposure.match_histograms(data, reference)
            matched /= 2**16
            img.xarray.data[r, ch] = deepcopy(matched)
    return img


def white_top_hat(img, wth_rad):
    """
    Perform white top hat filter on image.
    """
    footprint = morphology.disk(wth_rad)
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            for z in range(img.num_zplanes):
                img.xarray.data[r, ch, z] = cv2.morphologyEx(
                    img.xarray.data[r, ch, z], cv2.MORPH_TOPHAT, footprint
                )
    return img


def cli(
    input_dir: Path,
    output_dir: Path,
    clip_min: float = 95,
    clip_max: float = 99.9,
    aux_name: str = None,
    ch_per_reg: int = 1,
    background_name: str = None,
    anchor_name: str = None,
    high_sigma: int = None,
    decon_iter: int = 15,
    decon_sigma: int = None,
    low_sigma: int = None,
    rolling_rad: int = None,
    match_hist: bool = False,
    wth_rad: int = None,
    inline_log: bool = False,
):
    """
    clip_min: minimum value for ClipPercentileToZero

    aux_name: name of the aux view to align registration to

    chs_per_reg: Number of images/channels associated with each registration image.
    If registration images are duplicated so that the dimensions of primary and
    registration images match then keep set to 1 for 1-to-1 registration.

    background_name: name of the background view that will be subtracted, if provided.

    anchor_name: name of the aux view anchor round to perform processing on, if provided.

    high_sigma: Sigma value for high pass filter. High values remove less autofluorescence
        while lower values remove more. Won't need to change between data sets unless you
        had lots of autofluorescence.

    decon_iter: Number of iterations for deconvolution. High values remove more noise while
        lower values remove less. Won't need to change between data sets unless image is very noisy.

    decon_sigma: Sigma value for deconvolution. Should be approximately the expected spot size.

    low_sigma: Sigma value for lowpass filtering. Larger values result in stronger
    blurring. This should be low so can remain constant.

    rolling_rad: Radius for rolling ball background subtraction. Larger values lead to
        increased intensity evening effect. Likely doesn't need changed from 3.

    match_hist: If true, will perform histogram matching.

    wth_rad: Radius for white top hat filter. Should be slightly larger than the expected spot radius.
    """
    if not path.isdir(output_dir):
        makedirs(output_dir)

    if not inline_log:
        reporter = open(
            path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_img_processing.log")), "w"
        )
        sys.stdout = reporter
        sys.stderr = reporter

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    t0 = time()

    exp = starfish.core.experiment.experiment.Experiment.from_json(
        str(input_dir / "experiment.json")
    )
    for fov in exp.keys():
        img = exp[fov].get_image("primary")
        t1 = time()
        print("Fetched view " + fov)
        if aux_name:
            # If registration image is given calculate registration shifts for each image and apply them
            register = exp[fov].get_image(aux_name)
            print("\taligning to " + aux_name)
            img = register_primary(img, register, ch_per_reg)

        anchor = None
        if anchor_name:
            anchor = exp[fov].get_image(anchor_name)
            print("\tanchor image retrieved")

        if background_name:
            # If a background image is provided, subtract it from the primary image.
            bg = exp[fov].get_image(background_name)
            print("\tremoving existing backgound...")
            img = subtract_background(img, bg, register, 1)
            if anchor_name:
                print("\tremoving existing background from anchor image...")
                anchor = subtract_background(anchor, bg, register, 1)
        else:
            # If no background image is provided, estimate background using a large morphological
            # opening to subtract from primary images
            print("\tremoving estimated background...")
            img = subtract_background_estimate(img, cpu_count())
            if anchor_name:
                print("\tremoving estimated background from anchor image...")
                anchor = subtract_background(anchor, bg, register, 1)

        if high_sigma:
            # Remove cellular autofluorescence w/ gaussian high-pass filter
            print("\trunning high pass filter...")
            ghp = starfish.image.Filter.GaussianHighPass(sigma=high_sigma)
            # ghp.run(img, verbose=False, in_place=True)
            ghp.run(img, verbose=False, in_place=True, n_processes=cpu_count())

        if decon_sigma:
            # Increase resolution by deconvolving w/ point spread function
            print("\tdeconvolving point spread function...")
            dpsf = starfish.image.Filter.DeconvolvePSF(num_iter=decon_iter, sigma=decon_sigma)
            # dpsf.run(img, verbose=False, in_place=True)
            dpsf.run(img, verbose=False, in_place=True, n_processes=cpu_count())

        if low_sigma:
            # Blur image with lowpass filter
            print("\trunning low pass filter...")
            glp = starfish.image.Filter.GaussianLowPass(sigma=low_sigma)
            # glp.run(img, verbose=False, in_place=True)
            glp.run(img, verbose=False, in_place=True, n_processes=cpu_count())

        if wth_rad:
            print("\trunning white tophat filter...")
            img = white_top_hat(img, wth_rad)
            if anchor_name:
                print("\trunning white tophat filter on anchor image...")
                anchor = white_top_hat(anchor, wth_rad)

        if rolling_rad:
            # Apply rolling ball background subtraction method to even out intensities through each 2D image
            print("\tapplying rolling ball background subtraction...")
            img = rolling_ball(img, rolling_rad=rolling_rad, num_threads=cpu_count())

        if match_hist:
            # Use histogram matching to lower the intensities of each 3D image down to the same
            # intensity range as the least bright image. This is done so spot finding can be done.
            # BlobDetector doesn't do well when the intensities are in different ranges and c
            # lipping the values is not sufficient.
            print("\tapplying histogram matching...")
            img = match_hist_2_min(img)
            if anchor_name:
                print("\tapplying histogram matching to anchor image...")
                anchor = match_hist_2_min(anchor)

        print("\tclip and scaling...")
        # Scale image, clipping all but the highest intensities to zero
        clip = starfish.image.Filter.ClipPercentileToZero(
            p_min=clip_min, p_max=clip_max, is_volume=True, level_method=Levels.SCALE_BY_CHUNK
        )
        clip.run(img, in_place=True)
        if anchor_name:
            print("\tapplying clip and scale to anchor image...")
            clip.run(anchor, in_place=True)

        print(f"\tView {fov} complete")
        # save modified image
        saveImg(output_dir, f"primary-{fov}", img)

        # save all aux views while we're here
        for view in exp[fov].image_types:
            if view != "primary" and view != anchor_name:
                aux_img = exp[fov].get_image(view)
                saveImg(output_dir, f"{view}-{fov}", aux_img)
            elif view == anchor_name:
                saveImg(output_dir, f"{view}-{fov}", anchor)

        print(f"View {fov} saved")
        print(f"Time for {fov}: {time() - t1}")

    saveExp(input_dir, output_dir)
    print(f"\n\nTotal time elapsed for processing: {time() - t0}")


if __name__ == "__main__":

    output_dir = Path("3_processed")

    p = ArgumentParser()

    p.add_argument("--input-dir", type=Path)
    p.add_argument("--clip-min", type=float, default=95)
    p.add_argument("--clip-max", type=float, default=99.9)
    p.add_argument("--register-aux-view", type=str, nargs="?")
    p.add_argument("--ch-per-reg", type=int, nargs="?")
    p.add_argument("--background-view", type=str, nargs="?")
    p.add_argument("--anchor-view", type=str, nargs="?")
    p.add_argument("--high-sigma", type=int, nargs="?")
    p.add_argument("--decon-iter", type=int, nargs="?")
    p.add_argument("--decon-sigma", type=int, nargs="?")
    p.add_argument("--low-sigma", type=int, nargs="?")
    p.add_argument("--rolling-radius", type=int, nargs="?")
    p.add_argument("--match-histogram", dest="match_histogram", action="store_true")
    p.add_argument("--inline-log", dest="inline_log", action="store_true")
    p.add_argument("--tophat-radius", type=int, nargs="?")

    args = p.parse_args()

    cli(
        input_dir=args.input_dir,
        output_dir=output_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        aux_name=args.register_aux_view,
        ch_per_reg=args.ch_per_reg,
        background_name=args.background_view,
        anchor_name=args.anchor_view,
        high_sigma=args.high_sigma,
        decon_iter=args.decon_iter,
        decon_sigma=args.decon_sigma,
        low_sigma=args.low_sigma,
        rolling_rad=args.rolling_radius,
        match_hist=args.match_histogram,
        wth_rad=args.tophat_radius,
        inline_log=args.inline_log,
    )
