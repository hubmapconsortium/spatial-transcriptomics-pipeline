#!/usr/bin/env python

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
from os import path
from pathlib import Path
from time import time
from typing import List

import cv2
import numpy as np
import starfish
import tifffile as tiff
from scipy import ndimage
from skimage import exposure, morphology, restoration
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation
from starfish import Experiment, ImageStack
from starfish.types import Levels
from tqdm import tqdm
import pdb


def saveImg(loc: str, prefix: str, img: ImageStack):
    # save the individual slices of an image in the same format starfish does
    for r in range(img.num_rounds):
        for c in range(img.num_chs):
            for z in range(img.num_zplanes):
                tiff.imsave(
                    "{}{}-c{}-r{}-z{}.tiff".format(loc, prefix, c, r, z), img._data[r, c, z, :, :]
                )


def saveExp(
    source_dir: str, save_dir: str, exp: Experiment = None, selected_fovs: List[str] = None
):
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
        print(f"looking at {file}.")
        if "fov" in file:
            # images were only updated if we looked at that fov
            if (selected_fovs is None) or (True in [f in file for f in selected_fovs]):
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
                print(f"\tskipping file {file}")
        else:
            # if we're using a subset of fovs, we'll need to modify view files
            if (selected_fovs is not None) and (
                "json" in file and "codebook" not in file and "experiment" not in file
            ):
                data = json.load(open(str(source_dir) + "/" + file))
                new_data = {}
                for k, v in data.items():
                    if k == "contents":
                        # this is where we select only fovs that we care about
                        new_data["contents"] = {f: v[f] for f in selected_fovs}
                    else:
                        new_data[k] = v
                with open(str(save_dir) + "/" + file, "w") as f:
                    json.dump(new_data, f)
                print(f"\tsaved {file} with used FOVs.")
            else:
                # we can just copy the rest of the files
                shutil.copyfile(f"{source_dir}/{file}", f"{save_dir}/{file}")
                print(f"\tcopied {file}")


def register_primary_aux(img, reg_img, chs_per_reg):
    """
    Register primary images using provided registration images.
    chs_per_reg is number of primary image channels
    associated with each registration image.

    Calculates shifts between auxillary images using the first
    auxillary image as a reference, then applies shifts to primary images.
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
    for r, ch in shifts:
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


def register_primary_primary(img, reg_img):
    """
    Register primary images using provided registration images.

    Calculates shifts between primary images and the single auxillary image
    and then applies shifts to primary images.
    """

    # Calculate shifts from each primary image to the registration image
    shifts = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            shift, error, diffphase = phase_cross_correlation(
                reg_img.xarray.data[0, 0], img.xarray.data[r, ch], upsample_factor=100
            )
            shifts[(r, ch)] = shift

    # Create transformation matrices
    shape = img.raw_shape
    tforms = {}
    for r, ch in shifts:
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
                np.linalg.inv(tforms[(r, ch)]),
                output_shape=shape[2:],
            )
    return img


def subtract_background(img, background):
    """
    Subtract real background image from primary images. Will register to same reference as primary images were
    aligned to if reg_img is provided, assumes background is of same round/channel dimensions as reference.
    """

    # Subtract background images from primary
    bg_dat = background.xarray.data
    num_chs = background.num_chs
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            for z in range(img.num_zplanes):
                data = img.xarray.data[r, ch, z].astype("float32")
                data -= bg_dat[r, ch % num_chs, z].astype("float32")
                data[data < 0] = 0
                img.xarray.data[r, ch, z] = data.astype("uint16")

    return img


def morph_open(images):
    """
    Multiprocessing helper function to run morphological openings in parallel.
    """
    size = 100
    morphed = []
    for image in images:
        background = np.zeros_like(image)
        for z in range(image.shape[0]):
            background[z] = cv2.morphologyEx(image[z], cv2.MORPH_OPEN, disk(size))
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

    # Create list of lists of images corresponding to the above chunking
    chunked_imgs = []
    for rch_chunk in chunked_rchs:
        chunk = []
        for rch in rch_chunk:
            chunk.append(img.xarray.data[rch[0], rch[1]])
        chunked_imgs.append(chunk)

    # Run morph open in parallel
    with ProcessPoolExecutor() as pool:
        poolMap = pool.map(morph_open, [img_chunk for img_chunk in chunked_imgs])
        results = [x for x in poolMap]

    # Subtract background estimates from img
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
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            for z in range(img.num_zplanes):
                background = restoration.rolling_ball(
                    img.xarray.data[r, ch, z], radius=rolling_rad, num_threads=num_threads
                )
                img.xarray.data[r, ch, z] -= background
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

    # Use min image as reference for histogram matching
    reference = img.xarray.data[min_rch[0], min_rch[1]]
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            img.xarray.data[r, ch] = np.rint(
                exposure.match_histograms(img.xarray.data[r, ch], reference)
            )
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
    output_dir: str,
    n_processes: int,
    clip_min: float = 0,
    clip_max: float = 99.9,
    level_method: str = "",
    is_volume: bool = False,
    register_aux_view: str = None,
    register_to_primary: bool = False,
    ch_per_reg: int = 1,
    background_name: str = None,
    register_background: bool = False,
    anchor_name: str = None,
    high_sigma: int = None,
    decon_iter: int = 15,
    decon_sigma: int = None,
    low_sigma: int = None,
    rolling_rad: int = None,
    match_hist: bool = False,
    wth_rad: int = None,
    rescale: bool = False,
    selected_fovs: List[int] = None,
):
    """
    n_processes: If provided, the number of threads to use for processing. Otherwise, the max number of
        available CPUs will be used.

    clip_min: minimum value for ClipPercentileToZero

    is_volume: whether to treat the z-planes as a 3D image.

    level_method: Which level method to be applied to the Clip filter.

    aux_name: name of the aux view to align registration to

    chs_per_reg: Number of images/channels associated with each registration image.
    If registration images are duplicated so that the dimensions of primary and
    registration images match then keep set to 1 for 1-to-1 registration.

    background_name: name of the background view that will be subtracted, if provided.

    register_background: if true, the background image will be registered to 'aux_name'

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

    rescale: If true, will not run final clip and scale on image, because it is expected to rescale
        the images in the following decoding step.

    selected_fovs: If provided, only FOVs with the provided indicies will be run.
    """

    os.makedirs(output_dir, exist_ok=True)

    reporter = open(
        path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_img_processing.log")), "w"
    )
    sys.stdout = reporter
    sys.stderr = reporter

    print(locals())

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    if level_method and level_method.upper() == "SCALE_BY_CHUNK":
        level_method = Levels.SCALE_BY_CHUNK
    elif level_method and level_method.upper() == "SCALE_BY_IMAGE":
        level_method = Levels.SCALE_BY_IMAGE
    elif level_method and level_method.upper() == "SCALE_SATURATED_BY_CHUNK":
        level_method = Levels.SCALE_SATURATED_BY_CHUNK
    elif level_method and level_method.upper() == "SCALE_SATURATED_BY_IMAGE":
        level_method = Levels.SCALE_SATURATED_BY_IMAGE
    else:
        level_method = Levels.SCALE_BY_IMAGE

    t0 = time()
    exp = starfish.core.experiment.experiment.Experiment.from_json(
        str(input_dir / "experiment.json")
    )
    if selected_fovs is not None:
        fovs = ["fov_{:05}".format(int(f)) for f in selected_fovs]
    else:
        fovs = list(exp.keys())

    for fov in fovs:
        img = exp[fov].get_image("primary")
        t1 = time()
        print("Fetched view " + fov)

        anchor = None
        if anchor_name:
            anchor = exp[fov].get_image(anchor_name)
            print("\tanchor image retrieved")

        if background_name:
            # If a background image is provided, subtract it from the primary image.
            bg = exp[fov].get_image(background_name)
            print("\tremoving existing backgound...")
            # img = subtract_background(img, bg)
            if anchor_name:
                print("\tremoving existing background from anchor image...")
                anchor = subtract_background(anchor, bg)
        else:
            # If no background image is provided, estimate background using a large morphological
            # opening to subtract from primary images
            print("\tremoving estimated background...")
            img = subtract_background_estimate(img, n_processes)
            if anchor_name:
                print("\tremoving estimated background from anchor image...")
                anchor = subtract_background_estimate(anchor, n_processes)

        if high_sigma:
            # Remove cellular autofluorescence w/ gaussian high-pass filter
            print("\trunning high pass filter...")
            ghp = starfish.image.Filter.GaussianHighPass(sigma=high_sigma)
            # ghp.run(img, verbose=False, in_place=True)
            ghp.run(img, verbose=False, in_place=True, n_processes=n_processes)
            if anchor_name:
                print("\trunning high pass filter on anchor image...")
                ghp.run(anchor, verbose=False, in_place=True, n_processes=n_processes)

        if decon_sigma:
            # Increase resolution by deconvolving w/ point spread function
            print("\tdeconvolving point spread function...")
            dpsf = starfish.image.Filter.DeconvolvePSF(num_iter=decon_iter, sigma=decon_sigma)
            # dpsf.run(img, verbose=False, in_place=True)
            dpsf.run(img, verbose=False, in_place=True, n_processes=n_processes)
            if anchor_name:
                print("\tdeconvolving point spread function on anchor image...")
                dpsf.run(anchor, verbose=False, in_place=True, n_processes=n_processes)

        if low_sigma:
            # Blur image with lowpass filter
            print("\trunning low pass filter...")
            glp = starfish.image.Filter.GaussianLowPass(sigma=low_sigma)
            # glp.run(img, verbose=False, in_place=True)
            glp.run(img, verbose=False, in_place=True, n_processes=n_processes)

        if wth_rad:
            print("\trunning white tophat filter...")
            img = white_top_hat(img, wth_rad)
            if anchor_name:
                print("\trunning white tophat filter on anchor image...")
                anchor = white_top_hat(anchor, wth_rad)

        if rolling_rad:
            # Apply rolling ball background subtraction method to even out intensities through each 2D image
            print("\tapplying rolling ball background subtraction...")
            img = rolling_ball(img, rolling_rad=rolling_rad, num_threads=n_processes)
            if anchor_name:
                print("\tapplying rolling ball background subtraction to anchor image...")
                anchor = rolling_ball(anchor, rolling_rad=rolling_rad, num_threads=n_processes)

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

        if register_to_primary:
            # If register_to_primary calculate registration shifts between primary images and the single aux image and
            # apply to primary images
            register = exp[fov].get_image(register_aux_view)
            if register.shape["r"] != 1:
                raise Exception(
                    "If --register-primary-view is used, auxillary images must have only a single round/channel (use the --aux-single-round option)"
                )
            else:
                print("\taligning to " + register_aux_view)
                img = register_primary_primary(img, register)
        elif register_aux_view:
            # If not register_to_primary but still registering, calculate registration shifts between specified aux images and apply to primary images
            register = exp[fov].get_image(register_aux_view)
            if register.shape["r"] != img.shape["r"]:
                raise Exception(
                    "If --register-aux-view is used, auxillary image dimensions must match primary image dimensions"
                )
            else:
                print("\taligning to " + register_aux_view)
                img = register_primary_aux(img, register, ch_per_reg)

        if not rescale and not (clip_min == 0 and clip_max == 0):
            print("\tclip and scaling...")
            # Scale image, clipping all but the highest intensities to zero
            clip = starfish.image.Filter.ClipPercentileToZero(
                p_min=clip_min, p_max=clip_max, is_volume=is_volume, level_method=level_method
            )
            clip.run(img, in_place=True)
            if anchor_name:
                print("\tapplying clip and scale to anchor image...")
                clip = starfish.image.Filter.ClipPercentileToZero(
                    p_min=90, p_max=99.9, is_volume=is_volume, level_method=level_method
                )
                clip.run(anchor, in_place=True)

        else:
            print("\tskipping clip and scale.")
            # Clip values below 0 and greater than 1 (prevents errors in decoding)
            clip = starfish.image.Filter.ClipPercentileToZero(
                p_min=0, p_max=100, is_volume=is_volume, level_method=Levels.CLIP
            )
            clip.run(img, in_place=True)

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

    print(f"Saving updated .jsons for {fovs}, copying other jsons\n")
    saveExp(input_dir, output_dir, exp=None, selected_fovs=fovs)
    print(f"\n\nTotal time elapsed for processing: {time() - t0}")


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--tmp-prefix", type=str)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--clip-min", type=float, default=0)
    p.add_argument("--clip-max", type=float, default=99.9)
    p.add_argument("--level-method", type=str, nargs="?")
    p.add_argument("--is-volume", dest="is_volume", action="store_true")
    p.add_argument("--register-aux-view", type=str, nargs="?")
    p.add_argument("--register-to-primary", dest="register_to_primary", action="store_true")
    p.add_argument("--ch-per-reg", type=int, nargs="?")
    p.add_argument("--background-view", type=str, nargs="?")
    p.add_argument("--register-background", dest="register_background", action="store_true")
    p.add_argument("--anchor-view", type=str, nargs="?")
    p.add_argument("--high-sigma", type=int, nargs="?")
    p.add_argument("--decon-iter", type=int, nargs="?")
    p.add_argument("--decon-sigma", type=int, nargs="?")
    p.add_argument("--low-sigma", type=int, nargs="?")
    p.add_argument("--rolling-radius", type=int, nargs="?")
    p.add_argument("--match-histogram", dest="match_histogram", action="store_true")
    p.add_argument("--tophat-radius", type=int, nargs="?")
    p.add_argument("--rescale", dest="rescale", action="store_true")
    p.add_argument("--n-processes", type=int, nargs="?")
    p.add_argument("--selected-fovs", nargs="+", const=None)

    args = p.parse_args()

    output_dir = f"tmp/{args.tmp_prefix}/3_processed/"

    if args.n_processes:
        n_processes = args.n_processes
    else:
        try:
            # the following line is not guaranteed to work on non-linux machines.
            n_processes = len(os.sched_getaffinity(os.getpid()))
        except Exception:
            n_processes = 1

    cli(
        input_dir=args.input_dir,
        output_dir=output_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        level_method=args.level_method,
        is_volume=args.is_volume,
        register_aux_view=args.register_aux_view,
        register_to_primary=args.register_to_primary,
        ch_per_reg=args.ch_per_reg,
        background_name=args.background_view,
        register_background=args.register_background,
        anchor_name=args.anchor_view,
        high_sigma=args.high_sigma,
        decon_iter=args.decon_iter,
        decon_sigma=args.decon_sigma,
        low_sigma=args.low_sigma,
        rolling_rad=args.rolling_radius,
        match_hist=args.match_histogram,
        wth_rad=args.tophat_radius,
        rescale=args.rescale,
        n_processes=n_processes,
        selected_fovs=args.selected_fovs,
    )
