#!/usr/bin/env python

import functools
import hashlib
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from functools import partial, partialmethod
from os import makedirs, path
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import starfish
import tifffile as tiff
import xarray as xr
from scipy import ndimage
from skimage import morphology, registration, restoration
from skimage.morphology import ball, dilation, disk, opening
from skimage.registration import phase_cross_correlation
from starfish import BinaryMaskCollection, Experiment, ImageStack
from starfish.experiment.builder import write_experiment_json
from starfish.morphology import Binarize
from starfish.types import Levels
from tqdm import tqdm


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.
    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


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
    cp_files = [x for x in os.listdir(source_dir) if x[-5:] != ".tiff"]
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


def removeBg(opening_size: int, img: ImageStack):
    # Estimate background with a large morphological opening and divide out from image
    # Size of 10 seems to work well here (and for intron) and doesn't take too long (longer for 3D)
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            background = opening(img.xarray.data[r, ch, 0], selem=disk(opening_size))
            background /= background.max()
            img.xarray.data[r, ch] /= background


def registerImgs(img: ImageStack, aux_img: ImageStack):
    # Calculate registration shifts from images
    # Not sure how best to set this up but I made a 488 image (a fluorescent marker with wavelength of 488 was used
    # to mark every target in every hybridization round) for each 3D image even though the 488 images are the same
    # for each group of 3 primary images (so the images corresponding to round 0, channels 0, 1, and 2 all have the same
    # 488 image). There's probably a smarter way to do this without creating those duplicate images in memory. Then its
    # just like the previous step where you learn the transformation to align to the reference image and then apply those
    # transformations to the corresponding primary images. Registration images should be 3D with (z, y, x) dimension order
    reference = aux_img.xarray.data[0, 0, :, :, :]
    shifts = {}
    shape = aux_img.raw_shape
    for r in range(shape[0]):
        for ch in range(shape[1]):
            reg_img = aux_img.xarray.data[r, ch, :, :, :]
            shift, error, diffphase = phase_cross_correlation(
                reference, reg_img, upsample_factor=100
            )
            shifts[(r, ch)] = shift

    # Create transformation matrices
    tforms = {}
    for (r, ch) in shifts:
        tform = np.diag([1.0] * 4)
        if shape[2] == 1:
            start = 1
        else:
            start = 0
        for i in range(start, 3):
            tform[i, 3] = shifts[(r, ch)][i]
        tforms[(r, ch)] = tform

    # Register images
    same_size = aux_img.raw_shape == img.raw_shape

    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            ch_aux = 0
            if same_size:
                ch_aux = ch
            img.xarray.data[r, ch] = ndimage.affine_transform(
                img.xarray.data[r, ch], np.linalg.inv(tforms[(r, ch_aux)]), output_shape=shape[2:]
            )


def histoMatchImage(img: ImageStack):
    # Use histogram matching to lower the intensities of each 3D image down to the same intensity range as the least
    # bright image. This is done so spot finding can be done. BlobDetector doesn't do well when the intensities are in
    # different ranges and clipping the values is not sufficient (or at least wasn't for the intron images).
    # Uses a custom function that I pulled off stackoverflow because the starfish function that does this automatically
    # uses the median image reference which cannot be changed (without changing the actual code). This is non ideal
    # because it is not the dimmest image and calculating the median of large images (like in the intron set) takes A LOT
    # of memory. There is an skimage function that does this but is in the 0.19 version onward which starfish is not
    # compatible with

    # Calculate image medians to find min
    meds = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            meds[(r, ch)] = np.median(img.xarray.data[r, ch])
    min_rch = sorted(meds.items(), key=lambda item: item[1])[0][0]

    # Use min image as reference for histogram matching (need to convert to ints or it takes a VERY long time)
    reference = np.rint(img.xarray.data[min_rch[0], min_rch[1]] * 2 ** 16)
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            data = np.rint(img.xarray.data[r, ch] * 2 ** 16)
            matched = hist_match(data, reference)
            matched /= 2 ** 16
            img.xarray.data[r, ch] = deepcopy(matched)


def cli(
    input_dir: Path,
    output_dir: Path,
    clip_min: float = 95,
    opening_size: int = 10,
    aux_name: str = None,
):

    if not path.isdir(output_dir):
        makedirs(output_dir)

    reporter = open(
        path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_img_processing.log")), "w"
    )
    sys.stdout = reporter
    sys.stderr = reporter

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    exp = starfish.core.experiment.experiment.Experiment.from_json(
        str(input_dir / "experiment.json")
    )
    for fov in exp.keys():
        img = exp[fov].get_image("primary")
        register = exp[fov].get_image(aux_name)
        print("Fetched view " + fov)

        print("\tremoving background...")
        removeBg(opening_size, img)
        print("\tregistering...")
        registerImgs(img, register)
        print("\tmatching histo...")
        histoMatchImage(img)

        print("\tclip and scaling...")
        # Scale image, clipping all but the highest intensities to zero
        clip = starfish.image.Filter.ClipPercentileToZero(
            p_min=clip_min, p_max=99.9, is_volume=True, level_method=Levels.SCALE_BY_CHUNK
        )
        clip.run(img, in_place=True)

        print(f"\tView {fov} complete")
        # save modified image
        saveImg(output_dir, f"primary-{fov}", img)

        # save all aux views while we're here
        for view in exp[fov].image_types:
            if view != "primary":
                aux_img = exp[fov].get_image(view)
                saveImg(output_dir, f"{view}-{fov}", aux_img)

        print(f"View {fov} saved")

    saveExp(input_dir, output_dir)


if __name__ == "__main__":

    output_dir = Path("3_processed")

    p = ArgumentParser()

    p.add_argument("--input-dir", type=Path)
    p.add_argument("--clip-min", type=float, default=95)
    p.add_argument("--opening-size", type=int, default=10)
    p.add_argument("--register-aux-view", type=str, nargs="?")

    args = p.parse_args()

    cli(
        input_dir=args.input_dir,
        output_dir=output_dir,
        clip_min=args.clip_min,
        opening_size=args.opening_size,
        aux_name=args.register_aux_view,
    )
