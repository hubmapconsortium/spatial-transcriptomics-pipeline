#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partialmethod
from glob import glob
from os import makedirs, path
from pathlib import Path
from time import time
from typing import List, Mapping, Set

import numpy as np
import pandas as pd
import starfish
import starfish.data
from starfish import (
    BinaryMaskCollection,
    DecodedIntensityTable,
    ImageStack,
    IntensityTable,
)
from starfish.image import Filter as ImgFilter
from starfish.morphology import Binarize, Filter, Merge, Segment
from starfish.spots import AssignTargets
from starfish.types import Axes, Features, Levels
from tqdm import tqdm


def masksFromRoi(
    img_stack: List[ImageStack], roi_set: Path, file_formats: str
) -> List[BinaryMaskCollection]:
    """
    Return a list of masks from provided RoiSet.zip files.

    Parameters
    ----------
    img_stack: list[ImageStack]
        The images that the masks are to be applied to, provided per FOV.
    roi_set: Path
        Directory containing RoiSet files.
    file_formats: str
        String that will have .format() applied for each FOV.  Will be appended to roi_set.

    Returns
    -------
    list[BinaryMaskCollection]:
        Binary masks for each FOV.
    """
    masks = []
    for i in range(len(img_stack)):
        mask_name = ("{}/" + file_formats).format(roi_set, i)
        masks.append(BinaryMaskCollection.from_fiji_roi_set(mask_name, img_stack[i]))
    return masks


def masksFromLabeledImages(
    img_stack: List[ImageStack], labeled_image: Path, file_formats_labeled: str
) -> List[BinaryMaskCollection]:
    """
    Returns a list of masks from the provided labeled images.

    Parameters
    ----------
    img_stack: list[ImageStack]
        The images that the masks will be applied to, provided per FOV.
    labeled_image: Path
        Directory of labeled images with image segmentation data, such as from ilastik classification.
    file_formats_labeled: str
        Layout for name of each labelled image. Will be formatted with String.format([fov index])

    Returns
    -------
    list[BinaryMaskCollection]:
        Binary masks for each FOV.
    """
    masks = []
    for i in range(len()):
        label_name = ("{}/" + file_formats_labeled).format(labeled_image, i)
        masks.append(BinaryMaskCollection.from_external_labeled_image(label_name, img_stack[i]))
    return masks


def masksFromWatershed(
    img_stack: List[ImageStack],
    img_threshold: float,
    min_dist: int,
    min_size: int,
    max_size: int,
    masking_radius: int,
) -> List[BinaryMaskCollection]:
    """
    Runs a primitive thresholding and watershed pipeline to generate segmentation masks.

    Parameters
    ----------
    img_threshold: float
        Global threshold value for images.
    min_dist: int
        Minimum distance (pixels) between distance transformed peaks.
    min_size: int
        Minimum size for a cell (in pixels)
    max_size: int
        Maxiumum size for a cell (in pixels)
    masking_radius: int
        Radius for white tophat noise filter.

    Returns
    -------
    list[BinaryMaskCollection]:
        Binary masks for each FOV.
    """
    wt_filt = ImgFilter.WhiteTophat(masking_radius, is_volume=False)
    thresh_filt = Binarize.ThresholdBinarize(img_threshold)
    min_dist_label = Filter.MinDistanceLabel(min_dist, 1)
    area_filt = Filter.AreaFilter(min_area=min_size, max_area=max_size)
    area_mask = Filter.Reduce("logical_or", lambda shape: np.zeros(shape=shape, dtype=bool))
    segmenter = Segment.WatershedSegment()
    masks = []
    for img in img_stack:
        img_flat = img.reduce({Axes.ROUND}, func="max")
        working_img = wt_filt.run(img_flat, in_place=False)
        working_img = thresh_filt.run(working_img)
        labeled = min_dist_label.run(working_img)
        working_img = area_filt.run(labeled)
        working_img = area_mask.run(working_img)
        masks.append(segmenter.run(img_flat, labeled, working_img))
    return masks


def run(
    input_loc: Path,
    exp_loc: Path,
    output_loc: str,
    fov_count: int,
    aux_name: str,
    roiKwargs: dict,
    labeledKwargs: dict,
    watershedKwargs: dict,
):
    """
    Main class for generating and applying masks then saving output.

    Parameters
    ----------
    input_loc: Path
        Location of input cdf files, as formatted by starfishRunner.cwl
    exp_loc: Path
        Directory that contains "experiment.json" file for the experiment.
    output_loc: str
        Path to directory where output will be saved.
    fov_count: int
        The number of FOVs in the experiment.
    aux_name: str
        The name of the auxillary view to look at for image segmentation.
    roiKwargs: dict
        Dictionary with arguments for reading in masks from an RoiSet. See masksFromRoi.
    labeledKwargs: dict
        Dictionary with arguments for reading in masks from a labeled image. See masksFromLabeledImages.
    watershedKwargs: dict
        Dictionary with arguments for running basic watershed pipeline. See masksFromWatershed.
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # redirecting output to log
    reporter = open(
        path.join(output_dir, datetime.now().strftime("%Y-%d-%m_%H:%M_starfish_segmenter.log")),
        "w",
    )
    sys.stdout = reporter
    sys.stderr = reporter

    if not path.isdir(output_dir + "csv/"):
        makedirs(output_dir + "csv")
        print("made " + output_dir + "csv")

    if not path.isdir(output_dir + "cdf/"):
        makedirs(output_dir + "cdf")
        print("made " + output_dir + "cdf")

    # read in netcdfs based on how we saved prev step
    results = []
    keys = []
    masks = []
    for f in glob("{}/cdf/*_decoded.cdf".format(input_loc)):
        results.append(DecodedIntensityTable.open_netcdf(f))
        name = f[len(str(input_loc)) + 5 : -12]
        print("found fov key: " + name)
        keys.append(name)
        print("loaded " + f)

    # load in the images we want to look at
    exp = starfish.core.experiment.experiment.Experiment.from_json(
        str(exp_loc / "experiment.json")
    )
    img_stack = []
    for key in exp.keys():
        img_stack.append(exp[key].get_image(aux_name))

    # determine how we generate mask, then make it
    if len(roiKwargs.keys()) > 0:
        # then apply roi
        print("applying Roi mask")
        masks = masksFromRoi(img_stack, **roiKwargs)
    elif len(labeledKwargs.keys()) > 0:
        # then apply images
        print("applying labeled image mask")
        masks = masksFromLabeledImages(img_stack, **labeledKwargs)
    elif len(watershedKwargs.keys()) > 0:
        # then go thru watershed pipeline
        print("running basic threshold and watershed pipeline")
        masks = masksFromWatershed(img_stack, **watershedKwargs)
    else:
        # throw error
        raise Exception("Parameters do not specify means of defining mask.")

    # apply mask to tables, save results
    al = AssignTargets.Label()
    for i in range(fov_count):
        labeled = al.run(masks[i], results[i])
        labeled = labeled[labeled.cell_id != "nan"]
        labeled.to_decoded_dataframe().save_csv(
            output_dir + "csv/df_" + keys[i] + "_segmented.csv"
        )
        labeled.to_netcdf(output_dir + "cdf/df_" + keys[i] + "_segmented.cdf")
        labeled.to_expression_matrix().to_pandas().to_csv(
            output_dir + "csv/exp_" + keys[i] + "_segmented.csv"
        )
        labeled.to_expression_matrix().save(output_dir + "cdf/exp_" + keys[i] + "_segmented.cdf")
        print("saved fov key: {}, index {}".format(keys[i], i))

    sys.stdout = sys.__stdout__


def addKwarg(parser, kwargdict, var):
    result = getattr(parser, var)
    if result:
        kwargdict[var] = result


if __name__ == "__main__":
    output_dir = "5_Segmented/"
    p = ArgumentParser()

    p.add_argument("--decoded-loc", type=Path)
    p.add_argument("--fov-count", type=int)
    p.add_argument("--exp-loc", type=Path)
    p.add_argument("--aux-name", type=str)

    # for importing roi set
    p.add_argument("--roi-set", type=Path, nargs="?")
    p.add_argument("--file-formats", type=str, nargs="?")

    # for using a labeled image
    p.add_argument("--labeled-image", type=Path, nargs="?")
    p.add_argument("--file-formats-labeled", type=str, nargs="?")

    # for runnning basic watershed pipeline using starfish
    p.add_argument("--img-threshold", type=float, nargs="?")
    p.add_argument("--min-dist", type=int, nargs="?")
    p.add_argument("--min-size", type=int, nargs="?")
    p.add_argument("--max-size", type=int, nargs="?")
    p.add_argument("--masking-radius", type=int, nargs="?")

    args = p.parse_args()

    fov_count = args.fov_count
    input_dir = args.decoded_loc
    exp_dir = args.exp_loc
    aux_name = args.aux_name

    roiKwargs = {}
    addKwarg(args, roiKwargs, "roi_set")
    addKwarg(args, roiKwargs, "file_formats")

    labeledKwargs = {}
    addKwarg(args, labeledKwargs, "labeled_image")
    addKwarg(args, labeledKwargs, "file_formats_labeled")

    watershedKwargs = {}
    addKwarg(args, watershedKwargs, "img_threshold")
    addKwarg(args, watershedKwargs, "min_dist")
    addKwarg(args, watershedKwargs, "min_size")
    addKwarg(args, watershedKwargs, "max_size")
    addKwarg(args, watershedKwargs, "masking_radius")

    run(
        input_dir,
        exp_dir,
        output_dir,
        fov_count,
        aux_name,
        roiKwargs,
        labeledKwargs,
        watershedKwargs,
    )
