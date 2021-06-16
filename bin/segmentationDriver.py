#!/usr/bin/env python

from time import time
from datetime import datetime
from pathlib import Path
from os import path, makedirs
from glob import glob
from argparse import ArgumentParser
import sys
from typing import Set
import starfish
import starfish.data
from starfish.types import Levels, Axes, Features
from starfish import DecodedIntensityTable, BinaryMaskCollection, IntensityTable, ImageStack
from starfish.image import Filter as ImgFilter
from starfish.morphology import Binarize, Filter, Merge, Segment
from starfish.spots import AssignTargets
import numpy as np
import pandas as pd

def masksFromRoi(img_stack, roi_set, file_formats):
    masks = []
    for i in range(len(img_stack)):
        mask_name = ("{}/"+file_formats).format(roi_set, i)
        masks.append(BinaryMaskCollection.from_fiji_roi_set(mask_name, img_stack[i]))
    return masks

def masksFromLabeledImages(img_stack, labeled_image, file_formats_labeled):
    masks = []
    for i in range(len()):
        label_name = ("{}/"+file_formats_labeled).format(labeled_image,i)
        masks.append(BinaryMaskCollection.from_external_labeled_image(label_name, img_stack[i]))
    return masks

def masksFromWatershed(img_stack, img_thresh, min_dist, min_size, max_size, masking_radius):
    wt_filt = ImgFilter.WhiteTophat(masking_radius, is_volume=False)
    thresh_filt = Binarize.ThresholdBinarize(img_thresh)
    area_filt = Filter.AreaFilter(min_size, max_size)
    area_mask = Filter.Reduce("logical_or", lambda shape: np.zeros(shape=shape, dtype=np.bool))
    segmenter = Segment.WatershedSegment()
    masks = []
    for img in img_stack:
        img_flat = img.reduce({Axes.ROUND}, func="max")
        working_img = wt_filt.run(img_flat, in_place=False)
        working_img = thresh_filt.run(working_img)
        working_img = area_filt.run(working_img)
        masks.append(segmenter.run(img_flat,working_img))
    return masks

#def saveTable(table, savename):
#    intensities = IntensityTable(table.where(table[Features.PASSES_THRESHOLDS], drop=True))
#    traces = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))
#    traces = traces.to_features_dataframe()
#    traces.to_csv(savename)

def run(input_loc, exp_loc, output_loc, fov_count, aux_name, roiKwargs, labeledKwargs, watershedKwargs):
    # read in netcdfs based on how we saved prev step
    results = []
    masks = []
    for f in glob("{}/cdf/*_decoded.cdf".format(input_loc)):
        results.append(DecodedIntensityTable.open_netcdf(f))
        print("loaded "+f)

    if not path.isdir(output_dir):
        makedirs(output_dir)
        print("made "+output_dir)

    if not path.isdir(output_dir+"csv/"):
        makedirs(output_dir+"csv")
        print("made "+output_dir+"csv")

    if not path.isdir(output_dir+"cdf/"):
        makedirs(output_dir+"cdf")
        print("made "+output_dir+"cdf")

    # load in the images we want to look at
    exp = starfish.core.experiment.experiment.Experiment.from_json(str(exp_loc / "experiment.json"))
    img_stack = []
    for key in exp.keys():
        img_stack.append(exp[key].get_image(aux_name))

    # determine how we generate mask, then make it
    if len(roiKwargs.keys()) > 0:
        # then apply roi
        masks = masksFromRoi(img_stack, **roiKwargs)
    elif len(labeledKwargs.keys()) > 0:
        # then apply images
        masks = masksFromLabeledImages(img_stack, **labeledKwargs)
    elif len(watershedKwargs.keys()) > 0:
        # then go thru watershed pipeline
        masks = masksFromWatershed(img_stack, **watershedKwargs)
    else:
        #throw error
        raise Exception("Parameters do not specify means of defining mask.")

    # apply mask to tables, save results
    al = AssignTargets.Label()
    for i in range(fov_count):
        labeled = al.run(masks[i], results[i])
        labeled = labeled[labeled.cell_id != 'nan']
        labeled.to_decoded_dataframe().save_csv(output_dir+"csv/"+str(i)+"_segmented.csv")
        labeled.to_netcdf(output_dir+"csv/"+str(i)+"_segmented.cdf")


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
    p.add_argument("--img-thresh", type=float, nargs="?")
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
    addKwarg(args, watershedKwargs, "img_thresh")
    addKwarg(args, watershedKwargs, "min_dist")
    addKwarg(args, watershedKwargs, "min_size")
    addKwarg(args, watershedKwargs, "max_size")
    addKwarg(args, watershedKwargs, "masking_radius")

    run(input_dir, exp_dir, output_dir, fov_count, aux_name, roiKwargs, labeledKwargs, watershedKwargs)
