#!/usr/bin/env python

import starfish
import starfish.data
from datetime import datetime
from starfish.spots import AssignTargets
from starfish import Codebook, ImageStack, IntensityTable
from starfish.types import Axes, Coordinates, CoordinateValue, Features, TraceBuildingStrategies, Levels

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from os import path, makedirs
import sys
from typing import Set

from tqdm import tqdm
from functools import partialmethod

# TODO parameter specifications, docstrings

def imagePrePro(imgs, 
        flatten_axes: Set[Axes] = None, 
        gaussian_lowpass: float = None, 
        clip: bool = True, 
        zero_by_magnitude: float = None, 
        ref: bool= False):
    ret_imgs = {}
    ret_ref = {}
    for fov in imgs.keys():
        img = imgs[fov]
        if flatten_axes:
            img = img.reduce(flatten_axes, func="max")
        if clip:
            clip = starfish.image.Filter.Clip(p_max=99.9, is_volume=True, level_method=Levels.SCALE_BY_CHUNK)
            clip.run(img, in_place=True)
        if gaussian_lowpass:
            gaus = starfish.image.Filter.GaussianLowPass(gaussian_lowpass)
            gaus.run(img, in_place=True)
        if zero_by_magnitude:
            z_filt = starfish.image.Filter.ZeroByChannelMagnitude(thresh=zero_by_magnitude, normalize=False)
            z_filt.run(img, in_place=True)
        ref_img = None
        if ref:
            ref_img = img.reduce({Axes.CH, Axes.ROUND, Axes.ZPLANE}, func="max")
        ret_imgs[fov] = img
        ret_ref[fov] = ref_img
    return ret_imgs, ret_ref

def blobRunner(img, ref_img=None,
        min_sigma=(0.5,0.5,0.5), max_sigma=(8,8,8), num_sigma=10,
        threshold=0.1, is_volume=False, overlap=0.5):
    bd = starfish.spots.FindSpots.BlobDetector(min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
            threshold=threshold, is_volume=is_volume, overlap=overlap)
    results = None
    print(vars(bd))
    print(vars(img))
    if ref_img:
        results = bd.run(image_stack=img, reference_image=ref_img)
    else:
        results = bd.run(image_stack=img)
    return results

def decodeRunner(spots, codebook, decoderKwargs,
        callableDecoder=starfish.spots.DecodeSpots.PerRoundMaxChannel, 
        filtered_results=True):
    decoder = callableDecoder(codebook=codebook, **decoderKwargs)
    results = decoder.run(spots=spots)
    if filtered_results:
        results = results.loc[results[Features.PASSES_THRESHOLDS]]
        results = results[results.target != 'nan']
    return results

def blobDriver(imgs, ref_img, codebook, blobRunnerKwargs, decodeRunnerKwargs):
    fovs = imgs.keys()
    decoded = {}
    for fov in fovs:
        blobs = blobRunner(imgs[fov], ref_img=ref_img[fov] if ref_img else None, **blobRunnerKwargs)
        decoded[fov] = decodeRunner(blobs, codebook, **decodeRunnerKwargs)
    return decoded

def pixelDriver(imgs, codebook, pixelRunnerKwargs):
    fovs = imgs.keys()
    pixelRunner = starfish.spots.DetectPixels.PixelSpotDecoder(codebook=codebook, **pixelRunnerKwargs)
    decoded = {}
    for fov in fovs:
        decoded[fov] = pixelRunner.run(imgs[fov])[0]
    return decoded

def saveTable(table, savename):
    intensities = IntensityTable(table.where(table[Features.PASSES_THRESHOLDS], drop=True))
    traces = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    traces = traces.to_features_dataframe()
    traces.to_csv(savename)


def run(output_dir, experiment, blob_based, imagePreProKwargs, blobRunnerKwargs, decodeRunnerKwargs, pixelRunnerKwargs):
    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    if not path.isdir(output_dir+"csv/"):
        makedirs(output_dir+"csv")

    if not path.isdir(output_dir+"cdf/"):
        makedirs(output_dir+"cdf")

    reporter = open(path.join(output_dir,datetime.now().strftime("%Y-%d-%m_%H:%M_starfish_runner.log")),'w')
    sys.stdout = reporter
    sys.stderr = reporter

    print("output_dir: {}\nexp: {}\nblob_based: {}\nprepro: {}\nblobrunner: {}\ndecoderunner: {}\npixelrunner: {}\n".format(output_dir, experiment, blob_based, imagePreProKwargs, blobRunnerKwargs, decodeRunnerKwargs, pixelRunnerKwargs))

    #disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    imgs = {}
    for fov in experiment.keys():
        imgs[fov] = experiment[fov].get_image("primary")

    imgs, ref_imgs = imagePrePro(imgs, **imagePreProKwargs)
    
    decoded = {}
    if blob_based:
        decoded = blobDriver(imgs, ref_imgs, experiment.codebook, blobRunnerKwargs, decodeRunnerKwargs)
    else:
        decoded = pixelDriver(imgs, experiment.codebook, pixelRunnerKwargs)
    
    #saving 
    for fov in decoded.keys():
        saveTable(decoded[fov], output_dir+"csv/"+fov+"_decoded.csv")
        #decoded[fov].to_decoded_dataframe().save_csv(output_dir+fov+"_decoded.csv") 
        decoded[fov].to_netcdf(output_dir+"cdf/"+fov+"_decoded.cdf")

    sys.stdout = sys.__stdout__
    return 0

def addKwarg(parser, kwargdict, var):
    result = getattr(parser, var)
    if result:
        kwargdict[var] = result

if __name__ == "__main__":

    output_dir = "4_Decoded/"
    
    p = ArgumentParser()

    # inputs
    p.add_argument("--exp-loc", type=Path)

    # image processing args
    p.add_argument("--flatten-axes", type=str, nargs="*")
    p.add_argument("--gaussian-lowpass", type=float, nargs="?")
    p.add_argument("--clip-img", dest='clip_img', action='store_true')
    p.add_argument("--zero-by-magnitude", type=float, nargs="?")
    p.add_argument("--use-ref-img", dest='use_ref_img', action='store_true')
    p.set_defaults(clip_img=False)
    p.set_defaults(use_ref_img=False)

    # blobRunner kwargs
    p.add_argument("--min-sigma", type=float, nargs="*")
    p.add_argument("--max-sigma", type=float, nargs="*")
    p.add_argument("--num-sigma", type=int, nargs="?")
    p.add_argument("--threshold", type=float, nargs="?")
    p.add_argument("--overlap", type=float, nargs="?")
    p.add_argument("--is-volume", dest="is_volume", action='store_true')

    ### aside, are we going to want to include the ability to run a sweep?

    # decodeRunner kwargs
    p.add_argument("--decode-spots-method", type=str)
    p.add_argument("--trace-building-strategy", type=str, nargs="?") # only optional for SimpleLookupDecoder

    ## MetricDistance
    p.add_argument("--max-distance", type=float, nargs="?")
    p.add_argument("--min-intensity", type=float, nargs="?")
    p.add_argument("--metric", type=str, nargs="?") #NOTE also used in pixelRunner
    p.add_argument("--norm-order", type=int, nargs="?") #NOTE also used in pixelRunner
    p.add_argument("--anchor-round", type=int, nargs="?") # also used in PerRoundMaxChannel
    p.add_argument("--search-radius", type=int, nargs="?") # also used in PerRoundMaxChannel
    p.add_argument("--return-original-intensities", type=bool, nargs="?")
    p.add_argument("--filtered_results", dest='filtered_results', action='store_true') # defined by us
    p.set_defaults(filtered_results=False)

    # pixelRunner kwargs
    p.add_argument("--distance-threshold", type=float, nargs="?")
    p.add_argument("--magnitude-threshold", type=float, nargs="?")
    p.add_argument("--min-area", type=int, nargs="?")
    p.add_argument("--max-area", type=int, nargs="?")

    args = p.parse_args()

    #for item in vars(args):
    #    print(item, ':', vars(args)[item])

    exploc = args.exp_loc / "experiment.json" 
    experiment = starfish.core.experiment.experiment.Experiment.from_json(str(exploc))

    imagePreProKwargs = {}
    addKwarg(args, imagePreProKwargs, "gaussian_lowpass")
    addKwarg(args, imagePreProKwargs,"zero_by_magnitude")
    imagePreProKwargs["clip"] = args.clip_img
    imagePreProKwargs["ref"] = args.use_ref_img
    if args.flatten_axes:
        ax_set = set()
        for ax in args.flatten_axes:
            if ax == "ZPLANE":
                ax_set.add(Axes.ZPLANE)
            elif ax == "CH":
                ax_set.add(Axes.CH)
            elif ax == "ROUND":
                ax_set.add(Axes.ROUND)
        imagePreProKwargs["flatten_axes"] = ax_set
    else:
        imagePreProKwargs["flatten_axes"] = False

    blobRunnerKwargs = {}
    #addKwarg(args, blobRunnerKwargs, "min_sigma")
    #addKwarg(args, blobRunnerKwargs, "max_sigma")
    if args.min_sigma:
        blobRunnerKwargs["min_sigma"] = tuple(args.min_sigma)
    if args.max_sigma:
        blobRunnerKwargs["max_sigma"] = tuple(args.max_sigma)
    addKwarg(args, blobRunnerKwargs, "num_sigma")
    addKwarg(args, blobRunnerKwargs, "threshold")
    addKwarg(args, blobRunnerKwargs, "overlap")
    addKwarg(args, blobRunnerKwargs, "is_volume")

    pixelRunnerKwargs = {}
    addKwarg(args, pixelRunnerKwargs, "metric")
    addKwarg(args, pixelRunnerKwargs, "distance_threshold")
    addKwarg(args, pixelRunnerKwargs, "magnitude_threshold")
    addKwarg(args, pixelRunnerKwargs, "min_area")
    addKwarg(args, pixelRunnerKwargs, "max_area")
    addKwarg(args, pixelRunnerKwargs, "norm_order")

    decodeKwargs = {}
    
    method = args.decode_spots_method
    blob_based = args.distance_threshold is None
    if blob_based:
        if method == "PerRoundMaxChannel":
            method = starfish.spots.DecodeSpots.PerRoundMaxChannel
        elif method == "MetricDistance":
            method = starfish.spots.DecodeSpots.MetricDistance
        elif method == "SimpleLookupDecoder":
            method = starfish.spots.DecodeSpots.SimpleLookupDecoder
        else:
            raise Exception("DecodeSpots method "+str(method)+" is not a valid method.")

        trace_strat = args.trace_building_strategy
        if method != starfish.spots.DecodeSpots.SimpleLookupDecoder:
            if trace_strat == "SEQUENTIAL":
                trace_strat = TraceBuildingStrategies.SEQUENTIAL
            elif trace_strat == "EXACT_MATCH":
                trace_strat = TraceBuildingStrategies.EXACT_MATCH
            elif trace_strat == "NEAREST_NEIGHBOR":
                trace_strat = TraceBuildingStrategies.NEAREST_NEIGHBOR
            else:
                raise Exception("TraceBuildingStrategies "+str(trace_strat)+" is not valid.")
            decodeKwargs["trace_building_strategy"] = trace_strat

    addKwarg(args, decodeKwargs, "max_distance")
    addKwarg(args, decodeKwargs, "min_intensity")
    addKwarg(args, decodeKwargs, "metric")
    addKwarg(args, decodeKwargs, "norm_order")
    addKwarg(args, decodeKwargs, "anchor_round")
    addKwarg(args, decodeKwargs, "search_radius")
    
    decodeRunnerKwargs = {"decoderKwargs": decodeKwargs, "callableDecoder": method}
    addKwarg(args, decodeRunnerKwargs, "return_original_intensities")

    run(output_dir, experiment, blob_based, imagePreProKwargs, blobRunnerKwargs, decodeRunnerKwargs, pixelRunnerKwargs)

