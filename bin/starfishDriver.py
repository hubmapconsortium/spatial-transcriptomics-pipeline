#!/usr/bin/env python

import starfish
import starfish.data
from datetime import datetime
from starfish.spots import AssignTargets
from starfish import Codebook
from starfish.types import Axes, Coordinates, CoordinateValue, Features, TraceBuildingStrategies

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from os import path, makedirs
import sys

from tqdm import tqdm
from functools import partialmethod

# TODO parameter specifications, docstrings

def blobRunner(img, ref_img=None,
        min_sigma=(0.5,0.5,0.5), max_sigma=(8,8,8), num_sigma=10,
        threshold=0.1, overlap=0.5):
    bd = starfish.spots.FindSpots.BlobDetector(min_sigma, max_sigma, num_sigma, threshold, is_volume=False, overlap=overlap)
    results = None
    print(img)
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

def blobDriver(exp, blobRunnerKwargs, decodeRunnerKwargs):
    fovs = exp.keys()
    decoded = {}
    for fov in fovs:
        img = exp[fov].get_image("primary")
        blobs = blobRunner(img, **blobRunnerKwargs)
        decoded[fov] = decodeRunner(blobs, exp.codebook, **decodeRunnerKwargs)
    return decoded

def pixelDriver(exp, pixelRunnerKwargs):
    fovs = exp.keys()
    pixelRunner = starfish.spots.DetectPixels.PixelSpotDecoder(codebook=exp.codebook, **pixelRunnerKwargs)
    decoded = {}
    for fov in fovs:
        img = exp[fov].get_image("primary")
        decoded[fov] = pixelRunner(img)
    return decoded

def run(output_dir, experiment, blobRunnerKwargs, decodeRunnerKwargs, pixelRunnerKwargs):
    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    reporter = open(path.join(output_dir,datetime.now().strftime("%Y-%d-%m_%H:%M_starfish_runner.log")),'w')
    sys.stdout = reporter
    sys.stderr = reporter

    #disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # TODO add image processing options
    decoded = blobDriver(experiment, blobRunnerKwargs, decodeRunnerKwargs)
    # TODO add pixel based driver
    for fov in decoded.keys():
        decoded[fov].to_decoded_dataframe().to_csv(output_dir+fov+"_decoded.csv")
    
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
    # TODO LATER
    
    # blobRunner kwargs
    p.add_argument("--min-sigma", type=float, nargs="*")
    p.add_argument("--max-sigma", type=float, nargs="*")
    p.add_argument("--num-sigma", type=int, nargs="?")
    p.add_argument("--threshold", type=float, nargs="?")
    p.add_argument("--overlap", type=float, nargs="?")

    ### aside, are we going to want to include the ability to run a sweep?

    # decodeRunner kwargs
    p.add_argument("--decode-spots-method", type=str)
    p.add_argument("--trace-building-strategy", type=str, nargs="?") # only optional for SimpleLookupDecoder

    ## MetricDistance
    p.add_argument("--max-distance", type=float, nargs="?")
    p.add_argument("--min-intensity", type=float, nargs="?")
    p.add_argument("--metric", type=str, nargs="?") #NOTE also used in pixelRunner
    p.add_argument("--norm-order", type=int, nargs="?")
    p.add_argument("--anchor-round", type=int, nargs="?") # also used in PerRoundMaxChannel
    p.add_argument("--search-radius", type=int, nargs="?") # also used in PerRoundMaxChannel
    p.add_argument("--return-original-intensities", type=bool, nargs="?")
    p.add_argument("--filtered_results", type=bool, nargs="?") # defined by us

    # pixelRunner kwargs
    p.add_argument("--distance-threshold", type=float, nargs="?")
    p.add_argument("--magnitude-threshold", type=int, nargs="?")
    p.add_argument("--min-area", type=int, nargs="?")
    p.add_argument("--max-area", type=int, nargs="?")
    p.add_argument("--norm-order", type=int, nargs="?")

    args = p.parse_args()

    #for item in vars(args):
    #    print(item, ':', vars(args)[item])

    exploc = args.exp_loc / "experiment.json" 
    experiment = starfish.core.experiment.experiment.Experiment.from_json(str(exploc))

    blobRunnerKwargs = {}
    addKwarg(args, blobRunnerKwargs, "min_sigma")
    addKwarg(args, blobRunnerKwargs, "max_sigma")
    addKwarg(args, blobRunnerKwargs, "num_sigma")
    addKwarg(args, blobRunnerKwargs, "threshold")
    addKwarg(args, blobRunnerKwargs, "overlap")

    pixelRunnerKwargs = {}
    addKwarg(args, pixelRunnerKwargs, "metric")
    addKwarg(args, pixelRunnerKwargs, "distance_threshold")
    addKwarg(args, pixelRunnerKwargs, "magnitude_threshold")
    addKwarg(args, pixelRunnerKwargs, "min_area")
    addKwarg(args, pixelRunnerKwargs, "max_area")
    addKwarg(args, pixelRunnerKwargs, "norm_order")

    method = args.decode_spots_method
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

    decodeKwargs = {}
    addKwarg(args, decodeKwargs, "max_distance")
    addKwarg(args, decodeKwargs, "min_intensity")
    addKwarg(args, decodeKwargs, "metric")
    addKwarg(args, decodeKwargs, "norm_order")
    addKwarg(args, decodeKwargs, "anchor_round")
    addKwarg(args, decodeKwargs, "search_radius")
    
    if trace_strat:
        decodeKwargs["trace_building_strategy"] = trace_strat
    
    decodeRunnerKwargs = {"decodeKwargs": decodeKwargs, "callableDecoder": method}
    addKwarg(args, decodeRunnerKwargs, "return_original_intensities")

    run(output_dir, experiment, blobRunnerKwargs, decodeRunnerKwargs, pixelRunnerKwargs)

