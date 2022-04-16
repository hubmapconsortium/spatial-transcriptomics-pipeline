#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partialmethod
from os import makedirs, path
from pathlib import Path
from typing import Callable, Mapping, Set, Tuple

import numpy as np
import pandas as pd
import starfish
import starfish.data
from starfish import (
    Codebook,
    DecodedIntensityTable,
    Experiment,
    ImageStack,
    IntensityTable,
)
from starfish.core.types import SpotAttributes, SpotFindingResults
from starfish.spots import AssignTargets
from starfish.types import (
    Axes,
    Coordinates,
    CoordinateValue,
    Features,
    Levels,
    TraceBuildingStrategies,
)
from tqdm import tqdm


def blobRunner(
    img: ImageStack,
    ref_img: ImageStack = None,
    min_sigma: Tuple[float, float, float] = (0.5, 0.5),
    max_sigma: Tuple[float, float, float] = (8, 8),
    num_sigma: int = 10,
    threshold: float = 0.1,
    is_volume: bool = False,
    detector_method: str = "blob_log",
    overlap: float = 0.5,
) -> SpotFindingResults:
    """
    The driver method for running blob-based spot detection, given a set of parameters.

    Parameters
    ----------
    img : ImageStack
        The target image for spot detection
    ref_img : ImageStack
        If provided, the reference image to be used for spot detection.
    The remaining parameters are passed as-is to the starfish.spots.FindSpots.BlobDetector object.

    Returns
    -------
    SpotFindingResults:
        Starfish wrapper for an xarray with the spot data.

    """
    bd = starfish.spots.FindSpots.BlobDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        is_volume=is_volume,
        detector_method=detector_method,
        overlap=overlap,
    )
    results = None
    # print(vars(bd))
    # print(vars(img))
    if ref_img:
        results = bd.run(image_stack=img, reference_image=ref_img)
    else:
        results = bd.run(image_stack=img)
    return results


def decodeRunner(
    spots: SpotFindingResults,
    codebook: Codebook,
    decoderKwargs: dict,
    callableDecoder: Callable = starfish.spots.DecodeSpots.PerRoundMaxChannel,
    filtered_results: bool = True,
) -> DecodedIntensityTable:
    """
    The driver for decoding spots into barcodes.

    Parameters
    ----------
    spots: SpotFindingResults
        Input spots for decoding
    codebook: Codebook
        The codebook for the experiment, to be used for decoding.
    decoderKwargs: dict
        Dictionary with optional arguments to be passed to the decoder object.
    callabeDecoder: Callable
        The method for creating a decoder of the desired type. Defaults to PerRoundMaxChannel.
    filtered_results: bool
        If true, rows with no target or that do not pass thresholds will be removed.

    Returns
    -------
    DecodedIntensityTable:
        Starfish wrapper for an xarray with the labeled transcripts.
    """
    decoder = callableDecoder(codebook=codebook, **decoderKwargs)
    results = decoder.run(spots=spots)
    if filtered_results:
        results = results.loc[results[Features.PASSES_THRESHOLDS]]
        results = results[results.target != "nan"]
    return results


def blobDriver(
    imgs: Mapping[str, ImageStack],
    ref_img: Mapping[str, ImageStack],
    codebook: Codebook,
    blobRunnerKwargs: dict,
    decodeRunnerKwargs: dict,
    output_dir: str,
) -> Tuple[Mapping[str, SpotFindingResults], Mapping[str, DecodedIntensityTable]]:
    """
    Method to handle the blob-based version of the detection and decoding steps.

    Parameters
    ----------
    imgs : Mapping[str, ImageStack]
        The images to be processed, with the corresponding FOV name as the key.
    ref_img : Mapping[str, ImageStack]
        If provided, the reference image that will be used during spot detection.
    codebook : Codebook
        The codebook for the experiment, to be used for decoding.
    blobRunnerKwargs : dict
        A dictionary of optional parameters to be used for spot detection. Refer to blobRunner for specifics.
    decodeRunnerKwargs : dict
        A dictionary of optional parameters to be used in decoding. Refer to decodeRunner for specifics.
    output_dir: str
        The root location of the scripts output, to save blobs before they are put through the decoder.  Will not be saved if this is not passed.

    Returns
    -------
    SpotFindingResults:
        The container with information regarding spot locations.

    Mapping[str, DecodedIntensityTable]:
        A dictionary with the decoded tables stored by FOV name.
    """
    fovs = imgs.keys()
    blobs = {}
    decoded = {}
    for fov in fovs:
        blob = blobRunner(imgs[fov], ref_img=ref_img[fov] if ref_img else None, **blobRunnerKwargs)
        print("found total spots {}".format(blob.count_total_spots()))
        if ref_img:
            # Starfish doesn't apply threshold correctly when a ref image is used
            # so go through results and manually apply it.
            if blobRunnerKwargs["threshold"]:
                thresh = blobRunnerKwargs["threshold"]
            else:
                thresh = 0.1
            for k, v in blob.items():
                data = v.spot_attrs.data
                high = data[data["intensity"] > thresh]
                v.spot_attrs = SpotAttributes(high)
            print(f"removed spots below threshold, now {blob.count_total_spots()} total spots")
        blobs[fov] = blob
        if output_dir:
            blob.save(output_dir + "spots/" + fov + "_")
            print("spots saved.")
        decoded[fov] = decodeRunner(blob, codebook, **decodeRunnerKwargs)
    return blobs, decoded


def pixelDriver(
    imgs: Mapping[str, ImageStack], codebook: Codebook, pixelRunnerKwargs: dict
) -> Mapping[str, DecodedIntensityTable]:
    """
    Method to run Starfish's PixelSpotDecoder on the provided ImageStack

    Parameters
    ----------
    imgs : Mapping[str, ImageStack]
        The images to be processed, with the corresponding FOV name as the key.
    codebook : Codebook
        The codebook for the experiment, to be used for decoding.
    pixelRunnerKwargs : dict
        A dictionary of parameters to be passed to PixelSpotDecoder.


    Returns
    -------
    Mapping[str, DecodedIntensityTable]:
        A dictionary with the decoded tables stored by FOV name.

    """
    fovs = imgs.keys()
    pixelRunner = starfish.spots.DetectPixels.PixelSpotDecoder(
        codebook=codebook, **pixelRunnerKwargs
    )
    decoded = {}
    for fov in fovs:
        decoded[fov] = pixelRunner.run(imgs[fov])[0]
    return decoded


def saveTable(table: DecodedIntensityTable, savename: str):
    """
    Reformats and saves a DecodedIntensityTable.
    """
    intensities = IntensityTable(table.where(table[Features.PASSES_THRESHOLDS], drop=True))
    traces = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    # traces = table.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    traces = traces.to_features_dataframe()
    traces.to_csv(savename)


def run(
    output_dir: str,
    experiment: Experiment,
    blob_based: bool,
    use_ref: bool,
    blobRunnerKwargs: dict,
    decodeRunnerKwargs: dict,
    pixelRunnerKwargs: dict,
):
    """
    Main method for executing runs.  Sets up directories and calls appropriate driver methods.

    Parameters
    ----------
    output_dir: str
        Location to put all output from this tool.  Dir will be created if not present.
    experiment: Experiment
        Experiment object with corresponding images and codebook.
    blob_based: bool
        If true, use blob-detection and decoding methods. Else, use pixel-based methods.
    use_ref: bool
        If true, a reference image will be used and created by flattening the fov.
    blobRunnerKwargs: dict
        Dictionary with arguments for blob detection. Refer to blobRunner.
    decodeRunnerKwargs: dict
        Dictionary with arguments for spot-based decoding. Refer to decodeRunner.
    pixelRunnerKwargs: dict
        Dictionary with arguments for pixel-based detection and decoding.  Refer to starfish PixelSpotDecoder.

    """
    if not path.isdir(output_dir):
        makedirs(output_dir)

    if not path.isdir(output_dir + "csv/"):
        makedirs(output_dir + "csv")

    if not path.isdir(output_dir + "cdf/"):
        makedirs(output_dir + "cdf")

    if blob_based and not path.isdir(output_dir + "spots/"):
        makedirs(output_dir + "spots")

    reporter = open(
        path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_starfish_runner.log")), "w"
    )
    sys.stdout = reporter
    sys.stderr = reporter

    print(
        "output_dir: {}\nexp: {}\nblob_based: {}\nuse_ref: {}\nblobrunner: {}\ndecoderunner: {}\npixelrunner: {}\n".format(
            output_dir,
            experiment,
            blob_based,
            use_ref,
            blobRunnerKwargs,
            decodeRunnerKwargs,
            pixelRunnerKwargs,
        )
    )

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    imgs = {}
    for fov in experiment.keys():
        imgs[fov] = experiment[fov].get_image("primary")

    ref_imgs = None
    if use_ref:
        ref_imgs = {}
        for fov in experiment.keys():
            ref_imgs[fov] = imgs[fov].reduce({Axes.CH, Axes.ROUND, Axes.ZPLANE}, func="max")

    decoded = {}
    if blob_based:
        blobs, decoded = blobDriver(
            imgs, ref_imgs, experiment.codebook, blobRunnerKwargs, decodeRunnerKwargs, output_dir
        )
    else:
        decoded = pixelDriver(imgs, experiment.codebook, pixelRunnerKwargs)

    # saving
    for fov in decoded.keys():
        saveTable(decoded[fov], output_dir + "csv/" + fov + "_decoded.csv")
        # decoded[fov].to_decoded_dataframe().save_csv(output_dir+fov+"_decoded.csv")
        decoded[fov].to_netcdf(output_dir + "cdf/" + fov + "_decoded.cdf")

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

    # blobRunner kwargs
    p.add_argument("--min-sigma", type=float, nargs="*")
    p.add_argument("--max-sigma", type=float, nargs="*")
    p.add_argument("--num-sigma", type=int, nargs="?")
    p.add_argument("--threshold", type=float, nargs="?")
    p.add_argument("--overlap", type=float, nargs="?")
    p.add_argument("--detector-method", type=str, nargs="?")
    p.add_argument("--is-volume", dest="is_volume", action="store_true")
    p.add_argument("--use-ref-img", dest="use_ref_img", action="store_true")
    p.set_defaults(use_ref_img=False)
    ### aside, are we going to want to include the ability to run a sweep?

    # decodeRunner kwargs
    p.add_argument("--decode-spots-method", type=str)
    p.add_argument(
        "--trace-building-strategy", type=str, nargs="?"
    )  # only optional for SimpleLookupDecoder

    ## MetricDistance
    p.add_argument("--max-distance", type=float, nargs="?")
    p.add_argument("--min-intensity", type=float, nargs="?")
    p.add_argument("--metric", type=str, nargs="?")  # NOTE also used in pixelRunner
    p.add_argument("--norm-order", type=int, nargs="?")  # NOTE also used in pixelRunner
    p.add_argument("--anchor-round", type=int, nargs="?")  # also used in PerRoundMaxChannel
    p.add_argument(
        "--search-radius", type=int, nargs="?"
    )  # also used in PerRoundMaxChannel, CheckAll
    p.add_argument("--return-original-intensities", type=bool, nargs="?")
    p.add_argument(
        "--filtered_results", dest="filtered_results", action="store_true"
    )  # defined by us
    p.set_defaults(filtered_results=False)

    ## CheckAll
    p.add_argument("--error-rounds", type=int, nargs="?")
    p.add_argument("--mode", type=str, nargs="?")
    p.add_argument("--physical-coords", dest="physical_coords", action="store_true")

    # pixelRunner kwargs
    p.add_argument("--distance-threshold", type=float, nargs="?")
    p.add_argument("--magnitude-threshold", type=float, nargs="?")
    p.add_argument("--min-area", type=int, nargs="?")
    p.add_argument("--max-area", type=int, nargs="?")

    args = p.parse_args()

    # for item in vars(args):
    #    print(item, ':', vars(args)[item])

    exploc = args.exp_loc / "experiment.json"
    experiment = starfish.core.experiment.experiment.Experiment.from_json(str(exploc))

    blobRunnerKwargs = {}
    # addKwarg(args, blobRunnerKwargs, "min_sigma")
    # addKwarg(args, blobRunnerKwargs, "max_sigma")
    if args.min_sigma:
        blobRunnerKwargs["min_sigma"] = tuple(args.min_sigma)
    if args.max_sigma:
        blobRunnerKwargs["max_sigma"] = tuple(args.max_sigma)
    addKwarg(args, blobRunnerKwargs, "num_sigma")
    addKwarg(args, blobRunnerKwargs, "threshold")
    addKwarg(args, blobRunnerKwargs, "overlap")
    addKwarg(args, blobRunnerKwargs, "is_volume")
    addKwarg(args, blobRunnerKwargs, "detector_method")

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

        # checking dims on sigma, because scipy throws an unhelpful error
        # in the event of a mismatch.
        if args.min_sigma:
            minlen = len(tuple(args.min_sigma))
        else:
            minlen = 2

        if args.max_sigma:
            maxlen = len(tuple(args.max_sigma))
        else:
            maxlen = 2

        if args.is_volume:
            vol = args.is_volume
        else:
            vol = False

        if not (vol + 2 == minlen and vol + 2 == maxlen):
            raise Exception(
                f"is_volume is set to {vol}, but sigma dimensions are of length {minlen} and {maxlen}"
            )

        if method == "PerRoundMaxChannel":
            method = starfish.spots.DecodeSpots.PerRoundMaxChannel
        elif method == "MetricDistance":
            method = starfish.spots.DecodeSpots.MetricDistance
        elif method == "SimpleLookupDecoder":
            method = starfish.spots.DecodeSpots.SimpleLookupDecoder
        elif method == "CheckAll":
            method = starfish.spots.DecodeSpots.CheckAll
        else:
            raise Exception("DecodeSpots method " + str(method) + " is not a valid method.")

        trace_strat = args.trace_building_strategy
        if (
            method == starfish.spots.DecodeSpots.PerRoundMaxChannel
            or method == starfish.spots.DecodeSpots.MetricDistance
        ):
            if trace_strat == "SEQUENTIAL":
                trace_strat = TraceBuildingStrategies.SEQUENTIAL
            elif trace_strat == "EXACT_MATCH":
                trace_strat = TraceBuildingStrategies.EXACT_MATCH
            elif trace_strat == "NEAREST_NEIGHBOR":
                trace_strat = TraceBuildingStrategies.NEAREST_NEIGHBOR
            else:
                raise Exception("TraceBuildingStrategies " + str(trace_strat) + " is not valid.")
            decodeKwargs["trace_building_strategy"] = trace_strat

    addKwarg(args, decodeKwargs, "filtered_results")
    addKwarg(args, decodeKwargs, "error_rounds")
    addKwarg(args, decodeKwargs, "mode")
    addKwarg(args, decodeKwargs, "physical_coords")
    addKwarg(args, decodeKwargs, "max_distance")
    addKwarg(args, decodeKwargs, "min_intensity")
    addKwarg(args, decodeKwargs, "metric")
    addKwarg(args, decodeKwargs, "norm_order")
    addKwarg(args, decodeKwargs, "anchor_round")
    addKwarg(args, decodeKwargs, "search_radius")

    decodeRunnerKwargs = {"decoderKwargs": decodeKwargs, "callableDecoder": method}
    addKwarg(args, decodeRunnerKwargs, "return_original_intensities")

    use_ref = args.use_ref_img

    run(
        output_dir,
        experiment,
        blob_based,
        use_ref,
        blobRunnerKwargs,
        decodeRunnerKwargs,
        pixelRunnerKwargs,
    )
