#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from functools import partialmethod
from os import cpu_count, makedirs, path
from pathlib import Path
from typing import Callable, Mapping, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import starfish
import starfish.data
import xarray as xr
from starfish import (
    Codebook,
    DecodedIntensityTable,
    Experiment,
    ImageStack,
    IntensityTable,
)
from starfish.core.types import Number, SpotAttributes, SpotFindingResults
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
    min_sigma: Union[Number, Tuple[Number, ...]] = 0.5,
    max_sigma: Union[Number, Tuple[Number, ...]] = 8,
    num_sigma: int = 5,
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
    filtered_results: bool = False,
    n_processes: int = None,
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
    if n_processes:
        results = decoder.run(spots=spots, n_processes=n_processes)
    else:
        results = decoder.run(spots=spots)
    if filtered_results:
        results = results.loc[results[Features.PASSES_THRESHOLDS]]
        results = results[results.target != "nan"]
    return results


def blobDriver(
    img: ImageStack,
    ref_img: ImageStack,
    codebook: Codebook,
    blobRunnerKwargs: dict,
    decodeRunnerKwargs: dict,
    output_name: str,
) -> Tuple[SpotFindingResults, DecodedIntensityTable]:
    """
    Method to handle the blob-based version of the detection and decoding steps.

    Parameters
    ----------
    img : ImageStack
        The image to be processed.
    ref_img : ImageStack
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
    blob = blobRunner(img, ref_img=ref_img if ref_img else None, **blobRunnerKwargs)
    print("found total spots {}".format(blob.count_total_spots()))
    # if ref_img:
    #    # Starfish doesn't apply threshold correctly when a ref image is used
    #    # so go through results and manually apply it.
    #    if blobRunnerKwargs["threshold"]:
    #        thresh = blobRunnerKwargs["threshold"]
    #    else:
    #        thresh = 0.1
    #    for k, v in blob.items():
    #        data = v.spot_attrs.data
    #        high = data[data["intensity"] > thresh]
    #        v.spot_attrs = SpotAttributes(high)
    #    print(f"removed spots below threshold, now {blob.count_total_spots()} total spots")
    # blobs[fov] = blob
    if output_dir:
        blob.save(output_name)
        print("spots saved.")
    decoded = decodeRunner(blob, codebook, **decodeRunnerKwargs)
    return blob, decoded


def init_scale(img: ImageStack):
    # Initialize scaling factors for each image based on the relative positions of the 90th percentile
    # of their intensity histograms.

    # Build pixel histograms for each image
    pixel_histos = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            data = deepcopy(img.xarray.data[r, ch])
            data = np.rint(data * (2**16))
            data[data == 2**16] = (2**16) - 1
            hist = np.histogram(data, bins=range((2**16)))
            pixel_histos[(r, ch)] = hist[0]

    # Estimate scaling factors using cumulative distribution of each images intensities
    local_scale = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            cumsum = np.cumsum(pixel_histos[(r, ch)])
            cumsum = cumsum / cumsum[-1]
            diffs = np.abs(cumsum - 0.9)
            local_scale[(r, ch)] = np.where(diffs == np.min(diffs))[0][0] + 1

    # Normalize
    scale_mean = np.mean([x for x in local_scale.values()])
    for key in local_scale:
        local_scale[key] /= scale_mean

    return local_scale


def optimize_scale(
    img: ImageStack,
    scaling_factors,  # TODO typecast?
    codebook: Codebook,
    pixelRunnerKwargs: dict,
    is_volume: bool = False,
):
    """
    Parameters:
        img: Processed image.
        scaling_factors: Current intensity scaling factors
        codebook: Codebook object
        is_volume: Boolean whether to scale the image as a 3D volume or as individual 2D tiles.
        pixelRunnerKwargs: the parameters for running the decoder.
    """

    # Apply scaling factors
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            img.xarray.data[r, ch] = img.xarray.data[r, ch] / scaling_factors[(r, ch)]

    # Scale image
    pmin = 0
    pmax = 100
    clip = starfish.image.Filter.ClipPercentileToZero(
        p_min=pmin, p_max=pmax, is_volume=is_volume, level_method=Levels.SCALE_BY_IMAGE
    )
    clip.run(img, in_place=True)

    # Decode image
    decoded_targets, prop_results = pixelDriver(img, codebook, **pixelRunnerKwargs)
    decoded_targets = decoded_targets.loc[decoded_targets[Features.PASSES_THRESHOLDS]]

    # Calculate on-bit scaling factors for each bit
    local_pixel_vector = []
    for barcode in range(1, len(codebook) + 1):
        idx = np.where(prop_results.decoded_image == barcode)
        coords = [(idx[0][i], idx[1][i], idx[2][i]) for i in range(len(idx[0]))]
        if len(coords) > 0:
            local_traces = np.array([img.xarray.data[:, :, c[0], c[1], c[2]] for c in coords])
            local_mean = np.mean(local_traces, axis=0)
            local_pixel_vector.append(local_mean / np.linalg.norm(local_mean))
            local_pixel_vector[-1] /= len(coords)
    pixel_traces = np.array(local_pixel_vector)

    # Normalize pixel traces by l2 norm
    norms = np.linalg.norm(np.asarray(pixel_traces), axis=(1, 2))
    for i in range(len(norms)):
        pixel_traces[i] = pixel_traces[i] / norms[i]

    # Calculate average pixel trace for each barcode and normalize
    one_bit_int = np.mean(pixel_traces, axis=0)
    one_bit_int = one_bit_int / np.mean(one_bit_int)

    # Convert into dictionary with same keys as scaling_factors
    scaling_mods = {}
    for i in range(one_bit_int.shape[0]):
        for j in range(one_bit_int.shape[1]):
            scaling_mods[(i, j)] = one_bit_int[i, j]
    return scaling_mods


def scale_img(
    img, codebook, pixelRunnerKwargs: dict, level_method: Levels, is_volume: bool = False
):
    """
    Main method for image rescaling. Takes a set of images and rescales them to get the best
    pixel-based estimate.  Returns an ImageStack.
    """

    # Initialize scaling factors
    local_scale = init_scale(img)

    # Optimize scaling factors until convergence
    scaling_factors = deepcopy(local_scale)
    og_img = deepcopy(img)
    mod_mean = 1
    iters = 0
    while mod_mean > 0.01:

        scaling_mods = optimize_scale(img, scaling_factors, codebook, pixelRunnerKwargs, is_volume)

        # Apply modifications to scaling_factors
        for key in sorted(scaling_factors):
            scaling_factors[key] = scaling_factors[key] * scaling_mods[key]

        # Replace image with unscaled version
        img = deepcopy(og_img)

        # Update mod_mean and add to iteration number. If iters reaches 20 return current scaling factors
        # and print message
        mod_mean = np.mean(abs(np.array([x for x in scaling_mods.values()]) - 1))
        iters += 1
        if iters >= 20:
            print(
                "Scaling factors did not converge after 20 iterations. Returning initial estimate."
            )
            scaling_factors = deepcopy(local_scale)
            break

    # Scale with final factors
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            img.xarray.data[r, ch] = img.xarray.data[r, ch] / scaling_factors[(r, ch)]

    # Scale image
    pmin = 0
    pmax = 100
    clip = starfish.image.Filter.ClipPercentileToZero(
        p_min=pmin, p_max=pmax, is_volume=is_volume, level_method=level_method
    )
    clip.run(img, in_place=True)

    return img


def pixelDriver(
    img: ImageStack,
    codebook: Codebook,
    distance_threshold: float,
    magnitude_threshold: float,
    metric: str = "euclidean",
    min_area: int = 2,
    max_area: int = np.inf,
    norm_order: int = 2,
) -> DecodedIntensityTable:
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
    pixelRunner = starfish.spots.DetectPixels.PixelSpotDecoder(
        codebook=codebook,
        distance_threshold=distance_threshold,
        magnitude_threshold=magnitude_threshold,
        metric=metric,
        min_area=min_area,
        max_area=max_area,
        norm_order=norm_order,
    )
    return pixelRunner.run(img)


def saveTable(table: DecodedIntensityTable, savename: str):
    """
    Reformats and saves a DecodedIntensityTable.
    """
    if Features.PASSES_THRESHOLDS in table:
        intensities = IntensityTable(table.where(table[Features.PASSES_THRESHOLDS], drop=True))
    else:  # SimpleLookupDecoder will not have PASSES_THRESHOLDS
        intensities = IntensityTable(table)
    traces = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    # traces = table.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    traces = traces.to_features_dataframe()
    traces.to_csv(savename)


def run(
    output_dir: str,
    experiment: Experiment,
    blob_based: bool,
    use_ref: bool,
    anchor_name: str,
    rescale: bool,
    level_method: Levels,
    is_volume: bool,
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
    anchor_name: str
        If provided, that aux view will be flattened and used as the reference image.
    rescale: bool
        If true, the image will be rescaled until convergence before running the decoder.
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
        "output_dir: {}\nexp: {}\nblob_based: {}\nuse_ref: {}\nanchor: {}\nrescale: {}\nblobrunner: {}\ndecoderunner: {}\npixelrunner: {}\n".format(
            output_dir,
            experiment,
            blob_based,
            use_ref,
            anchor_name,
            rescale,
            blobRunnerKwargs,
            decodeRunnerKwargs,
            pixelRunnerKwargs,
        )
    )

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    for fov in experiment.keys():
        # we need to do this per fov to save memory
        img = experiment[fov].get_image("primary")

        ref_img = None
        if use_ref:
            ref_img = img.reduce({Axes.CH, Axes.ROUND, Axes.ZPLANE}, func="max")
        if anchor_name:
            ref_img = (
                experiment[fov]
                .get_image(anchor_name)
                .reduce({Axes.CH, Axes.ROUND, Axes.ZPLANE}, func="max")
            )
            clip = starfish.image.Filter.ClipPercentileToZero(
                p_min=20, p_max=99.9, is_volume=is_volume, level_method=Levels.SCALE_BY_CHUNK
            )
            clip.run(ref_img, in_place=True)

        if rescale:
            img = scale_img(img, experiment.codebook, pixelRunnerKwargs, level_method, is_volume)

        if blob_based:
            output_name = f"{output_dir}spots/{fov}_"
            blobs, decoded = blobDriver(
                img,
                ref_img,
                experiment.codebook,
                blobRunnerKwargs,
                decodeRunnerKwargs,
                output_name,
            )
            del blobs  # this is saved within the driver now
        else:
            decoded = pixelDriver(img, experiment.codebook, **pixelRunnerKwargs)[0]

        saveTable(decoded, output_dir + "csv/" + fov + "_decoded.csv")
        # decoded[fov].to_decoded_dataframe().save_csv(output_dir+fov+"_decoded.csv")
        decoded.to_netcdf(output_dir + "cdf/" + fov + "_decoded.cdf")

        # can run into memory problems, doing this preemptively.
        del img
        del ref_img
        del decoded

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
    p.add_argument("--is-volume", dest="is_volume", action="store_true")
    p.add_argument("--rescale", dest="rescale", action="store_true")
    p.add_argument("--level-method", type=str, nargs="?")
    p.add_argument("--anchor-view", type=str, nargs="?")

    # blobRunner kwargs
    p.add_argument("--min-sigma", type=float, nargs="*")
    p.add_argument("--max-sigma", type=float, nargs="*")
    p.add_argument("--num-sigma", type=int, nargs="?")
    p.add_argument("--threshold", type=float, nargs="?")
    p.add_argument("--overlap", type=float, nargs="?")
    p.add_argument("--detector-method", type=str, nargs="?")
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
        "--filtered-results", dest="filtered_results", action="store_true"
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
    blob_based = args.decode_spots_method is not None
    vol = False
    if blob_based:

        # checking dims on sigma, because scipy throws an unhelpful error
        # in the event of a mismatch.
        if args.min_sigma:
            minlen = len(tuple(args.min_sigma))

        if args.max_sigma:
            maxlen = len(tuple(args.max_sigma))

        if args.is_volume:
            vol = args.is_volume

        if not (args.min_sigma and vol + 2 == minlen) and (args.max_sigma and vol + 2 == maxlen):
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

    addKwarg(args, decodeKwargs, "error_rounds")
    addKwarg(args, decodeKwargs, "mode")
    addKwarg(args, decodeKwargs, "physical_coords")
    addKwarg(args, decodeKwargs, "max_distance")
    addKwarg(args, decodeKwargs, "min_intensity")
    if method == starfish.spots.DecodeSpots.MetricDistance:
        addKwarg(args, decodeKwargs, "metric")
        addKwarg(args, decodeKwargs, "norm_order")
        # notably including this when rescale is used with other decoders
        # leads to bugs since these two params aren't' an accepted decoder arg
    addKwarg(args, decodeKwargs, "anchor_round")
    addKwarg(args, decodeKwargs, "search_radius")

    decodeRunnerKwargs = {
        "decoderKwargs": decodeKwargs,
        "callableDecoder": method,
        "filtered_results": args.filtered_results,
    }
    if method == starfish.spots.DecodeSpots.CheckAll:
        decodeRunnerKwargs["n_processes"] = cpu_count()
    addKwarg(args, decodeRunnerKwargs, "return_original_intensities")

    use_ref = args.use_ref_img
    anchor_name = args.anchor_view
    rescale = args.rescale

    level_method = args.level_method
    if level_method == "SCALE_BY_CHUNK":
        level_method = Levels.SCALE_BY_CHUNK
    elif level_method == "SCALE_BY_IMAGE":
        level_method = Levels.SCALE_BY_IMAGE
    elif level_method == "SCALE_SATURATED_BY_CHUNK":
        level_method = Levels.SCALE_SATURATED_BY_CHUNK
    elif level_method == "SCALE_SATURATED_BY_IMAGE":
        level_method = Levels.SCALE_SATURATED_BY_IMAGE
    else:
        level_method = Levels.SCALE_BY_CHUNK

    run(
        output_dir,
        experiment,
        blob_based,
        use_ref,
        anchor_name,
        rescale,
        level_method,
        vol,
        blobRunnerKwargs,
        decodeRunnerKwargs,
        pixelRunnerKwargs,
    )
