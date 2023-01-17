#!/usr/bin/env python

import operator as op
import os
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from functools import partialmethod, reduce
from os import makedirs, path
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import starfish
import starfish.data
from scipy.spatial import distance
from starfish import (
    Codebook,
    DecodedIntensityTable,
    Experiment,
    ImageStack,
    IntensityTable,
)
from starfish.core.types import Number, SpotFindingResults
from starfish.types import Axes, Features, Levels, TraceBuildingStrategies
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
    print(locals())
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

    Returns
    -------
    DecodedIntensityTable:
        Starfish wrapper for an xarray with the labeled transcripts.
    """
    print(locals())
    decoder = callableDecoder(codebook=codebook, **decoderKwargs)
    if n_processes:
        results = decoder.run(spots=spots, n_processes=n_processes)
    else:
        results = decoder.run(spots=spots)

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
    print(locals())
    start = time.time()
    blob = blobRunner(img, ref_img=ref_img if ref_img else None, **blobRunnerKwargs)
    print("blobRunner", time.time() - start)
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
    if blob.count_total_spots() > 0:
        if output_dir:
            blob.save(output_name)
            print("spots saved.")
        start = time.time()
        decoded = decodeRunner(blob, codebook, **decodeRunnerKwargs)
        print("decodeRunner", time.time() - start)
        return blob, decoded
    else:
        print("Skipping decoding step.")
        return blob, DecodedIntensityTable()


def init_scale(img: ImageStack):
    """
    Initialize scaling factors for each image based on the relative positions
    of the 90th percentile of their intensity histograms.

    Parameters
    ----------
    img : ImageStack
        The target image for the initial scaling

    Returns
    -------
    Matrix of scaling factors (TODO: more specific + typecasting?)
    """

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

    # Crop edges
    crop = 40
    cropped_img = deepcopy(
        img.sel(
            {Axes.Y: (crop, img.shape[Axes.Y] - crop), Axes.X: (crop, img.shape[Axes.X] - crop)}
        )
    )

    # Initialize scaling factors
    local_scale = init_scale(cropped_img)

    # Optimize scaling factors until convergence
    scaling_factors = deepcopy(local_scale)
    og_img = deepcopy(cropped_img)
    mod_mean = 1
    iters = 0
    while mod_mean > 0.01:

        scaling_mods = optimize_scale(
            cropped_img, scaling_factors, codebook, pixelRunnerKwargs, is_volume
        )

        # Apply modifications to scaling_factors
        for key in sorted(scaling_factors):
            scaling_factors[key] = scaling_factors[key] * scaling_mods[key]

        # Replace image with unscaled version
        cropped_img = deepcopy(og_img)

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
            print(f"({r},{ch}): {scaling_factors[(r,ch)]}")
            img.xarray.data[r, ch] = img.xarray.data[r, ch] / scaling_factors[(r, ch)]

    # Scale image
    pmin = 0
    pmax = 100
    clip = starfish.image.Filter.ClipPercentileToZero(
        p_min=pmin, p_max=pmax, is_volume=is_volume, level_method=Levels.SCALE_BY_IMAGE
    )
    clip.run(img, in_place=True)

    print(f"Rescaled image in {iters} iterations, final mod_mean: {mod_mean}")

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
    intensities = IntensityTable(table)
    traces = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    # traces = table.stack(traces=(Axes.ROUND.value, Axes.CH.value))
    traces = traces.to_features_dataframe()
    traces.to_csv(savename)
    print(f"Saved decoded csv file {savename}.")


def nck(n, k):
    """
    n choose k function
    """
    k = min(k, n - k)
    numer = reduce(op.mul, range(n, n - k, -1), 1)
    denom = reduce(op.mul, range(1, k + 1), 1)
    return numer // denom


def add_corrected_rounds(codebook, decoded, ham_dist):
    """
    For MERFISH experiments, adds "corrected_rounds" field to DecodedIntensityTable which denotes the number of rounds
    that were corrected for in the experiment
    """

    # Make two dictionaries, both have target names as keys, neighbor_codes contains all neighboring codes to that
    # targets code within the hamming distance while perfect_words contains only the correct codeword for that target
    neighbor_codes = {}
    perfect_words = {}
    for code in codebook:
        target = str(code["target"].data)
        codeword = code.data
        codeword = codeword.flatten()
        C = [nck(x, ham_dist) for x in range(len(codeword))]
        codewords = np.tile(codeword, (len(C), 1))
        for i in range(len(C)):
            codewords[i, C[i]] = int(~np.array(codewords[i, C[i]], dtype=bool))
        neighbor_codes[target] = np.array([word / np.linalg.norm(word) for word in codewords])
        perfect_words[target] = codeword / np.linalg.norm(codeword)

    # For every transcript, check whether it's pixel vector is closer in distance to the true code or one of the
    # neighbor codes. If it is closer to a neighbor code, there was an error correction.
    corrected_rounds = []
    for i, transcript in enumerate(decoded):
        on_distance = np.linalg.norm(
            transcript.data.flatten() - perfect_words[str(transcript["target"].data)]
        )
        off_distances = distance.cdist(
            transcript.data.flatten().reshape((1, -1)),
            neighbor_codes[str(transcript["target"].data)],
        )
        if on_distance < np.min(off_distances):
            corrected_rounds.append(0)
        else:
            corrected_rounds.append(1)

    # Add corrected_rounds field to table and return
    return decoded.assign_coords(corrected_rounds=("features", corrected_rounds))


def run(
    output_dir: str,
    experiment: Experiment,
    blob_based: bool,
    use_ref: bool,
    anchor_name: str,
    rescale: bool,
    level_method: Levels,
    is_volume: bool,
    not_filtered_results: bool,
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
    not_filtered_results: bool
        If true, rows with no target or that do not pass thresholds will not be removed.
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

    print(locals())

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    for fov in experiment.keys():
        start0 = time.time()
        print("fov", fov)

        # we need to do this per fov to save memory
        start = time.time()
        img = experiment[fov].get_image("primary")
        print("Load Image", time.time() - start)

        ref_img = None
        if use_ref:
            ref_img = img.reduce({Axes.CH, Axes.ROUND}, func="max")
        if anchor_name:
            ref_img = (
                experiment[fov].get_image(anchor_name).reduce({Axes.CH, Axes.ROUND}, func="max")
            )

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
            print(f"Found {len(decoded)} transcripts with blobDriver")
        else:
            decoded = pixelDriver(img, experiment.codebook, **pixelRunnerKwargs)[0]
            print(f"Found {len(decoded)} transcripts with pixelDriver")

            # If applicable add "corrected_rounds" field

            # Check that codebook is not one-hot
            for row in experiment.codebook[0].data:
                row_sum = sum(row == 0)
                if row_sum != len(experiment.codebook["c"]) or row_sum != 0:
                    ham_dist = 1
                    decoded = add_corrected_rounds(experiment.codebook, decoded, ham_dist)

        # SimpleLookupDecoder will not have PASSES_THRESHOLDS
        if Features.PASSES_THRESHOLDS in decoded.coords and not not_filtered_results:
            decoded = decoded.loc[decoded[Features.PASSES_THRESHOLDS]]
            decoded = decoded[decoded.target != "nan"]

        if len(decoded) > 0:
            saveTable(decoded, output_dir + "csv/" + fov + "_decoded.csv")
            # decoded[fov].to_decoded_dataframe().save_csv(output_dir+fov+"_decoded.csv")
            decoded.to_netcdf(output_dir + "cdf/" + fov + "_decoded.cdf")
            print(f"Saved cdf file {output_dir}cdf/{fov}_decoded.cdf")
        else:
            print(f"No transcripts found for {fov}! Not saving a DecodedIntensityTable file.")

        # can run into memory problems, doing this preemptively.
        del img
        del ref_img
        del decoded
        print("Total driver time", time.time() - start0)
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
    p.add_argument("--not-filtered-results", dest="not_filtered_results", action="store_true")

    # blobRunner kwargs
    p.add_argument("--min-sigma", type=float, nargs="*")
    p.add_argument("--max-sigma", type=float, nargs="*")
    p.add_argument("--num-sigma", type=int, nargs="?")
    p.add_argument("--threshold", type=float, nargs="?")
    p.add_argument("--overlap", type=float, nargs="?")
    p.add_argument("--detector-method", type=str, nargs="?")
    p.add_argument("--use-ref-img", dest="use_ref_img", action="store_true")
    p.set_defaults(use_ref_img=False)
    # == aside, are we going to want to include the ability to run a sweep?

    # decodeRunner kwargs
    p.add_argument("--decode-spots-method", type=str)
    p.add_argument(
        "--trace-building-strategy", type=str, nargs="?"
    )  # only optional for SimpleLookupDecoder

    # == MetricDistance
    p.add_argument("--max-distance", type=float, nargs="?")
    p.add_argument("--min-intensity", type=float, nargs="?")
    p.add_argument("--metric", type=str, nargs="?")  # NOTE also used in pixelRunner
    p.add_argument("--norm-order", type=int, nargs="?")  # NOTE also used in pixelRunner
    p.add_argument("--anchor-round", type=int, nargs="?")  # also used in PerRoundMaxChannel
    p.add_argument(
        "--search-radius", type=float, nargs="?"
    )  # also used in PerRoundMaxChannel, CheckAll
    p.add_argument("--return-original-intensities", type=bool, nargs="?")

    # == CheckAll
    p.add_argument("--error-rounds", type=int, nargs="?")
    p.add_argument("--mode", type=str, nargs="?")
    p.add_argument("--physical-coords", dest="physical_coords", action="store_true")
    p.add_argument("--n-processes", type=int, nargs="?")

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
        if len(args.min_sigma) == 1:
            blobRunnerKwargs["min_sigma"] = args.min_sigma[0]
        else:
            blobRunnerKwargs["min_sigma"] = tuple(args.min_sigma)
    if args.max_sigma:
        if len(args.max_sigma) == 1:
            blobRunnerKwargs["max_sigma"] = args.min_sigma[0]
        else:
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
    }
    if method == starfish.spots.DecodeSpots.CheckAll:
        if args.n_processes:
            addKwarg(args, decodeRunnerKwargs, "n_processes")
        else:
            try:
                # the following line is not guaranteed to work on non-linux systems
                decodeRunnerKwargs["n_processes"] = len(os.sched_getaffinity(os.getpid()))
            except Exception:
                decodeRunnerKwargs["n_processes"] = 1
    addKwarg(args, decodeRunnerKwargs, "return_original_intensities")

    use_ref = args.use_ref_img
    anchor_name = args.anchor_view
    rescale = args.rescale
    not_filtered_results = args.not_filtered_results

    level_method = args.level_method
    if level_method and level_method == "SCALE_BY_CHUNK":
        level_method = Levels.SCALE_BY_CHUNK
    elif level_method and level_method == "SCALE_BY_IMAGE":
        level_method = Levels.SCALE_BY_IMAGE
    elif level_method and level_method == "SCALE_SATURATED_BY_CHUNK":
        level_method = Levels.SCALE_SATURATED_BY_CHUNK
    elif level_method and level_method == "SCALE_SATURATED_BY_IMAGE":
        level_method = Levels.SCALE_SATURATED_BY_IMAGE
    else:
        level_method = Levels.SCALE_BY_IMAGE

    run(
        output_dir,
        experiment,
        blob_based,
        use_ref,
        anchor_name,
        rescale,
        level_method,
        vol,
        not_filtered_results,
        blobRunnerKwargs,
        decodeRunnerKwargs,
        pixelRunnerKwargs,
    )
