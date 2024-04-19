#!/usr/bin/env python

import glob
import json
import operator as op
import os
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partialmethod, reduce
from os import makedirs, path
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import starfish
import starfish.data
import xarray as xr
from scipy.spatial import distance
from starfish import (
    Codebook,
    DecodedIntensityTable,
    Experiment,
    ImageStack,
    IntensityTable,
)
from starfish.core.types import (
    Number,
    PerImageSliceSpotResults,
    SpotAttributes,
    SpotFindingResults,
)
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
            hist = np.histogram(img.xarray.data[r, ch], bins=range((2**16)))
            pixel_histos[(r, ch)] = hist[0]

    # Estimate scaling factors using cumulative distribution of each images intensities
    local_scale = {}
    for r in range(img.num_rounds):
        for ch in range(img.num_chs):
            cumsum = np.cumsum(pixel_histos[(r, ch)])
            cumsum = cumsum / cumsum[-1]
            diffs = np.abs(cumsum - 0.9)
            local_scale[(r, ch)] = np.where(diffs == np.min(diffs))[0][0] + 1
            if local_scale[(r, ch)] == 1:
                local_scale[(r, ch)] = 0

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

    # Only calculate scaling factors if more than 10 transcripts, else return None
    if len(decoded_targets) >= 10:
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
    else:
        return None


def scale_img(fovs, experiment, pixelRunnerKwargs: dict, is_volume: bool = False):
    """
    Main method for image rescaling. Takes a list of fovs and randomly rescales them
    until 30 converge. Returns the average scaling factors across those 30 FOVs.
    """
    print("Intensity Rescaling...")
    scale_start = time.time()

    n = 30
    codebook = experiment.codebook
    rescale_fovs = random.sample(fovs, len(fovs))

    # Calculate scaling factors for shuffled FOVs until n FOVs converge
    num_converged = 0
    scaling_factors_set = []
    for fov in rescale_fovs:
        print(fov)
        img = experiment[fov].get_image("primary")

        # Crop edges
        crop = 40
        cropped_img = deepcopy(
            img.sel(
                {
                    Axes.Y: (crop, img.shape[Axes.Y] - crop),
                    Axes.X: (crop, img.shape[Axes.X] - crop),
                }
            )
        )

        # Initialize scaling factors
        local_scale = init_scale(cropped_img)
        # Optimize scaling factors until convergence
        scaling_factors = deepcopy(local_scale)
        og_img = deepcopy(cropped_img)
        mod_mean = 1
        mod_means = [5, 1]
        iters = 0
        while mod_mean > 0.01 and mod_means[-2] - mod_means[-1] > 0.01:
            scaling_mods = optimize_scale(
                cropped_img, scaling_factors, codebook, pixelRunnerKwargs, is_volume
            )

            # If scaling_mods == None, less than 10 transcripts were decodable and scaling_factors are all set to 1
            if scaling_mods is not None:
                # Apply modifications to scaling_factors
                for key in sorted(scaling_factors):
                    scaling_factors[key] = scaling_factors[key] * scaling_mods[key]

                # Replace image with unscaled version
                cropped_img = deepcopy(og_img)

                # Update mod_mean and add to iteration number. If iters reaches 20 return current scaling factors
                # and print message
                mod_mean = np.mean(abs(np.array([x for x in scaling_mods.values()]) - 1))
                mod_means.append(mod_mean)
                iters += 1
                if iters >= 20:
                    break

        # Check for convergence, save scaling factors and break out of loop if num_converged reaches n
        if iters < 20 and scaling_mods is not None:
            scaling_factors_set.append(scaling_factors)
            num_converged += 1
            if num_converged == n:
                break

    print(f"Scale factors calculated in: {time.time()-scale_start}")

    # If at least 1 FOV converged, calculate the average scaling factors and return them
    if num_converged >= 1:
        if num_converged < n:
            print(
                f"WARNING: Fewer than {n} FOVs converged during intensity rescaling. {num_converged}/{len(fovs)} did converge. Rescaling may not be accurate. Try adjusting processing parameters"
            )
        scaling_factors_list = defaultdict(list)
        for scaling_factors in scaling_factors_set:
            for rch in scaling_factors:
                scaling_factors_list[rch].append(scaling_factors[rch])

        scaling_factors_avg = {}
        for rch in scaling_factors_list:
            scaling_factors_avg[rch] = np.mean(scaling_factors_list[rch])

        return scaling_factors_avg
    # Returns None if no FOVs converged
    else:
        print(
            "WARNING: ZERO FOVs converged during intensity rescaling. Images will not be rescaled. Try adjusting processing parameters"
        )
        return None


def pixelDriver(
    img: ImageStack,
    codebook: Codebook,
    distance_threshold: float,
    magnitude_threshold: float,
    pnorm: int = 2,
    min_area: int = 2,
    max_area: int = np.inf,
    norm_order: int = 2,
    n_processes: int = 1,
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
        pnorm=pnorm,
        magnitude_threshold=magnitude_threshold,
        min_area=min_area,
        max_area=max_area,
        norm_order=norm_order,
        n_workers=n_processes,
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

    # Flateen rounds and channels dimensions and normalize codebook
    codebook_data = codebook.data.reshape(codebook.shape[0], codebook.shape[1] * codebook.shape[2])
    codebook_data_norm = codebook_data / np.linalg.norm(codebook_data, axis=1)[:, None]

    # Flatten rounds and channels dimensions of decoded vectors (already normalized)
    decoded_data = decoded.data.reshape(decoded.shape[0], decoded.shape[1] * decoded.shape[2])

    # Calculate distance of each decoded vector with the barcode it was assigned
    target_id_map = {target: target_id for target_id, target in enumerate(codebook["target"].data)}
    decoded_ids = [target_id_map[target] for target in decoded["target"].data]
    perfect_words = codebook_data_norm[np.array(decoded_ids)]
    on_distances = np.linalg.norm(decoded_data - perfect_words, axis=1)

    # Create dictionary of hamming distance ham_dist codes from each of the barcodes in the codebook
    neighbor_codes = {}
    for c, codeword in enumerate(codebook_data):
        target = codebook["target"].data[c]
        codeword = codeword.flatten()
        C = [nck(x, ham_dist) for x in range(len(codeword))]
        codewords = np.tile(codeword, (len(C), 1))
        for i in range(len(C)):
            codewords[i, C[i]] = int(~np.array(codewords[i, C[i]], dtype=bool))
        neighbor_codes[target] = np.array([word / np.linalg.norm(word) for word in codewords])

    # Create matrix of shape (rounds*channels, decoded transcript count, rounds*channels) where dimension 0 corresponds
    # to the rounds*channels different possible hamming distance ham_dist barcodes from each codebook word, dimension 1
    # corresponds to each decoded vector, and dimension 3 is the rounds*channels flattened barcoded
    neighbor_codes_mtx = []
    for i in range(codebook_data.shape[1]):
        this_neighbor_codes_mtx = []
        for target in decoded["target"].data:
            this_neighbor_codes_mtx.append(neighbor_codes[target][i])
        neighbor_codes_mtx.append(this_neighbor_codes_mtx)
    neighbor_codes_mtx = np.array(neighbor_codes_mtx)

    # Calculate distance of each decoded vector with the hamming distance ham_dist codes. Loops thround each of the
    # rounds*channels sets of error codes and calculates distances to decoded vectors and then takes the min
    off_distances = []
    for i in range(codebook.shape[1] * codebook.shape[2]):
        off_distances.append(np.linalg.norm(decoded_data - neighbor_codes_mtx[i], axis=1))
    off_distances = np.array(off_distances)
    min_off_distances = np.min(off_distances, axis=0)

    # Find decoded vectors where the difference between the on target barcode distance and minimum off-target distance
    # is positive, indicating that the off-target distance was smaller. Those vectors used error correction and are
    # assigned 1 in the new column
    diff = on_distances - min_off_distances
    diff = diff > 0
    corrected_rounds = diff.astype(int)

    # Add corrected_rounds field to table and return
    return decoded.assign_coords(corrected_rounds=("features", corrected_rounds))


def getCoords(exploc: str, selected_fovs: List[int]):
    """
    Extracts physical coordinates of each FOV from the primary-fov_*.json files. Used in creating composite images.
    """

    # Get json file names
    img_jsons = sorted(glob.glob(f"{str(exploc)[:-15]}/primary-fov_*.json"))

    # Filter by selcted_fovs (if set)
    if selected_fovs is not None:
        fovs = ["fov_{:05}".format(int(f)) for f in selected_fovs]
        img_jsons = [
            img_json
            for img_json in img_jsons
            if img_json.split("/")[-1].split("-")[-1].split(".")[0] in fovs
        ]

    # Get x_min, x_max, y_min, and y_max values for all FOVs and keep track of the absolute x_min and y_min
    composite_coords = defaultdict(dict)
    physical_coords = defaultdict(dict)
    x_min_all = np.inf
    x_max_all = 0
    y_min_all = np.inf
    y_max_all = 0
    for img_json in img_jsons:
        pos = img_json.split("/")[-1].split("_")[-1].split(".")[0]
        with open(img_json, "r") as file:
            metadata = json.load(file)

        # Convert physical coordinates to pixel coordinates to find each FOVs place in the composite image
        xc = metadata["tiles"][0]["coordinates"]["xc"]
        yc = metadata["tiles"][0]["coordinates"]["yc"]
        zc = metadata["tiles"][0]["coordinates"]["zc"]
        tile_shape = metadata["tiles"][0]["tile_shape"]
        x_size = (xc[1] - xc[0] + 1) / tile_shape["x"]
        y_size = (yc[1] - yc[0] + 1) / tile_shape["y"]
        z_size = (zc[1] - zc[0]) / metadata["shape"]["z"]

        # Save physical distance values for later use
        physical_coords[pos]["xc"] = xc
        physical_coords[pos]["yc"] = yc
        physical_coords[pos]["zc"] = zc
        physical_coords["x_size"] = x_size
        physical_coords["y_size"] = y_size
        physical_coords["z_size"] = z_size

        # Save min and max values for x and y for each FOV and track the min/max values across all FOVs
        composite_coords[pos]["x_min"] = int(xc[0] / x_size)
        x_min_all = (
            x_min_all
            if x_min_all < composite_coords[pos]["x_min"]
            else composite_coords[pos]["x_min"]
        )
        composite_coords[pos]["x_max"] = int(xc[1] / x_size)
        x_max_all = (
            x_max_all
            if x_max_all > composite_coords[pos]["x_max"]
            else composite_coords[pos]["x_max"]
        )
        composite_coords[pos]["y_min"] = int(yc[0] / y_size)
        y_min_all = (
            y_min_all
            if y_min_all < composite_coords[pos]["y_min"]
            else composite_coords[pos]["y_min"]
        )
        composite_coords[pos]["y_max"] = int(yc[1] / y_size)
        y_max_all = (
            y_max_all
            if y_max_all > composite_coords[pos]["y_max"]
            else composite_coords[pos]["y_max"]
        )

    # Subtract minimum coord values from xs and ys (ensures (0,0) is the top left corner)
    for img_json in img_jsons:
        pos = img_json.split("/")[-1].split("_")[-1].split(".")[0]
        composite_coords[pos]["x_min"] = composite_coords[pos]["x_min"] - x_min_all
        composite_coords[pos]["x_max"] = composite_coords[pos]["x_max"] - x_min_all
        composite_coords[pos]["y_min"] = composite_coords[pos]["y_min"] - y_min_all
        composite_coords[pos]["y_max"] = composite_coords[pos]["y_max"] - y_min_all

    y_max_all -= y_min_all
    x_max_all -= x_min_all
    physical_coords["y_offset"] = y_min_all
    physical_coords["x_offset"] = x_min_all

    return composite_coords, physical_coords, y_max_all, x_max_all, metadata["shape"]


def createComposite(
    experiment: Experiment,
    exploc: str,
    anchor_name: str,
    is_volume: bool,
    level_method: Levels,
    selected_fovs: List[int],
    composite_pmin: float = 0.0,
    composite_pmax: float = 100.0,
):
    """
    Creates a composite image by taking all FOVs and placing that according to their fov_positioning input. Used
    as an option for postcode decoding.
    """

    # Get physical coordinates
    composite_coords, physical_coords, y_max_all, x_max_all, shape = getCoords(
        exploc, selected_fovs
    )

    # Create empty combined images
    combined_img = np.zeros(
        (shape["r"], shape["c"], shape["z"], int(y_max_all) + 1, int(x_max_all) + 1),
        dtype="float32",
    )
    combined_anchor = np.zeros(
        (shape["r"], 1, shape["z"], int(y_max_all) + 1, int(x_max_all) + 1), dtype="float32"
    )

    # Get json file names
    img_jsons = sorted(glob.glob(f"{str(exploc)[:-15]}/primary-fov_*.json"))

    # Filter by selcted_fovs (if set)
    if selected_fovs is not None:
        fovs = ["fov_{:05}".format(int(f)) for f in selected_fovs]
        img_jsons = [
            img_json
            for img_json in img_jsons
            if img_json.split("/")[-1].split("-")[-1].split(".")[0] in fovs
        ]

    # Fill in image
    for img_json in img_jsons:
        pos = img_json.split("/")[-1].split("_")[-1].split(".")[0]

        fov = "fov_" + "0" * (5 - len(str(pos))) + str(pos)
        img = experiment[fov].get_image("primary")

        x_min = composite_coords[pos]["x_min"]
        x_max = composite_coords[pos]["x_max"]
        y_min = composite_coords[pos]["y_min"]
        y_max = composite_coords[pos]["y_max"]
        combined_img[:, :, :, y_min : y_max + 1, x_min : x_max + 1] = deepcopy(img.xarray.data)

        if anchor_name:
            anchor = experiment[fov].get_image(anchor_name)
            combined_anchor[:, :, :, y_min : y_max + 1, x_min : x_max + 1] = deepcopy(
                anchor.xarray.data
            )

    # Turn into ImageStacks and delete original arrays to save memory
    # If no anchor image was provided create one by take the max projection along the channel axis
    combined_starfish_img = ImageStack.from_numpy(combined_img)
    del combined_img
    if anchor_name:
        combined_starfish_anchor = ImageStack.from_numpy(combined_anchor)
        del combined_anchor
    else:
        combined_starfish_anchor = combined_starfish_img.reduce({Axes.CH}, func="max")

    # Scale composite images
    clip = starfish.image.Filter.ClipPercentileToZero(
        p_min=composite_pmin,
        p_max=composite_pmax,
        is_volume=is_volume,
        level_method=level_method,
    )
    clip.run(combined_starfish_img, in_place=True)

    # Anchor has fixed values
    clip = starfish.image.Filter.ClipPercentileToZero(
        p_min=90, p_max=99.9, is_volume=is_volume, level_method=level_method
    )
    clip.run(combined_starfish_anchor, in_place=True)

    return combined_starfish_img, combined_starfish_anchor


def saveCompositeResults(spots, decoded, exploc, selected_fovs, output_name):
    # Splits large spots object into lots of smaller ones
    spot_items = dict(spots.items())
    for rch in spot_items:
        spot_items[rch] = spot_items[rch].spot_attrs.data

    composite_coords, physical_coords, y_max_all, x_max_all, shape = getCoords(
        exploc, selected_fovs
    )

    # Get json file names
    img_jsons = sorted(glob.glob(f"{str(exploc)[:-15]}/primary-fov_*.json"))

    # Filter by selcted_fovs (if set)
    if selected_fovs is not None:
        fovs = ["fov_{:05}".format(int(f)) for f in selected_fovs]
        img_jsons = [
            img_json
            for img_json in img_jsons
            if img_json.split("/")[-1].split("-")[-1].split(".")[0] in fovs
        ]

    # Create a new SpotFindingResults object with only spots from each position and save separately
    for img_json in img_jsons:
        pos = img_json.split("/")[-1].split("_")[-1].split(".")[0]
        fov = "fov_{:0>5}".format(pos)
        spot_attrs = {}
        x_min = composite_coords[pos]["x_min"]
        x_max = composite_coords[pos]["x_max"]
        y_min = composite_coords[pos]["y_min"]
        y_max = composite_coords[pos]["y_max"]
        for rch in spot_items:
            if y_min == 0:
                spot_attrs[rch] = spot_items[rch][
                    (spot_items[rch]["y"] >= y_min)
                    & (spot_items[rch]["y"] <= y_max)
                    & (spot_items[rch]["x"] > x_min)
                    & (spot_items[rch]["x"] <= x_max)
                ]
            elif x_min == 0:
                spot_attrs[rch] = spot_items[rch][
                    (spot_items[rch]["y"] > y_min)
                    & (spot_items[rch]["y"] <= y_max)
                    & (spot_items[rch]["x"] >= x_min)
                    & (spot_items[rch]["x"] <= x_max)
                ]
            else:
                spot_attrs[rch] = spot_items[rch][
                    (spot_items[rch]["y"] > y_min)
                    & (spot_items[rch]["y"] <= y_max)
                    & (spot_items[rch]["x"] > x_min)
                    & (spot_items[rch]["x"] <= x_max)
                ]
            spot_attrs[rch] = spot_attrs[rch].reset_index(drop=True)
            spot_attrs[rch]["y"] = spot_attrs[rch]["y"] - y_min
            spot_attrs[rch]["y_min"] = spot_attrs[rch]["y_min"] - y_min
            spot_attrs[rch]["y_max"] = spot_attrs[rch]["y_max"] - y_min
            spot_attrs[rch]["x"] = spot_attrs[rch]["x"] - x_min
            spot_attrs[rch]["x_min"] = spot_attrs[rch]["x_min"] - x_min
            spot_attrs[rch]["x_max"] = spot_attrs[rch]["x_max"] - x_min

        spot_attrs_list = []
        for ch in range(shape["c"]):
            for r in range(shape["r"]):
                spot_results = PerImageSliceSpotResults(
                    spot_attrs=SpotAttributes(spot_attrs[(r, ch)]), extras=None
                )
                spot_attrs_list.append((spot_results, {Axes.ROUND: r, Axes.CH: ch}))

        # Correctly adds correct physical coordinate info for each FOV
        coords = {}
        xc = physical_coords[pos]["xc"]
        yc = physical_coords[pos]["yc"]
        zc = physical_coords[pos]["zc"]
        x_size = physical_coords["x_size"]
        y_size = physical_coords["y_size"]
        z_size = physical_coords["z_size"]
        x_offset = physical_coords["x_offset"]
        y_offset = physical_coords["y_offset"]
        xrange = np.arange(xc[0] + x_offset, xc[1] + x_offset, x_size)
        coords["xc"] = xr.DataArray(data=xrange, dims=["x"], coords=dict(xc=(["x"], xrange)))
        yrange = np.arange(yc[0] + y_offset, yc[1] + y_offset, y_size)
        coords["yc"] = xr.DataArray(data=yrange, dims=["y"], coords=dict(yc=(["y"], xrange)))
        zrange = np.arange(zc[0], zc[1], z_size)
        coords["zc"] = xr.DataArray(
            data=zrange,
            dims=["z"],
            coords=dict(z=(["z"], zrange), zc=(["z"], zrange)),
        )

        fov_spots = SpotFindingResults(
            imagestack_coords=coords, log=starfish.Log(), spot_attributes_list=spot_attrs_list
        )

        if fov_spots.count_total_spots() > 0:
            if output_name:
                fov_spots.save(f"{output_name}spots/{fov}_")
    print("spots saved.")

    # Save decoded transcripts
    for pos in composite_coords:
        fov = "fov_" + "0" * (5 - len(str(pos))) + str(pos)

        # Subset transcripts for this FOV based on the FOV's composite coordinates
        x_min = composite_coords[pos]["x_min"]
        x_max = composite_coords[pos]["x_max"]
        y_min = composite_coords[pos]["y_min"]
        y_max = composite_coords[pos]["y_max"]
        decoded_results = decoded[
            (decoded["y"] >= y_min)
            & (decoded["y"] < y_max)
            & (decoded["x"] >= x_min)
            & (decoded["x"] < x_max)
        ]

        # Adjusts x and y values back to thier FOV pixel values
        decoded_results["x"].data -= x_min
        decoded_results["y"].data -= y_min

        # Correctly adds correct physical coordinate info for each FOV
        xc = physical_coords[pos]["xc"]
        yc = physical_coords[pos]["yc"]
        zc = physical_coords[pos]["zc"]
        x_size = physical_coords["x_size"]
        y_size = physical_coords["y_size"]
        z_size = physical_coords["z_size"]
        x_offset = physical_coords["x_offset"]
        y_offset = physical_coords["y_offset"]
        decoded_results["xc"].data = [
            x * x_size + xc[0] + x_offset for x in decoded_results["x"].data
        ]
        decoded_results["yc"].data = [
            y * y_size + yc[0] + y_offset for y in decoded_results["y"].data
        ]
        decoded_results["zc"].data = [z * z_size + zc[0] for z in decoded_results["z"].data]

        if len(decoded_results) > 0:
            saveTable(decoded_results, f"{output_name}csv/{fov}_decoded.csv")
            decoded_results.to_netcdf(f"{output_name}cdf/{fov}_decoded.cdf")
            print(f"Saved cdf file {output_name}cdf/{fov}_decoded.cdf")
        else:
            print(f"No transcripts found for {fov}! Not saving a DecodedIntensityTable file.")


def run(
    output_dir: str,
    input_dir: Path,
    experiment: Experiment,
    blob_based: bool,
    use_ref: bool,
    anchor_name: str,
    rescale: bool,
    level_method: Levels,
    is_volume: bool,
    is_composite: bool,
    not_filtered_results: bool,
    selected_fovs: List[int],
    blobRunnerKwargs: dict,
    decodeRunnerKwargs: dict,
    pixelRunnerKwargs: dict,
    compositeKwargs: dict,
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
    level_method: Levels
        The level method to be used in clip and scale.
    is_volume: bool
        If true, zslices will be treated as 3D image, instead of being processed separately
    is_composite: bool
        If true, all fovs will be composited into one image. Only used for postcode decoding.
    not_filtered_results: bool
        If true, rows with no target or that do not pass thresholds will not be removed.
    selected_fovs: List[int]
        If provided, FOVs with the selected indices will be processed.
    blobRunnerKwargs: dict
        Dictionary with arguments for blob detection. Refer to blobRunner.
    decodeRunnerKwargs: dict
        Dictionary with arguments for spot-based decoding. Refer to decodeRunner.
    pixelRunnerKwargs: dict
        Dictionary with arguments for pixel-based detection and decoding.  Refer to starfish PixelSpotDecoder.
    compositeKwargs: dict
        Dictionary with arguments for creating and scaling composite image option for postcodeDecode. Refer to createComposite.
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
        path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f_starfish_runner.log")), "w"
    )
    sys.stdout = reporter
    sys.stderr = reporter

    print(locals())

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # If decoder is postcodeDecoder and composite_decode has been set to True then we combine images along edges
    # according to the information in fov_positioning input (extracted from json files) and then run spot finding
    # and decoding on the large composite image.
    if (
        decodeRunnerKwargs["callableDecoder"]
        and decodeRunnerKwargs["callableDecoder"].__name__ == "postcodeDecode"
        and is_composite
    ):
        if not anchor_name:
            print(
                "No anchor image detected. Using max projection of primary image as reference for spot finding (required for postcodeDecode)"
            )

        # Creates the big images. If not given an anchor_name then it takes the max projection of the primary image
        composite_img, composite_anchor = createComposite(
            experiment,
            input_dir,
            anchor_name,
            is_volume,
            level_method,
            selected_fovs,
            **compositeKwargs,
        )

        # Find spots and decode the composite image
        blobs, decoded = blobDriver(
            composite_img,
            composite_anchor.reduce({Axes.CH, Axes.ROUND}, func="max"),
            experiment.codebook,
            blobRunnerKwargs,
            decodeRunnerKwargs,
            output_name=f"{output_dir}spots/composite_",
        )

        # Save composite results
        if len(decoded) > 0:
            saveTable(decoded, output_dir + "csv/composite_decoded.csv")
            # decoded[fov].to_decoded_dataframe().save_csv(output_dir+fov+"_decoded.csv")
            decoded.to_netcdf(output_dir + "cdf/composite_decoded.cdf")
            print(f"Saved cdf file {output_dir}cdf/composite_decoded.cdf")
        else:
            print("No transcripts found for composite! Not saving a DecodedIntensityTable file.")

        # Saves per FOV spots and decoded results
        saveCompositeResults(blobs, decoded, input_dir, selected_fovs, output_name=f"{output_dir}")

    # Otherwise run on a per FOV basis
    else:
        if selected_fovs is not None:
            fovs = ["fov_{:05}".format(int(f)) for f in selected_fovs]
        else:
            fovs = [*experiment.keys()]

        # Calculate scaling factors for rescaling
        if rescale:
            scaling_factors = scale_img(fovs, experiment, pixelRunnerKwargs, is_volume)

        for fov in fovs:
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
                    experiment[fov]
                    .get_image(anchor_name)
                    .reduce({Axes.CH, Axes.ROUND}, func="max")
                )

            # Apply scaling factors for rescaling
            if rescale:

                # Scale with final factors (if not None)
                if scaling_factors is not None:
                    for r in range(img.num_rounds):
                        for ch in range(img.num_chs):
                            img.xarray.data[r, ch] = (
                                img.xarray.data[r, ch] / scaling_factors[(r, ch)]
                            )

                # Scale image
                pmin = 0
                pmax = 100
                clip = starfish.image.Filter.ClipPercentileToZero(
                    p_min=pmin, p_max=pmax, is_volume=is_volume, level_method=Levels.SCALE_BY_IMAGE
                )
                clip.run(img, in_place=True)

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
    p = ArgumentParser()

    # inputs
    p.add_argument("--exp-loc", type=Path)
    p.add_argument("--tmp-prefix", type=str)
    p.add_argument("--is-volume", dest="is_volume", action="store_true")
    p.add_argument("--rescale", dest="rescale", action="store_true")
    p.add_argument("--level-method", type=str, nargs="?")
    p.add_argument("--anchor-view", type=str, nargs="?")
    p.add_argument("--not-filtered-results", dest="not_filtered_results", action="store_true")
    p.add_argument("--selected-fovs", nargs="+", const=None)

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
    p.add_argument("--metric", type=str, nargs="?")
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
    p.add_argument("--n-processes", type=int, nargs="?")  # also used in PixelDecoder

    # == postcodeDecode
    p.add_argument("--composite-decode", dest="composite_decode", action="store_true")
    p.add_argument("--composite-pmin", type=float, nargs="?")
    p.add_argument("--composite-pmax", type=float, nargs="?")

    # pixelRunner kwargs
    p.add_argument("--distance-threshold", type=float, nargs="?")
    p.add_argument("--magnitude-threshold", type=float, nargs="?")
    p.add_argument("--pnorm", type=int, nargs="?")
    p.add_argument("--min-area", type=int, nargs="?")
    p.add_argument("--max-area", type=int, nargs="?")

    args = p.parse_args()

    # for item in vars(args):
    #    print(item, ':', vars(args)[item])

    output_dir = f"tmp/{args.tmp_prefix}/4_Decoded_{args.tmp_prefix}/"

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
    #addKwarg(args, pixelRunnerKwargs, "metric")
    addKwarg(args, pixelRunnerKwargs, "distance_threshold")
    addKwarg(args, pixelRunnerKwargs, "magnitude_threshold")
    addKwarg(args, pixelRunnerKwargs, "pnorm")
    addKwarg(args, pixelRunnerKwargs, "min_area")
    addKwarg(args, pixelRunnerKwargs, "max_area")
    addKwarg(args, pixelRunnerKwargs, "norm_order")
    addKwarg(args, pixelRunnerKwargs, "n_processes")
    if "magnitude_threshold" in pixelRunnerKwargs.keys():
        if pixelRunnerKwargs["magnitude_threshold"] < 1:
            pixelRunnerKwargs["magnitude_threshold"] *= 2**16

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
        elif method == "postcodeDecode":
            method = starfish.spots.DecodeSpots.postcodeDecode
            # Check that codebook is compatible with postcode
            codebook = experiment.codebook
            codebook_no_blanks = codebook[
                ["blank" not in target.lower() for target in codebook["target"].data]
            ]
            if len(codebook_no_blanks) >= len(codebook["c"]) ** len(codebook["r"]):
                raise Exception(
                    "PoSTcode decoder requires some unused barcode space or some blank codes in \
                                 the codebook. If you have used 100% of the barcode space for real codes, \
                                 then PoSTcode is not a valid decoding option."
                )
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

    compositeKwargs = {}
    addKwarg(args, compositeKwargs, "composite_pmin")
    addKwarg(args, compositeKwargs, "composite_pmax")

    use_ref = args.use_ref_img
    anchor_name = args.anchor_view
    rescale = args.rescale
    composite = args.composite_decode
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
        exploc,
        experiment,
        blob_based,
        use_ref,
        anchor_name,
        rescale,
        level_method,
        vol,
        composite,
        not_filtered_results,
        args.selected_fovs,
        blobRunnerKwargs,
        decodeRunnerKwargs,
        pixelRunnerKwargs,
        compositeKwargs,
    )
