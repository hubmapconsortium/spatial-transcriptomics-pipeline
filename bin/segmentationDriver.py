#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from functools import partialmethod
from glob import glob
from os import makedirs, path
from pathlib import Path

import cv2
import numpy as np
import skimage
import starfish
import starfish.data
from scipy import ndimage
from skimage.filters import threshold_local
from skimage.morphology import disk
from starfish import BinaryMaskCollection, DecodedIntensityTable, ImageStack
from starfish.image import Filter as ImgFilter
from starfish.morphology import Binarize, Filter, Segment
from starfish.spots import AssignTargets
from starfish.types import Axes, Levels
from tqdm import tqdm


def maskFromRoi(
    img: ImageStack, fov: str, roi_set: Path, file_formats: str
) -> BinaryMaskCollection:
    """
    Return a mask from provided RoiSet.zip files.

    Parameters
    ----------
    img: ImageStack
        The image that the mask is to be applied to.
    fov: str
        The name of the fov that we need to segment.
    roi_set: Path
        Directory containing RoiSet files.
    file_formats: str
        String that will have .format() applied for each FOV.  Will be appended to roi_set.

    Returns
    -------
    list[BinaryMaskCollection]:
        Binary masks for each FOV.
    """
    i = int(fov[-3:])
    mask_name = ("{}/" + file_formats).format(roi_set, i)
    return BinaryMaskCollection.from_fiji_roi_set(mask_name, img)


def maskFromLabeledImages(
    img: ImageStack, fov: str, labeled_image: Path, file_formats_labeled: str
) -> BinaryMaskCollection:
    """
    Returns a mask from the provided labeled images.

    Parameters
    ----------
    img: ImageStack
        The image that the masks will be applied to.
    fov: str
        The name of the fov that we need to segment.
    labeled_image: Path
        Directory of labeled images with image segmentation data, such as from ilastik classification.
    file_formats_labeled: str
        Layout for name of each labelled image. Will be formatted with String.format([fov index])

    Returns
    -------
    BinaryMaskCollection:
        Binary mask.
    """
    i = int(fov[-3:])
    label_name = ("{}/" + file_formats_labeled).format(labeled_image, i)
    return BinaryMaskCollection.from_external_labeled_image(label_name, img)


def maskFromWatershed(
    img: ImageStack,
    img_threshold: float,
    min_dist: int,
    min_size: int,
    max_size: int,
    masking_radius: int,
) -> BinaryMaskCollection:
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
    BinaryMaskCollection:
        Binary masks for this FOV.
    """
    wt_filt = ImgFilter.WhiteTophat(masking_radius, is_volume=False)
    thresh_filt = Binarize.ThresholdBinarize(img_threshold)
    min_dist_label = Filter.MinDistanceLabel(min_dist, 1)
    area_filt = Filter.AreaFilter(min_area=min_size, max_area=max_size)
    area_mask = Filter.Reduce("logical_or", lambda shape: np.zeros(shape=shape, dtype=bool))
    segmenter = Segment.WatershedSegment()
    img_flat = img.reduce({Axes.ROUND, Axes.CH}, func="max")
    working_img = wt_filt.run(img_flat, in_place=False)
    working_img = thresh_filt.run(working_img)
    labeled = min_dist_label.run(working_img)
    working_img = area_filt.run(labeled)
    working_img = area_mask.run(working_img)
    return segmenter.run(img_flat, labeled, working_img)


def _clip_percentile_to_zero(image, p_min, p_max, min_coeff=1, max_coeff=1):
    v_min, v_max = np.percentile(image, [p_min, p_max])
    v_min = min_coeff * v_min
    v_max = max_coeff * v_max
    return np.clip(image, v_min, v_max) - np.float32(v_min)


def segment_nuclei(
    nuclei, border_buffer=None, area_thresh=1.05, thresh_block_size=51, wtshd_ftprnt_size=100
):

    good_nuclei_all_z = np.zeros(nuclei.xarray.data[0, 0].shape, dtype="int32")
    all_nuclei_all_z = np.zeros(nuclei.xarray.data[0, 0].shape, dtype="int32")
    og_nuclei = deepcopy(nuclei)
    for z in range(nuclei.num_zplanes):

        nuclei = deepcopy(og_nuclei)

        # Even out background of nuclei image using large morphological opening
        size = 200
        sigma = 20
        background = cv2.morphologyEx(nuclei.xarray.data[0, 0, z], cv2.MORPH_OPEN, disk(size))
        smoothed = ndimage.gaussian_filter(background, sigma=sigma)
        nuclei.xarray.data[0, 0, z] = nuclei.xarray.data[0, 0, z] - smoothed
        nuclei.xarray.data[0, 0, z][nuclei.xarray.data[0, 0, z] < 0] = 0

        # Median filter
        nuclei.xarray.data[0, 0, z] = ndimage.median_filter(nuclei.xarray.data[0, 0, z], size=10)

        # Scale image intensities and convert low intensities to zero
        pmin = 50
        pmax = 100
        clip = starfish.image.Filter.ClipPercentileToZero(
            p_min=pmin, p_max=pmax, is_volume=False, level_method=Levels.SCALE_BY_CHUNK
        )
        scaled_nuclei = clip.run(nuclei, in_place=False)

        # Threshold locally
        param = 100
        local_thresh = threshold_local(
            scaled_nuclei.xarray.data[0, 0, z], thresh_block_size, offset=0, param=param
        )
        binary = scaled_nuclei.xarray.data[0, 0, z] > local_thresh

        # Morphological opening and closing to even edges out of nuclei
        size = 10
        kernel = np.ones((size, size), np.uint8)
        binary = cv2.morphologyEx(binary.astype("uint8"), cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary.astype("uint8"), cv2.MORPH_CLOSE, kernel)

        # Remove small objects
        cc_labels = skimage.measure.label(binary)
        props = skimage.measure.regionprops(cc_labels)
        areas = [p.area for p in props]
        outlier = 100
        area_small = [p.area < outlier for p in props]
        for x in range(1, len(area_small) + 1):
            if area_small[x - 1]:
                cc_labels[cc_labels == x] = 0
        cc_labels = skimage.segmentation.relabel_sequential(cc_labels)[0]

        # Convert back to binary
        binary = (cc_labels >= 1).astype("int16")

        # Find all non-overlapping nuclei (aka "good" nuclei)

        # Label each connect component as a single nucleus
        good_nuclei = skimage.measure.label(binary)

        # Remove border objects if specified
        if border_buffer is not None:
            good_nuclei = skimage.segmentation.clear_border(good_nuclei, buffer_size=border_buffer)
            good_nuclei = skimage.segmentation.relabel_sequential(good_nuclei)[0]

        # Identify objects with deformed borders (likely overlapping nuclei) by ratio of convex hull area with
        # normal area
        props = skimage.measure.regionprops(good_nuclei)
        deformed_border = []
        for x in range(1, len(props) + 1):

            object_area = props[x - 1].area
            con_hull_area = props[x - 1].convex_area

            if con_hull_area / object_area > area_thresh:
                deformed_border.append(False)
            else:
                deformed_border.append(True)

        for x in range(1, len(deformed_border) + 1):
            if not deformed_border[x - 1]:
                good_nuclei[good_nuclei == x] = 0
        good_nuclei = skimage.segmentation.relabel_sequential(good_nuclei)[0]

        # Remove big
        props = skimage.measure.regionprops(good_nuclei)
        if len(props) > 0:
            areas = [p.area for p in props]
            outlier = np.percentile(areas, 75) + 1.5 * (
                np.percentile(areas, 75) - np.percentile(areas, 25)
            )
            area_big = [p.area > outlier for p in props]

            for x in range(1, len(area_big) + 1):
                if area_big[x - 1]:
                    good_nuclei[good_nuclei == x] = 0
            good_nuclei = skimage.segmentation.relabel_sequential(good_nuclei)[0]

        # Segment all nuclei

        # Segment all nuclei using watershed
        distance = ndimage.distance_transform_edt(binary)
        max_coords = skimage.feature.peak_local_max(
            distance, labels=binary, footprint=np.ones((wtshd_ftprnt_size, wtshd_ftprnt_size))
        )
        local_maxima = np.zeros_like(binary, dtype=bool)
        local_maxima[tuple(max_coords.T)] = True
        markers = ndimage.label(local_maxima)[0]
        all_nuclei = skimage.segmentation.watershed(-distance, markers, mask=binary)

        # Fix nuclei segmentation errors
        all_nuclei = all_nuclei.astype("uint16")

        # Calculate number of pixels to set as threshold for merging objects. Need to set as a constant pixel
        # number and not ratio of areas to prevent large objects from blobbing up.
        # Calculated as (area_thresh - 1) * (average good nuclei area)
        props = skimage.measure.regionprops(all_nuclei.astype("int16"))
        areas = [p.area for p in props]
        px_area_thresh = np.mean(areas) * (area_thresh - 1)

        ymax, xmax = nuclei.shape["y"], nuclei.shape["x"]
        flag = False
        while not flag:
            # Find pairs of label that are adjacent (possible segmentation errors)
            pairs = {}
            for x in range(1, xmax - 1):
                for y in range(1, ymax - 1):
                    value = all_nuclei[y, x]
                    if value != 0:
                        neighbors = []
                        neighbors.append(all_nuclei[y - 1, x])
                        neighbors.append(all_nuclei[y + 1, x])
                        neighbors.append(all_nuclei[y, x - 1])
                        neighbors.append(all_nuclei[y, x + 1])
                        neighbors = [n for n in neighbors if n != 0]
                        for neighbor in neighbors:
                            if neighbor != value:
                                pairs[(int(value), int(neighbor))] = 0
            neighbor_pairs = list(set([tuple(sorted(pair)) for pair in pairs]))

            props = skimage.measure.regionprops(all_nuclei.astype("int16"))
            # Check ratio between area of the convex hull of both object and the original combined area, if it is low,
            # then the two objects are probably a single object and should be merged
            merge_count = 0
            removed_labels = []
            for pair in neighbor_pairs:

                if pair[0] not in removed_labels and pair[1] not in removed_labels:

                    pair_image1 = np.zeros((ymax, xmax))
                    bbox = props[pair[0] - 1].bbox
                    pair_image1[bbox[0] : bbox[2], bbox[1] : bbox[3]] = props[
                        pair[0] - 1
                    ].convex_image
                    pair_image2 = np.zeros((ymax, xmax))
                    bbox = props[pair[1] - 1].bbox
                    pair_image2[bbox[0] : bbox[2], bbox[1] : bbox[3]] = props[
                        pair[1] - 1
                    ].convex_image
                    pair_image = pair_image1.astype(bool) | pair_image2.astype(bool)
                    pair_props = skimage.measure.regionprops(pair_image.astype(int))

                    if pair_props[0].convex_area - pair_props[0].area < px_area_thresh:
                        all_nuclei[all_nuclei == pair[0]] = pair[1]
                        merge_count += 1
                        removed_labels.append(pair[0])
            all_nuclei = skimage.segmentation.relabel_sequential(all_nuclei.astype("int16"))[0]
            if merge_count == 0:
                flag = True

        # Remove small
        props = skimage.measure.regionprops(all_nuclei)
        for x in np.where([props[x].area < 100 for x in range(len(props))])[0]:
            all_nuclei[all_nuclei == x + 1] = 0
        all_nuclei = skimage.segmentation.relabel_sequential(all_nuclei)[0]

        good_nuclei_all_z[z] = deepcopy(good_nuclei)
        all_nuclei_all_z[z] = deepcopy(all_nuclei)

    # Adjust values so there are no repeats across slices
    for z in range(1, nuclei.num_zplanes):
        good_prev_max = np.max(good_nuclei_all_z[z - 1])
        good_nuclei_all_z[z] = good_nuclei_all_z[z] + good_prev_max
        good_nuclei_all_z[z][good_nuclei_all_z[z] == good_prev_max] = 0

        all_prev_max = np.max(all_nuclei_all_z[z - 1])
        all_nuclei_all_z[z] = all_nuclei_all_z[z] + all_prev_max
        all_nuclei_all_z[z][all_nuclei_all_z[z] == all_prev_max] = 0

    return good_nuclei_all_z, all_nuclei_all_z


def segment_cytoplasm(
    good_nuclei, all_nuclei, decoded_targets, border_buffer=None, label_exp_size=None
):

    # Convert transcripts to dataframe
    data = deepcopy(decoded_targets.to_features_dataframe())

    # Create empty arrays to put results in
    good_cyto_all_z = np.zeros(good_nuclei.shape, dtype="int32")
    all_cyto_all_z = np.zeros(all_nuclei.shape, dtype="int32")

    for z in range(all_nuclei.shape[0]):

        # Subset data for current z slice
        data = data[data["z"] == z]

        # Create image whose pixel intensities are proportional to the number of transcripts found in it's search
        # radius defined neighborhood
        density = np.zeros(good_nuclei[z].shape)
        for x, y in zip(data["x"], data["y"]):
            density[y, x] = 1

        # Dilate a bit
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(density, kernel, iterations=3)

        # Smooth density map with gaussian filter
        smoothed_density = ndimage.gaussian_filter(dilation, sigma=10)

        pmin = 0
        pmax = 80
        scaled_density = _clip_percentile_to_zero(smoothed_density, pmin, pmax)

        # Threshold locally
        block_size = 101
        param = 100
        local_thresh = threshold_local(scaled_density, block_size, offset=0, param=param)
        binary = scaled_density > local_thresh

        # Add in nuclei to ensure they are in the foreground
        binary = binary + all_nuclei[z]
        binary[binary > 1] = 1

        # Closing to fill small gaps
        size = 10
        kernel = np.ones((size, size), np.uint8)
        closed = cv2.morphologyEx(binary.astype("uint8"), cv2.MORPH_CLOSE, kernel)
        closed = closed.astype(bool)

        # Segment cytoplasms using nuclei centroids as markers and binary nuclei image as mask
        distance = ndimage.distance_transform_edt(closed)
        cyto_labels = skimage.segmentation.watershed(
            -distance, all_nuclei[z], mask=closed, compactness=0.001
        )

        # Remove cytoplasms associated with border or suspected overlapping nuclei
        bad_nuclei = all_nuclei[z].astype(bool) ^ good_nuclei[z].astype(bool)
        bad_labels = skimage.measure.label(bad_nuclei.astype(int))
        props = skimage.measure.regionprops(bad_labels)
        for x in np.where([props[x].area < 100 for x in range(len(props))])[0]:
            bad_labels[bad_labels == x + 1] = 0
        bad_nuclei = deepcopy(bad_labels).astype(bool)
        bad_cyto = np.unique(cyto_labels[bad_nuclei])
        final_labels = deepcopy(cyto_labels)
        final_labels[np.isin(cyto_labels, bad_cyto)] = 0

        # Remove small
        props = skimage.measure.regionprops(final_labels)
        area_small = [p.area < 20000 for p in props]
        for x in range(1, len(area_small) + 1):
            if area_small[x - 1]:
                final_labels[final_labels == x] = 0
        final_labels = skimage.segmentation.relabel_sequential(final_labels)[0]

        # Remove bg
        props = skimage.measure.regionprops(final_labels)
        area_big = [p.area > 200000 for p in props]
        for x in range(1, len(area_big) + 1):
            if area_big[x - 1]:
                final_labels[final_labels == x] = 0
        final_labels = skimage.segmentation.relabel_sequential(final_labels)[0]

        # Remove cytoplasm that have multiple nuclei in them
        for label in range(1, final_labels.max() + 1):
            nuclei_in_cyto = np.unique(all_nuclei[z][final_labels == label])
            if len(nuclei_in_cyto) > 2:
                final_labels[final_labels == label] = 0
        final_labels = skimage.segmentation.relabel_sequential(final_labels)[0]

        if border_buffer is not None:
            # Remove cytoplasms associated with border nuclei (for full cytoplasm set if border_buffer is set)
            all_nuclei_no_bord = skimage.segmentation.clear_border(
                all_nuclei[z], buffer_size=border_buffer
            )
            all_nuclei_no_bord = skimage.segmentation.relabel_sequential(all_nuclei_no_bord)[0]

            bad_nuclei = all_nuclei[z].astype(bool) ^ all_nuclei_no_bord.astype(bool)
            bad_labels = skimage.measure.label(bad_nuclei.astype(int))
            props = skimage.measure.regionprops(bad_labels)
            for x in np.where([props[x].area < 100 for x in range(len(props))])[0]:
                bad_labels[bad_labels == x + 1] = 0
            bad_nuclei = deepcopy(bad_labels).astype(bool)
            bad_cyto = np.unique(cyto_labels[bad_nuclei])
            cyto_labels_no_bord = deepcopy(cyto_labels)
            cyto_labels_no_bord[np.isin(cyto_labels_no_bord, bad_cyto)] = 0
            cyto_labels = deepcopy(cyto_labels_no_bord)

        # Expand labels
        if label_exp_size is not None:
            all_cyto_labels = skimage.segmentation.expand_labels(
                cyto_labels, distance=label_exp_size
            )
            good_cyto_labels = skimage.segmentation.expand_labels(
                final_labels, distance=label_exp_size
            )

        good_cyto_all_z[z] = deepcopy(good_cyto_labels)
        all_cyto_all_z[z] = deepcopy(all_cyto_labels)

    # Adjust values so there are no repeats across slices
    for z in range(1, good_cyto_all_z.shape[0]):
        good_prev_max = np.max(good_cyto_all_z[z - 1])
        good_cyto_all_z[z] = good_cyto_all_z[z] + good_prev_max
        good_cyto_all_z[z][good_cyto_all_z[z] == good_prev_max] = 0

        all_prev_max = np.max(all_cyto_all_z[z - 1])
        all_cyto_all_z[z] = all_cyto_all_z[z] + all_prev_max
        all_cyto_all_z[z][all_cyto_all_z[z] == all_prev_max] = 0

    return good_cyto_all_z, all_cyto_all_z


def segmentByDensity(
    nuclei,
    decoded_targets,
    cyto_seg=False,
    correct_seg=False,
    border_buffer=None,
    area_thresh=1.05,
    thresh_block_size=51,
    watershed_footprint_size=100,
    label_exp_size=20,
    nuclei_view="",
):

    """
    Parameters
        nuclei: Nuclei images.
        decoded_targets: Decoded transcripts (DecodedIntensityTable).
        cyto_seg: Boolean whether to segment the cytoplasm or not.
        correct_seg: Boolean whether to remove suspected nuclei/cytoplasms that have overlapping nuclei.
        border_buffer: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.
        area_thresh: Threshold used when determining if an object is one nucleus or two or more overlapping nuclei.
                     Objects whose ratio of convex hull area to normal area are above this threshold are removed if
                     the option to remove overlapping nuclei is set.
        thesh_block_size: Size of structuring element for local thresholding of nuclei. If nuclei interiors aren't
                          passing threshold, increase this value, if too much non-nuclei is passing threshold, lower
                          it.
        wtshd_ftprnt_size: Size of structuring element for watershed segmentation. Larger values will segment the
                           nuclei into larger objects and smaller values will result in smaller objects. Adjust
                           according to nucleus size.
        label_exp_size: Pixel size labels are dilated by in final step. Helpful for closing small holes that are
                        common from thresholding but can also cause cell boundaries to exceed their true boundaries
                        if set too high. Label dilation respects label borders and does not mix labels.
        nuclei_view: dummy variable to prevent issues with how we load in kwargs.
    """

    # TODO: Currently only works for 2D images, will modify to work with 3D images that are treated as stacks of
    # separate 2D images. This will not work for 3D volumes.

    # Since all processing pipelines register all images to the round 0 channel 0 image, this just takes that
    # one image to perform segmentation on since they should all be the same anyway and using the same image
    # that is used as reference for registration makes registering the image unnecessary. Might need to adjust
    # in the future as we run new datasets but this should work for now.

    # Segment nuclei images. Returns two label images: good_nuclei which has overlapping nuclei removed and
    # all_nuclei which includes all nuclei. If border_buffer has been set good_nuclei has had overlapping
    # nuclei removed but not all_nuclei (need border nuclei for accurate cytoplasm segmentation).
    good_nuclei, all_nuclei = segment_nuclei(
        nuclei,
        border_buffer=border_buffer,
        area_thresh=area_thresh,
        thresh_block_size=thresh_block_size,
        wtshd_ftprnt_size=watershed_footprint_size,
    )
    # Segment cytoplasm if specifed, returns two objects like previous function, cytoplasms with non-overlapping
    # nuclei and all cytoplasms (minus border if border_buffer is set).
    if cyto_seg:
        good_cyto_labels, all_cyto_labels = segment_cytoplasm(
            good_nuclei,
            all_nuclei,
            decoded_targets,
            border_buffer=border_buffer,
            label_exp_size=label_exp_size,
        )
        # If correct_seg is True returns cytoplasm labels that whose nuclei do not have overlaps otherwise
        # returns all cytoplasms (exception to those with border nuclei if border_buffer is set)
        if correct_seg:
            return good_cyto_labels
        else:
            return all_cyto_labels
    # If cyto_seg is False then we just return one of the nuclei images
    else:
        # Non overlapping nuclei
        if correct_seg:
            if label_exp_size is not None:
                for z in range(nuclei.num_zplanes):
                    good_nuclei[z] = skimage.segmentation.expand_labels(
                        good_nuclei[z], distance=label_exp_size
                    )
            return good_nuclei
        # All nuclei
        else:

            # Had to put this here because I need border nuclei for cytoplasm segmentation.
            if border_buffer is not None:
                for z in range(nuclei.num_zplanes):
                    all_nuclei[z] = skimage.segmentation.clear_border(
                        all_nuclei[z], buffer_size=border_buffer
                    )
                    all_nuclei[z] = skimage.segmentation.relabel_sequential(all_nuclei[z])[0]
            if label_exp_size is not None:
                for z in range(nuclei.num_zplanes):
                    all_nuclei[z] = skimage.segmentation.expand_labels(
                        all_nuclei[z], distance=label_exp_size
                    )

            return all_nuclei


def run(
    input_loc: Path,
    exp_loc: Path,
    output_loc: str,
    aux_name: str,
    roiKwargs: dict,
    labeledKwargs: dict,
    watershedKwargs: dict,
    densityKwargs: dict,
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
    aux_name: str
        The name of the auxillary view to look at for image segmentation.
    roiKwargs: dict
        Dictionary with arguments for reading in masks from an RoiSet. See masksFromRoi.
    labeledKwargs: dict
        Dictionary with arguments for reading in masks from a labeled image. See masksFromLabeledImages.
    watershedKwargs: dict
        Dictionary with arguments for running basic watershed pipeline. See masksFromWatershed.
    TODO: args for density segmentation
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # redirecting output to log
    reporter = open(
        path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_starfish_segmenter.log")),
        "w",
    )
    sys.stdout = reporter
    sys.stderr = reporter

    # if not path.isdir(output_dir + "csv/"):
    #    makedirs(output_dir + "csv")
    #    print("made " + output_dir + "csv")

    # if not path.isdir(output_dir + "cdf/"):
    #    makedirs(output_dir + "cdf")
    #    print("made " + output_dir + "cdf")

    # if not path.isdir(output_dir + "h5ad/"):
    #    makedirs(output_dir + "h5ad")
    #    print("made " + output_dir + "h5ad")

    # read in netcdfs based on how we saved prev step
    results = {}
    for f in glob("{}/cdf/*_decoded.cdf".format(input_loc)):
        name = f[len(str(input_loc)) + 5 : -12]
        print("found fov key: " + name)
        results[name] = DecodedIntensityTable.open_netcdf(f)
        print("loaded " + f)
        if not path.isdir(output_dir + name):
            makedirs(output_dir + name)
            print("made " + output_dir + name)

    # load in the images we want to look at
    exp = starfish.core.experiment.experiment.Experiment.from_json(
        str(exp_loc / "experiment.json")
    )
    print("loaded " + str(exp_loc / "experiment.json"))

    # IF WE'RE DOING DENSITY BASED, THAT's DIFFERENT'
    for key in results.keys():
        if "nuclei_view" in densityKwargs:
            nuclei_img = exp[key].get_image(densityKwargs["nuclei_view"])
            print(f"Segmenting {key}")
            raw_mask = segmentByDensity(
                nuclei=nuclei_img, decoded_targets=results[key], **densityKwargs
            )
            maskname = f"{output_dir}/{key}/mask.tiff"
            skimage.io.imsave(maskname, np.squeeze(raw_mask))
            mask = BinaryMaskCollection.from_external_labeled_image(maskname, nuclei_img)
        else:
            print(f"looking at {key}, {aux_name}")
            cur_img = exp[key].get_image(aux_name)
            # determine how we generate mask, then make it
            if len(roiKwargs.keys()) > 0:
                # then apply roi
                print("applying Roi mask")
                mask = maskFromRoi(cur_img, key, **roiKwargs)
            elif len(labeledKwargs.keys()) > 0:
                # then apply images
                print("applying labeled image mask")
                mask = maskFromLabeledImages(cur_img, key, **labeledKwargs)
            elif len(watershedKwargs.keys()) > 0:
                # then go thru watershed pipeline
                print("running basic threshold and watershed pipeline")
                mask = maskFromWatershed(cur_img, **watershedKwargs)
            else:
                # throw error
                raise Exception("Parameters do not specify means of defining mask.")

            # save masks to tiffs for later processing
            intmask = mask.to_label_image().xarray.values
            skimage.io.imsave(f"{output_dir}/{key}/mask.tiff", np.squeeze(intmask))

        # apply mask to tables, save results
        al = AssignTargets.Label()
        labeled = al.run(mask, results[key])
        # labeled = labeled[labeled.cell_id != "nan"]
        labeled.to_features_dataframe().to_csv(output_dir + key + "/segmentation.csv")
        labeled.to_netcdf(output_dir + key + "/df_segmented.cdf")
        labeled.to_expression_matrix().to_pandas().to_csv(output_dir + key + "/exp_segmented.csv")
        labeled.to_expression_matrix().save(output_dir + key + "/exp_segmented.cdf")
        labeled.to_expression_matrix().save_anndata(output_dir + key + "/exp_segmented.h5ad")
        print("saved fov key: {}".format(key))

    if len(results) == 0:
        print("No FOVs found! Did the decoding step complete correctly?")

    sys.stdout = sys.__stdout__


def addKwarg(parser, kwargdict, var):
    result = getattr(parser, var)
    if result:
        kwargdict[var] = result


if __name__ == "__main__":
    output_dir = "5_Segmented/"
    p = ArgumentParser()

    p.add_argument("--decoded-loc", type=Path)
    p.add_argument("--exp-loc", type=Path)
    p.add_argument("--aux-name", type=str, nargs="?")

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

    # for density-based segmentation
    p.add_argument("--nuclei-view", type=str, nargs="?")
    p.add_argument("--cyto-seg", dest="cyto_seg", action="store_true")
    p.add_argument("--correct-seg", dest="correct_seg", action="store_true")
    p.add_argument("--border-buffer", type=int, nargs="?")
    p.add_argument("--area-thresh", type=float, nargs="?")
    p.add_argument("--thresh-block-size", type=int, nargs="?")
    p.add_argument("--watershed-footprint-size", type=int, nargs="?")
    p.add_argument("--label-exp-size", type=int, nargs="?")

    args = p.parse_args()

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

    densityKwargs = {}
    addKwarg(args, densityKwargs, "nuclei_view")
    addKwarg(args, densityKwargs, "cyto_seg")
    addKwarg(args, densityKwargs, "correct_seg")
    addKwarg(args, densityKwargs, "border_buffer")
    addKwarg(args, densityKwargs, "area_thresh")
    addKwarg(args, densityKwargs, "thresh_block_size")
    addKwarg(args, densityKwargs, "watershed_footprint_size")
    addKwarg(args, densityKwargs, "label_exp_size")

    run(
        input_dir,
        exp_dir,
        output_dir,
        aux_name,
        roiKwargs,
        labeledKwargs,
        watershedKwargs,
        densityKwargs,
    )
