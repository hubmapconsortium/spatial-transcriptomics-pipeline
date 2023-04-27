#!/usr/bin/env python

import glob
from argparse import ArgumentParser
from copy import deepcopy
from os import makedirs
from pathlib import Path

import numpy as np
import skimage.measure
import skimage.segmentation
import tifffile
from scipy.ndimage import gaussian_filter
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable


def scale_img(image):
    image = image - np.min(image)
    image = image.astype("float32") / np.max(image)
    image = image * 2**16
    image[image == 2**16] = 2**16 - 1
    return np.rint(image).astype("uint16")


def _clip_percentile_to_zero(image, p_min, p_max, min_coeff=1, max_coeff=1):
    v_min, v_max = np.percentile(image, [p_min, p_max])
    v_min = min_coeff * v_min
    v_max = max_coeff * v_max
    return np.clip(image, v_min, v_max) - np.float32(v_min)


def cellpose_format(output_dir, input_dir, aux_ch_names, mRNA_dir, selected_fovs):
    # Get all fov names
    if selected_fovs is None:
        primary_jsons = glob.glob(f"{input_dir}/primary-*.json")
        fovs = [
            primary_json.split("/")[-1].split("-")[-1].split(".")[0]
            for primary_json in primary_jsons
        ]
    else:
        fovs = ["fov_{:03}".format(int(f)) for f in selected_fovs]

    # Get number of z slices by looking at the fov_000 files
    fov0_files = glob.glob(f"{input_dir}/primary-{fovs[0]}*.tiff")
    shape = tifffile.imread(fov0_files[0]).shape  # Get image xy shape (will need later)
    fov0_files = [file.split("/")[-1] for file in fov0_files]
    zs = np.max([int(file.split("-")[-1][1]) for file in fov0_files])

    # Make folder for cellpose inputs if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    # Create cellpose inputs
    # Images have to be 16 bit and have dimension order z, ch, y, x
    for fov in fovs:
        mRNA_ch = 1 if mRNA_dir else 0
        empty_ch = 1 if len(aux_ch_names) == 1 and not mRNA_ch else 0
        new_img = np.zeros(
            [zs + 1, len(aux_ch_names) + mRNA_ch + empty_ch] + list(shape), dtype="uint16"
        )
        for ch, aux_ch_name in enumerate(aux_ch_names):
            files = sorted(glob.glob(f"{input_dir}/{aux_ch_name}-{fov}-c0-r0-z*.tiff"))
            for z, file in enumerate(files):
                img = tifffile.imread(file)
                if np.max(img) <= 1:
                    img = np.rint(img * 2**16).astype("uint16")
                new_img[z, ch] = deepcopy(img)
            new_img[:, ch] = _clip_percentile_to_zero(new_img[:, ch], p_min=0, p_max=99.9)
            new_img[:, ch] = scale_img(new_img[:, ch])

        # Add mRNA density channel if specified
        # Each mRNA is plotted as a single point of maximum intensity and the the resulting image is then
        # blurred with a guassian filter.
        if mRNA_dir:
            dit = DecodedIntensityTable.open_netcdf(f"{mRNA_dir}/cdf/{fov}_decoded.cdf")
            coords = np.array(
                [[z, y, x] for z, y, x in zip(dit["z"].data, dit["y"].data, dit["x"].data)]
            )
            mRNAs = np.zeros([zs + 1] + list(shape), dtype="uint16")
            mRNAs[tuple(coords.T)] = 2**16 - 1
            for z in range(zs + 1):
                mRNAs[z] = gaussian_filter(
                    mRNAs[z], sigma=10, cval=0, truncate=4.0, mode="nearest"
                )
            new_img[:, -1] = scale_img(mRNAs)

        # Save result, squeeze out any size 1 dimensions
        tifffile.imsave(f"{output_dir}/{fov}_image.tiff", np.squeeze(new_img))


def filter_cellpose(
    output_dir, input_dir, border_buffer=None, label_exp_size=None, min_size=None, max_size=None
):
    # Make folder if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    # For each file, check for each function if it should be run then run it if yes
    files = glob.glob(f"{input_dir}/*cp_masks*")
    for file in files:
        # print(f"found {file}")
        mask = tifffile.imread(file)

        # Clear border objects
        if border_buffer is not None:
            # print(f"\tclearing border of size {border_buffer}")
            if mask.ndim == 3:
                for z in range(mask.shape[0]):
                    mask[z] = skimage.segmentation.clear_border(mask[z], buffer_size=border_buffer)
            else:
                mask = skimage.segmentation.clear_border(mask, buffer_size=border_buffer)
            mask = skimage.segmentation.relabel_sequential(mask)[0]

        # Expand labels
        if label_exp_size is not None:
            # print(f"\texpanding labels by size {label_exp_size}")
            if mask.ndim == 3:
                for z in range(mask.shape[0]):
                    mask[z] = skimage.segmentation.expand_labels(mask[z], distance=label_exp_size)
            else:
                mask = skimage.segmentation.expand_labels(mask, distance=label_exp_size)

        # Remove labels below a minimum size threshold
        if min_size is not None:
            # print(f"\tcells beneath size {min_size} being removed")
            props = skimage.measure.regionprops(mask)
            small_labels = np.where([p.area < min_size for p in props])[0] + 1
            mask[np.isin(mask, small_labels)] = 0
            mask = skimage.segmentation.relabel_sequential(mask)[0]

        # Remove labels above a maximum size threshold
        if max_size is not None:
            # print(f"\tcells above size {max_size} being removed")
            props = skimage.measure.regionprops(mask)
            big_labels = np.where([p.area > max_size for p in props])[0] + 1
            mask[np.isin(mask, big_labels)] = 0
            mask = skimage.segmentation.relabel_sequential(mask)[0]

        # Save result
        tifffile.imsave(f'{output_dir}/fov_{file.split("fov_")[-1][:3]}_masks.tiff', mask)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--input-dir", type=Path)
    p.add_argument("--tmp-prefix", type=str)
    p.add_argument("--selected-fovs", nargs="+", const=None)

    p.add_argument("--format", dest="format", action="store_true")
    p.add_argument("--aux-views", type=str, nargs="+")
    p.add_argument("--decoded-dir", type=Path, nargs="?")

    p.add_argument("--filter", dest="filter", action="store_true")
    p.add_argument("--border-buffer", type=int, nargs="?")
    p.add_argument("--label-exp-size", type=int, nargs="?")
    p.add_argument("--min-size", type=int, nargs="?")
    p.add_argument("--max-size", type=int, nargs="?")

    args = p.parse_args()

    if not (args.format ^ args.filter):
        raise ValueError("Script must be run with --format xor --filter. Terminating.")

    if args.format:
        cellpose_format(
            output_dir=f"tmp/{args.tmp_prefix}/5A_cellpose_input",
            input_dir=args.input_dir,
            aux_ch_names=args.aux_views,
            mRNA_dir=args.decoded_dir,
            selected_fovs=args.selected_fovs,
        )
    else:
        filter_cellpose(
            output_dir=f"tmp/{args.tmp_prefix}/5C_cellpose_filtered",
            input_dir=args.input_dir,
            border_buffer=args.border_buffer,
            label_exp_size=args.label_exp_size,
            min_size=args.min_size,
            max_size=args.max_size,
        )
        # because we know these two will be called as a part of the same cwl,
        # we don't need to re-clarify selected_fovs on the filter step
