#!/usr/bin/env python

import functools
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from time import time
from typing import List, Mapping, Union

import click
import numpy as np
import pandas as pd
import skimage.io
import starfish
from slicedimage import ImageFormat
from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features

# below is modified from parsing example from starfish docs

# We use this to cache images across tiles.  In the case of the osmFISH data set, volumes are saved
# together in a single file.  To avoid reopening and decoding the TIFF file, we use a single-element
# cache that maps between file_path and the npy file.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    # return skimage.io.imread(file_path, format="tiff")
    return skimage.io.imread(file_path)


class FISHTile(FetchedTile):
    def __init__(
        self,
        file_path: str,
        zplane: int,
        ch: int,
        rnd: int,
        is_aux: bool,
        cache_read_order: list,
        locs: Mapping[Axes, float] = None,
        voxel: Mapping[Axes, float] = None,
        shape: Mapping[Axes, int] = None,
    ):
        """
        Tile class generalized for most FISH experiments.
        Assumes that image format is .tiff

        Parameters
        ----------
        file_path: str
            Relative path to location of image file.
        zplane: int
            The number of the zplane layer.
        ch: int
            The number of the channel layer.
        rnd: int
            The number of the imaging round.
        is_aux: bool
            if false: The tile in question is primary.
            else: Is an aux image, var is an int with the number of the channel.
        cache_read_order: list
            Description of the order of the axes of the images. Each item in the list is one dimension in the image.
            The following strings will be converted to Axes objects and will be parsed based on the instance variables of the tile:
                -Z -> Axes.ZPLANE
                -CH -> Axes.CH
            All ofther values will be treated as an axis where the full contents will be read for each individual tile. (in pratice, this should only be Axes.X and Axes.Y)

        The following parameters are optional, and are only used if .coordinates() is called.  They may be used further downstream in analysis, in particular if there are multiple FOVs.
        For further details, see FISHTile.coordinates()

        locs: Mapping[Axes, float]
            The start location of the image, mapped to the corresponding Axes object (X, Y, or ZPLANE)
        voxel: Mapping[Axes, float]
            The size of each image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        shape: Mapping[Axes, int]
            The offset for the size of the image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        """
        self._file_path = file_path
        self._zplane = zplane
        self._ch = ch
        self._rnd = rnd
        self.is_aux = is_aux
        self.cache_read_order = cache_read_order
        if locs:
            self.locs = locs
        if voxel:
            self.voxel = voxel
        if shape:
            self.img_shape = shape
        self.coord_def = locs and voxel and shape
        # print("zpl {};ch {}".format(zplane,ch))

    @property
    def shape(self) -> Mapping[Axes, int]:
        """
        Gets image shape directly from the data. Note that this will result in the data being
        read twice, since the shape is retrieved from all tiles before the data is read, and thus
        single-file caching does not resolve the duplicated reads.

        Because the data here isn't tremendously large, this is acceptable in this instance.
        """
        raw_shape = self.tile_data().shape
        return {Axes.Y: raw_shape[0], Axes.X: raw_shape[1]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """
        Returns coordinate values based on values passed at initialization.

        Each cooridinate tuple is calculated by (locs*voxel, (locs+shape)*voxel), with values selected per axis.
        """
        if self.coord_def:
            # return value based on passed parameters
            return {
                Coordinates.X: (
                    self.locs[Axes.X] * self.voxel[Axes.X],
                    (self.locs[Axes.X] + self.img_shape[Axes.X]) * self.voxel[Axes.X],
                ),
                Coordinates.Y: (
                    self.locs[Axes.Y] * self.voxel[Axes.Y],
                    (self.locs[Axes.Y] + self.img_shape[Axes.Y]) * self.voxel[Axes.Y],
                ),
                Coordinates.Z: (
                    self.locs[Axes.ZPLANE] * self.voxel[Axes.ZPLANE],
                    (self.locs[Axes.ZPLANE] + self.img_shape[Axes.ZPLANE])
                    * self.voxel[Axes.ZPLANE],
                ),
            }
        else:
            # no defined location, retrun dummy
            return {
                Coordinates.X: (0.0, 1.0),
                Coordinates.Y: (0.0, 1.0),
                Coordinates.Z: (0.0, 0.1),
            }

    @property
    def format(self) -> ImageFormat:
        """Image Format for SeqFISH data is TIFF"""
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        """
        Squeeze dims of img based on pre-defined cache read order and internal channel and zplane values.

        Returns
        -------
        np.ndarray
            Image with number of dimensions equal to number of 'other' values specified in self.cache_read_order.
        """
        # print(cached_read_fn(self._file_path).shape)
        try:
            img = cached_read_fn(self._file_path)
            # iterate through and parse through the cache_read_order
            slices = []
            # print(self.cache_read_order)
            for i in range(len(self.cache_read_order)):
                axis = self.cache_read_order[i]
                if axis == Axes.ZPLANE:
                    slices.append(self._zplane)
                elif axis == Axes.CH:
                    if self.is_aux:
                        slices.append(self.is_aux)
                    else:
                        slices.append(self._ch)
                else:
                    slices.append(slice(0, img.shape[i]))
            slices = tuple(slices)
            # print(slices)
            img = img[slices]
            img = np.squeeze(img)
            return img
        except IndexError as e:
            print(
                "\t{}\nshape: {}\tcache read: {}\nzpl: {} ch: {} is aux: {}\nwith error {}".format(
                    self._file_path,
                    img.shape,
                    self.cache_read_order,
                    self._zplane,
                    self._ch,
                    self.is_aux,
                    e,
                )
            )
            return np.zeros((2048, 2048))


class PrimaryTileFetcher(TileFetcher):
    """
    Generic TileFetcher implementation for FISH experiments.
    """

    def __init__(
        self,
        input_dir: str,
        file_format: str = "",
        file_vars: str = "",
        cache_read_order: list = [],
        zplane_offset: int = 0,
        fov_offset: int = 0,
        round_offset: int = 0,
        channel_offset: int = 0,
        locs: List[Mapping[Axes, float]] = None,
        voxel: Mapping[Axes, float] = None,
        shape: Mapping[Axes, int] = None,
    ) -> None:
        """
        Implement a TileFetcher for a single Field of View.

        Parameters
        ----------
        input_dir: str
            Relative root location of image files.
        file_format: str
            String format for individual image files.  Appended to input_dir.
            Each "{}" within this string will be replaced the tile-specific values, as specified in the order of "file_vars"
        file_vars: str
            Variables to insert in file_format.  The following values are accepted:
                - channel
                - offset_channel (channel + channel_offset)
                - round
                - offset_round (round + round_offset)
                - fov
                - offset_fov (fov + fov_offset)
                - zplane
                - offset_zplane (zplane + zplane_offset)
        cache_read_order: list
            Description of the order of the axes of the images. Each item in the list is one dimension in the image.
            The following strings will be converted to Axes objects and will be parsed based on the instance variables of the tile:
                -Z -> Axes.ZPLANE
                -CH -> Axes.CH
            All ofther values will be treated as an axis where the full contents will be read for each individual tile. (in pratice, this should only be Axes.X and Axes.Y)
        zplane_offset: int
            Integer to be added to zplane names when looking for external file names, equal to the number of the first index.
        fov_offset: int
            Integer to be added to fov names when looking for external file names, equal to the number of the first index.
        round_offset: int
            Integer to be added to the round count when looking for external file names, equal to the number of the first index.
        channel_offset: int
            Integer to be added to channels when looking for external file names, equal to the number of the first index.

            The following parameters are optional, and are only used if .coordinates() is called.  They may be used further downstream in analysis, in particular if there are multiple FOVs.
        For further details, see FISHTile.coordinates()

        locs: List[Mapping[Axes, float]]
            Each list item refers to the corresponding indexed fov. The start location of the image, mapped to the corresponding Axes object (X, Y, or ZPLANE).
        voxel: Mapping[Axes, float]
            The size of each image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        shape: Mapping[Axes, int]
            The offset for the size of the image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        """
        self.file_format = file_format
        self.file_vars = file_vars
        self.input_dir = input_dir
        self.cache_read_order = cache_read_order
        self.zplane_offset = zplane_offset
        self.fov_offset = fov_offset
        self.round_offset = round_offset
        self.channel_offset = channel_offset
        self.locs = locs
        self.voxel = voxel
        self.img_shape = shape

    def get_tile(
        self, fov_id: int, round_label: int, ch_label: int, zplane_label: int
    ) -> FISHTile:
        """
        Extracts 2-d data from a multi-page TIFF containing images as defined by parameters passed at initialization. file_format, file_vars, input_dir, and cache_read_order are combined with tile-specific indicies to determine file name.
        Note: indecies with offsets can be specified by self.file_vars, but the index used when loading the tiff will always be the 0-indexed.

        Parameters
        ----------
        fov_id : int
            Number of the fov, before addition by fov_offset
        round_label : int
            Selects the imaging round from within the loaded TIFF file.
        ch : int
            Selects the channel from within the loaded TIFF file.
        zplane : int
            Selects the z-plane from within the loaded TIFF file.

        Returns
        -------
        FISHTile :
            FISH subclass of FetchedTile
        """
        print(
            "fov: {} round: {} channel: {} zplane: {}".format(
                fov_id, round_label, ch_label, zplane_label
            )
        )
        varTable = {
            "channel": ch_label,
            "offset_channel": ch_label + self.channel_offset,
            "round": round_label,
            "offset_round": round_label + self.round_offset,
            "fov": fov_id,
            "offset_fov": fov_id + self.fov_offset,
            "zplane": zplane_label,
            "offset_zplane": zplane_label + self.zplane_offset,
        }
        file_path = os.path.join(
            self.input_dir, self.file_format.format(*[varTable[arg] for arg in self.file_vars])
        )
        if self.locs:
            return FISHTile(
                file_path,
                zplane_label,
                ch_label,
                round_label,
                False,
                self.cache_read_order,
                self.locs[fov_id],
                self.voxel,
                self.img_shape,
            )
        else:
            return FISHTile(
                file_path, zplane_label, ch_label, round_label, False, self.cache_read_order
            )


class AuxTileFetcher(TileFetcher):
    """
    Alternate version of PrimaryTileFetcher for non-primary images.
    Primary difference relative to FISHTileFetcher: expects a single imaging channel.
    """

    def __init__(
        self,
        input_dir: str,
        file_format: str = "",
        file_vars: str = "",
        cache_read_order: list = [],
        zplane_offset: int = 0,
        fov_offset: int = 0,
        round_offset: int = 0,
        channel_offset: int = 0,
        fixed_channel: int = 0,
        locs: List[Mapping[Axes, float]] = None,
        voxel: Mapping[Axes, float] = None,
        shape: Mapping[Axes, int] = None,
    ) -> None:
        """
        Implement a TileFetcher for a single Field of View with a single channel.

        Parameters
        ----------
        input_dir: str
            Relative root location of image files.
        file_format: str
            String format for individual image files.  Appended to input_dir.
            Each "{}" within this string will be replaced the tile-specific values, as specified in the order of "file_vars"
        file_vars: str
            Variables to insert in file_format.  The following values are accepted:
                - channel
                - offset_channel (channel + channel_offset)
                - round
                - offset_round (round + round_offset)
                - fov
                - offset_fov (fov + fov_offset)
                - zplane
                - offset_zplane (zplane + zplane_offset)
        cache_read_order: list
            Description of the order of the axes of the images. Each item in the list is one dimension in the image.
            The following strings will be converted to Axes objects and will be parsed based on the instance variables of the tile:
                -Z -> Axes.ZPLANE
                -CH -> Axes.CH
            All ofther values will be treated as an axis where the full contents will be read for each individual tile. (in pratice, this should only be Axes.X and Axes.Y)
        fov_offset: int
            Integer to be added to fov names when looking for external file names, equal to the number of the first index.
        round_offset: int
            Integer to be added to the round count when looking for external file names, equal to the number of the first index.
        channel_offset: int
            Integer to be added to channels when looking for external file names, equal to the number of the first index.
        fixed_channel: int
            The single channel to look at for this tile.

            The following parameters are optional, and are only used if .coordinates() is called.  They may be used further downstream in analysis, in particular if there are multiple FOVs.
        For further details, see FISHTile.coordinates()

        locs: List[Mapping[Axes, float]]
            Each list item refers to the fov of the same index. The start location of the image, mapped to the corresponding Axes object (X, Y, or ZPLANE)
        voxel: Mapping[Axes, float]
            The size of each image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        shape: Mapping[Axes, float]
            The offset for the size of the image, mapped to the corresponding Axes object (X, Y, ZPLANE)
        """
        self.file_format = file_format
        self.file_vars = file_vars.split(";")
        self.input_dir = input_dir
        self.cache_read_order = cache_read_order
        self.zplane_offset = zplane_offset
        self.fov_offset = fov_offset
        self.round_offset = round_offset
        self.channel_offset = channel_offset
        self.fixed_channel = fixed_channel
        self.locs = locs
        self.voxel = voxel
        self.img_shape = shape

    def get_tile(
        self, fov_id: int, round_label: int, ch_label: int, zplane_label: int
    ) -> FISHTile:
        """
        Extracts 2-d data from a multi-page TIFF containing images as defined by parameters passed at initialization. file_format, file_vars, input_dir, and cache_read_order are combined with tile-specific indicies to determine file name.
        Note: indecies with offsets can be specified by self.file_vars, but the index used when loading the tiff will always be the 0-indexed.

        Parameters
        ----------
        fov_id : int
            Number of the fov, before addition by fov_offset
        round_label : int
            Selects the imaging round from within the loaded TIFF file.
        ch : int
            Selects the channel from within the loaded TIFF file.
        zplane : int
            Selects the z-plane from within the loaded TIFF file.

        Returns
        -------
        FISHTile :
            FISH subclass of FetchedTile
        """
        varTable = {
            "channel": ch_label,
            "offset_channel": ch_label + self.channel_offset,
            "round": round_label,
            "offset_round": round_label + self.round_offset,
            "fov": fov_id,
            "offset_fov": fov_id + self.fov_offset,
            "zplane": zplane_label,
            "offset_zplane": zplane_label + self.zplane_offset,
        }
        file_path = os.path.join(
            self.input_dir, self.file_format.format(*[varTable[arg] for arg in self.file_vars])
        )
        if self.locs:
            return FISHTile(
                file_path,
                zplane_label,
                self.fixed_channel,
                round_label,
                self.fixed_channel,
                self.cache_read_order,
                self.locs[fov_id],
                self.voxel,
                self.img_shape,
            )  # CHANNEL ID IS FIXED
        else:
            return FISHTile(
                file_path,
                zplane_label,
                self.fixed_channel,
                round_label,
                self.fixed_channel,
                self.cache_read_order,
            )  # CHANNEL ID IS FIXED


def parse_codebook(codebook_csv: str) -> Codebook:
    """Parses a codebook csv file provided by SeqFISH developers.

    Parameters
    ----------
    codebook_csv : str
        The codebook file is expected to contain a matrix whose rows are barcodes and whose columns
        are imaging rounds. Column IDs are expected to be sequential, and round identifiers (roman
        numerals) are replaced by integer IDs.

    Returns
    -------
    Codebook :
        Codebook object in SpaceTx format.
    """
    csv: pd.DataFrame = pd.read_csv(codebook_csv, index_col=0)
    genes = csv.index.values
    data_raw = csv.values
    rounds = csv.shape[1]
    channels = data_raw.max()

    # convert data_raw -> data, where data is genes x channels x rounds
    data = np.zeros((len(data_raw), rounds, channels))
    for b in range(len(data_raw)):
        for i in range(len(data_raw[b])):
            if data_raw[b][i] != 0:
                data[b][i][data_raw[b][i] - 1] = 1

    return Codebook.from_numpy(genes, rounds, channels, data)


def cli(
    input_dir: str,
    output_dir: str,
    file_format: str,
    file_vars: list,
    cache_read_order: list,
    counts: dict,
    aux_names: list = [],
    aux_file_formats: list = [],
    aux_file_vars: list = [],
    aux_cache_read_order: list = [],
    aux_fixed_channel: list = [],
    locs: List[Mapping[Axes, float]] = None,
    shape: Mapping[Axes, int] = None,
    voxel: Mapping[Axes, float] = None,
) -> int:
    """CLI entrypoint for spaceTx format construction for SeqFISH data

    Parameters
    ----------
    input_dir : str
        Directory containing folders for fovs.
    output_dir : str
        Directory containing output files. Will be created if it does not exist.
    file_format: str
        String format for individual image files of primary view.  Appended to input_dir.
        Each "{}" within this string will be replaced the tile-specific values, as specified in the order of "file_vars"
    file_vars: list
        Variables to insert in file_format.  The following values are accepted:
            - channel
            - offset_channel (channel + channel_offset)
            - round
            - offset_round (round + round_offset)
            - fov
            - offset_fov (fov + fov_offset)
            - zplane
            - offset_zplane (zplane + zplane_offset)
    cache_read_order: list
        Description of the order of the axes of the images. Each item in the list is one dimension in the image.
    counts: dict
        Dict with the counts for each dimension of the data. Expects values that correspond
        to keys of ["rounds","channels","zplanes","fovs"]
    aux_names: list
        A list containing the names of any auxilliary tile views.
    aux_file_formats: list
        The same as file_format, but for each individual aux view. Items within each list entry are semicolon (;) delimited.
    aux_file_vars: list
        The same as file_vars, but for each individual aux view. Items within each list entry are semicolon (;) delimited.
    aux_cache_read_order: list
        The same as cache_read_order, but for each individual aux view. Items within each list entry are semicolon (;) delimited.
    aux_fixed_channel: list
        The channel for each aux view to look at in their respective image files.
    locs: List[Mapping[Axes, float]]
        Each list item refers to the fov of the same index. The start location of the image, mapped to the corresponding Axes object (X, Y, or ZPLANE)
    shape: Mapping[Axes, int]
        The offset for the size of the image, mapped to the corresponding Axes object (X, Y, ZPLANE)
    voxel: Mapping[Axes, float]
        The size of each image, mapped to the corresponding Axes object (X, Y, ZPLANE)

    Returns
    -------
    int :
        Returns 0 if successful
    """

    t0 = time()

    os.makedirs(output_dir, exist_ok=True)

    reportFile = os.path.join(
        output_dir, datetime.now().strftime("%Y-%d-%m_%H:%M_TXconversion.log")
    )
    sys.stdout = open(reportFile, "w")

    image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: counts["rounds"],
        Axes.CH: counts["channels"],
        Axes.ZPLANE: counts["zplanes"],
    }

    aux_image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: counts["rounds"],
        Axes.CH: 1,
        Axes.ZPLANE: counts["zplanes"],
    }

    # file_format = "HybCycle_{}/MMStack_Pos{}.ome.tif"
    # file_vars = ["offset_round", "offset_fov"]

    cache_read_order_formatted = []
    for item in cache_read_order:
        if item == "Z":
            cache_read_order_formatted.append(Axes.ZPLANE)
        elif item == "CH":
            cache_read_order_formatted.append(Axes.CH)
        else:
            cache_read_order_formatted.append("other")

    primary_tile_fetcher = PrimaryTileFetcher(
        os.path.expanduser(input_dir),
        file_format,
        file_vars,
        cache_read_order_formatted,
        counts["zplane_offset"],
        counts["fov_offset"],
        counts["round_offset"],
        counts["channel_offset"],
        locs,
        shape,
        voxel,
    )

    aux_name_to_dimensions = {}
    aux_tile_fetcher = {}
    if aux_names:
        for i in range(len(aux_names)):
            name = aux_names[i]
            aux_name_to_dimensions[name] = aux_image_dimensions
            aux_cache_read_order_raw = aux_cache_read_order[i].split(";")
            aux_cache_read_order_formatted = []
            for item in aux_cache_read_order_raw:
                if item == "Z":
                    aux_cache_read_order_formatted.append(Axes.ZPLANE)
                elif item == "CH":
                    aux_cache_read_order_formatted.append(Axes.CH)
                else:
                    aux_cache_read_order_formatted.append("other")

            aux_tile_fetcher[name] = AuxTileFetcher(
                os.path.expanduser(input_dir),
                aux_file_formats[i],
                aux_file_vars[i],
                aux_cache_read_order_formatted,
                counts["zplane_offset"],
                counts["fov_offset"],
                counts["round_offset"],
                counts["channel_offset"],
                aux_fixed_channel[i],
                locs,
                shape,
                voxel,
            )
    # aux_tile_fetcher = {"DAPI": AuxTileFetcher(os.path.expanduser(input_dir), file_format, file_vars, counts["fov_offset"], counts["round_offset"],3)}
    # aux_name_to_dimensions = {"DAPI": aux_image_dimensions}

    t1 = time()
    print("Elapsed time to make experiment", t1 - t0)

    write_experiment_json(
        path=output_dir,
        fov_count=counts["fovs"],
        aux_tile_fetcher=aux_tile_fetcher,
        primary_tile_fetcher=primary_tile_fetcher,
        primary_image_dimensions=image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        tile_format=ImageFormat.TIFF,
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE),
    )

    os.remove(output_dir + "/codebook.json")

    t2 = time()
    print("Elapsed time for .json manipulation", t2 - t1)
    print("Operation complete, total elapsed time", t2 - t0)

    sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--codebook-csv", type=Path)
    p.add_argument("--codebook-json", type=Path)
    p.add_argument("--round-count", type=int)
    p.add_argument("--zplane-count", type=int)
    p.add_argument("--channel-count", type=int)
    p.add_argument("--fov-count", type=int)
    p.add_argument("--zplane-offset", type=int, default=0)
    p.add_argument("--round-offset", type=int, default=0)
    p.add_argument("--fov-offset", type=int, default=0)
    p.add_argument("--channel-offset", type=int, default=0)
    p.add_argument("--file-format", type=str)
    p.add_argument("--file-vars", nargs="+")
    p.add_argument("--cache-read-order", nargs="+")
    p.add_argument("--aux-names", nargs="+")
    p.add_argument("--aux-file-formats", nargs="+")
    p.add_argument("--aux-file-vars", nargs="+")
    p.add_argument("--aux-fixed-channel", type=int, nargs="+")
    p.add_argument("--aux-cache-read-order", nargs="+")
    p.add_argument("--x-pos-locs", type=str, nargs="?")
    p.add_argument("--x-pos-shape", type=int, nargs="?")
    p.add_argument("--x-pos-voxel", type=float, nargs="?")
    p.add_argument("--y-pos-locs", type=str, nargs="?")
    p.add_argument("--y-pos-shape", type=int, nargs="?")
    p.add_argument("--y-pos-voxel", type=float, nargs="?")
    p.add_argument("--z-pos-locs", type=str, nargs="?")
    p.add_argument("--z-pos-shape", type=int, nargs="?")
    p.add_argument("--z-pos-voxel", type=float, nargs="?")

    args = p.parse_args()

    if (
        len(
            {
                len(args.aux_names) if args.aux_names else 0,
                len(args.aux_file_formats) if args.aux_file_formats else 0,
                len(args.aux_file_vars) if args.aux_file_vars else 0,
                len(args.aux_fixed_channel) if args.aux_fixed_channel else 0,
                len(args.aux_cache_read_order) if args.aux_cache_read_order else 0,
            }
        )
        > 1
    ):
        print(
            args.aux_names,
            args.aux_file_formats,
            args.aux_file_vars,
            args.aux_fixed_channel,
            args.aux_cache_read_order,
        )
        raise Exception("Dimensions of all aux parameters must match.")

    output_dir = "3_tx_converted/"

    # parse loc info
    locs = []
    shape = {}
    voxel = {}

    # cwl spec says that if one of these dims is defined, they all must be.
    if args.x_pos_locs:
        # sanity check that length matches number of fovs:
        axis = [Axes.X, Axes.Y, Axes.ZPLANE]
        pos_locs = {}
        pos_locs[Axes.X] = args.x_pos_locs
        pos_locs[Axes.Y] = args.y_pos_locs
        pos_locs[Axes.ZPLANE] = args.z_pos_locs

        for ax in axis:
            if pos_locs[ax]:
                pos_locs[ax] = pos_locs[ax].split(",")
                if len(pos_locs[ax]) != args.fov_count:
                    raise Exception("Specified FOV locations must match fov_count.")

        for i in range(args.fov_count):
            this_loc = {}
            for ax in axis:
                if pos_locs[ax]:
                    this_loc[ax] = float(pos_locs[ax][i])
            locs.append(this_loc)

        shape[Axes.X] = args.x_pos_shape
        shape[Axes.Y] = args.y_pos_shape
        shape[Axes.ZPLANE] = args.z_pos_shape

        voxel[Axes.X] = args.x_pos_voxel
        voxel[Axes.Y] = args.y_pos_voxel
        voxel[Axes.ZPLANE] = args.z_pos_voxel

    counts = {
        "rounds": args.round_count,
        "channels": args.channel_count,
        "zplanes": args.zplane_count,
        "fovs": args.fov_count,
        "zplane_offset": args.zplane_offset,
        "round_offset": args.round_offset,
        "fov_offset": args.fov_offset,
        "channel_offset": args.channel_offset,
    }
    cli(
        args.input_dir,
        output_dir,
        args.file_format,
        args.file_vars,
        args.cache_read_order,
        counts,
        args.aux_names,
        args.aux_file_formats,
        args.aux_file_vars,
        args.aux_cache_read_order,
        args.aux_fixed_channel,
        locs,
        shape,
        voxel,
    )

    # Note: this must trigger AFTER write_experiment_json, as it will clobber the codebook with
    # a placeholder.
    if args.codebook_csv:
        codebook = parse_codebook(args.codebook_csv)
        codebook.to_json(output_dir + "codebook.json")
    if args.codebook_json:
        copyfile(args.codebook_json, output_dir + "codebook.json")
