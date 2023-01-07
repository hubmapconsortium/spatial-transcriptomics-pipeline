#!/usr/bin/env python

import functools
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Mapping, Union

import numpy as np
import pandas as pd
import skimage.io
import xarray as xr
from slicedimage import ImageFormat
from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue

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
        cache_read_order: list,
        locs: Mapping[Axes, float] = None,
        shape: Mapping[Axes, int] = None,
        voxel: Mapping[Axes, float] = None,
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
        cache_read_order: list
            Description of the order of the axes of the images. Each item in the list is one dimension in the image.
            The following strings will be converted to Axes objects and will be parsed based on the instance variables of the tile:
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
                    (self.locs[Axes.ZPLANE] + self._zplane) * self.voxel[Axes.ZPLANE],
                    (self.locs[Axes.ZPLANE] + self._zplane + 1) * self.voxel[Axes.ZPLANE],
                ),
            }
        else:
            # no defined location, return dummy
            return {
                Coordinates.X: (0.0, 1.0),
                Coordinates.Y: (0.0, 1.0),
                Coordinates.Z: (0.0, 1.0),
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
                "\t{}\nshape: {}\tcache read: {}\nzpl: {} ch: {}\nwith error {}".format(
                    self._file_path,
                    img.shape,
                    self.cache_read_order,
                    self._zplane,
                    self._ch,
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
        shape: Mapping[Axes, int] = None,
        voxel: Mapping[Axes, float] = None,
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
                self.cache_read_order,
                self.locs[fov_id],
                self.img_shape,
                self.voxel,
            )
        else:
            return FISHTile(file_path, zplane_label, ch_label, round_label, self.cache_read_order)


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
        cache_read_order: List[str] = [],
        channel_slope: float = 0,
        channel_intercept: int = 0,
        zplane_offset: int = 0,
        fov_offset: int = 0,
        round_offset: int = 0,
        channel_offset: int = 0,
        locs: List[Mapping[Axes, float]] = None,
        shape: Mapping[Axes, int] = None,
        voxel: Mapping[Axes, float] = None,
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
                - input_channel (see img channel below)
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
        channel_slope: float
            The slope for converting 0-indexed channel IDs to the channel ID within the image. Combined with the below variable with the following formula:

                (img channel) = int(slope*channel ID) + intercept

        channel_intercept: int
            The intercept for converting 0-index channel IDs to the channel ID within the image.
        zplane_offset: int
            Integer to be added to zplane names when looking for external files names, equal to the number of the first index.
        fov_offset: int
            Integer to be added to fov names when looking for external file names, equal to the number of the first index.
        round_offset: int
            Integer to be added to the round count when looking for external file names, equal to the number of the first index.
        channel_offset: int
            Integer to be added to channels when looking for external file names, equal to the number of the first index.
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
        self.slope = float(channel_slope)
        self.intercept = int(channel_intercept)
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
        varTable = {
            "channel": ch_label,
            "offset_channel": ch_label + self.channel_offset,
            "input_channel": int(int(self.slope * ch_label) + self.intercept),
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

        print(
            "fov: {} round: {} channel: {} zplane: {}".format(
                fov_id, round_label, ch_label, zplane_label
            )
        )

        print("aux view channel: int({} * {}) + {}".format(self.slope, ch_label, self.intercept))
        ch_label_adj = int(self.slope * int(ch_label)) + self.intercept

        if self.locs:
            return FISHTile(
                file_path,
                zplane_label,
                ch_label_adj,
                round_label,
                self.cache_read_order,
                self.locs[fov_id],
                self.img_shape,
                self.voxel,
            )
        else:
            return FISHTile(
                file_path,
                zplane_label,
                ch_label_adj,
                round_label,
                self.cache_read_order,
            )


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
    cache_read_order: List[str],
    counts: dict,
    aux_names: List[str] = [],
    aux_file_formats: List[str] = [],
    aux_file_vars: List[List[str]] = [],
    aux_cache_read_order: List[str] = [],
    aux_channel_count: List[int] = [],
    aux_channel_slope: List[float] = [],
    aux_channel_intercept: List[int] = [],
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
    aux_channel_count: list
        The total number of channels per aux view.
    aux_channel_slope: list
        The slope for converting 0-indexed channel IDs to the channel ID within the image.
    aux_channel_intercept: list
        The intercept for converting 0-index channel IDs to the channel ID within the image.
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

    reportFile = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M_TXconversion.log"))
    sys.stdout = open(reportFile, "w")

    image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: counts["rounds"],
        Axes.CH: counts["channels"],
        Axes.ZPLANE: counts["zplanes"],
    }

    cache_read_order_formatted = []
    for item in cache_read_order:
        if item.lower() == "z":
            cache_read_order_formatted.append(Axes.ZPLANE)
        elif item.lower() == "ch":
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
            aux_image_dimensions: Mapping[Union[str, Axes], int] = {
                Axes.ROUND: counts["rounds"],
                Axes.CH: int(aux_channel_count[i]),
                Axes.ZPLANE: counts["zplanes"],
            }
            aux_name_to_dimensions[name] = aux_image_dimensions
            aux_cache_read_order_raw = aux_cache_read_order[i].split(";")
            aux_cache_read_order_formatted = []
            for item in aux_cache_read_order_raw:
                if item.lower() == "z":
                    aux_cache_read_order_formatted.append(Axes.ZPLANE)
                elif item.lower() == "ch":
                    aux_cache_read_order_formatted.append(Axes.CH)
                else:
                    aux_cache_read_order_formatted.append("other")

            aux_tile_fetcher[name] = AuxTileFetcher(
                os.path.expanduser(input_dir),
                aux_file_formats[i],
                aux_file_vars[i],
                aux_cache_read_order_formatted,
                aux_channel_slope[i],
                aux_channel_intercept[i],
                counts["zplane_offset"],
                counts["fov_offset"],
                counts["round_offset"],
                counts["channel_offset"],
                locs,
                shape,
                voxel,
            )
    # aux_tile_fetcher = {"DAPI": AuxTileFetcher(os.path.expanduser(input_dir), file_format, file_vars, counts["fov_offset"], counts["round_offset"],3)}
    # aux_name_to_dimensions = {"DAPI": aux_image_dimensions}

    t1 = time()
    print("Elapsed time to make experiment", t1 - t0)

    print(image_dimensions)
    print(primary_tile_fetcher)
    # print(aux_name_to_dimensions)
    # print(aux_tile_fetcher)

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


def barcodeConv(lis, chs):
    barcode = np.zeros((len(lis), chs))
    for i in range(len(lis)):
        barcode[i][lis[i]] = 1
    return barcode


def incrBarcode(lis, chs):
    currInd = len(lis) - 1
    lis[currInd] += 1
    while lis[currInd] == chs:
        lis[currInd] = 0
        currInd -= 1
        lis[currInd] += 1
    return lis


def _view_row_as_element(array: np.ndarray) -> np.ndarray:
    nrows, ncols = array.shape
    dtype = {"names": ["f{}".format(i) for i in range(ncols)], "formats": ncols * [array.dtype]}
    return array.view(dtype)


def blank_codebook(real_codebook, num_blanks):
    """
    From a codebook of real codes, creates a codebook of those original codes plus a set of blank codes that
    follow the hamming distance rule of the codebook (auto-detects hamming distance of real codes). Resulting
    codebook will have num_blanks blank codes in addition to all the original real codes. If num_blanks is
    greater than the total number of blank codes found then all blanks will be added.
    """

    # Extract dimensions and create empty xarray for barcodes
    roundsN = len(real_codebook["r"])
    channelsN = len(real_codebook["c"])
    allCombo = xr.zeros_like(
        xr.DataArray(
            np.zeros((channelsN**roundsN, roundsN, channelsN)), dims=["target", "r", "c"]
        )
    )

    # Check that codebook is one-hot
    for row in real_codebook[0].data:
        row_sum = sum(row == 0)
        if row_sum == channelsN or row_sum == 0:
            raise ValueError(
                "Error: blank code generation only built for one-hot codebooks (codebooks where\
                              each round has exactly one active channel)."
            )

    # Calculate hamming distance rule in codebook
    codes = real_codebook.argmax(Axes.CH.value).data
    hamming_distance = 100
    for i in range(len(codes)):
        tmp_codes = np.append(codes[:i], codes[i + 1 :], axis=0)
        min_ham = np.min(roundsN - (codes[i] == tmp_codes).sum(axis=1))
        if min_ham < hamming_distance:
            hamming_distance = min_ham

    # Start from set of all possible codes
    barcode = [0] * roundsN
    for i in range(np.shape(allCombo)[0]):
        allCombo[i] = barcodeConv(barcode, channelsN)
        barcode = incrBarcode(barcode, channelsN)

    # If hamming distance is 0, a code is duplicated and we can't accurate find the hamming distance rule
    if hamming_distance == 0:
        raise ValueError("Error: codebook contains duplicate codes")

    # If hamming distance is 1, only need to drop real codes from list of all possible codes to get all blank codes
    elif hamming_distance == 1:
        keep = []
        for i, combo in enumerate(allCombo):
            if roundsN * channelsN not in (combo.data == real_codebook.data).sum(axis=1).sum(
                axis=1
            ):
                keep.append(i)
        blanks = allCombo[keep].data

    # if hamming distance is 2, need to drop real codes and those with hamming distance equal to 1
    elif hamming_distance == 2:

        # Remove codes that have hamming distance <= 1 to any code in the real codebook
        cb_codes = real_codebook.argmax(Axes.CH.value)
        drop_cb_codes = {}
        rounds = [True] * roundsN
        for r in range(roundsN):
            rounds[r] = False
            drop_codes = cb_codes.sel(r=rounds)
            drop_codes.values = np.ascontiguousarray(drop_codes.values)
            drop_codes = _view_row_as_element(drop_codes.values.reshape(drop_codes.shape[0], -1))
            drop_cb_codes[r] = drop_codes
            rounds[r] = True

        drop_combos = {}
        rounds = [True] * roundsN
        for r in range(roundsN):
            rounds[r] = False
            combo_codes = allCombo.argmax(Axes.CH.value)
            combo_codes = combo_codes.sel(r=rounds)
            combo_codes.values = np.ascontiguousarray(combo_codes.values)
            combo_codes = _view_row_as_element(
                combo_codes.values.reshape(combo_codes.shape[0], -1)
            )
            drop_combos[r] = combo_codes
            rounds[r] = True
        combo_codes = allCombo.argmax(Axes.CH.value)
        combo_codes.values = np.ascontiguousarray(combo_codes.values)
        combo_codes = _view_row_as_element(combo_codes.values.reshape(combo_codes.shape[0], -1))

        drop = []
        for i in range(len(combo_codes)):
            for r in range(roundsN):
                if np.any(drop_combos[r][i] == drop_cb_codes[r]):
                    drop.append(i)
                    break

        drop = set(drop)
        allCombo = allCombo[[x for x in range(len(combo_codes)) if x not in drop]]

        # Find set of codes that all have hamming distance of more than 1 to each other

        # Creates set of codebooks each with a different dropped round, can determine if two codes are 1 or fewer hamming
        # distances from each other by seeing if they match exactly when the same round is dropped for each code
        drop_combos = {}
        rounds = [True] * roundsN
        for r in range(roundsN):
            rounds[r] = False
            combo_codes = allCombo.argmax(Axes.CH.value)
            combo_codes = combo_codes.sel(r=rounds)
            combo_codes.values = np.ascontiguousarray(combo_codes.values)
            combo_codes = _view_row_as_element(
                combo_codes.values.reshape(combo_codes.shape[0], -1)
            )
            drop_combos[r] = combo_codes
            rounds[r] = True
        combo_codes = allCombo.argmax(Axes.CH.value)
        combo_codes.values = np.ascontiguousarray(combo_codes.values)
        combo_codes = _view_row_as_element(combo_codes.values.reshape(combo_codes.shape[0], -1))

        i = 0
        while i < len(combo_codes):
            drop = set()
            for r in range(roundsN):
                drop.update([x for x in np.nonzero(drop_combos[r][i] == drop_combos[r])[0]])
            drop.remove(i)
            inds = [x for x in range(len(combo_codes)) if x not in drop]
            combo_codes = combo_codes[inds]
            for r in range(roundsN):
                drop_combos[r] = drop_combos[r][inds]
            i += 1

        # Create Codebook object with blanks
        blanks = np.zeros((len(combo_codes), roundsN, channelsN))
        for i, code in enumerate(combo_codes):
            for j, x in enumerate(code[0]):
                blanks[i][j][x] = 1

    else:
        raise ValueError(
            "Error: can only generate blank codes for a one-hot codebook with a minimum hamming\
                          distance of 1 or 2"
        )

    blank_codebook = Codebook.from_numpy(
        code_names=["blank" + str(x) for x in range(len(blanks))],
        n_round=roundsN,
        n_channel=channelsN,
        data=blanks,
    )

    # Combine correct number of blank codes with real codebook and return combined codebook
    if num_blanks > len(blanks):
        num_blanks = len(blanks)
    rand_sample = random.sample(range(len(blanks)), num_blanks)
    combined = xr.concat([real_codebook, blank_codebook[rand_sample]], "target")

    return combined


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
    p.add_argument("--aux-names", nargs="+", const=None)
    p.add_argument("--aux-file-formats", nargs="+", const=None)
    p.add_argument("--aux-file-vars", nargs="+", const=None)
    p.add_argument("--aux-cache-read-order", nargs="+", const=None)
    p.add_argument("--aux-channel-count", nargs="+", const=None)
    p.add_argument("--aux-channel-slope", nargs="+", const=None)
    p.add_argument("--aux-channel-intercept", nargs="+", const=None)
    p.add_argument("--x-pos-locs", type=str, nargs="?")
    p.add_argument("--x-pos-shape", type=int, nargs="?")
    p.add_argument("--x-pos-voxel", type=float, nargs="?")
    p.add_argument("--y-pos-locs", type=str, nargs="?")
    p.add_argument("--y-pos-shape", type=int, nargs="?")
    p.add_argument("--y-pos-voxel", type=float, nargs="?")
    p.add_argument("--z-pos-locs", type=str, nargs="?")
    p.add_argument("--z-pos-shape", type=int, nargs="?")
    p.add_argument("--z-pos-voxel", type=float, nargs="?")
    p.add_argument("--add-blanks", dest="add_blanks", action="store_true")
    p.set_defaults(add_blanks=False)

    args = p.parse_args()

    aux_lens = []
    aux_vars = [
        args.aux_file_formats,
        args.aux_file_formats,
        args.aux_file_vars,
        args.aux_names,
        args.aux_cache_read_order,
        args.aux_channel_count,
        args.aux_channel_slope,
        args.aux_channel_intercept,
    ]

    for item in aux_vars:
        if isinstance(item, list):
            aux_lens.append(len(item))
        elif item is not None:
            aux_lens.append(1)
        else:
            aux_lens.append(0)

    if len(set(aux_lens)) > 1:
        print(aux_vars)
        print(aux_lens)
        raise Exception("Dimensions of all aux parameters must match.")

    output_dir = "2_tx_converted/"

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
                    raise Exception("Number of specified FOV locations must match fov_count.")

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
        args.aux_channel_count,
        args.aux_channel_slope,
        args.aux_channel_intercept,
        locs,
        shape,
        voxel,
    )

    # Note: this must trigger AFTER write_experiment_json, as it will clobber the codebook with
    # a placeholder.
    if args.codebook_csv:
        codebook = parse_codebook(args.codebook_csv)
    if args.codebook_json:
        codebook = Codebook.open_json(str(args.codebook_json))
    if args.add_blanks:
        codebook = blank_codebook(codebook, len(codebook))
    codebook.to_json(output_dir + "codebook.json")
