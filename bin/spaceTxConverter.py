#!/usr/bin/env python

from argparse import ArgumentParser
import functools
from typing import Mapping, Union
from pathlib import Path
import os

import click
import numpy as np
import pandas as pd
import skimage.io
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features
import starfish

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
            shape: Mapping[Axes, int] = None
    ):
        self._file_path = file_path
        self._zplane = ch
        self._ch = zplane
        self._rnd = rnd
        self.is_aux = is_aux
        self.cache_read_order = cache_read_order
        if locs:
            self.locs = locs
        if voxel:
            self.voxel = voxel
        if shape: 
            self.shape = shape
        self.coord_def = (locs and voxel and shape)
        #print("zpl {};ch {}".format(zplane,ch))

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
        """Returns coordinate values based on values passed at initialization"""
        if self.coord_def:
            # return value based on passed parameters
            return {
                    Coordinates.X: (self.locs[Axes.X]*self.voxel[Axes.X], (self.locs[Axes.X]+self.shape[Axes.X])*self.voxel[Axes.X]),
                    Coordinates.Y: (self.locs[Axes.Y]*self.voxel[Axes.Y], (self.locs[Axes.Y]+self.shape[Axes.Y])*self.voxel[Axes.Y]),
                    Coordinates.Z: (self.locs[Axes.Z]*self.voxel[Axes.Z], (self.locs[Axes.Z]+self.shape[Axes.Z])*self.voxel[Axes.Z])
                }
        else:
            # no defined location, retrun dummy
            return {
                    Coordinates.X: (0., 1.),
                    Coordinates.Y: (0., 1.),
                    Coordinates.Z: (0., 0.1)
                }

    @property
    def format(self) -> ImageFormat:
        """Image Format for SeqFISH data is TIFF"""
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        """squeeze dims of img based on pre-defined cache read order"""
        #print(cached_read_fn(self._file_path).shape)
        try:
            img = cached_read_fn(self._file_path)
            # iterate through and parse through the cache_read_order
            slices = []
            #print(self.cache_read_order)
            for i  in range(len(self.cache_read_order)):
                axis = self.cache_read_order[i]
                if axis == Axes.ZPLANE:
                    slices.append(self._zplane)
                elif axis == Axes.CH:
                    if self.is_aux:
                        slices.append(self.is_aux)
                    else:
                        slices.append(self._ch)
                else:
                    slices.append(slice(0,img.shape[i]))
            slices = tuple(slices)
            #print(slices)
            img = img[slices]
            img = np.squeeze(img)
            return img
        except IndexError as e:
            print("\t{}\nshape: {}\tcache read: {}\nzpl: {} ch: {} is aux: {}\nwith error {}".format(self._file_path, img.shape,self.cache_read_order, self._zplane, self._ch,self.is_aux, e))
            return np.zeros((2048,2048))
            
class PrimaryTileFetcher(TileFetcher):

    def __init__(self, input_dir: str, file_format: str="", file_vars: str="", cache_read_order: list=[], fov_offset: int=0, round_offset: int=0, channel_offset: int=0) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.file_format = file_format
        self.file_vars = file_vars
        self.input_dir = input_dir
        self.cache_read_order = cache_read_order
        self.fov_offset = fov_offset
        self.round_offset = round_offset
        self.channel_offset = channel_offset
        

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FISHTile:
        """Extracts 2-d data from a multi-page TIFF containing all Tiles for an imaging round

        Parameters
        ----------
        fov : int
            Number of the fov, offset but fov_offset
        r : int
            Imaging round. Selects the TIFF file to examine
        ch : int
            Selects the channel from within the loaded TIFF file
        zplane : int
            Selects the z-plane from within the loaded TIFF file

        Returns
        -------
        FISHTile :
            SeqFISH subclass of FetchedTile
        """
        #print("fov: {} round: {} channel: {} zplane: {}".format(fov_id, round_label, ch_label, zplane_label))
        varTable = {
                "channel": ch_label,
                "offset_channel": ch_label + self.channel_offset,
                "round": round_label,
                "offset_round": round_label + self.round_offset,
                "fov": fov_id,
                "offset_fov": fov_id + self.fov_offset,
                "zplane": zplane_label
        }
        file_path = os.path.join(self.input_dir, self.file_format.format(*[varTable[arg] for arg in self.file_vars]))
        return FISHTile(file_path, zplane_label, ch_label, round_label, False, self.cache_read_order)
    
class AuxTileFetcher(TileFetcher):
    # we define this separately to manually override parameters
    # this is used for the dapi images for registration
    # so only one channel and round are used.

    def __init__(self, input_dir: str, file_format: str="", file_vars: str= "", cache_read_order: list=[], fov_offset: int=0, round_offset: int=0, channel_offset: int=0, fixed_round: int=0) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.file_format = file_format
        self.file_vars = file_vars.split(";")
        self.input_dir = input_dir
        self.cache_read_order = cache_read_order
        self.fov_offset = fov_offset
        self.round_offset = round_offset
        self.channel_offset = channel_offset
        self.fixed_round = fixed_round

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FISHTile:
        """Extracts 2-d data from a multi-page TIFF containing all Tiles for an imaging round

        Parameters
        ----------
        fov : int
            Number of the fov, offset but fov_offset
        r : int
            Imaging round. Selects the TIFF file to examine
        ch : int
            Selects the channel from within the loaded TIFF file
        zplane : int
            Selects the z-plane from within the loaded TIFF file

        Returns
        -------
        FISHTile :
            SeqFISH subclass of FetchedTile
        """
        varTable = {
                "channel": ch_label,
                "offset_channel": ch_label + self.channel_offset,
                "round": round_label,
                "offset_round": round_label + self.round_offset,
                "fov": fov_id,
                "offset_fov": fov_id + self.fov_offset,
                "zplane": zplane_label
        }
        file_path = os.path.join(self.input_dir, self.file_format.format(*[varTable[arg] for arg in self.file_vars]))
        #print(file_path)
        return FISHTile(file_path, zplane_label, self.fixed_round, round_label, self.fixed_round, self.cache_read_order) # CHANNEL ID IS FIXED


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
    integer_round_ids = range(csv.shape[1])
    csv.columns = integer_round_ids

    mappings = []

    for gene, channel_series in csv.iterrows():
        mappings.append({
            Features.CODEWORD: [{
                Axes.ROUND.value: r, Axes.CH.value: c - 1, Features.CODE_VALUE: 1
            } for r, c in channel_series.items()],
            Features.TARGET: gene
        })

    return Codebook.from_code_array(mappings)

def cli(input_dir: str, output_dir: str, file_format: str, file_vars: list, cache_read_order:list, counts: dict, codebook_csv: str, aux_names: list = [], aux_file_formats: list = [], aux_file_vars: list = [], aux_cache_read_order: list = [], aux_fixed_channel: list = []) -> int:
    """CLI entrypoint for spaceTx format construction for SeqFISH data

    Parameters
    ----------
    input_dir : str
        directory containing folders for fovs
    output_dir : str
        directory containing output files. Will be created if it does not exist.
    counts: dict
        dict with the counts for each dimension of the data. Expects values that correspond
        to keys of ["rounds","channels","zplanes","fovs"]
    codebook_csv : str
        name of the codebook csv file containing barcode information for this field of view.

    Notes
    -----
    - each round is organized as [z, ch, [x|y], [x|y]] -- the order of x and y are not known, but
      since this script uses dummy coordinates, this distinction is not important.
    - The spatial organization of the field of view is not known to the starfish developers,
      so they are filled by dummy coordinates

    Returns
    -------
    int :
        Returns 0 if successful
    """

    os.makedirs(output_dir, exist_ok=True)

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
    
    #file_format = "HybCycle_{}/MMStack_Pos{}.ome.tif"
    #file_vars = ["offset_round", "offset_fov"]

    cache_read_order_formatted = []
    for item in cache_read_order:
        if item == "Z": cache_read_order_formatted.append(Axes.ZPLANE)
        elif item == "CH": cache_read_order_formatted.append(Axes.CH)
        else: cache_read_order_formatted.append("other")
   
    primary_tile_fetcher = PrimaryTileFetcher(os.path.expanduser(input_dir), file_format, file_vars, cache_read_order_formatted, counts["fov_offset"], counts["round_offset"], counts["channel_offset"])
    
    aux_name_to_dimensions = {}
    aux_tile_fetcher = {}
    if aux_names:
        for i in range(len(aux_names)):
            name = aux_names[i]
            aux_name_to_dimensions[name] = aux_image_dimensions
            aux_cache_read_order_raw = aux_cache_read_order[i].split(";")
            aux_cache_read_order_formatted = []
            for item in aux_cache_read_order_raw:
                if item == "Z": aux_cache_read_order_formatted.append(Axes.ZPLANE)
                elif item == "CH": aux_cache_read_order_formatted.append(Axes.CH)
                else: aux_cache_read_order_formatted.append("other")
            
            aux_tile_fetcher[name] = AuxTileFetcher(os.path.expanduser(input_dir), aux_file_formats[i], aux_file_vars[i], aux_cache_read_order_formatted, counts["fov_offset"], counts["round_offset"], counts["channel_offset"], aux_fixed_channel[i])
    #aux_tile_fetcher = {"DAPI": AuxTileFetcher(os.path.expanduser(input_dir), file_format, file_vars, counts["fov_offset"], counts["round_offset"],3)}
    #aux_name_to_dimensions = {"DAPI": aux_image_dimensions}
    

    write_experiment_json(
        path=output_dir,
        fov_count=counts["fovs"],
        aux_tile_fetcher=aux_tile_fetcher,
        primary_tile_fetcher=primary_tile_fetcher,
        primary_image_dimensions=image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        tile_format=ImageFormat.TIFF,
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )

    # Note: this must trigger AFTER write_experiment_json, as it will clobber the codebook with
    # a placeholder.
    codebook = parse_codebook(codebook_csv)
    codebook.to_json("codebook.json") # should this be renamed to include the output dir?

    return 0

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--codebook-csv", type=Path)
    p.add_argument("--round-count", type=int)
    p.add_argument("--zplane-count", type=int)
    p.add_argument("--channel-count", type=int)
    p.add_argument("--fov-count", type=int)
    p.add_argument("--round-offset", type=int, default=0)
    p.add_argument("--fov-offset", type=int, default=0)
    p.add_argument("--channel-offset", type=int, default=0)
    p.add_argument("--file-format", type=str)
    p.add_argument("--file-vars", nargs = '+')
    p.add_argument("--cache-read-order", nargs = '+')
    p.add_argument("--aux-names", nargs = "+")
    p.add_argument("--aux-file-formats", nargs = "+")
    p.add_argument("--aux-file-vars", nargs = "+")
    p.add_argument("--aux-fixed-channel", type=int, nargs = "+")
    p.add_argument("--aux-cache-read-order", nargs ='+')

    args = p.parse_args()

    if len({len(args.aux_names) if args.aux_names else 0,
            len(args.aux_file_formats) if args.aux_file_formats else 0, 
            len(args.aux_file_vars) if args.aux_file_vars else 0, 
            len(args.aux_fixed_channel) if args.aux_fixed_channel else 0,
            len(args.aux_cache_read_order) if args.aux_cache_read_order else 0}) > 1:
        print(args.aux_names, args.aux_file_formats, args.aux_file_vars, args.aux_fixed_channel, args.aux_cache_read_order)
        raise Exception("Dimensions of all aux parameters must match.")

    counts = {"rounds":         args.round_count,
             "channels":        args.channel_count,
             "zplanes":         args.zplane_count,
             "fovs":            args.fov_count,
             "round_offset":    args.round_offset,
             "fov_offset":      args.fov_offset,
             "channel_offset":  args.channel_offset}
    cli(args.input_dir, "tx_converted/", 
            args.file_format, args.file_vars, args.cache_read_order, counts, args.codebook_csv,
            args.aux_names, args.aux_file_formats, args.aux_file_vars, args.aux_cache_read_order, args.aux_fixed_channel)
