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


class SeqFISHTile(FetchedTile):
    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], CoordinateValue],
            zplane: int,
            ch: int,
            is_aux: bool,
    ):
        self._file_path = file_path
        self._zplane = ch
        self._ch = zplane
        self._coordinates = coordinates
        self.is_aux = is_aux
        #print("zpl {};ch {}".format(zplane,ch))

    @property
    def shape(self) -> Mapping[Axes, int]:
        """
        Gets image shape directly from the data. Note that this will result in the data being
        read twice, since the shape is retrieved from all tiles before the data is read, and thus
        single-file caching does not resolve the duplicated reads.

        Because the data here isn't tremendously large, this is acceptable in this instance.
        """
        #print(self.tile_data().shape)
        raw_shape = self.tile_data().shape
        return {Axes.Y: raw_shape[0], Axes.X: raw_shape[1]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Stores coordinate information passed from the TileFetcher"""
        return self._coordinates

    @property
    def format(self) -> ImageFormat:
        """Image Format for SeqFISH data is TIFF"""
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        """vary z the slowest, then channel -- each round has its own TIFF"""
        #print(cached_read_fn(self._file_path).shape)
        try:
            if self.is_aux: # aux tiles are always the 4th channel
                return cached_read_fn(self._file_path)[3,:,:,self._zplane]
            else:
                return cached_read_fn(self._file_path)[self._ch,:,:,self._zplane]
        except IndexError as e:
            print("\t{} zpl: {} ch: {} with error {}".format(self._file_path, self._zplane, self._ch, e))
            return np.zeros((2048,2048))
            
class SeqFISHTileFetcher(TileFetcher):

    def __init__(self, input_dir: str, fov_offset=0, round_offset=0) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.input_dir = input_dir
        self.fov_offset = fov_offset
        self.round_offset = round_offset
        

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Returns dummy coordinates because these pics don't overlap"""
        return {
            Coordinates.X: (0., 1.),
            Coordinates.Y: (0., 1.),
            Coordinates.Z: (0., 0.1),
        }

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> SeqFISHTile:
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
        SeqFISHTile :
            SeqFISH subclass of FetchedTile
        """
        #print("fov: {} round: {} channel: {} zplane: {}".format(fov_id, round_label, ch_label, zplane_label))
        file_path = os.path.join(self.input_dir, 
                                           f"HybCycle_{round_label + self.round_offset}/MMStack_Pos{fov_id + self.fov_offset}.ome.tif")
        return SeqFISHTile(file_path, self.coordinates, zplane_label, ch_label, False)
    
class SeqFISHAuxTileFetcher(TileFetcher):
    # we define this separately to manually override parameters
    # this is used for the dapi images for registration
    # so only one channel and round are used.

    def __init__(self, input_dir: str, fov_offset=0, round_offset=0) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.input_dir = input_dir
        self.fov_offset = fov_offset
        self.round_offset = round_offset

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Returns dummy coordinates because these pics don't overlap"""
        return {
            Coordinates.X: (0., 1.),
            Coordinates.Y: (0., 1.),
            Coordinates.Z: (0., 0.1),
        }

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> SeqFISHTile:
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
        SeqFISHTile :
            SeqFISH subclass of FetchedTile
        """
        file_path = os.path.join(self.input_dir, 
                                 f"HybCycle_{round_label + self.round_offset}/MMStack_Pos{fov_id + self.fov_offset}.ome.tif")
        #print(file_path)
        return SeqFISHTile(file_path, self.coordinates, zplane_label, 3, True) # CHANNEL ID IS FIXED


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

def cli(input_dir: str, output_dir: str, counts: dict, codebook_csv: str) -> int:
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
    
    primary_tile_fetcher = SeqFISHTileFetcher(os.path.expanduser(input_dir), counts["fov_offset"], counts["round_offset"])
    aux_tile_fetcher = {"DAPI": SeqFISHAuxTileFetcher(os.path.expanduser(input_dir), counts["fov_offset"], counts["round_offset"])}
    aux_name_to_dimensions = {"DAPI": aux_image_dimensions}
    

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
    p.add_argument("--round_offset", type=int, default=0)
    p.add_argument("--fov_offset", type=int, default=0)

    args = p.parse_args()

    counts = {"rounds":     args.round_count,
             "channels":    args.channel_count,
             "zplanes":     args.zplane_count,
             "fovs":        args.fov_count,
             "round_offset":args.round_offset,
             "fov_offset":  args.fov_offset}
    cli(args.input_dir, ".", counts, args.codebook_csv)
