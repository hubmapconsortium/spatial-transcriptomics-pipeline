#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, path
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import skimage.io
import starfish
import yaml
from PIL import Image
from starfish import Codebook


def parse_codebook(codebook_csv: str) -> Codebook:
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


def convert_codebook(
    oldbook: Codebook, cycles_conv: Dict[int, int], channels_conv: List[Dict[int, int]]
) -> Codebook:
    raw = oldbook.data
    targets = np.shape(raw)[0]
    rounds = len(cycles_conv)
    channels = len(channels_conv[0])
    new_data = np.empty((targets, rounds, channels), dtype=int)
    for t in range(targets):
        for pr in range(len(raw[t])):
            # annoying math because dicts are saved for the other direction
            pchannel = np.argmax(raw[t][pr])
            subChannel = [
                [tch for tch, pch in subchannel.items() if pch == pchannel]
                for subchannel in channels_conv
            ]
            subRound = np.argmax([len(per_round) for per_round in subChannel])
            tchannel = subChannel[subRound][0]
            tround = [tr for tr, pround in cycles_conv.items() if pround == pr][subRound]
            # print("channel {}->{} round {}->{}".format(pchannel,tchannel,pr,tround))
            new_data[t][tround][tchannel] = 1

    return Codebook.from_numpy(oldbook.coords["target"].data, rounds, channels, new_data)


def reformatter(
    cycles_conv: Dict[int, int],
    channels_conv: List[Dict[int, int]],
    input_dir: str,
    file_format: str = "",
    output_format: str = "",
    output_vars: List[str] = [],
    output_dir: str = "",
    file_vars: str = "",
    fov_count: int = 1,
    cache_read_order: List[str] = [],
    channel_slope: float = 1,
    channel_intercept: int = 0,
    fov_offset: int = 0,
    round_offset: int = 0,
    channel_offset: int = 0,
    aux_file_formats: List[str] = [],
    aux_file_vars: List[List[str]] = [],
    aux_names: List[str] = [],
    aux_cache_read_order: List[List[str]] = [],
    aux_channel_count: List[int] = [],
    aux_channel_slope: List[float] = [],
    aux_channel_intercept: List[int] = [],
):

    reportFile = path.join(output_dir, datetime.now().strftime("%Y%d%m_%H%M_psorting.log"))
    sys.stdout = open(reportFile, "w")

    combined_file_format = [file_format] + aux_file_formats
    combined_file_vars = [file_vars] + aux_file_vars
    combined_names = [""] + aux_names
    combined_cache_read_order = [cache_read_order] + aux_cache_read_order
    combined_channel_count = [len(channels_conv[0])] + aux_channel_count
    channel_slope = [1] + aux_channel_slope
    channel_intercept = [0] + aux_channel_intercept
    views = len(combined_names)

    for r in cycles_conv.keys():
        for c in range(max(combined_channel_count)):
            for fov in range(fov_count):
                varTable = {
                    "channel": c,
                    "offset_channel": c + channel_offset,
                    "round": r,
                    "offset_round": r + round_offset,
                    "fov": fov,
                    "offset_fov": fov + fov_offset,
                }
                for target in range(views):
                    if c < combined_channel_count[target]:
                        varTable["input_channel"] = int(
                            int(c * channel_slope[target]) + channel_intercept[target]
                        )
                        file_path = path.join(
                            input_dir,
                            combined_file_format[target].format(
                                *[varTable[arg] for arg in combined_file_vars[target]]
                            ),
                        )
                        print(varTable)
                        img = skimage.io.imread(file_path)
                        # img_out = img

                        # figure out what slice to take.
                        slices = []
                        for i in range(len(combined_cache_read_order[target])):
                            axis = combined_cache_read_order[target][i]
                            if axis.lower() == "ch":
                                c_adj = int(channel_slope[target] * c) + channel_intercept[target]
                                slices.append(int(c_adj))
                            elif axis.lower() == "round":
                                slices.append(r)
                            else:
                                slices.append(slice(0, img.shape[i]))

                        # take slices out of image and reduce unneeded dims
                        slices = tuple(slices)
                        print(slices)
                        img_out = np.squeeze(img[slices])

                        # convert to new rounds/channels
                        pr = cycles_conv[r]
                        pc = channels_conv[r % len(channels_conv)][c]

                        # get output string
                        varTableConv = {
                            "channel": pc,
                            "offset_channel": pc + channel_offset,
                            "round": pr,
                            "offset_round": pr + round_offset,
                            "fov": fov,
                            "offset_fov": fov + fov_offset,
                            "aux_name": combined_names[target],
                        }
                        output_path = path.join(
                            output_dir,
                            output_format.format(*[varTableConv[arg] for arg in output_vars]),
                        )
                        print("{}\n->{}".format(file_path, output_path))
                        print(np.shape(img_out))
                        skimage.io.imsave(output_path, img_out)

    sys.stdout = sys.__stdout__
    return True


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--codebook-csv", type=Path, nargs="?")
    p.add_argument("--codebook-json", type=Path, nargs="?")
    p.add_argument("--channel-yml", type=Path)
    p.add_argument("--cycle-yml", type=Path)
    p.add_argument("--file-format", type=str)
    p.add_argument("--file-vars", type=str, nargs="+")
    p.add_argument("--cache-read-order", type=str, nargs="+")
    p.add_argument("--z-plane-offset", type=int)
    p.add_argument("--fov-offset", type=int)
    p.add_argument("--round-offset", type=int)
    p.add_argument("--channel-offset", type=int)
    p.add_argument("--fov-count", type=int)
    p.add_argument("--channel-slope", type=float)
    p.add_argument("--channel-intercept", type=int)
    p.add_argument("--aux-file-formats", type=str, nargs="+", const=None)
    p.add_argument("--aux-file-vars", type=str, nargs="+", const=None)
    p.add_argument("--aux-names", type=str, nargs="+", const=None)
    p.add_argument("--aux-cache-read-order", type=str, nargs="+", const=None)
    p.add_argument("--aux-channel-count", type=int, nargs="+", const=None)
    p.add_argument("--aux-channel-slope", type=float, nargs="+", const=None)
    p.add_argument("--aux-channel-intercept", type=float, nargs="+", const=None)

    args = p.parse_args()

    aux_lens = []
    aux_vars = [
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

    output_dir = "1_pseudosort/"
    output_format = "PseudoCycle{}/MMStack_Pos{}_{}ch{}.ome.tif"
    output_vars = ["round", "fov", "aux_name", "channel"]

    with open(args.channel_yml, "r") as fl:
        channels_conv: List[Dict[int, int]] = yaml.load(fl, Loader=yaml.FullLoader)

    with open(args.cycle_yml, "r") as fl:
        cycles_conv: Dict[int, int] = yaml.load(fl, Loader=yaml.FullLoader)

    for i in range(len(set(cycles_conv.values()))):
        makedirs("{}PseudoCycle{}".format(output_dir, i))

    aux_file_vars = [item.split(";") for item in args.aux_file_vars]
    aux_cache_read_order = [item.split(";") for item in args.aux_cache_read_order]

    reformatter(
        cycles_conv=cycles_conv,
        channels_conv=channels_conv,
        input_dir=args.input_dir,
        file_format=args.file_format,
        output_format=output_format,
        output_vars=output_vars,
        output_dir=output_dir,
        file_vars=args.file_vars,
        fov_count=args.fov_count,
        cache_read_order=args.cache_read_order,
        channel_slope=args.channel_slope,
        channel_intercept=args.channel_intercept,
        fov_offset=args.fov_offset,
        round_offset=args.round_offset,
        channel_offset=args.channel_offset,
        aux_file_formats=args.aux_file_formats,
        aux_file_vars=aux_file_vars,
        aux_names=args.aux_names,
        aux_cache_read_order=aux_cache_read_order,
        aux_channel_count=args.aux_channel_count,
        aux_channel_slope=args.aux_channel_slope,
        aux_channel_intercept=args.aux_channel_intercept,
    )

    if args.codebook_csv:
        codebook = parse_codebook(args.codebook_csv)
    elif args.codebook_json:
        codebook = Codebook.open_json(args.codebook_json)
    else:
        print("Can't convert notebook, none provided.")

    conv_codebook = convert_codebook(codebook, cycles_conv, channels_conv)
    codebook.to_json(output_dir + "pround_codebook.json")
    conv_codebook.to_json(output_dir + "codebook.json")
