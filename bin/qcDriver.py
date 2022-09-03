#!/usr/bin/env python

import collections
import functools
import gc
import math
import pickle
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partialmethod
from glob import glob
from os import makedirs, path
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import skimage.io
import starfish
import starfish.data
import yaml
from astropy.stats import RipleysKEstimator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, skew
from starfish import BinaryMaskCollection, Codebook, DecodedIntensityTable, ImageStack
from starfish.core.types import (
    DecodedSpots,
    Number,
    PerImageSliceSpotResults,
    SpotAttributes,
    SpotFindingResults,
)
from starfish.types import Axes, Coordinates, CoordinateValue, Features
from tqdm import tqdm

# utility methods


def thresholdSpots(spots, spotThreshold):
    # takes a SpotFindingResults and removes all spots that have 'intensity'
    # beneath spotThreshold. Useful when a reference image is used.
    spot_attributes_list = []
    for k, v in spots.items():
        data = v.spot_attrs.data
        data = data[data["intensity"] > spotThreshold]
        data = data.reset_index(drop=True)
        spotResults = PerImageSliceSpotResults(spot_attrs=SpotAttributes(data), extras=None)
        spot_attributes_list.append((spotResults, {Axes.ROUND: k[0], Axes.CH: k[1]}))

    newCoords = {}
    renameAxes = {"x": Coordinates.X.value, "y": Coordinates.Y.value, "z": Coordinates.Z.value}
    for k in renameAxes.keys():
        newCoords[renameAxes[k]] = spots.physical_coord_ranges[k]

    newSpots = SpotFindingResults(
        imagestack_coords=newCoords, log=starfish.Log(), spot_attributes_list=spot_attributes_list
    )
    return newSpots


def filterSpots(spots, mask, oneIndex=False, invert=False):
    # takes a SpotFindingResults, ImageStack, and BinaryMaskCollection
    # to return a set of SpotFindingResults that are masked by the binary mask
    spot_attributes_list = []
    if isinstance(mask, BinaryMaskCollection):
        maskMat = mask.to_label_image().xarray.values
    else:
        maskMat = mask
    maskMat[maskMat > 1] = 1
    if invert:
        maskMat = 1 - maskMat
    maskSize = np.shape(maskMat)
    for item in spots.items():
        selectedSpots = item[1].spot_attrs.data
        selectedSpots = selectedSpots.reset_index(drop=True)

        selRow = []
        for ind, row in selectedSpots.iterrows():
            if maskMat[int(row["y"]) - oneIndex][int(row["x"]) - oneIndex] == 1:
                selRow.append(ind)

        selectedSpots = selectedSpots.iloc[selRow]
        selectedSpots = selectedSpots.drop_duplicates()  # unsure why the query necessitates this

        spotResults = PerImageSliceSpotResults(
            spot_attrs=SpotAttributes(selectedSpots), extras=None
        )
        spot_attributes_list.append((spotResults, {Axes.ROUND: item[0][0], Axes.CH: item[0][1]}))
    newCoords = {}

    renameAxes = {"x": Coordinates.X.value, "y": Coordinates.Y.value, "z": Coordinates.Z.value}
    for k in renameAxes.keys():
        newCoords[renameAxes[k]] = spots.physical_coord_ranges[k]

    newSpots = SpotFindingResults(
        imagestack_coords=newCoords, log=starfish.Log(), spot_attributes_list=spot_attributes_list
    )
    return newSpots


def monteCarloEnvelope(Kest, r, p, n, count):
    # Kest: Kest object from astropy
    # r: list of radii to test
    # p: confidence interval to compute on (0<p<1)
    # n: number of points to create in each simulation
    # count: number of simulations to run
    vals = []
    for i in range(count):
        z = np.random.uniform(low=Kest.x_min, high=Kest.x_max, size=(n, 2))
        # note that this assumes Kest is looking at a square area
        vals.append(Kest(data=z, radii=r, mode="ripley"))
        del z
    top = np.quantile(vals, q=1 - (1 - p) / 2, axis=0)
    mid = np.quantile(vals, q=0.5, axis=0)
    bot = np.quantile(vals, q=(1 - p) / 2, axis=0)
    return (top, mid, bot)


def plotRipleyResults(pdf, results, key, doMonte=False, text=None):
    if doMonte:
        res, mon = results
        monte = mon[key]
        kv, csr, r = res[key]
    else:
        kv, csr, r = results[key]

    fig, ax = plt.subplots()

    ax.plot(r, kv, label="Data")
    if doMonte:
        ax.plot(r, monte[0], "--r")
        ax.plot(r, monte[1], "-r", label="median monte carlo")
        ax.plot(r, monte[2], "--r")
    else:
        ax.plot(r, csr, "-r", label="CSR")

    if text is not None:
        ax.set_title(str(text))
    ax.set_ylabel("Ripley's K score")
    ax.set_xlabel("Radius (px)")
    ax.legend()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    # plt.show()
    pdf.savefig(fig)
    plt.close()


## Internal Metrics
# Spot Metrics


def getSpotDensity(spots, codebook):
    return spots.count_total_spots() / len(codebook.target)


def getSpatialDensity(spots, imgsize, steps=10, doMonte=False):
    allSpots = {}
    for i in range(imgsize[2]):
        for k in spots.keys():
            r, ch = k
            allSpots[(r, ch, i)] = pd.DataFrame(columns=["x", "y"])

    for k, v in spots.items():
        tempSpots = v.spot_attrs.data[["x", "y", "z"]]
        r, ch = k
        for i in range(imgsize[2]):
            allSpots[(r, ch, i)] = allSpots[(r, ch, i)].append(tempSpots[tempSpots["z"] == i])

    print("Sorted all relevant spots")
    # print(allSpots)

    results = {}
    savedKest = None
    for i in allSpots.keys():
        print("looking at " + str(i) + "\nsize: " + str(allSpots[i].shape))
        allSpots[i].drop_duplicates(inplace=True)
        print("dropped dupes, size is now " + str(allSpots[i].shape))
        allSpots[i].drop(columns="z", inplace=True)
        print("removed z column, size is " + str(allSpots[i].shape))

        ymin = 0
        xmin = 0
        ymax = imgsize[1]
        xmax = imgsize[0]
        area = abs(ymax - ymin) * abs(xmax - xmin)
        Kest = RipleysKEstimator(area=area, x_max=xmax, y_max=ymax, x_min=xmin, y_min=ymin)
        r = np.linspace(0, ((area) / 2) ** 0.5, steps)

        print("finding Kest")
        data = allSpots[i][["x", "y"]].to_numpy()
        # print(data)
        kvals = Kest(data=data, radii=r, mode="ripley")
        print("found Kest\n")
        # env = monteCarloEnvelope(Kest, r, .95, np.size(allSpots[i]), 100)
        csr = Kest.poisson(r)

        if doMonte:
            if savedKest is not None and savedKest.area != Kest.area:
                print(
                    "Note! Area different between Kest, monte estimation may be wrong.\n{} old, {} new".format(
                        savedKest.area, Kest.area
                    )
                )
            savedKest = Kest

        del Kest
        gc.collect()

        results[i] = (kvals, csr, r)

    savedSims = {}
    savedSimsSearchable = np.array([])
    numsim = 100
    if doMonte:
        monte = {}
        for k in allSpots.keys():
            simSize = allSpots[k].shape[0]
            searchResults = np.where(
                np.logical_and(
                    savedSimsSearchable >= simSize * 0.95, savedSimsSearchable <= simSize * 1.05
                )
            )
            if np.shape(searchResults)[1] > 0:
                closest = savedSimsSearchable[searchResults[0][0]]
                print(
                    "Similar simulation found, {} hits near size {}".format(
                        np.shape(searchResults)[1], simSize
                    )
                )

                # in the event of multiple matches in this range,
                # go with the closest
                for ind in searchResults[1:]:
                    it = savedSimsSearchable[ind[0]]
                    if abs(it - simSize) < abs(closest - simSize):
                        closest = it
                print("Using sim of size {}".format(closest))
                monte[k] = savedSims[closest]
            else:
                print(
                    "No close simulation saved for {}, running new sim with sample count {}".format(
                        k, allSpots[k].shape[0]
                    )
                )
                newSim = monteCarloEnvelope(savedKest, r, 0.95, simSize, numsim)
                monte[k] = newSim
                savedSims[simSize] = newSim
                savedSimsSearchable = np.append(savedSimsSearchable, simSize)

        return results, monte

    return results


def percentMoreClustered(results):
    # assumes results has monte included
    # returns list with % of each (r,ch,z) plane where calculated
    #    Kest > 95% monte null hypothesis
    planeWise = {}
    mean = 0
    for k in results[0].keys():
        topend = results[1][k][0]
        calc = results[0][k][0]
        planeWise[k] = sum([1 if topend[i] < calc[i] else 0 for i in range(len(topend))]) / len(
            topend
        )
        mean += planeWise[k]
    mean = mean / len(planeWise.keys())
    return mean, planeWise


def percentLessClustered(results):
    # assumes results has monte included
    # returns list with % of each (r,ch,z) plane where calculated
    #    Kest < 95% monte null hypothesis
    planeWise = {}
    mean = 0
    for k in results[0].keys():
        botend = results[1][k][2]
        calc = results[0][k][0]
        planeWise[k] = sum([1 if botend[i] < calc[i] else 0 for i in range(len(botend))]) / len(
            botend
        )
        mean += planeWise[k]
    mean = mean / len(planeWise.keys())
    return mean, planeWise


def getSpotRoundDist(spots, pdf=False):
    roundTallies = {}
    for k, v in spots.items():
        r, ch = k
        if r not in roundTallies.keys():
            roundTallies[r] = 0
        for i, item in v.spot_attrs.data.iterrows():
            roundTallies[r] += 1

    tally = [roundTallies[i] for i in range(max(roundTallies.keys()) + 1)]
    tally = [t / sum(tally) for t in tally]
    std = np.std(tally)
    skw = skew(tally)

    if pdf:
        fig, ax = plt.subplots()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(list(range(len(tally))), tally)
        plt.title("Spots per round")
        plt.xlabel("Round number")
        plt.ylabel("Spot count")

        avg = np.mean(tally)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
        plt.axhline(avg, color="black")
        plt.text(0, avg + offset, f"Average: {avg:.2f}")
        plt.axhline(avg + std, dashes=(2, 2), color="black")
        plt.axhline(avg - std, dashes=(2, 2), color="black")
        plt.text(0, avg - std + offset, f"Standard Deviation: {std:.2f}")

        pdf.savefig(fig)
        plt.close()

    return {"tally": tally, "stdev": std, "skew": skw}


def getSpotChannelDist(spots, pdf=False):
    channelTallies = {}
    for k, v in spots.items():
        r, ch = k
        if ch not in channelTallies.keys():
            channelTallies[ch] = 0
        for i, item in v.spot_attrs.data.iterrows():
            channelTallies[ch] += 1

    tally = [channelTallies[i] for i in range(max(channelTallies.keys()) + 1)]
    tally = [t / sum(tally) for t in tally]
    std = np.std(tally)
    skw = skew(tally)

    if pdf:
        fig, ax = plt.subplots()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(list(range(len(tally))), tally)
        plt.title("Spots per channel")
        plt.xlabel("Channel number")
        plt.ylabel("Fraction of spots")

        avg = np.mean(tally)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
        plt.axhline(avg, color="black")
        plt.text(0, avg + offset, f"Average: {avg:.2f}")
        plt.axhline(avg + std, dashes=(2, 2), color="black")
        plt.axhline(avg - std, dashes=(2, 2), color="black")
        plt.text(0, avg - std + offset, f"Standard Deviation: {std:.2f}")

        pdf.savefig(fig)
        plt.close()
    return {"tally": tally, "stdev": std, "skew": skw}


def maskedSpatialDensity(masked, unmasked, imgsize, steps, pdf=False):
    maskedDens = getSpatialDensity(masked, imgsize, steps, True)
    unmaskedDens = getSpatialDensity(unmasked, imgsize, steps, True)

    if pdf:
        for k in unmaskedDens[0].keys():
            plotRipleyResults(
                pdf=pdf,
                results=unmaskedDens,
                key=k,
                doMonte=True,
                text="Unmasked {}".format(str(k)),
            )
            plotRipleyResults(
                pdf=pdf, results=maskedDens, key=k, doMonte=True, text="Masked {}".format(str(k))
            )

    result = {}
    for val in ["more clustered", "less clustered"]:
        func = percentMoreClustered
        if val == "less clustered":
            func = percentLessClustered

        maskedPer = func(maskedDens)[0]
        unmaskedPer = func(unmaskedDens)[0]
        result[val] = {
            "ratio": unmaskedPer / maskedPer,
            "unmasked": unmaskedPer,
            "masked": maskedPer,
        }
    return result


# Transcript metrics


def getTranscriptDensity(transcripts, codebook):
    return np.shape(transcripts.data)[0] / len(codebook.target)


def getTranscriptsPerCell(segmented, pdf=False):
    counts = []
    # column to look at will be different depending if we've run baysor
    if "cell" in segmented.keys():
        cells = segmented["cell"]
    else:
        cells = segmented["cell_id"]

    # remove unassigned cells
    cells = [int(x) for x in cells if not np.isnan(x)]

    if len(cells) == 0:
        return ([0], 0, 0)

    for i in range(int(max(cells)) + 1):
        counts.append(len([x for x in cells if x == i]))

    counts.sort()
    counts = np.flip(counts)
    q1, mid, q3 = np.percentile(counts, [25, 50, 75])
    iqr_scale = 1.5

    if pdf:
        fig, ax = plt.subplots()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(list(range(len(counts))), counts)
        plt.axhline(
            y=mid - iqr_scale * (q3 - q1), dashes=(1, 1), color="gray", label="Outlier Threshold"
        )
        plt.axhline(y=q1, dashes=(2, 2), color="black", label="IQR")
        plt.axhline(y=mid, color="black", label="Median")
        plt.axhline(y=q3, dashes=(2, 2), color="black")
        plt.axhline(y=mid + iqr_scale * (q3 - q1), dashes=(1, 1), color="gray")
        plt.ylim(0)
        plt.xlim(0)
        plt.title("Transcript count per cell")
        plt.ylabel("Transcript count")
        plt.xlabel("Cells")
        plt.legend()

        pdf.savefig(fig)
        plt.close()
    return {
        "counts": counts,
        "quartiles": (q1, mid, q3),
        "stdev": np.std(counts),
        "skew": skew(counts),
    }


def getFractionSpotsUsed(spots, transcripts):
    spotCount = spots.count_total_spots()
    trspotCount = np.count_nonzero(transcripts.data)
    return trspotCount / spotCount


def getTranscriptRoundDist(transcripts, pdf=False):
    conv = np.where(transcripts.data > 0, 1, 0)
    counts = np.sum(conv, axis=(0, 2))
    counts = [c / sum(counts) for c in counts]
    std = np.std(counts)
    skw = skew(counts)

    if pdf:
        fig, ax = plt.subplots()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(range(len(counts)), counts)
        plt.title("Transcript source spot distribution across rounds")
        plt.ylabel("Spot count")
        plt.xlabel("Round ID")

        avg = np.mean(counts)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
        plt.axhline(avg, color="black")
        plt.text(0, avg + offset, f"Average: {avg:.2f}")
        plt.axhline(avg + std, dashes=(2, 2), color="black")
        plt.axhline(avg - std, dashes=(2, 2), color="black")
        plt.text(0, avg - std + offset, f"Standard Deviation: {std:.2f}")

        pdf.savefig(fig)
        plt.close()
    return {"counts": counts, "stdev": std, "skew": skw}


def getTranscriptChannelDist(transcripts, pdf=False):
    conv = np.where(transcripts.data > 0, 1, 0)
    counts = np.sum(conv, axis=(0, 1))
    counts = [c / sum(counts) for c in counts]
    std = np.std(counts)
    skw = skew(counts)

    if pdf:
        fig, ax = plt.subplots()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(range(len(counts)), counts)
        plt.title("Transcript source spot distribution across channels")
        plt.ylabel("Fraction of spots")
        plt.xlabel("Channel ID")

        avg = np.mean(counts)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
        plt.axhline(avg, color="black")
        plt.text(0, avg + offset, f"Average: {avg:.2f}")
        plt.axhline(avg + std, dashes=(2, 2), color="black")
        plt.axhline(avg - std, dashes=(2, 2), color="black")
        plt.text(0, avg - std + offset, f"Standard Deviation: {std:.2f}")

        pdf.savefig(fig)
        plt.close()
    return {"counts": counts, "stdev": std, "skew": skw}


def getFPR(segmentation, pdf=False):
    # remove unassigned transcripts, if the columns to do so are present.

    # remove unassigned transcripts, if the columns to do so are present.
    key = "cell"
    if "cell_id" in segmentation.keys():
        key = "cell_id"

    segmentation = segmentation[segmentation[key].notnull()]

    if len(segmentation) == 0:
        return None

    blank_counts_full = segmentation[segmentation["target"].str.contains("blank")]
    real_counts_full = segmentation[~segmentation["target"].str.contains("blank")]

    cell_count = int(real_counts_full[key].max()) + 1
    real_per_cell_full = np.histogram(real_counts_full[key], bins=cell_count)[0]
    blank_per_cell_full = np.histogram(blank_counts_full[key], bins=cell_count)[0]

    sort_ind = np.argsort(real_per_cell_full)
    sorted_reals_full = [
        real_per_cell_full[sort_ind[len(sort_ind) - i - 1]] for i in range(len(sort_ind))
    ]
    sorted_blanks_full = [
        blank_per_cell_full[sort_ind[len(sort_ind) - i - 1]] for i in range(len(sort_ind))
    ]

    results = {
        "FP": sum(blank_per_cell_full),
        "TP": sum(real_per_cell_full),
        "FPR": sum(blank_per_cell_full) / (sum(blank_per_cell_full) + sum(real_per_cell_full)),
    }

    if pdf:
        fig, ax = plt.subplots()

        plt.bar(
            range(len(real_per_cell_full)),
            sorted_reals_full,
            width=1,
            label="On-target",
            align="edge",
            color=(0, 119 / 256, 187 / 256),
        )
        plt.bar(
            range(len(blank_per_cell_full)),
            sorted_blanks_full,
            width=1,
            label="Off-target",
            align="edge",
            color=(204 / 256, 51 / 256, 17 / 256),
        )
        plt.plot(
            [0, len(real_per_cell_full)],
            [np.median(real_per_cell_full), np.median(real_per_cell_full)],
            color="black",
            label="Median count",
            linewidth=3,
        )
        plt.xlabel("Cells")
        plt.ylabel("Total barcodes per cell")
        plt.xlim([0, len(real_per_cell_full)])
        plt.ylim([0, max(max(real_per_cell_full), max(blank_per_cell_full)) * 1.1])
        plt.title("True positives vs False positives")

        plt.legend()

        pdf.savefig(fig)
        plt.close()

    return results


def plotSpotRatio(spots, transcripts, name, pdf):
    # Plots the channel/round distribution of spots and transcript sources on one graph
    fig, ax = plt.subplots()

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    avg = np.mean(spots)
    plt.scatter(
        [i for i in range(len(transcripts))], transcripts, label="spots used for transcripts"
    )
    plt.scatter([i for i in range(len(spots))], spots, label="all spots")
    plt.axhline(avg, color="black", lw=1)
    plt.ylim(0)
    plt.title(f"Comparison of spots and source spots for transcripts across {name}s")
    plt.ylabel("Fraction of spots")
    plt.xlabel(f"{name} ID")
    plt.legend()

    pdf.savefig(fig)
    plt.close()


def simplifyDict(ob):
    if isinstance(ob, collections.Mapping):
        return {k: simplifyDict(v) for k, v in ob.items()}
    elif isinstance(ob, np.ndarray):
        return ob.tolist()
    elif isinstance(ob, list):
        return [simplifyDict(k) for k in ob]
    elif isinstance(ob, tuple):
        return tuple([simplifyDict(k) for k in ob])
    elif isinstance(ob, np.generic):
        return ob.item()
    else:
        return ob


def runFOV(
    output_dir,
    transcripts,
    codebook,
    size,
    spots=None,
    spotThreshold=None,
    segmask=None,
    segmentation=None,
    doRipley=False,
    savePdf=False,
):

    t0 = time()

    # print("transcripts {}\ncodebook {}\nspots {}\nsegmented {}".format(transcripts, codebook, spots, segmentation))

    results = {}
    pdf = False
    if savePdf:
        pdf = PdfPages(output_dir + "graph_output.pdf")

    if spots:
        spotRes = {}
        print("finding spot metrics")

        if spotThreshold is not None:
            spots = thresholdSpots(spots, spotThreshold)

        relevSpots = spots
        if segmask is not None:
            relevSpots = filterSpots(spots, segmask, True)
            if relevSpots.count_total_spots() == 0:
                print("No spots inside segmentation area, are you sure the params are set right?")
                return None

        ts = time()
        spotRes["spot_counts"] = {
            "total_count": spots.count_total_spots(),
            "segmented_count": relevSpots.count_total_spots(),
            "ratio": relevSpots.count_total_spots() / spots.count_total_spots(),
        }
        spotRes["density"] = getSpotDensity(relevSpots, codebook)

        t1 = time()
        print("\tspot density elapsed time ", t1 - ts)

        spotRes["round_dist"] = getSpotRoundDist(relevSpots, pdf)

        t2 = time()
        print("\tround dist elapsed time ", t2 - t1)

        spotRes["channel_dist"] = getSpotChannelDist(relevSpots, pdf)

        t3 = time()
        print("\tchannel dist elapsed time ", t3 - t2)

        if doRipley:
            t = time()
            print("\n\tstarting ripley estimates")
            spatDens = getSpatialDensity(spots, size, doMonte=True)
            spotRes["spatial_density"] = percentMoreClustered(spatDens)
            if savePdf:
                for k in spatDens[0].keys():
                    plotRipleyResults(pdf, spatDens, k, True, str(k))
            if segmask is not None:
                invRelevSpots = filterSpots(spots, segmask, invert=True)
                spotRes["masked_spatial_density"] = maskedSpatialDensity(
                    relevSpots, invRelevSpots, size, 10, pdf
                )
            t1 = time()
            print("\tcompleted ripley estimates, elapsed time ", t - t1)

        results["spots"] = spotRes
        t = time()
        print("time for all spots metrics: " + str(t - t0))

    t1 = time()
    trRes = {}
    print("\nstarting transcript metrics")
    if segmentation is not None:
        trRes["per_cell"] = getTranscriptsPerCell(segmentation, pdf)
        trRes["FPR"] = getFPR(segmentation, pdf)
    trRes["density"] = getTranscriptDensity(transcripts, codebook)
    if spots:
        trRes["fraction_spots_used"] = getFractionSpotsUsed(relevSpots, transcripts)
    trRes["round_dist"] = getTranscriptRoundDist(transcripts, pdf)
    trRes["channel_dist"] = getTranscriptChannelDist(transcripts, pdf)
    if spots and pdf:
        plotSpotRatio(
            results["spots"]["channel_dist"]["tally"],
            trRes["channel_dist"]["counts"],
            "channel",
            pdf,
        )
        plotSpotRatio(
            results["spots"]["round_dist"]["tally"], trRes["round_dist"]["counts"], "round", pdf
        )

    results["transcripts"] = trRes
    t = time()
    print("time for all transcript metrics: " + str(t - t1))

    t = time()
    print("\nFOV Analysis complete\n\ttotal time elapsed " + str(t - t0))

    if savePdf:
        pdf.close()

    return results


def run(
    transcripts,
    codebook,
    size,
    fovs=None,
    spots=None,
    spot_threshold=None,
    segmask=None,
    segmentation=None,
    doRipley=False,
    savePdf=False,
):

    t0 = time()

    output_dir = "7_QC/"
    if not path.isdir(output_dir):
        makedirs(output_dir)

    reportFile = output_dir + datetime.now().strftime("%Y%m%d_%H%M_QC_metrics.log")
    sys.stdout = open(reportFile, "w")

    t = time()
    print("dir created " + str(t - t0))
    # print(f"fovs:\n\t{fovs}\nspots:\n\t{spots}\nsegmentation:\n\t{segmentation}\ndoRipley: {doRipley}\nsavePdf: {savePdf}")
    results = {}
    if fovs:
        for f in fovs:
            print(f"\ton fov '{f}'")
            fov_dir = "{}/{}_".format(output_dir, f)
            spot = False
            segmentOne = False
            if spots:
                spot = spots[f]
            if segmentation:
                segmentOne = segmentation[f]
            if segmask:
                segmaskOne = segmask[f]
            results[f] = runFOV(
                output_dir=fov_dir,
                transcripts=transcripts[f],
                codebook=codebook,
                size=size,
                spots=spot,
                spotThreshold=spot_threshold,
                segmask=segmaskOne,
                segmentation=segmentOne,
                doRipley=doRipley,
                savePdf=savePdf,
            )
    else:  # this only really happens if loading in pickles
        results = runFOV(
            output_dir=output_dir,
            transcripts=transcripts,
            codebook=codebook,
            size=size,
            spots=spots,
            spotThreshold=spot_threshold,
            segmask=segmask,
            segmentation=segmentation,
            doRipley=doRipley,
            savePdf=savePdf,
        )

    t = time()
    print("Analysis complete\n\ttotal time elapsed: " + str(t - t0))

    with open(output_dir + "QC_results.yml", "w") as fl:
        yaml.dump(simplifyDict(results), fl)
    print("Results saved.")

    sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":

    # disabling tdqm for pipeline runs
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    p = ArgumentParser()

    p.add_argument("--codebook-exp", type=Path)
    p.add_argument("--exp-output", type=Path)
    p.add_argument("--has-spots", dest="has_spots", action="store_true")

    p.add_argument("--codebook-pkl", type=Path)
    p.add_argument("--spots-pkl", type=Path)
    p.add_argument("--transcript-pkl", type=Path)
    p.add_argument("--segmentation-loc", type=Path, nargs="?")

    p.add_argument("--roi", type=Path)
    p.add_argument("--x-size", type=int, nargs="?")
    p.add_argument("--y-size", type=int, nargs="?")
    p.add_argument("--z-size", type=int, nargs="?")
    p.add_argument("--spot-threshold", type=float, nargs="?")
    p.add_argument("--run-ripley", dest="run_ripley", action="store_true")
    p.add_argument("--save-pdf", dest="save_pdf", action="store_true")
    args = p.parse_args()

    print(args)

    codebook = False
    roi = False

    if args.codebook_exp:
        codebook = Codebook.open_json(str(args.codebook_exp) + "/codebook.json")

        if (
            args.roi
        ):  # NOTE Going to assume 1 FOV for now. Largely used for debugging, not pipeline runs.
            exp = starfish.core.experiment.experiment.Experiment.from_json(
                str(args.codebook_exp) + "/experiment.json"
            )
            img = exp["fov_000"].get_image("primary")
            roi = BinaryMaskCollection.from_fiji_roi_set(
                path_to_roi_set_zip=args.roi, original_image=img
            )
        elif args.segmentation_loc:
            exp = starfish.core.experiment.experiment.Experiment.from_json(
                str(args.codebook_exp) + "/experiment.json"
            )
            roi = {}
            for f in exp.keys():
                maskloc = "{}/{}/mask.tiff".format(args.segmentation_loc, f)
                roi[f] = skimage.io.imread(maskloc)

    elif args.codebook_pkl:
        codebook = pickle.load(open(args.codebook_pkl, "rb"))

    transcripts = False
    if args.transcript_pkl:
        transcripts = pickle.load(open(args.transcript_pkl, "rb"))
    else:
        # load transcripts from exp dir
        transcripts = {}
        for f in glob("{}/cdf/*_decoded.cdf".format(args.exp_output)):
            name = f[len(str(args.exp_output)) + 5 : -12]
            transcripts[name] = DecodedIntensityTable.open_netcdf(f)

    segmentation = False
    if (
        args.segmentation_loc
    ):  # if this is true, going to assume baysorStaged dir-wise FOV structure
        segmentation = {}
        for name in transcripts.keys():
            segmentation[name] = pd.read_csv(
                "{}/{}/segmentation.csv".format(args.segmentation_loc, name)
            )
            # pre-filtering for nan targets, since this will crash QC code.
            segmentation[name] = segmentation[name][~segmentation[name]["target"].isna()]

    spots = False
    if args.spots_pkl:
        spots = pickle.load(open(args.spots_pkl, "rb"))
    elif args.has_spots:
        # load spots from exp dir
        spots = {}
        for k in transcripts.keys():
            spots[k] = SpotFindingResults.load(
                "{}/spots/{}_SpotFindingResults.json".format(args.exp_output, k)
            )

    spot_threshold = None
    if args.spot_threshold:
        spot_threshold = args.spot_threshold

    size = [0, 0, 0]
    if args.x_size:  # specify in CWL that all or none must be specified, only needed when doRipley
        size[0] = args.x_size
        size[1] = args.y_size
        size[2] = args.z_size

    fovs = False
    if args.exp_output:
        # reading in from experiment can have multiple FOVs
        fovs = [k for k in transcripts.keys()]
    run(
        transcripts=transcripts,
        codebook=codebook,
        size=size,
        fovs=fovs,
        spots=spots,
        spot_threshold=spot_threshold,
        segmask=roi,
        segmentation=segmentation,
        doRipley=args.run_ripley,
        savePdf=args.save_pdf,
    )
