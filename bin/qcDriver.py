#!/usr/bin/env python

import collections
import gc
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partialmethod
from glob import glob
from os import makedirs, path
from pathlib import Path
from time import time

import matplotlib.patches as mpatches
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
from matplotlib.lines import Line2D
from scipy.stats import norm, skew
from starfish import BinaryMaskCollection, Codebook, DecodedIntensityTable
from starfish.core.types import (
    PerImageSliceSpotResults,
    SpotAttributes,
    SpotFindingResults,
)
from starfish.types import Axes, Coordinates
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
    for item in spots.items():
        selectedSpots = item[1].spot_attrs.data
        selectedSpots = selectedSpots.reset_index(drop=True)

        selRow = []
        for ind, row in selectedSpots.iterrows():
            if (
                len(maskMat.shape) == 2
                and maskMat[int(row["y"]) - oneIndex][int(row["x"]) - oneIndex] == 1
            ) or (
                len(maskMat.shape) == 3
                and maskMat[int(row["z"]) - oneIndex][int(row["y"]) - oneIndex][
                    int(row["x"] - oneIndex)
                ]
                == 1
            ):
                selRow.append(ind)
            elif len(maskMat.shape) != 2 and len(maskMat.shape) != 3:
                raise Exception("Mask is not of 2 or 3 dimensions.")

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


# == Internal Metrics ==
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
    total = sum(tally)
    tally = [t / total for t in tally]
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

    return {"tally": tally, "stdev": std, "skew": skw, "total": total}


def getSpotChannelDist(spots, pdf=False):
    channelTallies = {}
    for k, v in spots.items():
        r, ch = k
        if ch not in channelTallies.keys():
            channelTallies[ch] = 0
        for i, item in v.spot_attrs.data.iterrows():
            channelTallies[ch] += 1

    tally = [channelTallies[i] for i in range(max(channelTallies.keys()) + 1)]
    total = sum(tally)
    tally = [t / total for t in tally]
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
    return {"tally": tally, "stdev": std, "skew": skw, "total": total}


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


def getTranscriptsPerCell(segmented=None, results=None, pdf=False):
    if segmented is None and results is None:
        raise Exception("Either segmented or results must be defined.")

    if segmented is not None:
        # column to look at will be different depending if we've run baysor
        if "cell" in segmented.keys():
            key = "cell"
        else:
            key = "cell_id"

        counts = pd.Series(
            collections.Counter(segmented[segmented[key].notnull()][key])
        ).sort_values(ascending=False)
    else:
        results.sort()
        results.reverse()
        counts = results

    if len(counts) == 0:
        return {"counts": [0], "quartiles": (0, 0, 0), "stdev": 0, "skew": 0}

    q1, mid, q3 = np.percentile(counts, [25, 50, 75])
    iqr_scale = 1.5

    if pdf:
        fig, ax = plt.subplots(figsize=(10, 10))

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.bar(list(range(len(counts))), counts, width=1)
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
        "counts": list(counts),
        "quartiles": [q1, mid, q3],
        "stdev": np.std(counts),
        "skew": skew(counts),
    }


def getFractionSpotsUsed(spots, transcripts):
    spotCount = spots.count_total_spots()
    trspotCount = np.count_nonzero(transcripts.data)
    return trspotCount / spotCount


def getTranscriptDist(transcripts):
    chlTally = [0 for i in range(len(transcripts.c))]
    rndTally = [0 for i in range(len(transcripts.r))]
    omitTally = [0 for i in range(len(transcripts.r))]
    for tr in range(len(transcripts)):
        for r in range(len(transcripts.r)):
            rvals = transcripts[tr].isel(r=r).data
            if not all(rvals == 0):
                rndTally[r] += 1
                chlTally[rvals.argmax()] += 1
            else:
                omitTally[r] += 1
    rndTotal = sum(rndTally)
    rndTally = [r / rndTotal for r in rndTally]
    chlTotal = sum(chlTally)
    chlTally = [r / chlTotal for r in chlTally]
    omitTotal = sum(omitTally)
    results = {
        "round": {
            "tally": rndTally,
            "stdev": np.std(rndTally),
            "skew": skew(rndTally),
            "total": rndTotal,
        },
        "channel": {
            "tally": chlTally,
            "stdev": np.std(chlTally),
            "skew": skew(chlTally),
            "total": chlTotal,
        },
    }
    if omitTotal != 0:
        omitTally = [r / omitTotal for r in omitTally]
        results["omit_round"] = {
            "tally": omitTally,
            "stdev": np.std(omitTally),
            "skew": skew(omitTally),
            "total": omitTotal,
        }
    return results


def plotTranscriptDist(counts, name, pdf, transcripts=True):
    std = np.std(counts)
    fig, ax = plt.subplots()

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.bar(range(len(counts)), counts)
    if transcripts:
        plt.title(f"Transcript source spot distribution across {name}s")
    else:
        plt.title(f"Spot distribution across {name}s")
        # we use this method with spots when using the combined fov.
    plt.ylabel("Spot count")
    plt.xlabel(f"{name} ID")

    avg = np.mean(counts)
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
    plt.axhline(avg, color="black")
    plt.text(0, avg + offset, f"Average: {avg:.2f}")
    plt.axhline(avg + std, dashes=(2, 2), color="black")
    plt.axhline(avg - std, dashes=(2, 2), color="black")
    plt.text(0, avg - std + offset, f"Standard Deviation: {std:.2f}")

    pdf.savefig(fig)
    plt.close()


# New version that plot bars in order of height
def getFPR(segmentation=None, results=None, pdf=False):
    if segmentation is None and results is None:
        raise Exception("Either segmentation or results must be provided.")

    if segmentation is not None:
        # remove unassigned transcripts, if the columns to do so are present.
        key = "cell"
        if "cell_id" in segmentation.keys():
            key = "cell_id"
        segmentation = segmentation[segmentation[key].notnull()]

        # Get real and blank target counts per cell for all barcodes
        blank_counts_all = segmentation[segmentation["target"].str.contains("blank", case=False)]
        real_counts_all = segmentation[~segmentation["target"].str.contains("blank", case=False)]
        real_per_cell_all = pd.Series(collections.Counter(real_counts_all[key]))
        blank_per_cell_all = pd.Series(collections.Counter(blank_counts_all[key]))

        # Add in cells that have reals codes but not blank codes and vice versa
        all_cells = set(segmentation[key])
        empty_cells = pd.Series(
            {cell: 0 for cell in all_cells if cell not in real_per_cell_all.index}
        )
        real_per_cell_all = real_per_cell_all.append(empty_cells)
        empty_cells = pd.Series(
            {cell: 0 for cell in all_cells if cell not in blank_per_cell_all.index}
        )
        blank_per_cell_all = blank_per_cell_all.append(empty_cells)
    else:
        real_per_cell_all = pd.Series(results["reals_all"])
        blank_per_cell_all = pd.Series(results["blanks_all"])

    # Sort counts by all real target counts
    sorted_reals_all = real_per_cell_all.sort_values(ascending=False)
    sorted_blanks_all = blank_per_cell_all[sorted_reals_all.index]

    if (sum(blank_per_cell_all) + sum(real_per_cell_all)) > 0:
        final_results = {
            "FP": sum(blank_per_cell_all),
            "TP": sum(real_per_cell_all),
            "FPR": sum(blank_per_cell_all) / (sum(blank_per_cell_all) + sum(real_per_cell_all)),
            "tally": {
                "reals_all": list(sorted_reals_all),
                "blanks_all": list(sorted_blanks_all),
            },
        }
    else:
        print("No cells within boundaries, was segmentation performed properly?")
        return {}

    # If error-correction is used, do the same for only non-error-corrected barcodes
    if (segmentation is not None and "corrected_rounds" in segmentation.keys()) or (
        results is not None and "reals_full" in results.keys()
    ):
        if segmentation is not None:
            # Get counts per cell
            full_counts = segmentation[segmentation["corrected_rounds"] == 0]
            blank_counts_full = full_counts[
                full_counts["target"].str.contains("blank", case=False)
            ]
            real_counts_full = full_counts[
                ~full_counts["target"].str.contains("blank", case=False)
            ]
            real_per_cell_full = pd.Series(collections.Counter(real_counts_full[key]))
            blank_per_cell_full = pd.Series(collections.Counter(blank_counts_full[key]))

            # Add in empty cells
            empty_cells = pd.Series(
                {cell: 0 for cell in all_cells if cell not in real_per_cell_full.index}
            )
            real_per_cell_full = real_per_cell_full.append(empty_cells)
            empty_cells = pd.Series(
                {cell: 0 for cell in all_cells if cell not in blank_per_cell_full.index}
            )
            blank_per_cell_full = blank_per_cell_full.append(empty_cells)
        else:
            real_per_cell_full = pd.Series(results["reals_full"])
            blank_per_cell_full = pd.Series(results["blanks_full"])

        # Sort
        sorted_reals_full = real_per_cell_full[sorted_reals_all.index]
        sorted_blanks_full = blank_per_cell_full[sorted_reals_all.index]

        final_results["tally"]["reals_full"] = list(sorted_reals_full)
        final_results["tally"]["blanks_full"] = list(sorted_blanks_full)

    if pdf:
        fig, ax = plt.subplots()

        all_on_color = (0 / 256, 153 / 256, 136 / 256)
        all_off_color = (238 / 256, 119 / 256, 51 / 256)
        full_on_color = (0 / 256, 119 / 256, 187 / 256)
        full_off_color = (204 / 256, 51 / 256, 17 / 256)

        # This next part orders the values and colors so that taller bars are plotted first and shorter bars later
        # to ensure that as many bars as possible are visible. Ties are broken according to the category order in
        # the hierarchy variable.
        per_cell_values = []
        per_cell_colors = []
        if (segmentation is not None and "corrected_rounds" in segmentation.keys()) or (
            results is not None and "reals_full" in results.keys()
        ):
            hierarchy = ["all_on", "full_on", "all_off", "full_off"][::-1]
            for ind in sorted_reals_all.index:
                values = pd.Series(
                    [
                        sorted_reals_all[ind],
                        sorted_reals_full[ind],
                        sorted_blanks_all[ind],
                        sorted_blanks_full[ind],
                    ],
                    index=["all_on", "full_on", "all_off", "full_off"],
                )
                sorted_values = values.sort_values(ascending=False)
                colors = pd.Series(
                    [all_on_color, full_on_color, all_off_color, full_off_color],
                    index=["all_on", "full_on", "all_off", "full_off"],
                )[sorted_values.index]
                counter = collections.Counter(sorted_values)
                if sum(np.array(list(counter.values())) > 1) > 0:
                    for value in counter:
                        if counter[value] > 1:
                            order = sorted_values[
                                [
                                    x
                                    for x in hierarchy
                                    if x in sorted_values[sorted_values == value].index
                                ]
                            ]
                            value1_ind = np.where(sorted_values.index == order.index[0])[0][0]
                            value2_ind = np.where(sorted_values.index == order.index[1])[0][0]
                            if value1_ind > value2_ind:
                                new_index = list(sorted_values.index)
                                new_index[value1_ind] = sorted_values.index[value2_ind]
                                new_index[value2_ind] = sorted_values.index[value1_ind]
                                sorted_values = sorted_values[new_index]
                                colors = colors[new_index]
                per_cell_colors.append(list(colors))
                per_cell_values.append(list(sorted_values))

            bars1 = [values[0] for values in per_cell_values]
            bars1_colors = [colors[0] for colors in per_cell_colors]
            bars2 = [values[1] for values in per_cell_values]
            bars2_colors = [colors[1] for colors in per_cell_colors]
            bars3 = [values[2] for values in per_cell_values]
            bars3_colors = [colors[2] for colors in per_cell_colors]
            bars4 = [values[3] for values in per_cell_values]
            bars4_colors = [colors[3] for colors in per_cell_colors]

        else:
            hierarchy = ["all_on", "all_off"][::-1]
            for ind in sorted_reals_all.index:
                values = pd.Series(
                    [sorted_reals_all[ind], sorted_blanks_all[ind]], index=["all_on", "all_off"]
                )
                sorted_values = values.sort_values(ascending=False)
                colors = pd.Series([all_on_color, all_off_color], index=["all_on", "all_off"])[
                    sorted_values.index
                ]
                counter = collections.Counter(sorted_values)
                if sum(np.array(list(counter.values())) > 1) > 0:
                    for value in counter:
                        if counter[value] > 1:
                            order = sorted_values[
                                [
                                    x
                                    for x in hierarchy
                                    if x in sorted_values[sorted_values == value].index
                                ]
                            ]
                            value1_ind = np.where(sorted_values.index == order.index[0])[0][0]
                            value2_ind = np.where(sorted_values.index == order.index[1])[0][0]
                            if value1_ind > value2_ind:
                                new_index = list(sorted_values.index)
                                new_index[value1_ind] = sorted_values.index[value2_ind]
                                new_index[value2_ind] = sorted_values.index[value1_ind]
                                sorted_values = sorted_values[new_index]
                                colors = colors[new_index]
                per_cell_colors.append(list(colors))
                per_cell_values.append(list(sorted_values))

            bars1 = [values[0] for values in per_cell_values]
            bars1_colors = [colors[0] for colors in per_cell_colors]

            bars2 = [values[1] for values in per_cell_values]
            bars2_colors = [colors[1] for colors in per_cell_colors]

        plt.bar(
            range(len(bars1)),
            bars1,
            width=1,
            align="edge",
            color=bars1_colors,
        )
        plt.bar(
            range(len(bars2)),
            bars2,
            width=1,
            align="edge",
            color=bars2_colors,
        )
        if (segmentation is not None and "corrected_rounds" in segmentation.keys()) or (
            results is not None and "reals_full" in results.keys()
        ):
            plt.bar(
                range(len(bars3)),
                bars3,
                width=1,
                align="edge",
                color=bars3_colors,
            )
            plt.bar(
                range(len(bars4)),
                bars4,
                width=1,
                align="edge",
                color=bars4_colors,
            )
        plt.plot(
            [0, len(real_per_cell_all)],
            [np.median(real_per_cell_all), np.median(real_per_cell_all)],
            color="black",
            linewidth=3,
        )
        if (segmentation is not None and "corrected_rounds" in segmentation.keys()) or (
            results is not None and "reals_full" in results.keys()
        ):
            plt.plot(
                [0, len(real_per_cell_full)],
                [np.median(real_per_cell_full), np.median(real_per_cell_full)],
                color="black",
                linestyle="dashed",
                linewidth=3,
            )

        # Create and plot legend
        if (segmentation is not None and "corrected_rounds" in segmentation.keys()) or (
            results is not None and "reals_full" in results.keys()
        ):
            proxy_all_on = mpatches.Patch(color=all_on_color, label="On-target EC+NC")
            proxy_full_on = mpatches.Patch(color=full_on_color, label="On-target NC")
            proxy_all_off = mpatches.Patch(color=all_off_color, label="Off-target EC+NC")
            proxy_full_off = mpatches.Patch(color=full_off_color, label="Off-target NC")
            solid_line = Line2D([0], [0], color="black", linestyle="solid", label="EC+NC Median")
            dashed_line = Line2D([0], [0], color="black", linestyle="dashed", label="NC Median")
            handles = [
                proxy_all_on,
                proxy_full_on,
                proxy_all_off,
                proxy_full_off,
                solid_line,
                dashed_line,
            ]
        else:
            proxy_on = mpatches.Patch(color=all_on_color, label="On-target")
            proxy_off = mpatches.Patch(color=all_off_color, label="Off-target")
            solid_line = Line2D(
                [0], [0], color="black", linestyle="solid", label="On-target Median"
            )
            handles = [proxy_on, proxy_off, solid_line]
        plt.legend(handles=handles)

        plt.xlabel("Cells")
        plt.ylabel("Total barcodes per cell")
        plt.xlim([0, len(real_per_cell_all)])
        plt.ylim([0, max(max(real_per_cell_all), max(blank_per_cell_all)) * 1.05])
        plt.title("True positives vs False positives")

        pdf.savefig(fig)
        plt.close()

    return final_results


def get_y_offset(cutoff, shift, ax):
    offset = ax.transData.transform((0, cutoff))[1] + shift
    return ax.transData.inverted().transform((0, offset))[1]


def plotBarcodeAbundance(pdf, decoded=None, results=None):
    if decoded is None and results is None:
        raise Exception("Either decoded or results must be defined.")

    fig, ax = plt.subplots()

    if decoded is not None:
        # Count targets and sort them in descending order
        targets = decoded["target"].data.tolist()
        all_counts = pd.Series(collections.Counter(targets)).sort_values(ascending=False)
    else:
        fulldict = results["reals_all"]
        fulldict.update(results["blanks_all"])
        all_counts = pd.Series(fulldict).sort_values(ascending=False)

    # Set scale, axis limits, axis labels, and plot title
    ax.set_yscale("log")
    plt.xlim([0, len(all_counts)])
    plt.ylim([0.9, max(all_counts) * 5])
    plt.xlabel("Barcodes")
    plt.ylabel("Total counts per barcode")
    plt.title("Relative abundance of barcodes")

    # Get range of display values (for plotting text later)
    disp_min = ax.transData.transform((0, 0.9))[1]
    disp_max = ax.transData.transform((0, max(all_counts) * 1.1))[1]
    disp_range = disp_max - disp_min

    # Assign colors to sorted targets based on whether they are blanks or real
    all_on_color = (0 / 256, 153 / 256, 136 / 256)
    all_off_color = (238 / 256, 119 / 256, 51 / 256)
    all_colors = [
        all_on_color if "blank" not in target.lower() else all_off_color
        for target in all_counts.index
    ]

    # Calculate upper 95% CI for blank codes and what proportion of real codes are above this value
    if decoded is not None:
        all_blank_counts_raw = collections.Counter([s for s in targets if "blank" in s.lower()])
        all_blank_counts = pd.Series(all_blank_counts_raw)
        all_real_counts_raw = collections.Counter([s for s in targets if "blank" not in s.lower()])
        all_real_counts = pd.Series(all_real_counts_raw)
    else:
        all_blank_counts_raw = results["blanks_all"]
        all_blank_counts = list(results["blanks_all"].values())
        all_real_counts_raw = results["reals_all"]
        all_real_counts = list(results["reals_all"].values())

    all_avg_bl = np.average(all_blank_counts)
    all_std_bl = max(1, np.std(all_blank_counts))
    all_conf = norm.interval(0.95, loc=all_avg_bl, scale=all_std_bl)[1]
    good_codes_all = sum(all_real_counts > all_conf) / len(all_real_counts)

    # Plot bars, upper 95% CI line, and text
    plt.bar(range(len(all_counts)), height=all_counts, color=all_colors, width=1, align="edge")
    plt.axhline(all_conf, color="black", label="Upper 95% CI EC+NC")
    plt.plot(
        [len(all_counts) * 0.4, len(all_counts) * 0.4],
        [all_conf, get_y_offset(all_conf, disp_range / 4.35, ax)],
        color="black",
    )
    top_txt_y = get_y_offset(all_conf, disp_range / 3.95, ax)
    plt.text(
        len(all_counts) * 0.4,
        top_txt_y,
        f"{good_codes_all*100:.2f}% barcodes above {all_conf:.2f} threshold",
        horizontalalignment="center",
        fontsize=8,
    )

    final_results = {
        "reals_all": dict(all_real_counts_raw),
        "blanks_all": dict(all_blank_counts_raw),
    }

    # Do all the same for only non-corrected barcodes if error-corrected barcodes are present
    if (decoded is not None and "corrected_rounds" in decoded.coords) or (
        results is not None and "reals_full" in results.keys()
    ):
        if decoded is not None:
            targets = decoded[decoded["corrected_rounds"] == 0]["target"].data.tolist()
            full_counts = pd.Series(collections.Counter(targets)).sort_values(ascending=False)
        else:
            fulldict = results["reals_full"]
            fulldict.update(results["blanks_full"])
            full_counts = pd.Series(fulldict.values()).sort_values(ascending=False)

        full_on_color = (0 / 256, 119 / 256, 187 / 256)
        full_off_color = (204 / 256, 51 / 256, 17 / 256)
        full_colors = [
            full_on_color if "blank" not in target.lower() else full_off_color
            for target in all_counts.index
        ]

        if decoded is not None:
            full_blank_counts_raw = collections.Counter(
                [s for s in targets if "blank" in s.lower()]
            )
            full_blank_counts = pd.Series(full_blank_counts_raw)
            full_real_counts_raw = collections.Counter(
                [s for s in targets if "blank" not in s.lower()]
            )
            full_real_counts = pd.Series(full_real_counts_raw)
        else:
            full_blank_counts_raw = results["blanks_full"]
            full_blank_counts = list(results["blanks_full"].values())
            full_real_counts_raw = results["reals_full"]
            full_real_counts = list(results["reals_full"].values())

        avg_bl = np.average(full_blank_counts)
        std_bl = max(1, np.std(full_blank_counts))
        full_conf = norm.interval(0.95, loc=avg_bl, scale=std_bl)[1]
        good_codes_full = sum(full_real_counts > full_conf) / len(full_real_counts)

        plt.bar(
            range(len(full_counts)), height=full_counts, color=full_colors, width=1, align="edge"
        )
        plt.axhline(full_conf, color="black", label="Upper 95% CI NC", linestyle="dashed")
        plt.plot(
            [len(all_counts) * 0.7, len(all_counts) * 0.7],
            [full_conf, get_y_offset(all_conf, disp_range / 10.9, ax)],
            color="black",
            linestyle="dashed",
        )
        plt.text(
            len(all_counts) * 0.7,
            get_y_offset(all_conf, disp_range / 8.7, ax),
            f"{good_codes_full*100:.2f}% barcodes above {full_conf:.2f} threshold",
            horizontalalignment="center",
            fontsize=8,
        )

        final_results["reals_full"] = dict(full_real_counts_raw)
        final_results["blanks_full"] = dict(full_blank_counts_raw)

    if top_txt_y > max(all_counts) * 1.1:
        plt.ylim([0.9, top_txt_y * 2])

    # Create and plot legend
    if (decoded is not None and "corrected_rounds" in decoded.coords) or (
        results is not None and "reals_full" in results.keys()
    ):
        proxy_positive_all = mpatches.Patch(color=all_on_color, label="On-target EC+NC")
        proxy_positive_full = mpatches.Patch(color=full_on_color, label="On-target NC")
        proxy_blank_all = mpatches.Patch(color=all_off_color, label="Off-target EC+NC")
        proxy_blank_full = mpatches.Patch(color=full_off_color, label="Off-target NC")
        solid_line = Line2D([0], [0], color="black", linestyle="solid", label="Upper 95% CI EC+NC")
        dashed_line = Line2D([0], [0], color="black", linestyle="dashed", label="Upper 95% CI NC")
        handles = [
            proxy_positive_all,
            proxy_positive_full,
            proxy_blank_all,
            proxy_blank_full,
            solid_line,
            dashed_line,
        ]
    else:
        proxy_positive = mpatches.Patch(color=all_on_color, label="On-target")
        proxy_blank = mpatches.Patch(color=all_off_color, label="Off-target")
        handles = [proxy_positive, proxy_blank]
    lgd = ax.legend(handles=handles, loc=(1.02, 0.5))

    pdf.savefig(fig, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()

    return {
        "cutoff": all_conf,
        "barcode_average": all_avg_bl,
        "barcode_std_used": all_std_bl,
        "tally": final_results,
    }


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


def flatten(lis):
    return [item for sublist in lis for item in sublist]


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
        trRes["per_cell"] = getTranscriptsPerCell(segmented=segmentation, pdf=pdf)
        trRes["FPR"] = getFPR(segmentation=segmentation, pdf=pdf)
    trRes["total"] = np.shape(transcripts.data)[0]
    trRes["density"] = getTranscriptDensity(transcripts, codebook)
    if spots:
        trRes["fraction_spots_used"] = getFractionSpotsUsed(relevSpots, transcripts)

    targets = [""]
    if len(transcripts["target"].str.contains("blank", case=False)) > 0:
        targets.append("_noblank")

    trRes["barcode_counts"] = plotBarcodeAbundance(decoded=transcripts, pdf=pdf)

    for t in targets:
        cur_trs = transcripts
        if t == "_noblank":
            cur_trs = transcripts[~transcripts["target"].str.contains("blank", case=False)]
        trDist = getTranscriptDist(cur_trs)
        for k, v in trDist.items():
            trRes[f"{k}{t}_dist"] = v
        if pdf:
            plotTranscriptDist(trRes[f"round{t}_dist"]["tally"], f"round{t}", pdf)
            plotTranscriptDist(trRes[f"channel{t}_dist"]["tally"], f"channel{t}", pdf)
        if spots and pdf:
            plotSpotRatio(
                results["spots"]["channel_dist"]["tally"],
                trRes[f"channel{t}_dist"]["tally"],
                f"channel{t}",
                pdf,
            )
            plotSpotRatio(
                results["spots"]["round_dist"]["tally"],
                trRes[f"round{t}_dist"]["tally"],
                f"round{t}",
                pdf,
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
        # don't bother running these if we only have one fov.
        if len(fovs) > 1:
            # running combined analysis on FOVs
            # a bit hacky, but we are less likely to run into memory issues if we take the `results`
            # output from the prior analyses and stich that together, compared to combining all
            # prior results objects.
            lens = {
                "round": len(results[fovs[0]]["transcripts"]["round_dist"]["tally"]),
                "channel": len(results[fovs[0]]["transcripts"]["channel_dist"]["tally"]),
            }
            if savePdf:
                pdf = PdfPages(output_dir + "combined_graph_output.pdf")
            else:
                pdf = None

            print("Computing QC for combined FOVs")

            dims = [["transcripts", "round"], ["transcripts", "channel"]]

            if "round_noblank_dist" in results[fovs[0]]["transcripts"].keys():
                dims.extend([["transcripts", "round_noblank"], ["transcripts", "channel_noblank"]])

            if "omit_round_dist" in results[fovs[0]]["transcripts"].keys():
                dims.extend([["transcripts", "omit_round"]])
                if "round_noblank_dist" in results[fovs[0]]["transcripts"].keys():
                    dims.extend([["transcripts", "omit_round_noblank"]])

            if spots:
                # these must be after the transcript dims
                # because we only check if we're on spots to decide to plot spot ratio
                dims.extend([["spots", "round"], ["spots", "channel"]])

            newRes = {}
            for target, axis in dims:
                # target: transcripts vs spots
                # axis: rounds vs channels
                thisLen = lens[
                    "round" if "round" in axis else "channel"
                ]  # don't want this to break with _noblank or omit_
                print(f"\tComputing combined {target}:{axis} distributions")
                total = sum([results[f][target][f"{axis}_dist"]["total"] for f in fovs])
                tally = [
                    sum(
                        [
                            results[f][target][f"{axis}_dist"]["tally"][i]
                            * results[f][target][f"{axis}_dist"]["total"]
                            for f in fovs
                        ]
                    )
                    / total
                    for i in range(thisLen)
                ]
                localRes = {"tally": tally, "stdev": np.std(tally), "skew": skew(tally)}
                if target not in newRes.keys():
                    newRes[target] = {}
                newRes[target][f"{axis}_dist"] = localRes
                if pdf:
                    plotTranscriptDist(tally, axis, pdf, target == "transcripts")
                    if target == "spots":
                        # if target is spots, then we can directly plot them with transcripts
                        plotSpotRatio(
                            tally,
                            newRes["transcripts"][f"{axis}_dist"]["tally"],
                            f"{axis}",
                            pdf,
                        )
                        if f"{axis}_noblank_dist" in newRes["transcripts"].keys():
                            # include noblank comparison
                            plotSpotRatio(
                                tally,
                                newRes["transcripts"][f"{axis}_noblank_dist"]["tally"],
                                f"{axis}_noblanks",
                                pdf,
                            )

            if spots:
                # recompute spot based metrics by combining across fovs
                print("\tComputing spot metrics")
                spotRes = {}
                spotRes["spot_counts"] = {
                    "total_count": sum(
                        [results[f]["spots"]["spot_counts"]["total_count"] for f in fovs]
                    ),
                    "segmented_count": sum(
                        [results[f]["spots"]["spot_counts"]["segmented_count"] for f in fovs]
                    ),
                }
                spotRes["spot_counts"]["ratio"] = (
                    spotRes["spot_counts"]["segmented_count"]
                    / spotRes["spot_counts"]["total_count"]
                )
                spotRes["density"] = spotRes["spot_counts"]["total_count"] / len(codebook)

            print("\tComputing transcript metrics")

            trRes = {}
            trRes["total"] = sum([results[f]["transcripts"]["total"] for f in fovs])
            trRes["density"] = (
                sum(
                    [
                        results[f]["transcripts"]["density"] * results[f]["transcripts"]["total"]
                        for f in fovs
                    ]
                )
                / trRes["total"]
            )

            if spots:
                trRes["fraction_spots_used"] = (
                    sum(
                        [
                            results[f]["transcripts"]["fraction_spots_used"]
                            * results[f]["transcripts"]["total"]
                            for f in fovs
                        ]
                    )
                    / trRes["total"]
                )

            if segmentation:
                print("\tRe-computing segmentation based metrics")
                cell_counts = flatten(
                    [results[f]["transcripts"]["per_cell"]["counts"] for f in fovs]
                )
                trRes["per_cell"] = getTranscriptsPerCell(results=cell_counts, pdf=pdf)
                if "FPR" in results[fovs[0]]["transcripts"].keys():
                    FPR_raw = {}
                    for k in results[fovs[0]]["transcripts"]["FPR"]["tally"].keys():
                        FPR_raw[k] = flatten(
                            [
                                results[f]["transcripts"]["FPR"]["tally"][k]
                                for f in fovs
                                if "FPR" in results[f]["transcripts"].keys()
                                and "tally" in results[f]["transcripts"]["FPR"]
                            ]
                        )
                    trRes["FPR"] = getFPR(results=FPR_raw, pdf=pdf)

            print("\tRe-computing barcode metrics")
            barcodeTallies = {}
            for k in results[fovs[0]]["transcripts"]["barcode_counts"]["tally"].keys():
                # take full list of barcodes across fovs, because barcodes with a count of 0 in a given fov won't be listed
                superBars = set(
                    flatten(
                        [
                            list(results[f]["transcripts"]["barcode_counts"]["tally"][k].keys())
                            for f in fovs
                        ]
                    )
                )
                # sum across fovs for each barcode
                barcodeTallies[k] = {
                    bar: sum(
                        [
                            results[f]["transcripts"]["barcode_counts"]["tally"][k][bar]
                            for f in fovs
                            if (
                                bar
                                in results[f]["transcripts"]["barcode_counts"]["tally"][k].keys()
                                and ("blank" in k or "blank" not in bar)
                            )
                        ]
                    )
                    for bar in superBars
                }
            trRes["barcode_counts"] = plotBarcodeAbundance(results=barcodeTallies, pdf=pdf)

            # don't forget to re-add the parametrized dist metrics
            trRes.update(newRes["transcripts"])
            if spots:
                spotRes.update(newRes["spots"])
                results["combined"] = {"spots": spotRes, "transcripts": trRes}
            else:
                results["combined"] = {"transcripts": trRes}

            if savePdf:
                pdf.close()

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
    p.add_argument("--selected-fovs", nargs="+", const=None)
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

    transcripts = {}
    if args.transcript_pkl:
        transcripts = pickle.load(open(args.transcript_pkl, "rb"))
    else:
        # load transcripts from exp dir
        for f in glob("{}/cdf/*_decoded.cdf".format(args.exp_output)):
            name = f[len(str(args.exp_output)) + 5 : -12]
            if "comp" not in name:
                transcripts[name] = DecodedIntensityTable.open_netcdf(f)

    segmentation = None
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

    spots = None
    if args.spots_pkl:
        spots = pickle.load(open(args.spots_pkl, "rb"))
    elif args.has_spots:
        # load spots from exp dir
        spots = {}
        for k in transcripts.keys():
            spots[k] = SpotFindingResults.load(
                "{}/spots/{}_SpotFindingResults.json".format(args.exp_output, k)
            )

    if args.codebook_exp:
        codebook = Codebook.open_json(str(args.codebook_exp) + "/codebook.json")

        if (
            args.roi
        ):  # NOTE Going to assume 1 FOV for now. Largely used for debugging, not pipeline runs.
            exp = starfish.core.experiment.experiment.Experiment.from_json(
                str(args.codebook_exp) + "/experiment.json"
            )
            img = exp[list(exp.keys())[0]].get_image("primary")
            roi = BinaryMaskCollection.from_fiji_roi_set(
                path_to_roi_set_zip=args.roi, original_image=img
            )
        elif args.segmentation_loc:
            roi = {}
            for f in transcripts.keys():
                maskloc = "{}/{}/mask.tiff".format(args.segmentation_loc, f)
                roi[f] = skimage.io.imread(maskloc)
                # If 3D check dimensions and make sure always (z, y, x). skimage sometimes
                # moves the z dimension last.
                if len(roi[f].shape) == 3:
                    if roi[f].shape[0] > roi[f].shape[2]:
                        roi[f] = np.transpose(roi[f], axes=[2, 0, 1])

    elif args.codebook_pkl:
        codebook = pickle.load(open(args.codebook_pkl, "rb"))

    spot_threshold = None
    if args.spot_threshold:
        spot_threshold = args.spot_threshold

    size = [0, 0, 0]
    if args.x_size:  # specify in CWL that all or none must be specified, only needed when doRipley
        size[0] = args.x_size
        size[1] = args.y_size
        size[2] = args.z_size

    fovs = False
    if args.selected_fovs is not None:
        # manually specified FOVs override anything else
        fovs = ["fov_{:03}".format(int(f)) for f in args.selected_fovs]
    elif args.exp_output:
        # reading in from experiment can have multiple FOVs
        fovs = [k for k in transcripts.keys()]
    if not args.exp_output or len(transcripts.keys()) > 0:
        # only run QC if we actually have input.
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
