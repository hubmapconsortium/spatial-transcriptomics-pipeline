#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from datetime import datetime
from os.path import join as pathjoin

import SimpleITK as sitk


class ImageTransformer:
    def __init__(self, destinationImageFiles, originImageFiles):
        self.checkDestinationImage(destinationImageFiles=destinationImageFiles)
        self.checkOriginImage(originImageFiles=originImageFiles)

    def checkDestinationImage(self, destinationImageFiles=None):
        try:
            sitk.ReadImage(destinationImageFiles)
            self.destinationImageFiles = destinationImageFiles
        except:
            raise FileNotFoundError("Destination image could not be read")

    def checkOriginImage(self, originImageFiles=None):
        try:
            sitk.ReadImage(originImageFiles)
            self.originImageFiles = originImageFiles
        except:
            raise FileNotFoundError("Origin image could not be read")

    def findTransformParameters(
        self,
        transform="affine",
        NumberOfResolutions=7,
        MaximumNumberOfIterations=1000,
        NumberOfSpatialSamples=4000,
    ):
        """running elastix on destination and origin images to find the transform parameter map between them"""
        self.transform = transform
        self.NumberOfResolutions = NumberOfResolutions
        self.MaximumNumberOfIterations = MaximumNumberOfIterations
        self.NumberOfSpatialSamples = NumberOfSpatialSamples

        self.elastixImageFilter = (
            sitk.ElastixImageFilter()
        )  # The basic object to do the transformation

        """ Setting the transformation parameters"""
        parameterMap = self.elastixImageFilter.GetDefaultParameterMap(
            self.transform
        )  # getting the dafault parameter map for our transformation of interest
        parameterMap["NumberOfHistogramBins"] = [
            "64"
        ]  # a parameter for the image comparison metric, AdvancedMattesMutualInformation, that we are using.
        parameterMap["MaximumNumberOfIterations"] = [
            str(self.MaximumNumberOfIterations)
        ]  # number of iterations per aligning each resolution
        parameterMap["NumberOfResolutions"] = [
            str(self.NumberOfResolutions)
        ]  # number of resolution-decreasing alignments. This is the most critical parameter
        parameterMap["NumberOfSpatialSamples"] = [
            str(self.NumberOfSpatialSamples)
        ]  # number of random samples drawn for image comparison during optimization
        parameterMap["WriteIterationInfo"] = [
            "true"
        ]  # This command writes the report in the current working directory, so we have to move the files later

        self.elastixImageFilter.SetParameterMap(
            parameterMap
        )  # setting the parameter map to our transformation object
        self.elastixImageFilter.SetMovingImage(
            self.readOriginImage()
        )  # Setting the origin image, the one we want to transform
        self.elastixImageFilter.SetFixedImage(
            self.readDestinationImage()
        )  # Setting the destination/final image
        self.elastixImageFilter.Execute()  # running the transformation
        self.transformParameterMap = (
            self.elastixImageFilter.GetTransformParameterMap()
        )  # saving the optimized transformation parameters

    def readOriginImage(self):
        origin3D = sitk.ReadImage(self.originImageFiles)
        origin2D = sitk.Extract(
            origin3D, (origin3D.GetWidth(), origin3D.GetHeight(), 0), (0, 0, 0)
        )
        return origin2D

    def readDestinationImage(self):
        dest3D = sitk.ReadImage(self.destinationImageFiles)
        dest2D = sitk.Extract(dest3D, (dest3D.GetWidth(), dest3D.GetHeight(), 0), (0, 0, 0))
        return dest2D

    def writeParameterFile(self, reportName):
        self.elastixImageFilter.WriteParameterFile(
            parameterMap=self.transformParameterMap[0], filename=reportName
        )

    def getTransformParameterMap(self):
        return self.transformParameterMap


class TwoDimensionalAligner:
    """Objects of this class align images taken from one "origin" cycle to images taken from "destination" cycle of the (probably) same position"""

    def __init__(
        self,
        originImagesFolder,
        destinationImagesFolder,
        originMatchingChannel,
        destinationMatchingChannel,
        imagesPosition,
        destinationCycle,
        originCycle,
        resultDirectory,
        MaximumNumberOfIterations=500,
    ):
        self.originImagesFolder = originImagesFolder
        self.destinationImagesFolder = destinationImagesFolder
        self.originMatchingChannel = originMatchingChannel
        self.destinationMatchingChannel = destinationMatchingChannel
        self.imagesPosition = imagesPosition
        self.destinationCycle = destinationCycle
        self.originCycle = originCycle
        self.MaximumNumberOfIterations = MaximumNumberOfIterations

        self.resultDirectory = resultDirectory
        if os.path.isdir(self.resultDirectory) == False:
            os.makedirs(self.resultDirectory)

        """ From the given folders, find image file names that we'll use for computing the transformation """
        self.setupInputFiles()

        self.setImageTransformer()  # finding the transformation between origin and destination images using the channel of interest

        self.transformAllOriginImages()  # transforming all origin images to destination images configuration

    def setImageTransformer(self):
        """This object will be our transformer from origin cycle to destination cycle"""
        print(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ") + self.destinationImagesFolder)
        self.imageTransformer = ImageTransformer(
            destinationImageFiles=[
                pathjoin(self.destinationImagesFolder, dsImgFile)
                for dsImgFile in self.destinationImageFilesByChannel[
                    self.destinationMatchingChannel
                ]
            ],
            originImageFiles=[
                pathjoin(self.originImagesFolder, ogImgFile)
                for ogImgFile in self.originImageFilesByChannel[self.originMatchingChannel]
            ],
        )
        print(
            datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ") + "Finding transform parameter started"
        )
        self.imageTransformer.findTransformParameters(
            transform="affine", MaximumNumberOfIterations=self.MaximumNumberOfIterations
        )
        print(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ") + "Finding transform parameter done")

        """ Creating a directory called metadata and write transformation parameters to it """
        if os.path.isdir(pathjoin(self.resultDirectory, "MetaData")) == False:
            os.mkdir(pathjoin(self.resultDirectory, "MetaData"))
        self.imageTransformer.writeParameterFile(
            reportName=pathjoin(
                self.resultDirectory,
                "MetaData",
                str(self.imagesPosition) + "_transformation report.txt",
            )
        )  # write the parameter map to folder MetaData

        # """ move every file in origin MetaData folder to the new MetaData folder """
        # import shutil
        # for file in os.listdir(pathjoin(self.originImagesFolder, "MetaData")):
        #    shutil.copy(src = pathjoin(self.originImagesFolder, "MetaData", file), dst = pathjoin(self.resultDirectory, "MetaData"))

    def setupInputFiles(self):
        """setting up origin file addresses"""
        originFolder_files = os.listdir(
            self.originImagesFolder
        )  # listing all files in the originImagesFolder

        """ finding the file names that corresponds to our desired cycle """
        #         relatedOriginImagesRE = re.compile(r"(Position" + self.imagesPosition + r")_z(\d+)_(ch\d+)(.tif)") # group(0) = string, group(1) = position, group(2) = z, group(3) = channel, group(4) = .tif
        #        relatedOriginImagesRE = re.compile(r"(" + self.imagesPosition + r")_z(\d+)_(ch\d+)(.tif)") # group(0) = whole string, group(1) = position, group(2) = z, group(3) = channel, group(4) = .tif
        relatedOriginImagesRE = re.compile(
            r"MIP_(" + self.originCycle + r")_(FOV\d+)_(ch\d+)(.tif)"
        )  # group(0) = whole string, group(1) = cycle, group(2) = position, group(3) = channel, group(4) = .tif
        print(relatedOriginImagesRE)
        originImageFiles_splitted = [
            relatedOriginImagesRE.search(filename) for filename in originFolder_files
        ]
        originImageFiles_splitted = [
            x for x in originImageFiles_splitted if x is not None
        ]  # name of all files in the origin directory splitted by above regex constraints
        originImageFiles_splitted.sort(
            key=lambda x: x.group(3)
        )  # sorting the file names based on channel values

        """ We have to group the origin images by channel, so that we used them for transformation later """
        self.originImageFilesByChannel = {}
        for channel in set(
            [x.group(3) for x in originImageFiles_splitted]
        ):  # iterate over the unique entries of channels
            self.originImageFilesByChannel[channel] = [
                x.group(0) for x in originImageFiles_splitted if x.group(3) == channel
            ]  # selecting those names that have the same channel

        """ setting up destination file addresses """
        destinationFolder_files = os.listdir(
            self.destinationImagesFolder
        )  # listing all files in the destinationImagesFolder

        """ finding all file names that correspond to our desired position """
        #         relatedDestinationImagesRE = re.compile(r"(Position" + self.imagesPosition + r")_z(\d+)_(ch\d+)(.tif)") # group(0) = string, group(1) = position, group(2) = z, group(3) = channel, group(4) = .tif
        #        relatedDestinationImagesRE = re.compile(r"(" + self.imagesPosition + r")_z(\d+)_(ch\d+)(.tif)") # group(0) = whole string, group(1) = position, group(2) = z, group(3) = channel, group(4) = .tif
        relatedDestinationImagesRE = re.compile(
            r"MIP_(" + self.destinationCycle + r")_(FOV\d+)_(ch\d+)(.tif)"
        )  # group(0) = whole string, group(1) = cycle, group(2) = position, group(3) = channel, group(4) = .tif
        destinationImageFiles_splitted = [
            relatedDestinationImagesRE.search(filename) for filename in destinationFolder_files
        ]
        destinationImageFiles_splitted = [
            x for x in destinationImageFiles_splitted if x is not None
        ]  # name of all files in the destination directory splitted by above regex constraints
        destinationImageFiles_splitted.sort(
            key=lambda x: x.group(3)
        )  # sorting the file names based on channel values

        """ We want to group the destination images by channel, just to be similar to origin images, but no real need """
        self.destinationImageFilesByChannel = {}
        for channel in set(
            [x.group(3) for x in destinationImageFiles_splitted]
        ):  # iterate over the unique entries of channels
            self.destinationImageFilesByChannel[channel] = [
                x.group(0) for x in destinationImageFiles_splitted if x.group(3) == channel
            ]  # selecting those names that have the same channel
        print(
            datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ")
            + "After input: "
            + str(len(self.destinationImageFilesByChannel))
        )
        print(
            datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ")
            + "After input: "
            + str(len(self.originImageFilesByChannel))
        )

    def transformAllOriginImages(self):
        """this function uses the transform parameter map found by `imageTransformer`
        to transform all origin images, i.e. to align them to destination images
        """
        self.transformixImageFilter = sitk.TransformixImageFilter()
        self.transformParameterMap = self.imageTransformer.getTransformParameterMap()
        self.transformParameterMap[0]["FinalBSplineInterpolationOrder"] = ["1"]
        self.transformixImageFilter.SetTransformParameterMap(
            self.transformParameterMap
        )  # object transformixImageFilter is now ready to transform images with feed to it.
        print(
            datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ") + "Transforming channel images started"
        )
        for channel in self.originImageFilesByChannel:
            print(
                datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ")
                + "Transforming images from channel "
                + channel
            )
            imagesPaths_input = [
                pathjoin(self.originImagesFolder, originSingleImage)
                for originSingleImage in self.originImageFilesByChannel[channel]
            ]
            images3D_input = sitk.ReadImage(imagesPaths_input)
            images2D_input = sitk.Extract(
                images3D_input,
                (images3D_input.GetWidth(), images3D_input.GetHeight(), 0),
                (0, 0, 0),
            )
            self.transformixImageFilter.SetMovingImage(images2D_input)
            self.transformixImageFilter.Execute()
            imagesPaths_output = pathjoin(
                self.resultDirectory, self.originImageFilesByChannel[channel][0]
            )
            sitk.WriteImage(
                sitk.Cast(self.transformixImageFilter.GetResultImage(), sitk.sitkUInt8),
                imagesPaths_output,
            )
        print(
            datetime.now().strftime("%Y-%d-%m_%H:%M:%S: ") + "Transforming channel images finished"
        )

    def getOriginImageFiles(self):
        return self.originAllImageFiles

    def getDestinationImageFiles(self):
        return self.destinationImageFiles
