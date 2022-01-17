#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["ls"]

inputs:
  segDir:
    type: Directory
    doc: Directory with output from starfish segmentation step.

outputs:
  csvs:
    type: File[]
    outputBinding:
      glob:"*/df_segmented.csv"
  priors:
    type: File[]
      glob: "*/mask.tiff"
