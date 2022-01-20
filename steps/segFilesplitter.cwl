#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool
baseCommand: [ls]

requirements:
  DockerRequirement:
    dockerPull: ubuntu:latest
  InitialWorkDirRequirement:
    listing:
      - $(inputs.segDir)

inputs:
  segDir:
    type: Directory
    doc: Directory with output from starfish segmentation step.

outputs:
  csvs:
    type:
      type: array
      items: File
    outputBinding:
      glob: "**/**/df_segmented.csv"
  priors:
    type:
      type: array
      items: File
    outputBinding:
      glob: "**/**/mask.tiff"
