#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["echo", "hello"]

requirements:
  InitialWorkDirRequirement:
    listing:
      - $(inputs.segDir)

inputs:
  segDir:
    type: Directory
    doc: Directory with output from starfish segmentation step.
    inputBinding:
      position: 1
      valueFrom: $(self.basename)

outputs:
  csvs:
    type:
      type: array
      items: File
    outputBinding:
      glob:"*/df_segmented.csv"
  priors:
    type:
      type: array
      items: File
    outputBinding:
      glob: "*/mask.tiff"
