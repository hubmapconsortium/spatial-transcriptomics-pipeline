#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/spaceTxConverter.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish/main:latest

inputs:
  tiffs:
    type: Directory
    inputBinding:
      position: 1
      prefix: --input-dir
    doc: The directory containing all .tiff files

  codebook_csv:
    type: File
    inputBinding:
      position: 2
      prefix: --codebook-csv
    doc: The codebook for this experiment in .csv format, as described [PLACE]

  round_count:
    type: int
    inputBinding:
      position: 3
      prefix: --round-count
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int
    inputBinding:
      position: 4
      prefix: --zplane-count
    doc: The number of z-planes in each image

  channel_count:
    type: int
    inputBinding:
      position: 5
      prefix: --channel-count
    doc: The number of total channels per imaging round

  fov_count:
    type: int
    inputBinding:
      position: 6
      prefix: --fov-count
    doc: The number of FOVs that are included in this experiment

  round_offset:
    type: int?
    inputBinding:
      position: 7
      prefix: --round_offset
    doc: The index of the first round.

  fov_offset:
    type: int?
    inputBinding:
      position: 8
      prefix: --fov_offset
    doc: The index of the first FOV.

outputs:
  spaceTx_converted:
    type: Directory
    outputBinding:
      glob: "tx_converted/"

