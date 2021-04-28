#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/spaceTxConverter.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish:latest

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
      prefix: --round-offset
    doc: The index of the first round.

  fov_offset:
    type: int?
    inputBinding:
      position: 8
      prefix: --fov-offset
    doc: The index of the first FOV.

  file_format:
    type: string
    inputBinding:
      position: 9
      prefix: --file-format
    doc: String with layout for .tiff files

  file_vars:
    type: string[]
    inputBinding:
      position: 10
      prefix: --file-vars
    doc: Variables to get substituted into the file_format string.

  aux_names:
    type: string[]?
    inputBinding:
      position: 11
      prefix: --aux-names
    doc: Names of the Auxillary tiles.

  aux_file_formats:
    type: string[]?
    inputBinding:
      position: 12
      prefix: --aux-file-formats
    doc: String layout for .tiff files of aux views.

  aux_file_vars:
    type: string[]?
    inputBinding:
      position: 13
      prefix: --aux-file-vars
    doc: Variables to be substituted into aux_file_formats. One entry per aux_name, with semicolon-delimited vars.

  aux_fixed_channel:
    type: int[]?
    inputBinding:
      position: 14
      prefix: --aux-fixed-channel
    doc: Which channel to refer to in aux images.

outputs:
  spaceTx_converted:
    type: Directory
    outputBinding:
      glob: "tx_converted/"

