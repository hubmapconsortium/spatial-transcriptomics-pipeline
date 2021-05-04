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
      prefix: --input-dir
    doc: The directory containing all .tiff files

  codebook_csv:
    type: File
    inputBinding:
      prefix: --codebook-csv
    doc: The codebook for this experiment in .csv format, where the rows are barcodes and the columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).

  round_count:
    type: int
    inputBinding:
      prefix: --round-count
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int
    inputBinding:
      prefix: --zplane-count
    doc: The number of z-planes in each image

  channel_count:
    type: int
    inputBinding:
      prefix: --channel-count
    doc: The number of total channels per imaging round

  fov_count:
    type: int
    inputBinding:
      prefix: --fov-count
    doc: The number of FOVs that are included in this experiment

  round_offset:
    type: int?
    inputBinding:
      prefix: --round-offset
    doc: The index of the first round (for file names).

  fov_offset:
    type: int?
    inputBinding:
      prefix: --fov-offset
    doc: The index of the first FOV (for file names).

  channel_offset:
    type: int?
    inputBinding: 
      prefix: --channel-offset
    doc: The index of the first channel (for file names).

  file_format:
    type: string
    inputBinding:
      prefix: --file-format
    doc: String with layout for .tiff files

  file_vars:
    type: string[]
    inputBinding:
      prefix: --file-vars
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]
    inputBinding:
      prefix: --cache-read-order
    doc: Order of non x,y dimensions within each image.

  aux_names:
    type: string[]?
    inputBinding:
      prefix: --aux-names
    doc: Names of the Auxillary tiles.

  aux_file_formats:
    type: string[]?
    inputBinding:
      prefix: --aux-file-formats
    doc: String layout for .tiff files of aux views.

  aux_file_vars:
    type: string[]?
    inputBinding:
      prefix: --aux-file-vars
    doc: Variables to be substituted into aux_file_formats. One entry per aux_name, with semicolon-delimited vars.

  aux_cache_read_order:
    type: string[]?
    inputBinding:
      prefix: --aux-cache-read-order
    doc: Order of non x,y dimensions within each image. One entry per aux_name, with semicolon-delimited vars.

  aux_fixed_channel:
    type: int[]?
    inputBinding:
      prefix: --aux-fixed-channel
    doc: Which channel to refer to in aux images.

outputs:
  spaceTx_converted:
    type: Directory
    outputBinding:
      glob: "tx_converted/"

