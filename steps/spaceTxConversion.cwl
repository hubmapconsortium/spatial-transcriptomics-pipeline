#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/spaceTxConverter.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.0

inputs:
  tiffs:
    type: Directory
    inputBinding:
      prefix: --input-dir
    doc: The directory containing all .tiff files

  codebook:
    type:
      - type: record
        name: csv
        fields:
          csv:
            type: File
            inputBinding:
              prefix: --codebook-csv
            doc: The codebook for this experiment in .csv format, where the rows are barcodes and the columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).
      - type: record
        name: json
        fields:
          json:
            type: File
            inputBinding:
              prefix: --codebook-json
            doc: The codebook for this experiment, already formatted in the spaceTx defined .json format.

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

  zplane_offset:
    type: int?
    inputBinding:
      prefix: --zplane-offset
    doc: The index of the first zplane (for file names).

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

  aux_tilesets:
    type:
      type: record
      name: aux_tilesets
      fields:
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
        aux_channel_count:
          type: int[]?
          inputBinding:
            prefix: --aux-channel-count
          doc: Count of channels in each aux image
        aux_channel_slope:
          type: float[]?
          inputBinding:
            prefix: --aux-channel-slope
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept
        aux_channel_intercept:
          type: int[]?
          inputBinding:
            prefix: --aux-channel-intercept
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept

  fov_positioning:
    - 'null'
    - type: record
      fields:
        - name: x-locs
          type: string
          inputBinding:
            prefix: --x-pos-locs
          doc: list of x-axis start locations per fov index
        - name: x-shape
          type: int
          inputBinding:
            prefix: --x-pos-shape
          doc: shape of each fov item in the x-axis
        - name: x-voxel
          type: float
          inputBinding:
            prefix: --x-pos-voxel
          doc: size of voxels in the x-axis
        - name: y-locs
          type: string
          inputBinding:
            prefix: --y-pos-locs
          doc: list of y-axis start locations per fov index
        - name: y-shape
          type: int
          inputBinding:
            prefix: --y-pos-shape
          doc: shape of each fov item in the y-axis
        - name: y-voxel
          type: float
          inputBinding:
            prefix: --y-pos-voxel
          doc: size of voxels in the y-axis
        - name: z-locs
          type: string
          inputBinding:
            prefix: --z-pos-locs
          doc: list of z-axis start locations per fov index
        - name: z-shape
          type: int
          inputBinding:
            prefix: --z-pos-shape
          doc: shape of each fov item in the z-axis
        - name: z-voxel
          type: float
          inputBinding:
            prefix: --z-pos-voxel
          doc: size of voxels in the z-axis

outputs:
  spaceTx_converted:
    type: Directory
    outputBinding:
      glob: "2_tx_converted/"

