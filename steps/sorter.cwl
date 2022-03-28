#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/pseudoSort.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.0

inputs:
  input_dir:
    type: Directory
    inputBinding:
      prefix: --input-dir
    doc: The root directory containing all images.

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

  channel_yml:
    type: File
    inputBinding:
      prefix: --channel-yml
    doc: PyYML-formatted list containing a dictionary outlining how the truechannels in imaging relate to the pseudochannels in the decoding codebook. The index of each dict within the list is the trueround % (count of pseudorounds). The keys of the dict are the channels within the image and the values are the pseudochannels in the converted notebook.

  cycle_yml:
    type: File
    inputBinding:
      prefix: --cycle-yml
    doc: PyYML-formatted dictionary outlining how the truerounds in imaging relate to the pseudorounds in the decoding codebook. The keys are truerounds and the values are the corresponding pseudorounds.

  file_format:
    type: string
    inputBinding:
      prefix: --file-format
    doc: String with layout for .tiff files. Will be formatted via str.format().

  file_vars:
    type: string[]
    inputBinding:
      prefix: --file-vars
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]
    inputBinding:
      prefix: --cache-read-order
    doc: Order of x,y,z,ch dimensions within each image.

  fov_count:
    type: int
    inputBinding:
      prefix: --fov-count
    doc: The number of FOVs that are included in this experiment

  round_offset:
    type: int?
    inputBinding:
      prefix: --round-offset
    default: 0
    doc: The index of the first round (for file names).

  fov_offset:
    type: int?
    inputBinding:
      prefix: --fov-offset
    default: 0
    doc: The index of the first FOV (for file names).

  channel_offset:
    type: int?
    inputBinding:
      prefix: --channel-offset
    default: 0
    doc: The index of the first channel (for file names).

  channel_slope:
    type: float?

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

outputs:
  pseudosorted_dir:
    type: Directory
    outputBinding:
      glob: "1_pseudosort/"

  log:
    type: stdout
