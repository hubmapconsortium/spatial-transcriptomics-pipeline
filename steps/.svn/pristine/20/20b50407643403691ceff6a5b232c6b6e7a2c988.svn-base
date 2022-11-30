#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  tiffs:
    type: Directory
    doc: The directory containing all .tiff files

  codebook_csv:
    type: File?
    doc: Flattened csv input, refer to record entry.

  codebook_json:
    type: File?
    doc: Flattened json input, refer to record entry.

  codebook:
    type:
      - 'null'
      - type: record
        name: csv
        fields:
          csv:
            type: File
            doc: The codebook for this experiment in .csv format, where the rows are barcodes and the columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).
      - type: record
        name: json
        fields:
          json:
            type: File
            doc: The codebook for this experiment, already formatted in the spaceTx defined .json format.

  parameter_json:
    type: File?
    doc: json file with values to be read into other input variables.

  round_count:
    type: int?
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int?
    doc: The number of z-planes in each image

  channel_count:
    type: int?
    doc: The number of total channels per imaging round

  fov_count:
    type: int?
    doc: The number of FOVs that are included in this experiment

  round_offset:
    type: int?
    doc: The index of the first round (for file names).
    default: 0

  fov_offset:
    type: int?
    doc: The index of the first FOV (for file names).
    default: 0

  channel_offset:
    type: int?
    doc: The index of the first channel (for file names).
    default: 0

  zplane_offset:
    type: int?
    doc: The index of the first zplane (for file names).
    default: 0

  file_format:
    type: string?
    doc: String with layout for .tiff files

  file_vars:
    type: string[]?
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]?
    doc: Order of non x,y dimensions within each image.

  aux_tilesets:
    - 'null'
    - type: record
      name: aux_tilesets
      fields:
        aux_names:
          type: string[]?
          doc: Names of the Auxillary tiles.
        aux_file_formats:
          type: string[]?
          doc: String layout for .tiff files of aux views.
        aux_file_vars:
          type: string[]?
          doc: Variables to be substituted into aux_file_formats. One entry per aux_name, with semicolon-delimited vars.
        aux_cache_read_order:
          type: string[]?
          doc: Order of non x,y dimensions within each image. One entry per aux_name, with semicolon-delimited vars.
        aux_channel_count:
          type: int[]?
          doc: Count of channels in each aux image
        aux_channel_slope:
          type: float[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept
        aux_channel_intercept:
          type: int[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept

  fov_positioning:
    - 'null'
    - type: record
      name: dummy
      fields:
        dummy:
          type: string?
          doc: Added to prevent cli parsing of the fov_positioning record.
    - type: record
      fields:
        - name: x_locs
          type: string
          doc: list of x-axis start locations per fov index
        - name: x_shape
          type: int
          doc: shape of each fov item in the x-axis
        - name: x_voxel
          type: float
          doc: size of voxels in the x-axis
        - name: y_locs
          type: string
          doc: list of y-axis start locations per fov index
        - name: y_shape
          type: int
          doc: shape of each fov item in the y-axis
        - name: y_voxel
          type: float
          doc: size of voxels in the y-axis
        - name: z_locs
          type: string
          doc: list of z-axis start locations per fov index
        - name: z_shape
          type: int
          doc: shape of each fov item in the z-axis
        - name: z_voxel
          type: float
          doc: size of voxels in the z-axis

  add_blanks:
    type: boolean?
    doc: If true, will add blanks with a hamming distance 1 from the existing codes.
    default: False

outputs:
  spaceTx_converted:
    type: Directory
    outputSource: execute_conversion/spaceTx_converted

steps:

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:2.10

      inputs:
        schema:
          type: string
          inputBinding:
            position: 1

      outputs:
        data:
          type: stdout

    in:
      schema:
        valueFrom: "/opt/spaceTxConversion.json"
    out: [data]

  stage_conversion:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [round_count, zplane_count, channel_count, fov_count, round_offset, fov_offset, zplane_offset, channel_offset, file_format, file_vars, cache_read_order, aux_tilesets_aux_names, aux_tilesets_aux_file_formats, aux_tilesets_aux_file_vars, aux_tilesets_aux_cache_read_order, aux_tilesets_aux_channel_count, aux_tilesets_aux_channel_slope, aux_tilesets_aux_channel_intercept,  fov_positioning_x_locs, fov_positioning_x_shape, fov_positioning_x_voxel, fov_positioning_y_locs, fov_positioning_y_shape, fov_positioning_y_voxel, fov_positioning_z_locs, fov_positioning_z_shape, fov_positioning_z_voxel, add_blanks]
    when: $(inputs.datafile != null)

  execute_conversion:
    run:
      class: CommandLineTool
      baseCommand: /opt/spaceTxConverter.py

      requirements:
        DockerRequirement:
            dockerPull: hubmap/starfish-custom:2.10
      inputs:
          tiffs:
            type: Directory
            inputBinding:
              prefix: --input-dir

          codebook:
            type:
              - type: record
                name: csv
                fields:
                  csv:
                    type: File
                    inputBinding:
                      prefix: --codebook-csv
              - type: record
                name: json
                fields:
                  json:
                    type: File
                    inputBinding:
                      prefix: --codebook-json

          round_count:
            type: int
            inputBinding:
              prefix: --round-count

          zplane_count:
            type: int
            inputBinding:
              prefix: --zplane-count

          channel_count:
            type: int
            inputBinding:
              prefix: --channel-count

          fov_count:
            type: int
            inputBinding:
              prefix: --fov-count

          round_offset:
            type: int?
            inputBinding:
              prefix: --round-offset

          fov_offset:
            type: int?
            inputBinding:
              prefix: --fov-offset

          channel_offset:
            type: int?
            inputBinding:
              prefix: --channel-offset

          zplane_offset:
            type: int?
            inputBinding:
              prefix: --zplane-offset

          file_format:
            type: string
            inputBinding:
              prefix: --file-format

          file_vars:
            type: string[]
            inputBinding:
              prefix: --file-vars

          cache_read_order:
            type: string[]
            inputBinding:
              prefix: --cache-read-order

          aux_tilesets:
            type:
              type: record
              name: aux_tilesets
              fields:
                aux_names:
                  type: string[]?
                  inputBinding:
                    prefix: --aux-names
                aux_file_formats:
                  type: string[]?
                  inputBinding:
                    prefix: --aux-file-formats
                aux_file_vars:
                  type: string[]?
                  inputBinding:
                    prefix: --aux-file-vars
                aux_cache_read_order:
                  type: string[]?
                  inputBinding:
                    prefix: --aux-cache-read-order
                aux_channel_count:
                  type: int[]?
                  inputBinding:
                    prefix: --aux-channel-count
                aux_channel_slope:
                  type: float[]?
                  inputBinding:
                    prefix: --aux-channel-slope
                aux_channel_intercept:
                  type: int[]?
                  inputBinding:
                    prefix: --aux-channel-intercept

          fov_positioning:
            - 'null'
            - type: record
              fields:
                - name: x_locs
                  type: string
                  inputBinding:
                    prefix: --x-pos-locs
                - name: x_shape
                  type: int
                  inputBinding:
                    prefix: --x-pos-shape
                - name: x_voxel
                  type: float
                  inputBinding:
                    prefix: --x-pos-voxel
                - name: y_locs
                  type: string
                  inputBinding:
                    prefix: --y-pos-locs
                - name: y_shape
                  type: int
                  inputBinding:
                    prefix: --y-pos-shape
                - name: y_voxel
                  type: float
                  inputBinding:
                    prefix: --y-pos-voxel
                - name: z_locs
                  type: string
                  inputBinding:
                    prefix: --z-pos-locs
                - name: z_shape
                  type: int
                  inputBinding:
                    prefix: --z-pos-shape
                - name: z_voxel
                  type: float
                  inputBinding:
                    prefix: --z-pos-voxel

          add_blanks:
            type: boolean
            inputBinding:
              prefix: --add-blanks

      outputs:
        spaceTx_converted:
          type: Directory
          outputBinding:
            glob: "2_tx_converted/"
    in:
      tiffs: tiffs
      codebook:
        source: [codebook, codebook_csv, codebook_json]
        linkMerge: merge_flattened
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]){
              return {csv: self[1]};
            } else {
              return {json: self[2]};
            }
          }
      round_count:
        source: [stage_conversion/round_count, round_count]
        pickValue: first_non_null
      zplane_count:
        source: [stage_conversion/zplane_count, zplane_count]
        pickValue: first_non_null
      channel_count:
        source: [stage_conversion/channel_count, channel_count]
        pickValue: first_non_null
      fov_count:
        source: [stage_conversion/fov_count, fov_count]
        pickValue: first_non_null
      round_offset:
        source: [stage_conversion/round_offset, round_offset]
        pickValue: first_non_null
      fov_offset:
        source: [stage_conversion/fov_offset, fov_offset]
        pickValue: first_non_null
      channel_offset:
        source: [stage_conversion/channel_offset, channel_offset]
        pickValue: first_non_null
      zplane_offset:
        source: [stage_conversion/zplane_offset, zplane_offset]
        pickValue: first_non_null
      file_format:
        source: [stage_conversion/file_format, file_format]
        pickValue: first_non_null
      file_vars:
        source: [stage_conversion/file_vars, file_vars]
        pickValue: first_non_null
      cache_read_order:
        source: [stage_conversion/cache_read_order, cache_read_order]
        pickValue: first_non_null
      aux_tilesets:
        source: [aux_tilesets, stage_conversion/aux_tilesets_aux_names, stage_conversion/aux_tilesets_aux_file_formats, stage_conversion/aux_tilesets_aux_file_vars, stage_conversion/aux_tilesets_aux_cache_read_order, stage_conversion/aux_tilesets_aux_channel_count, stage_conversion/aux_tilesets_aux_channel_slope, stage_conversion/aux_tilesets_aux_channel_intercept]
        valueFrom: |
          ${
            if(!self[1]){
              return self[0];
            } else {
              return {
                aux_names: self[1],
                aux_file_formats: self[2],
                aux_file_vars: self[3],
                aux_cache_read_order: self[4],
                aux_channel_count: self[5],
                aux_channel_slope: self[6],
                aux_channel_intercept: self[7]
              };
            };
          }
      fov_positioning:
        source: [fov_positioning, stage_conversion/fov_positioning_x_locs, stage_conversion/fov_positioning_x_shape, stage_conversion/fov_positioning_x_voxel, stage_conversion/fov_positioning_y_locs, stage_conversion/fov_positioning_y_shape, stage_conversion/fov_positioning_y_voxel, stage_conversion/fov_positioning_z_locs, stage_conversion/fov_positioning_z_shape, stage_conversion/fov_positioning_z_voxel]
        valueFrom: |
          ${
            if(!self[1]){
              return self[0];
            } else {
              return {
                x_locs: self[1],
                x_shape: self[2],
                x_voxel: self[3],
                y_locs: self[4],
                y_shape: self[5],
                y_voxel: self[6],
                z_locs: self[7],
                z_shape: self[8],
                z_voxel: self[9]
              };
            };
          }
      add_blanks:
        source: [stage_conversion/add_blanks, add_blanks]
        pickValue: first_non_null
    out: [spaceTx_converted]

