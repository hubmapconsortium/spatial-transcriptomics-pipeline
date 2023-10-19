#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  input_dir:
    type: Directory
    doc: The root directory containing all images.

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

  channel_yml:
    type: File
    doc: PyYML-formatted list containing a dictionary outlining how the truechannels in imaging relate to the pseudochannels in the decoding codebook. The index of each dict within the list is the trueround % (count of pseudorounds). The keys of the dict are the channels within the image and the values are the pseudochannels in the converted notebook.

  cycle_yml:
    type: File
    doc: PyYML-formatted dictionary outlining how the truerounds in imaging relate to the pseudorounds in the decoding codebook. The keys are truerounds and the values are the corresponding pseudorounds.

  parameter_json:
    type: File?
    doc: json file containing parameters for conversion

  file_format:
    type: string?
    doc: String with layout for .tiff files. Will be formatted via str.format().

  file_vars:
    type: string[]?
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]?
    doc: Order of x,y,z,ch dimensions within each image.

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

  channel_slope:
    type: float?
    default: 1

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
          type: float[]?
          doc: Count of channels in each aux image.
        aux_channel_slope:
          type: float[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept
        aux_channel_intercept:
          type: int[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept


outputs:
  pseudosorted_dir:
    type: Directory
    outputSource: execute_sort/pseudosorted_dir

steps:

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest

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
        valueFrom: "/opt/sorter.json"
    out: [data]

  tmpname:
    run: tmpdir.cwl
    in: []
    out: [tmp]

  file_sizer:
    run: fileSizer.cwl
    in:
      example_dir: input_dir
    out: [dir_size]

  stage_sort:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [round_count, fov_count, round_offset, fov_offset, channel_offset, channel_slope, file_format, file_vars, cache_read_order, aux_tilesets_aux_names, aux_tilesets_aux_file_formats, aux_tilesets_aux_file_vars, aux_tilesets_aux_cache_read_order, aux_tilesets_aux_channel_count, aux_tilesets_aux_channel_slope, aux_tilesets_aux_channel_intercept]
    when: $(inputs.datafile != null)

  execute_sort:
    run:
      class: CommandLineTool
      baseCommand: /opt/pseudoSort.py

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
        ResourceRequirement:
          tmpdirMin: $(inputs.dir_size * 1.2)
          outdirMin: $(inputs.dir_size * 1.2)

      inputs:

        dir_size:
          type: long

        tmp_prefix:
          type: string
          inputBinding: 
            prefix: --tmp-prefix

        input_dir:
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

        channel_yml:
          type: File
          inputBinding:
            prefix: --channel-yml

        cycle_yml:
          type: File
          inputBinding:
            prefix: --cycle-yml

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

        channel_slope:
          type: float?
          inputBinding:
            prefix: --channel-slope

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

      outputs:
        pseudosorted_dir:
          type: Directory
          outputBinding:
            glob: $("tmp/" + inputs.tmp_prefix + "/1_pseudosort/")

        log:
          type: stdout
    in:
      dir_size: file_sizer/dir_size
      tmp_prefix: tmpname/tmp
      input_dir: input_dir
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
      channel_yml: channel_yml
      cycle_yml: cycle_yml
      file_format:
        source: [stage_sort/file_format, file_format]
        pickValue: first_non_null
      file_vars:
        source: [stage_sort/file_vars, file_vars]
        pickValue: first_non_null
      cache_read_order:
        source: [stage_sort/cache_read_order, cache_read_order]
        pickValue: first_non_null
      fov_count:
        source: [stage_sort/fov_count, fov_count]
        pickValue: first_non_null
      round_offset:
        source: [stage_sort/round_offset, round_offset]
        pickValue: first_non_null
      fov_offset:
        source: [stage_sort/fov_offset, fov_offset]
        pickValue: first_non_null
      channel_offset:
        source: [stage_sort/channel_offset, channel_offset]
        pickValue: first_non_null
      channel_slope:
        source: [stage_sort/channel_slope, channel_slope]
        pickValue: first_non_null
      aux_tilesets:
        source: [aux_tilesets, stage_sort/aux_tilesets_aux_names, stage_sort/aux_tilesets_aux_file_formats, stage_sort/aux_tilesets_aux_file_vars, stage_sort/aux_tilesets_aux_cache_read_order, stage_sort/aux_tilesets_aux_channel_count, stage_sort/aux_tilesets_aux_channel_slope, stage_sort/aux_tilesets_aux_channel_intercept]
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
            }
          }
    out:
      [pseudosorted_dir]
