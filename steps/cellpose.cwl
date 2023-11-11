#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  exp_loc:
    type: Directory
    doc: Root directory containing space_tx formatted experiment

  dir_size:
    type: long?
    doc: Size of tiffs, in MiB. If provided, will be used to calculate ResourceRequirement.

  decoded_loc:
    type: Directory?
    doc: Location of directory that is output from the starfishRunner step, only needed if mRNA information is to be included.

  use_mrna:
    type: boolean?
    doc: If true and decoded_loc is provided, mrna data will be used in calculations.

  parameter_json:
    type: File?
    doc: json containing step parameters.

  selected_fovs:
    type: int[]?
    doc: If provided, segmentation will only be run on FOVs with these indices.

  zplane_count:
    type: int?
    doc: The number of z-planes in each image. All that matters is whether this is equal to 1 or not, retaining the same var name as conversion for simplification.

  aux_views:
    type: string[]?
    doc: The views to use for cellpose segmentation.

  pretrained_model_str:
    type: string?
    doc: Cellpose-provided model to use.

  pretrained_model_dir:
    type: File?
    doc: Manually trained cellpose model to use.

  diameter:
    type: float?
    doc: Expected diameter of cells. Should be 0 if a custom model is used.

  flow_threshold:
    type: float?
    doc: threshold for filtering cell segmentations (increasing this will filter out lower confidence segmentations), range is 0 to infinity

  stitch_threshold:
    type: float?
    doc: threshold for stitching together segmentations that occur at the same xy location but in adjacent z slices, range is 0 to 1. This should only be used when the image is 3D.

  cellprob_threshold:
    type: float?
    doc: determines the extent of the segmentations (0 is the default more negative values result in larger cells, more positive values result in smaller cells), range is -6 to 6.

  border_buffer:
    type: int?
    doc: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.

  label_exp_size:
    type: int?
    doc: Pixel size labels are dilated by in final step. Helpful for closing small holes that are common from thresholding but can also cause cell boundaries to exceed their true boundaries if set too high. Label dilation respects label borders and does not mix labels.

  min_allowed_size:
    type: int?
    doc: minimum size for a cell (in pixels)

  max_allowed_size:
    type: int?
    doc: maximum size for a cell (in pixels)

outputs:
  cellpose_input:
    type: Directory
    outputSource: execute_cellpose_prep/cellpose_input
  cellpose_output:
    type: Directory
    outputSource: execute_cellpose/cellpose_output
  cellpose_filtered:
    type: Directory
    outputSource: execute_filtering/cellpose_filtered

steps:

  tmpname:
    run: tmpdir.cwl
    in: []
    out: [tmp]

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
        ResourceRequirement:
          ramMin: 1000
          tmpdirMin: 1000
          outdirMin: 1000

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
        valueFrom: "/opt/cellpose.json"
    out: [data]

  stage_cellpose:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [use_mrna, zplane_count, selected_fovs, pretrained_model_str, diameter, flow_threshold, stitch_threshold, cellprob_threshold,  border_buffer, label_exp_size, min_allowed_size, max_allowed_size, aux_views]
    when: $(inputs.datafile != null)

  execute_cellpose_prep:
    run:
      class: CommandLineTool
      baseCommand: /opt/cellposeStaging.py

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
        ResourceRequirement:
          tmpdirMin: |
            ${
              if(inputs.dir_size === null) {
                return null;
              } else {
                return inputs.dir_size * 4;
              }
            }
          outdirMin: |
            ${
              if(inputs.dir_size === null) {
                return null;
              } else {
                return inputs.dir_size * 4;
              }
            }

      inputs:
        dir_size:
          type: long?

        tmp_prefix:
          type: string
          inputBinding:
            prefix: --tmp-prefix

        exp_loc:
          type: Directory
          doc: Root directory containing space_tx formatted experiment
          inputBinding:
            prefix: --input-dir

        decoded_loc:
          type: Directory?
          doc: Location of directory that is output from the starfishRunner step.
          inputBinding:
            prefix: --decoded-dir

        selected_fovs:
          type: int[]?
          doc: If provided, segmentation will only be run on FOVs with these indices.
          inputBinding:
            prefix: --selected-fovs

        aux_views:
          type: string[]?
          doc: The views to use for cellpose segmentation.
          inputBinding:
            prefix: --aux-views

        format:
          type: boolean
          doc: Used to specify method in python script
          default: true
          inputBinding:
            prefix: --format

      outputs:
        cellpose_input:
          type: Directory
          outputBinding:
            glob: $("tmp/" + inputs.tmp_prefix + "/5A_cellpose_input/")
    in:
      dir_size: dir_size
      tmp_prefix: tmpname/tmp
      exp_loc: exp_loc
      decoded_loc:
        source: [decoded_loc, stage_cellpose/use_mrna, use_mrna]
        valueFrom: |
          ${
            if(self[1] || self[2]){
              return self[0];
            } else {
              return null;
            }
          }
      selected_fovs:
        source: [stage_cellpose/selected_fovs, selected_fovs]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      aux_views:
        source: [stage_cellpose/aux_views, aux_views]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
    out: [cellpose_input]

  execute_cellpose:
    run:
      class: CommandLineTool
      baseCommand: ["cellpose"]

      requirements:
        DockerRequirement:
          dockerPull: hubmap/cellpose:latest
        InitialWorkDirRequirement:
          listing:
            - entry: $(inputs.input_dir)
              writable: true

      inputs:
        verbose:
          type: boolean?
          inputBinding:
            prefix: --verbose
          default: true
          doc: Enables verbose output

        input_dir:
          type: Directory
          inputBinding:
            prefix: --dir
#            valueFrom: $(self.basename)
          doc: Input directory for cellpose

        img_filter:
          type: string
          inputBinding:
            prefix: --img_filter
          default: _image
          doc: Glob filter for input

        z_axis:
          type: int?
          inputBinding:
            prefix: --z_axis
          doc: 0 if image is not flat, unset if flat.

        channel_axis:
          type: int?
          inputBinding:
            prefix: --channel_axis
          doc: 1 if image is flat, 0 if flat.

        all_channels:
          type: boolean?
          inputBinding:
            prefix: --all_channels
          default: true
          doc: Tells cellpose to use all channels in the order they appear

        save_tif:
          type: boolean?
          inputBinding:
            prefix: --save_tif
          default: true
          doc: Tells cellpose to save images as a tif, instead of defaulting to png

        savedir:
          type: string?
          inputBinding:
            prefix: --savedir
          default: "5B_cellpose_output"
          doc: Name of directory to save to.

        pretrained_model_str:
          type: string?
          inputBinding:
            prefix: --pretrained_model
          doc: Cellpose-provided model to use.

        pretrained_model_dir:
          type: File?
          inputBinding:
            prefix: --pretrained_model
          doc: Manually trained cellpose model to use.

        diameter:
          type: float?
          inputBinding:
            prefix: --diameter
          doc: Expected diameter of cells. Should be 0 if a custom model is used.

        flow_threshold:
          type: float?
          inputBinding:
            prefix: --flow_threshold
          default: 0.4
          doc: threshold for filtering cell segmentations (increasing this will filter out lower confidence segmentations), range is 0 to infinity

        stitch_threshold:
          type: float?
          inputBinding:
            prefix: --stitch_threshold
          doc: threshold for stitching together segmentations that occur at the same xy location but in adjacent z slices, range is 0 to 1. This should only be used when the image is 3D.

        cellprob_threshold:
          type: float?
          inputBinding:
            prefix: --cellprob_threshold
          doc: determines the extent of the segmentations (0 is the default more negative values result in larger cells, more positive values result in smaller cells), range is -6 to 6.

        net_avg:
          type: boolean?
          inputBinding:
            prefix: --net_avg
          default: true
          doc: tells cellpose to calculate 4 nets and take the average, improves performance.

      outputs:
        log:
          type: stdout
        cellpose_output:
          type: Directory
          outputBinding:
            glob: "5B_cellpose_output"

    in:
      input_dir: execute_cellpose_prep/cellpose_input
      z_axis:
        source: [stage_cellpose/zplane_count, zplane_count]
        valueFrom: |
          ${
            if(self[0]){
              if(self[0] > 1){
                return 0;
              } else {
                return null;
              }
            } else {
              if(self[1] > 1){
                return 0;
              } else {
                return null;
              }
            }
          }
      channel_axis:
        source: [stage_cellpose/zplane_count, zplane_count]
        valueFrom: |
          ${
            if(self[0]){
              if(self[0] > 1){
                return 1;
              } else {
                return 0;
              }
            } else {
              if(self[1] > 1){
                return 1;
              } else {
                return 0;
              }
            }
          }
      pretrained_model_str:
        source: [stage_cellpose/pretrained_model_str, pretrained_model_str]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      pretrained_model_dir: pretrained_model_dir
      diameter:
        source: [stage_cellpose/diameter, diameter, pretrained_model_dir]
        valueFrom: |
          ${
            if(self[2]){
              return 0;
            } else if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      flow_threshold:
        source: [stage_cellpose/flow_threshold, flow_threshold]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      stitch_threshold:
        source: [stage_cellpose/stitch_threshold, stitch_threshold, stage_cellpose/zplane_count, zplane_count]
        valueFrom: |
          ${
            if(self[2] == 1 || self[3] == 1){
              return null;
            }
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      cellprob_threshold:
        source: [stage_cellpose/cellprob_threshold, cellprob_threshold]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
    out: [cellpose_output]

  execute_filtering:
    run:
      class: CommandLineTool
      baseCommand: /opt/cellposeStaging.py

      requirements:
        DockerRequirement:
            dockerPull: hubmap/starfish-custom:latest
        ResourceRequirement:
          tmpdirMin: |
            ${
              if(inputs.dir_size === null) {
                return null;
              } else {
                return inputs.dir_size * 4;
              }
            }
          outdirMin: |
            ${
              if(inputs.dir_size === null) {
                return null;
              } else {
                return inputs.dir_size * 4;
              }
            }

      inputs:
        dir_size:
          type: long?

        tmp_prefix:
          type: string
          inputBinding:
            prefix: --tmp-prefix

        input_loc:
          type: Directory
          doc: Output from cellpose.
          inputBinding:
            prefix: --input-dir

        selected_fovs:
          type: int[]?
          doc: If provided, segmentation will only be run on FOVs with these indices.
          inputBinding:
            prefix: --selected-fovs

        border_buffer:
          type: int?
          doc: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.
          inputBinding:
            prefix: --border-buffer

        label_exp_size:
          type: int?
          doc: Pixel size labels are dilated by in final step. Helpful for closing small holes that are common from thresholding but can also cause cell boundaries to exceed their true boundaries if set too high. Label dilation respects label borders and does not mix labels.
          inputBinding:
            prefix: --label-exp-size

        max_allowed_size:
          type: int?
          doc: maximum size for a cell (in pixels)
          inputBinding:
            prefix: --max-size

        min_allowed_size:
          type: int?
          doc: minimum size for a cell (in pixels)
          inputBinding:
            prefix: --min-size

        filter:
          type: boolean
          doc: Used to specify method in python script
          default: true
          inputBinding:
            prefix: --filter

      outputs:
        cellpose_filtered:
          type: Directory
          outputBinding:
            glob: $("tmp/" + inputs.tmp_prefix + "/5C_cellpose_filtered")

    in:
      dir_size: dir_size
      tmp_prefix: tmpname/tmp
      input_loc: execute_cellpose/cellpose_output
      selected_fovs:
        source: [stage_cellpose/selected_fovs, selected_fovs]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      border_buffer:
        source: [stage_cellpose/border_buffer, border_buffer]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      label_exp_size:
        source: [stage_cellpose/label_exp_size, label_exp_size]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      min_allowed_size:
        source: [stage_cellpose/min_allowed_size, min_allowed_size]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      max_allowed_size:
        source: [stage_cellpose/max_allowed_size, max_allowed_size]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
    out: [cellpose_filtered]
