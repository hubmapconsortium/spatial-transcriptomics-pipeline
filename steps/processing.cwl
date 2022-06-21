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
    doc: Root directory containing space_tx formatted experiment

  parameter_json:
    type: File?
    doc: json containing step parameters.

  clip_min:
    type: float?
    doc: Pixels below this percentile are set to 0.

  register_aux_view:
    type: string?
    doc: The name of the auxillary view to be used for image registration.

  channels_per_reg:
    type: int?
    doc: The number of images associated with each channel in the registration image.

  background_view:
    type: string?
    doc: The name of the auxillary view to be used for background subtraction.  Background will be estimated if not provided.

  anchor_view:
    type: string?
    doc: The name of the auxillary view to be processed in parallel with primary view, such as for anchor round in ISS processing. Will not be included if not provided.

  high_sigma:
    type: int?
    doc: Sigma value for high pass gaussian filter. Will not be run if not provided.

  deconvolve_iter:
    type: int?
    doc: Number of iterations to perform for deconvolution. High values remove more noise while lower values remove less. The value 15 will work for most datasets unless image is very noisy. Will not be run if not provided.

  deconvolve_sigma:
    type: int?
    doc: Sigma value for deconvolution. Should be approximately the expected spot size.

  low_sigma:
    type: int?
    doc: Sigma value for low pass gaussian filter. Will not be run if not provided.

  rolling_radius:
    type: int?
    doc: Radius for rolling ball background subtraction. Larger values lead to increased intensity evening effect. The value of 3 will work for most datasets. Will not be run if not provided.

  match_histogram:
    type: boolean?
    doc: If true, histograms will be equalized.

  tophat_radius:
    type: int?
    doc: Radius for white top hat filter. Should be slightly larger than the expected spot radius. Will not be run if not provided.

outputs:
  processed_exp:
    type: Directory
    outputSource: execute_processing/processed_exp

steps:

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.05

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
        valueFrom: "/opt/processing.json"
    out: [data]

  stage_processing:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [clip_min, register_aux_view, channels_per_reg, background_view, anchor_view, high_sigma, deconvolve_iter, deconvolve_sigma, low_sigma, rolling_radius, match_histogram, tophat_radius]
    when: $(inputs.datafile != null)

  execute_processing:
    run:
      class: CommandLineTool
      baseCommand: /opt/imgProcessing.py

      requirements:
        DockerRequirement:
            dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.05

      inputs:
        input_dir:
          type: Directory
          inputBinding:
            prefix: --input-dir
          doc: Root directory containing space_tx formatted experiment

        clip_min:
          type: float?
          inputBinding:
            prefix: --clip-min
          doc: Pixels below this percentile are set to 0. Defaults to 95.

        register_aux_view:
          type: string?
          inputBinding:
            prefix: --register-aux-view
          doc: The name of the auxillary view to be used for image registration. Registration will not be performed if not provided.

        channels_per_reg:
          type: int?
          inputBinding:
            prefix: --ch-per-reg
          doc: The number of images associated with each channel of the registration image.  Defaults to 1.

        background_view:
          type: string?
          inputBinding:
            prefix: --background-view
          doc: The name of the auxillary view to be used for background subtraction.  Background will be estimated if not provided.

        anchor_view:
          type: string?
          inputBinding:
            prefix: --anchor-view
          doc: The name of the auxillary view to be processed in parallel with primary view, such as for anchor round in ISS processing. Will not be included if not provided.

        high_sigma:
          type: int?
          inputBinding:
            prefix: --high-sigma
          doc: Sigma value for high pass gaussian filter. Will not be run if not provided.

        deconvolve_iter:
          type: int?
          inputBinding:
            prefix: --decon-iter
          doc: Number of iterations to perform for deconvolution. High values remove more noise while lower values remove less. The value 15 will work for most datasets unless image is very noisy. Will not be run if not provided.

        deconvolve_sigma:
          type: int?
          inputBinding:
            prefix: --decon-sigma
          doc: Sigma value for deconvolution. Should be approximately the expected spot size.

        low_sigma:
          type: int?
          inputBinding:
            prefix: --low-sigma
          doc: Sigma value for low pass gaussian filter. Will not be run if not provided.

        rolling_radius:
          type: int?
          inputBinding:
            prefix: --rolling-radius
          doc: Radius for rolling ball background subtraction. Larger values lead to increased intensity evening effect. The value of 3 will work for most datasets. Will not be run if not provided.

        match_histogram:
          type: boolean?
          inputBinding:
            prefix: --match-histogram
          doc: If true, histograms will be equalized.

        tophat_radius:
          type: int?
          inputBinding:
            prefix: --tophat-radius
          doc: Radius for white top hat filter. Should be slightly larger than the expected spot radius. Will not be run if not provided.

      outputs:
        processed_exp:
          type: Directory
          outputBinding:
            glob: "3_processed"
    in:
      input_dir: input_dir
      clip_min:
        source: [stage_processing/clip_min, clip_min]
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
      register_aux_view:
        source: [stage_processing/register_aux_view, register_aux_view]
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
      channels_per_reg:
        source: [stage_processing/channels_per_reg, channels_per_reg]
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
      background_view:
        source: [stage_processing/background_view, background_view]
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
      anchor_view:
        source: [stage_processing/anchor_view, anchor_view]
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
      high_sigma:
       source: [stage_processing/high_sigma, high_sigma]
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
      deconvolve_iter:
        source: [stage_processing/deconvolve_iter, deconvolve_iter]
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
      deconvolve_sigma:
        source: [stage_processing/deconvolve_sigma, deconvolve_sigma]
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
      low_sigma:
        source: [stage_processing/low_sigma, low_sigma]
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
      rolling_radius:
        source: [stage_processing/rolling_radius, rolling_radius]
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
      match_histogram:
        source: [stage_processing/match_histogram, match_histogram]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if (self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      tophat_radius:
        source: [stage_processing/tophat_radius, tophat_radius]
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
    out: [processed_exp]

