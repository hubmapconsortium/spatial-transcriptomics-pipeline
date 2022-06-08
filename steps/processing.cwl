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

  opening_size:
    type: int?
    doc: Size of the morphological opening filter to be applied to the image

  register_aux_view:
    type: string?
    doc: The name of the auxillary view to be used for image registration.


outputs:
  processed_exp:
    type: Directory
    outputSource: execute_processing/processed_exp

steps:
  stage_processing:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema:
        valueFrom: |
          ${
            return {
              "class": "File",
              "location": "../input_schemas/processing.json"
            };
          }
    out: [clip_min, opening_size, register_aux_view]
    when: $(inputs.datafile != null)

  execute_processing:
    run:
      class: CommandLineTool
      baseCommand: /opt/imgProcessing.py

      requirements:
        DockerRequirement:
            dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.04

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
          doc: Pixels below this percentile are set to 0.

        opening_size:
          type: int?
          inputBinding:
            prefix: --opening-size
          doc: Size of the morphological opening filter to be applied to the image

        register_aux_view:
          type: string?
          inputBinding:
            prefix: --register-aux-view
          doc: The name of the auxillary view to be used for image registration.

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
      opening_size:
        source: [stage_processing/opening_size, opening_size]
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
    out: [processed_exp]

