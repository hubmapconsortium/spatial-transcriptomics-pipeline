#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  decoded_loc:
    type: Directory
    doc: Location of the directory that is the output from the starfishRunner step.

  exp_loc:
    type: Directory
    doc: Location of directory containing the 'experiment.json' file

  parameter_json:
    type: File?
    doc: File containing parameters to run this step.

  aux_name:
    type: string?
    doc: The name of the aux view to look at in the experiment file.

  fov_count:
    type: int?
    doc: The number of FOVs that are included in this experiment

  binary_mask:
    - 'null'
    - type: record
      name: roi_set
      fields:
        roi_set:
          type: Directory
          doc: Directory of RoiSet.zip for each fov, from fiji segmentation
        file_formats:
          type: string
          doc: Layout for name of each RoiSet.zip, per fov. Will be formatted with String.format([fov index]).
    - type: record
      name: labeled_image
      fields:
        labeled_image:
          type: Directory
          doc: Directory of labeled images with image segmentation data, such as from ilastik classification.
        file_formats_labeled:
          type: string
          doc: Layout for name of each labelled image. Will be formatted with String.format([fov index])
    - type: record
      name: basic_watershed
      fields:
        img_threshold:
          type: float
          doc: Global threshold value for images
        min_dist:
          type: int
          doc: minimum distance (pixels) between distance transformed peaks
        min_allowed_size:
          type: int
          doc: minimum size for a cell (in pixels)
        max_allowed_size:
          type: int
          doc: maxiumum size for a cell (in pixels)
        masking_radius:
          type: int
          doc: Radius for white tophat noise filter

outputs:
  segmented:
    type: Directory
    outputSource: execute_segmentation/segmented

steps:
  stage_segmentation:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema:
        valueFrom: |
          ${
            return {
              "class": "File",
              "location": "../input_schemas/segmentation.json"
            };
          }
    out: [aux_name, fov_count, binary_mask_img_threshold, binary_mask_min_dist, binary_mask_min_allowed_size, binary_mask_max_allowed_size, binary_mask_masking_radius]
    when: $(inputs.datafile != null)
  execute_segmentation:
    run:
      class: CommandLineTool
      baseCommand: /opt/segmentationDriver.py

      requirements:
        DockerRequirement:
          dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.04

      inputs:
        decoded_loc:
          type: Directory
          inputBinding:
            prefix: --decoded-loc

        exp_loc:
          type: Directory
          inputBinding:
            prefix: --exp-loc

        aux_name:
          type: string
          inputBinding:
            prefix: --aux-name

        fov_count:
          type: int
          inputBinding:
            prefix: --fov-count

        binary_mask:
          type:
            - type: record
              name: roi_set
              fields:
                roi_set:
                  type: Directory
                  inputBinding:
                    prefix: --roi-set
                file_formats:
                  type: string
                  inputBinding:
                    prefix: --file-formats
            - type: record
              name: labeled_image
              fields:
                labeled_image:
                  type: Directory
                  inputBinding:
                    prefix: --labeled-image
                file_formats_labeled:
                  type: string
                  inputBinding:
                    prefix: --file-formats-labeled
            - type: record
              name: basic_watershed
              fields:
                img_threshold:
                  type: float
                  inputBinding:
                    prefix: --img-threshold
                min_dist:
                  type: int
                  inputBinding:
                    prefix: --min-dist
                min_allowed_size:
                  type: int
                  inputBinding:
                    prefix: --min-size
                max_allowed_size:
                  type: int
                  inputBinding:
                    prefix: --max-size
                masking_radius:
                  type: int
                  inputBinding:
                    prefix: --masking-radius

      outputs:
        segmented:
          type: Directory
          outputBinding:
            glob: "5_Segmented/"

    in:
      decoded_loc: decoded_loc
      exp_loc: exp_loc
      aux_name:
        source: [stage_segmentation/aux_name, aux_name]
        pickValue: first_non_null
      fov_count:
        source: [stage_segmentation/fov_count, fov_count]
        pickValue: first_non_null
      binary_mask:
        source: [binary_mask, stage_segmentation/binary_mask_img_threshold, stage_segmentation/binary_mask_min_dist, stage_segmentation/binary_mask_min_allowed_size, stage_segmentation/binary_mask_max_allowed_size, stage_segmentation/binary_mask_masking_radius]
        valueFrom: |
          ${
            if(!self[1]){
              return self[0]
            } else {
              return {
                "img_threshold": self[1],
                "min_dist": self[2],
                "min_allowed_size": self[3],
                "max_allowed_size": self[4],
                "masking_radius": self[5]
              };
            }
          }
    out: [segmented]
