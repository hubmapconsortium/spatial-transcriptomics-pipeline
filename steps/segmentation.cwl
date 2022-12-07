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

  mask_roi_files:
    type: Directory?
    doc: Flattened directory input, refer to record entry "binary_mask"

  mask_roi_formats:
    type: string?
    doc: Flattened record input, refer to record entry "binary_mask"

  mask_labeled_files:
    type: Directory?
    doc: Flattened file input, refer to record entry "binary_mask"

  mask_labeled_formats:
    type: string?
    doc: Flattened record input, refer to record entry "binary_mask"

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
    - type: record
      name: density_based
      fields:
        nuclei_view:
          type: string
          doc: Name of the auxillary view with nuclei data
        cyto_seg:
          type: boolean
          doc: If true, the cytoplasm will be segmented
        correct_seg:
          type: boolean
          doc: If true, suspected nuclei/cytoplasms that overlap will be removed.
        border_buffer:
          type: int
          doc: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.
        area_thresh:
          type: float
          doc: Threshold used when determining if an object is one nucleus or two or more overlapping nuclei. Objects whose ratio of convex hull area to normal area are above this threshold are removed if the option to remove overlapping nuclei is set.
        thresh_block_size:
          type: int
          doc: Size of structuring element for local thresholding of nuclei. If nuclei interiors aren't passing threshold, increase this value, if too much non-nuclei is passing threshold, lower it.
        watershed_footprint_size:
          type: int
          doc: Size of structuring element for watershed segmentation. Larger values will segment the nuclei into larger objects and smaller values will result in smaller objects. Adjust according to nucleus size.
        label_exp_size:
          type: int
          doc:  Pixel size labels are dilated by in final step. Helpful for closing small holes that are common from thresholding but can also cause cell boundaries to exceed their true boundaries if set too high. Label dilation respects label borders and does not mix labels.

outputs:
  segmented:
    type: Directory
    outputSource: execute_segmentation/segmented

steps:
  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:2.11

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
        valueFrom: "/opt/segmentation.json"
    out: [data]

  stage_segmentation:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [aux_name, fov_count, binary_mask_img_threshold, binary_mask_min_dist, binary_mask_min_allowed_size, binary_mask_max_allowed_size, binary_mask_masking_radius, binary_mask_nuclei_view, binary_mask_cyto_seg, binary_mask_correct_seg, binary_mask_border_buffer, binary_mask_area_thresh, binary_mask_thresh_block_size, binary_mask_watershed_footprint_size, binary_mask_label_exp_size]
    when: $(inputs.datafile != null)
  execute_segmentation:
    run:
      class: CommandLineTool
      baseCommand: /opt/segmentationDriver.py

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:2.11

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
          type: string?
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
            - type: record
              name: density_based
              fields:
                nuclei_view:
                  type: string
                  inputBinding:
                    prefix: --nuclei-view
                cyto_seg:
                  type: boolean
                  inputBinding:
                    prefix: --cyto-seg
                correct_seg:
                  type: boolean
                  inputBinding:
                    prefix: --correct-seg
                border_buffer:
                  type: int
                  inputBinding:
                    prefix: --border-buffer
                area_thresh:
                  type: float
                  inputBinding:
                    prefix: --area-thresh
                thresh_block_size:
                  type: int
                  inputBinding:
                    prefix: --thresh-block-size
                watershed_footprint_size:
                  type: int
                  inputBinding:
                    prefix: --watershed-footprint-size
                label_exp_size:
                  type: int
                  inputBinding:
                    prefix: --label-exp-size

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
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if (self[1]){
              return self[1];
            } else {
              return null;
            }
          }
      fov_count:
        source: [stage_segmentation/fov_count, fov_count]
        pickValue: first_non_null
      binary_mask:
        source: [binary_mask, stage_segmentation/binary_mask_img_threshold, stage_segmentation/binary_mask_min_dist, stage_segmentation/binary_mask_min_allowed_size, stage_segmentation/binary_mask_max_allowed_size, stage_segmentation/binary_mask_masking_radius, stage_segmentation/binary_mask_nuclei_view, stage_segmentation/binary_mask_cyto_seg, stage_segmentation/binary_mask_correct_seg, stage_segmentation/binary_mask_border_buffer, stage_segmentation/binary_mask_area_thresh, stage_segmentation/binary_mask_thresh_block_size, stage_segmentation/binary_mask_watershed_footprint_size, stage_segmentation/binary_mask_label_exp_size, mask_roi_files, mask_roi_formats, mask_labeled_files, mask_labeled_formats]
        valueFrom: |
          ${
            if(self[14] && self[15]) {
              return {
                "roi_set": self[14],
                "file_formats": self[15]
              }
            } else if(self[16] && self[17]){
              return {
                "labeled_image": self[16],
                "file_formats_labeled": self[17]
              };
            } else if(!self[1] && !self[6]){
              return self[0];
            } else if(self[1]){
              return {
                "img_threshold": self[1],
                "min_dist": self[2],
                "min_allowed_size": self[3],
                "max_allowed_size": self[4],
                "masking_radius": self[5]
              };
            } else {
              return {
                "nuclei_view": self[6],
                "cyto_seg": self[7],
                "correct_seg": self[8],
                "border_buffer": self[9],
                "area_thresh": self[10],
                "thresh_block_size": self[11],
                "watershed_footprint_size": self[12],
                "label_exp_size": self[13]
              };
            }
          }
    out: [segmented]
