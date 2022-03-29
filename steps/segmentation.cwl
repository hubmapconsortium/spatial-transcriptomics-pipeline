#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/segmentationDriver.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:2.01

inputs:
  decoded_loc:
    type: Directory
    inputBinding:
      prefix: --decoded-loc
    doc: Location of the directory that is the output from the starfishRunner step.

  exp_loc:
    type: Directory
    inputBinding:
      prefix: --exp-loc
    doc: Location of directory containing the 'experiment.json' file

  aux_name:
    type: string
    inputBinding:
      prefix: --aux-name
    doc: The name of the aux view to look at in the experiment file.

  fov_count:
    type: int
    inputBinding:
      prefix: --fov-count
    doc: The number of FOVs that are included in this experiment

  binary_mask:
    type:
      - type: record
        name: roi_set
        fields:
          roi_set:
            type: Directory
            inputBinding:
              prefix: --roi-set
            doc: Directory of RoiSet.zip for each fov, from fiji segmentation
          file_formats:
            type: string
            inputBinding:
              prefix: --file-formats
            doc: Layout for name of each RoiSet.zip, per fov. Will be formatted with String.format([fov index]).
      - type: record
        name: labeled_image
        fields:
          labeled_image:
            type: Directory
            inputBinding:
              prefix: --labeled-image
            doc: Directory of labeled images with image segmentation data, such as from ilastik classification.
          file_formats_labeled:
            type: string
            inputBinding:
              prefix: --file-formats-labeled
            doc: Layout for name of each labelled image. Will be formatted with String.format([fov index])
      - type: record
        name: basic_watershed
        fields:
          img_threshold:
            type: float
            inputBinding:
              prefix: --img-threshold
            doc: Global threshold value for images
          min_dist:
            type: int
            inputBinding:
              prefix: --min-dist
            doc: minimum distance (pixels) between distance transformed peaks
          min_allowed_size:
            type: int
            inputBinding:
              prefix: --min-size
            doc: minimum size for a cell (in pixels)
          max_allowed_size:
            type: int
            inputBinding:
              prefix: --max-size
            doc: maxiumum size for a cell (in pixels)
          masking_radius:
            type: int
            inputBinding:
              prefix: --masking-radius
            doc: Radius for white tophat noise filter

outputs:
  segmented:
    type: Directory
    outputBinding:
      glob: "5_Segmented/"
