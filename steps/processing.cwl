#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/imgProcessing.py

requirements:
  DockerRequirement:
      dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:latest

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

