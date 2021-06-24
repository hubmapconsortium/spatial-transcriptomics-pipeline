#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/alignerDriver.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/alignment:1.0
  InitialWorkDirRequirement:
    listing:
      - $(inputs.raw_dir)

inputs:
  raw_dir:
    type: Directory
    inputBinding:
      prefix: --raw-dir
      valueFrom: $(self.basename)
    doc: Directory with image files

  fov_count:
    type: int
    inputBinding:
      prefix: --fov-count
    doc: The number of FoVs

  round_list:
    type: string[]
    inputBinding:
      prefix: --round-list
    doc: The names of the rounds

  sigma:
    type: float
    inputBinding:
      prefix: --sigma
    doc: Value used for Gaussian blur

  cycle_ref_ind:
    type: int
    inputBinding:
      prefix: --cycle-ref-ind
    doc: Which cycle to align to

  channel_DIC_reference:
    type: string
    inputBinding:
      prefix: --channel-dic-reference
    doc: DIC channel for reference cycle

  channel_DIC:
    type: string
    inputBinding:
      prefix: --channel-dic
    doc: DIC channel for (non-reference) decoding cycles (the channel we use for finding alignment parameters)

  cycle_other:
    type: string[]
    inputBinding:
      prefix: --cycle-other
    doc: if there are other data-containing folders which need to be aligned but are not named "CycleXX"

  channel_DIC_other:
    type: string[]
    inputBinding:
      prefix: --channel-dic-other
    doc: DIC channel for other data-containing folders

  skip_projection:
    type: boolean?
    inputBinding:
      prefix: --skip-projection
    doc: If true, will skip z-axis projection before alignment step.

  skip_align:
    type: boolean?
    inputBinding:
      prefix: --skip-align
    doc: If true, will skip alignment of images across rounds prior to spacetx conversion

outputs:
  projected:
    type: Directory
    outputBinding:
      glob: "1_Projected/"

  registered:
    type: Directory
    outputBinding:
      glob: "2_Registered/"

  tool_out:
    type: stdout

stdout: SITK_stdout.log
