class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/spaceTxConverter.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/main:latest

inputs:
  raw_dir:
    type: Directory
    inputBinding:
      prefix: --raw-dir
    doc: Directory with image files

  output_dir:
    type: string
    inputBinding:
      prefix: --output-dir
    doc: Directory for initial output images

  output_dir_aligned:
    type: string
    inputBinding:
      prefix: --output-dir-aligned
    doc: Directory for aligned output images

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

  channel-dic-other:
    type: string[]
    inputBinding:
      prefix: --channel-dic-other
    doc: DIC channel for other data-containing folders


outputs:
# look up how to re-stage output based on input strings/paths? later
