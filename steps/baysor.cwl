#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.2
baseCommand: ["baysor","run"]

requirements:
  DockerRequirement:
    dockerPull: vpetukhov/baysor@sha256:ce58af2bbd81ca29f7382497223afe9dbfbcc674e810155964722b447b676087
    #dockerPull: hubmap/baysor:latest

inputs:
  csv:
    type: File
    inputBinding:
      position: 5
    doc: csv with transcript information
  priors:
    type: File?
    inputBinding:
      position: 6
    doc: Binary Mask image with prior segmentation.
  scale:
    type: int?
    inputBinding:
      position: 1
      prefix: -s
    doc: Expected scale equal to cell radius in the same units as x, y, and z.
  x_col:
    type: string?
    inputBinding:
      position: 2
      prefix: -x
    doc: Name of the column with x information
    default: x
  y_col:
    type: string?
    inputBinding:
      position: 3
      prefix: -y
    default: y
    doc: Name of the column with y information
  gene_col:
    type: string?
    inputBinding:
      position: 4
      prefix: --gene
    default: target
    doc: Name of the column with gene names

outputs:
  segmented:
    type: File[]
    outputBinding:
      glob: "segmentation*"

stdout: baysor_stdout.log
