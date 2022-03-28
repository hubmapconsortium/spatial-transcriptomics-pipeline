#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.2
baseCommand: ["baysor","run"]

requirements:
  DockerRequirement:
    #dockerPull: vpetukhov/baysor:v0.5.0
    #dockerPull: waltsbaysor:latest
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/baysor:2.0

inputs:
  csv:
    type: File
    inputBinding:
      position: 6
    doc: csv with transcript information
  priors:
    type: File?
    inputBinding:
      position: 7
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
    default: x_min
  y_col:
    type: string?
    inputBinding:
      position: 3
      prefix: -y
    default: y_min
    doc: Name of the column with y information
  z_col:
    type: string?
    inputBinding:
      position: 5
      prefix: -z
    default: z_min
    doc: Name of the column with z information
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
