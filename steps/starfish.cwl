#!/usr/bin/env cwl-runner
cwlVersion: v1.1
class: CommandLineTool
requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/main:latest

inputs:
  data_dir: 
    type: Directory
    inputBinding:
      position: 1

outputs: 
  example_out:
    type: stdout

baseCommand: /opt/main.py
