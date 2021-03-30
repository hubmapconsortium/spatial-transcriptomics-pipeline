#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.1

inputs:
  data_dir:
    type: Directory

outputs: 
  example:
    type: File
    outputSource: starfish/example_out

steps:
  starfish:
    run: steps/starfish.cwl
    in:
      data_dir: data_dir
    out: [example_out]
