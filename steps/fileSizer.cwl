#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
  MultipleInputFeatureRequirement: {}
  InlineJavascriptRequirement: {}

inputs:
  example_dir:
    type: Directory

outputs:
  dir_size:
    type: long
    outputSource: formatter/dir_size

steps:
  size_dir:
    run:
      class: CommandLineTool
      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
        InitialWorkDirRequirement:
          listing:
            - $(inputs.example_dir)
      baseCommand: ["du", "-s", "--block-size=1MiB"]
      inputs:
        example_dir:
          type: Directory
          inputBinding:
            position: 0
      outputs:
        dir_size:
          type: stdout
    in:
      example_dir: example_dir
    out: [dir_size]

  formatter:
    run:
      class: ExpressionTool
      requirements:
        InlineJavascriptRequirement: {}
      expression: |
        ${
          return {dir_size: Number(inputs.len_str.contents.split("\t")[0])}
        }
      inputs:
        len_str:
          type: File
          loadContents: true
      outputs:
        dir_size:
          type: long
    in:
      len_str: size_dir/dir_size
    out: [dir_size]
