cwlVersion: v1.2
#class: ExpressionTool
class: Workflow
requirements:
  StepInputExpressionRequirement: {}
  ScatterFeatureRequirement: {}
  InlineJavascriptRequirement: {}

inputs:
  segDir:
    type: Directory
    doc: Directory with output from starfish segmentation step

outputs: []
  #csv:
  #  type: string[]
  #prior:
  #  type: string

steps:
  zero:
    run:
      class: ExpressionTool
      requirements: {InlineJavascriptRequirement: {}}
      inputs:
        dir: Directory
      expression: |
        ${
          return {
            "prior":inputs.segDir.listing
          };
        }
      outputs:
        prior: File[]
    in:
      dir: segDir
    out: [prior]
  one:
    run:
      class: CommandLineTool
      inputs:
        file:
          type: File
          inputBinding: {}
      baseCommand: echo
      outputs: []
    in:
      file: zero/prior
    scatter: file
    out: []
      
