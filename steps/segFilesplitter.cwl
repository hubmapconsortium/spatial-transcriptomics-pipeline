cwlVersion: v1.2
class: CommandLineTool
requirements:
  ShellCommandRequirement: {}

inputs:
  segDir:
    type: Directory
    doc: Directory with output from starfish segmentation step.

outputs:
  baysorDir:
    type: Directory[]
    doc: One directory for each FOV, containing the mask and csv files.
  outputBinding:
    glob: "baysor_*/"

arguments:
    - shellQuote: false
      valueFrom: >
        
