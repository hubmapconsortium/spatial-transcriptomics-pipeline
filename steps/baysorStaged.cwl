#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2
requirements:
  ScatterFeatureRequirement: {}
inputs:
  segmented: Directory
outputs:
  baysor:
    type: Directory
    outputSource: restage/pool_dir

steps:
  stage:
    run:
      class: CommandLineTool
      baseCommand: [ls]
      requirements:
        DockerRequirement:
          dockerPull: ubuntu:latest
        InitialWorkDirRequirement:
          listing:
            - $(inputs.segDir)
      inputs:
        segDir:
          type: Directory
          doc: Directory with output from starfish segmentation step.
      outputs:
        csvs:
          type:
            type: array
            items: File
          outputBinding:
            glob: "**/**/segmentation.csv"
        priors:
          type:
            type: array
            items: File
          outputBinding:
            glob: "**/**/mask.tiff"
    in:
      segDir: segmented
    out: [csvs, priors]
  baysor:
    run: baysor.cwl
    in:
      csv: stage/csvs
      priors: stage/priors
    scatter: [csv, priors]
    scatterMethod: dotproduct
    out: [segmented]
  restage:
    run:
      class: ExpressionTool
      requirements:
        InlineJavascriptRequirement: {}
      inputs:
        file_array:
          type:
            type: array
            items:
              type: array
              items: File
      outputs:
        pool_dir: Directory
      expression: |
        ${ var dir = [];
           for(var i=0;i<inputs.file_array.length; i++){
             dir.push({"class": "Directory", "basename": "fov_"+String(i).padStart(3,'0'), "listing": inputs.file_array[i]});
           }
           return {"pool_dir": {
             "class": "Directory",
             "basename": "6_Baysor",
             "listing": dir}
           }; }
    in:
      file_array: baysor/segmented
    out: [pool_dir]
