#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/qcDriver.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:latest

inputs:
  codebook:
    type:
      - type: record
        name: pkl
        fields:
          pkl:
            type: File
            inputBinding:
              prefix: --codebook-pkl
            doc: A codebook for this experiment, saved in a python pickle.
      - type: record
        name: exp
        fields:
          exp:
            type: Directory
            inputBinding:
              prefix: --codebook-exp
            doc: The location of an experiment.json file, which has the corresponding codebook for this experiment.
  segmentation_loc:
    type: Directory?
    inputBinding:
      prefix: --segmentation-loc
    doc: The location of the output from the segmentation step, if it was performed.

  data:
    type:
    - type: record
      name: pkl
      fields:
        spots:
          type: File?
          inputBinding:
            prefix: --spots-pkl
          doc: Spots found in this experiment, saved in a python pickle.
        transcripts:
          type: File
          inputBinding:
            prefix: --transcript-pkl
          doc: The output DecodedIntensityTable, saved in a python pickle.
    - type: record
      name: exp
      fields:
        exp:
          type: Directory
          inputBinding:
            prefix: --exp-output
          doc: The location of output of starfish runner step, 4_Decoded. Contains spots (if applicable) and netcdfs containing the DecodedIntensityTable.
        has_spots:
          type: boolean?
          inputBinding:
            prefix: --has-spots
          doc: If true, will look for spots within the experiment field.

  roi:
    type: File?
    inputBinding: 
      prefix: --roi
    doc: The location of the RoiSet.zip, if applicable.

  imagesize:
    - 'null'
    - type: record
      fields:
        - name: x-size
          type: int
          inputBinding:
            prefix: --x-size
          doc: x-dimension of image
        - name: y-size
          type: int
          inputBinding:
            prefix: --y-size
          doc: y-dimension of image
        - name: z-size
          type: int
          inputBinding:
            prefix: --z-size
          doc: number of z-stacks

  find_ripley:
    type: boolean?
    inputBinding:
      prefix: --run-ripley
    doc: If true, will run ripley K estimates to find spatial density measures.  Can be slow.
  
  save_pdf:
    type: boolean?
    inputBinding:
      prefix: --save-pdf
    doc: If true, will save graphical output to a pdf. Currently pdfs are bugged.

outputs:
  qc_metrics:
    type: Directory
    outputBinding:
      glob: "7_QC/"
