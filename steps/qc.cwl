#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/qcRunner.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish:latest

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
            type: File
            inputBinding:
              prefix: --codebook-exp
            doc: The location of an experiment.json file, which has the corresponding codebook for this experiment.
  spots:
    - 'null'
    - type:
        - type: record
          name: pkl
          fields:
            pkl:
              type: File
              inputBinding:
                prefix: --spots-pkl
              doc: Spots found in this experiment, saved in a python pickle.
        - type: record
          name: exp
          fields:
            exp:
              type: File
              inputBinding:
                prefix: --spots-exp
              doc: The location of OUTPUT FROM EXPERIMENT. NETCDF?

  transcripts:
    type:
      - type: record
        name: pkl
        fields:
          pkl:
            type: File
            inputBinding:
              prefix: --transcript-pkl
            doc: The output DecodedIntensityTable, saved in a python pickle.
      - type: record
        name: exp
        fields:
          exp:
            type: File
            inputBinding:
              prefix: --transcript-exp
            doc: The location of OUTPUT FROM EXPERIMENT. NETCDF?

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
            prefix:  --y-size
          doc: y-dimension of image
        - name: z-size
          type: int
          inputbinding:
            prefix: --z-size
          doc: number of z-stacks

  find-ripley:
    type: boolean?
    inputBinding:
      prefix: --run-ripley
    doc: If true, will run ripley K estimates to find spatial density measures.  Can be slow.
