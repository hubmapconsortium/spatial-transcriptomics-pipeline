#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  codebook:
    type:
      - type: record
        name: pkl
        fields:
          pkl:
            type: File
            doc: A codebook for this experiment, saved in a python pickle.
      - type: record
        name: exp
        fields:
          exp:
            type: Directory
            doc: The location of an experiment.json file, which has the corresponding codebook for this experiment.
  segmentation_loc:
    type: Directory?
    doc: The location of the output from the segmentation step, if it was performed.

  data:
    type:
    - type: record
      name: pkl
      fields:
        spots:
          type: File?
          doc: Spots found in this experiment, saved in a python pickle.
        transcripts:
          type: File
          doc: The output DecodedIntensityTable, saved in a python pickle.
    - type: record
      name: exp
      fields:
        exp:
          type: Directory
          doc: The location of output of starfish runner step, 4_Decoded. Contains spots (if applicable) and netcdfs containing the DecodedIntensityTable.

  has_spots:
    type: boolean?
    doc: If true, will look for spots within the experiment field.

  roi:
    type: File?
    doc: The location of the RoiSet.zip, if applicable.

  parameter_json:
    type: File?
    doc: The json with parameters to be read in for the following variables.

  imagesize:
    type:
      - 'null'
      - type: record
        fields:
          - name: x_size
            type: int
            doc: x-dimension of image
          - name: y_size
            type: int
            doc: y-dimension of image
          - name: z_size
            type: int
            doc: number of z-stacks

  find_ripley:
    type: boolean?
    doc: If true, will run ripley K estimates to find spatial density measures.  Can be slow.
    default: False

  save_pdf:
    type: boolean?
    doc: If true, will save graphical output to a pdf.
    default: True

outputs:
  qc_metrics:
    type: Directory
    outputSource: execute_qc/qc_metrics

steps:

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: ghcr.io/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:latest

      inputs:
        schema:
          type: string
          inputBinding:
            position: 1

      outputs:
        data:
          type: stdout

    in:
      schema:
        valueFrom: "/opt/qc.json"
    out: [data]

  stage_qc:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [find_ripley, save_pdf, fov_positioning_x_shape, fov_positioning_y_shape, fov_positioning_z_shape, decoding_decode_method]
    when: $(inputs.datafile != null)

  execute_qc:
    run:
      class: CommandLineTool
      baseCommand: /opt/qcDriver.py

      requirements:
        DockerRequirement:
          dockerPull: ghcr.io/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:latest

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
            - type: record
              name: exp
              fields:
                exp:
                  type: Directory
                  inputBinding:
                    prefix: --codebook-exp

        segmentation_loc:
          type: Directory?
          inputBinding:
            prefix: --segmentation-loc

        data:
          type:
          - type: record
            name: pkl
            fields:
              spots:
                type: File?
                inputBinding:
                  prefix: --spots-pkl
              transcripts:
                type: File
                inputBinding:
                  prefix: --transcript-pkl
          - type: record
            name: exp
            fields:
              exp:
                type: Directory
                inputBinding:
                  prefix: --exp-output

        has_spots:
          type: boolean?
          inputBinding:
            prefix: --has-spots

        roi:
          type: File?
          inputBinding: 
            prefix: --roi

        imagesize:
          - 'null'
          - type: record
            fields:
              - name: x_size
                type: int
                inputBinding:
                  prefix: --x-size
              - name: y_size
                type: int
                inputBinding:
                  prefix: --y-size
              - name: z_size
                type: int
                inputBinding:
                  prefix: --z-size

        find_ripley:
          type: boolean?
          inputBinding:
            prefix: --run-ripley

        save_pdf:
          type: boolean?
          inputBinding:
            prefix: --save-pdf

      outputs:
        qc_metrics:
          type: Directory
          outputBinding:
            glob: "7_QC/"
    in:
      codebook: codebook
      segmentation_loc: segmentation_loc
      has_spots:
        source: [stage_qc/decoding_decode_method, has_spots]
        valueFrom: |
          ${
             if(self[0] || self[1]){
               return true;
             } else {
               return false;
             }
          }
      data: data
      roi: roi
      imagesize:
        source: [imagesize, stage_qc/fov_positioning_x_shape, stage_qc/fov_positioning_y_shape, stage_qc/fov_positioning_z_shape]
        valueFrom: |
          ${
            if(!self[1]){
              return self[0];
            } else {
              return {
                "x_size": self[1],
                "y_size": self[2],
                "z_size": self[3]
              };
            }
          }
      find_ripley:
        source: [stage_qc/find_ripley, find_ripley]
        pickValue: first_non_null
      save_pdf:
        source: [stage_qc/save_pdf, save_pdf]
        pickValue: first_non_null

    out: [qc_metrics]
