#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:

  codebook_exp:
    type: Directory?
    doc: Flattened codebook input, refer to record entry.

  codebook_pkl:
    type: File?
    doc: Flattened codebook input, refer to record entry.

  locs_json:
    type: File?
    doc: Flattened json input, refer to record entry.

  codebook:
    type:
      - 'null'
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

  data_pkl_spots:
    type: File?
    doc: Flattened data input, refer to record entry.

  data_pkl_transcripts:
    type: File?
    doc: Flattened data input, refer to record entry.

  data_exp:
    type: Directory?
    doc: Flattened data input, refer to record entry.

  data:
    type:
    - 'null'
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

  selected_fovs:
    type: int[]?
    doc: If provided, QC will only be run on FOVs with these indices.

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
        name: locs
        fields:
          locs:
            type: File?
            doc: Input locations as a json file, using the same records as below.
      - type: record
        name: fov_positioning
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

  spot_threshold:
    type: float?
    doc: If has_spots is true and this is provided, spots with an intensity lower than this will not be included in qc metrics

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

  tmpname:
    run: tmpdir.cwl
    in: []
    out: [tmp]

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:3.4
        ResourceRequirement:
          ramMin: 1000
          tmpdirMin: 1000
          outdirMin: 1000

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
    out: [selected_fovs, find_ripley, save_pdf, fov_positioning_x_shape, fov_positioning_y_shape, fov_positioning_z_shape, decoding_decode_method, decoding_magnitude_threshold, decoding_decoder_min_intensity]
    when: $(inputs.datafile != null)

  codebook_grabber:
    run:
      class: ExpressionTool
      requirements:
        - class: InlineJavascriptRequirement
        - class: LoadListingRequirement

      inputs:
        experiment:
          type: Directory?
          doc: A directory containing a spaceTx-formatted experiment

      outputs:
        codebook:
          type: File?

      expression: |
        ${
          for(var i=0;i<inputs.experiment.listing.length; i++){
            if(inputs.experiment.listing[i].basename.includes("codebook.json")){
              return {"codebook": inputs.experiment.listing[i]};
            }
          }
          return null;
        }
    in:
      experiment:
        source: [codebook, codebook_exp]
        valueFrom: |
          ${
            if(self[1]){
              return self[1];
            } else if(self[0] && "exp" in self[0]) {
              return self[0].exp;
            } else {
              return null;
            }
          }
    out:
      [codebook]
    when: $(inputs.experiment != null)


  execute_qc:
    run:
      class: CommandLineTool
      baseCommand: /opt/qcDriver.py

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:3.4
        InitialWorkDirRequirement:
          listing:
            - entryname: "$('input_dir_' + inputs.tmp_prefix)"
              writable: true
              entry: "$(inputs.data_exp_dir)"

      inputs:
        tmp_prefix:
          type: string
          inputBinding:
            prefix: --tmp-prefix
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
                  type: File
                  inputBinding:
                    prefix: --codebook-file

        segmentation_loc:
          type: Directory?
          inputBinding:
            prefix: --segmentation-loc

        data_exp_dir:
          type: Directory

        data:
          type:
          - 'null'
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

        data_exp_staged:
          type: string
          inputBinding:
            prefix: --exp-output

        selected_fovs:
          type: int[]?
          inputBinding:
            prefix: --selected-fovs
          doc: If provided, processing will only be run on FOVs with these indices.

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
            name: locs
            fields:
              locs:
                type: File?
                inputBinding:
                  prefix: --loc-json
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

        spot_threshold:
          type: float?
          inputBinding:
            prefix: --spot-threshold

        find_ripley:
          type: boolean?
          inputBinding:
            prefix: --run-ripley
          default: False

        save_pdf:
          type: boolean?
          inputBinding:
            prefix: --save-pdf
          default: True

      outputs:
        qc_metrics:
          type: Directory
          outputBinding:
            glob: $("tmp/" + inputs.tmp_prefix + "/7_QC/")
    in:
      tmp_prefix: tmpname/tmp
      codebook:
        source: [codebook, codebook_grabber/codebook, codebook_pkl]
        valueFrom: |
          ${
            if(self[1]){
              return {exp: self[1]};
            } else if(self[0]) {
              return self[0];
            } else {
              return {pkl: self[2]}
            }
          }
      segmentation_loc: segmentation_loc
      selected_fovs:
        source: [stage_qc/selected_fovs, selected_fovs]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      has_spots:
        source: [stage_qc/decoding_decode_method, has_spots]
        valueFrom: |
          ${
             if((self[0] && self[0].length) || self[1]){
               return true;
             } else {
               return false;
             }
          }
      data:
        source: [data, data_pkl_spots, data_pkl_transcripts]
        valueFrom: |
          ${
            if(self[0] && !("exp" in self[0])){
              return self[0];
            } else if(self[2]){
              return {pkl: {spots: self[1], transcripts: self[2]}};
            } else {
              return null;
            }
          }
      data_exp_dir:
        source: [data, data_exp]
        valueFrom: |
          ${
            if(self[0] && "exp" in self[0]){
              return self[0].exp;
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      data_exp_staged:
        valueFrom: $("input_dir_" + inputs.tmp_prefix)
      roi: roi
      imagesize:
        source: [imagesize, stage_qc/fov_positioning_x_shape, stage_qc/fov_positioning_y_shape, stage_qc/fov_positioning_z_shape, locs_json]
        valueFrom: |
          ${
            if(!self[1]){
              if(self[0]){
                return self[0];
              } else if(self[4]){
                return {"locs": self[4]};
              } else {
                return null;
              }
            } else {
              return {
                "x_size": self[1],
                "y_size": self[2],
                "z_size": self[3]
              };
            }
          }
      spot_threshold:
        source: [stage_qc/decoding_decoder_min_intensity, stage_qc/decoding_magnitude_threshold, spot_threshold]
        valueFrom: |
          ${
             if(self[0]){
               return self[0];
             } else if(self[1]) {
               return self[1];
             } else if(self[2]){
               return self[2];
             } else {
               return null;
             }
          }
      find_ripley:
        source: [stage_qc/find_ripley, find_ripley]
        pickValue: first_non_null
      save_pdf:
        source: [stage_qc/save_pdf, save_pdf]
        pickValue: first_non_null

    out: [qc_metrics]
