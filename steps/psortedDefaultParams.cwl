#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  exp_dir:
    type: Directory
    doc: Converted experiment with converted codebook from prior step.
  parameter_json:
    type: File?
    doc: Json containing information for the dataset
  aux_names:
    type: string[]?
    doc: list of the aux view names.  Assumed none if blank.
  cache_read_order:
    type: string[]?
    doc: Cache read order for files. Will strip any CH dimensions.
  aux_cache_read_order:
    type: string[]?
    doc: Cache read order for aux views.  Will strip any CH dimensions.

outputs:
  codebook:
    type: File
    outputSource: execute_defaults/codebook
  round_count:
    type: int
    outputSource: stage_defaults/round_count
  fov_count:
    type: int
    outputSource: stage_defaults/fov_count
  channel_count:
    type: int
    outputSource: stage_defaults/channel_count
  zplane_count:
    type: int
    outputSource: stage_defaults/zplane_count
  round_offset:
    type: int
    outputSource: execute_defaults/round_offset
  fov_offset:
    type: int
    outputSource: execute_defaults/fov_offset
  channel_offset:
    type: int
    outputSource: execute_defaults/channel_offset
  zplane_offset:
    type: int
    outputSource: execute_defaults/zplane_offset
  file_format:
    type: string
    outputSource: execute_defaults/file_format
  file_vars:
    type: string[]
    outputSource: execute_defaults/file_vars
  cache_read_order:
    type: string[]
    outputSource: execute_defaults/cache_read_order
  aux_names:
    type: string[]
    outputSource: execute_defaults/aux_names
  aux_file_formats:
    type: string[]
    outputSource: execute_defaults/aux_file_formats
  aux_file_vars:
    type: string[]
    outputSource: execute_defaults/aux_file_vars
  aux_cache_read_order:
    type: string[]
    outputSource: execute_defaults/aux_cache_read_order
  aux_channel_count:
    type: int[]
    outputSource: execute_defaults/aux_channel_count
  aux_channel_slope:
    type: string[]
    outputSource: execute_defaults/aux_channel_slope
  aux_channel_intercept:
    type: string[]
    outputSource: execute_defaults/aux_channel_intercept

steps:
  stage_defaults:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema:
        valueFrom: |
          ${
            return {
              "class": "File",
              "location": "../input_schemas/psortedDefaultParams.json"
            };
          }
    out: [round_count, fov_count, channel_count, zplane_count, aux_tilesets_aux_names, cache_read_order, aux_tilesets_aux_cache_read_order, aux_tilesets_aux_channel_count]
    when: $(inputs.datafile != null)

  execute_defaults:
    run:
      cwlVersion: v1.2
      class: ExpressionTool

      requirements:
        InlineJavascriptRequirement: {}
        LoadListingRequirement:
          loadListing: shallow_listing
        InitialWorkDirRequirement:
          listing:
            - $(inputs.exp_dir)

      expression: |
        ${
          var cb = "BLANK";
          var aux = {};
          var aux_names = [];
          var aux_file_formats = [];
          var aux_file_vars = [];
          var aux_cache_read_order = [];
          var aux_channel_slope = [];
          var aux_channel_intercept = [];
          var cache = inputs.cache_read_order;
          var aux_channel_count = inputs.aux_channel_count;
          var ind = cache.indexOf("CH"); // remove channel if it was in the read order
          if(ind > -1){
            cache.splice(ind, 1);
          }
          for(var i=0; i<inputs.exp_dir.listing.length; i++){
            if(inputs.exp_dir.listing[i].basename=="pround_codebook.json"){
              cb = inputs.exp_dir.listing[i];
            }
          }
          for(var i=0; i<inputs.aux_names.length; i++){
            var aux_cache = inputs.aux_cache_read_order[i];
            aux_cache = aux_cache.split(";");
            var aux_ind = aux_cache.indexOf("CH");
            if(aux_ind > -1){
              aux_cache.splice(aux_ind, 1);
            }
            aux_cache = aux_cache.join(";");
            aux_names.push(inputs.aux_names[i]);
            aux_file_formats.push("PseudoCycle{}/MMStack_Pos{}_"+inputs.aux_names[i]+"ch{}.ome.tif");
            aux_file_vars.push("round;fov;channel");
            aux_cache_read_order.push(aux_cache);
            aux_channel_slope.push(1);
            aux_channel_intercept.push(0);
          }
          return {"codebook":              cb,
                  "round_offset":          0,
                  "fov_offset":            0,
                  "channel_offset":        0,
                  "zplane_offset":         0,
                  "file_format":           "PseudoCycle{}/MMStack_Pos{}_ch{}.ome.tif",
                  "file_vars":             ["round", "fov", "channel"],
                  "cache_read_order":      cache,
                  "aux_names":             aux_names,
                  "aux_file_formats":      aux_file_formats,
                  "aux_file_vars":         aux_file_vars,
                  "aux_cache_read_order":  aux_cache_read_order,
                  "aux_channel_slope":     aux_channel_slope,
                  "aux_channel_intercept": aux_channel_intercept,
                  "aux_channel_count":     aux_channel_count
                  };
         }

      inputs:
        exp_dir:
          type: Directory
        aux_names:
          type: string[]?
        cache_read_order:
          type: string[]
        aux_cache_read_order:
          type: string[]?
        aux_channel_count:
          type: int[]?

      outputs:
        codebook:
          type: File
        round_offset:
          type: int
        fov_offset:
          type: int
        channel_offset:
          type: int
        zplane_offset:
          type: int
        file_format:
          type: string
        file_vars:
          type: string[]
        cache_read_order:
          type: string[]
        aux_names:
          type: string[]
        aux_file_formats:
          type: string[]
        aux_file_vars:
          type: string[]
        aux_cache_read_order:
          type: string[]
        aux_channel_slope:
          type: string[]
        aux_channel_intercept:
          type: string[]
        aux_channel_count:
          type: int[]
    in:
      exp_dir: exp_dir
      aux_names:
        source: [stage_defaults/aux_tilesets_aux_names, aux_names]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]){
              return self[1];
            } else {
              return null;
            }
          }
      cache_read_order:
        source: [stage_defaults/cache_read_order, cache_read_order]
        pickValue: first_non_null
      aux_channel_count:
        source: stage_defaults/aux_tilesets_aux_channel_count
      aux_cache_read_order:
        source: [stage_defaults/aux_tilesets_aux_cache_read_order, aux_cache_read_order]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if (self[1]){
              return self[1];
            } else {
              return null;
            }
          }

    out: [codebook, round_offset, fov_offset, channel_offset, zplane_offset, file_format, file_vars, cache_read_order, aux_names, aux_file_formats, aux_file_vars, aux_cache_read_order, aux_channel_slope, aux_channel_intercept, aux_channel_count]
