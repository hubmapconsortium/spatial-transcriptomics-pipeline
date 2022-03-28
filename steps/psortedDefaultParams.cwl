#!/usr/bin/env cwl-runner

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
            "aux_channel_intercept": aux_channel_intercept
            };
   }

inputs:
  exp_dir:
    type: Directory
    doc: Converted experiment with converted codebook from prior step.
  aux_names:
    type: string[]?
    doc: list of the aux view names.  Assumed none if blank.
  cache_read_order:
    type: string[]
    doc: Cache read order for files. Will strip any CH dimensions.
  aux_cache_read_order:
    type: string[]?
    doc: Cache read order for aux views.  Will strip any CH dimensions.


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
