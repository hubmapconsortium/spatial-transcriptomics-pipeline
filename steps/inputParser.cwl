#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: ExpressionTool

requirements:
  - class: InlineJavascriptRequirement

inputs:
  datafile:
    type: File
    inputBinding:
      loadContents: true

  schema:
    type: File
    inputBinding:
      loadContents: true

# all possible outputs from tool must be listed here.
# not every output needs to be present in workflows that call this.
outputs:
  round_count: int
  zplane_count: int
  channel_count: int
  fov_count: int
  round_offset: int
  fov_offset: int
  zplane_offset: int
  channel_offset: int
  channel_slope: float
  file_format: string
  file_vars: string[]
  cache_read_order: string[]
  aux_tilesets_aux_names: string[]
  aux_tilesets_aux_file_formats: string[]
  aux_tilesets_aux_file_vars: string[]
  aux_tilesets_aux_cache_read_order: string[]
  aux_tilesets_aux_single_round: string[]
  aux_tilesets_aux_channel_count: float[]
  aux_tilesets_aux_channel_slope: float[]
  aux_tilesets_aux_channel_intercept: int[]
  fov_positioning_x_locs: string
  fov_positioning_x_shape: int
  fov_positioning_x_voxel: float
  fov_positioning_y_locs: string
  fov_positioning_y_shape: int
  fov_positioning_y_voxel: float
  fov_positioning_z_locs: string
  fov_positioning_z_shape: int
  fov_positioning_z_voxel: float
  add_blanks: boolean
  skip_formatting: boolean
  skip_processing: boolean
  selected_fovs: int[]
  clip_min: float
  clip_max: float
  level_method: string
  register_aux_view: string
  register_to_primary: boolean
  channels_per_reg: int
  background_view: string
  register_background: boolean
  anchor_view: string
  high_sigma: int
  deconvolve_iter: int
  deconvolve_sigma: int
  low_sigma: int
  rolling_radius: int
  match_histogram: boolean
  tophat_radius: int
  use_ref_img: boolean
  is_volume: boolean
  rescale: boolean
  not_filtered_results: boolean
  n_processes: int
  decoding_min_sigma: float[]
  decoding_max_sigma: float[]
  decoding_num_sigma: int
  decoding_threshold: float
  decoding_overlap: float
  decoding_decode_method: string
  decoding_decoder_trace_building_strategy: string
  decoding_decoder_max_distance: float
  decoding_decoder_min_intensity: float
  decoding_decoder_metric: string
  decoding_decoder_norm_order: int
  decoding_decoder_anchor_round: int
  decoding_decoder_search_radius: int
  decoding_decoder_return_original_intensities: boolean
  decoding_decoder_error_rounds: int
  decoding_decoder_mode: string
  decoding_decoder_physical_coords: boolean
  decoding_metric: string
  decoding_distance_threshold: float
  decoding_magnitude_threshold: float
  decoding_min_area: int
  decoding_max_area: int
  decoding_norm_order: int
  decoding_composite_decode: boolean
  decoding_composite_pmin: float
  decoding_composite_pmax: float
  run_cellpose: boolean
  use_mrna: boolean
  pretrained_model_str: string
  diameter: float
  flow_threshold: float
  stitch_threshold: float
  cellprob_threshold: float
  border_buffer: int
  label_exp_size: int
  min_allowed_size: int
  max_allowed_size: int
  aux_views: string[]
  aux_name: string
  binary_mask_img_threshold: float
  binary_mask_min_dist: int
  binary_mask_min_allowed_size: int
  binary_mask_max_allowed_size: int
  binary_mask_masking_radius: int
  binary_mask_nuclei_view: string
  binary_mask_cyto_seg: string
  binary_mask_correct_seg: boolean
  binary_mask_border_buffer: int
  binary_mask_area_thresh: float
  binary_mask_thresh_block_size: int
  binary_mask_watershed_footprint_size: int
  binary_mask_label_exp_size: int
  run_baysor: boolean
  skip_qc: boolean
  find_ripley: boolean
  save_pdf: boolean

# input schema describes the expected layout of variables in json format.
# inputs are stored in an array.
# any items that are treated as records for cwl input are stored in an object, where the key is the prefix on all items in the object.
# the value in an object is an array or an array of arrays.
# if there are two nested arrays, the sub-array with the closest match is used, ie the sub-arrays are mutually exclusive.
# all items in an object's array must be included in the json file, unless the item ends with a question mark.
# objects can be nested inside other objects, and all of their prefixes will apply to all items.
expression: |
  ${   var data = JSON.parse(inputs.datafile.contents);
       var schema = JSON.parse(inputs.schema.contents);
       function enforce_record(data, key, items, output_dict){
           if(Array.isArray(items[0])){
                // Record where one of a mutually exclusive set of sublists is defined
                // Find closest match and enforce that
                var subind = 0;
                var coverage = 0.0;
                for(var i=0;i<items.length;i++){
                        var tally = 0.0;
                        for(var j=0;j<items[i].length;j++){
                                if(items[i][j].constructor != Object){
                                        var subkey = items[i][j].replace("?","");
                                        if(subkey in data){
                                                tally++;
                                        }
                                        // we want to define this key as null so that
                                        // all possible outputs from schema are accounted for,
                                        // even for non-selected schemas
                                        output_dict[key + "_" + subkey] = null;
                                } else {
                                        var sbk = Object.keys(items[i][j])[0];
                                        for(var k=0;k<items[i][j][sbk].length; k++){
                                                var sublis = items[i][j][sbk][k];
                                                for(var m=0;m<sublis.length;m++){
                                                        var cleaned = sublis[m].replace("?","");
                                                        output_dict[key + "_" + sbk + "_" + cleaned] = null;
                                                }
                                        }
                                }
                        }
                        var new_coverage = tally / items[i].length;
                        if(new_coverage > coverage){
                                coverage = new_coverage;
                                subind = i;
                        }
                }
                enforce_record(data, key, items[subind], output_dict);
           } else {
                   for(var i=0;i<items.length;i++){
                        if(items[i].constructor == Object){
                                // Record within a record, make recursive call
                                var subkey = Object.keys(items[i])[0];
                                enforce_record(data[subkey], key+"_"+subkey, items[i][subkey], output_dict);
                        } else {
                                // If not present, throw an error unless marked with "?"
                                var item_comp = items[i].replace("?","");
                                if(!(item_comp in data)){
                                        if(!items[i].includes("?")){
                                                throw 'If '+key+' is defined, then all of '+items+' must be defined.';
                                        }
                                } else {
                                // Add val to output dict
                                        output_dict[key + '_' + item_comp] = data[item_comp];
                                }
                        }
                   }
                   return output_dict;
           }
       }

       var values_dict = {};
       for(var i=0;i<schema.length;i++){
           if(schema[i].constructor == Object){
                var key = Object.keys(schema[i])[0];
                var lis = schema[i][key];
                enforce_record(data[key], key, lis, values_dict);
           } else {
                var sch = schema[i].replace("?","");
                if(sch in data){
                     values_dict[sch] = data[sch];
                } else {
                     values_dict[sch] = null;
                }
           }
       }
       return values_dict;
  }
