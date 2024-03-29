#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement
   - class: ScatterFeatureRequirement

inputs:
  exp_loc:
    type: Directory
    doc: Location of directory containing starfish experiment.json file

  dir_size:
    type: long?
    doc: Size of exp_loc, in MiB. If provided, will be used to calculate ResourceRequirement.

  parameter_json:
    type: File?
    doc: JSON formatted input parameters.

  selected_fovs:
    type: int[]?
    doc: If provided, starfish will only be run on FOVs with these indices.

  fov_count:
    type: int?
    doc: The number of FOVs that are included in this experiment

  use_ref_img:
    type: boolean?
    doc: Whether to generate a reference image and use it alongside spot detection.
    default: False

  level_method:
    type: string?
    doc: Levelling method for clip and scale application. Defaults to SCALE_BY_IMAGE.

  anchor_view:
    type: string?
    doc: The name of the auxillary view to be used as a reference view, such as for anchor round in ISS processing. Will not be included if not provided.

  is_volume:
    type: boolean?
    doc: Whether to treat the zplanes as a 3D image.
    default: False

  rescale:
    type: boolean?
    doc: Whether to iteratively rescale images before running the decoder.

  not_filtered_results:
    type: boolean?
    doc: Will not remove genes that do not match a target and do not meet criteria.

  n_processes:
    type: int?
    doc: If provided, the number of processes that will be spawned for processing. Otherwise, the maximum number of available CPUs will be used.

  scatter_into_n:
    type: int?
    doc: If provided, the step to run decoding will be split into n batches, where each batch is (FOV count/n) FOVs big.

  decoding_blob:
    - 'null'
    - type: record
      name: dummy
      fields:
        dummy:
          type: string?
          doc: Added to prevent cli parsing of the decoding_blob record.
    - type: record
      name: blob
      fields:
        min_sigma:
          type: float[]?
          doc: Minimum sigma tuple to be passed to blob detector
        max_sigma:
          type: float[]?
          doc: Maximum sigma tuple to be passed to blob detector
        num_sigma:
          type: int?
          doc: The number of sigma values to be tested, passed to blob detector
        threshold:
          type: float?
          doc: Threshold of blob detection
        overlap:
          type: float?
          doc: Amount of overlap allowed between blobs, passed to blob detector
        detector_method:
          type: string?
          doc: Name of the scikit-image spot detection method to use
        composite_decode:
          type: boolean?
          doc: Whether to composite all FOVs into one image, typically for PoSTcode decoding.
        composite_pmin:
          type: float?
          doc: pmin value for clip and scale of composite image.
        composite_pmax:
          type: float?
          doc: pmax value for clip and scale of composite image.
        decode_method:
          type: string
          doc: Method name for spot decoding. Refer to starfish documentation.
        decoder:
          type:
            - type: record
              name: metric_distance
              fields:
                trace_building_strategy:
                  type: string
                  doc: Which tracing strategy to use.  See starfish docs.
                max_distance:
                  type: float
                  doc: Maximum distance between spots.
                min_intensity:
                  type: float
                  doc: Minimum intensity of spots.
                pnorm:
                  type: int?
                  doc: Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance 2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance A finite large p may cause a ValueError if overflow can occur.
                norm_order:
                  type: int?
                  doc: Refer to starfish documentation for metric_distance
                anchor_round:
                  type: int?
                  doc: Anchor round for comparison.
                search_radius:
                  type: float?
                  doc: Distance to search for matching spots.
                return_original_intensities:
                  type: boolean?
                  doc: Return original intensities instead of normalized ones.
            - type: record
              name: per_round_max
              fields:
                trace_building_strategy:
                  type: string
                  doc: Which tracing strategy to use.  See starfish docs.
                anchor_round:
                  type: int?
                  doc: Round to refer to.  Required for nearest_neighbor.
                search_radius:
                  type: float?
                  doc: Distance to search for matching spots.
            - type: record
              name: check_all
              fields:
                search_radius:
                  type: float?
                  doc: Distance to search for matching spots.
                error_rounds:
                  type: int?
                  doc: Maximum hamming distance a barcode can be from its target and still be uniquely identified.
                mode:
                  type: string?
                  doc: Accuracy mode to run in.  Can be 'high', 'med', or 'low'.
                physical_coords:
                  type: boolean?
                  doc: Whether to use physical coordinates or pixel coordinates 


  decoding_pixel:
    - 'null'
    - type: record
      name: dummy
      fields:
        dummy:
          type: string?
          doc: Added to prevent cli parsing of the decoding_blob record.
    - type: record
      name: pixel
      fields:
        distance_threshold:
          type: float
          doc: Spots whose codewords are more than this metric distance from an expected code are filtered
        magnitude_threshold:
          type: float
          doc: spots with intensity less than this value are filtered.
        min_area:
          type: int?
          doc: Spots with total area less than this value are filtered. Defaults to 2.
        max_area:
          type: int?
          doc: Spots with total area greater than this value are filtered. Defaults to `np.inf`.
        pnorm:
          type: int?
          doc: Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance 2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance A finite large p may cause a ValueError if overflow can occur.
        norm_order:
          type: int?
          doc: Order of L_p norm to apply to intensities and codes when using metric_decode to pair each intensities to its closest target (default = 2)

outputs:
  decoded:
    type: Directory
    outputSource: restage/pool_dir

steps:

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
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
        valueFrom: "/opt/starfishRunner.json"
    out: [data]

  stage_runner:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [fov_count, selected_fovs, level_method, use_ref_img, is_volume, anchor_view, rescale, not_filtered_results, n_processes, scatter_into_n, decoding_min_sigma, decoding_max_sigma, decoding_num_sigma, decoding_threshold, decoding_overlap, decoding_decode_method, decoding_decoder_trace_building_strategy, decoding_decoder_max_distance, decoding_decoder_min_intensity, decoding_decoder_pnorm, decoding_decoder_norm_order, decoding_decoder_anchor_round, decoding_decoder_search_radius, decoding_decoder_return_original_intensities, decoding_decoder_error_rounds, decoding_decoder_mode, decoding_decoder_physical_coords, decoding_pnorm, decoding_distance_threshold, decoding_magnitude_threshold, decoding_min_area, decoding_max_area, decoding_norm_order, decoding_composite_decode, decoding_composite_pmin, decoding_composite_pmax]
    when: $(inputs.datafile != null)

  scatter_generator:
    run:
      class: ExpressionTool
      expression: |
        ${ var fovs = inputs.selected_fovs;
           if(fovs === null){
             fovs = [];
             for (let i=0; i<inputs.fov_count; i++) {
               fovs.push(Number(i));
             }
           }
           if(inputs.scatter_into_n === null){
             return {"scatter_out": new Array(fovs)};
           } else {
             var scattered = new Array(inputs.scatter_into_n);
             var chunkSize = Math.ceil(fovs.length / inputs.scatter_into_n);
             var loc = 0;
             for (let i = 0; i<fovs.length; i += chunkSize) {
               var subs = [];
               for (let j=i; j<i + chunkSize && j<fovs.length; j +=1) {
                 subs.push(Number(fovs[j]));
               }
               scattered[loc] = subs;
               loc += 1;
             }
             return {"scatter_out": scattered};
           }; }
      inputs:
        scatter_into_n:
          type: int?
        selected_fovs:
          type: int[]?
        fov_count:
          type: int
      outputs:
        scatter_out:
          type:
            type: array
            items:
              type: array
              items: int
    in:
      scatter_into_n:
        source: [stage_runner/scatter_into_n, scatter_into_n]
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
      selected_fovs:
        source: [stage_runner/selected_fovs, selected_fovs]
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
      fov_count:
        source: [stage_runner/fov_count, fov_count]
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
    out: [scatter_out]

  tmpname:
    run: tmpdir.cwl
    scatter: sc_count
    in:
      sc_count: scatter_generator/scatter_out
    out: [tmp]

  fileDivider:
    scatter: scatter
    run:
      class: ExpressionTool
      requirements:
        - class: InlineJavascriptRequirement
        - class: LoadListingRequirement

      inputs:
        experiment:
          type: Directory
          doc: Directory containing spaceTx-formatted experiment

        scatter:
          type:
            type: array
            items: int
          doc: List describing the FOVs in this specific scatter.

      outputs:
        out: File[]

      expression: |
        ${
          var dir_lis = [];
          for(var i=0;i<inputs.experiment.listing.length; i++){
            var id = inputs.experiment.listing[i].basename;
            if(id.includes("json")){
              dir_lis.push(inputs.experiment.listing[i])
            } else {
              for(var j=0;j<inputs.scatter.length; j++) {
                if(id.includes("fov_"+String(inputs.scatter[j]).padStart(5,'0'))){
                  dir_lis.push(inputs.experiment.listing[i])
                }
              }
            }
          }
          return {"out": dir_lis};
        }
    in:
      experiment: exp_loc
      scatter: scatter_generator/scatter_out
    out:
      [out]

  execute_runner:
    scatter: [selected_fovs, tmp_prefix, exp_files]
    scatterMethod: dotproduct
    run:
      class: CommandLineTool
      baseCommand: /opt/starfishDriver.py

      requirements:
        InitialWorkDirRequirement:
          listing:
            - entryname: "$('input_dir_'+inputs.tmp_prefix)"
              writable: true
              entry: "$({class: 'Directory', listing: inputs.exp_files})"
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:latest
        ResourceRequirement:
          tmpdirMin: |
            ${
              if(inputs.dir_size === null) {
                return null;
              } else {
                return inputs.dir_size;
              }
            }
          outdirMin: |
            ${
              return 1000;
            }
          coresMin: |
            ${
              if(inputs.n_processes === null) {
                return null;
              } else {
                return inputs.n_processes;
              }
            }
          ramMin: |
            ${
              if(inputs.dir_size === null){
                return null;
              } else {
                if(inputs.decoding_blob === null){
                  return parseInt((inputs.dir_size/inputs.fov_count) * 4);
                } else if(inputs.decoding_blob.min_intensity !== null) {
                  return parseInt((inputs.dir_size/inputs.fov_count) * 10);
                } else if(inputs.decoding_blob.mode !== null) {
                  return parseInt((inputs.dir_size/inputs.fov_count) * 10);
                } else {
                  return parseInt((inputs.dir_size/inputs.fov_count) * 10);
                }
              }
            }

      inputs:
        dir_size:
          type: long?

        tmp_prefix:
          type: string
          inputBinding:
            prefix: --tmp-prefix

        exp_files:
          type: File[]
          doc: Formatted input from fileDivider step.

        exp_loc:
          type: string
          inputBinding:
            prefix: --exp-loc

        selected_fovs:
          type:
            type: array
            items: int
          inputBinding:
            prefix: --selected-fovs
          doc: If provided, processing will only be run on FOVs with these indices.

        fov_count:
          type: int

        use_ref_img:
          type: boolean?
          inputBinding:
            prefix: --use-ref-img

        anchor_view:
          type: string?
          inputBinding:
            prefix: --anchor-view

        is_volume:
          type: boolean?
          inputBinding:
            prefix: --is-volume

        rescale:
          type: boolean?
          inputBinding:
            prefix: --rescale

        level_method:
          type: string?
          inputBinding:
            prefix: --level-method

        not_filtered_results:
          type: boolean?
          inputBinding:
            prefix: --not-filtered-results

        n_processes:
          type: int?
          inputBinding:
            prefix: --n-processes

        decoding_blob:
          - 'null'
          - type: record
            name: blob
            fields:
              min_sigma:
                type: float[]?
                inputBinding:
                  prefix: --min-sigma
              max_sigma:
                type: float[]?
                inputBinding:
                  prefix: --max-sigma
              num_sigma:
                type: int?
                inputBinding:
                  prefix: --num-sigma
              threshold:
                type: float?
                inputBinding:
                  prefix: --threshold
              overlap:
                type: float?
                inputBinding:
                  prefix: --overlap
              detector_method:
                type: string?
                inputBinding:
                  prefix: --detector-method
              composite_decode:
                type: boolean?
                inputBinding:
                  prefix: --composite-decode
              composite_pmin:
                type: float?
                inputBinding:
                  prefix: --composite-pmin
              composite_pmax:
                type: float?
                inputBinding:
                  prefix: --composite-pmax
              decode_method:
                type: string
                inputBinding:
                  prefix: --decode-spots-method
              decoder:
                type:
                  - type: record
                    name: metric_distance
                    fields:
                      trace_building_strategy:
                        type: string
                        inputBinding:
                          prefix: --trace-building-strategy
                      max_distance:
                        type: float
                        inputBinding:
                          prefix: --max-distance
                      min_intensity:
                        type: float
                        inputBinding:
                          prefix: --min-intensity
                      pnorm:
                        type: int?
                        inputBinding:
                          prefix: --int
                      norm_order:
                        type: int?
                        inputBinding:
                          prefix: --norm-order
                      anchor_round:
                        type: int?
                        inputBinding:
                          prefix: --anchor-round
                      search_radius:
                        type: float?
                        inputBinding:
                          prefix: --search-radius
                      return_original_intensities:
                        type: boolean?
                        inputBinding:
                          prefix: --return-original-intensities
                  - type: record
                    name: per_round_max
                    fields:
                      trace_building_strategy:
                        type: string
                        inputBinding:
                          prefix: --trace-building-strategy
                      anchor_round:
                        type: int?
                        inputBinding:
                          prefix: --anchor-round
                      search_radius:
                        type: float?
                        inputBinding:
                          prefix: --search-radius
                  - type: record
                    name: check_all
                    fields:
                      search_radius:
                        type: float?
                        inputBinding:
                          prefix: --search-radius
                      error_rounds:
                        type: int?
                        inputBinding:
                          prefix: --error-rounds
                      mode:
                        type: string?
                        inputBinding:
                          prefix: --mode
                      physical_coords:
                        type: boolean?
                        inputBinding:
                          prefix: --physical-coords

        decoding_pixel:
           - 'null'
           - type: record
             name: pixel
             fields:
               pnorm:
                 type: int?
                 inputBinding:
                   prefix: --pnorm
               distance_threshold:
                 type: float
                 inputBinding:
                   prefix: --distance-threshold
               magnitude_threshold:
                 type: float
                 inputBinding:
                   prefix: --magnitude-threshold
               min_area:
                 type: int?
                 inputBinding:
                   prefix: --min-area
               max_area:
                 type: int?
                 inputBinding:
                   prefix: --max-area
               norm_order:
                 type: int?
                 inputBinding:
                   prefix: --norm-order

      outputs:
        decoded:
          type: Directory
          outputBinding:
            glob: $("tmp/" + inputs.tmp_prefix + "/4_Decoded_" + inputs.tmp_prefix + "/")

    in:
      tmp_prefix: tmpname/tmp
      dir_size: dir_size
      exp_files: fileDivider/out
      exp_loc:
        valueFrom: $("input_dir_" + inputs.tmp_prefix)
      selected_fovs: scatter_generator/scatter_out
      fov_count:
        source: [stage_runner/fov_count, fov_count]
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
      use_ref_img:
        source: [stage_runner/use_ref_img, use_ref_img]
        pickValue: first_non_null
      anchor_view:
        source: [stage_runner/anchor_view, anchor_view]
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
      is_volume:
        source: [stage_runner/is_volume, is_volume]
        pickValue: first_non_null
      rescale:
        source: [stage_runner/rescale, rescale]
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
      level_method:
        source: [stage_runner/level_method, level_method]
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
      not_filtered_results:
        source: [stage_runner/not_filtered_results, not_filtered_results]
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
      n_processes:
        source: [stage_runner/n_processes, n_processes]
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
      decoding_blob:
        source: [decoding_blob, stage_runner/decoding_min_sigma, stage_runner/decoding_max_sigma, stage_runner/decoding_num_sigma, stage_runner/decoding_threshold, stage_runner/decoding_overlap, stage_runner/decoding_decode_method, stage_runner/decoding_decoder_trace_building_strategy, stage_runner/decoding_decoder_max_distance, stage_runner/decoding_decoder_min_intensity, stage_runner/decoding_decoder_pnorm, stage_runner/decoding_decoder_norm_order, stage_runner/decoding_decoder_anchor_round, stage_runner/decoding_decoder_search_radius, stage_runner/decoding_decoder_return_original_intensities, stage_runner/decoding_decoder_error_rounds, stage_runner/decoding_decoder_mode, stage_runner/decoding_decoder_physical_coords, stage_runner/decoding_composite_decode, stage_runner/decoding_composite_pmin, stage_runner/decoding_composite_pmax]
        valueFrom: |
          ${
            if(!self[6]){
              return self[0];
            } else {
              var decode = {
                min_sigma: self[1],
                max_sigma: self[2],
                num_sigma: self[3],
                threshold: self[4],
                overlap: self[5],
                decode_method: self[6],
                composite_decode: self[18],
                composite_pmin: self[19],
                composite_pmax: self[20]
              };
              if(self[9]){
                /* metric distance decoder */
                decode["decoder"] = {
                  trace_building_strategy: self[7],
                  max_distance: self[8],
                  min_intensity: self[9],
                  pnorm: self[10],
                  norm_order: self[11],
                  anchor_round: self[12],
                  search_radius: self[13],
                  return_original_intensities: self[14]
                };
              } else if(self[16]){
                /* check all decoder */
                decode["decoder"] = {
                  search_radius: self[13],
                  error_rounds: self[15],
                  mode: self[16],
                  physical_coords: self[17]
                };
              } else {
                /* per round max decoder */
                decode["decoder"] = {
                  trace_building_strategy: self[7],
                  anchor_round: self[12],
                  search_radius: self[13]
                };
              };
              return decode;
            };
          }
      decoding_pixel:
        source: [decoding_pixel, stage_runner/decoding_pnorm, stage_runner/decoding_distance_threshold, stage_runner/decoding_magnitude_threshold, stage_runner/decoding_min_area, stage_runner/decoding_max_area, stage_runner/decoding_norm_order]
        valueFrom: |
          ${
            if(!self[2]){
              return self[0]
            } else {
              return {
                pnorm: self[1],
                distance_threshold: self[2],
                magnitude_threshold: self[3],
                min_area: self[4],
                max_area: self[5],
                norm_order: self[6]
              };
            };
          }
    out: [decoded]
  restage:
    run:
      class: ExpressionTool
      requirements:
        InlineJavascriptRequirement: {}
        LoadListingRequirement:
          loadListing: deep_listing
      expression: |
        ${
          var listing = [];
          var csv = [];
          var cdf = [];
          var spots = [];
          for(var i=0;i<inputs.file_array.length;i++){
            for(var j=0;j<inputs.file_array[i].listing.length;j++){
              var item = inputs.file_array[i].listing[j];
              if(item.class == "Directory") {
                if(item.basename === "csv") {
                  for(var k=0;k<item.listing.length;k++){
                    csv.push(item.listing[k]);
                  }
                } else if(item.basename === "cdf") {
                  for(var k=0;k<item.listing.length;k++){
                    cdf.push(item.listing[k]);
                  }
                } else {
                  for(var k=0;k<item.listing.length; k++){
                    spots.push(item.listing[k]);
                  }
                }
              } else {
                listing.push(item);
              }
            }
          }
          listing.push({"class":"Directory","basename":"csv","listing":csv});
          listing.push({"class":"Directory","basename":"cdf","listing":cdf});
          if(spots.length > 0){
            listing.push({"class":"Directory","basename":"spots","listing":spots});
          }
          return {"pool_dir": {
            "class": "Directory",
            "basename": "4_Decoded",
            "listing": listing,
          }};
        }
      inputs:
        dir_size:
          type: long

        file_array:
          type:
            type: array
            items: Directory

      outputs:
        pool_dir:
          type: Directory

    in:
      file_array: execute_runner/decoded
      dir_size: dir_size
    out: [pool_dir]
