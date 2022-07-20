#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:
  exp_loc:
    type: Directory
    doc: Location of directory containing starfish experiment.json file

  parameter_json:
    type: File?
    doc: JSON formatted input parameters.

  use_ref_img:
    type: boolean?
    doc: Whether to generate a reference image and use it alongside spot detection.
    default: False

  is_volume:
    type: boolean?
    doc: Whether to treat the zplanes as a 3D image.
    default: False

  rescale:
    type: boolean?
    doc: Whether to iteratively rescale images before running the decoder.

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
        decode_method:
          type: string
          doc: Method name for spot decoding. Refer to starfish documentation.
        filtered_results:
          type: boolean?
          doc: Automatically remove genes that do not match a target and do not meet criteria.
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
                metric:
                  type: string?
                  doc: Metric name to be used for determining distance.
                norm_order:
                  type: int?
                  doc: Refer to starfish documentation for metric_distance
                anchor_round:
                  type: int?
                  doc: Anchor round for comparison.
                search_radius:
                  type: int?
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
                  type: int?
                  doc: Distance to search for matching spots.
            - type: record
              name: check_all
              fields:
                search_radius:
                  type: int?
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
        metric:
          type: string?
          doc: The sklearn metric string to pass to NearestNeighbors. Defaults to 'euclidean'
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
        norm_order:
          type: int?
          doc: Order of L_p norm to apply to intensities and codes when using metric_decode to pair each intensities to its closest target (default = 2)

outputs:
  decoded:
    type: Directory
    outputSource: execute_runner/decoded

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
        valueFrom: "/opt/starfishRunner.json"
    out: [data]

  stage_runner:
    run: inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [use_ref_img, is_volume, rescale, decoding_min_sigma, decoding_max_sigma, decoding_num_sigma, decoding_threshold, decoding_overlap, decoding_decode_method, decoding_filtered_results, decoding_decoder_trace_building_strategy, decoding_decoder_max_distance, decoding_decoder_min_intensity, decoding_decoder_metric, decoding_decoder_norm_order, decoding_decoder_anchor_round, decoding_decoder_search_radius, decoding_decoder_return_original_intensities, decoding_decoder_error_rounds, decoding_decoder_mode, decoding_decoder_physical_coords, decoding_metric, decoding_distance_threshold, decoding_magnitude_threshold, decoding_min_area, decoding_max_area, decoding_norm_order]
    when: $(inputs.datafile != null)

  execute_runner:
    run:
      class: CommandLineTool
      #baseCommand: [sudo /opt/mountDriver.sh]
      baseCommand: /opt/starfishDriver.py

      requirements:
        DockerRequirement:
          dockerPull: ghcr.io/hubmapconsortium/spatial-transcriptomics-pipeline/starfish-custom:latest

      inputs:
        exp_loc:
          type: Directory
          inputBinding:
            prefix: --exp-loc

        use_ref_img:
          type: boolean?
          inputBinding:
            prefix: --use-ref-img

        is_volume:
          type: boolean?
          inputBinding:
            prefix: --is-volume

        rescale:
          type: boolean?
          inputBinding:
            prefix: --rescale

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
              decode_method:
                type: string
                inputBinding:
                  prefix: --decode-spots-method
              filtered_results:
                type: boolean?
                inputBinding:
                  prefix: --filtered-results
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
                      metric:
                        type: string?
                        inputBinding:
                          prefix: --metric
                      norm_order:
                        type: int?
                        inputBinding:
                          prefix: --norm-order
                      anchor_round:
                        type: int?
                        inputBinding:
                          prefix: --anchor-round
                      search_radius:
                        type: int?
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
                        type: int?
                        inputBinding:
                          prefix: --search-radius
                  - type: record
                    name: check_all
                    fields:
                      search_radius:
                        type: int?
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
               metric:
                 type: string?
                 inputBinding:
                   prefix: --metric
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
            glob: "4_Decoded/"

    in:
      exp_loc: exp_loc
      use_ref_img:
        source: [stage_runner/use_ref_img, use_ref_img]
        pickValue: first_non_null
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
      decoding_blob:
        source: [decoding_blob, stage_runner/decoding_min_sigma, stage_runner/decoding_max_sigma, stage_runner/decoding_num_sigma, stage_runner/decoding_threshold, stage_runner/decoding_overlap, stage_runner/decoding_decode_method, stage_runner/decoding_filtered_results, stage_runner/decoding_decoder_trace_building_strategy, stage_runner/decoding_decoder_max_distance, stage_runner/decoding_decoder_min_intensity, stage_runner/decoding_decoder_metric, stage_runner/decoding_decoder_norm_order, stage_runner/decoding_decoder_anchor_round, stage_runner/decoding_decoder_search_radius, stage_runner/decoding_decoder_return_original_intensities, stage_runner/decoding_decoder_error_rounds, stage_runner/decoding_decoder_mode, stage_runner/decoding_decoder_physical_coords]
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
                filtered_results: self[7]
              };
              if(self[9]){
                /* metric distance decoder */
                decode["decoder"] = {
                  trace_building_strategy: self[8],
                  max_distance: self[9],
                  min_intensity: self[10],
                  metric: self[11],
                  norm_order: self[12],
                  anchor_round: self[13],
                  search_radius: self[14],
                  return_original_intensities: self[15]
                };
              } else if(self[17]){
                /* check all decoder */
                decode["decoder"] = {
                  search_radius: self[14],
                  error_rounds: self[16],
                  mode: self[17],
                  physical_coords: self[18]
                };
              } else {
                /* per round max decoder */
                decode["decoder"] = {
                  trace_building_strategy: self[8],
                  anchor_round: self[13],
                  search_radius: self[14]
                };
              };
              return decode;
                /* pixel decoding */
                return {
                  metric: self[19],
                  distance_threshold: self[20],
                  magnitude_threshold: self[21],
                  min_area: self[22],
                  max_area: self[23],
                  norm_order: self[24]
                };
            }
          }
      decoding_pixel:
        source: [decoding_pixel, stage_runner/decoding_metric, stage_runner/decoding_distance_threshold, stage_runner/decoding_magnitude_threshold, stage_runner/decoding_min_area, stage_runner/decoding_max_area, stage_runner/decoding_norm_order]
        valueFrom: |
          ${
            if(!self[2]){
              return self[0]
            } else {
              return {
                metric: self[1],
                distance_threshold: self[2],
                magnitude_threshold: self[3],
                min_area: self[4],
                max_area: self[5],
                norm_order: self[6]
              };
            };
          }
    out: [decoded]
