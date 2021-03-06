#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: /opt/starfishDriver.py

requirements:
  DockerRequirement:
    dockerPull: docker.pkg.github.com/hubmapconsortium/spatial-transcriptomics-pipeline/starfish:latest

inputs:
  exp_loc:
    type: Directory
    inputBinding:
      prefix: --exp-loc
    doc: Location of directory containing starfish experiment.json file

  flatten_axes:
    type: string[]?
    inputBinding:
      prefix: --flatten-axes
    doc: Which axes, if any, to compress in the image preprocessing steps.

  clip_img:
    type: boolean?
    inputBinding:
      prefix: --clip-img
    doc: Whether to rescale and clip images across rounds.

  use_ref_img:
    type: boolean?
    inputBinding:
      prefix: --use-ref-img
    doc: Whether to generate a reference image and use it alongside spot detection.

  gaussian_lowpass:
    type: float?
    inputBinding:
      prefix: --gaussian-lowpass
    doc: If included, standard deviation for gaussian kernel in lowpass filter

  zero_by_magnitude:
    type: float?
    inputBinding:
      prefix: --zero-by-magnitude
    doc: If included, pixels in each round that have a L2 norm across channels below this threshold are set to 0.

  decoding:
    type:
      - type: record
        name: blob
        fields:
          min_sigma:
            type: float[]?
            inputBinding:
              prefix: --min-sigma
            doc: Minimum sigma tuple to be passed to blob detector
          max_sigma:
            type: float[]?
            inputBinding:
              prefix: --max-sigma
            doc: Maximum sigma tuple to be passed to blob detector
          num_sigma:
            type: int?
            inputBinding:
              prefix: --num-sigma
            doc: The number of sigma values to be tested, passed to blob detector
          threshold:
            type: float?
            inputBinding:
              prefix: --threshold
            doc: Threshold of blob detection
          is_volume:
            type: boolean?
            inputBinding:
              prefix: --is-volume
            doc: If True, passes 3d volumes to func, else pass 2d tiles to func.
          overlap:
            type: float?
            inputBinding:
              prefix: --overlap
            doc: Amount of overlap allowed between blobs, passed to blob detector
          decode_method:
            type: string
            inputBinding:
              prefix: --decode-spots-method
            doc: Method name for spot decoding. Refer to starfish documentation.
          filtered_results:
            type: boolean?
            inputBinding:
              prefix: --filtered-results
            doc: Automatically remove genes that do not match a target and do not meet criteria.
          decoder:
            type:
              - type: record
                name: metric_distance
                fields:
                  trace_building_strategy:
                    type: string
                    inputBinding:
                      prefix: --trace-building-strategy
                    doc: Which tracing strategy to use.  See starfish docs.
                  max_distance:
                    type: float
                    inputBinding:
                      prefix: --max-distance
                    doc: Maximum distance between spots.
                  min_intensity:
                    type: float
                    inputBinding:
                      prefix: --min-intensity
                    doc: Minimum intensity of spots.
                  metric:
                    type: string
                    inputBinding:
                      prefix: --metric
                    doc: Metric name to be used for determining distance.
                  norm_order:
                    type: int
                    inputBinding:
                      prefix: --norm-order
                    doc: Refer to starfish documentation for metric_distance
                  anchor_round:
                    type: int?
                    inputBinding:
                      prefix: --anchor-round
                    doc: Anchor round for comparison.
                  search_radius:
                    type: int
                    inputBinding:
                      prefix: --search-radius
                    doc: Distance to search for matching spots.
                  return_original_intensities:
                    type: boolean?
                    inputBinding:
                      prefix: --return-original-intensities
                    doc: Return original intensities instead of normalized ones.
              - type: record
                name: per_round_max
                fields:
                  trace_building_strategy:
                    type: string
                    inputBinding:
                      prefix: --trace-building-strategy
                    doc: Which tracing strategy to use.  See starfish docs.
                  anchor_round:
                    type: int?
                    inputBinding:
                      prefix: --anchor-round
                    doc: Round to refer to.  Required for nearest_neighbor.
                  search_radius:
                    type: int?
                    inputBinding:
                      prefix: --search-radius
                    doc: Distance to search for matching spots.

      - type: record
        name: pixel
        fields:
          metric:
            type: string
            inputBinding:
              prefix: --metric
            doc: The sklearn metric string to pass to NearestNeighbors
          distance_threshold:
            type: float
            inputBinding:
              prefix: --distance-threshold
            doc: Spots whose codewords are more than this metric distance from an expected code are filtered
          magnitude_threshold:
            type: float
            inputBinding:
              prefix: --magnitude-threshold
            doc: spots with intensity less than this value are filtered.
          min_area:
            type: int
            inputBinding:
              prefix: --min-area
            doc: Spots with total area less than this value are filtered
          max_area:
            type: int
            inputBinding:
              prefix: --max-area
            doc: Spots with total area greater than this value are filtered
          norm_order:
            type: int?
            inputBinding:
              prefix: --norm-order
            doc: Order of L_p norm to apply to intensities and codes when using metric_decode to pair each intensities to its closest target (default = 2)

outputs:
  decoded:
    type: Directory
    outputBinding:
      glob: "4_Decoded/"
