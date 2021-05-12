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
          dummy:
            type: string?

outputs:
  decoded:
    type: Directory
    outputBinding:
      glob: "4_Decoded/"
