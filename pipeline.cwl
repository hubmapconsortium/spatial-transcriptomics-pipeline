#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.1

inputs:
# step 1 - align
  raw_dir:
    type: Directory
    inputBinding:
      prefix: --raw-dir
      valueFrom: $(self.basename)
    doc: Directory with image files

  fov_count:
    type: int
    inputBinding:
      prefix: --fov-count
    doc: The number of FoVs

  round_list:
    type: string[]
    inputBinding:
      prefix: --round-list
    doc: The names of the rounds

  sigma:
    type: float
    inputBinding: 
      prefix: --sigma
    doc: Value used for Gaussian blur

  cycle_ref_ind:
    type: int
    inputBinding:
      prefix: --cycle-ref-ind
    doc: Which cycle to align to

  channel_DIC_reference:
    type: string
    inputBinding:
      prefix: --channel-dic-reference
    doc: DIC channel for reference cycle

  channel_DIC:
    type: string
    inputBinding:
      prefix: --channel-dic
    doc: DIC channel for (non-reference) decoding cycles (the channel we use for finding alignment parameters)

  cycle_other:
    type: string[]
    inputBinding:
      prefix: --cycle-other
    doc: if there are other data-containing folders which need to be aligned but are not named "CycleXX"

  channel_DIC_other:
    type: string[]
    inputBinding:
      prefix: --channel-dic-other
    doc: DIC channel for other data-containing folders

  skip_projection:
    type: boolean?
    inputBinding:
      prefix: --skip-projection
    doc: If true, will skip z-axis projection before alignment step.

  skip_align:
    type: boolean?
    inputBinding:
      prefix: --skip-align
    doc: If true, will skip alignment of images across rounds prior to spacetx conversion

#step 2 - spaceTxConversion

  tiffs:
    type: Directory
    inputBinding:
      prefix: --input-dir
    doc: The directory containing all .tiff files

  codebook:
    type:
      - type: record
        name: csv
        fields:
          csv:
            type: File
            inputBinding:
              prefix: --codebook-csv
            doc: The codebook for this experiment in .csv format, where the rows are barcodes and the columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).
      - type: record
        name: json
        fields:
          json:
            type: File
            inputBinding:
              prefix: --codebook-json
            doc: The codebook for this experiment, already formatted in the spaceTx defined .json format.

  round_count:
    type: int
    inputBinding:
      prefix: --round-count
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int
    inputBinding:
      prefix: --zplane-count
    doc: The number of z-planes in each image

  channel_count:
    type: int
    inputBinding:
      prefix: --channel-count
    doc: The number of total channels per imaging round

  round_offset:
    type: int?
    inputBinding:
      prefix: --round-offset
    doc: The index of the first round (for file names).

  fov_offset:
    type: int?
    inputBinding:
      prefix: --fov-offset
    doc: The index of the first FOV (for file names).

  channel_offset:
    type: int?
    inputBinding: 
      prefix: --channel-offset
    doc: The index of the first channel (for file names).

  file_format:
    type: string
    inputBinding:
      prefix: --file-format
    doc: String with layout for .tiff files

  file_vars:
    type: string[]
    inputBinding:
      prefix: --file-vars
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]
    inputBinding:
      prefix: --cache-read-order
    doc: Order of non x,y dimensions within each image.

  aux_tilesets:
    type:
      type: record
      name: aux_tilesets
      fields:
        aux_names:
          type: string[]?
          inputBinding:
            prefix: --aux-names
          doc: Names of the Auxillary tiles.
        aux_file_formats:
          type: string[]?
          inputBinding:
            prefix: --aux-file-formats
          doc: String layout for .tiff files of aux views.
        aux_file_vars:
          type: string[]?
          inputBinding:
            prefix: --aux-file-vars
          doc: Variables to be substituted into aux_file_formats. One entry per aux_name, with semicolon-delimited vars.
        aux_cache_read_order:
          type: string[]?
          inputBinding:
            prefix: --aux-cache-read-order
          doc: Order of non x,y dimensions within each image. One entry per aux_name, with semicolon-delimited vars.
        aux_fixed_channel:
          type: int[]?
          inputBinding:
            prefix: --aux-fixed-channel
          doc: Which channel to refer to in aux images.

  fov_positioning:
    - 'null'
    - type: record
      fields:
        - name: x-locs
          type: string
          inputBinding:
            prefix: --x-pos-locs
          doc: list of x-axis start locations per fov index
        - name: x-shape
          type: int
          inputBinding:
            prefix: --x-pos-shape
          doc: shape of each fov item in the x-axis
        - name: x-voxel
          type: float
          inputBinding:
            prefix: --x-pos-voxel
          doc: size of voxels in the x-axis
        - name: y-locs 
          type: string
          inputBinding:
            prefix: --y-pos-locs
          doc: list of y-axis start locations per fov index
        - name: y-shape
          type: int
          inputBinding:
            prefix: --y-pos-shape
          doc: shape of each fov item in the y-axis
        - name: y-voxel
          type: float
          inputBinding:
            prefix: --y-pos-voxel
          doc: size of voxels in the y-axis
        - name: z-locs
          type: string
          inputBinding:
            prefix: --z-pos-locs
          doc: list of z-axis start locations per fov index
        - name: z-shape
          type: int
          inputBinding:
            prefix: --z-pos-shape
          doc: shape of each fov item in the z-axis
        - name: z-voxel
          type: float
          inputBinding:
            prefix: --z-pos-voxel
          doc: size of voxels in the z-axis

# step 3 - starfishRunner

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
          min_sigma_blob:
            type: float[]?
            inputBinding:
              prefix: --min-sigma
            doc: Minimum sigma tuple to be passed to blob detector
          max_sigma_blob:
            type: float[]?
            inputBinding:
              prefix: --max-sigma
            doc: Maximum sigma tuple to be passed to blob detector
          num_sigma_blob:
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
          min_area_pixel:
            type: int
            inputBinding:
              prefix: --min-area
            doc: Spots with total area less than this value are filtered
          max_area_pixel:
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
  1_Projected:
    type: Directory
    outputSource: align/projected_dir
  2_Registered:
    type: Directory
    outputSource: align/registered_dir
  3_tx_converted:
    type: Directory
    outputSource: spaceTxConversion/tx_converted_dir
  4_Decoded:
    type: Directory
    outputSource: starfishRunner/decoded_dir

steps:
  align:
    run: steps/aligner.cwl
    in:
      raw_dir: raw_dir
      fov_count: fov_count
      round_list: round_list
      sigma: sigma
      cycle_ref_ind: cycle_ref_ind
      channel_DIC_reference: channel_DIC_reference
      channel_DIC: channel_DIC
      cycle_other: cycle_other
      channel_DIC_other: channel_DIC_other
      skip_projection: skip_projection
      skip_align: skip_align
    out: [projected_dir, registered_dir]

  spaceTxConversion:
    run: steps/spaceTxConversion.cwl
    in:
      tiffs: align/registered_dir 
      codebook:
        csv: codebook_csv
        json: codebook_json
      round_count: round_count
      zplane_count: zplane_count
      channel_count: channel_count
      fov_count: fov_count
      round_offset: round_offset
      fov_offset: fov_offset
      channel_offset: channel_offset
      file_format: file_format
      file_vars: file_vars
      cache_read_order: cache_read_order
      aux_tilesets:
        aux_names: aux_names
        aux_file_formats: aux_file_formats
        aux_file_vars: aux_file_vars
        aux_cache_read_order: aux_cache_read_order
        aux_fixed_channel: aux_fixed_channel
      fov_positioning:
        x-locs: x-locs
        x-shape: x-shape
        x-voxel: x-voxel
        y-locs: y-locs
        y-shape: y-shape
        y-voxel: y-voxel
        z-locs: z-locs
        z-shape: z-shape
        z-voxel: z-voxel
    out: [tx_converted_dir]

  starfishRunner:
    run: steps/starfishRunner.cwl
    in:
      exp_loc: spaceTxConversion/tx_converted_dir
      flatten_axes: flatten_axes
      clip_img: clip_img
      use_ref_img: use_ref_img
      gaussian_lowpass: gaussian_lowpass
      zero_by_magnitude: zero_by_magnitude
      decoding:
        min_sigma: min_sigma_blob
        max_sigma: max_sigma_blob
        num_sigma: num_sigma_blob
        threshold: threshold
        overlap: overlap
        decode_method: decode_method
        filtered_results: filtered_results
        decoder:
          trace_building_strategy: trace_building_strategy
          max_distance: max_distance
          min_distance: min_distance
          min_intensity: min_intensity
          metric: metric
          norm_order: norm_order
          anchor_round: anchor_round
          search_radius: search_radius
          return_original_intensities: return_original_intensities
        metric: metric
        distance_threshold: distance_threshold
        magnitude_threshold: magnitude_threshold
        min_area: min_area_pixel
        max_area: max_area_pixel
        norm_order: norm_order
    out:
      [decoded_dir]
