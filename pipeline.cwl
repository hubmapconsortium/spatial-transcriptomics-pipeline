#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.1

inputs:
# step 1 - align
  raw_dir:
    type: Directory
    doc: Directory with image files

  fov_count:
    type: int
    doc: The number of FoVs

  round_list:
    type: string[]
    doc: The names of the rounds

  sigma:
    type: float
    doc: Value used for Gaussian blur

  cycle_ref_ind:
    type: int
    doc: Which cycle to align to

  channel_DIC_reference:
    type: string
    doc: DIC channel for reference cycle

  channel_DIC:
    type: string
    doc: DIC channel for (non-reference) decoding cycles (the channel we use for finding alignment parameters)

  cycle_other:
    type: string[]
    doc: if there are other data-containing folders which need to be aligned but are not named "CycleXX"

  channel_DIC_other:
    type: string[]
    doc: DIC channel for other data-containing folders

  skip_projection:
    type: boolean?
    doc: If true, will skip z-axis projection before alignment step.

  skip_align:
    type: boolean?
    doc: If true, will skip alignment of images across rounds prior to spacetx conversion

#step 2 - spaceTxConversion

  codebook:
    type:
      - type: record
        name: csv
        fields:
          csv:
            type: File
            doc: The codebook for this experiment in .csv format, where the rows are barcodes and the columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).
      - type: record
        name: json
        fields:
          json:
            type: File
            doc: The codebook for this experiment, already formatted in the spaceTx defined .json format.

  round_count:
    type: int
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int
    doc: The number of z-planes in each image

  channel_count:
    type: int
    doc: The number of total channels per imaging round

  round_offset:
    type: int?
    doc: The index of the first round (for file names).

  fov_offset:
    type: int?
    doc: The index of the first FOV (for file names).

  channel_offset:
    type: int?
    doc: The index of the first channel (for file names).

  file_format:
    type: string
    doc: String with layout for .tiff files

  file_vars:
    type: string[]
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]
    doc: Order of non x,y dimensions within each image.

  aux_tilesets:
    type:
      type: record
      name: aux_tilesets
      fields:
        aux_names:
          type: string[]?
          doc: Names of the Auxillary tiles.
        aux_file_formats:
          type: string[]?
          doc: String layout for .tiff files of aux views.
        aux_file_vars:
          type: string[]?
          doc: Variables to be substituted into aux_file_formats. One entry per aux_name, with semicolon-delimited vars.
        aux_cache_read_order:
          type: string[]?
          doc: Order of non x,y dimensions within each image. One entry per aux_name, with semicolon-delimited vars.
        aux_fixed_channel:
          type: int[]?
          doc: Which channel to refer to in aux images.

  fov_positioning:
    - 'null'
    - type: record
      fields:
        - name: x-locs
          type: string
          doc: list of x-axis start locations per fov index
        - name: x-shape
          type: int
          doc: shape of each fov item in the x-axis
        - name: x-voxel
          type: float
          doc: size of voxels in the x-axis
        - name: y-locs
          type: string
          doc: list of y-axis start locations per fov index
        - name: y-shape
          type: int
          doc: shape of each fov item in the y-axis
        - name: y-voxel
          type: float
          doc: size of voxels in the y-axis
        - name: z-locs
          type: string
          doc: list of z-axis start locations per fov index
        - name: z-shape
          type: int
          doc: shape of each fov item in the z-axis
        - name: z-voxel
          type: float
          doc: size of voxels in the z-axis

# step 3 - starfishRunner

  flatten_axes:
    type: string[]?
    doc: Which axes, if any, to compress in the image preprocessing steps.

  clip_img:
    type: boolean?
    doc: Whether to rescale and clip images across rounds.

  use_ref_img:
    type: boolean?
    doc: Whether to generate a reference image and use it alongside spot detection.

  gaussian_lowpass:
    type: float?
    doc: If included, standard deviation for gaussian kernel in lowpass filter

  zero_by_magnitude:
    type: float?
    doc: If included, pixels in each round that have a L2 norm across channels below this threshold are set to 0.

  decoding:
    type:
      - type: record
        name: blob
        fields:
          min_sigma_blob:
            type: float[]?
            doc: Minimum sigma tuple to be passed to blob detector
          max_sigma_blob:
            type: float[]?
            doc: Maximum sigma tuple to be passed to blob detector
          num_sigma_blob:
            type: int?
            doc: The number of sigma values to be tested, passed to blob detector
          threshold:
            type: float?
            doc: Threshold of blob detection
          is_volume:
            type: boolean?
            doc: If true, pass 3d tiles to func, else pass 2d tiles to func.
          overlap:
            type: float?
            doc: Amount of overlap allowed between blobs, passed to blob detector
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
                    type: string
                    doc: Metric name to be used for determining distance.
                  norm_order:
                    type: int
                    doc: Refer to starfish documentation for metric_distance
                  anchor_round:
                    type: int?
                    doc: Anchor round for comparison.
                  search_radius:
                    type: int
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
        name: pixel
        fields:
          metric:
            type: string
            doc: The sklearn metric string to pass to NearestNeighbors
          distance_threshold:
            type: float
            doc: Spots whose codewords are more than this metric distance from an expected code are filtered
          magnitude_threshold:
            type: float
            doc: spots with intensity less than this value are filtered.
          min_area:
            type: int
            doc: Spots with total area less than this value are filtered
          max_area:
            type: int
            doc: Spots with total area greater than this value are filtered
          norm_order:
            type: int?
            doc: Order of L_p norm to apply to intensities and codes when using metric_decode to pair each intensities to its closest target (default = 2)

# 4 - Segmentation

  aux_name:
    type: string
    doc: The name of the aux view to look at in the experiment file for image segmentation.

  binary_mask:
    type:
      - type: record
        name: roi_set
        fields:
          roi_set:
            type: Directory
            doc: Directory of RoiSet.zip for each fov, from fiji segmentation
          file_formats:
            type: string
            doc: Layout for name of each RoiSet.zip, per fov. Will be formatted with String.format([fov index]).
      - type: record
        name: labeled_image
        fields:
          labeled_image:
            type: Directory
            doc: Directory of labeled images with image segmentation data, such as from ilastik classification.
          file_formats_labeled:
            type: string
            doc: Layout for name of each labelled image. Will be formatted with String.format([fov index])
      - type: record
        name: basic_watershed
        fields:
          img_threshold:
            type: float
            doc: Global threshold value for images
          min_dist:
            type: int
            doc: minimum distance (pixels) between distance transformed peaks
          min_allowed_size:
            type: int
            doc: minimum size for a cell (in pixels)
          max_allowed_size:
            type: int
            doc: maxiumum size for a cell (in pixels)
          masking_radius:
            type: int
            doc: Radius for white tophat noise filter

outputs:
  1_Projected:
    type: Directory
    outputSource: align/projected
  2_Registered:
    type: Directory
    outputSource: align/registered
  2_Registered_log:
    type: File
    outputSource: align/tool_out
  3_tx_converted:
    type: Directory
    outputSource: spaceTxConversion/spaceTx_converted
  4_Decoded:
    type: Directory
    outputSource: starfishRunner/decoded
  5_Segmented:
    type: Directory
    outputSource: segmentation/segmented

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
    out: [projected, registered, tool_out]

  spaceTxConversion:
    run: steps/spaceTxConversion.cwl
    in:
      tiffs: align/registered
      codebook: codebook
#        csv: codebook_csv
#        json: codebook_json
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
      aux_tilesets: aux_tilesets
    out: [spaceTx_converted]

  starfishRunner:
    run: steps/starfishRunner.cwl
    in:
      exp_loc: spaceTxConversion/spaceTx_converted
      flatten_axes: flatten_axes
      clip_img: clip_img
      use_ref_img: use_ref_img
      gaussian_lowpass: gaussian_lowpass
      zero_by_magnitude: zero_by_magnitude
      decoding: decoding
    out:
      [decoded]

  segmentation:
    run: steps/segmentation.cwl
    in:
      decoded_loc: starfishRunner/decoded
      exp_loc: spaceTxConversion/spaceTx_converted
      aux_name: aux_name
      fov_count: fov_count
      binary_mask: binary_mask
    out:
      [segmented]
