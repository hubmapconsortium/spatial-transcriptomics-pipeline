#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:

# pseudochannel sorting vars, if present then it will be assumed that sorting must be performed.

  channel_yml:
    type: File?
    doc: PyYML-formatted list containing a dictionary outlining how the truechannels in imaging relate to the pseudochannels in the decoding codebook. The index of each dict within the list is the trueround % (count of pseudorounds). The keys of the dict are the channels within the image and the values are the pseudochannels in the converted notebook.

  cycle_yml:
    type: File?
    doc: PyYML-formatted dictionary outlining how the truerounds in imaging relate to the pseudorounds in the decoding codebook. The keys are truerounds and the values are the corresponding pseudorounds.


# format of input vars
# can be read into converter or sorter, followed by string literal input will be used for conversion

  tiffs:
    type: Directory
    doc: The directory containing all .tiff files

  codebook:
  # NOTE: if running psort, this is assumed to be json.
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

  fov_count:
    type: int
    doc: The number of FOVs that are included in this experiment

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
        aux_channel_count:
          type: float[]?
          doc: Count of channels in each aux image.
        aux_channel_slope:
          type: float[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept
        aux_channel_intercept:
          type: int[]?
          doc: Used to convert 0-indexed channel IDs to the channel index within the image.  Calculated as (image index) = int(index*slope) + intercept

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

# image processing

  skip_processing:
    type: boolean?
    doc: If true, image processing step will be skipped.
    default: false

  clip_min:
    type: float?
    doc: Pixels below this percentile are set to 0.

  opening_size:
    type: int?
    doc: Size of the morphological opening filter to be applied to the image

  register_aux_view:
    type: string?
    doc: The name of the auxillary view to be used for image registration.

# starfishRunner

  use_ref_img:
    type: boolean?
    doc: Whether to generate a reference image and use it alongside spot detection.

  decoding:
    type:
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
                name: check_all
                fields:
                  search_radius:
                    type: int?
                    doc: Distance to search for matching spots.
                  error_rounds:
                    type: int?
                    doc: Maximum hamming distance a barcode can be from its target and still be uniquely identified.
                  mode:
                    type: string
                    doc: Accuracy mode to run in.  Can be 'high', 'med', or 'low'.
                  physical_coords:
                    type: boolean?
                    doc: Whether to use physical coordinates or pixel coordinates

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

# segmentation

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

# QC
  skip_baysor:
    type: boolean?
    doc: If true, the baysor step will be skipped.
    default: false

  find_ripley:
    type: boolean?
    doc: If true, will run ripley K estimates to find spatial density measures.  Can be slow.
    default: False
  save_pdf:
    type: boolean?
    doc: If true, will save graphical output to a pdf.
    default: True

outputs:
  1_Pseudosort:
    type: Directory
    outputSource: sorter/pseudosorted_dir
  2_tx_converted:
    type: Directory
    outputSource: spaceTxConversion/spaceTx_converted
  3_Processed:
    type: Directory
    outputSource: processing/processed_exp
  4_Decoded:
    type: Directory
    outputSource: starfishRunner/decoded
  5_Segmented:
    type: Directory
    outputSource: segmentation/segmented
  6_Baysor:
    type: Directory
    outputSource: baysorStaged/baysor
  7_QC:
    type: Directory
    outputSource: qc/qc_metrics

steps:

  sorter:
    run: steps/sorter.cwl
    in:
      channel_yml: channel_yml
      cycle_yml: cycle_yml
      input_dir: tiffs
      codebook: codebook
      round_count: round_count
      fov_count: fov_count
      round_offset: round_offset
      fov_offset: fov_offset
      channel_offset: channel_offset
      file_format: file_format
      file_vars: file_vars
      cache_read_order: cache_read_order
      aux_tilesets: aux_tilesets
    when: $(inputs.channel_yml != null)
    out: [pseudosorted_dir]

  stagedSorted:
    run: steps/psortedDefaultParams.cwl
    in:
      channel_yml: channel_yml
      exp_dir: sorter/pseudosorted_dir
      aux_names:
        source: aux_tilesets
        valueFrom: |
          ${
            return self.aux_names;
          }
      cache_read_order: cache_read_order
      aux_cache_read_order:
        source: aux_tilesets
        valueFrom: |
          ${
            return self.aux_cache_read_order
          }
    when: $(inputs.channel_yml != null)
    out: [codebook, round_offset, fov_offset, channel_offset, zplane_offset, file_format, file_vars, cache_read_order, aux_names, aux_file_formats, aux_file_vars, aux_cache_read_order, aux_channel_slope, aux_channel_intercept]

  spaceTxConversion:
    run: steps/spaceTxConversion.cwl
    in:
      tiffs:
        source: [sorter/pseudosorted_dir, tiffs]
        pickValue: first_non_null
      codebook:
        source: [stagedSorted/codebook, codebook]
        linkMerge: merge_flattened
        valueFrom: |
          ${
            if(!self[0]){
              return self[1];
            }
            return {json: self[0]};
          }
      round_count: round_count
      zplane_count: zplane_count
      channel_count: channel_count
      fov_count: fov_count
      round_offset:
        source: [stagedSorted/round_offset, round_offset]
        pickValue: first_non_null
      fov_offset:
        source: [stagedSorted/fov_offset, fov_offset]
        pickValue: first_non_null
      channel_offset:
        source: [stagedSorted/channel_offset, channel_offset]
        pickValue: first_non_null
      file_format:
        source: [stagedSorted/file_format, file_format]
        pickValue: first_non_null
      file_vars:
        source: [stagedSorted/file_vars, file_vars]
        pickValue: first_non_null
      cache_read_order:
        source: [stagedSorted/cache_read_order, cache_read_order]
        pickValue: first_non_null
      aux_tilesets:
        source: [aux_tilesets, stagedSorted/aux_names, stagedSorted/aux_file_formats, stagedSorted/aux_file_vars, stagedSorted/aux_cache_read_order, stagedSorted/aux_channel_slope, stagedSorted/aux_channel_intercept]
        valueFrom: |
          ${
            if(!self[1]){
              return {
                  aux_names: self[0].aux_names,
                  aux_file_formats: self[0].aux_file_formats,
                  aux_file_vars: self[0].aux_file_vars,
                  aux_cache_read_order: self[0].aux_cache_read_order,
                  aux_channel_count: self[0].aux_channel_count,
                  aux_channel_slope: self[0].aux_channel_slope,
                  aux_channel_intercept: self[0].aux_channel_intercept
              };
            } else {
              return {
                  aux_names: self[1],
                  aux_file_formats: self[2],
                  aux_file_vars: self[3],
                  aux_cache_read_order: self[4],
                  aux_channel_count: self[0].aux_channel_count,
                  aux_channel_slope: self[5],
                  aux_channel_intercept: self[6]
              };
            };
          }
    out: [spaceTx_converted]

  processing:
    run: steps/processing.cwl
    in:
      skip_processing: skip_processing
      input_dir: spaceTxConversion/spaceTx_converted
      clip_min: clip_min
      opening_size: opening_size
      register_aux_view: register_aux_view
    when: $(inputs.skip_processing == false)
    out:
      [processed_exp]

  starfishRunner:
    run: steps/starfishRunner.cwl
    in:
      exp_loc:
        source: [processing/processed_exp, spaceTxConversion/spaceTx_converted]
        pickValue: first_non_null
      use_ref_img: use_ref_img
      decoding: decoding
    out:
      [decoded]

  segmentation:
    run: steps/segmentation.cwl
    in:
      skip_baysor: skip_baysor
      decoded_loc: starfishRunner/decoded
      exp_loc: spaceTxConversion/spaceTx_converted
      aux_name: aux_name
      fov_count: fov_count
      binary_mask: binary_mask
    out:
      [segmented]

  baysorStaged:
    run: steps/baysorStaged.cwl
    in:
      segmented: segmentation/segmented
    when: $(inputs.skip_baysor == false)
    out:
      [baysor]

  qc:
    run: steps/qc.cwl
    in:
      codebook:
        source: [sorter/pseudosorted_dir, spaceTxConversion/spaceTx_converted]
        pickValue: first_non_null
        valueFrom: |
          ${
            return {exp: self};
          }
      segmentation_loc:
        source: [baysorStaged/baysor, segmentation/segmented]
        pickValue: first_non_null
      imagesize:
        source: fov_positioning
        valueFrom: |
          ${
            return {
              "x-size": self['x-shape'],
              "y-size": self['y-shape'],
              "z-size": self['z-shape']
            };
          }
      find_ripley: find_ripley
      save_pdf: save_pdf
      data:
        source: [starfishRunner/decoded, decoding]
        linkMerge: merge_flattened
        valueFrom: |
          ${
            return {
              exp: self[0],
              has_spots: typeof self[1].decode_method != 'undefined'
            };
          }
    out:
      [qc_metrics]
