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

  codebook_csv:
    type: File?
    doc: Flattened csv input, refer to record entry.

  codebook_json:
    type: File?
    doc: Flattened json input, refer to record entry.

  mask_roi_files:
    type: Directory?
    doc: Flattened directory input, refer to record entry "binary_mask"

  mask_roi_formats:
    type: string?
    doc: Flattened record input, refer to record entry "binary_mask"

  mask_labeled_files:
    type: Directory?
    doc: Flattened file input, refer to record entry "binary_mask"

  mask_labeled_formats:
    type: string?
    doc: Flattened record input, refer to record entry "binary_mask"

  codebook:
    type:
      - 'null'
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

  parameter_json:
    type: File?
    doc: json file containing parameters for the whole experiment.  If variable is present here, it will supercede any passed in the yml.

  round_count:
    type: int?
    doc: The number of imaging rounds in the experiment

  zplane_count:
    type: int?
    doc: The number of z-planes in each image

  channel_count:
    type: int?
    doc: The number of total channels per imaging round

  fov_count:
    type: int?
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
    type: string?
    doc: String with layout for .tiff files

  file_vars:
    type: string[]?
    doc: Variables to get substituted into the file_format string.

  cache_read_order:
    type: string[]?
    doc: Order of non x,y dimensions within each image.

  aux_tilesets:
    - 'null'
    - type: record
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
      name: dummy
      fields:
        dummy:
          type: string?
          doc: Added to prevent cli parsing of the fov_positioning record.
    - type: record
      name: fov_positioning
      fields:
        - name: x_locs
          type: string?
          doc: list of x-axis start locations per fov index
        - name: x_shape
          type: int?
          doc: shape of each fov item in the x-axis
        - name: x_voxel
          type: float?
          doc: size of voxels in the x-axis
        - name: y_locs
          type: string?
          doc: list of y-axis start locations per fov index
        - name: y_shape
          type: int?
          doc: shape of each fov item in the y-axis
        - name: y_voxel
          type: float?
          doc: size of voxels in the y-axis
        - name: z_locs
          type: string?
          doc: list of z-axis start locations per fov index
        - name: z_shape
          type: int?
          doc: shape of each fov item in the z-axis
        - name: z_voxel
          type: float?
          doc: size of voxels in the z-axis

  add_blanks:
    type: boolean?
    doc: If true, will add blanks with a hamming distance 1 from existing codes.

# image processing

  skip_processing:
    type: boolean?
    doc: If true, image processing step will be skipped.
    default: false

  clip_min:
    type: float?
    doc: Pixels below this percentile are set to 0.

  clip_max:
    type: float?
    doc: Pixels above this percentile are set to 1.

  level_method:
    type: string?
    doc: Levelling method for clip and scale application. Defaults to SCALE_BY_CHUNK.

  register_aux_view:
    type: string?
    doc: The name of the auxillary view to be used for image registration.

  background_view:
    type: string?
    doc: The name of the auxillary view to be used for background subtraction.  Background will be estimated if not provided.

  register_background:
    type: boolean?
    doc: If true, `background_view` will be aligned to `aux_name`.

  anchor_view:
    type: string?
    doc: The name of the auxillary view to be processed in parallel with primary view, such as for anchor round in ISS processing. Will not be included if not provided.

  high_sigma:
    type: int?
    doc: Sigma value for high pass gaussian filter. Will not be run if not provided.

  deconvolve_iter:
    type: int?
    doc: Number of iterations to perform for deconvolution. High values remove more noise while lower values remove less. The value 15 will work for most datasets unless image is very noisy. Will not be run if not provided.

  deconvolve_sigma:
    type: int?
    doc: Sigma value for deconvolution. Should be approximately the expected spot size.

  low_sigma:
    type: int?
    doc: Sigma value for low pass gaussian filter. Will not be run if not provided.

  rolling_radius:
    type: int?
    doc: Radius for rolling ball background subtraction. Larger values lead to increased intensity evening effect. The value of 3 will work for most datasets. Will not be run if not provided.

  match_histogram:
    type: boolean?
    doc: If true, histograms will be equalized.

  tophat_radius:
    type: int?
    doc: Radius for white top hat filter. Should be slightly larger than the expected spot radius. Will not be run if not provided.

# starfishRunner

  use_ref_img:
    type: boolean?
    doc: Whether to generate a reference image and use it alongside spot detection.

  is_volume:
    type: boolean?
    doc: Whether to treat the zplanes as a 3D image.
    default: False

  rescale:
    type: boolean?
    doc: Whether to rescale images before running decoding.

  not_filtered_results:
    type: boolean?
    doc: Pipeline will not remove genes that do not match a target and do not meet criteria.

  n_processes:
    type: int?
    doc: If provided, the number of processes that will be spawned for processing. Otherwise, the maximum number of available CPUs will be used.

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
          doc: Added to prevent cli parsing of decoding_pixel parameters.
    - type: record
      name: pixel
      fields:
        metric:
          type: string?
          doc: The sklearn metric string to pass to NearestNeighbors. Defaults to euclidean.
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


# segmentation

  aux_name:
    type: string?
    doc: The name of the aux view to look at in the experiment file for image segmentation.

  binary_mask:
    - 'null'
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
    - type: record
      name: density_based
      fields:
        nuclei_view:
          type: string
          doc: Name of the auxillary view with nuclei data
        cyto_seg:
          type: boolean
          doc: If true, the cytoplasm will be segmented
        correct_seg:
          type: boolean
          doc: If true, suspected nuclei/cytoplasms that overlap will be removed.
        border_buffer:
          type: int
          doc: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.
        area_thresh:
          type: float
          doc: Threshold used when determining if an object is one nucleus or two or more overlapping nuclei. Objects whose ratio of convex hull area to normal area are above this threshold are removed if the option to remove overlapping nuclei is set.
        thresh_block_size:
          type: int
          doc: Size of structuring element for local thresholding of nuclei. If nuclei interiors aren't passing threshold, increase this value, if too much non-nuclei is passing threshold, lower it.
        watershed_footprint_size:
          type: int
          doc: Size of structuring element for watershed segmentation. Larger values will segment the nuclei into larger objects and smaller values will result in smaller objects. Adjust according to nucleus size.
        label_exp_size:
          type: int
          doc:  Pixel size labels are dilated by in final step. Helpful for closing small holes that are common from thresholding but can also cause cell boundaries to exceed their true boundaries if set too high. Label dilation respects label borders and does not mix labels.

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

  read_schema:
    run:
      class: CommandLineTool
      baseCommand: cat

      requirements:
        DockerRequirement:
          dockerPull: hubmap/starfish-custom:2.12

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
        valueFrom: "/opt/pipeline.json"
    out: [data]

  stage:
    run: steps/inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [skip_baysor, skip_processing, register_aux_view, fov_positioning_x_locs, fov_positioning_x_shape, fov_positioning_x_voxel, fov_positioning_y_locs, fov_positioning_y_shape, fov_positioning_y_voxel, fov_positioning_z_locs, fov_positioning_z_shape, fov_positioning_z_voxel, add_blanks]
    when: $(inputs.datafile != null)

  sorter:
    run: steps/sorter.cwl
    in:
      channel_yml: channel_yml
      cycle_yml: cycle_yml
      parameter_json: parameter_json
      input_dir: tiffs
      codebook:
        source: [codebook, codebook_csv, codebook_json]
        linkMerge: merge_flattened
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]){
              return {csv: self[1]};
            } else {
              return {json: self[2]};
            }
          }
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
      parameter_json: parameter_json
      aux_names:
        source: aux_tilesets
        valueFrom: |
          ${
            if(self){
                return self.aux_names;
            } else {
                return null;
            }
          }
      cache_read_order: cache_read_order
      channel_count: channel_count
      aux_cache_read_order:
        source: aux_tilesets
        valueFrom: |
          ${
            if(self) {
                return self.aux_cache_read_order;
            } else {
                return null;
            }
          }
    when: $(inputs.channel_yml != null)
    out: [codebook, round_count, fov_count, channel_count, zplane_count, round_offset, fov_offset, channel_offset, zplane_offset, file_format, file_vars, cache_read_order, aux_names, aux_file_formats, aux_file_vars, aux_cache_read_order, aux_channel_count, aux_channel_slope, aux_channel_intercept]

  spaceTxConversion:
    run: steps/spaceTxConversion.cwl
    in:
      tiffs:
        source: [sorter/pseudosorted_dir, tiffs]
        pickValue: first_non_null
      codebook:
        source: [stagedSorted/codebook, codebook, codebook_csv, codebook_json]
        linkMerge: merge_flattened
        valueFrom: |
          ${
            if(self[0]){
              return {json: self[0]};
            } else if(self[1]) {
              return self[1];
            } else if(self[2]) {
              return {csv: self[2]};
            } else {
              return {json: self[3]};
            }
          }
      parameter_json:
        source: [parameter_json, sorter/pseudosorted_dir]
        valueFrom: |
          ${
            if(self[1]){
              return null;
            } else {
              return self[0];
            }
          }
      round_count:
        source: [stagedSorted/round_count, round_count]
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
      zplane_count:
        source: [stagedSorted/zplane_count, zplane_count]
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
      channel_count:
        source: [stagedSorted/channel_count, channel_count]
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
      fov_count:
        source: [stagedSorted/fov_count, fov_count]
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
      round_offset:
        source: [stagedSorted/round_offset, round_offset]
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
      fov_offset:
        source: [stagedSorted/fov_offset, fov_offset]
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
      channel_offset:
        source: [stagedSorted/channel_offset, channel_offset]
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
      file_format:
        source: [stagedSorted/file_format, file_format]
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
      file_vars:
        source: [stagedSorted/file_vars, file_vars]
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
      cache_read_order:
        source: [stagedSorted/cache_read_order, cache_read_order]
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
      aux_tilesets:
        source: [aux_tilesets, stagedSorted/aux_names, stagedSorted/aux_file_formats, stagedSorted/aux_file_vars, stagedSorted/aux_cache_read_order, stagedSorted/aux_channel_count, stagedSorted/aux_channel_slope, stagedSorted/aux_channel_intercept]
        valueFrom: |
          ${
            if(!self[1] && self[0]){
              return {
                  aux_names: self[0].aux_names,
                  aux_file_formats: self[0].aux_file_formats,
                  aux_file_vars: self[0].aux_file_vars,
                  aux_cache_read_order: self[0].aux_cache_read_order,
                  aux_channel_count: self[0].aux_channel_count,
                  aux_channel_slope: self[0].aux_channel_slope,
                  aux_channel_intercept: self[0].aux_channel_intercept
              };
            } else if(self[1]) {
              var count = self[5];
              if(self[0] && self[0].aux_channel_count){
                count = self[0].aux_channel_count;
              }
              return {
                  aux_names: self[1],
                  aux_file_formats: self[2],
                  aux_file_vars: self[3],
                  aux_cache_read_order: self[4],
                  aux_channel_count: count,
                  aux_channel_slope: self[6],
                  aux_channel_intercept: self[7]
              };
            } else {
              return null;
            }
          }
      fov_positioning:
        source: [fov_positioning, stage/fov_positioning_x_locs, stage/fov_positioning_x_shape, stage/fov_positioning_x_voxel, stage/fov_positioning_y_locs, stage/fov_positioning_y_shape, stage/fov_positioning_y_voxel, stage/fov_positioning_z_locs, stage/fov_positioning_z_shape, stage/fov_positioning_z_voxel]
        valueFrom: |
          ${
            if(self[1]) {
              return {
                x_locs: self[1],
                x_shape: self[2],
                x_voxel: self[3],
                y_locs: self[4],
                y_shape: self[5],
                y_voxel: self[6],
                z_locs: self[7],
                z_shape: self[8],
                z_voxel: self[9]
              };
            } else if (self[0]) {
              return self[0];
            } else {
              return null;
            }
          }
      add_blanks:
        source: [add_blanks, stage/add_blanks]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return false;
            }
          }
    out: [spaceTx_converted]

  processing:
    run: steps/processing.cwl
    in:
      skip_processing:
        source: [stage/skip_processing, skip_processing]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
      input_dir: spaceTxConversion/spaceTx_converted
      parameter_json: parameter_json
      clip_min: clip_min
      clip_max: clip_max
      level_method: level_method
      is_volume: is_volume
      register_aux_view: register_aux_view
      channels_per_reg:
        source: [stagedSorted/aux_names, stagedSorted/aux_channel_count, stagedSorted/channel_count, register_aux_view, stage/register_aux_view]
        valueFrom: |
          ${
            if(self[0] && self[1] && self[2]){
              var name = "";
              if(self[3]){
                name = self[3];
              } else {
                name = self[4];
              }
              var aux_ind = self[0].indexOf(name);
              var aux_count = self[1][aux_ind];
              return Math.round(self[2] / aux_count)
            } else {
              return null;
            }
          }
      background_view: background_view
      register_background: register_background
      anchor_view: anchor_view
      high_sigma: high_sigma
      deconvolve_iter: deconvolve_iter
      deconvolve_sigma: deconvolve_sigma
      low_sigma: low_sigma
      rolling_radius: rolling_radius
      match_histogram: match_histogram
      tophat_radius: tophat_radius
      n_processes: n_processes
    when: $(inputs.skip_processing == false)
    out:
      [processed_exp]

  starfishRunner:
    run: steps/starfishRunner.cwl
    in:
      exp_loc:
        source: [processing/processed_exp, spaceTxConversion/spaceTx_converted]
        pickValue: first_non_null
      parameter_json: parameter_json
      level_method: level_method
      use_ref_img: use_ref_img
      anchor_view: anchor_view
      is_volume: is_volume
      rescale: rescale
      not_filtered_results: not_filtered_results
      n_processes: n_processes
      decoding_blob: decoding_blob
      decoding_pixel: decoding_pixel
    out:
      [decoded]

  segmentation:
    run: steps/segmentation.cwl
    in:
      decoded_loc: starfishRunner/decoded
      exp_loc: spaceTxConversion/spaceTx_converted
      parameter_json: parameter_json
      aux_name: aux_name
      fov_count: fov_count
      binary_mask:
        source: [binary_mask, mask_roi_files, mask_roi_formats, mask_labeled_files, mask_labeled_formats]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1] && self[2]){
              return {
                "roi_set": self[1],
                "file_formats": self[2]
              };
            } else if(self[3] && self[4]){
              return {
                "labeled_image": self[3],
                "file_formats_labeled": self[4]
              };
            } else {
              return null;
            }
          }
    out:
      [segmented]

  baysorStaged:
    run: steps/baysorStaged.cwl
    in:
      skip_baysor:
        source: [stage/skip_baysor, skip_baysor]
        valueFrom: |
          ${
            if(self[0]||self[1]){
              return true;
            } else {
              return false;
            };
          }
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
      parameter_json: parameter_json
      imagesize:
        source: fov_positioning
        valueFrom: |
          ${
            if(self &&
               'x_shape' in self && self['x_shape'] != null &&
               'y_shape' in self && self['y_shape'] != null &&
               'z_shape' in self && self['z_shape'] != null){
              return {
                "x_size": self['x_shape'],
                "y_size": self['y_shape'],
                "z_size": self['z_shape']
              };
            } else {
              return null;
            }
          }
      has_spots:
        source: decoding_blob
        valueFrom: |
          ${
            if(self) {
              return typeof self.decode_method != 'undefined';
            } else {
              return null;
            }
          }
      spot_threshold:
        source: decoding_blob
        valueFrom: |
          ${
             if(self && 'decoder' in self && 'min_intensity' in self['decoder']){
               return self['decoder']['min_intensity'];
             } else if(self && 'magnitude_threshold' in self){
               return self['mangitude_threshold'];
             } else {
               return null;
             }
          }
      find_ripley: find_ripley
      save_pdf: save_pdf
      data:
        source: [starfishRunner/decoded]
        valueFrom: |
          ${
            return {
              "exp": self
            };
          }
    out:
      [qc_metrics]
