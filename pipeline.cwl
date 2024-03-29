#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.2

requirements:
   - class: SubworkflowFeatureRequirement
   - class: InlineJavascriptRequirement
   - class: StepInputExpressionRequirement
   - class: MultipleInputFeatureRequirement

inputs:

  skip_formatting:
    type: boolean?
    doc: If true, will skip first two stages and start at processing.

# pseudochannel sorting vars, if present then it will be assumed that sorting must be performed.

  channel_yml:
    type: File?
    doc: PyYML-formatted list containing a dictionary outlining how the truechannels in imaging relate to the pseudochannels in the decoding codebook. The index of each dict within the list is the trueround % (count of pseudorounds). The keys of the dict are the channels within the image and the values are the pseudochannels in the converted notebook.

  cycle_yml:
    type: File?
    doc: PyYML-formatted dictionary outlining how the truerounds in imaging relate to the pseudorounds in the decoding codebook. The keys are truerounds and the values are the corresponding pseudorounds.


  selected_fovs:
    type: int[]?
    doc: If provided, steps after conversion will only be run on FOVs with these indices.

# format of input vars
# can be read into converter or sorter, followed by string literal input will be used for conversion

  tiffs:
    type: Directory?
    doc: The directory containing all .tiff files

  codebook_csv:
    type: File?
    doc: Flattened csv input, refer to record entry.

  codebook_json:
    type: File?
    doc: Flattened json input, refer to record entry.

  locs_json:
    type: File?
    doc: Flattened json input, refer to record entry.

  data_org_file:
    type: File?
    doc: The data org file used to describe .dax formatted images.

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
        aux_single_round:
          type: string[]?
          doc: If True, the specified aux view will only have one round.
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
      name: locs
      fields:
        locs:
          type: File?
          doc: Input locations as a json file, using the same records as below.
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

  input_dir:
    type: Directory?
    doc: Root directory containing space_tx formatted experiment. Only used if skip_formatting is true.

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

  register_to_primary:
    type: boolean?
    doc: If true, registration will be performed between the first round of register_aux_view and the primary view.

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

  exp_loc:
    type: Directory?
    doc: Location of directory containing starfish experiment.json file. Only used when both skip_formatting and skip_processing are true.

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
          doc: Added to prevent cli parsing of decoding_pixel parameters.
    - type: record
      name: pixel
      fields:
        pnorm:
          type: int?
          doc: Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance 2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance A finite large p may cause a ValueError if overflow can occur.
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

  skip_seg:
    type: boolean?
    doc: If true, segmentation (and QC) will be skipped.

## cellpose-specific vars

  run_cellpose:
    type: boolean?
    doc: If true, cellpose will be run.

  use_mrna:
    type: boolean?
    doc: If true, mrna data will be used in cellpose calculations.

  aux_views:
    type: string[]?
    doc: The views to use for cellpose segmentation.

  pretrained_model_str:
    type: string?
    doc: Cellpose-provided model to use.

  pretrained_model_dir:
    type: File?
    doc: Manually trained cellpose model to use.

  diameter:
    type: float?
    doc: Expected diameter of cells. Should be 0 if a custom model is used.

  flow_threshold:
    type: float?
    doc: threshold for filtering cell segmentations (increasing this will filter out lower confidence segmentations), range is 0 to infinity

  stitch_threshold:
    type: float?
    doc: threshold for stitching together segmentations that occur at the same xy location but in adjacent z slices, range is 0 to 1. This should only be used when the image is 3D.

  cellprob_threshold:
    type: float?
    doc: determines the extent of the segmentations (0 is the default more negative values result in larger cells, more positive values result in smaller cells), range is -6 to 6.

  border_buffer:
    type: int?
    doc: If not None, removes cytoplasms whose nuclei lie within the given distance from the border.

  label_exp_size:
    type: int?
    doc: Pixel size labels are dilated by in final step. Helpful for closing small holes that are common from thresholding but can also cause cell boundaries to exceed their true boundaries if set too high. Label dilation respects label borders and does not mix labels.

  min_allowed_size:
    type: int?
    doc: minimum size for a cell (in pixels)

  max_allowed_size:
    type: int?
    doc: maximum size for a cell (in pixels)

## built-in segmentation methods

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
  run_baysor:
    type: boolean?
    doc: If true, the baysor step will be run.
    default: False

  skip_qc:
    type: boolean?
    doc: If true, QC will not be run.
    default: False

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
  5A_cellpose_input:
    type: Directory
    outputSource: cellpose/cellpose_input
  5B_cellpose_output:
    type: Directory
    outputSource: cellpose/cellpose_output
  5C_cellpose_filtered:
    type: Directory
    outputSource: cellpose/cellpose_filtered
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
        valueFrom: "/opt/pipeline.json"
    out: [data]

  stage:
    run: steps/inputParser.cwl
    in:
      datafile: parameter_json
      schema: read_schema/data
    out: [run_baysor, aux_views, skip_formatting, skip_processing, register_aux_view, fov_positioning_x_locs, fov_positioning_x_shape, fov_positioning_x_voxel, fov_positioning_y_locs, fov_positioning_y_shape, fov_positioning_y_voxel, fov_positioning_z_locs, fov_positioning_z_shape, fov_positioning_z_voxel, run_cellpose, add_blanks, skip_seg, skip_qc]
    when: $(inputs.datafile != null)

  sizer:
    run: steps/fileSizer.cwl
    in:
      exp_loc: exp_loc
      input_dir: input_dir
      tiffs: tiffs
      example_dir:
        valueFrom: |
          ${
            if (inputs.exp_loc !== null) {
              return inputs.exp_loc;
            } else if(inputs.input_dir !== null) {
              return inputs.input_dir;
            } else {
              return inputs.tiffs;
            }
          }
    out: [dir_size]

  sorter:
    run: steps/sorter.cwl
    in:
      dir_size: sizer/dir_size
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
      skip_formatting:
        source: [stage/skip_formatting, skip_formatting]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
    when: $(inputs.channel_yml != null && !inputs.skip_formatting)
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
      skip_formatting:
        source: [stage/skip_formatting, skip_formatting]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
    when: $(inputs.channel_yml != null && !inputs.skip_formatting)
    out: [codebook, round_count, fov_count, channel_count, zplane_count, round_offset, fov_offset, channel_offset, zplane_offset, file_format, file_vars, cache_read_order, aux_names, aux_file_formats, aux_file_vars, aux_cache_read_order, aux_channel_count, aux_channel_slope, aux_channel_intercept]

  spaceTxConversion:
    run: steps/spaceTxConversion.cwl
    in:
      dir_size: sizer/dir_size
      tiffs:
        source: [sorter/pseudosorted_dir, tiffs]
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
      data_org_file: data_org_file
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
        source: [fov_positioning, stage/fov_positioning_x_locs, stage/fov_positioning_x_shape, stage/fov_positioning_x_voxel, stage/fov_positioning_y_locs, stage/fov_positioning_y_shape, stage/fov_positioning_y_voxel, stage/fov_positioning_z_locs, stage/fov_positioning_z_shape, stage/fov_positioning_z_voxel, locs_json]
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
            } else if (self[10]) {
              return {"locs": self[10]};
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
      skip_formatting:
        source: [stage/skip_formatting, skip_formatting]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
    when: $(inputs.skip_formatting == false)
    out: [spaceTx_converted]

  processing:
    run: steps/processing.cwl
    in:
      dir_size: sizer/dir_size
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
      input_dir:
        source: [spaceTxConversion/spaceTx_converted, input_dir]
        valueFrom: |
          ${
            if(self[0]){
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            };
          }
      parameter_json: parameter_json
      selected_fovs: selected_fovs
      fov_count: fov_count
      clip_min: clip_min
      clip_max: clip_max
      level_method: level_method
      is_volume: is_volume
      register_aux_view: register_aux_view
      register_to_primary: register_to_primary
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
      scatter_into_n: scatter_into_n
    when: $(inputs.skip_processing == false)
    out:
      [processed_exp]

  starfishRunner:
    run: steps/starfishRunner.cwl
    in:
      dir_size: sizer/dir_size
      exp_loc:
        source: [processing/processed_exp, spaceTxConversion/spaceTx_converted, exp_loc]
        pickValue: first_non_null
      parameter_json: parameter_json
      selected_fovs: selected_fovs
      fov_count: fov_count
      level_method: level_method
      use_ref_img: use_ref_img
      anchor_view: anchor_view
      is_volume: is_volume
      rescale: rescale
      not_filtered_results: not_filtered_results
      n_processes: n_processes
      scatter_into_n: scatter_into_n
      decoding_blob: decoding_blob
      decoding_pixel: decoding_pixel
    out:
      [decoded]

  cellpose:
    run: steps/cellpose.cwl
    in:
      dir_size: sizer/dir_size
      exp_loc:
        source: [processing/processed_exp, spaceTxConversion/spaceTx_converted, exp_loc]
        pickValue: first_non_null
      decoded_loc: starfishRunner/decoded
      parameter_json: parameter_json
      selected_fovs: selected_fovs
      use_mrna: use_mrna
      zplane_count: zplane_count
      aux_views: aux_views
      pretrained_model_str: pretrained_model_str
      pretrained_model_dir: pretrained_model_dir
      diameter: diameter
      flow_threshold: flow_threshold
      stitch_threshold: stitch_threshold
      cellprob_threshold: cellprob_threshold
      border_buffer: border_buffer
      label_exp_size: label_exp_size
      min_allowed_size: min_allowed_size
      max_allowed_size: max_allowed_size
      run_cellpose:
        source: [stage/run_cellpose, run_cellpose]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            }
          }
    when: $(inputs.run_cellpose == true)
    out:
      [cellpose_input, cellpose_output, cellpose_filtered]

  segmentation:
    run: steps/segmentation.cwl
    in:
      skip_seg:
        source: [stage/skip_seg, skip_seg]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
      decoded_loc: starfishRunner/decoded
      exp_loc:
        source: [processing/processed_exp, spaceTxConversion/spaceTx_converted, exp_loc]
        pickValue: first_non_null
      parameter_json: parameter_json
      selected_fovs: selected_fovs
      aux_name:
        source: [aux_name, aux_views, stage/aux_views]
        valueFrom: |
          ${
            if(self[1]){
              return self[1][0];
            } else if(self[2]){
              return self[2][0];
            } else {
              return self[0];
            }
          }
      binary_mask:
        source: [binary_mask, mask_roi_files, mask_roi_formats, mask_labeled_files, mask_labeled_formats, cellpose/cellpose_filtered]
        valueFrom: |
          ${
            if(self[5]){
              return {
                "labeled_image": self[5],
                "file_formats_labeled": "fov_{:05d}_masks.tiff"
              }
            } else {
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
          }
    out:
      [segmented]
    when: $(inputs.skip_seg == false)

  baysorStaged:
    run: steps/baysorStaged.cwl
    in:
      run_baysor:
        source: [stage/run_baysor, run_baysor]
        valueFrom: |
          ${
            if(self[0]||self[1]){
              return true;
            } else {
              return false;
            };
          }
      segmented: segmentation/segmented
    when: $(inputs.run_baysor == true)
    out:
      [baysor]

  qc:
    run: steps/qc.cwl
    in:
      skip_seg:
        source: [stage/skip_seg, skip_seg]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            };
          }
      skip_qc:
        source: [stage/skip_qc, skip_qc]
        valueFrom: |
          ${
            if(self[0] || self[1]){
              return true;
            } else {
              return false;
            }
          }
      codebook:
        source: [sorter/pseudosorted_dir, spaceTxConversion/spaceTx_converted, processing/processed_exp, exp_loc]
        pickValue: first_non_null
        valueFrom: |
          ${
            return {exp: self};
          }
      segmentation_loc:
        source: [baysorStaged/baysor, segmentation/segmented]
        valueFrom: |
          ${
            if(self[0]) {
              return self[0];
            } else if(self[1]) {
              return self[1];
            } else {
              return null;
            }
          }
      parameter_json: parameter_json
      imagesize:
        source: [fov_positioning, locs_json]
        valueFrom: |
          ${
            if(self[0] &&
               'x_shape' in self[0] && self[0]['x_shape'] != null &&
               'y_shape' in self[0] && self[0]['y_shape'] != null &&
               'z_shape' in self[0] && self[0]['z_shape'] != null){
              return {
                "x_size": self[0]['x_shape'],
                "y_size": self[0]['y_shape'],
                "z_size": self[0]['z_shape']
              };
            } else if(self[0] && "locs" in self[0]){
              return self[0];
            } else if(self[1]){
              return {"locs": self[1]};
            } else {
              return null;
            }
          }
      selected_fovs: selected_fovs
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
    when: $(inputs.skip_qc == false && inputs.skip_seg == false)
    out:
      [qc_metrics]
