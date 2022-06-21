[![Build Status](https://travis-ci.com/hubmapconsortium/spatial-transcriptomics-pipeline.svg?branch=master)](https://travis-ci.com/hubmapconsortium/spatial-transcriptomics-pipeline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# HuBMAP Spatial Transcriptomics Pipeline
A [CWL](https://www.commonwl.org/) pipeline for processing spatial transcriptomics data.

## Steps

Folder         | Generated By            | Used By
---------------|-------------------------|----------------------
0_Raw          |                         | `sorter.cwl` or `spaceTxConversion.cwl`
1_Pseudosort   | `sorter.cwl`            | `spaceTxConversion.cwl`
2_tx_converted | `spaceTxConversion.cwl` | `processing.cwl`
3_Processed    | `processing.cwl`        | `starfishRunner.cwl`
4_Decoded      | `starfishRunner.cwl`    | `segmentation.cwl` and `qc.cwl`
5_Segmented    | `segmentation.cwl`      | `baysorStaged.cwl` (scatters to `baysor.cwl`)
6_Baysor       | `baysorStaged.cwl`      | `qc.cwl`
7_QC           | `qc.cwl`                |


## Inputs
The general input folders for each step are defined in the above table. The remaining input parameters are as follows:

### Psuedoround Sorting
For experiments where the true rounds and channels are converted to stacked “pseudo” rounds, the following information is needed to convert the images from trueround to pseudoround format.  *Note that it is assumed that the images are the only thing referred to by their respective true imaging rounds, and that all the other variables provided above and in the codebook refer to pseudorounds and pseudochannels.*

The following are needed as inputs to `sorter.cwl`, as well as those specified in the "Dataset Formatting" section.

- `channel_yml` *file*
  PyYML-formatted list containing dictionaries outlining how the imaging truechannels relate to the pseudochannels in the decoding codebook. The index of each dict within the list is the trueround % (count of pseudorounds). The keys of the dict are the channels within the image and the values are the pseudochannels in the converted notebook.
- `cycle_yml` *file*
  PyYML-formatted dictionary outlining how the truerounds in imaging relate to the pseudorounds in the decoding codebook. The keys are truerounds and the values are the corresponding pseudorounds.

### Dataset Information

If these fields were not specified for `sorter.cwl`, they must be satisfied for `spaceTxConversion.cwl`.
- `fov_count` *int*
  The number of FOVs in the experiment
- `codebook` *File*
  CSV containing codebook information for the experiment. Rows are barcodes and columns are imaging rounds. Column IDs are expected to be sequential, and round identifiers are expected to be integers (not roman numerals).
- `round_count` *int*
  The number of imaging rounds in the experiment.
- `zplane_count` *int*
  The number of z-planes in each image.
- `channel_count` *int*
  The number of total channels per imaging round.
- `cache_read_order` *string[]*
  Describes the order of the axes within each individual image file.  Accepted values are CH, X, Y, Z.

Optional variables that describe the real-world locations of each fov and size of each voxel.  These must be all unspecified or all specified.  Required for QC to run.
- `x_locs`, `y_locs`, `z_locs` *float[]*
  List of the n-axis start location per FOV index.
- `x_shape`, `y_shape`, `z_shape` *int*
  Length of the n-axis across all FOVs.
- `x_shape`, `y_shape`, `z_shape` *float*
  Size of voxels along the n-axis.

#### File Formatting
- `round_offset` *int*
  The index of the first round in the file name. Defaults to 0.
- `fov_offset` *int*
  The index of the first FOV in the file name. Defaults to 0.
- `channel_offset` *int*
  The index of the first channel in the file name. Defaults to 0.
- `file_format` *string*
  String with directory/file layout for .tiff files. Tile-specific values will be inserted into this string using the [python .format function](https://docs.python.org/3/library/string.html#formatstrings), with the order specified in “file_vars”.
- `file_vars` *string[]*
List of strings with variables to insert into “file_format”.  The following values are accepted.
  - `channel`
  - `offset_channel` (channel + channel_offset)
  - `round`
  - `offset_round` (round + round_offset)
  - `fov`
  - `offset_fov` (fov + fov_offset)
  - `zplane`
  - `offset_zplane` (zplane + zplane_offset)

#### Auxiliary View Formatting
The following variables describe the structure of auxillary views, if any are present.  There must be as many entries in these lists as there are aux views, unless otherwise noted.
- `aux_names` *string[]*
  The name for each auxiliary view.
- `aux_file_formats` *string[]*
  The same as “file_format”, for each auxiliary view.
- `aux_file_vars` *string[][]*
  The same as “file_vars”, for each auxiliary view.
- `aux_cache_read_order` *string[]*
  The same as “cache_read_order”, for each auxiliary view.
- `aux_channel_count` *int[]*
  The number of channels in each aux view.
- `aux_channel_slope` *float[]*
  Used to convert 0-indexed channel IDs to the channel within the image, in the case that aux views use higher-indexed channels in an image already used in another view.  Defaults to 1.
  Calculated as :
    `Image index = int(index * channel_slope) + channel_intercept`
- `aux_channel_intercept` *int[]*
  Used to convert 0-indexed channel IDs to the channel within the image, in the case that aux views use higher-indexed channels in an image already used in another view.  See above calculation. Defaults to 0.

### Image Processing
The following are used in `processing.cwl` for basic image formatting.  Alternatively, if `skip_processing` is set to `True`, this step will be skipped.
- `clip_min` *float*
  Pixels with brightness below this percentile are set to zero. Defaults to 95.
- `register_aux_view` *string*
  The name of the auxiliary view to be used for image registration.  If not provided, no registration will happen.
- `channels_per_reg` *int*
  The number of images associated with each channel in the registration image. Defaults to 1.
- `background_view` *string*
  The name of the auxillary view to be used for background subtraction.  Background will be estimated if not provided.

- `anchor_view` *string*
  The name of the auxillary view to be processed in parallel with primary view, such as for anchor round in ISS processing. Will not be included if not provided.
- `high_sigma` *int*
  Sigma value for high pass gaussian filter. Will not be run if not provided.
- `deconvolve_iter` *int*
  Number of iterations to perform for deconvolution. High values remove more noise while lower values remove less. The value 15 will work for most datasets unless image is very noisy. Will not be run if not provided.
- `deconvolve_sigma` *int*
  Sigma value for deconvolution. Should be approximately the expected spot size. Must be provided if `deconvolve_iter` is provided.
- `low_sigma` *int*
  Sigma value for low pass gaussian filter. Will not be run if not provided.
- `rolling_radius` *int*
  Radius for rolling ball background subtraction. Larger values lead to increased intensity evening effect. The value of 3 will work for most datasets. Will not be run if not provided.
- `match_histogram` *boolean*
  If true, histograms will be equalized. Defaults to false.
- `tophat_radius` *int*
  Radius for white top hat filter. Should be slightly larger than the expected spot radius. Will not be run if not provided.

### Image Decoding
The following are used in `starfishRunner.cwl`, which is effectively a wrapper for the starfish spot detection/decoding methods.
- `use_reference_image` *boolean*
If true, a max projection image will be generated and used as a reference for decoding.
- **Blob-based Decoding**
  Will run decoding in two steps, [blob detection](https://spacetx-starfish.readthedocs.io/en/stable/api/spots/index.html?highlight=blobdetector#starfish.spots.FindSpots.BlobDetector) followed by decoding.
  - `min_sigma` *float[]*
    Minimum sigma tuple to be passed to blob detector. Defaults to (0.5, 0.5, 0.5).
  -` max_sigma` *float[]*
    Maximum sigma tuple to be passed to blob detector. Defaults to (8, 8, 8).
  - `num_sigma` *int*
    Number of sigma values to be tested. Defaults to 10.
  - `threshold` *float*
    Threshold of likeliness for a spot to be valid. Defaults to 0.1.
  - `is_volume` *boolean*
    If true, passes 3D volumes to spot spot detector.  If false, passes 2D tiles. Defaults to False.
  - `overlap` *float*
    Amount of overlap allowed between blob. Defaults to 0.5.
  - `detector_method` *string*
    The name of the sciki-image spot detection method to use.  Valid options are `blob_dog`, `blob_doh`, and `blob_log` (default).
  - `decode_method` *string*
    Method name for spot decoding.  Valid options and corresponding required parameters are listed below.
    - `SimpleLookupDecoder`
      Does not have any futher parameters.
    - [`MetricDistance`](https://spacetx-starfish.readthedocs.io/en/stable/api/spots/index.html#starfish.spots.DecodeSpots.MetricDistance)
      - `trace_building_strategy` *string*
        Which tracing strategy to use.  Valid options are `SEQUENTIAL`, `EXACT_MATCH`, `NEAREST_NEIGHBOR`.
      - `max_distance` *float*
        Spots greater than this distance from their nearest target are not decoded.
      - `min_intensity` *float*
        Spots dimmer than this intensity are not decoded.
      - `metric` *string*
        Can be any metric that satisfies the triangle inequality that is implemented by [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance). Defaults to `euclidean`.
      - `norm_order` *int*
        the norm to use to normalize the magnitudes of spots and codes. Defaults to 2, the L2 norm.
      - `anchor_round` *int*
        Round to use as “anchor” for comparison. Optional.
      - `search_radius` *int*
        Distance to search for matching spots, only if `trace_building_strategy` is set to `NEAREST_NEIGHBOR`. Optional.
      - `return_original_intensities` *boolean*
        Return original intensities instead of normalized ones.  Defaults to False.
    - [`PerRoundMaxChannel`](https://spacetx-starfish.readthedocs.io/en/stable/api/spots/index.html#starfish.spots.DecodeSpots.PerRoundMaxChannel)
      - `trace_building_strategy` *string*
         Which tracing strategy to use.  Valid options are `SEQUENTIAL`, `EXACT_MATCH`, `NEAREST_NEIGHBOR`.
      - `anchor_round` *int*
         Round to use as “anchor” for comparison. Optional.
      - `search_radius` *int*
         Distance to search for matching spots, only if `trace_building_strategy` is set to `NEAREST_NEIGHBOR`. Optional.
    - [`CheckAll`](https://github.com/nickeener/starfish/blob/master/starfish/core/spots/DecodeSpots/check_all_decoder.py)
      - `search_radius` *int*
         Distance to search for matching spots.  Defaults to 3.
      - `filter_rounds` *int*
         The number of rounds that a barcode must be identified in to pass filters.  Defaults to (number of rounds) - 1 or (number of rounds - error rounds), if error_rounds > 0.
      - `error_rounds` *int*
         Maximum hamming distance a barcode can be from its target and still be uniquely identified.  Defaults to 0.
- **Pixel-based Decoding**
  Will run decoding in one step using the [`PixelSpotDecoder`](https://spacetx-starfish.readthedocs.io/en/stable/api/spots/index.html#starfish.spots.DetectPixels.PixelSpotDecoder)
  - `metric` *string*
    The sklearn metric to pass to NearestNeighbors.
  - `distance_threshold` *float*
    Spots whose codewords are more than this metric distance from an expected code are filtered.
  - `magnitude_threshold` *float*
    Spots with less than this intensity value are filtered.
  - `min_area` *int*
    Spots with total area less than this value are filtered.
  - `max_area` *int*
    Spots with total area greater than this value are filtered.
  - `norm_order` *int*
    Order of L_p norm to apply to intensities and codes when using metric_decode to pair each intensity to its closest target.  Defaults to 2.

### Segmentation
Used by `segmentation.cwl`. Results from this step will automatically be run through Baysor unless `skip_baysor` is set to `True`.  Depending on pre-existing segmentation data, one of three methods can be used.
- `aux_name` *string*
  The name of the aux view to look at for segmentation.
- Provided roi_set
  - `roi_set` *directory*
     Directory containing `RoiSet.zip` for each FOV.
  - `file_formats` *string*
     Layout of the name of each zip folder, per FOV.  Will be formatted as `file_formats.format([fov index])`.
- Provided labeled binary image
  - `labeled_image` *directory*
     Directory containing labeled segmentation images, such as from ilastik classification.
  - `file_formats_labeled` *string*
     Layout of the name of each labelled image.  Will be formatted with `file_formats_labeled.format([fov index])`.
- Threshold and watershed segmentation (no provided data)
  - `img_threshold` *float*
    Global threshold value for images.
  - `min_dist` *int*
    Minimum distance (in pixels) between transformed peaks in watershedding.
  - `min_allowed_size` *int*
    Minimum size for a cell (in pixels).
  - `max_allowed_size` *int*
    Maximum allowed size for a cell (in pixels).
  - `masking_radius` *int*
    Radius for white tophat noise filter.

## Development
Code in this repository is formatted with [black](https://github.com/psf/black) and
[isort](https://pypi.org/project/isort/), and this is checked via Travis CI.

A [pre-commit](https://pre-commit.com/) hook configuration is provided, which runs `black` and `isort` before committing.
Run `pre-commit install` in each clone of this repository which you will use for development (after `pip install pre-commit`
into an appropriate Python environment, if necessary).

## Building Docker images
Run `build_docker_images` in the root of the repository, assuming you have an up-to-date
installation of the Python [`multi-docker-build`](https://github.com/mruffalo/multi-docker-build)
package.

## Release process

The `master` branch is intended to be production-ready at all times, and should always reference Docker containers
with the `latest` tag.

Publication of tagged "release" versions of the pipeline is handled with the
[HuBMAP pipeline release management](https://github.com/hubmapconsortium/pipeline-release-mgmt) Python package. To
release a new pipeline version, *ensure that the `master` branch contains all commits that you want to include in the release,*
then run
```shell
tag_release_pipeline v0.whatever
```
See the pipeline release managment script usage notes for additional options, such as GPG signing.
