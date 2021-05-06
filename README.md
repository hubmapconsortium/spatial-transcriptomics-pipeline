[![Build Status](https://travis-ci.com/hubmapconsortium/spatial-transcriptomics-pipeline.svg?branch=master)](https://travis-ci.com/hubmapconsortium/spatial-transcriptomics-pipeline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# HuBMAP Spatial Transcriptomics Pipeline
A [CWL](https://www.commonwl.org/) pipeline for processing spatial transcriptomics data.

## Steps

Folder         | Input to              | Output from
--------------------------------------------------------------
0_Raw          | aligner.cwl           | 
1_Projected    |                       | aligner.cwl
2_Registered   | spaceTxConversion.cwl | aligner.cwl
3_tx_converted |                       | spaceTxConversion.cwl

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
tag_releae_pipeline v0.whatever
```
See the pipeline release managment script usage notes for additional options, such as GPG signing.
