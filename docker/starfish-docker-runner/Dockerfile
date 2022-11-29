FROM quay.io/ucsc_cgl/toil:5.7.1-b5cae9634820d76cb6c13b2a6312895122017d54-py3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y subversion && rm -rf /var/lib/apt/lists/*

RUN svn checkout https://github.com/hubmapconsortium/spatial-transcriptomics-pipeline/branches/release/steps
RUN wget https://raw.githubusercontent.com/hubmapconsortium/spatial-transcriptomics-pipeline/release/pipeline.cwl
