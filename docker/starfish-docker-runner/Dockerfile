FROM quay.io/ucsc_cgl/toil:5.11.0a1-f6dda143d2bbd1e0a28138263cb34e2deca04377-py3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y subversion && rm -rf /var/lib/apt/lists/*

RUN git clone --branch release https://github.com/hubmapconsortium/spatial-transcriptomics-pipeline.git
