FROM quay.io/ucsc_cgl/toil:5.9.0a1-e1da83f8b3b45bad69facfacf86b5938aa07b92f-py3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y subversion && rm -rf /var/lib/apt/lists/*

RUN svn checkout https://github.com/hubmapconsortium/spatial-transcriptomics-pipeline/branches/release/steps
RUN wget https://raw.githubusercontent.com/hubmapconsortium/spatial-transcriptomics-pipeline/release/pipeline.cwl
