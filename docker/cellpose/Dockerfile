#FROM ubuntu:18.04
FROM continuumio/miniconda3:23.3.1-0

RUN conda create -y -n cellpose -c andfoy python=3.8 pyqt
RUN conda init bash; . /root/.bashrc; conda activate cellpose

RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev wget git
#RUN apt install -y python3.8-dev python3.8-venv python3-pip

#ENV VIRTUAL_ENV=/opt/venv
#RUN python3.8 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel numpy cython
RUN conda install -c conda-forge imagecodecs

# Can't run on GPU inside a cwl, no sense installing this.
#RUN pip install torch cuda-python==11.7

RUN pip install git+https://www.github.com/nickeener/cellpose.git

CMD ["/bin/bash"]
