FROM ubuntu:20.04 

COPY requirements_aligner.txt /opt

RUN apt-get update
RUN apt-get install -y cmake protobuf-compiler swig python3 python3-dev tcl tcl-dev tk tk-dev
RUN apt-get install -y git
RUN git clone https://github.com/SuperElastix/SimpleElastix
RUN mkdir /build
RUN cd /build
RUN cd /build;cmake -D PYTHON_EXECUTABLE=`which python3` ../SimpleElastix/SuperBuild
RUN cd /build;make -j `nproc`
RUN cd /build/SimpleITK-build/Wrapping/Python/Packaging;python3 setup.py install

RUN python3 -m pip install -r /opt/requirements_aligner.txt \
 && rm -rf /root/cache/.pip /opt/requirements_aligner.txt

COPY bin /opt
RUN chmod +x /opt/*.py

CMD ["/bin/bash"]
