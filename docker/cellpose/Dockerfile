FROM ubuntu:18.04

RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev wget git
RUN apt install -y python3.8-dev python3.8-venv python3-pip

ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel cython numpy
RUN pip install torch cuda-python==11.7

RUN pip install git+https://www.github.com/nickeener/cellpose.git

CMD ["/bin/bash"]
