FROM python:3.7-stretch

COPY requirements.txt /opt

RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/cache/.pip /opt/requirements.txt
RUN echo "import matplotlib" | python

RUN git clone --branch ctcisar-hubmap https://github.com/ctcisar/starfish.git
RUN cd /starfish;make install-dev

COPY bin /opt
RUN chmod +x /opt/*.py

CMD ["/bin/bash"]
