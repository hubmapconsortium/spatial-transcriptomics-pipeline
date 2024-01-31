FROM python:3.8-buster

COPY requirements.txt /opt

RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/cache/.pip /opt/requirements.txt

RUN git clone --branch ctcisar-hubmap https://github.com/ctcisar/starfish.git
RUN cd /starfish;make install-dev

COPY bin /opt
COPY input_schemas /opt
RUN chmod +x /opt/*.*

CMD ["/bin/bash"]
