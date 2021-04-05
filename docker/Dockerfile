FROM python:3.7-slim-stretch

COPY requirements.txt /opt

RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/cache/.pip /opt/requirements.txt

COPY bin /opt
CMD ["/bin/bash"]
