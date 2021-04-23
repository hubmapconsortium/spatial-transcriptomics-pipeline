FROM python:3.7-stretch

COPY requirements.txt /opt

RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/cache/.pip /opt/requirements.txt

COPY bin /opt
RUN chmod +x /opt/*.py

CMD ["/bin/bash"]
