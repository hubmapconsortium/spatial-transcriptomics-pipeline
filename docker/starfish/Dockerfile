FROM python:3.7-stretch

COPY requirements.txt /opt

RUN python3 -m pip install -r /opt/requirements.txt \
 && python3 -m pip install starfish \
 && rm -rf /root/cache/.pip /opt/requirements.txt

RUN echo "import matplotlib.pyplot as plt" | python

COPY bin /opt
COPY input_schemas /opt
RUN chmod +x /opt/*.py

CMD ["/bin/bash"]
