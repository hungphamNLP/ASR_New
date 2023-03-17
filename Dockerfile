FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/anaconda3 && \
    rm ~/anaconda.sh

ENV PATH="/opt/anaconda3/bin:${PATH}"

COPY . /app
RUN pip install -r requirement.txt
EXPOSE 8080

RUN echo "completed!"
CMD ["python","api.py"]