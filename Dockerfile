# FROM ubuntu:18.04
# FROM nvidia/cuda:9.0-base
FROM tensorflow/tensorflow:1.15.0-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive 
# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing && apt-get install -y build-essential libcap-dev libsm6 libxext6 libxrender-dev python3-tk
# RUN apt-get update --fix-missing && \
#     apt-get install -y wget bzip2 ca-certificates curl git sudo && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* 
    

# ENV PATH /opt/conda/bin:$PATH

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     sudo /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     sudo /opt/conda/bin/conda clean -tipsy && \
#     sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc


# ENV TINI_VERSION v0.16.1
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN sudo chmod +x /usr/bin/tini

# RUN sudo apt-get update && sudo apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev build-essential libcap-dev python3 python3-pip

# COPY beepose.yml beepose.yml

COPY beepose/ beepose/
COPY plotbee/ plotbee/
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY scripts/ scripts/

# RUN conda env create -f beepose.yml
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "beepose", "/bin/bash", "-c"]
RUN pip3 install cython==0.29.14 && pip3 install -r requirements.txt && cd plotbee  && pip3 install -r requirements.txt && pip3 install . 
RUN python setup.py install 

# RUN pip install ./plotbee && pip install . && echo "conda activate beepose && cd /home/beepose/ && python setup.py install" >> ~/.bashrc
# ENV PATH /opt/conda/envs/beepose/bin:$PATH
# RUN echo $PATH

# RUN /bin/bash -c "conda init bash"
ENTRYPOINT ["./scripts/beepose"]
CMD ["-h"]
