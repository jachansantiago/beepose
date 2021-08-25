# FROM ubuntu:18.04
# FROM nvidia/cuda:9.0-base
FROM tensorflow/tensorflow:1.15.0-gpu-py3
ARG JUPYTERHUB_VERSION=1.4.2

# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y sudo
RUN adduser --disabled-password --gecos '' beepose
RUN adduser beepose sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# RUN useradd -m beepose
ENV HOME=/home/beepose
WORKDIR $HOME
USER beepose
ENV SHELL=/bin/bash
RUN sudo apt-get update --fix-missing && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libcap-dev libsm6 libxext6 libxrender-dev python3-tk git wget

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
RUN sudo pip3 install cython==0.29.14 && sudo pip3 install -r requirements.txt && cd plotbee  && sudo pip3 install -r requirements.txt && sudo pip3 install . 
RUN sudo python setup.py install

RUN sudo pip3 install --no-cache jupyterhub==$JUPYTERHUB_VERSION && sudo pip3 install notebook && sudo pip3 install jupyterlab && sudo pip3 install widgetsnbextension
RUN jupyter nbextension enable --py widgetsnbextension && sudo pip3 install jupyterlab-widgets
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /home/beepose/conda && \
    rm ~/miniconda.sh && \
    # . /home/beepose/conda/bin/conda clean -tipsy && \
    echo ". /home/beepose/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    # echo "conda activate beepose" >> ~/.bashrc
# RUN pip install ./plotbee && pip install . && echo "conda activate beepose && cd /home/beepose/ && python setup.py install" >> ~/.bashrc
# ENV PATH /opt/conda/envs/beepose/bin:$PATH
# RUN echo $PATH

RUN /home/beepose/conda/bin/conda create -n beepose python=3.6
# Make RUN commands use the new environment:
SHELL ["/home/beepose/conda/bin/conda", "run", "-n", "beepose", "/bin/bash", "-c"]
RUN git clone --recurse-submodules https://github.com/jachansantiago/plotbee.git plotbee2 \
&& cd plotbee2 \
&& pip install -r requirements.txt \
&& python setup.py install \
&& conda install ipykernel -y \
&& conda install -n beepose -c conda-forge ipywidgets \
&& python -m ipykernel install --user --name beepose --display-name "PlotbeeTF2"
# RUN /bin/bash -c "conda init bash"
# ENTRYPOINT ["beepose"]
# CMD ["-h"]
CMD ["jupyterhub-singleuser"]
