FROM nvidia/cuda:9.0-base
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1005 beepose
RUN useradd -rm -d /home/beepose -s /bin/bash -g beepose -G sudo -u 1005 beepose && echo "beepose ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER beepose
# WORKDIR /home/beepose
WORKDIR /home/beepose

ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    sudo /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    sudo /opt/conda/bin/conda clean -tipsy && \
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN sudo chown -R beepose /opt/conda
RUN sudo chmod 770 -R /opt/conda

#RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN sudo chmod +x /usr/bin/tini



 
RUN sudo apt-get update && sudo apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev build-essential libcap-dev


COPY beepose.yml /home/beepose/beepose.yml
# This line is for debuging
RUN cd /home/beepose/ && conda env create -f beepose.yml && echo "conda activate beepose && cd beepose && pip install -e beeid && python setup.py develop" >> ~/.bashrc
ENV PATH /opt/conda/envs/beepose/bin:$PATH
RUN echo $PATH
# TODO: install python setup.py in beepose
#RUN /bin/bash -c "source ~/.bashrc && conda activate beepose && cd /home/beepose/ && python setup.py install"
#RUN cd /home/beepose/ && python setup.py install
# RUN /bin/bash -c ". activate beepose &&  cd /home/beepose/" 

CMD /bin/bash
