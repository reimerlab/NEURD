FROM ubuntu:18.04 AS base
LABEL mantainer="Brendan Celii <brendanacelii@gmail.com>"
# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get install -y software-properties-common \
                       build-essential \
                       git \
                       git-lfs \
                       wget \
                       vim \
                       curl \
                       zip \
                       zlib1g-dev \
                       unzip \
                       pkg-config \
                       libblas-dev \
                       liblapack-dev \
                       python3-tk \
                       python3-wheel \
                       graphviz \
                       libhdf5-dev \
                       python3.8 \
                       python3.8-dev \
                       python3.8-distutils \
                       swig \
                       fish \
                       # necessary for building ipygraph (old docker guild)
                       # libxml2 \
                       # libxml2-dev \
                       # zlib1g-dev
                       &&\



    apt-get clean &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Installing node
# ENV NODE_VERSION=18.0.0
# RUN apt install -y curl
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# ENV NVM_DIR=/root/.nvm
# RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
# RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
# RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
# ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
# RUN node --version
# RUN npm --version


# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     npm
# RUN npm install npm@latest -g && \
#     npm install n -g && \
#     n latest

# RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
#     && apt-get install -y nodejs

# RUN conda install -c conda-forge jupyterlab
# RUN conda install -c conda-forge nodejs

# Install essential Python packages
RUN pip3 install setuptools=='59.6.0' \
    && pip3 install --upgrade pip==21.3.1


RUN python3 -m pip --no-cache-dir install \
    notebook==6.5.4 \
    jupyterlab==4.0.2 \
    jupyter_contrib_nbextensions \
    simplejson \
    jedi==0.17.2 \
    ipywidgets \
    ipywebrtc \
    jupyterlab_widgets \
    ipyvolume

RUN jupyter nbextension enable --py --sys-prefix ipyvolume \
    && jupyter nbextension enable --py --sys-prefix widgetsnbextension

    #&& jupyter labextension install ipyvolume \
    #&& jupyter nbextension enable --py widgetsnbextension \
    #&& jupyter labextension install jupyter-webrtc \
    #&& jupyter labextension install @jupyter-widgets/jupyterlab-manager \ 

# -- installing datajoint and dotmotif (moved to the ) ====
# WORKDIR /
# RUN git clone https://github.com/aplbrain/grandiso-networkx.git \
#     && git clone https://github.com/celiibrendan/dotmotif.git
# ADD ./grandiso_patch_files/queues.py /grandiso-networkx/grandiso/queues.py
# RUN pip3 install /grandiso-networkx/ --no-cache-dir \ 
#     && pip3 install /dotmotif/ --no-cache-dir \
#     && pip3 install git+https://github.com/spapa013/datajoint-python.git --no-cache-dir


#-------------------------------------------------------------
FROM base AS mesh_lite

# RUN pip3 install numpy=='1.19.5' \
#     && pip3 install cython=='0.29.21'

ADD ./python_bash_files/embree.bash /src/embree.bash

RUN apt-get update && apt-get install -q -y \
    python3-rtree \
    cmake-curses-gui \
    libtbb-dev \
    libglfw3-dev \
    && chmod +x /src/embree.bash && /src/embree.bash


#-------------------------------------------------------------
FROM mesh_lite AS mesh_heavy

# --- cgal and meshlab dependencies----
RUN apt-get update && apt-get install -q -y \
    build-essential \
    g++ \
    libeigen3-dev \
    qt5-qmake \
    qtscript5-dev \
    libqt5xmlpatterns5-dev \
    libqt5opengl5-dev \
    assimp-utils \
    nano \
    xvfb \
    && rm -rf /var/lib/apt/lists/*


# CGAL Installation ########################################################

RUN add-apt-repository universe \
    && apt-get update \
    && apt-get -y install libboost-all-dev libgmp-dev libmpfr-dev libcgal-dev libboost-wave-dev libeigen3-dev

# MESHLAB Installation ########################################################

ARG QMAKE_FLAGS="-spec linux-g++ CONFIG+=release CONFIG+=qml_release CONFIG+=c++11 QMAKE_CXXFLAGS+=-fPIC QMAKE_CXXFLAGS+=-std=c++11 QMAKE_CXXFLAGS+=-fpermissive INCLUDEPATH+=/usr/include/eigen3 LIBS+=-L/meshlab/src/external/lib/linux-g++"
ARG MAKE_FLAGS="-j"

ENV PATH="/meshlab/src/distrib:${PATH}"
ENV PIP_ROOT_USER_ACTION=ignore
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

#### manually running the steps from https://hub.docker.com/r/hamzamerzic/meshlab/ ###

# add all of the lines that will compile the meshlabserver
WORKDIR /
RUN git clone https://github.com/hamzamerzic/vcglib -b devel \
    && git clone https://github.com/hamzamerzic/meshlab -b devel

#replace the certain files in vcglib and meshlab so will compile
RUN rm /vcglib/wrap/io_trimesh/import_nvm.h /vcglib/wrap/io_trimesh/import_out.h /meshlab/src/meshlab_mini.pro

ADD ./meshlab_patch_files/import_nvm.h /vcglib/wrap/io_trimesh/import_nvm.h
ADD ./meshlab_patch_files/import_out.h /vcglib/wrap/io_trimesh/import_out.h
ADD ./meshlab_patch_files/meshlab_mini.pro /meshlab/src/meshlab_mini.pro


# ----- this was not working so had to comment it out
#WORKDIR /meshlab/src/external
#RUN qmake -qt=5 external.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src/common
RUN qmake -qt=5 common.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src
RUN qmake -qt=5 meshlab_mini.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src/meshlabplugins/filter_ao/
RUN qmake -qt=5 filter_ao.pro $QMAKE_FLAGS && make $MAKE_FLAGS


# --- installing cgal python wrappers
RUN . /etc/profile
ADD ./CGAL /src/CGAL
#ADD ./CGAL/cgal_skeleton_param /src/CGAL/cgal_skeleton_param 


# ---- for some reason this isnt installling correctly but doing from container works fine
RUN \
    pip3 install -e /src/CGAL/cgal_segmentation \
    && pip3 install -e /src/CGAL/cgal_skeleton \ 
    && pip3 install -e /src/CGAL/cgal_skeleton_param 


# FROM BASE AS TORCH_CPU

# RUN \
#     python3 -m pip --no-cache-dir install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
#     && python3 -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whltorch-1.10.2+cpu.html \
#     && python3 -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cpu.html \
#     && python3 -m pip --no-cache-dir install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
#     && python3 -m pip --no-cache-dir install \
#         torch-geometric \
#         tensorboardX \
        

# ---- enabling the jupyter lab launch ----
WORKDIR /

EXPOSE 8888

ADD ./jupyter/run_jupyter_unix.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter_unix.sh"]

