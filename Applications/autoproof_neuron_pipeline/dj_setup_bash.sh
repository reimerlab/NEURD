#!/bin/bash


#  --- datajoint modules
pip3 install git+https://github.com/spapa013/datajoint-python.git --no-cache-dir
pip3 install git+https://github.com/Cpapa97/minnie-config.git
    
cd /
if [ ! -d "/microns-materialization" ]; then
    git clone https://github.com/cajal/microns-materialization.git
    pip3 install /microns-materialization/python/microns-materialization --no-cache-dir
fi

if [ ! -d "/microns-morphology" ]; then
    git clone https://github.com/cajal/microns-morphology.git
    pip3 install /microns-morphology/python/microns-morphology --no-cache-dir
fi


# ---- neuroglancer related modules and setup
pip3 install annotationframeworkclient nglui graphviz caveclient
#pip3 install cloud-volume==5.2.0


if [ ! -d "/root/.cloudvolume/secrets" ]; then
    mkdir -p /root/.cloudvolume/secrets
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cp "$SCRIPT_DIR/cloudvolume_secrets/cave-secret.json" /root/.cloudvolume/secrets/cave-secret.json
cp "$SCRIPT_DIR/cloudvolume_secrets/chunkedgraph-secret.json" /root/.cloudvolume/secrets/chunkedgraph-secret.json


# ---- google cloud ---
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update 
apt-get install google-cloud-cli


Applications/*.off
Applications/*.mls
Applications/*.cgal
Applications/*.pbz2


