#!/bin/bash
celiib/mesh_tools

cd /neurd_packages

pip3 install --use-deprecated=legacy-resolver -e ./datasci_tools/

pip3 install --use-deprecated=legacy-resolver -e ./machine_learning_tools/
pip3 install --use-deprecated=legacy-resolver -e ./neuron_morphology_tools/
pip3 install --use-deprecated=legacy-resolver -e ./graph_tools/
pip3 install -e ./mesh_tools/

pip3 install -e ./NEURD/

python3 -m pip install --upgrade six


# 
pip3 install load_dotenv
pip3 install pipreqs
pip3 install cloudvolume caveclient