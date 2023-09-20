#!/bin/bash

cd /neurd_packages

pip3 install --use-deprecated=legacy-resolver -e ./python_tools/

pip3 install --use-deprecated=legacy-resolver -e ./machine_learning_tools/
pip3 install --use-deprecated=legacy-resolver -e ./neuron_morphology_tools/
pip3 install --use-deprecated=legacy-resolver -e ./graph_tools/
pip3 install --use-deprecated=legacy-resolver -e ./mesh_tools/

pip3 install --use-deprecated=legacy-resolver -e ./NEURD/

python3 -m pip install --upgrade six