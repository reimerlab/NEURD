# the original docker build is the docker build from the mesh_tools repo:
# https://github.com/bacelii/mesh_tools/tree/main/docker
FROM celiib/mesh_tools:v3

RUN pip3 install --upgrade six 
RUN pip3 install \
    datasci_stdlib_tools==1.0.0 \
    machine_learning_tools==1.0.0 \
    neuron_morphology_tools==1.0.0 \
    graph_nx_tools==1.0.0 \
    mesh_processing_tools==1.0.0 \
    code_structure_tools==1.0.0 
    

RUN pip3 install \
    cloud-volume==8.27.0 \
    caveclient==5.11.0

RUN pip3 install git+https://github.com/reimerlab/tamarind.git
RUN pip3 install git+https://github.com/reimerlab/dotmotif/
