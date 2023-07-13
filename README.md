# NEURD

NEURD: A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions

repository is in the process of being updated

# Setup:To Run Inside Docker Env

docker pull celiib/mesh_tools:v2
mkdir notebooks
docker container run -it \
 -p 8890:8888 \
 -v ./notebooks:/notebooks \
 celiib/mesh_tools:v2

-- go to http://localhost:8890/lab and in terminal run
cd /
git clone https://github.com/reimerlab/NEURD.git;
pip3 install --use-deprecated=legacy-resolver -e ./NEURD/;
