# NEURD

---

NEURD: A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions

Repository is in the process of being updated

## Setup: Installation inside docker env

---

### Download Docker Image

```bash
docker pull celiib/mesh_tools:v2
```

### Run Docker Container

```bash
mkdir notebooks

docker container run -it \
    -p 8890:8888 \
    -v ./notebooks:/notebooks \
    celiib/mesh_tools:v2
```

### Installing NEURD inside Docker Container

go to http://localhost:8890/lab and open terminal

```bash
cd /
git clone https://github.com/reimerlab/NEURD.git;
pip3 install --use-deprecated=legacy-resolver -e ./NEURD/;
```
