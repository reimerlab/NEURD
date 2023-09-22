# NEURD

---

NEURD: A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions

Repository is in the process of being updated

## Setup: Installation inside docker env

---

### Download Docker Image

```bash
docker pull celiib/mesh_tools:v3
```

### Run Docker Container (from CLI)

```bash
mkdir notebooks

docker container run -it \
    -p 8890:8888 \
    -v ./notebooks:/notebooks \
    celiib/mesh_tools:v3
```

### Installing NEURD inside Docker Container

go to http://localhost:8890/lab and open terminal

```bash
cd /
git clone https://github.com/reimerlab/NEURD.git;
pip3 install --use-deprecated=legacy-resolver -e ./NEURD/;
```

## Documentation

### Tutorials

All of the tutorials made for showing the decomposition/autoproofreading pipeline (and other features like spine detection and proximity detection) are in .ipynb inside Applications>Tutorials.

A video tutorial of the decomposition/autoproofreading pipeline can be found here: https://youtu.be/ObGoIE8q70Y
