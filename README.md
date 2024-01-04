# NEURD

---

NEURD: A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions

publication: https://www.biorxiv.org/content/10.1101/2023.03.14.532674v1

## Setup: Installation inside docker env

---

### Download Docker Image

```bash
docker pull celiib/neurd:v1
```

### Run Docker Container (from CLI)

```bash
mkdir notebooks
docker container run -it \
    -p 8890:8888 \
    -v ./notebooks:/notebooks \
    celiib/neurd:v1
```

### Installing NEURD inside Docker Container

go to http://localhost:8890/lab and open terminal

```bash
cd /
git clone https://github.com/reimerlab/NEURD.git;
pip3 install ./NEURD/;
```

## Documentation

Documentation Site: https://reimerlab.github.io/NEURD/

### Tutorials

All of the tutorials made for showing the decomposition/autoproofreading pipeline (and other features like spine detection and proximity detection) are in .ipynb files inside Applications>Tutorials.
