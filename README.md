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

#### Highlighted Tutorials:

---

1. Auto Proofreading Pipeline:

   - Multi Soma: Applications/Tutorials/Auto_Proof_Pipeline/Double_Soma/neuron_pipeline_vp5_double_soma.ipynb
   - Single Soma Excitatory: Applications/Tutorials/Auto_Proof_Pipeline/Single_Soma_Exc/neuron_pipeline_vp5_single_demo_exc.ipynb
   - Single Soma Inhibitory: Applications/Tutorials/Auto_Proof_Pipeline/Single_Soma_Inh/neuron_pipeline_vp5_single_demo_inh.ipynb

2. Neuron Object:

   - Hierarchical Organization and Access: Applications/Tutorials/Neuron_Features/Neuron_Limb_Branch_Hierarchical_Data_Structure.ipynb
   - Neuron Feature Tutorial: Applications/Tutorials/Neuron_Features/Neuron_Features_Tutorial.ipynb
     \*\*\* See Neuron_Feature_Documentation sheet below for detailed descriptions \*\*\*

3. Proximities:

   - How to calculate proximities: Applications/Tutorials/Proximities/Tutorial_Proximities_vp2.ipynb
   - SWC Output and Anlaysis with 3rd Party Software: Applications/Tutorials/SWC_Output_and_Analysis/SWC_output_and_morphopy_analysis.ipynb

4. Volume Data Interface (VDI) Override Implementations:

   - H01 (Human Dataset) VDI Override: Applications/Tutorials/VDI_override/Tutorial_Making_Vdi_Override_H01.ipynb
   - MICrONS Caveclient VDI Override: Applications/Tutorials/VDI_microns_caveclient/vdi_microns_caveclient_demo.ipynb
   - Fake Data VDI Override: Applications/Tutorials/VDI_override/Tutorial_Making_Vdi_Override_Whale.ipynb

5. Connectivity Analysis:

   - Conversion Rate of Groups of Cells: Applications/Tutorials/Auto_Proof_Pipeline/Single_Soma_Inh/neuron_pipeline_vp5_single_demo_inh.ipynb

6. Visualizations :
   - Skeleton and Compartments of Auto Proofread Neuron: Applications/Tutorials/Visualizing_Auto_Proof_Neurons/Visualizing_Neuron_Skeletons_and_Compartments.ipynb

#### Documentation sheets:

---

1. Neuron_Feature_Documentation:
   https://docs.google.com/spreadsheets/d/1DBFTMUY7RpRoDQM3TWEb4aZoJGc-7cxq0_D-zkJ4JSE/edit?usp=sharing
2. NEURD module overview:
   https://docs.google.com/spreadsheets/d/1B5FqA1jQjadnEuQPjmbhHZFthm21NW3tGNcoLVrEUW4/edit?usp=sharing
3. NEURD paper N Table (for figures in publication)
   https://docs.google.com/spreadsheets/d/1OHeZjenEdYGDCl_5wM6wTdxFV_ouT3sQoJw5lVvgatg/edit?usp=sharing
