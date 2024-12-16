"""
DANDI: centralized place to shared datasets related to brain (usually in NWB formats)
- file format supports: NWB (Neurodata Without Borders), BIDS (Brain Imaging Data Structure, for MRI, fMRI)

Features: 
1) Version control
2) Follows FAIR principle
    FAIR: Findable, Accessible, Interoperable, Reusable
3) Handle massive datasets

Typical process: 
1) create NWB file (using PyNWB)
2) upload the data to DANDI archive using the command-line toold dandi-cli
3) share datasets or cite in publications
4) Other labs can download the datasets using the web interface
"""