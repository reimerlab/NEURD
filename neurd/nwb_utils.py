"""
Purpose of NWB: to store (mainly in hdf5 files) large scale neuroscience data easily

Data types typically supported:
1) electrophysiology recordings (spikes)
2) behavioral data (movement tracking)
3) optimal imaging (calcium data)
4) experimental metadata

Advantages:
1) hdf5 based - handle lard scale datasets
2) schema for metadata about experiment
3) has PyNWB that helps interact with NWB files


Typical process:
1) create a NWB file object (with metadata)
2) adds a subject object to the file.subject attribute
3) Creates data, stores inside another object, then adds to nwb file object with (add_aquisition)
"""

try:
    from pynwb import NWBFile, TimeSeries
    from pynwb.file import Subject
    from pynwb import NWBHDF5IO
except:
    NWBFile = None
    TimeSeries = None
    Subject = None
    NWBHDF5IO = None
    
from datetime import datetime
import numpy as np

def example_nwb_file():
    
    # Step 1: Create a new NWBFile
    nwbfile = NWBFile(
        session_description='Example NWB file creation',
        identifier='NWB123',  # unique ID for this file
        session_start_time=datetime.now(),  # the start time of the experiment
        experimenter='Dr. Jane Doe',  # who conducted the experiment
        lab='Neuroscience Lab',
        institution='Example University',
        experiment_description='An example experiment for demonstrating NWB',
        session_id='001',  # session ID
    )
    
    # creates a subject object to the file object
    nwbfile.subject = Subject(
        subject_id='Mouse123',
        description='A wild-type mouse used for recording',
        species='Mus musculus',
        sex='F',
        age='P90D'  # Postnatal day 90
    )
    
    # Step 3: Creates some dataa to store
    # Simulate some data
    data = np.random.randn(1000)  # 1000 data points
    timestamps = np.linspace(0, 100, num=1000)  # 100 seconds of recording

    # Step 4: Stores the data inside a TimeSeries object that will be added to the file (with some metadata around it)
    timeseries = TimeSeries(
        name='example_timeseries',
        data=data,
        unit='mV',  # data unit (e.g., millivolts for electrophysiology data)
        timestamps=timestamps,  # timestamps for the data
        description='Simulated neural data'
    )
    
    # Add TimeSeries data to the NWB file
    nwbfile.add_acquisition(timeseries)
    
    # Step 4: Write the NWBFile to disk
    with NWBHDF5IO('example_nwb_file.nwb', 'w') as io:
        io.write(nwbfile)

    print("NWB file 'example_nwb_file.nwb' has been created successfully.")

    