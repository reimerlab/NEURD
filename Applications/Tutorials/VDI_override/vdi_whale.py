import numpy as np
from pathlib import Path


from neurd import vdi_default as vdi_def

config_filepath = str(Path("./parameters_config_whale.py").absolute())

default_settings = dict(
    source = "whale",
    
    parameters_config_filepaths = config_filepath,
    
    synapse_filepath = None,   
)

class DataInterfaceWhale(vdi_def.DataInterfaceDefault):
    
    def __init__(
        self,
        **kwargs
        ):
        
        kwargs.update(default_settings)
        super().__init__(
            **kwargs
        )
        
    @property
    def voxel_to_nm_scaling(self):
        return np.array([5,5,90])
    
    def segment_id_to_synapse_df(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_df(
            *args,
            **kwargs
        )
    
    def get_align_matrix(self,*args,**kwargs):
        return np.array([
            [0,1,0],
            [0,0,1],
            [1,0,0]
        ])
    
    
    
volume_data_interface = DataInterfaceWhale()