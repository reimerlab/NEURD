import numpy as np
from pathlib import Path


from neurd import vdi_default as vdi_def


default_settings = dict(
    source = "chen",
    parameters_config_filepaths = "/neurd_packages/neuron_mesh_tools/CONNECTS/Chen_Lab/parameters_config_chen.py",
    synapse_filepath = None,   
)


class DataInterfaceChen(vdi_def.DataInterfaceDefault):
    
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
        return np.array([1,1,1])
    
    def segment_id_to_synapse_df(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_df(
            *args,
            **kwargs
        )
    
    def get_align_matrix(self,*args,**kwargs):
        return None 

    
volume_data_interface = DataInterfaceChen()