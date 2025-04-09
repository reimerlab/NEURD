import numpy as np
from pathlib import Path
from datasci_tools import numpy_utils as nu


from . import vdi_default as vdi_def

parameters_config_filename = "parameters_config_microns.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")
).absolute())

default_settings = dict(
    source = "microns",
    
    parameters_config_filepaths = config_filepath,
    
    synapse_filepath = None,   
)


class DataInterfaceMicrons(vdi_def.DataInterfaceDefault):
    
    def __init__(
        self,
        **kwargs
        ):
        
        for k,v in default_settings.items():
            if k == "parameters_config_filepaths":
                parameters_config_filepaths = kwargs.get(k,[])
                parameters_config_filepaths = nu.to_list(parameters_config_filepaths)
                parameters_config_filepaths = nu.to_list(v) + parameters_config_filepaths
                kwargs["parameters_config_filepaths"] = parameters_config_filepaths
            else:
                if k not in kwargs:
                    kwargs[k] = v
                
        
        super().__init__(
            **kwargs
        )
        
    @property
    def voxel_to_nm_scaling(self):
        return np.array([4,4,40])
    
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
    
    
    
volume_data_interface = DataInterfaceMicrons()

