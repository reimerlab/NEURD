import numpy as np
from pathlib import Path

from . import vdi_default as vdi_def

parameters_config_filename = "parameters_config_h01.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)

from . import h01_volume_utils as hvu
class DataInterfaceMicrons(vdi_def.DataInterfaceDefault):
    
    def __init__(
        self,
        synapse_filepath = None,
        source = "microns",
        parameters_config_filepath = config_filepath,
        ):
        
        super().__init__(
            synapse_filepath = synapse_filepath,
            source=source,
        )
        
        self.parameters_config_filepath = parameters_config_filepath
        self.set_parameters_obj_from_filepath()
        
    @property
    def voxel_to_nm_scaling(self):
        return hvu.voxel_to_nm_scaling
        
    @property
    def default_low_degree_graph_filters(self):
        from . import graph_filters as gf
        return [
            gf.axon_webbing_filter,
            gf.thick_t_filter,
            gf.axon_double_back_filter,
            gf.fork_divergence_filter,
            gf.fork_min_skeletal_distance_filter,
        ]
        
    def align_array(self,*args,**kwargs):
        return hvu.align_array(*args,**kwargs)
    
    def align_mesh(self,*args,**kwargs):
        return hvu.align_mesh(*args,**kwargs)
    
    def align_skeleton(self,*args,**kwargs):
        return hvu.align_skeleton(*args,**kwargs)
    
    def align_neuron_obj(self,*args,**kwargs):
        return hvu.align_neuron_obj(*args,**kwargs)

    def unalign_neuron_obj(self,*args,**kwargs):
        return hvu.unalign_neuron_obj(*args,**kwargs) 
    
    def segment_id_to_synapse_dict(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_dict(
            *args,
            **kwargs
        )
        
        
    
volume_data_interface = DataInterfaceMicrons()