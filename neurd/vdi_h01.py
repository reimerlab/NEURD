import numpy as np
from pathlib import Path

from . import vdi_default as vdi_def

parameters_config_filename = "parameters_config_h01.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")
).absolute())

default_settings = dict(
    source = "h01",
    parameters_config_filepaths = config_filepath,
    synapse_filepath = None,
)

from . import h01_volume_utils as hvu
class DataInterfaceH01(vdi_def.DataInterfaceDefault):
    
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
        return np.array([8,8,33])
    
    def segment_id_to_synapse_dict(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_dict(
            *args,
            **kwargs
        )
        
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
        
        
    def get_align_matrix(
        self,
        neuron_obj=None,
        soma_center = None,
        rotation = None,
        verbose = False,
        **kwargs
        ):
        """
        Purpose: generating the alignment matrix from the
        soma center (which shows rotation only dependent on 
        location of cell in volume)
        """
    
        if rotation is None:
            if soma_center is None:
                soma_center = neuron_obj["S0"].mesh_center
            
            if verbose:
                print(f"soma_center = {soma_center}")
                
            rotation=hvu.rotation_from_soma_center(
                soma_center = soma_center,
                verbose = verbose,
            )
            
        align_matrix = hvu.align_matrix_from_rotation(rotation = rotation)
            
        if verbose:
            print(f"align_matrix = {align_matrix}")
            
        return align_matrix
        
    
volume_data_interface = DataInterfaceH01()