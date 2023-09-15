import numpy as np
from pathlib import Path


from . import vdi_default as vdi_def

parameters_config_filename = "parameters_config_microns.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)

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
        return np.array([4,4,40])
    
        
    def align_array(self):
        pass

    def align_mesh(self):
        pass

    def align_skeleton(self):
        pass 
    
    def align_neuron_obj(self):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        pass
    
    

    def unalign_neuron_obj(self):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        pass
    
    def segment_id_to_synapse_dict(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_dict(
            *args,
            **kwargs
        )
        
        
volume_data_interface = DataInterfaceMicrons()