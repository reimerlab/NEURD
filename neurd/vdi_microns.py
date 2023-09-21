import numpy as np
from pathlib import Path


from . import vdi_default as vdi_def

parameters_config_filename = "parameters_config_microns.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)

default_settings = dict(
    meshes_table = None,
    parameters_config_filename=parameters_config_filename
    
)

class DataInterfaceMicrons(vdi_def.DataInterfaceDefault):
    
    def __init__(
        self,
        synapse_filepath = None,
        source = "microns",
        **kwargs
        ):
        
        kwargs.update(default_settings)
        
        super().__init__(
            synapse_filepath = synapse_filepath,
            source=source,
            **kwargs
        )
        
        self.set_parameters_obj_from_filepath()
        
    @property
    def voxel_to_nm_scaling(self):
        return np.array([4,4,40])
    
    def align_array(self,array):
        return array
    
    def align_mesh(self,mesh):
        return mesh

    def align_skeleton(self,skeleton):
        return skeleton 
    
    def align_neuron_obj(self,neuron_obj):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        return neuron_obj
    
    def unalign_neuron_obj(self,neuron_obj):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        return neuron_obj
    
    def segment_id_to_synapse_dict(
        self,
        *args,
        **kwargs):
        return super().segment_id_to_synapse_dict(
            *args,
            **kwargs
        )
        
        
    # def cell_type_from_segment_id(segment_id):
    #     (db_table & dict(segment_id)).fetch1('cell_type')
        
    # def cell_type_from_segment_id(
    #     neuron_obj
    #     ):
        
    #     return neuron_obj.cell_type
    # def fetch_segment_id_mesh(
    #     self,segment_id,table=None):
    #     if table is None:
    #         table = self.meshes_table
            
    #     return (table & dict(segment_id=segment_id)).fetch1("mesh")
        
    
        
volume_data_interface = DataInterfaceMicrons()