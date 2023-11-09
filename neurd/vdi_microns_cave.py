import numpy as np
from pathlib import Path


from . import vdi_default as vdi_def
from .cave_client_utils import CaveInterface

parameters_config_filename = "parameters_config_microns.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")
).absolute())

release_name_default = "minnie65_public"

default_settings = dict(
    source = "microns",
    parameters_config_filepaths = config_filepath,
    
    synapse_filepath = None,   
)


class DataInterfaceMicrons(vdi_def.DataInterfaceDefault):
    
    def __init__(
        self,
        
        # --- cave interface
        release_name = None,
        env_filepath = None,
        cave_token = None,
        client = None,
        
        **kwargs
        ):
        
        
        
        kwargs.update(default_settings)
        super().__init__(
            **kwargs
        )
        
        if release_name is None:
            release_name = release_name_default
        
        self.client = CaveInterface(
            release_name = release_name,
            env_filepath = env_filepath,
            cave_token = cave_token,
            client = client,
        )
        
    @property
    def voxel_to_nm_scaling(self):
        return self.client.voxel_to_nm_scaling
    
    def segment_id_to_synapse_df(
        self,
        segment_id,
        *args,
        **kwargs
        ):
        
        syn_df = self.client.synapse_df_from_seg_id(
            seg_id=segment_id,
            voxel_to_nm_scaling = np.array([1,1,1])
        )

        syn_df = pu.map_column_with_dict(
            syn_df,
            "prepost",
            dict_map = dict(
                pre = "presyn",
                post = "postsyn",
            )
        )
        
        return syn_df
        
    # def segment_id_to_synapse_dict(
    #     self,
    #     segment_id,
    #     *args,
    #     **kwargs):

    #     syn_df = self.client.synapse_df_from_seg_id(
    #             seg_id=segment_id,
    #             voxel_to_nm_scaling = self.voxel_to_nm_scaling,
    #     )

    #     syn_df = pu.map_column_with_dict(
    #         syn_df,
    #         "prepost",
    #         dict_map = dict(
    #             pre = "presyn",
    #             post = "postsyn",
    #         )
    #     )
    #     return syu.synapse_dict_from_synapse_df(
    #         df = syn_df,
    #         scaling = np.array([1,1,1]),
    #         coordinates_nm = False,
    #     )
    
    def get_align_matrix(self,*args,**kwargs):
        return None 
    
    def fetch_segment_id_mesh(
        self,
        segment_id=None,
        plot = False,
        ): 
        """
        """
        
        mesh = self.client.mesh_from_seg_id(
            seg_id=segment_id
        )
            
        return mesh
    
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import pandas_utils as pu

from . import synapse_utils as syu  

volume_data_interface = DataInterfaceMicrons()