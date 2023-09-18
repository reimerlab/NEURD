from abc import (ABC,abstractmethod,)
from pathlib import Path
import numpy as np

# --- for parameters
parameters_config_filename = "parameters_config_default.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)


default_locations = dict(
    parameters_config_filepath = config_filepath,
    
    # --- mesh locations ---
    meshes_folder = "./",
    meshes_folder_undecimated = "./",
    
    # --- synapse locations ---
    synapse_filepath = None,
)

class DataInterfaceDefault(ABC):
    def __init__(
        self,
        source = "default",
        **kwargs
        ):
        for k,v in default_locations.items():
            setattr(self,k,v)
            
        for k,v in kwargs.items():
            setattr(self,k,v)
        
        self.set_parameters_obj_from_filepath()
    
    # @property
    # @abstractmethod 
    # def parameters_config_filepath(self):
    #     return None
        
    def set_parameters_obj_from_filepath(self,filepath=None):
        
        if not hasattr(self,"parameters_obj"):
            setattr(self,"parameters_obj",paru.PackageParameters())
        
        if filepath is None:
            filepath = self.parameters_config_filepath
            
        if filepath is None:
            return 
        
        self.parameters_obj_curr = paru.parameters_from_filepath(
            filepath = filepath
        )
        
        self.parameters_obj.update(
            self.parameters_obj_curr
        )
            
    def set_parameters_for_directory_modules(
        self,
        directory = None,
        verbose = False,
        **kwargs):
        
        
        paru.set_parameters_for_directory_modules_from_obj(
            obj = self,
            directory = directory,
            verbose_loop = verbose,
            **kwargs
        )
        
    # -------- abstract properties
    @property
    @abstractmethod
    def voxel_to_nm_scaling(self):
        return np.array([1,1,1])
    
    # --------------------------

    @abstractmethod
    def align_array(self,array):
        return array

    @abstractmethod
    def align_mesh(self,mesh):
        return mesh

    @abstractmethod
    def align_skeleton(self,skeleton):
        return skeleton

    @abstractmethod
    def align_neuron_obj(self,neuron_obj):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        return neuron_obj

    @abstractmethod
    def unalign_neuron_obj(self,neuron_obj):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        return neuron_obj
    
    def set_synapse_filepath(self,synapse_filepath):
        self.synapse_filepath = synapse_filepath
        
        
    # ------- Functions for fetching -----
    def fetch_segment_id_mesh(
        self,
        segment_id,
        meshes_folder = None,
        plot = False,
        ext = "off"
        ): 
        """
        

        Args:
            segment_id (_type_): _description_
            .... other attributes
            plot (bool, optional): _description_. Defaults to False.
            ext (str, optional): _description_. Defaults to "off".

        Returns:
            mesh obj
        """
        
        if meshes_folder is None:
            meshes_folder = self.meshes_folder
        
        mesh_filepath = Path(meshes_folder) / Path(
            f"{segment_id}.{ext}")
        
        mesh = tu.load_mesh_no_processing(mesh_filepath)

        if plot: 
            nviz.plot_objects(mesh)
            
        return mesh
    
    def fetch_undecimated_segment_id_mesh(
        self,
        segment_id,
        meshes_folder = None,
        plot = False,
        ext = "off",
        ):
        
        if meshes_folder is None:
            meshes_folder = self.meshes_folder_undecimated
        
        return self.fetch_segment_id_mesh(
            segment_id,
            meshes_folder = meshes_folder,
            plot = plot,
            ext = ext
        )
    
    @abstractmethod
    def segment_id_to_synapse_dict(
        self,
        *args,
        **kwargs):
        #raise Exception("")
        if kwargs.get("synapse_filepath",None) is None:
            if self.synapse_filepath is None:
                raise Exception("No synapse filepath set")
            kwargs["synapse_filepath"] = self.synapse_filepath
        
        return syu.synapse_dict_from_synapse_csv(**kwargs)

    
    
    def nuclei_from_segment_id(
        self,
        segment_id,
        return_centers=True,
        return_nm=True,
        ):
        """
        Purpose: To returns the nuclues
        information corresponding to a segment
        """
        nuclues_ids = None
        nucleus_centers = None
        
        if return_centers:
            return nuclues_ids,nucleus_centers
        else:
            return nucleus_ids
           
    def nuclei_classification_info_from_nucleus_id(
        self,
        nuclei,
        *args,
        **kwargs,
        ):
        """
        Purpose: To return a dictionary of cell type
        information (same structure as from the allen institute of brain science CAVE client return)
    

        Example Returns: 
        
        {
            'external_cell_type': 'excitatory',
            'external_cell_type_n_nuc': 1,
            'external_cell_type_fine': '23P',
            'external_cell_type_fine_n_nuc': 1,
            'external_cell_type_fine_e_i': 'excitatory'
        }
        
        """
        
        return {
            'external_cell_type': None,
            'external_cell_type_n_nuc': None,
            'external_cell_type_fine': None,
            'external_cell_type_fine_n_nuc': None,
            'external_cell_type_fine_e_i': None,
        }
        
        
    # ---------- Functions that should be reviewed or overriden --
    
    # ---- the filters used for autoproofreading -- 
    def exc_filters_auto_proof(*args,**kwargs):
        return pru.v7_exc_filters(
            *args,**kwargs
    )
        
    def inh_filters_auto_proof(*args,**kwargs):
        
        return pru.v7_inh_filters(
            *args,**kwargs
        )
    
    @property
    def default_low_degree_graph_filters(self):
        return [
            gf.axon_webbing_filter,
            gf.thick_t_filter,
            gf.axon_double_back_filter,
            gf.fork_divergence_filter,
            gf.fork_min_skeletal_distance_filter,
            gf.axon_spine_at_intersection_filter,
            gf.min_synapse_dist_to_branch_point_filter,
        ]
        
    # ---------- used by autoproofreading --------------
    def multiplicity(self,neuron_obj):
        """
        For those who don't store the output of each stage in the neuron obj
        this function could be redefined to pull from a database
        """
        return neuron_obj.multiplicity
        
    def nucleus_id(self,neuron_obj):
        return neuron_obj.nucleus_id
    
    def cell_type(self,neuron_obj):
        return neuron_obj.nucleus_id
        
        
        
        
        
        
        
    @property
    def vdi(self):
        return self
    
    
"""
Functions there: 

segment_id_to_synapse_table
decomposition_with_spine_recalculation
segment_to_nuclei
vdi.fetch_segment_id_mesh
save_proofread_faces
save_proofread_faces
save_proofread_skeleton
voxel_to_nm_scaling
voxel_to_nm_scaling
save_proofread_faces
"""

from mesh_tools import trimesh_utils as tu

from . import graph_filters as gf
from . import proofreading_utils as pru
from . import parameter_utils as paru
from . import synapse_utils as syu
from . import neuron_visualizations as nviz