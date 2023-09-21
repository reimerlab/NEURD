from abc import (ABC,abstractmethod,)
from pathlib import Path
import numpy as np
import pandas as pd

# --- for parameters
parameters_config_filename = "parameters_config_default.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)


default_settings = dict(
    parameters_config_filepath = config_filepath,
    
    # --- mesh locations ---
    meshes_directory = "./",
    meshes_undecimated_directory = "./",
    
    neuron_obj_directory = "./",
    neuron_obj_auto_proof_directory = "./",
    neuron_obj_auto_proof_suffix = "_auto_proof",
    # --- synapse locations ---
    synapse_filepath = None,
)

from functools import wraps

def neuron_obj_func(func):
    @wraps(func)
    def new_func(self,*args,**kwargs):
        if len(args) > 0:
            segment_id = args[0]
        else:
            segment_id = kwargs[list(kwargs.keys())[0]]
        if not isinstance(segment_id,neuron.Neuron):
            raise Exception("Only implemented for neuron_obj input (not segment_id yet)")
        return func(self,*args,**kwargs)
        
    return new_func
        
    

class DataInterfaceDefault(ABC):
    proof_version = 7
    
    def __init__(
        self,
        source = "default",
        **kwargs
        ):
        for k,v in default_settings.items():
            setattr(self,k,v)
            
        for k,v in kwargs.items():
            setattr(self,k,v)
        
        self.set_parameters_obj_from_filepath()
    
    # @property
    # @abstractmethod 
    # def parameters_config_filepath(self):
    #     return None

    @property
    def vdi(self):
        return self
        
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
        meshes_directory = None,
        plot = False,
        ext = "off"
        ): 
        """
        """
        
        if meshes_directory is None:
            meshes_directory = self.meshes_directory
        
        mesh_filepath = Path(meshes_directory) / Path(
            f"{segment_id}.{ext}")
        
        mesh = tu.load_mesh_no_processing(mesh_filepath)

        if plot: 
            ipvu.plot_objects(mesh)
            
        return mesh
    
    def fetch_undecimated_segment_id_mesh(
        self,
        segment_id,
        meshes_directory = None,
        plot = False,
        ext = "off",
        ):
        
        if meshes_directory is None:
            meshes_directory = self.meshes_undecimated_directory
        
        return self.fetch_segment_id_mesh(
            segment_id,
            meshes_directory = meshes_directory,
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
        
        
    def save_neuron_obj(
        self,
        neuron_obj,
        directory = None,
        filename = None,
        suffix = '',
        verbose = False,):
        
        if directory is None:
            directory = self.neuron_obj_directory
        
        if filename is None:
            filename = f"{neuron_obj.segment_id}{suffix}"
            
        filepath = nru.save_compressed_neuron(
            neuron_obj,
            output_folder = directory,
            file_name = filename,
            return_file_path = True,
        )
        
        if verbose:
            print(f"saved neuron filepath = {filepath}")

        return str(filepath) + ".pbz2"
    
        
    def load_neuron_obj(
        self,
        segment_id,
        mesh_decimated = None,
        meshes_directory = None,
        filepath = None,
        directory = None,
        filename = None,
        suffix = "",
        verbose = False,
        **kwargs):
        
        if mesh_decimated is None:
            mesh_decimated = self.fetch_segment_id_mesh(
                segment_id,
                meshes_directory = meshes_directory,)
        
        if filepath is None:
            if directory is None:
                directory = self.neuron_obj_directory
            
            if filename is None:
                filename = f"{segment_id}{suffix}"
                
            filepath = Path(directory) / Path(filename)
            
        return nru.decompress_neuron(
            filepath = filepath,
            original_mesh = mesh_decimated,
            **kwargs
        ) 
          
        
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
    
    
    def save_neuron_obj_auto_proof(
        self,
        neuron_obj,
        directory = None,
        filename = None,
        suffix = None,
        verbose = False,):
        if directory is None:
            directory = self.neuron_obj_auto_proof_directory
            
        if suffix is None:
            suffix = self.neuron_obj_auto_proof_suffix
            
        return self.save_neuron_obj(
            neuron_obj=neuron_obj,
            directory = directory,
            filename = filename,
            suffix = suffix,
            verbose = verbose,
        )
        
    def load_neuron_obj_auto_proof(
        self,
        segment_id,
        mesh_decimated = None,
        directory = None,
        **kwargs):
        
        
        if directory is None:
            directory = self.neuron_obj_auto_proof_directory
            
        if "suffix" not in kwargs:
            kwargs["suffix"] = self.neuron_obj_auto_proof_suffix
            
        return self.load_neuron_obj(
            segment_id,
            mesh_decimated = mesh_decimated,
            directory=directory,
            **kwargs
        )
        
        
    # ---- used by proximities
    def segment_id_and_split_index(
        self,
        segment_id,
        split_index = 0,
        return_dict = False):
        
        
        single_flag = False
        if not nu.is_array_like(segment_id):
            segment_id=[segment_id]
            split_index = [split_index]*len(segment_id)
            single_flag = True
        else:
            if not nu.is_array_like(split_index):
                split_index = [split_index]*len(segment_id)
        
        seg_ids_final = []
        sp_idxs_final = []
        for seg_id,sp_idx in zip(segment_id,split_index):
            if sp_idx == None:
                sp_idx = 0

            try:
                if "_" in seg_id:
                    seg_id,sp_idx = self.segment_id_and_split_index_from_node_name(seg_id)
            except:
                pass
            
            seg_ids_final.append(seg_id)
            sp_idxs_final.append(sp_idx)
            
        if single_flag:
            seg_ids_final = seg_ids_final[0]
            sp_idxs_final = sp_idxs_final[0]

        if not return_dict:
            return seg_ids_final,sp_idxs_final
        else:
            if single_flag:
                return dict(segment_id=seg_ids_final,split_index = sp_idxs_final)
            else:
                return [dict(segment_id=k,split_index = v) for k,v in zip(seg_ids_final,sp_idxs_final)]
    

    def segment_id_to_synapse_table_optimized_connectome(
        self,
        segment_id,
        split_index = 0,
        synapse_type = None,
        coordinates_nm = False,
        **kwargs
        ):
        """
        Purpose: to return a dataframe
        of the valid connections in connectome with
        the constraint of one segment id as a presyn or postsyn
        """
        return None

    def segment_id_to_synapse_table_optimized(
        self,
        segment_id,
        synapse_type = None,
        filter_away_self_synapses = True,
        coordinates_nm = True,
        synapse_filepath = None,
        **kwargs):
        if isinstance(segment_id,neuron.Neuron):
            if synapse_filepath is None:
                synapse_filepath = getattr(
                    segment_id,"synapse_filepath",None
                )
            segment_id = segment_id.segment_id
            
        
        if synapse_filepath is None:
            if self.synapse_filepath is None:
                raise Exception("No synapse filepath set")
            synapse_filepath = self.synapse_filepath
        
        return_df = syu.synapse_df_from_csv(
            synapse_filepath=synapse_filepath,
            segment_id=segment_id,
            coordinates_nm=coordinates_nm,
            **kwargs
        )
        
        return_df = pu.rename_columns(
            return_df,
            dict(
                segment_id = 'primary_seg_id',
                segment_id_secondary = "secondary_seg_id"
            )
        )
        
        if synapse_type is not None:
            return_df = return_df.query(f"prepost == '{synapse_type}'").reset_index(drop=True)

        return return_df
    
    @neuron_obj_func
    def segment_id_to_synapse_table_optimized_proofread(
        self,
        segment_id,
        split_index = 0,
        synapse_type = None,
        **kwargs
        ):
        neuron_obj = segment_id
        
        orig_syn_df = self.segment_id_to_synapse_table_optimized(
            neuron_obj,
            synapse_type = synapse_type,
            #synapse_filepath = "../Auto_Proof_Pipeline/Single_Soma_Inh/864691135567721964_synapses.csv"
        )

        syn_df = syu.synapses_df(
            neuron_obj.synapses_valid,
            add_compartment_coarse_fine = True,
            decode_head_neck_shaft_idx = True
        )
        
        syn_df = syn_df.query(f"compartment_coarse != 'error'").reset_index(drop=True)
        

        syn_df = pu.rename_columns(
            syn_df,
            dict(
                syn_type = "prepost",
                syn_id = "synapse_id",
                head_neck_shaft = "spine_bouton",
                soma_distance = "skeletal_distance_to_soma",
            )
        )

        merge_df = pd.merge(
            orig_syn_df,
            syn_df,
            on=['synapse_id','prepost'],
        ).reset_index(drop=True)

        return merge_df

    @neuron_obj_func
    def soma_nm_coordinate(
        self,
        segment_id,
        split_index = 0,
        return_dict = False,
        **kwargs):
            
        neuron_obj = segment_id
        
        return_value  = neuron_obj["S0"].mesh_center

        if return_dict:
            return {f"centroid_{ax}_nm":val for ax,val in zip(["x","y","z"],return_value)}
        else:
            return return_value
    
    @neuron_obj_func
    def graph_obj_from_proof_stage(
        self,
        segment_id,
        split_index = 0,
        clean = True,
        verbose = False,
        **kwargs
        ):
        """
        Purpose: to get the neuron_obj from 
        decomposition cell type
        """
        neuron_obj = segment_id
        G = neuron_obj.neuron_graph_after_proof
            
        if clean: 
            G = nxu.clean_G(G,verbose = verbose,**kwargs)
            
        return G
    
    @neuron_obj_func
    def fetch_proofread_mesh(
        self,
        segment_id,
        split_index = 0,
        plot_mesh = False,
        **kwargs
        
        ):
        
        neuron_obj = segment_id
        mesh = neuron_obj.mesh_from_branches
        
        if plot_mesh:
            ipvu.plot_objects(mesh)
            
        return mesh
    
    @neuron_obj_func
    def fetch_soma_mesh(
        self,
        segment_id,
        split_index = 0,
        plot_mesh = False,
        **kwargs
        
        ):
        
        neuron_obj = segment_id
        mesh = neuron_obj["S0"].mesh
        
        if plot_mesh:
            ipvu.plot_objects(mesh)
            
        return mesh
    
    @neuron_obj_func
    def pre_post_synapse_ids_coords_from_connectome(
        self,
        segment_id_pre,
        segment_id_post,
        split_index_pre=0,
        split_index_post=0,
        synapse_pre_df = None,
        verbose= False,
        ):
        
        if synapse_pre_df is None:
            synapse_pre_df = self.segment_id_to_synapse_table_optimized_proofread(
                segment_id_pre,
                split_index = split_index_pre,
                synapse_type="presyn",
            )

        def synapse_coordinates_from_df(df):
            return df[["synapse_x_nm","synapse_y_nm","synapse_z_nm"]].to_numpy().astype('float')

        # gets the pre and post synapses
        
        if len(synapse_pre_df) > 0:
            syn_ids = syu.synapse_df(segment_id_post.synapses_valid)["syn_id"].to_list()

            
            synapse_pre_post_df = synapse_pre_df.query(
            f"(secondary_seg_id == {segment_id_post.segment_id})")
            
            synapse_pre_post_df = synapse_pre_post_df.query(
            f"(synapse_id in {syn_ids})"
            )


            synapse_pre_post_coords = synapse_coordinates_from_df(synapse_pre_post_df)
            synapse_pre_post_ids = synapse_pre_post_df["synapse_id"].to_numpy()
        else:
            synapse_pre_post_coords = []
            synapse_pre_post_ids = []

        if verbose:
            print(f"synapse_pre_post_coords = {synapse_pre_post_coords}")
            print(f"synapse_pre_post_ids = {synapse_pre_post_ids}")

        return synapse_pre_post_ids,synapse_pre_post_coords
        
    
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
from python_tools import pandas_utils as pu
from python_tools import ipyvolume_utils as ipvu

from mesh_tools import trimesh_utils as tu

from neuron_morphology_tools import neuron_nx_utils as nxu

from . import graph_filters as gf
from . import proofreading_utils as pru
from . import parameter_utils as paru
from . import synapse_utils as syu
from . import neuron_visualizations as nviz
from . import neuron_utils as nru
from . import neuron



