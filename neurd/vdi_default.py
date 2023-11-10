from abc import (ABC,abstractmethod,)
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from typing import List,Union
from .neuron import Neuron
import networkx as nx

# --- for parameters
parameters_config_filename = "parameters_config_default.py"
config_filepath = str((
    Path(__file__).parents[0]
    / Path(f"parameter_configs/{parameters_config_filename}")).absolute()
)


default_settings = dict(
    parameters_config_filepath_default = config_filepath,
    
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
        

class DataInterfaceBoilerplate(ABC):
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
        
        self.set_parameters_obj()
        

    @property
    def vdi(self):
        return self
        
    def set_parameters_obj(
        self
        ):
        """
        Purpose: To set the parameters obj using the
        """
        
        
        parameters_config_filepath_default = getattr(
            self,
            "parameters_config_filepath_default",
            None)
        
        filepaths = []
        if parameters_config_filepath_default is not None:
            filepaths.append(parameters_config_filepath_default)
            
            
        override_filepaths = getattr(self,"parameters_config_filepaths",[])
        override_filepaths = nu.to_list(override_filepaths)
        
        filepaths += override_filepaths
        
        for f in filepaths:
            #print(f"Setting parameters from {f}")
            self.set_parameters_obj_from_filepath(
                filepath = f
            )
        
    def set_parameters_obj_from_filepath(
        self,
        filepath=None,
        set_module_parameters = True):
        
        if not hasattr(self,"parameters_obj"):
            setattr(self,"parameters_obj",paru.PackageParameters())
        
        if filepath is None:
            filepath = self.parameters_config_filepath
            
        if filepath is None:
            return 
        
        
        parameters_obj_curr = paru.parameters_from_filepath(
            filepath = filepath
        )
        
        self.parameters_obj.update(
            parameters_obj_curr
        )
        
        if set_module_parameters:
            self.set_parameters_for_directory_modules()
            
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
        
    # --------------------------
    def align_array(self,array,align_matrix = None,**kwargs):
        #print(f"inside vdi align array")
        return nru.align_array(array,align_matrix = align_matrix)

    def align_mesh(self,mesh,align_matrix = None,**kwargs):
        #print(f"inside vdi align mesh")
        return nru.align_mesh(mesh,align_matrix=align_matrix)

    def align_skeleton(self,skeleton,align_matrix = None,**kwargs):
        #print(f"inside vdi align skeleton")
        return nru.align_skeleton(skeleton=skeleton,align_matrix = align_matrix)
    
    def align_neuron_obj_from_align_matrix(
        self,
        neuron_obj,
        align_matrix=None,
        **kwargs):
        return nru.align_neuron_obj_from_align_matrix(
            neuron_obj,
            align_matrix=align_matrix,
            align_array = self.align_array,
            align_mesh=self.align_mesh,
            align_skeleton=self.align_skeleton,
            **kwargs)
    
    def unalign_neuron_obj_from_align_matrix(self,neuron_obj,align_matrix=None,**kwargs):
        return nru.unalign_neuron_obj_from_align_matrix(neuron_obj,align_matrix=align_matrix,**kwargs)

    def align_neuron_obj(self,neuron_obj,align_matrix = None,**kwargs):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        if align_matrix is None:
            align_matrix = self.get_align_matrix(neuron_obj,**kwargs)
        return self.align_neuron_obj_from_align_matrix(
            neuron_obj,
            align_matrix = align_matrix,
            **kwargs
        )

    def unalign_neuron_obj(self,neuron_obj,align_matrix = None,**kwargs):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        if align_matrix is None:
            align_matrix = self.get_align_matrix(neuron_obj,**kwargs)
        return self.unalign_neuron_obj_from_align_matrix(
            neuron_obj,
            align_matrix = align_matrix,
            **kwargs
        )
    
    # -------- save and load for neuron obj
    def save_neuron_obj(
        self,
        neuron_obj,
        directory = None,
        filename = None,
        suffix = '',
        verbose = False,):
        """
        

        Parameters
        ----------
        neuron_obj : _type_
            
        directory : _type_, optional
            by default None
        filename : _type_, optional
            by default None
        suffix : str, optional
            by default ''
        verbose : bool, optional
            by default False

        Returns
        -------
        _type_
            
        """
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
        segment_id=None,
        mesh_decimated = None,
        mesh_filepath = None,
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
                meshes_directory = meshes_directory,
                mesh_filepath=mesh_filepath)
        
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
    
    
    def add_nm_to_synapse_df(self,df):
        return syu.add_nm_to_synapse_df(
            df,
            scaling=self.voxel_to_nm_scaling,
        )
    def segment_id_to_synapse_dict(
        self,
        segment_id,
        **kwargs):
        """
        
        Purpose: return a dictionary containing the presyn and postsyn information for a certain segment from the backend datasource implmeneted for the data. The structure of the returned dictionary should in the following format where all coordinates and sizes ARE SCALED TO NM ALREADY
        
        syn_dict = dict(
            presyn = dict(
                synapse_ids= np.array (N),
                synapse_coordinates = np.array (Nx3),
                synapse_sizes = np.array (N),
            ),
            postsyn = dict(
                synapse_ids= np.array (N),
                synapse_coordinates = np.array (Nx3),
                synapse_sizes = np.array (N),
            )
        )
        
        
        The default implementation assumes there is a local synapse csv file (whose path needs to be passed as an argument or set with as an object attribute) with the following columns
        
        segment_id,
        segment_id_secondary,
        synapse_id,
        prepost, # presyn or postsyn
        synapse_x, # in voxel coordinates
        synapse_y, # in voxel coordinates
        synapse_z, # in voxel coordinates
        synapse_size, # in voxel coordinates
        
        
        Example Implementation
        ----------------------
        cave_client_utils.synapse_df_from_seg_id
        
        """
        
        
        syn_df = self.segment_id_to_synapse_df(
            segment_id,
            **kwargs,
        )
        
        df = self.add_nm_to_synapse_df(syn_df)
        
        syn_dict = syu.synapse_dict_from_synapse_df(
            df,
            scaling = None,
            coordinates_nm = True,
            **kwargs
        )
        
        return syn_dict
    
    def segment_id_to_synapse_table_optimized(
        self,
        segment_id,
        synapse_type = None,
        filter_away_self_synapses = True,
        coordinates_nm = True,
        synapse_filepath = None,
        **kwargs):
        """
        Purpose: Given a segment id (or neuron obj)
        will retrieve the synapses from a backend synapse implementation renamed in a particular manner
        

        Parameters
        ----------
        segment_id : int or neuron.Neuron
            
        synapse_type : _type_, optional
            by default None
        filter_away_self_synapses : bool, optional
            by default True
        coordinates_nm : bool, optional
            by default True
        synapse_filepath : _type_, optional
            by default None

        Returns
        -------
        _type_
            

        Raises
        ------
        Exception
            
        """
        if isinstance(segment_id,neuron.Neuron):
            if synapse_filepath is None:
                synapse_filepath = getattr(
                    segment_id,"synapse_filepath",None
                )
            segment_id = segment_id.segment_id
            
        return_df = self.segment_id_to_synapse_df(
            segment_id=segment_id,
            synapse_filepath = synapse_filepath,
            coordinates_nm=coordinates_nm,
        )
        
        if coordinates_nm:
            return_df = self.add_nm_to_synapse_df(return_df)
        
        # if synapse_filepath is None:
        #     if self.synapse_filepath is None:
        #         raise Exception("No synapse filepath set")
        #     synapse_filepath = self.synapse_filepath
        
        # return_df = syu.synapse_df_from_csv(
        #     synapse_filepath=synapse_filepath,
        #     segment_id=segment_id,
        #     coordinates_nm=coordinates_nm,
        #     **kwargs
        # )
        
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
    
    
class DataInterfaceDefault(DataInterfaceBoilerplate):
    """
    Class to outline what functions to overload in implement a volume data interface that will work with NEURD. All methods exposed fall under the following categories
    
    1) required abstract method
    2) data fetchers/setters
    3) autoproofreading filter settings
    
    All fetchers and setters have a default implementation where data is stored locally in csv (for synapses) or locally in the neuron object. If exporting data to non-local source (ex: database) override these functions to pull from other these sources
    """
    def __init__(
        self,
        *args,
        **kwargs
        ):
        
        super().__init__(*args,**kwargs)
        
    @property    
    @abstractmethod
    def voxel_to_nm_scaling(self):
        """
        ***REQUIRED OVERRIDE***
        
        Purpose: Provide a 1x3 numpy matrix representing the scaling of voxel units to nm units. If the data is already in nm format then just assign a ones matrix
        
        Returns
        -------
        scaling_vector : np.array
            vector that can convert a matrix or vector of 3D voxel coordinates to 3D nm coordinates (default: np.array([1,1,1]))
        """
        return np.array([1,1,1])
    
    @abstractmethod
    def get_align_matrix(self,*args,**kwargs):
        """
        ***REQUIRED OVERRIDE***
        
        Purpose: a transformation matrix (call A, 3x3) that when applied to a matrix of 3D coordinates (call B, Nx3) as a matrix multiplication of C = BA will produce a new matrix of rotated coordinates (call C, Nx3) so that all coordinates or a mesh or skeleton are rotated to ensure that the apical of the neuron is generally direted in the positive z direction.
        
        """
        return None
    
    @abstractmethod
    def segment_id_to_synapse_df(
        self,
        segment_id,
        **kwargs,
        ):
        """
        ***REQUIRED OVERRIDE***
        
        Purpose: return a dataframe with the presyn
        and postsyn information for a certain segment from the backend data source. The structure of the dataframe should return the following columns
        
        segment_id,
        segment_id_secondary,
        synapse_id,
        prepost, # presyn or postsyn
        synapse_x, # in voxel coordinates
        synapse_y, # in voxel coordinates
        synapse_z, # in voxel coordinates
        synapse_size, # in voxel coordinates
        
        The default implementation assumes there is a local synapse csv file (whose path needs to be passed as an argument or set with as an object attribute)
    
        Parameters
        ___________
        
        segment_id: int
        coordinates_nm: bool
            Whether to scale the coordinate to nm units
        scaling: np.array
            The scaling factor to use
        
        Returns
        -------
        pd.DataFrame
            dataframe with all of the relevant synapse information for one segment id
        """
        if scaling is None:
            scaling = self.voxel_to_nm_scaling
            
        if kwargs.get("synapse_filepath",None) is None:
            if self.synapse_filepath is None:
                raise Exception("No synapse filepath set")
            kwargs["synapse_filepath"] = self.synapse_filepath
        
        df = syu.synapse_df_from_csv(
            synapse_filepath,
            segment_id = segment_id,
            coordinates_nm = False,
            scaling = None,
            **kwargs
        )
        
        return df
    
    # ---- Data Fetching and Savinig Functions
    def fetch_segment_id_mesh(
        self,
        segment_id:int=None,
        meshes_directory:str = None,
        mesh_filepath:str = None,
        plot:bool = False,
        ext:str = "off"
        ) -> trimesh.Trimesh:
        """
        Purpose: retrieve a decimated segment id mesh. Current implementation assumes a local filepath storing all meshes.

        Parameters
        ----------
        segment_id : int, optional
            neuron segment id, by default None
        meshes_directory : str, optional
            location of decimated mesh files, by default None
        mesh_filepath : str, optional
            complete path of location and filename for neuron , by default None
        plot : bool, optional
            by default False
        ext : str, optional
            the file extension for mesh storage, by default "off"

        Returns
        -------
        trimesh.Trimesh
            decimated mesh for segment id
        """
        
        if mesh_filepath is None:
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
        segment_id:int,
        meshes_directory:str = None,
        plot:bool = False,
        ext:str = "off",
        ) -> trimesh.Trimesh:
        """
        

        Parameters
        ----------
        segment_id : int
            
        meshes_directory : str, optional
            by default None
        plot : bool, optional
            by default False
        ext : str, optional
            by default "off"

        Returns
        -------
        trimesh.Trimesh
            undecimated mesh for segment id
        """
        
        if meshes_directory is None:
            meshes_directory = self.meshes_undecimated_directory
        
        return self.fetch_segment_id_mesh(
            segment_id,
            meshes_directory = meshes_directory,
            plot = plot,
            ext = ext
        )
    
    def set_synapse_filepath(
        self,
        synapse_filepath:str
        ) -> None:
        """
        sets the location and filename of the synapse csv for the default implementation that loads synapses from a local csv file

        Parameters
        ----------
        synapse_filepath : str
            complete folder path and filename for synapse csv
        """ 
        self.synapse_filepath = synapse_filepath

    def nuclei_from_segment_id(
        self,
        segment_id:int,
        return_centers:bool=True,
        return_nm:bool=True,
        )->np.array:
        """
        retrieves the nuclei id (and possibly the 
        nuclei centers) from an external database. No external database currently set so currently set to None returns. 

        Parameters
        ----------
        segment_id : int
            
        return_centers : bool, optional
            whether to return the nuclei center coordinates along with the ids, by default True
        return_nm : bool, optional
            whether to return nuclei center coordinates in nm units, by default True

        Returns
        -------
        nuclei_ids: np.array (N,)
            nuclei ids corresponding to segment_id
        nuclei_centers: np.array (N,3), optional
            center locations for the corresponding nuclei
        """
        nuclues_ids = None
        nucleus_centers = None
        
        if return_centers:
            return nuclues_ids,nucleus_centers
        else:
            return nucleus_ids
           
    def nuclei_classification_info_from_nucleus_id(
        self,
        nuclei:int,
        *args,
        **kwargs,
        )->dict:
        """
        Purpose: To return a dictionary of cell type
        information (same structure as from the allen institute of brain science CAVE client return) from an external database. No external database currently set up so None filled dictionary returned.   
    

        Example Returns: 
        
        {
            'external_cell_type': 'excitatory',
            'external_cell_type_n_nuc': 1,
            'external_cell_type_fine': '23P',
            'external_cell_type_fine_n_nuc': 1,
            'external_cell_type_fine_e_i': 'excitatory'
        }

        Parameters
        ----------
        nuclei : int
            

        Returns
        -------
        dict
            nuclei info about classification (fine and coarse)
        """
        
        return {
            'external_cell_type': None,
            'external_cell_type_n_nuc': None,
            'external_cell_type_fine': None,
            'external_cell_type_fine_n_nuc': None,
            'external_cell_type_fine_e_i': None,
        }
    
        
    def save_neuron_obj_auto_proof(
        self,
        neuron_obj:Neuron,
        directory:str=None,
        filename:str = None,
        suffix:str = None,
        verbose:bool = False,) -> str:
        """
        Save a neuron object in the autoproofreading directory (using the default pbz2 compressed method that does not save the mesh along with it). Typical  This is the current local implementation, should be overriden if the proofreading neuron objects are to be saved in an external store 
        
        Default filename: {segment_id}.pbz2 

        Parameters
        ----------
        neuron_obj : Neuron
            
        directory : str, optional
            location for storing .pbz2 files, by default None
        filename : str, optional
            a custom name for compressed neuron file to replace the default name, by default None
        suffix : str, optional
            change filename to {segment_id}{suffix}.pbz2 , by default None
        verbose : bool, optional
            by default False

        Returns
        -------
        str
            filepath of saved neuron file
        """
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
        segment_id:str,
        mesh_decimated:trimesh.Trimesh = None,
        directory:str = None,
        **kwargs) -> Neuron:
        """
        Loading an external neuron file into a python object. Current implementation assumes the default .pbz2 method of compression that does not store the mesh information, which is why the mesh object needs to be passed as an argument

        Parameters
        ----------
        segment_id : str
            
        mesh_decimated : trimesh.Trimesh, optional
            the original decimated mesh before any proofreaidng, by default None
        directory : str, optional
            filepath location of saved .pbz2 file, by default self.neuron_obj_auto_proof_directory

        Returns
        -------
        Neuron
            
        """
        
        
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
        
        
        
    # --------- functions for setting autoproofreading ---
    def exc_filters_auto_proof(*args,**kwargs):
        """
        All autoproofreading filters (referenced in proofreading_utils.py) that will be used for excitatory cells

        Returns
        -------
        List[filter objects]
            
        """
        return pru.v7_exc_filters(
            *args,**kwargs
    )
        
    def inh_filters_auto_proof(*args,**kwargs):
        """
        All autoproofreading filters (referenced in proofreading_utils.py) that will be used for inhibitory cells

        Returns
        -------
        List[filter functions]
            
        """
        return pru.v7_inh_filters(
            *args,**kwargs
        )
    
    @property
    def default_low_degree_graph_filters(self):
        """
        The graph filters to be using the 'exc_low_degree_branching_filter' for autoproofreading that inspects axon branches with exactly 2 downstream nodes and classifies as an error based on if one fo the following graph filters has a successful match. Overriding this function could be simply excluding some filters that are not applicable/work for your volume even with parameters tuned

        Returns
        -------
        List[graph filter functions]
        """
        return [
            gf.axon_webbing_filter,
            gf.thick_t_filter,
            gf.axon_double_back_filter,
            gf.fork_divergence_filter,
            gf.fork_min_skeletal_distance_filter,
            gf.axon_spine_at_intersection_filter,
            gf.min_synapse_dist_to_branch_point_filter,
        ]
        
        
    # --- Fetching functions of data (default to searching neuron obj because no default database set up)
    @neuron_obj_func
    def fetch_proofread_mesh(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        plot_mesh:bool = False,
        **kwargs
        )-> trimesh.Trimesh:
        """
        Retrieve mesh after autoproofreading filtering. Default implementation uses a local solution of extracting the mesh from the neuron object, but the proofreading mesh could be stored in an external database with only the segment id and split index needed to retrieve. 

        Parameters
        ----------
        segment_id : Union[int,Neuron]
            proofread neuron object from which the mesh can be extracted or an int representing the segment id for external database implementation where saved products indexed by unique segment_id and split index
        split_index : int, optional
            for external database implementation where saved products indexed by unique segment_id and split index, by default 0
        plot_mesh : bool, optional
            by default False

        Returns
        -------
        trimesh.Trimesh
            auto proofread mesh
        """
        
        neuron_obj = segment_id
        mesh = neuron_obj.mesh_from_branches
        
        if plot_mesh:
            ipvu.plot_objects(mesh)
            
        return mesh
    
    @neuron_obj_func
    def fetch_soma_mesh(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        plot_mesh:bool = False,
        **kwargs
        
        ):
        """
        Retrieve soma mesh. Default implementation uses a local solution of extracting the soma mesh from the neuron object, but the soma mesh could be stored in an external database with only the segment id and split index needed to retrieve. 

        Parameters
        ----------
        segment_id : Union[int,Neuron]
            neuron object from which the mesh can be extracted or an int representing the segment id for external database implementation where saved products indexed by unique segment_id and split index
        split_index : int, optional
            for external database implementation where saved products indexed by unique segment_id and split index, by default 0
        plot_mesh : bool, optional
            by default False

        Returns
        -------
        trimesh.Trimesh
            auto proofread mesh
        """
        
        neuron_obj = segment_id
        mesh = neuron_obj["S0"].mesh
        
        if plot_mesh:
            ipvu.plot_objects(mesh)
            
        return mesh
    
    def segment_id_to_synapse_table_optimized_connectome(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        synapse_type:str = None,
        coordinates_nm:bool = False,
        **kwargs
        ):
        """
        Purpose: to return a dataframe
        of the connections before proofreading with
        the constraint of one segment_id/split_index as a presyn or postsyn. Not implemented for local storage
        """
        return None

    
    
    @neuron_obj_func
    def segment_id_to_synapse_table_optimized_proofread(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        synapse_type:str = None,
        **kwargs
        ):
        """
        Purpose: to return a dataframe
        of the valid connections in the proofread segment/split. Currently only implemented for local solution of where synapse information stored in local csv and proofrad synapses are stored in neuron object. Could override to pull original or proofread synapses from an external source.

        Parameters
        ----------
        segment_id : Union[int,Neuron]
            neuron obj with proofread synapses, or just segment id if synapses stored externally
        split_index : int, optional
            identifier for segment if stored externally, by default 0
        synapse_type : str, optional
            presyn or postsyn restriction, by default None

        Returns
        -------
        synapse_df : pd.DataFrame
        """
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
        
        if synapse_type is not None:
            syn_df = syn_df.query(f"prepost == '{synapse_type}'").reset_index(drop=True) 

        merge_df = pd.merge(
            orig_syn_df,
            syn_df,
            on=['synapse_id','prepost'],
        ).reset_index(drop=True)

        return merge_df

    @neuron_obj_func
    def soma_nm_coordinate(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        return_dict:bool = False,
        **kwargs)->np.array:
        """
        Return the soma coordinate for a segment. Implemented with local solution of accepting neuron object but could override with external store fetching.

        Parameters
        ----------
        segment_id : Union[int,Neuron]
            
        split_index : int, optional
            by default 0
        return_dict : bool, optional
            by default False

        Returns
        -------
        soma coordinate: np.array (3,)
        """
            
        neuron_obj = segment_id
        
        return_value  = neuron_obj["S0"].mesh_center

        if return_dict:
            return {f"centroid_{ax}_nm":val for ax,val in zip(["x","y","z"],return_value)}
        else:
            return return_value

    
    @neuron_obj_func
    def graph_obj_from_proof_stage(
        self,
        segment_id:Union[int,Neuron],
        split_index:int = 0,
        clean:bool = True,
        verbose:bool = False,
        **kwargs
        ) -> nx.DiGraph:
        """
        
        Purpose: Retrieve the lite neuron_obj (implemented). Local implementation where retrieved from pipeline products of neuron obj but could override to fetch from an external store using the segment id and split index

        Parameters
        ----------
        segment_id : Union[int,Neuron]
            
        split_index : int, optional
            by default 0
        clean : bool, optional
            by default True
        verbose : bool, optional
            by default False

        Returns
        -------
        neuron_obj_lite: nx.DiGraph
        """
        
        
        neuron_obj = segment_id
        G = neuron_obj.neuron_graph_after_proof
            
        if clean: 
            G = nxu.clean_G(G,verbose = verbose,**kwargs)
            
        return G
    
    
        

from datasci_tools import pandas_utils as pu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import numpy_utils as nu
from datasci_tools import mesh_utils as meshu

from mesh_tools import trimesh_utils as tu

from neuron_morphology_tools import neuron_nx_utils as nxu

from . import graph_filters as gf
from . import proofreading_utils as pru
from . import parameter_utils as paru
from . import synapse_utils as syu
from . import neuron_visualizations as nviz
from . import neuron_utils as nru
from . import neuron



