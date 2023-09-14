from abc import (
  ABC,
  abstractmethod,)


class DataInterface(ABC):
    def __init__(
        self,
        synapse_filepath = None,
        parameters_config_filepath = None,
        ):
        self.source = source
        self.synapse_filepath = synapse_filepath
        self.parameters_config_filepath = parameters_config_filepath
        self.set_parameters_obj_from_filepath()
    
    @property
    @abstractmethod 
    def parameters_config_filepath(self):
        return None
        
    def set_parameters_obj_from_filepath(self,filepath=None):
        import parameters_utils as paru
        
        if filepath is None:
            filepath = self.parameters_config_filepath
            
        if filepath is None:
            return 
        
        self.parameters_obj = paru.parameters_from_filepath(
            filepath = filepath
        )
            
    def set_module_parameters_from_parameters_obj(
        self,
        directory = None,
        **kwargs):
        
        paru.set_parameters_for_directory_modules_from_obj(
            obj = self.parameters_obj,
            directory = directory,
            **kwargs
        )
        
    # -------- abstract properties
    @property
    @abstractmethod
    def voxel_to_nm_scaling(self):
        pass
    
    @property
    @abstractmethod
    def source(self):
        pass
    
    # --------------------------

    @abstractmethod
    def align_array(self):
        pass

    @abstractmethod
    def align_mesh(self):
        pass

    @abstractmethod
    def align_skeleton(self):
        pass

    @abstractmethod
    def align_neuron_obj(self):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        pass

    @abstractmethod
    def unalign_neuron_obj(self):
        """
        Keep the body of function as "pass" unless the neuron obj needs to be rotated so axon is pointing down
        """
        pass
    
    def set_synapse_filepath(self,synapse_filepath):
        self.synapse_filepath = synapse_filepath
        
    @abstractmethod
    def segment_id_to_synapse_dict(
        self,
        **kwargs):
        from . import synapse_utils as syu
        
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
        from . import proofreading_utils as pru
        return pru.v7_exc_filters(
            *args,**kwargs
        )
        
    def inh_filters_auto_proof(*args,**kwargs):
        from . import proofreading_utils as pru
        return pru.v7_inh_filters(
            *args,**kwargs
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
            gf.axon_spine_at_intersection_filter,
            gf.min_synapse_dist_to_branch_point_filter,
        ]
