
from abc import (
  ABC,
  abstractmethod,)
class DataInterface(ABC):
    def __init__(
        self, 
        source,
        voxel_to_nm_scaling = None,
        synapse_filepath = None
        ):
        self.source = source
        self.voxel_to_nm_scaling = voxel_to_nm_scaling
        self.synapse_filepath = synapse_filepath
        
    def set_synapse_filepath(self,synapse_filepath):
        self.synapse_filepath = synapse_filepath

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
        pass

    @abstractmethod
    def unalign_neuron_obj(self):
        pass
    
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
    