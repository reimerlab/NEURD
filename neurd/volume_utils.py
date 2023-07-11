from abc import (
  ABC,
  abstractmethod,
)


class DataInterface(ABC):
  def __init__(
        self, 
        source,
        voxel_to_nm_scaling = None,
        ):
    self.source = source
    self.voxel_to_nm_scaling = None
    
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
  
  
      