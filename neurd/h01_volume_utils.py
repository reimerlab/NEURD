
import copy
from python_tools import numpy_dep as np
from . import microns_volume_utils as mvu

voxel_to_nm_scaling = np.array([8,8,33])
source = "h01"
current_nucleus_version = 0
# -------------- Functions for finding the rotation matrix needed -------
top_radius =700_000 #when magn = 1
bottom_radius = 350_000 #when magn = 0
max_rotation_global = -30
upward_vector_middle = np.array([ 0.94034618, -0.34021915,  0.        ])
upward_vector_middle_non_scaled = np.array([1808188.88892619, -654206.39541785,       0.        ])
upward_vector_start_point = np.array([1041738.17659344, 1785911.29763922,  125032.57443884])
align_vector = np.array([ 0.85082648, -0.52544676,  0.        ])
upward_vector_top_left = np.array([])



def aligning_matrix_3D(upward_vector=align_vector,#upward_vector_middle,
                target_vector = mvu.top_of_layer_vector,
                       rotation = None,
                       verbose = False,
                      ):
    """
    Will come up with an alignment matrix
    """
    upward_vector = upward_vector.copy()
    
    if rotation is not None:
        if verbose:
            print(f"upward_vector before = {upward_vector}")
        upward_vector[:2] = lu.rotation_matrix_2D(rotation) @ upward_vector[:2]
        if verbose:
            print(f"upward_vector AFTER = {upward_vector}")
    return nu.aligning_matrix_3D(upward_vector,target_vector)

def align_mesh_from_rotation(mesh,align_mat = None,upward_vector = None,
               rotation= None,
                             verbose = False,
               **kwargs):
    """
    Need a better version of rotation 
    
    """
    #verbose = True,
    #print(f"Inside align mesh")
    if upward_vector is not None:
        kwargs["upward_vector"] = upward_vector
    if rotation is not None:
        kwargs["rotation"] = rotation

    
    if align_mat is None:
        align_mat = aligning_matrix_3D(**kwargs)
        
    if verbose:
        print(f"align_mat = {align_mat}  ")
    
    return rotate_mesh_from_matrix(mesh,align_mat)


    

def radius_for_rotation_from_proj_magn(magn):
    return (top_radius-bottom_radius)*magn + bottom_radius
    



def rotation_from_proj_error_and_radius(
    proj_error,
    radius_for_rotation,
    max_rotation = max_rotation_global,
    verbose = False
    ):
    """
    Purpose: To calculate the amount of rotation necessary 
    based on the current radius of rotation and error
    magnitude of the projection
    
    """
    magn_err = np.linalg.norm(proj_error)
    if verbose:
        print(f"magn_err = {magn_err}")
        
    rotation = max_rotation*(magn_err/radius_for_rotation)
    
    if verbose:
        print(f"rotation = {rotation} (max_rotation = {max_rotation})")
        
    return rotation



def rotation_signed_from_middle_vector(
    coordinate,
    origin_coordinate=upward_vector_start_point,
    middle_vector = upward_vector_middle_non_scaled,
    zero_out_z_coord = True,
    verbose = False,
    ):
    """
    Purpose: Determine the direction
    and amount of rotation needed for a 
    neuron based on the location of the soma

    Pseudocode: 
    1) Compute the new relative vector to starting vector
    2) Find the magnitude of projection of new point onto upward middle vector non scaled
    3) Use the magnitude of the projection to find the slope of the rotation function
    4) Find the error distance between point and projection distance
    5) Determine the amount of rotation needed based on radius and error projection magnitude
    6) Determine the sign of the rotation
    
    Ex: 
    rotation_signed_from_middle_vector(
    coordinate = orienting_coords["bottom_far_right"],
        verbose = True
    )
    """
    #print(f"verbose rotation_signed_from_middle_vector = {verbose}")
    #1) Compute the new relative vector to starting vector
    v = coordinate - origin_coordinate
    m = middle_vector

    
    if zero_out_z_coord:
        idx_for_projection = np.arange(0,2)
    else:
        idx_for_projection = np.arange(0,3)

    if verbose:
        print(f"new vector = {v}")

    #2) Find the magnitude of projection of new point onto upward middle vector non scaled
    proj_v,magn = lu.projection(
        vector_to_project=v,
        line_of_projection=m,
        idx_for_projection=idx_for_projection,
        verbose=verbose,
        return_magnitude=True
    )

    #3) Use the magnitude of the projection to find the slope of the rotation function
    radius_rot = radius_for_rotation_from_proj_magn(magn)
    if verbose:
        print(f"radius_rot= {radius_rot}")

    #4) Find the error distance between point and projection distance
    proj_er = lu.error_from_projection(
        vector_to_project=v,
        line_of_projection=m,
        idx_for_projection=idx_for_projection,
        verbose=False
    )

    #5) Determine the amount of rotation needed based on radius and error projection magnitude
    curr_rotation = rotation_from_proj_error_and_radius(
        proj_error = proj_er,
        radius_for_rotation = radius_rot,
        verbose = False
    )

    #6) Determine the sign of the rotation
    if zero_out_z_coord:
        m_perp = lu.perpendicular_vec_2D(m[idx_for_projection])
        rotation_sign = np.sign(m_perp @ proj_er)
    else:
        rotation_sign = None

    if verbose:
        print(f"rotation_sign = {rotation_sign}")
        print(f"curr_rotation = {curr_rotation}")
    
    return curr_rotation*rotation_sign

def rotation_from_soma_center(soma_center,
                             verbose = False,
                             **kwargs):
    """
    Purpose: To get the amout r tation necessary from soma
    center of neuron
    
    """
    soma_center = np.array(soma_center).reshape(-1)
    rotation = rotation_signed_from_middle_vector(soma_center,
                                                     verbose = verbose,
                                                     **kwargs)
    if verbose:
        print(f"rotation = {rotation}")
    return rotation
    

def align_mesh(
    mesh,
    soma_center=None,
    rotation = None,
    align_matrix = None,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To align a mesh by a soma coordinate
    
    Ex: 
    # rotating the mesh
    nviz.plot_objects(align_mesh_from_soma_coordinate(mesh,
                                                         soma_center=soma_mesh_center
                                                        ))
    """
    #print(f"verbose align_mesh_from_soma_coordinate = {verbose}")
    if len(mesh.faces) <= 0:
        return mesh
    if align_matrix is None:
        if rotation is None:
            soma_center = np.array(soma_center).reshape(-1)
            rotation = rotation_signed_from_middle_vector(soma_center,
                                                             verbose = verbose,
                                                             **kwargs)
            if verbose:
                print(f"rotation = {rotation}")

        return align_mesh_from_rotation(mesh,rotation=rotation)
    else:
        if verbose:
            print(f"Using matrix align_matrix = {align_matrix}")
        return rotate_mesh_from_matrix(mesh,align_matrix)



def align_matrix_from_rotation(upward_vector=None,
                              rotation=None,
                              **kwargs):
    if upward_vector is not None:
        kwargs["upward_vector"] = upward_vector
    if rotation is not None:
        kwargs["rotation"] = rotation
    return aligning_matrix_3D(**kwargs)

def align_matrix_from_soma_coordinate(
    soma_center,
    verbose = False,
    **kwargs
    ):
    #print(f"verbose = {verbose}")
    """
    Purpose: To align a mesh by a soma coordinate
    
    Ex: 
    # rotating the mesh
    nviz.plot_objects(align_mesh_from_soma_coordinate(mesh,
                                                         soma_center=soma_mesh_center
                                                        ))
    """
    #print(f"verbose align_matrix_from_soma_coordinate = {verbose}")
    soma_center = np.array(soma_center).reshape(-1)
    rotation = rotation_signed_from_middle_vector(soma_center,
                                                     verbose=verbose,
                                                     **kwargs)
    if verbose:
        print(f"rotation = {rotation}")
        
    return align_matrix_from_rotation(rotation=rotation)


def align_array(
    array,
    soma_center=None,
    rotation = None,
    align_matrix = None,
    verbose = False,
    **kwargs
    ):
    #print(f"verbose align_array_from_soma_coordinate = {verbose}")
    """
    Purpose: Will align a coordinate or skeleton
    (or any array) with the rotation matrix
    determined from the soam center
    """
    if len(array) <= 0:
        return array
    
    
    curr_shape = array.shape
    #verbose = True
    
    if align_matrix is None:
        if rotation is None:
            rotation = rotation_from_soma_center(soma_center,
                                     verbose = False,
                                     **kwargs)
            if verbose:
                print(f"rotation = {rotation}")
        
    
        align_matrix = align_matrix_from_rotation(
            rotation = rotation,
            verbose = verbose,
        )
        
        if verbose:
            print(f"align_matrix = {align_matrix}")

    new_coords = array.reshape(-1,3) @ align_matrix
    new_array = new_coords.reshape(*curr_shape)
    
    return new_array

def align_skeleton(
    array,
    soma_center=None,
    rotation=None,
    align_matrix = None,
    verbose = False,
    **kwargs
    ):
    #print(f"verbose align_skeleton_from_soma_coordinate = {verbose}")
    
    if len(array) <= 0:
        return array
    
    return align_array(
    array,
    soma_center=soma_center,
    rotation=rotation,
    align_matrix = align_matrix,
    verbose = verbose,
    **kwargs
    )


def align_attribute(obj,attribute_name,
                    soma_center=None,
                    rotation=None,
                   align_matrix = None,):
    setattr(obj,f"{attribute_name}",align_array(
                getattr(obj,f"{attribute_name}"),
        soma_center=soma_center,
        rotation=rotation,
        align_matrix = align_matrix))
    

def rotate_mesh_from_matrix(mesh,matrix):
    new_mesh = mesh.copy()
    new_mesh.vertices = new_mesh.vertices @ matrix
    return new_mesh

    

def align_neuron_obj(
    neuron_obj,
    mesh_center = None,
    rotation = None,
    align_matrix = None,
    in_place = False,
    verbose = False,
    plot_final_neuron = False,
    align_synapses=True,
    **kwargs):
    """
    Purpose: To rotate all of the meshes
    and skeletons of a neuron object
    
    Ex: 
    neuron_obj_rot = copy.deepcopy(neuron_obj)
    mesh_center = neuron_obj["S0"].mesh_center
    for i in range(0,10):
        neuron_obj_rot = align_neuron_obj(neuron_obj_rot,
                                             mesh_center=mesh_center,
                                             verbose =True)
    nviz.visualize_neuron(
        neuron_obj_rot,limb_branch_dict = "all")
    
    
    """
    if not in_place:
        neuron_obj = copy.deepcopy(neuron_obj)

    

    if align_matrix is None:
        if rotation is None:
            if mesh_center is None:
                soma_center = neuron_obj["S0"].mesh_center

            if verbose:
                print(f"soma_center = {soma_center}")

            align_matrix = rotation_from_soma_center(soma_center,
                                     verbose = False,)
            
            
        align_matrix = align_matrix_from_rotation(rotation)
        
    if verbose:
        print(f"align_matrix = {align_matrix}")
        
    neuron_obj.align_matrix = align_matrix
    

    for j,limb_obj in enumerate(neuron_obj):
        for branch_obj in limb_obj:
            branch_obj.mesh = align_mesh(
                                branch_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
            )

            branch_obj.mesh_center = tu.mesh_center_vertex_average(branch_obj.mesh)

            branch_obj.skeleton = align_skeleton(
                                branch_obj.skeleton,
                                align_matrix=align_matrix,
                                verbose = False
            )
            branch_obj.endpoints = align_array(branch_obj.endpoints,
                                                    align_matrix=align_matrix,)
            
            
            
            if align_synapses:
                for syn in branch_obj.synapses:
                    for att in syu.synapse_coordinate_system_dependent_attributes:
                        align_attribute(syn,att,align_matrix=align_matrix,)
                        
            #doing the spine alignment
            if branch_obj.spines is not None:
                branch_obj.spines = [align_mesh(k,align_matrix=align_matrix) for k in branch_obj.spines]
            
            if branch_obj.spines_obj is not None:
                for s_obj in branch_obj.spines_obj:
                    s_obj.mesh = align_mesh(s_obj.mesh,align_matrix=align_matrix)
                
                        
            
        
        #changing the concept network
        all_concept_network_data = []
        att_to_change = ["starting_endpoints","starting_coordinate","touching_soma_vertices"]
        
        for k in limb_obj.all_concept_network_data:
            new_data = copy.deepcopy(k)
            for att in att_to_change:
                new_data[att] = align_array(k[att],align_matrix=align_matrix,)
            all_concept_network_data.append(new_data)
            
        for att in att_to_change:
            setattr(limb_obj,f"current_{att}",align_array(
                getattr(limb_obj,f"current_{att}"),align_matrix=align_matrix,))

        limb_obj.mesh = align_mesh(
                                limb_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
            )
        limb_obj.all_concept_network_data = copy.deepcopy(all_concept_network_data)
        limb_obj.set_concept_network_directional()
        
    neuron_obj.mesh = align_mesh(
                                neuron_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
                                )
#     for s in neuron_obj.get_soma_node_names():
#         neuron_obj[s].mesh = align_mesh(
#                                 neuron_obj[s].mesh,
#                                 align_matrix=align_matrix,
#                                 verbose = False
#                                 )
        
#         neuron_obj[s].mesh_center = align_array(neuron_obj[s].mesh_center,
#                                                     align_matrix=align_matrix,)
        
    #finishing soma mesh stuff
    for s_name in neuron_obj.get_soma_node_names():
        neuron_obj[s_name].mesh = align_mesh(
            neuron_obj[s_name].mesh ,
            align_matrix=align_matrix,
        )
        
        if align_synapses:
            for syn in neuron_obj[s_name].synapses:
                for att in syu.synapse_coordinate_system_dependent_attributes:
                    align_attribute(syn,att,align_matrix=align_matrix,)

        neuron_obj[s_name].mesh_center = tu.mesh_center_vertex_average(neuron_obj[s_name].mesh)
        #print(f"neuron_obj[s_name].mesh_center = {neuron_obj[s_name].mesh_center}")
        
        
    #aligning the preprocessing stuff
    
        
    if plot_final_neuron:
        nviz.visualize_neuron(neuron_obj,limb_branch_dict = "all")
    return neuron_obj

def unalign_neuron_obj(neuron_obj,
                       align_attribute = "align_matrix",
                       verbose = False,
                       plot_final_neuron = False,
                      **kwargs):
    align_matrix = getattr(neuron_obj,align_attribute,None)
    
    if align_matrix is None:
        raise Exception(f"No {align_attribute} found in neuron object")
        
    if align_attribute == "rotation":
        align_matrix = -1*align_matrix
    elif align_attribute == "align_matrix":
        align_matrix = np.linalg.inv(align_matrix)
    else:
        raise Exception(f"Unimplemented align_attribute: {align_attribute}")
    
    if verbose:
        print(f"new {align_attribute} = {align_matrix}")
        
    kwargs[align_attribute] = align_matrix
        
    curr_neuron = align_neuron_obj(neuron_obj,
                               verbose = verbose,
                               plot_final_neuron=plot_final_neuron,
                              **kwargs)
    
    setattr(curr_neuron,align_attribute,None)
    
    return curr_neuron

from . import volume_utils
class DataInterface(volume_utils.DataInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def align_array(self,*args,**kwargs):
        return align_array(*args,**kwargs)
    
    def align_mesh(self,*args,**kwargs):
        return align_mesh(*args,**kwargs)
    
    def align_skeleton(self,*args,**kwargs):
        return align_skeleton(*args,**kwargs)
    
    def align_neuron_obj(self,*args,**kwargs):
        return align_neuron_obj(*args,**kwargs)

    def unalign_neuron_obj(self,*args,**kwargs):
        return unalign_neuron_obj(*args,**kwargs) 
    
    def segment_id_to_synapse_dict(
        self,
        segment_id = None,
        synapse_filepath=None,
        **kwargs
        ):
        
        return super().segment_id_to_synapse_dict(
            synapse_filepath=synapse_filepath,
            segment_id = segment_id,
            **kwargs
        )
        

data_interface = DataInterface(
    source = "h01",
    voxel_to_nm_scaling = voxel_to_nm_scaling
)


#--- from neurd_packages ---
from . import microns_volume_utils as mvu
from . import neuron_visualizations as nviz
from . import synapse_utils as syu
from . import volume_utils

#--- from mesh_tools ---
from mesh_tools import trimesh_utils as tu

#--- from python_tools ---
from python_tools import linalg_utils as lu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu

from . import h01_volume_utils as hvu