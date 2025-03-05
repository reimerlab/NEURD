'''




How to adjust the features on synapses
based on the closest skeleton point

# to calculate the closest skeleton point
syn_coord_sk = sk.closest_skeleton_coordinate(curr_branch.skeleton,
                            face_coord)
#after have closest skeleton coordinate
syu.calculate_endpoints_dist()
syu.calculate_upstream_downstream_dist_from_down_idx(syn,down_idx)






'''
import copy
import operator
import pandas as pd
import time
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu


class Synapse:
    """
    Classs that will hold information about 
    the synapses that will be attributes of a neuron object
    
    Attributes
    ----------
    
    synapse_id: 
    synapse volume
    upstream_dist: skeletal distance from the closest upstream branch point
    downstream_dist: skeletal distance from the closest downstream branch point or endpoint
    coordinate: 3D location in space: 
    closest_sk_coordinate: 3D location in space of closest skeletal point on branch for which synapse is located
    closest_face_coordinate: center coordinate of closest mesh face  on branch for which synapse is located
    closest_face_dist: distance from synapse coordinate to closest_face_coordinate
    soma_distance: skeletal walk distance from synapse to soma
    soma_distance_euclidean: straight path distance from synapse to soma center
    head_neck_shaft: whether the synapse is located on a spine head, spine neck or neurite shaft (decoding of integer label is in spine_utils)
    compartment: the compartment of the branch that the synapse is located on
    limb_idx: the limb identifier that the synapse is located on
    branch_idx: the branch identifier that the synapse is located on

    Note: features like head_neck_shaft, compartment are not populated until later stages (cell typing, autoproofreading) when that information is available for the branches
    """
    def __init__(self,synapse_obj=None,**kwargs):
        for a in synapse_attributes:
            setattr(self,a,None)
        
        if synapse_obj is not None:
            kwargs.update(synapse_obj.export())
        #if synapse_dict is not None:
        for k,v in kwargs.items():
            if k in synapse_attributes:
                setattr(self,k,v)
                
    def export(self):
        return syu.export(self)
    


    
synapse_attributes = ["syn_type",
                      "syn_id",
                     "volume",
                     "endpoints_dist",
                      "upstream_dist",
                      "downstream_dist",
                     "coordinate",
                     "closest_sk_coordinate",
                     "closest_face_idx",
                      "closest_branch_face_idx",
                     "closest_face_dist",
                      "closest_face_coordinate",
                      "soma_distance",
                      "soma_distance_euclidean",
                      "head_neck_shaft",
                      "compartment",
                      "limb_idx",
                      "branch_idx"
                     ]
synapse_coordinate_system_dependent_attributes = [
    "closest_sk_coordinate",
    "coordinate",
    "closest_face_coordinate"
]


#soma_synapse_offset = nru.soma_face_offset

synapse_error_types = ["distance_errored","mesh_errored"]

synapse_type_names = dict(presyn=dict(valid="valid_syn_centers_presyn",
                                     error="errored_syn_centers_presyn"),
                         postsyn = dict(valid="valid_syn_centers_postsyn",
                                       error = "errored_syn_centers_postsyn"))


valid_pre_color = "yellow"
valid_post_color = "blue"
error_pre_color = "black"
error_post_color = "orange"
distance_errored_synapses_pre_color = "tan"
distance_errored_synapses_post_color = "pink"
mesh_errored_synapses_pre_color = "brown"
mesh_errored_synapses_post_color = "lime"
soma_pre_color = "aqua"
soma_post_color = "purple"

default_synapse_size = 0.3

synapse_types = dict(
    soma="synapses_somas",
    limb_branch="synapses",
    mesh_errored="mesh_errored_synapses",
    distance_errored="distance_errored_synapses"
)






def get_synapse_types():
    return list(synapse_types.values())

# ------ different queries for synapses ------- #
presyns_on_dendrite_as_errors = True

def set_presyns_on_dendrite_as_errors_default():
    print(f"set_presyns_on_dendrite_as_errors to default of True")
    presyns_on_dendrite_as_errors = True
    
def set_presyns_on_dendrite_as_errors(value):
    print(f"set_presyns_on_dendrite_as_errors to {value}")
    presyns_on_dendrite_as_errors = value


def error_query():
    #query = f"(compartment=='error')"
    query = f"(soma_distance == -1) or (compartment == 'error') or (label in ['mesh_errored','distance_errored'])"
    if presyns_on_dendrite_as_errors:
        query += f" or {presyns_on_dendrite_query}"
    return query

def valid_query():
    #query = f"(compartment !='error')"
    query = f"not ({error_query()})"
    if presyns_on_dendrite_as_errors:
        query += f" and not({presyns_on_dendrite_query})"
    return query
    





    
    
# ------- Synapse Plotting ------------- #




def plot_valid_error_synpases(neuron_obj = None,
                             synapse_dict=None,
                            mesh = None,
                            original_mesh = None,
                             synapses_type_to_plot = None,
                             synapses_type_to_not_plot = None,
                             keyword_to_plot = None,
                             verbose=False,
                             TP_color="yellow",
                            TN_color="aqua",
                            FP_color="black",
                            FN_color="orange",
                             synapse_scatter_size = 0.3,
                             
                             #for plotting the actual mesh parts to go along
                             plot_only_axon_skeleton = True,
                             error_mesh_color = "red",
                             valid_mesh_color = "green",
                             valid_skeleton_color = "black",
                             mesh_alpha = 0.3,
                             print_color_key = True,
#                              mapping = dict(TP= "valid_syn_centers_presyn",
#                                               FP = "errored_syn_centers_presyn",
#                                               TN = "valid_syn_centers_postsyn",
#                                               FN = "errored_syn_centers_postsyn",),
                            mapping = None,
                            **kwargs):
    """
    Purpose: Will plot the synapse centers against 
    a proofread neuron
    
    Ex: 
    output_syn_dict = syu.synapse_dict_mesh_labels_to_synapse_coordinate_dict(synapse_mesh_labels_dict=mesh_label_dict,
                                                           synapse_dict=synapse_dict)
    syu.plot_valid_error_synpases(
                             synapse_dict=output_syn_dict,
                             neuron_obj = None,
                            mesh = mesh,
                            original_mesh = original_mesh,
                            keyword_to_plot = "error",
        synapse_scatter_size=2,
    )
    """
    
    if synapse_dict is None:
        synapse_dict = syu.synapse_pre_post_valid_errror_coordinates_dict(neuron_obj)

    if synapses_type_to_plot is not None:

        new_dict = dict(synapse_dict)
        for k in synapse_dict.keys():
            if k not in synapses_type_to_plot:
                del new_dict[k]

        synapse_dict = new_dict

    if synapses_type_to_not_plot is not None:

        new_dict = dict(synapse_dict)
        for k in synapse_dict.keys():
            if k in synapses_type_to_not_plot:
                del new_dict[k]
        synapse_dict = new_dict
        
    if keyword_to_plot is not None:
        new_dict = dict(synapse_dict)
        for k in synapse_dict.keys():
            if keyword_to_plot not in k:
                del new_dict[k]

        synapse_dict = new_dict

    
    
    if mapping is not None:
        synapse_types = ["valid_syn_centers_presyn","errored_syn_centers_presyn",
                    "valid_syn_centers_postsyn","errored_syn_centers_postsyn"]
        synapse_dict_pre = {k:np.array([]) for k in synapse_types}
        for k,v in synapse_dict.items():
            synapse_dict_pre[mapping[k]] = v
        synapse_dict = synapse_dict_pre

    color_dict = dict(TP_color=TP_color,
                        TN_color=TN_color,
                        FP_color=FP_color,
                        FN_color=FN_color,)

    # ---- Part B: Prepares the Mesh Part of the Visual --------#

    
    """
    Purpose: Plot the valid faces and the invalid faces of neuron
    (including maybe the axon skeleton of the valid portion), and then to plot the synapses

    Pseudocode: 
    1) Get the error mesh from the original mesh and then the mesh after proofreading
    2) Get the valid skeleton
    """
    if neuron_obj is not None:
        print(f"Using the mesh from the neuron object")
        import validation_utils as vu
        
        error_mesh, valid_mesh = vu.mesh_errored_after_neuron_proofreading(neuron_obj,return_valid_mesh=True)
        if plot_only_axon_skeleton:
            valid_skeleton = neuron_obj.axon_skeleton
        else:
            valid_skeleton = neuron_obj.skeleton
        skeletons = [valid_skeleton]
        
    else:
        valid_mesh = mesh
        if original_mesh is not None:
            error_mesh = tu.subtract_mesh(original_mesh,valid_mesh)
        skeletons = []
        

    meshes = [error_mesh,valid_mesh]
    meshes_colors = [error_mesh_color,valid_mesh_color]
    
    skeletons_colors = [valid_skeleton_color]
    
    #print(f"synapse_dict = {synapse_dict}")
    nviz.plot_valid_error_synapses(neuron_obj=None,
                           synapse_dict =synapse_dict,
                           meshes=meshes,
                           meshes_colors=meshes_colors,
                           synapse_scatter_size=synapse_scatter_size,

#                                 scatters=scatters,
#                                 scatter_size=scatter_size,
#                                    scatters_colors=scatters_colors,

                          valid_presyn_color=TP_color,
                        valid_postsyn_color=TN_color,
                        error_presyn_color=FP_color,
                        error_postsyn_color=FN_color,

                            plot_error_synapses=True,

                            skeletons=skeletons,
                              skeletons_colors=skeletons_colors, 
                            mesh_alpha=mesh_alpha,
                               **kwargs
                          )
    if print_color_key:
        curr_colors = [TP_color,TN_color,FP_color,FN_color]
        curr_types = ["valid_presyn_color","valid_postsyn_color","error_presyn_color","error_postsyn_color"]
        #print(f"\nColor Key:")
        for c_type,col in zip(curr_types,curr_colors):
            print(f"{c_type}:{col}")




def soma_synapses_to_scatter_info(neuron_obj,
                             pre_color=soma_pre_color,
                             post_color = soma_post_color,
                                 scatter_size = default_synapse_size):
    """
    Purpose: To Turn the soma synapses into plottable scatters
    
    Ex: syu.soma_synapses_to_scatter_info(neuron_obj)
    """
    scatters = []
    scatters_colors = []
    scatter_size_list = []
    
    for soma_name in neuron_obj.get_soma_node_names():
        soma_syns_pre = neuron_obj[soma_name].synapses_pre
        soma_syns_post = neuron_obj[soma_name].synapses_post
        
        scatters.append(syu.synapses_to_coordinates(soma_syns_pre))
        scatters.append(syu.synapses_to_coordinates(soma_syns_post))
        
        scatters_colors += [pre_color,post_color]
        scatter_size_list += [scatter_size,scatter_size]
        
    return scatters,scatters_colors,scatter_size_list



def error_synapses_to_scatter_info(neuron_obj,
                                   error_synapses_names = None,
                                  pre_color=error_pre_color,
                                 post_color = error_post_color,
                                   color_mapping=dict(distance_errored_synapses_pre = distance_errored_synapses_pre_color,
                                                     distance_errored_synapses_post = distance_errored_synapses_post_color,
                                                     mesh_errored_synapses_pre = mesh_errored_synapses_pre_color,
                                                     mesh_errored_synapses_post = mesh_errored_synapses_post_color,),
                                     scatter_size = default_synapse_size):
    """
    To turn the error synapses into plottable scatters
    
    Ex: syu.error_synapses_to_scatter_info(neuron_obj)
    """
    
    scatters = []
    scatters_colors = []
    scatter_size_list = []
    
    default_color_mapping = dict(pre=pre_color,
                                post=post_color)
    
    if error_synapses_names is None:
        error_synapses_names = syu.get_errored_synapses_names(neuron_obj)
    for err_syn in error_synapses_names:
        for syn_type in ["pre","post"]:
            
            err_syn_type = f"{err_syn}_{syn_type}"
            if err_syn_type in color_mapping.keys():
                curr_color = color_mapping[err_syn_type]
            else:
                curr_color = default_color_mapping[syn_type]
            
            curr_synapses = getattr(neuron_obj,err_syn_type)
            
            scatters.append(syu.synapses_to_coordinates(curr_synapses))
            scatters_colors.append(curr_color)
            scatter_size_list.append(scatter_size)
            
    return scatters,scatters_colors,scatter_size_list

def limb_branch_synapses_to_scatter_info(neuron_obj,
                                        limb_branch_dict="all",
                                        pre_color = valid_pre_color,
                                        post_color = valid_post_color,
                                        scatter_size = default_synapse_size,
                                        synapse_type = "synapses"):
    """
    Purpose: To make the synapses on the limb and branches
    plottable
    
    Ex: limb_branch_synapses_to_scatter_info(neuron_obj)
    """
    if limb_branch_dict == "all":
        limb_branch_dict = neuron_obj.limb_branch_dict
    
    pre_syn = nru.concatenate_feature_over_limb_branch_dict(neuron_obj,
                                                 limb_branch_dict=limb_branch_dict,
                                                 feature="synapses_pre")
    post_syn = nru.concatenate_feature_over_limb_branch_dict(neuron_obj,
                                                 limb_branch_dict=limb_branch_dict,
                                                 feature="synapses_post")
    if synapse_type == "synapses":
        scatters = [syu.synapses_to_coordinates(pre_syn),
                    syu.synapses_to_coordinates(post_syn)]
        scatters_colors = [pre_color,post_color]
        scatter_size_list = [scatter_size,scatter_size]
    elif synapse_type == "synapses_pre":
        scatters = [syu.synapses_to_coordinates(pre_syn)]
        scatters_colors = [pre_color]
        scatter_size_list = [scatter_size]
    elif synapse_type == "synapses_post":
        scatters = [syu.synapses_to_coordinates(post_syn)]
        scatters_colors = [post_color]
        scatter_size_list = [scatter_size]
    else:
        raise Exception(f"Unknown synapse type = {synapse_type}")
    
    return scatters,scatters_colors,scatter_size_list


def append_synapses_to_plot(
    neuron_obj,
    total_synapses = False,
    total_synapses_size = default_synapse_size,

    limb_branch_dict = "all",
    limb_branch_synapses = False,
    limb_branch_size = default_synapse_size,
    limb_branch_synapse_type = "synapses",

    distance_errored_synapses = False,
    distance_errored_size = default_synapse_size,

    mesh_errored_synapses = False,
    mesh_errored_size = default_synapse_size,

    soma_synapses = False,
    soma_size = default_synapse_size,


    return_plottable = False,
    append_figure = True,
    show_at_end = False,
    verbose = False):
    """
    Purpose: To add synapse scatter plots
    to an existing plot

    """

    scatters = []
    scatters_colors = []
    scatter_size_list = []




    if total_synapses:
        limb_branch_size = total_synapses_size
        distance_errored_size = total_synapses_size
        mesh_errored_size = total_synapses_size
        soma_size = total_synapses_size
        
        limb_branch_synapses = True
        distance_errored_size = True
        mesh_errored_synapses = True
        soma_synapses = True
        
        limb_branch_dict = "all"


    if limb_branch_synapses:
        curr_sc,curr_c,curr_sz = syu.limb_branch_synapses_to_scatter_info(neuron_obj,
                                                                         scatter_size = limb_branch_size,
                                                                         limb_branch_dict=limb_branch_dict,
                                                                         synapse_type=limb_branch_synapse_type)
        
        scatters+= curr_sc
        scatters_colors+=curr_c
        scatter_size_list += curr_sz

    if mesh_errored_synapses:
        curr_sc,curr_c,curr_sz = syu.error_synapses_to_scatter_info(neuron_obj,
                                                                    error_synapses_names=["mesh_errored_synapses"],
                                                                         scatter_size = mesh_errored_size)
        scatters+= curr_sc
        scatters_colors+=curr_c
        scatter_size_list += curr_sz

    if distance_errored_synapses:
        curr_sc,curr_c,curr_sz = syu.error_synapses_to_scatter_info(neuron_obj,
                                                                    error_synapses_names=["distance_errored_synapses"],
                                                                         scatter_size = distance_errored_size)
        scatters+= curr_sc
        scatters_colors+=curr_c
        scatter_size_list += curr_sz

    if soma_synapses:
        curr_sc,curr_c,curr_sz = syu.soma_synapses_to_scatter_info(neuron_obj,
                                                                         scatter_size = soma_size)
        scatters+= curr_sc
        scatters_colors+=curr_c
        scatter_size_list += curr_sz
        
    scatters = [np.array(k).reshape(-1,3) for k in scatters]

    if verbose:
        print(f"scatters = {scatters}")
        print(f"scatters_colors= {scatters_colors}")
        print(f"scatter_size_list = {scatter_size_list}")

    if len(scatters) == 0 or len(np.vstack(scatters)) == 0:
        if verbose:
            print(f"No Synapses to plot")
    else:
        nviz.plot_objects(scatters=scatters,
                         scatters_colors=scatters_colors,
                         scatter_size=scatter_size_list,
                         append_figure=append_figure,
                         show_at_end=show_at_end)
        
    if return_plottable:
        return [scatters,scatter_size_list,scatter_size_list]

def plot_synapses(
    neuron_obj,
    synapse_type = "synapses",
    total_synapses=False,
    limb_branch_size = default_synapse_size,
    distance_errored_size = default_synapse_size,
    mesh_errored_size = default_synapse_size,
    soma_size = default_synapse_size,
    **kwargs
    ):
    """
    The synapse types
    
    """
    nviz.visualize_neuron(
        neuron_obj,
        limb_branch_dict="all",#dict(L2="all"),
        limb_branch_synapses=True,
        limb_branch_synapse_type = synapse_type,
        total_synapses=total_synapses,
        limb_branch_size = limb_branch_size,
        distance_errored_size = distance_errored_size,
        mesh_errored_size = mesh_errored_size,
        soma_size = soma_size,
        **kwargs
        )
# --------- End of Synapse Plotting -------#
                    
def export(synapse_obj):
    return {k:getattr(synapse_obj,k,None) for k in synapse_attributes}
                    
    
       
 
def combine_synapse_dict_into_presyn_postsyn_valid_error_dict(synapse_dict,
                                                             verbose = False):
    """
    Purpose: To concatenate all of the valid and error synapses
    into one synapse dict (application: which can eventually be plotted)

    Pseudocode: 
    1) iterate through presyn,postsyn
        2) iterate through error valid
           Find all the keys that have the following in the name
           Concatenate the lists
           Store 

    """
    output_dict = dict()
    for k in ["presyn","postsyn"]:
        synapse_info = synapse_dict[k]
        synapse_info_keys = list(synapse_info.keys())
        for t in ["valid","error"]:
            curr_name = synapse_type_names[k][t]
            curr_keys = [j for j in synapse_info_keys if t in j]
            if verbose:
                print(f"\nFor {curr_name} using keys {curr_keys}")
                
            if len(curr_keys) > 0:
                curr_arrays = np.vstack([np.array(synapse_info[h]) for h in curr_keys])
            else:
                curr_arrays = np.array([])
                
            if verbose:
                print(f"# of coordinates = {len(curr_arrays)}")
                
            output_dict[curr_name] = curr_arrays
            
    return output_dict


def synapse_dict_mesh_labels_to_synapse_attribute_dict(synapse_mesh_labels_dict,
                                                        synapse_dict,
                                                       attribute,
                                                      return_presyn_postsyn_valid_error_dict = False ):
    coord_dict = dict()
    for synapse_type,syn_info in synapse_mesh_labels_dict.items():
        coord_dict[synapse_type] = dict()
        for l,l_ids in syn_info.items():
            current_indices = nu.intersect_indices(synapse_dict[synapse_type]["synapse_ids"],l_ids)
            coord_dict[synapse_type][l] = synapse_dict[synapse_type][attribute][current_indices]
            
    if return_presyn_postsyn_valid_error_dict:
        coord_dict = syu.combine_synapse_dict_into_presyn_postsyn_valid_error_dict(coord_dict,
                                                          verbose = False)
    return coord_dict

def synapse_dict_mesh_labels_to_synapse_coordinate_dict(synapse_mesh_labels_dict,
                                                        synapse_dict,
                                                      return_presyn_postsyn_valid_error_dict = True ):
    return synapse_dict_mesh_labels_to_synapse_attribute_dict(synapse_mesh_labels_dict,
                                                        synapse_dict,
                                                       attribute = "synapse_coordinates",
                                return_presyn_postsyn_valid_error_dict = return_presyn_postsyn_valid_error_dict )
    
def synapse_dict_mesh_labels_to_synapse_volume_dict(synapse_mesh_labels_dict,
                                                        synapse_dict,
                                                      return_presyn_postsyn_valid_error_dict = True ):
    return synapse_dict_mesh_labels_to_synapse_attribute_dict(synapse_mesh_labels_dict,
                                                        synapse_dict,
                                                       attribute = "synapse_sizes",
                                return_presyn_postsyn_valid_error_dict = return_presyn_postsyn_valid_error_dict )



            



def n_synapses(neuron_obj):
    if type(neuron_obj) != list:
        synapses = neuron_obj.synapses
    else:
        synapses = neuron_obj
    return len(synapses)

def synapse_density(neuron_obj,synapses=None,density_type = "skeletal_length"):
    if synapses is None:
        synapses = neuron_obj.synapses
        
    
    skeletal_length = getattr(neuron_obj,density_type)
    if skeletal_length > 0:
        spine_density = len(synapses)/skeletal_length
    else:
        spine_density = 0
    return spine_density

def synapses_pre(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="syn_type",
                                    comparison_value="presyn")
def synapses(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
        
    return curr_synapses

def synapses_post(neuron_obj=None):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="syn_type",
                                    comparison_value="postsyn")

def n_synapses_pre(neuron_obj):
    return len(syu.synapses_pre(neuron_obj))

def n_synapses_post(neuron_obj):
    return len(syu.synapses_post(neuron_obj))

def synapse_density_pre(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_pre(neuron_obj),
                          density_type=density_type)
def synapse_density_post(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_post(neuron_obj),
                          density_type=density_type)
def synapse_pre_perc(neuron_obj):
    total_synapses = syu.n_synapses(neuron_obj)
    if total_synapses > 0:
        return syu.n_synapses_pre(neuron_obj)/total_synapses
    else:
        return 0
    

    
def synapse_post_perc(neuron_obj):
    total_synapses = syu.n_synapses(neuron_obj)
    if total_synapses > 0:
        return syu.n_synapses_post(neuron_obj)/total_synapses
    else:
        return 0
    
    
# -------- 7/19: Has all of the head_neck_shaft classifications ----- #
def synapses_head(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["head"])
def synapses_neck(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["neck"])
def synapses_shaft(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["shaft"])

def synapses_no_head(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["no_head"])

def synapses_non_bouton(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["non_bouton"])

def synapses_bouton(neuron_obj):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
    return syu.synapses_with_feature(curr_synapses,
                                    feature_name="head_neck_shaft",
                                    comparison_value=spu.head_neck_shaft_dict["bouton"])

def synapses_type_and_head_neck_shaft(neuron_obj,
                                    syn_type, 
                                     head_neck_shaft_type):
    if type(neuron_obj) != list:
        curr_synapses = neuron_obj.synapses
    else:
        curr_synapses = neuron_obj
        
    if len(curr_synapses) == 0:
        return []
    
    return syu.query_synapses(curr_synapses,
                       query=(f"(syn_type == '{syn_type}') and "
                             f"(head_neck_shaft == {spu.head_neck_shaft_dict[head_neck_shaft_type]})"),
                       return_synapses=True
                      )

def synapses_post_head(neuron_obj):
    return synapses_type_and_head_neck_shaft(neuron_obj,"postsyn","head")

def synapses_post_neck(neuron_obj):
    return synapses_type_and_head_neck_shaft(neuron_obj,"postsyn","neck")

def synapses_post_shaft(neuron_obj):
    return synapses_type_and_head_neck_shaft(neuron_obj,"postsyn","shaft")

def synapses_post_no_head(neuron_obj):
    return synapses_type_and_head_neck_shaft(neuron_obj,"postsyn","no_head")

def synapses_post_spine(neuron_obj):
    return synapses_post_neck(neuron_obj) + synapses_post_head(neuron_obj) + synapses_post_no_head(neuron_obj)

def synapses_pre_shaft(neuron_obj):
    return synapses_type_and_head_neck_shaft(neuron_obj,"presyn","shaft")

def synapses_spine(neuron_obj):
    return syu.synapses_head(neuron_obj) + syu.synapses_neck(neuron_obj) + syu.synapses_no_head(neuron_obj)

def n_synapses_head(neuron_obj):
    return len(syu.synapses_head(neuron_obj))
def n_synapses_neck(neuron_obj):
    return len(syu.synapses_neck(neuron_obj))
def n_synapses_shaft(neuron_obj):
    return len(syu.synapses_shaft(neuron_obj))

def n_synapses_no_head(neuron_obj):
    return len(syu.synapses_no_head(neuron_obj))

def n_synapses_spine(neuron_obj):
    return len(syu.synapses_spine(neuron_obj))

def n_synapses_post_head(neuron_obj):
    return len(syu.synapses_post_head(neuron_obj))

def n_synapses_post_neck(neuron_obj):
    return len(syu.synapses_post_head(neuron_obj))

def n_synapses_post_no_head(neuron_obj):
    return len(syu.synapses_post_no_head(neuron_obj))

def n_synapses_post_spine(neuron_obj):
    return len(syu.synapses_post_spine(neuron_obj))

def n_synapses_post_shaft(neuron_obj):
    return len(syu.synapses_post_shaft(neuron_obj))

def n_synapses_pre_shaft(neuron_obj):
    return len(syu.synapses_pre_shaft(neuron_obj))


def synapse_density_head(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_head(neuron_obj),density_type=density_type)
def synapse_density_neck(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_neck(neuron_obj),density_type=density_type)
def synapse_density_shaft(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_shaft(neuron_obj))
def synapse_density_no_head(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_no_head(neuron_obj),density_type=density_type)

def synapse_density_spine(neuron_obj,density_type = "skeletal_length"):
    return synapse_density(neuron_obj,
                          synapses=syu.synapses_spine(neuron_obj),density_type=density_type)

def synapse_head_perc(neuron_obj):
    total_synapses = syu.n_synapses(neuron_obj)
    if total_synapses > 0:
        return syu.n_synapses_head(neuron_obj)/total_synapses
    else:
        return 0
    
def synapse_spine_perc(neuron_obj):
    total_synapses = syu.n_synapses(neuron_obj)
    if total_synapses > 0:
        return syu.n_synapses_spine(neuron_obj)/total_synapses
    else:
        return 0


# --------- End of head neck shaft --------------

def synapses_with_feature(synapses,
                         feature_name,
                          comparison_value,
                         operator_func=operator.eq,
                          verbose = False
                         ):
    """
    Purpose: Will find synapses with a certain feature
    
    Possible operators:
    operator.eq
    operator.gt
    operator.ge
    operator.lt
    operator.le
    
    Ex: 
    synapses_with_feature(neuron_obj.synapses,
                     feature_name="syn_type",
                     comparison_value="postsyn",
                     verbose=True)
                     
    Ex: 
    import operator
    syu.calculate_neuron_soma_distance(neuron_obj)
    syn_from_soma = syu.synapses_with_feature(neuron_obj.synapses,
                              feature_name = "soma_distance",
                              comparison_value = 4000,
                              operator_func=operator.le,
                              verbose = True)

    syn_coords = syu.synapses_to_coordinates(syn_from_soma)

    nviz.plot_objects(neuron_obj.mesh,
                     scatters=[syn_coords])
    """
    if not nu.is_array_like(feature_name):
        feature_name = [feature_name]
    if not nu.is_array_like(comparison_value):
        comparison_value = [comparison_value]
    if not nu.is_array_like(operator_func):
        operator_func = [operator_func]
    
    match_synapses = synapses
    for feat,cv,of in zip(feature_name,comparison_value,operator_func):
        match_synapses = [k for k in match_synapses if 
                              of(getattr(k,feat),cv)]
    
        if verbose:
            print(f"# of match_synapses with {feature_name} {operator_func} to value {comparison_value} = {len(match_synapses)}")
        
    return match_synapses



def exports_to_synapses(exports):
    if exports is None:
        return None
    return_list = []
    for j in exports:
        if type(j) == dict:
            return_list.append(syu.Synapse(**j))
        elif j.__class__ == syu.Synapse().__class__:
            return_list.append(j)
        else:
            raise Exception(f"Unknown type: {type(j)}")
    return return_list

def synapses_to_exports(synapses):
    return [k.export() for k in synapses]

def calculate_limb_synapse_soma_distances(
    limb_obj,
    calculate_endpoints_dist_if_empty=False,
    verbose=False):
    """
    Purpose: To store the distances to the soma 
    for all of the synapses

    Computing the upstream soma distance
    for each branch
    1) calculate the upstream distance
    2) Calcualte the upstream endpoint
        For each synapse:
        3) Soma distance = endpoint_dist
        
    Ex: 
    calculate_limb_synapse_soma_distances(limb_obj = neuron_obj[2],
    verbose = True)

    """

    for branch_idx in limb_obj.get_branch_names():
        branch_obj = limb_obj[branch_idx]

        #1) Calculate the upstream distance
        upstream_dist = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        upstream_endpoint_idx = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=True)

        for syn in branch_obj.synapses:
            endpoint_dist = syn.endpoints_dist[upstream_endpoint_idx]
            if endpoint_dist == -1:
                if calculate_endpoints_dist_if_empty:
                    syu.calculate_endpoints_dist(branch_obj,syn)
                    syu.calculate_upstream_downstream_dist(limb_obj,branch_idx,syn)
                    #endpoint_dist = syn.endpoints_dist[upstream_endpoint_idx]
                    endpoint_dist = syn.upstream_dist
                else:
                    raise Exception("Endpoint distance was not calculated yet and calculate_endpoint_dist_if_empty not set")
            syn.soma_distance = endpoint_dist + upstream_dist
            
def calculate_endpoints_dist(branch_obj,syn):
    """
    Purpose: Will calculate the endpoint distance for a synapse
    """
    bau.calculate_endpoints_dist(branch_obj,syn)
    

def calculate_upstream_downstream_dist(limb_obj,
                                       branch_idx,
                                       syn):
    bau.calculate_upstream_downstream_dist(limb_ob,branch_idx,syn)
    
def calculate_upstream_downstream_dist_from_down_idx(
    syn,down_idx):
    bau.calculate_upstream_downstream_dist_from_down_idx(syn,down_idx)

def calculate_upstream_downstream_dist_from_up_idx(
    syn,up_idx):
    bau.calculate_upstream_downstream_dist_from_up_idx(syn,up_idx)
    
def calculate_neuron_soma_distance(neuron_obj,
                                  verbose  =False,
                                  store_soma_placeholder = True,
                                  store_error_placeholder = True):
    """
    Purpose: To calculate all of the soma distances for all the valid synapses
    on limbs
    
    Ex: 
    calculate_neuron_soma_distance(neuron_obj,
                              verbose = True)
    """
    st = time.time()
    for limb_name in neuron_obj.get_limb_names():
        st_loc = time.time()
        
        limb_obj = neuron_obj[limb_name]
        syu.calculate_limb_synapse_soma_distances(limb_obj = limb_obj,
            verbose = False)
        if verbose:
            print(f"\n--- Limb {limb_name} soma calculation time = {np.round(time.time() - st_loc,3)}")
    
        
    if store_soma_placeholder:
        if verbose:
            print(f"Putting Soma Placeholders")
        for s_idx in neuron_obj.get_soma_indexes():
            st_loc = time.time()
            soma_synapses = neuron_obj[f"S{s_idx}"].synapses
            for syn in soma_synapses:
                syn.soma_distance = -1*(nru.soma_face_offset + s_idx)
            if verbose:
                print(f"\n--- Soma {s_idx} soma calculation time = {np.round(time.time() - st_loc,3)}")
                
    if store_error_placeholder:
        st_loc = time.time()
        if verbose:
            print(f"Putting Error Placeholders")
        for syn in neuron_obj.synapses_error:
            syn.soma_distance = -1
        if verbose:
            print(f"\n--- Error soma calculation time = {np.round(time.time() - st_loc,3)}")
                
                
    if verbose:
        print(f"Total soma distance calculation time = {time.time() - st}")
        

        
# -------------- 6/9 For Applying the synapses ------------ #

def add_valid_soma_synapses_to_neuron_obj(neuron_obj,
                                          verbose=False,
                                          validation=False,
                                          **kwargs):
    return syu.add_valid_synapses_to_neuron_obj(neuron_obj,
                                               verbose=verbose,
                                                validation=validation,
                                                add_only_soma_synapses = True,
                                                **kwargs
                                               )
def add_valid_synapses_to_neuron_obj(neuron_obj,
                                    synapse_dict=None,
                                    mesh_label_dict=None,
                                     validation = False,
                                    verbose = False,
                                    debug_time = True,
                                    calualate_endpoints_dist = True,
                                     limb_branch_dict_to_add_synapses = None,
                                     #set_head_neck_shaft = False,
                                     original_mesh = None,
                                     add_only_soma_synapses = False,
                                    **kwargs):
    """
    Purpose: To add valid synapses to a neuron object
    
    """
    
    # ----- Phase 0: Getting the Synapse Info ------ #
    if synapse_dict is None:
        synapse_dict = vdi.segment_id_to_synapse_dict(neuron_obj.segment_id,
                                                 validation=validation,
                                                 verbose=verbose)
        
    if mesh_label_dict is None:
        mesh_label_dict = syu.fetch_synapse_dict_by_mesh_labels(
            mesh = nru.neuron_mesh_from_branches(neuron_obj),
            segment_id=neuron_obj.segment_id,
            synapse_dict = synapse_dict,
            original_mesh = original_mesh,
            #original_mesh_kd=original_mesh_kd,
            validation=validation,
            plot_synapses=False,
            verbose = verbose)
    



    # ----- Phase 1: Computing the features of the Synapses ------ #
    st = time.time()

    valid_synapse_dict = dict(presyn=dict(valid=mesh_label_dict["presyn"]["valid"]),
                              postsyn=dict(valid=mesh_label_dict["postsyn"]["valid"]))
    
    
        
    
    valid_synapse_dict_coord = syu.synapse_dict_mesh_labels_to_synapse_coordinate_dict(valid_synapse_dict,
                                                            synapse_dict,
                                                            return_presyn_postsyn_valid_error_dict = False)
    valid_synapse_dict_volume = syu.synapse_dict_mesh_labels_to_synapse_volume_dict(valid_synapse_dict,
                                                            synapse_dict,
                                                            return_presyn_postsyn_valid_error_dict = False)

    if debug_time:
        print(f"Synapse dict: {np.round(time.time() - st,4)}")
        st = time.time()

    original_mesh_proof = neuron_obj.mesh_from_branches
    original_mesh_kdtree_proof = tu.mesh_to_kdtree(original_mesh_proof)

    if debug_time:
        print(f"Original Mesh: {np.round(time.time() - st,4)}")
        st = time.time()

    limb_branch_info = dict()
    for k in ["presyn","postsyn"]:
        limb_branch_info[k] = dict()
        limb_branch_idx,dist,closest_faces = nru.coordinates_to_closest_limb_branch(neuron_obj,
                                               coordinates=valid_synapse_dict_coord[k]["valid"],
                                                original_mesh = original_mesh_proof,
                                                original_mesh_kdtree = original_mesh_kdtree_proof,
                                               return_distances_to_limb_branch = True,
                                              return_closest_faces=True)
        if add_only_soma_synapses:
            if verbose:
                print(f"Restricting to only soma synapses")
                
            keep_map = limb_branch_idx[:,0] < -1
            valid_synapse_dict[k]["valid"] = valid_synapse_dict[k]["valid"][keep_map]
            valid_synapse_dict_coord[k]["valid"] = valid_synapse_dict_coord[k]["valid"][keep_map]
            valid_synapse_dict_volume[k]["valid"] = valid_synapse_dict_volume[k]["valid"][keep_map]
            limb_branch_idx = np.array(limb_branch_idx)[keep_map]
            dist = np.array(dist)[keep_map]
            closest_faces = np.array(closest_faces)[keep_map]
            
        limb_branch_info[k]["limb_idx"] = limb_branch_idx[:,0].astype("int")
        limb_branch_info[k]["branch_idx"] = limb_branch_idx[:,1].astype("int")
        limb_branch_info[k]["closest_face_dist"] = dist
        limb_branch_info[k]["closest_face_idx"] = closest_faces
        

        limb_branch_info[k]["closest_face_coordinate"] = original_mesh_proof.triangles_center[closest_faces]

        if debug_time:
            print(f"Closest Branch: {np.round(time.time() - st,4)}")
            st = time.time()

        #computing the closest skeleton vertex and endpoint distance
        limb_branch_info[k]["closest_sk_coordinate"] = np.zeros(limb_branch_info[k]["closest_face_coordinate"].shape)
        limb_branch_info[k]["endpoints_dist"] = np.ones((limb_branch_info[k]["closest_face_coordinate"].shape[0],2))*-1
        limb_branch_info[k]["upstream_dist"] = np.ones((limb_branch_info[k]["closest_face_coordinate"].shape[0]))*-1
        limb_branch_info[k]["downstream_dist"] = np.ones((limb_branch_info[k]["closest_face_coordinate"].shape[0]))*-1

        
        downstream_endpoint_idx_dict = dict()
        for j,(limb_idx,branch_idx,face_coord,curr_face_idx) in enumerate(zip(limb_branch_info[k]["limb_idx"],
                                                  limb_branch_info[k]["branch_idx"],
                                                  limb_branch_info[k]["closest_face_coordinate"],
                                                limb_branch_info[k]["closest_face_idx"])):


            if limb_idx < 0 or branch_idx < 0:
                syn_coord_sk = face_coord
                endpoints_dist = [0,0]
                upstream_dist = 0
                downstream_dist = 0
#                 if set_head_neck_shaft:
#                     head_neck_shaft_val = spu.head_neck_shaft_dict["no_label"]

            else:
                if limb_branch_dict_to_add_synapses is not None:
                    if not nru.in_limb_branch_dict(limb_branch_dict_to_add_synapses,limb_idx,branch_idx):
                        continue
                
                curr_branch = neuron_obj[limb_idx][branch_idx]
                
#                 if set_head_neck_shaft:
#                     head_neck_shaft_val = curr_branch.head_neck_shaft_idx[curr_face_idx]
                
                
                syn_coord_sk = sk.closest_skeleton_coordinate(curr_branch.skeleton,
                            face_coord)
                #getting the distances to endpoint1 and endpoint2

                if calualate_endpoints_dist:
                    endpoints_dist = [sk.skeleton_path_between_skeleton_coordinates(
                                starting_coordinate = syn_coord_sk,
                                destination_node=j,
                                skeleton_graph = curr_branch.skeleton_graph,
                                only_skeleton_distance = True,) for j in curr_branch.endpoints_nodes]
    #                 if debug_time:
    #                     print(f"endpoints_dist {limb_idx},{branch_idx}: {np.round(time.time() - st,4)}")
    #                     st = time.time()
                    """
                    Need to figure out which index is upstream and which is downstream
                    """


                    if limb_idx not in downstream_endpoint_idx_dict.keys():
                        downstream_endpoint_idx_dict[limb_idx] = dict()
            
                    if branch_idx not in downstream_endpoint_idx_dict[limb_idx].keys():
                        downstream_endpoint_idx_dict[limb_idx][branch_idx] = nru.downstream_endpoint(neuron_obj[limb_idx],branch_idx,return_endpoint_index=True)
                        
                    down_idx = downstream_endpoint_idx_dict[limb_idx][branch_idx]
                    downstream_dist = endpoints_dist[down_idx]
                    upstream_dist = endpoints_dist[1-down_idx]
    
                    
                else:
                    upstream_dist = -1
                    downstream_dist = -1
                    endpoints_dist = [upstream_dist,downstream_dist]
                    

            limb_branch_info[k]["endpoints_dist"][j] = endpoints_dist
            limb_branch_info[k]["upstream_dist"][j] = upstream_dist
            limb_branch_info[k]["downstream_dist"][j] = downstream_dist
            limb_branch_info[k]["closest_sk_coordinate"][j] = syn_coord_sk
#             limb_branch_info[k]["head_neck_shaft"][j] = head_neck_shaft_val

        # THIS IS VERY LONG STEP IN THE KUBERNETES VERSION
        if debug_time:
            print(f"Closest Skeleton Branch and distance from endpoint: {np.round(time.time() - st,4)}")
            st = time.time()


    # ----- Phase 2: Creating the Synapse Objects ------ #

    st = time.time()

    #print(f"limb_branch_dict_to_add_synapses = {limb_branch_dict_to_add_synapses}")
    
    limb_branch_to_synapse_list = dict()
    for syn_type,valid_dict in valid_synapse_dict.items():
        syn_ids = valid_dict["valid"]

        for syn_idx,syn_id in enumerate(syn_ids):
            limb_idx = limb_branch_info[syn_type]["limb_idx"][syn_idx]
            branch_idx = limb_branch_info[syn_type]["branch_idx"][syn_idx]
            if limb_branch_dict_to_add_synapses is not None:
                if not nru.in_limb_branch_dict(limb_branch_dict_to_add_synapses,limb_idx,branch_idx):
                    continue
            syn_coord = valid_synapse_dict_coord[syn_type]["valid"][syn_idx]
            syn_volume = valid_synapse_dict_volume[syn_type]["valid"][syn_idx]
            syn_obj_dict = dict(syn_type=syn_type,
                            syn_id = syn_id,
                            volume = syn_volume,
                                coordinate = syn_coord,
                               )
            syn_obj_dict.update({k:v[syn_idx] for k,v in limb_branch_info[syn_type].items() if k not in ["limb_idx","branch_idx"]})
            


    #         curr_branch = neuron_obj[limb_idx][branch_idx]
    #         nviz.plot_objects(curr_branch.mesh,
    #                          scatters=[curr_branch.endpoints[0],
    #                                   curr_branch.endpoints[1],
    #                                   syn_obj_dict["coordinate"]],
    #                          scatters_colors=["red","blue","orange"],
    #                          scatter_size=0.5)


            if limb_idx not in limb_branch_to_synapse_list.keys():
                limb_branch_to_synapse_list[limb_idx] = dict()
            if branch_idx not in limb_branch_to_synapse_list[limb_idx].keys():
                limb_branch_to_synapse_list[limb_idx][branch_idx] = []

            limb_branch_to_synapse_list[limb_idx][branch_idx].append(syu.Synapse(**syn_obj_dict))
            
    #return limb_branch_to_synapse_list

    #----- Phase 2: Adding the Synapse Objects ------ #
    soma_synapses = {k:[] for k in neuron_obj.get_soma_indexes()}
    for l,limb_data in limb_branch_to_synapse_list.items():
        limb_name = nru.get_limb_string_name(l)
        if limb_branch_dict_to_add_synapses is not None:
            if limb_name not in limb_branch_dict_to_add_synapses.keys():
                continue
        for b,branch_synapses in limb_data.items():
            if limb_branch_dict_to_add_synapses is not None:
                if b not in limb_branch_dict_to_add_synapses[limb_name]:
                    continue
            if l < 0 or b < 0:
                soma_synapses[-l - nru.soma_face_offset]+= branch_synapses
            else:
                neuron_obj[l][b].synapses = branch_synapses

    # storing the soma synapses
    if limb_branch_dict_to_add_synapses is None:
        for k,v in soma_synapses.items():
#             for v_syn in v:
#                 v.compartment = "soma"
            neuron_obj[f"S{k}"].synapses = v

#     print(f"soma_synapses= {soma_synapses}")
#     print(f"neuron_obj.syanspes_somas = {neuron_obj.synapses_somas}")

    if verbose:
        print(f"Total time for valid synapse objects = {time.time() - st}")
    
    
def add_error_synapses_to_neuron_obj(
    neuron_obj,
    synapse_dict=None,
    mesh_label_dict=None,
    validation = False,
    verbose = False,
    original_mesh = None,
    ):

    """
    Pseudocode: 
    0) Get the coordinates and volumes of each 
    For each error type
    a) Create a list for storage
    For presyn/postsyn:
        c) Build the synapses from the information
        d) store in the list

    """
    st = time.time()
    
    # ----- Phase 0: Getting the Synapse Info ------ #
    if synapse_dict is None:
        synapse_dict = vdi.segment_id_to_synapse_dict(neuron_obj.segment_id,
                                                 validation=validation,
                                                 verbose=verbose)
        
    if mesh_label_dict is None:
        mesh_label_dict = syu.fetch_synapse_dict_by_mesh_labels(
            mesh = nru.neuron_mesh_from_branches(neuron_obj),
            segment_id=neuron_obj.segment_id,
            synapse_dict = synapse_dict,
            original_mesh = original_mesh,
            #original_mesh_kd=original_mesh_kd,
            validation=validation,
            plot_synapses=False,
            verbose = verbose)                        
                    

    # ------ Phase 1: Getting the Error Dictionaries set Up -------
    error_synapse_dict = dict()
    for t in ["presyn","postsyn"]:
        error_synapse_dict[t] = {k:v for k,v in mesh_label_dict[t].items() if "error" in k}



    error_synapse_dict_coord = syu.synapse_dict_mesh_labels_to_synapse_coordinate_dict(error_synapse_dict,
                                                            synapse_dict,
                                                            return_presyn_postsyn_valid_error_dict = False)
    error_synapse_dict_volume = syu.synapse_dict_mesh_labels_to_synapse_volume_dict(error_synapse_dict,
                                                            synapse_dict,
                                                            return_presyn_postsyn_valid_error_dict = False)

    # ------ Phase 2: Creating the Error Synapses From the Dictionary Info -------
    syn_error_lists = {k:[] for k in syu.synapse_error_types}
    for error_type in syu.synapse_error_types:
        if verbose:
            print(f"Working on error_type = {error_type}")
        for syn_type,valid_dict in error_synapse_dict.items():
            syn_ids = valid_dict[error_type]

            for syn_idx,syn_id in enumerate(syn_ids):
                syn_coord = error_synapse_dict_coord[syn_type][error_type][syn_idx]
                syn_volume = error_synapse_dict_volume[syn_type][error_type][syn_idx]
                syn_obj_dict = dict(syn_type=syn_type,
                                syn_id = syn_id,
                                volume = syn_volume,
                                    coordinate = syn_coord,
                                    soma_distance = -1,
                                   )
                syn_error_lists[error_type].append(syu.Synapse(**syn_obj_dict))


        setattr(neuron_obj,f"{error_type}_synapses",syn_error_lists[error_type])
    if verbose:
        print(f"Total time for valid synapse objects = {time.time() - st}")
        
        
def add_synapses_to_neuron_obj(
    neuron_obj,
    segment_id = None,
    validation = False,
    verbose  = False,
    original_mesh = None,
    plot_valid_error_synapses = False,
    calculate_synapse_soma_distance = False,
    add_valid_synapses = True,
    add_error_synapses = True,
    limb_branch_dict_to_add_synapses = None,
    **kwargs
    #set_head_neck_shaft=True
    ):

    """
    Purpose: To add the synapse information 
    to the neuron object

    Pseudocode: 
    0) Get the KDTree of the original mesh
    1) Get 


    """
    if verbose:
        print(f"\n---Step 1: Computing synapse_dict---")
        
    
        
    if segment_id is None:
        segment_id = neuron_obj.segment_id
        
    synapse_dict = vdi.segment_id_to_synapse_dict(
        segment_id = segment_id,
        validation=validation,
        verbose=verbose,
        **kwargs
    )


    

    #original_mesh_kd = tu.mesh_to_kdtree(original_mesh)
    if verbose:
        print(f"\n---Step 2: Computing mesh_label_dict---")
    mesh_label_dict = syu.fetch_synapse_dict_by_mesh_labels(
        mesh = nru.neuron_mesh_from_branches(neuron_obj),
        segment_id=segment_id,
        synapse_dict = synapse_dict,
        original_mesh = original_mesh,
        #original_mesh_kd=original_mesh_kd,
        validation=validation,
        plot_synapses=plot_valid_error_synapses,
        verbose = verbose)


    # Apply the Valid Synapses
    if add_valid_synapses:
        if verbose:
            print(f"\n---Step 3: add_valid_synapses_to_neuron_obj---")
        syu.add_valid_synapses_to_neuron_obj(neuron_obj,
                                            synapse_dict=synapse_dict,
                                            mesh_label_dict=mesh_label_dict,
                                             validation = validation,
                                            verbose = verbose,
                                            debug_time = verbose,
                                            calualate_endpoints_dist = True,
                                            limb_branch_dict_to_add_synapses = limb_branch_dict_to_add_synapses
                                             #set_head_neck_shaft=set_head_neck_shaft
                                            )
    
    if add_error_synapses:
        if verbose:
            print(f"\n---Step 4: add_error_synapses_to_neuron_obj---")

        syu.add_error_synapses_to_neuron_obj(neuron_obj,
                                            synapse_dict=synapse_dict,
                                            mesh_label_dict=mesh_label_dict,
                                             validation = validation,
                                            verbose = verbose)

    if calculate_synapse_soma_distance:
        if verbose:
            print(f"\n---Step 5: Adding Soma distances to synapse objects---")
        syu.calculate_neuron_soma_distance(neuron_obj)
        
        
    syu.set_limb_branch_idx_to_synapses(neuron_obj)

    return neuron_obj

def synapses_to_coordinates(synapses,
                           coordinate_type = "coordinate"):
    """
    Will export the coordinates of a list of synapse
    
    Other coordinate types:
    1) coordinate
    2) closest_sk_coordinate
    3) closest_face_coordinate
    """
    return np.array([getattr(k,coordinate_type) for k in synapses]).reshape(-1,3)

def synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict,
                                  synapse_type = "synapses"):
    """
    To gather all of the synapses over a limb branch dict restriction
    
    Ex: 
    syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict=dict(L2=[5,6,7]),
                                  synapse_type = "synapses")
    """
    return nru.concatenate_feature_over_limb_branch_dict(neuron_obj,
                                             limb_branch_dict=limb_branch_dict,
                                            feature=synapse_type)

def n_synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict,
                                  synapse_type = "synapses"):
    """
    To gather all of the synapses over a limb branch dict restriction
    
    Ex: 
    syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict=dict(L2=[5,6,7]),
                                  synapse_type = "synapses")
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                             limb_branch_dict=limb_branch_dict,
                                            feature=f"n_{synapse_type}")

def n_synapses_pre_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict,):
    """
    To gather all of the synapses over a limb branch dict restriction
    
    Ex: 
    syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict=dict(L2=[5,6,7]),
                                  synapse_type = "synapses")
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                             limb_branch_dict=limb_branch_dict,
                                            feature=f"n_synapses_pre")

def n_synapses_post_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict,):
    """
    To gather all of the synapses over a limb branch dict restriction
    
    Ex: 
    syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict=dict(L2=[5,6,7]),
                                  synapse_type = "synapses")
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                             limb_branch_dict=limb_branch_dict,
                                            feature=f"n_synapses_post")
def synapse_pre_perc_over_limb_branch_dict(neuron_obj,
                                           limb_branch_dict
                                          ):
    """
    Purpose: Will compute the percentage of synapses that are presyn over a limb branch dict
    
    Ex: 
    branches = [0,5,6,7]
    lb = dict(L0=branches)
    syn_pre_perc = syu.synapse_pre_perc_over_limb_branch_dict(neuron_obj_exc_syn_sp,
                                                             lb)
    """
    curr_syns = syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict)
    return syu.synapse_pre_perc(curr_syns)


def synapse_post_perc_over_limb_branch_dict(neuron_obj,
                                           limb_branch_dict
                                          ):
    """
    Purpose: Will compute the percentage of synapses that are postsyn over a limb branch dict
    
    Ex: 
    lb = dict(L0=branches)
    syn_post_perc = syu.synapse_post_perc_over_limb_branch_dict(neuron_obj_exc_syn_sp,
                                                             lb)
    """
    curr_syns = syu.synapses_over_limb_branch_dict(neuron_obj,
                                  limb_branch_dict)
    return syu.synapse_post_perc(curr_syns)


def get_errored_synapses_names(neuron_obj):
    return [k for k in neuron_obj.__dict__.keys() if "errored_synapses" in k]


def synapses_somas(neuron_obj):
    total_synapses = []
    for s in neuron_obj.get_soma_node_names():
        total_synapses += neuron_obj[s].synapses
    return total_synapses

soma_synapses = synapses_somas

def synapses_somas_postsyn(
    neuron_obj,
    verbose = False,
    plot= False,
    **kwargs):
    
    return syu.query_synapses(
    neuron_obj,
    query = "(compartment == 'soma') and (syn_type == 'postsyn') ",
    plot = plot,
    verbose = verbose
    )

def n_synapses_somas_postsyn(neuron_obj,**kwargs):
    return len(syu.synapses_somas_postsyn(neuron_obj,**kwargs))
    

    

def synapses_valid_old(neuron_obj,include_soma=True):
    """
    Will return all valid synapses and somas
    """
    valid_synapses = neuron_obj.synapses
    if include_soma:
        valid_synapses += neuron_obj.synapses_somas
    return valid_synapses

def synapses_valid(neuron_obj,include_soma=True):
    """
    Will return all valid synapses and somas
    """
    valid_syns = syu.query_synapses(neuron_obj,
                      query = valid_query(),
                        return_synapses=True,
                      verbose = False)
    return valid_syns

def n_synapses_valid(neuron_obj,include_soma=True):
    return len(syu.synapses_valid(neuron_obj))

def synapses_valid_pre(neuron_obj):
    return syu.synapses_pre(syu.synapses_valid(neuron_obj))

def synapses_valid_post(neuron_obj):
    return syu.synapses_post(syu.synapses_valid(neuron_obj))

def n_synapses_valid_pre(neuron_obj):
    return len(syu.synapses_valid_pre(neuron_obj))

def n_synapses_valid_post(neuron_obj):
    return len(syu.synapses_valid_post(neuron_obj))

def synapses_error_old(neuron_obj,
                    error_synapses_names=None,
                    presyns_on_dendrite_as_errors = presyns_on_dendrite_as_errors,
                    verbose=False):
    """
    Will get all of the errored synapses stored in the object
    
    syu.error_synpases(neuron_obj,
                verbose = True)
    """
    
    total_error_synapses = []
    if error_synapses_names is None:
        error_synapses_names = syu.get_errored_synapses_names(neuron_obj)
        
    if verbose:
        print(f"error_synapses_names = {error_synapses_names}")
    for err_syn in error_synapses_names:
        total_error_synapses += getattr(neuron_obj,err_syn)
        
    return total_error_synapses

def synapses_error(neuron_obj,
                    error_synapses_names=None,
                    presyns_on_dendrite_as_errors = presyns_on_dendrite_as_errors,
                    verbose=False):
    """
    Will get all of the errored synapses stored in the object
    
    syu.error_synpases(neuron_obj,
                verbose = True)
    """
    
    
    error_syns = syu.query_synapses(neuron_obj,
                      query = error_query(),
                    return_synapses=True,
                      verbose = False)
    return error_syns

def n_synapses_error(neuron_obj,
                    error_synapses_names=None,
                    presyns_on_dendrite_as_errors = presyns_on_dendrite_as_errors,
                    verbose=False):
    return len(syu.synapses_error(neuron_obj,
                    error_synapses_names=error_synapses_names,
                    presyns_on_dendrite_as_errors = presyns_on_dendrite_as_errors,
                    verbose=verbose))

def synapses_error_pre(neuron_obj):
    return syu.synapses_pre(syu.synapses_error(neuron_obj))

def synapses_error_post(neuron_obj):
    return syu.synapses_post(syu.synapses_error(neuron_obj))

def n_synapses_error_pre(neuron_obj):
    return len(syu.synapses_error_pre(neuron_obj))

def n_synapses_error_post(neuron_obj):
    return len(syu.synapses_error_post(neuron_obj))


def synapses_total(neuron_obj):
    """
    Purpose: to find the total number of synapses stored
    in the neuron object
    
    Pseducode: 
    1) get all of the soma synapses
    2) get all of the errored synapses
    3) get all of the limb_branch synapses
    """
#     # old way of compiling
#     soma_synapses = neuron_obj.synapses_somas
#     errored_synapses = neuron_obj.synapses_error
#     limb_branch_synapses = neuron_obj.synapses
#     soma_synapses + errored_synapses + limb_branch_synapses
    
    total_synapses = []
    for s_type,s_type_att in syu.synapse_types.items():
        curr_syns = getattr(neuron_obj,s_type_att)
        total_synapses += curr_syns
    
    return total_synapses


def synapses_to_synapses_df(
    synapses,
    label="no_label",
    add_compartment_coarse_fine = False,
    decode_head_neck_shaft_idx = False,):
    synapse_dicts = [dict(k.export(),
                                   label=label,
                                   )
                                   for k in synapses]
    df = pu.dicts_to_dataframe(synapse_dicts)
    
    df = annotate_synapse_df(
        df,
        add_compartment_coarse_fine=add_compartment_coarse_fine,
        decode_head_neck_shaft_idx = decode_head_neck_shaft_idx,
    )
    
    return df

def synapses_df(
    neuron_obj,
    synapse_types_to_process = None,
    verbose = False,
    add_compartment_coarse_fine = False,
    decode_head_neck_shaft_idx = False,
    **kwargs):
    """
    Purpose: To create a dataframe with all of the features
    of the synapses so the synapses can be queried
    """
    if isinstance(neuron_obj,neuron.Limb) or (isinstance(neuron_obj,neuron.Branch)):
        neuron_obj = neuron_obj.synapses

    if type(neuron_obj) == list:
        return syu.synapses_to_synapses_df(
            neuron_obj,
            add_compartment_coarse_fine=add_compartment_coarse_fine,
            decode_head_neck_shaft_idx=decode_head_neck_shaft_idx,
            **kwargs)
        
    
        
    if synapse_types_to_process is not None:
        curr_synapse_types = {k:v for k,v in 
                              syu.synapse_types.items() if k in synapse_types_to_process}
    else:
        curr_synapse_types = syu.synapse_types

    synapse_dicts = []
    for s_type,s_type_att in curr_synapse_types.items():
        if s_type != "limb_branch":
            if s_type == "soma":
                compartment = "soma"
            elif "error" in s_type:
                compartment = "error"
            else:
                raise Exception(f"Unknown stype = {s_type}")
                
            curr_syns = getattr(neuron_obj,s_type_att)
            curr_synapse_dicts = []
            for ss in curr_syns:
                curr_dict = dict(label=s_type,
                                       limb_idx = -1,
                                       branch_idx = -1,
                                       compartment = compartment,)
                curr_dict.update({j:b for j,b in ss.export().items() if (b is not None
                                                                        or j not in ["label","limb_idx","brnach_idx","compartment"])})
                curr_synapse_dicts.append(curr_dict)
#             curr_synapse_dicts = [dict(k.export(),
#                                        label=s_type,
#                                        limb_idx = -1,
#                                        branch_idx = -1,
#                                        compartment = compartment,
#                                        )
#                                        for k in curr_syns]
            synapse_dicts += curr_synapse_dicts
        else:
            for limb_idx in neuron_obj.get_limb_names(return_int=True):
                limb_obj = neuron_obj[limb_idx]
                for branch_idx in limb_obj.get_branch_names():
                    curr_synapse_dicts = []
                    branch_obj = limb_obj[branch_idx]
                    for ss in branch_obj.synapses:
                        curr_dict = dict(label=s_type,
                                        limb_idx = limb_idx,
                                       branch_idx = branch_idx,
                                        compartment = limb_obj[branch_idx].axon_compartment,
                                        #width = 
                                        )
                        curr_dict.update({j:b for j,b in ss.export().items() if (b is not None
                                                                        or j not in ["label","limb_idx","brnach_idx","compartment"])})
                        curr_synapse_dicts.append(curr_dict)
#                         curr_synapse_dicts = [dict(k.export(),
#                                        label=s_type,
#                                         limb_idx = limb_idx,
#                                        branch_idx = branch_idx,
#                                         compartment = limb_obj[branch_idx].axon_compartment,
#                                        )
#                                        for k in limb_obj[branch_idx].synapses]
                    synapse_dicts += curr_synapse_dicts

    if verbose:
        print(f"# of synapses processed = {len(synapse_dicts)}")
        
    df = pu.dicts_to_dataframe(synapse_dicts)
    df = nru.add_limb_branch_combined_name_to_df(
        df,
    )
    
    df = annotate_synapse_df(
        df,
        add_compartment_coarse_fine=add_compartment_coarse_fine,
        decode_head_neck_shaft_idx = decode_head_neck_shaft_idx,
        )
    return df

def annotate_synapse_df(
    df,
    add_compartment_coarse_fine=False,
    decode_head_neck_shaft_idx = False,
    ):
    
    if add_compartment_coarse_fine:
        df = apu.add_compartment_coarse_fine_to_df(
            df
        )
        
    if decode_head_neck_shaft_idx:
        df["head_neck_shaft"] = spu.decode_head_neck_shaft_idx(
            df.head_neck_shaft
        )
    
    return df

synapse_df = synapses_df


def restrict_synapses_df_by_limb_branch_dict(
    df,
    limb_branch_dict,
    ):
    
    return df.query(f"limb_branch in {list(nru.limb_branch_str_names_from_limb_branch_dict(limb_branch_dict))}")
def query_synapses(neuron_obj,
                  query,
                  return_df=False,
                  return_index=False,
                   return_synapses=False,
                  return_column = "syn_id",
                  local_dict = dict(),
                   limb_branch_dict = None,
                  verbose = False,
                   plot = False,
                   synapse_size = 1,
                  **kwargs):
    """
    Purpose: To return a dataframe 
    
    Pseudocode:
    1) Create synapse dataframe
    2) Use the query function to reduce the dataframe
    3) Return the desired output
    
    Note: YOU CAN SEND THIS FUNCTION A LIST OF SYNAPSES NOW
    
    EX: 
    syn_type = "presyn"
    head_neck_shaft_type = "shaft"
    syu.query_synapses(neuron_obj_exc_syn_sp[0][0].synapses,
                       query=(f"(syn_type == '{syn_type}') and "
                             f"(head_neck_shaft == {spu.head_neck_shaft_dict[head_neck_shaft_type]})"),
                       return_synapses=True
                      )
    """

    synapse_df = syu.synapses_df(neuron_obj,**kwargs)
    query_df = synapse_df.query(query,
                    local_dict=local_dict)
    
    if limb_branch_dict is not None and len(query_df)>0:
        if len(limb_branch_dict) > 0:
            query_df = syu.restrict_synapses_df_by_limb_branch_dict(
               query_df,
               limb_branch_dict
            )

    if verbose:
        print(f"# of synapses in query = {len(query_df)}")
        
    if plot:
        syn_indexes = query_df.index.to_numpy()
        synapse_objs = syu.synapse_indexes_to_synapses(neuron_obj,
                                          syn_indexes)
        syu.plot_synapses_objs(neuron_obj,
                  synapse_objs,
                synapse_size=synapse_size)
    
    if return_synapses:
        syn_indexes = query_df.index.to_numpy()
        if nu.is_array_like(neuron_obj):
            return [neuron_obj[k] for k in syn_indexes]
        else:
            return synapse_indexes_to_synapses(neuron_obj,
                                          syn_indexes)
    elif return_df:
        return query_df
    elif return_index:
        return query_df.index.to_numpy()
    else:
        return query_df[return_column].to_numpy()


# ---------- 6:11 For creating datajoint entries ----------
def synapses_dj_export_dict_valid(synapse,
                                 output_spine_str=True,):
    syn=synapse
    spine_label = syn.head_neck_shaft
    
    compartment_coarse,compartment_fine = apu.coarse_fine_compartment_from_label(syn.compartment)
        
    return_dict = dict(synapse_id=syn.syn_id,
          synapse_type=syn.syn_type,
          skeletal_distance_to_soma = np.round(syn.soma_distance,2),
          limb_idx = syn.limb_idx,
          branch_idx = syn.branch_idx,
          compartment_coarse=compartment_coarse,
          compartment_fine=compartment_fine)
    
    if output_spine_str:
        spine_label = spu.spine_str_label(spine_label)
        return_dict["spine_bouton"] = spine_label
    
    return return_dict
    
def synapses_dj_export_dict_error(synapse,**kwargs):
    syn=synapse
    return dict(synapse_id=syn.syn_id,
          synapse_type=syn.syn_type,)
        


def synapses_to_dj_keys_old(
    neuron_obj,
    valid_synapses = True,
    verbose = False,
    nucleus_id = None,
    split_index = None,
    output_spine_str=True,
    ver=None):
    """
    Pseudocode: 
    1) Get either the valid of invalid synapses
    2) For each synapses export the following as dict

    synapse_id=syn,
    synapse_type=synapse_type,
    nucleus_id = nucleus_id,
    segment_id = segment_id,
    split_index = split_index,
    skeletal_distance_to_soma=np.round(syn_dist[syn_i],2),

    return the list
    
    Ex: 
    from datasci_tools import numpy_dep as np
    dj_keys = syu.synapses_to_dj_keys(neuron_obj,
                           verbose = True,
                           nucleus_id=12345,
                           split_index=0)
    np.unique([k["compartment"] for k in dj_keys],return_counts=True)
    
    
    Ex:  How to get error keys with version:
    dj_keys = syu.synapses_to_dj_keys(neuron_obj,
                                  valid_synapses = False,
                       verbose = True,
                       nucleus_id=12345,
                       split_index=0,
                                 ver=158)

    """
    segment_id = neuron_obj.segment_id
    
    if nucleus_id is not None:
        neuron_obj.nucleus_id = nucleus_id
    if split_index is not None:
        neuron_obj.split_index = split_index
    
    if valid_synapses:
        synapses = neuron_obj.synapses_valid
        export_func = syu.synapses_dj_export_dict_valid 
    else:
        synapses = neuron_obj.synapses_error
        export_func = syu.synapses_dj_export_dict_error
        
    for att in ["nucleus_id","segment_id","split_index"]:
#         globs = globals()
#         locs = locals()
        if getattr(neuron_obj,att) is None and eval(att) is None:
            raise Exception(f"{att} is None")
    st = time.time()
#     syn_keys = [dict(synapse_id=syn.syn_id,
#                      synapse_type=syn.syn_type,
#                     nucleus_id=neuron_obj.nucleus_id,
#                     segment_id=neuron_obj.segment_id,
#                     split_index = neuron_obj.split_index,
#         skeletal_distance_to_soma = np.round(syn.soma_distance,2)) for syn in synapses]

    syn_keys = [dict(export_func(syn,output_spine_str=output_spine_str),
                                nucleus_id=getattr(neuron_obj,"nucleus_id",nucleus_id),
                    segment_id=neuron_obj.segment_id,
                    split_index = getattr(neuron_obj,"split_index",split_index)) for syn in synapses]
    
    #adding on the secondary seg
    
    if verbose:
        print(f"valid_synapses = {valid_synapses}")
        print(f"Time for {len(syn_keys)} synapse entries = {time.time() - st}")
    
    if ver is not None:
        syn_keys = [dict(k,ver=ver) for k in syn_keys]
    
    return syn_keys



def presyn_on_dendrite_synapses(
    neuron_obj,
    split_index=0,
    nucleus_id=0,
    return_dj_keys = False,
    verbose = True,
    **kwargs
    ):


    syns = syu.query_synapses(
        neuron_obj,query="(compartment=='dendrite') and (syn_type=='presyn')",
        return_synapses=True
    )
    
    if verbose:
        print(f"# of presyns on dendrite = {len(syns)}")

    if return_dj_keys:
        return syu.synapses_to_dj_keys(
            neuron_obj,
            synapses=syns,
            valid_synapses=False,
            nucleus_id=nucleus_id,
            split_index = split_index,
            verbose = verbose)
    return syns

def presyn_on_dendrite_synapses_non_axon_like(
    neuron_obj,
    limb_branch_dict = None,
    plot = False,
    **kwargs):
    """
    Purpose: To get the presyns on dendrites
    where the dendrites are restricted to 
    those that aren't axon-like
    """
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.non_axon_like_limb_branch_on_dendrite
    return syu.query_synapses(
        neuron_obj,
        query = syu.presyns_on_dendrite_query,
        limb_branch_dict=limb_branch_dict,
        plot = plot,
        **kwargs
    )




def presyns_on_dendrite(neuron_obj,
                       verbose = False,
                        return_df=False,
                        return_column = "syn_id",
                       ):
    """
    Purpose: Be able to find the synapses that are presyn on dendrite

    Pseudocode: 
    1) query the synapses_df for "(label=='limb_branch') and (compartment=='dendrite') and (syn_type=='presyn')"
    """
    query_df = syu.query_synapses(neuron_obj,
                    query=presyns_on_dendrite_query,
                   return_df=True,
                      verbose = False)
    
    if return_df:
        return query_df
    else:
        return query_df[return_column].to_numpy()
    
def axon_synapses(neuron_obj,
                ):
    return syu.query_synapses(neuron_obj,
                    query="compartment=='axon'",
                   return_df=False,
                      verbose = False)
    
    
def n_presyns_on_dendrite(neuron_obj):
    return len(syu.presyns_on_dendrite(neuron_obj))
    
def synapse_ids_to_synapses(neuron_obj,
                           synapse_ids,
                           verbose = True):
    """
    If have list of synapse ids and want to find the corresponding objects
    
    (not used in queries at all)
    
    """
    st = time.time()
    
    match_syn = [k for k in neuron_obj.synapses_total
                if k.syn_id in set(synapse_ids)]
    
    if verbose:
        print(f"Total time for retrieving synapses: {time.time() - st}")
        
    return match_syn

def synapse_indexes_to_synapses(neuron_obj,
                           synapse_indexes,
                           verbose = False):
    st = time.time()
    
    match_syn = list(np.array(neuron_obj.synapses_total)[synapse_indexes])
    
    if verbose:
        print(f"Total time for retrieving synapses: {time.time() - st}")
        
    return match_syn

def synapses_to_feature(synapses,feature):
    return [getattr(k,feature) for k in synapses]
def synapses_to_synapse_ids(synapses):
    return syu.synapses_to_feature(synapses,feature="syn_id")
def synapses_to_coordinates(synapses):
    return syu.synapses_to_feature(synapses,feature="coordinate")
    
    
def synapses_error_pre_coordinates(neuron_obj):
    return syu.synapses_to_coordinates(syu.synapses_error_pre(neuron_obj))

def synapses_error_post_coordinates(neuron_obj):
    return syu.synapses_to_coordinates(syu.synapses_error_post(neuron_obj))

def synapses_valid_pre_coordinates(neuron_obj):
    return syu.synapses_to_coordinates(syu.synapses_valid_pre(neuron_obj))

def synapses_valid_post_coordinates(neuron_obj):
    return syu.synapses_to_coordinates(syu.synapses_valid_post(neuron_obj))
    
def synapse_pre_post_valid_errror_stats_dict(neuron_obj):
    presyn_error_syn_non_axon_ids = syu.presyns_on_dendrite(neuron_obj)
    stats_dict = dict(n_valid_syn_presyn = len(syu.synapses_valid_pre(neuron_obj)),
    n_valid_syn_postsyn = len(syu.synapses_valid_post(neuron_obj)),
    n_errored_syn_presyn = len(syu.synapses_error_pre(neuron_obj)),
    n_errored_syn_postsyn = len(syu.synapses_error_post(neuron_obj)),
    presyn_error_syn_non_axon_ids = presyn_error_syn_non_axon_ids,
    n_errored_syn_presyn_non_axon = len(presyn_error_syn_non_axon_ids))
    
    return stats_dict

def synapses_error_ids(neuron_obj):
    return syu.query_synapses(neuron_obj,
                      query=error_query())

def synapse_pre_post_valid_errror_coordinates_dict(neuron_obj):
    """
    Purpose: To make a dictionary that has
    the valid and error synapses

    Pseudocode: 
    For valid and error: 
        for presyn and postsyn
        1) Get the corresponding synapses
        2) extract the coordinates from the synapses
        3) store in the dictionary
    """
    return dict(valid_syn_centers_presyn=syu.synapses_valid_pre_coordinates(neuron_obj),
    errored_syn_centers_presyn = syu.synapses_error_pre_coordinates(neuron_obj),
    valid_syn_centers_postsyn = syu.synapses_valid_post_coordinates(neuron_obj),
    errored_syn_centers_postsyn = syu.synapses_error_post_coordinates(neuron_obj))




# ------------ Function to replace the old filtering function ----------- #
def synapse_filtering_vp2(
    neuron_obj,
    split_index = None,
    nucleus_id = None,
    original_mesh = None,
    verbose = True,
    compute_synapse_to_soma_skeletal_distance = False,
    return_synapse_filter_info = True,
    return_error_synapse_ids = True,
    return_synapse_center_data = False,
    return_errored_synapses_ids_non_axons = True,
    return_error_table_entries = True,
    return_neuron_obj = False,
    apply_non_axon_presyn_errors = True,
    plot_synapses = False,
    validation = False):

    """
    Purpose: Applying the synpase filtering 
    by using the synapses incorporated in the
    neuron object

    """


    syu.add_synapses_to_neuron_obj(neuron_obj,
                                validation = validation,
                                verbose  = verbose,
                                original_mesh = original_mesh,
                                plot_valid_error_synapses = False,
                                calculate_synapse_soma_distance = False,
                                add_valid_synapses = False,
                                  add_error_synapses=True)
    
    #---- 8/29: Will add limb and branch idx to synapses ------
    syu.set_limb_branch_idx_to_synapses(neuron_obj)

    # prework: adding the nucleus_id,split-index and calculating the soma distances
    if nucleus_id is not None:
        neuron_obj.nucleus_id = nucleus_id

    if split_index is not None:
        neuron_obj.split_index = split_index


    syu.presyns_on_dendrite_as_errors = apply_non_axon_presyn_errors

    if compute_synapse_to_soma_skeletal_distance:
        syu.calculate_neuron_soma_distance(neuron_obj,verbose = True)

    keys_to_write_without_version = syu.synapses_to_dj_keys(neuron_obj,
                       valid_synapses = True,
                       verbose = True)

    keys_to_write_without_version_errors = syu.synapses_to_dj_keys(neuron_obj,
                       valid_synapses = False,
                       verbose = True)

    synapse_stats = syu.synapse_pre_post_valid_errror_stats_dict(neuron_obj)

    total_error_synapse_ids = syu.synapses_error_ids(neuron_obj)
    if verbose:
        print(f"# of total_error_synapse_ids = {len(total_error_synapse_ids)}")

    
    
    if return_synapse_center_data:
        synapse_center_coordinates = syu.synapse_pre_post_valid_errror_coordinates_dict(neuron_obj)
    
    if plot_synapses:
        print("Displaying the Synapse Classifications")
        syu.plot_valid_error_synpases(neuron_obj)
    
    
    syu.set_presyns_on_dendrite_as_errors_default()
    
    if ((not return_synapse_filter_info)
        and (not return_synapse_center_data) 
        and (not return_error_synapse_ids)
        and (not return_error_table_entries)
        and (not return_neuron_obj)
       ):
        return data_to_write
    
    return_value = [keys_to_write_without_version]
    
    
    if return_synapse_filter_info:
        return_value.append(synapse_stats)
    if return_synapse_center_data:
        return_value.append(synapse_center_coordinates)
    if return_error_synapse_ids:
        return_value.append(total_error_synapse_ids)
    if return_error_table_entries:
        return_value.append(keys_to_write_without_version_errors)
    if return_neuron_obj:
        return_value.append(neuron_obj)
    
    return return_value

# ------------- 6/25: For the synapse near the endpoint -----------
def synapse_endpoint_dist_upstream_downstream(limb_obj,
                                              branch_idx,
                                              direction,
                                         synapses = None,
                                             synapse_type = "synapses",
                                             verbose = False):
    """
    Purpose: Will compute the upstream or downstream
    distance of a synapse or group of synapses
    
    Pseudocode:
    1) Get the upstream or downstream endpoint index
    
    For each synapse
    2) Get the upstream or downstream distance
    
    3) Return the list
    
    Ex: 
    from neurd import synapse_utils as syu
    syu.synapse_endpoint_dist_upstream_downstream(limb_obj,
                                             branch_idx = 16,
                                             direction="downstream",
                                             verbose = True)
    """
    singular_flag = False
    if synapses is None:
        synapses = getattr(limb_obj[branch_idx],synapse_type)
    else:
        if not nu.is_array_like(synapses):
            singular_flag = True
        synapses = [synapses]
        
    #1) Get the upstream or downstream endpoint index
    if direction == "upstream":
        endpoint_idx = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=True)
    elif direction == "downstream":
        endpoint_idx = nru.downstream_endpoint(limb_obj,branch_idx,return_endpoint_index=True)
        
    if verbose:
        print(f"using direction {direction}, endpoint_idx = {endpoint_idx}")
        
    dist_from_endpoint = np.array([syn.endpoints_dist[endpoint_idx] for syn in synapses])
    
    if verbose:
        print(f"dist_from_endpoint = {dist_from_endpoint}")
    
    if singular_flag:
        return dist_from_endpoint[0]
    return dist_from_endpoint

def synapse_endpoint_dist_downstream(limb_obj,
                                              branch_idx,
                                         synapses = None,
                                             synapse_type = "synapses",
                                             verbose = False):
    return synapse_endpoint_dist_upstream_downstream(limb_obj,
                                              branch_idx,
                                              direction="downstream",
                                         synapses = synapses,
                                             synapse_type = synapse_type,
                                             verbose = verbose)

def synapse_endpoint_dist_upstream(limb_obj,
                                              branch_idx,
                                         synapses = None,
                                             synapse_type = "synapses",
                                             verbose = False):
    return synapse_endpoint_dist_upstream_downstream(limb_obj,
                                              branch_idx,
                                              direction="upstream",
                                         synapses = synapses,
                                             synapse_type = synapse_type,
                                             verbose = verbose)

def plot_synapses_objs(neuron_obj,
                     synapses,
                      plot_with_spines = False,
                      synapse_color = "red",
                      synapse_size = 0.15,
                     **kwargs):
    """
    Purpose: To plot a certain group of synapses on top
    of the neuron object
    
    Ex: 
    syu.plot_synapse_objs(neuron_obj_exc_syn_sp,
                 synapses = syu.synapses_shaft(neuron_obj_exc_syn_sp),
                  synapse_color="yellow"
                 )
    """
    if nru.is_neuron_obj(neuron_obj):
        if not plot_with_spines:
            nviz.visualize_neuron_lite(neuron_obj,
                                      show_at_end=False)
        else:
            spu.plot_spines_head_neck(neuron_obj,
                                     show_at_end=False)
    else:
        nviz.plot_objects(neuron_obj,
                         show_at_end=False)
        
    nviz.plot_objects(
                     scatters=[syu.synapses_to_coordinates(synapses)],
                     scatter_size=synapse_size,
                     scatters_colors=synapse_color,
                     append_figure=True)
    
def plot_head_neck_shaft_synapses(neuron_obj,
                                  plot_with_spines = True,
                      head_color = "yellow",
                        neck_color = "blue",
                        shaft_color = "black",
                        no_head_color = "purple",
                        bouton_color = "pink",
                        non_bouton_color = "brown",
                      synapse_size = 0.15,
                                  verbose = False,
                     **kwargs):
    """
    Purpose: To plot all of the head neck and shaft spines of a neuron
    
    Ex: 
    syu.plot_head_neck_shaft_synapses(neuron_obj_exc_syn_sp,
                                 synapse_size=0.08)
    """
    if not plot_with_spines or neuron_obj.n_spines == 0:
        nviz.visualize_neuron_lite(neuron_obj,
                                  show_at_end=False)
    else:
        spu.plot_spines_head_neck(neuron_obj,
                                 show_at_end=False)
    
    synapses = []
    synapses_colors = []
    
    for t,col in zip(["head","neck","shaft","no_head","bouton","non_bouton"],
                     [head_color,neck_color,shaft_color,no_head_color,bouton_color,non_bouton_color]):
        syns = getattr(syu,f"synapses_{t}")(neuron_obj)
        if verbose:
            print(f"# of {t} = {len(syns)} ({col}) ")
        if len(syns) > 0:
            
            synapses.append(np.array(syu.synapses_to_coordinates(syns)).reshape(-1,3))
            synapses_colors.append(col)
            
    nviz.plot_objects(
                     scatters=synapses,
                     scatter_size=synapse_size,
                     scatters_colors=synapses_colors,
                     append_figure=True,
                    **kwargs)
    


def synapses_over_limb_branch_dict(neuron_obj,
                                   limb_branch_dict,
                                  synapse_type = "synapses",
                                  plot_synapses=False):
    synapses = nru.feature_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=limb_branch_dict,
                                     branch_func_instead_of_feature=getattr(syu,f"{synapse_type}"))
    if plot_synapses:
        syu.plot_synapses_objs(neuron_obj,
                       synapses=synapses)
    return synapses
    
    
def synapse_density_over_limb_branch(neuron_obj,
                                     limb_branch_dict,
                                    synapse_type = "synapses",
                                    multiplier = 1,
                                     verbose = False,
                                     return_skeletal_length = False,
                                     density_type = "skeletal_length",
                                    ):
    """
    Purpose: To calculate the 
    synapse density over lmb branch

    Application: To be used for cell type (E/I)
    classification

    Pseudocode: 
    1) Restrict the neuron branches to be processed
    for postsynaptic density
    2) Calculate the skeletal length over the limb branch
    3) Find the number of postsyns over limb branch
    4) Compute postsynaptic density
    
    Ex: 
    syu.synapse_density_over_limb_branch(neuron_obj = neuron_obj_exc_syn_sp,
                                     limb_branch_dict=syn_dens_limb_branch,
    #neuron_obj = neuron_obj_inh_syn_sp,
    verbose = True,
    synapse_type = "synapses_post",
    #synapse_type = "synapses_head",
    #synapse_type = "synapses_shaft",
    multiplier = 1000)

    """
    sk_length = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict,
                                         feature=density_type)

    n_synapses = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict,
                                         branch_func_instead_of_feature=getattr(syu,f"n_{synapse_type}"))
    if sk_length != 0:
        density = n_synapses/sk_length
    else:
        density = 0

    density = density*multiplier

    if verbose:
        print(f"{density_type} = {sk_length}")
        print(f"# of {synapse_type} = {n_synapses}")
        print(f"Density = {density}")

    if return_skeletal_length:
        return density,sk_length
    else:
        return density
    
# ----------------- 7/20: helping with axon and dendrite merge errors ---------- #
def synapse_pre_perc_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    """
    Purpose: Find the downstream
    downstream of a branch
    
    syu.synapse_pre_perc_downstream(   limb_obj = neuron_obj_exc_syn_sp[0],
    branch_idx = 6,
    verbose = True,
    )

    """
    synapses_downstream = cnu.synapses_downstream(limb_obj,branch_idx,
                                                 only_non_branching=False)
    pre_perc = syu.synapse_pre_perc(list(synapses_downstream))
    if verbose:
        print(f"# of downstream synapses = {len(synapses_downstream)}")
        print(f"pre_perc = {pre_perc}")

    return pre_perc

def synapse_post_perc_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    """
    Purpose: Find the downstream
    downstream of a branch
    
    Ex: 
    syu.synapse_post_perc_downstream(   limb_obj = neuron_obj_exc_syn_sp[0],
    branch_idx = 6,
    verbose = True,
    )

    """
    synapses_downstream = cnu.synapses_downstream(limb_obj,branch_idx,
                                                 only_non_branching=False)
    post_perc = syu.synapse_post_perc(list(synapses_downstream))
    if verbose:
        print(f"# of downstream synapses = {len(synapses_downstream)}")
        print(f"post_perc = {post_perc}")

    return post_perc


def synapses_downstream(limb_obj,
                       branch_idx,
                       verbose = False):
    synapses_downstream = cnu.synapses_downstream(limb_obj,branch_idx,
                                                 only_non_branching=False)
    if verbose:
        print(f"# of downstream synapses = {len(synapses_downstream)}")

    return synapses_downstream

def n_synapses_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    return len(synapses_downstream(limb_obj,
                               branch_idx,
                               verbose))

def synapses_post_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    """
    Purpose: Find the downstream
    downstream of a branch
    
    syu.synapse_pre_perc_downstream(   limb_obj = neuron_obj_exc_syn_sp[0],
    branch_idx = 6,
    verbose = True,
    )

    """
    synapses_downstream = cnu.synapses_downstream(limb_obj,branch_idx,
                                                 only_non_branching=False)
    post_synapses = syu.synapses_post(list(synapses_downstream))
    if verbose:
        print(f"# of downstream synapses = {len(synapses_downstream)}")
        print(f"post_synapses = {post_synapses}")

    return post_synapses

def synapses_pre_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    """
    Purpose: Find the downstream
    downstream of a branch
    
    syu.synapse_pre_perc_downstream(   limb_obj = neuron_obj_exc_syn_sp[0],
    branch_idx = 6,
    verbose = True,
    )

    """
    synapses_downstream = cnu.synapses_downstream(limb_obj,branch_idx,
                                                 only_non_branching=False)
    pre_synapses = syu.synapses_pre(list(synapses_downstream))
    if verbose:
        print(f"# of downstream synapses = {len(synapses_downstream)}")
        print(f"pre_synapses = {pre_synapses}")

    return pre_synapses
def n_synapses_post_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    return len(synapses_post_downstream(limb_obj,
                               branch_idx,
                               verbose))

def n_synapses_pre_downstream(limb_obj,
                               branch_idx,
                               verbose = False):
    return len(synapses_pre_downstream(limb_obj,
                               branch_idx,
                               verbose))
# --------------- 7/26: Help with axon identification -------------
def synapses_within_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = True):
    """
    Purpose: Will measure the number of synapses
    within a certain distance of the upstream or downstream endpoint
    
    Pseudocode: 
    1) Get the desired synapses
    2) Query the synapses based on the direction attribute
    
    Ex: 
    branch_obj = neuron_obj[0][2]
    synapses_within_upstream_downstream_endpoint(branch_obj,
                                         direction="downstream",
                                         distance =15000,
                                         synapse_type="synapses_post",
                                         )
    """
    synapses = getattr(branch_obj,synapse_type)
    
    if verbose:
        print(f"# of unfiltered synapses = {len(synapses)}")
        
    if synapses is not None and len(synapses) > 0:
        syn_filtered = syu.query_synapses(synapses,
                          query = f"{direction}_dist < {distance}",
                          return_synapses=True)

        if verbose:
            print(f"# of syn_filtered = {len(syn_filtered)}")
        
        return syn_filtered
    else:
        return []
    
def n_synapses_within_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    return len(syu.synapses_within_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs))

def n_synapses_within_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction="downstream",
                                      distance=distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs)
def n_synapses_within_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                    synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction="upstream",
                                      distance=distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_pre_within_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_pre",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_post_within_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_post",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_pre_within_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_pre",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_post_within_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_downstream_endpoint(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_post",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_spine_within_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_within_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_spine",
                                     verbose = verbose,
                                    **kwargs)


def synapse_density_post_within_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    n_post_syns = syu.n_synapses_within_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_post",
                                     verbose = verbose,
                                    **kwargs)
    return n_post_syns/branch_obj.skeletal_length

# ---- offset n_synapses ------------
def synapses_offset_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = True):
    """
    Purpose: Will measure the number of synapses
    within a certain distance of the upstream or downstream endpoint
    
    Pseudocode: 
    1) Get the desired synapses
    2) Query the synapses based on the direction attribute
    
    Ex: 
    branch_obj = neuron_obj[0][2]
    synapses_within_upstream_downstream_endpoint(branch_obj,
                                         direction="downstream",
                                         distance =15000,
                                         synapse_type="synapses_post",
                                         )
    """
#     verbose = True
    if verbose:
        print(f"distance = {distance}")
        
    synapses = getattr(branch_obj,synapse_type)
    
    if verbose:
        print(f"# of unfiltered synapses = {len(synapses)}")
        
    if synapses is not None and len(synapses) > 0:
        syn_filtered = syu.query_synapses(synapses,
                          query = f"{direction}_dist > {distance}",
                          return_synapses=True)

        if verbose:
            print(f"# of syn_filtered = {len(syn_filtered)}")
        
        return syn_filtered
    else:
        return []

def n_synapses_offset_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    
    return len(syu.synapses_offset_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction,
                                      distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs))

def n_synapses_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                      synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction="downstream",
                                      distance=distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs)
def n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                    synapse_type = "synapses",
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_upstream_downstream(branch_obj,
                                      direction="upstream",
                                      distance=distance,
                                      synapse_type = synapse_type,
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_pre_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_pre",
                                     verbose = verbose,
                                    **kwargs)



def n_synapses_pre_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_pre",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_post_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_post",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_post_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance=10_000,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_post",
                                     verbose = verbose,
                                    **kwargs)

def n_synapses_spine_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_spine",
                                     verbose = verbose,
                                    **kwargs)
def n_synapses_spine_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    return syu.n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                      synapse_type = "synapses_spine",
                                     verbose = verbose,
                                    **kwargs)


def synapse_density_post_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    n_post_syns = syu.n_synapses_post_offset_distance_of_endpoint_downstream(branch_obj,
                                      distance=distance,
                                     verbose = verbose,
                                    **kwargs)
    return n_post_syns/branch_obj.skeletal_length

def synapse_density_post_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    n_post_syns = syu.n_synapses_post_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                     verbose = verbose,
                                    **kwargs)
    return n_post_syns/branch_obj.skeletal_length

def synapse_density_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance,
                                     verbose = False,
                                                   **kwargs):
    n_post_syns = syu.n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                      distance=distance,
                                     verbose = verbose,
                                    **kwargs)
    return n_post_syns/branch_obj.skeletal_length
        
    
def plot_synapses_on_limb(neuron_obj,
                        limb_idx,
                          limb_branch_synapse_type = "synapses",
                          **kwargs
                        ):
    limb_name = nru.get_limb_string_name(limb_idx)
    nviz.visualize_neuron(neuron_obj,
                         limb_branch_dict={limb_name:"all"},
                         limb_branch_synapses = True,
                          limb_branch_synapse_type=limb_branch_synapse_type,
                         **kwargs)
    
def limb_branch_with_synapses(neuron_obj,
                             min_n_synapses = 1,
                             synapse_type = "synapses"):
    syn_name = f"n_{synapse_type}"
    return ns.query_neuron(neuron_obj,
               functions_list=[syn_name],
               query = f"{syn_name} > {min_n_synapses}",)

def set_branch_synapses_attribute(branch_obj,
                                         synapse_attribute,
                                         branch_func,
                                          catch_errors = False,
                                         default_value=None,
                                          verbose = False,
                                         ):
    """
    Purpose: Will set the synapse attributes on a branch object
    
    Ex: 
    def new_func(branch_obj):
        raise Exception("")
    syu.set_branch_synapses_attribute(neuron_obj[0][0],
                                 synapse_attribute="compartment",
                                 #branch_func = apu.compartment_label_from_branch_obj,
                                      branch_func = new_func,
                                      catch_errors=True,
                                      default_value="exception_label",
                                 verbose = True)
    """
    try:
        branch_val = branch_func(branch_obj)
    except Exception as e:
        if catch_errors: 
            branch_val = default_value
        else:
            raise Exception(f"{str(e)}")
            
    if verbose:
        print(f"branch_val = {branch_val}")
        
    for s in branch_obj.synapses:
        setattr(s,synapse_attribute,branch_val)
        
def set_branch_synapses_compartment(branch_obj,
                                          catch_errors = False,
                                         default_value=None,
                                          verbose = False,
                                         ):
    syu.set_branch_synapses_attribute(branch_obj,
                                         synapse_attribute="compartment",
                                         branch_func=apu.compartment_label_from_branch_obj,
                                          catch_errors = catch_errors,
                                         default_value=default_value,
                                          verbose = verbose,
                                         )
    
def set_neuron_synapses_compartment(neuron_obj,
                                   ):
    """
    Purpose: Will set the compartment labels of all synapses
    based on the compartment label of te branch
    
    """
    nru.set_branch_attribute_over_neuron(neuron_obj,
                                 branch_func=syu.set_branch_synapses_compartment,
                                  catch_errors = False,
                                 default_value=None,
                                  verbose = False,)
    for s_name in neuron_obj.get_soma_node_names():
        for s_syn in neuron_obj[s_name].synapses:
            s_syn.compartment = "soma"
    
    
'''        
def set_neuron_synapses_attribute(neuron_obj,
                                 synapse_attribute,
                                 branch_func,
                                  catch_errors = False,
                                 default_value=None,
                                  verbose = False,):
    """
    Purpose: To set the  attribute of synapses in an entire neuron
    based on a value computed from the branch
    """
    nru.set_branch_attribute_over_neuron(neuron_obj,
                                        branch_func=branch_func,
                                        verbose = verbose,
                                        synapse_attribute=synapse_attribute,
                                        catch_errors=catch_errors,
                                        default_value=default_value)
'''
def soma_face_offset_value(soma_name):
    soma_idx = int(soma_name[1:])
    soma_value = -1*(nru.soma_face_offset + soma_idx)

def set_limb_branch_idx_to_synapses(neuron_obj):
    """
    Purpose: Will add limb and branch indexes for
    all synapses in a Neuron object
    """
    for limb_idx in neuron_obj.get_limb_names(return_int=True):
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            branch_obj = limb_obj[branch_idx]
            for s in branch_obj.synapses:
                s.limb_idx = limb_idx
                s.branch_idx = branch_idx
                
    for soma_name in neuron_obj.get_soma_node_names():
        soma_value = syu.soma_face_offset_value(soma_name)
        for s in neuron_obj[soma_name].synapses:
            s.limb_idx = soma_value
            s.branch_idx = soma_value
            
            
def get_synapses_compartment(neuron_obj,
                            compartments,
                            verbose = False):
    """
    Purpose: will get synapses of a certain 
    compartment if synapses are labeled
    
    Ex: 
    synapses = syu.get_synapses_compartment(o_neuron,
                                           compartments=["apical","oblique"],
                                           verbose = True)
    """
    compartments = nu.convert_to_array_like(compartments)
    query = f"compartment in {list(compartments)}"
    if verbose:
        print(f"query = {query}")
    synapse_objs = syu.query_synapses(neuron_obj,
                  query = query,
                  return_synapses=True)
    if verbose: 
        print(f"Synapses in {compartments}: {len(synapse_objs)}")
    return synapse_objs
    
def plot_synapses_compartment(neuron_obj,
                             compartments,
                              synapse_size = 1,
                              verbose =False,
                             ):
    """
    Purpose: To plot all synapses of a certain compartment
    
    Ex: 
    synapses = syu.get_synapses_compartment(o_neuron,
                                           compartments=["apical","oblique"],
                                           verbose = True)
    """
    
    
    synapses = syu.get_synapses_compartment(neuron_obj,
                                           compartments,
                                           verbose = verbose)
    syu.plot_synapses_objs(neuron_obj,
                  synapses,
                      synapse_size=synapse_size)
    
def plot_synapses_query(neuron_obj,
                       query,
                       synapse_size = 1,
                       verbose = False):
    """
    Purpose: To plot the synapses from a query
    
    Ex: 
    syu.plot_synapses_query(neuron_obj,
                       query = "syn_type=='presyn'")
    """
    synapse_objs = syu.query_synapses(neuron_obj,
                  query = query,
                  return_synapses=True)
    
    if verbose:
        print(f"# of synapses in query = {len(synapse_objs)}")
    
    syu.plot_synapses_objs(neuron_obj,
                  synapse_objs,
                      synapse_size=synapse_size)
    
def plot_synapses_presyn_dendrite_errors(neuron_obj,
                                        verbose = False):
    syu.plot_synapses_query(neuron_obj,
                       syu.presyns_on_dendrite_query,
                       verbose = verbose)
    
def plot_synapses_soma(neuron_obj,
                       synapse_size = 3,
                       verbose =False,
                      ):
    """
    Ex: syu.plot_synapses_soma(o_neuron)
    """
    syu.plot_synapses_compartment(neuron_obj,
                             compartments="soma",
                              synapse_size = synapse_size,
                              verbose =verbose,
                             )
    
def plot_synapses_valid_from_neuron_obj(neuron_obj,
                                       synapse_size = 0.3,
                                        verbose = True
                                      ):
    valid_syn = neuron_obj.synapses_valid
    if verbose:
        print(f"# of valid synapses = {len(valid_syn)}")
    syu.plot_synapses_objs(neuron_obj,
                  valid_syn,
                      synapse_size=synapse_size)
def plot_synapses_error_from_neuron_obj(neuron_obj,
                                       synapse_size = 0.3,
                                        verbose = True
                                      ):
    valid_syn = neuron_obj.synapses_error
    if verbose:
        print(f"# of error synapses = {len(valid_syn)}")
    syu.plot_synapses_objs(neuron_obj,
                  valid_syn,
                      synapse_size=synapse_size)
    
def synapse_compartment_spine_type_title(
    compartment_label=None,
    spine_label = None,
    syn_type = None,
    add_n_syn_to_title = False,
    verbose = False
    ):
    c = compartment_label
    sp_l= spine_label
    syn_l = syn_type
    
    
    title_str = [k for k in [c,sp_l,syn_l] if k is not None]
    if len(title_str)>1: 
        title = "_".join(title_str)
    else:
        title = title_str[0]
    if verbose:
        print(f"  {title}: {n_syns}")
        
    if add_n_syn_to_title:
        title = f"n_syn_{title}"
        
    return title

def synapses_by_compartment_spine_type(
    neuron_obj,
    compartment_label = None,
    spine_label = None,
    syn_type = None,
    
    plot_synapses=False,
    synapse_size=0.2,
    
    verbose = False,
    return_title = False,
    add_n_syn_to_title = False,
    **kwargs):
    """
    Purpose: To be able to 
    get the synapses of any specification
    of compartment and head_neck_shaft

    Pseudocode: 
    1) Get the spine int label
    2) Get the compartment string label
    3) Decide whether should be presyn or postsyn
    4) Assemble query
    5) Query the synapses
    6) Return the synapses

    """
    compartment_label_original = copy.copy(compartment_label)
    spine_label_original = copy.copy(spine_label)
    syn_type_original = copy.copy(syn_type)
    
    compartment_query = None
    if compartment_label is not None:
        if not nu.is_array_like(compartment_label):
            compartment_label = apu.compartment_label_to_all_labels(compartment_label)

        compartment_query = f"(compartment in {compartment_label})"

    spine_query = None

    if spine_label is not None:
        if not nu.is_array_like(spine_label):
            spine_label = [spine_label]

        spine_label = [spu.spine_int_label(k) for k in spine_label]

        spine_query = f"(head_neck_shaft in {spine_label})"

    syn_type_query = None

    if syn_type is None and compartment_label is not None:
        if len(np.intersect1d(apu.dendrite_labels,compartment_label)) > 0:
            syn_type = ["postsyn"]
#         elif len(np.intersect1d(apu.dendrite_labels,["axon"])) > 0:
#             syn_type = ["presyn"]
        else:
            syn_type = ["presyn","postsyn"]
    elif type(syn_type) == str:
        syn_type = nu.convert_to_array_like(syn_type)
    elif syn_type is None or None in syn_type:
        syn_type = ["presyn","postsyn"]
    else:
        syn_type = nu.convert_to_array_like(syn_type)

    syn_type_query = f"(syn_type in {syn_type})"


    if verbose:
        print(f"compartment_query = {compartment_query}")
        print(f"spine_query = {spine_query}")
        print(f"syn_type_query = {syn_type_query}")

    non_none_queries = [k for k in [compartment_query,spine_query,syn_type_query] if k is not None]

    if len(non_none_queries) == 0:
        raise Exception("No non None queries")
    elif len(non_none_queries) == 1:
        final_query = non_none_queries[0]
    else:
        final_query = " and ".join(non_none_queries)

    if verbose:
        print(F"final_query = {final_query}")

    returned_syns =syu.query_synapses(neuron_obj,
                      query=final_query,
                       return_synapses=True,
                      **kwargs)

    if verbose:
        print(f"# of synapses in query = {len(returned_syns)}")

    if plot_synapses:
        if len(returned_syns) > 0:
            syu.plot_synapses_objs(neuron_obj,
                             returned_syns,
                                  synapse_size=synapse_size)
        else:
            print(f"No synapses to plot")

            
    if return_title:
        title = syu.synapse_compartment_spine_type_title(
            compartment_label = compartment_label_original ,
            spine_label = spine_label_original ,
            syn_type = syn_type_original ,
            add_n_syn_to_title = add_n_syn_to_title
        )
        return returned_syns,title
    else:
        return returned_syns

def n_synapses_by_compartment_spine_type(
    neuron_obj,
    compartment_label = None,
    spine_label = None,
    syn_type = None,
    
    plot_synapses=False,
    synapse_size=0.2,
    
    verbose = False,
    
    return_title = False,
    add_n_syn_to_title = True,
    **kwargs):

    """
    Return the number of synapses
    of a certain type
    """
    return_value = syu.synapses_by_compartment_spine_type(
    neuron_obj,
    compartment_label = compartment_label,
    spine_label = spine_label,
    syn_type = syn_type,
    
    plot_synapses=plot_synapses,
    synapse_size=synapse_size,
        
    return_title = return_title,
    add_n_syn_to_title = add_n_syn_to_title,
    
    verbose = verbose,
    **kwargs)
    
    if return_title:
        return len(return_value[0]),return_value[1]
    else:
        return len(return_value)

    
        
def n_synapses_all_compartment_spine_type(neuron_obj,
                                     compartment_labels = None,
                                     spine_labels = None,
                                     syn_types = None,
                                     verbose = False,
                                      return_synapse_objs = False,
                                      add_n_syn_in_keys = True,
                                     **kwargs):
    """
    Purpose: To get all combinations
    of compartments, spine labels and synapse types
    that should be computed

    """

    synapse_dict = dict()

    if compartment_labels is None:
        compartment_labels = apu.compartment_labels_for_synapses_stats()

    # if spine_labels is None:
    #     spine_labels = spu.spine_labels()

    # if syn_types is None:
    #     syn_types = ["presyn","postsyn"]

    if verbose:
        print(f"compartment_labels = {compartment_labels}")
    #     print(f"spine_labels = {spine_labels}")
    #     print(f"syn_types = {syn_types}")

    all_combs = []

    if not return_synapse_objs:
        func = syu.n_synapses_by_compartment_spine_type
    else:
        func = syu.synapses_by_compartment_spine_type
    
    for c in compartment_labels:
        cp = c
        if verbose:
            print(f"--- Working on comparment {c} ---")
            
        if spine_labels is None:
            c_spine_labels = apu.spine_labels_from_compartment(c)
        else:
            c_spine_labels = spine_labels
            
        if syn_types is None: 
            c_syn_type = apu.syn_type_from_compartment(c)
        else:
            c_syn_type = syn_types

        for sp_l in c_spine_labels:
            for syn_l in c_syn_type:
                #print(f"{c}_{sp_l}_{syn_l}")
                n_syns,title = func(
                    neuron_obj,
                    compartment_label = c,
                    spine_label = sp_l,
                    syn_type = syn_l,
                    plot_synapses = False,
                    verbose = False,
                    return_title = True,
                    add_n_syn_to_title = add_n_syn_in_keys
                    )
                

                synapse_dict[f"{title}"] = n_syns

#     if add_n_syn_in_keys:
#         synapse_dict = {f"n_syn_{k}":v for k,v in synapse_dict.items()}

    return synapse_dict

def n_synapses_all_compartments(neuron_obj,
                                     compartment_labels = None,
                                     verbose = False,
                                      return_synapse_objs = False,
                                     **kwargs):
    """
    Purpose: To get all combinations
    of compartments, spine labels and synapse types
    that should be computed

    """
    return n_synapses_all_compartment_spine_type(neuron_obj,
                                     compartment_labels = compartment_labels,
                                     spine_labels = [None],
                                     syn_types = [None],
                                     verbose = verbose,
                                      return_synapse_objs = return_synapse_objs,
                                     **kwargs)

def n_synapses_all_spine_labels(neuron_obj,
                                     compartment_labels = None,
                                     verbose = False,
                                      return_synapse_objs = False,
                                     **kwargs):
    """
    Purpose: To get all combinations
    of compartments, spine labels and synapse types
    that should be computed

    """
    return n_synapses_all_compartment_spine_type(neuron_obj,
                                     compartment_labels = [None],
                                     spine_labels = spu.spine_labels(include_no_label=True),
                                     syn_types = [None],
                                     verbose = verbose,
                                      return_synapse_objs = return_synapse_objs,
                                     **kwargs)

def synapses_mesh_errored(neuron_obj):
    return syu.query_synapses(neuron_obj,
                      query="label == 'mesh_errored'",
                      return_synapse_objs=True)
def synapses_distance_errored(neuron_obj):
    return syu.query_synapses(neuron_obj,
                      query="label == 'distance_errored'",
                      return_synapse_objs=True)
def n_synapses_mesh_errored(neuron_obj):
    return len(syu.synapses_mesh_errored(neuron_obj))

def n_synapses_distance_errored(neuron_obj):
    return len(syu.synapses_distance_errored(neuron_obj))

def n_synapses_all_valid_error(neuron_obj):
    return dict(
        n_syn_valid = syu.n_synapses_valid(neuron_obj),
        n_syn_valid_pre = syu.n_synapses_valid_pre(neuron_obj),
        n_syn_valid_post = syu.n_synapses_valid_post(neuron_obj),
        n_syn_error = syu.n_synapses_error(neuron_obj),
        n_syn_error_pre = syu.n_synapses_error_pre(neuron_obj),
        n_syn_error_post = syu.n_synapses_error_post(neuron_obj),
        n_syn_presyns_on_dendrite = syu.n_presyns_on_dendrite(neuron_obj),
        n_syn_mesh_errored=syu.n_synapses_mesh_errored(neuron_obj),
        n_syn_distance_errored=syu.n_synapses_distance_errored(neuron_obj),
        
    )

def complete_n_synapses_analysis(neuron_obj,
                                include_axon_ais_syn = True):
    syn_valid_error_dict = syu.n_synapses_all_valid_error(neuron_obj)
    syn_spine_dict = syu.n_synapses_all_spine_labels(neuron_obj)
    syn_compartment_dict = syu.n_synapses_all_compartments(neuron_obj)
    syn_comp_spine_type_dict = syu.n_synapses_all_compartment_spine_type(neuron_obj,
                                                                    verbose = False)
    
    syn_dict = dict()
    if include_axon_ais_syn:
        syn_dict["n_syn_axon_ais_postsyn"] = syu.n_axon_ais_synapses(neuron_obj)
        
    return gu.merge_dicts([syn_valid_error_dict,
                           syn_spine_dict,
                           syn_compartment_dict,
                           syn_comp_spine_type_dict,
                          syn_dict])

def n_synapses_analysis_axon_dendrite(
    neuron_obj,
    verbose = True,
    include_axon_ais_syn = True,
    include_soma_syn = True
    ):
    """
    Puporse: calculating synapses
    """
    
    syn_dict = syu.n_synapses_all_compartment_spine_type(
    neuron_obj,
    compartment_labels=["axon","dendrite"],
    verbose = verbose
    )
    
    if include_axon_ais_syn:
        syn_dict["n_syn_axon_ais_postsyn"] = syu.n_axon_ais_synapses(neuron_obj)
        
    if include_soma_syn:
        syn_dict["n_syn_soma_postsyn"] = syu.n_synapses_somas_postsyn(neuron_obj)
    
    return syn_dict



synapse_attribute_to_dj_key_map = dict(
syn_type="synapse_type",
syn_id="synapse_id",
soma_distance = "skeletal_distance_to_soma",
)

dj_key_to_synapse_attribute_map = gu.invert_mapping(synapse_attribute_to_dj_key_map,one_to_one=True)
dj_key_to_synapse_attribute_map

def synapse_obj_from_dj_synapse_dict(synapse_dict,
                       ):
    """
    Purpose: To convert a list of dictionaries
    to synapses objects
    
    Pseudocode: 
    1) convert the compartment_fine to just one label
    2) convert the spine_bouton to the number
    3) Send the new dictionary to a spine object
    """
    if "compartment_fine" in synapse_dict.keys():
        if synapse_dict["compartment_fine"] is not None:
            compartment = synapse_dict["compartment_fine"] 
        else:
            compartment = synapse_dict["compartment_coarse"] 
    else:
        compartment =  "error"
        
    if "spine_bouton" in synapse_dict.keys():   
        head_neck_shaft = spu.spine_int_label(synapse_dict["spine_bouton"] )
    else:
        head_neck_shaft = None
        
    if "skeletal_distance_to_soma" not in synapse_dict.keys():
        synapse_dict["skeletal_distance_to_soma"] = -1
    
    new_key = {k:v for k,v in synapse_dict.items() if k not in ["compartment_coarse",
                                                               "compartment_fine",
                                                               "spine_bouton"]}
    
    new_key.update(dict(compartment=compartment,head_neck_shaft = head_neck_shaft,))
    for k,v in dj_key_to_synapse_attribute_map.items():
        if k in new_key.keys():
            new_key[v] = new_key[k]
    
    #return new_key
    return syu.Synapse(**new_key)


def synapses_obj_groups_from_queries(synapses_obj,
                                    queries,
                                    verbose = False):
    """
    Purpose; Will divide synapses ob into groups based on 
    a xeries of queries
    """
    return_syns = [syu.query_synapses(synapses_obj,q,
                                      verbose=verbose,
                                     return_synapses=True) for q in queries]
    return return_syns

def valid_error_groups_from_synapses_obj(synapses_obj):
    syn_groups = syu.synapses_obj_groups_from_queries(synapses_obj,
                                    queries = [syu.valid_query(),
                                              syu.error_query()]) 
    return syn_groups



def compartment_groups_from_synapses_obj(synapses_obj,
                                        compartments=None,
                                        verbose = False):
    if compartments is None:
        compartments = apu.compartments_to_plot
    queries = [f"compartment == '{c}'" for c in compartments]
    
    if verbose:
        print(f"For compartments: {compartments}")
        
    return syu.synapses_obj_groups_from_queries(synapses_obj,
                                    queries = queries,
                                    verbose = verbose)

def spine_bouton_groups_from_synapses_obj(synapses_obj,
                                         spine_bouton_labels = None,
                                         verbose = False):
    if spine_bouton_labels is None:
        spine_bouton_labels = spu.spine_labels(include_no_label=True)
        
    queries = [f"head_neck_shaft == '{spu.spine_int_label(c)}'" for c in spine_bouton_labels]
    
    if verbose:
        print(f"For spine_bouton_labels: {spine_bouton_labels}")
        
    return syu.synapses_obj_groups_from_queries(synapses_obj,
                                    queries = queries,
                                    verbose = verbose)

def presyn_postsyn_groups_from_synapses_obj(synapses_obj,
                                           verbose = False):
    return [syu.synapses_pre(synapses_obj),syu.synapses_post(synapses_obj)]

def synapse_plot_items_by_type_or_query(
    synapses_objs,
    synapses_size = 0.15,
    synapse_plot_type = "spine_bouton",#"compartment"#  "valid_error", "soma"
    synapse_compartments = None,
    synapse_spine_bouton_labels = None,
    plot_error_synapses = True,
    valid_synapses_color = "orange",
    error_synapses_color = "aliceblue",
    synapse_queries = None,
    synapse_queries_colors = None,

    verbose = False,
    print_spine_colors = True):
    
    if synapse_compartments is None:
        synapse_compartments = apu.compartments_to_plot
    
    """
    Purpose: will  
    
    """
    if synapse_spine_bouton_labels is None:
        synapse_spine_bouton_labels = spu.spine_bouton_labels_to_plot()
    def print_color_dict(color_dict):
        print(f"Synapse Colors:")
        for k,v in color_dict.items():
            print(f"  {k}:{v}")
    
    already_plotted_errors = False

    if synapse_queries is None:
        if synapse_plot_type == "valid_error":
            synapse_groups = syu.valid_error_groups_from_synapses_obj(synapses_objs)
            synapse_colors = [valid_synapses_color,error_synapses_color]
            
            
            color_dict = dict(valid=valid_synapses_color,error=error_synapses_color)
            
            already_plotted_errors = True
        elif synapse_plot_type == "compartment":
            synapse_groups = syu.compartment_groups_from_synapses_obj(synapses_objs,
                                                     compartments = synapse_compartments,
                                                    verbose=verbose)
            synapse_colors = apu.colors_from_compartments(synapse_compartments)
            color_dict = {k:v for k,v in zip(synapse_compartments,synapse_colors)}
        elif synapse_plot_type == "spine_bouton":
            synapse_groups = syu.spine_bouton_groups_from_synapses_obj(
                synapses_objs,
                spine_bouton_labels = synapse_spine_bouton_labels,
                verbose = verbose)
            synapse_colors = spu.colors_from_spine_bouton_labels(synapse_spine_bouton_labels)
            color_dict = {k:v for k,v in zip(synapse_spine_bouton_labels,synapse_colors)}
        elif synapse_plot_type == "valid_presyn_postsyn":
            valid_syns = syu.synapses_obj_groups_from_queries(synapses_objs,[syu.valid_query()])[0]
            synapse_groups = presyn_postsyn_groups_from_synapses_obj(valid_syns)
            color_dict = dict(presyn="yellow",postsyn="blue")
            synapse_colors = ["yellow","blue"]
        elif synapse_plot_type == "soma":
            valid_syns = syu.synapses_obj_groups_from_queries(synapses_objs,["compartment == 'soma'"])[0]
            synapse_groups = presyn_postsyn_groups_from_synapses_obj(valid_syns)
            print(f"# of syn_soma_pre = {len(synapse_groups[0])}" + f", # of syn_soma_post = {len(synapse_groups[1])}")
            color_dict = dict(presyn="yellow",postsyn="blue")
            synapse_colors = ["yellow","blue"]
        else:
            raise Exception(f"Unimplemented synapse_plot_type = {synapse_plot_type}")
    else:
        synapse_groups = syu.synapses_obj_groups_from_queries(synapses_objs,
                                                             queries=synapse_queries,
                                                             verbose = verbose)
        synapse_colors = synapse_queries_colors
        color_dict = {f"query_{i+1}":c for i,c in enumerate(synapse_colors)}

    if not already_plotted_errors and plot_error_synapses:
        error_syn = syu.synapses_error(synapses_objs)
        synapse_groups.append(error_syn)
        synapse_colors.append(error_synapses_color)
        color_dict["error"] = error_synapses_color
        
    if print_spine_colors:
        print_color_dict(color_dict)
        
    scatters = [syu.synapses_to_coordinates(k) for k in synapse_groups]
    scatters_colors = synapse_colors
    scatters_sizes = [synapses_size]*len(synapse_groups)
    
    
    return scatters,scatters_colors,scatters_sizes


# --------------- 10/25 --------------------#
def calculate_neuron_soma_distance_euclidean(neuron_obj,
                                  verbose  =False,
                                  store_soma_placeholder = True,
                                  store_error_placeholder = True):
    """
    Purpose: To calculate all of the soma distances for all the valid synapses
    on limbs
    
    Ex: 
    calculate_neuron_soma_distance(neuron_obj,
                              verbose = True)
    """
    st = time.time()
    soma_center = neuron_obj["S0"].mesh_center
    
    for syn in neuron_obj.synapses:
        syn.soma_distance_euclidean = np.linalg.norm(soma_center - syn.coordinate)
        
    if store_soma_placeholder:
        if verbose:
            print(f"Putting Soma Placeholders")
        for s_idx in neuron_obj.get_soma_indexes():
            st_loc = time.time()
            soma_synapses = neuron_obj[f"S{s_idx}"].synapses
            for syn in soma_synapses:
                syn.soma_distance_euclidean = -1*(nru.soma_face_offset + s_idx)
            if verbose:
                print(f"\n--- Soma {s_idx} soma calculation time = {np.round(time.time() - st_loc,3)}")
                
    if store_error_placeholder:
        st_loc = time.time()
        if verbose:
            print(f"Putting Error Placeholders")
        for syn in neuron_obj.synapses_error:
            syn.soma_distance_euclidean = -1
        if verbose:
            print(f"\n--- Error soma calculation time = {np.round(time.time() - st_loc,3)}")
                
                
    if verbose:
        print(f"Total soma distance calculation time = {time.time() - st}")
        
        
# -------- attempting to get ais synapses -------
def axon_ais_synapses(
    neuron_obj,
    max_ais_distance_from_soma = None,
    plot = False,
    verbose = False,
    return_synapses = False,
    **kwargs
    ):
    
    """
    Purpose: to get the postsyns synapses of those
    on the axon within a certain distance of 
    the soma (so ideally the ais)
    
    Ex: 
    """
    if max_ais_distance_from_soma is None:
        max_ais_distance_from_soma = au.max_ais_distance_from_soma
    
    syu.calculate_neuron_soma_distance(neuron_obj)
    
    curr_syns = syu.query_synapses(neuron_obj,
                query="(compartment == 'axon') and (syn_type=='postsyn')"
                    f" and (soma_distance < {max_ais_distance_from_soma})",
                return_synapses=return_synapses,
                plot=plot)
    
    if verbose:
        print(f"# of AIS postsyns = {len(curr_syns)}")
        
    return curr_syns

def n_axon_ais_synapses(
    neuron_obj,
    plot = False,
    verbose = False,
    **kwargs
    ):
    return len(syu.axon_ais_synapses(
            neuron_obj,
            plot = plot,
            verbose = verbose,
            **kwargs
            ))



def adjust_obj_with_face_offset(
    synapse_obj,
    face_offset,
    attributes_not_to_adjust = ("closest_face_idx",),
    verbose = False,
    ):
    """
    Purpose: To adjust the spine properties that
    would be affected by a different face idx
    
    Ex: 
    b_test = neuron_obj[0][18]
    sp_obj = b_test.spines_obj[0]
    sp_obj.export()

    spu.adjust_spine_obj_with_face_offset(
        sp_obj,
        face_offset = face_offset,
        verbose = True
    ).export()
    """
    new_obj = copy.deepcopy(synapse_obj)
    for k,v in synapse_obj.export().items():
        if "face_idx" not in k:
            continue
        
        if v is None:
            continue
        if k in attributes_not_to_adjust:
            #print(f"skipping {v} to not adjust")
            continue
            
        if verbose:
            print(f"Adjusting {k} because face_idx and not None")
            
        setattr(new_obj,k,v + face_offset)
        
    return new_obj


def endpoint_dist_extrema_over_syn(
    branch_obj,
    endpoint_type="downstream",
    extrema_type="max",
    default_dist = 100000000,
    verbose = False,
    **kwargs):
    
    syns = branch_obj.synapses
    if len(syns) > 0:
        dists = np.array([getattr(k,f"{endpoint_type}_dist") for k in syns])
        return_value = getattr(np,extrema_type)(dists)
    else:
        return_value = default_dist
        
    if verbose:
        print(f"{extrema_type} {endpoint_type} dist = {return_value}")
        
    return return_value

def downstream_dist_min_over_syn(
    branch_obj,
    verbose = False,
    **kwargs
    ):
    
    return syu.endpoint_dist_extrema_over_syn(
    branch_obj,
    verbose = verbose,
    endpoint_type="downstream",
    extrema_type="min",
    **kwargs
    )

def downstream_dist_max_over_syn(
    branch_obj,
    verbose = False,
    **kwargs
    ):
    
    return syu.endpoint_dist_extrema_over_syn(
    branch_obj,
    verbose = verbose,
    endpoint_type="downstream",
    extrema_type="max",
    **kwargs
    )

def synapse_coordinates_from_synapse_df(df):
    return np.vstack(df["coordinate"].to_numpy()).reshape(-1,3)

def axon_on_dendrite_synapses(
    neuron_obj,
    plot_limb_branch = False,
    verbose = False,
    ):
    ax_on_d_lb = pru.axon_on_dendrite_plus_downstream(
        neuron_obj,
        plot = plot_limb_branch,
    )

    axon_on_dendrite_synapses = syu.synapses_over_limb_branch_dict(
        neuron_obj,
        ax_on_d_lb
    )

    if verbose:
        print(f"# of axon on dendrite merge synapses: {len(axon_on_dendrite_synapses)}")

    return axon_on_dendrite_synapses

def presyn_on_dendrite_synapses_after_axon_on_dendrite_filter_away(
    neuron_obj,
    axon_on_dendrite_limb_branch_dict = None,
    plot = False,
    **kwargs
    ):
    
    if axon_on_dendrite_limb_branch_dict is None:
        axon_on_dendrite_limb_branch_dict = ax_on_d_lb = pru.axon_on_dendrite_plus_downstream(
            neuron_obj,
            plot = False,
        )
        
    lb = nru.limb_branch_invert(neuron_obj,axon_on_dendrite_limb_branch_dict)
        
    if len(lb) == 0:
        return np.array([])
    return syu.query_synapses(
        neuron_obj,
        query = syu.presyns_on_dendrite_query,
        limb_branch_dict=lb,
        plot = plot,
        **kwargs
    )
    
def synapses_to_dj_keys(
        self,
        neuron_obj,
        valid_synapses = True,
        verbose = False,
        nucleus_id = None,
        split_index = None,
        output_spine_str=True,
        add_secondary_segment = True,
        ver=None,
        synapses = None,
        key = None):
        """
        Pseudocode: 
        1) Get either the valid of invalid synapses
        2) For each synapses export the following as dict

        synapse_id=syn,
        synapse_type=synapse_type,
        nucleus_id = nucleus_id,
        segment_id = segment_id,
        split_index = split_index,
        skeletal_distance_to_soma=np.round(syn_dist[syn_i],2),

        return the list
        
        Ex: 
        import numpy as np
        dj_keys = syu.synapses_to_dj_keys(neuron_obj,
                            verbose = True,
                            nucleus_id=12345,
                            split_index=0)
        np.unique([k["compartment"] for k in dj_keys],return_counts=True)
        
        
        Ex:  How to get error keys with version:
        dj_keys = syu.synapses_to_dj_keys(neuron_obj,
                                    valid_synapses = False,
                        verbose = True,
                        nucleus_id=12345,
                        split_index=0,
                                    ver=158)

        """
        segment_id = neuron_obj.segment_id
        
        if nucleus_id is not None:
            neuron_obj.nucleus_id = nucleus_id
        if split_index is not None:
            neuron_obj.split_index = split_index
        
        if synapses is not None:
            if valid_synapses:
                export_func = syu.synapses_dj_export_dict_valid 
            else:
                export_func = syu.synapses_dj_export_dict_error
        elif valid_synapses:
            synapses = neuron_obj.synapses_valid
            export_func = syu.synapses_dj_export_dict_valid 
        else:
            synapses = neuron_obj.synapses_error
            export_func = syu.synapses_dj_export_dict_error
            
        for att in ["nucleus_id","segment_id","split_index"]:
    #         globs = globals()
    #         locs = locals()
            if getattr(neuron_obj,att) is None and eval(att) is None:
                raise Exception(f"{att} is None")
        st = time.time()

        syn_keys = [dict(export_func(syn,output_spine_str=output_spine_str),
                                    primary_nucleus_id=getattr(neuron_obj,"nucleus_id",nucleus_id),
                        primary_seg_id=neuron_obj.segment_id,
                        split_index = getattr(neuron_obj,"split_index",split_index)) for syn in synapses]
        
        #adding on the secondary seg
        """
        Purpose: To add on the secondary 
        
        """
        if add_secondary_segment and len(syn_keys) > 0:
            syn_keys_df = pd.DataFrame(syn_keys)
            
            df = self.segment_id_to_synapse_table_optimized(
                neuron_obj.segment_id,
                return_df = True)
            df_secondary = df[["synapse_id","secondary_seg_id"]]
            syn_keys= pu.df_to_dicts(pd.merge(syn_keys_df,df_secondary,on="synapse_id"))
                
        
        if verbose:
            print(f"valid_synapses = {valid_synapses}")
            print(f"Time for {len(syn_keys)} synapse entries = {time.time() - st}")
        
        if ver is not None:
            syn_keys = [dict(k,ver=ver) for k in syn_keys]
        if key is not None:
            syn_keys = [gu.merge_dicts([k,key.copy()]) for k in syn_keys]
        
        return syn_keys


def synapse_df_from_synapse_dict(
    synapse_dict,
    segment_id = None,
    ):
    """
    Purpose: convert synapse dict into a dataframe

    """
    all_results = []
    for prepost,p_data in synapse_dict.items():
        for syn_id,syn_coord,syn_size in zip(
            p_data["synapse_ids"],
            p_data["synapse_coordinates"],
            p_data["synapse_sizes"],
            ):
            x,y,z = syn_coord
            curr_dict = dict(
                prepost = prepost,
                synapse_id = syn_id,
                synapse_x = x,
                synapse_y = y,
                synapse_z = z,
                synapse_size = syn_size
            )
            
            all_results.append(curr_dict)
            
            
    df = pd.DataFrame.from_records(all_results)
    if segment_id is not None:
        df['segment_id'] = segment_id
        
    return df

def add_nm_to_synapse_df(
    df,
    scaling,
    ):
    df[
        ["synapse_x_nm",'synapse_y_nm','synapse_z_nm']
    ] = df[
        ["synapse_x",'synapse_y','synapse_z']
    ].to_numpy() * scaling
    
    df["synapse_size_nm"] = df["synapse_size"]*(scaling.prod())
    
    return df
def synapse_df_from_csv(
    synapse_filepath,
    segment_id = None,
    coordinates_nm = True,
    scaling = None,
    verbose = True,
    **kwargs,
    ):
    """
    Purpose: to read in a csv file 
    """
    if scaling is None:
        scaling = vdi.voxel_to_nm_scaling
    
    df = pu.csv_to_df(
        synapse_filepath
    )

    if segment_id is not None:
        df = df.query(f"segment_id == {segment_id}").reset_index(drop=True)
        
    if coordinates_nm:
        df = add_nm_to_synapse_df(
            df,
            scaling=scaling,
        )
    
    return df

def synapse_dict_from_synapse_df(
    df,
    scaling = None,
    verbose = True,
    coordinates_nm = True,
    syn_types = ["presyn","postsyn"],
    **kwargs
    ):
    
    if coordinates_nm:
        syn_coord_names = ["synapse_x_nm",'synapse_y_nm','synapse_z_nm','synapse_size_nm']
    else:
        syn_coord_names = ["synapse_x",'synapse_y','synapse_z',"synapse_size"]

    synapse_dict = dict()
    for synapse_type in syn_types:
        df_curr = df.query(f"prepost == '{synapse_type}'").reset_index(drop=True)
        synapse_ids, centroid_xs, centroid_ys, centroid_zs,synapse_sizes = df_curr[
            ["synapse_id"] + syn_coord_names
        ].to_numpy().T
        if len(synapse_ids) > 0:
            synapse_centers = np.vstack([centroid_xs,centroid_ys,centroid_zs]).T
            if scaling is not None:
                synapse_centers = synapse_centers* scaling
        else:
            synapse_centers = np.array([])
            synapse_ids = np.array([])
            synapse_sizes = np.array([])

        synapse_dict[synapse_type] = dict(
            synapse_ids = synapse_ids,
            synapse_coordinates= synapse_centers,
            synapse_sizes = synapse_sizes
        )

        if verbose:
            print(f"# of {synapse_type}: {len(synapse_dict[synapse_type]['synapse_coordinates'] )}")
            
    return synapse_dict
    

def synapse_dict_from_synapse_csv(
    synapse_filepath,
    segment_id = None,
    scaling = None,
    verbose = True,
    coordinates_nm = True,
    **kwargs
    ):
    """
    Purpose: to injest the synapse information for a segment id in some manner

    Example implementation: injesting synapse inforation from a csv

    Pseudocode: 
    1) Read in the csv
    2) Filter to the segment id
    3) Creates the prepost dictionary to be filled:
    Iterates through pre and post (call preprost):
        a) filters for certain prepost
        b) Gets the synapse id, x, y, z and synapse size
        c) Stacks the syz and scales them if need
        d) Stores all data in dictionary for that prepost

    """
    df = synapse_df_from_csv(
        synapse_filepath,
        segment_id = segment_id,
        coordinates_nm = coordinates_nm,
        scaling = scaling,
        verbose = verbose,
    )
    
    return synapse_dict_from_synapse_df(
        df,
        scaling = None,
        verbose = verbose,
        coordinates_nm = coordinates_nm,
        **kwargs
        )
    


def fetch_synapse_dict_by_mesh_labels(
        segment_id,
        mesh,
        synapse_dict = None,
        original_mesh = None,
        original_mesh_kd = None,
        validation = False,
        verbose = False,
        original_mesh_method = True,
        mapping_threshold = 500,
        
        plot_synapses=False,
        plot_synapses_type = None,
        **kwargs):
        """
        Purpose: To return a synapse dictionary mapping

        type (presyn/postsyn)---> mesh_label  --> list of synapse ids

        for a segment id based on which original mesh face
        the synapses map to

        Pseudocode: 
        1) get synapses for the segment id
        Iterate through presyn and postsyn
            a. Find the errors because of distance
            b. Find the errors because of mesh cancellation
            c. Find the valid mesh (by set difference)
            store all error types in output dict
        2) Plot the synapses if requested
        
        Example:
        mesh_label_dict = syu.fetch_synapse_dict_by_mesh_labels(
            segment_id=o_neuron.segment_id,
            mesh = nru.neuron_mesh_from_branches(o_neuron),
            original_mesh = du.fetch_segment_id_mesh(segment_id),
            validation=True,
            plot_synapses=True,
            verbose = True)
        """

        if synapse_dict is None:
            synapse_dict = vdi.segment_id_to_synapse_dict(
                segment_id,
                validation=validation,
                verbose=verbose,
                **kwargs)

        mesh_label_dict = dict()


        for synapse_type in ["presyn","postsyn"]:
            if verbose:
                print(f"-- Working on {synapse_type}--")
            synapse_centers_scaled = synapse_dict[synapse_type]["synapse_coordinates"]
            synapse_ids = synapse_dict[synapse_type]["synapse_ids"]

            distance_errored_syn_idx = np.array([],dtype="int")
            mesh_errored_syn_idx = np.array([],dtype="int")
            valid_syn_idx = np.array([],dtype="int")

            local_label_dict = dict()

            if len(synapse_centers_scaled) > 0:
                if not original_mesh_method:

                    distance_errored_syn_idx = tu.valid_coordiantes_mapped_to_mesh(mesh=mesh,
                                            coordinates=synapse_centers_scaled,
                                            mapping_threshold = mapping_threshold,
                                            return_idx = True,
                                            return_errors = True)
                    valid_syn_idx = np.delete(np.arange(len(synapse_centers_scaled)),distance_errored_syn_idx)

                else:
                    """
                    Pseudocode:

                    """
                    if verbose:
                        print(f"Using original_mesh_method")
                        
                    if original_mesh is None:
                        original_mesh = vdi.fetch_segment_id_mesh(segment_id)
                    distance_errored_syn_idx = tu.valid_coordiantes_mapped_to_mesh(mesh=original_mesh,
                                            coordinates=synapse_centers_scaled,
                                            mapping_threshold = mapping_threshold,
                                            return_idx = True,
                                            return_errors = True)

                    mesh_errored_syn_idx_pre = tu.valid_coordiantes_mapped_to_mesh(mesh=mesh,
                                                                            original_mesh = original_mesh,
                                                                            original_mesh_kdtree=original_mesh_kd,
                                            coordinates=synapse_centers_scaled,
                                            mapping_threshold = mapping_threshold,
                                            return_idx = True,
                                            return_errors = True)
                    mesh_errored_syn_idx = np.setdiff1d(mesh_errored_syn_idx_pre,distance_errored_syn_idx)


                    total_error_syn_idx = np.hstack([distance_errored_syn_idx,mesh_errored_syn_idx])
                    valid_syn_idx = np.delete(np.arange(len(synapse_centers_scaled)),total_error_syn_idx)

                if verbose:
                    print(f"# of distance_errored_syn_idx = {len(distance_errored_syn_idx)}")
                    print(f"# of mesh_errored_syn_idx = {len(mesh_errored_syn_idx)}")
                    print(f"# of valid_syn_idx = {len(valid_syn_idx)}")

            local_label_dict["distance_errored"] = synapse_ids[distance_errored_syn_idx]
            local_label_dict["mesh_errored"] = synapse_ids[mesh_errored_syn_idx]
            local_label_dict["valid"] = synapse_ids[valid_syn_idx]

            mesh_label_dict[synapse_type] = local_label_dict

        if plot_synapses:
            output_syn_dict = syu.synapse_dict_mesh_labels_to_synapse_coordinate_dict(synapse_mesh_labels_dict=mesh_label_dict,
                                                        synapse_dict=synapse_dict)
            syu.plot_valid_error_synpases(neuron_obj = None,
                                    synapse_dict=output_syn_dict,
                                    mesh = mesh,
                                    original_mesh = original_mesh,
                                    keyword_to_plot=plot_synapses_type)
        return mesh_label_dict

def synapse_df_abridged(
    neuron_obj,
    ):

    syn_df = syu.synapses_df(syu.synapses_valid(neuron_obj))
    syn_df_new = syn_df[["syn_id","syn_type","coordinate","volume","compartment","limb_idx","branch_idx","soma_distance"]]
    syn_df_new[["synapse_x_nm","synapse_y_nm","synapse_z_nm"]] = np.vstack(syn_df_new["coordinate"].to_numpy())
    syn_df_new = pu.rename_columns(
        syn_df_new,
        dict(
            syn_type = "prepost",
            syn_id = "synapse_id",
            volume = "synapse_size",
        )
    )

    return syn_df_new

# ------------- Setting up parameters -----------

# -- default
attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling,
    vdi = mvu.data_interface
)    
global_parameters_dict_default = dict(
    #max_ais_distance_from_soma = 50_000
)

# -- microns
global_parameters_dict_microns = {}
attributes_dict_microns = {}

#-- h01--
attributes_dict_h01 = dict(
    voxel_to_nm_scaling = hvu.voxel_to_nm_scaling,
    vdi = hvu.data_interface
)
global_parameters_dict_h01 = dict()
       
# data_type = "default"
# algorithms = None
# modules_to_set = [syu]

# def set_global_parameters_and_attributes_by_data_type(data_type,
#                                                      algorithms_list = None,
#                                                       modules = None,
#                                                      set_default_first = True,
#                                                       verbose=False):
#     if modules is None:
#         modules = modules_to_set
    
#     modu.set_global_parameters_and_attributes_by_data_type(modules,data_type,
#                                                           algorithms=algorithms_list,
#                                                           set_default_first = set_default_first,
#                                                           verbose = verbose)
    
# set_global_parameters_and_attributes_by_data_type(data_type,
#                                                    algorithms)

# def output_global_parameters_and_attributes_from_current_data_type(
#     modules = None,
#     algorithms = None,
#     verbose = True,
#     lowercase = True,
#     output_types = ("global_parameters"),
#     include_default = True,
#     algorithms_only = False,
#     **kwargs):
    
#     if modules is None:
#         modules = modules_to_set
    
#     return modu.output_global_parameters_and_attributes_from_current_data_type(
#         modules,
#         algorithms = algorithms,
#         verbose = verbose,
#         lowercase = lowercase,
#         output_types = output_types,
#         include_default = include_default,
#         algorithms_only = algorithms_only,
#         **kwargs,
#         )


#--- from neurd_packages ---
from . import apical_utils as apu
from . import axon_utils as au
from . import branch_attr_utils as bau
from . import concept_network_utils as cnu
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mvu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import proofreading_utils as pru
from . import spine_utils as spu

presyns_on_dendrite_query = ("(label=='limb_branch') and ((compartment=='dendrite') or "
                            f" (compartment in {apu.dendrite_compartment_labels()})) and (syn_type=='presyn')")

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import module_utils as modu 
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu

from . import neuron
from . import synapse_utils as syu