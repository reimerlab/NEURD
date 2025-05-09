

from datasci_tools import numpy_dep as np

def all_paths_to_leaf_nodes(
    limb_obj,
    verbose = False):
    return xu.all_paths_to_leaf_nodes(
            G = limb_obj.concept_network_directional,
            verbose = verbose,
        )
# ---------- statistics ----------------

default_relation_value = -1

def parent_skeletal_angle(
    limb_obj,
    branch_idx,
    verbose = False,
    default_value = None,
    skeletal_angle_attr = "skeleton_vector_[dir]",
    skeleton_attribute = "skeleton",
    **kwargs):
    """
    Purpose: to get the branching angle with parent
    from the skeleton vectors

    Pseudocode: 
    1) Get parent branch
    2) Get parent and child vector
    3) Get the angle between the two
    
    Ex: 
    from neurd import limb_utils as lu
    lu.parent_skeletal_angle(
    branch_idx = 2,
    limb_obj = neuron_obj[1],
    verbose = True,
    )
    """
    
    skeletal_angle_attr=skeletal_angle_attr.replace("skeleton",skeleton_attribute)
    
    upstream_attr = skeletal_angle_attr.replace('[dir]','upstream')
    downstream_attr = skeletal_angle_attr.replace('[dir]','downstream')
    
    if limb_obj[branch_idx].endpoints_upstream_downstream_idx is None:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
    
    parent_idx = nru.parent_node(limb_obj,branch_idx)
    if verbose:
        print(f"parent_idx = {parent_idx}")
    
    if  parent_idx is None:
        return default_value
    
    branch_obj = limb_obj[parent_idx]
    branch_obj_2 = limb_obj[branch_idx]
    return np.round(nu.angle_between_vectors(
        getattr(branch_obj,downstream_attr),
        getattr(branch_obj_2,upstream_attr)),2
    )
    
def parent_skeletal_angle_extra_offset(
    limb_obj,
    branch_idx,
    verbose = False,
    default_value = None,
    skeletal_angle_attr = "skeleton_vector_[dir]",
    **kwargs):
    return parent_skeletal_angle(
        limb_obj,
        branch_idx,
        verbose = verbose,
        default_value = default_value,
        skeletal_angle_attr = "skeleton_vector_[dir]_extra_offset",
        **kwargs)

def relation_skeletal_angle(
    limb_obj,
    branch_idx,
    relation,
    nodes_idx = None,
    default_value = None,
    verbose = False,
    extrema_value = None,
    return_dict = True,
    skeleton_attribute = "skeleton",
    **kwargs
    ):
    """
    Purpose: To find the sibling angles with all siblings
    """
    #print(f"nodes_idx = {nodes_idx}")
    
    if limb_obj[branch_idx].endpoints_upstream_downstream_idx is None:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
    
    if default_value is None:
        default_value = default_relation_value
    
    singular_flag = False
    
    if relation == "children":
        node_func = nru.children_nodes
        branch_vec_func = "skeleton_vector_downstream"
    elif relation == "siblings":
        node_func = nru.sibling_nodes
        branch_vec_func = "skeleton_vector_upstream"
    else:
        raise Exception("")
    
    branch_vec_func = branch_vec_func.replace("skeleton",skeleton_attribute)
    
    if nodes_idx is None:
        nodes_idx = node_func(limb_obj,branch_idx)
    elif not nu.is_array_like(nodes_idx):
        singular_flag = True
        nodes_idx = nu.convert_to_array_like(nodes_idx)
    else:
        pass
    
    if nodes_idx is None or len(nodes_idx) == 0:
        return default_value 

    sibling_idx = nu.convert_to_array_like(nodes_idx)
    
    if verbose:
        print(f"{relation}_idx= {sibling_idx}")
    
    branch_obj = limb_obj[branch_idx]
    sibling_angles = {}
    for s in sibling_idx:
        branch_obj_2 = limb_obj[s]
        curr_angle = np.round(nu.angle_between_vectors(
            getattr(branch_obj,branch_vec_func),
            branch_obj_2.skeleton_vector_upstream),2
        )
        if verbose:
            print(f"{branch_idx} and {s} angle: {curr_angle}")
        sibling_angles[s] = curr_angle
        
    if singular_flag and return_dict:
        sibling_angles = sibling_angles[sibling_idx[0]]
            
    if extrema_value:
        return getattr(np,extrema_value)(list(sibling_angles.values()))
        
    if return_dict:
        return sibling_angles
    else:
        return list(sibling_angles.values())

def siblings_skeletal_angle(
    limb_obj,
    branch_idx,
    sibling_idx = None,
    default_value = None,
    verbose = False,
    **kwargs
    ):
    
    return lu.relation_skeletal_angle(limb_obj,
    branch_idx,
    relation="siblings",
    nodes_idx = sibling_idx,
    default_value = default_value,
    verbose = verbose,
    **kwargs
    )

def children_skeletal_angle(
    limb_obj,
    branch_idx,
    nodes_idx = None,
    default_value = None,
    verbose = False,
    **kwargs
    ):
    
    return lu.relation_skeletal_angle(limb_obj,
    branch_idx,
    relation="children",
    nodes_idx = nodes_idx,
    default_value = default_value,
    verbose = verbose,
    **kwargs
    )

def siblings_skeletal_angle_max(
    limb_obj,
    branch_idx,
    **kwargs
    ):
    
    return lu.siblings_skeletal_angle(
        limb_obj,
        branch_idx,
        extrema_value = "max",
        **kwargs
    )

def siblings_skeletal_angle_min(
    limb_obj,
    branch_idx,
    **kwargs
    ):
    
    return lu.siblings_skeletal_angle(
        limb_obj,
        branch_idx,
        extrema_value = "min",
        **kwargs
    )

def children_skeletal_angle_max(
    limb_obj,
    branch_idx,
    **kwargs
    ):
    
    return lu.children_skeletal_angle(
        limb_obj,
        branch_idx,
        extrema_value = "max",
        **kwargs
    )

def children_skeletal_angle_min(
    limb_obj,
    branch_idx,
    **kwargs
    ):
    
    return lu.children_skeletal_angle(
        limb_obj,
        branch_idx,
        extrema_value = "min",
        **kwargs
    )


def sibling_angle_smooth(
    limb_obj,
    branch_1,
    branch_2,
    extra_offset = False,
    suppress_errors = True,
    default_value = -1,
    verbose=False,
    ):
    if extra_offset:
        attr = "skeleton_smooth_vector_upstream_extra_offset"
    else:
        attr = "skeleton_smooth_vector_upstream"

    if verbose:
        print(f"Angle attr = {attr}")
    obj1,obj2 = limb_obj[branch_1],limb_obj[branch_2]
    if obj1.endpoints_upstream_downstream_idx is None:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)

    try:
        v1 = getattr(obj1,attr)
        v2 = getattr(obj2,attr)
    except Exception as e:
        if suppress_errors:
            return default_value
        else:
            raise Exception(e)
    angle = nu.angle_between_vectors(v1,v2)
    return np.round(angle,2)

def sibling_angle_smooth_extra_offset(
    limb_obj,
    branch_1,
    branch_2,
    suppress_errors = True,
    default_value = -1,
    verbose=False,
    ):

    return sibling_angle_smooth(
        limb_obj,
        branch_1,
        branch_2,
        extra_offset = True,
        suppress_errors = suppress_errors,
        default_value = default_value,
        verbose=verbose,
    )

def most_usptream_endpoints_of_branches_on_limb(
    limb_obj,
    branches_idx,
    verbose = False,
    plot = False,
    scatter_size = 0.5,
    group_by_conn_comp = True,
    include_downstream_endpoint = True,
    **kwargs):
    """
    Purpose: To get all of the upstream endpoints of the
    connected components of a list of branches
    
    Pseudocode: 
    1) Get the connected components of the branches
    2) For each connected component find
        i) the most upstream branch
        ii) the upstream coordinate for that branch
            (could be a little offset of the upstream branch to prevent overal)
    """
    if len(branches_idx) == 0:
        return []
    
    if group_by_conn_comp:
        conn_comp = nru.connected_components_from_branches(
            limb_obj,
            branches=branches_idx,
        )
    else:
        conn_comp = [[k] for k in np.unique(branches_idx)]
    
    coordinates = []
    
    for i,cc in enumerate(conn_comp):
        if verbose:
            print(f"\n--Working on conn comp {i}: # of branches{len(cc)}")
        
        #i) the most upstream branch
        upstream_branch = nru.most_upstream_branch(limb_obj,cc)
        
        if verbose:
            print(f"upstream_branch = {upstream_branch}")
            
        #ii) the upstream coordinate for that branch
        #    (could be a little offset of the upstream branch to prevent overal)
        
        try:
            branch_obj = limb_obj[upstream_branch]
            branch_obj.endpoint_upstream
        except:
            bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
            branch_obj = limb_obj[upstream_branch]
        
        #coord = branch_obj.endpoint_upstream_with_offset
        sk_coords = branch_obj.skeletal_coordinates_upstream_to_downstream
        if include_downstream_endpoint:
            coord = np.array(sk_coords[[1,-1]])
        else:
            coord = np.array(sk_coords[[1]])
        
        if verbose:
            print(f"coord = {coord}")
        
        coordinates.append(coord)
    
    #coordinates = np.vstack(coordinates).reshape(-1,3)
    
    if plot:
        nviz.plot_objects(
            limb_obj.mesh,
            meshes = [limb_obj[k].mesh for k in branches_idx],
            meshes_colors = "blue",
            scatters=coordinates,
            scatters_colors="red",
            scatter_size=scatter_size,
            **kwargs)
        
    return coordinates

def most_upstream_endpoints_of_limb_branch(
    neuron_obj,
    limb_branch_dict,
    verbose = False,
    verbose_most_upstream=False,
    plot = False,
    return_array = False,
    group_by_conn_comp = True,
    include_downstream_endpoint = True,
    ):
    
    """
    Pseudocode:
    
    Ex: 
    lu.most_upstream_endpoints_of_limb_branch_conn_comp(
        neuron_obj,
        limb_branch_dict=dict(L1=[1,2],L2=[19,16]),
        verbose = False,
        verbose_most_upstream=False,
        plot = False,
        return_array = True,
        )
    
    """
    return_limb_coords = {}
    if len(limb_branch_dict) == 0:
        pass
    else:
        for limb_name,branches in limb_branch_dict.items():
            if verbose:
                print(f"\n---Working on limb {limb_name}: {branches}-----")
            coords = lu.most_usptream_endpoints_of_branches_on_limb(
                neuron_obj[limb_name],
                branches_idx=branches,
                verbose = verbose_most_upstream,
                plot = plot,
                group_by_conn_comp = group_by_conn_comp,
                include_downstream_endpoint = include_downstream_endpoint,)
            return_limb_coords[limb_name] = coords
    
    if return_array:
        if len(return_limb_coords) > 0:
            return np.vstack(list(return_limb_coords.values()))
        else:
            return np.array([])
    else:
        return return_limb_coords
    
def width_upstream(
    limb_obj,
    branch_idx,
    verbose = False,
    min_skeletal_length = 2000,
    skip_low_skeletal_length_upstream = True,
    default_value = 10000000):
    """
    Purpoose: To get the width of the upstream segement
    
    Pseudocode:
    1) Get the parent node
    2) Get the parent width
    
    Ex: 
    from neurd import limb_utils as lu
    lu.width_upstream(neuron_obj[1],5,verbose = True)
    """
    parent_node = nru.parent_node(limb_obj,branch_idx)
    
    if min_skeletal_length is None:
        min_skeletal_length = 0
        
    parent_sk_length = None
    width = default_value
    
    while parent_node is not None:
        parent_sk_length = limb_obj[parent_node].skeletal_length
        if parent_sk_length < min_skeletal_length:
            parent_node = nru.parent_node(limb_obj,parent_node)
        else:
            break

    if parent_node is not None:
        width = nru.width(limb_obj[parent_node])
        parent_sk_length = limb_obj[parent_node].skeletal_length
    
    if verbose:
        print(f"parent_node = {parent_node} (width = {width}, parent_sk_length = {parent_sk_length})")
        
    return width


def width_path_to_start(
    limb_obj,
    branch_idx,
    nodes_to_ignore = None,
    remove_zeros = True,
    width_func = None,
    verbose= False,
    skeletal_length_min = 0,
    remove_start_branch = True,
    return_branch_path = True,
    ):
    """
    To find the path from a branch to the start of the limb while accounting 
    for some branches being ing
    """
    if nodes_to_ignore is None:
        nodes_to_ignore = []
    if width_func is None:
        width_func = bu.width_max#au.axon_width

    path_to_start = nru.branch_path_to_start_node(limb_obj = limb_obj,
        branch_idx = branch_idx,
        include_branch_idx = False,
        skeletal_length_min = skeletal_length_min,
        include_last_branch_idx = remove_start_branch,
        verbose = False
    )

    # filtering the path to start
    path_to_start_filtered = np.array([k for k in path_to_start if k not in nodes_to_ignore])
    if verbose:
        print(f"path_to_start before_filter = {path_to_start}")
        print(f"path_to_start_filtered = {path_to_start_filtered}")

    path_widths = np.array([width_func(limb_obj[k]) for k in path_to_start_filtered])

    if verbose:
        print(f"path_widths = {path_widths}")
        
    if remove_zeros:
        filt = path_widths>0
        path_widths = list(path_widths[filt])
        path_to_start_filtered = path_to_start_filtered[filt]
        
        if verbose:
            print(f"path_widths AFTER REMOVING ZEROS= {path_widths}")

    if return_branch_path:
        return path_widths,path_to_start_filtered
    else:
        return path_widths
    
def downstream_endnode_skeletal_distance_from_soma(limb,branch_idx):
    
    path_to_soma = nru.branch_path_to_soma(limb,branch_idx)
    downstream_skeletal_length_to_soma = np.sum([limb[k].skeletal_length for k in path_to_soma])
    return downstream_skeletal_length_to_soma

# ------------ automatically create limb functions out of existing functions ------

def upstream_mesh(limb,branches,plot = False):
    branches = nu.to_list(branches)
    upstream_path = []
    for b in branches:
        upstream_path += nru.branch_path_to_soma(limb,b)
    upstream_path = set(upstream_path)
    upstream_mesh = tu.combine_meshes([limb[k].mesh for k in upstream_path])

    if plot:
        ipvu.plot_objects(
            upstream_mesh,
            meshes = [limb[k].mesh for k in branches],
            meshes_colors="red"
        )
    return upstream_mesh




#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu


from mesh_tools import trimesh_utils as tu
from datasci_tools import (
    numpy_utils as nu,
    ipyvolume_utils as ipvu,
)

from . import limb_utils as lu

ns.set_limb_functions_for_search(lu,verbose = False)

def skeletal_angles_df(neuron_obj,
    functions_list=(lu.parent_skeletal_angle_limb_ns,
                    lu.siblings_skeletal_angle_max_limb_ns,
                    lu.children_skeletal_angle_max_limb_ns)
    ):
    angles_df = nst.stats_df(neuron_obj,functions_list)
        
    return angles_df

def root_width(limb_obj):
    return limb_obj[limb_obj.current_starting_node].width_upstream

def best_feature_match_in_descendents(
    limb,
    branch_idx,
    feature,
    verbose = True,
    ):
    child_nodes = xu.all_children_nodes(limb.concept_network,branch_idx,depth_limit = None)
    child_features = [getattr(limb[c],feature) for c in child_nodes]
    
    branch_feature = getattr(limb[branch_idx],feature)
    
    abs_diff = [np.abs(cf - branch_feature) for cf in child_features]
    
    min_idx = np.argmin(abs_diff)
    min_child = child_nodes[min_idx]
    min_child_value = child_features[min_idx]
    
    if verbose:
        print(f"child_nodes = {child_nodes}")
        print(f"All children {feature} = {child_features}")
        print(f"Best match of Branch {branch_idx} {feature} ({branch_feature:.2f}):")
        print(f"   Child Node {min_child}, {feature} = {min_child_value:.2f}")
        
    return min_child

def root_skeleton_vector_from_soma(
    neuron_obj,
    limb_idx,
    soma_name = "S0",
    normalize = True):

    limb = neuron_obj[limb_idx]
    root_skeleton_vector_from_soma = limb.current_starting_coordinate - neuron_obj[soma_name].mesh_center
    if normalize:
        root_skeleton_vector_from_soma = root_skeleton_vector_from_soma/np.linalg.norm(root_skeleton_vector_from_soma)
    return root_skeleton_vector_from_soma
