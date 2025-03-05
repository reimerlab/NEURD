
import copy
from pykdtree.kdtree import KDTree
import time
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu

non_optional_features = [
    "mesh",
    "mesh_face_idx",
    
    "spines",
    'spines_obj',
    'spines_volume',
    
    'boutons', 
    'boutons_cdfs',
    'boutons_volume',
    'head_neck_shaft_idx',
    'synapses',
    
    
]

only_keep_upstream = [
    'web',
    'web_cdf',
]

non_optional_features_recalculate = [
    "_mesh_volume",
    "mesh_center",
]

optional_features = [
    "labels",
    
    "endpoints",
    "skeleton",
    'width',
    'width_array',
    'width_new'
]

optional_features_recalculate = [
    "_endpoints_nodes",
    "_skeleton_graph",
]


def skeleton_adjust(
    branch_obj,
    skeleton = None,
    skeleton_append = None,):
    """
    Purpose: To adjust the skeleton 
    of a branch and then have the
    endpoints readjusted

    Pseudocode: 
    1) Adjust skeleton (by stacking or reassingning)
    2) recalculate the new endpoints
    3) pass back the branch

    """
    if skeleton_append is not None:
        skeleton = sk.stack_skeletons([branch_obj.skeleton,
                                      skeleton_append])
        
    if skeleton is None:
        raise Exception("")
        
    branch_obj.skeleton = skeleton
    branch_obj.endpoints = None
    branch_obj.calculate_endpoints()
    branch_obj._skeleton_graph = None
    branch_obj._endpoints_nodes = None
    
    return branch_obj

def add_jitter_to_endpoint(
    branch_obj,
    endpoint,
    jitter = 2,
    verbose = False,
    ):
    """
    Purpose: to add jitter to a 
    branches coordinate to move it by a certain amount
    (and pass back the branch moved)

    Pseudocode: 
    1) Create jitter segment
    2) Adjust the branch skeleton
    3) Pass back the jitter segment
    """


    jitter_sk = sk.jitter_skeleton_from_coordinate(
        coordinate =endpoint,
        verbose = verbose,
    )

    bu.skeleton_adjust(
        branch_obj,
        skeleton_append = jitter_sk,
    )

    return jitter_sk




def combine_attr_lists(list_1,list_2,verbose = False):
    if list_1 is None and list_2 is None:
        if verbose:
            print(f"Both None")
        return None
    if list_1 is None:
        if verbose:
            print(f"list_1 None")
        return copy.deepcopy(list_2)
    elif list_2 is None:
        if verbose:
            print(f"list_2 None")
        return copy.deepcopy(list_1)
    else:
        if type(list_1) == list and type(list_2) == list:
            if verbose:
                print(f"both lists")
            return list_1 + list_2
        else:
            if verbose:
                print(f"both arrays")
            return np.concatenate([list_1,list_2])

def combine_branches(
    branch_upstream,
    branch_downstream,
    add_skeleton = True,
    add_labels = False,
    verbose = True,
    common_endpoint = None,
    return_jitter_segment = False,
    ):
    """
    Purpose: To combine two branch objects together
    WHERE IT IS ASSUMED THEY SHARE ONE COMMON ENDPOINT
    
    
    Ex: 
    from neurd import branch_utils as bu

    branch_upstream = copy.deepcopy(neuron_obj[0][upstream_branch])
    branch_downstream= copy.deepcopy(neuron_obj[0][downstream_branch])

    branch_upstream.labels = ["hellow"]
    branch_downstream.labels = ["my","new","labels"]

    b_out = bu.combine_branches(
        branch_upstream,
        branch_downstream,
        verbose = True,
        add_skeleton = False,
        add_labels = False
    )
    """
    
    for branch_obj in [branch_upstream,branch_downstream]:
        if (len(branch_obj.width_array["no_spine_median_mesh_center"]) != 
            len(branch_obj.skeletal_coordinates_upstream_to_downstream) - 1):
                raise Exception("")
    
    debug_time = False
    st = time.time()

    b_up = branch_upstream
    b_d = branch_downstream
    
    b_obj = copy.deepcopy(b_up)
    
    if debug_time:
        print(f"Copying branch: {time.time() - st}")
        st = time.time()

    # --working on the non-optional_features
    face_offset = len(b_up.mesh.faces)
    b_obj.mesh = tu.combine_meshes([b_up.mesh,
                                    b_d.mesh
                                   ])
    if verbose:
        print(f"b_up.mesh.faces = {b_up.mesh.faces.shape}")
        print(f"b_d.mesh.faces = {b_d.mesh.faces.shape}")
        print(f"b_obj.mesh  = {b_obj.mesh.faces.shape}")
        
    if debug_time:
        print(f"Combining meshes: {time.time() - st}")
        st = time.time()


    b_obj.mesh_face_idx = np.concatenate([
        b_up.mesh_face_idx,
        b_d.mesh_face_idx])

    # if verbose:
    #     print(f"b_up.mesh_face_idx = {b_up.mesh_face_idx}")
    #     print(f"b_d.mesh_face_idx = {b_d.mesh_face_idx}")
    #     print(f"b_obj.mesh_face_idx= {b_obj.mesh_face_idx}")

    # --------- setting the spine info -----------
    spine_objs_to_add = []
    if b_d.spines is not None:
        if b_d.spines_obj is not None:
            spine_objs_to_add = [spu.adjust_obj_with_face_offset(k,
                                                                 face_offset=face_offset)
                       for k in b_d.spines_obj]
        else:
            spine_objs_to_add = []

        for sp_attr in ["spines","spines_volume"]:
            if getattr(b_obj,sp_attr) is None:
                setattr(b_obj,sp_attr,[])

            if sp_attr == "spines_obj":
                curr_value = spine_objs_to_add
            else:
                curr_value = getattr(b_d,sp_attr)

            setattr(b_obj,sp_attr,getattr(b_obj,sp_attr) + curr_value)
        
            
    if debug_time:
        print(f"Spine adjustment: {time.time() - st}")
        st = time.time()

    if verbose:
        print(f"Total number of spines = {b_obj.n_spines}")
        
    if debug_time:
        print(f"computing number of spines : {time.time() - st}")
        st = time.time()
        
    # --------- setting the bouton and spines info -----------
    for k in ['boutons', 
              'boutons_cdfs',
              'boutons_volume',
              "head_neck_shaft_idx",]:
        if debug_time:
            print(f"combining {k}: {time.time() - st}")
            st = time.time()
        setattr(b_obj,k,combine_attr_lists(getattr(b_up,k),getattr(b_d,k)))

    if debug_time:
        print(f"combining lists: {time.time() - st}")
        st = time.time()

        
    #------ adjusting the synapses ------------------
    synapses_to_add = [syu.adjust_obj_with_face_offset(k,face_offset)
                     for k in b_d.synapses]
    
    if debug_time:
        print(f"synapses offset: {time.time() - st}")
        st = time.time()


    # ---------------- PARTS TO RECALCULATE 
    b_obj._mesh_volume = None
    b_obj.mesh_volume
    if verbose:
        print(f"b_obj.mesh_volume = {b_obj.mesh_volume}")



    # ---------------- Do skeleton combination ------------------------------
    try:
        match_idx_1,match_idx_2 = sk.matching_endpoint_singular(
                b_obj.endpoints,b_d.endpoints,
                return_indices=True,
                verbose = False
                )
        jitter_segment = None
    except Exception as e:
        if common_endpoint is None:
            raise Exception(f"{e}")
        
        match_idx_1 = nu.matching_row_index(
                b_obj.endpoints,common_endpoint)
        match_idx_2 = nu.matching_row_index(
                b_d.endpoints,common_endpoint)
        
        jitter_segment = bu.add_jitter_to_endpoint(
                        b_d,
                        b_d.endpoints[1-match_idx_2],
                        verbose = verbose
                        )
        
        print(f"b_d.endpoints = {b_d.endpoints}")
        
        
    upstream_coordinate = b_obj.endpoints[1-match_idx_1]
        
        
        

    if add_skeleton:
        if verbose:
            print(f"Adding skeleton of downstream branch")

        sk_len_weights = [b_up.skeletal_length,b_d.skeletal_length]
        skeleton_add = b_d.skeleton
        width_array_add = b_d.width_array_upstream_to_downstream
        new_width_array = {k:np.concatenate([v,width_array_add[k]]) 
                               for k,v in b_obj.width_array_upstream_to_downstream.items()}
        
        #calculate a new width array distances to keep track if not computed normally
        width_array_skeletal_lengths_d = b_d.width_array_skeletal_lengths_upstream_to_downstream
        if width_array_skeletal_lengths_d is None:
            width_array_skeletal_lengths_d = bu.skeletal_coordinates_dist_upstream_to_downstream(b_d,cumsum=False)
            
        width_array_skeletal_lengths_b = b_obj.width_array_skeletal_lengths_upstream_to_downstream
        if width_array_skeletal_lengths_b is None:
            width_array_skeletal_lengths_b = bu.skeletal_coordinates_dist_upstream_to_downstream(b_obj,cumsum=False)

        new_width_array_skeletal_lengths = np.concatenate([width_array_skeletal_lengths_b,width_array_skeletal_lengths_d])
        
        b_obj.width = nu.weighted_average([b_up.width,b_d.width],sk_len_weights)
        b_obj.width_new = {k:nu.weighted_average([v,b_d.width_new[k]],sk_len_weights)
                          for k,v in b_obj.width_new.items()}

        
        b_obj.skeleton = sk.stack_skeletons([b_up.skeleton,skeleton_add])
        b_obj.calculate_endpoints()
        b_obj.order_skeleton_by_smallest_endpoint()
        # ------------ Parts to recalculates ----------
        b_obj._skeleton_graph = None
        b_obj._endpoints_nodes = None
        b_obj.skeleton_graph
        b_obj.endpoints_nodes
        
        if not np.array_equal(b_obj.endpoints[0],upstream_coordinate):
            new_width_array = {k:np.flip(v) for k,v in new_width_array.items()}
            new_width_array_skeletal_lengths = np.flip(new_width_array_skeletal_lengths)
        
        b_obj.width_array = new_width_array
        b_obj.width_array_skeletal_lengths = new_width_array_skeletal_lengths
        

        if verbose:
            print(f"Original lengths = {sk_len_weights}")
            print(f"New skeleton length = {b_obj.skeletal_length}")
            print(f"New endpoints calculated = {sk.find_skeleton_endpoint_coordinates(b_obj.skeleton)}")
            print(f"Adjusted endpoints = {b_obj.endpoints}")
    else:
        if verbose:
            print(f"Not adding skeleton")
        # adjust the closest skeleton coordinates of all the synapses to add
        for sy in synapses_to_add:
            sy.closest_sk_coordinate = sk.closest_skeleton_coordinate(b_obj.skeleton,
                                                                    sy.closest_face_coordinate)
        for sp in spine_objs_to_add:
            sp.closest_sk_coordinate = sk.closest_skeleton_coordinate(b_obj.skeleton,
                                                                    sp.closest_face_coordinate)

        
    if (len(b_obj.width_array["no_spine_median_mesh_center"]) != 
        len(b_obj.skeletal_coordinates_upstream_to_downstream) - 1):
            raise Exception("")

    
            
    if debug_time:
        print(f"Skeleton adjustment: {time.time() - st}")
        st = time.time()

    if debug_time:
        print(f"Ccalculating skeleton graph and endpoints: {time.time() - st}")
        st = time.time()

    # ----- adjusting the synapses --------
    if verbose:
        print(f"--- Adjusting the synapse features based on new skeleton additions ------")

    b_obj.synapses = combine_attr_lists(b_obj.synapses,synapses_to_add)
    b_obj.spines_obj = combine_attr_lists(b_obj.spines_obj,spine_objs_to_add)
    
    bu.set_endpoints_upstream_downstream_idx_from_upstream_coordinate(b_obj,upstream_coordinate)
    up_idx = b_obj.endpoints_upstream_downstream_idx[0]
    
    for syn in b_obj.synapses:
        syu.calculate_endpoints_dist(b_obj,syn)
        syu.calculate_upstream_downstream_dist_from_up_idx(syn,up_idx=up_idx)
        
    if b_obj.spines_obj is not None:
        for sp in b_obj.spines_obj:
            spu.calculate_endpoints_dist(b_obj,sp)
            spu.calculate_upstream_downstream_dist_from_up_idx(sp,up_idx=up_idx)
        
    if debug_time:
        print(f"synapse distances: {time.time() - st}")
        st = time.time()
        
    # --- resetting the parameters that need to be reset
    b_obj._skeleton_vector_downstream = None
    b_obj._width_upstream = None
    b_obj._width_downstream = None
    b_obj._skeleton_vector_upstream = None

    # --------- adding labels -------
    if add_labels:
        b_obj.labels += b_d.labels
        
    if return_jitter_segment:
        return b_obj,jitter_segment
    else:
        return b_obj
    
# --------------- setting branch attributes --------------
def set_branch_attr_on_limb(
    limb_obj,
    func,
    attr_name,
    branch_idxs=None,
    **kwargs):
    """
    Purpose: To set the upstream and downstream order of the
    endpoints of a branch in a limb
    """
    if branch_idxs is None:
        branch_idxs = limb_obj.get_branch_names()
    for branch_idx in branch_idxs:
        setattr(limb_obj[branch_idx],attr_name,func(limb_obj,branch_idx,**kwargs))
        
    return limb_obj

def set_branches_endpoints_upstream_downstream_idx_on_limb(
    limb_obj,
    **kwargs
    ):
    return set_branch_attr_on_limb(
        limb_obj,
        func = nru.upstream_downstream_endpoint_idx,
        attr_name = "endpoints_upstream_downstream_idx",
        **kwargs,
    )
    
def set_branch_attr_on_limb_on_neuron(
    neuron_obj,
    func,
    attr_name,
    verbose = False,
    **kwargs):
    """
    Purpose: To set the upstream and downstream order of the
    endpoints of a branch in a limb
    """
    for limb_idx in neuron_obj.get_limb_names():
        neuron_obj[limb_idx] = bu.set_branch_attr_on_limb(neuron_obj[limb_idx],func,attr_name,**kwargs)
    return neuron_obj
        
def set_branches_endpoints_upstream_downstream_idx(
    neuron_obj,
    **kwargs
    ):
    
    neuron_obj = bu.set_branch_attr_on_limb_on_neuron(
        neuron_obj,
        func = nru.upstream_downstream_endpoint_idx,
        attr_name = "endpoints_upstream_downstream_idx",
        **kwargs
    )
    
    #print(f"{neuron_obj[0][0].endpoints_upstream_downstream_idx}")
    return neuron_obj


def skeleton_vector_endpoint(
    branch_obj,
    endpoint_type,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    offset=500,
    comparison_distance=3_000,#2_000,
    **kwargs
    ):
    """
    Purpose: To restrict a skeleton
    to its upstream or downstream vector ()

    The vector is always in the direction of most
    upstream skeleton point to downstream skeletal point

    Example: 
    bu.skeleton_vector_endpoint(
    branch_obj,
    endpoint_type = "downstream",
    #endpoint_coordinate = np.array([2504610. ,  480431. ,   33741.2])
    plot_restricted_skeleton = True,
    verbose = True,
    )
    """
    if offset is None:
        offset = offset_skeleton_vector_global
    
    if comparison_distance is None:
        comparison_distance = comparison_distance_skeleton_vector_global
    

    if endpoint_coordinate is None:
        endpoint_coordinate = getattr(branch_obj,f"endpoint_{endpoint_type}")
        if verbose:
            print(f"{endpoint_type} endpoint: {endpoint_coordinate}")

    curr_vec = sk.vector_away_from_endpoint(
        skeleton = branch_obj.skeleton,
        endpoint = endpoint_coordinate,
        verbose = False,
        plot_restricted_skeleton = plot_restricted_skeleton,
        offset=offset,
        comparison_distance=comparison_distance,#2_000,
        **kwargs
    )



    mult_number = 1
    if endpoint_type == directional_flow:
        mult_number = -1

    final_vec=  mult_number*curr_vec

    if verbose:
        print(f"curr_vec = {curr_vec}")
        print(f"mult_number = {mult_number}")
        print(f"final_vec = {final_vec}")
        
    return final_vec


def skeleton_vector_upstream(
    branch_obj,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    **kwargs
    
    ):
    return bu.skeleton_vector_endpoint(
        branch_obj,
        endpoint_type = "upstream",
        directional_flow = directional_flow,
        endpoint_coordinate = endpoint_coordinate,
        verbose = verbose,
        plot_restricted_skeleton= plot_restricted_skeleton,
        **kwargs
    )

def skeleton_vector_downstream(
    branch_obj,
    directional_flow = "downstream",
    endpoint_coordinate = None,
    verbose = False,
    plot_restricted_skeleton= False,
    **kwargs
    
    ):
    return bu.skeleton_vector_endpoint(
        branch_obj,
        endpoint_type = "downstream",
        directional_flow = directional_flow,
        endpoint_coordinate = endpoint_coordinate,
        verbose = verbose,
        plot_restricted_skeleton= plot_restricted_skeleton,
        **kwargs
    )


def width_endpoint(
    branch_obj,
    endpoint, # if None then will select most upstream endpoint of branch
    #parameters for the restriction
    offset=0,
    comparison_distance=2000,
    skeleton_segment_size=1000,
    verbose = False,
    ):
    """
    Purpose: To compute the width of a branch
    around a comparison distance and offset of an endpoint
    on it's skeleton

    """

    if verbose:
        print(f"endpoint = {endpoint}")


    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(branch_obj,
                              common_endpoint=endpoint,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=False,
                             )


    branch_width = np.mean(base_final_widths)
    overall_ais_width = branch_obj.width_new
    if verbose:
        print(f"base_final_widths = {base_final_widths}")
        print(f"overall_branch_width = {overall_ais_width}")
        print(f"branch_width = {branch_width}")
        
    return branch_width

def width_upstream(
    branch_obj,
    **kwargs
    ):
    
    return bu.width_endpoint(
        branch_obj,
        endpoint = branch_obj.endpoint_upstream,
        **kwargs
    )

def width_downstream(
    branch_obj,
    **kwargs
    ):
    
    return bu.width_endpoint(
        branch_obj,
        endpoint = branch_obj.endpoint_downstream,
        **kwargs
    )


# ---------- synapse dists ------------
def min_dist_synapse_endpoint(
    branch_obj,
    synapse_type,
    endpoint_type,
    verbose = False,
    default_value = np.inf,
    ):
    if synapse_type == "synapses":
        syns = branch_obj.synapses
    else:
        if "post" in synapse_type:
            syns = branch_obj.synapses_post
        elif "pre" in synapse_type:
            syns = branch_obj.synapses_pre
        else:
            raise Exception("")
        
    if len(syns) == 0:
        return default_value
    
    dists = [getattr(k,f"{endpoint_type}_dist") for k in syns]
    min_dist = np.min(dists)
    
    if verbose:
        print(f"For {synapse_type}, {endpoint_type}:min_dist={min_dist} \n   dists = {dists}")
    return min_dist


def min_dist_synapses_pre_upstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="pre",
    endpoint_type="upstream",
    **kwargs
    )

def min_dist_synapses_post_upstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="post",
    endpoint_type="upstream",
    **kwargs
    )

def min_dist_synapses_pre_downstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="pre",
    endpoint_type="downstream",
    **kwargs
    )

def min_dist_synapses_post_downstream(
    branch_obj,
    **kwargs):
    return min_dist_synapse_endpoint(
    branch_obj,
    synapse_type="post",
    endpoint_type="downstream",
    **kwargs
    )


def closest_mesh_skeleton_dist(
    obj,
    verbose = False):
    """
    Purpose: To find the closest distance between mesh and 
    the skeleton of a branch
    """
    coordinates = np.array(obj.skeleton).reshape(-1,3)
    mesh_kd = KDTree(obj.mesh.triangles_center)
    dist,face_idx = mesh_kd.query(coordinates)
    min_dist = np.min(dist)
    
    if verbose:
        print(f"Closest face dist = {min_dist}")
        
    return min_dist

def mesh_shaft(
    obj,
    plot = False,
    return_mesh = True):
    """
    Purpose: To export the shaft mesh of the branch
    (aka the mesh without the spine meshes)
    """
    from mesh_tools import trimesh_utils as tu

    shaft_mesh = tu.subtract_mesh(
        obj.mesh,
        [k.mesh for k in obj.spines_obj],
        return_mesh = return_mesh)

    if plot:
        if not return_mesh:
            plot_mesh = obj.mesh.submesh([shaft_mesh],append=True)
        else:
            plot_mesh = shaft_mesh
        nviz.plot_objects(obj.mesh,
                         meshes = [plot_mesh],
                         meshes_colors="red")
        
    return shaft_mesh
    
def mesh_shaft_idx(obj,
    plot = False,):
    return mesh_shaft(obj,plot=plot,return_mesh=False)

def is_skeleton_upstream_to_downstream(branch_obj,verbose = False):
    upstream_endpoint = branch_obj.endpoint_upstream
    if verbose:
        print(f"endpoints = {branch_obj.endpoints}")
        print(f"upstream_endpoint= {upstream_endpoint}")
    if np.array_equal(branch_obj.endpoints[0],upstream_endpoint):
        up_flag = True
    elif np.array_equal(branch_obj.endpoints[1],upstream_endpoint):
        up_flag = False
    else:
        raise Exception("")
    
    if verbose:
        print(f"up_flag = {up_flag}")
    
    return up_flag


def width_array_upstream_to_downstream(branch_obj,verbose = False):
    is_upstream = bu.is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        return branch_obj.width_array
    else:
        if verbose:
            print(f"Applying Flip")
        return {k:np.flip(v) for k,v in branch_obj.width_array.items()}
    
def width_array_skeletal_lengths_upstream_to_downstream(branch_obj,verbose = False):
    if branch_obj.width_array_skeletal_lengths is None:
        return None
    
    is_upstream = bu.is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        return branch_obj.width_array_skeletal_lengths
    else:
        return np.flip(branch_obj.width_array_skeletal_lengths)
    
    
def skeletal_coordinates_upstream_to_downstream(
    branch_obj,
    verbose = False,
    skeleton = None,
    coordinate_dists = None,
    resize=True):
    
    if not resize:
        skeleton = branch_obj.skeleton
        
    if skeleton is None:
        array = wu.skeleton_resized_ordered(branch_obj.skeleton)
    else:
        skeleton = sk.order_skeleton(skeleton)
        array = skeleton
        
        
    is_upstream = bu.is_skeleton_upstream_to_downstream(branch_obj,verbose)
    if is_upstream:
        pass
    else:
        if verbose:
            print(f"Applying Flip")
        array= sk.flip_skeleton(array)
        
    if branch_obj.width_array_skeletal_lengths_upstream_to_downstream is not None:
        coordinate_dists = np.concatenate([[0],branch_obj.width_array_skeletal_lengths_upstream_to_downstream])
        
    #coordinate_dists = branch_obj.width_array_skeletal_lengths_upstream_to_downstream
        
    if coordinate_dists is not None:
        coordinate_dists = np.cumsum(coordinate_dists)
        coordinates = sk.coordinates_from_downstream_dist(
            array,
            coordinate_dists,
            verbose = False,
            segment_width=0,
            plot = False
        )
    else:
        coordinates = sk.skeleton_coordinate_path_from_start(array)
    
    return coordinates

def skeletal_coordinates_dist_upstream_to_downstream(
    branch_obj,
    verbose = False,
    cumsum = True,
    skeleton = None,
    **kwargs):
    
    if skeleton is None:
        array = bu.skeletal_coordinates_upstream_to_downstream(branch_obj,**kwargs)
    else:
        array = skeleton
        
    dist_array = np.linalg.norm(array[1:] - array[:-1],axis=1)
    if cumsum:
        return np.cumsum(dist_array)
    else:
        return dist_array
    
    
def endpoint_upstream_idx(branch_obj,coordinate = None):
    if coordinate is None:
        coordinate = branch_obj.endpoint_upstream
    return nu.matching_row_index(branch_obj.endpoints,coordinate)

def endpoint_downstream_idx(branch_obj,coordinate = None):
    if coordinate is None:
        coordinate = branch_obj.endpoint_downstream
    return nu.matching_row_index(branch_obj.endpoints,coordinate)

def set_endpoints_upstream_downstream_idx_from_upstream_coordinate(
    branch_obj,
    upstream_coordinate=None,
    up_idx = None,
    ):
    """
    Purpose: Set the branch upstream, downstream by a coordinate
    """
    if up_idx is None:
        up_idx = bu.endpoint_upstream_idx(branch_obj,upstream_coordinate)
    branch_obj.endpoints_upstream_downstream_idx = (up_idx,1-up_idx)
    
def set_endpoints_upstream_downstream_idx_on_branch(
    limb_obj,
    branch_idx,):
    
    if limb_obj[branch_idx].endpoints_upstream_downstream_idx is None:
        up_idx = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=True)
        bu.set_endpoints_upstream_downstream_idx_from_upstream_coordinate(
            limb_obj[branch_idx],
            up_idx = up_idx,
        )
    
    
synapse_dynamics_attrs = [
    "upstream_dist",
    "head_neck_shaft",
    "syn_id",
    "volume",
    "syn_type",  
    #"soma_distance"
]

spine_dynamics_attrs = [
    "upstream_dist",
    "volume",
    "area",
    "spine_id"
    #"soma_distance"
]

def width_array_upstream_to_dowstream_with_skeletal_points(
    branch_obj,
    width_name = "no_spine_median_mesh_center",
    ):
    """
    Purpose: Want to get the width at a certain
    point on the branch where that certain point
    is the closest distcretization to another coordinate
    """
    skeleton_coords = branch_obj.skeletal_coordinates_upstream_to_downstream
    skeleton_coords_mid = (skeleton_coords[:-1] + skeleton_coords[1:])/2
    
    return branch_obj.width_array_upstream_to_downstream[width_name],skeleton_coords_mid

def width_array_value_closest_to_coordinate(
    branch_obj,
    coordinate,
    verbose = False,):
    """
    Purpose: To find the width closest to certain coordinates
    on a branch obj
    """
    coordinate = np.array(coordinate)
    if coordinate.ndim == 1:
        single_flag = True
    else:
        single_flag = False
        
    coordinate = np.array(coordinate).reshape(-1,3)
    widths,sk_coordinates = bu.width_array_upstream_to_dowstream_with_skeletal_points(branch_obj)
    closest_widths = widths[nu.closest_idx_for_each_coordinate(
        coordinate,
        sk_coordinates,
        closest_idx_algorithm = "linalg")
    ]
    
    if verbose:
        print(f"closest_widths= {closest_widths}")
        
    if single_flag:
        return closest_widths[0]
    else:
        return closest_widths

def branch_dynamics_attr_dict_dynamics_from_node(
    branch_obj,
    width_name = "no_spine_median_mesh_center"
    ):
    """
    Purpose: To save off all of the necessary information 
    for branch dynamics (of spines,width,synapses) to 
    """
    sp_atts = []
    if branch_obj.spines_obj is not None:
        sp_atts = [{att:getattr(spu.Spine(sp),att)
                    for att in spine_dynamics_attrs} for sp in branch_obj.spines_obj]
    
    syn_atts = []
    if branch_obj.synapses is not None:
        syn_atts = [{att:getattr(sp,att) if att != "head_neck_shaft"
                    else  spu.spine_str_label(getattr(sp,att))
                     for att in synapse_dynamics_attrs} for sp in branch_obj.synapses]
        
    skeleton_coords = branch_obj.skeletal_coordinates_upstream_to_downstream
    skeleton_coords_dists = branch_obj.skeletal_coordinates_dist_upstream_to_downstream
    
    width_attrs = [dict(
       upstream_dist = skeleton_coords_dists[i],
       width = k,
    ) for i,k in enumerate(branch_obj.width_array_upstream_to_downstream[width_name])]
    
    
    return_dict = dict(
        spine_data = sp_atts,
        synapse_data = syn_atts,
        width_data = width_attrs,
        skeleton_data = skeleton_coords,
    )
    
    return return_dict


def refine_width_array_to_match_skeletal_coordinates(
    neuron_obj,
    verbose = False):
    """
    Purpose: To update the widths of those that don't match the 
    skeletal coordinates
    """
    bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    
    def width_array_length(branch_obj,width_name="no_spine_median_mesh_center",**kwargs):
        return len(branch_obj.width_array[width_name])
    def skeletal_coordinates_upstream_to_downstream_length(branch_obj,**kwargs):
        return len(branch_obj.skeletal_coordinates_upstream_to_downstream)

    from neurd import neuron_searching as ns
    lb = ns.query_neuron(
        neuron_obj,
        functions_list=[
            width_array_length,
            skeletal_coordinates_upstream_to_downstream_length],
        query = "width_array_length != (skeletal_coordinates_upstream_to_downstream_length-1)",  
    )
    
    if verbose:
        print(f"limb branch to update = {lb}")
    
    wu.neuron_width_calculation_standard(
        neuron_obj,
        verbose = verbose,
        limb_branch_dict=lb,
    )
    
    return neuron_obj

def endpoint_type_with_offset(
    branch_obj,
    endpoint_type="upstream",
    offset =1000,
    plot = False,
    verbose= False,
    ):
    """
    Purpose: To get the skeleton point
    a little offset from the current endpoint
    """
    endpoint_coordinate = getattr(branch_obj,f"endpoint_{endpoint_type}")
    coordinate = sk.skeleton_coordinate_offset_from_endpoint(
        branch_obj.skeleton,
        offset_distance = offset,
        endpoint_coordinate = endpoint_coordinate,
        plot_coordinate = plot,
    )
    
    if verbose:
        print(f"coordinate = {coordinate} (with enpoint coord = {endpoint_coordinate})")
        
    return coordinate
    
def endpoint_upstream_with_offset(
    branch_obj,
    offset =1000,
    plot = False,
    verbose= False,
    ):
    
    """
    Ex: 
    bu.endpoint_upstream_with_offset(
        branch_obj = limb_obj[26],
        verbose = True,
        offset = 200,
        plot = True
    )
    
    """
    
    return bu.endpoint_type_with_offset(
        branch_obj=branch_obj,
        endpoint_type="upstream",
        offset =offset,
        plot = plot,
        verbose= verbose,
        )

def endpoint_downstream_with_offset(
    branch_obj,
    offset =1000,
    plot = False,
    verbose= False,
    ):
    
    return bu.endpoint_type_with_offset(
        branch_obj=branch_obj,
        endpoint_type="downstream",
        offset =offset,
        plot = plot,
        verbose= verbose,
        )



def skeleton_angle_from_top(
    branch_obj,
    top_of_layer_vector = None):
    if top_of_layer_vector is None:
        top_of_layer_vector = nst.top_of_layer_vector


    sk_vector = branch_obj.skeleton_vector_upstream
    angle_from_top = np.round(nu.angle_between_vectors(nst.top_of_layer_vector,sk_vector),4)
    return angle_from_top

# -------------------------------------------------------


global_parameters_dict_default = dict(
    offset_skeleton_vector = 500,
    comparison_distance_skeleton_vector = 3000
)
attributes_dict_default = dict()    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01 = dict()
attributes_dict_h01 = dict()



# data_type = "default"
# algorithms = None
# modules_to_set = [bu]

# def set_global_parameters_and_attributes_by_data_type(dt,
#                                                      algorithms_list = None,
#                                                       modules = None,
#                                                      set_default_first = True,
#                                                       verbose=False):
#     if modules is None:
#         modules = modules_to_set
    
#     modu.set_global_parameters_and_attributes_by_data_type(modules,dt,
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
from . import branch_attr_utils as bau
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import spine_utils as spu
from . import synapse_utils as syu
from . import width_utils as wu
from . import neuron_statistics as nst

 

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import module_utils as modu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu

from . import branch_utils as bu