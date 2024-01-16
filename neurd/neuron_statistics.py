
import copy
import networkx as nx
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu

top_of_layer_vector = np.array([0,-1,0])

def neuron_path_analysis(neuron_obj,
                        N = 3,
                        plot_paths = False,
                        return_dj_inserts = True,
                        verbose = False):

    """
    Pseudocode: 
    1) Get all the errored branches
    For Each Limb:
    2) Remove the errored branches
    3) Find all branches that are N steps away from starting node (and get the paths)
    4) Filter away paths that do not have all degrees of 2 on directed network
    *** Those are the viable paths we would analyze***
    5) Extract the statistics

    """

    # 0) Compute the mesh center of the soma
    curr_soma = neuron_obj["S0"].mesh
    curr_soma_center = tu.mesh_center_vertex_average(curr_soma)
    y_vector = top_of_layer_vector

    # 1) Getting all Errored Branches 
    error_branches = ed.error_branches_by_axons(neuron_obj,visualize_errors_at_end=False)

    # ----------- Loop that will iterate through all branches ----------- #
    neuron_path_inserts_by_limb = dict()


    total_paths = dict()
    for curr_limb_idx,curr_limb_obj in enumerate(neuron_obj):
        l_name = f"L{curr_limb_idx}"
        if l_name in error_branches.keys():
            curr_limb_error_branches = error_branches[l_name]
        else:
            curr_limb_error_branches = []

        # 2) Remove the errored branches
        if len(curr_limb_obj.all_concept_network_data)>1:
            raise Exception(f"More than one starting node for limb {curr_limb_idx}")

        st_node = curr_limb_obj.current_starting_node
        st_coordinates = curr_limb_obj.current_starting_coordinate



        #2-4: Getting the paths we want

        G = nx.Graph(curr_limb_obj.concept_network)

        target_to_path = nx.single_source_shortest_path(G, source=st_node)#, cutoff=N+1) 
        paths_of_certain_length = [v for k,v in target_to_path.items() if (len(v) == N) ]
        if verbose:
            print(f"Number of paths with {N} nodes = {len(paths_of_certain_length)}")

        #remove the paths with errors on
        paths_of_certain_length_no_errors = [v for v in paths_of_certain_length if len(np.intersect1d(curr_limb_error_branches,
                                                                                                     v))==0]

        if verbose:
            print(f"Number of paths with {N} nodes and no Errors = {len(paths_of_certain_length_no_errors)}")

        #need to filter away for high degree nodes along path
        """
        1) Turn network into directional
        2) Find all of the upstream nodes
        3) Filter those paths away where existence of greater than 2 degrees
        """
        #1) Turn network into directional
        G_directional = nx.DiGraph(curr_limb_obj.concept_network_directional)

        final_paths = []
        for ex_path in paths_of_certain_length_no_errors:
            path_degree = np.array([len(xu.downstream_edges_neighbors(G_directional,k)) for k in ex_path[:-1]] )
            if np.sum(path_degree!=2) == 0:
                final_paths.append(ex_path)
            else:
                if verbose:
                    print(f"Ignoring path because path degrees are {path_degree}")

        if verbose: 
            print(f"Number of paths after filtering away high degree nodes = {len(final_paths)} ")

        if plot_paths:
            if len(final_paths) > 0:
                curr_nodes = np.unique(np.concatenate(final_paths))
            else:
                curr_nodes = []
            total_paths.update({f"L{curr_limb_idx}":curr_nodes})


        # Step 5: Calculating the Statistics on the branches


        """
        Pseudocode: 
        1) Get starting angle of branch

        For all branches not in the starting node
        a) Get the width (all of them)
        b) Get the number of spines, spines_volume, and spine density


        d) Skeletal distance (distance to next branch point)
        e) Angle between parent branch and current branch
        f) Angle between sibling branch and current


        """

        #1) Get starting angle of branch
        st_vector = st_coordinates - curr_soma_center
        st_vector_norm = st_vector/np.linalg.norm(st_vector)
        angle_from_top = np.round(nu.angle_between_vectors(y_vector,st_vector_norm),2)



        limb_path_dict = dict()
        for zz,curr_path in enumerate(final_paths):

            local_dict = dict(soma_angle=angle_from_top)
            for j,n in enumerate(curr_path[1:]):
                curr_name = f"n{j}_"
                curr_node = curr_limb_obj[n]

                #a) Get the width (all of them)
                for w_name,w_value in curr_node.width_new.items():
                    local_dict[curr_name +"width_"+ w_name] = np.round(w_value,2)

                #b) Get the number of spines, spines_volume, and spine density
                attributes_to_export = ["n_spines","total_spine_volume","spine_volume_median",
                                       "spine_volume_density","skeletal_length"]
                for att in attributes_to_export:
                    local_dict[curr_name + att] = np.round(getattr(curr_node,att),2)

                #e) Angle between parent branch and current branch
                local_dict[curr_name + "parent_angle"] = nru.find_parent_child_skeleton_angle(curr_limb_obj,child_node=n)

                #f) Angle between sibling branch and current
                local_dict[curr_name + "sibling_angle"]= list(nru.find_sibling_child_skeleton_angle(curr_limb_obj,child_node=n).values())[0]

            limb_path_dict[zz] = local_dict


        neuron_path_inserts_by_limb[curr_limb_idx] =  limb_path_dict

    if plot_paths:
        nviz.visualize_neuron(neuron_obj,
                          visualize_type=["mesh"],
                          limb_branch_dict=total_paths,
                         mesh_color="red",
                         mesh_whole_neuron=True)   

    if return_dj_inserts:
        # Need to collapse this into a list of dictionaries to insert
        dj_inserts = []
        for limb_idx,limb_paths in neuron_path_inserts_by_limb.items():
            for path_idx,path_dict in limb_paths.items():
                dj_inserts.append(dict(path_dict,limb_idx=limb_idx,path_idx=path_idx))
        return dj_inserts
    else:
        return neuron_path_inserts_by_limb
    
    
def soma_starting_vector(limb_obj=None,
                        neuron_obj=None,
                        limb_idx=None,
                        soma_idx = 0,
                        soma_group_idx = None,
                        soma_center = None):
    """
    Will find the angle between the vector pointing to the
    top of the volume and the angle from the soma center to the starting skeleton
    coordinate
    
    
    """
    if limb_obj is None:
        curr_limb_obj = neuron_obj[limb_idx]
        
    else:
        curr_limb_obj = limb_obj
        
    if soma_group_idx is not None:
        curr_limb_obj.set_concept_network_directional(starting_soma=soma_idx,
                                                     soma_group_idx=soma_group_idx)
        
    st_coordinates = curr_limb_obj.current_starting_coordinate
    
    if soma_center is None:
        curr_soma = neuron_obj[f"S{soma_idx}"].mesh
        soma_center = tu.mesh_center_vertex_average(curr_soma)
    
    
    #1) Get starting angle of branch
    st_vector = st_coordinates - soma_center
    st_vector_norm = st_vector/np.linalg.norm(st_vector)
    return st_vector_norm

def soma_starting_angle(limb_obj=None,
                        neuron_obj=None,
                        limb_idx=None,
                        soma_idx = 0,
                        soma_group_idx = None,
                        soma_center = None,
                        y_vector = np.array([0,-1,0])):
    st_vector_norm = nst.soma_starting_vector(limb_obj=limb_obj,
                        neuron_obj=neuron_obj,
                        limb_idx=limb_idx,
                        soma_idx = soma_idx,
                        soma_group_idx = soma_group_idx,
                        soma_center = soma_center,)
    angle_from_top = np.round(nu.angle_between_vectors(y_vector,st_vector_norm),2)
    
    return angle_from_top

def fork_divergence_from_skeletons(upstream_skeleton,
                    downstream_skeletons,
                    downstream_starting_endpoint = None,
                    comparison_distance = 500,
                    skeletal_segment_size = 50,
                    plot_restrictions = False,
                    combining_function = np.sum,
                                  verbose=False):
    """
    Purpose: To compute the number for the fork splitting

    Pseudocode: 
    1) Find intersection point of all 3 branches
    2) for 2 downstream branch: 
       - restrict the skeleton to a certain distance from the start
    3) discretize the skeeletons so have x pieces
    4) Measure distance between each indexed point
    5) Have one way of aggregating the distances (median, mean)

    Application: If below a certain value then can indicate incorrect branching


    Ex:
    from neurd import neuron_statistics as nst
    nst.fork_divergence(upstream_skeleton = upstream_sk,
                        downstream_skeletons = downstream_sk,
                        comparison_distance = 500,
                        skeletal_segment_size = 50,
                        plot_restrictions = True,
                        combining_function = np.mean)
    
    """
    downstream_sk = downstream_skeletons
    upstream_sk = upstream_skeleton
    
    if downstream_starting_endpoint is None:
        joining_endpoint_1 = sk.shared_coordiantes(list(downstream_sk) + [upstream_skeleton],
                                 return_one=True)
    else:
        joining_endpoint_1 = downstream_starting_endpoint
        
    if verbose:
        print(f"joining_endpoint_1 = {joining_endpoint_1}")

    # resize the skeletons
    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                          for k in downstream_skeletons]
    
    if verbose:
#         print(f"After resizing:\n{[sk.calculate_skeleton_segment_distances(k) for k in d_skeletons_resized]}")
        pass
        

    d_skeletons_restricted = [sk.restrict_skeleton_to_distance_from_coordinate(k,
                                             coordinate = joining_endpoint_1,
                                             distance_threshold = comparison_distance+1,)
                              for k in d_skeletons_resized]
    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                           for k in d_skeletons_restricted]
                           
    if verbose:
        #print(f"After restricting:\n{[sk.calculate_skeleton_distance(k) for k in d_skeletons_restricted]}")
        pass
        
    if verbose:
        print(f"Segment sizes after restriction")
        for d in d_skeletons_resized:
            print(sk.calculate_skeleton_segment_distances(d))

    #align the skeletons

    if plot_restrictions:
        nviz.plot_objects(skeletons = downstream_sk + d_skeletons_resized,
                         skeletons_colors=["black"]*len(d_skeletons_resized) + ["red"]*len(d_skeletons_restricted),
                         scatters=[joining_endpoint_1.reshape(-1,3)],
                         scatter_size=0.5)

    d_ordered_paths = [sk.skeleton_coordinate_path_from_start(k,start_endpoint_coordinate=joining_endpoint_1) for k in d_skeletons_restricted]
    
    
    if verbose:
        #print(f"Before skeleton path from start: {d_skeletons_restricted}")
        #print(f"After skeleton path from start: {d_ordered_paths}")
        for d in d_ordered_paths:
            print(sk.calculate_skeleton_segment_distances(sk.skelton_coordinate_path_to_skeleton(d)))
        
    min_length = np.min([len(k) for k in d_ordered_paths])
    if verbose:
        print(f"min_length = {min_length}")
        
    path_distances = np.linalg.norm(d_ordered_paths[0][:min_length]-d_ordered_paths[1][:min_length],axis=1)

    return_value = combining_function(path_distances)
    
    return return_value


def fork_divergence_from_branch(limb_obj,
                                branch_idx,
                                verbose = False,
                                error_not_2_downstream=True,
                                total_downstream_skeleton_length_threshold = 4000,
                                individual_branch_length_threshold = 3000,
                                skip_value = np.inf,
                                plot_fork_skeleton = False,
                                upstream_sk_color="red",
                                downstream_sk_colors = None,
                                
                                #arguments for the fork divergence measurement
                                comparison_distance = 400,
                                skeletal_segment_size = 40,
                                plot_restrictions = False,
                                combining_function = np.mean,
                                **kwargs
                               ):
    """
    Purpose

    Pseudocode: 
    1) Get the branch where the error is
    2) Get all the upstream branch and all of the downstream branches of that
    3) Measure the sibling angles
    4) Collect together the skeletons in a list
    5) Run the fork splitting function

    Note: should only be done on 2 forks
    

    """

    return_value = None
    #2) Get all the upstream branch and all of the downstream branches of that
    upstream_node = xu.upstream_node(limb_obj.concept_network_directional,
                                     branch_idx)
    if upstream_node is None:
        return skip_value
    
    downstream_nodes = xu.downstream_nodes(limb_obj.concept_network_directional,upstream_node)

    if error_not_2_downstream and len(downstream_nodes) != 2:
        raise Exception(f"Not exactly 2 downstream nodes: {downstream_nodes}")


    upstream_sk = limb_obj[upstream_node].skeleton
    downstream_sk = [limb_obj[d].skeleton for d in downstream_nodes]

    if verbose:
        print(f"Upstream Node = {upstream_node}, downstream nodes = {downstream_nodes}")


    if (total_downstream_skeleton_length_threshold is not None and
        individual_branch_length_threshold is not None):
        d_len = np.array([sk.calculate_skeleton_distance(limb_obj[k].skeleton) for 
                         k in downstream_nodes])
        d_skeletal_len = np.array([nru.skeletal_length_over_downstream_branches(limb_obj,
                                                d,
                                                verbose=False) for d in downstream_nodes])
        below_threshold = np.where((d_skeletal_len<total_downstream_skeleton_length_threshold) | 
                                   (d_len < individual_branch_length_threshold))[0]
        if len(below_threshold) > 0:
            if verbose:
                print(f"Skipping this intersection because some of downstream skeletal lengths too short (min {total_downstream_skeleton_length_threshold}):")
                print(f" or the individual branch length was too short (min {individual_branch_length_threshold})")
                for j,(d,d_len) in enumerate(zip(downstream_nodes,d_skeletal_len)):
                    if j in below_threshold:
                        print(f"Brnach {d}: length {d_l} {d_len}")

            return_value = skip_value


    if return_value is None:
        #3) Measure the sibling angles
        sibling_angle = list(nru.find_sibling_child_skeleton_angle(curr_limb_obj=limb_obj,
                                             child_node=branch_idx).values())
        if verbose:
            print(f"sibling_angle = {sibling_angle}")

        #4) Collect together the skeletons in a list


        if downstream_sk_colors is None:
            from datasci_tools import matplotlib_utils as mu
            downstream_sk_colors = mu.generate_color_list(n_colors = len(downstream_nodes),
                                  colors_to_omit = [upstream_sk_color])

        if plot_fork_skeleton:
            downstream_meshes = [limb_obj[d].mesh for d in downstream_nodes]
            upstream_mesh = limb_obj[upstream_node].mesh
            nviz.plot_objects(main_mesh=upstream_mesh,
                              main_mesh_color=upstream_sk_color,
                                main_skeleton=upstream_sk,
                              main_skeleton_color=upstream_sk_color,
                              meshes=downstream_meshes,
                              meshes_colors=downstream_sk_colors,
                             skeletons=downstream_sk,
                             skeletons_colors=downstream_sk_colors)

        
        return_value = nst.fork_divergence_from_skeletons(upstream_skeleton = upstream_sk,
                            downstream_skeletons = downstream_sk,
                            comparison_distance = comparison_distance,
                            skeletal_segment_size = skeletal_segment_size,
                            plot_restrictions = plot_restrictions,
                            combining_function = combining_function,
                                                         verbose=verbose)


    if verbose:
        print(f"Final Fork Divergence = {return_value}")
        
    return return_value


def n_small_children(limb_obj,
    branch_idx,
    width_maximum = 80,
    verbose = False):
    """
    Purpose: Will measure the number
    of small width immediate downstream nodes

    Pseudocode: 
    1) Find the number of downstream nodes
    2) Find the width of the downstream nodes
    3) Count how many are below the threshold
    
    Ex:
    from neurd import neuron_statistics as nst
    nst.n_small_children(limb_obj = neuron_obj[6],
        branch_idx = 5,
        width_maximum = 80,
    verbose = False)
    """
    
    downstream_n = xu.downstream_nodes(limb_obj.concept_network_directional,
                        branch_idx)
    downstream_width = np.array([au.axon_width(limb_obj[k]) for k in downstream_n])

    n_small_downstream = len(np.where(downstream_width < width_maximum)[0])
    
    if verbose: 
        print(f"downstream_n = {downstream_n}, downstream_width = {downstream_width},"
              f" n_small_downstream = {n_small_downstream}")
    return n_small_downstream

def child_angles(limb_obj,
    branch_idx,
    verbose = False,
    comparison_distance=1500,
    ):

    """
    Purpose: To measure all of the angles betweent he children nodes

    Psuedocode: 
    1) Get the downstream nodes --> if none or one then return empty dictionary

    For all downstream nodes:
    2) choose one of the downstream nodes and send to nru.find_sibling_child_skeleton_angle
    3) create a dictionary with the nodes in a tuple as the key and the angle between 
    them as the values

    will error if more than 2 children current
    
    Ex: 
    nst.child_angles(limb_obj = neuron_obj[6],
    branch_idx = 22,
    verbose = False,
    )

    """




    #1) Get the downstream nodes --> if none or one then return empty dictionary
    downstream_n = xu.downstream_nodes(limb_obj.concept_network_directional,
                            branch_idx)

    if len(downstream_n) in [0,1]:
        return {}
    elif len(downstream_n) > 2:
        raise Exception(f"Not implemented for number of children more than 2 and currently there are {len(downstream_n) }")
    else:
        pass

    #2) choose one of the downstream nodes and send to nru.find_sibling_child_skeleton_angle
    sibling_angles = nru.find_sibling_child_skeleton_angle(limb_obj,
                                         downstream_n[0],
                                            comparison_distance=comparison_distance)
    return_dict = {(downstream_n[0],k):v for k,v in sibling_angles.items()}

    if verbose:
        print(f"return_dict= {return_dict}")
    return return_dict


def angle_from_top(vector,
    vector_pointing_to_top = np.array([0,-1,0]),
                   verbose=True
    ):
    """
    Purpose: Will find the angle between
    a vector and the vector pointing towards the top
    of the volume
    
    """
    if vector.ndim == 2:
        vector = vector[-1] - vector[0] 
    st_vector_norm = vector/np.linalg.norm(vector)
    angle_from_top = np.round(nu.angle_between_vectors(vector_pointing_to_top,st_vector_norm),2)
    return angle_from_top


def children_skeletal_lengths(limb_obj,
                                        branch_idx,
                                        verbose = False,
                                        return_dict = True):
    """
    Purpose: To generate the downstream skeletal lengths of all children
    
    Pseudocode:
    1) Find the downstream nodes
    2) Compute the downstream skeletal length for each
    3) return as dictionary
    """
    
    vals = dict([(k,nru.skeletal_length_over_downstream_branches(limb_obj,
                                             branch_idx = k,
                                             include_branch_skeletal_length=True))
                  for k in xu.downstream_nodes(limb_obj.concept_network_directional,branch_idx)])
    if verbose:
        print(f"downstream values = {vals}")
    if return_dict:
        return vals
    else:
        return list(vals.values())

def children_skeletal_lengths_min(limb_obj,
                                        branch_idx,
                                        verbose = False):
    return_sk_len = nst.children_skeletal_lengths(limb_obj,
                                        branch_idx,
                                        verbose = verbose,
                                        return_dict = False)
    if len(return_sk_len) == 0:
        return 0
    else:
        return np.min(return_sk_len)

    
def children_feature(limb_obj,
                       branch_idx,
                       feature_func,
                       verbose = False,
                       return_dict = True,
                    **kwargs):
    """
    To compute a feature over all of the children nodes of
    a traget branchn
    
    """
    vals = dict([(k,feature_func(limb_obj,
                                             branch_idx = k,
                                **kwargs))
                  for k in xu.downstream_nodes(limb_obj.concept_network_directional,branch_idx)])
    if verbose:
        print(f"{feature_func.__name__} values = {vals}")
    if return_dict:
        return vals
    else:
        return list(vals.values())
    
def children_axon_width(limb_obj,
                       branch_idx,
                       verbose = False,
                       return_dict = True):
    """
    Computes the axon width of all the children
    """
    feature_func = au.axon_width
    vals = dict([(k,feature_func(limb_obj[k],
                                ))
                  for k in xu.downstream_nodes(limb_obj.concept_network_directional,branch_idx)])
    if verbose:
        print(f"Childrean Axon Width values = {vals}")
    if return_dict:
        return vals
    else:
        return list(vals.values())
    return vals

def children_axon_width_max(limb_obj,
                       branch_idx,
                       verbose = False,
                       **kwargs):
    children_widths = children_axon_width(limb_obj,
                       branch_idx,
                       verbose = verbose,
                       return_dict = False)
    return np.max(children_widths)
    
    
def upstream_axon_width(limb_obj,
                       branch_idx,
                        default = np.inf,
                        **kwargs):
    """
    Purpose: To return the widh of the upstream branch
    
    Psuedocode: 
    1) Get the upstream branch
    2) return the width
    """
    up_node = xu.upstream_node(limb_obj.concept_network_directional,branch_idx)
    if up_node is None:
        return default
    else:
        return au.axon_width(limb_obj[up_node])
    
def upstream_skeletal_length(limb_obj,
                       branch_idx,
                        default = np.inf,
                        **kwargs):
    """
    Purpose: To return the skeletal length of the upstream branch
    
    Psuedocode: 
    1) Get the upstream branch
    2) return the width
    """
    up_node = xu.upstream_node(limb_obj.concept_network_directional,branch_idx)
    if up_node is None:
        return default
    else:
        return limb_obj[up_node].skeletal_length
    
    
def skeletal_length_along_path(limb_obj,
                              branch_path):
    return np.sum([limb_obj[k].skeletal_length for k in branch_path])
def total_upstream_skeletal_length(limb_obj,
                                  branch_idx,
                                   include_branch=False,
                                  **kwargs):
    """
    Purpose: To get all of the skeleton length from current branch to 
    starting branch
    
    """
    #print(f"include_branch = {include_branch}")
    
    branch_path = nru.branch_path_to_start_node(limb_obj,
                             branch_idx,
                             include_branch_idx = include_branch,
                            skeletal_length_min = None,
                            verbose = False)
    return skeletal_length_along_path(limb_obj,branch_path)

# --------- 6/17: Functions for helping pair branches with each other ----------#
def width_new(branch,width_new_name="no_spine_mean_mesh_center",
              width_new_name_backup = "no_spine_median_mesh_center",
              **kwargs):
    try:
        return branch.width_new[width_new_name]
    except:
        return branch.width_new[width_new_name_backup]


def width_diff_basic(limb_obj,
                 branch_1_idx,
                 branch_2_idx,
                 width_func = width_new,):
    """
    Ex: 
    from neurd import neuron_statistics as nst
    nst.width_diff_percentage_basic(n_obj_syn[0],1,2)
    """
    return np.abs(width_func(limb_obj[branch_1_idx]) - width_func(limb_obj[branch_2_idx]))


def width_diff(limb_obj,
                 branch_1_idx,
                 branch_2_idx,
                 width_func = None,
                   branch_1_direction = "upstream",
                   branch_2_direction = "downstream",
                  comparison_distance = 10000,
                  nodes_to_exclude=None,
                 return_individual_widths = False,
              verbose = False):
    
    if width_func is None:
        width_func = au.axon_width
    
    branch_1_width = cnu.width_upstream_downstream(limb_obj,
                                                  branch_1_idx,
                                                  direction=branch_1_direction,
                                                  distance=comparison_distance,
                                                  width_func=width_func,
                                                  nodes_to_exclude=nodes_to_exclude)
    
    branch_2_width = cnu.width_upstream_downstream(limb_obj,
                                                  branch_2_idx,
                                                  direction=branch_2_direction,
                                                  distance=comparison_distance,
                                                  width_func=width_func,
                                                  nodes_to_exclude=nodes_to_exclude)
    if verbose:
        print(f"branch_1_width = {branch_1_width}, branch_2_width = {branch_2_width}")
    
    width_d = np.abs(branch_1_width - branch_2_width)
    if return_individual_widths:
        return width_d,branch_1_width,branch_1_width
    else:
        return width_d

def width_max(limb_obj,
             branches_idxs,
             width_func = None):
    if width_func is None:
        width_func = au.axon_width
    return np.max([width_func(limb_obj[k]) for k in branches_idxs])

def width_diff_percentage_basic(limb_obj,
                 branch_1_idx,
                 branch_2_idx,
                 width_func = width_new,
                         verbose = False):
    """
    Ex: 
    from neurd import neuron_statistics as nst
    nst.width_diff_percentage_basic(n_obj_syn[0],1,2)
    """
    w_diff = width_diff(limb_obj,
                       branch_1_idx=branch_1_idx,
                       branch_2_idx=branch_2_idx,
                       width_func=width_func)
    max_width = width_max(limb_obj,[branch_1_idx,branch_2_idx],width_func=width_func)
    if max_width > 0:
        w_diff_perc = w_diff/max_width
    else:
        w_diff_perc = 0
        
    if verbose:
        print(f"w_diff= {w_diff}, max_width = {max_width}, w_diff_perc = {w_diff_perc}")
    return w_diff_perc

def width_diff_percentage(limb_obj,
                 branch_1_idx,
                 branch_2_idx,
                 width_func = None,
                branch_1_direction = "upstream",
                   branch_2_direction = "downstream",
                  comparison_distance = 10000,
                  nodes_to_exclude=None,
              verbose = False):
                  
    if width_func is None:
        width_func = au.axon_width
    
    w_diff, b1_w, b2_w= width_diff(limb_obj,
                       branch_1_idx=branch_1_idx,
                       branch_2_idx=branch_2_idx,
                       width_func=width_func,
                       branch_1_direction = branch_1_direction,
                   branch_2_direction = branch_2_direction,
                  comparison_distance = comparison_distance,
                  nodes_to_exclude=nodes_to_exclude,
              verbose = verbose,
                       return_individual_widths=True)
    
    
    max_width = np.max([b1_w, b2_w])
    if max_width > 0:
        w_diff_perc = w_diff/max_width
    else:
        w_diff_perc = 0
        
    if verbose:
        print(f"w_diff= {w_diff}, max_width = {max_width}, w_diff_perc = {w_diff_perc}, for individual widths: {b1_w,b2_w}")
    return w_diff_perc



def parent_child_sk_angle(limb_obj,
                                branch_1_idx,
                                branch_2_idx,
                         **kwargs):
    return nst.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                        branch_1_idx,
                                                        branch_2_idx,
                                                        branch_1_type="upstream",
                                                        branch_2_type = "downstream",
                                                        use_upstream_skeleton_restriction=True,
                                                        **kwargs)

def sibling_sk_angle(limb_obj,
                                branch_1_idx,
                                branch_2_idx,
                         **kwargs):
    return nst.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                        branch_1_idx,
                                                        branch_2_idx,
                                                        branch_1_type="downstream",
                                                        branch_2_type = "downstream",
                                                        use_upstream_skeleton_restriction=True,
                                                        **kwargs)
# def none_to_some_synapses(limb_obj,
#                           branch_1_idx,
#                          branch_2_idx,
#                          synapse_type=None,):
#     """
#     Purpose: To indicate if there were no synapses and then some synapses
#     """
#     attr_name = "synapses"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = len(getattr(limb_obj[branch_1_idx],attr_name))
#     n_syn_branch_2 = len(getattr(limb_obj[branch_2_idx],attr_name))
    
#     if n_syn_branch_1 > 0 and n_syn_branch_2 > 0:
#         return False
    
#     if n_syn_branch_1 > 0 or n_syn_branch_2 > 0: 
#         return True
    
#     return False
    
# def n_synapses_diff(limb_obj,
#                       branch_1_idx,
#                       branch_2_idx,
#                   synapse_type=None,
#                   verbose = False):
#     """
#     Purpose: Will return the different in number of synapses
    
#     """
#     attr_name = "synapses"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = len(getattr(limb_obj[branch_1_idx],attr_name))
#     n_syn_branch_2 = len(getattr(limb_obj[branch_2_idx],attr_name))
    
#     if verbose:
#         print(f"Using {attr_name}")
#         print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
#     return np.abs(n_syn_branch_1 - n_syn_branch_2)

def n_synapses_diff(limb_obj,
                      branch_1_idx,
                      branch_2_idx,
                    synapse_type="synapses",
                    branch_1_direction = "upstream",
                    branch_2_direction = "downstream",
                    comparison_distance = 10000,
                    nodes_to_exclude=None,
                  verbose = False,
                   **kwargs):
    """
    Purpose: Will return the different in number of synapses
    
    """
    #print(f"synapse_type inside nst = {synapse_type}")
    n_syn_branch_1 = cnu.synapses_upstream_downstream(limb_obj,
                                                            branch_1_idx,
                                                             synapse_type=synapse_type,
                                                            direction=branch_1_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    n_syn_branch_2 = cnu.synapses_upstream_downstream(limb_obj,
                                                            branch_2_idx,
                                                             synapse_type=synapse_type,
                                                            direction=branch_2_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
        
    n_syn_branch_1 = len(n_syn_branch_1)
    n_syn_branch_2 = len(n_syn_branch_2)
    
    if verbose:
        print(f"Using {synapse_type}")
        print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
    return np.abs(n_syn_branch_1 - n_syn_branch_2)

# def synapse_density_diff(limb_obj,
#                       branch_1_idx,
#                       branch_2_idx,
#                   synapse_type=None,
#                   verbose = False):
#     """
#     Purpose: Will return the different in number of synapses
    
#     """
#     attr_name = "synapse_density"
#     if synapse_type is not None:
#         attr_name += f"_{synapse_type}"
        
#     n_syn_branch_1 = getattr(limb_obj[branch_1_idx],attr_name)
#     n_syn_branch_2 = getattr(limb_obj[branch_2_idx],attr_name)
    
#     if verbose:
#         print(f"Using {attr_name}")
#         print(f"n_syn_branch_1 = {n_syn_branch_1}, n_syn_branch_2 = {n_syn_branch_2}")
    
#     return np.abs(n_syn_branch_1 - n_syn_branch_2)

def synapse_density_diff(limb_obj,
                      branch_1_idx,
                      branch_2_idx,
                  synapse_type="synapse_density",
                    branch_1_direction = "upstream",
                    branch_2_direction = "downstream",
                    comparison_distance = 10000,
                    nodes_to_exclude=None,
                  verbose = False,
                        **kwargs):
    """
    Purpose: Will return the different in number of synapses
    
    """
        
    n_syn_branch_1 = cnu.synapse_density_upstream_downstream(limb_obj,
                                                            branch_1_idx,
                                                             synapse_density_type=synapse_type,
                                                            direction=branch_1_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    n_syn_branch_2 = cnu.synapse_density_upstream_downstream(limb_obj,
                                                            branch_2_idx,
                                                             synapse_density_type=synapse_type,
                                                            direction=branch_2_direction,
                                                            distance=comparison_distance,
                                                            nodes_to_exclude=nodes_to_exclude)
    
    if verbose:
        print(f"Using {synapse_type}")
        print(f"syn_density_branch_1 = {n_syn_branch_1}, syn_density_branch_2 = {n_syn_branch_2}")
    
    return np.abs(n_syn_branch_1 - n_syn_branch_2)


def compute_edge_attributes_locally(G,
                                          limb_obj,
                                         nodes_to_compute,
                                         edge_functions,
                                         arguments_for_all_edge_functions = None,
                                         verbose=False,
                                         directional = False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                         **kwargs):
    """
    Purpose: To iterate over graph edges and compute
    edge properties and store

    Pseudocode: 
    For each nodes to compute:
        get all of the edges for that node
        For each downstream partner:
            For each function:
                compute the value and store it in the edge
    Ex: 
    G = complete_graph_from_node_ids(all_branch_idx)

    nodes_to_compute = [upstream_branch]
    edge_functions = dict(sk_angle=nst.parent_child_sk_angle,
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage)

    compute_edge_attributes_between_nodes(G,
                                             nodes_to_compute,
                                             edge_functions,
                                             verbose=True,
                                             directional = False)

    """
    G = copy.deepcopy(G)
    
    if directional:
        neighbors_func = xu.downstream_nodes
    else:
        neighbors_func = xu.get_neighbors

    func_value_dict = dict()
    for n in nodes_to_compute:
        func_value_dict[n] = dict()
        down_nodes = neighbors_func(G,n)
        if verbose:
            print(f"Working on node {n}")

        for d in down_nodes:
            func_value_dict[n][d] = dict()
            if verbose:
                print(f"   Neighbor {d}")
            for func_name,func_info in edge_functions.items():
                if not callable(func_info):
                    func = func_info["function"]
                    if "arguments" in func_info.keys():
                        args = func_info["arguments"]
                    else:
                        args = dict()
                else:
                    func = func_info
                    args = dict()
                    
                if arguments_for_all_edge_functions is not None:
                    func_args = dict(arguments_for_all_edge_functions)
                else:
                    func_args = dict()
                
                func_args.update(args)
                
                func_value = func(limb_obj,n,d,**func_args)
            
                if verbose:
                    print(f"      {func_name}: {func_value}")
                func_value_dict[n][d][func_name] = func_value
    
    xu.apply_edge_attribute_dict_to_graph(G,func_value_dict)
    
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
                
    return G

def compute_edge_attributes_locally_upstream_downstream(
            limb_obj,
            upstream_branch,
            downstream_branches,
            offset=1500,
            comparison_distance = 2000,
            plot_extracted_skeletons = False,
            concept_network_comparison_distance = 10000,
            synapse_density_diff_type = "synapse_density_pre",
            n_synapses_diff_type = "synapses_pre",
            
    
    ):
    """
    To compute a graph storing the values for the
    edges between the nodes
    """
    all_branch_idx = np.hstack([downstream_branches,[upstream_branch]])
    G = xu.complete_graph_from_node_ids(all_branch_idx)
    
    nodes_to_compute = [upstream_branch]
    
    arguments_for_all_edge_functions = dict(
                                        #nodes_to_exclude=nodes_to_exclude,
                                           branch_1_direction="upstream",
                                            branch_2_direction="downstream",
                                           comparison_distance = concept_network_comparison_distance)
    
    edge_functions = dict(sk_angle=dict(function=nst.parent_child_sk_angle,
                                        arguments=dict(offset=offset,
                                                      comparison_distance=comparison_distance,
                                                      plot_extracted_skeletons=plot_extracted_skeletons)),
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage,
                         synapse_density_diff=dict(function=nst.synapse_density_diff,
                                               arguments=dict(synapse_type = synapse_density_diff_type)),
                          n_synapses_diff = dict(function=nst.n_synapses_diff,
                                                 arguments=dict(synapse_type=n_synapses_diff_type)),
                          #none_to_some_synapses = nst.none_to_some_synapses
                         )

    G_e_1 = nst.compute_edge_attributes_locally(G,
                                              limb_obj,
                                             nodes_to_compute,
                                             edge_functions,
                                                arguments_for_all_edge_functions=arguments_for_all_edge_functions,
                                             verbose=False,
                                             directional = False)
    
    nodes_to_compute = downstream_branches
    
    arguments_for_all_edge_functions = dict(
                                        #nodes_to_exclude=nodes_to_exclude,
                                           branch_1_direction="downstream",
                                            branch_2_direction="downstream",
                                           comparison_distance = concept_network_comparison_distance)
    
    edge_functions = dict(
                          sk_angle=dict(function=nst.sibling_sk_angle,
                                        arguments=dict(offset=offset,
                                                      comparison_distance=comparison_distance,
                                                plot_extracted_skeletons=plot_extracted_skeletons)),
                         width_diff = nst.width_diff,
                          width_diff_percentage = nst.width_diff_percentage,
                         synapse_density_diff=dict(function=nst.synapse_density_diff,
                                               arguments=dict(synapse_type = synapse_density_diff_type)),
                          n_synapses_diff = dict(function=nst.n_synapses_diff,
                                                 arguments=dict(synapse_type=n_synapses_diff_type)),
                         #none_to_some_synapses = nst.none_to_some_synapses
    )

    G_e_2 = nst.compute_edge_attributes_locally(G_e_1,
                                              limb_obj,
                                             nodes_to_compute,
                                             edge_functions,
                                                arguments_for_all_edge_functions=arguments_for_all_edge_functions,
                                             verbose=False,
                                             directional = False)
    return G_e_2
                
                
# --------- For the global deleteion functions ------------

def compute_edge_attributes_globally(G,
                                     edge_functions,
                                     edges_to_compute=None,
                                     arguments_for_all_edge_functions = None,
                                     verbose=False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                     **kwargs):
    """
    Purpose: to compute edge attributes
    that need the whole graph to be computed
    
    """
    G = copy.deepcopy(G)
    
    if edges_to_compute is None:
        edges_to_compute = xu.edges(G)
    
    other_edges_to_remove = []
    for e in edges_to_compute:
        if verbose:
            print(f"   Working on Edge {e}")
            
        for func_name,func_info in edge_functions.items():
            if not callable(func_info):
                func = func_info["function"]
                if "arguments" in func_info.keys():
                    args = func_info["arguments"]
                else:
                    args = dict()
            else:
                func = func_info
                args = dict()
                
                
            if arguments_for_all_edge_functions is not None:
                func_args = dict(arguments_for_all_edge_functions)
            else:
                func_args = dict()

            func_args.update(args)
                
            global_edge_dict = func(G,e[0],e[1],**func_args)
            
            if verbose:
                print(f"      {func_name}: {global_edge_dict}")
                
            xu.apply_edge_attribute_dict_to_graph(G,global_edge_dict,label=func_name)
            
            
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
    return G
            
def edges_to_delete_from_threshold_and_buffer(G,
                                              u,
                                              v,
                                        edge_attribute="sk_angle",
                                        threshold = 45,
                                        buffer = 15,
                                       verbose = False,
                                             **kwargs):
    """
    4) Create definite pairs by looking for edges that meet:
    - match threshold
    - have buffer better than other edges
    ** for those edges, eliminate all edges on those
    2 nodes except that edge

    Pseudocode: 
    Iterate through each edge:
    a) get the current weight of this edge
    b) get all the other edges that are touching the two nodes and their weights
    c) Run the following test on the edge:
       i) Is it in the match limit
       ii) is it less than other edge weightbs by the buffer size
    d) If pass the tests then delete all of the other edges from the graph
    
    Ex: 
    edges_to_delete = edges_to_delete_from_threshold_and_buffer(G,
                                                            225,
                                                            226,
                                          threshold=100,
                                          buffer= 13,
                                         verbose = True)
    """
    #verbose = True
    e = [u,v]
    
    edges_to_delete_dict = dict()
    e = np.sort(e)
    if verbose:
        print(f"--Working on edge {e}-- (attribute = {edge_attribute}, buffer = {buffer}, threshold= {threshold})")
    #e_weight = xu.get_edge_weight(G,e)
    e_weight = G[e[0]][e[1]][edge_attribute]
    all_edges = np.unique(
                np.sort(
                np.array(xu.node_to_edges(G,e[0]) + xu.node_to_edges(G,e[1])),axis=1)
                ,axis=0)


    #b) get all the other edges that are touching the two nodes and their weights
    other_edges = nu.setdiff2d(all_edges,e.reshape(-1,2))

    if len(other_edges) == 0:
        other_edge_min = np.inf
    else:
        other_edge_weights = [G[edg[0]][edg[1]][edge_attribute] for edg in other_edges]
        other_edge_min = np.min(other_edge_weights)

    edge_buffer = other_edge_min - e_weight
    if verbose:
        print(f"edge_buffer = {edge_buffer} (other_edge_min = {other_edge_min}, e_weight = {e_weight})")
    
    if e_weight <= threshold and edge_buffer > buffer:
        if verbose:
            print(f"Edge {e} is matches definite match threshold with: "
                 f"\nEdge Buffer of {edge_buffer} (buffer = {buffer})"
                 f"\nEdge distane of {e_weight} (threshold = {threshold})")
            print(f"other_edges = {other_edges}")
            
        edges_to_delete_dict = xu.edge_attribute_dict_from_edges(other_edges)
        edges_to_keep_dict = xu.edge_attribute_dict_from_edges([e],value_to_store=False)
        edges_to_delete_dict = xu.combine_edge_attributes([edges_to_delete_dict,edges_to_keep_dict])
        if verbose:
            print(f"edges_to_delete_dict = {edges_to_delete_dict}")
        
    return edges_to_delete_dict



# -------------- For node level edge attributes ------------- 3
def compute_edge_attributes_around_node(G,
                                        edge_functions,
                                        edge_functions_args = dict(),
                                        nodes_to_compute=None,
                                        arguments_for_all_edge_functions = None,
                                        verbose=False,
                                        #directional = False,
                                         set_default_at_end = True,
                                         default_value_at_end = None,
                                        **kwargs):
    """
    Purpose: To use all the edges around a node 
    to compute edge features

    """
    G = copy.deepcopy(G)
    
    if nodes_to_compute is None:
        nodes = list(G.nodes())
    
    if not nu.is_array_like(nodes_to_compute):
        nodes_to_compute = [nodes_to_compute]
        
    if verbose:
        print(f"nodes_to_compute = {nodes_to_compute}")

    for n in nodes_to_compute:
        node_edges = xu.node_to_edges(G,n)
        for func_name,func_info in edge_functions.items():
            if not callable(func_info):
                func = func_info["function"]
                if "arguments" in func_info.keys():
                    args = func_info["arguments"]
                else:
                    args = dict()
            else:
                func = func_info
                args = dict()
                
            if arguments_for_all_edge_functions is not None:
                func_args = dict(arguments_for_all_edge_functions)
            else:
                func_args = dict()

            func_args.update(args)
                
            node_edge_dict = func(G,node_edges,**func_args)
            
            if verbose:
                print(f"      {func_name}: {node_edge_dict}")
                
            xu.apply_edge_attribute_dict_to_graph(G,node_edge_dict,label=func_name)
            
            
    if set_default_at_end:
        for func_name,func_info in edge_functions.items():
            d_value = default_value_at_end
            try:
                if "default_value" in func_info:
                    d_value = func_info["defualt_value"]
            except:
                pass
            xu.set_edge_attribute_defualt(G,func_name,d_value)
    return G
            
def edges_to_delete_on_node_above_threshold_if_one_below(G,
                                                  node_edges,
                                                   threshold,
                                                   edge_attribute="sk_angle",
                                                  verbose = False):
    """
    Purpose: To mark edges that should
    be deleted if there is another node
    that is already below the threshold
    
    Pseudocode:
    1) Get the values of the attribute for all of the edges
    2) Get the number of these values below the threshold
    3) If at least one value below, then get the edges that are above 
    the threshold, turn them into an edge_attribute dict and return
    """
    if len(node_edges) == 0:
        return dict()
    
    edge_values = np.array([G[e[0]][e[1]][edge_attribute] for e in node_edges])
    under_threshold_mask = edge_values<=threshold
    n_below_match = np.sum(under_threshold_mask)
    
    if verbose:
        print(f"for edge_values = {edge_values} \nn_below_match = {n_below_match} (with threshold = {threshold})")
        
    if n_below_match == 0:
        return dict()
    
    node_edges = np.array(node_edges)
    edges_to_delete = node_edges[edge_values>threshold]
    
    if verbose:
        print(f"edges_to_delete = {edges_to_delete}")
    return xu.edge_attribute_dict_from_edges(edges_to_delete)



def find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                         branch_1_idx,
                                                         branch_2_idx,
                                                         branch_1_type = "upstream",
                                                         branch_2_type = "downstream",
                                                        verbose = False,
                                                        offset=1500,
                                                        min_comparison_distance = 1000,
                                                        comparison_distance = 2000,
                                                         skeleton_resolution = 100,
                                                        plot_extracted_skeletons = False,
                                                         use_upstream_skeleton_restriction=True,
                                                         use_downstream_skeleton_restriction = True,
                                                         nodes_to_exclude = None,
                                                         **kwargs
                                                        ):
    """
    Purpose: to find the skeleton angle between a designated
    upstream and downstream branch
    
    Ex: 
    nru.find_parent_child_skeleton_angle_upstream_downstream(
        limb_obj = neuron_obj[0],
    branch_1_idx = 223,
    branch_2_idx = 224,
        plot_extracted_skeletons = True
    )
    
    
    Ex: 
    branch_idx = 140
    nru.find_parent_child_skeleton_angle_upstream_downstream(limb_obj,
                                                            nru.upstream_node(limb_obj,branch_idx),branch_idx,
                                                            verbose = True,
                                                            plot_extracted_skeletons=True,
                                                            comparison_distance=40000,
                                                            use_upstream_skeleton_restriction=True)
    """
    output_sk = []
    endpoints = []
    for b_idx,b_type in zip([branch_1_idx,branch_2_idx],
                            [branch_1_type,branch_2_type]):
        
        upstream_func = False
        if b_type == "downstream":
            b_endpt = nru.upstream_endpoint(limb_obj,
                                 b_idx)
            if use_downstream_skeleton_restriction:
                downstream_func = True
        else: 
            b_endpt = nru.downstream_endpoint(limb_obj,
                                 b_idx)
            if use_upstream_skeleton_restriction:
                upstream_func = True
        
        if upstream_func:
            curr_sk = nru.restrict_skeleton_from_start_plus_offset_upstream(
                                                                limb_obj,
                                                                b_idx,
                                                                offset=offset,
                                                                comparison_distance=comparison_distance,
                                                                min_comparison_distance=min_comparison_distance,
                                                                verbose=verbose,
                                                                start_coordinate=b_endpt,
                                                                skeleton_resolution = skeleton_resolution,
                                                                nodes_to_exclude = nodes_to_exclude)
        elif downstream_func:
            curr_sk = nru.restrict_skeleton_from_start_plus_offset_downstream(
                                                                limb_obj,
                                                                b_idx,
                                                                offset=offset,
                                                                comparison_distance=comparison_distance,
                                                                min_comparison_distance=min_comparison_distance,
                                                                verbose=verbose,
                                                                start_coordinate=b_endpt,
                                                                skeleton_resolution = skeleton_resolution,
                                                                nodes_to_exclude=nodes_to_exclude)
        else:
            curr_sk = sk.restrict_skeleton_from_start_plus_offset(limb_obj[b_idx].skeleton,
                                                   offset=offset,
                                                comparison_distance=comparison_distance,
                                                    min_comparison_distance=min_comparison_distance,
                                                verbose=verbose,
                                                 start_coordinate=b_endpt,
                                                                  skeleton_resolution = skeleton_resolution
                                                   )
        output_sk.append(curr_sk)
        endpoints.append(b_endpt)
        
        if verbose:
            print(f"{b_type} {b_idx} endpoint: {b_endpt}")

    up_sk = output_sk[0]
    d_sk = output_sk[1]
    
#     sk.restrict_skeleton_from_start_plus_offset(limb_obj[branch_1_idx].skeleton,
#                                                    offset=offset,
#                                                 comparison_distance=comparison_distance,
#                                                     min_comparison_distance=min_comparison_distance,
#                                                 verbose=verbose,
#                                                  start_coordinate=b_1_endpt,
#                                                    )

#     d_sk = sk.restrict_skeleton_from_start_plus_offset(limb_obj[branch_2_idx].skeleton,
#                                                    offset=offset,
#                                                 comparison_distance=comparison_distance,
#                                                     min_comparison_distance=min_comparison_distance,
#                                                 verbose=verbose,
#                                                  start_coordinate=b_2_endpt,
#                                                    )

    curr_angle = sk.parent_child_skeletal_angle(up_sk,d_sk)
    if verbose:
        print(f"curr_angle = {curr_angle}")
        
    if plot_extracted_skeletons:
        bs = [branch_1_idx,branch_2_idx]
        parent_color = "red"
        child_color = "blue"
        print(f"Parent ({branch_1_idx}):{parent_color}, child ({branch_2_idx}):{child_color}")
        c = [parent_color,child_color]
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in bs],
                         meshes_colors=c,
                         skeletons =[up_sk,d_sk],
                         skeletons_colors=c,
                         scatters=endpoints,
                         scatters_colors = c)

    return curr_angle

    
def ray_trace_perc(branch_obj,percentile=85):
    return tu.mesh_size(branch_obj.mesh,'ray_trace_percentile',percentile)

def parent_width(limb_obj,branch_idx,width_func=None,verbose = False,**kwargs):
    if width_func is None:
        width_func = au.axon_width
    upstream_node = nru.upstream_node(limb_obj,branch_idx)
    if upstream_node is None:
        return 0
    
    return width_func(limb_obj[upstream_node])

def min_synapse_dist_to_branch_point(limb_obj,
    branch_idx,
    downstream_branches= None,
    downstream_distance=0,
    default_value = np.inf,
    plot_closest_synapse = False,
    nodes_to_exclude = None,
    synapse_type = None, #either pre or post
    verbose = False):
    """
    Purpose: To check if any of the synapses on the 
    branch or downstream branches has a synapse
    close to the branching point

    Pseudocode: 
    2) Get all of the downstream synapses (not including the current branch)
    3) Get all of the distances upstream
    4) Get the synapses for the current branch
    5) Get all fo the distances dowsntream 
    6) Concatenate the distances
    7) Find the minimum distance (if none then make inf)
    
    Ex: 
    from neurd import neuron_statistics as nst
    nst.min_synapse_dist_to_branch_point(limb_obj,
        branch_idx = 16,
        downstream_distance = 0,
        default_value = np.inf,
        plot_closest_synapse = True,
        verbose = True)

    """
    #2) Get all of the downstream synapses (not including the current branch)
    
    if downstream_branches is None:
        down_syn,down_nodes = cnu.synapses_downstream(limb_obj,branch_idx,
                                   distance = downstream_distance, 
                                   include_branch_in_dist = False,
                                   include_branch_idx=False,
                                    only_non_branching=False,
                                    nodes_to_exclude=nodes_to_exclude,
                                   plot_synapses=False,
                                          return_nodes=True)
    else:
        down_nodes = downstream_branches
        down_syn = np.concatenate([limb_obj[k].synapses for k in downstream_branches])
    
    if verbose:
        print(f"down_syn = {down_syn}, down_nodes = {down_nodes}")
        
    if synapse_type is not None:
        if "pre" in synapse_type:
            synapse_type = "pre"
        if "post" in synapse_type:
            synapse_type = "post"
        if verbose:
            print(f"Downstream synapses before synapse_type: {len(down_syn)}")
        down_syn = getattr(syu,f"synapses_{synapse_type}")(list(down_syn))
        if verbose:
            print(f"Downstream synapses AFTER synapse_type: {len(down_syn)}")
        

    #3) Get all of the distances upstream
    upstream_dist = [k.upstream_dist for k in down_syn]

    if verbose:
        print(f"upstream_dist from downstreams = {upstream_dist}")

    #4) Get the synapses for the current branch
    curr_syn = nst.synapses_upstream(limb_obj,
                                     branch_idx,
                                     nodes_to_exclude = nodes_to_exclude)
    
    if synapse_type is not None:
        if verbose:
            print(f"Upstream synapses before synapse_type: {len(curr_syn)}")
        curr_syn = getattr(syu,f"synapses_{synapse_type}")(list(curr_syn))
        if verbose:
            print(f"Upstream synapses AFTER synapse_type: {len(curr_syn)}")

    downstream_dist = [k.downstream_dist for k in curr_syn]

    if verbose:
        print(f"downstream_dist from current node = {downstream_dist}")

    all_dist = np.concatenate([upstream_dist,downstream_dist])

    if len(all_dist) > 0:
        min_distance_from_branch_point = np.min(all_dist)
    else:
        min_distance_from_branch_point = default_value

    if verbose:
        print(f"All distances: {all_dist}")
        print(f"min_distance_from_branch_point: {min_distance_from_branch_point}")

    if plot_closest_synapse:
        if len(all_dist) == 0:
            print(f"No synapses to plot")
        else:
            branch_point = nru.downstream_endpoint(limb_obj,branch_idx)
            closest_coordinate = [k.coordinate for k in np.concatenate([curr_syn,down_syn])
                                         if (k.upstream_dist == min_distance_from_branch_point 
                                             or k.downstream_dist == min_distance_from_branch_point)][0]
            nviz.plot_branch_with_neighbors(limb_obj,branch_idx,down_nodes,
                                        scatters=[branch_point.reshape(-1,3),closest_coordinate.reshape(-1,3)],
                                            scatters_colors=["red","yellow"],
                                        verbose = False)

    return min_distance_from_branch_point



# ---------------- Getting Attributes Using the concept network walking ----------- #

def skeleton_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeleton_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def skeleton_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeleton_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)


def skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)

def skeletal_length_downstream_total(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            include_branch_in_dist = True,
                            **kwargs):
    return cnu.skeletal_length_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                            include_branch_in_dist=include_branch_in_dist,
                            only_non_branching=False,
                                         **kwargs)

def skeletal_length_upstream_total(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            include_branch_in_dist = True,
                            **kwargs):
    return cnu.skeletal_length_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                            include_branch_in_dist=include_branch_in_dist,
                            only_non_branching=False,
                                         **kwargs)

def width_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.width_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                       **kwargs)

def width_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                            **kwargs):
    return cnu.width_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)

def synapses_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                           synapse_type="synapses",
                            **kwargs):
    return cnu.synapses_upstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=nodes_to_exclude,
                                synapse_type=synapse_type,
                                       **kwargs)

def synapses_downstream(limb_obj,
                            branch_idx,
                            nodes_to_exclude=None,
                        synapse_type="synapses",
                            **kwargs):
    return cnu.synapses_downstream(limb_obj,
                            branch_idx,
                            synapse_type=synapse_type,
                            nodes_to_exclude=nodes_to_exclude,
                                         **kwargs)
def n_synapses_upstream(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_upstream(limb_obj,
                            branch_idx,
                                       **kwargs))

def n_synapses_downstream(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_upstream(limb_obj,
                            branch_idx,
                                       **kwargs))
def synapses_downstream_total(limb_obj,
                            branch_idx,
                             distance=np.inf,
                            **kwargs):
    return nst.synapses_downstream(limb_obj,
                            branch_idx,
                                   distance=distance,
                                   only_non_branching=False,
                                       **kwargs)
def synapses_upstream_total(limb_obj,
                            branch_idx,
                             distance=np.inf,
                            **kwargs):
    return nst.synapses_upstream(limb_obj,
                            branch_idx,
                                   distance=distance,
                                   only_non_branching=False,
                                       **kwargs)
def n_synapses_upstream_total(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_upstream_total(limb_obj,
                            branch_idx,
                                       **kwargs))

def n_synapses_downstream_total(limb_obj,
                       branch_idx,**kwargs):
    return len(nst.synapses_downstream_total(limb_obj,
                            branch_idx,
                                       **kwargs))


def synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses",
                            distance = 5000,
                                    
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    """
    purpose: to find thenumber of downstream
    postsyns within a certain downstream distance

    """

    post_syns = cnu.synapses_downstream(limb_obj,
                           branch_idx = branch_idx,
                           distance = distance,
                            only_non_branching=False,
                            include_branch_in_dist=False,
                            include_branch_idx=False,
                            synapse_type=synapse_type,
                                        **kwargs
                           )
    if verbose:
        print(f"# of postsyns = {len(post_syns)}")

    if plot_synapses:
        syn_coords = syu.synapses_to_coordinates(post_syns)
        nviz.plot_objects(limb_obj.mesh,
                         meshes=[limb_obj[branch_idx].mesh],
                         meshes_colors="red",
                         scatters=[syn_coords],
                         scatter_size=1)
    return post_syns

def synapses_post_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_post",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,)

def synapses_pre_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_pre",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,)

def n_synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return len(synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,))

def n_synapses_post_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return len(synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_post",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,))

def n_synapses_pre_downstream_within_dist(limb_obj,
                            branch_idx,
                            distance = 5000,
                            plot_synapses = False,
                            verbose = False,
                            **kwargs,):
    return len(synapses_downstream_within_dist(limb_obj,
                            branch_idx,
                            synapse_type="synapses_pre",
                            distance = distance,
                            plot_synapses = plot_synapses,
                            verbose = verbose,
                            **kwargs,))



def fork_divergence(limb_obj,
    branch_idx,
    downstream_idxs = None,

    #arguments for skeletons
    skeleton_distance = 10000,

    error_not_2_downstream = True,

    #arguments for enforcing skipping rule for fork check
    total_downstream_skeleton_length_threshold = 0,#3000#4000
    individual_branch_length_threshold = 2000,#3000
    skip_value = np.inf,

    # for visualizing the current plot
    plot_fork_skeleton = False,

    #arguments for the fork divergence measurement
    comparison_distance = 400,
    skeletal_segment_size = 40,
    plot_restrictions = False,
    combining_function = np.mean,
                    
                nodes_to_exclude = None,
                   verbose = False):

    """
    Purpose: To run the fork divergence the children
    of an upstream node

    Pseudocode: 
    1) Get downstream nodes
    2) Apply skeletal length restrictions if any
    3) compute the fork divergence from the skeletons

    """


    upstream_node = branch_idx

    if downstream_idxs is None:
        downstream_nodes = cnu.downstream_nodes(limb_obj,branch_idx)
    else:
        downstream_nodes = downstream_idxs

    if verbose:
        print(f"downstream_nodes = {downstream_nodes}")

    upstream_sk = limb_obj[upstream_node].skeleton
    downstream_sk = [cnu.skeleton_downstream(limb_obj,d,distance=skeleton_distance)
                     for d in downstream_nodes]


    return_value = None


    if error_not_2_downstream and len(downstream_nodes) != 2:
        raise Exception(f"Not exactly 2 downstream nodes: {downstream_nodes}")

    if (total_downstream_skeleton_length_threshold is not None and
        individual_branch_length_threshold is not None):
        d_len = np.array([sk.calculate_skeleton_distance(limb_obj[k].skeleton) for 
                         k in downstream_nodes])
        d_skeletal_len = np.array([nru.skeletal_length_over_downstream_branches(limb_obj,
                                                d,
                                                verbose=False) for d in downstream_nodes])
        if verbose:
            print(f"skeletal length = {d_len}")
            print(f"downstream skeletal length = {d_skeletal_len}")

        below_threshold = np.where((d_skeletal_len<total_downstream_skeleton_length_threshold) | 
                                   (d_len < individual_branch_length_threshold))[0]
        if len(below_threshold) > 0:
            if verbose:
                print(f"Skipping this intersection because some of downstream skeletal lengths too short (min {total_downstream_skeleton_length_threshold}):")
                print(f" or the individual branch length was too short (min {individual_branch_length_threshold})")
                for j,(d,d_len) in enumerate(zip(downstream_nodes,d_skeletal_len)):
                    if j in below_threshold:
                        print(f"Brnach {d}: length {d_len}")

            return_value = skip_value


    if plot_fork_skeleton:
        nviz.plot_branch_with_neighbors(limb_obj,30,main_skeleton = upstream_sk,
                                       skeletons=downstream_sk)

    if return_value is None:
        return_value = nst.fork_divergence_from_skeletons(upstream_skeleton = upstream_sk,
                                downstream_skeletons = downstream_sk,
                                comparison_distance = comparison_distance,
                                skeletal_segment_size = skeletal_segment_size,
                                plot_restrictions = plot_restrictions,
                                combining_function = combining_function,
                                                             verbose=verbose)

    if verbose:
        print(f"return_value = {return_value}")
        
    return return_value

def compute_node_attributes(G,
                            limb_obj,
                            node_functions,
                           verbose = False):
    """
    Purpose: To Compute node attributes given:
    - function
    - arguments for function
    - nodes to compute for (so can explicitely do for upstream and downstream)

    Each of this will be stored in a list of dictionaries
    """


    G_nodes = list(G.nodes())
    att_dict = dict([(k,dict()) for k in G_nodes])
    
    for f_info  in node_functions:
        f_name = f_info["name"]
        func = f_info["function"]
        nodes_to_compute = f_info.get("nodes_to_compute",G_nodes)
        if not nu.is_array_like(nodes_to_compute):
            nodes_to_compute = [nodes_to_compute]
        
        args = f_info.get("arguments",dict())
        default = f_info.get("default",None)
        
        if verbose:
            print(f"Working on {f_name} with args = {args}")

        nodes_not_compute = np.setdiff1d(G_nodes,nodes_to_compute)

        for n in nodes_to_compute:
            att_dict[n][f_name] = func(limb_obj=limb_obj,
                                       branch_idx=n,
                                       **args)

        if default is not None:
            for n in nodes_not_compute:
                att_dict[n][f_name] = default

    xu.set_node_attributes_dict(G,att_dict)
    return G


def node_functions_default(upstream_branch,
                   downstream_branches,
                  ):
    """
    To create the defautl node attributes
    wanting to compute
    """
    return [
    dict(name="skeletal_length_downstream",
     function = nst.skeletal_length_downstream,
     nodes_to_compute = downstream_branches),

    dict(name="skeletal_length_downstream_total",
     function = nst.skeletal_length_downstream_total,
     nodes_to_compute = downstream_branches,),
        
    dict(name="skeletal_length_upstream_total",
     function = nst.skeletal_length_upstream_total,
     nodes_to_compute = upstream_branch,),
        
    dict(name="skeletal_length_upstream",
     function = nst.skeletal_length_upstream,
     nodes_to_compute = upstream_branch,),
    
    dict(name="width_downstream",
     function = nst.width_downstream,
     nodes_to_compute = downstream_branches,),
    
    dict(name="width_upstream",
     function = nst.width_upstream,
     nodes_to_compute = upstream_branch,), 
    
    dict(name="n_synapses_upstream",
        function = nst.n_synapses_upstream,
        nodes_to_compute = upstream_branch),
    
    dict(name="n_synapses_downstream",
        function = nst.n_synapses_downstream,
        nodes_to_compute = downstream_branches),
    
    dict(name="n_synapses_downstream_total",
        function = nst.n_synapses_downstream_total,
        nodes_to_compute = downstream_branches),
    
    ]

def compute_node_attributes_upstream_downstream(G,
                                                limb_obj,
                                               upstream_branch,
                                               downstream_branches,
                                                node_functions=None,
                                               verbose = False):
    """
    Purpose: To attach node properties to a graph
    that references branches on a limb
    
    """
    if node_functions is None:
        node_functions = node_functions_default(upstream_branch=upstream_branch,
                            downstream_branches=downstream_branches)
        
    node_type_dict = gu.merge_dicts([{upstream_branch:dict(node_type="upstream")},
                        {k:dict(node_type="downstream") for k in downstream_branches}])
    
    xu.set_node_attributes_dict(G,node_type_dict)

    nst.compute_node_attributes(G,
                           limb_obj,
                            node_functions=node_functions,
                           verbose = verbose)
    
    if verbose:
        print(xu.node_df(G))
        
    return G


def synapse_closer_to_downstream_endpoint_than_upstream(branch_obj):
    """
    Purpose: Will indicate if there is a synapse that is closer to the downstream endpoint than upstream endpoint
    
    """
    return_value = False
    for syn in branch_obj.synapses:
        if syn.downstream_dist < syn.upstream_dist:
            return_value = True
    
    return return_value


    

def downstream_upstream_diff_of_most_downstream_syn(branch_obj,
                                                   default_value = 0):
    """
    Purpose: Determine the difference between 
    the closest downstream dist and 
    the farthest upstream dist

    Pseudocode: 
    1) Get the synapse with min of downstream dist
    2) Get the difference between downstream dist and upstream dist
    3) Return the difference
    """
    return_value = default_value
    syns = branch_obj.synapses
    if len(syns) > 0:
        d_dists = np.array([k.downstream_dist for k in syns])
        syn_idx = np.argmin(d_dists)
        d_min = d_dists[syn_idx]
        u_max = syns[syn_idx].upstream_dist
        return_value = d_min - u_max
    return return_value


# ------------ New Filter: 7/8 ----------
def fork_min_skeletal_distance_from_skeletons(downstream_skeletons,
                                             comparison_distance = 3000,
                                             offset = 700,
                                              skeletal_segment_size = 40,
                                             verbose = False,
                                             plot_skeleton_restriction=False,
                                              plot_min_pair = False
                                             ):
    """
    Purpose: To determine the min distance from two diverging
    skeletons with an offset
    """
    joining_endpoint_1 = sk.shared_coordiantes(downstream_skeletons,
                                 return_one=True)

    d_skeletons_resized = [sk.resize_skeleton_branch(k,segment_width=skeletal_segment_size)
                              for k in downstream_skeletons]

    new_sks = [sk.restrict_skeleton_from_start_plus_offset(k,
                                               offset = offset,
                                               comparison_distance = comparison_distance,
                                                start_coordinate = joining_endpoint_1,
                                               ) for k in d_skeletons_resized]

    if plot_skeleton_restriction:
        nviz.plot_objects(skeletons=new_sks, # + d_skeletons_resized,
                         skeletons_colors=["red","blue"],#,"yellow","yellow"],
                         scatters=[joining_endpoint_1])

    distances_between_skeletons = sk.closest_distances_from_skeleton_vertices_to_base_skeleton(new_sks[0],
                                                              new_sks[1],
                                                              verbose= verbose,
                                                              plot_min_pair=plot_min_pair)
    min_dist = np.min(distances_between_skeletons)
    if verbose:
        print(f"min_dist= {min_dist}")
        print(f"distances_between_skeletons = {distances_between_skeletons}")
        
    return min_dist

def fork_min_skeletal_distance(limb_obj,
    branch_idx,
    downstream_idxs = None,

    #arguments for skeletons
    skeleton_distance = 10000,

    error_not_2_downstream = True,

    #arguments for enforcing skipping rule for fork check
    total_downstream_skeleton_length_threshold = 0,#3000#4000
    individual_branch_length_threshold = 2000,#3000
    skip_value = np.inf,

    #arguments for the fork divergence measurement
    comparison_distance = 2000,
    offset = 700,#2000,#700,
    skeletal_segment_size = 40,
    plot_skeleton_restriction = False,  
    plot_min_pair=False,
    nodes_to_exclude = None,
    verbose = False):

    """
    Purpose: To run the fork divergence the children
    of an upstream node

    Pseudocode: 
    1) Get downstream nodes
    2) Apply skeletal length restrictions if any
    3) compute the fork skeleton min distance

    Ex: 
    from neurd import neuron_statistics as nst

    upstream_branch = 68
    downstream_branches = [55,64]
    verbose = False
    div = nst.fork_min_skeletal_distance(limb_obj,upstream_branch,
                                      downstream_idxs = downstream_branches,
                                  total_downstream_skeleton_length_threshold=0,
                                  individual_branch_length_threshold = 0,
                       plot_skeleton_restriction = False,
                       verbose = verbose)
    """


    upstream_node = branch_idx

    if downstream_idxs is None:
        downstream_nodes = cnu.downstream_nodes(limb_obj,branch_idx)
    else:
        downstream_nodes = downstream_idxs

    if verbose:
        print(f"downstream_nodes = {downstream_nodes}")

    upstream_sk = limb_obj[upstream_node].skeleton
    downstream_sk = [cnu.skeleton_downstream(limb_obj,d,distance=skeleton_distance)
                     for d in downstream_nodes]


    return_value = None


    if error_not_2_downstream and len(downstream_nodes) != 2:
        raise Exception(f"Not exactly 2 downstream nodes: {downstream_nodes}")

    if (total_downstream_skeleton_length_threshold is not None and
        individual_branch_length_threshold is not None):
        d_len = np.array([sk.calculate_skeleton_distance(limb_obj[k].skeleton) for 
                         k in downstream_nodes])
        d_skeletal_len = np.array([nru.skeletal_length_over_downstream_branches(limb_obj,
                                                d,
                                                verbose=False) for d in downstream_nodes])
        if verbose:
            print(f"skeletal length = {d_len}")
            print(f"downstream skeletal length = {d_skeletal_len}")

        below_threshold = np.where((d_skeletal_len<total_downstream_skeleton_length_threshold) | 
                                   (d_len < individual_branch_length_threshold))[0]
        if len(below_threshold) > 0:
            if verbose:
                print(f"Skipping this intersection because some of downstream skeletal lengths too short (min {total_downstream_skeleton_length_threshold}):")
                print(f" or the individual branch length was too short (min {individual_branch_length_threshold})")
                for j,(d,d_len) in enumerate(zip(downstream_nodes,d_skeletal_len)):
                    if j in below_threshold:
                        print(f"Brnach {d}: length {d_len}")

            return_value = skip_value


    if return_value is None:
        
        return_value = nst.fork_min_skeletal_distance_from_skeletons(
                                downstream_skeletons = downstream_sk,
                                comparison_distance = comparison_distance,
                                offset=offset,
                                skeletal_segment_size = skeletal_segment_size,
                                plot_skeleton_restriction = plot_skeleton_restriction,
                                plot_min_pair=plot_min_pair,
                                verbose=verbose)
    if verbose:
        print(f"return_value = {return_value}")
        
    return return_value


def shortest_distance_from_soma_multi_soma(limb_obj,
                      branches,
                      somas=None,
                      include_node_skeleton_dist = False,
                      verbose = False,
                       return_dict = False,
                       **kwargs
                      ):
    """
    Purpose: To find the distance of a branch from the soma 
    (if there are multiple somas it will check for shortest distance between all of them)
    
    Ex: 
    nst.shortest_distance_from_soma_multi_soma(neuron_obj_exc_syn_sp[0],190)
    """
    singular_flag = False
    if not nu.is_array_like(branches):
        branches = [branches]
        singular_flag = True
        
    return_d = nru.skeletal_distance_from_soma(limb_obj,
                                                  branches=branches,
                    somas = somas,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=include_node_skeleton_dist,
                    print_flag = verbose,
                    **kwargs)
    
    if return_dict:
        return return_d
    return_value = [return_d[k] for k in branches]
    
    if singular_flag:
        return_value = return_value[0]
        
    return return_value

def distance_from_soma(limb_obj,
                      branch_idx,
                      include_node_skeleton_dist = False,
                      verbose = False,
                      **kwargs):
    """
    Purpose: To find the distance away from the soma
    for a given set of branches
    
    Ex: 
    nst.distance_from_soma(limb_obj,190)
    """
    
    singular_flag = False
    if not nu.is_array_like(branch_idx):
        branches = [branch_idx]
        singular_flag = True
    else:
        branches = branch_idx
        
    branch_paths = [nru.branch_path_to_start_node(
                    limb_obj,k,include_branch_idx=include_node_skeleton_dist) for k in branches]
    
    return_value = [nru.sum_feature_over_branches(limb_obj,k,
                                                 feature_name = "skeletal_length") for k in branch_paths]
    
    if verbose:
        print(f"branch_paths = {branch_paths}")
        print(f"Path lengths = {return_value}")
        
        
    if singular_flag:
        return_value = return_value[0]
        
    return return_value

def distance_from_soma_euclidean(limb_obj,
                                branch_idx,):
    """
    Will return the euclidean distance of the upstream endpoint
    to the starting coordinate of the limb
    
    Ex: 
    branch_idx = 0
    limb_obj = neuron_obj_proof[0]
    nst.distance_from_soma_euclidean(limb_obj,branch_idx)
    """
    upstream_endpoint = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=False)
    return np.linalg.norm(limb_obj.current_starting_coordinate - upstream_endpoint)

def distance_from_soma_candidate(neuron_obj,candidate):
    """
    Purpose: Will return the distance of a candidate
    
    """
    a = candidate 
    limb_obj = neuron_obj[a["limb_idx"]]
    branch_idx = a["start_node"]
    downstream_nodes = a["branches"]
    return  nst.distance_from_soma(limb_obj,branch_idx)


def width_basic(branch_obj):
    return branch_obj.width
                            
    
def farthest_dendrite_branch_from_soma(neuron_obj):
    from neurd import neuron_searching as ns
    dist_from_soma_df = ns.query_neuron(neuron_obj,
                    functions_list=[ns.distance_from_soma],
                   query="distance_from_soma > -1",
                   return_dataframe=True,
                    limb_branch_dict_restriction=neuron_obj.dendrite_limb_branch_dict)
    max_distance = np.max(dist_from_soma_df["distance_from_soma"].to_numpy())
    return dist_from_soma_df[dist_from_soma_df["distance_from_soma"] == max_distance]

def trajectory_angle_from_start_branch_and_subtree(limb_obj,
                                                  subtree_branches,
                                                   start_branch_idx=None,
                                                  nodes_to_exclude = None,
                                                   downstream_distance = 10000,
                                                plot_skeleton_before_restriction = False,
                                                plot_skeleton_after_restriction = False,
                                                plot_skeleton_endpoints = False,
                                                return_max_min = False,
                                                return_n_angles = False,
                                                verbose = False,
                                                  ):

    """
    Purpose: To figure out the 
    initial trajectory of a subtree
    of branches if given the initial 
    branch of the subtree and all
    the branches of the subtree

    Pseudocode: 
    1) Get all branches that are within a certain distance of the starting branch
    2) Get the upstream coordinate of start branch
    3) Restrict the skeleton to the downstream distance
    4) Find all endpoints of the restricted skeleton
    5) Calculate the vectors and angle from the top of the start coordinate and all the endpoints

    Ex: 
    nst.trajectory_angle_from_start_branch_and_subtree(
    limb_obj = neuron_obj_exc_syn_sp[limb_idx],
    start_branch_idx = 31,
    subtree_branches = [31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101],
    nodes_to_exclude = nodes_to_exclude,
    plot_skeleton_endpoints = plot_skeleton_endpoints,
    return_max_min=True,
    return_n_angles=True
)
    """
    if start_branch_idx is None:
        G = limb_obj.concept_network_directional
        G_subtree = G.subgraph(subtree_branches)
        start_branch_idx = xu.starting_node_from_DiG(G_subtree)
        if verbose:
            print(f"start_branch computed as {start_branch_idx} (because original set to None)")
    
    
    if verbose:
        print(f"nodes_to_exclude = {nodes_to_exclude}")

    downstream_skeleton = cnu.skeleton_downstream(limb_obj,
                                                 branch_idx = start_branch_idx,
                                                 distance = downstream_distance,
                                                 only_non_branching = False,
                                                  nodes_to_exclude=nodes_to_exclude
                                                 )
    starting_coordinate = nru.upstream_endpoint(limb_obj,start_branch_idx)

    if plot_skeleton_before_restriction:
        nviz.plot_objects(limb_obj.mesh,
                         skeletons=[downstream_skeleton],
                         scatters=[starting_coordinate],
                         scatter_size=1)

    restricted_sk = sk.restrict_skeleton_to_distance_from_coordinate(downstream_skeleton,
                                                    coordinate=starting_coordinate,
                                                    distance_threshold=downstream_distance)

    if plot_skeleton_after_restriction:
        nviz.plot_objects(limb_obj.mesh,
                         skeletons=[restricted_sk],
                         scatters=[starting_coordinate],
                         scatter_size=1)

    endpoints = sk.find_skeleton_endpoint_coordinates(restricted_sk,coordinates_to_exclude = starting_coordinate)

    if verbose:
        print(f"endpoints = {endpoints}")

    if plot_skeleton_endpoints:
        start_color = "red"
        endpoint_color = "blue"
        print(f"Starting point = {start_color}, endpoints = {endpoint_color}")
        nviz.plot_objects(limb_obj.mesh,
                         skeletons=[restricted_sk],
                         scatters=[starting_coordinate,endpoints],
                          scatters_colors=[start_color,endpoint_color],
                         scatter_size=0.5)

    subtree_vectors = [nu.vector_from_endpoints(starting_coordinate,k) for k in endpoints] 
    if verbose:
        print(f"subtree_vectors = {subtree_vectors}")

    subtree_angles = [nst.angle_from_top(k) for k in subtree_vectors]

    if verbose:
        print(f"subtree_angles = {subtree_angles}")

    if return_max_min:
        max_angle = np.max(subtree_angles)
        min_angle = np.min(subtree_angles)
        if return_n_angles:
            return max_angle,min_angle, len(subtree_angles)
        else:
            return max_angle,min_angle
    else:
        return subtree_angles
    

# ------- 7/26: To help identify axons  -----------
distance_away_from_endpoint= 6_000
#distance_away_from_endpoint= 10_000
def synapse_density_post_near_endpoint_downstream(branch_obj,
                                                 distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    """
    Purpose: To get the synapse density near the downstream endpoint
    """
    return syu.synapse_density_post_within_distance_of_endpoint_downstream(branch_obj,
                                                                          distance=distance,
                                                                          verbose=verbose)

def n_synapses_spine_within_distance_of_endpoint_downstream(branch_obj,
                                                            distance = distance_away_from_endpoint,
                                                            verbose = False,
                                                            **kwargs
                                                           ):
    return syu.n_synapses_spine_within_distance_of_endpoint_downstream(branch_obj,
                                                                          distance=distance,
                                                                          verbose=verbose)

def synapse_density_post_offset_endpoint_upstream(branch_obj,
                                                 distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    return syu.synapse_density_post_offset_distance_of_endpoint_upstream(branch_obj,
                                                                    distance=distance,
                                                                    verbose = verbose,)

def synapse_density_offset_endpoint_upstream(branch_obj,
                                                 distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    return syu.synapse_density_offset_distance_of_endpoint_upstream(branch_obj,
                                                                    distance=distance,
                                                                    verbose = verbose,)

def n_synapses_offset_endpoint_upstream(branch_obj,
                                                   distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    return syu.n_synapses_offset_distance_of_endpoint_upstream(branch_obj,
                                                                    distance=distance,
                                                                    verbose = verbose,)

def n_synapses_spine_offset_endpoint_upstream(branch_obj,
                                                 distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    return syu.n_synapses_spine_offset_distance_of_endpoint_upstream(branch_obj,
                                                                    distance=distance,
                                                                    verbose = verbose,)

def n_synapses_pre_offset_endpoint_upstream(branch_obj,
                                                 distance = distance_away_from_endpoint,
                                                  verbose = False,
                                                  **kwargs
                                                 ):
    return syu.n_synapses_pre_offset_distance_of_endpoint_upstream(branch_obj,
                                                                    distance=distance,
                                                                    verbose = verbose,)



# ------------- 7/28: for apical -----------------
def filter_limbs_by_soma_starting_angle(neuron_obj,
                                       soma_angle,
                                        angle_less_than = True,
                                       verbose = False,
                                       return_int_names = True):
    """
    Purpose: Will return the limb names that satisfy the
    soma angle requirement
    
    Ex: nst.filter_limbs_by_soma_starting_angle(neuron_obj,60,verbose=True)
    """
    
    if angle_less_than:
        query = f"soma_starting_angle<={soma_angle}"
    else:
        query = f"soma_starting_angle>={soma_angle}"
        
    if verbose:
        print(f"Restricting limbs to {query}")
        
    soma_center = neuron_obj["S0"].mesh_center
    
    possible_apical_limbs_dict = ns.query_neuron(neuron_obj,
                    query=query,
                   functions_list=[ns.soma_starting_angle],
                   function_kwargs=dict(soma_center=soma_center,
                                       verbose=verbose))

    possible_apical_limbs = list(possible_apical_limbs_dict.keys())
    
    if return_int_names:
        possible_apical_limbs = [nru.get_limb_int_name(k) for k in possible_apical_limbs]
        
    return possible_apical_limbs



def skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: 
    To find the percentage of skeleton that is
    within a certain angle of a comparison vector

    Pseudocode: 
    1) Resize the branch skeleton and order the skeleton
    2) For each segments of the skeleton:
    - extract the vector from the skeleton
    - find the angle betweeen the reference bector and the segment vector
    - if angle is below the threshold then count it as a match
    3) return the percentage of the matches

    """


    branch_obj = limb_obj[branch_idx]
    if plot_branch:
        nviz.plot_branches_with_spines(branch_obj,)

    #branch_obj
    #1) Resize the branch skeleton and order the skeleton

    perc_match,dist_match = sk.percentage_skeleton_match_to_ref_vector(branch_obj.skeleton,
                                              reference_vector=reference_vector,
                                               max_angle=max_angle,
                                                min_angle=min_angle,
                                              start_endpoint_coordinate=nru.upstream_endpoint(limb_obj,branch_idx),
                                            segment_width=skeleton_resize_distance,
                                             plot_skeleton=False,
                                              verbose=verbose,
                                              return_match_length = True)
    if verbose:
        print(f"perc_match = {perc_match}")
        print(f"dist_match = {dist_match}")
        
    return perc_match,dist_match

def skeleton_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: To return the amount of skeletal distance that matches
    a comparison vector
    
    Ex: 
    nst.skeleton_dist_match_ref_vector(neuron_obj[0],
                                       11,
                                       verbose = True)
    
    """
    perc_match,dist_match = nst.skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = max_angle,
    min_angle = min_angle,
    reference_vector = reference_vector,
    skeleton_resize_distance = skeleton_resize_distance,
    plot_branch = plot_branch,
    verbose = verbose,)
    
    return dist_match

def skeleton_perc_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = 30,
    min_angle = None,
    reference_vector = np.array([0,-1,0]),
    skeleton_resize_distance = 8000,
    plot_branch = False,
    verbose = False,
    **kwargs):
    """
    Purpose: To return the percentage of skeletal distance that matches
    a comparison vector
    
    """
    perc_match,dist_match = nst.skeleton_perc_dist_match_ref_vector(limb_obj,
    branch_idx,
    max_angle = max_angle,
    min_angle = min_angle,
    reference_vector = reference_vector,
    skeleton_resize_distance = skeleton_resize_distance,
    plot_branch = plot_branch,
    verbose = verbose,)
    
    return perc_match

def get_stat(obj,stat,**kwargs):
    """
    Purpose: Will either run the function 0n the  object
    of get the property of the function if it is a string
    
    Ex: 
    nst.get_stat(limb_obj[0],syu.n_synapses_pre)
    nst.get_stat(limb_obj[0],"skeletal_length")
    
    """
    if type(stat) == str:
        return getattr(obj,stat)
    else:
        return stat(obj,**kwargs)
    
    
# --------- Statisitcs for canddidates --------------
def skeletal_length_over_candidate(neuron_obj,
                                  candidate,
                                  **kwargs):
    sk_len = nru.skeletal_length_over_limb_branch(neuron_obj,nru.nru.limb_branch_from_candidate(candidate))
    return sk_len

def width_over_candidate(
    neuron_obj,
    candidate,
    **kwargs):
    
    return nst.width_weighted_over_branches(
        neuron_obj[candidate['limb_idx']],
        branches = candidate['branches'],
        **kwargs)

def max_layer_height_over_candidate(neuron_obj,candidate,**kwargs):
    """
    Purpose: To determine the maximum height in the
    layer
    """
    cand_sk = nru.skeleton_over_candidate(neuron_obj,candidate)
    return np.max(mcu.coordinates_to_layer_height(cand_sk.reshape(-1,3)))

def max_layer_distance_above_soma_over_candidate(neuron_obj,candidate,
                                                **kwargs):
    soma_max_height = mcu.coordinates_to_layer_height(neuron_obj["S0"].mesh_center)
    cand_height = nst.max_layer_height_over_candidate(neuron_obj,candidate)
    return cand_height - soma_max_height


def downstream_dist_match_ref_vector_over_candidate(neuron_obj,
                                                    candidate,
                                                    verbose = False,
                                                    max_angle = 65,
                                                   **kwargs):
    """
    Purpose: Measure the amount of downstream branch
    length that is at a certain angle

    1) Get all of the nodes that are downstream
    of all of the branches
    2) Add up the amount of distance on each
    branch that matches the angle specified
    
    Ex: 
    nst.downstream_dist_match_ref_vector_over_candidate(neuron_obj,
                                               candidate = winning_candidates[0],max_angle=65)
    """


    all_downstream_branches = nru.all_downstream_branches_from_candidate(neuron_obj,
                                                                        candidate)

    if verbose:
        print(f"all_downstream_branches = {all_downstream_branches}")

    if len(all_downstream_branches) > 0:
        angle_match_dist = np.sum([nst.skeleton_dist_match_ref_vector(neuron_obj[candidate['limb_idx']],
                                                  branch_idx = k,
                                                  max_angle=max_angle,
                                                 verbose = verbose) for k in all_downstream_branches])
    else: 
        angle_match_dist = 0

    if verbose:
        print(f"angle_match_dist = {angle_match_dist}")

    return angle_match_dist


# ------- help with searching for labels -------------- #
def upstream_node_is_apical_shaft(limb_obj,branch_idx,verbose,**kwargs):
    return nru.upstream_node_has_label(limb_obj,branch_idx,label="apical_shaft",verbose = verbose)
def upstream_node_is_apical(limb_obj,branch_idx,verbose,**kwargs):
    return nru.upstream_node_has_label(limb_obj,branch_idx,label="apical",verbose = verbose)

def is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   ):
    """
    Purpose: To test if a label is in the downstream 
    nodes 

    1) Get all the downstream labels
    2) return the test if a certain label is in downstream labels
    
    Ex: 
    nst.is_label_in_downstream_branches(neuron_obj[1],5,"apical_shaft",verbose = True)
    """
    if not nu.is_array_like(label):
        label = [label]
        
    downstream_labels = nru.downstream_labels(limb_obj,branch_idx,
                          all_downstream_nodes = all_downstream_nodes,
                         verbose = verbose)
    
    common_labels = np.intersect1d(downstream_labels,label)
    if verbose:
        print(f"common_labels= {common_labels}")
        
    if len(common_labels) > 0:
        return True
    else:
        return False
    
def is_apical_shaft_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   **kwargs):
    """
    Ex: 
    nst.is_apical_shaft_in_downstream_branches(neuron_obj[1],4,verbose = True)
    """
    return is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label="apical_shaft",
                                    all_downstream_nodes = all_downstream_nodes,
                                     verbose = verbose,
                                   )

def is_axon_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    all_downstream_nodes = False,
                                     verbose = False,
                                   **kwargs):
    """
    Ex: 
    nst.is_apical_shaft_in_downstream_branches(neuron_obj[1],4,verbose = True)
    """
    return is_label_in_downstream_branches(limb_obj,
                                    branch_idx,
                                    label="axon",
                                    all_downstream_nodes = all_downstream_nodes,
                                     verbose = verbose,
                                   )


# ---------- Functions over upstream and downstream branches ----------- #
def width_weighted_over_branches(limb_obj,
                                branches,
                                 width_func = None,
                                verbose = False):
    """
    Purpose: Find weighted width over branches

    Ex: 
    nst.width_weighted_over_branches(n_obj_2[6],
                                branches = [24,2])
    """
    if width_func is None:
        width_func = nst.width_new
        
    weight_width = cnu.weighted_feature_over_branches(limb_obj = limb_obj,
                                branches =branches,
                               direction=None,
                               verbose = verbose,
                               feature_function=width_func
    )
    
    return weight_width


def skeleton_dist_match_ref_vector_sum_over_branches(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=None,
                                                     direction = None,
                                                     verbose = False,
                                                    **kwargs):
    """
    Purpose: Find the amount of upstream skeletal distance
    that matches a certain angle

    """
    sk_dist = cnu.sum_feature_over_branches(limb_obj = limb_obj,
                                branches =branches,
                               direction=direction,
                               verbose = verbose,
                               feature_function=nst.skeleton_dist_match_ref_vector,
                                  use_limb_obj_and_branch_idx = True,
                                  max_angle=max_angle,
                                  min_angle = min_angle,)
    return sk_dist

def skeleton_dist_match_ref_vector_sum_over_branches_upstream(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=None,
                                                     verbose = False,
                                                    **kwargs):
    return skeleton_dist_match_ref_vector_sum_over_branches(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=min_angle,
                                                     direction = "upstream",
                                                     verbose = verbose,
                                                    **kwargs)
def skeleton_dist_match_ref_vector_sum_over_branches_downstream(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=None,
                                                     verbose = False,
                                                    **kwargs):
    """
    Purpose: To find the skeletal length of the downstream 
    branch portions that match a certain angle
    
    Ex: 
    nst.skeleton_dist_match_ref_vector_sum_over_branches_downstream(
    limb_obj = n_obj_2[6],
    branches = [23,14,27],
    max_angle = 10000,
    min_angle = 40,
    verbose = True)
    """
    return skeleton_dist_match_ref_vector_sum_over_branches(limb_obj,
                                                    branches,
                                                    max_angle,
                                                    min_angle=min_angle,
                                                     direction = "downstream",
                                                     verbose = verbose,
                                                    **kwargs)

def stats_dict_over_limb_branch(
    neuron_obj,
   limb_branch_dict=None,
   stats_to_compute = ("skeletal_length","area","mesh_volume","n_branches"),
    ):
    """
    Purpose: To get a statistics 
    over a limb branch dict

    Stats to retrieve:
    1) skeletal length
    2) surface area
    3) volume
    
    Ex: 
    from neurd import neuron_statistics as nst
    nst.stats_dict_over_limb_branch(
        neuron_obj = neuron_obj_proof,
        limb_branch_dict = apu.apical_limb_branch_dict(neuron_obj_proof))
    """
    if limb_branch_dict is None:
        limb_branch_dict= neuron_obj.limb_branch_dict
    
    s_dict = {k:nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                   limb_branch_dict,
                                                   feature=k) for k in stats_to_compute}


    return s_dict

def features_from_neuron_skeleton_and_soma_center(
    neuron_obj,
    limb_branch_dict = None,
    neuron_obj_aligned = None, 
    **kwargs
    ):
    
    if limb_branch_dict is not None:
        skeleton = nru.skeleton_over_limb_branch_dict(
            neuron_obj,
            limb_branch_dict,
        )
    else:
        skeleton = neuron_obj.skeleton
        
    if neuron_obj_aligned is not None:
        if limb_branch_dict is not None:
            skeleton_aligned = nru.skeleton_over_limb_branch_dict(
                neuron_obj_aligned,
                limb_branch_dict,
            )
        else:
            skeleton_aligned = neuron_obj_aligned.skeleton
            
        soma_center = neuron_obj_aligned["S0"].mesh_center
    else:
        skeleton_aligned = None 
        soma_center = neuron_obj["S0"].mesh_center
    
        
    
    return nst.features_from_skeleton_and_soma_center(
    skeleton,
    soma_center = soma_center,#neuron_obj["S0"].mesh_center,
    skeleton_aligned = skeleton_aligned,
    **kwargs
    )
    
def features_from_skeleton_and_soma_center(
    skeleton,
    soma_center,
    short_threshold = 6000,
    long_threshold = 100000,
    volume_divisor = 1_000_000_000_000_000,#(10**14),
    verbose = False,
    name_prefix = None,
    features_to_exclude = None,
    skeleton_aligned = None,
    in_um = True,):
    """
    Purpose: To calculate features about a skeleton
    representing a subset of the neuron (
    features specifically in relation to soma)
    
    """
    if len(skeleton) == 0:
        axon_dict =  dict(

                    length = 0,
                    branch_length_median = 0,
                    branch_length_mean = 0,

                    n_branches = 0,
                    n_short_branches = 0,
                    n_long_branches = 0,
                    n_medium_branches = 0,

                    bbox_volume=0,
                    bbox_x_min=0,
                    bbox_y_min=0,
                    bbox_z_min=0,
                    bbox_x_max=0,
                    bbox_y_max=0,
                    bbox_z_max=0,

                    bbox_x_min_soma_relative=0,
                    bbox_y_min_soma_relative=0,
                    bbox_z_min_soma_relative=0,
                    bbox_x_max_soma_relative=0,
                    bbox_y_max_soma_relative=0,
                    bbox_z_max_soma_relative=0,

                    )
    else:

        # Calculating the boudning box
        sk_branches = sk.decompose_skeleton_to_branches(skeleton)

        sk_branches_dist = np.array([sk.calculate_skeleton_distance(k) for k in sk_branches])

        n_branches = len(sk_branches)
        n_short_branches = np.sum(sk_branches_dist<short_threshold)
        n_long_branches = np.sum(sk_branches_dist>long_threshold)
        n_medium_branches = np.sum((sk_branches_dist<=long_threshold) & 
                                  (sk_branches_dist>=short_threshold))

        if verbose:
            print(f"Total Number of Branches = {(n_branches)}")
            print(f"n_short_branches = {n_short_branches}, n_medium_branches = {n_medium_branches}, n_long_branches = {n_long_branches}")

        # calculating the skeletal lengths
        if in_um:
            divisor = 1000
        else:
            divisor = 1

        axon_length = np.sum(sk_branches_dist)/divisor
        axon_branch_length_median = np.median(sk_branches_dist)/divisor
        axon_branch_length_mean = np.mean(sk_branches_dist)/divisor

        if verbose:
            print(f"axon_length = {axon_length}, axon_branch_length_median = {axon_branch_length_median}, axon_branch_length_mean = {axon_branch_length_mean}")

        bbox_volume = sk.bbox_volume(skeleton)/volume_divisor
        bbox_corners = sk.bounding_box_corners(skeleton)
        bbox_corners_soma_relative = bbox_corners - soma_center
        if skeleton_aligned is not None:
            if verbose:
                print(f"Using skeleton aligned")
                print(f"Previous bbox_corners_soma_relative = {bbox_corners_soma_relative}")
            bbox_corners_aligned = sk.bounding_box_corners(skeleton_aligned)
            bbox_corners_soma_relative = bbox_corners_aligned - soma_center
            if verbose:
                print(f"NEW ALIGNED bbox_corners_soma_relative = {bbox_corners_soma_relative}")
            

        if verbose:
            print(f"bbox_volume = {bbox_volume}")
            print(f"bbox_corners = {bbox_corners}")
            print(f"bbox_corners_soma_relative = {bbox_corners_soma_relative}")


        axon_dict = dict(

                        length = axon_length,
                        branch_length_median = axon_branch_length_median,
                        branch_length_mean = axon_branch_length_mean,

                        n_branches = n_branches,
                        n_short_branches = n_short_branches,
                        n_long_branches = n_long_branches,
                        n_medium_branches = n_medium_branches,

                        bbox_volume=bbox_volume,
                        bbox_x_min=bbox_corners[0][0],
                        bbox_y_min=bbox_corners[0][1],
                        bbox_z_min=bbox_corners[0][2],
                        bbox_x_max=bbox_corners[1][0],
                        bbox_y_max=bbox_corners[1][1],
                        bbox_z_max=bbox_corners[1][2],

                        bbox_x_min_soma_relative=bbox_corners_soma_relative[0][0],
                        bbox_y_min_soma_relative=bbox_corners_soma_relative[0][1],
                        bbox_z_min_soma_relative=bbox_corners_soma_relative[0][2],
                        bbox_x_max_soma_relative=bbox_corners_soma_relative[1][0],
                        bbox_y_max_soma_relative=bbox_corners_soma_relative[1][1],
                        bbox_z_max_soma_relative=bbox_corners_soma_relative[1][2],

                        )
    
    if features_to_exclude is not None:
        axon_dict = {k:v for k,v in axon_dict.items() if k not in features_to_exclude}
    if name_prefix is not None:
        axon_dict = {f"{name_prefix}_{k}":v for k,v in axon_dict.items()}
        
    
    
    return axon_dict






def branch_stats_over_limb_branch(
    neuron_obj,
    limb_branch_dict,
    features = ("skeletal_length",
               "width_with_spines",
               "width_no_spines"),
    stats_to_compute = ("mean","median","percentile_70"),
    verbose = False,
    
    ):
    """
    Purpose: to compute some stats over a limb branch

    Things want to find out about dendrites: 

    - widths
    - lengths

    and then summary statistics about it
    - mean/median
    - 70th percentile

    """
    branch_lengths = nru.feature_over_limb_branch_dict(neuron_obj,
                                                     limb_branch_dict,
                                                     feature="skeletal_length")

    branch_widths_with_spine = nru.feature_over_limb_branch_dict(neuron_obj,
                                                     limb_branch_dict,
                                                     feature_from_fuction=nst.width_new,
                                                    feature_from_fuction_kwargs=dict(width_new_name = "median_mesh_center"))
    branch_widths_no_spine = nru.feature_over_limb_branch_dict(neuron_obj,
                                                     limb_branch_dict,
                                                     feature_from_fuction=nst.width_new,
                                                    feature_from_fuction_kwargs=dict(width_new_name = "no_spine_median_mesh_center"))
    branch_data = dict(skeletal_length = branch_lengths,
        width_with_spines = branch_widths_with_spine,
        width_no_spines = branch_widths_no_spine)

    # if verbose:
    #     print(f"branch_widths_with_spine = {branch_widths_with_spine}")
    #     print(f"branch_widths_no_spine = {branch_widths_no_spine}")

    branch_data_stats = dict()
    for k,v in branch_data.items():
        if k not in features:
            continues
        for s in stats_to_compute:
            s_kwargs = dict()
            if "percentile" in s:
                stat_name,percentile = s.split("_")
                s_kwargs = dict(q = int(percentile))
            else:
                stat_name = s

            branch_data_stats[f"{k}_{s}"] = getattr(np,stat_name)(v,**s_kwargs)

    return branch_data_stats
    
    
    
#-------------- 12/9 Developed for work with cell typing -----------------
def soma_distance_branch_set(neuron_obj,
                                     attr_name,
                                     attr_func,
                                    ):
    """
    Purpose: Will set the skeletal distance to soma
    on each branch
    
    Pseudocode: 
    1) iterate through all of the limbs and branches
    2) Find the distnace from soma and store
    """
    for limb_idx in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            s_dist = attr_func(limb_obj,branch_idx)
            setattr(limb_obj[branch_idx],attr_name,s_dist)
            
            
def soma_distance_skeletal_branch_set(neuron_obj,
                                     attr_name = "soma_distance_skeletal",):
    nst.soma_distance_branch_set(neuron_obj,attr_name=attr_name,
                             attr_func = nst.distance_from_soma,
                            )
    
def soma_distance_euclidean_branch_set(neuron_obj,
                                     attr_name = "soma_distance_euclidean",):
    nst.soma_distance_branch_set(neuron_obj,attr_name=attr_name,
                             attr_func = nst.distance_from_soma_euclidean,
                            )
def upstream_endpoint_branch_set(neuron_obj,
                                attr_name = "upstream_endpoint"):
    for limb_idx in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            starting_coordinate = nru.upstream_endpoint(limb_obj,branch_idx)
            setattr(limb_obj[branch_idx],attr_name,starting_coordinate)
            
def centroid_stats_from_neuron_obj(neuron_obj,
                                  voxel_adjustment_vector=None,
                                  include_volume=True):
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector=voxel_to_nm_scaling
        
    soma_x_nm,soma_y_nm,soma_z_nm = nru.soma_centers(neuron_obj,
                                       soma_name="S0",
                                       voxel_adjustment=False,
                                       return_int_form=False)
    soma_x,soma_y,soma_z = nru.soma_centers(neuron_obj,
                                           soma_name="S0",
                                            voxel_adjustment = True,
                                           voxel_adjustment_vector=voxel_adjustment_vector,
                                           return_int_form=True)
    return_dict = dict(
        centroid_x_nm=soma_x_nm,
        centroid_y_nm=soma_y_nm,
        centroid_z_nm=soma_z_nm,
        centroid_x=soma_x,
        centroid_y=soma_y,
        centroid_z=soma_z 
    )
    
    if include_volume:
        return_dict["centroid_volume"] = neuron_obj["S0"].volume
        
    return return_dict
    

def skeleton_stats_from_neuron_obj(neuron_obj,
                                  include_centroids = True,
                                  voxel_adjustment_vector=None,
                                  verbose= False,
                                   limb_branch_dict = None,
                                   neuron_obj_aligned=None,
                                  ):
    """
    Compute all the statistics for a neurons skeleton (should have only one soma)
    """
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector = voxel_to_nm_scaling
    
    sk_dict = nst.stats_dict_over_limb_branch(
        neuron_obj,
        limb_branch_dict=limb_branch_dict,
        stats_to_compute=["skeletal_length","n_branches"])

    sk_dict_2 = nst.features_from_neuron_skeleton_and_soma_center(
        neuron_obj,
        verbose = verbose,
        limb_branch_dict=limb_branch_dict,
        features_to_exclude=("length","n_branches"),
        neuron_obj_aligned=neuron_obj_aligned,
        )
    sk_dict.update(sk_dict_2)
    sk_dict["n_limbs"] = neuron_obj.n_limbs
    
    if include_centroids:
        cent_stats = nst.centroid_stats_from_neuron_obj(neuron_obj,
                                                       voxel_adjustment_vector=voxel_adjustment_vector)
        sk_dict.update(cent_stats)
    
    return sk_dict

def skeleton_stats_compartment(
    neuron_obj,
    compartment,
    include_compartmnet_prefix=True,
    include_centroids = False,
    **kwargs):
    
    limb_branch_dict = getattr(neuron_obj,f"{compartment}_limb_branch_dict")
    return_dict = nst.skeleton_stats_from_neuron_obj(
        neuron_obj,
        limb_branch_dict = limb_branch_dict,
        include_centroids=include_centroids,
        **kwargs
        )
    if include_compartmnet_prefix:
        return_dict = {f"{compartment}_{k}":v for k,v in return_dict.items()}
        
    return return_dict
        
def skeleton_stats_dendrite(
    neuron_obj,
    **kwargs):
    return nst.skeleton_stats_compartment(
            neuron_obj,
            compartment="dendrite",
            **kwargs)
def skeleton_stats_axon(
    neuron_obj,
    **kwargs):
    return nst.skeleton_stats_compartment(
            neuron_obj,
            compartment="axon",
            **kwargs)


def width_near_branch_endpoint(
    limb_obj,
    branch_idx,
    endpoint = None, # if None then will select most upstream endpoint of branch

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
    if endpoint is None:
        endpoint = nru.closest_branch_endpoint_to_limb_starting_coordinate(
            limb_obj=limb_obj,
            branches=[branch_idx],
        )

    if verbose:
        print(f"endpoint = {endpoint}")


    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(limb_obj[branch_idx],
                              common_endpoint=endpoint,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=False,
                             )


    branch_width = np.mean(base_final_widths)
    overall_ais_width = limb_obj[branch_idx].width_new
    if verbose:
        print(f"base_final_widths = {base_final_widths}")
        print(f"overall_branch_width = {overall_ais_width}")
        print(f"branch_width = {branch_width}")
        
    return branch_width

# --------------s / 9 ---------------
def farthest_distance_from_skeleton_to_mesh(
    obj,
    verbose = False,
    plot = False,
    **kwargs
    ):
    """
    Purposee: find the coordinate
    of the skeleton that has the 
    longest closest distance to the mesh
    
    Ex: 
    farthest_distance_from_skeleton_to_mesh(
    branch_obj,
    verbose = True,
    plot = True
    )
    """
    return tu.farthest_coordinate_to_faces(
        obj.mesh,
        obj.skeleton,
        return_distance = True,
        verbose = verbose,
        plot = plot,
    )

def limb_branch_from_stats_df(
    df
    ):
    """
    Purpose: To convert a dataframe to a limb branch dict
    """
    limb_branch_pairings = df[["limb","node"]].to_numpy()

    #gets a dictionary where key is the limb and value is a list of all the branches that were in the filtered dataframe
    limb_to_branch = dict([(k,np.sort(limb_branch_pairings[:,1][np.where(limb_branch_pairings[:,0]==k)[0]]).astype("int")) 
                           for k in np.unique(limb_branch_pairings[:,0])])
    return limb_to_branch

def coordinates_function_list(
    coordinates = None
    ):
    if coordinates is None:
        coordinates = ("mesh_center",
               "endpoint_downstream",
              "endpoint_upstream")
    return  np.concatenate([
        [f"{k}_{x}" for x in ["x","y","z"]] for k in coordinates
    ])

def coordinates_stats_df(
    neuron_obj,
    coordinates = None,
    limb_branch_dict_restriction = None,
    verbose = False
    ):
    """
    Purpose: To create a dataframe of centers 
    for a limb branch

    """
    functions_list=nst.coordinates_function_list(coordinates)
    
    if (np.any(["endpoint" in k for k in functions_list])):
        bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)


    coordinates_df = nst.stats_df(
        neuron_obj,
        functions_list=functions_list,
        limb_branch_dict_restriction=limb_branch_dict_restriction
    )

    return coordinates_df

def stats_df(
    neuron_obj,
    functions_list=None,
    query = None,
    limb_branch_dict_restriction = None,
    function_kwargs=None,
    include_coordinates = False,
    coordinates = None,
    check_nans=False,
    ):
    """
    Purpose: To return the stats on neuron branches 
    that is used by the neuron searching to filter down
    
    Ex: 
    from neurd import neuron_statistics as nst

    limb_obj = neuron_obj[6]

    s_df = nst.stats_df(
        neuron_obj,
        functions_list = [ns.width_new,
        ns.skeletal_length,
        ns.n_synapses_post_downstream],
        limb_branch_dict_restriction=dict(L6=limb_obj.get_branch_names())
            )
    s_df
    """
    
    if functions_list is None:
        functions_list= []
        
    if query is not None:
        functions_list += ns.functions_list_from_query(query)
    
    if include_coordinates:
        neuron_obj = bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
        functions_list += list(nst.coordinates_function_list(coordinates))
        
    return ns.query_neuron(neuron_obj,
                functions_list=functions_list,
                function_kwargs = function_kwargs,       
                return_dataframe_before_filtering=True,
                           limb_branch_dict_restriction=limb_branch_dict_restriction,
               query="",
                check_nans=check_nans)
    
    
        


def neuron_stats(
    neuron_obj,
    stats_to_ignore=None,
    include_skeletal_stats = False,
    include_centroids= False,
    voxel_adjustment_vector = None,
    cell_type_mode = False,
    **kwargs):
    
    """
    Purpose: Will compute a wide range of statistics 
    on a neurons object
    """
    
    if cell_type_mode:
        stats_to_ignore = [
            "n_not_processed_soma_containing_meshes",
            "n_error_limbs",
            "n_same_soma_multi_touching_limbs",
            "n_multi_soma_touching_limbs",
            "n_somas",
            "spine_density"
            ]
        include_skeletal_stats = False
        include_centroids= True


    stats_dict = dict(
                    n_vertices = neuron_obj.n_vertices,
                    n_faces = neuron_obj.n_faces,

                    axon_length = neuron_obj.axon_length,
                    axon_area = neuron_obj.axon_area,

                    max_soma_volume = neuron_obj.max_soma_volume,
                    max_soma_n_faces = neuron_obj.max_soma_n_faces,
                    max_soma_area = neuron_obj.max_soma_area,


                    n_not_processed_soma_containing_meshes = len(neuron_obj.not_processed_soma_containing_meshes),
                    n_error_limbs=neuron_obj.n_error_limbs,
                    n_same_soma_multi_touching_limbs=len(neuron_obj.same_soma_multi_touching_limbs),
                    n_multi_soma_touching_limbs = len(neuron_obj.multi_soma_touching_limbs),
                    n_somas=neuron_obj.n_somas,
                    n_limbs=neuron_obj.n_limbs,
                    n_branches=neuron_obj.n_branches,
                    max_limb_n_branches=neuron_obj.max_limb_n_branches,

                    skeletal_length=neuron_obj.skeletal_length,
                    max_limb_skeletal_length=neuron_obj.max_limb_skeletal_length,
                    median_branch_length=neuron_obj.median_branch_length,

                    width_median=neuron_obj.width_median, #median width from mesh center without spines removed
                    width_no_spine_median=neuron_obj.width_no_spine_median, #median width from mesh center with spines removed
                    width_90_perc=neuron_obj.width_90_perc, # 90th percentile for width without spines removed
                    width_no_spine_90_perc=neuron_obj.width_no_spine_90_perc,  # 90th percentile for width with spines removed

                    n_spines=neuron_obj.n_spines,
                    n_boutons=neuron_obj.n_boutons,

                    spine_density=neuron_obj.spine_density, # n_spines/ skeletal_length
                    spines_per_branch=neuron_obj.spines_per_branch,

                    skeletal_length_eligible=neuron_obj.skeletal_length_eligible, # the skeletal length for all branches searched for spines
                    n_spine_eligible_branches=neuron_obj.n_spine_eligible_branches,
                    spine_density_eligible = neuron_obj.spine_density_eligible,
                    spines_per_branch_eligible = neuron_obj.spines_per_branch_eligible,

                    total_spine_volume=neuron_obj.total_spine_volume, # the sum of all spine volume
                    spine_volume_median = neuron_obj.spine_volume_median,
                    spine_volume_density=neuron_obj.spine_volume_density, #total_spine_volume/skeletal_length
                    spine_volume_density_eligible=neuron_obj.spine_volume_density_eligible, #total_spine_volume/skeletal_length_eligible
                    spine_volume_per_branch_eligible=neuron_obj.spine_volume_per_branch_eligible, #total_spine_volume/n_spine_eligible_branche



    )

    if stats_to_ignore is not None:
        for s in stats_to_ignore:
            del stats_dict[s]

    if include_skeletal_stats:
        sk_dict = nst.features_from_neuron_skeleton_and_soma_center(neuron_obj,
                                              verbose = False,
                                              features_to_exclude=("length","n_branches"),
                                             )
        stats_dict.update(sk_dict)


    if include_centroids:
        cent_stats = nst.centroid_stats_from_neuron_obj(neuron_obj,
                                                       voxel_adjustment_vector=voxel_adjustment_vector)
        stats_dict.update(cent_stats)


    return stats_dict

def branch_stats_dict_from_df(df,limb_name,branch_idx):
    """
    Ex: limb_df = nst.stats_df(neuron_obj,
        functions_list=[eval(f"lu.{k}_{ns.limb_function_append_name}") 
                        for k in ctcu.branch_attrs_limb_based_for_G])

    limb_name = "L0"
    branch_name = 4
    nst.branch_stats_dict_from_df(limb_df,limb_name,branch_name)
    """
    limb_name = nru.get_limb_string_name(limb_name)
    return pu.df_to_dicts(pu.delete_columns(df.query(f"(limb=='{limb_name}') and (node == {branch_idx})"),["limb","node"]))[0]

def euclidean_distance_from_soma_limb_branch(
    neuron_obj,
    less_than = False,
    distance_threshold = 10_000,
    endpoint_type = "downstream",
    verbose = False,
    plot = False,
    ):
    """
    Purpose: Find limb branch dict within or
    farther than a certain euclidean distance
    from all the soma pieces

    Pseudocode: 
    1) get the upstream endpoints of all
    """
   
    

    bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    soma_kd = tu.mesh_to_kdtree(neuron_obj["S0"].mesh)



    lb_dict = dict()
    for limb_idx in neuron_obj.get_limb_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            branch_obj = limb_obj[branch_idx]
            dist,__ = soma_kd.query(getattr(branch_obj,f"endpoint_{endpoint_type}").reshape(-1,3))
            if less_than:
                add_flag = dist[0] < distance_threshold
            else:
                add_flag = dist[0] >= distance_threshold

            if add_flag:
                if limb_idx not in lb_dict:
                    lb_dict[limb_idx] = []
                lb_dict[limb_idx].append(branch_idx)

                if verbose:
                    print(f"Adding {limb_idx}, {branch_idx} because dist {dist[0]}")


    lb_dict = {k:np.array(v) for k,v in lb_dict.items()}
    
    if plot:
        nviz.plot_limb_branch_dict(neuron_obj,lb_dict)
    
    return lb_dict

def euclidean_distance_close_to_soma_limb_branch(
    neuron_obj,
    distance_threshold = 10_000,
    verbose = False,
    plot = False,
    ):
    
    return nst.euclidean_distance_from_soma_limb_branch(
    neuron_obj,
    less_than = True,
    distance_threshold = distance_threshold,
    verbose = verbose,
    plot = plot,
    )

def euclidean_distance_farther_than_soma_limb_branch(
    neuron_obj,
    distance_threshold = 10_000,
    verbose = False,
    plot = False,
    ):
    
    return nst.euclidean_distance_from_soma_limb_branch(
    neuron_obj,
    less_than = False,
    distance_threshold = distance_threshold,
    verbose = verbose,
    plot = plot,
    )

# ----------------- Parameters ------------------------


global_parameters_dict_default = dict(
)

attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling
)    

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01 = {}


attributes_dict_h01 = dict(
    voxel_to_nm_scaling = hvu.voxel_to_nm_scaling
)

# data_type = "default"
# algorithms = None


# modules_to_set = [nst]

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
from . import axon_utils as au
from . import branch_utils as bu
from . import concept_network_utils as cnu
from . import error_detection as ed
from . import h01_volume_utils as hvu 
from . import microns_volume_utils as mcu
from . import microns_volume_utils as mvu
from . import neuron_searching as ns
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import synapse_utils as syu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu

from . import neuron_statistics as nst