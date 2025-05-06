
import networkx as nx
from datasci_tools import numpy_dep as np

non_branching_upstream = False

def distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph,):
    """
    Purpose: To determine the distance
    between two nodes along a path
    
    """
    branch_path = nru.branch_path_to_node(limb_obj,
                       start_idx=start_idx,
                       destination_idx=destination_idx,
                        include_branch_idx=False,
                        include_last_branch_idx=False,
                        reverse_di_graph=reverse_di_graph)
    
    if branch_path is None:
        return np.inf
    
    return nst.skeletal_length_along_path(limb_obj,branch_path)

def distance_between_nodes_di_upstream(limb_obj,
                      start_idx,
                      destination_idx):
    """
    Purpose: To determine the upstream distance
    from branch_idx to start_idx
    
    """
    return cnu.distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph=True)

def distance_between_nodes_di_downstream(limb_obj,
                      start_idx,
                      destination_idx):
    """
    Purpose: To determine the upstream distance
    from branch_idx to start_idx
    
    """
    return cnu.distance_between_nodes_di(limb_obj,
                      start_idx,
                      destination_idx,
                      reverse_di_graph=False)

def branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func,
                            distance_threshold,
                            include_branch_idx = False):
    """
    Purpose: To find all branches with a certain downstream distance
    """
    branch_names = np.array(limb_obj.get_branch_names())
    branch_dist = np.array([dist_func(limb_obj,branch_idx,k) for k in branch_names])
    #print(f"branch_names and dist = \n {np.vstack([branch_names,branch_dist]).T.astype("int")}")
    
    branches_within_dist = branch_names[(branch_dist<=distance_threshold) & 
                                       (branch_dist != np.inf)]
    
    if include_branch_idx:
        return branches_within_dist
    else:
        return branches_within_dist[branches_within_dist != branch_idx]


def branches_within_distance_upstream(limb_obj,
                             branch_idx,
                                     distance_threshold,
                                     include_branch_idx = False):
    return branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func=cnu.distance_between_nodes_di_upstream,
                                   distance_threshold=distance_threshold,
                                   include_branch_idx = include_branch_idx)

def branches_within_distance_downstream(limb_obj,
                             branch_idx,
                                     distance_threshold,
                                       include_branch_idx = False):
    """
    Ex: 
    nst.branches_within_distance_downstream(limb_obj,223,
                                       2000)
    """
    return branches_within_distance(limb_obj,
                             branch_idx,
                            dist_func=cnu.distance_between_nodes_di_downstream,
                                   distance_threshold=distance_threshold,
                                   include_branch_idx = include_branch_idx)

#find all end nodes within a downstream threshold
def endnode_branches_of_branches_within_distance_downtream(
    limb_obj,
    branch_idx,
    skip_distance=2000,
    return_skipped_branches=False,
    **kwargs
    ):
    """
    Purpose: To get the branches that are a certain distance
    away from a branch but to only return the furthermost branches
    
    Ex: 
    limb_obj = neuron_obj[0]
    branch_idx = 223

    cnu.endnode_branches_of_branches_within_distance_downtream(limb_obj,
                                                          branch_idx,
                                                          0)
    
    """
    downstream_branches = cnu.branches_within_distance_downstream(
        limb_obj,
        branch_idx,
        skip_distance
    )
    G = limb_obj.concept_network_directional
    G_sub = G.subgraph(downstream_branches)

    end_nodes = xu.end_nodes_of_digraph(G_sub)
    
    if return_skipped_branches:
        skipped_branches = np.setdiff1d(downstream_branches,end_nodes)
        return end_nodes,skipped_branches
    else:
        return end_nodes


def branches_with_parent_branching(limb_obj):
    return xu.nodes_with_parent_branching(limb_obj.concept_network_directional)

def branches_with_parent_non_branching(limb_obj):
    """
    Purpose: To see if a branch had a parent node
    that branched off into multiple branches
    """
    return xu.nodes_with_parent_non_branching(limb_obj.concept_network_directional)




def subgraph_around_branch(limb_obj,
                           branch_idx,
                           upstream_distance=0,
                           downstream_distance=0,
                           distance = None,
                           distance_attribute = "skeletal_length",
                          include_branch_in_upstream_dist=True,
                          include_branch_in_downstream_dist=True,
                          only_non_branching_downstream=True,
                           only_non_branching_upstream = non_branching_upstream, 
                           include_branch_idx = True,
                          return_branch_idxs=True,
                           plot_subgraph = False,
                           nodes_to_exclude=None,
                           nodes_to_include = None,
                           verbose = False
                          ):
    """
    Purpose: To return a subgraph around a certain 
    branch to find all the nodes upstream and/or downstream
    
    Pseudocode: 
    1) Find all the branches upstream of branch 
    (subtract the skeletal length of branch if included in upstream dist )
    2) Find all the branches downstrem of branch 
    (subtract the skeletal length of branch if included in upstream dist )
    3) Find the upstream and downstream nodes a certain distance away
    
    Ex: 
    cnu.subgraph_around_branch(limb_obj,
                           branch_idx=97,
                           upstream_distance=1000000,
                           downstream_distance=1000000,
                           distance = None,
                           distance_attribute = "skeletal_length",
                          include_branch_in_upstream_dist=True,
                          include_branch_in_downstream_dist=True,
                          only_non_branching_downstream=True,
                           include_branch_idx = False,
                          return_branch_idxs=True,
                           plot_subgraph = True,
                           verbose = False
                          )
    """
    
    if distance is not None:
        upstream_distance = downstream_distance = distance
    
    branch_idx_dist = getattr(limb_obj[branch_idx],distance_attribute)
    
    if include_branch_in_upstream_dist:
        upstream_distance = upstream_distance - branch_idx_dist
            
    if include_branch_in_downstream_dist:
        downstream_distance = downstream_distance - branch_idx_dist
    if verbose:
        print(f"branch_idx_dist = {branch_idx_dist}")
        print(f"upstream_distance = {upstream_distance}")
        print(f"downstream_distance = {downstream_distance}")
    
    upstream_branches = cnu.branches_within_distance_upstream(limb_obj,
                                                          branch_idx,
                                                          upstream_distance,
                                                          include_branch_idx=include_branch_idx)
    downstream_branches = cnu.branches_within_distance_downstream(limb_obj,
                                                          branch_idx,
                                                          downstream_distance,
                                                          include_branch_idx=include_branch_idx)
    
    if verbose:
        print(f"upstream_branches= {upstream_branches}")
        print(f"downstream_branches = {downstream_branches}")
        
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
        
    #print(f"nodes_to_exclude = {nodes_to_exclude}")
    
    if nodes_to_exclude is not None:
        downstream_branches = np.setdiff1d(downstream_branches,nodes_to_exclude)
        upstream_branches = np.setdiff1d(upstream_branches,nodes_to_exclude)
        if verbose:
            print(f"Excluding Nodes: {nodes_to_exclude}")
            print(f"After exclusion:\n downstream_branches = {downstream_branches}\nupstream_branches = {upstream_branches}")
    
    if nodes_to_include is not None:
        downstream_branches = np.intersect1d(downstream_branches,nodes_to_include)
        upstream_branches = np.intersect1d(upstream_branches,nodes_to_include)
        if verbose:
            print(f"Limiting to only Include Nodes: {nodes_to_include}")
            print(f"After inclusion:\n downstream_branches = {downstream_branches}\nupstream_branches = {upstream_branches}")
    
        
    if only_non_branching_upstream:
        upstream_branches = np.intersect1d(upstream_branches,cnu.upstream_nodes_without_branching(limb_obj,branch_idx,
                                                                                                  nodes_to_exclude=nodes_to_exclude))
        if verbose:
            print(f"\nAfter only_non_branching_upstream:\nupstream_branches = {upstream_branches} ")
    
    if only_non_branching_downstream:
#         downstream_branches = np.setdiff1d(downstream_branches,cnu.branches_with_parent_branching(limb_obj))
        downstream_branches = np.intersect1d(downstream_branches,cnu.downstream_nodes_without_branching(limb_obj,branch_idx,
                                                                                                       nodes_to_exclude=nodes_to_exclude))
        if verbose:
            print(f"\nAfter only_non_branching_downstream:\ndownstream_branches = {downstream_branches} ")
        
    total_branches = np.hstack([upstream_branches,downstream_branches])
    if include_branch_idx:
        if branch_idx not in total_branches:
            total_branches = np.hstack([total_branches,[branch_idx]])
        
    if verbose:
        print(f"total_branches = {total_branches}")
    
    if return_branch_idxs:
        return total_branches
    
    sub_G = limb_obj.concept_network_directional.subgraph(total_branches)
    
    if plot_subgraph:
        nx.draw(sub_G,with_labels=True)
        
    return sub_G

def downstream_nodes_without_branching(limb_obj,
                                      branch_idx,
                                      nodes_to_exclude = None):
    """
    Purpose: To return all nodes that are 
    downstream of a branch but not after a branching point
    """
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
    
    downstream_nodes = []
    curr_node = branch_idx
    for i,b in enumerate(limb_obj.get_branch_names()):
        d_nodes = nru.downstream_nodes(limb_obj,curr_node)
        
        if nodes_to_exclude is not None:
            d_nodes = np.setdiff1d(d_nodes,nodes_to_exclude)
        
        if len(d_nodes) == 0 or len(d_nodes) > 1:
            return downstream_nodes
        elif len(d_nodes) == 1:
            curr_node = d_nodes[0]
            downstream_nodes.append(curr_node)
            
    return downstream_nodes

def upstream_nodes_without_branching(limb_obj,
                                      branch_idx,
                                    nodes_to_exclude = None):
    """
    Purpose: To return all nodes that are 
    downstream of a branch but not after a branching point
    """
    if nodes_to_exclude is None:
        nodes_to_exclude = limb_obj.nodes_to_exclude
    
    upstream_nodes = []
    curr_node = branch_idx
    for i,b in enumerate(limb_obj.get_branch_names()):
        u_node = nru.upstream_node(limb_obj,curr_node)
        if u_node is None:
            return upstream_nodes
        
        d_nodes = nru.downstream_nodes(limb_obj,u_node)
        
        if nodes_to_exclude is not None:
            d_nodes = np.setdiff1d(d_nodes,nodes_to_exclude)
        
        if len(d_nodes) > 1:
            return upstream_nodes
        elif len(d_nodes) == 1:
            curr_node = u_node
            upstream_nodes.append(curr_node)
            
    return upstream_nodes


# ------ 6/25: Helps find attributes that are downstream or upstream -------

'''def downstream_attribute(limb_obj,
                                 branch_idx,
                                 attribute_name,
                                 concat_func = np.concatenate,
                         downstream_distance = np.inf,
                         include_branch_in_downstream_dist = True,
                         only_non_branching_downstream = True,
                         include_branch_idx = True,
                         verbose = False,
                         nodes_to_exclude = None,
                         return_nodes = False,
                                 ):

    """
    Purpose: To retrieve and concatenate
    the attributes of a branch and
    all of the branches downsream
    of the branch until there is a branching point
    or within a certain distance

    Pseudocode: 
    1) Get all of the branches that are downstream
    (either up to branch point or within certain distance)
    2) Get the attributes of the branch and all those downstream
    3) concatenate the attributes using the prescribed function

    """

    # 1) Get all of the branches that are downstream
    # (either up to branch point or within certain distance)
    all_downstream_nodes = cnu.subgraph_around_branch(limb_obj,
                                                      branch_idx = branch_idx,
                                                      include_branch_idx=include_branch_idx,
                                                      include_branch_in_downstream_dist = include_branch_in_downstream_dist,
                                    downstream_distance = downstream_distance,
                                                      upstream_distance = -1,
                                    only_non_branching_downstream=only_non_branching_downstream,
                                                      verbose = verbose
                                    )

    if verbose:
        print(f"With downstream_distance= {downstream_distance}, only_non_branching_downstream = {only_non_branching_downstream}")
        print(f"all_downstream_nodes = {all_downstream_nodes}")
        
    if nodes_to_exclude is not None:
        all_downstream_nodes = np.setdiff1d(all_downstream_nodes,nodes_to_exclude)
        if verbose:
            print(f"Excluding Nodes: {nodes_to_exclude}")
            print(f"After exclusion: all_downstream_nodes = {all_downstream_nodes}")

    #2) Get the attributes of the branch and all those downstream
    down_attr = [getattr(limb_obj[k],attribute_name) for k in all_downstream_nodes]

    #3) concatenate the attributes using the prescribed function
    if len(down_attr) > 0:
        down_attr_concat = concat_func(down_attr)
    else:
        down_attr_concat = down_attr 
        

    if return_nodes:
        return down_attr_concat,all_downstream_nodes
    else:
        return down_attr_concat
'''
def other_direction(direction):
    if direction=="upstream":
        return "downstream"
    elif direction == "downstream":
        return "upstream"
    else:
        raise Exception(f"unknown direction: {direction}")

def nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction,
                             distance = np.inf,include_branch_in_dist = True,
                         only_non_branching = True,
                         include_branch_idx = True,
                         verbose = False,
                         nodes_to_exclude = None,
                             nodes_to_include=None):
    """
    Will return nodes that are upstream or downstream by a certain dist
    """
    arg_dict = {f"include_branch_in_{direction}_dist":include_branch_in_dist,
                f"only_non_branching_{direction}":only_non_branching,
                f"{direction}_distance":distance,
                f"{other_direction(direction)}_distance":-1}
                

    # 1) Get all of the branches that are downstream
    # (either up to branch point or within certain distance)
    all_downstream_nodes = cnu.subgraph_around_branch(limb_obj,
                                                      branch_idx = branch_idx,
                                                      include_branch_idx=include_branch_idx,
                                                      verbose = verbose,
                                                      nodes_to_exclude=nodes_to_exclude,
                                                      nodes_to_include=nodes_to_include,
                                                      **arg_dict
                                    )

    if verbose:
        print(f"With direction = {direction}, distance= {distance}, only_non_branching = {only_non_branching}, include_branch_in_dist = {include_branch_in_dist}")
        print(f"all_{direction}_nodes = {all_downstream_nodes}")
        
    return all_downstream_nodes

def nodes_downstream(limb_obj,
                             branch_idx,
                             distance = np.inf,
                          include_branch_in_dist = False,
                         only_non_branching = False,
                         include_branch_idx = False,
                         verbose = False,
                         nodes_to_exclude = None,
                    nodes_to_include = None):
    return nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction="downstream",
                             distance = distance,
                        include_branch_in_dist = include_branch_in_dist,
                         only_non_branching = only_non_branching,
                         include_branch_idx = include_branch_idx,
                         verbose = verbose,
                         nodes_to_exclude = nodes_to_exclude,
                                    nodes_to_include=nodes_to_include,)

def nodes_upstream(limb_obj,
                             branch_idx,
                             distance = np.inf,
                          include_branch_in_dist = False,
                         only_non_branching = False,
                         include_branch_idx = False,
                         verbose = False,
                         nodes_to_exclude = None,
                          nodes_to_include = None):
    return nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction="upstream",
                             distance = distance,
                        include_branch_in_dist = include_branch_in_dist,
                         only_non_branching = only_non_branching,
                         include_branch_idx = include_branch_idx,
                         verbose = verbose,
                         nodes_to_exclude = nodes_to_exclude,
                                    nodes_to_include=nodes_to_include,)
                    
                     
        
def attribute_upstream_downstream(limb_obj,
                                 branch_idx,
                                  direction,
                                  attribute_name=None,
                                  attribute_func = None,
                                 concat_func = np.concatenate,
                                 distance = np.inf,
                         include_branch_in_dist = True,
                         only_non_branching = True,
                         include_branch_idx = True,
                         verbose = False,
                         nodes_to_exclude = None,
                         return_nodes = False,
                                 ):

    """
    Purpose: To retrieve and concatenate
    the attributes of a branch and
    all of the branches downsream
    of the branch until there is a branching point
    or within a certain distance

    Pseudocode: 
    1) Get all of the branches that are downstream
    (either up to branch point or within certain distance)
    2) Get the attributes of the branch and all those downstream
    3) concatenate the attributes using the prescribed function

    """
    '''    
    arg_dict = {f"include_branch_in_{direction}_dist":include_branch_in_dist,
                f"only_non_branching_{direction}":only_non_branching,
                f"{direction}_distance":distance,
                f"{other_direction(direction)}_distance":-1}
                

    # 1) Get all of the branches that are downstream
    # (either up to branch point or within certain distance)
    all_downstream_nodes = cnu.subgraph_around_branch(limb_obj,
                                                      branch_idx = branch_idx,
                                                      include_branch_idx=include_branch_idx,
                                                      verbose = verbose,
                                                      nodes_to_exclude=nodes_to_exclude,
                                                      **arg_dict
                                    )

    if verbose:
        print(f"With direction = {direction}, distance= {distance}, only_non_branching = {only_non_branching}, include_branch_in_dist = {include_branch_in_dist}")
        print(f"all_{direction}_nodes = {all_downstream_nodes}")'''

    all_downstream_nodes = cnu.nodes_upstream_downstream(limb_obj,
                             branch_idx,
                             direction,
                             distance = distance,
                            include_branch_in_dist = include_branch_in_dist,
                         only_non_branching = only_non_branching,
                         include_branch_idx = include_branch_idx,
                         verbose = verbose,
                         nodes_to_exclude = nodes_to_exclude,)
        

    #2) Get the attributes of the branch and all those downstream
    if attribute_func is None:
        down_attr = [getattr(limb_obj[k],attribute_name) for k in all_downstream_nodes]
    else:
        down_attr = [attribute_func(limb_obj[k]) for k in all_downstream_nodes]

    #3) concatenate the attributes using the prescribed function
    if len(down_attr) > 0 and concat_func is not None:
        down_attr_concat = concat_func(down_attr)
    else:
        down_attr_concat = down_attr 
        

    if return_nodes:
        return down_attr_concat,all_downstream_nodes
    else:
        return down_attr_concat
'''
def upstream_attribute(limb_obj,
                                 branch_idx,
                                 attribute_name,
                                 concat_func = np.concatenate,
                         upstream_distance = np.inf,
                         include_branch_in_upstream_dist = True,
                         only_non_branching_upstream = non_branching_upstream,
                         include_branch_idx = True,
                         verbose = False,
                       return_nodes=False,
                       nodes_to_exclude = None,
                                 **kwargs):

    """
    Purpose: To retrieve and concatenate
    the attributes of a branch and
    all of the branches downsream
    of the branch until there is a branching point
    or within a certain distance

    Pseudocode: 
    1) Get all of the branches that are downstream
    (either up to branch point or within certain distance)
    2) Get the attributes of the branch and all those downstream
    3) concatenate the attributes using the prescribed function

    """

    # 1) Get all of the branches that are downstream
    # (either up to branch point or within certain distance)
    all_upstream_nodes = cnu.subgraph_around_branch(limb_obj,
                                                      branch_idx = branch_idx,
                                                      include_branch_idx=include_branch_idx,
                                                      include_branch_in_upstream_dist = include_branch_in_upstream_dist,
                                                    only_non_branching_upstream = only_non_branching_upstream
                                                      upstream_distance = upstream_distance,
                                                      downstream_distance = -1,
                                                      verbose = verbose
                                    )

    if verbose:
        print(f"With upstream_distance= {upstream_distance}, only_non_branching_upstream = {only_non_branching_upstream}, include_branch_idx = {include_branch_idx}")
        print(f"all_upstream_nodes = {all_upstream_nodes}")
        
    if nodes_to_exclude is not None:
        all_upstream_nodes = np.setdiff1d(all_upstream_nodes,nodes_to_exclude)
        if verbose:
            print(f"Excluding Nodes: {nodes_to_exclude}")
            print(f"After exclusion: all_upstream_nodes = {all_upstream_nodes}")

    #2) Get the attributes of the branch and all those downstream
    up_attr = [getattr(limb_obj[k],attribute_name) for k in all_upstream_nodes]

    #3) concatenate the attributes using the prescribed function
    up_attr_concat = concat_func(up_attr)

    if return_nodes:
        return up_attr_concat,all_upstream_nodes
    else:
        return up_attr_concat
'''

def downstream_nodes_mesh_connected(limb_obj,branch_idx,
                                   n_points_of_contact = None,
                                    downstream_branches=None,
                                   verbose = False):
    """
    Purpose: will determine if at least N number of points of
    contact between the upstream and downstream meshes
    
    Ex: 
    nst.downstream_nodes_mesh_connected(limb_obj,147,
                               verbose=True)
    """
    upstream_node = branch_idx
    if downstream_branches is None:
        downstream_nodes = cnu.downstream_nodes(limb_obj,upstream_node)
    else:
        downstream_nodes = downstream_branches
    
    if n_points_of_contact is None:
        n_points_of_contact = len(downstream_nodes)
    conn_array = tu.mesh_list_connectivity(meshes=[limb_obj[k].mesh for k in [upstream_node] + list(downstream_nodes) ],
                             main_mesh = limb_obj.mesh)
    #intersect_array = nu.intersect2d(conn_array,np.array([[0,1],[0,2]]))
    if verbose:
        print(f"conn_array = {conn_array}")
        print(f"n_points_of_contact = {n_points_of_contact}")
    
    if len(conn_array) >= n_points_of_contact:
        return True
    else:
        return False

def skeleton_upstream_downstream(limb_obj,
                       branch_idx,
                                 direction,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_idx = True,
                        include_branch_in_dist=True,
                        plot_skeleton = False,
                       verbose = False,
                        **kwargs
                       ):
    """
    Purpose: To get the downstream skeleton of a branch
    
    Ex: 
    skel = downstream_skeleton(limb_obj,
                    96,
                           only_non_branching_downstream = False,
                           downstream_distance = 30000
                   )

    """
    
    skel = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    attribute_name = "skeleton",
    direction = direction,
    concat_func = sk.stack_skeletons,
    include_branch_idx=include_branch_idx,
    include_branch_in_dist = include_branch_in_dist,
    distance = distance,
    only_non_branching = only_non_branching,
    verbose = verbose,
     **kwargs)

    if plot_skeleton:
        nviz.plot_objects(skeletons=[skel])
        
    return skel

def skeleton_downstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_idx = True,
                        include_branch_in_dist = True,
                        plot_skeleton = False,
                       verbose = False,
                        **kwargs
                       ):
    return skeleton_upstream_downstream(limb_obj,
                       branch_idx,
                        direction="downstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_idx = include_branch_idx,
                        include_branch_in_dist = include_branch_in_dist,
                        plot_skeleton = plot_skeleton,
                       verbose = verbose,
                        **kwargs
                       )

def skeleton_upstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_idx = True,
                        plot_skeleton = False,
                       verbose = False,
                        **kwargs
                       ):
    return skeleton_upstream_downstream(limb_obj,
                       branch_idx,
                        direction="upstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_idx = include_branch_idx,
                        plot_skeleton = plot_skeleton,
                       verbose = verbose,
                        **kwargs
                       )


def synapses_upstream_downstream(limb_obj,
                       branch_idx,
                        direction,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                       verbose = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    """
    Purpose: To get the downstream synapses at a branch
    
    Ex: 
    syns = downstream_synapses(limb_obj,16,downstream_distance = 0, include_branch_in_downstream_dist = False,
                    only_non_branching_downstream=False,
                   plot_synapses=True)
                   
    E

    """
    syns,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = synapse_type,
    concat_func = np.concatenate,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    verbose = verbose,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)
    
    if verbose:
        print(f"# of syns = {len(syns)}")

    if plot_synapses:
        downstream_nodes = [k for k in nodes if k != branch_idx]
        branch_idx_color = "red"
        d_color = "blue"
        scatter_color = "yellow"
        scatter_size = 1
        print(f"branch_idx ({branch_idx}): {branch_idx_color}\n {direction} nodes ({downstream_nodes}): {d_color}")
        
        nviz.plot_objects(limb_obj[branch_idx].mesh,
                         main_mesh_color=branch_idx_color,
                         meshes = [limb_obj[k].mesh for k in downstream_nodes],
                         meshes_colors=d_color,
                         scatters=[[s.coordinate for s in syns]],
                         scatter_size=scatter_size)
    if return_nodes:
        return syns,nodes
    else:
        return syns
    
def synapses_downstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                       verbose = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    return synapses_upstream_downstream(limb_obj,
                       branch_idx,
                         direction="downstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_in_dist = include_branch_in_dist,
                        include_branch_idx = include_branch_idx,
                        plot_synapses = plot_synapses,
                       verbose = verbose,
                        synapse_type=synapse_type,
                        return_nodes = return_nodes,
                        nodes_to_exclude = nodes_to_exclude,
                        **kwargs
                       )

def synapses_upstream(limb_obj,
                       branch_idx,
                       distance = np.inf,
                       only_non_branching=True,
                        include_branch_in_dist = True,
                        include_branch_idx = True,
                        plot_synapses = False,
                       verbose = False,
                        synapse_type="synapses",
                        return_nodes = False,
                        nodes_to_exclude = None,
                        **kwargs
                       ):
    return synapses_upstream_downstream(limb_obj,
                       branch_idx,
                         direction="upstream",
                       distance = distance,
                       only_non_branching=only_non_branching,
                        include_branch_in_dist = include_branch_in_dist,
                        include_branch_idx = include_branch_idx,
                        plot_synapses = plot_synapses,
                       verbose = verbose,
                        synapse_type=synapse_type,
                        return_nodes = return_nodes,
                        nodes_to_exclude = nodes_to_exclude,
                        **kwargs
                       )
    

def weighted_attribute_upstream_downstream(limb_obj,
                                          branch_idx,
                                           direction,
                                          attribute_name,
                                           attribute_func = None,
                                           verbose = False,
                                           filter_away_zero_sk_lengths=True,
                                          **kwargs):
    sk_lengths,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = "skeletal_length",
    concat_func = None,
     return_nodes=True,
    verbose = verbose,
     **kwargs)

    attr_values,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = attribute_name,
    attribute_func=attribute_func,
    concat_func = None,
    return_nodes=True,           
     **kwargs)

    sk_lengths = np.array(sk_lengths)
    attr_values = np.array(attr_values)

    if verbose:
        print(f"sk_lengths = {sk_lengths}")
        print(f"{attribute_name} (aka attribute value) = {attr_values}")

    # if filter_away_zero_widths:
    if filter_away_zero_sk_lengths:
        keep_mask = sk_lengths > 0

        attr_values = attr_values[keep_mask]
        sk_lengths = sk_lengths[keep_mask]

        if verbose:
            print(f"filter_away_zero_sk_lengths Set:")
            print(f"sk_lengths = {sk_lengths}")
            print(f"{attribute_name} (aka attribute value) = {attr_values}")

    if len(attr_values) != len(sk_lengths):
        raise Exception("")

    if len(attr_values) > 0:
        return nu.weighted_average(attr_values,sk_lengths)
    else:
        return 0
    
def width_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
    **kwargs):
    
    if width_func is None:
        width_func = au.axon_width
    
    return cnu.weighted_attribute_upstream_downstream(limb_obj,
                                          branch_idx,
                                          direction=direction,
                                          attribute_name=width_attribute,
                                        attribute_func = width_func,
                                           verbose = verbose,
                                        include_branch_idx=include_branch_idx,
                                        distance = distance,
                                        only_non_branching = only_non_branching,
                                        include_branch_in_dist = include_branch_in_dist,
                                        nodes_to_exclude=nodes_to_exclude,
                                          **kwargs)

'''
def width_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = au.axon_width,
    width_attribute = None,
    return_nodes = False,
    nodes_to_exclude = None,
    filter_away_zero_sk_lengths = True,
                              **kwargs):
    """
    Purpose: To find the up and downstream width

    Pseudocode: 
    1) Get all up/down sk lengths
    2) Get all up/down widths
    3) Filter away non-zeros widths if argument set
    4) If arrays are non-empty, computed the weighted average

    """

    sk_lengths,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = "skeletal_length",
    concat_func = None,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    verbose = verbose,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)

    widths,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = width_attribute,
    attribute_func = width_func,
    concat_func = None,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    verbose = verbose,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)

    sk_lengths = np.array(sk_lengths)
    widths = np.array(widths)

    if verbose:
        print(f"sk_lengths = {sk_lengths}")
        print(f"widths = {widths}")

    # if filter_away_zero_widths:
    if filter_away_zero_sk_lengths:
        keep_mask = sk_lengths > 0

        widths = widths[keep_mask]
        sk_lengths = sk_lengths[keep_mask]

        if verbose:
            print(f"filter_away_zero_sk_lengths Set:")
            print(f"sk_lengths = {sk_lengths}")
            print(f"widths = {widths}")

    if len(widths) != len(sk_lengths):
        raise Exception("")

    if len(widths) > 0:
        return nu.weighted_average(widths,sk_lengths)
    else:
        return 0
'''
    
def width_upstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
                              **kwargs):
    """
    cnu.width_downstream(limb_obj,
    branch_idx = 65,
    distance = np.inf,
    only_non_branching=False,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = au.axon_width,
    width_attribute = None,
    return_nodes = False,
    nodes_to_exclude = None,)
    """
    
    if width_func is None:
        width_func = au.axon_width
    
    return width_upstream_downstream(limb_obj,
    branch_idx,
    direction="upstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    width_func = width_func,
    width_attribute = width_attribute,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def width_downstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = None,
    width_attribute = None,
    nodes_to_exclude = None,
    **kwargs):
    
    if width_func is None:
        width_func = au.axon_width
    
    return width_upstream_downstream(limb_obj,
    branch_idx,
    direction="downstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    width_func = width_func,
    width_attribute = width_attribute,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    """
    Purpose: To find the up and downstream width

    Pseudocode: 
    1) Get all up/down sk lengths
    2) Get all up/down widths
    3) Filter away non-zeros widths if argument set
    4) If arrays are non-empty, computed the weighted average

    """

    sk_len,nodes = cnu.attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = "skeletal_length",
    concat_func = np.sum,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    verbose = verbose,
    nodes_to_exclude=nodes_to_exclude,
    return_nodes = True,
     **kwargs)
        
        
    if verbose:
        print(f"sk_len = {sk_len} for {len(nodes)} branches ({nodes})")

    if return_nodes:
        return sk_len,nodes
    else:
        return sk_len
    
    
def skeletal_length_upstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    
    return skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction = "upstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    return_nodes = return_nodes,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def skeletal_length_downstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    return_nodes = False,
    nodes_to_exclude = None,
    **kwargs):
    
    return skeletal_length_upstream_downstream(limb_obj,
    branch_idx,
    direction = "downstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    return_nodes = return_nodes,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)


# ------------ synapse density ---------- #



def synapse_density_upstream_downstream(limb_obj,
    branch_idx,
    direction,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    synapse_density_type = "synapse_density",
    nodes_to_exclude = None,
                              **kwargs):
    """
    Purpose: To find the up and downstream width

    Pseudocode: 
    1) Get all up/down sk lengths
    2) Get all up/down widths
    3) Filter away non-zeros widths if argument set
    4) If arrays are non-empty, computed the weighted average

    """


    return cnu.weighted_attribute_upstream_downstream(limb_obj = limb_obj,
    branch_idx = branch_idx,
    direction=direction,
    attribute_name = synapse_density_type,
    include_branch_idx=include_branch_idx,
    distance = distance,
    only_non_branching = only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    verbose = verbose,
    nodes_to_exclude=nodes_to_exclude,
     **kwargs)

    
def synapse_density_upstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    synapse_density_type = "synapse_density",
    nodes_to_exclude = None,
    filter_away_zero_widths = True,
                              **kwargs):
    """
    cnu.width_downstream(limb_obj,
    branch_idx = 65,
    distance = np.inf,
    only_non_branching=False,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    width_func = au.axon_width,
    width_attribute = None,
    return_nodes = False,
    nodes_to_exclude = None,
    filter_away_zero_widths = True,)
    """
    
    return synapse_density_upstream_downstream(limb_obj,
    branch_idx,
    direction="upstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    synapse_density_type=synapse_density_type,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def synapse_density_downstream(limb_obj,
    branch_idx,
    distance = np.inf,
    only_non_branching=True,
    include_branch_in_dist = True,
    include_branch_idx = True,
    verbose = False,
    synapse_density_type= "synapse_density",
    nodes_to_exclude = None,
                              **kwargs):
    
    return synapse_density_upstream_downstream(limb_obj,
    branch_idx,
    direction="downstream",
    distance = distance,
    only_non_branching=only_non_branching,
    include_branch_in_dist = include_branch_in_dist,
    include_branch_idx = include_branch_idx,
    verbose = verbose,
    synapse_density_type=synapse_density_type,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs)

def downstream_nodes(limb_obj,
                    branch_idx):
    """
    Will give the downstream nodes excluding the 
    nodes to be excluded
    
    """
    down_nodes = nru.downstream_nodes(limb_obj,branch_idx)
    return np.setdiff1d(down_nodes,limb_obj.nodes_to_exclude)

def all_downtream_branches(limb_obj,
                          branch_idx):
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,branch_idx)

def all_downtream_branches_including_branch(limb_obj,
                          branch_idx):
    return xu.all_downstream_nodes_including_node(limb_obj.concept_network_directional,branch_idx)

def all_upstream_branches(limb_obj,
                          branch_idx):
    return xu.all_upstream_nodes(limb_obj.concept_network_directional,branch_idx)

def all_upstream_branches_including_branch(limb_obj,
                          branch_idx):
    return xu.all_upstream_nodes_including_node(limb_obj.concept_network_directional,branch_idx)




def skeleton_downstream_restricted(limb_obj,
                                  branch_idx,
                                   downstream_skeletal_length,
                                  downstream_nodes=None,
                                  nodes_to_exclude=None,
                                plot_downstream_skeleton = False,
                                plot_restricted_skeleton = False,
                                verbose = False,
                                  ):
    """
    Purpose: To get restricted downstream skeleton 
    starting from the upstream node
    and going a certain distance

    Application: will help select a part of skeleton
    that we want to find the width around (for axon identification purposes)

    Psuedocode: 
    1) Get the downstream skeleton
    2) Get the upstream coordinate and restrict the skeleton to
    a certain distance away from the upstream coordinate
    3) Calculate the new width based on the skeleton and the meshes

    """
    #1) Get the downstream skeleton
    if downstream_nodes is not None:
        nodes_to_exclude = np.setdiff1d(list(limb_obj.get_branch_names()),downstream_nodes)

    if verbose:
        print(f"nodes_to_exclude = {nodes_to_exclude}")


    downstream_sk = cnu.skeleton_downstream(limb_obj,
                           branch_idx=branch_idx,
                           distance=downstream_skeletal_length,
                           only_non_branching=False,
                           nodes_to_exclude=nodes_to_exclude,
                           plot_skeleton=False)

    upstream_coordinate = nru.upstream_endpoint(limb_obj,branch_idx)
    if verbose:
        print(f"upstream_coordinate = {upstream_coordinate}")

    if plot_downstream_skeleton:
        print(f"Plotting Downstream skeleton:")
        nviz.plot_objects(limb_obj.mesh,
                         skeletons=[downstream_sk],
                         scatters=[upstream_coordinate],
                         scatter_size=1)

    restr_sk = sk.restrict_skeleton_to_distance_from_coordinate(downstream_sk,
               coordinate = upstream_coordinate,
               distance_threshold=downstream_skeletal_length)

    if plot_restricted_skeleton:
        print(f"Plotting Restricted skeleton:")
        nviz.plot_objects(limb_obj.mesh,
                         skeletons=[restr_sk],
                         scatters=[upstream_coordinate],
                         scatter_size=1)

    return restr_sk


def width_downstream_restricted(limb_obj,
                               branch_idx,
                               downstream_skeletal_length,
                               downstream_nodes = None,
                               plot_restricted_skeleton = False,
                                remove_spines_from_mesh = True,
                                verbose = False,
                                **kwargs):
    """
    Purpose: To find the width around a 
    skeleton starting from a certain branch
    and uptream coordinate
    
    Ex: 
    
    from neurd import concept_network_utils as cnu
    
    cnu.width_downstream_restricted(
    limb_obj = neuron_obj_exc_syn_sp[0],
    branch_idx = 21,
    downstream_skeletal_length = 30_000,
    downstream_nodes = [21,26,30,35],
    nodes_to_exclude=None,
    plot_restricted_skeleton = True,
    remove_spines_from_mesh = True,
    verbose = True)

    """

    #1) Get the downstream skeleton
    if downstream_nodes is not None:
        nodes_to_exclude = np.setdiff1d(list(limb_obj.get_branch_names()),downstream_nodes)
    if verbose:
        print(f"nodes_to_exclude = {nodes_to_exclude}")

    old_width = cnu.width_downstream(limb_obj,
                                    branch_idx,
                                    distance=downstream_skeletal_length,
                                    nodes_to_exclude=nodes_to_exclude,
                                    #width_func=nst.width_new,
                                     width_func=nst.width_basic,
                                    )

    if verbose:
        print(f"old_width = {old_width}")


    restr_sk = cnu.skeleton_downstream_restricted(    
        limb_obj = limb_obj,
        branch_idx = branch_idx,
        downstream_skeletal_length = downstream_skeletal_length,
        downstream_nodes = downstream_nodes,
        plot_restricted_skeleton=False,
        **kwargs
    )

    ref_mesh = limb_obj.mesh
    if remove_spines_from_mesh:
        spine_meshes = limb_obj.spines

        if verbose:
            print(f"ref_mesh before spine remove = {ref_mesh}")

        if len(spine_meshes) > 0:
            ref_mesh = tu.subtract_mesh(ref_mesh,tu.combine_meshes(spine_meshes),
                                       error_for_exact_match=False)

        if verbose:
            print(f"ref_mesh after spine_remove = {ref_mesh}")

    if plot_restricted_skeleton:
        print(f"Plotting Restricted skeleton:")
        nviz.plot_objects(ref_mesh,
                         skeletons=[restr_sk],)

    new_width = wu.new_width_from_mesh_skeleton(restr_sk,
                                ref_mesh,
                                backup_width=old_width,
                                verbose = False,
                                   )
    if verbose:
        print(f"new_width = {new_width}")

    return new_width

def G_weighted_from_limb(limb_obj,
                                      weight_name = "weight",
                                    upstream_attribute_for_weight = "skeletal_length",
                                      node_properties = None):
    """
    Purpose: Convert the concept_network_directional
    to a graph with weighted edges being 
    the length of the upstream edge

    Pseudocode: 
    1) Copy the concept network directional
    2) Add the edge weight property
    3) Add any node properties requested
    
    Ex: 
    G = cnu.G_weighted_from_limb(limb_obj,
                                  weight_name = "weight",
                                upstream_attribute_for_weight = "skeletal_length",
                                  node_properties = [nst.width_new])

    from datasci_tools import numpy_utils as nu
    nu.turn_off_scientific_notation()
    xu.get_node_attributes(G,"width_new",24)
    xu.get_edges_with_weights(G)

    """

    G = xu.copy_G_without_data(limb_obj.concept_network_directional)

    for (n1,n2) in G.edges():
        G[n1][n2][weight_name] = nst.get_stat(limb_obj[n1],upstream_attribute_for_weight)

    if node_properties is not None:
        for n_prop in node_properties:
            for n in G.nodes():
                if type(n_prop) == str:
                    curr_name = n_prop
                else:
                    curr_name = str(n_prop.__name__)

                G.nodes[n][curr_name] = nst.get_stat(limb_obj[n],n_prop)
    return G
def all_downstream_branches_from_branches(limb_obj,
                                         branches,
                                         include_original_branches=False,
                                         verbose = False):
    if not nu.is_array_like(branches):
        branches = [branches]
        
    all_downs = np.concatenate([cnu.all_downtream_branches(limb_obj,
                                           k) for k in branches])
    if include_original_branches:
        all_downs = np.concatenate([all_downs,branches])
    else:
        all_downs = np.setdiff1d(all_downs,branches)
        
    downstream_nodes = np.unique(all_downs)
    
    if verbose:
        print(f"downstream_nodes = {downstream_nodes}")
        
    return downstream_nodes

def all_upstream_branches_from_branches(limb_obj,
                                         branches,
                                         include_original_branches=False,
                                         verbose = False):
    if not nu.is_array_like(branches):
        branches = [branches]
        
    all_downs = np.concatenate([cnu.all_upstream_branches(limb_obj,
                                           k) for k in branches])
    if include_original_branches:
        all_downs = np.concatenate([all_downs,branches])
    else:
        all_downs = np.setdiff1d(all_downs,branches)
        
    upstream_nodes = np.unique(all_downs)
    
    if verbose:
        print(f"upstream_nodes = {upstream_nodes}")
        
    return upstream_nodes


# ---- helps with developing statistics over current/above/below branches
def feature_over_branches(
    limb_obj,
    branches,
    direction = None,#downstream or upstream
    include_original_branches_in_direction = False,
    # argument for computing the feature
    feature_name=None,
    feature_function=None,
    combining_function=None,
    return_skeletal_length = False,
    verbose = False,
    **kwargs):
    """
    Purpose: To find the average value over a list of branches

    Pseudocode: 
    1) convert the branches list into the branches
    that will be used to compute the statistic
    2) Compute the skeletal length for all the branches
    3) Compute the statistic for all the nodes
    
    Ex: 
    feature_over_branches(limb_obj = n_obj_2[6],
                                branches = [24,2],
                               direction="upstream",
                               verbose = True,
                               feature_function=ns.width
    )
    """

    #1) convert the branches list into the branches
    #that will be used to compute the statistic
    if direction is not None:
        branches = getattr(cnu,f"all_{direction}_branches_from_branches")(limb_obj,
                                                                         branches,
                                                                         include_original_branches=include_original_branches_in_direction)
        if verbose:
            print(f"New branches computed with direction ({direction}): {branches}")

    #2) Compute the skeletal length for all the branches
    sk_len = [limb_obj[k].skeletal_length for k in branches]

    if verbose:
        print(f"sk_len = {sk_len}")

    #3) compute statistics over branches
    branches_val = nru.feature_over_branches(limb_obj,
                                            branch_list = branches,
                                            feature_name=feature_name,
                                            feature_function=feature_function,
                                            combining_function=combining_function,
                                            **kwargs)
    if verbose:
        print(f"branches_val = {branches_val}")
    
    if return_skeletal_length:
        return branches_val,sk_len
    else:
        return branches_val
    
def weighted_feature_over_branches(
    limb_obj,
    branches,
    direction = None,#downstream or upstream
    include_original_branches_in_direction = False,
    # argument for computing the feature
    feature_name=None,
    feature_function=None,
    combining_function=None,
    default_value = 0,
    verbose = False,
    **kwargs):
    """
    Purpose: To find the average value over a list of branches

    Pseudocode: 
    1) Find features over branches with skeletal length
    4) Do a weighted average based on skeletal length
    
    Ex: 
    weighted_feature_over_branches(limb_obj = n_obj_2[6],
                                branches = [24,2],
                               direction="upstream",
                               verbose = True,
                               feature_function=ns.width
    )
    """
    branches_val,sk_len = cnu.feature_over_branches(
    limb_obj,
    branches,
    direction = direction,#downstream or upstream
    include_original_branches_in_direction = include_original_branches_in_direction,
    # argument for computing the feature
    feature_name=feature_name,
    feature_function=feature_function,
    combining_function=combining_function,
    return_skeletal_length = True,
    verbose = verbose,
    **kwargs)
    
    if len(sk_len) == 0:
        return_value = default_value
    else:
        return_value = nu.weighted_average(branches_val,sk_len)

    if verbose:
        print(f"Weighted value = {return_value}")

    return return_value

def sum_feature_over_branches(
    limb_obj,
    branches,
    direction = None,#downstream or upstream
    include_original_branches_in_direction = False,
    # argument for computing the feature
    feature_name=None,
    feature_function=None,
    combining_function=None,
    default_value = 0,
    verbose = False,
    **kwargs):
    """
    Purpose: To find the average value over a list of branches

    Pseudocode: 
    1) Find features over branches with skeletal length
    4) Do a weighted average based on skeletal length
    
    Ex: 
    cnu.sum_feature_over_branches(limb_obj = n_obj_2[6],
                                branches = [24,2],
                               direction="upstream",
                               verbose = True,
                               feature_function=ns.width
    )
    """
    branches_val,sk_len = cnu.feature_over_branches(
    limb_obj,
    branches,
    direction = direction,#downstream or upstream
    include_original_branches_in_direction = include_original_branches_in_direction,
    # argument for computing the feature
    feature_name=feature_name,
    feature_function=feature_function,
    combining_function=combining_function,
    return_skeletal_length = True,
    verbose = verbose,
    **kwargs)
    
    if len(sk_len) == 0:
        return_value = default_value
    else:
        return_value = np.sum(branches_val)

    if verbose:
        print(f"Sum value = {return_value}")

    return return_value
    
def all_downstream_nodes(limb_obj,branch_idx):
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,
                                   branch_idx)

def upstream_branches_in_branches_list(limb_obj,
                                       branches):
    """
    Purpose: To return branch idxs where 
    other branch idxs are in the downstream 
    nodes of a current branch idx

    Pseudocode: 
    For each branch idx
    1) Get all the downstream nodes
    2) Add it to the upstream list if intersect exists
    """


    upstream_nodes = []
    for b in branches:
        try:
            all_down = cnu.all_downstream_nodes(limb_obj,b)
        except:
            continue
        if len(np.intersect1d(all_down,branches)) > 0:
            upstream_nodes.append(b)

    return upstream_nodes

def downstream_nodes_with_skip_distance(
    limb_obj,
    branch_idx,
    skip_distance = 0,
    skip_nodes = None,
    return_skipped=False,
    verbose = False,
    max_iterations = 1000,
    ):
    """
    Purpose: Find the branches that are immediately downstream of a parent node
    (where branches between a certain threshold are skipped, according to skip distance)
    * could skip multiple branches in a row if all below skip distance
    
    Pseudocode:
    0. Initialize a skipped_nodes,downstream_nodes
    1. Add current node to list of parent nodes
    Iterate through parent nodes until list is empty
        a. get all the downstream nodes
        b. if any of the nodes have a skeletal length less than the skip_distance, add to skip_list and parent nodes
        b2. For any nodes above the skip_distance, add to downstream nodes
    """

    
    G = limb_obj.concept_network_directional
    
    parent_nodes = []
    processed_nodes = set()
    downstream_nodes = set()
    
    def node_distance(n):
        return limb_obj[n].skeletal_length
    
    parent_nodes.append(branch_idx)
    counter = 0
    while len(parent_nodes) > 0:
        p = parent_nodes.pop(0)
        processed_nodes.add(p)
        if verbose:
            print(f"-- Working on parent {p} --")
            
        # children of parent
        children = list(G[p].keys())

        if verbose:
            print(f"children = {children}")
        if len(children) == 0:
            continue
        for c in children:
            if skip_nodes is None:
                node_dist = node_distance(c)
                skip_value = node_dist < skip_distance
                skip_reason = "(dist = {node_dist})"
            else:
                skip_value = c in skip_nodes
                skip_reason = f"(due to argument skip_nodes = {skip_nodes})"
                
            #print(f"skip_reason = {skip_reason}")
            if not skip_value:
                if verbose:
                    print(f"   child {c}: adding to downstream nodes {skip_reason}")
                downstream_nodes.add(c)
            else:
                if verbose:
                    print(f"   child {c}: skipped and added to parent list {skip_reason}")
                if c in processed_nodes:
                    raise Exception("child already in processed nodes")
                parent_nodes.append(c)
        counter += 1
        if counter > max_iterations:
            raise Exception("Max iterations in downstream nodes loop reached")
    
    processed_nodes.remove(branch_idx)
    if skip_nodes is None:
        skip_nodes = list(processed_nodes)
    downstream_nodes = list(downstream_nodes)
    
    if verbose:
        print(f"\n -- Results --\nfor branch {branch_idx} with skip_distance = {skip_distance}:")
        print(f"downstream_nodes = {downstream_nodes}\nskipped_nodes = {skip_nodes}")

    if return_skipped:
        return downstream_nodes,skip_nodes
    return downstream_nodes

#--- from neurd_packages ---
from . import axon_utils as au   
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import width_utils as wu  

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu

from . import concept_network_utils as cnu