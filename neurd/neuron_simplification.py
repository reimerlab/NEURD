'''



For functions that operate over the whole neuron object





'''
import copy 
import time
from python_tools import numpy_dep as np

# --------- functions that will carry out deletion ---------
def branch_idx_map_from_branches_to_delete_on_limb(
    limb_obj,
    branches_to_delete,
    verbose = False,
    ):
    """
    Purpose: To generate a mapping dictionary 
    from nodes to delete
    
    Ex: 
    from neurd import neuron_simplification as nsimp
    nsimp.branch_idx_map_from_branches_to_delete_on_limb(
        limb_obj,
        branches_to_delete = [0,1,5],
        verbose = True

    )

    """
    new_node_name_dict = dict()

    valid_counter = 0
    delete_counter = -1
    for n in limb_obj.get_branch_names():
        if n in branches_to_delete:
            new_node_name_dict[n] = delete_counter
            delete_counter -= 1
        else:
            new_node_name_dict[n] = valid_counter
            valid_counter += 1

    if verbose:
        print(f"new_node_name_dict = {new_node_name_dict}")

    return new_node_name_dict

def reset_concept_network_branch_endpoints(
    limb_obj,
    verbose=False):
    """
    Purpose: To recalculate endpoints of branches on concept network
    """
    for branch_idx in limb_obj.get_branch_names():
        branch_obj = limb_obj[branch_idx]
        limb_obj.concept_network.nodes[branch_idx]["endpoints"] =  branch_obj.endpoints
        
def all_concept_network_data_updated(limb_obj):
    """
    Purpose: To revise the all concept network data
    for a limb object, assuming after the concept network has been reset
    """
    nsimp.reset_concept_network_branch_endpoints(limb_obj)
    starting_node = nru.get_starting_node_from_limb_concept_network(limb_obj)
    all_concept_network_data_revised = limb_obj.all_concept_network_data.copy()
    all_concept_network_data_revised[0]["starting_node"] = starting_node
    all_concept_network_data_revised[0]["starting_endpoints"] = limb_obj[starting_node].endpoints
    all_concept_network_data_revised[0]["concept_network"] = limb_obj.concept_network

    return all_concept_network_data_revised

def delete_branches_from_limb(
    neuron_obj,
    limb_idx,
    branches_to_delete,
    verbose = True,
    
    ):
    """
    Purpose: To adjust a whole limb object
    after floating branch pieces or path branches
    combines (so there is not change to the whole limb mesh)

    Pseudocode:
    1) Find a mapping of old node names to new node names
    2) Renmae the nodes in the concept network
    3) Delete the nodes from the concept network
    4) Fix all of the starting network info using the name map
    """
    st = time.time()
    limb_obj = neuron_obj[limb_idx]
    
    #1) Find a mapping of old node names to new node names
    node_map = nsimp.branch_idx_map_from_branches_to_delete_on_limb(
        limb_obj,
        branches_to_delete = branches_to_delete,
        verbose = verbose

    )
    
    #2) Renmae the nodes in the concept network
    xu.relabel_node_names(limb_obj.concept_network,mapping = node_map)
    
    if verbose:
        print(f"AFter relabeling branch names: {limb_obj.get_branch_names()}")
    
    #3) Delete the nodes from the concept network
    xu.remove_nodes_from(limb_obj.concept_network,[node_map[k] for k in branches_to_delete])
    if verbose:
        print(f"After branch deletion, names: {limb_obj.get_branch_names()}")
    
    #4) Fix all of the starting network info using the name map
    if verbose:
        print(f"Current starting node BEFORE reset: {limb_obj.current_starting_node}")
        
    limb_obj.all_concept_network_data = nsimp.all_concept_network_data_updated(limb_obj)
    limb_obj.set_concept_network_directional(starting_soma=0)
    
    if verbose:
        print(f"Current starting node after reset: {limb_obj.current_starting_node}")
    
    # compute the new limb correspondence
    
    #seting the attributes correctly in the preprocessed data
    limb_idx = nru.limb_idx(limb_idx)
    nru.set_preprocessed_data_from_limb_no_mesh_change(neuron_obj,limb_idx=limb_idx,limb_obj = limb_obj)
    
    if verbose:
        print(f"Total time for deletion: {time.time() - st}")
        
        
def delete_branches_from_neuron(
        neuron_obj,
        limb_branch_dict,
        plot_final_neuron = False,
        verbose = False,
        inplace = False,
    ):
    """
    Purpose: To delete a limb_branch_dict
    from a neuron object if there is no mesh loss

    Pseudocode: 
    """
    if not inplace:
        neuron_obj = nru.copy_neuron(neuron_obj)
    
    if len(limb_branch_dict) == 0:
        final_neuron_obj = neuron_obj
    else:
        for limb_name,branches_to_delete in limb_branch_dict.items():
            if verbose:
                print(f"\n---Working on limb {limb_name}, deleting {branches_to_delete}")
        
            nsimp.delete_branches_from_limb(
                neuron_obj,
                limb_name,
                branches_to_delete=branches_to_delete,
                verbose=verbose
                )
            
    if plot_final_neuron:
        print(f"Plotting final neuron after deletion")
        nviz.visualize_neuron_lite(neuron_obj)
    
    return neuron_obj
    
        
# ----------------------------------------



def combine_path_branches_on_limb(
    limb_obj,
    one_downstream_node_branches = None,
    verbose = True,
    return_branches_to_delete = True,
    inplace = False
    ):
    """
     Purpose: To combine all branches that are along a 
    non-branching path into one branch FOR JUST ONE BRANCH

    1) Find all the nodes with only one child if not already passed
    2) Get all the children of ups (for that branch) and convert into one list
    3) Find the connected components
    For each connected component:
        a) Order the branches from most upstream to least
        b) Determine the most upstream node
        c) Combine each d stream node sequentially with upstream node
        d) Add all downstream nodes to branches to delete


    """
    
    if not inplace:
        limb_obj = copy.deepcopy(limb_obj)


    #1) Find all the nodes with only one child if not already passed
    #if one_downstream_node_branches is None:
    one_downstream_node_branches = [k for k in limb_obj.get_branch_names()
                                   if nru.n_downstream_nodes(limb_obj,k) == 1]

    if verbose:
        print(f"one_downstream_node_branches = {one_downstream_node_branches}")

        
    #3) Find the connected components
    conn_comp_pre = nru.connected_components_from_branches(limb_obj,
                                                           one_downstream_node_branches,
                                                           verbose = False,
                                                          use_concept_network_directional = True)
    if verbose:
        print(f"conn_comp_pre in func = {conn_comp_pre}")
        
    verbose_child = False
    conn_comp = []
    for c in conn_comp_pre:
        if verbose_child:
            print(f"Working on  comp = {c}")
        all_children = []
        for b in c:
            curr_children = nru.children_nodes(limb_obj,b)
            if len(curr_children) != 1:
                raise Exception(f"Branch {b} did not have one child")
            if verbose_child:
                print(f"branch {b} had children {curr_children[0]}")
            all_children.append(curr_children[0])
            
        if verbose_child:
            print(f"all_children = {all_children}")
        new_conn_com = list(np.unique(list(c) + all_children))
#         if verbose:
#             print(f"for conn comp {c}: all_children = {all_children}")
#             print(f"new_conn_com = {new_conn_com}")
        conn_comp.append(new_conn_com)
        
    if verbose:
        print(f"conn_comp in func = {conn_comp}")
        
    """

    #2) Get all the children of ups (for that branch) and convert into one list
    all_nodes = []
    for n in one_downstream_node_branches:
        curr_down = nru.downstream_nodes(limb_obj,n)
        if len(curr_down) != 1:
            raise Exception("")
        if verbose:
            print(f"Branch {n} downstream nodes = {curr_down}")

        all_nodes += [n, curr_down[0]]

    if verbose:
        print(f"All nodes to process = {all_nodes}")
        
    """

    
        
    debug_time = False


    #4) For each connected component:
    branches_to_delete = []
    for j,c in enumerate(conn_comp):
        if verbose:
            print(f"\n---working on comp {j}: {c}")
        #a) Order the branches from most upstream to least
        st = time.time()
        ordered_branches = nru.order_branches_by_skeletal_distance_from_soma(
        limb_obj,
        c,
        verbose = verbose)
        #b) Determine the most upstream node
        up_branch = ordered_branches[0]
        down_branches = ordered_branches[1:]
        
        if debug_time:
            print(f"ordering branches = {time.time() - st}")
            st = time.time()

        #c) Combine each d stream node sequentially with upstream node
        for d in down_branches:
            if verbose:
                print(f"\n-- merging downstream {d} with {up_branch}")
                
            common_endpoint = nru.downstream_endpoint(limb_obj,up_branch)    
            original_downstream_last_endpoint = nru.downstream_endpoint(limb_obj,d) 
            
            new_branch_obj,jitter_segment = bu.combine_branches(
                branch_upstream=limb_obj[up_branch],
                branch_downstream = limb_obj[d],
                verbose = verbose,
                add_skeleton = True,
                add_labels = False,
                common_endpoint = common_endpoint,
                return_jitter_segment = True
            )
            limb_obj[up_branch] = new_branch_obj
            
            if verbose:
                print(f"jitter_segment = {jitter_segment}")
                print(f"b_d.endpoints = {limb_obj[d].endpoints}")
            
            if jitter_segment is not None:
                childs = nru.children_nodes(limb_obj,d)
                if verbose:
                    print(f"Applyig jitter segment to {childs}")
                for c in childs:
                    bu.skeleton_adjust(
                    limb_obj[c],
                    skeleton_append=jitter_segment,
                    )
            
            if debug_time:
                print(f"combining branches = {time.time() - st}")
                st = time.time()
                
        #d) Add all downstream nodes to branches to delete
        branches_to_delete += list(down_branches)
        
        #adjust the concept network to connect to 
        """
        Purpose: To remove from the concept network the connections 
        for the branches that were removed (and then reconnect to original master branch)
        
        Pseudocode: 
        1) Get all of the downstream nodes for all down branches
        For each downstream node: 
        a. delete the upstream to downstream branch
        """
        down_branches_down_nodes = [nru.children_nodes(limb_obj,k) for k in down_branches]

        if verbose:
            print(f"down_branches_down_nodes = {down_branches_down_nodes}")

        for d_idx,(d_branch,d_downs) in enumerate(zip(down_branches,down_branches_down_nodes)):

            if d_idx == 0:
                upstream_edges_add = [[up_branch,d_branch]]
                if verbose:
                    print(f"Removing upstream edges")

                limb_obj.concept_network.remove_edges_from(upstream_edges_add)
                limb_obj.concept_network_directional.remove_edges_from(upstream_edges_add)


            delete_edges = [[d_branch,kkk] for kkk in d_downs]
            if verbose:
                print(f"Adjusting deleting downstream edges {delete_edges}")
            limb_obj.concept_network.remove_edges_from(delete_edges)
            limb_obj.concept_network_directional.remove_edges_from(delete_edges)

            if d_idx == len(down_branches)-1:
                up_add_edges = [[up_branch,kkk] for kkk in d_downs]
                if verbose:
                    print(f"Adjusting the upstream edges with {up_add_edges}")
                limb_obj.concept_network.add_edges_from([[up_branch,kkk] for kkk in d_downs])
                limb_obj.concept_network_directional.add_edges_from([[up_branch,kkk] for kkk in d_downs])

        
        

    if return_branches_to_delete:
        return limb_obj,branches_to_delete
    else:
        return limb_obj
    
    

def combine_path_branches(
    neuron_obj,
    plot_downstream_path_limb_branch = False,
    verbose = True,
    plot_final_neuron= False,
    return_copy = True,
    ):
    """
    Purpose: To combine all branches that are along a 
    non-branching path into one branch in neuron object

    1) Find all nodes with one downstream node (call ups)
    2) For each limb: combine the branches and pass back the ones to delete
    3) Delete all branches on limbs that need deletion and pass back neuron object
    """


    downstream_path_limb_branch = ns.query_neuron(neuron_obj,
                    functions_list=[ns.n_downstream_nodes],
                    query="n_downstream_nodes == 1",
                    return_dataframe=False,
            #limb_branch_dict_restriction=None,
                         plot_limb_branch_dict=plot_downstream_path_limb_branch
                   )
    if verbose:
        print(f"downstream_path_limb_branch= {downstream_path_limb_branch}")
        
    if len(downstream_path_limb_branch) == 0:
        return neuron_obj
    
    if return_copy:
        neuron_obj = copy.deepcopy(neuron_obj)

    limb_branch_to_delete = dict()
    for limb_name,one_node_branches in downstream_path_limb_branch.items():
        if verbose:
            print(f"\n\n---Working on {limb_name}: one_node_branches = {one_node_branches} ")
        new_limb,branches_to_delete = nsimp.combine_path_branches_on_limb(
            limb_obj = neuron_obj[limb_name],
            one_downstream_node_branches = one_node_branches,
            verbose = verbose,
            return_branches_to_delete = True
        )

        neuron_obj[limb_name] = new_limb
        limb_branch_to_delete[limb_name] = branches_to_delete

    if verbose:
        print(f"limb_branch_to_delete= {limb_branch_to_delete}")


    new_neuron_obj = nsimp.delete_branches_from_neuron(
        neuron_obj,
        limb_branch_dict = limb_branch_to_delete,
        plot_final_neuron = plot_final_neuron
    )

    return new_neuron_obj


def floating_end_nodes_limb_branch(
    neuron_obj,
    limb_branch_dict_restriction = "dendrite",
    width_max  = 300,
    max_skeletal_length = 7000,#6000,#5000,
    min_distance_from_soma = 10_000,
    #min_farthest_skeletal_dist = 0,
    return_df = False,
    verbose = False,
    plot = False,
    
    ):
    """
    Purpose: To find a limb branch dict of pieces
    that were probably stitched to the mesh but
    probably dont want splitting the skeleton
    """
    
    if limb_branch_dict_restriction == "dendrite":
        limb_branch_dict_restriction = neuron_obj.dendrite_limb_branch_dict
    
    query =("(n_downstream_nodes == 0) "
          f" and (n_siblings > 0) "
          #f" and (distance_from_soma > )"
          f"and (width_new < {width_max})"
          #f"and (farthest_distance_from_skeleton_to_mesh > {min_farthest_skeletal_dist})"
        f"and (is_branch_mesh_connected_to_neighborhood == False)"
          f" and (skeletal_length < {max_skeletal_length})")
    
    if verbose:
        print(f"query = {query}\n\n")
        
    functions_list=[ns.n_downstream_nodes,
                                   ns.width_new,
                                    "skeletal_length",
                                    "n_siblings",
                                    #"farthest_distance_from_skeleton_to_mesh",
                                    "is_branch_mesh_connected_to_neighborhood"
                                   ]

    limb_br = ns.query_neuron(neuron_obj,
                    functions_list=functions_list,
                    query=query,
                    return_dataframe=False,
            limb_branch_dict_restriction=limb_branch_dict_restriction,
                         plot_limb_branch_dict=plot,

                   )
    if verbose: 
        print(f"floating stitch limb branch: {limb_br}")

    if return_df:
        limb_br_df = ns.query_neuron(neuron_obj,
                        functions_list=functions_list,
                        query=query,
                        return_dataframe=True,
                limb_branch_dict_restriction=limb_branch_dict_restriction,
                             plot_limb_branch_dict=False,

                       )
        
        return limb_br,limb_br_df
    else:
        return limb_br
    

def merge_floating_end_nodes_to_parent(
    neuron_obj,
    floating_end_nodes_limb_branch_dict = None,
    plot_floating_end_nodes_limb_branch_dict = False,
    add_merge_label = True,
    verbose = True,
    plot_final_neuron = False,
    return_copy = True,
    **kwargs
    ):
    """
    Purpose: To combine the floating end nodes
    with their parent branch

    Psueodocode: 
    1) Find all the floating endnodes

    For each limb and branch that is a floating endnode:
        1) Find the parent node
        2) Combine it with parent node

    Create new limb object by deleteing all the end nodes
    """
    if return_copy:
        neuron_obj = copy.deepcopy(neuron_obj)
    

    if floating_end_nodes_limb_branch_dict is None:
        floating_end_nodes_limb_branch_dict = nsimp.floating_end_nodes_limb_branch(
            neuron_obj,
            verbose = verbose,
            plot = plot_floating_end_nodes_limb_branch_dict)

    branches_to_delete = dict()
    for limb_idx,branches in floating_end_nodes_limb_branch_dict.items():
        if verbose:
            print(f"\n-- Working on limb {limb_idx}--")
    
        limb_obj = neuron_obj[limb_idx]
        
        for branch_idx in branches:
            
            parent_node = nru.parent_node(limb_obj,branch_idx)
            if parent_node is None:
                continue
                
            if verbose:
                print(f"\n-- merging downstream {branch_idx} into parent {parent_node}")
            
            limb_obj[parent_node] = bu.combine_branches(
                branch_upstream=limb_obj[parent_node],
                branch_downstream = limb_obj[branch_idx],
                verbose = verbose,
                add_skeleton = False,
                add_labels = False
            )
            
            if add_merge_label:
                limb_obj[parent_node].labels += [f"merged_{branch_idx}"]
                
            if limb_idx not in branches_to_delete:
                branches_to_delete[limb_idx] = []
            
            branches_to_delete[limb_idx].append(branch_idx)
            
            
        neuron_obj[limb_idx] = limb_obj

    if verbose:
        print(f"branches_to_delete= {branches_to_delete}")


    new_neuron_obj = nsimp.delete_branches_from_neuron(
        neuron_obj,
        limb_branch_dict = branches_to_delete,
        plot_final_neuron = plot_final_neuron,
    )
    
    return new_neuron_obj


def branching_simplification(
    neuron_obj,
    return_copy = True,
    
    #floating endpiece arguments
    plot_floating_end_nodes_limb_branch_dict = False,
    plot_final_neuron_floating_endpoints = False,
    return_before_combine_path_branches = False,
    
    # combine path arguments
    plot_downstream_path_limb_branch = False,
    plot_final_neuron_path = False,
    
    
    verbose_merging = False,
    verbose = False,
    plot_after_simplification = False,
    **kwargs,
    ):
    """
    Purpose: Total simplification of neuron object where
    1) eliminates floating end nodes
    2) simplifies path on neuron object

    """
    st = time.time()
    bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    
    
    original_len_dict = {}
    if verbose:
        print(f"N_branches on limbs before simplification")
        for limb_idx in neuron_obj.get_limb_node_names():
            curr_len = len(neuron_obj[limb_idx])
            print(f"{limb_idx}: {curr_len}")
            original_len_dict[limb_idx] = curr_len

    if verbose:
        print(f"--- STARTING merge_floating_end_nodes_to_parent----")
    new_neuron_obj = nsimp.merge_floating_end_nodes_to_parent(
        neuron_obj,
        verbose = verbose_merging,
        plot_floating_end_nodes_limb_branch_dict = plot_floating_end_nodes_limb_branch_dict,
        plot_final_neuron = plot_final_neuron_floating_endpoints,
        return_copy=return_copy
    )
    
    merge_float_len_dict = {}
    if verbose:
        print(f"\n\n\n---N_branches on limbs AFTER merge_floating_end_nodes_to_parent---")
        for limb_idx in new_neuron_obj.get_limb_node_names():
            curr_len = len(new_neuron_obj[limb_idx])
            print(f"{limb_idx}: {curr_len} (difference of {original_len_dict[limb_idx] - curr_len})")
            merge_float_len_dict[limb_idx] = curr_len
    
    if verbose:
        print(f"\n\n\n--- STARTING COMBINING BRANCHES----")
        
    if return_before_combine_path_branches:
        return new_neuron_obj
    
    n_obj_ret = nsimp.combine_path_branches(
        new_neuron_obj,
        plot_downstream_path_limb_branch = plot_downstream_path_limb_branch,
        verbose = verbose_merging,
        plot_final_neuron= plot_final_neuron_path,
        return_copy = False
        )
    
    
    if verbose:
        print(f"\n\n\n---N_branches on limbs AFTER combine_path_branches---")
        for limb_idx in n_obj_ret.get_limb_node_names():
            curr_len = len(n_obj_ret[limb_idx])
            print(f"{limb_idx}: {curr_len} (difference of {merge_float_len_dict[limb_idx] - curr_len})")
            
            
    if verbose:
        print(f"\n\n\n---N_branches on limbs AFTER total simplification---")
        for limb_idx in n_obj_ret.get_limb_node_names():
            curr_len = len(n_obj_ret[limb_idx])
            print(f"{limb_idx}: {curr_len} (difference of {original_len_dict[limb_idx] - curr_len})")
            
    if verbose:
        print(f"\n***Total time for branch simplification = {time.time() - st}")
        
    if plot_after_simplification:
        nviz.visualize_neuron(
            n_obj_ret,
            limb_branch_dict="all"
        )
            
    return n_obj_ret


#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_searching as ns
from . import neuron_utils as nru
from . import proofreading_utils as pru
from . import neuron_visualizations as nviz

#--- from python_tools ---
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np

from . import neuron_simplification as nsimp