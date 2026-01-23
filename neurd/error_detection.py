
import copy
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from pykdtree.kdtree import KDTree
import time
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from datasci_tools import data_struct_utils as dsu

double_back_threshold_axon_thick = 120
double_back_threshold_axon_thin = 127

double_back_threshold_axon_thick_inh = 135
double_back_threshold_axon_thin_inh = 140

# min_upstream_skeletal_distance_global = 500

# #for the high and low degree matches check that will enforce skipping nodes 7000 nm away from soma
# min_distance_from_soma_for_proof_global = 10000




def calculate_skip_distance_poly(
    x=None,
    y=None,
    degree=1):
    
    if x is None:
        x = skip_distance_poly_x_global
    if y is None:
        y = skip_distance_poly_y_global
    
    return nu.polyfit(x,y,degree)

    
def skip_distance_from_branch_width(
    width,
    max_skip = 2300,
    skip_distance_poly=None):
    """
    Purpose: To return the skip distance of of the
    upstream branch based on the width

    Pseudocode: 
    1) Evaluate the skip distance polynomial
    at the certain branch width
    """
    if skip_distance_poly is None:
        skip_distance_poly = ed.calculate_skip_distance_poly()
    skip_value = nu.polyval(skip_distance_poly,width)
    if max_skip is not None:
        if skip_value > max_skip:
            skip_value = max_skip
    return skip_value
    
def width_jump_edges(limb,
                    width_name = "no_spine_median_mesh_center",
                     width_jump_threshold = 100,
                     verbose=False,
                     path_to_check = None,
                    ):
    """
    Will only look to see if the width jumps up by a width_jump_threshold threshold ammount
    and if it does then will save the edges according to that starting soma group
    
    Example: 
    ed = reload(ed)
    ed.width_jump_edges(neuron_obj[5],verbose=True)
    """
    curr_limb = copy.deepcopy(limb)


    width_start_time = time.time()

    error_edges = dict()
    for k in curr_limb.all_concept_network_data:
        curr_soma = k["starting_soma"]
        curr_soma_group = k["soma_group_idx"]
        
        if verbose:
            print(f"Working on Soma {curr_soma} and Soma touching group {curr_soma_group}")
        
        
        curr_limb.set_concept_network_directional(starting_soma=curr_soma,
                                                 soma_group_idx=curr_soma_group,
                                                 suppress_disconnected_errors=True)
        curr_net = curr_limb.concept_network_directional

        if verbose: 
            print(f'Working on soma group {k["soma_group_idx"]}')

        curr_error_edges = []
        for current_nodes in tqdm(curr_net.edges()):
            if not path_to_check is None:
                if len(np.intersect1d(current_nodes,path_to_check)) < 2:
#                     if verbose:
#                         print(f"Skipping edge {current_nodes} because not on path to check: {path_to_check}")
                    continue
            if verbose:
                print(f"  Edge: {current_nodes}")

            up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                                  edge=current_nodes,
                                width_name=width_name,
                                #offset=0,
                                verbose=False)

            downstream_jump = d_width-up_width

            if downstream_jump > width_jump_threshold:
                if verbose:
                    print(f"Adding error edge {current_nodes} because width jump was {downstream_jump}")
                curr_error_edges.append(list(current_nodes))
        
        if curr_soma not in error_edges.keys():
            error_edges[curr_soma] = dict()
        error_edges[curr_soma][curr_soma_group] = curr_error_edges
        
    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    return error_edges

def path_to_edges(path,skip_nodes=[]):
    path = np.array(path)
    for ni in skip_nodes:
        path = path[path != ni]
    return np.vstack([path[:-1],path[1:]]).T

def width_jump_edges_path(limb, #assuming the concept network is already set
                          path_to_check,
                    width_name = "no_spine_median_mesh_center",
                     width_jump_threshold = 100,
                     verbose=False,
                          return_all_edge_info = True,
                          comparison_distance=3000,
                          offset=1000,
                    skip_nodes=[],
                          
                    ):
    """
    Will only look to see if the width jumps up by a width_jump_threshold threshold ammount
    
    **but only along a certain path**
    
    
    Example: 
    curr_limb.set_concept_network_directional(starting_node = 4)
    err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                            path_to_check=np.flip(soma_to_soma_path),
                                        width_jump_threshold=200  )

    err_edges,edges,edges_width_jump
    """


    width_start_time = time.time()
    curr_net = limb.concept_network_directional
    edges = path_to_edges(path_to_check,skip_nodes=skip_nodes)
    edges_width_jump = []
    error_edges = []
    
    
    for current_nodes in edges:
        
        skip_nodes_present = np.intersect1d(skip_nodes,current_nodes)
        if len(skip_nodes_present)>0:
            if verbose:
                print(f"-->Skipping Edge {current_nodes} because had at least on skip node: {skip_nodes_present}")
            continue

        up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(limb,
                              edge=current_nodes,
                            comparison_distance=comparison_distance,
                            width_name=width_name,
                            offset=offset,
                            verbose=False)

        downstream_jump = d_width-up_width
        edges_width_jump.append(downstream_jump)
        
        if verbose:
            print(f"  Edge: {current_nodes}: jump = {np.round(downstream_jump,2)}")
            
        if downstream_jump >= width_jump_threshold:
            if verbose:
                print(f"Adding error edge {current_nodes} because width jump was {downstream_jump}")
            error_edges.append(list(current_nodes))

    
    edges_width_jump = np.array(edges_width_jump)
    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    if return_all_edge_info:
        return error_edges,edges,edges_width_jump
    else:
        return error_edges



def double_back_edges(
    limb,
    double_back_threshold = 130,
    verbose = True,
    comparison_distance=3000,
    offset=0,
    path_to_check=None):

    """
    Purpose: To get all of the edges where the skeleton doubles back on itself

    Application: For error detection


    """

    curr_limb = copy.deepcopy(limb)


    width_start_time = time.time()

    error_edges = dict()
    for k in curr_limb.all_concept_network_data:
        curr_soma = k["starting_soma"]
        curr_soma_group = k["soma_group_idx"]

        if verbose:
            print(f"Working on Soma {curr_soma} and Soma touching group {curr_soma_group}")

        
            
        curr_limb.set_concept_network_directional(starting_soma=curr_soma,
                                                 soma_group_idx=curr_soma_group,
                                                 suppress_disconnected_errors=True)
        curr_net = curr_limb.concept_network_directional

        if verbose: 
            print(f'Working on soma group {k["soma_group_idx"]}')

        curr_error_edges = []
        for current_nodes in tqdm(curr_net.edges()):
            if verbose:
                print(f"  Edge: {current_nodes}")
                
            if not path_to_check is None:
                if len(np.intersect1d(current_nodes,path_to_check)) < 2:
#                     if verbose:
#                         print(f"Skipping edge {current_nodes} because not on path to check: {path_to_check}")
                    continue

            up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                                  edge=current_nodes,
                                  comparison_distance = comparison_distance,
                                offset=offset,
                                verbose=False)

            """
            Pseudocode:
            1) Flip the upstream skeleton (the downstream one should be in right direction)
            2) Get the endpoints from first and last of skeleton coordinates for both to find the vectors
            3) Find the angle between them

            """
            up_sk_flipped = sk.flip_skeleton(up_sk)

            up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
            d_vec = d_sk[-1][-1] - d_sk[0][0]

            
            
            curr_angle = nu.angle_between_vectors(up_vec,d_vec)

            if curr_angle > double_back_threshold:
                curr_error_edges.append(list(current_nodes))

        if curr_soma not in error_edges.keys():
            error_edges[curr_soma] = dict()
        error_edges[curr_soma][curr_soma_group] = curr_error_edges

    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    
    return error_edges



def double_back_edges_path(
    limb,
    path_to_check,
    double_back_threshold = 130,
    verbose = True,
    comparison_distance=3000,
    offset=0,
    return_all_edge_info = True,
    skip_nodes=[]):

    """
    Purpose: To get all of the edges where the skeleton doubles back on itself
    **but only along a certain path**

    Application: For error detection
    
    
    Example: 
    curr_limb.set_concept_network_directional(starting_node = 2)
    err_edges,edges,edges_width_jump = ed.double_back_edges_path(curr_limb,
                            path_to_check=soma_to_soma_path )

    err_edges,edges,edges_width_jump


    """

    curr_limb = limb


    width_start_time = time.time()
    
    curr_net = limb.concept_network_directional
    edges = path_to_edges(path_to_check,skip_nodes=skip_nodes)
    edges_doubling_back = []
    error_edges = []
    
    
    for current_nodes in tqdm(edges):

        skip_nodes_present = np.intersect1d(skip_nodes,current_nodes)
        if len(skip_nodes_present)>0:
            if verbose:
                print(f"-->Skipping Edge {current_nodes} because had at least on skip node: {skip_nodes_present}")
            continue

        up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                              edge=current_nodes,
                              comparison_distance = comparison_distance,
                            offset=offset,
                            verbose=False)

        """
        Pseudocode:
        1) Flip the upstream skeleton (the downstream one should be in right direction)
        2) Get the endpoints from first and last of skeleton coordinates for both to find the vectors
        3) Find the angle between them

        """
        up_sk_flipped = sk.flip_skeleton(up_sk)

        up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
        d_vec = d_sk[-1][-1] - d_sk[0][0]
        
        curr_angle = nu.angle_between_vectors(up_vec,d_vec)
        edges_doubling_back.append(curr_angle)
        
        if verbose:
                print(f"  Edge: {current_nodes}: curr_angle = {np.round(curr_angle,2)}")
                
        
        
        if curr_angle > double_back_threshold:
            error_edges.append(list(current_nodes))

    if verbose: 
        print(f"Total time for doubling_back = {time.time() - width_start_time}")
    if return_all_edge_info:
        return error_edges,edges,edges_doubling_back
    else:
        return error_edges


# ----------- 1/31: This will only compare doubling back and width transitions for big nodes
def width_jump_double_back_edges_path(limb_obj, #assuming the concept network is already set
    path,
    starting_coordinate=None,
    width_name = "no_spine_median_mesh_center",
    width_name_backup = "no_spine_median_mesh_center",

    skeletal_length_to_skip=5000,

    # parameters for the boundary transition
    comparison_distance = 4000,
    offset=2000, #have to make the offset larger because the spines are cancelled out 2000 from the endpoints


    #the thresholds for determining if there are errors
    width_jump_threshold = 200,
    width_jump_axon_like_threshold = 250,
    running_width_jump_method=False,
    
                                      
                                      
    double_back_threshold = 120,
    double_back_axon_like_threshold = None,

    perform_double_back_errors = True,
    perform_width_errors = True,
    perform_axon_width_errors = True,
    skip_double_back_errors_for_axon = True,


    allow_axon_double_back_angle_with_top = None,
    allow_axon_double_back_angle_with_top_width_min = 110,
                                      
    verbose=True,
    return_all_edge_info = True,
    axon_comparison_distance = None):
    
    """
    To get the double back and width jumps along a path of a limb
    (but only for those branches that are deemed significant by a long enough skeletal length)

    -- have options to set for both width and doubling back
    -- option to set that will skip the doubling back if axon (or axon-like) or not
    -- have option for axon width jump (so if want different than dendritic)

    Pseducodde: 
    1) Get the order of coordinates on te path
    2) Calculate the skeletal lengths of branches 
    3) Determine the branches that are too small skeletal wise (deemed insignificant) and remove from path

    -- IF THERE IS AT LEAST 2 BRANCHES LEFT TO TEST --

    4) Revise the ordered coordinates by deleted the indexes that are too small
    5) Compute the enw edges to test
    6) Get the pairs of endpoints for each edge

    7) Iterate through all of the edges to test
        - find if any of the branches are labeled as axon or axon-like
        a. get the skeleton and width boundary
        b. Get the width jump (and record)
        c. Get the skeleton angle (and record)
        d. Depending on the conditions set add the start node and then next node 
           in the original path to the error edges if violates one of the rules

    8) Return the error edges and all of the skeleton angle, width jump data for the path analyzed
    """
    
    if False:
        print(f"running_width_jump_method={running_width_jump_method}")
        print(f"double_back_threshold={double_back_threshold}")
        print(f"double_back_axon_like_threshold={double_back_axon_like_threshold}")
        print(f"skip_double_back_errors_for_axon =  {skip_double_back_errors_for_axon}")
        
    verbose = False

    endpoints_verbose = False
    

    #------------------------------------------------
    path =  np.array(path)
    width_start_time = time.time()

    #0) Figure out the starting coordinate if not specified
    if starting_coordinate is None:
        unique_coordinate = nu.setdiff2d(limb_obj[path[0]].endpoints,limb_obj[path[1]].endpoints)
        if len(unique_coordinate) != 1:
            raise Exception("starting_coordinate was None and there was not just one unique coordinate "
                           f" between 1st and 2nd node on path: {unique_coordinate}")
        else:
            starting_coordinate = unique_coordinate[0]

        if endpoints_verbose:
            print(f"Found starting coordinate = {starting_coordinate}")

    #1) Get the order of coordinates on the path
    ordered_coordinates = nru.ordered_endpoints_on_branch_path(limb_obj = limb_obj,
                path = path,
                starting_endpoint_coordinate = starting_coordinate)

    if endpoints_verbose:
        print(f"ordered_coordinates = \n{ordered_coordinates}")

    #2) Calculate the skeletal lengths of branches 
    branches_sk_len = np.array([limb_obj[k].skeletal_length for k in path])

    #3) Determine the branches that are too small skeletal wise (deemed insignificant) and remove from path
    branch_idx_for_revised_path = np.where(branches_sk_len>skeletal_length_to_skip)[0]
    branch_idx_removed = np.where(branches_sk_len<=skeletal_length_to_skip)[0]

    if len(branch_idx_for_revised_path) < 2:
        if return_all_edge_info:
            return [],[],[],[],[]
        else:
            return []

    revised_path = path[branch_idx_for_revised_path]

    if verbose:
        print(f"branches_removed = {path[branch_idx_removed]}")
        print(f"path: {path} --> revised path: {revised_path}")

    #4) Revise the ordered coordinates by deleted the indexes that are too small
    revised_ordered_coordinates = ordered_coordinates[branch_idx_for_revised_path]

    if endpoints_verbose:
        print(f"revised_ordered_coordinates = \n{revised_ordered_coordinates}")

    #5) Compute the enw edges to test
    revised_edges = ed.path_to_edges(revised_path)

    if endpoints_verbose:
        print(f"revised_edges= {revised_edges}")

    #6) Get the pairs of endpoints for each edge
    common_endpoints_revised = revised_ordered_coordinates.reshape(-1,3)[1:-1].reshape(-1,2,3)

    if endpoints_verbose:
        print(f"common_endpoints_revised = \n{common_endpoints_revised}")

    """
    7) Iterate through all of the edges to test
        - find if any of the branches are labeled as axon or axon-like
        a. get the skeleton and width boundary
        b. Get the width jump (and record)
        c. Get the skeleton angle (and record)
        d. Depending on the conditions set add the start node and then next node 
           in the original path to the error edges if violates one of the rules

    """

    edges_doubling_back = []
    edges_width_jump = []

    error_edges_doubling_back = []
    error_edges_width_jump= []

    width_min = np.inf
    for current_nodes,nodes_common_endpoints in zip(revised_edges,common_endpoints_revised):

        #- find if any of the branches are labeled as axon or axon-like
        curr_labels = limb_obj[current_nodes[0]].labels + limb_obj[current_nodes[1]].labels

        if verbose:
            print(f"curr_labels = {curr_labels}")

        if "axon" in curr_labels:
            axon_flag = True
        else:
            axon_flag = False


        if "axon-like" in curr_labels:
            axon_like_flag = True
        else:
            axon_like_flag = False

        if verbose:
            print(f"Working on edge: {current_nodes}")
            
        if axon_flag and axon_comparison_distance is not None:
            current_comparison_distance = axon_comparison_distance
        else:
            current_comparison_distance = comparison_distance
            
        

        #a. get the skeleton and width boundary
        up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(limb_obj,
                              edge=current_nodes,
                            upstream_common_endpoint=nodes_common_endpoints[0],
                            downstream_common_endpoint=nodes_common_endpoints[1],
                            error_on_no_network_connection=False,
                            comparison_distance=current_comparison_distance,
                            offset=offset,
                            width_name=width_name,
                            verbose=False)

        if up_width < width_min:
            width_min = up_width
            
        #b. Get the width jump (and record)
        
        
        if running_width_jump_method:
            
            downstream_jump = d_width-width_min
        else:
            downstream_jump = d_width-up_width
        
        edges_width_jump.append(downstream_jump)

        #c. Get the skeleton angle (and record)
        curr_angle = sk.parent_child_skeletal_angle(up_sk,d_sk)
        edges_doubling_back.append(curr_angle)

        if verbose:
            print(f"downstream_jump = {downstream_jump}")
            print(f"curr_angle = {curr_angle}")

        #d. Depending on the conditions set add the start node and then next node 
        #in the original path to the error edges if violates one of the rules

        # -- 4/23 Revisions ---------
        """
        What we want is to error out the edge that doubled back and the preceeding one
        """
#         upstream_node = current_nodes[0]
#         downstream_node_for_error = path[np.where(path == upstream_node)[0] + 1][0]

        downstream_node_for_error = current_nodes[-1]
        upstream_node = path[np.where(path == downstream_node_for_error)[0] -1][0]
        
        edge_to_error = [upstream_node,downstream_node_for_error]

        if verbose:
            print(f"POTENTIALLY edge_to_error = {edge_to_error}")



        ''' OLD WAY WITHOUT IMPROVED DOUBLE BACK LOGIC
        if perform_double_back_errors:
            if curr_angle > double_back_threshold:
                if skip_double_back_errors_for_axon and not axon_flag:
                    if verbose:
                        print("Appending edge to double back errors")
                    error_edges_doubling_back.append(edge_to_error)
                else:
                    if verbose:
                        print("Skipping the double back check because axon_flag was set (OTHERWISE WOULD)")
        '''
        if perform_double_back_errors:
            if axon_flag and skip_double_back_errors_for_axon:
                if verbose:
                    print("Skipping the double back check because axon_flag was set (OTHERWISE WOULD)")
            else:
                if axon_flag and double_back_axon_like_threshold is not None:
                    curr_double_back_threshold = double_back_axon_like_threshold
                else:
                    curr_double_back_threshold = double_back_threshold

                if curr_angle > curr_double_back_threshold:
                    append_flag = True
                    
                    # ---- 4/23 Addition --------#
                    #will allow for axon to double back if it is pointing back to top
                    if allow_axon_double_back_angle_with_top is not None:
                        if allow_axon_double_back_angle_with_top_width_min is None:
                            allow_axon_double_back_angle_with_top_width_min = -1
                        d_vec_child = d_sk[-1][-1] - d_sk[0][0]
                        angle_with_top = nst.angle_from_top(d_vec_child)
                        upstream_node_width = au.axon_width(limb_obj[upstream_node])
                        #print(f"upstream_node_width = {upstream_node_width}, allow_axon_double_back_angle_with_top_width_min = {allow_axon_double_back_angle_with_top_width_min}")
                        #print(f"angle_with_top = {angle_with_top} (threshold = {allow_axon_double_back_angle_with_top})")
                        if ((angle_with_top < allow_axon_double_back_angle_with_top ) and 
                           (upstream_node_width > allow_axon_double_back_angle_with_top_width_min)):
                            print("Skipping the double back even though double back threshold violated because "
                                  f"the angle with the top is {angle_with_top} which is less than set threshold {allow_axon_double_back_angle_with_top} "
                                  f"\nand upstream_node_width ({upstream_node_width}) is greater than threshold {allow_axon_double_back_angle_with_top_width_min}")
                            append_flag = False
                            
                    if append_flag:
                        if verbose:
                            print(f"Appending edge to double back errors with threshold: {curr_double_back_threshold}")
                        error_edges_doubling_back.append(edge_to_error)
#                         from datasci_tools import system_utils as su
#                         su.compressed_pickle(up_sk,"up_sk")
#                         su.compressed_pickle(d_sk,"d_sk")
#                         raise Exception("")
                else:
                    if verbose:
                        print(f"Skipping the double back becuase {curr_angle} angle not larger than threshold {curr_double_back_threshold}")
                        

        if perform_width_errors:
            if axon_like_flag or axon_flag:
                curr_wdith_threshold = width_jump_axon_like_threshold
            else:
                curr_wdith_threshold = width_jump_threshold
                
            add_width_errors_flag = True
            
            if axon_flag and not perform_axon_width_errors:
                add_width_errors_flag = False

            if downstream_jump > curr_wdith_threshold:
                if add_width_errors_flag:
                    if verbose:
                        print("Appending edge to width errors")
                    error_edges_width_jump.append(edge_to_error)
                

    all_error_edges = error_edges_doubling_back + error_edges_width_jump
    if len(all_error_edges) > 0:
        all_error_edges = nu.unique_rows(np.vstack(all_error_edges).reshape(-1,2))




    if return_all_edge_info:
        return all_error_edges,error_edges_doubling_back,error_edges_width_jump,edges_doubling_back,edges_width_jump
    else:
        return all_error_edges


# ----------------------------------------------------- #
        
        
    
def resolving_crossovers(limb_obj,
                        coordinate,
                        match_threshold = 65,
                         #match_threshold = 60,
                        verbose = False,
                         return_new_edges = True,
                        return_subgraph=False,
                        plot_intermediates=False,
                         offset=1000,
                         comparison_distance = 1000,
                         
                         apply_width_filter = None,
                         best_match_width_diff_max = None,
                         best_match_width_diff_max_perc = None,
                         best_match_width_diff_min = None,
                         
                         best_singular_match=None,
                         lowest_angle_sum_for_pairs=None,
                         return_existing_edges = True,
                         
                         
                         
                         edges_to_avoid = None,
                         
                         no_non_cut_disconnected_comps = None, #should be true
                         branches_to_disconnect = None, #will distinguish branches that need cutting from those that dont
                         
                        **kwargs):
    
    """
    Purpose: To determine the connectivity that should be at the location
    of a crossover (the cuts that should be made and the new connectivity)

    Pseudocode: 
    1) Get all the branches that correspond to the coordinate
    2) For each branch
    - get the boundary cosine angle between the other branches
    - if within a threshold then add edge
    3) Ge the subgraph of all these branches:
    - find what edges you have to cut
    4) Return the cuts/subgraph
    
    Ex: 
    resolving_crossovers(limb_obj = copy.deepcopy(curr_limb),
                     coordinate = high_degree_coordinates[0],
                    match_threshold = 40,
                    verbose = False,
                     return_new_edges = True,
                    return_subgraph=True,
                    plot_intermediates=False)

    """
    if apply_width_filter is None:
        apply_width_filter = apply_width_filter_global
        
    if best_match_width_diff_max is None:
        best_match_width_diff_max = best_match_width_diff_max_global
        
    if best_match_width_diff_max_perc is None:
        best_match_width_diff_max_perc = best_match_width_diff_max_perc_global
        
    if best_match_width_diff_min is None:
        best_match_width_diff_min = best_match_width_diff_min_global
        
    if no_non_cut_disconnected_comps is None:
        no_non_cut_disconnected_comps = no_non_cut_disconnected_comps_global
    
    if best_singular_match is None:
        best_singular_match = best_singular_match_global
        
    if lowest_angle_sum_for_pairs is None:
        lowest_angle_sum_for_pairs = lowest_angle_sum_for_pairs_global
        
        
    
#     print(f"comparison_distance = {comparison_distance}")
#     print(f"offset= {offset}")
    
    
    debug = False
    if debug:
        debug_dict = dict(apply_width_filter = apply_width_filter,
        best_match_width_diff_max = best_match_width_diff_max,
        best_match_width_diff_max_perc = best_match_width_diff_max_perc,
        best_match_width_diff_min = best_match_width_diff_min,
        best_singular_match=best_singular_match,
        lowest_angle_sum_for_pairs=lowest_angle_sum_for_pairs,)
        print(f"Inisde resolving_crossovers: debug_dict=/n{debug_dict}")
    
    
    
    #1) Get all the branches that correspond to the coordinate
    sk_branches = [br.skeleton for br in limb_obj]

    coordinate_branches = np.sort(sk.find_branch_skeleton_with_specific_coordinate(sk_branches,coordinate))
    curr_colors = ["red","aqua","purple","green"]

    
    if verbose: 
        print(f"coordinate = {coordinate}")
        print(f"coordinate_branches = {list(coordinate_branches)}")
        for c,col in zip(coordinate_branches,curr_colors):
            print(f"{c} = {col}")
    
    
    
    if plot_intermediates:
        
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=[limb_obj[k].skeleton for k in coordinate_branches],
                         skeletons_colors=curr_colors)
    
    
    # 2) For each branch
    # - get the boundary cosine angle between the other branches
    # - if within a threshold then add edge

    match_branches = []
    match_branches_angle = []
    
    all_aligned_skeletons = []
    
    if verbose:
        print(f"edges_to_avoid= {edges_to_avoid}")
    
    for br1_idx in coordinate_branches:
        for br2_idx in coordinate_branches:
            if br1_idx>=br2_idx:
                continue
            
            if edges_to_avoid is not None:
                if len(nu.intersect2d(np.sort(edges_to_avoid,axis = 1),
                           np.sort(np.array([br1_idx,br2_idx])).reshape(-1,2))) > 0:
                    if verbose:
                        print(f"Skipping edge: {[br1_idx,br2_idx]} because in edges_to_avoid ")
                    continue
                
                
            edge = [br1_idx,br2_idx]
            edge_skeletons = [sk_branches[e] for e in edge]
            aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                             offset=offset,
                                                                             comparison_distance=comparison_distance,
                                                                             common_endpoint=coordinate)
            

            curr_angle = sk.parent_child_skeletal_angle(aligned_sk_parts[0],aligned_sk_parts[1])
            
            
            if verbose:
                print(f"Angle between {br1_idx} and {br2_idx} = {curr_angle} ")

            # - if within a threshold then add edge
            if curr_angle <= match_threshold:
                """
                ----- 7/15: Will now eliminate all edges where the width jump is too large ----
        
                best_match_width_diff_max = 75,
                best_match_width_diff_max_perc = 0.60,
                """
                add_edge_flag = True
                if apply_width_filter:
                    
                    width_diff = nst.width_diff_basic(limb_obj,br1_idx,br2_idx)
                    width_diff_perc = nst.width_diff_percentage_basic(limb_obj,br1_idx,br2_idx)
                    
                    if verbose:
                        print(f"width_diff = {width_diff}, width_diff_perc = {width_diff_perc}\n")
                    
                    if (((width_diff > best_match_width_diff_max) and (width_diff_perc > best_match_width_diff_max_perc))
                        and (width_diff_perc > best_match_width_diff_min)):
                        if verbose:
                            print(f"Not adding edge {[br1_idx,br2_idx]} because width_diff= {width_diff}, width_diff_perc= = {width_diff_perc}")
                        add_edge_flag = False
                
                if add_edge_flag:
                    match_branches.append([br1_idx,br2_idx])
                    match_branches_angle.append(curr_angle)
                
            if plot_intermediates:
                #saving off the aligned skeletons to visualize later
                all_aligned_skeletons.append(aligned_sk_parts[0])
                all_aligned_skeletons.append(aligned_sk_parts[1])
    
    if verbose: 
        print(f"Final Matches = {match_branches}")
        
    
    
        
    # -------- 12 / 31 : Will attempt to only keep the best singular match between nodes ------- #
    
    if lowest_angle_sum_for_pairs:
        """
        Psuedocode: (if matched_branches is not empty)
        1) Turn the matched branhes and the match_branches_angle
        into a weighted graph
        2) Get the lowest weight graph and output the edges
        3) Turn to lists and reassign as the matched branches
        
        """
        #print(f"Using lowest_angle_sum_for_pairs optimization")
        if len(match_branches)>0:
            
            ''' JUST COMPILED PROCESS INTO FUNCTION
            curr_branches_G = xu.edges_and_weights_to_graph(match_branches,
                                                            match_branches_angle)
            match_branches,match_branches_angle = xu.degree_1_max_edge_min_max_weight_graph(
                G = curr_branches_G,
                verbose = False,
                plot_winning_graph = False,
                return_edge_info=True)
            '''
            match_branches,match_branches_angle  = xu.lowest_weighted_sum_singular_matches(
                                                            match_branches,
                                                            match_branches_angle)
            
            match_branches = list(match_branches)
            match_branches_angle = list(match_branches_angle)
        
        
    elif best_singular_match:
        """
        Pseudocode: 
        0) Create an alredy_matched list
        1) Get the sorting indexes by match angle
        2) Iterate through match angle indexes in order:
        a. if both nodes in edge have not been matched 
            - add edge to final list
            - add edge nodes to already_matched list
        b. else:
            - skip edge
            
        """
        
        already_matched = []
        matched_branches_revised = []
        smallest_to_largest_angles = np.argsort(match_branches_angle)
        for e_i in smallest_to_largest_angles:
            curr_edge = match_branches[e_i]
            
            if len(np.intersect1d(already_matched,curr_edge))==0:
                matched_branches_revised.append(curr_edge)
                already_matched += list(curr_edge)
        if verbose:
            print(f"matched_branches_revised = {matched_branches_revised}")
        match_branches = matched_branches_revised
    else:
        if verbose:
            print("Not using any edge optimized pairing")
        
    
    
    if plot_intermediates:
        print("Aligned Skeleton Parts")
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=all_aligned_skeletons)
    
    if plot_intermediates:
        for curr_match in match_branches:
            nviz.plot_objects(meshes=[limb_obj[k].mesh for k in curr_match],
                 meshes_colors=curr_colors,
                 skeletons=[limb_obj[k].skeleton for k in curr_match],
                 skeletons_colors=curr_colors)
            
    
    # find what cuts and connections need to make
    limb_subgraph = limb_obj.concept_network.subgraph(coordinate_branches)
    
    if verbose:
        print("Original graph")
        nx.draw(limb_subgraph,with_labels=True)
        plt.show()
        print(f"match_branches = {match_branches}")
    
    """ Older way of doing that just accounted for edges in current graph
    sorted_edges = np.sort(limb_subgraph.edges(),axis=1)
    if len(match_branches)>0:
        
        sorted_confirmed_edges = np.sort(match_branches,axis=1)


        edges_to_delete = []

        for ed in sorted_edges:
            if not return_existing_edges:
                if len(nu.matching_rows_old(sorted_confirmed_edges,ed))==0:
                    edges_to_delete.append(ed)
            else:
                edges_to_delete.append(ed)

        edges_to_create = []

        for ed in sorted_confirmed_edges:
            if not return_existing_edges:
                if len(nu.matching_rows_old(sorted_edges,ed))==0:
                    edges_to_create.append(ed)
            else:
                edges_to_create.append(ed)
    else:
        edges_to_delete = sorted_edges
        edges_to_create = []
    """
    
    """
    1/10/22: 
    Purpose: Don't want edges that are not 
    the ones subject to cutting to be disconnected even if not find pairing
    
    Pseudocode: 
    1) Get the current edges to create
    2) Build a graph from those matched edges and nodes considering
    3) For each node not in nodes to cut:
       a) If not
    """
    if no_non_cut_disconnected_comps:
        
        
        
        if branches_to_disconnect is None:
            raise Exception("no_non_cut_disconnected_comps but branches_to_disconnect is None in resolving_crossovers")
            
        branches_to_disconnect = np.intersect1d(coordinate_branches,branches_to_disconnect)
        
        all_start_nodes = limb_obj.all_starting_nodes
        branches_to_avoid = np.setdiff1d(all_start_nodes,branches_to_disconnect)
        if verbose:
            print(f"branches_to_avoid= {branches_to_avoid}")
        
        G = nx.Graph()
        G.add_nodes_from(coordinate_branches)
        G.add_edges_from(match_branches)
        
        new_neighbors = []
        for b in np.setdiff1d(coordinate_branches,
                              branches_to_disconnect,
                              ):
            if b in branches_to_avoid:
                continue
            curr_neighbors = xu.get_neighbors_simple(G,b)
            if len(np.intersect1d(curr_neighbors
                              ,branches_to_disconnect))==0:
                if verbose:
                    print(f"{b}: No Pair so adding back old edge")
                old_neighbors = xu.get_neighbors_simple(limb_subgraph,b)
                
                if verbose:
                    print(f"{b}: Old neighbors = {old_neighbors}")

                new_neighbors+= [list(np.sort([b,o])) for o in old_neighbors if o not in branches_to_avoid]
           
        if verbose:     
            print(f"new_neighbors = {new_neighbors}")
        
        match_branches += new_neighbors
        
        
    # ----------- 1/8/21 New iteration that just says what edges should be there and what shouldn't
    edges_to_create = np.array(match_branches).tolist()
    possible_edges = np.sort(np.array(list(itertools.combinations(coordinate_branches, 2))),axis=1)
    edges_to_delete = nu.setdiff2d(possible_edges,match_branches).tolist()
    
    
            
    if verbose: 
        print(f"edges_to_delete (resolve crossover) = {edges_to_delete}")
        print(f"edges_to_create (resolve crossover) = {edges_to_create}")
    
    return_value = [edges_to_delete] 
    if return_new_edges:
        return_value.append(edges_to_create)
    if return_subgraph:
        #actually creating the new sugraph
        graph_copy = nx.Graph(limb_obj.concept_network)
        graph_copy.remove_edges_from(edges_to_delete)
        graph_copy.add_edges_from(edges_to_create)
        
        if verbose:
            print(f"n_components in adjusted graph = {nx.number_connected_components(graph_copy)}")
        return_value.append(graph_copy)
        
        
    return return_value





# ------------ part that will error all floating axon pieces ----------- #


def error_branches_by_axons(neuron_obj,verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 65):
    
    if neuron_obj.n_limbs == 0:
        if return_axon_non_axon_faces:
            axon_faces = np.array([])
            non_axon_faces = np.arange(len(neuron_obj.mesh.faces))
            return np.array([]),axon_faces,non_axon_faces
        return np.array([])
    
    axon_seg_dict = au.axon_like_segments(neuron_obj,include_ais=False,
                                          filter_away_end_false_positives=True,
                                          visualize_at_end=False,
                                         )

    # Step 2: Get the branches that should not be considered for axons


    to_keep_limb_names = nru.filter_limbs_below_soma_percentile(neuron_obj,verbose=False)

    axons_to_consider = dict([(k,v) for k,v in axon_seg_dict.items() if k in to_keep_limb_names])
    axons_to_not_keep = dict([(k,v) for k,v in axon_seg_dict.items() if k not in to_keep_limb_names])
    

    if verbose:
        print(f"Axons not keeping because of soma: {axons_to_not_keep}")

    # Step 3: Erroring out the axons based on projections

    valid_axon_branches_by_limb = dict()
    not_valid_axon_branches_by_limb = dict()


    
    axon_vector = np.array([0,1,0])

    for curr_limb_name,curr_axon_nodes in axons_to_consider.items():
        if verbose:
            print(f"\n----- Working on {curr_limb_name} ------")
        # curr_limb_name = "L0"
        # curr_axon_nodes = axons_to_consider[curr_limb_name]
        curr_limb_idx = int(curr_limb_name[1:])
        curr_limb = neuron_obj[curr_limb_idx]


        #1) Get the nodes that are axons


        #2) Group into connected components
        curr_limb_network = nx.from_edgelist(curr_limb.concept_network.edges())
        axon_subgraph = curr_limb_network.subgraph(curr_axon_nodes)
        axon_connected_components = list(nx.connected_components(axon_subgraph))

        valid_axon_branches = []

        #3) Iterate through the connected components
        for ax_idx,ax_group in enumerate(axon_connected_components):
            valid_flag = False
            if verbose:
                print(f"-- Axon Group {ax_idx} of size {len(ax_group)}--")
            for soma_idx in curr_limb.touching_somas():
                all_start_node = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_node")
                all_start_coord = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_coordinate")
                for start_node,start_coord in zip(all_start_node,all_start_coord):
                    if verbose:
                        print(f"   Working on soma {soma_idx}, starting_node {start_node}")

                    #find the shortest path between the axon group and the starting node
                    current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(curr_limb_network,[start_node],list(ax_group))
                    

                    #get the skeleton of the path
                    path_skeletons = sk.stack_skeletons([curr_limb[k].skeleton for k in current_shortest_path])

                    #order the skeleton by a certain coordinate
                    ordered_path_skeleton = sk.order_skeleton(path_skeletons,start_endpoint_coordinate=start_coord)

                    #check and see if skeletal distance is lower than distance check and if it is then use a different angle check
                    if sk.calculate_skeleton_distance(ordered_path_skeleton)< min_skeletal_path_threshold:
                        if verbose:
                            print(f"Using AIS angle threshold {ais_angle_threshold}")
                        curr_angle_threshold = ais_angle_threshold
                    else:
                        if verbose:
                            print("Not using AIS angle threshold")
                        curr_angle_threshold = non_ais_angle_threshold


                    #get the first skeletal distance of threshold
                    keep_skeleton_indices = np.where(sk.calculate_skeleton_segment_distances(ordered_path_skeleton)<=sub_skeleton_length)[0]

                    
                    restricted_skeleton = ordered_path_skeleton[keep_skeleton_indices]
                    restricted_skeleton_endpoints_sk = np.array([restricted_skeleton[0][0],restricted_skeleton[-1][-1]]).reshape(-1,2,3)
                    restricted_skeleton_vector = np.array(restricted_skeleton[-1][-1]-restricted_skeleton[0][0])
                    restricted_skeleton_vector = restricted_skeleton_vector/np.linalg.norm(restricted_skeleton_vector)

                    #angle between going down and skeleton vector
                    sk_angle = nu.angle_between_vectors(axon_vector,restricted_skeleton_vector)
                    if verbose:
                        print(f"sk_angle= {sk_angle}")

                    if sk_angle > curr_angle_threshold:
                        if verbose:
                            print("*****Path to axon group not valid******")
                    else:
                        if verbose:
                            pass
                            #print("Path to axon group valid so adding them as valid axon segments")
                        valid_axon_branches.append(list(ax_group))
                        valid_flag = True
                        break

    #                 if curr_limb_name == "L1":
    #                     raise Exception()

                if valid_flag:
                    break


        
            

        
        if len(valid_axon_branches) > 0:
            valid_axon_branches_by_limb[curr_limb_name] = np.concatenate(valid_axon_branches)
            not_valid_axon_branches_by_limb[curr_limb_name] = list(np.setdiff1d(curr_axon_nodes,np.concatenate(valid_axon_branches)))
        else:
            valid_axon_branches_by_limb[curr_limb_name] = []
            not_valid_axon_branches_by_limb[curr_limb_name] = list(curr_axon_nodes)
        
        if verbose:
            print(f"\n\nFor limb {curr_limb_idx} the valid axon branches are {valid_axon_branches_by_limb[curr_limb_name] }")
            print(f"The following are not valid: {not_valid_axon_branches_by_limb[curr_limb_name]}")

    # Step 4: Compiling all the errored faces


    final_error_axons = copy.copy(axons_to_not_keep)
    final_error_axons.update(not_valid_axon_branches_by_limb)
    
    if verbose:
        print(f"final_error_axons = {final_error_axons}")
    
    if visualize_errors_at_end:
        nviz.visualize_neuron(neuron_obj,
                              visualize_type=["mesh"],
                              limb_branch_dict=final_error_axons,
                             mesh_color="red",
                             mesh_whole_neuron=True)
        
    return final_error_axons


def error_faces_by_axons(neuron_obj,error_branches = None,
                         verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 65,
                         return_axon_non_axon_faces=False):
    """
    Purpose: Will return the faces that are errors after computing 
    the branches that are errors
    
    
    
    """
    
    if error_branches is None:
        final_error_axons = error_branches_by_axons(neuron_obj,verbose=verbose,
                                                    visualize_errors_at_end=False,
                        min_skeletal_path_threshold = min_skeletal_path_threshold,
                                sub_skeleton_length = sub_skeleton_length,
                                ais_angle_threshold = ais_angle_threshold,
                                non_ais_angle_threshold = non_ais_angle_threshold)
    else:
        final_error_axons = error_branches
        
    
    # Step 5: Getting all of the errored faces
    error_faces = []
    for curr_limb_name,error_branch_idx in final_error_axons.items():
        curr_limb = neuron_obj[curr_limb_name]
        curr_error_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                            [curr_limb[k].mesh for k in error_branch_idx],
                                       matching=True,
                                       print_flag=False)


        #curr_error_faces = np.concatenate([new_limb_mesh_face_idx[curr_limb[k].mesh_face_idx] for k in error_branch_idx])
        error_faces.append(curr_error_faces)

    if len(error_faces) > 0:
        error_faces_concat = np.concatenate(error_faces)
    else:
        error_faces_concat = error_faces
        
    error_faces_concat = np.array(error_faces_concat).astype("int")
        
    if verbose:
        print(f"\n\n -------- Total number of error faces = {len(error_faces_concat)} --------------")

    if visualize_errors_at_end:
        nviz.plot_objects(main_mesh = neuron_obj.mesh,
            meshes=[neuron_obj.mesh.submesh([error_faces_concat],append=True)],
                         meshes_colors=["red"])
        
    if return_axon_non_axon_faces:
        if verbose:
            print("Computing the axon and non-axonal faces")
        axon_faces = nru.limb_branch_dict_to_faces(neuron_obj,valid_axon_branches_by_limb)
        non_axon_faces = np.delete(np.arange(len(neuron_obj.mesh.faces)),axon_faces)
        return error_faces_concat,axon_faces,non_axon_faces
        
    return error_faces_concat

''' Old Way that did not have the function split up
def error_faces_by_axons(neuron_obj,verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 50,
                         return_axon_non_axon_faces=False):
    
    if neuron_obj.n_limbs == 0:
        if return_axon_non_axon_faces:
            axon_faces = np.array([])
            non_axon_faces = np.arange(len(neuron_obj.mesh.faces))
            return np.array([]),axon_faces,non_axon_faces
        return np.array([])
    
    axon_seg_dict = au.axon_like_segments(neuron_obj,include_ais=False,
                                          filter_away_end_false_positives=True,
                                          visualize_at_end=False,
                                         )

    # Step 2: Get the branches that should not be considered for axons


    to_keep_limb_names = nru.filter_limbs_below_soma_percentile(neuron_obj,verbose=False)

    axons_to_consider = dict([(k,v) for k,v in axon_seg_dict.items() if k in to_keep_limb_names])
    axons_to_not_keep = dict([(k,v) for k,v in axon_seg_dict.items() if k not in to_keep_limb_names])
    

    if verbose:
        print(f"Axons not keeping because of soma: {axons_to_not_keep}")

    # Step 3: Erroring out the axons based on projections

    valid_axon_branches_by_limb = dict()
    not_valid_axon_branches_by_limb = dict()


    
    axon_vector = np.array([0,1,0])

    for curr_limb_name,curr_axon_nodes in axons_to_consider.items():
        if verbose:
            print(f"\n----- Working on {curr_limb_name} ------")
        # curr_limb_name = "L0"
        # curr_axon_nodes = axons_to_consider[curr_limb_name]
        curr_limb_idx = int(curr_limb_name[1:])
        curr_limb = neuron_obj[curr_limb_idx]


        #1) Get the nodes that are axons


        #2) Group into connected components
        curr_limb_network = nx.from_edgelist(curr_limb.concept_network.edges())
        axon_subgraph = curr_limb_network.subgraph(curr_axon_nodes)
        axon_connected_components = list(nx.connected_components(axon_subgraph))

        valid_axon_branches = []

        #3) Iterate through the connected components
        for ax_idx,ax_group in enumerate(axon_connected_components):
            valid_flag = False
            if verbose:
                print(f"-- Axon Group {ax_idx} of size {len(ax_group)}--")
            for soma_idx in curr_limb.touching_somas():
                all_start_node = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_node")
                all_start_coord = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_coordinate")
                for start_node,start_coord in zip(all_start_node,all_start_coord):
                    if verbose:
                        print(f"   Working on soma {soma_idx}, starting_node {start_node}")

                    #find the shortest path between the axon group and the starting node
                    current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(curr_limb_network,[start_node],list(ax_group))
                    

                    #get the skeleton of the path
                    path_skeletons = sk.stack_skeletons([curr_limb[k].skeleton for k in current_shortest_path])

                    #order the skeleton by a certain coordinate
                    ordered_path_skeleton = sk.order_skeleton(path_skeletons,start_endpoint_coordinate=start_coord)

                    #check and see if skeletal distance is lower than distance check and if it is then use a different angle check
                    if sk.calculate_skeleton_distance(ordered_path_skeleton)< min_skeletal_path_threshold:
                        if verbose:
                            print(f"Using AIS angle threshold {ais_angle_threshold}")
                        curr_angle_threshold = ais_angle_threshold
                    else:
                        if verbose:
                            print("Not using AIS angle threshold")
                        curr_angle_threshold = non_ais_angle_threshold


                    #get the first skeletal distance of threshold
                    keep_skeleton_indices = np.where(sk.calculate_skeleton_segment_distances(ordered_path_skeleton)<=sub_skeleton_length)[0]

                    
                    restricted_skeleton = ordered_path_skeleton[keep_skeleton_indices]
                    restricted_skeleton_endpoints_sk = np.array([restricted_skeleton[0][0],restricted_skeleton[-1][-1]]).reshape(-1,2,3)
                    restricted_skeleton_vector = np.array(restricted_skeleton[-1][-1]-restricted_skeleton[0][0])
                    restricted_skeleton_vector = restricted_skeleton_vector/np.linalg.norm(restricted_skeleton_vector)

                    #angle between going down and skeleton vector
                    sk_angle = nu.angle_between_vectors(axon_vector,restricted_skeleton_vector)
                    if verbose:
                        print(f"sk_angle= {sk_angle}")

                    if sk_angle > curr_angle_threshold:
                        if verbose:
                            print("*****Path to axon group not valid******")
                    else:
                        if verbose:
                            pass
                            #print("Path to axon group valid so adding them as valid axon segments")
                        valid_axon_branches.append(list(ax_group))
                        valid_flag = True
                        break

    #                 if curr_limb_name == "L1":
    #                     raise Exception()

                if valid_flag:
                    break


        
            

        
        if len(valid_axon_branches) > 0:
            valid_axon_branches_by_limb[curr_limb_name] = np.concatenate(valid_axon_branches)
            not_valid_axon_branches_by_limb[curr_limb_name] = list(np.setdiff1d(curr_axon_nodes,np.concatenate(valid_axon_branches)))
        else:
            valid_axon_branches_by_limb[curr_limb_name] = []
            not_valid_axon_branches_by_limb[curr_limb_name] = list(curr_axon_nodes)
        
        if verbose:
            print(f"\n\nFor limb {curr_limb_idx} the valid axon branches are {valid_axon_branches_by_limb[curr_limb_name] }")
            print(f"The following are not valid: {not_valid_axon_branches_by_limb[curr_limb_name]}")

    # Step 4: Compiling all the errored faces


    final_error_axons = copy.copy(axons_to_not_keep)
    final_error_axons.update(not_valid_axon_branches_by_limb)
    
    if verbose:
        print(f"final_error_axons = {final_error_axons}")


    # Step 5: Getting all of the errored faces

    error_faces = []
    for curr_limb_name,error_branch_idx in final_error_axons.items():
        curr_limb = neuron_obj[curr_limb_name]
        curr_error_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                            [curr_limb[k].mesh for k in error_branch_idx],
                                       matching=True,
                                       print_flag=False)


        #curr_error_faces = np.concatenate([new_limb_mesh_face_idx[curr_limb[k].mesh_face_idx] for k in error_branch_idx])
        error_faces.append(curr_error_faces)

    if len(error_faces) > 0:
        error_faces_concat = np.concatenate(error_faces)
    else:
        error_faces_concat = error_faces
        
    error_faces_concat = np.array(error_faces_concat).astype("int")
        
    if verbose:
        print(f"\n\n -------- Total number of error faces = {len(error_faces_concat)} --------------")

    if visualize_errors_at_end:
        nviz.plot_objects(main_mesh = neuron_obj.mesh,
            meshes=[neuron_obj.mesh.submesh([error_faces_concat],append=True)],
                         meshes_colors=["red"])
        
    if return_axon_non_axon_faces:
        if verbose:
            print("Computing the axon and non-axonal faces")
        axon_faces = nru.limb_branch_dict_to_faces(neuron_obj,valid_axon_branches_by_limb)
        non_axon_faces = np.delete(np.arange(len(neuron_obj.mesh.faces)),axon_faces)
        return error_faces_concat,axon_faces,non_axon_faces
        
    return error_faces_concat

'''




    
# ---- 4/22 v4 Error Detection Rules ----------

'''
def axon_fork_divergence_errors_limb_branch_dict(neuron_obj,
                                           divergence_threshold_mean = 160,
                                            upstream_width_threshold = 80,
                                           verbose = False,
                                                ):
    """
    Purpose: Will create a limb branch dict of all the skinny forking errors
    on an axon

    Pseudocode: 
    1) Find the axon limb of the neuron (if none then return emptty dictionary)
    2) Restrict the neuron to only axon pieces, with a width below certain threshold and having one sibling
    3) Run the fork divergence function 
    4) Return the limb branch dict highlight the errors where occured

    """
    
    
    if neuron_obj.axon_limb_name is None:
        return {}
    
    axon_brancehs = ns.query_neuron_by_labels(neuron_obj,
                                         matching_labels = ["axon"])

    two_downstream_thick_axon_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list = ["n_siblings","axon_width"],
                   query = f"(n_siblings == 1) and (axon_width)<{downstream_width_threshold}",
                   return_dataframe=False,
                    limb_branch_dict_restriction=axon_brancehs,
                   limbs_to_process=[neuron_obj.axon_limb_name])
    if verbose:
        print(f"two_downstream_thick_axon_limb_branch = {two_downstream_thick_axon_limb_branch}")

    fork_div_limb_branch = ns.query_neuron(neuron_obj,
               functions_list = ["fork_divergence"],
               query = f"fork_divergence < {divergence_threshold_mean}",
               return_dataframe=False,
                limb_branch_dict_restriction=two_downstream_thick_axon_limb_branch,
               limbs_to_process=[neuron_obj.axon_limb_name])
    
    if verbose:
        print(f"With divergence_threshold_mean = {divergence_threshold_mean}\nfork_div_limb_branch = {fork_div_limb_branch}")

    return fork_div_limb_branch

'''
def attempt_width_matching_for_fork_divergence(neuron_obj,
                                              fork_div_limb_branch,
                                              width_match_threshold = 10,
                                              width_match_buffer = 10,
                                              verbose = False):
    """
    Purpose: To see if there is a possible winner in the forking
    based on width matching, and if there is then 
    remove it from the error branches

    Pseudocode:
    1) Divide the branches into sibling groups
    2) For each sibling group:

    a. Get the upstream node and its width
    b. Get the widths of all of the sibling nodes 
    c. subtract the upstream nodes with from them and take the absolute value
    d. get the minimum differences and check 2 things:
        i) less than width_match_threshold
        2) less than maximum difference by width_match_buffer

    e1. If yes --> then only add the argmax to the error branches
    e2. If no --> then add both to the error branches

    """
    #1) Divide the branches into sibling groups
    fork_div_limb_branch_rev = dict()

    for limb_name,branch_list in fork_div_limb_branch.items():

        limb_obj = neuron_obj[limb_name]
        fork_div_limb_branch_rev[limb_name] = []

        #1) Divide the branches into sibling groups
        sib_groups = xu.group_nodes_into_siblings(G = neuron_obj[limb_name].concept_network_directional,
            nodes = branch_list,
            verbose = False)

        if verbose:
            print(f"For {limb_name} sib_groups= {sib_groups}")

        for s in sib_groups:
            #a. Get the upstream node and its width
            if len(s) != 2:
                if verbose:
                    print(f"Not processing {s} because there was not 2 nodes in pair")
                continue
                
            up_node = xu.upstream_node(limb_obj.concept_network_directional,s[0])
            up_node_width = au.axon_width(limb_obj[up_node])

            d_widths = np.array([au.axon_width(limb_obj[k]) for k in s])
            width_differences = np.abs(d_widths - up_node_width)

            if verbose:
                print(f"For sibling group {s}: upstream node = {up_node}")
                print(F"Widths are {d_widths}, upstream_width = {up_node_width}")
                print(f"width_differences= {width_differences}")

            """
            d. get the minimum differences and check 2 things:
            i) less than width_match_threshold
            2) less than maximum difference by width_match_buffer

            """
            winning_idx = None

            min_idx = np.argmin(width_differences)
            max_idx = 1 - min_idx

            if ((width_differences[min_idx]  < width_match_threshold) and
                (width_differences[min_idx] + width_match_buffer < width_differences[max_idx])):
                if verbose:
                    print(f"With min_idx= {min_idx}, {s[min_idx]} was a matching node to the upstream node")
                winning_idx = min_idx

            """
            e1. If yes --> then only add the argmax to the error branches
            e2. If no --> then add both to the error branches
            """
            if winning_idx is None:
                fork_div_limb_branch_rev[limb_name] += s
            else:
                fork_div_limb_branch_rev[limb_name] += [s[max_idx]]
    return fork_div_limb_branch_rev


def axon_fork_divergence_errors_limb_branch_dict(neuron_obj,
                                           divergence_threshold_mean = 160,
                                            width_threshold = 90,
                                            upstream_width_max = 90,
                                            verbose = False,
                                            plot_two_downstream_thick_axon_limb_branch = False,
                                            plot_fork_div_limb_branch = False,
                                            
                                            #arguments for attempting a matching of the one of the 2 parts of fork
                                            attempt_width_matching = True,
                                            width_match_threshold = 10,
                                            width_match_buffer = 10,
                                                 
                                                ):
    """
    Purpose: Will create a limb branch dict of all the skinny forking errors
    on an axon

    Pseudocode: 
    1) Find the axon limb of the neuron (if none then return emptty dictionary)
    2) Restrict the neuron to only axon pieces, with a width below certain threshold and having one sibling
    3) Run the fork divergence function 
    4) Return the limb branch dict highlight the errors where occured

    """
    
    
    if neuron_obj.axon_limb_name is None:
        return {}

    axon_brancehs = ns.query_neuron_by_labels(neuron_obj,
                                         matching_labels = ["axon"])

    two_downstream_thick_axon_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list = ["n_siblings","axon_width","upstream_axon_width"],
                   query = f"(n_siblings == 1) and (axon_width<{width_threshold})"
                           f" and (upstream_axon_width < {upstream_width_max})",
                   return_dataframe=False,
                    limb_branch_dict_restriction=axon_brancehs,
                   limbs_to_process=[neuron_obj.axon_limb_name])
    if verbose:
        print(f"two_downstream_thick_axon_limb_branch = {two_downstream_thick_axon_limb_branch}")

    if plot_two_downstream_thick_axon_limb_branch:
        nviz.plot_limb_branch_dict(neuron_obj,
                              two_downstream_thick_axon_limb_branch)

    fork_div_limb_branch = ns.query_neuron(neuron_obj,
               functions_list = ["fork_divergence"],
               query = f"fork_divergence < {divergence_threshold_mean}",
               return_dataframe=False,
                limb_branch_dict_restriction=two_downstream_thick_axon_limb_branch,
               limbs_to_process=[neuron_obj.axon_limb_name])

    if verbose:
        print(f"With divergence_threshold_mean = {divergence_threshold_mean}\nfork_div_limb_branch = {fork_div_limb_branch}")
        
    if attempt_width_matching:
        fork_div_limb_branch = ed.attempt_width_matching_for_fork_divergence(neuron_obj,
                                              fork_div_limb_branch,
                                              width_match_threshold = width_match_threshold,
                                              width_match_buffer = width_match_buffer,
                                              verbose = verbose)
    if plot_fork_div_limb_branch:
        nviz.plot_limb_branch_dict(neuron_obj,
                              fork_div_limb_branch)
    
    final_limb_branch = {k:v for k,v in fork_div_limb_branch.items() if len(v) > 0}

    return final_limb_branch



def webbing_t_errors_limb_branch_dict(neuron_obj,
                                     axon_only = True,
                                #child_width_maximum = 75,
                                child_width_maximum = 75,
                                parent_width_maximum = 75,
                                plot_two_downstream_thin_axon_limb_branch = False,
                                plot_wide_angled_children = False,
                                error_if_web_is_none = True,
                                verbose = True,

                                #arguments for the web thresholding
                                web_size_threshold=120,
                                web_size_type="ray_trace_median",
                                web_above_threshold = True,

                                plot_web_errors = False,
                                    child_skeletal_threshold = 10000,
                                     ignore_if_child_mesh_not_touching=True):
    """
    Purpose: Return all of the branches that are errors based on the 
    rule that when the axon is small and forms a wide angle t then 
    there should be a characteristic webbing that is wide enough 
    (if not then it is probably just a merge error)

    Pseudocode: 
    1) Find all of the candidate branches in the axon
    2) Find all those that have a webbing t error
    3) find all of the downstream nodes of that nodes 
    and add them to a limb branch dict that gets returned


    """
    
    wide_angled_children = au.wide_angle_t_candidates(neuron_obj,
                                                     axon_only = axon_only,
                                                child_width_maximum = child_width_maximum,
                                                parent_width_maximum = parent_width_maximum,
                                                plot_two_downstream_thin_axon_limb_branch = plot_two_downstream_thin_axon_limb_branch,
                                                plot_wide_angled_children = plot_wide_angled_children,
                                                child_skeletal_threshold = child_skeletal_threshold,
                                                verbose = verbose)

    if ignore_if_child_mesh_not_touching:
        if verbose:
            print(f"wide_angled_children before ignoring non touhing meshes = {wide_angled_children}")
        wide_angled_children = ns.query_neuron(neuron_obj,
                functions_list=["downstream_nodes_mesh_connected"],
                query="downstream_nodes_mesh_connected == True",
               limb_branch_dict_restriction=wide_angled_children)
        if verbose:
            print(f"wide_angled_children AFTER ignoring non touhing meshes = {wide_angled_children}")
    
    invalid_branches_from_webbing = dict()
    for l_name,error_web_branches in wide_angled_children.items():

        limb_obj = neuron_obj[l_name]
        local_errors = []

        for w in error_web_branches:
            curr_web = limb_obj[w].web

            add_downstream_nodes = False
            if error_if_web_is_none and curr_web is None:
                add_downstream_nodes = True
            elif curr_web is None:
                add_downstream_nodes = False
            elif not au.valid_web_for_t(curr_web,
                                   size_threshold = web_size_threshold,
                                   size_type = web_size_type,
                                   above_threshold = web_above_threshold,
                                       verbose=verbose):
                add_downstream_nodes = True

            if add_downstream_nodes:
                down_nodes = list(xu.downstream_nodes(limb_obj.concept_network_directional,w))
                if len(down_nodes) > 0:
                    if verbose:
                        print(f"From limb {l_name}, branch {w}, Adding the downstream nodes {down_nodes}  ")
                    local_errors += down_nodes

        if len(local_errors)>0:
            invalid_branches_from_webbing[l_name] = local_errors

    if verbose:
        print(f"Final web t error limb branch dict = {invalid_branches_from_webbing}")

    if plot_web_errors:
        nviz.plot_limb_branch_dict(neuron_obj,
                                     invalid_branches_from_webbing)
    return invalid_branches_from_webbing

def webbing_t_errors_limb_branch_dict_old(neuron_obj,
                                     axon_only = True,
                                #child_width_maximum = 75,
                                child_width_maximum = 75,
                                parent_width_maximum = 75,
                                plot_two_downstream_thin_axon_limb_branch = False,
                                plot_wide_angled_children = False,
                                error_if_web_is_none = True,
                                verbose = True,

                                #arguments for the web thresholding
                                web_size_threshold=120,
                                web_size_type="ray_trace_median",
                                web_above_threshold = True,

                                plot_web_errors = False,
                                    child_skeletal_threshold = 10000,
                                     ignore_if_child_mesh_not_touching=True):
    """
    Purpose: Return all of the branches that are errors based on the 
    rule that when the axon is small and forms a wide angle t then 
    there should be a characteristic webbing that is wide enough 
    (if not then it is probably just a merge error)

    Pseudocode: 
    1) Find all of the candidate branches in the axon
    2) Find all those that have a webbing t error
    3) find all of the downstream nodes of that nodes 
    and add them to a limb branch dict that gets returned


    """
    
    wide_angled_children = au.wide_angle_t_candidates(neuron_obj,
                                                     axon_only = axon_only,
                                                child_width_maximum = child_width_maximum,
                                                parent_width_maximum = parent_width_maximum,
                                                plot_two_downstream_thin_axon_limb_branch = plot_two_downstream_thin_axon_limb_branch,
                                                plot_wide_angled_children = plot_wide_angled_children,
                                                child_skeletal_threshold = child_skeletal_threshold,
                                                verbose = verbose)


    invalid_branches_from_webbing = dict()
    upstream_nodes_for_error = dict()
    for l_name,error_web_branches in wide_angled_children.items():

        limb_obj = neuron_obj[l_name]
        local_upstream = []

        for w in error_web_branches:
            curr_web = limb_obj[w].web

            add_downstream_nodes = False
            if error_if_web_is_none and curr_web is None:
                add_downstream_nodes = True
            elif curr_web is None:
                add_downstream_nodes = False
            elif not au.valid_web_for_t(curr_web,
                                   size_threshold = web_size_threshold,
                                   size_type = web_size_type,
                                   above_threshold = web_above_threshold,
                                       verbose=verbose):
                add_downstream_nodes = True

            if add_downstream_nodes:
                local_upstream.append(w)
        
        if len(local_upstream) > 0:
            upstream_nodes_for_error[l_name] = local_upstream
    
    if ignore_if_child_mesh_not_touching:
        if verbose:
            print(f"local_upstream before ignoring non touhing meshes = {local_upstream}")
        upstream_nodes_for_error = ns.query_neuron(neuron_obj,
                functions_list=["downstream_nodes_mesh_connected"],
                query="downstream_nodes_mesh_connected == True",
               limb_branch_dict_restriction=upstream_nodes_for_error)
        if verbose:
            print(f"local_upstream AFTER ignoring non touhing meshes = {local_upstream}")
                
                
    for l_name,error_web_branches in upstream_nodes_for_error.items():
        local_errors = []
        for w in error_web_branches:
            down_nodes = list(xu.downstream_nodes(limb_obj.concept_network_directional,w))
            if len(down_nodes) > 0:
                if verbose:
                    print(f"From limb {l_name}, branch {w}, Adding the downstream nodes {down_nodes}  ")
                local_errors += down_nodes

        if len(local_errors)>0:
            invalid_branches_from_webbing[l_name] = local_errors

    if verbose:
        print(f"Final web t error limb branch dict = {invalid_branches_from_webbing}")

    if plot_web_errors:
        nviz.plot_limb_branch_dict(neuron_obj,
                                     invalid_branches_from_webbing)
    return invalid_branches_from_webbing

    
# -------- New Rule 4: High Degree Branching ----------#
    
def matched_branches_by_angle(limb_obj,
                              branches,
                             **kwargs):
    
    coordinate = nru.shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                    branches[0],
                                                    branches[1],
                                                    check_concept_network_connectivity=False)
    
    
    return matched_branches_by_angle_at_coordinate(limb_obj,
    coordinate=coordinate,
    coordinate_branches = branches,
    **kwargs)

def matched_branches_by_angle_at_coordinate(limb_obj,
    coordinate,
    coordinate_branches = None,
    offset=1000,
    comparison_distance = 1000,
    match_threshold = 45,
    verbose = False,
    plot_intermediates = False,
    return_intermediates = False,
    plot_match_intermediates = False,
    less_than_threshold = True):
    """
    Purpose: Given a list of branch indexes on a limb that all touch, find:
    a) the skeleton angle between them all
    b) apply a threshold on the angle between to only keep those below/above
    
    Ex: 
    from neurd import error_detection as ed
    ed.matched_branches_by_angle_at_coordinate(limb_obj,
                                            coordinate,
                                            offset=1500,
                                            comparison_distance = 1000,
                                            match_threshold = 40,
                                            verbose = True,
                                            plot_intermediates = False,
                                            plot_match_intermediates = False)
    """

    if coordinate_branches is None:
        coordinate_branches= nru.find_branch_with_specific_coordinate(limb_obj,coordinate)

    if len(coordinate_branches) != 4:
        curr_colors = mu.generate_non_randon_named_color_list(len(coordinate_branches))
    else:
        curr_colors = ["red","aqua","purple","green"]
    if verbose: 
        print(f"coordinate_branches = {list(coordinate_branches)}")
        for c,col in zip(coordinate_branches,curr_colors):
            print(f"{c} = {col}")

    if plot_intermediates:
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=[limb_obj[k].skeleton for k in coordinate_branches],
                         skeletons_colors=curr_colors)

    match_branches = []
    match_branches_angle = []

    all_aligned_skeletons = []

    for br1_idx in coordinate_branches:
        for br2_idx in coordinate_branches:
            if br1_idx>=br2_idx:
                continue


            edge = [br1_idx,br2_idx]
            edge_skeletons = [limb_obj[e].skeleton for e in edge]
            aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                             offset=offset,
                                                                             comparison_distance=comparison_distance,
                                                                             common_endpoint=coordinate)


            curr_angle = sk.parent_child_skeletal_angle(aligned_sk_parts[0],aligned_sk_parts[1])


            if verbose:
                print(f"Angle between {br1_idx} and {br2_idx} = {curr_angle} ")

            # - if within a threshold then add edge
            if less_than_threshold:
                if curr_angle <= match_threshold:
                    match_branches.append([br1_idx,br2_idx])
                    match_branches_angle.append(curr_angle)
            else:
                if curr_angle >= match_threshold:
                    match_branches.append([br1_idx,br2_idx])
                    match_branches_angle.append(curr_angle)

            if plot_intermediates:
                #saving off the aligned skeletons to visualize later
                all_aligned_skeletons.append(aligned_sk_parts[0])
                all_aligned_skeletons.append(aligned_sk_parts[1])

    if verbose: 
        print(f"Final Matches = {match_branches}, Final Matches Angle = {match_branches_angle}")


    if plot_match_intermediates:
        print("Aligned Skeleton Parts")
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=all_aligned_skeletons)

    if plot_match_intermediates:
        for curr_match in match_branches:
            nviz.plot_objects(meshes=[limb_obj[k].mesh for k in curr_match],
                 meshes_colors=curr_colors,
                 skeletons=[limb_obj[k].skeleton for k in curr_match],
                 skeletons_colors=curr_colors)
            
    if return_intermediates:
        return match_branches,match_branches_angle,all_aligned_skeletons,curr_colors
    else:
        return match_branches,match_branches_angle
    
    

'''
def high_degree_upstream_match_old(
    limb_obj,
    coordinate = None,
    upstream_branch = None,
    downstream_branches = None,

    #arguments for the angle checking
    offset=1500,
    comparison_distance = 2000,
    worst_case_match_threshold = 65,
    plot_intermediates = False,
    plot_match_intermediates = False,

    #args for width matching
    width_diff_max = 75,#np.inf,100,
    width_diff_perc = 60,

    #args for definite pairs
    match_threshold = 45,
    angle_buffer = 15,
    
    max_degree_to_resolve = 6,
    max_degree_to_resolve_wide = 8,
    max_degree_to_resolve_width_threshold = 200,
    
    axon_dependent = True,
    
    width_max = 170,

    #args for picking the final winner
    match_method = "best_match", #other option is "best_match"
    
    remove_short_thick_endnodes = True,
    short_thick_endnodes_to_remove = None,
    min_degree_to_resolve = 4,
    verbose = False,
    
    kiss_check = True,
    kiss_check_bbox_longest_side_threshold = 450,
    
    ):
    """
    Purpose: To figure out which downstream
    node is the most likely continuation of the 
    upstream node

    Pseudocode: 
    0) Determine branches touching coordinate and which node is the upstream node and which are downstream
    1) Compute the skeletal angles between all branches
    2) Create a skeletal graph where make the edges between
    all nodes that meet the worst case scenario
    3) Compute the width difference between all branches connected by an edge
    Remove all the edges that violate the width difference threshold
    4) Create definite pairs by looking for edges that meet:
    - match threshold
    - have buffer better than other edges
    ** for those edges, eliminate all edges on those
    2 nodes except that edge

    5) If the upstream node has at least one valid 
    match then eliminate others above the match threshold

    6) Get a subgraph that includes the upstream node:
    if there are other nodes in the group use on of the following to determine winner
        a) best match
        b) least sum angle

    7) Return the winning edge, and optionally all of the other
    downstream nodes that are errored out

    """
    

    #0) Determine which node is the upstream node and which are downstream
    if upstream_branch is None or downstream_branches is None:
        branches_at_coord = nru.find_branch_with_specific_coordinate(limb_obj,coordinate)

        upstream_branch, downstream_branches = nru.classify_upstream_downsream(limb_obj,
                                   branch_list = branches_at_coord,
                                    verbose = verbose)
    else:
        branches_at_coord = np.hstack([downstream_branches,[upstream_branch]])

    if verbose:
        print(f"branches_at_coord = {branches_at_coord}")
        
        
    """
    4/24 Addition: Will remove the short thick axon endnodes
    """
    if remove_short_thick_endnodes:
        if short_thick_endnodes_to_remove is None:
            short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
                                 verbose = False)
        branches_at_coord = np.setdiff1d(branches_at_coord,short_thick_endnodes_to_remove)
            
    if len(branches_at_coord) < min_degree_to_resolve:
        if verbose:
            print(f"Number of branches ({len(branches_at_coord)}, aka branches_at_coord = {branches_at_coord}) was less than min_degree_to_resolve ({min_degree_to_resolve}) so returning no error branches")
        return None,np.array([])
        
    # -- end of short thick addition --------
    

    if max_degree_to_resolve_wide is not None:
        up_width = au.axon_width(limb_obj[upstream_branch])
        if up_width > max_degree_to_resolve_width_threshold:
            max_degree_to_resolve = max_degree_to_resolve_wide
            print(f"Changing max_degree_to_resolve = {max_degree_to_resolve_wide} because upstream width was {up_width} ")
    
    if len(branches_at_coord) > max_degree_to_resolve:
        if verbose:
            print(f"Number of branches ({len(branches_at_coord)}) was more than max_degree_to_resolve ({max_degree_to_resolve}) so returning all downstream as error branches")
        return None,branches_at_coord
    
    widths_in_branches = np.array([au.axon_width(limb_obj[b]) for b in branches_at_coord])
    if verbose:
        print(f"widths_in_branches = {widths_in_branches}")
        
    widths_in_branches = widths_in_branches[widths_in_branches != 0]
    
    if width_max is not None:
        if len(widths_in_branches[widths_in_branches>width_max]) == len(widths_in_branches):
            if verbose:
                print(f"Returning No errors because widths are too thick for skeletons to be trusted")
            return None,[]
    
    if axon_dependent:
        for b in branches_at_coord:
            if "axon" not in limb_obj[b].labels:
                if verbose:
                    print(f"Returning No errors because not all branches were axons")
                return None,[]

    #1) Compute the skeletal angles between all branches
    matched_edges, matched_edges_angles = ed.matched_branches_by_angle_at_coordinate(limb_obj,
                                            coordinate,
                                               coordinate_branches = branches_at_coord,
                                            offset=offset,
                                            comparison_distance = comparison_distance,
                                            match_threshold = worst_case_match_threshold,
                                            verbose = verbose,
                                            plot_intermediates = plot_intermediates,
                                            plot_match_intermediates = plot_match_intermediates,
                                            )
    if verbose:
        print(f"matched_edges = {matched_edges}"
              f"matched_edges_angles = {matched_edges_angles}")



    # 2) Create a skeletal graph where make the edges between
    # all nodes that meet the worst case scenario

    G = nx.Graph()
    G.add_nodes_from(branches_at_coord)
    G.add_weighted_edges_from([k+[v] for k,v in zip(matched_edges,matched_edges_angles)])

    if verbose:
        print(f"Step 2: Edges with worst case scenario matching = {worst_case_match_threshold}")
        print(f"Remaining Edges = {G.edges()}, Remaining Nodes = {G.nodes()}")
    #     nx.draw(G,with_labels=True)
    #     plt.show()


    """
    3) Compute the width difference between all branches connected by an edge
    Remove all the edges that violate the width difference threshold
    """
    
    
    edges_to_remove_by_width = []
    for e in G.edges():
        b1,b2 = e
        b1_width = au.axon_width(limb_obj[b1])
        b2_width = au.axon_width(limb_obj[b2])    
        width_difference = np.abs(b1_width-b2_width)
        
        if width_diff_perc is not None:
            width_diff_max_perc = width_diff_perc*np.max([b1_width,b2_width])/100
            #print(f"Computed width_diff_max as {width_diff_max_perc} using width_diff_perc = {width_diff_perc} and width_diff_max = {width_diff_max}")
            if width_diff_max is not None:
                width_diff_max = np.max([width_diff_max_perc,width_diff_max])
                
                #print(f"The maximum width chosen was {width_diff_max}")
            else:
                width_diff_max = width_diff_max_perc
                
        if width_difference > width_diff_max:
            if verbose:
                print(f"Removing edges {e} because width difference {width_difference}")
            edges_to_remove_by_width.append(e)
    if verbose:
        print(f"edges_to_remove_by_width = {edges_to_remove_by_width}")

    G.remove_edges_from(edges_to_remove_by_width)

    if verbose:
        print(f"Step 2: Edges after widht mismatch")
        print(f"Remaining Edges = {G.edges()}, Remaining Nodes = {G.nodes()}")
    #     nx.draw(G,with_labels=True)
    #     plt.show()

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
    """
    other_edges_to_remove = []
    for e in G.edges():
        e = np.sort(e)
        if verbose:
            print(f"--Working on edge {e}--")
        e_weight = xu.get_edge_weight(G,e)
        all_edges = np.unique(
                    np.sort(
                    np.array(xu.node_to_edges(G,e[0]) + xu.node_to_edges(G,e[1])),axis=1)
                    ,axis=0)


        #b) get all the other edges that are touching the two nodes and their weights
        other_edges = nu.setdiff2d(all_edges,e.reshape(-1,2))

        if len(other_edges) == 0:
            other_edge_min = np.inf
        else:
            other_edge_weights = [xu.get_edge_weight(G,edg) for edg in other_edges]
            #print(f"other_edge_weights = {other_edge_weights}")
            other_edge_min = np.min(other_edge_weights)
            #print(f"other_edge_min = {other_edge_min}")

        edge_buffer = other_edge_min - e_weight
        if e_weight <= match_threshold and edge_buffer > angle_buffer:
            if verbose:
                print(f"Edge {e} is matches definite match threshold with: "
                     f"\nEdge Buffer of {edge_buffer} (angle_buffer = {angle_buffer})"
                     f"\nEdge Angle of {e_weight} (match_threshold = {match_threshold})")
            other_edges_to_remove += list(other_edges)
    G.remove_edges_from(other_edges_to_remove)

    if verbose:
        print(f"Step 4: Definite Edges")
        print(f"Remaining Edges = {G.edges()}, Remaining Nodes = {G.nodes()}")
    #     nx.draw(G,with_labels=True)
    #     plt.show()


    """
    5) If the upstream node has at least one valid 
    match then eliminate others above the match threshold
    """

    upstream_subgraph = np.array([list(k) for k in nx.connected_components(G) 
                                  if upstream_branch in k][0])
    upstream_G = G.subgraph(upstream_subgraph)

    if verbose:
        print(f"upstream_subgraph = {upstream_subgraph}")

    poss_connections = np.array(xu.get_neighbors(upstream_G,upstream_branch))
    poss_connections_weights = np.array([xu.get_edge_weight(G,(upstream_branch,k)) for k in poss_connections])

    if verbose:
        print(f"Possible Connections = {poss_connections}, angles = {poss_connections_weights}")

    n_below_match = len(np.where(poss_connections_weights<=match_threshold)[0])
    if n_below_match > 0:
        e_to_delete = [(upstream_branch,k) for k in 
                       poss_connections[poss_connections_weights>match_threshold]]
        if verbose:
            print(f"Deleting the following nodes because above match threshold while {n_below_match} are: {e_to_delete}")
        G.remove_edges_from(e_to_delete)

    if verbose:
        print(f"Step 5: Removing worst case edges")
        print(f"Remaining Edges = {G.edges()}")

        
    
        
    """
    ---------- 4/29 Addition: Kiss Filter -----------

    Pseudocode:
    0) Get the offset skeleton coordinates for all nodes in graph
    1) find all the possible partitions of the remaining nodes

    """ 
    if kiss_check:
        upstream_matches = xu.get_neighbors(G,upstream_branch)
        if len(upstream_matches)>1:
            print(f"Working on Kissing check because possible upstream matches greater than 1: {upstream_matches}")

            G = ed.cut_kissing_graph_edges(G,limb_obj,
                kiss_check_bbox_longest_side_threshold = kiss_check_bbox_longest_side_threshold,
                coordinate = coordinate,
                offset=offset,
                comparison_distance = comparison_distance,
                plot_offset_skeletons = False,
                plot_source_sink_vertices = False,
                plot_cut_vertices = False,
                plot_cut_bbox = False,
                verbose = False
                            )

            if verbose:
                print(f"Step 5b: Removing kissing edges")
                print(f"Remaining Edges = {G.edges()}")
        else:
            if verbose:
                print(f"Not doing kiss check because upstream_matches = {upstream_matches}")


    


    """
    Part 6:
    if there are other nodes in the group use on of the following to determine winner
        a) best match
        b) least sum angle
    """
    upstream_subgraph = np.array([list(k) for k in nx.connected_components(G) 
                                  if upstream_branch in k][0])
    if len(upstream_subgraph) == 1:
        winning_node = None
        error_branches = downstream_branches
    else:
        if match_method == "best_match":
            if verbose:
                print(f"Using best match method")
            winning_node = xu.get_neighbor_min_weighted_edge(G,upstream_branch)
        elif match_method == "lowest_angle_sum":
            if verbose:
                print(f"Using lowest_angle_sum method")
            raise Exception("hasn't been fixed to make sure the upstream node is guaranteed to be in the output graph")
            G_final = xu.graph_to_lowest_weighted_sum_singular_matches(G,
            verbose = verbose,
            return_graph = True)
            
            
            winning_node = xu.get_neighbors(G_final,upstream_branch)
            if len(winning_node) != 1:
                raise Exception(f"Not just one winning node: {winning_node}")
            else:
                winning_node = winning_node[0]
        else:
            raise Exception(f"Unimplemented match_method : {match_method} ")


        error_branches = downstream_branches[downstream_branches!= winning_node]

        if verbose:
            print(f"for upstream node {upstream_branch}, winning_node = {winning_node}, error_branches = {error_branches}")

    
    """
    --- 5/12: Not having the short thick end nodes in the errors to remove
    """
    if remove_short_thick_endnodes:    
        #print(f"short_thick_endnodes_to_remove = {short_thick_endnodes_to_remove}")
        error_branches = np.setdiff1d(error_branches,short_thick_endnodes_to_remove)
    
    return winning_node,error_branches

'''



    
'''
def high_degree_branch_errors_limb_branch_dict_old(neuron_obj,
                                              **kwargs):

    """
    Purpose: To resolve high degree nodes for the axon branches
    and to return a limb branch dict of all of the errors

    Pseudocode: 
    1) Get the axon limb
    2) Find all of the high degree coordinates on the axon limb

    For each high degree coordinate
    a. Send the coordinate to the high_degree_upstream_match
    b. Get the error limbs back and if non empty then add to the limb branch dict

    return the limb branch dict

    """

    axon_name = neuron_obj.axon_limb_name
    if axon_name is None:
        return dict()
    limb_obj = neuron_obj[axon_name]

    short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
                                 verbose = False)
    
    
    
    exactly_equal = False
    crossover_coordinates = nru.high_degree_branching_coordinates_on_limb(limb_obj,min_degree_to_find=4,
                                                                             exactly_equal=exactly_equal,
                                                                             )
    limb_branch_dict = dict()
    error_branches = []
    for j,c in enumerate(crossover_coordinates):
        #if verbose:
        print(f"\n\n ----- Working on coordinate {j}: {c}--------")
        winning_downstream,error_downstream = ed.high_degree_upstream_match(limb_obj,
                              coordinate = c,
                            plot_intermediates = False,
                            plot_match_intermediates = False,
                            short_thick_endnodes_to_remove = short_thick_endnodes_to_remove,
                                                                           **kwargs)
        print(f"winning_downstream = {winning_downstream},error_downstream = {error_downstream} ")
        #if verbose:
        print(f"coordinate {c} had error branches {error_downstream}--------")
        if len(error_downstream) > 0:
            error_branches += list(error_downstream)

    if len(error_branches) > 0:
        limb_branch_dict[axon_name] = np.array(error_branches)

    return limb_branch_dict
    
'''




def thick_t_errors_limb_branch_dict(neuron_obj,
        axon_only = True,
        parent_width_maximum = 70,
        min_child_width_max = 78,#85,
        child_skeletal_threshold = 7000,
        plot_two_downstream_thin_axon_limb_branch = False,
        plot_wide_angled_children = False,
        plot_thick_t_crossing_limb_branch = False,
        plot_t_error_limb_branch = False,
        verbose = False):
    """
    Purpose: To generate a limb branch dict of the for branches
    where probably a thick axon has crossed a smaller axon

    Application: Will then be used to filter away in proofreading


    Pseudocode: 
    1) Find all of the thin axon branches with 2 downstream nodes
    2) Filter list down to those with 
    a) a high enough sibling angles
    b) high min child skeletal length
    c) min max child width

    ** those branches that pass that filter are where error occurs

    For all error branches
    i) find the downstream nodes
    ii) add the downstream nodes to the error branch list
    
    Example:
    ed.thick_t_errors_limb_branch_dict(filt_neuron,
                               plot_two_downstream_thin_axon_limb_branch = False,
                            plot_wide_angled_children = False,
                            plot_thick_t_crossing_limb_branch = False,
                            plot_t_error_limb_branch = True,
                            verbose = True)

    """
    wide_angle_T_thin_parent = au.wide_angle_t_candidates(neuron_obj,
                                axon_only = axon_only,
                                child_width_maximum = np.inf,
                                parent_width_maximum = parent_width_maximum,
                                plot_two_downstream_thin_axon_limb_branch = plot_two_downstream_thin_axon_limb_branch,
                                plot_wide_angled_children = plot_wide_angled_children,
                                child_skeletal_threshold = child_skeletal_threshold,
                                verbose = verbose)
    thick_t_crossing_limb_branch= ns.query_neuron(neuron_obj,
                   functions_list=["children_axon_width_max"],
                   query=f"children_axon_width_max > {min_child_width_max}",
                   limb_branch_dict_restriction = wide_angle_T_thin_parent)

    if verbose:
        print(f"thick_t_crossing_limb_branch= {thick_t_crossing_limb_branch}")

    if plot_thick_t_crossing_limb_branch:
        print(f"plotting plot_thick_t_crossing_limb_branch = {thick_t_crossing_limb_branch}")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  thick_t_crossing_limb_branch)

    """
    For all error branches
    i) find the downstream nodes
    ii) add the downstream nodes to the error branch list
    """

    t_error_limb_branch = dict()
    for limb_name,branch_list in thick_t_crossing_limb_branch.items():

        limb_obj = neuron_obj[limb_name]
        t_error_limb_branch[limb_name] = []
        for b in branch_list:
            downstream_nodes = xu.downstream_nodes(limb_obj.concept_network_directional,b)
            t_error_limb_branch[limb_name] += list(downstream_nodes)
        t_error_limb_branch[limb_name]  = np.array(t_error_limb_branch[limb_name])

    if verbose:
        print(f"t_error_limb_branch= {t_error_limb_branch}")

    if plot_t_error_limb_branch:
        print(f"plotting plot_t_error_limb_branch")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  t_error_limb_branch)

    return t_error_limb_branch


def cut_kissing_graph_edges(G,
                            limb_obj,
                            coordinate,
    kiss_check_bbox_longest_side_threshold = 450,
    offset=1500,
    comparison_distance = 2000,
    only_process_partitions_with_valid_edges = True,
    plot_offset_skeletons = False,
    plot_source_sink_vertices = False,
    plot_cut_vertices = False,
    plot_cut_bbox = False,
    verbose = False,
                            
                ):
    """
    Purpose: To remove edges
    in a connectivity graph that 
    are between nodes that come from
    low mesh bridging (usually due to merge errors)

    Pseudocode:
    1) Get the mesh intersection
    2) Get the offset skeletons and the endpoints
    3) Find all possible partions of the branch 
    """

    branches = np.array(list(G.nodes()))
    upstream_branch = branches[0]

    # ---- 4/30 Change: Want all meshes at that coordinate so doesn't disconnect the mesh --- #
#     mesh_inter = nru.branches_combined_mesh(limb_obj,branches,
#                               plot_mesh=False)
    branches_at_coord = nru.find_branch_with_specific_coordinate(limb_obj,coordinate)
    mesh_inter = nru.branches_combined_mesh(limb_obj,branches_at_coord,
                              plot_mesh=False)
    

    offset_skeletons,skeleton_offset_points = nru.coordinate_to_offset_skeletons(limb_obj,
                                      coordinate=coordinate,
                                       branches= branches,
                                       offset=offset,
                                        comparison_distance = comparison_distance,
                                       plot_offset_skeletons=plot_offset_skeletons,
                                       verbose = verbose,
                                       return_skeleton_endpoints = True,
                                      )

    skeleton_offset_points_dict = {b:sk_o for b,sk_o in zip(branches,skeleton_offset_points)}
    if verbose:
        print(f"skeleton_offset_points_dict = {skeleton_offset_points_dict}")

    if verbose:
        print(f"node_partitions")    
    node_partitions = nu.all_partitions(branches,verbose=verbose)

    for j,partition in enumerate(node_partitions):
        part1,part2 = partition
        if verbose:
            print(f"Working on partition {j}: {part1},{part2}")
            
        if only_process_partitions_with_valid_edges:
            """
            Pseudocode: check that there is at least one valid edge
            in each of the partitions, or else continue
            
            
            """
            skip_partition = False
            for p in partition:
                part_G = G.subgraph(p)
                if len(part_G.edges()) == 0:
                    if verbose:
                        print(f"Skipping Partition {j}: {partition} because there was not a valid edge in {p}")
                    skip_partition = True
            
            if skip_partition:
                continue
            
            else:
                if verbose:
                    print(f"Continuing with Partition because valid edge")
        
        source_coordinates = [skeleton_offset_points_dict[k] for k in part1]
        sink_coordinates = [skeleton_offset_points_dict[k] for k in part2]

        cut_coordinates = tu.min_cut_to_partition_mesh_vertices(mesh_inter,
                                              source_coordinates,
                                              sink_coordinates,
                                               plot_source_sink_vertices= plot_source_sink_vertices,
                                              verbose = verbose,
                                              return_edge_midpoint = True,
                                                        plot_cut_vertices = plot_cut_vertices)

    #     if verbose:
    #         print(f"cut_coordinates = {cut_coordinates}")


        if cut_coordinates is None:
            if verbose:
                print(f"Output was None so continuing")
            continue


        cut_bbox = tu.coordinates_to_bounding_box(cut_coordinates)
        cut_bbox_volume = cut_bbox.volume
        cut_bbox_longest_side = tu.bounding_box_longest_side(cut_bbox)

        if verbose:
            print(f"cut_bbox_volume = {cut_bbox_volume}, cut_bbox_longest_side = {cut_bbox_longest_side}")

        if plot_cut_bbox:
            nviz.plot_objects(main_mesh=mesh_inter,
                              meshes=[cut_bbox],
                              meshes_colors="red",
                              skeletons=[limb_obj[k].skeleton for k in branches],
                              skeletons_colors="random"
                )

        # apply the kiss threshold o see if edges should be cut
        if cut_bbox_longest_side < kiss_check_bbox_longest_side_threshold:
            #if verbose:
            print(f"** triggered kiss check cut becuase cut_bbox_longest_side = {cut_bbox_longest_side}***")
            #then delete the edges that are not contained within the partitions
            G = xu.remove_inter_partition_edges(G,
                                                partition,
                                                verbose = False)
            if verbose:
                print(f"Edges after removing partition = {xu.get_edges_with_weights(G)}")

    if verbose:
        print(f"\n\n----Final Edges After Kissing Processing = {xu.get_edges_with_weights(G)}")
    return G

# ------ Rule 7: Width Jump and Double Back Revision ----------- #
def width_jump_from_upstream_min(limb_obj,
                      branch_idx,
                      skeletal_length_min = 2000,
                       verbose = False,
                        **kwargs):
    """
    Purpose: To Find the width jump up
    of a current branch from those 
    proceeding it

    Pseudocode: 
    1) Find the minimum proceeding width
    2) Find the current width
    3) Subtract and Return
    
    Ex: 
    from neurd import error_detection as ed
    ed.width_jump_from_upstream_min(limb_obj=neuron_obj[0],
    branch_idx=318,
    skeletal_length_min = 2000,
    verbose = False)

    """

    min_upstream_width = nru.min_width_upstream(limb_obj,
                          branch_idx,
                          skeletal_length_min = skeletal_length_min,
                            verbose = verbose)

    current_width = nru.width(limb_obj[branch_idx])
    width_jump = current_width - min_upstream_width
    
    if verbose:
        print(f"min_upstream_width = {min_upstream_width}")
        print(f"current_width = {current_width}")
        print(f"width_jump = {width_jump}")
    return width_jump


def width_jump_up_error_limb_branch_dict(
    neuron_obj,
    limb_branch_dict_restriction=None,
    upstream_skeletal_length_min = 10000,
    branch_skeletal_length_min = 6000,
    upstream_skeletal_length_min_for_min= 4000,
    width_jump_max = 75,
    plot_final_width_jump = False,
    verbose = False,
    ignore_large_skeleton_endpoint_jump = False,
    max_skeleton_endpoint_jump = None,
    **kwargs
    ):


    """
    Purpose: To find all branches
    that hae a jump up of width 
    from the minimum of the upsream widths 
    (that are indicative of an error)

    Pseudocode: 
    0) Given a starting limb branch dict
    1) Query the neuron for those branhes
    that have a certain upstream width and have
    a certain skeletal width
    2) Query the neuron for those
    with a width jump above a certain amount
    3) Graph the query 

    """
    if verbose:
        print(f"Before any restrictions in width jump, limb_branch_dict_restriction = {limb_branch_dict_restriction}")
    limb_branch_dict_restriction = ns.restrict_by_branch_and_upstream_skeletal_length(neuron_obj, 
                                                limb_branch_dict_restriction=limb_branch_dict_restriction,
                                                upstream_skeletal_length_min = upstream_skeletal_length_min,
                                                      branch_skeletal_length_min=branch_skeletal_length_min,
                                                     **kwargs)
    

    if verbose:
        print(f"After skeletal restrictions, limb_branch_dict_restriction = {limb_branch_dict_restriction}")

    if len(limb_branch_dict_restriction) == 0:
        return limb_branch_dict_restriction

    
    functions_list=["width_jump_from_upstream_min"]
    query=f"(width_jump_from_upstream_min>{width_jump_max})"
    
    if ignore_large_skeleton_endpoint_jump:
        query += f" and (max_skeleton_endpoint_dist < {max_skeleton_endpoint_jump})"
        functions_list.append("max_skeleton_endpoint_dist")
    
    
    width_jump_limb_branch = ns.query_neuron(neuron_obj,
                    functions_list=functions_list,
                    query=query,
                    function_kwargs=dict(skeletal_length_min=upstream_skeletal_length_min_for_min),
                    return_dataframe=False,
            limb_branch_dict_restriction=limb_branch_dict_restriction)

    if plot_final_width_jump:
        print(f"width_jump_limb_branch (WITH threshold {width_jump_from_upstream_min}) = {width_jump_limb_branch}")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  width_jump_limb_branch)
        
    return width_jump_limb_branch


def width_jump_up_axon(
    neuron_obj,
    upstream_skeletal_length_min = None,#5000,
    branch_skeletal_length_min = None,#8000,
    upstream_skeletal_length_min_for_min = None,#4000,
    width_jump_max = None,#55,
    axon_width_threshold_max = None,#au.axon_thick_threshold,
    plot_width_errors = False,
    **kwargs):
    """
    Purpose: To apply the width 
    jump up check on the axon segments of neuron

    Pseudocode: 
    0) Set the width parameters corectly for axon
    1) Find all of the axon branches
    2) Run the width jump check
    """
    from neurd import limb_utils as lu
    if upstream_skeletal_length_min is None:
        upstream_skeletal_length_min = upstream_skeletal_length_min_width_j_axon_global
    if branch_skeletal_length_min is None:
        branch_skeletal_length_min = branch_skeletal_length_min_width_j_axon_global
    if upstream_skeletal_length_min_for_min is None:
        upstream_skeletal_length_min_for_min = upstream_skeletal_length_min_for_min_width_j_axon_global
    if width_jump_max is None:
        width_jump_max = width_jump_max_width_j_axon_global
    if axon_width_threshold_max is None:
        axon_width_threshold_max = axon_width_threshold_max_width_j_axon_global
    


    axon_limb_branch = ns.query_neuron_by_labels(neuron_obj,
                                                            matching_labels=["axon"])
    
    axon_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list=[
                       #"axon_width",
                       lu.width_upstream_limb_ns,
                   ],
                   query=(
                       #f"axon_width <= {axon_width_threshold_max}"
                       f"width_upstream <= {axon_width_threshold_max}"
                   ),
                   limb_branch_dict_restriction=axon_limb_branch)
    

    width_errors = ed.width_jump_up_error_limb_branch_dict(neuron_obj,
                                         limb_branch_dict_restriction = axon_limb_branch,
                                         upstream_skeletal_length_min = upstream_skeletal_length_min,
                                         branch_skeletal_length_min = branch_skeletal_length_min,
                                         upstream_skeletal_length_min_for_min= upstream_skeletal_length_min_for_min,
                                        width_jump_max = width_jump_max,
                                                          **kwargs)
    
    if plot_width_errors:
        if len(width_errors) == 0:
            print("No Width Errors To Plot")
        else:
            print(f"width_errors = {width_errors}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      width_errors)
    return width_errors


#dendrite_trunk_width = 500

def dendrite_branch_restriction(neuron_obj,
                               width_max = None,#dendrite_trunk_width,
                                upstream_skeletal_length_min = None,#5000,
                                plot = False,
                                verbose= False
                               ):
    from neurd import limb_utils as lu
    
    if width_max is None:
        width_max = width_max_dendr_restr_global
    if upstream_skeletal_length_min is None:
        upstream_skeletal_length_min = upstream_skeletal_length_min_dendr_restr_global
    
    current_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                                                        not_matching_labels=["axon"])
    
    if verbose:
        print(f"Before any querying started: {current_limb_branch_dict}")
    
    
    query=(
                       #f"(width_neuron <= {width_max}) "
                       f"(width_upstream <= {width_max})"
                       f"and (total_upstream_skeletal_length>{upstream_skeletal_length_min})"
                       )
    
    
    current_limb_branch_dict = ns.query_neuron(neuron_obj,
                   functions_list=[
                       #"width_neuron",
                       lu.width_upstream_limb_ns,
                       "total_upstream_skeletal_length"],
                   query=query,
                   limb_branch_dict_restriction=current_limb_branch_dict)
    
    if verbose:
        print(f"query = {query}")
        print(f"current_limb_branch_dict = {current_limb_branch_dict}")
        
    if plot:
        #print(f"current_limb_branch_dict = {current_limb_branch_dict}")
        if len(current_limb_branch_dict) > 0:
            nviz.plot_limb_branch_dict(neuron_obj,current_limb_branch_dict)
    
    return current_limb_branch_dict



def width_jump_up_dendrite(
    neuron_obj,
    upstream_skeletal_length_min = None,#5000,
    branch_skeletal_length_min = None,#7000,
    upstream_skeletal_length_min_for_min = None,#4000,
    width_jump_max = None,#200,
    plot_width_errors = False,
    ignore_large_skeleton_endpoint_jump = None,
    max_skeleton_endpoint_jump = None,
                          **kwargs):
    """
    Purpose: To apply the width 
    jump up check on the axon segments of neuron

    Pseudocode: 
    0) Set the width parameters corectly for axon
    1) Find all of the axon branches
    2) Run the width jump check
    """
    
    if upstream_skeletal_length_min is None:
        upstream_skeletal_length_min = upstream_skeletal_length_min_width_j_dendr_global
    if branch_skeletal_length_min is None:
        branch_skeletal_length_min = branch_skeletal_length_min_width_j_dendr_global
    if upstream_skeletal_length_min_for_min is None:
        upstream_skeletal_length_min_for_min = upstream_skeletal_length_min_for_min_width_j_dendr_global
    if width_jump_max is None:
        width_jump_max = width_jump_max_width_j_dendr_global
    if ignore_large_skeleton_endpoint_jump is None:
        ignore_large_skeleton_endpoint_jump = ignore_large_skeleton_endpoint_jump_global
    if max_skeleton_endpoint_jump is None:
        max_skeleton_endpoint_jump = max_skeleton_endpoint_jump_global



    dendrite_limb_branch = ed.dendrite_branch_restriction(neuron_obj)
    #print(f"dendrite_limb_branch = {dendrite_limb_branch}")
    
    width_errors = ed.width_jump_up_error_limb_branch_dict(neuron_obj,
                                         limb_branch_dict_restriction = dendrite_limb_branch,
                                         upstream_skeletal_length_min = upstream_skeletal_length_min,
                                         branch_skeletal_length_min = branch_skeletal_length_min,
                                         upstream_skeletal_length_min_for_min= upstream_skeletal_length_min_for_min,
                                        width_jump_max = width_jump_max,
                                        ignore_large_skeleton_endpoint_jump=ignore_large_skeleton_endpoint_jump,
                                        max_skeleton_endpoint_jump=max_skeleton_endpoint_jump,
                                        **kwargs)
    
    if plot_width_errors:
        if len(width_errors) == 0:
            print("No Width Errors To Plot")
        else:
            print(f"width_errors = {width_errors}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      width_errors)
    return width_errors


# -------------- Doubling Back Errors ---------------------- #


def double_back_error_limb_branch_dict(
    neuron_obj,
    double_back_threshold=120,
    branch_skeletal_length_min=4000,

    limb_branch_dict_restriction=None,
    upstream_skeletal_length_min = 5000,

    comparison_distance = 3000,
    offset = 0,
    plot_final_double_back = False,
    verbose = False,
    
    angle_func_type = None,
    skeleton_attribute="skeleton",
    **kwargs):
    """
    Purpose: To find all branches that have a skeleton that
    doubles back by a certain degree
    
    Pseudocode: 
    0)

    """
    
    if angle_func_type is None:
        angle_func_type = double_back_angle_func_type_global
    
    limb_branch_dict_restriction = ns.restrict_by_branch_and_upstream_skeletal_length(neuron_obj, 
                                                limb_branch_dict_restriction=limb_branch_dict_restriction,
                                                upstream_skeletal_length_min = upstream_skeletal_length_min,
                                                      branch_skeletal_length_min=branch_skeletal_length_min,
                                                     **kwargs)

    if verbose:
        print(f"After skeletal restrictions, limb_branch_dict_restriction = {limb_branch_dict_restriction}")

    if len(limb_branch_dict_restriction) == 0:
        return limb_branch_dict_restriction

    #print(f"inside double_back_error_limb_branch_dict")
    double_back_limb_branch_dict = ns.query_neuron(neuron_obj,
                    functions_list=["parent_angle"],
                    query=f"parent_angle>{double_back_threshold}",
                    function_kwargs=dict(comparison_distance=comparison_distance,
                                         angle_func_type=angle_func_type,
                                        offset=offset,
                                        check_upstream_network_connectivity=False,
                                        skeleton_attribute = skeleton_attribute),
                    return_dataframe=False,
            limb_branch_dict_restriction=limb_branch_dict_restriction)

    if plot_final_double_back:
        print(f"double_back_limb_branch_dict (WITH threshold {double_back_threshold}) = {double_back_limb_branch_dict}")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  double_back_limb_branch_dict)
    return double_back_limb_branch_dict


def double_back_dendrite(
    neuron_obj,
    double_back_threshold=None,#120,
    comparison_distance = None,#3000,
    offset = None,#0,
    branch_skeletal_length_min = None,#7000, #deciding which branches will be skipped because of length
    width_max = None,
    plot_starting_limb_branch = False,
    plot_double_back_errors = False,
    skeleton_attribute = "skeleton_smooth",
    **kwargs
    ):
    """
    Purpose: To find all skeletal double 
    back errors on dendrite port
    
    
    """
    if double_back_threshold is None:
        double_back_threshold = double_back_threshold_double_b_dendrite_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_double_b_dendrite_global
    if offset is None:
        offset = offset_double_b_dendrite_global
    if branch_skeletal_length_min is None:
        branch_skeletal_length_min = branch_skeletal_length_min_double_b_dendrite_global
    if width_max is None:
        width_max = width_max_dendr_double_back_restr_global
    
    current_limb_branch_dict = ed.dendrite_branch_restriction(neuron_obj,width_max=width_max)
    
    if plot_starting_limb_branch:
        if len(current_limb_branch_dict) == 0:
            print("No limb_branch_dict To Plot")
        else:
            print(f"current_limb_branch_dict = {current_limb_branch_dict}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      current_limb_branch_dict)
                         
    double_back_errors = ed.double_back_error_limb_branch_dict(neuron_obj,
                                          double_back_threshold=double_back_threshold,
                                      limb_branch_dict_restriction=current_limb_branch_dict,
                                      plot_final_double_back=False,
                                         comparison_distance = comparison_distance,
                                           offset = offset,
                                        branch_skeletal_length_min=branch_skeletal_length_min,
                                        skeleton_attribute=skeleton_attribute,
                                                              **kwargs)
    
    if plot_double_back_errors:
        if len(double_back_errors) == 0:
            print("No Double Back Errors To Plot")
        else:
            print(f"double_back_errors = {double_back_errors}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      double_back_errors)
    
    return double_back_errors


def double_back_axon_thin(neuron_obj,
                          axon_width_threshold = None,
                        double_back_threshold=135,
                         comparison_distance = 1000,
                         offset = 0,
                         branch_skeletal_length_min = 4000, #deciding which branches will be skipped because of length
                         plot_starting_limb_branch = False,
                         plot_double_back_errors = False,
                         **kwargs
                        ):
    """
    Purpose: To find all skeletal double 
    back errors on dendrite port

    """
    if axon_width_threshold is None:
        axon_width_threshold = au.axon_thick_threshold
        
    
    current_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                                                        matching_labels=["axon"])
    
    current_limb_branch_dict = ns.query_neuron(neuron_obj,
                   functions_list=["axon_width"],
                   query=f"axon_width <= {axon_width_threshold}",
                   limb_branch_dict_restriction=current_limb_branch_dict)
    
    if plot_starting_limb_branch:
        if len(current_limb_branch_dict) == 0:
            print("No limb_branch_dict To Plot")
        else:
            print(f"current_limb_branch_dict = {current_limb_branch_dict}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      current_limb_branch_dict)
                         
    double_back_errors = ed.double_back_error_limb_branch_dict(neuron_obj,
                                          double_back_threshold=double_back_threshold,
                                      limb_branch_dict_restriction=current_limb_branch_dict,
                                      plot_final_double_back=False,
                                         comparison_distance = comparison_distance,
                                           offset = offset,
                                        branch_skeletal_length_min=branch_skeletal_length_min,
                                                              **kwargs)
    
    if plot_double_back_errors:
        if len(double_back_errors) == 0:
            print("No Double Back Errors To Plot")
        else:
            print(f"double_back_errors = {double_back_errors}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      double_back_errors)
    
    return double_back_errors


def double_back_axon_thick(neuron_obj,
                          axon_width_threshold = None,
                           axon_width_threshold_max = None,
                        double_back_threshold=120,
                         comparison_distance = 1000,
                         offset = 0,
                         branch_skeletal_length_min = 4000, #deciding which branches will be skipped because of length
                         plot_starting_limb_branch = False,
                         plot_double_back_errors = False,
                         **kwargs
                        ):
    """
    Purpose: To find all skeletal double 
    back errors on dendrite port

    """
    if axon_width_threshold is None:
        axon_width_threshold = au.axon_thick_threshold
        
    if axon_width_threshold_max is None:
        axon_width_threshold_max = au.axon_ais_threshold
        
    
    current_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                                                        matching_labels=["axon"])
    
    current_limb_branch_dict = ns.query_neuron(neuron_obj,
                   functions_list=["axon_width"],
                   query=f"(axon_width > {axon_width_threshold}) and (axon_width < {axon_width_threshold_max})",
                   limb_branch_dict_restriction=current_limb_branch_dict)
    
    if plot_starting_limb_branch:
        if len(current_limb_branch_dict) == 0:
            print("No limb_branch_dict To Plot")
        else:
            print(f"current_limb_branch_dict = {current_limb_branch_dict}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      current_limb_branch_dict)
                         
    double_back_errors = ed.double_back_error_limb_branch_dict(neuron_obj,
                                          double_back_threshold=double_back_threshold,
                                      limb_branch_dict_restriction=current_limb_branch_dict,
                                      plot_final_double_back=False,
                                         comparison_distance = comparison_distance,
                                           offset = offset,
                                        branch_skeletal_length_min=branch_skeletal_length_min,
                                                              **kwargs)
    
    if plot_double_back_errors:
        if len(double_back_errors) == 0:
            print("No Double Back Errors To Plot")
        else:
            print(f"double_back_errors = {double_back_errors}")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      double_back_errors)
    
    return double_back_errors

# -------------- 6/21: Version 6 Erro Detection Rules --------- #


# --------- High degree branching -------------- #



def calculate_skip_distance(limb_obj,
                            branch_idx,
                            calculate_skip_distance_including_downstream = True,
                           verbose = False):
    if calculate_skip_distance_including_downstream:
        #1) Get all downstream branhes (with an optional skip distance)
        downstream_branches = nru.downstream_nodes(limb_obj,branch_idx)
        all_nodes = list(downstream_branches) + [branch_idx]
#             print(f"all_nodes = {all_nodes}")
        curr_width = [au.axon_width(limb_obj[k]) for k in all_nodes]
        skip_distance = [ed.skip_distance_from_branch_width(k) for k in curr_width]
#             print(f"curr_width = {curr_width}")
#             print(f"skip_distance = {skip_distance}")
        skip_distance = np.max(skip_distance)
        #if verbose:
        print(f"Current node skip distance was {ed.skip_distance_from_branch_width(au.axon_width(limb_obj[branch_idx]))} but max skip distance was {skip_distance}")
    else:
        curr_width = au.axon_width(limb_obj[branch_idx])
        skip_distance = ed.skip_distance_from_branch_width(curr_width)
        
    if verbose:
        print(f"For {branch_idx} the skip distance was {skip_distance} (for width {curr_width})")

    return skip_distance


def high_low_degree_upstream_match_preprocessing(
    limb_obj,
    branch_idx,
    
    #arguments for determining downstream nodes
    skip_distance = None,
    min_upstream_skeletal_distance = None,
    min_distance_from_soma_for_proof = None,
    short_thick_endnodes_to_remove = None,
    axon_spines = None,
    min_degree_to_resolve = None,# 3,
    
    # helps determine the max degrees to resolve
    width_func = None,
    max_degree_to_resolve_absolute = None,#1000,
    max_degree_to_resolve = None,#1000,
    max_degree_to_resolve_wide = None,#1000,
    max_degree_to_resolve_width_threshold = None,#200,
    skip_greater_than_max_degree_to_resolve = False,
    
    
    # parameter checking to see if high degree resolve can be used
    width_min = None,#35,
    width_max = None,#170,
    upstream_width_max = None,#None,
    axon_dependent = None,#True, 

    #arguments for what to return
    return_skip_info=True,
    verbose = False,
    ):
    
    """
    Purpose: 
    To take a node on a limb and determine
    a) if the node should even be processed (and if it shouldn't what is the return value)
    b) What the downstream nodes to be processed should be
    c) What the skip distance and skip nodes are
    
    What want to return: 
    - return value
    - skip distance
    - skipped_nodes
    - downstream_branches
    
    Pseudocode: 
    1) Calulate the skip distance
    
    """
    #verbose = True
    # ----- setting the parameters --------
    if min_distance_from_soma_for_proof is None:
        min_distance_from_soma_for_proof = min_distance_from_soma_for_proof_global
        
    if min_degree_to_resolve is None:
        min_degree_to_resolve = min_degree_to_resolve_global
    if max_degree_to_resolve_absolute is None:
        max_degree_to_resolve_absolute = max_degree_to_resolve_absolute_global
    if max_degree_to_resolve is None:
        max_degree_to_resolve = max_degree_to_resolve_global
    if max_degree_to_resolve_wide is None:
        max_degree_to_resolve_wide = max_degree_to_resolve_wide_global
    if max_degree_to_resolve_width_threshold is None:
        max_degree_to_resolve_width_threshold = max_degree_to_resolve_width_threshold_global
    if width_min is None:
        width_min = width_min_global
    if width_max is None:
        width_max = width_max_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_global
    if axon_dependent is None:
        axon_dependent = axon_dependent_global
    if width_func is None:
        width_func = au.axon_width

    return_value = []
    downstream_branches = None
    skipped_nodes = None
    
    nodes_to_exclude = []
    
    if short_thick_endnodes_to_remove is not None:
        nodes_to_exclude += list(short_thick_endnodes_to_remove)
        if branch_idx in short_thick_endnodes_to_remove:
            if verbose:
                print(f"Skipping because in short_thick_endnodes_to_remove")
            return_value = [None,np.array([])]

    if axon_spines is not None:
        nodes_to_exclude += list(axon_spines)
        if branch_idx in axon_spines:
            if verbose:
                print(f"Skipping because in axon_spines")
            return_value = [None,np.array([])]
            
    
    
    if min_upstream_skeletal_distance is not None:
        #curr_sk_len = limb_obj[branch_idx].skeletal_length
        curr_sk_len = cnu.skeletal_length_upstream(limb_obj,branch_idx,nodes_to_exclude=nodes_to_exclude)
        if  curr_sk_len < min_upstream_skeletal_distance:
            if verbose:
                print(f"Skipping because skeletal length ({curr_sk_len}) was less than min_upstream_skeletal_distance = {min_upstream_skeletal_distance}")
            return_value = [None,np.array([])]
            
    if min_distance_from_soma_for_proof is not None:
        dist = nst.distance_from_soma(limb_obj,branch_idx,include_node_skeleton_dist=True)
        if dist < min_distance_from_soma_for_proof:
            if verbose:
                print(f"Skipping because distance away from soma ({dist}) was less than min_distance_from_soma_for_proof = {min_distance_from_soma_for_proof}")
            return_value = [None,np.array([])]

    # # check if any downstream branches at all, and if not then return
    # if len(nru.downstream_nodes(limb_obj,branch_idx)) == 0:
    #     return_value = [None,np.array([])]
                
    
    if len(return_value) > 0:
        if return_skip_info:
            return return_value,downstream_branches,skip_distance,skipped_nodes
        else:
            return return_value,downstream_branches
                            
    
    
    if skip_distance is None:
        #print(f'Calculating skip distance')
        skip_distance = ed.calculate_skip_distance(limb_obj,
                                                  branch_idx)
    # ----- Phase A: Preprocessing before matching -----------
    
    
    #1) Get all downstream branhes (with an optional skip distance)
    downstream_branches = cnu.endnode_branches_of_branches_within_distance_downtream(limb_obj,
                                                              branch_idx,
                                                              skip_distance=skip_distance)
    
    all_downstream_branches = cnu.branches_within_distance_downstream(limb_obj,
                                                                     branch_idx,
                                                                     distance_threshold=skip_distance)
    skipped_nodes = np.setdiff1d(all_downstream_branches,downstream_branches)

    if verbose:
        print(f"downstream_branches = {downstream_branches}")
        print(f"skipped_nodes = {skipped_nodes}")

        
    #2) Remove short thick endnodes from possible branches in the high degree point
    if short_thick_endnodes_to_remove is not None:
        downstream_branches = np.setdiff1d(downstream_branches,short_thick_endnodes_to_remove)

        if verbose:
            print(f"Total number of short_thick_endnodes_to_remove = {len(short_thick_endnodes_to_remove)}")
            print(f"downstream_branches after remove_short_thick_endnodes = {downstream_branches}")
            
    if axon_spines is not None:
        downstream_branches = np.setdiff1d(downstream_branches,axon_spines)
        if verbose:
            print(f"Total number of axon_spines = {len(axon_spines)}")
            print(f"downstream_branches after remove_short_thick_endnodes = {downstream_branches}")

    #3) Return if not enough branches at the intersection
    if len(downstream_branches) < min_degree_to_resolve:
        if verbose:
            print(f"Number of branches ({len(downstream_branches)}), aka downstream_branches = {downstream_branches}) was less than min_degree_to_resolve ({min_degree_to_resolve}) so returning no error branches")
        return_value =  [None,np.array([])]
        
    
    if len(return_value) > 0:
        if return_skip_info:
            return return_value,downstream_branches,skip_distance,skipped_nodes
        else:
            return return_value,downstream_branches
    
    # -------- 8/1: Sets a limit on the maximum branch degree to be resolved ---------
    upstream_branch = branch_idx
    
    #print(f"!!!!inside ed. skip_greater_than_max_degree_to_resolve = {skip_greater_than_max_degree_to_resolve}")
    if max_degree_to_resolve_absolute is not None and len(downstream_branches) > max_degree_to_resolve_absolute:
        if verbose:
            print(f"Number of branches ({len(downstream_branches)}) was more than max_degree_to_resolve ({max_degree_to_resolve_absolute}) so returning all downstream as error branches")
        if skip_greater_than_max_degree_to_resolve:
            return_value =  [None,[]]
        else:
            return_value= [None,downstream_branches]
    
    if len(return_value) > 0:
        if return_skip_info:
            return return_value,downstream_branches,skip_distance,skipped_nodes
        else:
            return return_value,downstream_branches 
       
    # 4) If the branch being considered is thick enough then increase the max degree to resolve
    
    
    if upstream_width_max is not None:
        upstream_w = cnu.width_downstream(limb_obj,upstream_branch,
                                                        width_func=width_func,
                                                        nodes_to_exclude=nodes_to_exclude)
        if upstream_w > upstream_width_max:
            if verbose:
                print(f"Returning No errors because upstream width ({upstream_w}) is greaeter than the upstream_width_max {upstream_width_max}")
            return_value =  [None,[]]
    
    
    if len(return_value) > 0:
        if return_skip_info:
            return return_value,downstream_branches,skip_distance,skipped_nodes
        else:
            return return_value,downstream_branches 
        
    # ----------- Returning if the upstream widht is too great ------------- #
    
    if max_degree_to_resolve_wide is not None:
        up_width = width_func(limb_obj[upstream_branch])
        if up_width > max_degree_to_resolve_width_threshold:
            max_degree_to_resolve = max_degree_to_resolve_wide
            if verbose:
                print(f"Changing max_degree_to_resolve = {max_degree_to_resolve_wide} because upstream width was {up_width} ")

    #5) Return all downstream branches as errors if number of branches at intersection is too large
    if max_degree_to_resolve is not None and len(downstream_branches) > max_degree_to_resolve:
        if verbose:
            print(f"Number of branches ({len(downstream_branches)}) was more than max_degree_to_resolve ({max_degree_to_resolve}) so returning all downstream as error branches")
        if skip_greater_than_max_degree_to_resolve:
            return_value =  [None,[]]
        else:
            return_value= [None,downstream_branches]

        
    if len(return_value) > 0:
        if return_skip_info:
            return return_value,downstream_branches,skip_distance,skipped_nodes
        else:
            return return_value,downstream_branches    
    

    all_branch_idx = np.hstack([downstream_branches,[upstream_branch]])

    
    #widths_in_branches = np.array([width_func(limb_obj[b]) for b in all_branch_idx])
    widths_in_branches = np.array([cnu.width_downstream(limb_obj,b,
                                                        width_func=width_func,
                                                        nodes_to_exclude=nodes_to_exclude)
                                   for b in all_branch_idx])
    if verbose:
        print(f"widths_in_branches = {widths_in_branches}")

    widths_in_branches = widths_in_branches[widths_in_branches != 0]

    # 6) Do not process the intersection if all the branches are thick or not all are axons (return no errors)
    if width_max is not None:
        if len(widths_in_branches[widths_in_branches>width_max]) == len(widths_in_branches):
            if verbose:
                print(f"Returning No errors because widths are too thick for skeletons to be trusted")
            return_value =  [None,[]]

    if axon_dependent:
        for b in downstream_branches:
            if "axon" not in limb_obj[b].labels:
                if verbose:
                    print(f"Returning No errors because not all branches were axons")
                return_value = [None,[]]
                
    
    upstream_width = au.axon_width(limb_obj[branch_idx])
    if upstream_width < width_min:
        if verbose:
            print(f"Upstream width is too small (under {width_min}) so not processing")
        return_value =  [None,np.array([])]
    
    
    if return_skip_info:
        return return_value,downstream_branches,skip_distance,skipped_nodes
    else:
        return return_value,downstream_branches
    


# def high_degree_upstream_match_old_2(
#     limb_obj,
#     branch_idx,
    
#     #--- Phase A: arguments for determining downstream nodes ------
#     skip_distance = None,#3000,
#     min_upstream_skeletal_distance = None,
#     remove_short_thick_endnodes = True,
#     axon_spines = None,
    
#     short_thick_endnodes_to_remove = None,
#     min_degree_to_resolve = 3,
    
#     # helps determine the max degrees to resolve
#     width_func = au.axon_width,
#     max_degree_to_resolve_absolute = 1000,
#     max_degree_to_resolve = 1000,
#     max_degree_to_resolve_wide = 1000,
#     max_degree_to_resolve_width_threshold = 200,
    
#     # parameter checking to see if high degree resolve can be used
#     width_max = 170,
#     axon_dependent = True,
    
#     plot_starting_branches = False,

#     # --- Phase B.1: parameters for local edge attributes ------
#     offset=1500,
#     comparison_distance = 2000,
#     plot_extracted_skeletons = False,
    
    
#     # --- Phase B.2: parameters for local edge query ------
    
#     worst_case_sk_angle_match_threshold = 65,
    
#     width_diff_max = 75,#np.inf,100,
#     width_diff_perc = 0.60,
    
#     perform_synapse_filter = True,
#     synapse_density_diff_threshold = 0.00015, #was 0.00021
#     n_synapses_diff_threshold = 6,
    
#     plot_G_local_edge = False,
    
#     # ----- Phase B.3: parameters for global attributes ---
#     #args for definite pairs
#     sk_angle_match_threshold = 45,
#     sk_angle_buffer = 15,
    
#     width_diff_perc_threshold = 0.15,
#     width_diff_perc_buffer = 0.30,

#     # ----- Phase B.4 paraeters for global query ---
#     plot_G_global_edge = False,
    
#     # ----- Phase B.6 paraeters for ndoe query ---
#     plot_G_node_edge = False,
    
#     # ---- Phase C: Optional Kiss filter ----
#     kiss_check = False,
#     kiss_check_bbox_longest_side_threshold = 450,

#     # ---- Phase D: Picking the final winner -----
#     plot_final_branch_matches = False,
#     match_method = "all_error_if_not_one_match",# "best_match", #other option is "best_match"
#     use_exclusive_partner = True,
#     plot_G_exclusive_partner_edge = False,
    
#     verbose = False,
#     ):
#     #print(f"perform_synapse_filter = {perform_synapse_filter}")
#     """
#     Purpose: To Determine if branches downstream from a certain
#     branch should be errored out based on crossovers and 
#     high degree branching downstream
    
#     Pseudocode: 
#     Phase A:
#     #1) Get all downstream branhes (with an optional skip distance)
#     #2) Remove short thick endnodes from possible branches in the high degree point
#     #3) Return if not enough branches at the intersection
#     #4) If the branch being considered is thick enough then increase the max degree to resolve
#     #5) Return all downstream branches as errors if number of branches at intersection is too large
#     #6) Do not process the intersection if all the branches are thick or not all are axons (return no errors)
    
#     Phase B: 
#     #1) Compute features of a complete graph that connets all upsream and downsream edges
#     #(slightly different computation for upstream than downstream edges)
#     """
# #     if branch_idx == 13:
# #         verbose = True
#     plot_G_local_edge = True
#     plot_G_global_edge = True
#     plot_G_node_edge= True

#     if remove_short_thick_endnodes:
#         if short_thick_endnodes_to_remove is None:
#             short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
#                                  verbose = False)
            
#         limb_obj.short_thick_endnodes = short_thick_endnodes_to_remove
    
#     if axon_spines is not None:
#         limb_obj.axon_spines = axon_spines
            
#     # ---------- Phase A: Figure out if branch needs to be processed at all (and if so compute the downstream branches ---
#     (return_value,
#     downstream_branches,
#     skip_distance,
#     skipped_nodes) = high_low_degree_upstream_match_preprocessing(
#                         limb_obj,
#                         branch_idx,

#                         #arguments for determining downstream nodes
#                         skip_distance = skip_distance,
#                         min_upstream_skeletal_distance = min_upstream_skeletal_distance,
#                         short_thick_endnodes_to_remove = limb_obj.short_thick_endnodes,
#                         axon_spines = limb_obj.axon_spines,
#                         min_degree_to_resolve = min_degree_to_resolve,

#                         # helps determine the max degrees to resolve
#                         width_func = width_func,
#                         max_degree_to_resolve_absolute = max_degree_to_resolve_absolute,
#                         max_degree_to_resolve = max_degree_to_resolve,
#                         max_degree_to_resolve_wide = max_degree_to_resolve_wide,
#                         max_degree_to_resolve_width_threshold = max_degree_to_resolve_width_threshold,

#                         # parameter checking to see if high degree resolve can be used
#                         width_max = width_max,
#                         axon_dependent = axon_dependent, 

#                         #arguments for what to return
#                         return_skip_info=True,
        
#                         verbose=verbose,
#                         )
    
#     if len(return_value) > 0:
#         return return_value
    
#     # ---------- Phase B: Start the filtering of downstream branches for the match ----
    
#     if verbose:
#         print(f"***Branch being considered after filters = {branch_idx}***")
    
#     #1) Compute features of a complete graph that connets all upsream and downsream edges
#     #(slightly different computation for upstream than downstream edges)
    
#     upstream_branch = branch_idx
#     all_branch_idx = np.hstack([downstream_branches,[upstream_branch]])
    
#     G = xu.complete_graph_from_node_ids(all_branch_idx)
    
    
#     if plot_starting_branches:
#         nviz.plot_branch_groupings(limb_obj = limb_obj,
#         groupings = [[k] for k in G.nodes],
#         verbose = False,
#         plot_meshes = True,
#         plot_skeletons = True,
#         extra_group = skipped_nodes)
        
        
#     G_e_2=nst.compute_edge_attributes_locally_upstream_downstream(
#             limb_obj,
#             upstream_branch,
#             downstream_branches,
#             offset=offset,
#             comparison_distance = comparison_distance,
#             plot_extracted_skeletons = plot_extracted_skeletons,
    
#     )
        
#     '''=
#     arguments_for_all_edge_functions = dict(
#                                         #nodes_to_exclude=nodes_to_exclude,
#                                            branch_1_direction="upstream",
#                                             branch_2_direction="downstream",
#                                            comparison_distance = 10000)
    
#     nodes_to_compute = [upstream_branch]
#     edge_functions = dict(sk_angle=dict(function=nst.parent_child_sk_angle,
#                                         arguments=dict(offset=offset,
#                                                       comparison_distance=comparison_distance,
#                                                       plot_extracted_skeletons=plot_extracted_skeletons)),
#                          width_diff = nst.width_diff,
#                           width_diff_percentage = nst.width_diff_percentage,
#                          synapse_density_diff=nst.synapse_density_diff,
#                           n_synapses_diff = nst.n_synapses_diff,
#                           none_to_some_synapses = nst.none_to_some_synapses
#                          )

#     G_e_1 = nst.compute_edge_attributes_locally(G,
#                                               limb_obj,
#                                              nodes_to_compute,
#                                              edge_functions,
#                                              verbose=False,
#                                                 arguments_for_all_edge_functions=arguments_for_all_edge_functions,
#                                              directional = False)
    
#     nodes_to_compute = downstream_branches
    
#     arguments_for_all_edge_functions = dict(
#                                         #nodes_to_exclude=nodes_to_exclude,
#                                            branch_1_direction="downstream",
#                                             branch_2_direction="downstream",
#                                            comparison_distance = 10000)
    
#     edge_functions = dict(
#                           sk_angle=dict(function=nst.sibling_sk_angle,
#                                         arguments=dict(offset=offset,
#                                                       comparison_distance=comparison_distance,
#                                                 plot_extracted_skeletons=plot_extracted_skeletons)),
#                          width_diff = nst.width_diff,
#                           width_diff_percentage = nst.width_diff_percentage,
#                          synapse_density_diff=nst.synapse_density_diff,
#                           n_synapses_diff = nst.n_synapses_diff,
#                          none_to_some_synapses = nst.none_to_some_synapses)

#     G_e_2 = nst.compute_edge_attributes_locally(G_e_1,
#                                               limb_obj,
#                                              nodes_to_compute,
#                                              edge_functions,
#                                              verbose=False,
#                                                 arguments_for_all_edge_functions=arguments_for_all_edge_functions,
#                                              directional = False)
        
#     '''
        
#     #2) Filter the edges by local properties
#     synapse_query = (f"((synapse_density_diff<{synapse_density_diff_threshold}) or" 
#                         f" (n_synapses_diff < {n_synapses_diff_threshold}))")

#     branch_match_query = (f"(((width_diff < {width_diff_max}) or (width_diff_percentage < {width_diff_perc}))"
#                           f" and (sk_angle < {worst_case_sk_angle_match_threshold}))")

#     if perform_synapse_filter:
#         branch_match_query += f"and {synapse_query}"

#     if verbose:
#         print(f"branch_match_query = :\n{branch_match_query}")

#     G_edge_filt = xu.d(G_e_2,
#                                       edge_query=branch_match_query,
#                                       verbose=verbose)
#     if plot_G_local_edge:
#         print(f"\n--- Before Local Query ---")
#         print(xu.edge_df(G_e_2))
#         print("Afer Local query: ")
#         print(xu.edge_df(G_edge_filt))
#         nx.draw(G_edge_filt,with_labels=True) 
#         plt.show()
    
    
#     G = G_edge_filt
#     if len(G_edge_filt.edges()) > 0:
#         # ------------- Phase B.2: Looking at global features for query ------- #
#         print(f"plot_G_global_edge = {plot_G_global_edge}")
#         if verbose:
#             print(f"Performing global features query")

#         # 3) computes the global fetures
#         edge_functions_global = dict(definite_partner_sk_delete=dict(function=nst.edges_to_delete_from_threshold_and_buffer,
#                                                               arguments=dict(threshold=sk_angle_match_threshold,
#                                                                                   buffer= sk_angle_buffer,
#                                                                            verbose = False,
#                                                                             edge_attribute = "sk_angle")),
#                                 definite_partner_width_delete=dict(function=nst.edges_to_delete_from_threshold_and_buffer,
#                                                               arguments=dict(threshold=width_diff_perc_threshold,
#                                                                                   buffer= width_diff_perc_buffer,
#                                                                            verbose = False,
#                                                                             edge_attribute = "width_diff_percentage"))

#                          )

#         # 4) Filtering Graph by global properties (applying the definite filter pair)
#         G_edge_filt_with_att = nst.compute_edge_attributes_globally(G_edge_filt,
#                                              edge_functions_global)
#         G_global_1 = xu.query_to_subgraph(G_edge_filt_with_att,
#                                           edge_query="(definite_partner_sk_delete == False) or ((definite_partner_sk_delete != True) and (definite_partner_width_delete != True))",
#                                           verbose=verbose)

#         if plot_G_global_edge:
#             print(f"\n--- Before Global Query ---")
#             print(xu.edge_df(G_edge_filt_with_att))
#             print("Afer Global query: ")
#             print(xu.edge_df(G_global_1))
#             nx.draw(G_global_1,with_labels=True) 
#             plt.show()
            
#         G = G_global_1
#         if len(G_global_1.edges())>0:
#             if verbose:
#                 print(f"Performing node features query")
    
#             # 5) Computing NOde features (for sfiltering on the upstream node edges)
#             edge_functions_node_global = dict(above_threshold_delete=dict(
#                                         function=nst.edges_to_delete_on_node_above_threshold_if_one_below,
#                                         arguments=dict(threshold=sk_angle_match_threshold,
#                                            verbose = False)
#                                         )
#                              )

#             if use_exclusive_partner:
#                 nodes_to_compute = list(G_global_1.nodes())
#             else:
#                 nodes_to_compute = branch_idx
                
#             G_edge_filt_with_node_att = nst.compute_edge_attributes_around_node(G_global_1,
#                                              edge_functions_node_global,
#                                                 nodes_to_compute=nodes_to_compute,
#                                              )

#             # 6) Filtering graph based on node features
#             G_global_2 = xu.query_to_subgraph(G_edge_filt_with_node_att,
#                                           edge_query="above_threshold_delete != True",
#                                           verbose=verbose)

#             if plot_G_node_edge:
#                 print(f"\n--- Before Node Query ---")
#                 print(xu.edge_df(G_edge_filt_with_node_att))
#                 print("Afer Node query: ")
#                 print(xu.edge_df(G_global_2))
#                 nx.draw(G_global_2,with_labels=True) 
#                 plt.show()
                
#             G = G_global_2
        
        
#     upstream_branch = branch_idx
    
    
#     # ------- Phase C: Optional Kiss Filter ------
#     """
#     ---------- 4/29 Addition: Kiss Filter -----------

#     Pseudocode:
#     0) Get the offset skeleton coordinates for all nodes in graph
#     1) find all the possible partitions of the remaining nodes

#     """ 
#     if kiss_check:
#         if verbose:
#             print(f"Attempting to perform Kiss check")
#         coordinate = nru.downstream_endpoint(limb_obj,upstream_branch)
#         upstream_matches = xu.get_neighbors(G,upstream_branch)
#         if len(upstream_matches)>1:
#             print(f"Working on Kissing check because possible upstream matches greater than 1: {upstream_matches}")

#             G = ed.cut_kissing_graph_edges(G,limb_obj,
#                 kiss_check_bbox_longest_side_threshold = kiss_check_bbox_longest_side_threshold,
#                 coordinate = coordinate,
#                 offset=offset,
#                 comparison_distance = comparison_distance,
#                 plot_offset_skeletons = False,
#                 plot_source_sink_vertices = False,
#                 plot_cut_vertices = False,
#                 plot_cut_bbox = False,
#                 verbose = False
#                             )

#             if verbose:
#                 print(f"Step 5b: Removing kissing edges")
#                 print(f"Remaining Edges = {G.edges()}")
#         else:
#             if verbose:
#                 print(f"Not doing kiss check because upstream_matches = {upstream_matches}")


    
#     # ------- Phase D: Picking the Winner of the upstream branch and error branches ------
    
    
#     """
#     Part 6:
#     if there are other nodes in the group use on of the following to determine winner
#         a) best match
#         b) least sum angle
#     """
    
    
#     upstream_subgraph = np.array([list(k) for k in nx.connected_components(G) 
#                                   if upstream_branch in k][0])
#     if len(upstream_subgraph) == 1:
#         winning_node = None
#         error_branches = downstream_branches
#     else:
#         if match_method == "best_match":
#             if verbose:
#                 print(f"Using best match method")
#             winning_node = xu.get_neighbor_min_weighted_edge(G,upstream_branch)
#         elif match_method == "lowest_angle_sum":
#             if verbose:
#                 print(f"Using lowest_angle_sum method")
#             raise Exception("hasn't been fixed to make sure the upstream node is guaranteed to be in the output graph")
#             G_final = xu.graph_to_lowest_weighted_sum_singular_matches(G,
#             verbose = verbose,
#             return_graph = True)
            
            
#             winning_node = xu.get_neighbors(G_final,upstream_branch)
#             if len(winning_node) != 1:
#                 raise Exception(f"Not just one winning node: {winning_node}")
#             else:
#                 winning_node = winning_node[0]
#         elif match_method == "all_error_if_not_one_match":
#             error_branches = downstream_branches
#             if len(upstream_subgraph) == 2:
#                 winning_node = upstream_subgraph[upstream_subgraph!=upstream_branch][0]
#             else:
#                 winning_node = None
#         else:
#             raise Exception(f"Unimplemented match_method : {match_method} ")


#         error_branches = downstream_branches[downstream_branches!= winning_node]

#     if verbose:
#         print(f"for upstream node {upstream_branch}, winning_node = {winning_node}, error_branches = {error_branches}")

#     if plot_final_branch_matches:
#         nviz.plot_branch_groupings(limb_obj = limb_obj,
#         groupings = G,
#         verbose = False,
#         plot_meshes = True,
#         plot_skeletons = True,
#             extra_group = skipped_nodes,)
    
#     """
#     --- 5/12: Not having the short thick end nodes in the errors to remove
#     """
#     if remove_short_thick_endnodes:    
#         #print(f"short_thick_endnodes_to_remove = {short_thick_endnodes_to_remove}")
#         error_branches = np.setdiff1d(error_branches,short_thick_endnodes_to_remove)
        
#     if axon_spines is not None:
#         error_branches = np.setdiff1d(error_branches,axon_spines)
    
#     return winning_node,error_branches

def high_degree_false_positive_low_sibling_filter(
    limb_obj,
    branch_idx,
    downstream_idx,
    width_min = None,#320,
    sibling_skeletal_angle_max = None,#90,
    verbose = False,
    ):
    """
    Purpose: to not error out high degree branches
    that have a degree of 4 and the error branches
    have a very low sibling angle

    Pseudocode: 
    1) If have 2 error branches
    2) If the width is above a threshold
    3) Find the skeletal angle between the two components
    4) Return no errors if less than certain skeletal length
    
    Ex: 
    high_degree_false_positive_low_sibling_filter(
        neuron_obj[2],
        3,
        [1,2],
        verbose = True,
        width_min = 400,
        #sibling_skeletal_angle_max=80
    )
    """
    bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
    
    
    print(f"Inside high_degree_false_positive_low_sibling_filter ****")
    verbose = True
    
    if width_min is None:
        width_min = width_min_high_degree_false_positive_global
    if sibling_skeletal_angle_max is None:
        sibling_skeletal_angle_max =sibling_skeletal_angle_max_high_degree_false_positive_global
    
    error_downstream = downstream_idx
    if branch_idx is None:
        if verbose:
            print(f"No winning branch so returning")
        return error_downstream
    
    if error_downstream is None:
        if verbose:
            print(f"None error_downstream so returning")
        return error_downstream
    
    if len(error_downstream) != 2:
        if verbose:
            print(f"Not exactly 2 downstream errors so returning")
        return error_downstream
    
    upstream_b = limb_obj[branch_idx]
    upstream_b_width = nst.width_new(upstream_b)
    
    if upstream_b_width < width_min:
        if verbose:
            print(f"Upstream width ({upstream_b_width}) less than width_min({width_min})")
        return error_downstream
    
    #3) Find the skeletal angle between the two components
    b1 = limb_obj[error_downstream[0]]
    b2 = limb_obj[error_downstream[1]]
    
    try:
        skel_angle = nu.angle_between_vectors(
            b1.skeleton_vector_upstream,
            b2.skeleton_vector_upstream
        )
    except:
        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)
        b1 = limb_obj[error_downstream[0]]
        b2 = limb_obj[error_downstream[1]]
        skel_angle = nu.angle_between_vectors(
            b1.skeleton_vector_upstream,
            b2.skeleton_vector_upstream
        )
    
    if verbose:
        print(f"skel_angle between downstream branches = {skel_angle}")
        
    #4) Return no errors if less than certain skeletal length
    if skel_angle < sibling_skeletal_angle_max:
        if verbose:
            print(f"Sibling angle less than max so returning no branches")
        return []
    else:
        if verbose:
            print(f"Sibling angle greater than max so returning original errors")
        return error_downstream
    
def high_degree_upstream_match(
    limb_obj,
    branch_idx,
    
    #--- Phase A: arguments for determining downstream nodes ------
    skip_distance = None,#3000,
    min_upstream_skeletal_distance = None,
    remove_short_thick_endnodes = True,
    axon_spines = None,
    
    
    short_thick_endnodes_to_remove = None,
    
    # ----------- default parameters for these set in the preprocessing function
    min_degree_to_resolve = None,#3,
    
    # helps determine the max degrees to resolve
    width_func = None,
    max_degree_to_resolve_absolute = None,#1000,
    max_degree_to_resolve = None,#1000,
    max_degree_to_resolve_wide = None,#1000,
    max_degree_to_resolve_width_threshold = None,#200,
    skip_greater_than_max_degree_to_resolve=False,
    
    # parameter checking to see if high degree resolve can be used
    width_max = None,#170,
    upstream_width_max = None,#None,
    axon_dependent = True,
    
    plot_starting_branches = False,

    # --- Phase B.1: parameters for local edge attributes ------
    offset=None,#1000,#1500,
    comparison_distance = None,#2000,
    plot_extracted_skeletons = False,
    
    
    # --- Phase B.2: parameters for local edge query ------
    
    worst_case_sk_angle_match_threshold = None,#65,
    
    width_diff_max = None,#75,#np.inf,100,
    width_diff_perc = None,#0.60,
    
    perform_synapse_filter =None,# True,
    synapse_density_diff_threshold = None,#0.00015, #was 0.00021
    n_synapses_diff_threshold = None,#6,
    
    plot_G_local_edge = False,
    
    # ----- Phase B.3: parameters for global attributes ---
    #args for definite pairs
    sk_angle_match_threshold = None,#45,
    sk_angle_buffer = None,#25,
    
    width_diff_perc_threshold = None,#0.15,
    width_diff_perc_buffer = None,#0.30,

    # ----- Phase B.4 paraeters for global query ---
    plot_G_global_edge = False,
    
    # ----- Phase B.6 paraeters for ndoe query ---
    plot_G_node_edge = False,
    
    # ---- Phase C: Optional Kiss filter ----
    kiss_check = None,#False,
    kiss_check_bbox_longest_side_threshold = None,#450,

    # ---- Phase D: Picking the final winner -----
    plot_final_branch_matches = False,
    match_method = None,#"all_error_if_not_one_match",# "best_match", #other option is "best_match"
    use_exclusive_partner = None,#True,
    
    #false positive low sibling filter
    use_high_degree_false_positive_filter = None,
    
    verbose = False,
    ):
    #print(f"perform_synapse_filter = {perform_synapse_filter}")
    """
    Purpose: To Determine if branches downstream from a certain
    branch should be errored out based on crossovers and 
    high degree branching downstream
    
    Pseudocode: 
    Phase A:
    #1) Get all downstream branhes (with an optional skip distance)
    #2) Remove short thick endnodes from possible branches in the high degree point
    #3) Return if not enough branches at the intersection
    #4) If the branch being considered is thick enough then increase the max degree to resolve
    #5) Return all downstream branches as errors if number of branches at intersection is too large
    #6) Do not process the intersection if all the branches are thick or not all are axons (return no errors)
    
    Phase B: 
    #1) Compute features of a complete graph that connets all upsream and downsream edges
    #(slightly different computation for upstream than downstream edges)
    """
#     if branch_idx == 3:
#         verbose = True

    if width_func is None:
        width_func = au.axon_width

    if offset is None:
        offset = offset_high_d_match_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_high_d_match_global
    if worst_case_sk_angle_match_threshold is None:
        worst_case_sk_angle_match_threshold = worst_case_sk_angle_match_threshold_high_d_match_global
    if width_diff_max is None:
        width_diff_max = width_diff_max_high_d_match_global
    if width_diff_perc is None:
        width_diff_perc = width_diff_perc_high_d_match_global
    if perform_synapse_filter is None:
        perform_synapse_filter = perform_synapse_filter_high_d_match_global
    if synapse_density_diff_threshold is None:
        synapse_density_diff_threshold = synapse_density_diff_threshold_high_d_match_global
    if n_synapses_diff_threshold is None:
        n_synapses_diff_threshold = n_synapses_diff_threshold_high_d_match_global
    if sk_angle_match_threshold is None:
        sk_angle_match_threshold = sk_angle_match_threshold_high_d_match_global
    if sk_angle_buffer is None:
        sk_angle_buffer = sk_angle_buffer_high_d_match_global
    if width_diff_perc_threshold is None:
        width_diff_perc_threshold = width_diff_perc_threshold_high_d_match_global
    if width_diff_perc_buffer is None:
        width_diff_perc_buffer = width_diff_perc_buffer_high_d_match_global
    if kiss_check is None:
        kiss_check = kiss_check_high_d_match_global
    if kiss_check_bbox_longest_side_threshold is None:
        kiss_check_bbox_longest_side_threshold = kiss_check_bbox_longest_side_threshold_high_d_match_global
    if match_method is None:
        match_method = match_method_high_d_match_global
    if use_exclusive_partner is None:
        use_exclusive_partner = use_exclusive_partner_high_d_match_global
        
    if use_high_degree_false_positive_filter is None:
        use_high_degree_false_positive_filter = use_high_degree_false_positive_filter_global

    if remove_short_thick_endnodes:
        if short_thick_endnodes_to_remove is None:
            short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
                                 verbose = False)
            
        limb_obj.short_thick_endnodes = short_thick_endnodes_to_remove
    
    if axon_spines is not None:
        limb_obj.axon_spines = axon_spines
        
    #print(f"width_diff_perc_threshold = {width_diff_perc_threshold}")
    #print(f"width_diff_perc_buffer = {width_diff_perc_buffer}")
    #print(f"skip_distance = {skip_distance}")
            
    # ---------- Phase A: Figure out if branch needs to be processed at all (and if so compute the downstream branches ---
    (return_value,
    downstream_branches,
    skip_distance,
    skipped_nodes) = high_low_degree_upstream_match_preprocessing(
                        limb_obj,
                        branch_idx,

                        #arguments for determining downstream nodes
                        skip_distance = skip_distance,
                        min_upstream_skeletal_distance = min_upstream_skeletal_distance,
                        short_thick_endnodes_to_remove = limb_obj.short_thick_endnodes,
                        axon_spines = limb_obj.axon_spines,
                        min_degree_to_resolve = min_degree_to_resolve,

                        # helps determine the max degrees to resolve
                        width_func = width_func,
                        max_degree_to_resolve_absolute = max_degree_to_resolve_absolute,
                        max_degree_to_resolve = max_degree_to_resolve,
                        max_degree_to_resolve_wide = max_degree_to_resolve_wide,
                        max_degree_to_resolve_width_threshold = max_degree_to_resolve_width_threshold,
                        skip_greater_than_max_degree_to_resolve=skip_greater_than_max_degree_to_resolve,

                        # parameter checking to see if high degree resolve can be used
                        width_max = width_max,
                        upstream_width_max = upstream_width_max,
                        axon_dependent = axon_dependent, 

                        #arguments for what to return
                        return_skip_info=True,
        
                        verbose=verbose,
                        )
    
    if len(return_value) > 0:
        return return_value
    
    # ---------- Phase B: Start the filtering of downstream branches for the match ----
    
    
    if verbose:
        print(f"***Branch being considered after filters = {branch_idx}***")
    
#     if branch_idx == 78:
#         verbose = True
#         plot_G_local_edge = True
#         plot_G_global_edge = True
#         plot_final_branch_matches = True
    
    winning_node,error_branches=gf.upstream_pair_singular(limb_obj,
                          upstream_branch=branch_idx,
                          downstream_branches=downstream_branches,
                           plot_starting_branches = plot_starting_branches,
                        offset=offset,
                        comparison_distance = comparison_distance,
                        plot_extracted_skeletons = plot_extracted_skeletons,


                        worst_case_sk_angle_match_threshold = worst_case_sk_angle_match_threshold,

                        width_diff_max = width_diff_max,#np.inf,100,
                        width_diff_perc = width_diff_perc,

                        perform_synapse_filter = perform_synapse_filter,
                        synapse_density_diff_threshold = synapse_density_diff_threshold, #was 0.00021
                        n_synapses_diff_threshold = n_synapses_diff_threshold,

                        plot_G_local_edge = plot_G_local_edge,

                        # ----- Phase B.3: parameters for global attributes ---
                        #args for definite pairs
                        perform_global_edge_filter = True,
                        sk_angle_match_threshold = sk_angle_match_threshold,
                        sk_angle_buffer = sk_angle_buffer,

                        width_diff_perc_threshold = width_diff_perc_threshold,
                        width_diff_perc_buffer = width_diff_perc_buffer,

                        # ----- Phase B.4 paraeters for global query ---
                        plot_G_global_edge = plot_G_global_edge,


                        # ------- For Node Query ----#
                        perform_node_filter = True,
                        use_exclusive_partner = use_exclusive_partner,
                        plot_G_node_edge = plot_G_node_edge,
                           
                        kiss_check = kiss_check,
                        kiss_check_bbox_longest_side_threshold = kiss_check_bbox_longest_side_threshold,

                        # ---- Phase D: Picking the final winner -----
                        plot_final_branch_matches = plot_final_branch_matches,
                        match_method = match_method,# "best_match", #other option is "best_match"

                        verbose = verbose,
                          )
    
    
    """
    --- 5/12: Not having the short thick end nodes in the errors to remove
    """
    if remove_short_thick_endnodes:    
        #print(f"short_thick_endnodes_to_remove = {short_thick_endnodes_to_remove}")
        error_branches = np.setdiff1d(error_branches,short_thick_endnodes_to_remove)
        
    if axon_spines is not None:
        error_branches = np.setdiff1d(error_branches,axon_spines)
        
    if use_high_degree_false_positive_filter:
        error_branches = high_degree_false_positive_low_sibling_filter(
            limb_obj,
            winning_node,
            error_branches,
            verbose = verbose,
        )
        if verbose:
            print(f"Using use_high_degree_false_positive_filter and after error_branches = {error_branches} ")
    
    return winning_node,error_branches



    

def high_degree_branch_errors_limb_branch_dict(neuron_obj,
                                               limb_branch_dict = "axon",
                                               # parameters to add as more filters for the branches to check
                                               skip_distance = None,
                                               min_upstream_skeletal_distance = None,
                                               plot_limb_branch_pre_filter = False,
                                               plot_limb_branch_post_filter = False,
                                               plot_limb_branch_errors = False,
                                               verbose = False,
                                               high_degree_order_verbose = False,
                                               filter_axon_spines = True,
                                               axon_spines_limb_branch_dict = None,
                                               filter_short_thick_endnodes = True,
                                               debug_branches = None,
                                              **kwargs):

    """
    Purpose: To resolve high degree nodes for a neuron 

    Pseudocode: 
    0) get the limb branch dict to start over
    2) Find all of the high degree coordinates on the axon limb

    For each high degree coordinate
    a. Send the coordinate to the high_degree_upstream_match
    b. Get the error limbs back and if non empty then add to the limb branch dict

    return the limb branch dict

    """
    #debug_branches = [4]
    
    if min_upstream_skeletal_distance is None:
        min_upstream_skeletal_distance = min_upstream_skeletal_distance_global
    
    
#     high_degree_order_verbose = True
#     verbose = True
    
    
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict
    elif limb_branch_dict in ["axon","dendrite"]:
        limb_branch_dict = getattr(neuron_obj,f"{limb_branch_dict}_limb_branch_dict")
    else:
        pass
    
    if plot_limb_branch_pre_filter:
        print(f"The initial limb branch dict before the skip distance and skeletal length ")
        nviz.plot_limb_branch_dict(neuron_obj,
                                   limb_branch_dict)
    
# ------ Moved this filter into the preprocessing stage of high_lower_degree branching preprocessing ----
    
#     high_degree_limb_branch = ns.query_neuron(neuron_obj,
#                     functions_list=["skeletal_length"],
#                     query = f"(skeletal_length>{min_skeletal_distance})",
#                     function_kwargs=dict(skip_distance=skip_distance),
#                    limb_branch_dict_restriction=limb_branch_dict)

#     if plot_limb_branch_post_filter:
#         print(f"The initial limb branch dict after min_skeletal_distance = {min_skeletal_distance} ")
#         nviz.plot_limb_branch_dict(neuron_obj,
#                                    high_degree_limb_branch)
    
    if filter_axon_spines and axon_spines_limb_branch_dict is None:
        axon_spines_limb_branch_dict = au.axon_spines_limb_branch_dict(neuron_obj)
    elif axon_spines_limb_branch_dict is None:
        axon_spines_limb_branch_dict = dict()
    else:
        pass
    
    if filter_short_thick_endnodes:
        short_thick_endnodes_to_remove_limb_branch = au.short_thick_branches_limb_branch_dict(neuron_obj,
                                                                                         verbose = False)
    else:
        short_thick_endnodes_to_remove_limb_branch = dict()
    
    limb_branch_dict_errors = dict()
    for limb_name,branch_list in limb_branch_dict.items():
        if verbose:
            print(f"\n\n ----- Working on limb {limb_name}-------")
        limb_obj = neuron_obj[limb_name]
#         short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
#                                      verbose = False)

        if limb_name in short_thick_endnodes_to_remove_limb_branch.keys():
            #short_thick_endnodes_to_remove = short_thick_endnodes_to_remove_limb_branch[limb_name]
            limb_obj.short_thick_endnodes = short_thick_endnodes_to_remove_limb_branch[limb_name]
        else:
            limb_obj.short_thick_endnodes = []

        
        if limb_name in axon_spines_limb_branch_dict.keys():
            limb_obj.axon_spines = axon_spines_limb_branch_dict[limb_name]
        else:
            limb_obj.axon_spines = []

        error_branches = []
        for j,b in enumerate(branch_list):
            if debug_branches is not None:
                if b not in debug_branches:
                    continue
                
                kwargs["plot_starting_branches"] = True
                kwargs["plot_G_local_edge"] = True
                kwargs["plot_G_global_edge"] = True
                kwargs["plot_G_node_edge"] = True
                kwargs["plot_final_branch_matches"] = True
                high_degree_order_verbose = True
                verbose = True
            
                
            if verbose:
                print(f"\n\n ----- Working on branch {j}/{len(branch_list)}: {b}--------")
            winning_downstream,error_downstream = ed.high_degree_upstream_match(limb_obj,
                                                                                branch_idx=b,
                                                                                skip_distance=skip_distance,
                                short_thick_endnodes_to_remove = limb_obj.short_thick_endnodes,
                                                                                verbose = high_degree_order_verbose,
                                                                                axon_spines = limb_obj.axon_spines,
                                                                                min_upstream_skeletal_distance=min_upstream_skeletal_distance,
                                                                               **kwargs)
            
            #winning_downstream,error_downstream = [],[]
        
            if verbose:
                print(f"winning_downstream = {winning_downstream},error_downstream = {error_downstream} ")
            if len(error_downstream) > 0:
                error_branches += list(error_downstream)

        if len(error_branches) > 0:
            limb_branch_dict_errors[limb_name] = np.array(error_branches)
            
    if plot_limb_branch_errors:
        print(f"After high degree branch filter errors: limb_branch_dict_errors = {limb_branch_dict_errors} ")
        nviz.plot_limb_branch_dict(neuron_obj,
                                   limb_branch_dict_errors)

    return limb_branch_dict_errors





def high_degree_branch_errors_dendrite_limb_branch_dict(
    neuron_obj,

    # parameters for high_degree_branch_errors_limb_branch_dict
    skip_distance = None,#1_500,
    #min_upstream_skeletal_distance = None,#10_000,
    plot_limb_branch_pre_filter = False,
    plot_limb_branch_post_filter = False,
    plot_limb_branch_errors = False,
    verbose = False,
    high_degree_order_verbose = False,
    filter_axon_spines = True,
    filter_short_thick_endnodes = False,
    debug_branches = None,
    
    
    #--------high degree upstream match parameters-----
    width_max = None,#800,
    upstream_width_max = None,#1000000,
    offset = None,#1_500,
    comparison_distance = None,#3_000,
    width_diff_max = None,#150,
    perform_synapse_filter = None,#False,
    width_diff_perc_threshold = None,
    width_diff_perc_buffer = None,
    
    #---- parameters for limb branch restriction -------
    min_skeletal_length_endpoints = None,#4_000,
    plot_endpoints_filtered = False,
    min_distance_from_soma_mesh = None,#7_000,
    plot_soma_restr = False,
    
    use_high_degree_false_positive_filter = None,
    
    **kwargs):
    
    if width_diff_perc_threshold is None:
        width_diff_perc_threshold = width_diff_perc_threshold_high_d_match_dendr_global
    if width_diff_perc_buffer is None:
        width_diff_perc_buffer = width_diff_perc_buffer_high_d_match_dendr_global
    
    if skip_distance is None:
        skip_distance = skip_distance_high_degree_dendr_global
#     if min_upstream_skeletal_distance is None:
#         min_upstream_skeletal_distance = min_upstream_skeletal_distance_high_degree_dendr_global
    if width_max is None:
        width_max = width_max_high_degree_dendr_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_high_degree_dendr_global
    if offset is None:
        offset = offset_high_degree_dendr_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_high_degree_dendr_global
    if width_diff_max is None:
        width_diff_max = width_diff_max_high_degree_dendr_global
    if perform_synapse_filter is None:
        perform_synapse_filter = perform_synapse_filter_high_degree_dendr_global
    if min_skeletal_length_endpoints is None:
        min_skeletal_length_endpoints = min_skeletal_length_endpoints_high_degree_dendr_global
    if min_distance_from_soma_mesh is None:
        min_distance_from_soma_mesh = min_distance_from_soma_mesh_high_degree_dendr_global
        
    if use_high_degree_false_positive_filter is None:
        use_high_degree_false_positive_filter = use_high_degree_false_positive_filter_dendr_global
    
    print(f"width_max = {width_max}")
    print(f"upstream_width_max = {upstream_width_max}")
    
    # purpose: To find the small end nodes
    
    limb_branch_too_close = nst.euclidean_distance_close_to_soma_limb_branch(
        neuron_obj,
        distance_threshold=min_distance_from_soma_mesh,
        plot = plot_soma_restr
    )    
    
    axon_spines_limb_branch = ns.query_neuron(
        neuron_obj,
        query = f"(skeletal_length < {min_skeletal_length_endpoints}) and (n_downstream_nodes == 0)",
        limb_branch_dict_restriction=neuron_obj.dendrite_limb_branch_dict,
        plot_limb_branch_dict=plot_endpoints_filtered
    )
    
    limb_b_restr = nru.limb_branch_setdiff([neuron_obj.dendrite_limb_branch_dict,limb_branch_too_close])
    
    if verbose:
        print(f"limb_branch_too_close = {limb_branch_too_close}")
        print(f"axon_spines_limb_branch = {axon_spines_limb_branch}")
    
    #print(f"skip_distance = {skip_distance}")
    return ed.high_degree_branch_errors_limb_branch_dict(
        neuron_obj,
        limb_branch_dict = limb_b_restr,
        skip_distance = skip_distance,
        #min_upstream_skeletal_distance = min_upstream_skeletal_distance,
        plot_limb_branch_pre_filter = plot_limb_branch_pre_filter,
        plot_limb_branch_post_filter = plot_limb_branch_post_filter,
        plot_limb_branch_errors = plot_limb_branch_errors,
        verbose = verbose,
        high_degree_order_verbose = high_degree_order_verbose,
        filter_axon_spines = filter_axon_spines,
        axon_spines_limb_branch_dict=axon_spines_limb_branch,
        filter_short_thick_endnodes = filter_short_thick_endnodes,
        debug_branches = debug_branches,

        width_max = width_max,
        upstream_width_max = upstream_width_max,
        axon_dependent = False,
        offset = offset,
        comparison_distance = comparison_distance,
        width_diff_max = width_diff_max,
        perform_synapse_filter = perform_synapse_filter,
        
        width_diff_perc_threshold = width_diff_perc_threshold,
        width_diff_perc_buffer = width_diff_perc_buffer,
        
        use_high_degree_false_positive_filter=use_high_degree_false_positive_filter,

        **kwargs

    )





# ---------- The low degree error detection ----------- #
low_degree_filters_default = []
def low_degree_upstream_match(
    limb_obj,
    branch_idx,

    #--- Phase A: arguments for determining downstream nodes ------
    skip_distance = None,#0,#3000,
    min_upstream_skeletal_distance = None,#2000,
    remove_short_thick_endnodes = True,
    short_thick_endnodes_to_remove = None,
    axon_spines = None,
    
    
    min_degree_to_resolve = None,#2,
    max_degree_to_resolve_wide = None,#2,
    
    # helps determine the max degrees to resolve
    width_func = None,
    max_degree_to_resolve_absolute = None,#1000,
    max_degree_to_resolve = None,#2,
    skip_greater_than_max_degree_to_resolve = False,
    #max_width_to_resolve = None,
    
    # parameter checking to see if high degree resolve can be used
    width_max = None,#170,
    upstream_width_max = None,#None,
    axon_dependent = True,
    
    plot_starting_branches = False,

    # --- Phase B.1: parameters for local edge attributes ------
    offset=None,#1000,#1500,
    comparison_distance = None,#2000,
    plot_extracted_skeletons = False,
    
    
    # --- Phase B.2: parameters for local edge query ------
    
    worst_case_sk_angle_match_threshold = None,#65,
    
    width_diff_max = None,#75,#np.inf,100,
    width_diff_perc = None,#0.60,
    
    perform_synapse_filter =None,# True,
    synapse_density_diff_threshold = None,#0.00015, #was 0.00021
    n_synapses_diff_threshold = None,#6,
    
    plot_G_local_edge = False,
    
    filters_to_run = None,
    
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To Determine if branches downstream from a certain
    branch should be errored out based on forking rules
    
    
    1) Determine if branch should even be processed
    if should be processed
    2) Calculate the edge attributes for this local graph
    3) Iterate through all of the filters filters_to_run
        a. Send the limb, graph to the filter to run
        b. 
    """
#     if branch_idx == 13:
#         verbose = True

    if width_func is None:
        width_func = au.axon_width
    if skip_distance is None:
        skip_distance = skip_distance_low_d_match_global
    if min_upstream_skeletal_distance is None:
        min_upstream_skeletal_distance = min_upstream_skeletal_distance_low_d_match_global
    if min_degree_to_resolve is None:
        min_degree_to_resolve = min_degree_to_resolve_low_d_match_global
    if max_degree_to_resolve_wide is None:
        max_degree_to_resolve_wide = max_degree_to_resolve_wide_low_d_match_global
    if max_degree_to_resolve_absolute is None:
        max_degree_to_resolve_absolute = max_degree_to_resolve_absolute_low_d_match_global
    if max_degree_to_resolve is None:
        max_degree_to_resolve = max_degree_to_resolve_low_d_match_global
    if width_max is None:
        width_max = width_max_low_d_match_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_low_d_match_global
    if offset is None:
        offset = offset_low_d_match_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_low_d_match_global
    if worst_case_sk_angle_match_threshold is None:
        worst_case_sk_angle_match_threshold = worst_case_sk_angle_match_threshold_low_d_match_global
    if width_diff_max is None:
        width_diff_max = width_diff_max_low_d_match_global
    if width_diff_perc is None:
        width_diff_perc = width_diff_perc_low_d_match_global
    if perform_synapse_filter is None:
        perform_synapse_filter = perform_synapse_filter_low_d_match_global
    if synapse_density_diff_threshold is None:
        synapse_density_diff_threshold = synapse_density_diff_threshold_low_d_match_global
    if n_synapses_diff_threshold is None:
        n_synapses_diff_threshold = n_synapses_diff_threshold_low_d_match_global

    if remove_short_thick_endnodes:
        if short_thick_endnodes_to_remove is None:
            short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
                                 verbose = False)
            
    
    if short_thick_endnodes_to_remove is not None:
        limb_obj.short_thick_endnodes = short_thick_endnodes_to_remove
    if axon_spines is not None:
        limb_obj.axon_spines = axon_spines
        
    # ---------- Phase A: Figure out if branch needs to be processed at all (and if so compute the downstream branches ---
    (return_value,
    downstream_branches,
    skip_distance,
    skipped_nodes) = ed.high_low_degree_upstream_match_preprocessing(
                        limb_obj,
                        branch_idx,

                        #arguments for determining downstream nodes
                        skip_distance = skip_distance,
                        min_upstream_skeletal_distance = min_upstream_skeletal_distance,
                        short_thick_endnodes_to_remove = limb_obj.short_thick_endnodes,
                        axon_spines = limb_obj.axon_spines ,
                        min_degree_to_resolve = min_degree_to_resolve,

                        # helps determine the max degrees to resolve
                        width_func = width_func,
                        max_degree_to_resolve_absolute = max_degree_to_resolve_absolute,
                        max_degree_to_resolve = max_degree_to_resolve,
                        max_degree_to_resolve_wide = None,
                        max_degree_to_resolve_width_threshold = None,
                        skip_greater_than_max_degree_to_resolve=skip_greater_than_max_degree_to_resolve,

                        # parameter checking to see if high degree resolve can be used
                        width_max = width_max,
                        upstream_width_max = upstream_width_max,
                        axon_dependent = axon_dependent, 

                        #arguments for what to return
                        return_skip_info=True,
        
                        verbose=verbose,
                        )
    
    if len(return_value) > 0:
        return return_value[-1],return_value[0]
    
    if verbose:
        print(f"Running local filtering for branch {branch_idx}")
    
    upstream_branch = branch_idx
    G_e_2 = nst.compute_edge_attributes_locally_upstream_downstream(
                limb_obj,
                upstream_branch = upstream_branch,
                downstream_branches = downstream_branches,
                offset=offset,
                comparison_distance=comparison_distance,
                plot_extracted_skeletons=plot_extracted_skeletons,
        )
    
    if plot_G_local_edge:
        print(f"\n--- After Edge Attributes ---")
        print(xu.edge_df(G_e_2,with_node_attributes=False))
        
    G = nst.compute_node_attributes_upstream_downstream(
                            G=G_e_2,
                            limb_obj=limb_obj,
                           upstream_branch=upstream_branch,
                            downstream_branches=downstream_branches,
                           )
    
    if plot_G_local_edge:
        print(f"\n--- After node attributes ---")
        print(xu.node_df(G))
    
    
    #return G
    
    if filters_to_run is None:
        filters_to_run = gf.default_low_degree_graph_filters
        
    # ------- Part that will now run the filters -------#
    error_branches = []
    #verbose = True
    filter_triggered = None
    for filt_func in filters_to_run:
        error_branches = filt_func(G,
                                  limb_obj,
                                  verbose = verbose,
                                   **kwargs
                                  )
        
        if len(error_branches) > 0:
            filter_triggered = filt_func.__name__
            verbose = True
            if verbose:
                print(f"filter_triggered = {filter_triggered}, error_branches = {error_branches}")
            break
    
#     if branch_idx = 30:
#         raise Exception("")
    if short_thick_endnodes_to_remove is not None:    
        error_branches = np.setdiff1d(error_branches,short_thick_endnodes_to_remove)
        
    if axon_spines is not None:
        error_branches = np.setdiff1d(error_branches,axon_spines)
    
    return error_branches,filter_triggered
    
    
    

def low_degree_branch_errors_limb_branch_dict(neuron_obj,
                                               limb_branch_dict = "axon",
                                               # parameters to add as more filters for the branches to check
                                               skip_distance = 0,
                                               min_upstream_skeletal_distance = None,
                                               plot_limb_branch_pre_filter = False,
                                               plot_limb_branch_post_filter = False,
                                               plot_limb_branch_errors = False,
                                               verbose = False,
                                               low_degree_order_verbose = False,
                                               filter_axon_spines = True,
                                              filters_to_run = None,
                                               debug_branches = None,
                                              **kwargs):

    """
    Purpose: To resolve low degree nodes for a neuron 

    Pseudocode: 
    0) get the limb branch dict to start over
    2) Find all of the high degree coordinates on the axon limb

    For each high degree coordinate
    a. Send the coordinate to the high_degree_upstream_match
    b. Get the error limbs back and if non empty then add to the limb branch dict

    return the limb branch dict
    
    Ex: 
    from neurd import error_detection as ed
    ed.low_degree_branch_errors_limb_branch_dict(filt_neuron,
                                                 verbose = True,
                                                low_degree_order_verbose=True,
                                                filters_to_run = [gf.axon_double_back_filter],
                                                plot_G_local_edge = True)
                                                
    Ex on how to debug a certain filter on a certain branch:
    

    """
#     high_degree_order_verbose = True
    if min_upstream_skeletal_distance is None:
        min_upstream_skeletal_distance = min_upstream_skeletal_distance_global
    
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict
    elif limb_branch_dict == "axon":
        limb_branch_dict = neuron_obj.axon_limb_branch_dict
    
    if plot_limb_branch_pre_filter:
        print(f"The initial limb branch dict before the skip distance and skeletal length ")
        nviz.plot_limb_branch_dict(neuron_obj,
                                   limb_branch_dict)
    
    
    if filter_axon_spines:
        axon_spines_limb_branch_dict = au.axon_spines_limb_branch_dict(neuron_obj)
    else:
        axon_spines_limb_branch_dict = dict()
    
    short_thick_endnodes_to_remove_limb_branch = au.short_thick_branches_limb_branch_dict(neuron_obj,
                                                                                         verbose = False)
    
    limb_branch_dict_errors = dict()
    for limb_name,branch_list in limb_branch_dict.items():
        if verbose:
            print(f"\n\n ----- Working on limb {limb_name}-------")
        limb_obj = neuron_obj[limb_name]
#         short_thick_endnodes_to_remove = au.short_thick_branches_from_limb(limb_obj,
#                                      verbose = False)

        if limb_name in short_thick_endnodes_to_remove_limb_branch.keys():
            #short_thick_endnodes_to_remove = short_thick_endnodes_to_remove_limb_branch[limb_name]
            limb_obj.short_thick_endnodes = short_thick_endnodes_to_remove_limb_branch[limb_name]
        else:
            limb_obj.short_thick_endnodes = []

        
        if limb_name in axon_spines_limb_branch_dict.keys():
            limb_obj.axon_spines = axon_spines_limb_branch_dict[limb_name]
        else:
            limb_obj.axon_spines = []

        error_branches = []
        for j,b in enumerate(branch_list):
            if debug_branches is not None:
                if b not in debug_branches:
                    continue
                kwargs["plot_starting_branches"] = True
                kwargs["plot_G_local_edge"] = True
                kwargs["plot_G_global_edge"] = True
                kwargs["plot_G_node_edge"] = True
                kwargs["plot_final_branch_matches"] = True
                
                low_degree_order_verbose = True
                verbose = True
            
            if verbose:
                print(f"\n\n ----- Working on branch {j}/{len(branch_list)}: {b}--------")
            error_downstream,triggered_filter = ed.low_degree_upstream_match(limb_obj,
                                                                                branch_idx=b,
                                                                                skip_distance=skip_distance,
                                short_thick_endnodes_to_remove = limb_obj.short_thick_endnodes,
                                                                                verbose = low_degree_order_verbose,
                                                                                axon_spines = limb_obj.axon_spines,
                                                                                min_upstream_skeletal_distance=min_upstream_skeletal_distance,
                                                                              filters_to_run=filters_to_run,
                                                                               **kwargs)
            #if verbose:
            if triggered_filter is not None:
                print(f"{b} triggered {triggered_filter}")

            #winning_downstream,error_downstream = [],[]
        
            if verbose:
                print(f"error_downstream = {error_downstream},triggered_filter = {triggered_filter} ")
            if len(error_downstream) > 0:
                error_branches += list(error_downstream)

        if len(error_branches) > 0:
            limb_branch_dict_errors[limb_name] = np.array(error_branches)
            
    if plot_limb_branch_errors:
        print(f"After low degree branch filter errors: limb_branch_dict_errors = {limb_branch_dict_errors} ")
        nviz.plot_limb_branch_dict(neuron_obj,
                                   limb_branch_dict_errors)

    return limb_branch_dict_errors
    

def double_back_threshold_axon_by_width(limb_obj = None,
                               branch_idx = None,
                                width=None,
                                axon_thin_width_max = None,
                                nodes_to_exclude=None,
                                double_back_threshold_thin = None,
                                double_back_threshold_thick = None,
                              ):
    """
    Purpose: Will compute the dobule back threshold to use 
    based on the upstream width
    """
    if double_back_threshold_thick is None:
        double_back_threshold_thick = double_back_threshold_axon_thick
    if double_back_threshold_thin is None:
        double_back_threshold_thin = double_back_threshold_axon_thin
    if axon_thin_width_max is None:
        axon_thin_width_max = au.axon_thick_threshold
    
    if limb_obj is not None or branch_idx is not None:
        width = nst.width_upstream(limb_obj,branch_idx,
                                  nodes_to_exclude=nodes_to_exclude)
    
    if width < axon_thin_width_max:
        thresh = double_back_threshold_axon_thin
    else:
        thresh = double_back_threshold_axon_thick
    return thresh


def upstream_node_from_G(G):
    upstream_nodes = xu.get_nodes_with_attributes_dict(G,
                                  dict(node_type="upstream"))
    if len(upstream_nodes) != 1:
        raise Exception(f"Not 1 upstream node: {upstream_nodes}")
    else:
        return upstream_nodes[0]
    
def downstream_nodes_from_G(G):
    downstream_nodes = xu.get_nodes_with_attributes_dict(G,
                                  dict(node_type="downstream"))
    return downstream_nodes


def debug_branches_low_degree(neuron_obj,debug_branches=None,filters_to_run=None):
    """
    ed.debug_branches_low_degree(neuron_obj,debug_branches=[68])
    
    """
    return ed.low_degree_branch_errors_limb_branch_dict(neuron_obj,
                                         filters_to_run = filters_to_run,
                                         debug_branches=debug_branches
                                            )

def debug_branches_high_degree(neuron_obj,debug_branches=None):
    return ed.high_degree_branch_errors_limb_branch_dict(neuron_obj,
                                         debug_branches=debug_branches
                                            )



# ------------- parameters for stats ---------------

global_parameters_dict_default_auto_proof = dsu.DictType(

    double_back_threshold_axon_thick_inh = 135,
    double_back_threshold_axon_thin_inh = 140,
    
    min_upstream_skeletal_distance = 500,
    min_distance_from_soma_for_proof = 10000,
    
    # ---- high_low_degree_upstream_match_preprocessing ----
    min_degree_to_resolve = 3,
    
    max_degree_to_resolve_absolute = 1000,
    max_degree_to_resolve = 1000,
    max_degree_to_resolve_wide = 1000,
    max_degree_to_resolve_width_threshold = 200,
    
    width_min = 35,
    width_max = 170,
    upstream_width_max = (None,"int unsigned"),
    axon_dependent = True, 
    
    # **** filter 2 *****
    # -----high_degree_upstream_match ----
    skip_distance_poly_x = ((80,200),"blob"),
    skip_distance_poly_y = ((1500,2000),"blob"),
    
    # --- Phase B.1: parameters for local edge attributes ------
    offset_high_d_match=1000,#1500,
    comparison_distance_high_d_match = 2000,
    
    
    # --- Phase B.2: parameters for local edge query ------
    worst_case_sk_angle_match_threshold_high_d_match = 65,
    
    width_diff_max_high_d_match = 75,#np.inf,100,
    width_diff_perc_high_d_match = 0.60,
    
    perform_synapse_filter_high_d_match = True,
    synapse_density_diff_threshold_high_d_match = 0.00015, #was 0.00021
    n_synapses_diff_threshold_high_d_match = 6,
    
    # ----- Phase B.3: parameters for global attributes ---
    #args for definite pairs
    sk_angle_match_threshold_high_d_match = 45,
    sk_angle_buffer_high_d_match = 25,
    
    width_diff_perc_threshold_high_d_match = 0.15,
    width_diff_perc_buffer_high_d_match = 0.30,
    
    # ---- Phase C: Optional Kiss filter ----
    kiss_check_high_d_match = False,
    kiss_check_bbox_longest_side_threshold_high_d_match = 450,

    # ---- Phase D: Picking the final winner -----
    match_method_high_d_match = "all_error_if_not_one_match",# "best_match", #other option is "best_match"
    use_exclusive_partner_high_d_match = True,
    
    # -- filtering out false positives --
    use_high_degree_false_positive_filter = True,
    width_min_high_degree_false_positive = 250,
    sibling_skeletal_angle_max_high_degree_false_positive = 110,
    
    
    
    # **** filter 3 *****
    # -----low_degree_upstream_match ----\
    skip_distance_low_d_match = 0,#3000,
    min_upstream_skeletal_distance_low_d_match = 2000,
    
    min_degree_to_resolve_low_d_match = 2,
    max_degree_to_resolve_wide_low_d_match = 3,
    
    # helps determine the max degrees to resolve
    max_degree_to_resolve_absolute_low_d_match = 1000,
    max_degree_to_resolve_low_d_match = 3,
    
    width_max_low_d_match = 170,
    upstream_width_max_low_d_match = (None,"int unsigned"),
    
    offset_low_d_match=1000,#1500,
    comparison_distance_low_d_match = 2000,
    
    # --- Phase B.2: parameters for local edge query ------
    
    worst_case_sk_angle_match_threshold_low_d_match = 65,
    
    width_diff_max_low_d_match = 75,#np.inf,100,
    width_diff_perc_low_d_match = 0.60,
    
    perform_synapse_filter_low_d_match = True,
    synapse_density_diff_threshold_low_d_match = 0.00015, #was 0.00021
    n_synapses_diff_threshold_low_d_match = 6,
    
    
    #---*** width restriction --**    
    width_max_dendr_restr = 500,
    width_max_dendr_double_back_restr = 500,
    upstream_skeletal_length_min_dendr_restr = 5000,
    
    # **** filter 4 ***** #
    #--- width_jump_dendrite ---
    upstream_skeletal_length_min_width_j_dendr = 5000,
    branch_skeletal_length_min_width_j_dendr = 7000,
    upstream_skeletal_length_min_for_min_width_j_dendr = 4000,
    width_jump_max_width_j_dendr = 200,
    
    # -- 4/22/25 addition to prevent width jump when skeleton jumps
    ignore_large_skeleton_endpoint_jump = False,
    max_skeleton_endpoint_jump = 4500,
    
    # **** filter 5 ***** #
    #--- width_jump_axon ---
    upstream_skeletal_length_min_width_j_axon = 5000,
    branch_skeletal_length_min_width_j_axon = 8000,
    upstream_skeletal_length_min_for_min_width_j_axon = 4000,
    width_jump_max_width_j_axon = 55,
    axon_width_threshold_max_width_j_axon = 100,
    
    # **** filter 6 ***** #
    # ---- dendrite double back -----
    double_back_threshold_double_b_dendrite=120,
    comparison_distance_double_b_dendrite = 3000,
    offset_double_b_dendrite = 0,
    branch_skeletal_length_min_double_b_dendrite = 7000, #deciding which branches will be skipped because of length
    double_back_angle_func_type = "parent_skeletal_angle",
    
    
    
    
    
)

global_parameters_dict_default_high_degree_dendr = dict(
    skip_distance_high_degree_dendr = 1_200,
    #min_upstream_skeletal_distance_high_degree_dendr = 10_000,
    width_max_high_degree_dendr = 300,#550,
    upstream_width_max_high_degree_dendr = 300,#550,
    offset_high_degree_dendr = 1_500,
    comparison_distance_high_degree_dendr = 3_000,
    width_diff_max_high_degree_dendr = 150,
    perform_synapse_filter_high_degree_dendr = False,
    width_diff_perc_threshold_high_d_match_dendr = 0.15,
    width_diff_perc_buffer_high_d_match_dendr = 0.20,
    
    use_high_degree_false_positive_filter_dendr = False,
    #---- parameters for limb branch restriction -------
    min_skeletal_length_endpoints_high_degree_dendr = 8_000,#4_000,
    min_distance_from_soma_mesh_high_degree_dendr = 7_000,
    
    
 )


global_parameters_dict_default_crossover = dict(
    # ---- resolve crossover parameters ----
    apply_width_filter = True,
    best_match_width_diff_max = 75,
    best_match_width_diff_max_perc = 0.60,
    best_match_width_diff_min = 0.25,
    no_non_cut_disconnected_comps = True,
    best_singular_match = True,
    lowest_angle_sum_for_pairs=False,
    )

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_auto_proof,
    global_parameters_dict_default_crossover,
    global_parameters_dict_default_high_degree_dendr,
    
])

attributes_dict_default = dict(
    double_back_threshold_axon_thick = 120,
    double_back_threshold_axon_thin = 127,
)    

# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}

# ====== h01 -----------
attributes_dict_h01 = dict()

global_parameters_dict_h01_auto_proof= dict(
    skip_distance_poly_x = ((80,400),"blob"),
    skip_distance_poly_y = ((1500,1700),"blob"),
    
    #high degree match 
    use_high_degree_false_positive_filter = True,
    width_min_high_degree_false_positive = 300,
    sibling_skeletal_angle_max_high_degree_false_positive = 110,
    
    
    
    #low degree match
    width_max_low_d_match = 400,
    
    
    width_jump_axon = 80,
    
    #doubling back angle
    double_back_threshold_double_b_dendrite=100,
    
    #---*** width restriction --**    
    width_max_dendr_restr = 700,
    width_max_dendr_double_back_restr = 10_000,
    upstream_skeletal_length_min_dendr_restr = 10_000,
)

global_parameters_dict_h01_high_degree_dendr = dict(
    skip_distance_high_degree_dendr = 3_000,#1_500,
    width_max_high_degree_dendr = 400,#550,
    upstream_width_max_high_degree_dendr = 400,#550,
)

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_auto_proof,
    global_parameters_dict_h01_high_degree_dendr
    
])



global_parameters_dict_h01_split = dict()


# data_type = "default"
# algorithms = None
# modules_to_set = [ed,(au,"auto_proof"),gf]

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


# if skip_distance_poly is None:
#     skip_distance_poly = calculate_skip_distance_poly()


#--- from neurd_packages ---
from . import axon_utils as au
from . import branch_utils as bu
from . import concept_network_utils as cnu
from . import graph_filters as gf
from . import neuron
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import data_struct_utils as dsu
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm

from . import error_detection as ed