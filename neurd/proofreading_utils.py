import copy
import itertools
import networkx as nx
from pykdtree.kdtree import KDTree
import time
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu
from . import proofreading_utils as pru

current_proof_version = 6 #has the inhibitory rules and safer overall axon rules
current_proof_version = 7 #has the inhibitory rules and safer overall axon rules

split_version = 1 #fixed the problem with split from suggestions
split_version = 2 # adding all of the non-soma touching pieces to all of the splits
split_version = 3 # added width thresholding for which edges are allowed in the resolving edges high degree branching
split_version = 4 # startw with the double back check at beginning

merge_type_color_map = {
    "axon_on_dendrite_merges":"brown",
    "high_degree_branching":"aqua",
    "low_degree_branching":"purple",
    "high_degree_branching_dendrite":"orange",
    "width_jump_up_dendrite":"black",
    "width_jump_up_axon":"yellow",
    "double_back_dendrite":"pink"    
}

def merge_type_to_color(merge_type):
    merge_type = merge_type.replace("_red_blue_suggestions","")
    return merge_type_color_map[merge_type]

def print_merge_type_color_map(color_map = None):
    if color_map is None:
        color_map = merge_type_color_map
    print(f"Merge Type Colors:")
    print(f"------------------")
    for k,v in color_map.items():
        print(f"\t{k} : {v}")

def find_high_degree_coordinates_on_path(limb_obj,curr_path_to_cut,
                                   degree_to_check=4):
    """
    Purpose: Find coordinates on a skeleton of the path speciifed (in terms of node ids)
    that are above the degree_to_check (in reference to the skeleton)
    
    
    """
    path_divergent_points = [sk.find_branch_endpoints(limb_obj[k].skeleton) for k in curr_path_to_cut]
    endpoint_coordinates = np.unique(np.concatenate(path_divergent_points),axis=0)

    limb_sk_gr = sk.convert_skeleton_to_graph(limb_obj.skeleton)
    endpoint_degrees = xu.get_coordinate_degree(limb_sk_gr,endpoint_coordinates)
    high_degree_endpoint_coordinates = endpoint_coordinates[endpoint_degrees>=degree_to_check]
    
    return high_degree_endpoint_coordinates

proofreading_rule_to_clean_name_dict = {
 'axon_on_dendrite_merges': 'Axon On Dendrite',
 'high_degree_branching': 'High Degree Axon',
 'low_degree_branching': 'Low Degree Axon',
 'high_degree_branching_dendrite': 'High Degree Dendrite',
 'width_jump_up_dendrite': 'Width Jump Dendrite',
 'width_jump_up_axon': 'Width Jump Axon',
 'double_back_dendrite': 'Double Back Dendrite'
}

proofreading_rule_color_dict = {
 'axon_on_dendrite_merges': 'red',
 'high_degree_branching': 'blue',
 'low_degree_branching': 'green',
 'high_degree_branching_dendrite': 'orange',
 'width_jump_up_dendrite': 'brown',
 'width_jump_up_axon': 'aqua',
 'double_back_dendrite': 'grey'   
}

def get_best_cut_edge(curr_limb,
                      cut_path,
                      
                      remove_segment_threshold=None,#the segments along path that should be combined
                      remove_segment_threshold_round_2 = None,
                      consider_path_neighbors_for_removal = None,
                      
                      #paraeters for high degree nodes
                      offset_high_degree = None,#2500,#1500,
                      comparison_distance_high_degree = None,
                      match_threshold_high_degree = None,#65,#35,#45,#35,
                      plot_intermediates=False,
                      
                      #parameter for both width and doubling back
                      # This will prevent the edges that were added to extend to the soma from causing the doulbing back or width threshold errors
                      skip_small_soma_connectors = None,
                      small_soma_connectors_skeletal_threshold = None,
                      
                      # parameters for the doubling back
                      double_back_threshold = None,#100,# 130,
                      offset_double_back = None,
                      comparison_distance_double_back = None,
                      
                      #parameters for the width threshold
                      width_jump_threshold = None,
                      verbose=False,
                      
                      high_degree_endpoint_coordinates_tried = [], #is a list of high degree end_nodes that have already been tried
                      simple_path_of_2_cut = None,
                      
                      apply_double_back_first = None,
                      double_back_threshold_at_first = None,
                      return_split_reasons = False,
                    **kwargs):
    """
    Purpose: To choose the best path to cut to disconnect
    a path based on the heuristic hierarchy of 
    
    Cut in descending priority
    1) high degree coordinates
    2) Doubling Back 
    3) Width Jump
    
    Pseudocode: 
    0) Combine close nodes if requested
    1) Get any high degree cordinates on path
    --> if there are then pick the first one and perform the cuts
    
    2) Check the doubling backs (and pick the highest one if above threshold)
    3) Check for width jumps (and pick the highest one)
    4) Record the cuts that will be made
    5) Make the alterations to the graph (can be adding and creating edges)
    """
    if remove_segment_threshold is None:
        remove_segment_threshold = remove_segment_threshold_global
        
    if remove_segment_threshold_round_2 is None:
        remove_segment_threshold_round_2 = remove_segment_threshold_round_2_global
        
    if consider_path_neighbors_for_removal is None:
        consider_path_neighbors_for_removal  = consider_path_neighbors_for_removal_global
        
    if offset_high_degree is None:
        offset_high_degree = offset_high_degree_global
        
    if comparison_distance_high_degree is None:
        comparison_distance_high_degree = comparison_distance_high_degree_global
        
    if match_threshold_high_degree is None:
        match_threshold_high_degree = match_threshold_high_degree_global
        
    if skip_small_soma_connectors is None:
        skip_small_soma_connectors = skip_small_soma_connectors_global
        
    if small_soma_connectors_skeletal_threshold is None:
        small_soma_connectors_skeletal_threshold = small_soma_connectors_skeletal_threshold_global
        
    if double_back_threshold is None:
        double_back_threshold =double_back_threshold_global
        
    if offset_double_back is None:
        offset_double_back = offset_double_back_global
        
    if comparison_distance_double_back is None:
        comparison_distance_double_back = comparison_distance_double_back_global
    
    if width_jump_threshold is None:
        width_jump_threshold = width_jump_threshold_global
        
    if simple_path_of_2_cut is None:
        simple_path_of_2_cut = simple_path_of_2_cut_global
        
    if apply_double_back_first is None:
        apply_double_back_first = apply_double_back_first_global
    
    if double_back_threshold_at_first is None:
        double_back_threshold_at_first =double_back_threshold_at_first_global
    
    #verbose=True
    removed_branches=[]
    
    cut_path = np.array(cut_path)
    split_reason = None
    
    debug = False
    if debug:
        debug_dict = dict(remove_segment_threshold=remove_segment_threshold,#the segments along path that should be combined
        remove_segment_threshold_round_2 = remove_segment_threshold_round_2,
        consider_path_neighbors_for_removal = consider_path_neighbors_for_removal,

        #paraeters for high degree nodes
        offset_high_degree = offset_high_degree,#2500,#1500,
        comparison_distance_high_degree = comparison_distance_high_degree,
        match_threshold_high_degree = match_threshold_high_degree,#65,#35,#45,#35,
        plot_intermediates=plot_intermediates,

        #parameter for both width and doubling back
        # This will prevent the edges that were added to extend to the soma from causing the doulbing back or width threshold errors
        skip_small_soma_connectors = skip_small_soma_connectors,
        small_soma_connectors_skeletal_threshold = small_soma_connectors_skeletal_threshold,

        # parameters for the doubling back
        double_back_threshold = double_back_threshold,#100,# 130,
        offset_double_back = offset_double_back,
        comparison_distance_double_back = comparison_distance_double_back,

        #parameters for the width threshold
        width_jump_threshold = width_jump_threshold,
        verbose=verbose,

        high_degree_endpoint_coordinates_tried = high_degree_endpoint_coordinates_tried, #is a list of high degree end_nodes that have already been tried
        simple_path_of_2_cut = simple_path_of_2_cut,

        apply_double_back_first = apply_double_back_first,
        double_back_threshold_at_first = double_back_threshold_at_first,
        return_split_reasons = return_split_reasons,)
        
        debug_dict.update(kwargs)
        print(f"Inisde get_best_cut_edge: debug_dict=/n{debug_dict}")
    
    # ---------- 2/2: will just cut the path if current path is 2 ----------------#
    if len(cut_path) == 2 and simple_path_of_2_cut:
        if verbose:
            print(f"Path was only 2 segments ({cut_path}) so just cutting them")
        edges_to_delete = [[cut_path[0],cut_path[1]]]
        edges_to_create = []
        curr_high_degree_coord_list = [] 
        split_reason = "2_branch_cut_at_middle"
    
    else:
        if (not remove_segment_threshold is None) and (remove_segment_threshold > 0):
            

            path_without_ends = cut_path[1:-1]

            if consider_path_neighbors_for_removal:
                curr_neighbors = np.unique(np.concatenate([xu.get_neighbors(curr_limb.concept_network,k) for k in cut_path]))
                segments_to_consider = curr_neighbors[(curr_neighbors != cut_path[0]) & (curr_neighbors != cut_path[-1])]

                # ----------- 1/19: Have to remove the starting nodes from possible collapsing ------
                segments_to_consider = np.setdiff1d(segments_to_consider,curr_limb.all_starting_nodes)
                
                if verbose:
                    print(f"consider_path_neighbors_for_removal is set so segments_to_consider = {segments_to_consider}")
            else:
                segments_to_consider = np.array(path_without_ends)


            sk_lens = np.array([sk.calculate_skeleton_distance(curr_limb[k].skeleton) for k in segments_to_consider])

            short_segments = segments_to_consider[np.where(sk_lens<remove_segment_threshold)[0]]
            if len(short_segments) == 0:
                if verbose:
                    print(f"Trying combining short segments with a larger threshold {remove_segment_threshold_round_2}")
                short_segments = segments_to_consider[np.where(sk_lens<remove_segment_threshold_round_2)[0]]

            if verbose:
                print(f"Short segments to combine = {short_segments}")

            if len(short_segments)>0:
                if verbose:
                    print("\n\n-------- Removing Segments -------------")
                curr_limb = pru.collapse_branches_on_limb(curr_limb,
                                                     branch_list=short_segments,
                                                     plot_new_limb=False,
                                                    verbose=verbose)
                
                curr_limb.set_concept_network_directional(starting_node=cut_path[0],
                                                         suppress_disconnected_errors=True)
                

                removed_branches = list(short_segments)

                for s in short_segments:
                    cut_path = cut_path[cut_path != s]
                if verbose:
                    print(f"Revised cut path = {cut_path}")
                    print("\n-------- Done Removing Segments -------------\n\n")







        high_degree_endpoint_coordinates = find_high_degree_coordinates_on_path(curr_limb,cut_path)
        high_degree_endpoint_coordinates = nu.setdiff2d(high_degree_endpoint_coordinates,high_degree_endpoint_coordinates_tried)

        edges_to_create = None
        edges_to_delete = None

        resolve_crossover_at_end = True
        curr_high_degree_coord_list = [] #will store what high degree end-node was tried
        if verbose:
            print(f"Found {len(high_degree_endpoint_coordinates)} high degree coordinates to cut")
            
            
            
            
        curr_limb.set_concept_network_directional(starting_node = cut_path[0],suppress_disconnected_errors=True)

        if apply_double_back_first:
            # ------------- 12 /28 addition that allows us to skip end nodes if too small ------------------
            skip_nodes = []
            if skip_small_soma_connectors:
                revised_cut_path = np.array(cut_path)
                for endnode in [cut_path[0],cut_path[-1]]:
                    curr_sk_distance = sk.calculate_skeleton_distance(curr_limb[endnode].skeleton)
                    if curr_sk_distance<small_soma_connectors_skeletal_threshold:
                        print(f"Skipping endnode {endnode} because skeletal distance was {curr_sk_distance} and threshold was {small_soma_connectors_skeletal_threshold}")
                        revised_cut_path = revised_cut_path[revised_cut_path != endnode]
                        skip_nodes.append(endnode)


                if len(revised_cut_path) <2 :
                    if verbose:
                        print("Could not used the revised endnodes path because empty")
                    skip_nodes = []
            else:
                revised_cut_path = cut_path

            if verbose:
                print(f"skip_nodes = {skip_nodes}")

            if edges_to_delete is None:
                if verbose: 
                    print("\nAttempting the doubling back check (symmetrical so don't need to check from both sides)")

                err_edges,edges,edges_double_back = ed.double_back_edges_path(curr_limb,
                                        path_to_check=revised_cut_path,
                                      double_back_threshold = double_back_threshold_at_first,
                                      comparison_distance = comparison_distance_double_back,
                                     offset=offset_double_back,
                                    verbose = verbose,
                                    skip_nodes=skip_nodes)


                if len(err_edges) > 0:

                    largest_double_back = np.argmax(edges_double_back)
                    largest_double_back_value = int(edges_double_back[largest_double_back])
                    winning_err_edge = edges[largest_double_back]

                    if verbose:
                        print(f"There were {len(err_edges)} edges that passed doubling back threshold of {double_back_threshold_at_first}")
                        print(f"Winning edge {winning_err_edge} had a doubling back of {edges_double_back[largest_double_back]}")

                    edges_to_delete = [winning_err_edge]
                    split_reason = f"double_back_angle_first_pass_{largest_double_back_value}_threshold_{double_back_threshold_at_first}"



        # -------------- 1/27: Will organize the high degree nodes by the average width of the branches they are touching --- #

        if len(high_degree_endpoint_coordinates) > 0 and edges_to_delete is None:

            sk_branches = [br.skeleton for br in curr_limb]

            high_degree_endpoint_coordinates_widths = []
            for coordinate in high_degree_endpoint_coordinates:
                coordinate_branches = np.sort(sk.find_branch_skeleton_with_specific_coordinate(sk_branches,coordinate))
                if len(coordinate_branches) > 0:
                    mean_width = np.mean([curr_limb[b].width_new["no_spine_median_mesh_center"] for b in coordinate_branches])
                else:
                    mean_width = np.inf

                high_degree_endpoint_coordinates_widths.append(mean_width)

            high_degree_order = np.argsort(high_degree_endpoint_coordinates_widths)

            if verbose:
                print(f"high_degree_endpoint_coordinates_widths = {high_degree_endpoint_coordinates_widths}")
                print(f"high_degree_order = {high_degree_order}")


            #for curr_high_degree_coord in high_degree_endpoint_coordinates:
            for curr_high_degree_coord_idx in high_degree_order:
                curr_high_degree_coord = high_degree_endpoint_coordinates[curr_high_degree_coord_idx]

                curr_high_degree_coord_list.append(curr_high_degree_coord)

                if verbose:
                    print(f"Picking {curr_high_degree_coord} high degree coordinates to cut")
                    print(f"curr_limb.deleted_edges 4={curr_limb.deleted_edges}")
                edges_to_delete_pre,edges_to_create_pre = ed.resolving_crossovers(curr_limb,
                                                        coordinate = curr_high_degree_coord,
                                                        offset=offset_high_degree,
                                                        comparison_distance=comparison_distance_high_degree,
                                                                                  match_threshold =match_threshold_high_degree,
                                                                                  plot_intermediates=plot_intermediates,
                                                        verbose = verbose,
                                                         branches_to_disconnect = cut_path,
                                                        
                                                        **kwargs
                                       )
                
                #raise Exception("Done Crossover")
                #  --- 1/19: Add in a check that will see if divides up the path, if not then set to empty  --- #

                if len(edges_to_delete_pre)>0:

                    cut_path_edges = np.vstack([cut_path[:-1],cut_path[1:]]).T


                    G = nx.from_edgelist(cut_path_edges)
                    if verbose:
                        print(f"nx.number_connected_components(G) before = {nx.number_connected_components(G)}")
                    G.remove_edges_from(edges_to_delete_pre)
                    
                    if verbose:
                        print(f"G.edges() = {G.edges()}")
                        print(f"G.nodes() = {G.nodes()}")
                        print(f"nx.number_connected_components(G) after = {nx.number_connected_components(G)}")

                    if (len(G.nodes()) < len(cut_path)) or nx.number_connected_components(G)>1:
                        if verbose:
                            print("Using the resolve crossover delete edges because will help seperate the path")
                        edges_to_delete = edges_to_delete_pre
                        edges_to_create = edges_to_create_pre
                        resolve_crossover_at_end = False
                        split_reason = "high_degree_node"
                        break

                    else:
                        if verbose:
                            print("NOT USING the resolve crossover delete edges because not help resolve the cut")
                        continue








        curr_limb.set_concept_network_directional(starting_node = cut_path[0],suppress_disconnected_errors=True)

        # ------------- 12 /28 addition that allows us to skip end nodes if too small ------------------
        skip_nodes = []
        if skip_small_soma_connectors:
            revised_cut_path = np.array(cut_path)
            for endnode in [cut_path[0],cut_path[-1]]:
                curr_sk_distance = sk.calculate_skeleton_distance(curr_limb[endnode].skeleton)
                if curr_sk_distance<small_soma_connectors_skeletal_threshold:
                    print(f"Skipping endnode {endnode} because skeletal distance was {curr_sk_distance} and threshold was {small_soma_connectors_skeletal_threshold}")
                    revised_cut_path = revised_cut_path[revised_cut_path != endnode]
                    skip_nodes.append(endnode)


            if len(revised_cut_path) <2 :
                if verbose:
                    print("Could not used the revised endnodes path because empty")
                skip_nodes = []

        if verbose:
            print(f"skip_nodes = {skip_nodes}")

        if edges_to_delete is None:
            if verbose: 
                print("\nAttempting the doubling back check (symmetrical so don't need to check from both sides)")

            err_edges,edges,edges_double_back = ed.double_back_edges_path(curr_limb,
                                    path_to_check=cut_path,
                                  double_back_threshold = double_back_threshold,
                                  comparison_distance = comparison_distance_double_back,
                                 offset=offset_double_back,
                                verbose = verbose,
                                skip_nodes=skip_nodes)


            if len(err_edges) > 0:

                largest_double_back = np.argmax(edges_double_back)
                largest_double_back_value = int(edges_double_back[largest_double_back])
                winning_err_edge = edges[largest_double_back]

                if verbose:
                    print(f"There were {len(err_edges)} edges that passed doubling back threshold of {double_back_threshold}")
                    print(f"Winning edge {winning_err_edge} had a doubling back of {edges_double_back[largest_double_back]}")

                edges_to_delete = [winning_err_edge]
                
                split_reason = f"double_back_angle_second_pass_{largest_double_back_value}_threshold_{double_back_threshold}" 

        if edges_to_delete is None:
            if verbose: 
                print("\nAttempting the width jump check (ARTIFICIALLY ATTEMPTING FROM BOTH SIDES)")
                print(f"width_jump_threshold = {width_jump_threshold}")

            possible_starting_nodes = [cut_path[0],cut_path[-1]]


            first_error_edges = []
            first_error_sizes = []
            first_error_locations = []

            """  Old way of doing 
            for s_node in possible_starting_nodes:

                curr_limb.set_concept_network_directional(starting_node = s_node,suppress_disconnected_errors=True)

                if cut_path[0] != s_node:
                    cut_path = np.flip(cut_path)
                    if cut_path[0] != s_node:
                        raise Exception("Neither of cut path end nodes are starting node")

                err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                                                                            path_to_check=cut_path,
                                                                            width_jump_threshold=width_jump_threshold,
                                                                            offset=offset,
                                                                            verbose=verbose,
                                                                            skip_nodes=skip_nodes
                                        )

                if verbose:
                    print(f"Path starting at {s_node} had err_edges: {err_edges}")

                err_edges_mask = edges_width_jump>=width_jump_threshold

                if np.any(err_edges_mask):
                    first_error_edges.append(edges[err_edges_mask][0])
                    first_error_sizes.append(edges_width_jump[err_edges_mask][0])
                else:
                    first_error_edges.append(None)
                    first_error_sizes.append(-np.inf)
            """

            # ----------- 1/19: New Way of Splitting ------------- #
            curr_limb.set_concept_network_directional(starting_node = cut_path[0],suppress_disconnected_errors=True)



            err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                                                                        path_to_check=cut_path,
                                                                        width_jump_threshold=width_jump_threshold,
                                                                        offset=offset_double_back,
                                                                        verbose=verbose,
                                                                        skip_nodes=skip_nodes
                                    )

            # 1) Doing the forwards way

            err_edges_mask = np.where(edges_width_jump>=width_jump_threshold)[0]

            if np.any(err_edges_mask):
                first_error_edges.append(edges[err_edges_mask[0]])
                first_error_sizes.append(edges_width_jump[err_edges_mask][0])
                first_error_locations.append(err_edges_mask[0])
            else:
                first_error_edges.append(None)
                first_error_sizes.append(-np.inf)
                first_error_locations.append(np.inf)

            # 2) Doing the backwards way

            edges_width_jump_flipped = np.flip(edges_width_jump)
            edges_flipped = np.flip(edges)

            err_edges_mask = np.where(edges_width_jump_flipped <=-width_jump_threshold)[0]

            if np.any(err_edges_mask):
                first_error_edges.append(edges_flipped[err_edges_mask[0]])
                first_error_sizes.append(-1*(edges_width_jump_flipped[err_edges_mask[0]]))
                first_error_locations.append(err_edges_mask[0])
            else:
                first_error_edges.append(None)
                first_error_sizes.append(-np.inf)
                first_error_locations.append(np.inf)

            """
            Pseudocode: 
            1) Check if both error edges are not empty
            2) Get the starting error 


            """
            if (not first_error_edges[0] is None) or (not first_error_edges[1] is None):
                """  Old way that did it on the biggest size, new way does it on the first jump
                winning_path = np.argmax(first_error_sizes)
                """

                winning_path = np.argmin(first_error_locations)
                winning_err_edge = first_error_edges[winning_path]
                if verbose: 
                    print(f"first_error_sizes = {first_error_sizes}, first_error_locations = {first_error_locations}, winning_path = {winning_path}")


                edges_to_delete = [winning_err_edge]
                split_reason = f"width_jump_{int(first_error_sizes[winning_path])}_threshold_{width_jump_threshold}" 
            else:
                if verbose:
                    print(f"Did not find an error edge in either of the paths")


        # need to resolve cross over at this point
        if resolve_crossover_at_end and (not edges_to_delete is None):
            if verbose:
                print(f"inside resolve_crossover_at_end at end: \ncurrent edges_to_delete = {edges_to_delete} ")
            cut_e = edges_to_delete[0]
            suggested_cut_point = sk.shared_endpoint(curr_limb[cut_e[0]].skeleton,
                                                    curr_limb[cut_e[1]].skeleton)

            edges_to_delete_new,edges_to_create_new = ed.resolving_crossovers(
                curr_limb,
                coordinate=suggested_cut_point,
                offset=offset_high_degree,
                comparison_distance=comparison_distance_high_degree,
                match_threshold =match_threshold_high_degree,
                verbose = verbose,
                edges_to_avoid = edges_to_delete,
                return_subgraph=False,
                branches_to_disconnect = cut_e,
                **kwargs)
            
            if verbose:
                print(f"After Resolving crossovers at end: ")
                print(f"edges_to_delete_new = {edges_to_delete_new}")
                print(f"edges_to_create_new = {edges_to_create_new}")

            
            if len(edges_to_delete_new) > 0:
                edges_to_delete = np.vstack([edges_to_delete,edges_to_delete_new])
                edges_to_delete = list(np.unique(np.sort(np.array(edges_to_delete),axis=1),axis=0))



            if not edges_to_create is None:
                edges_to_create += edges_to_create_new
                edges_to_create = list(np.unique(np.sort(np.array(edges_to_create),axis=1),axis=0))
            else:
                edges_to_create = edges_to_create_new

            # want to limit the edges to only those with one of the disconnected edges in it
            edges_to_create_final = []
            for e_c1 in edges_to_create:
                if len(np.intersect1d(e_c1,cut_e)) == 1:
                    edges_to_create_final.append(e_c1)
                else:
                    if verbose:
                        print(f"Rejecting creating edge {e_c1} becuase did not involve only 1 node in the deleted edge")
            edges_to_create = edges_to_create_final




    curr_limb,edges_to_create_final = pru.cut_limb_network_by_edges(curr_limb,
                                                    edges_to_delete,
                                                    edges_to_create,
                                                    return_accepted_edges_to_create=True,
                                                    verbose=verbose)
    
    edges_to_create=edges_to_create_final
    
        

    if verbose:
        conn_comp = list(nx.connected_components(curr_limb.concept_network))
        print(f"Number of connected components = {len(conn_comp)}")
        for j,k in enumerate(conn_comp):
            print(f"Comp {j} = {k}")
    if nu.is_array_like(edges_to_delete):
        edges_to_delete = list(edges_to_delete)
    if nu.is_array_like(edges_to_create):
        edges_to_create = list(edges_to_create)
    if nu.is_array_like(removed_branches):
        removed_branches = list(removed_branches)
        
    if return_split_reasons:
        return edges_to_delete,edges_to_create,curr_limb,removed_branches,curr_high_degree_coord_list,split_reason
    else:
        return edges_to_delete,edges_to_create,curr_limb,removed_branches,curr_high_degree_coord_list

def get_all_coordinate_suggestions(suggestions,concatenate=True,
                                  voxel_adjustment=True):
    """
    Getting all the coordinates where there should be cuts
    
    
    """
    if len(suggestions) == 0:
        return []
    
    all_coord = []
    for limb_idx,sugg_v in suggestions.items():
        curr_coords = np.array(get_attribute_from_suggestion(suggestions,curr_limb_idx=limb_idx,
                                 attribute_name="coordinate_suggestions"))
        if voxel_adjustment and len(curr_coords)>0:
            curr_coords = curr_coords/np.array([4,4,40])
                
        if len(curr_coords) > 0:
            all_coord.append(curr_coords)
    
    if voxel_adjustment:
        all_coord
    if concatenate:
        if len(all_coord) > 0:
            return list(np.unique(np.vstack(all_coord),axis=0))
        else:
            return list(all_coord)
    else:
        return list(all_coord)
    
    
def get_n_paths_not_cut(limb_results):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return 0
    
    n_paths_not_cut = 0

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_not_cut"]) > 0:
                n_paths_not_cut += 1   
                
    return n_paths_not_cut

def split_type_from_title(title):
    somas = title.split(" from ")
    somas = [k.replace(" ","").split("_")[0] for k in somas]
    if somas[0] == somas[1]:
        return 'multi-touch'
    else:
        return 'multi-soma'
    
def get_n_paths_cut(
    limb_results,
    return_multi_touch_multi_soma = False,
    verbose = False,):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return 0
    
    n_paths_cut = 0
    n_paths_multi_soma = 0
    n_paths_multi_touch = 0

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_cut"]) > 0:
                n_paths_cut += 1  
                
                if split_type_from_title(path_cut_info['title']) == 'multi-touch':
                    n_paths_multi_touch += 1
                else:
                    n_paths_multi_soma += 1
                    
    if verbose:
        print(f"n_paths_multi_soma = {n_paths_multi_soma}")
        print(f"n_paths_multi_touch = {n_paths_multi_touch}")
        print(f"n_paths_cut = {n_paths_cut}")
        
        
        
    if return_multi_touch_multi_soma:
        return n_paths_multi_touch,n_paths_multi_soma
    return n_paths_cut

def get_all_cut_and_not_cut_path_coordinates(limb_results,voxel_adjustment=True,
                                            ):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return [],[]
    
    cut_path_coordinates = []
    not_cut_path_coordinates = []

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_cut"]) > 0:
                cut_path_coordinates.append(np.vstack(path_cut_info["paths_cut"]))
            if len(path_cut_info["paths_not_cut"]) > 0:
                not_cut_path_coordinates.append(np.vstack(path_cut_info["paths_not_cut"]))

    if len(cut_path_coordinates)>0:
        total_cut_path_coordinates = np.vstack(cut_path_coordinates)
    else:
        total_cut_path_coordinates = []

    if len(not_cut_path_coordinates)>0:
        total_not_cut_path_coordinates = list(np.vstack(not_cut_path_coordinates))
    else:
        total_not_cut_path_coordinates = []


    if len(total_not_cut_path_coordinates)>0 and len(total_cut_path_coordinates)>0:
        total_cut_path_coordinates_revised = list(nu.setdiff2d(total_cut_path_coordinates,total_not_cut_path_coordinates))
    else:
        total_cut_path_coordinates_revised = list(total_cut_path_coordinates)
        
    if voxel_adjustment:
        voxel_divider = np.array([4,4,40])
        total_cut_path_coordinates_revised = [k/voxel_divider for k in total_cut_path_coordinates_revised if len(k)>0]
        total_not_cut_path_coordinates = [k/voxel_divider for k in total_not_cut_path_coordinates if len(k)>0]
        
    
        
    return list(total_cut_path_coordinates_revised),list(total_not_cut_path_coordinates)

    
    
def get_attribute_from_suggestion(suggestions,curr_limb_idx=None,
                                 attribute_name="edges_to_delete"):
    if type(suggestions) == dict:
        if curr_limb_idx is None:
            raise Exception("No specified limb idx when passed all the suggestions")
        suggestions = suggestions[curr_limb_idx]
        
    total_attribute = []
    for cut_s in suggestions:
        total_attribute += cut_s[attribute_name]
        
    return total_attribute

def get_edges_to_delete_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="edges_to_delete")
def get_edges_to_create_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="edges_to_create")
def get_removed_branches_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="removed_branches")
    

def cut_limb_network_by_suggestions(curr_limb,
                                   suggestions,
                                   curr_limb_idx=None,
                                    return_copy=True,
                                   verbose=False):
    if type(suggestions) == dict:
        if curr_limb_idx is None:
            raise Exception("No specified limb idx when passed all the suggestions")
        suggestions = suggestions[curr_limb_idx]
    
    return cut_limb_network_by_edges(curr_limb,
                                    edges_to_delete=pru.get_edges_to_delete_from_suggestion(suggestions),
                                    edges_to_create=pru.get_edges_to_create_from_suggestion(suggestions),
                                     removed_branches=pru.get_removed_branches_from_suggestion(suggestions),
                                    verbose=verbose,
                                     return_copy=return_copy
                                    )
    
def cut_limb_network_by_edges(curr_limb,
                                    edges_to_delete=None,
                                    edges_to_create=None,
                                    removed_branches=[],
                                    perform_edge_rejection=False,
                                    return_accepted_edges_to_create=False,
                                    return_copy=True,
                                    return_limb_network = False,
                                    verbose=False):
    if return_copy:
        curr_limb = copy.deepcopy(curr_limb)
    if (not removed_branches is None) and len(removed_branches)>0:
        curr_limb = pru.collapse_branches_on_limb(curr_limb,
                                                 branch_list=removed_branches)
    
    if not edges_to_delete is None:
        if verbose:
            print(f"edges_to_delete (cut_limb_network) = {edges_to_delete}")
        curr_limb.concept_network.remove_edges_from(edges_to_delete)
        
        curr_limb.deleted_edges += edges_to_delete
        
    #apply the winning cut
    accepted_edges_to_create = []
    if not edges_to_create is None:
        if verbose:
            print(f"edges_to_create = {edges_to_create}")
            
        if perform_edge_rejection:
            for n1,n2 in edges_to_create:
                curr_limb.concept_network.add_edge(n1,n2)
                counter = 0
                for d1,d2 in edges_to_delete:
                    try:
                        ex_path = np.array(nx.shortest_path(curr_limb.concept_network,d1,d2))
                    except:
                        pass
                    else:
                        counter += 1
                        break
                if counter > 0:
                    curr_limb.concept_network.remove_edge(n1,n2)
                    if verbose:
                        print(f"Rejected edge ({n1,n2})")
                else:
                    if verbose:
                        print(f"Accepted edge ({n1,n2})")
                    accepted_edges_to_create.append([n1,n2])
        else:
            accepted_edges_to_create = edges_to_create
            curr_limb.concept_network.add_edges_from(accepted_edges_to_create)

        #add them to the properties
        curr_limb.created_edges += accepted_edges_to_create
    
    if return_accepted_edges_to_create:
        return curr_limb,accepted_edges_to_create
    
    if return_limb_network:
        return curr_limb.concept_network
    else:
        return curr_limb


def soma_connections_from_split_title(title):
    split_title = title.split(" ")
    return split_title[0],split_title[-2]
def soma_names_from_split_title(title,return_idx=False):
    split_title = title.split(" ")
    soma_idx_1 = split_title[0].split("_")[0]
    soma_idx_2 = split_title[-2].split("_")[0]
    
    return_value = [soma_idx_1,soma_idx_2]
    if return_idx:
        return_value  = [nru.get_soma_int_name(k) for k in return_value]
    return return_value


def multi_soma_split_suggestions(
    neuron_obj,
    verbose=False,
    max_iterations=100,
    plot_suggestions=False,
    plot_intermediates=False,
    plot_suggestions_scatter_size=0.4,
    remove_segment_threshold = None,
    plot_cut_coordinates = False,
    only_multi_soma_paths = False, # to restrict to only different soma-soma paths
    default_cut_edge = "last",#None,
    debug = False,

    #for red blue suggestions
    output_red_blue_suggestions = True,
    split_red_blue_by_common_upstream = True,

    one_hop_downstream_error_branches_max_distance=4_000,
    offset_distance_for_points=3_000,#500,
    n_points=1,#3,
    plot_final_blue_red_points = False,

    only_outermost_branches = True,
    include_removed_branches = False,
    min_error_downstream_length_total = 5_000,
    apply_valid_upstream_branches_restriction = True,
    debug_red_blue = False,
    **kwargs):
    """
    Purpose: To come up with suggestions for splitting a multi-soma

    Pseudocode: 

    1) Iterate through all of the limbs that need to be processed
    2) Find the suggested cuts until somas are disconnected or failed
    3) Optional: Visualize the nodes and their disconnections

    """
    if remove_segment_threshold is None:
        remove_segment_threshold = remove_segment_threshold_global
    
    #sprint("inside multi_soma_split_suggestions")

    multi_soma_limbs = nru.multi_soma_touching_limbs(neuron_obj)
    multi_touch_limbs = nru.same_soma_multi_touching_limbs(neuron_obj)
    
    if only_multi_soma_paths:
        multi_touch_limbs = []
    
    if verbose: 
        print(f"multi_soma_limbs = {multi_soma_limbs}")
        print(f"multi_touch_limbs = {multi_touch_limbs}")
    
    total_limbs_to_process = np.unique(
        np.concatenate(
            [multi_soma_limbs,multi_touch_limbs]
        )
    )

    limb_results = dict()
    red_blue_split_dict = dict()
    
    red_blue_counter = 0
    

    for curr_limb_idx in total_limbs_to_process:
        curr_limb_idx = int(curr_limb_idx)
        if verbose:
            print(f"\n\n -------- Working on limb {curr_limb_idx}------------")
#         if curr_limb_idx != 2:
#             continue
        curr_limb_copy = copy.deepcopy(neuron_obj[curr_limb_idx])
        
        if output_red_blue_suggestions:
            limb_obj_collapsed = copy.deepcopy(neuron_obj[curr_limb_idx])
            red_blue_split_dict[curr_limb_idx] = []

        #----- starting the path cutting ------ #
        """
        Find path to cut:
        1) Get the concept network
        2) Get all of the starting nodes for somas
        3) Get the shortest path between each combination of starting nodes

        """
        """
        OLD METHOD OF FINDING THE COMBINATIONS
        #2) Get all of the starting nodes for somas
        all_starting_nodes = [k["starting_node"] for k in curr_limb_copy.all_concept_network_data]

        starting_node_combinations = list(itertools.combinations(all_starting_nodes,2))
        starting_node_combinations = nu.unique_non_self_pairings(starting_node_combinations)
        """
        
        starting_node_combinations= nru.starting_node_combinations_of_limb_sorted_by_microns_midpoint(
            neuron_obj,
            limb_idx = curr_limb_idx,
            verbose = False,
            only_multi_soma_paths = only_multi_soma_paths,
            return_soma_names = False,
        )

        if verbose or debug:
            print(f"Starting combinations to process = {starting_node_combinations}")

        results = []

        
        
        for st_n_1,st_n_2 in starting_node_combinations:
            local_results = dict(starting_node_1=st_n_1,
                                starting_node_2 = st_n_2)
            st_n_1_soma,st_n_1_soma_group_idx = curr_limb_copy.get_soma_by_starting_node(st_n_1),curr_limb_copy.get_soma_group_by_starting_node(st_n_1)
            st_n_2_soma,st_n_2_soma_group_idx = curr_limb_copy.get_soma_by_starting_node(st_n_2),curr_limb_copy.get_soma_group_by_starting_node(st_n_2)

            soma_title = f"S{st_n_1_soma}_{st_n_1_soma_group_idx} from S{st_n_2_soma}_{st_n_2_soma_group_idx} "
            local_results["title"] = soma_title
            
            

            total_soma_paths_to_cut = []
            total_soma_paths_to_add = []
            total_removed_branches = []
            total_split_reasons = dict()
            
            # need to keep cutting until no path for them
            if verbose:
                print(f"\n\n---- working on disconnecting {st_n_1} and {st_n_2}")
                print(f"---- This disconnects {soma_title} ")

            if debug:
                print(f"Currnet title = {soma_title}")
            counter = 0
            success = False
            
            high_degree_endpoint_coordinates_tried = []
            local_paths_cut = []
            local_paths_not_cut = []
            
            while True:
                if verbose:
                    print(f" Cut iteration {counter}")
                seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
                if verbose:
                    print(f"Total number of graphs at the end of the split BEFORE DIRECTIONAL = {len(seperated_graphs)}")
                    
                    
                curr_limb_copy.set_concept_network_directional(starting_node=st_n_1,suppress_disconnected_errors=True)
                
                
                seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
                if verbose:
                    print(f"Total number of graphs at the end of the split AFTER DIRECTIONAL = {len(seperated_graphs)}")
                try:
                    soma_to_soma_path = np.array(nx.shortest_path(curr_limb_copy.concept_network,st_n_1,st_n_2))
                    
                    
                    
                    
                except:
                    if verbose or debug:
                        print("No valid path so moving onto the next connection")
                    
                    success = True
                    break
                
                #try and figure out the endpoints along the path
                local_paths_cut.append(nru.skeleton_points_along_path(curr_limb_copy,soma_to_soma_path))
                
                if verbose:
                    print(f"Shortest path = {list(soma_to_soma_path)}")

                # say we found the cut node to make
                
                if verbose:
                    print(f"remove_segment_threshold = {remove_segment_threshold}")
                    print(f"high_degree_endpoint_coordinates_tried = {high_degree_endpoint_coordinates_tried}")
                
                (cut_edges, 
                added_edges, 
                curr_limb_copy,
                removed_branches,
                curr_high_degree_coord,
                 curr_split_reason,
                ) = pru.get_best_cut_edge(curr_limb_copy,soma_to_soma_path,
                                                                remove_segment_threshold=remove_segment_threshold,
                                                                               verbose=verbose,
                                      high_degree_endpoint_coordinates_tried=high_degree_endpoint_coordinates_tried,
                                                                plot_intermediates=plot_intermediates,
                                          return_split_reasons=True,
                                                                              **kwargs)
                
                
                high_degree_endpoint_coordinates_tried += curr_high_degree_coord
                
                if verbose or debug:
                    print(f"curr_limb_copy.deleted_edges = {curr_limb_copy.deleted_edges}")
                    print(f"curr_limb_copy.created_edges = {curr_limb_copy.created_edges}")
                
                if verbose or debug:
                    print(f"After get best cut: cut_edges = {cut_edges}, added_edges = {added_edges}")
                
                
                if cut_edges is None:
                    if verbose:
                        print("***** there was no suggested cut for this limb even though it is still connnected***")

                    
                    if default_cut_edge is not None:
                        if verbose:
                            print(f"--> So Setting the default_cut_edge to {default_cut_edge}")
                        if default_cut_edge == "first":
                            cut_edges = [[soma_to_soma_path[0],soma_to_soma_path[1]]]
                            curr_split_reason = "default_edge_first"
                        elif default_cut_edge == "last":
                            cut_edges = [[soma_to_soma_path[-2],soma_to_soma_path[-1]]]
                            curr_split_reason = "default_edge_last"
                        else:
                            raise Exception(f"Unimplemented default_cut_edge = {default_cut_edge}")
                            
                        curr_limb_copy,_ = pru.cut_limb_network_by_edges(curr_limb_copy,
                                                    cut_edges,
                                                    [],
                                                    return_accepted_edges_to_create=True,
                                                    verbose=verbose)
                        #print(f"cut edges after returned: {cut_edges}")
                    else:
                        break
                
                

                #------ 1/8 Addition: check if any new edges cut and if not then break
                #print(f"cut_edges = {cut_edges}, total_soma_paths_to_cut = {total_soma_paths_to_cut}")
                edge_diff = nu.setdiff2d(cut_edges,total_soma_paths_to_cut)
                if verbose:
                    print(f"edge_diff = {edge_diff}")
                if len(nu.intersect2d(cut_edges,total_soma_paths_to_cut)) == len(cut_edges):
                    if verbose:
                        print("**** there were no NEW suggested cuts")
                    break
                
                if verbose:
                    print(f"total_soma_paths_to_cut = {total_soma_paths_to_cut}")
                if not cut_edges is None:
                    suggested_cut_points = []
                    for cut_e in cut_edges:
                        shared_endpoints = sk.shared_endpoint(curr_limb_copy[cut_e[0]].skeleton,
                                                        curr_limb_copy[cut_e[1]].skeleton,
                                                             return_possibly_two=True)
                        if shared_endpoints.ndim == 1:
                            suggested_cut_points.append(shared_endpoints)
                        else:
                            for s_e in shared_endpoints:
                                if len(nu.matching_rows(high_degree_endpoint_coordinates_tried,s_e)) > 0:
                                    suggested_cut_points.append(s_e)

                    suggested_cut_points = np.unique(np.array(suggested_cut_points).reshape(-1,3),axis=0)

                    if curr_split_reason not in total_split_reasons.keys():
                        total_split_reasons[curr_split_reason] = []
                    
                    if len(suggested_cut_points) > 0:
                        total_split_reasons[curr_split_reason] += list(suggested_cut_points)
                    
                    if plot_cut_coordinates:
                        
                        path_skeleton_points = nru.skeleton_points_along_path(curr_limb_copy,
                                                                              skeletal_distance_per_coordinate=3000,
                                                                             branch_path=soma_to_soma_path)
                        
                        if verbose:
                            print(f"\n\nsuggested_cut_points = {suggested_cut_points}\n\n")
                        nviz.plot_objects(curr_limb_copy.mesh,
                                          skeletons=[curr_limb_copy.skeleton],
                                          scatters = [path_skeleton_points,suggested_cut_points],
                                          scatters_colors=["blue","red"],
                                          scatter_size = 0.3
                        
                        )
                        
                    
                    
                    total_soma_paths_to_cut += cut_edges
                    
                    # ---------- 11/30: Calculate the coordinates affected by the cut edge
                    
                    
                    
                if not added_edges is None:
                    total_soma_paths_to_add += added_edges
                if len(removed_branches)>0:
                    total_removed_branches += removed_branches
                    
                
                if verbose:
                    print(f"-----------counter = {counter}------------")
                counter += 1

                if counter > max_iterations:
                    if verbose:
                        print(f"Breaking because hit max iterations {max_iterations}")
                    
                    break

            if not success:
                local_paths_not_cut.append(nru.skeleton_points_along_path(curr_limb_copy,soma_to_soma_path))
                   
            local_results["edges_to_delete"] = total_soma_paths_to_cut
            local_results["edges_to_create"] = total_soma_paths_to_add
            local_results["removed_branches"] = total_removed_branches

            suggested_cut_points = []
            for cut_e in total_soma_paths_to_cut:
                high_degree_endpoint_coordinates_tried
                shared_endpoints = sk.shared_endpoint(curr_limb_copy[cut_e[0]].skeleton,
                                                curr_limb_copy[cut_e[1]].skeleton,
                                                     return_possibly_two=True)
                if shared_endpoints.ndim == 1:
                    suggested_cut_points.append(shared_endpoints)
                else:
                    for s_e in shared_endpoints:
                        if len(nu.matching_rows(high_degree_endpoint_coordinates_tried,s_e)) > 0:
                            suggested_cut_points.append(s_e)

            local_results["coordinate_suggestions"] =suggested_cut_points
            local_results["successful_disconnection"] = success
            local_results["paths_not_cut"] = local_paths_not_cut
            local_results["paths_cut"] = local_paths_cut
            local_results["split_reasons"] = total_split_reasons
            results.append(local_results)
            
            # ------------ Part where going to output the red/blue points -------------#
            """
            Psuedocode:
            1) Decide whether to output if need red blue points (if have )
            2) 


            """
            if output_red_blue_suggestions:
                
                if (len(local_results["edges_to_delete"]) == 0
                    and len(local_results["edges_to_create"]) == 0):
                    if verbose:
                        print(f"\n Not doing red blue splits because no paths to delete or add")
                    continue


                if verbose:
                     print(f"\n\n**** Computing red blue splits****")

                soma_starts = [st_n_1,st_n_2]
                local_red_blue = dict(soma_path = soma_title,
                                         node_path = soma_starts)
                
                if debug_red_blue:
                    su.compressed_pickle(limb_obj_collapsed,f"pre_limb_obj_collapsed_{red_blue_counter}")

                limb_obj_collapsed = pru.collapse_branches_on_limb(limb_obj_collapsed,
                                                     branch_list=local_results["removed_branches"],
                                                     plot_new_limb=False,
                                                    verbose=False)
                
#                 if verbose:
#                     print(f"RIGHT BEFORE RED BLUE SPLIT: curr_limb_copy.created_edges = {curr_limb_copy.created_edges}")
#                     print(f"limb_obj_collapsed.created_edges = {limb_obj_collapsed.created_edges}")
#                 limb_obj_collapsed.created_edges = curr_limb_copy.created_edges
#                 if verbose:
#                     print(f"limb_obj_collapsed.created_edges = {limb_obj_collapsed.created_edges}")
                
                

                conn_comps = xu.connected_components(curr_limb_copy.concept_network)
                conn_comp_splits = [xu.connected_component_with_node(k,connected_components=conn_comps,return_only_one=True)
                            for k in soma_starts]

                if verbose:
                    print(f"Conn comp size before expansion = {[len(k) for k in conn_comp_splits]}")

                if verbose:
                    print(f"Not expanding to removed branches")

                all_red_blue_splits = dict() 
                soma_idxs = soma_names_from_split_title(soma_title,return_idx=True)


                for j,sm_name in enumerate(soma_connections_from_split_title(soma_title)):
                    if verbose:
                        print(f"\n--Doing Red/Blue splits for {sm_name}")

#                     limb_obj_collapsed.set_concept_network_directional(starting_soma = soma_idxs[1 - j],
#                                                                       suppress_disconnected_errors=True)
                    
                    limb_obj_collapsed.set_concept_network_directional(starting_node=soma_starts[1-j],
                                                                       suppress_disconnected_errors=True,
                                                                       no_cycles=False,
                                                                      )
        
                    if apply_valid_upstream_branches_restriction:
                        valid_upstream_branches_restriction = conn_comp_splits[1-j]
                    else:
                        valid_upstream_branches_restriction = None
                        
                    error_branches_for_restriction = np.setdiff1d(np.union1d(conn_comp_splits[j],edge_diff.ravel()),
                                                                  conn_comp_splits[1-j])
                                                               
                        
                    if verbose:
                        print(f"error_branches={error_branches_for_restriction}")
                        print(f"valid_upstream_branches_restriction = {valid_upstream_branches_restriction}")
                        #print(f"group_all_conn_comp_together = {group_all_conn_comp_together}")

                    red_blue_splits = pru.limb_errors_to_cancel_to_red_blue_group(
                                    limb_obj_collapsed,
                                    #error_branches=conn_comp_splits[j],
                                    error_branches = error_branches_for_restriction,
                                    valid_upstream_branches_restriction = valid_upstream_branches_restriction,
                                    group_all_conn_comp_together = True,
                                    neuron_obj=neuron_obj,
                                    verbose = False,
                                    plot_final_blue_red_points=plot_final_blue_red_points,
                                    offset_distance_for_points=offset_distance_for_points,
                                    one_hop_downstream_error_branches_max_distance=one_hop_downstream_error_branches_max_distance,
                                    n_points=n_points,
                                    only_outermost_branches = only_outermost_branches,
                                    min_error_downstream_length_total = min_error_downstream_length_total,
                                    limb_idx = curr_limb_idx,
                                    split_red_blue_by_common_upstream=split_red_blue_by_common_upstream)
                    if verbose:
                        print(f"# of red_blue_splits made = {len(red_blue_splits)}")
                    all_red_blue_splits[sm_name] = red_blue_splits
                
                
                red_blue_split_dict[curr_limb_idx].append(all_red_blue_splits)
                
                if debug_red_blue:
                    su.compressed_pickle(curr_limb_copy,f"curr_limb_copy_{red_blue_counter}")
                    su.compressed_pickle(limb_obj_collapsed,f"limb_obj_collapsed_{red_blue_counter}")
                    red_blue_counter += 1
            


            
            
            

        seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
        if verbose:
            print(f"Total number of graphs at the end of the split = {len(seperated_graphs)}: {[np.array(list(k)) for k in seperated_graphs]}")
            

        limb_results[curr_limb_idx] = results
        
        
        
        
    if plot_suggestions:
        nviz.plot_split_suggestions_per_limb(neuron_obj,
                                    limb_results,
                                    scatter_size = plot_suggestions_scatter_size)
        
    if output_red_blue_suggestions:
        return limb_results,red_blue_split_dict
    else:
        return limb_results


def split_suggestions_to_concept_networks(neuron_obj,limb_results,
                                         apply_changes_to_limbs=False):
    """
    Will take the output of the multi_soma_split suggestions and 
    return the concept network with all fo the cuts applied
    
    """
    
    
    new_concept_networks = dict()
    for curr_limb_idx,path_cut_info in limb_results.items():
        if not apply_changes_to_limbs:
            limb_cp = copy.deepcopy(neuron_obj[curr_limb_idx])
        else:
            limb_cp = neuron_obj[curr_limb_idx]
        #limb_nx = nx.Graph(neuron_obj[curr_limb_idx].concept_network)
        
        curr_limb = pru.cut_limb_network_by_suggestions(limb_cp,
                                                       path_cut_info,
                                                       return_copy=not apply_changes_to_limbs)
        
        
        new_concept_networks[curr_limb_idx] = curr_limb.concept_network
        
    
    return new_concept_networks


def split_suggestions_to_concept_networks_old(neuron_obj,limb_results,
                                         apply_changes_to_limbs=False):
    """
    Will take the output of the multi_soma_split suggestions and 
    return the concept network with all fo the cuts applied
    
    """
    
    
    new_concept_networks = dict()
    for curr_limb_idx,path_cut_info in limb_results.items():
        if not apply_changes_to_limbs:
            limb_cp = copy.deepcopy(neuron_obj[curr_limb_idx])
        else:
            limb_cp = neuron_obj[curr_limb_idx]
        #limb_nx = nx.Graph(neuron_obj[curr_limb_idx].concept_network)
        for cut in path_cut_info:
            limb_cp=pru.collapse_branches_on_limb(limb_cp,branch_list=cut["removed_branches"])
            limb_cp.concept_network.remove_edges_from(cut["edges_to_delete"])
            limb_cp.concept_network.add_edges_from(cut["edges_to_create"])
        new_concept_networks[curr_limb_idx] = limb_cp.concept_network
        
    raise Exception("Debugging the cuts")
    return new_concept_networks


# --------------- Functions that do the actual limb and Neuron Splitting --------- #


def split_neuron_limb_by_seperated_network(neuron_obj,
                     curr_limb_idx,
                    seperate_networks=None,
                    cut_concept_network=None,#could send a cut concept network that hasn't been seperated
                    split_current_concept_network=True, #if the stored concept network has already been cut then use that
                    error_on_multile_starting_nodes=True,
                    delete_limb_if_empty=True,
                     verbose = False):
    """
    Purpose: To Split a neuron limb up into sepearte limb graphs specific

    Arguments:
    neuron_obj
    seperated_graphs
    limb_idx


    """
    
    
    # -------- Getting the mesh and correspondence information --------- #
    """
    1) Assemble all the faces of the nodes and concatenate them
    - copy the data into the new limb correspondence
    - save the order they were concatenated in the new limb correspondence
    - copy of 
    2) Use the concatenated faces idx to obtain the new limb mesh
    3) index the concatenated faces idx into the limb.mesh_face_idx to get the neew limb.mesh_face_idx
    """
    if seperate_networks is None:
        if split_current_concept_network:
            seperated_graphs = [list(k) for k in nx.connected_components(neuron_obj[curr_limb_idx].concept_network)]
        elif cut_concept_network is not None:
            seperated_graphs = [list(k) for k in nx.connected_components(cut_concept_network)]
        else:
            seperated_graphs = [neuron_obj[curr_limb_idx].concept_network]
    else:
        seperated_graphs = seperate_networks
        
        
    new_limb_data = []
    curr_limb = neuron_obj[curr_limb_idx]
    
    if len(seperated_graphs) == 0:
        sep_graph_data = dict()
        curr_labels = ["empty"]
        sep_graph_data["Limb_obj"] = nru.empty_limb_object(labels=curr_labels)
        sep_graph_data["limb_correspondence"] = dict()
        sep_graph_data["limb_labels"] = curr_labels
        sep_graph_data["limb_concept_networks"] = dict()
        sep_graph_data["limb_network_stating_info"]=dict()
        sep_graph_data["limb_meshes"] = tu.empty_mesh()
        new_limb_data.append(sep_graph_data)
    else:
        for seg_graph_idx,sep_G in enumerate(seperated_graphs):

            curr_subgraph = list(sep_G)

            #will store all of the relevant info in the 
            sep_graph_data = dict()

            if len(curr_subgraph) == 0:
                curr_labels = ["empty"]
                sep_graph_data["Limb_obj"] = nru.empty_limb_object(labels=curr_labels)
                sep_graph_data["limb_correspondence"] = dict()
                sep_graph_data["limb_labels"] = curr_labels
                sep_graph_data["limb_concept_networks"] = dict()
                sep_graph_data["limb_network_stating_info"]=dict()
                sep_graph_data["limb_meshes"] = tu.empty_mesh()
                new_limb_data.append(sep_graph_data)
                continue


            if verbose:
                print(f"\n\n----Working on seperate_graph {seg_graph_idx}----")




            fixed_node_objects = dict()

            
            build_from_mesh_face_idx = False
            
            
            if build_from_mesh_face_idx:
                limb_face_idx_concat = []
            else:
                new_limb_mesh_list = []
                
            face_counter = 0
            old_node_to_new_node_mapping = dict()
            
            
            for i,n_name in enumerate(curr_subgraph):
                #store the mapping for the new names
                old_node_to_new_node_mapping[n_name] = i

                fixed_node_objects[i] = copy.deepcopy(curr_limb[n_name])
                
                #sometimes these can't be trusted....
                if build_from_mesh_face_idx:
                    curr_mesh_face_idx = fixed_node_objects[i].mesh_face_idx
                    limb_face_idx_concat.append(curr_mesh_face_idx)
                    fixed_node_objects[i].mesh_face_idx = np.arange(face_counter,face_counter+len(curr_mesh_face_idx))
                    face_counter += len(curr_mesh_face_idx)
                else:
                    new_limb_mesh_list.append( fixed_node_objects[i].mesh)
                    curr_new_face_idx = np.arange(face_counter,face_counter+len(fixed_node_objects[i].mesh.faces))
                    fixed_node_objects[i].mesh_face_idx = curr_new_face_idx
                    face_counter += len(curr_new_face_idx)

            if build_from_mesh_face_idx:
                total_limb_face_idx = np.concatenate(limb_face_idx_concat)
                new_limb_mesh = curr_limb.mesh.submesh([total_limb_face_idx],append=True,repair=False)
            else:
                new_limb_mesh = tu.combine_meshes(new_limb_mesh_list)
            


            new_limb_mesh_face_idx = tu.original_mesh_faces_map(neuron_obj.mesh, new_limb_mesh,
                                       matching=True,
                                       print_flag=False)

            #recovered_new_limb_mesh = neuron_obj.mesh.submesh([new_limb_mesh_face_idx],append=True,repair=False)
            sep_graph_data["limb_meshes"] = new_limb_mesh

            # ------- How to get the new concept network starting info --------- #

            #get all of the starting dictionaries that match a node in the subgraph
            curr_all_concept_network_data = [k for k in curr_limb.all_concept_network_data if k["starting_node"] in list(curr_subgraph)]
            if len(curr_all_concept_network_data) > 1:
                warning_string = f"There were more not exactly one starting dictinoary: {curr_all_concept_network_data} "
                if error_on_multile_starting_nodes:
                    raise Exception(warning_string)
                else:
                    if verbose:
                        print(warning_string)


            limb_corresp_for_networks = dict([(i,dict(branch_skeleton=k.skeleton,
                                                     width_from_skeleton=k.width,
                                                     branch_mesh=k.mesh,
                                                     branch_face_idx=k.mesh_face_idx)) for i,k in fixed_node_objects.items()])

            floating_flag = False
            if len(curr_all_concept_network_data) == 0:
                #pick a random endpoint to start from the skeleton
                
                if False:
                    total_skeleton = sk.stack_skeletons([v["branch_skeleton"] for v in limb_corresp_for_networks.values()])
                    all_endpoints = sk.find_skeleton_endpoint_coordinates(total_skeleton)
                    chosen_endpoint = all_endpoints[0]
                else:
                    limb_keys = list(limb_corresp_for_networks.keys())
                    chosen_endpoint = sk.find_branch_endpoints(limb_corresp_for_networks[limb_keys[0]]["branch_skeleton"])[0]
                    

                if verbose:
                    print(f"There was no starting information so doing to put dummy information and random starting endpoint = {chosen_endpoint}")
                curr_limb_network_stating_info = {-1:{-1:{"touching_verts":None,
                                                         "endpoint":chosen_endpoint}}}
                floating_flag = True
            else:
                #curr_all_concept_network_data[0]["soma_group_idx"] = 0
                curr_all_concept_network_data = nru.clean_all_concept_network_data(curr_all_concept_network_data)
                curr_limb_network_stating_info = nru.all_concept_network_data_to_dict(curr_all_concept_network_data)

            #calculate the concept networks


            sep_graph_data["limb_correspondence"] = limb_corresp_for_networks

            sep_graph_data["limb_network_stating_info"] = curr_limb_network_stating_info

            #raise Exception("Checking on limb starting network info")

            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(limb_corresp_for_networks,
                                                                                            curr_limb_network_stating_info,
                                                                                            run_concept_network_checks=False,
                                                                                verbose=verbose
                                                                                           )   

            sep_graph_data["limb_concept_networks"] = limb_to_soma_concept_networks


            # --------------- Making the new limb object -------------- #
            limb_str_name = f"split_limb_from_limb_{curr_limb_idx}_part_{seg_graph_idx}"
            if floating_flag:
                limb_str_name += "_floating"

            new_labels = [limb_str_name]

            new_limb_obj = neuron.Limb(mesh=new_limb_mesh,
                         curr_limb_correspondence=limb_corresp_for_networks,
                         concept_network_dict=limb_to_soma_concept_networks,
                         mesh_face_idx=new_limb_mesh_face_idx,
                        labels=new_labels,
                         branch_objects = fixed_node_objects,#this will have a dictionary mapping to the branch objects if provided
                       )


            sep_graph_data["limb_labels"] = new_labels
            sep_graph_data["Limb_obj"] = new_limb_obj

            new_limb_data.append(sep_graph_data)

    
    
    # Phase 2: ------------- Adjusting the existing neuron object --------------- #
    
    neuron_obj_cp = neuron.Neuron(neuron_obj)
    #1) map the new neuron objects to unused limb names
    new_limb_dict = dict()
    new_limb_idxs = [curr_limb_idx] + [len(neuron_obj_cp) + i for i in range(len(new_limb_data[1:]))]
    new_limb_string_names = [f"L{k}" for k in new_limb_idxs]
    for l_i,limb_data in zip(new_limb_idxs,new_limb_data):
        new_limb_dict[l_i] = limb_data


    #3) Delete the old limb data in the preprocessing dictionary (Adjust the soma_to_piece_connectivity)
    attr_to_update = ['limb_meshes', 'limb_correspondence', 'limb_network_stating_info', 'limb_concept_networks', 'limb_labels']
    for attr_upd in attr_to_update:
        del neuron_obj_cp.preprocessed_data[attr_upd][curr_limb_idx]

    # --- revise the soma_to_piece_connectivity -- #
    somas_to_delete_from = np.unique(neuron_obj_cp[curr_limb_idx].touching_somas())

    for sm_d in somas_to_delete_from:
        neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d].remove(curr_limb_idx)

    #4) Delete the old limb from the neuron concept network   
    neuron_obj_cp.concept_network.remove_node(f"L{curr_limb_idx}")

    #5) Add the new limb nodes with edges to the somas they are touching
    for l_i,limb_data in new_limb_dict.items():
        curr_limb_obj = limb_data["Limb_obj"]
        curr_limb_touching_somas = curr_limb_obj.touching_somas()



        str_node_name = f"L{l_i}"
        
        neuron_obj_cp.concept_network.add_node(str_node_name)

        xu.set_node_data(curr_network=neuron_obj_cp.concept_network,
                                         node_name=str_node_name,
                                         curr_data=curr_limb_obj,
                                         curr_data_label="data")

        for sm_d in curr_limb_touching_somas:
            neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d].append(l_i)
            neuron_obj_cp.concept_network.add_edge(str_node_name,f"S{sm_d}")

        for attr_upd in attr_to_update:
            if attr_upd == "limb_meshes":
                neuron_obj_cp.preprocessed_data[attr_upd].insert(l_i,limb_data[attr_upd])
            else:
                neuron_obj_cp.preprocessed_data[attr_upd][l_i] = limb_data[attr_upd]

    for sm_d in neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"]:
        neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d] = list(np.unique(neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d]))
    
    
    return neuron_obj_cp






# ----------------12/27: Whole Neuron Splitting ------------------------ #





def split_neuron_limbs_by_suggestions(neuron_obj,
                                split_suggestions,
                                      plot_soma_limb_network=False,
                                verbose=False):
    """
    Purpose: 
    
    Will take the suggestions of the splits and 
    split the necessary limbs of the neuron object and 
    return the split neuron
    
    """
    
    split_neuron_obj = neuron.Neuron(neuron_obj)
    limb_results = split_suggestions
    
    #this step is where applies the actual changes to the neuron obj
    new_concept_networks = pru.split_suggestions_to_concept_networks(neuron_obj,limb_results,
                                                                    apply_changes_to_limbs=True)
   
    for curr_limb_idx,curr_limb_nx in new_concept_networks.items():
        curr_limb_idx = int(curr_limb_idx)
        
        #delete the node sthat should be deleted
        removed_branches_list  = pru.get_removed_branches_from_suggestion(limb_results,curr_limb_idx)
        split_neuron_obj[curr_limb_idx] = pru.collapse_branches_on_limb(split_neuron_obj[curr_limb_idx],removed_branches_list)
        
        

        conn_comp = list(nx.connected_components(curr_limb_nx))

        if verbose:
            print(f"\n\n---Working on Splitting Limb {curr_limb_idx} with {len(conn_comp)} components----")

        split_neuron_obj = pru.split_neuron_limb_by_seperated_network(split_neuron_obj,
                                                                      seperate_networks=conn_comp,
                                                                 curr_limb_idx = curr_limb_idx,
                                                                error_on_multile_starting_nodes=False,
                                                                verbose=verbose)
        
    if plot_soma_limb_network:
        nviz.plot_soma_limb_concept_network(split_neuron_obj)
        
    return split_neuron_obj

def split_disconnected_neuron(neuron_obj,
                              plot_seperated_neurons=False,
                             verbose=False,
                             save_original_mesh_idx=True,
                             filter_away_remaining_error_limbs=True,
                             return_errored_limbs_info=True,
                             add_split_to_description=True,
                             copy_all_non_soma_touching=True,
                              
                             ):
    
    """
    Purpose: If a neuron object has already been disconnected
    at the limbs, this function will then split the neuron object
    into a list of multiple neuron objects
    
    Pseudocode: 
    1) check that there do not exist any error limbs
    2) Do the splitting process
    3) Visualize results if requested
    
    """
    split_neuron_obj = neuron_obj
    
    
    
    #--------Part 1: check that all the limbs have beeen split so that there are no more error limbs
    same_soma_error_limbs = split_neuron_obj.same_soma_multi_touching_limbs
    multi_soma_error_limbs = split_neuron_obj.multi_soma_touching_limbs
    
    curr_error_limbs = nru.error_limbs(split_neuron_obj)

    if len(curr_error_limbs) > 0:
        if not filter_away_remaining_error_limbs:
            raise Exception(f"There were still error limbs before trying the neuron object split: error limbs = {curr_error_limbs}")
        else:
            if verbose:
                print(f"Still remaining error limbs ({curr_error_limbs}), but will filter them away")
    
    
    
    
    # ------ Part 2: start the splitting process
    
    # get all the somas that we will split into
    soma_names = split_neuron_obj.get_soma_node_names()
    
    
    neuron_obj_list = []
    neuron_obj_errored_limbs_area = []
    neuron_obj_errored_limbs_skeletal_length = []
    neuron_obj_n_multi_soma_errors = []
    neuron_obj_n_same_soma_errors = []
    

    for curr_soma_idx,curr_soma_name in enumerate(soma_names):
        if verbose:
            print(f"\n\n------ Working on Soma {curr_soma_idx} -------")

        neuron_cp = split_neuron_obj

        #getting all the soma information we will need for preprocessing
        soma_obj = neuron_cp[curr_soma_name]
        curr_soma_meshes = [soma_obj.mesh]
        curr_soma_sdfs = [soma_obj.sdf]
        curr_soma_synapses = [soma_obj.synapses]
        curr_soma_volume = [soma_obj._volume]
        curr_soma_volume_ratios = [soma_obj.volume_ratio]





        # getting the limb information and new soma connectivity
        limb_neighbors = np.sort(xu.get_neighbors(neuron_cp.concept_network,curr_soma_name)).astype("int")
        limb_neighbors = [int(k) for k in limb_neighbors]
        
        if verbose:
            print(f"limb_neighbors = {limb_neighbors}")

        soma_to_piece_connectivity = neuron_cp.preprocessed_data["soma_to_piece_connectivity"][curr_soma_idx]
        
        

        if len(np.intersect1d(limb_neighbors,soma_to_piece_connectivity)) < len(soma_to_piece_connectivity):
            raise Exception(f"piece connectivity ({soma_to_piece_connectivity}) not match limb neighbors ({limb_neighbors})")
            
        if filter_away_remaining_error_limbs:# and len(curr_error_limbs)>0:
            if verbose:
                print(f"limb_neighbors BEFORE error limbs removed = {limb_neighbors}")
            original_len = len(limb_neighbors)
            
            
            # ---------- 1/28: More error limb information ----------------- #
            curr_neuron_same_soma_error_limbs = list(np.intersect1d(limb_neighbors,same_soma_error_limbs))
            curr_neuron_multi_soma_error_limbs = list(np.intersect1d(limb_neighbors,multi_soma_error_limbs))
            
            #print(f"curr_neuron_multi_soma_error_limbs = {curr_neuron_multi_soma_error_limbs}")
            
            curr_n_multi_soma_limbs_cancelled = len(curr_neuron_multi_soma_error_limbs)
            curr_n_same_soma_limbs_cancelled = len(curr_neuron_same_soma_error_limbs)
            
            unique_error_limbs = np.unique(curr_neuron_same_soma_error_limbs + curr_neuron_multi_soma_error_limbs)
            
            curr_error_limbs_cancelled_area = [split_neuron_obj[k].area/len(np.unique(split_neuron_obj[k].touching_somas()))
                                               for k in unique_error_limbs]
            
            curr_error_limbs_cancelled_skeletal_length = [split_neuron_obj[k].skeletal_length/len(np.unique(split_neuron_obj[k].touching_somas()))
                                               for k in unique_error_limbs]
            
            
            limb_neighbors = np.setdiff1d(limb_neighbors,curr_error_limbs)
            
            if verbose:
                print(f"limb_neighbors AFTER error limbs removed = {limb_neighbors}")
            
        else:
            curr_n_multi_soma_limbs_cancelled = 0
            curr_n_same_soma_limbs_cancelled = 0
            curr_error_limbs_cancelled_area = []
            
        if verbose:
            print(f"curr_n_multi_soma_limbs_cancelled = {curr_n_multi_soma_limbs_cancelled}")
            print(f"curr_n_same_soma_limbs_cancelled = {curr_n_same_soma_limbs_cancelled}")
            print(f"n_errored_lims = {len(curr_error_limbs_cancelled_area)}")
            print(f"curr_error_limbs_cancelled_area = {curr_error_limbs_cancelled_area}")
            
        neuron_obj_errored_limbs_area.append(curr_error_limbs_cancelled_area)
        neuron_obj_errored_limbs_skeletal_length.append(curr_error_limbs_cancelled_skeletal_length)
        neuron_obj_n_multi_soma_errors.append(curr_n_multi_soma_limbs_cancelled)
        neuron_obj_n_same_soma_errors.append(curr_n_same_soma_limbs_cancelled)

        curr_soma_to_piece_connectivity = {0:list(np.arange(0,len(limb_neighbors)))}






        #getting the whole mesh and limb face correspondence
        mesh_list_for_whole = [soma_obj.mesh]

        #for the limb meshes
        limb_meshes = []

        #for the limb mesh faces idx
        counter = len(curr_soma_meshes[0].faces)
        face_idx_list = [np.arange(0,counter)]

        old_node_to_new_node_mapping = dict()


        for i,k in  enumerate(limb_neighbors):

            #getting the name mapping
            old_node_to_new_node_mapping[k] = i

            #getting the meshes of the limbs
            limb_mesh = neuron_cp[k].mesh
            limb_meshes.append(limb_mesh)


            mesh_list_for_whole.append(limb_mesh)
            face_length = len(limb_mesh.faces)
            face_idx_list.append(np.arange(counter,counter + face_length))
            counter += face_length

        whole_mesh = tu.combine_meshes(mesh_list_for_whole)






        # generating the new limb correspondence:
        #curr_limb_correspondence = dict([(i,neuron_cp.preprocessed_data["limb_correspondence"][k]) for i,k in enumerate(limb_neighbors)])
        curr_limb_correspondence = dict([(i,neuron_cp[k].limb_correspondence) for i,k in enumerate(limb_neighbors)])




        # concept network generation
        curr_limb_network_stating_info = dict()


        for k in limb_neighbors:

            local_starting_info = neuron_cp.preprocessed_data["limb_network_stating_info"][k]

            #making sure the soma has the right name
            soma_keys = list(local_starting_info.keys())
            if len(soma_keys) > 1:
                raise Exception("More than one soma connection")
            else:
                soma_key = soma_keys[0]

            if soma_key != 0:
                local_starting_info = {0:local_starting_info[soma_key]}


            #making sure the soma group has the right name
            starting_group_keys = list(local_starting_info[0].keys())
            if len(starting_group_keys) > 1 or starting_group_keys[0] != 0:
                raise Exception("Touching group was not equal to 0")

            #save the new starting info
            curr_limb_network_stating_info[old_node_to_new_node_mapping[k]] = local_starting_info

        # creating the new concept networks from the starting info
        curr_limb_concept_networks=dict()

        
        for curr_limb_idx,new_limb_correspondence_indiv in curr_limb_correspondence.items():
            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(new_limb_correspondence_indiv,
                                                                                curr_limb_network_stating_info[curr_limb_idx],
                                                                                run_concept_network_checks=True,
                                                                               )   

            curr_limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks








        #limb labels:
        curr_limb_labels = dict()

        for k in limb_neighbors:
            local_limb_labels = neuron_cp.preprocessed_data["limb_labels"][k]
            if local_limb_labels is None or local_limb_labels == "Unlabeled":
                local_limb_labels = []

            local_limb_labels.append(f"Soma_{curr_soma_idx}_originally")
            curr_limb_labels[old_node_to_new_node_mapping[k]] = local_limb_labels



        meshes_to_concat = [[kv["branch_mesh"] for k,kv in jv.items()] for j,jv in curr_limb_correspondence.items()]
        
        if len(meshes_to_concat) > 0:
            whole_branch_mesh = tu.combine_meshes(np.concatenate(meshes_to_concat))
            
            floating_indexes = tu.mesh_pieces_connectivity(neuron_obj.mesh,
                         whole_branch_mesh,
                        periphery_pieces=neuron_obj.non_soma_touching_meshes,
                           print_flag=False)
        
            local_floating_meshes = list(np.array(neuron_obj.non_soma_touching_meshes)[floating_indexes])

            if verbose:
                print(f"local_floating_meshes = {local_floating_meshes}")
            
            """
            5/9: Attemps to copy all floating pieces to each of the splits
            so if want to add them back  later then can (DecompositionAxon)
            
            """
            if copy_all_non_soma_touching:
                local_floating_meshes = neuron_obj.non_soma_touching_meshes
                
            whole_mesh  = whole_mesh + local_floating_meshes
        else:
            local_floating_meshes = []
            whole_mesh  = whole_mesh + local_floating_meshes
            

        
        

        #using all of the data to create new preprocessing info
        new_preprocessed_data = preprocessed_data= dict(
                #soma data
                soma_meshes = curr_soma_meshes,
                soma_sdfs = curr_soma_sdfs,
                soma_synapses = curr_soma_synapses,
                soma_volumes = curr_soma_volume,
                soma_volume_ratios=curr_soma_volume_ratios,
                

                #soma connectivity
                soma_to_piece_connectivity = curr_soma_to_piece_connectivity,

                # limb info
                limb_correspondence=curr_limb_correspondence,
                limb_meshes=limb_meshes,
                limb_mehses_face_idx = face_idx_list,
                limb_labels=curr_limb_labels,

                #concept network info
                limb_concept_networks=curr_limb_concept_networks,
                limb_network_stating_info=curr_limb_network_stating_info,


                # the other mesh pieces that will not be included
                insignificant_limbs=[],
                not_processed_soma_containing_meshes=[],
                glia_faces = [],
                non_soma_touching_meshes=local_floating_meshes,
                inside_pieces=[],


                )

        limb_to_branch_objects = dict()
        for k in limb_neighbors:
            limb_obj = neuron_cp[int(k)]
            branch_dict = dict([(b,limb_obj[int(b)]) for b in limb_obj.get_branch_names()])
            limb_to_branch_objects[old_node_to_new_node_mapping[k]] = branch_dict

        segment_id = neuron_cp.segment_id
        if add_split_to_description:
            description = f"{neuron_cp.description}_soma_{curr_soma_idx}_split"
        else:
            description = f"{neuron_cp.description}"

        #---------- 1/24: Will save off the original mesh idx so can use the old mesh file to load -------- #
        if save_original_mesh_idx:
            original_mesh_idx = tu.original_mesh_faces_map(neuron_obj.mesh,whole_mesh,
                          exact_match=True)
            if neuron_obj.original_mesh_idx is not None:
                original_mesh_idx = neuron_obj.original_mesh_idx[original_mesh_idx]
        else:
            original_mesh_idx = None


        # new neuron object:

        single_split_neuron_obj = neuron.Neuron(mesh=whole_mesh,
                 segment_id=segment_id,
                 description=description,
                 preprocessed_data=new_preprocessed_data,
                 limb_to_branch_objects=limb_to_branch_objects,
                 widths_to_calculate=[],
                 original_mesh_idx=original_mesh_idx,
                suppress_output=not verbose,
                suppress_all_output = not verbose)


        neuron_obj_list.append(single_split_neuron_obj)
        
        
        
    # ------ Part 3: Visualize the Results
    if verbose:
        print(f"\n\nNumber of seperate neuron objects = {len(neuron_obj_list)}")

    if plot_seperated_neurons:
        for n_obj in neuron_obj_list:
            nviz.visualize_neuron(n_obj,
                                 visualize_type=["mesh","skeleton"],
                                 limb_branch_dict="all")

            
    
    if return_errored_limbs_info:
        return (neuron_obj_list,
                neuron_obj_errored_limbs_area,
                neuron_obj_errored_limbs_skeletal_length,
                neuron_obj_n_multi_soma_errors,
                neuron_obj_n_same_soma_errors)
    else:
        return neuron_obj_list



def split_neuron(neuron_obj,
                 limb_results=None,
                 plot_crossover_intermediates=False,
                 plot_neuron_split_results=False,
                 
                 plot_soma_limb_network=False,
                 plot_seperated_neurons=False,
                verbose=False,
                 filter_away_remaining_error_limbs=True,
                 return_error_info = False,
                 
                 min_skeletal_length_limb= None,
                 
                 
                **kwargs):
    """
    Purpose: To take in a whole neuron that could have any number of somas
    and then to split it into multiple neuron objects

    Pseudocode: 
    1) Get all of the split suggestions
    2) Split all of the limbs that need splitting
    3) Once have split the limbs, split the neuron object into mutliple objects


    """
    if min_skeletal_length_limb is None:
        min_skeletal_length_limb = min_skeletal_length_limb_global
    
    if plot_crossover_intermediates:
        kwargs["plot_intermediates"] = True
    
    if neuron_obj.n_error_limbs == 0 and neuron_obj.n_somas == 1:
        print("No error limbs to processs so just returning the original neuron")
        
        neuron_list = [neuron_obj]
        
        if return_error_info:
            return (neuron_list,[[]],[[]],[0],[0])
        else:
            return neuron_list
        
    
    neuron_obj = neuron.Neuron(neuron_obj)
    
    # ---------- 1/22 Addition that cleans the concept network info ------------ #
    nru.clean_neuron_all_concept_network_data(neuron_obj)
    
    #1) Get all of the split suggestions
    if limb_results is None:
        limb_results = pru.multi_soma_split_suggestions(neuron_obj,
                                                   verbose = verbose,
                                                       **kwargs)
    else:
        if verbose:
            print("using precomputed split suggestions")
            
    if plot_neuron_split_results:
        nviz.plot_split_suggestions_per_limb(neuron_obj,
                                        limb_results)
    
    #2) Split all of the limbs that need splitting
    split_neuron_obj = pru.split_neuron_limbs_by_suggestions(neuron_obj,
                                split_suggestions=limb_results,
                                plot_soma_limb_network=plot_soma_limb_network,
                                verbose=verbose)
    
    
    nru.clean_neuron_all_concept_network_data(split_neuron_obj)
    
    if not filter_away_remaining_error_limbs:
        curr_error_limbs = nru.error_limbs(split_neuron_obj)

        if len(curr_error_limbs) > 0:
            raise Exception(f"There were still error limbs before the splitting: {curr_error_limbs}")
            
    
    #3) Once have split the limbs, split the neuron object into mutliple objects
    (neuron_list,
     neuron_list_errored_limbs_area,
     neuron_list_errored_limbs_skeletal_length,
    neuron_list_n_multi_soma_errors,
    neuron_list_n_same_soma_errors) = pru.split_disconnected_neuron(split_neuron_obj,
                         plot_seperated_neurons=plot_seperated_neurons,
                        filter_away_remaining_error_limbs=filter_away_remaining_error_limbs,
                         verbose =verbose)
    
    #2b) Check that all the splits occured
    for i,n_obj in enumerate(neuron_list):
        curr_error_limbs = nru.error_limbs(n_obj)

        if len(curr_error_limbs) > 0:
            raise Exception(f"There were still error limbs after splitting for neuron obj {i}")
            
            
    #3) Filter the limbs by length if necessary
    if min_skeletal_length_limb is not None:
        if verbose:
            print(f"Filter limb lengths for limb size of {min_skeletal_length_limb}")
        neuron_list = [
            nru.filter_away_neuron_limbs_by_min_skeletal_length(
                n_obj,min_skeletal_length_limb = min_skeletal_length_limb,verbose = verbose)
            for n_obj in neuron_list]
        

    if return_error_info:
        return (neuron_list,neuron_list_errored_limbs_area,
                neuron_list_errored_limbs_skeletal_length,
                                        neuron_list_n_multi_soma_errors,
                                        neuron_list_n_same_soma_errors)
    else:
        return neuron_list


'''
def collapse_branches_on_limb_old(limb_obj,branch_list,
                             plot_new_limb=False,
                              reassign_mesh=True,
                              store_placeholder_for_removed_nodes=True,
                              debug_time = True,
                             verbose=False):
    """
    Purpose: To remove 1 or more branches from the concept network
    of a limb and to adjust the underlying skeleton
    
    Application: To be used in when trying to split 
    a neuron and want to combine nodes that are really close
    
    
    *** currently does not does not reallocate the mesh part of the nodes that were deleted
    
    
    Pseudocode: 
    
    For each branch to remove
    0) Find the branches that were touching the soon to be deleted branch
    1) Alter the skeletons of those that were touching that branch
    
    After revised all nodes
    2) Remove the current node
    3) Generate the limb correspondence, network starting info to generate soma concept networks
    4) Create a new limb object and return it


    Ex: new_limb_obj = nru.collapse_branches_on_limb(curr_limb,[30,31],plot_new_limb=True,verbose=True)
    """
    
    limb_time = time.time()
    
    curr_limb_cp = copy.deepcopy(limb_obj)
    
    if debug_time:
        print(f"deepcopy limb time = {time.time() - limb_time}")
        limb_time = time.time()
    
    
    if not nu.is_array_like(branch_list ):
        branch_list = [branch_list]
        
    seg_id_lookup = np.arange(0,len(curr_limb_cp))
    
    for curr_short_seg in branch_list:
        #need to look up the new seg id if there is one now based on previous deletions
        
        curr_short_seg_revised = seg_id_lookup[curr_short_seg]
        if verbose:
            print(f"curr_short_seg_revised = {curr_short_seg_revised}")
        
        
        branch_obj = curr_limb_cp[curr_short_seg_revised]
        
        touching_branches,touching_endpoints = nru.skeleton_touching_branches(curr_limb_cp,branch_idx=curr_short_seg_revised)
        
        if debug_time:
            print(f"skeleton branch touchings = {time.time() - limb_time}")
            limb_time = time.time()
        
        
        #deciding whether to take the average or just an endpoint as the new stitch point depending on if nodes on both ends
        touch_len = np.array([len(k) for k in touching_branches])
        
        if verbose:
            print(f"np.sum(touch_len>0) = {np.sum(touch_len>0)}")
            
        if np.sum(touch_len>0)==2:
            if verbose:
                print("Using average stitch point")
            new_stitch_point = np.mean(touching_endpoints,axis=0)
            middle_node = True
        else:
            if verbose:
                print("Using ONE stitch point")
            new_stitch_point = touching_endpoints[np.argmax(touch_len)]
            middle_node = False
        
        if verbose:
            print(f"touching_endpoints = {touching_endpoints}")
            print(f"new_stitch_point = {new_stitch_point}")
            
        if debug_time:
            print(f"Getting new stitch point time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #1a) Decide which branch to give the mesh to
        if middle_node and reassign_mesh:
            """
            Pseudocode:
            1) Get the skeletal angles between all the touching branches and the branch to be deleted
            2) Find the branch with the smallest skeletal angle
            3) Things that need to be updated for the winning branch:
            - mesh
            -mesh_face_idx
            - n_spines
            - spines
            - spines_volume
            
            
            """
            all_touching_branches = np.concatenate(touching_branches)
            all_touching_branches_angles = [sk.offset_skeletons_aligned_parent_child_skeletal_angle(
                            curr_limb_cp[br].skeleton,
                            curr_limb_cp[curr_short_seg_revised].skeleton) for br in all_touching_branches]
            winning_branch =all_touching_branches[np.argmin(all_touching_branches_angles)]
            
            
            if verbose:
                print(f"Angles for {all_touching_branches} are {all_touching_branches_angles}")
                print(f"Branch that will absorb mesh of {curr_short_seg} is {winning_branch} ")
                
            if debug_time:
                print(f"branch angles = {time.time() - limb_time}")
                limb_time = time.time()
            
            curr_limb_cp[winning_branch].mesh_face_idx = np.concatenate([curr_limb_cp[winning_branch].mesh_face_idx,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh_face_idx])
            curr_limb_cp[winning_branch].mesh = tu.combine_meshes([curr_limb_cp[winning_branch].mesh,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh])
            if not curr_limb_cp[curr_short_seg_revised].spines is None:
                if curr_limb_cp[winning_branch].spines is not None:
                    curr_limb_cp[winning_branch].spines += curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume += curr_limb_cp[curr_short_seg_revised].spines_volume
                else:
                    curr_limb_cp[winning_branch].spines = curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume = curr_limb_cp[curr_short_seg_revised].spines_volume
                    
            
            if debug_time:
                print(f"Resolving spines time = {time.time() - limb_time}")
                limb_time = time.time()
            

        #1b) Alter the skeletons of those that were touching
        for t_branches,t_endpoints in zip(touching_branches,touching_endpoints):
            for br in t_branches:
                curr_limb_cp[br].skeleton = sk.add_and_smooth_segment_to_branch(curr_limb_cp[br].skeleton,
                                                                                new_stitch_point=new_stitch_point,
                                                                            skeleton_stitch_point=t_endpoints)
                curr_limb_cp[br].calculate_endpoints()
                
        if debug_time:
            print(f"smoothing the segments and calculating endpoints = {time.time() - limb_time}")
            limb_time = time.time()
        """
        
        
        
        """
        
            
        #2) Remove the current node
        if not store_placeholder_for_removed_nodes:
            curr_limb_cp.concept_network.remove_node(curr_short_seg_revised)
            #update the seg_id_lookup
            seg_id_lookup = np.insert(seg_id_lookup,curr_short_seg,-1)[:-1]
            run_concept_network_checks=True
        else:
            curr_mesh_center = curr_limb_cp[curr_short_seg_revised].mesh_center
            center_deviation = 5.2345
            curr_limb_cp[curr_short_seg_revised].skeleton = np.array([curr_mesh_center,curr_mesh_center+center_deviation]).reshape(-1,2,3)
            curr_limb_cp[curr_short_seg_revised].calculate_endpoints()
            run_concept_network_checks = False
            
        if debug_time:
            print(f" Adding placeholder branch time = {time.time() - limb_time}")
            limb_time = time.time()

        limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(curr_limb_cp.limb_correspondence,
                                                                                    curr_limb_cp.network_starting_info,
                                                                                    run_concept_network_checks=False,
                                                                              )  
        if debug_time:
            print(f"Calculating limb concept networks time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #print(f"curr_limb.deleted_edges 6={curr_limb_cp.deleted_edges}")
        curr_limb_cp = neuron.Limb(curr_limb_cp.mesh,
                                 curr_limb_cp.limb_correspondence,
                                 concept_network_dict=limb_to_soma_concept_networks,
                                 mesh_face_idx = curr_limb_cp.mesh_face_idx,
                                 labels=curr_limb_cp.labels,
                                 branch_objects=curr_limb_cp.branch_objects,
                                  deleted_edges=curr_limb_cp.deleted_edges,
                                  created_edges=curr_limb_cp.created_edges)
        
        if debug_time:
                print(f"Creating new limb object time = {time.time() - limb_time}")
                limb_time = time.time()
        
        #print(f"curr_limb.deleted_edges 7={curr_limb_cp.deleted_edges}")
        
        
        
        
    
    if plot_new_limb:
        nviz.plot_limb_correspondence(curr_limb_cp.limb_correspondence)
    
    return curr_limb_cp
'''

def collapse_branches_on_limb(limb_obj,branch_list,
                             plot_new_limb=False,
                              reassign_mesh=True,
                              store_placeholder_for_removed_nodes=True,
                              debug_time = False,
                             verbose=False):
    """
    Purpose: To remove 1 or more branches from the concept network
    of a limb and to adjust the underlying skeleton
    
    ** this is more if want to remove the presence of a branch but not
    remove the mesh associated with it (so just collapsing the node)
    
    Application: To be used in when trying to split 
    a neuron and want to combine nodes that are really close
    
    
    *** currently does not does not reallocate the mesh part of the nodes that were deleted
    
    
    Pseudocode: 
    
    For each branch to remove
    0) Find the branches that were touching the soon to be deleted branch
    1) Alter the skeletons of those that were touching that branch
    
    After revised all nodes
    2) Remove the current node
    3) Generate the limb correspondence, network starting info to generate soma concept networks
    4) Create a new limb object and return it


    Ex: new_limb_obj = nru.collapse_branches_on_limb(curr_limb,[30,31],plot_new_limb=True,verbose=True)
    """
    
    limb_time = time.time()
    
    curr_limb_cp = copy.deepcopy(limb_obj)
    
    if debug_time:
        print(f"deepcopy limb time = {time.time() - limb_time}")
        limb_time = time.time()
    
    
    if not nu.is_array_like(branch_list ):
        branch_list = [branch_list]
        
    seg_id_lookup = np.arange(0,len(curr_limb_cp))
    
    for curr_short_seg in branch_list:
        #need to look up the new seg id if there is one now based on previous deletions
        
        curr_short_seg_revised = seg_id_lookup[curr_short_seg]
        if verbose:
            print(f"curr_short_seg_revised = {curr_short_seg_revised}")
        
        
        branch_obj = curr_limb_cp[curr_short_seg_revised]
        
        touching_branches,touching_endpoints = nru.skeleton_touching_branches(curr_limb_cp,branch_idx=curr_short_seg_revised)
        
        if debug_time:
            print(f"skeleton branch touchings = {time.time() - limb_time}")
            limb_time = time.time()
        
        
        #deciding whether to take the average or just an endpoint as the new stitch point depending on if nodes on both ends
        touch_len = np.array([len(k) for k in touching_branches])
        
        if verbose:
            print(f"np.sum(touch_len>0) = {np.sum(touch_len>0)}")
            
        if np.sum(touch_len>0)==2:
            if verbose:
                print("Using average stitch point")
            new_stitch_point = np.mean(touching_endpoints,axis=0)
            middle_node = True
        else:
            if verbose:
                print("Using ONE stitch point")
            new_stitch_point = touching_endpoints[np.argmax(touch_len)]
            middle_node = False
        
        if verbose:
            print(f"touching_endpoints = {touching_endpoints}")
            print(f"new_stitch_point = {new_stitch_point}")
            
        if debug_time:
            print(f"Getting new stitch point time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #1a) Decide which branch to give the mesh to
        if middle_node and reassign_mesh:
            """
            Pseudocode:
            1) Get the skeletal angles between all the touching branches and the branch to be deleted
            2) Find the branch with the smallest skeletal angle
            3) Things that need to be updated for the winning branch:
            - mesh
            -mesh_face_idx
            - n_spines
            - spines
            - spines_volume
            
            
            """
            all_touching_branches = np.concatenate(touching_branches)
            all_touching_branches_angles = [sk.offset_skeletons_aligned_parent_child_skeletal_angle(
                            curr_limb_cp[br].skeleton,
                            curr_limb_cp[curr_short_seg_revised].skeleton) for br in all_touching_branches]
            winning_branch =all_touching_branches[np.argmin(all_touching_branches_angles)]
            
            
            if verbose:
                print(f"Angles for {all_touching_branches} are {all_touching_branches_angles}")
                print(f"Branch that will absorb mesh of {curr_short_seg} is {winning_branch} ")
                
            if debug_time:
                print(f"branch angles = {time.time() - limb_time}")
                limb_time = time.time()
            
            curr_limb_cp[winning_branch].mesh_face_idx = np.concatenate([curr_limb_cp[winning_branch].mesh_face_idx,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh_face_idx])
            curr_limb_cp[winning_branch].mesh = tu.combine_meshes([curr_limb_cp[winning_branch].mesh,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh])
            if not curr_limb_cp[curr_short_seg_revised].spines is None:
                if curr_limb_cp[winning_branch].spines is not None:
                    curr_limb_cp[winning_branch].spines += curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume += curr_limb_cp[curr_short_seg_revised].spines_volume
                else:
                    curr_limb_cp[winning_branch].spines = curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume = curr_limb_cp[curr_short_seg_revised].spines_volume
                    
            
            if debug_time:
                print(f"Resolving spines time = {time.time() - limb_time}")
                limb_time = time.time()
            

        #1b) Alter the skeletons of those that were touching
        for t_branches,t_endpoints in zip(touching_branches,touching_endpoints):
            for br in t_branches:
                curr_limb_cp[br].skeleton = sk.add_and_smooth_segment_to_branch(curr_limb_cp[br].skeleton,
                                                                                new_stitch_point=new_stitch_point,
                                                                            skeleton_stitch_point=t_endpoints)
                curr_limb_cp[br].calculate_endpoints()
                
        if debug_time:
            print(f"smoothing the segments and calculating endpoints = {time.time() - limb_time}")
            limb_time = time.time()
        """
        
        
        
        """
        
            
        #2) Remove the current node
        if not store_placeholder_for_removed_nodes:
            curr_limb_cp.concept_network.remove_node(curr_short_seg_revised)
            #update the seg_id_lookup
            seg_id_lookup = np.insert(seg_id_lookup,curr_short_seg,-1)[:-1]
            run_concept_network_checks=True
        else:
            curr_mesh_center = curr_limb_cp[curr_short_seg_revised].mesh_center
            center_deviation = 5.2345
            curr_limb_cp[curr_short_seg_revised].skeleton = np.array([curr_mesh_center,curr_mesh_center+center_deviation]).reshape(-1,2,3)
            curr_limb_cp[curr_short_seg_revised].calculate_endpoints()
            run_concept_network_checks = False
            
        if debug_time:
            print(f" Adding placeholder branch time = {time.time() - limb_time}")
            limb_time = time.time()
            

    limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(curr_limb_cp.limb_correspondence,
                                                                                curr_limb_cp.network_starting_info,
                                                                                run_concept_network_checks=False,
                                                                          )  
    if debug_time:
        print(f"Calculating limb concept networks time = {time.time() - limb_time}")
        limb_time = time.time()

    #print(f"curr_limb.deleted_edges 6={curr_limb_cp.deleted_edges}")
    curr_limb_cp = neuron.Limb(curr_limb_cp.mesh,
                             curr_limb_cp.limb_correspondence,
                             concept_network_dict=limb_to_soma_concept_networks,
                             mesh_face_idx = curr_limb_cp.mesh_face_idx,
                             labels=curr_limb_cp.labels,
                             branch_objects=curr_limb_cp.branch_objects,
                              deleted_edges=curr_limb_cp.deleted_edges,
                              created_edges=curr_limb_cp.created_edges)

    if debug_time:
            print(f"Creating new limb object time = {time.time() - limb_time}")
            limb_time = time.time()


    if plot_new_limb:
        nviz.plot_limb_correspondence(curr_limb_cp.limb_correspondence)
    
    return curr_limb_cp



# ----------------- 1/26: Final Proofreading Rules splitting ------------#



def delete_branches_from_limb(neuron_obj,
                              branches_to_delete,
                              limb_idx=None,
                              limb_name=None,
                              verbose=False,
                                ):
    """
    Will delete branches from a certain limb
    
    
    
    """
    if limb_idx is None:
        limb_idx = int(limb_name[1:])
    
    if verbose:
        print(f"---- Working on Limb {limb_idx} ----")
        print(f"length of concept network BEFORE elimination = {len(neuron_obj[limb_idx].concept_network.nodes())} ")

    #1) Remove the nodes in the limb branch dict
    concept_graph = nx.Graph(neuron_obj[limb_idx].concept_network)
    concept_graph.remove_nodes_from(branches_to_delete)
    seperate_networks = [list(k) for k in nx.connected_components(concept_graph)]

    if verbose:
        print(f"length of concept network AFTER elimination = {len(concept_graph.nodes())} ")

    split_neuron = pru.split_neuron_limb_by_seperated_network(neuron_obj,
                                                             curr_limb_idx=limb_idx,
                                                              #seperate_networks = [list(concept_graph.nodes())]
                                                              seperate_networks = seperate_networks
                                                             )
    return split_neuron

def delete_branches_from_neuron(neuron_obj,
                                limb_branch_dict,
                                plot_neuron_after_cancellation = False,
                                plot_final_neuron = False,
                                verbose = False,
                                add_split_to_description = False,
                                **kwargss
                               ):
    

    """
    Purpose: To eliminate the error cells and downstream targets
    given limb branch dict of nodes to eliminate

    Pseudocode: 

    For each limb in branch dict
    1) Remove the nodes in the limb branch dict

    2) Send the neuron to 
       i) split_neuron_limbs_by_suggestions
       ii) split_disconnected_neuron


    If a limb is empty or has no more connetion to the
    starting soma then it will be deleted in the end

    """



    if verbose:
        print(f"Number of branches at beginning = {neuron_obj.n_branches} ")

    #For each limb in branch dict
    n_branches_at_start = int(neuron_obj.n_branches)

    split_neuron = neuron_obj
    for limb_name,branches_to_delete in limb_branch_dict.items():
        
        split_neuron = delete_branches_from_limb(split_neuron,
                              branches_to_delete=branches_to_delete,
                              limb_name=limb_name,
                              verbose=verbose,
                                )
        

    if plot_neuron_after_cancellation:
        nviz.visualize_neuron(split_neuron,
                             limb_branch_dict="all")

    if verbose:
        print(f"Number of branches after split_neuron_limb_by_seperated_network = {split_neuron.n_branches} ")

    disconnected_neuron = pru.split_disconnected_neuron(split_neuron,
                                                        return_errored_limbs_info=False,
                                                       add_split_to_description=add_split_to_description)

    if len(disconnected_neuron) != 1:
        raise Exception(f"The disconnected neurons were not 1: {disconnected_neuron}")

    return_neuron = disconnected_neuron[0]

    if verbose: 
        print(f"Number of branches after split_disconnectd neuron = {return_neuron} ")

    if plot_final_neuron:
        nviz.visualize_neuron(return_neuron,
                             limb_branch_dict="all")
    
    return return_neuron
    
    
# ----------- 1/28 Proofreading Rules that will help filter a neuron object --------------- #

def filter_away_limb_branch_dict(neuron_obj,
                                 limb_branch_dict=None,
                                 limb_edge_dict=None,
                                plot_limb_branch_filter_away=False,
                                 plot_limb_branch_filter_with_disconnect_effect=False,
                                return_error_info=True,
                                 plot_final_neuron=False,
                                 verbose=False,
                                 **kwargs
                                ):
    """
    Purpose: To filter away a limb branch dict from a single neuron
    
    
    """
    
    if len(limb_branch_dict) == 0:
        if verbose:
            print("limb_branch_dict was empty so returning original neuron")
        if return_error_info:
            return neuron_obj,0,0
        else:
            return neuron_obj
    
    if plot_limb_branch_filter_away:
        print("\n\nBranches Requested to Remove (without disconnect effect)")
        nviz.plot_limb_branch_dict(neuron_obj,
                             limb_branch_dict)


    #2) Find the total branches that will be removed using the axon-error limb branch dict

    removed_limb_branch_dict = nru.limb_branch_after_limb_branch_removal(neuron_obj=neuron_obj,
                                          limb_branch_dict = limb_branch_dict,
                                 return_removed_limb_branch = True,
                                )

    if verbose:
        print(f"After disconnecte effect, removed_limb_branch_dict = {removed_limb_branch_dict}")
        
    if plot_limb_branch_filter_with_disconnect_effect:
        print("\n\nBranches Requested to Remove (WITH disconnect effect)")
        nviz.plot_limb_branch_dict(neuron_obj,
                             removed_limb_branch_dict)
    
    #3) Calculate the total skeleton length and error faces area for what will be removed
    if return_error_info:
        total_area = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                           limb_branch_dict=removed_limb_branch_dict,
                                           feature="area")

        total_sk_distance = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                               limb_branch_dict=removed_limb_branch_dict,
                                               feature="skeletal_length")/1000
        
        if verbose:
            print(f"total_sk_distance = {total_sk_distance}, total_area = {total_area}")
        
    #4) Delete the brnaches from the neuron
    new_neuron = pru.delete_branches_from_neuron(neuron_obj,
                                    limb_branch_dict = removed_limb_branch_dict,
                                    plot_neuron_after_cancellation = False,
                                    plot_final_neuron = False,
                                    verbose = False,
                                    add_split_to_description=False,
                                    )

    if plot_final_neuron:
        nviz.visualize_neuron(new_neuron,
                             visualize_type=["mesh"],
                             mesh_whole_neuron=True)
    
    if return_error_info:
        return new_neuron,total_area,total_sk_distance
    else:
        return new_neuron
    
    
def filter_away_axon_on_dendrite_merges_old_limb_branch_dict(neuron_obj,**kwargs):
    return ns.query_neuron_by_labels(neuron_obj,matching_labels=["axon-error"])

def filter_away_axon_on_dendrite_merges_old(
    neuron_obj,
    perform_deepcopy = True,
    
    axon_merge_error_limb_branch_dict = None,
    perform_axon_classification = False,
    use_pre_existing_axon_labels = False,

    return_error_info=True,

    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect=False,
    plot_final_neuron = False,

    verbose = False,
    return_limb_branch_dict_to_cancel = False,
    prevent_errors_on_branches_with_all_postsyn = True,
    return_limb_branch_before_filter_away=False,
    **kwargs):

    """
    Pseudocode: 

    If error labels not given
    1a) Apply axon classification if requested
    1b) Use the pre-existing error labels if requested

    2) Find the total branches that will be removed using the axon-error limb branch dict
    3) Calculate the total skeleton length and error faces area for what will be removed
    4) Delete the brnaches from the neuron
    5) Return the neuron

    Example: 
    
    filter_away_axon_on_dendrite_merges(
    neuron_obj = neuron_obj_1,
    perform_axon_classification = True,
    return_error_info=True,
    verbose = True)
    """
    if perform_deepcopy:
        neuron_obj = neuron.Neuron(neuron_obj)

    if axon_merge_error_limb_branch_dict is None:
        if perform_axon_classification == False and use_pre_existing_axon_labels == False:
            raise Exception("Need to set either perform_axon_classification or use_pre_existing_axon_labels because"
                           f" axon_merge_error_limb_branch_dict is None")

        if use_pre_existing_axon_labels:

            if verbose:
                print("using pre-existing labels for axon-error detection")

            axon_merge_error_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,matching_labels=["axon-error"])
        else:

            if verbose:
                print("performing axon classification for axon-error detection")


            axon_limb_branch_dict,axon_merge_error_limb_branch_dict = clu.axon_classification(neuron_obj,
                            return_axon_labels=True,
                            return_error_labels=True,
                           plot_axons=False,
                           plot_axon_errors=False,
                            verbose=True,
                           )


    if prevent_errors_on_branches_with_all_postsyn:
        pass
        
    new_neuron,total_area,total_sk_distance = filter_away_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=axon_merge_error_limb_branch_dict,
                                    plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                     plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                    return_error_info=True,
                                     plot_final_neuron=plot_final_neuron,
                                     verbose=verbose,
                                                                           **kwargs
                                                                           
                                    )
    
    axon_merge_error_limb_branch_dict_after = nru.limb_branch_after_limb_branch_removal(neuron_obj=neuron_obj,
                                          limb_branch_dict = axon_merge_error_limb_branch_dict,
                                 return_removed_limb_branch = True,
                                )
    
    if return_error_info:
        if return_limb_branch_dict_to_cancel:
            if return_limb_branch_before_filter_away:
                return new_neuron,total_area,total_sk_distance,axon_merge_error_limb_branch_dict_after,axon_merge_error_limb_branch_dict
            else:
                return new_neuron,total_area,total_sk_distance,axon_merge_error_limb_branch_dict_after
        else:
            return new_neuron,total_area,total_sk_distance
    else:
        return new_neuron

#---------- Rule 2: Removing Dendritic Merges on Axon ------------- #

def filter_away_dendrite_on_axon_merges_old(
    neuron_obj,
    perform_deepcopy=True,
    
    limb_branch_dict_for_search=None,
    use_pre_existing_axon_labels=False,
    perform_axon_classification=False,
    
    dendritic_merge_on_axon_query=None,
    dendrite_merge_skeletal_length_min = 20000,
    dendrite_merge_width_min = 100,
    dendritie_spine_density_min = 0.00015,
    
    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    return_error_info = False,
    plot_final_neuron = False,
    return_limb_branch_dict_to_cancel = False,
    verbose=False,
    ):
    """
    Purpose: To filter away the dendrite parts that are 
    merged onto axon pieces
    
    if limb_branch_dict_for_search is None then 
    just going to try and classify the axon and then 
    going to search from there
    
    
    """
    
    
    if perform_deepcopy:
        neuron_obj = neuron.Neuron(neuron_obj)

    if limb_branch_dict_for_search is None:
        if use_pre_existing_axon_labels == False and perform_axon_classification == False:
            raise Exception("Need to set either perform_axon_classification or use_pre_existing_axon_labels because"
                           f" limb_branch_dict_for_search is None")
        if not use_pre_existing_axon_labels:
            axon_limb_branch_dict,axon_error_limb_branch_dict = clu.axon_classification(neuron_obj,
                                        return_error_labels=True,
                                        plot_candidates=False,
                                        plot_axons=False,
                                        plot_axon_errors=False)
        else:
            if verbose:
                print("Using pre-existing axon and axon-like labels")


    if dendritic_merge_on_axon_query is None:
        dendritic_merge_on_axon_query = (f"labels_restriction == True and "
                    f"(median_mesh_center > {dendrite_merge_width_min}) and  "
                    f"(skeletal_length > {dendrite_merge_skeletal_length_min}) and "
                    f"(spine_density) > {dendritie_spine_density_min}")


    function_kwargs = dict(matching_labels=["axon"],
                          not_matching_labels=["axon-like"])

    dendritic_branches_merged_on_axon = ns.query_neuron(neuron_obj,
                    functions_list=["labels_restriction","median_mesh_center",
                                   "skeletal_length","spine_density"],
                    query=dendritic_merge_on_axon_query,
                    function_kwargs=function_kwargs)

    if verbose:
        print(f"dendritic_branches_merged_on_axon = {dendritic_branches_merged_on_axon}")
        
        
    

    (dendrite_stripped_neuron,
        total_area_dendrite_stripped,
         total_sk_distance_stripped) = pru.filter_away_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=dendritic_branches_merged_on_axon,
                                        plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                         plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        return_error_info=True,
                                         plot_final_neuron=plot_final_neuron,
                                         verbose=verbose
                                        )
    
    if return_error_info:
        if return_limb_branch_dict_to_cancel:
            return (dendrite_stripped_neuron,
                    total_area_dendrite_stripped,
                     total_sk_distance_stripped,
                   dendritic_branches_merged_on_axon)
        else:
            return (dendrite_stripped_neuron,
                    total_area_dendrite_stripped,
                     total_sk_distance_stripped)
    else:
        return dendrite_stripped_neuron
    
    
# ------------ Rule 3: Filtering away axon mess ----------- #

def filter_away_limb_branch_dict_with_function(
    neuron_obj,
    limb_branch_dict_function,
    perform_deepcopy=True,
    
    
    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    return_error_info = False,
    plot_final_neuron = False,
    print_limb_branch_dict_to_cancel = True,
    verbose=False,
    return_limb_branch_dict_to_cancel = False,
    return_limb_branch_before_filter_away = False,
    return_created_edges = False,
    apply_after_removal_to_limb_branch_before = True,
    **kwargs #These argument will be for running the function that will come up with limb branch dict
    ):
    """
    Purpose: To filter away a limb branch dict from
    a neuron using a function that generates a limb branch dict
    
    """
    
    
    if perform_deepcopy:
        neuron_obj = neuron.Neuron(neuron_obj)

        
    ret_val = limb_branch_dict_function(neuron_obj,
                             verbose=verbose,
                             **kwargs)
    if return_created_edges:
        limb_branch_dict_to_cancel,created_edges = ret_val
    else:
        limb_branch_dict_to_cancel = ret_val
        
    # doing the cancelling beforehand
    limb_branch_dict_to_cancel_pre = limb_branch_dict_to_cancel.copy()
    if apply_after_removal_to_limb_branch_before:
        limb_branch_dict_to_cancel = nru.limb_branch_after_limb_branch_removal(neuron_obj=neuron_obj,
                                              limb_branch_dict = limb_branch_dict_to_cancel,
                                     return_removed_limb_branch = True,
                                    )

    
    
    if print_limb_branch_dict_to_cancel:
        print(f"limb_branch_dict_to_cancel = {limb_branch_dict_to_cancel}")

    (dendrite_stripped_neuron,
    total_area_dendrite_stripped,
     total_sk_distance_stripped) = pru.filter_away_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict_to_cancel,
                                        plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                         plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        return_error_info=True,
                                         plot_final_neuron=plot_final_neuron,
                                         verbose=verbose
                                        )
    
    if return_error_info:
        if not return_limb_branch_dict_to_cancel:
            if return_created_edges:
                return (dendrite_stripped_neuron,
                        total_area_dendrite_stripped,
                         total_sk_distance_stripped,
                       created_edges)
            else:
                return (dendrite_stripped_neuron,
                        total_area_dendrite_stripped,
                         total_sk_distance_stripped,
                       )
        else:
            if return_limb_branch_before_filter_away:
                if return_created_edges:
                    return (dendrite_stripped_neuron,
                            total_area_dendrite_stripped,
                             total_sk_distance_stripped,
                           limb_branch_dict_to_cancel,
                            limb_branch_dict_to_cancel_pre,
                           created_edges)
                else:
                    return (dendrite_stripped_neuron,
                            total_area_dendrite_stripped,
                             total_sk_distance_stripped,
                           limb_branch_dict_to_cancel,
                            limb_branch_dict_to_cancel_pre,
                        
                           )
            else:
                if return_created_edges:
                    return (dendrite_stripped_neuron,
                            total_area_dendrite_stripped,
                             total_sk_distance_stripped,
                           limb_branch_dict_to_cancel,
                           created_edges)
                else:
                    return (dendrite_stripped_neuron,
                            total_area_dendrite_stripped,
                             total_sk_distance_stripped,
                           limb_branch_dict_to_cancel,
                        
                           )
    else:
        if not return_limb_branch_dict_to_cancel:
            return dendrite_stripped_neuron
        else:
            if return_limb_branch_before_filter_away:
                if return_created_edges:
                    return dendrite_stripped_neuron, limb_branch_dict_to_cancel,limb_branch_dict_to_cancel_pre,created_edges
                else:
                    return dendrite_stripped_neuron, limb_branch_dict_to_cancel,limb_branch_dict_to_cancel_pre
            else:
                if return_created_edges:
                    return dendrite_stripped_neuron, limb_branch_dict_to_cancel,created_edges
                else:
                    return dendrite_stripped_neuron, limb_branch_dict_to_cancel


def filter_away_low_branch_length_clusters(neuron_obj,
                                           max_skeletal_length=5000,
                                           min_n_nodes_in_cluster=4,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=nru.low_branch_length_clusters,
                 max_skeletal_length=max_skeletal_length,
                 min_n_nodes_in_cluster=min_n_nodes_in_cluster,
                 return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)
    
    
# ---------- Rule 4: Width and doubling back rules ---------------- #


def edges_to_cut_by_doubling_back_and_width_change(
    limb_obj,
    
    #--- Parameter for the width_jump_double_back_edges_path function
    skeletal_length_to_skip=5000,

    # parameters for the boundary transition
    comparison_distance = 3000,
    offset=1000,


    #the thresholds for determining if there are errors
    width_jump_threshold = 300,
    width_jump_axon_like_threshold = 250,
    running_width_jump_method=True, 
    
                                     
    double_back_threshold = 120,
    double_back_axon_like_threshold = 145,

    perform_double_back_errors = True,
    perform_width_errors = True,
    skip_double_back_errors_for_axon = True,
    
    verbose = False,
    **kwargs
    ):
    """
    Purpose: Getting the edges of the concept
    network to cut for a limb object based on 
    the width and doubling back rules
    
    Application: Will then feed these edges in to cut the limb
    when automatic proofreading

    """
    #verbose = True


    curr_limb =limb_obj
    curr_limb.set_concept_network_directional()
    dir_nx = nx.Graph(curr_limb.concept_network_directional)
    paths_to_end_nodes = xu.all_path_from_start_to_end_nodes(dir_nx,curr_limb.current_starting_node)


    edges_to_cut_from_double_back = []
    edges_to_cut_from_width = []

    starting_coordinate = curr_limb.current_starting_coordinate

    checked_edges = []
    
    for jj,ex_path in enumerate(paths_to_end_nodes):
        if verbose:
            print(f"\n--- Working on path {jj}: {ex_path} -- #")
    

        all_cuts,double_back_cuts,width_cuts,all_sk_angles,all_widths = ed.width_jump_double_back_edges_path(
                        limb_obj = curr_limb,
                        path = ex_path,
                        starting_coordinate=starting_coordinate,
                        
                        skeletal_length_to_skip=skeletal_length_to_skip,

                        # parameters for the boundary transition
                        comparison_distance = comparison_distance,
                        offset=offset,


                        #the thresholds for determining if there are errors
                        width_jump_threshold = width_jump_threshold,
                        width_jump_axon_like_threshold = width_jump_axon_like_threshold,
                        running_width_jump_method=running_width_jump_method,
            
            
                        double_back_threshold = double_back_threshold,
                        double_back_axon_like_threshold = double_back_axon_like_threshold,
        
    

                        perform_double_back_errors = perform_double_back_errors,
                        perform_width_errors = perform_width_errors,
                        skip_double_back_errors_for_axon = skip_double_back_errors_for_axon,
            
                        verbose=False,
            
                        **kwargs)
        if verbose:
            print(f"all cuts = {all_cuts}")
            print(f"double_back_cuts = {double_back_cuts}")
            print(f"width_cuts = {width_cuts}")
            print(f"all_sk_angles = {all_sk_angles}")
            print(f"all_widths = {all_widths}")

        if len(double_back_cuts) > 0:
            edges_to_cut_from_double_back.append(double_back_cuts)

        if len(width_cuts) > 0:
            edges_to_cut_from_width.append(width_cuts)

    if len(edges_to_cut_from_double_back)>0:
        edges_to_cut_from_double_back = list(np.vstack(edges_to_cut_from_double_back).reshape(-1,2))
    else:
        edges_to_cut_from_double_back = []

    if len(edges_to_cut_from_width)>0:
        edges_to_cut_from_width = list(np.vstack(edges_to_cut_from_width).reshape(-1,2))
    else:
        edges_to_cut_from_width = []

    return edges_to_cut_from_double_back,edges_to_cut_from_width

def edges_to_create_and_delete_by_doubling_back_and_width(limb_obj,
                                                          verbose=False,
                                                         **kwargs):
    """
    Wrapper for the doubling back and width cuts that will generate edges to delete
    and create so can fit the edge pipeline
    
    """
    
    double_back_cuts_total, width_cuts_total = edges_to_cut_by_doubling_back_and_width_change(limb_obj,
                                                                                              verbose=False,
                                                  **kwargs)
    edges_to_delete = double_back_cuts_total + width_cuts_total
    edges_to_create = []
    
    return edges_to_create,edges_to_delete



def doubling_back_and_width_elimination_limb_branch_dict(neuron_obj,
    
                                            #--- Parameter for the width_jump_double_back_edges_path function
                                            skeletal_length_to_skip=5000,

                                            # parameters for the boundary transition
                                            comparison_distance = 4000,
                                            offset=2000, #have to make the offset larger because the spines are cancelled out 2000 from the endpoints


                                            #the thresholds for determining if there are errors
                                            width_jump_threshold = 300,#200,
                                            width_jump_axon_like_threshold = 250,
                                            double_back_threshold = 140,

                                            perform_double_back_errors = True,
                                            perform_width_errors = True,
                                            skip_double_back_errors_for_axon = True,

                                            verbose = False,
                                                **kwargs):
    
    
    
    double_back_width_limb_branch_dict = nru.limb_branch_from_edge_function(neuron_obj,
                                   edge_function = pru.edges_to_create_and_delete_by_doubling_back_and_width,
                                                                            
                                    skeletal_length_to_skip=skeletal_length_to_skip,

                                    # parameters for the boundary transition
                                    comparison_distance = comparison_distance,
                                    offset=offset,


                                    #the thresholds for determining if there are errors
                                    width_jump_threshold = width_jump_threshold,
                                    width_jump_axon_like_threshold = width_jump_axon_like_threshold,
                                    double_back_threshold = double_back_threshold,

                                    perform_double_back_errors = perform_double_back_errors,
                                    perform_width_errors = perform_width_errors,
                                    skip_double_back_errors_for_axon = skip_double_back_errors_for_axon,

                                    verbose = verbose,
                                                                            **kwargs
                                    
                                  )
    
    return double_back_width_limb_branch_dict



def filter_away_large_double_back_or_width_changes(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=pru.doubling_back_and_width_elimination_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)
    
    
# ------------- Rule 5: Resolving Crossing Traintracks --------------------- #


def edges_to_create_and_delete_crossover(
                                        limb_obj,
                                        offset = None,
                                        comparison_distance = None,
                                        match_threshold = None,
                                        axon_dependent = True,
                                        require_two_pairs = True,
                                        verbose = False,
                                        
                                        ):

    """
    Purpose: To seperate train track crossovers if there are perfect matches
    and ignore them if there are not

    Pseudocode: 
    1) Find all 4 degree skeleton coordinates
    2) For each coordinate:
    a) if axon dependnet --> check that all of them are axon (if not then continue)
    b) do resolve the crossover with the best pairs
    c) if 2 perfect pairs --> then add the delete and remove edges to the big list
        not 2 pairs --> skip


    """
    if comparison_distance is None:
        comparison_distance = comparison_distance_high_degree_global
        
    if offset is None:
        offset = offset_high_degree_global
        
    if match_threshold is None:
        match_threshold = match_threshold_high_degree_global
        
    
    if require_two_pairs:
        exactly_equal=True
    else:
        exactly_equal = False

    crossover_coordinates = nru.high_degree_branching_coordinates_on_limb(limb_obj,min_degree_to_find=4,
                                                                         exactly_equal=exactly_equal,
                                                                         )

    if verbose:
        print(f"crossover_coordinates = {crossover_coordinates}")

    edges_to_create = []
    edges_to_delete = []

    #2) For each oe resolve the crossover with the best pairs
    for i,c_coord in enumerate(crossover_coordinates):

        if axon_dependent:
            continue_flag=False
            branches_at_coord = list(nru.find_branch_with_specific_coordinate(limb_obj,c_coord))
            
            
            if not nru.axon_only_group(limb_obj,
                                   branches_at_coord,
                                   use_axon_like=True,
                                   verbose=verbose):
                continue
                
#             for b in branches_at_coord:
#                 if "axon-like" not in limb_obj[b].labels and "axon" not in limb_obj[b].labels:
#                     if verbose:
#                         print(f"Skipping coordinate {i} ({c_coord}) because branch {b} was not an axon-like or axon piece ")
#                     continue_flag=True
#                     break

#             if continue_flag:
#                 continue

        curr_edges_to_delete,curr_edges_to_create = ed.resolving_crossovers(limb_obj,
                            coordinate =c_coord,
                             offset=offset,
                             comparison_distance=comparison_distance,
                            match_threshold = match_threshold,
                            require_two_pairs=require_two_pairs,
                            #verbose = False,
                            verbose = False,
                             return_new_edges = True,
                            #plot_intermediates=False,
                            plot_intermediates=False,
                            )
        
        if len(curr_edges_to_create) == 2 or not require_two_pairs:
            if verbose:
                print(f"Found matching train tracks for coordinate {i} = {c_coord}")
            edges_to_create += list(curr_edges_to_create)
            edges_to_delete += list(curr_edges_to_delete)
            
        else:
            if verbose:
                print(f"Only found {len(curr_edges_to_create)} successful matches so skipping")

    return edges_to_create,edges_to_delete

def crossover_elimination_limb_branch_dict(neuron_obj,
                                        offset = 2500,
                                        comparison_distance = None,
                                        match_threshold = 35,
                                           require_two_pairs=True,
                                        axon_dependent = True,
                                                       **kwargs):
    if comparison_distance is None:
        comparison_distance = comparison_distance_high_degree_global
    
    high_degree_limb_branch = nru.limb_branch_from_edge_function(neuron_obj,
                                   edge_function = pru.edges_to_create_and_delete_crossover,
                                                                  offset = offset,
                                                                    comparison_distance = comparison_distance,
                                                                    match_threshold = match_threshold,
                                                                    axon_dependent = axon_dependent,
                                                                 require_two_pairs=require_two_pairs,
                                                                  **kwargs)
                                                                  
    return high_degree_limb_branch    

def filter_away_crossovers(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=pru.crossover_elimination_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

# ------------- Rule 6: Filtering Away High Degree Edges --------------------- #

def edges_to_create_and_delete_high_degree_coordinates(limb_obj,
                                                     min_degree_to_find = 5,
                                                       axon_dependent = True,
                                                     verbose=False):

    """
    Purpose: Cut all edges at branches
    grouped around a high degree skeleton node

    Pseudocode:
    1) Find Branches Grouped around a high degree skeleton node
    2) Get all combinations of the branches
    3) Make those combinations the edges to delete, and make edges to create empty
    
    Ex: 
    limb_obj =  neuron_obj[0]
    min_degree_to_find = 5

    edges_to_create_and_delete_high_degree_coordinate(limb_obj,
                                                         min_degree_to_find = min_degree_to_find,
                                                         verbose=True)
    """
    
    #1) Find Branches Grouped around a high degree skeleton node
    high_degree_branch_groups = nru.branches_at_high_degree_coordinates(limb_obj,
                                                                       min_degree_to_find=min_degree_to_find,
                                                                       verbose=verbose)
    if verbose:
        print(f"high_degree_branch_groups = {high_degree_branch_groups}")
        
    #2) Get all combinations of the branches
    edges_to_delete = []
    edges_to_create = []
    
    for b_group in high_degree_branch_groups:
        if axon_dependent:
            if not nru.axon_only_group(limb_obj,
                                   b_group,
                                   use_axon_like=True,
                                   verbose=verbose):
                continue
            
        
        all_pairings = list(nu.all_unique_choose_2_combinations(b_group))
        
        if verbose:
            print(f"For branch group ({b_group}), the edges to cut were: {all_pairings}")
        edges_to_delete += all_pairings
    
    return edges_to_create,edges_to_delete

def high_degree_coordinates_elimination_limb_branch_dict(neuron_obj,
                                                       min_degree_to_find=5,
                                                         axon_dependent=True,
                                                       **kwargs):
    
    high_degree_limb_branch = nru.limb_branch_from_edge_function(neuron_obj,
                                   edge_function = pru.edges_to_create_and_delete_high_degree_coordinates,
                                                                  min_degree_to_find=min_degree_to_find,
                                                                 axon_dependent=axon_dependent,
                                                                  **kwargs)
                                                                  
    return high_degree_limb_branch    


def filter_away_high_degree_coordinates(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=pru.high_degree_coordinates_elimination_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)
                                                                  
#  ---- 2/2: Pipeline for Applying Filters ------:
def make_filter_dict(filter_name,
                    filter_function,
                    filter_kwargs=None,
                     catch_error = False
                    ):
    
    if filter_kwargs is None:
        filter_kwargs = dict()
        
    return dict(filter_name=filter_name,
               filter_function=filter_function,
               filter_kwargs=filter_kwargs,
               catch_error=catch_error)


def apply_proofreading_filters_to_neuron(
    input_neuron,
    filter_list,
    plot_limb_branch_filter_with_disconnect_effect=False,
    plot_limb_branch_filter_away=False,
    plot_final_neuron=False,
    return_error_info=False,
    verbose=False,
    verbose_outline=True,
    return_limb_branch_dict_to_cancel = False,
    return_red_blue_splits = False,
    return_split_locations = False,
    save_intermediate_neuron_objs = False,
    combine_path_branches = False,
    ):
    """
    Purpose: To apply a list of filters to a neuron
    and collect all of the information on what was filtered 
    away and the remaining neuron after

    - Be able to set certain run arguments that could
    help with plotting along the way 


    Pseudocode: 
    1) Recieve the input neuron

    For each filter:
    a) print the name of the function, and the arguments to go along
    b) Run the function with the function argument and run arguments
    c) print the time it takes
    d) Print the erro information
    e) Store the following information
    - neuron
    - error skeleton
    - error area
    time

    f) Make the output neuron as the new input neuron

    # -- Adding Optional parameter that allows a filter to recover from an error gracefully -- #
    
    
    """

    #print(f"filter_list = \n{filter_list}")

    output_info = dict()

    total_apply_filter_time = time.time()
    

    for z,current_filter in enumerate(filter_list):
        #print(f"input_neuron.description = {input_neuron.description}")
        
        if type(current_filter) == dict:
            filter_name = current_filter["filter_name"]
            filter_function = current_filter["filter_function"]
            filter_kwargs = current_filter["filter_kwargs"]
            filter_catch_error = current_filter.get("catch_error",False)
        elif gu.is_function(current_filter):
            filter_function = current_filter
            filter_name = str(current_filter.__name__)
            filter_kwargs = dict()
            filter_catch_error = False
        else:
            raise Exception(f"Unknown type for filter: {current_filter} of type {type(current_filter)}")
        

        if verbose or verbose_outline:
            print(f"\n--- Working on filter {z}:\n"
                 f"function = {filter_name}\n"
                  f"function __name__ = {filter_function.__name__}\n"
                 f"function arguments = {filter_kwargs}\n\n")

        if verbose:
            print("----------------------- Running Filter ------------------")
            
        filter_time = time.time()
        
#         try:
        output_res= filter_function(input_neuron,
            plot_limb_branch_filter_with_disconnect_effect = plot_limb_branch_filter_with_disconnect_effect,
            plot_limb_branch_filter_away = plot_limb_branch_filter_away,
            plot_final_neuron=plot_final_neuron,
                                            return_error_info=True,
                                            return_limb_branch_dict_to_cancel = return_limb_branch_dict_to_cancel,
                                            verbose=verbose,
                                            return_limb_branch_before_filter_away = True,
                                            **filter_kwargs
                                           )
    
    
    
        limb_branch_dict_to_cancel = {}
        red_blue_suggestions = {}
        created_edges = None
        split_locations = {}
        split_locations_before = {}
        
        if return_limb_branch_dict_to_cancel:
            if len(output_res) == 5:
                (output_neuron,
                 total_area_current,
                 total_sk_distance_current,
                limb_branch_dict_to_cancel,
                limb_branch_dict_to_cancel_before_filt)  = output_res

                created_edges = None
            else:
                (output_neuron,
                 total_area_current,
                 total_sk_distance_current,
                limb_branch_dict_to_cancel,
                 limb_branch_dict_to_cancel_before_filt.
                created_edges)  = output_res

            """
            -- 5/10 Want to generate the red and blue points as well


            """
#             red_blue_suggestions = pru.extract_blue_red_points_from_limb_branch_dict_to_cancel(
#                 input_neuron,
#                 limb_branch_dict_to_cancel)

            if return_red_blue_splits:
                red_blue_suggestions = pru.limb_branch_dict_to_cancel_to_red_blue_groups(neuron_obj=input_neuron,
                                                limb_branch_dict_to_cancel=limb_branch_dict_to_cancel,
                                                            created_edges = created_edges,
                                                            plot_all_blue_red_groups = False,
                                                             verbose = False)
            if return_split_locations:
                from neurd import limb_utils as lu
                split_locations_before_filter = lu.most_upstream_endpoints_of_limb_branch(
                    input_neuron,
                    limb_branch_dict = limb_branch_dict_to_cancel_before_filt,
                    group_by_conn_comp = False,# "axon_on_dendrite" in filter_name,
                    verbose = False
                    )
                split_locations= lu.most_upstream_endpoints_of_limb_branch(
                    input_neuron,
                    limb_branch_dict = limb_branch_dict_to_cancel,
                    group_by_conn_comp = False,
                    verbose = False
                    )
            
                
            

        else:

            (output_neuron,
             total_area_current,
             total_sk_distance_current) = output_res

            
                
#         except Exception as e:
#             if not filter_catch_error:
#                 raise Exception(str(e))
#             else:
#                 print(f"Becasue catch_error = {filter_catch_error}, Gracefullly recovering from error {str(e)}")
                
#                 output_neuron = input_neuron
#                 total_area_current = 0
#                 total_sk_distance_current = 0
#                 limb_branch_dict_to_cancel = {}
#                 red_blue_suggestions = {}
            
        
        
        if verbose:
            print("----------------------- FINISHED Running Filter ------------------\n\n")

        elapsed_time = time.time() - filter_time

        #c) print the time it takes
        #d) Print the erro information
        if False:
            print(f"\n --Filter {filter_name} Results --")
            print(f"Time = {elapsed_time}")
            print(f"error_area={total_area_current}")
            print(f"error_length={total_sk_distance_current}")
            print(f"limb_branch_dict_to_cancel = {limb_branch_dict_to_cancel}")

        """
        e) Store the following information
        - neuron
        - error skeleton
        - error area
        """
        local_results = {
                            f"{filter_name}_time":elapsed_time,
                            f"{filter_name}_error_area":total_area_current,
                            f"{filter_name}_error_length":total_sk_distance_current}
        
        if combine_path_branches:
            #print(f"\n\n\n***combining path branches****\n\n")
            output_neuron = nsimp.combine_path_branches(
                output_neuron,
                plot_downstream_path_limb_branch = False,
                verbose = verbose,
            )
        
        
        
        if save_intermediate_neuron_objs:
            local_results.update({f"{filter_name}_neuron":output_neuron,})
            
        if return_limb_branch_dict_to_cancel:
            local_results.update({f"{filter_name}_limb_branch_dict_to_cancel":limb_branch_dict_to_cancel,
                                  f"{filter_name}_created_edges":created_edges,
                                 f"{filter_name}_red_blue_suggestions":red_blue_suggestions,
                                 })
        if return_split_locations:
            local_results.update({f"{filter_name}_split_locations":split_locations,
                                  f"{filter_name}_split_locations_before_filter":split_locations_before_filter
                                 })

        if verbose:
            print(f"\n --Filter {filter_name} Results --")
            #print(f"local_results = {local_results}")

        output_info.update(local_results)

        #f) Make the output neuron as the new input neuron
        input_neuron = output_neuron

    if verbose or verbose_outline:
        print(f"\n\n\n ---- Total time for applying filter: {time.time() - total_apply_filter_time} -----")
    
    if return_error_info:
        return input_neuron,output_info
    else:
        return input_neuron
    
def get_exc_filters():
    exc_axon_on_dendrite_merges_filter = pru.make_filter_dict("axon_on_dendrite_merges",
                                         pru.filter_away_axon_on_dendrite_merges_old,
                                         dict(use_pre_existing_axon_labels=True)
                                              
                                        )

    exc_low_branch_clusters_filter = pru.make_filter_dict("low_branch_clusters",
                                            pru.filter_away_low_branch_length_clusters,
                                            dict())

    exc_dendrite_on_axon_merges_filter = pru.make_filter_dict("dendrite_on_axon_merges",
                                                         pru.filter_away_dendrite_on_axon_merges_old,
                                                          dict(use_pre_existing_axon_labels=True)
                                                         )
    exc_double_back_and_width_change_filter = pru.make_filter_dict("double_back_and_width_change",
                                                         pru.filter_away_large_double_back_or_width_changes,
                                                          dict(perform_double_back_errors=True,
                                                              skip_double_back_errors_for_axon=False,
                                                              #double_back_threshold = 140,
                                                               
                                                               width_jump_threshold = 250,
                                                               running_width_jump_method=True, 
                                                               
                                                               
                                                               double_back_axon_like_threshold=145,
                                                               #double_back_threshold = 115,
                                                               double_back_threshold = 120,
                                                              ),
                                                                   catch_error=True,
                                                         )
    exc_crossovers_filter = pru.make_filter_dict("crossovers",
                                                         pru.filter_away_crossovers,
                                                          dict(axon_dependent=True,
                                                              match_threshold = 30)
                                                         )
    
    exc_high_degree_coordinates_filter = pru.make_filter_dict("high_degree_coordinates",
                                                         pru.filter_away_high_degree_coordinates,
                                                          dict(axon_dependent=True,min_degree_to_find=4)
                                                         )

    
    exc_filters = [exc_axon_on_dendrite_merges_filter,
                  exc_low_branch_clusters_filter,
                  exc_dendrite_on_axon_merges_filter,
                  exc_double_back_and_width_change_filter,
                  exc_crossovers_filter,
                  exc_high_degree_coordinates_filter,]
    
    return exc_filters


def get_inh_filters():
    inh_low_branch_clusters_filter = pru.make_filter_dict("low_branch_clusters",
                                            pru.filter_away_low_branch_length_clusters,
                                            dict())
    # --------- 2/16: No longer making the cross overs axon dependent ----------- #
    inh_crossovers_filter = pru.make_filter_dict("crossovers",
                                                         pru.filter_away_crossovers,
                                                          dict(axon_dependent=False,
                                                              match_threshold = 30)
                                                         )

    inh_double_back_and_width_change_filter = pru.make_filter_dict("double_back_and_width_change",
                                                         pru.filter_away_large_double_back_or_width_changes,
                                                          dict(perform_double_back_errors=True,
                                                              skip_double_back_errors_for_axon=False,
                                                               
                                                               width_jump_threshold = 250,
                                                              running_width_jump_method=True, 
                                                               
                                                               
                                                               double_back_axon_like_threshold=120,
                                                               #double_back_threshold = 115,
                                                               double_back_threshold = 115,
                                                              
                                                              ),
                                                                   catch_error=True,
                                                         )
    
    inh_high_degree_coordinates_filter = pru.make_filter_dict("high_degree_coordinates",
                                                         pru.filter_away_high_degree_coordinates,
                                                          dict(axon_dependent=False,min_degree_to_find=4)
                                                         )



    inh_filters = [inh_low_branch_clusters_filter,
                   inh_crossovers_filter,
                   inh_double_back_and_width_change_filter,
                   inh_high_degree_coordinates_filter,
                  ]

    return inh_filters


def get_exc_filters_high_fidelity_axon_preprocessing_old():
    exc_dendrite_on_axon_merges_filter = pru.make_filter_dict("dendrite_on_axon_merges",
                                                         pru.filter_away_dendrite_on_axon_merges_old,
                                                          dict(use_pre_existing_axon_labels=True)
                                                         )
    return [exc_dendrite_on_axon_merges_filter]

def get_exc_filters_high_fidelity_axon_postprocessing_old():
    exc_axon_on_dendrite_merges_filter = pru.make_filter_dict("axon_on_dendrite_merges",
                                         pru.filter_away_axon_on_dendrite_merges_old,
                                         dict(use_pre_existing_axon_labels=True)
                                              
                                        )

    exc_low_branch_clusters_filter = pru.make_filter_dict("low_branch_clusters",
                                            pru.filter_away_low_branch_length_clusters,
                                            dict())

    exc_double_back_and_width_change_filter = pru.make_filter_dict("double_back_and_width_change",
                                                         pru.filter_away_large_double_back_or_width_changes,
                                                          dict(perform_double_back_errors=True,
                                                              skip_double_back_errors_for_axon=False,
                                                              #double_back_threshold = 140,
                                                               
                                                               width_jump_threshold = 250,
                                                               running_width_jump_method=True, 
                                                               
                                                               
                                                               double_back_axon_like_threshold=145,
                                                               #double_back_threshold = 115,
                                                               double_back_threshold = 120,
                                                              ),catch_error=True,
                                                         )
    exc_crossovers_filter = pru.make_filter_dict("crossovers",
                                                         pru.filter_away_crossovers,
                                                          dict(axon_dependent=True,
                                                              match_threshold = 50,
                                                              require_two_pairs=False)
                                                         )
    
    exc_high_degree_coordinates_filter = pru.make_filter_dict("high_degree_coordinates",
                                                         pru.filter_away_high_degree_coordinates,
                                                          dict(axon_dependent=True,min_degree_to_find=4)
                                                         )
    
    
    exc_filters = [exc_axon_on_dendrite_merges_filter,
                   exc_crossovers_filter,
                  #exc_low_branch_clusters_filter,
                  exc_double_back_and_width_change_filter,
                  exc_high_degree_coordinates_filter,]
    
    return exc_filters


def proofread_neuron(
    
    input_neuron,

    attempt_to_split_neuron = False,
    plot_neuron_split_results = False,

    plot_neuron_before_filtering = False,

    plot_axon = False,
    plot_axon_like = False,

    # -- for the filtering loop
    plot_limb_branch_filter_with_disconnect_effect = True,
    plot_final_filtered_neuron = False,

    # -- for the output --
    return_process_info = True,

    debug_time = True,
    verbose = True,
    verbose_outline=True,
    
    #will apply the high fidelity axon process on excitatory cells
    high_fidelity_axon_on_excitatory=True,
    
    # --------- Arguments to make it a streamlined process  ----------- #
    inh_exc_class = None,
    perform_axon_classification = True,

    ):
    
    
    """
    Purpose: To apply all of the 
    proofreading rules to a neuron (or a pre-split neuron)
    and to return the proofread neuron and all of the 
    error information

    Pseudocode: 
    1) If requested try and split the neuron
    2) Put the neuron(s) into a list

    For each neuron
    a) Check that there are not any error limbs or
       multiple somas 
    b) Run the axon classification
    c) Run the excitatory and inhibitory classification (save results in dict)
    d) Based on cell type--> get the filters going to use
    e) Apply the filters to the neuron --> save the error information

    3) If not requested to split neuron, then just return the 
    just the single neuron



    Ex: 
    
    pru.proofread_neuron(
    
    input_neuron = neuron_obj_original,

    attempt_to_split_neuron = True,
    plot_neuron_split_results = False,

    plot_neuron_before_filtering = False,

    plot_axon = False,
    plot_axon_like = False,

    # -- for the filtering loop
    plot_limb_branch_filter_with_disconnect_effect = True,
    plot_final_filtered_neuron = True,

    # -- for the output --
    return_process_info = True,

    debug_time = True,
    verbose = False,
    verbose_outline=True

    )
    """
    
    
    neuron_obj_original = input_neuron
    proof_time = time.time()

    # -------------------

    #1) If requested try and split the neuron
    if attempt_to_split_neuron:
        if verbose or verbose_outline:
            print("---- Part A: Attempting to split neuron --------")


        filter_time = time.time()
        
        split_results = pru.multi_soma_split_suggestions(neuron_obj_original,
                                                         verbose = verbose,
                                                        )

        if plot_neuron_split_results:
            nviz.plot_split_suggestions_per_limb(neuron_obj_original,
                                            split_results,
                                                verbose=verbose)

        (neuron_list,
        neuron_list_errored_limbs_area,
         neuron_list_errored_limbs_skeletal_length,
        neuron_list_n_multi_soma_errors,
        neuron_list_n_same_soma_errors) = pru.split_neuron(neuron_obj_original,
                        limb_results=split_results,
                                       verbose=False,
                                        return_error_info=True
                                            ) 

        if debug_time:
            print(f"Time for Split Neuron = {time.time() - filter_time}")
            filter_time = time.time()

        split_error_info = dict(errored_limbs_area=neuron_list_errored_limbs_area,
                               errored_limbs_skeletal_length=neuron_list_errored_limbs_skeletal_length,
                                n_multi_soma_errors=neuron_list_n_multi_soma_errors,
                                n_same_soma_errors = neuron_list_n_same_soma_errors

                               )

        if verbose: 
            print(f" # of neurons = {len(neuron_list)}")
            print(f"neuron_list_errored_limbs_area = {neuron_list_errored_limbs_area}")
            print(f"neuron_list_errored_limbs_skeletal_length = {neuron_list_errored_limbs_skeletal_length}")
            print(f"neuron_list_n_multi_soma_errors = {neuron_list_n_multi_soma_errors}")
            print(f"neuron_list_n_same_soma_errors = {neuron_list_n_same_soma_errors}")
            
        if verbose_outline and not verbose:
            print(f" # of neurons found after split= {len(neuron_list)}")
    else:
        if verbose or verbose_outline:
            print("---- Part A: NOT Attempting to split neuron --------")

        split_error_info  = {'errored_limbs_area': [[]],
         'errored_limbs_skeletal_length': [[]],
         'n_multi_soma_errors': [ 0],
         'n_same_soma_errors': [ 0]}

        neuron_list = [input_neuron]


    """
    For each neuron
    a) Check that there are not any error limbs or
       multiple somas 
    b) Run the axon classification
    c) Run the excitatory and inhibitory classification (save results in dict)
    d) Based on cell type--> get the filters going to use
    e) Apply the filters to the neuron --> save the error information
    """
    neuron_list_results = []
    for n_idx,neuron_obj in enumerate(neuron_list):

        local_split_error_info = {k:v[n_idx] for k,v in split_error_info.items()}


        if verbose or verbose_outline:
            print(f"\n--- Working on Neuron {n_idx} ---")

        #a) Check that there are not any error limbs or multiple somas 
        if plot_neuron_before_filtering:
            nviz.visualize_neuron_lite(neuron_obj)

        #b) Run the axon classification

        if neuron_obj.n_somas > 1:
            raise Exception(f"Number of Somas was greater than 1: {neuron_obj.n_somas}")

        if neuron_obj.n_error_limbs > 0:
            raise Exception(f"Number of Error Limbs was greater than 1: {neuron_obj.n_error_limbs}")




        #b) Run the axon classification
        filter_time = time.time()

        if verbose or verbose_outline:
            print(f"\n\n ------ Part B: Axon Classification ---- \n\n")
            
        if perform_axon_classification:
            axon_limb_branch_dict,axon_angles = clu.axon_classification(neuron_obj,
                                                                    return_error_labels=False,
                                                                    verbose=False,
                                                                    plot_axons=False,
                                                                        best_axon=True,
                                                                    label_axon_errors=True,
                                                                    return_axon_angles=True)

            if debug_time:
                print(f"Axon Classification = {time.time() - filter_time}")
                filter_time = time.time()
        else:
            if verbose or verbose_outline:
                print(f"Skipping Axon Classification")
            axon_limb_branch_dict=neuron_obj.axon_limb_branch_dict
            from neurd import axon_utils as au
            axon_angles = au.axon_angles(neuron_obj)



        if plot_axon:
            axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                                                             matching_labels=["axon"])
            nviz.plot_limb_branch_dict(neuron_obj,
                                      axon_limb_branch_dict)

        if plot_axon_like:
            axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                                                             matching_labels=["axon-like"])
            nviz.plot_limb_branch_dict(neuron_obj,
                                      axon_limb_branch_dict)





        #c) Run the excitatory and inhibitory classification (save results in dict)
        if verbose or verbose_outline:
            print(f"\n\n ------ Part C: Inhibitory Excitatory Classification ---- \n\n")

        filter_time = time.time()

        if inh_exc_class is None:
            (inh_exc_class,
                         spine_category,
                         axon_angles,
                         n_axons,
                         n_apicals,
                         neuron_spine_density,
                         n_branches_processed,
                         skeletal_length_processed,
                         n_branches_in_search_radius,
                         skeletal_length_in_search_radius
                         ) = clu.inhibitory_excitatory_classifier(neuron_obj,
                                                            return_spine_classification=True,
                                                            return_axon_angles=True,
                                                             return_n_axons=True,
                                                             return_n_apicals=True,
                                                             return_spine_statistics=True,
                                                                 axon_limb_branch_dict_precomputed=axon_limb_branch_dict,
                                                            axon_angles_precomputed=axon_angles,
                                                                 verbose=verbose)
            #4) Get the maximum of the axon angles:
            all_axon_angles = []
            for limb_idx,limb_data in axon_angles.items():
                for candidate_idx,cand_angle in limb_data.items():
                    all_axon_angles.append(cand_angle)

            if len(axon_angles)>0:
                axon_angle_maximum = np.max(all_axon_angles)
            else:
                axon_angle_maximum = 0



            if debug_time:
                print(f"Inhibitory Excitatory Classification = {time.time() - filter_time}")
                filter_time = time.time()

            if verbose:
                print("\n -- Cell Type Classification Results --")
                print(f"inh_exc_class={inh_exc_class}")
                print(f"spine_category={spine_category}")
                print(f"axon_angles={axon_angles}")
                print(f"n_axons={n_axons}")
                print(f"n_apicals={n_apicals}")
                print(f"neuron_spine_density={neuron_spine_density}")
                print(f"n_branches_processed={n_branches_processed}")
                print(f"skeletal_length_processed={skeletal_length_processed}")
                print(f"n_branches_in_search_radius={n_branches_in_search_radius}")
                print(f"skeletal_length_in_search_radius={skeletal_length_in_search_radius}")

            if verbose_outline and (not verbose):
                print(f"cell type = {inh_exc_class}")

            cell_type_info = dict(
                        inh_exc_class=inh_exc_class,
                         spine_category=spine_category,
                         axon_angles=axon_angles,
                        axon_angle_maximum = axon_angle_maximum,
                         n_axons=n_axons,
                         n_apicals=n_apicals,
                         neuron_spine_density=neuron_spine_density,
                         n_branches_processed=neuron_spine_density,
                         skeletal_length_processed=skeletal_length_processed,
                         n_branches_in_search_radius=n_branches_in_search_radius,
                         skeletal_length_in_search_radius=skeletal_length_in_search_radius,

            )
        
        else:
            print(f"Skipping the Cell Type classification because predetermined as {inh_exc_class}")
            cell_type_info = dict()


        #d) Based on cell type--> get the filters going to use

        if verbose or verbose_outline:
            print(f"\n\n ------ Part D: Neuron Filtering ---- \n\n ")

        o_neuron,filtering_info = pru.proofread_neuron_class_predetermined(neuron_obj=neuron_obj,
            inh_exc_class = inh_exc_class,
            plot_limb_branch_filter_with_disconnect_effect = plot_limb_branch_filter_with_disconnect_effect,
            verbose = verbose,
            verbose_outline = verbose_outline,
            high_fidelity_axon_on_excitatory = high_fidelity_axon_on_excitatory,
            plot_final_filtered_neuron = plot_final_filtered_neuron,)

        if verbose:
            print(f"\n\n ------ Part E: Save Neuron  ---- \n\n ")

        neuron_list_results.append(dict(filtered_neuron=o_neuron,
                                    split_info=local_split_error_info,
                                    cell_type_info=cell_type_info,
                                    filtering_info = filtering_info))
        
        


        
    if verbose_outline or verbose:
        print(f"\n\nTotal time for Neuron Proofreading: {time.time() - proof_time}")
        
    if attempt_to_split_neuron:
        if return_process_info:
            return_value = neuron_list_results
        else:
            return_value =  [k["filtered_neuron"] for k in neuron_list_results]
    else:
        if return_process_info:
            return_value = neuron_list_results[0]
        else:
            return_value = k["filtered_neuron"]

    return return_value


# --------------- Proofreading The Synapses ------------------------- #

def synapse_filtering(neuron_obj,
                split_index,
                nucleus_id,
                segment_id=None,
                return_synapse_filter_info = True,
                return_synapse_center_data = False,
                return_error_synapse_ids = True,
                return_valid_synapse_centers = False,
                return_errored_synapses_ids_non_axons=False,
                return_error_table_entries=True,
                
                mapping_threshold = 500,
                plot_synapses=False,
                original_mesh_method = True,
                original_mesh = None,
                original_mesh_kdtree=None,
                valid_faces_on_original_mesh=None, 
                axon_faces_on_original_mesh = None,
                apply_non_axon_presyn_errors = True,
                      
                # if already have the synapse data (for instance we use when validating)
                # NEEDS TO BE DICTIONARY
                precomputed_synapse_dict= None,
                validation=False,
                
                verbose = False):

    """
    Psuedocode:
    1) Get the synapses that are presyn or postsyn to segment id (but not both)
    2) Build a KDTree of the mesh final

    ------ For presyn and postsyn (as type): -------------------
    2) Restrict the table to when segment id is that type
    3) Fetch the synapses and scale the centers
    4) Find the distance of synapses to mesh
    5) If within distance threshold then consider valid
    6) For synapses to keep create a list of dictionaries saving off:
    synaps_id
    type (presyn or postsyn)
    segment_id
    split_id
    nucleus_id
    7) Save of the stats on how many synapses of that type you started with and how many you finished with
    8) Save of synapse centers into valid and error groups


    ------ End Loop -------------------
    9) Compiles all stats on erroring
    10) Compile all synapse centers

    Return the dictionaries to write and also:
    - stats
    - synapse centers

    """
    

    if segment_id is None:
        segment_id = neuron_obj.segment_id
        
        
    # --------- Check Axon Classification if requested ---------- #
    if (axon_faces_on_original_mesh is None and apply_non_axon_presyn_errors):
        raise Exception("axon_faces_on_original_mesh was None")
    
        
    if axon_faces_on_original_mesh is None:
        axon_faces_on_original_mesh = []
    axon_faces_on_original_mesh = np.array(axon_faces_on_original_mesh)
    
    # ----------------------- 
    #1) Get the synapses that are presyn or postsyn to segment id (but not both)

    if precomputed_synapse_dict is None:
        beginning_direct_connections = vdi.segment_id_to_synapse_table_optimized(segment_id,
                                                                         validation=validation)
    #2) Build a KDTree of the mesh final
    if not original_mesh_method:
        neuron_kd = KDTree(neuron_obj.mesh.triangles_center)


    data_to_write = []
    data_to_write_errors = []
    synapse_stats = {}
    synapse_center_coordinates = {}
    total_error_synapse_ids = dict()

    for synapse_type in ["presyn","postsyn"]:
        
        
        #3) Fetch the synapses and scale the centers
        if precomputed_synapse_dict is None:
            table_for_type = beginning_direct_connections & f"{synapse_type}={segment_id}"
            synapse_ids, centroid_xs, centroid_ys, centroid_zs = table_for_type.fetch("synapse_id","synapse_x","synapse_y","synapse_z")
            synapse_centers = np.vstack([centroid_xs,centroid_ys,centroid_zs]).T
            synapse_centers_scaled = synapse_centers* [4, 4, 40]
        else:
            synapse_ids = precomputed_synapse_dict[synapse_type]["synapse_ids"]
            if "synapse_ceneters_scaled" in precomputed_synapse_dict[synapse_type].keys():
                synapse_ceneters_scaled = precomputed_synapse_dict[synapse_type]["synapse_ceneters_scaled"]
            else:
                synapse_centers = precomputed_synapse_dict[synapse_type]["synapse_centers"]
                synapse_centers_scaled = synapse_centers* [4, 4, 40]


        #4) Find the distance of synapses to mesh
        
        
        
        if not original_mesh_method:
            dist,closest_face = neuron_kd.query(synapse_centers_scaled)
            
            
            errored_synapses_idx = np.where(dist>mapping_threshold)[0]
            valid_synapses_idx = np.delete(np.arange(len(synapse_centers_scaled)),errored_synapses_idx)
            error_presyn_dendrites_idx = np.array([]).astype("int")
        else:
            """
            Pseudocode:
            
            """
            neuron_mesh_labels = np.zeros(len(original_mesh.triangles_center))
            neuron_mesh_labels_original = np.copy(neuron_mesh_labels)
            neuron_mesh_labels[valid_faces_on_original_mesh] = 1
            
            
            dist,closest_face = original_mesh_kdtree.query(synapse_centers_scaled)
            
            closest_face_labels = neuron_mesh_labels[closest_face]
            
            errored_synapses_idx = np.where((dist>mapping_threshold) | ((closest_face_labels==0)))[0]
            
            
            """  2/15: Adding the parts that will error out synapses that are presyn
            but are not the axon
            
            
            """
        
            if apply_non_axon_presyn_errors and synapse_type == "presyn":
                print("Apply the presyn non_error")
                axon_faces_on_original_mesh = axon_faces_on_original_mesh.astype("int")
                neuron_mesh_labels_original[axon_faces_on_original_mesh] = 2
                closest_face_labels_axon = neuron_mesh_labels_original[closest_face]
                error_presyn_dendrites_idx_pre = np.where(closest_face_labels_axon!=2)[0]
                error_presyn_dendrites_idx = np.setdiff1d(error_presyn_dendrites_idx_pre,errored_synapses_idx).astype("int")
                
                errored_synapses_idx = np.concatenate([errored_synapses_idx,error_presyn_dendrites_idx])
                valid_synapses_idx = np.delete(np.arange(len(synapse_centers_scaled)),errored_synapses_idx)
            
            else:
                valid_synapses_idx = np.delete(np.arange(len(synapse_centers_scaled)),errored_synapses_idx)
                error_presyn_dendrites_idx = np.array([]).astype("int")
                
                
            

        n_errored_synapses = len(errored_synapses_idx)
        n_valid_synapses = len(valid_synapses_idx)
        n_error_presyn_non_axon = len(error_presyn_dendrites_idx)
        total_synapses = n_errored_synapses + n_valid_synapses

        if verbose:
            print(f"For {synapse_type}: # valid synapses = {n_valid_synapses}, # error synapses  = {n_errored_synapses}, # error presyns = {n_error_presyn_non_axon}")

        """
        6) For synapses to keep create a list of dictionaries saving off:
        synaps_id
        type (presyn or postsyn)
        segment_id
        split_id
        nucleus_id
        """

        errored_synapses = synapse_ids[errored_synapses_idx]
        errored_synapses_non_axons = synapse_ids[error_presyn_dendrites_idx]
        non_errored_synapses = synapse_ids[valid_synapses_idx]
        val_syn_centers = synapse_centers_scaled[valid_synapses_idx]
        
        #print(f"errored_synapses_non_axons = {errored_synapses_non_axons}")

        
        total_error_synapse_ids[synapse_type] = errored_synapses
        
        
        if return_valid_synapse_centers:
            if verbose:
                print(f"Computing the distance to soma for the synapses")
                
            syn_dist = nru.synapse_skeletal_distances_to_soma(neuron_obj=neuron_obj,
                                               synapse_coordinates=val_syn_centers,
                                               original_mesh = original_mesh,
                                               original_mesh_kdtree = original_mesh_kdtree,
                                               verbose = False,

                                              )
        else:
            syn_dist = [None]*len(non_errored_synapses)
            syn_dist = [1]*len(non_errored_synapses)

        local_data_to_write = [dict(synapse_id=syn,
                              synapse_type=synapse_type,
                              nucleus_id = nucleus_id,
                              segment_id = segment_id,
                              split_index = split_index,
                            skeletal_distance_to_soma=np.round(syn_dist[syn_i],2),)
                               
                               for syn_i,syn in enumerate(non_errored_synapses)]
        
        
        
        data_to_write += local_data_to_write
        
        if return_error_table_entries: 
            total_err_for_table = list(errored_synapses)+list(errored_synapses_non_axons)
            #print(f"total_err_for_table = {total_err_for_table}")
            local_data_to_write_errors = [dict(synapse_id=syn,
                              synapse_type=synapse_type,
                              nucleus_id = nucleus_id,
                              segment_id = segment_id,
                              split_index = split_index,
                            skeletal_distance_to_soma=-1,)
                            for syn_i,syn in enumerate(list(errored_synapses)+list(errored_synapses_non_axons))]
            #print(f"local_data_to_write_errors = {local_data_to_write_errors}")
        
            data_to_write_errors += local_data_to_write_errors
            


        #7) Save of the stats on how many synapses of that type you started with and how many you finished with
        local_synapse_stats = {f"n_valid_syn_{synapse_type}":n_valid_synapses,
                               f"n_errored_syn_{synapse_type}":n_errored_synapses,
                              }
        if synapse_type == "presyn":
            local_synapse_stats[f"n_errored_syn_presyn_non_axon"] = n_error_presyn_non_axon
            if return_errored_synapses_ids_non_axons:
                synapse_stats["presyn_error_syn_non_axon_ids"] = errored_synapses_non_axons
        
        synapse_stats.update(local_synapse_stats)

        #8) Save of synapse centers into valid and error groups
        local_synapse_centers = {f"valid_syn_centers_{synapse_type}":synapse_centers_scaled[valid_synapses_idx],
                               f"errored_syn_centers_{synapse_type}":synapse_centers_scaled[errored_synapses_idx],
                              }
        
        if synapse_type == "presyn":
            local_synapse_centers[f"errored_syn_centers_presyn_non_axon"] = synapse_centers_scaled[error_presyn_dendrites_idx]
        
        synapse_center_coordinates.update(local_synapse_centers)
        
    

    if plot_synapses:
        print("Displaying the Synapse Classifications")
        nviz.plot_valid_error_synapses(neuron_obj,
                              synapse_center_coordinates)

    if ((not return_synapse_filter_info)
        and (not return_synapse_center_data) 
        and (not return_error_synapse_ids)
        and (not return_error_table_entries)
       ):
        return data_to_write
    
    return_value = [data_to_write]
    
    
    if return_synapse_filter_info:
        return_value.append(synapse_stats)
    if return_synapse_center_data:
        return_value.append(synapse_center_coordinates)
    if return_error_synapse_ids:
        return_value.append(total_error_synapse_ids)
    if return_error_table_entries:
        return_value.append(data_to_write_errors)
    
    return return_value



def calculate_error_rate(total_error_synapse_ids_list,
                        synapse_stats_list,
                        verbose=True):
    """
    Calculates all of the synapse erroring stats
    for the neuron after all the runs
    """
    
    all_presyn_errors = nu.intersect1d_multi_list([k["presyn"] for k in total_error_synapse_ids_list])
    all_postsyn_errors = nu.intersect1d_multi_list([k["postsyn"] for k in total_error_synapse_ids_list])

    n_presyn_error_syn = len(all_presyn_errors)
    n_postsyn_error_syn = len(all_postsyn_errors)

    if verbose:
        print(f"n_presyn_error_syn = {n_presyn_error_syn}, n_postsyn_error_syn = {n_postsyn_error_syn}")



    curr_split_idx = 0
    total_presyns = (synapse_stats_list[curr_split_idx]["n_valid_syn_presyn"] +
                     synapse_stats_list[curr_split_idx]["n_errored_syn_presyn"])

    total_postsyns = (synapse_stats_list[curr_split_idx]["n_valid_syn_postsyn"] +
                     synapse_stats_list[curr_split_idx]["n_errored_syn_postsyn"])

    if verbose:
        print(f"total_presyns = {total_presyns}, total_postsyns = {total_postsyns}")





    # finding the total number of presyns and postsyns
    if total_presyns>0:
        perc_error_presyn = np.round(n_presyn_error_syn/total_presyns,4)
    else:
        perc_error_presyn = None

    if total_postsyns>0:
        perc_error_postsyn = np.round(n_postsyn_error_syn/total_postsyns,4)
    else:
        perc_error_postsyn = None

    if verbose:
        print(f"perc_error_presyn = {perc_error_presyn}, perc_error_postsyn = {perc_error_postsyn}")




    total_synapses = total_presyns + total_postsyns
    total_error_synapses = n_presyn_error_syn + n_postsyn_error_syn


    if total_synapses>0:
        overall_percent_error = np.round(total_error_synapses/total_synapses,4)
    else:
        overall_percent_error = None

    if verbose:
        print(f"total_error_synapses = {total_error_synapses}, total_synapses = {total_synapses}"
         f"\noverall_percent_error= {overall_percent_error}")




    return_dict = dict(n_presyn_error_syn=n_presyn_error_syn,
                       n_postsyn_error_syn=n_postsyn_error_syn,
                       total_error_synapses=total_error_synapses,

                       total_presyns=total_presyns,
                       total_postsyns=total_postsyns,
                       total_synapses=total_synapses,

                       perc_error_presyn=perc_error_presyn,
                       perc_error_postsyn=perc_error_postsyn,

                       overall_percent_error=overall_percent_error,


                    )
    return return_dict




def proofreading_table_processing(key,
                                  proof_version,
                                  axon_version,
                                 ver=None, 
                                  compute_synapse_to_soma_skeletal_distance=True,
                                  return_errored_synapses_ids_non_axons=False,
                                  validation=False,
                                  soma_center_in_nm= False,
                                  perform_axon_classification = True,
                                  high_fidelity_axon_on_excitatory = True,
                                  perform_nucleus_pairing = True,
                                  add_synapses_before_filtering = False,
                                 verbose=True,):
    """
    Purpose: To do the proofreading and synapse filtering 
    for the datajoint tables

    """
    if ver is None:
        ver = key["ver"]
    
    # 1) Pull Down All of the Neurons
    segment_id = key["segment_id"]

    print(f"\n\n------- AutoProofreadNeuron {segment_id}  ----------")

    neuron_objs,neuron_split_idxs = vdi.decomposition_with_spine_recalculation(segment_id)

    if verbose:
        print(f"Number of Neurons found ={len(neuron_objs)}")


    # 2)  ----- Pre-work ------

    nucleus_ids,nucleus_centers = vdi.segment_to_nuclei(segment_id,
                                                       nuclei_version=ver)

    if verbose:
        print(f"Number of Corresponding Nuclei = {len(nucleus_ids)}")
        print(f"nucleus_ids = {nucleus_ids}")
        print(f"nucleus_centers = {nucleus_centers}")



    original_mesh = vdi.fetch_segment_id_mesh(segment_id)
    original_mesh_kdtree = KDTree(original_mesh.triangles_center)



    # 3) ----- Iterate through all of the Neurons and Proofread --------

    # lists to help save stats until write to ProofreadStats Table
    filtering_info_list = []
    synapse_stats_list = []
    total_error_synapse_ids_list = []


    AutoProofreadSynapse_keys = []
    AutoProofreadSynapseErrors_keys = []
    AutoProofreadNeurons_keys = []
    neuron_mesh_list = []
    axon_mesh_list = []
    axon_skeleton_list = []
    dendrite_skeleton_list = []
    neuron_skeleton_list = []
    
    
    for split_index,neuron_obj_pre_split in zip(neuron_split_idxs,neuron_objs):

        whole_pass_time = time.time()

        if verbose:
            print(f"\n-----Working on Neuron Split {split_index}-----")



        neuron_obj = neuron_obj_pre_split
    #             if neuron_obj_pre_split.n_error_limbs > 0:
    #                 if verbose:
    #                     print(f"   ---> Pre-work: Splitting Neuron Limbs Because still error limbs exist--- ")
    #                 neuron_objs_split = pru.split_neuron(neuron_obj_pre_split,
    #                                              verbose=False)
    #                 if len(neuron_objs_split) > 1:
    #                     raise Exception(f"After splitting the neuron there were more than 1: {neuron_objs_split}")

    #                 neuron_obj= neuron_objs_split[0]
    #             else:
    #                 neuron_obj = neuron_obj_pre_split


        if verbose:
            print(f"\n ----> Pre =-work: Adding the Synapses to Neuron object)---")
            
        if add_synapses_before_filtering:
            syu.add_synapses_to_neuron_obj(neuron_obj,
                                validation = validation,
                                verbose  = True,
                                original_mesh = original_mesh,
                                plot_valid_error_synapses = False,
                                calculate_synapse_soma_distance = False,
                                add_valid_synapses = True,
                                  add_error_synapses=False)
        
        # Part A: Proofreading the Neuron
        if verbose:
            print(f"\n   --> Part A: Proofreading the Neuron ----")


    #     nviz.visualize_neuron(neuron_obj,
    #                       limb_branch_dict="all")



        output_dict= pru.proofread_neuron(neuron_obj,
                            plot_limb_branch_filter_with_disconnect_effect=False,
                            plot_final_filtered_neuron=False,
                            perform_axon_classification = perform_axon_classification,
                            high_fidelity_axon_on_excitatory = high_fidelity_axon_on_excitatory,
                            verbose=True)

        filtered_neuron = output_dict["filtered_neuron"]
        cell_type_info = output_dict["cell_type_info"]
        filtering_info = output_dict["filtering_info"]





        # Part B: Getting Soma Centers and Matching To Nuclei
        if verbose:
            print(f"\n\n    --> Part B: Getting Soma Centers and Matching To Nuclei ----")


        if perform_nucleus_pairing:
            winning_nucleus_id, nucleus_info = nru.pair_neuron_obj_to_nuclei(filtered_neuron,
                                     "S0",
                                      nucleus_ids,
                                      nucleus_centers,
                                     nuclei_distance_threshold = 15000,
                                      return_matching_info = True,
                                     verbose=True)
        else:
            winning_nucleus_id = 12345
            nucleus_info = dict()
            nucleus_info["nucleus_id"] = winning_nucleus_id
            nucleus_info["nuclei_distance"] = 0
            nucleus_info["n_nuclei_in_radius"] = 1
            nucleus_info["n_nuclei_in_bbox"] = 1

        if verbose:
            print(f"nucleus_info = {nucleus_info}")
            print(f"winning_nucleus_id = {winning_nucleus_id}")
        






        # Part C: Getting the Faces of the Original Mesh
        if verbose:
            print(f"\n\n    --> Part C: Getting the Faces of the Original Mesh ----")

        original_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                    filtered_neuron.mesh,
                                                    exact_match=True,
                                                    original_mesh_kdtree=original_mesh_kdtree)

        original_mesh_faces_file = vdi.save_proofread_faces(original_mesh_faces,
                                                          segment_id=segment_id,
                                                          split_index=split_index,
                                    file_name_ending=f"proofv{proof_version}_neuron")




    #     nviz.plot_objects(recovered_mesh)


        # Part C.2: Getting the axon information to use for the synapse erroring
        axon_limb_branch_dict = clu.axon_limb_branch_dict(filtered_neuron)

        axon_skeletal_length = nru.sum_feature_over_limb_branch_dict(filtered_neuron,
                                         limb_branch_dict=axon_limb_branch_dict,
                                         feature="skeletal_length")

        axon_mesh_area = nru.sum_feature_over_limb_branch_dict(filtered_neuron,
                                             limb_branch_dict=axon_limb_branch_dict,
                                             feature="area")

        axon_face_labels = clu.axon_faces_from_labels_on_original_mesh(filtered_neuron,
                                               original_mesh=original_mesh,
                                               original_mesh_kdtree=original_mesh_kdtree,
                                                plot_axon=False,
                                               verbose=False,)

        original_mesh_faces_file_axon = vdi.save_proofread_faces(axon_face_labels,
                                                          segment_id=segment_id,
                                                          split_index=split_index,
                                                    file_name_ending=f"proofv{proof_version}_axon")

        
        # Part D: Getting the Synapse Information
        if verbose:
            print(f"\n\n    --> Part D: Getting the Synapse Information ----")


        """
        # -------- old version that did not use the synapse objects ----------
        
        
        (keys_to_write_without_version,
         synapse_stats,
         total_error_synapse_ids,
         keys_to_write_without_version_errors,
         
        ) = pru.synapse_filtering(filtered_neuron,
                        split_index,
                        nucleus_id=winning_nucleus_id,
                        segment_id=None,
                        return_synapse_filter_info = True,
                        return_synapse_center_data = False,
                        return_error_synapse_ids = True,
                       return_valid_synapse_centers=compute_synapse_to_soma_skeletal_distance,
                        return_errored_synapses_ids_non_axons=return_errored_synapses_ids_non_axons,
                        return_error_table_entries = True,
                        mapping_threshold = 500,
                          plot_synapses=False,
                        verbose = True,
                        original_mesh_method = True,
                        original_mesh = original_mesh,
                        original_mesh_kdtree = original_mesh_kdtree,
                        valid_faces_on_original_mesh=original_mesh_faces, 
                        axon_faces_on_original_mesh=axon_face_labels,
                                  
                        #will only apply the filter if it is excitatory
                        apply_non_axon_presyn_errors=cell_type_info["inh_exc_class"] == "excitatory",
                        validation=validation,

                        )
        
        """
        (keys_to_write_without_version,
         synapse_stats,
         total_error_synapse_ids,
         keys_to_write_without_version_errors,

        ) = syu.synapse_filtering_vp2(filtered_neuron,
                                split_index,
                                nucleus_id=winning_nucleus_id,
                                return_synapse_filter_info = True,
                                return_synapse_center_data = False,
                                return_error_synapse_ids = True,
                                compute_synapse_to_soma_skeletal_distance=compute_synapse_to_soma_skeletal_distance,
                                return_errored_synapses_ids_non_axons=return_errored_synapses_ids_non_axons,
                                return_error_table_entries = True,
                                plot_synapses=False,
                                verbose = True,
                                original_mesh = original_mesh,
                                #will only apply the filter if it is excitatory
                                apply_non_axon_presyn_errors=cell_type_info["inh_exc_class"] == "excitatory",
                                validation=validation)

        # -- 2/15: Will calculate the synapse distances ---- #

        keys_to_write = [dict(k,ver=key["ver"])
                                     for k in keys_to_write_without_version]
        
        #print(f"keys_to_write_without_version_errors = {keys_to_write_without_version_errors}")
        keys_to_write_errors = [dict(k,ver=key["ver"])
                                     for k in keys_to_write_without_version_errors]



        soma_x,soma_y,soma_z = nru.soma_centers(filtered_neuron,
                                           soma_name="S0",
                                           voxel_adjustment=not soma_center_in_nm)





        #7) Creating the dictionary to insert into the AutoProofreadNeuron
        new_key = dict(key,
                       split_index = split_index,
                       proof_version = proof_version,
                       axon_version= axon_version,

                       multiplicity = len(neuron_objs),

                       # -------- Important Excitatory Inhibitory Classfication ------- #
                    cell_type_predicted = cell_type_info["inh_exc_class"],
                    spine_category=cell_type_info["spine_category"],

                    n_axons=cell_type_info["n_axons"],
                    n_apicals=cell_type_info["n_axons"],




                    # ----- Soma Information ----#
                    nucleus_id         = nucleus_info["nucleus_id"],
                    nuclei_distance      = np.round(nucleus_info["nuclei_distance"],2),
                    n_nuclei_in_radius   = nucleus_info["n_nuclei_in_radius"],
                    n_nuclei_in_bbox     = nucleus_info["n_nuclei_in_bbox"],

                    soma_x           = soma_x,
                    soma_y           =soma_y,
                    soma_z           =soma_z,

                    # ---------- Mesh Faces ------ #


                    # ------------- The Regular Neuron Information (will be computed in the stats dict) ----------------- #



                       # ------ Information Used For Excitatory Inhibitory Classification -------- 
                    axon_angle_maximum=cell_type_info["axon_angle_maximum"],
                    spine_density_classifier=cell_type_info["neuron_spine_density"],
                    n_branches_processed=cell_type_info["n_branches_processed"],
                    skeletal_length_processed=cell_type_info["skeletal_length_processed"],
                    n_branches_in_search_radius=cell_type_info["n_branches_in_search_radius"],
                    skeletal_length_in_search_radius=cell_type_info["skeletal_length_in_search_radius"],




                       run_time=np.round(time.time() - whole_pass_time,4)
                      )




        


        stats_dict = filtered_neuron.neuron_stats()
        new_key.update(stats_dict)
        
        AutoProofreadSynapse_keys.append(keys_to_write)
        AutoProofreadSynapseErrors_keys.append(keys_to_write_errors)
        AutoProofreadNeurons_keys.append(new_key)
        
        #     #saving following information for later processing:
        filtering_info_list.append(filtering_info)
        synapse_stats_list.append(synapse_stats)
        total_error_synapse_ids_list.append(total_error_synapse_ids)
        
        neuron_mesh_list.append(original_mesh_faces_file)
        axon_mesh_list.append(original_mesh_faces_file_axon)
        
        # ---- 2/27: saving off the skeletons ------------
        axon_skeleton_file = vdi.save_proofread_skeleton(filtered_neuron.axon_skeleton,
                                                          segment_id=segment_id,
                                                          split_index=split_index,
                                    file_name_ending=f"proofv{proof_version}_axon_skeleton")
        
        dendrite_skeleton_file = vdi.save_proofread_skeleton(filtered_neuron.dendrite_skeleton,
                                                          segment_id=segment_id,
                                                          split_index=split_index,
                                    file_name_ending=f"proofv{proof_version}_dendrite_skeleton")
        
        neuron_skeleton_file = vdi.save_proofread_skeleton(filtered_neuron.skeleton,
                                                          segment_id=segment_id,
                                                          split_index=split_index,
                                    file_name_ending=f"proofv{proof_version}_neuron_skeleton")
        
        
        
        axon_skeleton_list.append(axon_skeleton_file)
        dendrite_skeleton_list.append(dendrite_skeleton_file)
        neuron_skeleton_list.append(neuron_skeleton_file)
    
    dict_to_return = dict(AutoProofreadSynapse_keys=AutoProofreadSynapse_keys,
                          AutoProofreadSynapseErrors_keys = AutoProofreadSynapseErrors_keys,
                          AutoProofreadNeurons_keys=AutoProofreadNeurons_keys,
                        filtering_info_list=filtering_info_list,
                        synapse_stats_list=synapse_stats_list,
                        total_error_synapse_ids_list=total_error_synapse_ids_list,

                        neuron_mesh_list=neuron_mesh_list,
                        axon_mesh_list = axon_mesh_list,
                          neuron_split_idxs=neuron_split_idxs,
                          
                        axon_skeleton_list = axon_skeleton_list,
                        dendrite_skeleton_list = dendrite_skeleton_list,
                          neuron_skeleton_list = neuron_skeleton_list,
       
    
    
    )
    return dict_to_return
    
    
    

def refine_axon_for_high_fidelity_skeleton(
    neuron_obj,
    plot_new_axon_limb_correspondence = False,
    plot_new_limb_object = False,
    plot_final_revised_axon_branch=False,
    verbose = False,
    **kwargs):
    """
    Purpose: To replace the axon branches with a higher
    fidelity representation within the neuron object
    (aka replacing all of the branch objects)

    ** Note: The Neuron should already have axon classification up to this point **
    
    
    Pseudocode:
    0) Get the limb branch dict for the axon (if empty then return)
    #1) Generate the new limb correspondence for the axon (will pass back the starting info as well)
    2) Combine the data with any left over branches that still exist 
    in the limb object

    a. Figure out which starting info to use (previous one or axon one)

    b. Delete all replaced branches

    c. Rename the existing branches so not incorporate any of the new names
    from the correspondence

    d. Save the computed dict of all existing branches

    e. export a limb correspondence for those existing branches

    3) Send all the limb correspondence info to create a limb object

    4) Part 4: Computing of all the feautres:
    a) Add back the computed dict
    b) Re-compute the median mesh width  and add no spines for all the new ones
        (have option where can set spines to 0)
    6) Recompoute median mesh no spine

    5) Adding new limb
    a)  replace old limb with new one
    b) Run the function that will go through and fix the limbs
    """
    #0) Get the limb branch dict for the axon (if empty then return)
    axon_limb_branch_dict= neuron_obj.axon_limb_branch_dict
    if len(neuron_obj.axon_limb_branch_dict) == 0:
        return neuron_obj

    #1) Generate the new limb correspondence for the axon (will pass back the starting info as well)
    new_limb_corr, st_info = pre.high_fidelity_axon_decomposition(neuron_obj,
                                         plot_new_axon_limb_correspondence=False,
                                         plot_connecting_skeleton_fix = plot_new_axon_limb_correspondence,
                                         return_starting_info=True,
                                    verbose = True,)
    
    
    
    # ---- Part 2: Combine Axon Data
    neuron_obj_cp = neuron.Neuron(neuron_obj)
    axon_limb_name = neuron_obj_cp.axon_limb_name
    curr_limb = copy.deepcopy(neuron_obj_cp[axon_limb_name]) 
    axon_branches = axon_limb_branch_dict[axon_limb_name]
    axon_starting_branch = neuron_obj_cp.axon_starting_branch
    if verbose:
        print(f"axon_limb_name= {axon_limb_name}")
        print(f"Limb starting node = {curr_limb.current_starting_node}")
        print(f"Axon starting node = {axon_starting_branch}")
        print(f"axon_branches = {axon_branches}")
        
        
    #a. Figure out starting network info
    if axon_starting_branch == curr_limb.current_starting_node:
        if verbose:
            print("Using Axon Starting Info")
        network_starting_info = st_info
    else:
        if verbose:
            print("Using Original Starting Info")
        network_starting_info = curr_limb.network_starting_info
        
        
    #b. Delete all replaced branches
    curr_limb.concept_network.remove_nodes_from(axon_branches)
    
    high_fid_node_name = np.array(list(new_limb_corr.keys())).astype('int')
    curr_node_names = list(curr_limb.concept_network.nodes())

    new_node_names = np.arange(len(curr_node_names)) + len(high_fid_node_name)

    node_name_mapping = dict([(k,v) for k,v in zip(curr_node_names,new_node_names)])

    if verbose:
        print(f"Leftover Nodes After Axon Deletion = {curr_node_names}")
        print(f"New Node names = {new_node_names}")
        print(f"Leftover Nodes Mapping = {node_name_mapping}")
    
    
    #c. Rename the existing branches so not incorporate any of the new names
    #from the correspondence

    import networkx as nx
    curr_limb.concept_network = nx.relabel_nodes(curr_limb.concept_network,
                     mapping=node_name_mapping,
                     copy=True)
    
    #d. Save the computed dict of all existing branches
    computed_data_non_axon = curr_limb.get_computed_attribute_data()
    
    #e. export a limb correspondence for those existing branches
    kept_limb_correspondence = curr_limb.limb_correspondence
    kept_limb_correspondence
    
    
    
    
    
    # --Part 3: Creating the Limb Object ---
    """
    3) Send all the limb correspondence info to create a limb object

    Purpose: Create new limb object from all our information
    of limb correspondences

    Pseudocode:
    1) Create the joint limb correspondence
    2) Calculate the new concept network
    3) Send all the information 

    """
    combined_limb_correspondence = dict(new_limb_corr)
    combined_limb_correspondence.update(kept_limb_correspondence)

    limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(combined_limb_correspondence,
                                                                            network_starting_info,
                                                                            run_concept_network_checks=True,
                                                                           )
    limb_mesh = neuron_obj_cp[axon_limb_name].mesh
    limb_mesh_face_idx = neuron_obj_cp[axon_limb_name].mesh_face_idx

    new_limb_object = neuron.Limb( mesh=limb_mesh,
                                 curr_limb_correspondence=combined_limb_correspondence,
                                 concept_network_dict=limb_to_soma_concept_networks,
                                 mesh_face_idx=limb_mesh_face_idx,
                                 verbose=True)


    if plot_new_limb_object:
        print(f"New Limb Object")   
        nviz.visualize_limb_obj(limb_obj = new_limb_object)
        
        
        
    # -- Part 4: Assigning New Limb and Computing All the Features
    
    #a) Add back the computed dict
    new_limb_object.set_computed_attribute_data(computed_data_non_axon)
    
    #b) Make none of the axon have spines or spine volume
    for b_idx in high_fid_node_name:
        new_limb_object[b_idx].spines = []
        new_limb_object[b_idx].spines_volume = []
        new_limb_object[b_idx].labels = ["axon-like","axon"]
        
    #c) setting the new limb object before calulating new widths
    nru.neuron_limb_overwrite(neuron_obj_cp,
                         limb_name = axon_limb_name,
                         limb_obj=new_limb_object)
    
    #d) Recalculate the widths

    new_axon_limb_branch_dict = {axon_limb_name:high_fid_node_name}

    neuron_obj_cp.calculate_new_width(no_spines=False,
                                           distance_by_mesh_center=True,
                                           summary_measure="median",
                            limb_branch_dict=new_axon_limb_branch_dict,
                                     verbose=verbose)

    neuron_obj_cp.calculate_new_width(no_spines=True,
                                           distance_by_mesh_center=True,
                                           summary_measure="median",
                            limb_branch_dict=new_axon_limb_branch_dict,
                                     verbose=verbose)


    
    
    # ---- Part 5: Re-processing the Limb to Make sure everything is good
    neuron_obj_revised = pru.split_neuron_limb_by_seperated_network(neuron_obj_cp,
                                                               curr_limb_idx=int(axon_limb_name[1:]),
                                                               verbose = True)
    
    if plot_final_revised_axon_branch:
        nviz.visualize_neuron(neuron_obj_revised,
                              limb_branch_dict={axon_limb_name:"all"})
    
    return neuron_obj_revised

def proofread_neuron_class_predetermined(neuron_obj,
    inh_exc_class,
    perform_axon_classification=False,
                                         
    plot_limb_branch_filter_with_disconnect_effect = True,

    high_fidelity_axon_on_excitatory = True,
    plot_final_filtered_neuron = False,

    #arguments for the axon high fidelity:
    plot_new_axon_limb_correspondence = False,
    plot_new_limb_object = False,
    plot_final_revised_axon_branch = False,

    verbose = False,
    verbose_outline = True,
    return_limb_branch_dict_to_cancel = True,
    filter_list=None,
                                         
    return_red_blue_splits = True,
    return_split_locations = True,
    neuron_simplification = True
                                        
    ):

    """
    Purpose: To apply filtering rules to a neuron that
    has already been classified
    """

    if perform_axon_classification:
        clu.axon_classification(neuron_obj)

    if inh_exc_class == "inhibitory":
        
        if filter_list is None:
            curr_filters = inh_filters_auto_proof()
        else:
            curr_filters = filter_list

        o_neuron, filtering_info = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj,
                                        filter_list = curr_filters,
                    plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        plot_limb_branch_filter_away=False,
                                        plot_final_neuron=False,

                                        return_error_info=True,
                                         verbose=verbose,
                                        verbose_outline=verbose_outline,
                                        return_limb_branch_dict_to_cancel = return_limb_branch_dict_to_cancel,
                                        return_red_blue_splits=return_red_blue_splits,
                                        return_split_locations = return_split_locations )


    elif inh_exc_class == "excitatory":
        if filter_list is None:
            #curr_filters = pru.get_exc_filters()
            #curr_filters = pru.v4_exc_filters()
            #curr_filters = pru.v5_exc_filters()
            curr_filters = exc_filters_auto_proof()
        else:
            curr_filters = filter_list

        if not high_fidelity_axon_on_excitatory:
            o_neuron, filtering_info = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj,
                                        filter_list = curr_filters,
                    plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        plot_limb_branch_filter_away=False,
                                        plot_final_neuron=False,

                                        return_error_info=True,
                                         verbose=verbose,
                                        verbose_outline=verbose_outline,
                                        return_limb_branch_dict_to_cancel=return_limb_branch_dict_to_cancel,
                                        return_red_blue_splits=return_red_blue_splits,
                                        return_split_locations = return_split_locations )
        else:
            """
            2/25 Addition:

            Pseudocode:
            1) Run the Dendrite on Axon Proofreading
            2) Run the High Fidelity Axon replacement
            3) Run the Post processing Filters
            """
            if verbose or verbose_outline:
                print(f"\n\n Using high_fidelity_axon_on_excitatory")
                print(f"\n\n---Step 1: Applying Dendrite on Axon Filtering")

            pre_filters = pru.get_exc_filters_high_fidelity_axon_preprocessing()
            post_filters = pru.get_exc_filters_high_fidelity_axon_postprocessing()

            #1) Run the Dendrite on Axon Proofreading
            o_neuron_pre, filtering_info_pre = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj,
                                        filter_list = pre_filters,
                    plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        plot_limb_branch_filter_away=False,
                                        plot_final_neuron=False,

                                        return_error_info=True,
                                         verbose=verbose,
                                        verbose_outline=verbose_outline,
                                        return_limb_branch_dict_to_cancel=return_limb_branch_dict_to_cancel)

            if verbose or verbose_outline:
                print(f"\n\n---- Step 2: Applying High Fidelity Axon -----")

            #2) Run the High Fidelity Axon replacement
            neuron_obj_high_fidelity_axon = pru.refine_axon_for_high_fidelity_skeleton(neuron_obj=o_neuron_pre,
                            plot_new_axon_limb_correspondence = plot_new_axon_limb_correspondence,
                            plot_new_limb_object = plot_new_limb_object,
                            plot_final_revised_axon_branch=plot_final_revised_axon_branch,
                             verbose = verbose,)

            if verbose or verbose_outline:
                print(f"\n\n---- Step 3: Applying Excitatory Filters Post-processing -----")

            o_neuron, filtering_info = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj_high_fidelity_axon,
                                        filter_list = post_filters,
                    plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        plot_limb_branch_filter_away=False,
                                        plot_final_neuron=False,

                                        return_error_info=True,
                                         verbose=verbose,
                                        verbose_outline=verbose_outline,
                                    return_limb_branch_dict_to_cancel=return_limb_branch_dict_to_cancel,
                                                return_red_blue_splits=return_red_blue_splits,
                                                return_split_locations = return_split_locations 
                                                                               )
            filtering_info.update(filtering_info_pre)



    else:
        raise Exception(f"Invalid cell type: {inh_exc_class}")


    if neuron_simplification:
        o_neuron = nsimp.branching_simplification(o_neuron,verbose = True)

    if plot_final_filtered_neuron:
        print("\n               ******* Plotting Final Neuron ***************")
        nviz.visualize_neuron_lite(o_neuron)
        
    
        
    return o_neuron,filtering_info

    
# ---------- For Allen Requested Proofreading ---- #
def plot_limb_to_red_blue_groups(neuron_obj,
                                limb_to_red_blue_groups,
                                error_color = "red",
                                valid_color = "blue",
                                scatter_size=0.1):
    """
    Purpose: To plot a picture of all the limb to red blue groups
    information
    
    """


    limb_branch_dict_plot = dict()
    limb_branch_dict_plot_color = dict()
    total_valid_points = []
    total_error_points = []
    for limb_name,error_comp_dict in limb_to_red_blue_groups.items():

        if limb_name not in limb_branch_dict_plot:
            limb_branch_dict_plot[limb_name] = []
            limb_branch_dict_plot_color[limb_name] = dict()

        for comp_idx,p_info in error_comp_dict.items():
            for points_info in p_info:
                limb_branch_dict_plot[limb_name] += list(points_info["error_branches"]) + list(points_info["valid_branches"])
                total_valid_points.append(points_info["valid_points"])
                total_error_points.append(points_info["error_points"])

                for e in points_info["error_branches"]:
                    limb_branch_dict_plot_color[limb_name][e] = error_color

                for e in points_info["valid_branches"]:
                    limb_branch_dict_plot_color[limb_name][e] = valid_color

    try:
        total_valid_points = np.concatenate(total_valid_points).reshape(-1,3)
    except:
        total_valid_points = np.array([])

    try:
        total_error_points = np.concatenate(total_error_points).reshape(-1,3)
    except:
        total_error_points = np.array([])

    nviz.visualize_neuron(neuron_obj,
                          visualize_type=["mesh","skeleton"],
                         limb_branch_dict=limb_branch_dict_plot,
                          mesh_color=limb_branch_dict_plot_color,
                          skeleton_color=limb_branch_dict_plot_color,
                         scatters=[total_valid_points,total_error_points],
                          mesh_whole_neuron=True,
        scatters_colors=[valid_color,error_color],
                         scatter_size=scatter_size,)
    

def limb_errors_to_cancel_to_red_blue_group(
    limb_obj,
    error_branches,
    neuron_obj = None,
    limb_idx = None,

    plot_error_graph_before_create_edges = False,
    plot_error_branches = False,
    created_edges = None,
    plot_error_graph_after_create_edges = False,

    plot_error_connected_components = False,
    include_one_hop_downstream_error_branches = None,#True,
    one_hop_downstream_error_branches_max_distance = None,#4_000,#10_000,

    offset_distance_for_points_valid = None,#3_000,#500,#1000,
    offset_distance_for_points_error = None,#3_000
    n_points = None,#1,#3,
    n_red_points = None,
    n_blue_points = None,
    red_blue_points_method = None,#"closest_mesh_face", #other options: "skeleton",
    plot_final_blue_red_points = False,
    scatter_size=0.3,
    pair_conn_comp_by_common_upstream = None,#False
    pair_conn_comp_errors =None,# True,
    group_all_conn_comp_together = None,#False,
    only_outermost_branches = None,#True,
    
    min_error_downstream_length_total = None,#5_000,# None,#5_000,
    verbose = False,
    valid_upstream_branches_restriction = None,
    
    split_red_blue_by_common_upstream = None,#True,#False,
    use_undirected_graph = None,#False,
    avoid_one_red_or_blue = None,#True,
    
    min_cancel_distance_absolute = None,#2000,
    
    min_cancel_distance_absolute_all_points = None,
    add_additional_point_to_no_children_branches = True,
    
    return_error_skeleton_points = True,
    return_synapse_points = True,
    add_extensive_parent_downstream_features = True,
    **kwargs
    ):
    """
    Purpose: To lay down red and blue points on a limb
    given error branches
    
    """
    if include_one_hop_downstream_error_branches is None:
        include_one_hop_downstream_error_branches = include_one_hop_downstream_error_branches_red_blue_global
    if one_hop_downstream_error_branches_max_distance is None:
        one_hop_downstream_error_branches_max_distance = one_hop_downstream_error_branches_max_distance_red_blue_global
    if offset_distance_for_points_valid is None:
        offset_distance_for_points_valid = offset_distance_for_points_valid_red_blue_global
    if offset_distance_for_points_error is None:
        offset_distance_for_points_error = offset_distance_for_points_error_red_blue_global
    if n_points is None:
        n_points = n_points_red_blue_global
    if n_red_points is None:
        n_red_points = n_red_points_red_blue_global
    if n_blue_points is None:
        n_blue_points = n_blue_points_red_blue_global
    if red_blue_points_method is None:
        red_blue_points_method = red_blue_points_method_red_blue_global
    if pair_conn_comp_by_common_upstream is None:
        pair_conn_comp_by_common_upstream = pair_conn_comp_by_common_upstream_red_blue_global
    if pair_conn_comp_errors is None:
        pair_conn_comp_errors = pair_conn_comp_errors_red_blue_global
        
    if group_all_conn_comp_together is None:
        group_all_conn_comp_together = group_all_conn_comp_together_red_blue_global
    if only_outermost_branches is None:
        only_outermost_branches = only_outermost_branches_red_blue_global
    if min_error_downstream_length_total is None:
        min_error_downstream_length_total = min_error_downstream_length_total_red_blue_global
    if split_red_blue_by_common_upstream is None:
        split_red_blue_by_common_upstream = split_red_blue_by_common_upstream_red_blue_global
    if use_undirected_graph is None:
        use_undirected_graph = use_undirected_graph_red_blue_global
    if avoid_one_red_or_blue is None:
        avoid_one_red_or_blue = avoid_one_red_or_blue_red_blue_global
    if min_cancel_distance_absolute is None:
        min_cancel_distance_absolute = min_cancel_distance_absolute_red_blue_global
    
    if n_red_points is None:
        n_red_points = n_points
    if n_blue_points is None:
        n_blue_points = n_points
        

    red_blue_dict = dict()
    n_obj = neuron_obj
    
    if created_edges is None:
        created_edges = []


    if len(error_branches) == 0:
        raise Exception(f"Error branches list empty in error_limb_branch_dict for limb")

    if verbose:
        print(f"Error Branches = {error_branches}")

    curr_limb = limb_obj
    
    if not use_undirected_graph:
        error_subgraph = nx.Graph(curr_limb.concept_network_directional.subgraph(error_branches))
    else:
        error_subgraph = nx.Graph(curr_limb.concept_network.subgraph(error_branches))

    if plot_error_graph_before_create_edges:
        print(f"plot_error_graph_before_create_edges")
        nx.draw(error_subgraph,with_labels=True)
        plt.show()

    if plot_error_branches and n_obj is not None:
        nviz.visualize_neuron_path(n_obj,
                                  limb_idx = limb_idx,
                                  path=error_branches)

    for e in created_edges:
        if e[0] in error_branches and e[1] in error_branches:
            if verbose:
                print(f"Adding error edge: {e}")
            error_subgraph.add_edge(*e)

    if plot_error_graph_after_create_edges:
        print(f"plot_error_graph_after_create_edges")
        nx.draw(error_subgraph,with_labels=True)
        plt.show()




    #1) Find the connected components of error branches
    conn_comp_errors = [list(k) for k in nx.connected_components(error_subgraph)]

    if verbose:
        print(f"\n**conn_comp_errors = {conn_comp_errors}")

        
    if group_all_conn_comp_together:
        if verbose:
            print(f"Grouping all conn comp together")
            
        conn_comp_errors = [list(np.hstack(conn_comp_errors))]
        
        if verbose:
            print(f"\n**conn_comp_errors AFTER PAIRING = {conn_comp_errors}")
    elif pair_conn_comp_by_common_upstream:
        if verbose:
            print(f"Attempting to pair_conn_comp_by_common_upstream")
        conn_comp_errors = nru.pair_branch_connected_components_by_common_upstream(
            curr_limb,
            conn_comp = conn_comp_errors,
            verbose = verbose
        )
        
        if verbose:
            print(f"\n**conn_comp_errors AFTER PAIRING = {conn_comp_errors}")
            
    elif pair_conn_comp_errors:
        if verbose:
            print(f"Attempting to pair_conn_comp_errors")
        conn_comp_errors = nru.pair_branch_connected_components(limb_obj=curr_limb,
                                conn_comp = conn_comp_errors,
                                plot_conn_comp_before_combining = False,
                                                            verbose = verbose,
                                                               **kwargs)

        if verbose:
            print(f"\n**conn_comp_errors AFTER PAIRING = {conn_comp_errors}")
    else:
        pass



    """
    2) For each connected component we will build a red and a blue team

    a) find all upstream nodes of error branches THAT AREN'T ERRORS:
    -> include the error branches hat these upstream valid branches came from and the 
    skeleton point that connectes them

    b) Find all valid downstream nodes from te upstream valid ones
    --. include te skeleton points that connect them

    c) Optional: Choose the downstream error branches of current boundary error branches
    """

    for conn_comp_idx,curr_conn_comp in enumerate(conn_comp_errors):

        if verbose:
            print(f"Working on connected component: {curr_conn_comp}")

        valid_upstream_branches = []
        error_border_branches = []

        error_border_coordinates = []

        for b in curr_conn_comp:
            if not use_undirected_graph:
                curr_upstream_nodes = nu.convert_to_array_like(xu.upstream_node(curr_limb.concept_network_directional,b,return_single=False))
            else:
                curr_upstream_nodes = xu.get_neighbors(curr_limb.concept_network,b)
                
#             if verbose:
#                 print(f"curr_upstream_nodes= {curr_upstream_nodes}")
            
            for curr_upstream_node in curr_upstream_nodes:
                if valid_upstream_branches_restriction is not None:
                    if curr_upstream_node not in valid_upstream_branches_restriction:
                        if verbose:
                            print(f"Skipping upstream node {curr_upstream_node} because was not in valid_upstream_branches_restriction")
                        continue

                if curr_upstream_node not in curr_conn_comp:

                    error_border_branches.append(b)

                    """
                    - 5/10 Addition: Accounts for if the error node is the starting node

                    """
                    if curr_upstream_node is None:
                        valid_upstream_branches.append(-2)
                        error_border_coordinates.append(curr_limb.current_starting_coordinate)
                    else:
                        #if curr_upstream_node not in error_branches:
                        valid_upstream_branches.append(curr_upstream_node)
                        common_endpoint = np.array(sk.shared_endpoint(curr_limb[b].skeleton,
                                                                  curr_limb[curr_upstream_node].skeleton,
                                                                  return_possibly_two=True)).reshape(-1,3)

                        error_border_coordinates.append(common_endpoint)

                        if verbose:
                            print(f"Branch {b} had a valid upstream node {curr_upstream_node} with common endpoint {common_endpoint}")
                        
        
        

        if verbose:
            print(f"valid_upstream_branches = {valid_upstream_branches}")
            print(f"error_border_branches = {error_border_branches}")
            print(f"error_border_coordinates = {error_border_coordinates}")
            print(f"")

        #b) Find all valid downstream nodes from te upstream valid ones
        #--> include te skeleton points that connect them
        
        
        # ---------------- COULD FORM GROUPS OF error_border_branches ---------------- #
        if split_red_blue_by_common_upstream:
            if verbose:
                print(f"Grouping Red Blue into Common Upstream Groups")
            
            valid_upstream_branches_total = []
            error_border_branches_total = []
            error_border_coordinates_total = []
            
            named_upstream_dict = dict()
            for upstream,error_branch,error_coord in zip(valid_upstream_branches,
                                                        error_border_branches,
                                                        error_border_coordinates):
                if upstream == -2:
                    valid_upstream_branches_total.append([upstream])
                    error_border_branches_total.append([error_branch])
                    error_border_coordinates_total.append([error_coord])
                else:
                    if upstream not in named_upstream_dict.keys():
                        named_upstream_dict[upstream] = dict(up=[],err=[],err_c = [])
                    
                    named_upstream_dict[upstream]["up"].append(upstream)
                    named_upstream_dict[upstream]["err"].append(error_branch)
                    named_upstream_dict[upstream]["err_c"].append(error_coord)
                
            for up_key,data_dict in named_upstream_dict.items():
                valid_upstream_branches_total.append(data_dict["up"])
                error_border_branches_total.append(data_dict["err"])
                error_border_coordinates_total.append(data_dict["err_c"])
            
        else:
            valid_upstream_branches_total = [valid_upstream_branches]
            error_border_branches_total = [error_border_branches]
            error_border_coordinates_total = [error_border_coordinates]
            
            
        local_red_blue = []
        for upstream_idx,(valid_upstream_branches,
             error_border_branches,
             error_border_coordinates) in enumerate(zip(valid_upstream_branches_total,
                                                       error_border_branches_total,
                                                       error_border_coordinates_total)):
            
            valid_upstream_branches_unique = np.unique(valid_upstream_branches)

            for v in valid_upstream_branches_unique:
                if v == -2:
                    continue
                    
                    
                if not use_undirected_graph:
                    downstream_nodes = xu.downstream_nodes(curr_limb.concept_network_directional,v)
                else:
                    downstream_nodes = [k for k in xu.get_neighbors(curr_limb.concept_network,v)
                                   if len(np.intersect1d(error_border_branches,xu.get_neighbors(curr_limb.concept_network,k))) > 0]
                    
                    
                    
                    
                #non_error_downstream = np.setdiff1d(downstream_nodes,curr_conn_comp)
                non_error_downstream = np.setdiff1d(downstream_nodes,error_branches)

                for d in non_error_downstream:
                    valid_upstream_branches.append(d)
                    error_border_branches.append(-1)

                    common_endpoint = np.array(sk.shared_endpoint(curr_limb[v].skeleton,
                                                                  curr_limb[d].skeleton,
                                                                  return_possibly_two=True)).reshape(-1,3)

                    error_border_coordinates.append(common_endpoint)

                    if verbose:
                        print(f"Valid Branch {v} had a valid downstream node {d} with common endpoint {common_endpoint}")

            if verbose:
                    print(f"\nAfter Adding Downstream Valid Nodes")
                    print(f"error_border_branches = {error_border_branches}")
                    print(f"valid_upstream_branches = {valid_upstream_branches}")
                    print(f"error_border_coordinates = {error_border_coordinates}")
                    print(f"")


            error_border_branches_copy = np.copy(error_border_branches)
            if include_one_hop_downstream_error_branches:
                for v in error_border_branches_copy:
                    if v == -1:
                        continue
                    if curr_limb[v].skeletal_length > one_hop_downstream_error_branches_max_distance:
                        if verbose:
                            print(f"Skipping Branch {v} one hop downstream because skeletal distance"
                                  f" ({curr_limb[v].skeletal_length}) larger than threshold one_hop_downstream_error_branches_max_distance")
                        continue
                        
                    
                    if not use_undirected_graph:
                        downstream_error_nodes = xu.downstream_nodes(curr_limb.concept_network_directional,v)
                    else:
                        downstream_error_nodes = np.intersect1d(xu.get_neighbors(curr_limb.concept_network,v),error_branches)

                    for d in downstream_error_nodes:
                        error_border_branches.append(d)
                        valid_upstream_branches.append(-1)
                        common_endpoint = np.array(sk.shared_endpoint(curr_limb[v].skeleton,
                                                                      curr_limb[d].skeleton,
                                                                      return_possibly_two=True)).reshape(-1,3)

                        error_border_coordinates.append(common_endpoint)

                        if verbose:
                            print(f"Error Branch {v} had an error downstream node {d} with common endpoint {common_endpoint}")
            if verbose:
                    print(f"\nAfter Adding Downstream Error Nodes")
                    print(f"error_border_branches = {error_border_branches}")
                    print(f"valid_upstream_branches = {valid_upstream_branches}")
                    print(f"error_border_coordinates = {error_border_coordinates}")
                    print(f"")




            """
            d) for each node in the group
                For each endpoint that is included in a boundary
                    i) Attempt to restrict the skeleton by X distance from that endoint 
                    (if two small then pick other endpoint)
                    ii) Find the closest traingle face to that point on that branch mesh and use that
            """


    #             blue_points = []
    #             red_points = []

            processed_valid = []

            curr_node_names = ["blue","red"]
            blue_red_points = [[],[]]
            

            if only_outermost_branches:
                error_branches_to_skip = cnu.upstream_branches_in_branches_list(
                                            limb_obj = curr_limb,
                                            branches = error_border_branches,
                                            )
                if verbose:
                    print(f"error_branches_to_skip = {error_branches_to_skip} after upsream branches in branches list")
            else:
                error_branches_to_skip = []


            """
            10/5 Edit: making the min error downstream just part of the skip branches
            
            Old way ---------- 
            if min_error_downstream_length_total is not None:
                error_border_branches_down_len = np.array([nst.skeletal_length_downstream_total(curr_limb,n)
                                                 for n in error_border_branches if n >= 0])
                if len(error_border_branches_down_len) == 0 or np.sum(error_border_branches_down_len>=min_error_downstream_length_total) == 0:
                    #print(f"changing min_error_downstream_length_total to None because otherwise no red points")
                    min_error_downstream_length_total = None
            """
            
            e_nodes = np.unique(error_border_branches)
            e_nodes = e_nodes[e_nodes != -1]
            
            if min_error_downstream_length_total is not None:
                possible_skip_nodes = np.array(error_border_branches)[np.array(error_border_branches) >= 0]
                
                error_border_branches_down_len = np.array([nst.skeletal_length_downstream_total(curr_limb,n)
                                                 for n in possible_skip_nodes])
                
                
                filtered_skip_nodes = possible_skip_nodes[error_border_branches_down_len<=min_error_downstream_length_total]
                
                if verbose:
                    print(f"filtered_skip_nodes according to min_error_downstream_length_total ({min_error_downstream_length_total}) = {filtered_skip_nodes}")
                
                # --- 10/7 Addition: Making sure not going to skip any starting nodes or those with downstream nodes ----
                # ---- don't need the number of downstream ones because downstream skeleton already acccounts for it ---------
#                 filtered_skip_nodes_n_downstream = np.array([nru.n_downstream_nodes(curr_limb,k) for k in filtered_skip_nodes])
#                 filtered_skip_nodes_with_downstream = filtered_skip_nodes[filtered_skip_nodes_n_downstream > 0]
#                 nodes_not_to_skip = np.concatenate([curr_limb.all_starting_nodes,filtered_skip_nodes_with_downstream])

                ## nodes skipping: starting nodes

                nodes_not_to_skip = np.concatenate([curr_limb.all_starting_nodes,
                            [k for k in filtered_skip_nodes if len(np.intersect1d(curr_limb.all_starting_nodes,
                            nru.downstream_nodes(curr_limb,k))) > 0]]).astype("int")
                
                if verbose:
                    print(f"nodes_not_to_skip= {nodes_not_to_skip}")
                filtered_skip_nodes = np.setdiff1d(filtered_skip_nodes,nodes_not_to_skip)
                
                # need to filter for 
                
                #if len(possible_skip_nodes) > len(filtered_skip_nodes): 
                new_total_error_branches_to_skip = np.concatenate([error_branches_to_skip,filtered_skip_nodes])
                if len(filtered_skip_nodes) > 0 and len(np.setdiff1d(e_nodes,new_total_error_branches_to_skip)) > 0: 
                    error_branches_to_skip = new_total_error_branches_to_skip
                    if verbose:
                        print(f"Because of min_error_downstream_length_total = {min_error_downstream_length_total}"
                              f"\n --> Expanding to skip nodes to {error_branches_to_skip}")
                    

            """
            10/5 Addition: Will compute the number of branches to be processed
            and whether a double point needs to be layed down
            
            """
            v_nodes = np.unique(valid_upstream_branches)
            v_nodes = v_nodes[v_nodes != -1]
            e_nodes = np.setdiff1d(e_nodes,error_branches_to_skip)
            
            double_point_flags = [True if (len(nodes) <= 1 and -2 not in nodes) else False
                                for nodes in [v_nodes,e_nodes]]
            
#             su.compressed_pickle(curr_limb,"curr_limb")
#             raise Exception("")
            
            if verbose:
                print(f"Before laying down points")
                print(f"valid nodes to process: {v_nodes}")
                print(f"Error nodes to process: {e_nodes}")
                print(f"double_point_flags = {double_point_flags}")


            for y,(v,e,coord) in enumerate(zip(valid_upstream_branches,
                                 error_border_branches,
                                error_border_coordinates)):
                curr_nodes = [v,e]
                
                downstream_flag = False
                if add_additional_point_to_no_children_branches:
                    if e >= 0:
                        downstream_errors = np.intersect1d(e_nodes,
                                                       nru.all_downstream_branches(curr_limb,e))
                        if verbose:
                            print(f"downstream_errors = {downstream_errors}")
                        if len(downstream_errors) == 0:
                            downstream_flag = True

                if verbose:
                    print(f"For Pair {y}: coordinate {coord}")
                for j,(n,lab,off_dist,n_pts) in enumerate(zip(curr_nodes,["valid","error"],
                                                        [offset_distance_for_points_valid,offset_distance_for_points_error],
                                                       [n_blue_points,n_red_points])):
                    if n == -1 or n in processed_valid:
                        continue

                    """
                    - 5/10 Addition: Accounts for if the error node is the starting node

                    """
                    if n == -2:
                        blue_red_points[j].append(curr_limb.current_touching_soma_vertices)
                        continue

                    if n in error_branches_to_skip:
                        continue
                        
                    # 10 /5 
                    dp_flag = double_point_flags[j]
                    
                    
                    
                    if (dp_flag or (downstream_flag and j == 1)) and n_pts == 1:
                        if verbose:
                            print(f"Incrementing the number of points")
                        local_n_points = n_pts + 1
                    else:
                        local_n_points = n_pts

                    """
                    if min_error_downstream_length_total is not None and lab == "error":
                        total_downstream_length = nst.skeletal_length_downstream_total(curr_limb,n)
                        #print(f"total_downstream_length = {total_downstream_length}")
                        if total_downstream_length < min_error_downstream_length_total:
                            if verbose:
                                print(f"Skipping node {n} because total_downstream_length ({total_downstream_length}) less than min_error_downstream_length_total ({min_error_downstream_length_total})")
                            continue
                            
                    """

                    branch_obj = curr_limb[n]
                    """
                    ---5/12: Where can lay down multiple points

                    """
                    
                    '''  OLD WAY OF LAYING DOWN POINTS
                    tried_offsets = [0]
                    n_fails = 3
                    for point_idx in range(1,local_n_points+1):
                        if tried_offsets[-1] == np.max(tried_offsets):
                            curr_offset = tried_offsets[-1] + offset_distance_for_points
                        else:
                            curr_offset = (tried_offsets[-1] + tried_offsets[-2])/2


                        sk_point,success = sk.skeleton_coordinate_offset_from_endpoint(branch_obj.skeleton,
                                                        coord,
                                                        offset_distance=curr_offset,
                                                        return_success = True)

                        for jj in range(n_fails):
                            if success:
                                break
                            curr_offset = (curr_offset + tried_offsets[-1])/2
                            sk_point,success = sk.skeleton_coordinate_offset_from_endpoint(branch_obj.skeleton,
                                                        coord,
                                                        offset_distance=curr_offset,
                                                        return_success = True)
                        #print(f"For point {point_idx}: curr_offset = {curr_offset}")
                        tried_offsets.append(curr_offset)
                        

                        if red_blue_points_method == "skeleton":
                            curr_points = np.array(sk_point).reshape(-1,3)

                        elif red_blue_points_method == "closest_mesh_face":
                            curr_points = np.array(tu.closest_face_to_coordinate(branch_obj.mesh,sk_point,
                                             return_face_coordinate=True)).reshape(-1,3)
                        else:
                            raise Exception(f"Unimplemented red_blue_points_method {red_blue_points_method}")

                        if verbose:
                            print(f"{curr_node_names[j]} {n} node offset skeleton coordinate is {sk_point}")
                            print(f" --> using {red_blue_points_method} method curr_points = {curr_points}")


                        blue_red_points[j].append(curr_points)
                        
                        '''
                    
                    
                    
                    mesh_to_map = branch_obj.mesh
                    
                    
                    if min_cancel_distance_absolute_all_points is not None and min_cancel_distance_absolute_all_points > 0:
                        for i in range(0,2):
                            if len(blue_red_points[0]) > 0 or len(blue_red_points[1]) > 0:
                                blue_red_combined= np.vstack([np.vstack(k) for k in blue_red_points if len(k) > 0])

                                if len(blue_red_combined) > 0:
                                    mesh_to_map = tu.faces_farther_than_distance_of_coordinates(
                                            mesh_to_map,
                                            coordinate = blue_red_combined,
                                            distance_threshold = min_cancel_distance_absolute_all_points,
                                            return_mesh = True
                                        )
                                if len(mesh_to_map.faces) == 0:
                                    mesh_to_map = nru.branch_neighbors_mesh(curr_limb,
                                                    n,
                                                    verbose = verbose,
                                                   )
                                else:
                                    break
                            
                            else:
                                break

                    
                    point_array = sk.coordinates_along_skeleton_offset_from_start(
                        branch_obj.skeleton,
                        coord,
                            offset = off_dist,
                            n_points = local_n_points,
                            plot_points=False
                        )
                    
                    if verbose:
                        print(f"point_array= {point_array}")
                        
                        
                    
                    if local_n_points>1:
                        min_cancel_distance = np.min(nu.all_pairwise_distances_between_coordinates(point_array))
                        
                        if min_cancel_distance_absolute is not None:
                            if verbose:
                                print(f"Deciding between min_cancel_distance = {min_cancel_distance}, and min_cancel_distance_absolute = {min_cancel_distance_absolute}")
                            min_cancel_distance = np.max([min_cancel_distance_absolute,min_cancel_distance])
                            
                            if verbose:
                                print(f"After min: min_cancel_distance = {min_cancel_distance}")
                    
                    for jj,sk_point in enumerate(point_array):
                        if tu.n_faces(mesh_to_map) == 0:
                            #raise Exception("mesh_to_map empty")
                            use_mesh_map = False
                        else:
                            use_mesh_map = True
                        
                        if red_blue_points_method == "skeleton" or not use_mesh_map:
                            curr_points = np.array(sk_point).reshape(-1,3)

                        elif red_blue_points_method == "closest_mesh_face":
                            curr_points = np.array(tu.closest_face_to_coordinate(mesh_to_map,sk_point,
                                             return_face_coordinate=True)).reshape(-1,3)
                        else:
                            raise Exception(f"Unimplemented red_blue_points_method {red_blue_points_method}")

                        if verbose:
                            print(f"{curr_node_names[j]} {n} node offset skeleton coordinate is {sk_point}")
                            print(f" --> using {red_blue_points_method} method curr_points = {curr_points}")
                        
                        # check to see if the point already exists
                        blue_red_points[j].append(curr_points)
                        
                        if local_n_points > 1 and use_mesh_map:
                            mesh_to_map = tu.faces_farther_than_distance_of_coordinates(
                                mesh_to_map,
                                coordinate = blue_red_points[j],
                                distance_threshold = min_cancel_distance,
                                return_mesh = True
                            )
                            
                            if tu.n_faces(mesh_to_map) == 0:
                                if verbose:
                                    print(f"Having to use surrounding neighbors mesh ")
                                mesh_to_map = nru.branch_neighbors_mesh(curr_limb,
                                                n,
                                                verbose = verbose,
                                               )
                                
                                if min_cancel_distance_absolute_all_points is not None and min_cancel_distance_absolute_all_points > 0:
                                    if len(blue_red_points[0]) > 0 or len(blue_red_points[1]) > 0:
                                        blue_red_combined= np.vstack([np.vstack(k) for k in blue_red_points if len(k) > 0])

                                        if len(blue_red_combined) > 0:
                                            mesh_to_map = tu.faces_farther_than_distance_of_coordinates(
                                                    mesh_to_map,
                                                    coordinate = blue_red_combined,
                                                    distance_threshold = min_cancel_distance_absolute_all_points,
                                                    return_mesh = True
                                                )
                                    
                                """
                                Old way: 
                                for jj_old in range(0,j+1):
                                    mesh_to_map = tu.faces_farther_than_distance_of_coordinates(
                                        mesh_to_map,
                                        coordinate = sk_point,
                                        distance_threshold = min_cancel_distance,
                                        return_mesh = True
                                    )
                                """
                                
                                
                                mesh_to_map = tu.faces_farther_than_distance_of_coordinates(
                                    mesh_to_map,
                                    coordinate = blue_red_points[j],
                                    distance_threshold = min_cancel_distance,
                                    return_mesh = True
                                )
                        
                        
                        

                    #print(f"blue_red_points = {blue_red_points}")

                processed_valid.append(v)
                if verbose:
                    print("")


            if len(blue_red_points[0]) > 0:
                final_blue_points = np.concatenate(blue_red_points[0])
            else:
                final_blue_points = np.array([])
                
            final_red_points = np.concatenate(blue_red_points[1])
            
            if verbose:
                print(f"final_blue_points = {final_blue_points}, voxels = {final_blue_points/vdi.voxel_to_nm_scaling}")
                print(f"final_red_points = {final_red_points}, voxels = {final_blue_points/vdi.voxel_to_nm_scaling}")
                print(f"")
            
            
            curr_conn_comp_div = nru.all_downstream_branches_from_multiple_branhes(
                                    curr_limb,
                                    branches_idx=error_border_branches,
                                    )
            
            #print(f"plot_final_blue_red_points = {plot_final_blue_red_points}")
            #raise Exception("")
            if plot_final_blue_red_points and n_obj is not None and limb_idx is not None:
                print(f"Plotting L{limb_idx} red blue points")
                nviz.visualize_neuron_path(n_obj,
                                      limb_idx=limb_idx,
                                      path=curr_conn_comp_div,
                                      scatters=[final_blue_points.reshape(-1,3),
                                               final_red_points.reshape(-1,3)],
                                      scatter_color_list=["blue","red"],
                                      scatter_size=scatter_size)


            error_branches_skeleton_length = np.sum([sk.calculate_skeleton_distance(curr_limb[k].skeleton) 
                                                     for  k in curr_conn_comp_div])
            
            

            try:
                parent_branch = valid_upstream_branches[0]
                parent_branch_width = nru.width(curr_limb[parent_branch])
                parent_branch_axon = "axon" in curr_limb[parent_branch].labels
            except:
                parent_branch = error_border_branches[0]
                parent_branch_width = nru.width(curr_limb[parent_branch])
                parent_branch_axon = "axon" in curr_limb[parent_branch].labels
                
            e_branches_for_syns = np.array(curr_conn_comp_div)
            e_branches_for_syns = e_branches_for_syns[e_branches_for_syns>=0]
            e_branches_for_syns = np.unique(e_branches_for_syns)

            v_branches_for_syns = np.array(v_nodes)
            v_branches_for_syns = v_branches_for_syns[v_branches_for_syns>=0]
            v_branches_for_syns = np.unique(v_branches_for_syns)

            curr_local_red_blue = dict(error_branches=np.array(e_branches_for_syns),
                                                                     error_branches_skeleton_length = error_branches_skeleton_length,
                                                                     valid_branches=np.array(v_branches_for_syns),
                                                                     parent_branch = parent_branch,
                                                                     parent_branch_width = nru.width(curr_limb[parent_branch]),
                                                                     parent_branch_axon = "axon" in curr_limb[parent_branch].labels,
                                                                     n_error_branches = len(e_branches_for_syns),
                                                                   valid_points = final_blue_points,
                                                                   error_points = final_red_points,
                                       coordinate = coord,
                                      )
            
            if add_extensive_parent_downstream_features:
                #print(f"#### ---- Need to fix so computed statistic over all downstream branches")
                curr_local_red_blue.update(
                    nst.parent_and_downstream_branches_feature_dict(
                        curr_limb,
                        parent_idx = parent_branch,
                        branches = e_branches_for_syns,
                        #branches = error_branches,
                    )
                )
            if return_error_skeleton_points:
                curr_local_red_blue["error_branches_skeleton_points"] = nru.skeleton_nodes_from_branches_on_limb(
                                                    curr_limb,
                                                    curr_conn_comp_div,
                                                    plot_nodes = False,
                                                )
                
            if return_synapse_points:
                
                for b_type,b in zip(["error","valid"],[e_branches_for_syns,v_branches_for_syns]):
                    for syn_type in ["pre","post"]:
                        name = f"{b_type}_{syn_type}"
                        if len(b) == 0:
                            curr_syns = []
                        else:
                            curr_syns = np.concatenate([getattr(curr_limb[bidx],f"synapses_{syn_type}") for bidx in b])
                        
                        curr_local_red_blue[f"{name}_ids"] = np.array([x.syn_id for x in curr_syns])
                        curr_local_red_blue[f"{name}_coordinates"] = np.array([x.coordinate for x in curr_syns])
                
                
            local_red_blue.append(curr_local_red_blue)
            
        if split_red_blue_by_common_upstream:
            red_blue_dict[conn_comp_idx] = local_red_blue
        else:
            red_blue_dict[conn_comp_idx] = local_red_blue[0]
    return red_blue_dict

def limb_branch_dict_to_cancel_to_red_blue_groups(neuron_obj,
                                                  limb_branch_dict_to_cancel,
                                                plot_error_graph_before_create_edges = False,
                                                plot_error_branches = False,
                                                created_edges = None,
                                                plot_error_graph_after_create_edges = False,

                                                plot_error_connected_components = False,
#                                                 include_one_hop_downstream_error_branches = True,
#                                                 one_hop_downstream_error_branches_max_distance = 4_000,#10_000,

#                                                 offset_distance_for_points = 3_000,#500,#1000,
#                                                 n_points = 1,#3,
#                                                 red_blue_points_method = "closest_mesh_face", #other options: "skeleton",
                                                
                                                  plot_final_blue_red_points = False,
                                                  scatter_size=0.3,
                                                plot_all_blue_red_groups = False,
                                                  pair_conn_comp_errors = True,
                                                 verbose = False,
                                                  
                                                  return_error_skeleton_points = True,
                                                 **kwargs):

    """
    Purpose: To create groups that should be split using blue and red team
    and then find the split points


    Psuedocode: 

    For each limb: 
    0a) Get subgraph of error branches
    0b) Add any edges that were created that are
    between these error branches
    1) Find the connected components of error branches
    2) For each connected component we will build a red and a blue team

    a) find all upstream nodes of error branches THAT AREN'T ERRORS:
    -> include the error branches hat these upstream valid branches came from and the 
    skeleton point that connectes them

    b) Find all valid downstream nodes from te upstream valid ones
    --. include te skeleton points that connect them

    c) Optional: Choose the downstream error branches of current boundary error branches

    At this point: Have the red and blue branches and the connecting points


    d) for each node in the group
        For each endpoint that is included in a boundary
            i) Attempt to restrict the skeleton by X distance from that endoint 
            (if two small then pick other endpoint)
            ii) Find the closest traingle face to that point on that branch mesh and use that

    """

    import networkx as nx
    import matplotlib.pyplot as plt
    from datasci_tools import networkx_utils as xu


    #print(f"plot_final_blue_red_points = {plot_final_blue_red_points}")


    error_limb_branch_dict = limb_branch_dict_to_cancel# axon_merge_error_limb_branch_dict
    error_limb_branch_dict = nru.limb_branch_after_limb_branch_removal(neuron_obj=neuron_obj,
                                          limb_branch_dict = error_limb_branch_dict,
                                 return_removed_limb_branch = True,
                                )
    
    n_obj = neuron_obj #neuron_obj_high_fidelity_axon

    if created_edges is None:
        created_edges = []




    limb_to_red_blue_groups = dict()

    #For each limb:
    for limb_name in error_limb_branch_dict.keys():

        limb_idx = int(limb_name[1:])

        limb_to_red_blue_groups[limb_name] = dict()

        if verbose:
            print(f"-- Working on {limb_name} --")
            
        #getting the error branches
        error_branches = error_limb_branch_dict[limb_name]
        limb_obj = n_obj[limb_name]

        limb_to_red_blue_groups[limb_name] = pru.limb_errors_to_cancel_to_red_blue_group(
                                                    limb_obj,
                                                    error_branches,
                                                    limb_idx=limb_idx,
                                                    neuron_obj = n_obj,

                                                    plot_error_graph_before_create_edges = plot_error_graph_before_create_edges,
                                                    plot_error_branches = plot_error_branches,
                                                    created_edges = created_edges,
                                                    plot_error_graph_after_create_edges = plot_error_graph_after_create_edges,

                                                    plot_error_connected_components = plot_error_connected_components,
                                                    
#                                                     include_one_hop_downstream_error_branches = include_one_hop_downstream_error_branches,
#                                                     one_hop_downstream_error_branches_max_distance = one_hop_downstream_error_branches_max_distance,

#                                                     offset_distance_for_points = offset_distance_for_points,#1000,
#                                                     n_points = n_points,
#                                                     red_blue_points_method = red_blue_points_method, #other options: "skeleton",
                                                    plot_final_blue_red_points = plot_final_blue_red_points,
                                                    scatter_size=scatter_size,
                                                    pair_conn_comp_errors = pair_conn_comp_errors,
                                                    verbose = verbose,
                                                    return_error_skeleton_points = return_error_skeleton_points,
                                                    **kwargs
                                                    )

    if plot_all_blue_red_groups:
        pru.plot_limb_to_red_blue_groups(n_obj,
                                    limb_to_red_blue_groups,
                                    error_color = "red",
                                    valid_color = "blue",
                                    scatter_size=scatter_size)   
        
    return limb_to_red_blue_groups

def valid_synapse_records_to_unique_synapse_df(synapse_records):
    """
    To turn the records of the synapses into a dataframe of the unique synapses
    
    Application: For turning the synapse filtering output into a valid dataframe
    
    Ex:
    pru.valid_synapse_records_to_unique_synapse_df(keys_to_write_without_version)
    """
    curr_df = pd.DataFrame.from_dict(synapse_records)
    v_synapse_ids, v_synapse_id_counts = np.unique(curr_df["synapse_id"].to_list(),return_counts=True)
    v_synapse_ids_unique = v_synapse_ids[np.where(v_synapse_id_counts==1)[0]]
    curr_df_unique = curr_df[curr_df["synapse_id"].isin(v_synapse_ids_unique)]
    return curr_df_unique


# ------------- Version 4 Rules ----------------- 
def filter_away_small_axon_fork_divergence(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.axon_fork_divergence_errors_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_webbing_t_merges(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.webbing_t_errors_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_high_degree_branching(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.high_degree_branch_errors_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_high_degree_branching_dendrite(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.high_degree_branch_errors_dendrite_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_thick_t_merge(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.thick_t_errors_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)




exc_axon_on_dendrite_merges_filter = pru.make_filter_dict("axon_on_dendrite_merges",
                                         pru.filter_away_axon_on_dendrite_merges_old,
                                         dict(use_pre_existing_axon_labels=True)
                                              
                                        )
    
exc_double_back_and_width_change_filter = pru.make_filter_dict("double_back_and_width_change",
                                                     pru.filter_away_large_double_back_or_width_changes,
                                                      dict(perform_double_back_errors=True,
                                                          skip_double_back_errors_for_axon=False,
                                                          #double_back_threshold = 140,

                                                           width_jump_threshold = 250,
                                                           running_width_jump_method=True, 


                                                           double_back_axon_like_threshold=145,
                                                           #double_back_axon_like_threshold=130,
                                                           #axon_comparison_distance = 1500,
                                                           #double_back_threshold = 115,
                                                           double_back_threshold = 120,

                                                           #allow_axon_double_back_angle_with_top = 39,
                                                           allow_axon_double_back_angle_with_top = None,
                                                           #allow_axon_double_back_angle_with_top_width_min = 140,
                                                           allow_axon_double_back_angle_with_top_width_min = 120,
                                                           skeletal_length_to_skip = 4000,
                                                           comparison_distance = 3000,


                                                            perform_width_errors = True,
                                                           perform_axon_width_errors = False,
                                                           

                                                          ),
                                                               catch_error=True,
                                                     )

exc_axon_fork_divergence_filter = pru.make_filter_dict("axon_fork_divergence",
                                     pru.filter_away_small_axon_fork_divergence,
                                     dict(divergence_threshold_mean=165)

                                    )
exc_axon_webbing_t_merges_filter = pru.make_filter_dict("axon_webbing_t_merges",
                                     pru.filter_away_webbing_t_merges,
                                     dict(child_width_maximum = 75,
                                        parent_width_maximum = 75,
                                         axon_only = True,
                                         error_if_web_is_none=True,
                                         web_size_threshold=120,
                                        web_size_type="ray_trace_median",
                                        web_above_threshold = True,)

                                    )

#     exc_crossovers_filter = pru.make_filter_dict("crossovers",
#                                                          pru.filter_away_crossovers,
#                                                           dict(axon_dependent=True,
#                                                               match_threshold = 30,
#                                                               comparison_distance = 2500,
#                                                               offset=2000,)
#                                                          )

exc_high_degree_branching_filter_old = pru.make_filter_dict("high_degree_branching",
                                                     pru.filter_away_high_degree_branching,
                                                      dict(
                                                          #arguments for the angle checking
                                                        offset=1500,
                                                        comparison_distance = 2000,
                                                        worst_case_sk_angle_match_threshold = 65,

                                                        #args for width matching
                                                        width_diff_max = 75,#np.inf,#100,
                                                        width_diff_perc = 60,

                                                        #args for definite pairs
                                                        sk_angle_match_threshold = 45,
                                                        sk_angle_buffer = 15,

                                                        max_degree_to_resolve = 6,
                                                        max_degree_to_resolve_wide = 8,

                                                        #args for picking the final winner
                                                        match_method = "best_match", #other option is "best_match"
                                                          
                                                        kiss_check = True,
                                                        kiss_check_bbox_longest_side_threshold = 450,

                                                      ),
                                                         catch_error=False,
                                                     )

exc_thick_t_merge_filter = pru.make_filter_dict("thick_t_merge",
                            pru.filter_away_thick_t_merge,
                            dict(
                            
                            ))

exc_high_degree_coordinates_filter = pru.make_filter_dict("high_degree_coordinates",
                                                         pru.filter_away_high_degree_coordinates,
                                                          dict(axon_dependent=True,min_degree_to_find=4)
                                                         )
    
def v4_exc_filters():

    print(f"\n*****Using v4 Filters!!!\n\n")
    exc_filters = [
        exc_axon_on_dendrite_merges_filter(),
        exc_high_degree_branching_filter_old,
        exc_axon_webbing_t_merges_filter,
        exc_thick_t_merge_filter,
        exc_double_back_and_width_change_filter,
        exc_axon_fork_divergence_filter,
        #exc_high_degree_coordinates_filter,
    ]
    

    return exc_filters


def extract_from_filter_info(filter_info,
                            #name_to_extract="limb_branch_dict_to_cancel"
                            name_to_extract = "red_blue_suggestions",
                            name_must_be_ending = False,):
    if name_must_be_ending:
        return {k:v for k,v in filter_info.items() if k[-len(name_to_extract):] == name_to_extract}
    else:
        return {k:v for k,v in filter_info.items() if name_to_extract in k}

def extract_blue_red_points_from_limb_branch_dict_to_cancel(neuron_obj,
                                                           limb_branch_dict_to_cancel,):
    total_limb_to_red_blue_groups = dict()
    for curr_cancel_key,curr_limb_branch_dict in limb_branch_dict_to_cancel.items():#["axon_on_dendrite_merges_limb_branch_dict_to_cancel"]
        limb_to_red_blue_groups = pru.limb_branch_dict_to_cancel_to_red_blue_groups(neuron_obj=neuron_obj,
                                            limb_branch_dict_to_cancel=curr_limb_branch_dict,
                                                        plot_all_blue_red_groups = False,
                                                         verbose = False)
        limb_branch_st_idx = curr_cancel_key.find("limb_branch_dict_to_cancel")

        total_limb_to_red_blue_groups[curr_cancel_key[:limb_branch_st_idx-1]] = limb_to_red_blue_groups

    return total_limb_to_red_blue_groups



# ----------- 5/27: version 5 Additions with better width jump/ doubling back that is axon/dendrite specific ----

def filter_away_width_jump_up_dendrite(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.width_jump_up_dendrite,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_width_jump_up_dendrite_filter = pru.make_filter_dict("width_jump_up_dendrite",
                                     pru.filter_away_width_jump_up_dendrite,
                                     dict()
                                    )

def filter_away_width_jump_up_axon(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.width_jump_up_axon,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_width_jump_up_axon_filter = pru.make_filter_dict("width_jump_up_axon",
                                     pru.filter_away_width_jump_up_axon,
                                     dict()
                                    )


def filter_away_double_back_dendrite(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.double_back_dendrite,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_double_back_dendrite_filter = pru.make_filter_dict("double_back_dendrite",
                                     pru.filter_away_double_back_dendrite,
                                     dict()
                                    )

def filter_away_double_back_axon_thin(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.double_back_axon_thin,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_double_back_axon_thin_filter = pru.make_filter_dict("double_back_axon_thin",
                                     pru.filter_away_double_back_axon_thin,
                                     dict()
                                    )

def filter_away_double_back_axon_thick(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.double_back_axon_thick,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_double_back_axon_thick_filter = pru.make_filter_dict("double_back_axon_thick",
                                     pru.filter_away_double_back_axon_thick,
                                     dict()
                                    )


def v5_exc_filters():

    print(f"\n*****Using v5 Filters!!!\n\n")
    exc_filters = [
        exc_axon_on_dendrite_merges_filter(),
        exc_high_degree_branching_filter_old,
        exc_axon_webbing_t_merges_filter,
        exc_thick_t_merge_filter,
        
        # exc_double_back_and_width_change_filter,
        
        exc_width_jump_up_dendrite_filter(),
        exc_width_jump_up_axon_filter(),
        exc_double_back_dendrite_filter(),
        exc_double_back_axon_thin_filter,
        exc_double_back_axon_thick_filter,
        
        exc_axon_fork_divergence_filter,
        #exc_high_degree_coordinates_filter,
    ]
    
    return exc_filters


# --------------v6 excitatory filters ------------------ #
exc_high_degree_branching_filter_v6 = pru.make_filter_dict("high_degree_branching",
                                                     pru.filter_away_high_degree_branching,
                                                      dict(
                                                          #perform_synapse_filter = False
                                                      ),
                                                         catch_error=False,
                                                     )


def filter_away_low_degree_branching(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=ed.low_degree_branch_errors_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

exc_low_degree_branching_filter_v6 = pru.make_filter_dict("low_degree_branching",
                                                     pru.filter_away_low_degree_branching,
                                                      dict(
                                                          #perform_synapse_filter = False
                                                      ),
                                                         catch_error=False,
                                                     )
def v6_exc_filters_old():

    print(f"\n*****Using v6 Filters!!!\n\n")
    exc_filters = [
        exc_axon_on_dendrite_merges_filter(),
        exc_high_degree_branching_filter(),
        exc_axon_webbing_t_merges_filter,
        exc_thick_t_merge_filter,
        
        # exc_double_back_and_width_change_filter,
        
        exc_width_jump_up_dendrite_filter(),
        exc_width_jump_up_axon_filter(),
        exc_double_back_dendrite_filter(),
        exc_double_back_axon_thin_filter,
        exc_double_back_axon_thick_filter,
        
        exc_axon_fork_divergence_filter,
        #exc_high_degree_coordinates_filter,
    ]
    
    return exc_filters


def v6_exc_filters():

    print(f"\n*****Using v6 Filters!!!\n\n")
    exc_filters = [
        exc_axon_on_dendrite_merges_filter(),
        exc_high_degree_branching_filter(),
        exc_low_degree_branching_filter(),
        exc_width_jump_up_dendrite_filter(),
        exc_width_jump_up_axon_filter(),
        exc_double_back_dendrite_filter(),
    ]
    
    return exc_filters
    

# ----------------- 7/22 New Axon Preprocessing Filters ----------- #
def filter_away_axon_on_dendrite_merges(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=au.axon_on_dendrite_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_dendrite_on_axon_merges(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=au.dendrite_on_axon_limb_branch_dict,
                return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def get_exc_filters_high_fidelity_axon_preprocessing():
    exc_dendrite_on_axon_merges_filter = pru.make_filter_dict("dendrite_on_axon_merges",
                                                         pru.filter_away_dendrite_on_axon_merges,
                                                         )
    return [exc_dendrite_on_axon_merges_filter]


# ----------- Inhibitory Filters ------------------------
# corresponding to the function ed.high_degree_branch_errors_limb_branch_dict
# which calls on function: ed.high_degree_upstream_match
inh_high_degree_branching_filter_v6 = pru.make_filter_dict("high_degree_branching",
                                                     pru.filter_away_high_degree_branching,
                                                      dict(
                                                          width_max = 140, #setting the width max for crossovers a little smaller
                                                          upstream_width_max = 180,
                                                      ),
                                                         catch_error=False,
                                                     )




# corresponding to the function ed.low_degree_branch_errors_limb_branch_dict
# which calls on function ed.low_degree_upstream_match
# inh_low_degree_branching_filter_v6 = pru.make_filter_dict("low_degree_branching",
#                                                      pru.filter_away_low_degree_branching,
#                                                       dict(
#                                                           #perform_synapse_filter = False
#                                                           width_max = 140,
#                                                           upstream_width_max = 180,
#                                                           max_degree_to_resolve_absolute = 5,
#                                                           filters_to_run = [
#                                                              gf.axon_webbing_filter,
#                                                              gf.thick_t_filter,
#                                                              #gf.axon_double_back_filter,
#                                                               gf.axon_double_back_inh_filter,
#                                                              gf.fork_divergence_filter,
#                                                              gf.fork_min_skeletal_distance_filter,
                                                              
#                                                           ]
#                                                       ),
#                                                          catch_error=False,
#                                                      )

def v6_inh_filters():
    print(f"\n*****Using v6 INHIBITORY Filters!!!\n\n")
    inh_filters = [
        exc_axon_on_dendrite_merges_filter(),
        inh_high_degree_branching_filter(),
        inh_low_degree_branching_filter(),
        exc_width_jump_up_dendrite_filter(),
        exc_width_jump_up_axon_filter(),
        exc_double_back_dendrite_filter(),
    ]
    
    return inh_filters


def proofread_neuron_full(
    neuron_obj,
    
    # arguments for processing down in DecompositionCellType
    cell_type=None,
    add_valid_synapses = False,
    validation = False,

    add_spines = False,
    add_back_soma_synapses = True,

    perform_axon_processing = False,

    return_after_axon_processing = False,

    #arguments for processing after DecompositionCellType to Proofread Neuron
    plot_head_neck_shaft_synapses = False,
    
    plot_soma_synapses = False,
    
    proofread_verbose = False,
    verbose_outline = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    plot_final_filtered_neuron = False,
    plot_synapses_after_proofread = False,
    
    plot_compartments = False,
    
    plot_valid_synapses = False,
    plot_error_synapses = False,

    return_filtering_info = True,
    verbose = False,
    debug_time = True,
    return_red_blue_splits = True,
    return_split_locations = True,
    filter_list=None,
    
    add_spine_distances = False,
    original_mesh = None,
    use_refactored_proofread_neuron_full=None,
    filters_dataset = None,
    ):
    """
    Purpose: To proofread the neuron after it has already been:
    
    1) cell typed
    2) Found the axon (can be optionally performed)
    3) Synapses have been added (can be optionally performed)
    
    """
    if use_refactored_proofread_neuron_full is None:
        use_refactored_proofread_neuron_full = use_refactored_proofread_neuron_full_global
    if filters_dataset is None:
        filters_dataset = filters_dataset_global
    
    if add_valid_synapses:
        st = time.time()
        if verbose:
            print(f"\n--Adding valid synapses")
        neuron_obj = syu.add_synapses_to_neuron_obj(neuron_obj,
                            validation = validation,
                            verbose  = verbose,
                            original_mesh = original_mesh,
                            plot_valid_error_synapses = False,
                            calculate_synapse_soma_distance = False,
                            add_valid_synapses = True,
                              add_error_synapses=False,)
        if verbose:
            print(f"Done adding synapses: {time.time() - st}")
            
    if add_spines:
        st = time.time()
        if verbose:
            print(f"\n--Adding Spines:")
        neuron_obj = spu.add_head_neck_shaft_spine_objs(neuron_obj,
                                                               verbose = verbose
                                                              )
        if verbose:
            print(f"Done adding spines: {time.time() - st}")
            
    if cell_type is None:
        if verbose:
            print(f"\n--Calculating the cell type because was none ")
            st = time.time()
        baylor_e_i,baylor_cell_type_info = ctu.e_i_classification_from_neuron_obj(neuron_obj,
                                                                                     verbose = True,
                                                                                     return_cell_type_info = True)
        
        if verbose:
            print(f"baylor_e_i = {baylor_e_i}: {time.time() - st}")
        cell_type = baylor_e_i
        
    
    if perform_axon_processing: 
        st = time.time()
        if verbose:
            print(f"\n--Performing axon processing")
            st = time.time()
        neuron_obj,filtering_info_axon,axon_angles_dict=au.complete_axon_processing(
                    neuron_obj,
                    cell_type = cell_type,
                     add_synapses_and_head_neck_shaft_spines = False,
                    validation = validation,
                    plot_initial_axon=False,
                    plot_axon_on_dendrite=False,
                    return_filtering_info = True,
                     return_axon_angle_info = True,
                    verbose = verbose)
        
        if verbose:
            print(f"Doen Axon Processing: {time.time() - st}")
            
    if return_after_axon_processing:
        return neuron_obj
            
    # --- a) adding head neck shaft bouton labels (and setting the labels to synapses) ===
    st = time.time()
    
    if verbose:
        print(f"\n--- a) adding head neck shaft bouton labels (and setting the labels to synapses)")

    spu.set_neuron_head_neck_shaft_idx(neuron_obj)
    spu.set_neuron_synapses_head_neck_shaft(neuron_obj)
    
    if debug_time:
        print(f"\na) Time for head/neck/shaft/bouton labels and syn label: {time.time() - st}")
    
    if plot_head_neck_shaft_synapses:
        syu.plot_head_neck_shaft_synapses(neuron_obj)
        
    
    # ---- b) Adding Back the Soma Synapses -----
    if add_back_soma_synapses:
        st = time.time()
        if verbose:
            print(f"\n---- b) Adding Back the Soma Synapses ")
        syu.add_valid_soma_synapses_to_neuron_obj(neuron_obj,
                                            verbose = True)
        if debug_time:
            print(f"\nb) Time for adding back soma synapses: {time.time() - st}")

        if plot_soma_synapses:
            syu.plot_synapses(neuron_obj,total_synapses=True)
            
    
    # --- c) Proofreading the Neuron
    if verbose:
        print(f"\n--- c) Proofreading the Neuron")

    st = time.time()
    
    high_fidelity_axon_on_excitatory = False
    
    if not use_refactored_proofread_neuron_full:
        o_neuron,filtering_info = pru.proofread_neuron_class_predetermined(
            neuron_obj=neuron_obj,
            inh_exc_class = cell_type,
            plot_limb_branch_filter_with_disconnect_effect = plot_limb_branch_filter_with_disconnect_effect,
            verbose = proofread_verbose,
            verbose_outline = verbose_outline,
            high_fidelity_axon_on_excitatory = high_fidelity_axon_on_excitatory,
            plot_final_filtered_neuron = plot_final_filtered_neuron,
            filter_list=filter_list,
            return_red_blue_splits=return_red_blue_splits,
            return_split_locations = return_split_locations 
                                                                    )
    else:
        from . import graph_filter_pipeline as pipe
        
        o_neuron,filtering_info = pipe.proofread_neuron_full_refactored(
            neuron_obj,
            cell_type=cell_type,
            filters_dataset=filters_dataset,
            verbose = True,
            verbose_time = True,
        )
    
    
    
    if debug_time:
        print(f"\nTime for proofreading rules: {time.time() - st}")
    
    if plot_synapses_after_proofread:
        syu.plot_synapses(o_neuron,total_synapses=True)
        
    # ---- d) Compartment Classification
    if verbose:
        print(f"\n---- d) Compartment Classification")
    st = time.time()
    o_neuron = apu.compartment_classification_by_cell_type(
        o_neuron,
        cell_type=cell_type,
        plot_compartments=plot_compartments)
    
    if debug_time:
        print(f"\n Time for cell compartments: {time.time() - st}")
    
    # --- e) Add the error synapses back to the neuorn (because done proofreading)
    
    if verbose:
        print(f"\n--- e) Add the error synapses back to the neuorn (because done proofreading)")
    
    st = time.time()
    syu.add_synapses_to_neuron_obj(o_neuron,
                               segment_id=None,
                            validation = False,
                            verbose  = verbose,
                            original_mesh = original_mesh,
                            plot_valid_error_synapses = False,
                            calculate_synapse_soma_distance = False,
                            add_valid_synapses = False,
                              add_error_synapses=True,
                               limb_branch_dict_to_add_synapses = None,
                              #set_head_neck_shaft=True
                              )
    if debug_time:
        print(f"\nb) Time for adding back error synapses: {time.time() - st}")
        
        
    # ---f) setting the limb/branch idx and soma distances for synapses
    
    if verbose:
        print(f"\n---f) setting the limb/branch idx and soma distances for synapses")
    
    st = time.time()
    
    syu.set_limb_branch_idx_to_synapses(o_neuron)
    syu.calculate_neuron_soma_distance(o_neuron,
                                  verbose = verbose)
    spu.set_soma_synapses_spine_label(neuron_obj)
    
    if debug_time:
        print(f"\nb) Time for setting limb/branch idx and soma distances for synapses: {time.time() - st}")
        
        
    if plot_valid_synapses:
        syu.plot_synapses_valid_from_neuron_obj(o_neuron)

    if plot_error_synapses:
        syu.plot_synapses_error_from_neuron_obj(o_neuron)    
        
    if add_spine_distances:
        if verbose:
            print(f"Adding spine distances")
        o_neuron= spu.calculate_spine_obj_attr_for_neuron(
            o_neuron,
        verbose = verbose)
    
    
    if (not return_filtering_info):
        return o_neuron
    
    return_value = [o_neuron]
    if return_filtering_info:
        return_value.append(filtering_info)
        
    return return_value



def save_off_meshes_skeletons(
    neuron_obj,
    save_off_compartments=True,
    save_off_entire_neuron=True,
    file_name_ending = "",
    return_file_paths = True,
    split_index = None,
    verbose = False):
    """
    Purpose: To save off the skeletons and mesh of a neuron
    and the compartments
    
    """
    file_path_dict = dict()
    
    segment_id = neuron_obj.segment_id
    
    original_mesh = vdi.fetch_segment_id_mesh(segment_id)
    original_mesh_kdtree = tu.mesh_to_kdtree(original_mesh)
    
    
    if split_index is None:
        split_index = neuron_obj.split_index
        
    if split_index is None:
        raise Exception("Split index is None")
    
    #saving off the meshes
    
    if save_off_compartments:
        if verbose:
            print(f"---Working on Compartments")
        comp_meshes = apu.compartments_mesh(neuron_obj,verbose = False)
        if verbose:
            print(f"---Working on Meshes")
        for comp,c_mesh in comp_meshes.items():
            comp = "_".join(comp.split("_")[:-1])
            if verbose:
                print(f"  Working on {comp}")
            original_c_faces = tu.original_mesh_faces_map(original_mesh,
                                                        c_mesh,
                                                        exact_match=True,
                                                        original_mesh_kdtree=original_mesh_kdtree)

            c_mesh_file = vdi.save_proofread_faces(original_c_faces,
                                                          segment_id=neuron_obj.segment_id,
                                                          split_index=split_index,
                                                file_name_ending=f"{file_name_ending}_{comp}_mesh_faces")
            file_path_dict[f"{comp}_mesh_faces"] = c_mesh_file


        #saving off the skeleton
        if verbose:
            print(f"---Working on Skeleton")
        comp_sks = apu.compartments_skeleton(neuron_obj,verbose=False)
        for comp,c_sk in comp_sks.items():
            comp = "_".join(comp.split("_")[:-1])
            if verbose:
                print(f"  Working on {comp}")
            c_skeleton_file = vdi.save_proofread_skeleton(c_sk,
                                                          segment_id=neuron_obj.segment_id,
                                                          split_index=split_index,
                                                file_name_ending=f"{file_name_ending}_{comp}_skeleton")
            file_path_dict[f"{comp}_skeleton"] = c_skeleton_file
    
    
    if save_off_entire_neuron:
        if verbose:
            print(f"\n---Saving off entire neuron")
        comp = "neuron"
        original_c_faces = tu.original_mesh_faces_map(original_mesh,
                                                            nru.neuron_mesh_from_branches(neuron_obj),
                                                            exact_match=True,
                                                            original_mesh_kdtree=original_mesh_kdtree)

        c_mesh_file = vdi.save_proofread_faces(original_c_faces,
                                                      segment_id=neuron_obj.segment_id,
                                                      split_index=split_index,
                                            file_name_ending=f"{file_name_ending}_{comp}_mesh_faces")
        file_path_dict[f"{comp}_mesh_faces"] = c_mesh_file
        
        c_sk = neuron_obj.skeleton
        
        c_skeleton_file = vdi.save_proofread_skeleton(c_sk,
                                                      segment_id=neuron_obj.segment_id,
                                                      split_index=split_index,
                                            file_name_ending=f"{file_name_ending}_{comp}_skeleton")
        file_path_dict[f"{comp}_skeleton"] = c_skeleton_file
        
    if verbose:
        print(f"file_path_dict = \n{file_path_dict}")
    if return_file_paths:
        return file_path_dict

    
def split_success(neuron_obj):
    if neuron_obj.n_error_limbs == 0:
        split_success = 0
    elif neuron_obj.multi_soma_touching_limbs == 0:
        split_successs = 1
    elif neuron_obj.same_soma_multi_touching_limbs == 0:
        split_success = 2
    else:
        split_success = 3
        
    return split_success


# ---------------------- Version 7 Filters -------------------------
def low_branch_length_large_clusters(
    neuron_obj,
    max_skeletal_length = None,#8_000,
    min_n_nodes_in_cluster = None,#16,
    limb_branch_dict_restriction = None,
    skeletal_distance_from_soma_min = None,#15_000
    plot = False,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: 
    To identify large clusters of small length
    branches that usually signifify dendrite that
    was converted to axon or glia pieces
    
    Ex: 
    from neurd import proofreading_utils as pru
    _ = pru.low_branch_length_large_clusters_dendrite(neuron_obj,plot = True,
                                                      max_skeletal_length = 9000,
                                                 min_n_nodes_in_cluster = 20)
    """
    if max_skeletal_length is None:
        max_skeletal_length = max_skeletal_length_low_branch_clusters_axon_global
    if min_n_nodes_in_cluster is None:
        min_n_nodes_in_cluster = min_n_nodes_in_cluster_low_branch_clusters_axon_global
    if skeletal_distance_from_soma_min is None:
        skeletal_distance_from_soma_min = skeletal_distance_from_soma_min_axon_global
    
        
    return nru.low_branch_length_clusters(
        neuron_obj,
        max_skeletal_length = max_skeletal_length,
        min_n_nodes_in_cluster = min_n_nodes_in_cluster,
        limb_branch_dict_restriction =limb_branch_dict_restriction,
        skeletal_distance_from_soma_min = skeletal_distance_from_soma_min,
        plot = plot,
        verbose = verbose
                                                )

def low_branch_length_large_clusters_dendrite(
    neuron_obj,
    max_skeletal_length = None,#8_000,
    min_n_nodes_in_cluster = None,#16,
    **kwargs
    ):
    
    if max_skeletal_length is None:
        max_skeletal_length = max_skeletal_length_low_branch_clusters_dendrite_global
    if min_n_nodes_in_cluster is None:
        min_n_nodes_in_cluster = min_n_nodes_in_cluster_low_branch_clusters_dendrite_global

    
    return pru.low_branch_length_large_clusters(
        neuron_obj,
        max_skeletal_length=max_skeletal_length,
        min_n_nodes_in_cluster=min_n_nodes_in_cluster,
        limb_branch_dict_restriction = neuron_obj.dendrite_limb_branch_dict,
        **kwargs
        
    )

def low_branch_length_large_clusters_axon(
    neuron_obj,
    max_skeletal_length = None,#8_000,
    min_n_nodes_in_cluster = None,#16,
    **kwargs
    ):
    
    if max_skeletal_length is None:
        max_skeletal_length = max_skeletal_length_low_branch_clusters_axon_global
    if min_n_nodes_in_cluster is None:
        min_n_nodes_in_cluster = min_n_nodes_in_cluster_low_branch_clusters_axon_global
    
    
    return pru.low_branch_length_large_clusters(
        neuron_obj,
        max_skeletal_length=max_skeletal_length,
        min_n_nodes_in_cluster=min_n_nodes_in_cluster,
        limb_branch_dict_restriction = neuron_obj.axon_limb_branch_dict,
        **kwargs
        
    )

def filter_away_low_branch_length_clusters_dendrite(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=pru.low_branch_length_large_clusters_dendrite,
                 return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)

def filter_away_low_branch_length_clusters_axon(neuron_obj,
                                           return_error_info=False,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=pru.low_branch_length_large_clusters_axon,
                 return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)


low_branch_length_clusters_axon_filter = pru.make_filter_dict("low_branch_length_clusters_axon",
                                     pru.filter_away_low_branch_length_clusters_axon,
                                     dict()
                                    )

low_branch_length_clusters_dendrite_filter = pru.make_filter_dict("low_branch_length_clusters_dendrite",
                                     pru.filter_away_low_branch_length_clusters_dendrite,
                                     dict()
                                    )


# -------------- version 7 filters ---------------------------------
def exc_axon_on_dendrite_merges_filter(
    **kwargs,
    ):
    return pru.make_filter_dict("axon_on_dendrite_merges",
                                         pru.filter_away_axon_on_dendrite_merges_old,
                                         gu.merge_dicts([
                                             dict(use_pre_existing_axon_labels=True),
                                             kwargs,])
                                              
                                        )

def exc_high_degree_branching_filter(
    catch_error = False,
    **kwargs
    ):
    
    return pru.make_filter_dict("high_degree_branching",
                         pru.filter_away_high_degree_branching,
                          gu.merge_dicts([
                                        dict(),
                                        kwargs,]),
                             catch_error=catch_error,
                         )

def exc_high_degree_branching_dendrite_filter(
    catch_error = False,
    **kwargs
    ):
    
    return pru.make_filter_dict("high_degree_branching_dendrite",
                         pru.filter_away_high_degree_branching_dendrite,
                          gu.merge_dicts([
                                        dict(),
                                        kwargs,]),
                             catch_error=catch_error,
                         )

def exc_low_degree_branching_filter(
    catch_error = False,
    **kwargs):
    return pru.make_filter_dict("low_degree_branching",
             pru.filter_away_low_degree_branching,
              gu.merge_dicts([
                                        dict(),
                                        kwargs,]),
                 catch_error=catch_error,
             )

def exc_width_jump_up_dendrite_filter(
    **kwargs,):
    return pru.make_filter_dict("width_jump_up_dendrite",
                                     pru.filter_away_width_jump_up_dendrite,
                                     gu.merge_dicts([
                                        dict(),
                                        kwargs,]),
                                    )

def exc_width_jump_up_axon_filter(
    **kwargs):
    return pru.make_filter_dict("width_jump_up_axon",
                                     pru.filter_away_width_jump_up_axon,
                                     gu.merge_dicts([
                                        dict(),
                                        kwargs,]),
                                    )

def exc_double_back_dendrite_filter(
    **kwargs
    ):
    return pru.make_filter_dict("double_back_dendrite",
                                     pru.filter_away_double_back_dendrite,
                                     dict()
                                    )

def inh_double_back_dendrite_filter(
    double_back_threshold = None,
    **kwargs
    ):
    if double_back_threshold is None:
        double_back_threshold = double_back_threshold_inh_double_b_global
    
    return pru.make_filter_dict("double_back_dendrite",
                                     pru.filter_away_double_back_dendrite,
                                     dict(double_back_threshold=double_back_threshold)
                                    )

def inh_high_degree_branching_filter(
    width_max = None,
    upstream_width_max = None,
    catch_error = False,
    **kwargs):
    
    if width_max is None:
        width_max = width_max_high_low_degree_inh_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_high_low_degree_inh_global
    
    return pru.make_filter_dict(
        "high_degree_branching",
        pru.filter_away_high_degree_branching,
        gu.merge_dicts([
                dict(
                      width_max = width_max, #setting the width max for crossovers a little smaller
                      upstream_width_max = upstream_width_max,
                  ),
                kwargs,]),
          
             catch_error=catch_error,
         )

def inh_high_degree_branching_dendrite_filter(
    width_max = None,
    upstream_width_max = None,
    catch_error = False,
    **kwargs):
    
    if width_max is None:
        width_max = width_max_high_high_degree_inh_dendr_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_high_high_degree_inh_dendr_global
    
    return pru.make_filter_dict(
        "high_degree_branching_dendrite",
        pru.filter_away_high_degree_branching_dendrite,
        gu.merge_dicts([
                dict(
                      width_max = width_max, #setting the width max for crossovers a little smaller
                      upstream_width_max = upstream_width_max,
                  ),
                kwargs,]),
          
             catch_error=catch_error,
         )


def axon_on_dendrite_plus_downstream(neuron_obj,plot = False,):
    lb = ns.query_neuron_by_labels(neuron_obj,matching_labels=["axon-error"])
    if len(lb) > 0:
        lb = nru.all_donwstream_branches_from_limb_branch(
            neuron_obj,
            limb_branch_dict=lb,
            plot = plot
        )
    return lb


def inh_low_degree_branching_filter(
    width_max = None,
    upstream_width_max = None,
    max_degree_to_resolve_absolute = None,
    filters_to_run = None,
    catch_error = False,
    **kwargs,
    ):
    
    if width_max is None:
        width_max = width_max_high_low_degree_inh_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_high_low_degree_inh_global
    if max_degree_to_resolve_absolute is None:
        max_degree_to_resolve_absolute = max_degree_to_resolve_absolute_low_degree_inh_global
        
    
    if filters_to_run is None:
        filters_to_run = [
                     gf.axon_webbing_filter,
                     gf.thick_t_filter,
                     #gf.axon_double_back_filter,
                      gf.axon_double_back_inh_filter,
                     gf.fork_divergence_filter,
                     gf.fork_min_skeletal_distance_filter,

                  ]
    
    return pru.make_filter_dict("low_degree_branching",
         pru.filter_away_low_degree_branching,
         gu.merge_dicts([
          dict(
              #perform_synapse_filter = False
              width_max = width_max,
              upstream_width_max = upstream_width_max,
              max_degree_to_resolve_absolute = max_degree_to_resolve_absolute,
              filters_to_run=filters_to_run
              
          ),
             kwargs,]),
             catch_error=catch_error,
         )

    
def v7_exc_filters(dendrite_branching_filters = None):
    
    if dendrite_branching_filters is None:
        dendrite_branching_filters = dendrite_branching_filters_global

    print(f"\n*****Using v7 Filters!!!\n\n")
        
    if dendrite_branching_filters:
        
        exc_filters = [
            exc_axon_on_dendrite_merges_filter(),
            exc_high_degree_branching_filter(),
            exc_low_degree_branching_filter(),
            exc_high_degree_branching_dendrite_filter(),
            exc_width_jump_up_dendrite_filter(),
            exc_width_jump_up_axon_filter(),
            exc_double_back_dendrite_filter(),
        ]
    else:
        exc_filters = [
            exc_axon_on_dendrite_merges_filter(),
            exc_high_degree_branching_filter(),
            exc_low_degree_branching_filter(),
            #exc_high_degree_branching_dendrite_filter(),
            exc_width_jump_up_dendrite_filter(),
            exc_width_jump_up_axon_filter(),
            exc_double_back_dendrite_filter(),
        ]
        
    
    return exc_filters

def v7_inh_filters(dendrite_branching_filters = None):
    
    if dendrite_branching_filters is None:
        dendrite_branching_filters = dendrite_branching_filters_inh_global
    print(f"\n*****Using v7 INHIBITORY Filters!!!\n\n")
    
    if dendrite_branching_filters:
        inh_filters = [
            exc_axon_on_dendrite_merges_filter(),
            inh_high_degree_branching_filter(),
            inh_low_degree_branching_filter(),
            inh_high_degree_branching_dendrite_filter(),
            exc_width_jump_up_dendrite_filter(),
            exc_width_jump_up_axon_filter(),
            inh_double_back_dendrite_filter(),
        ]
    else:
        inh_filters = [
            exc_axon_on_dendrite_merges_filter(),
            inh_high_degree_branching_filter(),
            inh_low_degree_branching_filter(),
            exc_width_jump_up_dendrite_filter(),
            exc_width_jump_up_axon_filter(),
            inh_double_back_dendrite_filter(),
        ]
        
    
    return inh_filters


def merge_error_red_blue_suggestions_clean(
    red_blue_suggestions):

    red_blue_merge_error_suggesions = dict()

    for error_filter,red_blue_splits in red_blue_suggestions.items():
        red_blue_merge_error_suggesions[error_filter] = []
        for limb_idx,limb_split_info in red_blue_splits.items():
            #red_blue_merge_error_suggesions[error_filter][limb_idx] = []
            for split_dic_key,split_dict_list in limb_split_info.items():
                for split_dic in split_dict_list:
                    red_blue_merge_error_suggesions[error_filter].append(split_dic)

    return red_blue_merge_error_suggesions


# ------------- parameters for stats ---------------

global_parameters_dict_default_split = dict(
    # ------------- parameters for the splitting ----------------
    
    remove_segment_threshold=1500,#the segments along path that should be combined
    remove_segment_threshold_round_2 = 2500,
    consider_path_neighbors_for_removal = True,

    #paraeters for high degree nodes
    offset_high_degree = 2000,#2500,#1500,
    comparison_distance_high_degree = 2000,
    match_threshold_high_degree = 65,#65,#35,#45,#35,

    #parameter for both width and doubling back
    # This will prevent the edges that were added to extend to the soma from causing the doulbing back or width threshold errors
    skip_small_soma_connectors = True,
    small_soma_connectors_skeletal_threshold = 2500,

    # parameters for the doubling back
    double_back_threshold = 80,#100,# 130,
    offset_double_back = 1000,
    comparison_distance_double_back = 6000,

    #parameters for the width threshold
    width_jump_threshold = 200,


    simple_path_of_2_cut = False,

    apply_double_back_first = True,
    double_back_threshold_at_first = 110,
    
    #--- split neuron function parameters --
    min_skeletal_length_limb= 15_000,

)

global_parameters_dict_default_low_branch_clusters = dict(
    # ---- for filtering away large clusters -------
    max_skeletal_length_low_branch_clusters_dendrite = 8_000,
    min_n_nodes_in_cluster_low_branch_clusters_dendrite = 16,
    
    max_skeletal_length_low_branch_clusters_axon = 8_000,
    min_n_nodes_in_cluster_low_branch_clusters_axon = 16,
    
    skeletal_distance_from_soma_min_axon = 10_000,
)

global_parameters_dict_default_auto_proof = dict(
    # ---- inh_high_degree_branching_filter and inh_low_degree_branching_filter---
    width_max_high_low_degree_inh = 140, #setting the width max for crossovers a little smaller
    upstream_width_max_high_low_degree_inh = 180,
    max_degree_to_resolve_absolute_low_degree_inh = 5,
    
    width_max_high_high_degree_inh_dendr = 500,
    upstream_width_max_high_high_degree_inh_dendr = 500,
    
    dendrite_branching_filters = False,
    dendrite_branching_filters_inh = False,
    
    
    #--double back filters 
    double_back_threshold_inh_double_b = None,
    use_refactored_proofread_neuron_full = None,
    filters_dataset = 'microns',
    
)

global_parameters_dict_default_red_blue_multi_soma = dict(
    include_one_hop_downstream_error_branches_red_blue = True,
    one_hop_downstream_error_branches_max_distance_red_blue = 4_000,#10_000,
    offset_distance_for_points_valid_red_blue = 3_000,#500,#1000,
    offset_distance_for_points_error_red_blue = 3_000,#500,#1000,
    n_points_red_blue = 1,#3,
    n_red_points_red_blue = 3,#None,
    n_blue_points_red_blue = 2,#None,
    red_blue_points_method_red_blue = "closest_mesh_face",
    pair_conn_comp_by_common_upstream_red_blue = True,#False,
    pair_conn_comp_errors_red_blue = True,
    group_all_conn_comp_together_red_blue = False,
    only_outermost_branches_red_blue = True,
    min_error_downstream_length_total_red_blue = 5_000,# None,#5_000,

    split_red_blue_by_common_upstream_red_blue = True,#False,
    use_undirected_graph_red_blue = False,
    avoid_one_red_or_blue_red_blue = True,

    min_cancel_distance_absolute_red_blue = 1000,#2000,
)

global_parameters_dict_default_red_blue_multi_axon_on_dendrite = dict(
    include_one_hop_downstream_error_branches_red_blue = True,
    one_hop_downstream_error_branches_max_distance_red_blue = 1_000,#10_000,
    offset_distance_for_points_valid_red_blue = 1700,#3_000,#500,#1000,
    offset_distance_for_points_error_red_blue = 1000,#3_000,#500,#1000,
    n_points_red_blue = 1,#3,
    n_red_points_red_blue = 3,#None,
    n_blue_points_red_blue = 2,#None,
    red_blue_points_method_red_blue = "closest_mesh_face",
    pair_conn_comp_by_common_upstream_red_blue = True,#False,
    pair_conn_comp_errors_red_blue = True,
    group_all_conn_comp_together_red_blue = False,
    only_outermost_branches_red_blue = True,
    min_error_downstream_length_total_red_blue = 5_000,# None,#5_000,

    split_red_blue_by_common_upstream_red_blue = True,#False,
    use_undirected_graph_red_blue = False,
    avoid_one_red_or_blue_red_blue = True,

    min_cancel_distance_absolute_red_blue = 1000,#2000,
)


global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_split,
    global_parameters_dict_default_low_branch_clusters,
    global_parameters_dict_default_auto_proof,
    global_parameters_dict_default_red_blue_multi_axon_on_dendrite,
    
])



# print(f"mvu.data_interface.voxel_to_nm_scaling = {mvu.data_interface.voxel_to_nm_scaling}")

attributes_dict_default = dict(
    vdi = mvu.data_interface,
    exc_filters_auto_proof = v7_exc_filters,
    inh_filters_auto_proof = v7_inh_filters,
)    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}
global_parameters_dict_microns_split = dict()
global_parameters_dict_microns_auto_proof = dict()
global_parameters_dict_microns_low_branch_clusters = {}
global_parameters_dict_default_red_blue = dict(
    n_red_points_red_blue = 4,
)

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_microns_split,
    global_parameters_dict_microns_low_branch_clusters,
    global_parameters_dict_microns_auto_proof,
    global_parameters_dict_default_red_blue
])

# --------- spiltting -------------
global_parameters_dict_h01_split = dict()
global_parameters_dict_h01_auto_proof = dict(
    width_max_high_low_degree_inh = 160, #setting the width max for crossovers a little smaller
    upstream_width_max_high_low_degree_inh = 200,
    
    width_max_high_high_degree_inh_dendr = 650,
    upstream_width_max_high_high_degree_inh_dendr = 650,
    
    dendrite_branching_filters = True,
    dendrite_branching_filters_inh = False,
    
    #--double back filters 
    double_back_threshold_inh_double_b = 120,

)
global_parameters_dict_h01_low_branch_clusters = {}

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_split,
    global_parameters_dict_h01_low_branch_clusters,
    global_parameters_dict_h01_auto_proof
])

attributes_dict_h01 = dict(
    vdi = hvu.data_interface
)



# data_type = "default"
# algorithms = None
# modules_to_set = [pru,ed,nst]

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
from . import apical_utils as apu
from . import axon_utils as au
from . import classification_utils as clu
from . import concept_network_utils as cnu
from . import error_detection as ed
from . import graph_filters as gf
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mvu
from . import neuron
from . import neuron_searching as ns
from . import neuron_simplification as nsimp
from . import neuron_simplification as nsimp 
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import preprocess_neuron as pre
from . import spine_utils as spu
from . import synapse_utils as syu
from . import cell_type_utils as ctu
from . import neuron_statistics as nst


#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su

from . import proofreading_utils as pru
