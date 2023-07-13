'''


Utils for helping with the classification of a neuron
for compartments like axon, apical, basal...



'''
import copy
import networkx as nx
from pykdtree.kdtree import KDTree
import time




top_volume_vector = np.array([0,-1,0])

def axon_candidates(neuron_obj,
                    possible_axon_limbs = None,
                   ais_threshold=20000,
                   plot_close_branches=False,
                    plot_candidats_after_elimination=False,
                    plot_candidates_after_adding_back=False,
                   verbose=False,
                   **kwargs):
    """
    Purpose: To return with a list of the possible 
    axon subgraphs of the limbs of a neuron object
    
    Pseudocode: 
    1) Find all the branches in the possible ais range and delete them from the concept networks
    2) Collect all the leftover branches subgraph as candidates
    3) Add back the candidates that were deleted
    4) Combining all the candidates in one list
    """
    curr_neuron_obj = neuron_obj
    
    if possible_axon_limbs is None:
        possible_axon_limbs = nru.get_limb_names_from_concept_network(curr_neuron_obj.concept_network)
    
    
    
    close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=[ns.skeletal_distance_from_soma],
                                            query=f"skeletal_distance_from_soma<{ais_threshold}",
                                            function_kwargs=dict(limbs_to_process=possible_axon_limbs),
                                             #return_dataframe=False


                                            )
    outside_bubble_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=[ns.skeletal_distance_from_soma],
                                            query=f"skeletal_distance_from_soma>={ais_threshold}",
                                            function_kwargs=dict(limbs_to_process=possible_axon_limbs),
                                             #return_dataframe=False


                                            )
    
    if plot_close_branches:
        colors_dict_returned = nviz.visualize_neuron(curr_neuron_obj,
                              limb_branch_dict=close_limb_branch_dict,
                             mesh_color="red",
                             mesh_color_alpha=1,
                             mesh_whole_neuron=True,
                             return_color_dict=True)

        
        
        
    
    # 2) --------Delete the nodes from the branch graph and then group into connected ocmponents
    # into candidates

    limbs_to_check = [nru.get_limb_string_name(k) for k in possible_axon_limbs]

    sub_limb_color_dict = dict()
    total_sub_limbs = dict() #will map the limbs to the connected components


    for limb_idx in limbs_to_check:
        if verbose:
            print(f"\nPhase 2: Working on Limb {limb_idx}")

        #initializing the candidate list and the color dictionary for visualization
        total_sub_limbs[limb_idx] = []
        sub_limb_color_dict[limb_idx] = dict()



        curr_limb = curr_neuron_obj[limb_idx]

        if limb_idx in close_limb_branch_dict.keys():
            nodes_to_eliminate = close_limb_branch_dict[limb_idx]
        else:
            nodes_to_eliminate = []

        #the nodes that were eliminated we need to show deleted colors
        for n in nodes_to_eliminate:
            sub_limb_color_dict[limb_idx][n] = mu.color_to_rgba("black", alpha=1)

        if verbose:
            print(f"nodes_to_eliminate = {nodes_to_eliminate}")

        curr_filt_network = nx.Graph(curr_limb.concept_network_directional)
        curr_filt_network.remove_nodes_from(nodes_to_eliminate)

        if len(curr_filt_network) == 0:
            if verbose:
                print("The filtered network is empty so just leaving the candidates as empty lists")
            continue

        curr_limb_conn_comp = list(nx.connected_components(curr_filt_network))


        total_sub_limbs[limb_idx] = [list(k) for k in curr_limb_conn_comp]

        colors_to_use = mu.generate_unique_random_color_list(n_colors=len(curr_limb_conn_comp),colors_to_omit=["black","midnightblue"])
        for j,(c_comp,curr_random_color) in enumerate(zip(curr_limb_conn_comp,colors_to_use)):

            for n in c_comp:
                sub_limb_color_dict[limb_idx][n] = curr_random_color


                
    if plot_candidats_after_elimination:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict="all",
                             mesh_color=sub_limb_color_dict,
                             mesh_fill_color="green")
        
        
        
        
        
        
        
    # ----------- Part 3: ------------------#
        
        
    """
    3) Adding back all endpoints that were eliminated in step 2: Process is below

    For each limb

    0) Get all of the endpoint nodes in the whole directed concept network
    - remove the starting node from the list
    1) Find the shortest path from every endpoint to the starting node
    2) Concatenate shortest paths into dictionary mapping node to number of
    occurances in the shortest paths
    3) Find all of the endpoints that were eliminated with the restriction
    4) Filter those endpoint paths for nodes that only have an 
    occurance of one for the lookup dictionary
    5) Add all deleted endpoint filtered shortest paths as candidates

    How to handle corner cases:
    1) If only starting node that got deleted
    --> just add that as a candidate
    2) If all of network got deleted, current way will work

    """

    removed_candidates = dict()

    for limb_idx in limbs_to_check:
        if verbose:
            print(f"\n----Working on Limb {limb_idx}-----")

        curr_limb = curr_neuron_obj[limb_idx]    

        removed_candidates[limb_idx] = []

        if limb_idx in close_limb_branch_dict.keys():
            nodes_to_eliminate = close_limb_branch_dict[limb_idx]
        else:
            nodes_to_eliminate = []
            if verbose:
                print("No nodes were eliminated so don't need to add back any candidates")
            continue


        curr_network = nx.Graph(curr_limb.concept_network_directional)
        curr_starting_node = curr_limb.current_starting_node

        #covering the corner case that only the root node existed
        #and it was deleted
        if len(nodes_to_eliminate) == 1 and len(curr_network)==1:
            if verbose:
                print("network was only of size 1 and that node was eliminated so returning that as the only candidate")
            removed_candidates[limb_idx] = [[curr_starting_node]]

            #adding the color
            curr_random_color = mu.generate_unique_random_color_list(n_colors=1,colors_to_omit=["black","midnightblue"])[0]
            sub_limb_color_dict[limb_idx][n] = curr_random_color

        else:
            #0) Get all of the endpoint nodes in the whole directed concept network
            #- remove the starting node from the list
            curr_endpoints = xu.get_nodes_of_degree_k(curr_network,1)
            if curr_starting_node in curr_endpoints:
                curr_endpoints.remove(curr_starting_node)


            #3) Find all of the endpoints that were eliminated with the restriction
            endpoints_eliminated = [k for k in curr_endpoints if k in nodes_to_eliminate]

            if len(endpoints_eliminated) == 0:
                if verbose:
                    print("No endpoints were eliminated so don't need to add back any candidates")
                continue

            #1) Find the shortest path from every endpoint to the starting node
            shortest_paths_endpoints = dict()
            for en in curr_endpoints:
                en_shortest_path = nx.shortest_path(curr_network,
                                source = en,
                                 target = curr_starting_node)
                shortest_paths_endpoints[en] = en_shortest_path

            #2) Concatenate shortest paths into dictionary mapping node to number of
            #occurances in the shortest paths
            node_occurance = dict()
            for curr_path in shortest_paths_endpoints.values():
                for n in curr_path:
                    if n not in node_occurance.keys():
                        node_occurance[n] = 1
                    else:
                        node_occurance[n] += 1

            #4) Filter those endpoint paths for nodes that only have an 
            #occurance of one for the lookup dictionary
            added_back_candidates = []
            for en_elim in endpoints_eliminated:
                filtered_path = [k for k in shortest_paths_endpoints[en_elim] if node_occurance[k] == 1]
                added_back_candidates.append(filtered_path)

            if verbose:
                print(f"New candidates added back: {added_back_candidates}")

            removed_candidates[limb_idx] = added_back_candidates

        #5) Adding the new paths to the color dictionary for visualization 
        colors_to_use = mu.generate_unique_random_color_list(n_colors=len(removed_candidates[limb_idx]),colors_to_omit=["black","midnightblue"])
        for add_path,curr_random_color in zip(removed_candidates[limb_idx],colors_to_use):
            for n in add_path:
                sub_limb_color_dict[limb_idx][n] = curr_random_color

    # checking that adding back the candidates went well

    if plot_candidates_after_adding_back:
        
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict="all",
                             mesh_color=sub_limb_color_dict,
                             mesh_fill_color="green")
        
        
        
        
    # --------- Part 4: Combining All the Candidates ------------ #
    
    all_candidates = dict()
    for limb_name in limbs_to_check:
        all_candidates[int(limb_name[1:])] = total_sub_limbs[limb_name] + removed_candidates[limb_name]

    if verbose:
        print("Final Candidates")
        for limb_name, list_of_subgraphs in all_candidates.items():
            print(f"\nLimb {limb_name}")
            for jj,sg in enumerate(list_of_subgraphs):
                curr_color = mu.convert_rgb_to_name(sub_limb_color_dict[f"L{limb_name}"][sg[0]])
                print(f"Candidate {jj} ({curr_color}): {repr(np.array(sg))}")
                
    return all_candidates


def candidate_starting_skeletal_angle(limb_obj,candidate_nodes,
                                      offset = 10000,#20000,
                                    axon_sk_direction_comparison_distance = 10000,#5000,
                                    buffer_for_skeleton = 5000,
                                      top_volume_vector = np.array([0,-1,0]),
                                      plot_skeleton_paths_before_restriction=False,
                                      plot_skeleton_paths_after_restriction=False,
                                      return_restricted_skeletons=False,
                                      branches_not_to_consider_for_end_nodes = None,
                                      verbose=False,
                                     ):
    
    """
    Purpose: To get the skeleton that represents the starting skeleton
    --> and then find the projection angle to filter it away or not

    Pseudocode: 
    1) convert the graph into a skeleton (this is when self touches could be a problem)
    2) Find all skeleton points that are within a certain distance of the starting coordinate
    3) Find all end-degree nodes (except for the start)
    4) Find path back to start for all end-nodes
    5) Find paths that are long enough for the offset plus test --> if none then don't filter
    anyway

    For each valid path (make them ordered paths):
    6) Get the offset + test subskeletons for all valid paths
    7) Get the angle of the sksletons vectors
    """
    
    debug_cancellation = False
    
    # -- Renaming the variables --
    subgraph_branches = candidate_nodes
    curr_limb = limb_obj
    
    
    sk_size_to_compare = axon_sk_direction_comparison_distance + offset
    total_distance = offset + axon_sk_direction_comparison_distance + buffer_for_skeleton
    
    
    
    #1) convert the graph into a skeleton (this is when self touches could be a problem)
    candidate_sk= sk.stack_skeletons([curr_limb[k].skeleton for k in subgraph_branches])
    candidate_sk_graph = sk.convert_skeleton_to_graph(candidate_sk)

    #2) Find all skeleton points that are within a certain distance of the starting coordinate
    starting_sk_coord = curr_limb.current_starting_coordinate
    starting_sk_node = xu.get_graph_node_by_coordinate(candidate_sk_graph,starting_sk_coord)
    skeletons_nodes_for_comparison = xu.find_nodes_within_certain_distance_of_target_node(
        candidate_sk_graph,
        starting_sk_node,total_distance)
    np.array(list(skeletons_nodes_for_comparison))
    comparison_subgraph = candidate_sk_graph.subgraph(skeletons_nodes_for_comparison)
    
    #3) Find all edn-degree nodes (except for the start)
    all_endnodes = xu.get_nodes_of_degree_k(comparison_subgraph,1)
    starting_coordinate_endnode = xu.get_graph_node_by_coordinate(
        comparison_subgraph,
        starting_sk_coord)
    
    
        
    
    endnodes_to_test = np.setdiff1d(all_endnodes,[starting_coordinate_endnode])
    
    
    # ------------ 1/24 Addition: Will get rid of end nodes that are on dendritic portions ----- #
    if branches_not_to_consider_for_end_nodes is not None:
        """
        Pseudocode: 
        1) Get coordinate of end node
        2) Get the branches that belong to that coordinate
        3) Subtract off the branches that shouldn't be considered
        4) If empty then skip, if not then add
        
        """
        debug_cancellation = False
        new_endnodes_to_test = []
        
        if debug_cancellation:
            if verbose:
                print(f"branches_not_to_consider_for_end_nodes: {branches_not_to_consider_for_end_nodes}")
            
        for curr_endpoint in endnodes_to_test:
            if debug_cancellation:
                print(f"working on endpoint: {curr_endpoint}")
            curr_endpoint_coordinate = xu.get_coordinate_by_graph_node(comparison_subgraph,
                                                                      curr_endpoint)
            
            branches_of_endnode = nru.find_branch_with_specific_coordinate(limb_obj,
                                                                           curr_endpoint_coordinate)
            viable_branches = np.setdiff1d(branches_of_endnode,branches_not_to_consider_for_end_nodes)
            
            if debug_cancellation:
                print(f"branches_of_endnode: {branches_of_endnode}")
                print(f"viable_branches: {viable_branches}")
                
            
            if len(viable_branches)>0:
                new_endnodes_to_test.append(curr_endpoint)
        
        endnodes_to_test = np.array(new_endnodes_to_test)
        
        if debug_cancellation:
            print(f"endnodes_to_test AFTER FILTERING= {endnodes_to_test}")

    if verbose:
        print(f"endnodes_to_test = {endnodes_to_test}")
        
        
    # nviz.plot_objects(curr_limb.mesh,
    #             skeletons=[sk.convert_graph_to_skeleton(comparison_subgraph)],
    #                  )
    
    
    if len(endnodes_to_test) == 0:
        if return_restricted_skeletons:
            return None,None
        else:
            return None
        
        
        
    #4) Find path back to start for all end-nodes
    
    paths_to_test = [nx.shortest_path(comparison_subgraph,
                                      starting_coordinate_endnode,
                                      k
                                     ) for k in endnodes_to_test]
    sk_paths_to_test = [sk.convert_graph_to_skeleton(comparison_subgraph.subgraph(k))
                           for k in paths_to_test]
    sk_paths_to_test_ordered = [sk.order_skeleton(k,
                                                  start_endpoint_coordinate = starting_sk_coord)
                               for k in sk_paths_to_test]
    


    if len(sk_paths_to_test_ordered) <= 0: 
        raise Exception("Found no skeleton paths")
    
    if plot_skeleton_paths_before_restriction:
        endpoint_scatters = xu.get_coordinate_by_graph_node(comparison_subgraph,endnodes_to_test)
        for k,sc_point in zip(sk_paths_to_test_ordered,endpoint_scatters):
            nviz.plot_objects(curr_limb.mesh,
                             skeletons=[k],
                             scatters=[sc_point.reshape(-1,3)])
            
            
    #5) Find paths that are long enough for the offset plus test --> if none then don't filter any
    sk_distances = np.array([sk.calculate_skeleton_distance(k) for k in sk_paths_to_test_ordered])
    filtered_indexes = np.where(sk_distances>=sk_size_to_compare)[0]


    if len(filtered_indexes)> 0:
        filtered_skeletons = [sk_paths_to_test_ordered[k] for k in filtered_indexes]
    else:
        filtered_skeletons = sk_paths_to_test_ordered

    if verbose:
        print(f"Skeleton paths distances = {sk_distances}")
        print(f"Filtered indexes = {filtered_indexes}")
        print(f"len(filtered_skeletons) = {len(filtered_skeletons)}")
        
        
        
    
    #6) Get the offset + test subskeletons for all valid paths
    filtered_skeletons_restricted = [sk.restrict_skeleton_from_start_plus_offset(k,
                                        offset=offset,
                                        comparison_distance=axon_sk_direction_comparison_distance)
             for k in filtered_skeletons]
    
    

    if plot_skeleton_paths_after_restriction:
        endpoint_scatters = xu.get_coordinate_by_graph_node(comparison_subgraph,endnodes_to_test)
        for k,sc_point in zip(filtered_skeletons_restricted,endpoint_scatters):
            nviz.plot_objects(curr_limb.mesh,
                             skeletons=[k],
                             scatters=[sc_point.reshape(-1,3),filtered_skeletons_restricted[0][0].reshape(-1,3)],
                                      scatters_colors=["red","blue"])
            
    #7) Get the angle of the sletons vectors

    #angle between going down and skeleton vector
    sk_vectors = [sk.skeleton_endpoint_vector(k) for k in filtered_skeletons_restricted]
    sk_angles = np.array([nu.angle_between_vectors(top_volume_vector,k) for k in sk_vectors])

    

    if verbose:
        print(f"sk_angles = {sk_angles}")
        
        
    if return_restricted_skeletons:
        return sk_angles,filtered_skeletons_restricted
    else:
        return sk_angles
    
    

def filter_axon_candiates_old(neuron_obj,
    axon_subgraph_candidates,
    axon_angle_threshold_relaxed = 110,#90,
    axon_angle_threshold = 120,
    relaxation_percentage = 0.85,
    relaxation_axon_length = np.inf,#40_000,
                              
                          
    #parameters for computing the skeletal angle
     
    skeletal_angle_offset = 10000,
    skeletal_angle_comparison_distance = 10000,
    skeletal_angle_buffer = 5000,
                          
    axon_like_limb_branch_dict = None,
                          
    min_ais_width=85,#85,
    use_beginning_ais_for_width_filter=True,
                          
    extra_ais_checks= False, #feature that is not needed with new flip_dendrite_to_axon added at axon-like identification step
    extra_ais_width_threshold =  650,
    extra_ais_spine_density_threshold = 0.00015,
    extra_ais_angle_threshold = 150,
                          
    verbose = False,
    
                          
    return_axon_angles = True,
    best_axon=False,
    best_axon_skeletal_legnth_ratio = 20,
    **kwargs
    ):
    
    

    """
    Pseudocode: 

    For each candidate: 

    0) If all Axon? (Have a more relaxed threshold for the skeleton angle)
    1) Find the starting direction, and if not downwards --> then not axon
    2) ------------- Check if too thin at the start --> Not Axon (NOT GOING TO DO THIS) -------------
    3) If first branch is axon --> classify as axon
    4) Trace back to starting node and add all branches that are axon like

    """
    
    
    if axon_like_limb_branch_dict is None:
        axon_like_limb_branch_dict = ns.query_neuron(neuron_obj,
                functions_list=["matching_label"],
               query="matching_label==True",
               function_kwargs=dict(labels=["axon-like"]),
               )
        
        
#     su.compressed_pickle(axon_subgraph_candidates,"axon_subgraph_candidates")
#     su.compressed_pickle(axon_like_limb_branch_dict,"axon_like_limb_branch_dict")
#     raise Exception("")
        
    final_axon_like_classification = axon_like_limb_branch_dict
    curr_neuron_obj = neuron_obj
    
    axon_candidate_filtered = dict()
    axon_candidate_filtered_angles = dict()
    
    
    best_axon_candidate_filtered = dict()
    best_axon_candidate_filtered_angles = dict()
    best_axon_soma_angle = -1
    best_axon_axon_angle = -1
    best_axon_skeletal_length = 0
    debug = False
    
    candidate_filt_dicts = []
        
    for curr_limb_idx,limb_candidate_grouped_branches in axon_subgraph_candidates.items():
    
        curr_limb_name = nru.get_limb_string_name(curr_limb_idx)
        curr_limb = curr_neuron_obj[curr_limb_idx]
        
        
        
        for curr_candidate_idx,curr_candidate_subgraph in enumerate(limb_candidate_grouped_branches):
            curr_candidate_subgraph = np.array(curr_candidate_subgraph)
            
            if verbose:
                print(f"\n\n --- Working on limb {curr_limb_idx}, candidate # {curr_candidate_idx}")
            
            if debug:
                print(f"curr_candidate_subgraph = {curr_candidate_subgraph}")
                
            
            # ---------- Part Prework:  --------------------------------------------- #
            
            # ----------- Getting the path to the starting soma ----#
            curr_limb.set_concept_network_directional(starting_soma = 0)

            undirectional_limb_graph = nx.Graph(curr_limb.concept_network_directional)

            current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(
                undirectional_limb_graph,[curr_limb.current_starting_node],
                curr_candidate_subgraph)
            candidate_starting_node = current_shortest_path[-1]
            
            if debug:
                print(f"current_shortest_path = {current_shortest_path}")
                print(f"candidate_starting_node = {candidate_starting_node}")
            
            
            # -------- getting the axon branches ------------- #
            if curr_limb_name in final_axon_like_classification.keys():
                axon_branches_on_limb = final_axon_like_classification[curr_limb_name]
            else:
                axon_branches_on_limb = []
            
            
            # ------ 2/3 Addition: Will add back AIS if not there and meets strict requirements ------ #
            #get the soma angle
            curr_soma_angle = nst.soma_starting_angle(neuron_obj=neuron_obj,
                                                      limb_obj=curr_limb)
            
            if extra_ais_checks and candidate_starting_node not in axon_branches_on_limb:
                
                curr_ais_width  = curr_limb[candidate_starting_node].width_new["median_mesh_center"]
                curr_ais_spine_density = curr_limb[candidate_starting_node].spine_density
                
                
                
                
                add_back_ais = ((curr_ais_width < extra_ais_width_threshold) and
                                (curr_ais_spine_density < extra_ais_spine_density_threshold) and 
                                       (curr_soma_angle >extra_ais_angle_threshold))
                
                if verbose:
                    print("Apply extra conditions in order to add back AIS to axons")
                    print(f"curr_soma_angle = {curr_soma_angle}")
                    print(f"curr_ais_width = {curr_ais_width}")
                    print(f"curr_ais_spine_density = {curr_ais_spine_density}")
                    print(f"add_back_ais = {add_back_ais}")
                    
                if add_back_ais:
                    if verbose:
                        print(f"Adding the starting candidate ({candidate_starting_node}) to the axon_branches_on_limb")
                    axon_branches_on_limb  = list(axon_branches_on_limb) + [candidate_starting_node]
                    

            
            # ------------- Part A: Filtering For Axon Composition ------------------

            """
            Pseudocode: 
            1) Get the number of branches in the candidate that are axons
            2a) If all are axons --> choose the relaxed axon angle threshold
            2b) If none are axons --> remove as not a candidate
            2c) if some are --> use standard axon threshold

            """
            
                
                
                
                
                
            
                
            axon_branches_on_subgraph = np.intersect1d(axon_branches_on_limb,curr_candidate_subgraph)
            

            axon_percentage_n_nodes = len(axon_branches_on_subgraph)/len(curr_candidate_subgraph)
            axon_percentage = (np.sum([curr_limb[k].skeletal_length for k in axon_branches_on_subgraph])/
                               np.sum([curr_limb[k].skeletal_length for k in curr_candidate_subgraph]))
            axon_length_over_nodes = np.sum([curr_limb[k].skeletal_length for k in axon_branches_on_subgraph])

            if verbose:
                print(f"{len(axon_branches_on_subgraph)} out of {len(curr_candidate_subgraph)} branches are axons")
                print(f"Axon percentage = {axon_percentage}")
                print(f"axon_percentage_n_nodes = {axon_percentage_n_nodes}")
                print(f"axon_length_over_nodes = {axon_length_over_nodes}")
                
            if axon_percentage > relaxation_percentage or axon_length_over_nodes > relaxation_axon_length:
                curr_axon_angle_threshold = axon_angle_threshold_relaxed
                
            elif len(axon_branches_on_subgraph) == 0:
                if verbose:
                    print(f"Not adding candidate no axon branches detected ")
                continue
            else:
                curr_axon_angle_threshold = axon_angle_threshold

            if verbose:
                print(f"curr_axon_angle_threshold = {curr_axon_angle_threshold}")






            # ---------  Part B: Filtering For Starting Skeleton Angle -------------
            


            candidate_nodes = np.unique(np.hstack([curr_candidate_subgraph,current_shortest_path]))
            
            # ----- 1/24: Filtering out the nodes that are on branches that are not axons --------- #
            non_axon_branches_on_subgraph = np.setdiff1d(candidate_nodes,axon_branches_on_limb)
            
            if verbose:
                print(f"candidate_nodes = {candidate_nodes}")
                print(f"non_axon_branches_on_subgraph = {non_axon_branches_on_subgraph}")



            candidate_angles,restr_skels = clu.candidate_starting_skeletal_angle(limb_obj=curr_limb,
                              candidate_nodes=candidate_nodes,
                                  offset = skeletal_angle_offset,
                                axon_sk_direction_comparison_distance = skeletal_angle_comparison_distance,
                                buffer_for_skeleton = skeletal_angle_buffer,
                                  top_volume_vector = np.array([0,-1,0]),
                                  plot_skeleton_paths_before_restriction=False,
                                  plot_skeleton_paths_after_restriction=False,
                                                 return_restricted_skeletons=True,
                                  verbose=verbose,
                                   branches_not_to_consider_for_end_nodes = non_axon_branches_on_subgraph,
                                **kwargs
                                 )
            #print(f"candidate_angles,restr_skels = {candidate_angles,restr_skels}")
            
            if candidate_angles is not None:
                sk_passing_threshold = np.where(candidate_angles>curr_axon_angle_threshold)[0]
            else:
                sk_passing_threshold = []

            if len(sk_passing_threshold) == 0:
                if verbose:
                    print(f"Not adding candidate because no angles ({candidate_angles})"
                          f" passed the threhold {curr_axon_angle_threshold} ")
                continue





            # -----------Part C: Filtering by Axon Being the Current Starting Piece -------------

            if candidate_starting_node not in axon_branches_on_limb:
                if verbose:
                    print(f"Not adding candidate the first branch was not an axon ")
                continue
                
            if not use_beginning_ais_for_width_filter:
                ais_width = curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]
            
            else:
                try:
                    #get the closest endpoint
                    endpoint_closest_to_limb_starting_coordinate = nru.closest_branch_endpoint_to_limb_starting_coordinate(limb_obj=curr_limb,
                                                    branches=[candidate_starting_node],
                                                       )
                    
                    
                    (base_final_skeleton,
                    base_final_widths,
                    base_final_seg_lengths) = nru.align_and_restrict_branch(curr_limb[candidate_starting_node],
                                              common_endpoint=endpoint_closest_to_limb_starting_coordinate,
                                             offset=0,
                                             comparison_distance=2000,
                                             skeleton_segment_size=1000,
                                              verbose=False,
                                             )
                    
                    ais_width = np.mean(base_final_widths)
                    overall_ais_width = curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]
                    if verbose:
                        print(f"base_final_widths = {base_final_widths}")
                        print(f"overall_ais_width = {overall_ais_width}")
                        print(f"ais_width = {ais_width}")
                    
                except:
                    print("Problem with calculating restricted ais so just using overall segment width")
                    ais_width = curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]
                
                
            if verbose:
                print(f"ais_width  = {ais_width}")
            if (ais_width < min_ais_width):
                    if verbose:
                        print(f'Not adding candidate the because AIS width was not higher than threshold ({min_ais_width}): {ais_width} ')
                    continue
                
            
             
                




            # ----Part D: Add all of the candidates branches and those backtracking to mesh that are axon-like
            extra_nodes_to_add = np.intersect1d(axon_branches_on_limb,current_shortest_path[:-1])
            true_axon_branches = np.hstack([curr_candidate_subgraph,extra_nodes_to_add])

            

            if curr_limb_idx not in axon_candidate_filtered:
                axon_candidate_filtered[curr_limb_idx] = dict()
                axon_candidate_filtered_angles[curr_limb_idx] = dict()

            axon_candidate_filtered[curr_limb_idx][curr_candidate_idx] = true_axon_branches
            max_axon_angle = np.max(candidate_angles)
            axon_candidate_filtered_angles[curr_limb_idx][curr_candidate_idx] = max_axon_angle
            
            if verbose:
                print(f"Adding the following branches as true axons: {true_axon_branches}\n"
                     f"curr_soma_angle = {curr_soma_angle}\n"
                     f"max_axon_angle = {max_axon_angle}")
            
            """
            Pseudocode: 
            1) if the soma angle is better and the other angle is not greater than 150, replace
            2) if both soma angles are being compared are 150, then choose the best
            candidate based on then choose based on best axon angle
            
            **might want to add something where not choose the really short one over
            the longer axon **
            
            """
            comparison_angle_threshold = 110
            replace_flag = False
            
            if verbose:
                print(f"----- Atempting to decide to replace best candidate")
                print(f"curr_soma_angle  = {curr_soma_angle}, best_axon_soma_angle = {best_axon_soma_angle} (comparison_angle_threshold = {comparison_angle_threshold})")
                
                
                
            current_candidate_limb_branch_dict = {curr_limb_idx:curr_candidate_subgraph}
            candidate_filt_dicts.append(dict(
                candidate_idx = curr_candidate_idx,
                limb_idx = curr_limb_idx,
                soma_angle=curr_soma_angle,
                axon_angle = max_axon_angle,
                axon_skeletal_length =  nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                            {curr_limb_idx:{curr_candidate_idx:true_axon_branches}},
                                                                                 "skeletal_length"),
                best_axon_candidate_filtered = {curr_limb_idx:{curr_candidate_idx:true_axon_branches}},
                
                
                
            ))    
                
            
            
            if curr_soma_angle > best_axon_soma_angle and  best_axon_soma_angle < comparison_angle_threshold:
                
                replace_flag = True
                
            elif curr_soma_angle >= best_axon_soma_angle or curr_soma_angle >= comparison_angle_threshold:
                
                #want to compute the lengths of each of these axons
                
                current_candidate_limb_branch_dict = {curr_limb_idx:curr_candidate_subgraph}
                curr_skeletal_length = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                                             current_candidate_limb_branch_dict,
                                                                             "skeletal_length")
                skeletal_length_ratio = (best_axon_skeletal_length/curr_skeletal_length)
                
                if verbose:
                    print(f"curr_skeletal_length = {curr_skeletal_length}, best_axon_skeletal_length= {best_axon_skeletal_length}, skeletal_length_ratio = {skeletal_length_ratio}")
                
                
                if max_axon_angle >= best_axon_axon_angle or (best_axon_axon_angle > 120 and skeletal_length_ratio<(1/best_axon_skeletal_legnth_ratio)):
                    if skeletal_length_ratio<best_axon_skeletal_legnth_ratio:
                        replace_flag = True
                    else:
                        if verbose:
                            print(f"Not replacing best axon because skeletal length ration ({skeletal_length_ratio}) was greater than threshold of {best_axon_skeletal_legnth_ratio} ")
                else:
                    if verbose:
                        print(f"Not replacing best axon because angle ({max_axon_angle}) was less than the {best_axon_axon_angle} \nand/or "
                             f"skeletal_length_ratio ({skeletal_length_ratio}) was not less than 1/best_axon_skeletal_legnth_ratio ({1/best_axon_skeletal_legnth_ratio})")
            
            if replace_flag:
                
                if verbose:
                    print("Changing to a better axon candidate")
                best_axon_axon_angle = max_axon_angle
                best_axon_soma_angle = curr_soma_angle
                best_axon_candidate_filtered = {curr_limb_idx:{curr_candidate_idx:true_axon_branches}}
                best_axon_candidate_filtered_angles = {curr_limb_idx:{curr_candidate_idx:max_axon_angle}}
                best_axon_skeletal_length = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                                                 best_axon_candidate_filtered,
                                                                                 "skeletal_length")
    
    #compiling list into limb_branch dict that is easy to use
    limb_branch_dict = dict()
    
    for limb_idx, limb_info in axon_candidate_filtered.items():
        curr_branches = []
        for cand_idx,cand_list in limb_info.items():
            curr_branches.append(cand_list)
        limb_branch_dict[f"L{limb_idx}"] = np.concatenate(curr_branches)
        
    #compiling list into limb_branch dict that is easy to use
    best_limb_branch_dict = dict()
    
    for limb_idx, limb_info in best_axon_candidate_filtered.items():
        curr_branches = []
        for cand_idx,cand_list in limb_info.items():
            curr_branches.append(cand_list)
        best_limb_branch_dict[f"L{limb_idx}"] = np.concatenate(curr_branches)
    
    if best_axon:
        # --- 2_17: Will add back axon parts that should be accounted for 
        
        """
        Pseudocode: 
        if there even is an axon
        1) Get the limb name of the best axon
        2) Get the concept network of the limb and the starting node
        3) Get the branches that would be axon
        4) Delete the branches from the concept network
        5) For each connected component in the leftover network
        - if not have starting node and all in axon-like: 
        add to list to add to the true axon

        6) add the new nodes to the axon group
        """


        if len(best_limb_branch_dict) > 0:

            #1) Get the limb name of the best axon
            limb_name_of_axon = list(best_limb_branch_dict.keys())
            if len(limb_name_of_axon) > 1:
                raise Excpetion("More than 1 axon key")

            limb_name_of_axon = limb_name_of_axon[0]

            curr_limb = neuron_obj[limb_name_of_axon]
            #2) Get the concept network of the limb and the starting node
            conc_net = nx.Graph(curr_limb.concept_network)
            start_node = curr_limb.current_starting_node

            #3) Get the branches that would be axon
            axon_branches = best_limb_branch_dict[limb_name_of_axon]
            axon_like_branches = axon_like_limb_branch_dict[limb_name_of_axon]

            #4) Delete the branches from the concept network
            conc_net.remove_nodes_from(axon_branches)


            #5) For each connected component in the leftover network
            new_axon_branches = []
            for conn_comp in nx.connected_components(conc_net):
                conn_comp = list(conn_comp)

                #- if not have starting node and all in axon-like: 
                #add to list to add to the true axon
                axon_like_in_conn_comp = np.intersect1d(axon_like_branches,conn_comp)
                if start_node not in conn_comp and len(axon_like_in_conn_comp) == len(conn_comp):
                    new_axon_branches += conn_comp

    
        
            if len(new_axon_branches) > 0:
                best_limb_branch_dict[limb_name_of_axon] = np.array(list(axon_branches) + new_axon_branches)


        if verbose:
            print("Using the best axon approach")
                    
        limb_branch_dict = best_limb_branch_dict 
        axon_candidate_filtered_angles = best_axon_candidate_filtered_angles
        
        # ------- 2/17: Wil add back axon branches that have been cut off by best axon feature
#         from python_tools import system_utils as su
#         su.compressed_pickle(axon_like_limb_branch_dict,"axon_like_limb_branch_dict")
#         su.compressed_pickle(best_limb_branch_dict,"best_limb_branch_dict")
        
    if return_axon_angles:
        return limb_branch_dict, axon_candidate_filtered_angles
    else:
        return limb_branch_dict

    
# --------- arguments to use for the axon identification -----------
axon_width_like_requirement = "(median_mesh_center < 200)"# or no_spine_median_mesh_center < 150)"
ais_axon_width_like_requirement = "(median_mesh_center < 600)" #this will get all of axons including the axon 
def axon_width_like_query_revised(width_to_use,
                                  spine_limit,
                                 spine_density=None):
   
    axon_width_like_query = f"((n_spines < 7) and ({width_to_use}) and spine_density <= 0.00008)"
    return axon_width_like_query

def axon_width_like_segments(current_neuron,
                      current_query=None,
                      current_functions_list=None,
                             include_ais=True,
                             verbose=False,
                             
                             #arguments for the axon finding
                             non_ais_width = None,#200,
                             ais_width = None,#600,
                             max_n_spines = None,#7
                             max_spine_density = None,#0.00008
                             width_to_use=None,
                            
                            plot = False):
    """
    Purpose: Will get all of the branches that look like spines
    based on width and spine properties
    
    """
    
    
    if non_ais_width is None:
        non_ais_width = non_ais_width_axon_global
    if ais_width is None:
        ais_width = ais_width_axon_global
    if max_n_spines is None:
        max_n_spines = max_n_spines_axon_global
    if max_spine_density is None:
        max_spine_density = max_spine_density_axon_global
    
    functions_list = ["median_mesh_center",
                     "n_spines",
                     "spine_density"]
    
    if current_functions_list is None:
        current_functions_list = functions_list
        
    if current_query is None:
        if not width_to_use is None:
            width_expression  = f"(median_mesh_center < {width_to_use})"
        else:
            if include_ais:
                spine_limit = 5
                width_expression = f"(median_mesh_center < {ais_width})"
            else:
                spine_limit = 3
                width_expression = f"(median_mesh_center < {non_ais_width})"
                
                
        current_query = f"((n_spines < {max_n_spines}) and ({width_expression}) and spine_density <= {max_spine_density})"
        #current_query = axon_width_like_query_revised(width_expression,spine_limit)
        
        
    if verbose:
        print(f"current_query = {current_query}")
        
    limb_branch_dict = ns.query_neuron(current_neuron,
                                       #query="n_spines < 4 and no_spine_average_mesh_center < 400",
                                       query=current_query,
                                       #return_dataframe=True,
                   functions_list=current_functions_list)
    
    if plot:
        print(f"limb_branch_dict= {limb_branch_dict}")
        nviz.plot_limb_branch_dict(current_neuron,limb_branch_dict)
    return limb_branch_dict


    
def axon_like_limb_branch_dict(neuron_obj,
                              downstream_face_threshold=3000,
                                width_match_threshold=50,
                               downstream_non_axon_percentage_threshold=0.3,
                               distance_for_downstream_check=40000,
                               max_skeletal_length_can_flip=70000,
                              include_ais=True,
                              plot_axon_like=False,
                              verbose=False):
    neuron_obj = copy.deepcopy(neuron_obj)
    axon_like_limb_branch_dict = clu.axon_width_like_segments(neuron_obj,
                                                        include_ais=include_ais)
    if verbose:
        print(f"axon_like_limb_branch_dict = {axon_like_limb_branch_dict}")


    

    current_functions_list = ["axon_segment"]
    final_axon_like_classification = ns.query_neuron(neuron_obj,

                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=downstream_face_threshold,
                                                            width_match_threshold=width_match_threshold,
                                                            downstream_non_axon_percentage_threshold=downstream_non_axon_percentage_threshold,
                                                            distance_for_downstream_check=distance_for_downstream_check,
                                                            max_skeletal_length_can_flip=max_skeletal_length_can_flip,
                                                            
                                                           print_flag=verbose),
                                       functions_list=current_functions_list)
    
    if verbose:
        print(f"final_axon_like_classification = {final_axon_like_classification}")
    
    if plot_axon_like:
        nviz.plot_limb_branch_dict(neuron_obj,
                                  final_axon_like_classification)

    return final_axon_like_classification
    
    
def clear_axon_labels_from_dendritic_paths_to_starter_node(limb_obj,
                                                          axon_branches=None,
                                                          dendritic_branches=None,
                                                          verbose=False):
    
    """
    Purpose: To make sure that no axon branches are on path of dendritic branches
    back to the starting node of that limb
    
    Pseudocode:
    1a) if dendritic branches are None then use axon branches to figure out
    1b) If axon branches are None....
    2) If dendritic branches or axon branches are empty then just return original axon branches
    3) Find starting node of branch
    4) for all dendritic branches:
        i) find the shortest path back to starting node
        ii) Add those nodes on path to a list to make sure is not included in axons
    5) Subtract all the non-axon list from the axon branches
    6) Return the new axon list
    
    Ex:
    final_axon_branches = clu.clear_axon_labels_from_dendritic_paths_to_starter_node(limb_obj=neuron_obj["L4"],
                                                          axon_branches=neuron_obj.axon_limb_branch_dict["L4"],
                                                          dendritic_branches=None,
                                                          verbose=True)
    final_axon_branches
    """
    curr_limb_branches = limb_obj.get_branch_names()
    
    if verbose:
        print(f"curr_limb_branches = {curr_limb_branches}")
        
    if dendritic_branches is None and axon_branches is None:
        raise Exception("At least dendritic_branches or axon_branches needs to be non-None ")
    
    # 1a) if dendritic branches are None then use axon branches to figure out
    if dendritic_branches is None:
        dendritic_branches = np.setdiff1d(curr_limb_branches,axon_branches)
    
    #1b) If axon branches are None....
    if axon_branches is None:
        axon_branches = np.setdiff1d(curr_limb_branches,dendritic_branches)
    
    if verbose:
        print(f"dendritic_branches = {dendritic_branches}")
        print(f"axon_branches = {axon_branches}")
        
        
    #2) If dendritic branches or axon branches are empty then just return original axon branches
    if len(dendritic_branches) == 0 or len(axon_branches) == 0:
        if verbose:
            print("No processing needed beause dendrites or axon are empty")
        return axon_branches
    
    #3) Find starting node of branch
    starting_branch = limb_obj.current_starting_node
    
    dendritic_nodes_by_path_check = []
    branches_to_check = np.array(dendritic_branches)
    
    while len(branches_to_check) > 0:
        curr_b = branches_to_check[0]
        curr_shortest_path = nx.shortest_path(limb_obj.concept_network,
                                              curr_b,
                                             starting_branch)

        dendritic_nodes_by_path_check += curr_shortest_path
        
        branches_to_check = np.setdiff1d(branches_to_check,curr_shortest_path)
    
    final_non_axonal_branches = np.unique(dendritic_nodes_by_path_check)
    
    if verbose:
        print(f"final_non_axonal_branches = {final_non_axonal_branches}")
    
    final_axon_list = np.setdiff1d(axon_branches,final_non_axonal_branches)
    return final_axon_list

def axon_classification(neuron_obj,
                        
                        
    error_on_multi_soma = True,
    ais_threshold = 14_000,#9000,# 5000,#10000,
                        
                        #Part 1: for axon-like classification
                        downstream_face_threshold=3000,
                        width_match_threshold=50,
                        plot_axon_like_segments=False,
                        
                        #Part 2: Filter limbs by starting angle
                        axon_soma_angle_threshold = 70,
                        
                        #Part 3: Creating Candidates
                        plot_candidates = False,
                        
                        #Part 4: Filtering Candidates
                        plot_axons = False,
                        plot_axon_errors=False,
                        
                        axon_angle_threshold_relaxed = 95,#110,
                        axon_angle_threshold = 120,
                        
    
    add_axon_labels = True,
    clean_prior_axon_labels=True,  
    label_axon_errors =True,
                        
    error_width_max = 140,
    error_length_min = None,#5000,
                        
    return_axon_labels=True,
    return_axon_angles=False,
                        
    return_error_labels=True,
    best_axon = True,
                        
    no_dendritic_branches_off_axon=True,
                        
                        
    verbose = False,
                        
    **kwargs
    ):
    """
    Purpose: 
    To put the whole axon classificatoin steps 
    together into one function that will labels
    branches as axon-like, axon and error
    
    1) Classify All Axon-Like Segments
    2) Filter Limbs By Starting Angle
    3) Get all of the Viable Candidates
    4) Filter Candidates
    5) Apply Labels
    
    """
    
    plot_candidates= False
    plot_axon_like_segments = False
    
    
    curr_neuron_obj = neuron_obj
    
    soma_names = curr_neuron_obj.get_soma_node_names()
    
    if len(soma_names)>1:
        soma_print = f"More than 1 soma: {soma_names}"
        if error_on_multi_soma:
            raise Exception(soma_print)
        else:
            print(soma_print)

    soma_name = soma_names[0]
    
    
    
    
    
    #  ------------- Part 1: Classify All Axon-Like Segments ----------------------------
    
    
    '''  Old way that did not use the classification
    
    axon_like_limb_branch_dict = ns.axon_width_like_segments(curr_neuron_obj,
                                                        include_ais=True)

    # nviz.visualize_neuron(curr_neuron_obj,
    #                       visualize_type=["mesh"],
    #                      limb_branch_dict=axon_like_limb_branch_dict,
    #                      mesh_color="red",
    #                       mesh_color_alpha=1,
    #                      mesh_whole_neuron=True)

    current_functions_list = ["axon_segment"]
    final_axon_like_classification = ns.query_neuron(curr_neuron_obj,

                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=downstream_face_threshold,
                                                            width_match_threshold=width_match_threshold,
                                                           print_flag=False),
                                       functions_list=current_functions_list)
    '''
    
    final_axon_like_classification = axon_like_limb_branch_dict(curr_neuron_obj,
                                                                downstream_face_threshold=downstream_face_threshold,
                                                                width_match_threshold=width_match_threshold,
                                                               )

    if plot_axon_like_segments:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=final_axon_like_classification,
                             mesh_color="red",
                              mesh_color_alpha=1,
                             mesh_whole_neuron=True)
        
        
    
    if verbose:
        print(f"\nPart 1: Axon like branchese \n{final_axon_like_classification}")
        
        
        
    #------------------ Part 2: Filter Limbs By Starting Angle  ------------------
    
    
    

    soma_center = curr_neuron_obj["S0"].mesh_center

    possible_axon_limbs_dict = ns.query_neuron(curr_neuron_obj,
                    query=f"soma_starting_angle>{axon_soma_angle_threshold}",
                   functions_list=[ns.soma_starting_angle],
                   function_kwargs=dict(soma_center=soma_center,
                                       verbose=verbose))

    possible_axon_limbs = list(possible_axon_limbs_dict.keys())
    possible_axon_limbs = [nru.get_limb_int_name(k) for k in possible_axon_limbs]
    
    if verbose: 
        print(f'\nPart 2: possible_axon_limbs = {possible_axon_limbs}')
              
              
                
                
    
    #---------------------- Part 3: Get all of the Viable Candidates ----------------------
    
    
    axon_subgraph_candidates = clu.axon_candidates(curr_neuron_obj,
                   possible_axon_limbs=possible_axon_limbs,
                        ais_threshold=ais_threshold,
                   plot_candidates_after_adding_back=plot_candidates,
                   verbose=verbose,
                                                   
                                                  **kwargs)
              
    if verbose:
        print(f"Part 3: axon_subgraph_candidates = {axon_subgraph_candidates}")
        
        
        
        
        
    
    #---------------------- Part 4: Filtering The Candidates ----------------------
    
    curr_result = clu.filter_axon_candiates(
                            curr_neuron_obj,
                            axon_subgraph_candidates,
                            verbose = verbose,
                            axon_like_limb_branch_dict=final_axon_like_classification,
                                return_axon_angles=True,
                                axon_angle_threshold_relaxed = axon_angle_threshold_relaxed,
                                axon_angle_threshold = axon_angle_threshold,
                                best_axon=best_axon,
                            **kwargs
                            )
    
    
    
    
    final_true_axons, axon_angles = curr_result
    
    if no_dendritic_branches_off_axon:
        if verbose:
            print(f"Using {no_dendritic_branches_off_axon}")
        final_true_axons_new= dict()
        for limb_name,axon_branches in final_true_axons.items():
            
            new_axon_branches = clu.clear_axon_labels_from_dendritic_paths_to_starter_node(limb_obj=neuron_obj[limb_name],
                                                          axon_branches=axon_branches,
                                                          verbose=False)
            if verbose:
                print(f"Limb {limb_name}: Axon branches before dendritic path filter = {axon_branches}")
                print(f"Limb {limb_name}: Axon branches AFTER dendritic path filter = {new_axon_branches}")
                
            final_true_axons_new[limb_name] = new_axon_branches
        final_true_axons = final_true_axons_new
    
    
    if verbose:
        print(f"\n\nPart 4: final_true_axons = {final_true_axons}")
    
    if plot_axons:
        if len(final_true_axons)>0:
            nviz.visualize_neuron(curr_neuron_obj,
                                  visualize_type=["mesh"],
                                 limb_branch_dict=final_true_axons,
                                 mesh_color="red",
                                  mesh_color_alpha=1,
                                 mesh_whole_neuron=True)
        else:
            print("NO AXON DETECTED FOR PLOTTING")
        
        
        
    #---------------------- Part 5: Adding Labels ----------------------
    """
    Pseudocode: 
    1) Clear the labels if option set
    2) Label all the true axon branches
    
    
    """
    
    if clean_prior_axon_labels and add_axon_labels:
        nru.clear_all_branch_labels(curr_neuron_obj,["axon","axon-like","axon-error"])
    
#     nru.add_branch_label(curr_neuron_obj,
#                     limb_branch_dict=final_axon_like_classification,
#                     labels="axon-like")
    
    #adding the labels
    if add_axon_labels:
        nru.add_branch_label(curr_neuron_obj,
                        limb_branch_dict=final_true_axons,
                        labels="axon")

        nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=final_axon_like_classification,
                    labels="axon-like")
        
        
    if (label_axon_errors or return_error_labels) and add_axon_labels:
        
        
        if error_width_max is None and error_length_min is None:
            error_limb_branch_dict = ns.query_neuron_by_labels(curr_neuron_obj,
                                 matching_labels = ["axon-like"],
                                 not_matching_labels = ["axon"]
                                 )
        else:
            if error_length_min is None:
                error_length_min = 0
            if error_width_max is None:
                error_width_max = np.inf
            
            error_limb_branch_dict = ns.query_neuron(neuron_obj,
                            query=f"(labels_restriction == True) and (median_mesh_center < {error_width_max}) "
                                                     f"and (skeletal_length > {error_length_min})",
                   functions_list=["labels_restriction","median_mesh_center","skeletal_length"],
                   function_kwargs=dict(matching_labels=["axon-like"],
                                        not_matching_labels=["axon"]
                                       )
                           )

        if label_axon_errors:
            nru.add_branch_label(curr_neuron_obj,
                            limb_branch_dict=error_limb_branch_dict,
                            labels="axon-error")
    
        if plot_axon_errors:
            if len(error_limb_branch_dict) > 0:
                nviz.visualize_neuron(curr_neuron_obj,
                                      visualize_type=["mesh"],
                                     limb_branch_dict=error_limb_branch_dict,
                                     mesh_color="red",
                                      mesh_color_alpha=1,
                                     mesh_whole_neuron=True)
            else:
                print("NO AXON ERRORS DETECTED FOR PLOTTING!!")
        

    if return_axon_labels and return_error_labels:
        if return_axon_angles:
            return final_true_axons,axon_angles,error_limb_branch_dict
        else:
            return final_true_axons,error_limb_branch_dict
    elif return_axon_labels:
        if return_axon_angles:
            return final_true_axons,axon_angles
        else:
            return final_true_axons
    elif return_error_labels:
        return error_limb_branch_dict
                                                    
        

        
# ----------- 2/15: Visualizing the Axon Classification --------- #


def axon_limb_branch_dict(neuron_obj):
    axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         matching_labels=["axon"])
    return axon_limb_branch_dict

def dendrite_limb_branch_dict(neuron_obj):
    dendrite_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         not_matching_labels=["axon"])
    return dendrite_limb_branch_dict

def dendrite_branches_on_limb(neuron_obj,limb_name):
    dendrite_limb_branch_dict = clu.dendrite_limb_branch_dict(neuron_obj)
    if type(limb_name) != str:
        limb_name = f"L{limb_name}"
    if limb_name in dendrite_limb_branch_dict:
        return dendrite_limb_branch_dict[limb_name]
    else:
        return np.array([])

def axon_mesh_from_labels(neuron_obj,
                         verbose=False,
                         plot_axon=False):
    """
    Will compile the axon mesh from the 
    labels stored in the neuron object
    
    
    Ex: 
    clu.axon_mesh_from_labels(neuron_obj,
                     plot_axon=False,
                     verbose=True)
    
    """
    axon_limb_branch = axon_limb_branch_dict(neuron_obj)
    
    axon_meshes = nru.feature_over_limb_branch_dict(neuron_obj,
                                                    axon_limb_branch,
                                                    feature="mesh",
                                                   )
    axon_mesh = tu.combine_meshes(axon_meshes)
    
    if verbose:
        print(f"Final axon_mesh = {axon_mesh}")
        
    if plot_axon:
        nviz.plot_objects(axon_mesh)
        
    return axon_mesh

def axon_faces_from_labels_on_original_mesh(neuron_obj,
                                           original_mesh,
                                           original_mesh_kdtree=None,
                                            plot_axon=False,
                                           verbose=False,
                                           **kwargs):
    """
    Purpose: To get the axon face indices on the original mesh
    
    Pseudocode:
    1) Get the original mesh if not passed
    2) Get the axon mesh of the neuron object
    3) Map the axon mesh to the original mesh
    
    Ex: 
    clu.axon_faces_from_labels_on_original_mesh(neuron_obj,
                                            plot_axon=True,
                                           verbose=True,
                                           original_mesh=original_mesh,
                                           original_mesh_kdtree=original_mesh_kdtree)
    """
        
    axon_mesh = axon_mesh_from_labels(neuron_obj,
                                     verbose=verbose,
                                     **kwargs)
    
    axon_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                        axon_mesh,
                                                        exact_match=True,
                                                        original_mesh_kdtree=original_mesh_kdtree)
    
    if plot_axon:
        nviz.plot_objects(original_mesh,
                          meshes=[original_mesh.submesh([axon_mesh_faces],append=True)],
                         meshes_colors="red")
        
    return axon_mesh_faces

        
        
        
# ----------- 1/22: Apical Classification (The beginning parts): ----------------- #


def apical_branch_candidates_on_limb(limb_obj,
                                     
                                     
                                    apical_check_distance_max = 90000,
                                    apical_check_distance_min = 25000,
                                    plot_restricted_skeleton = False,
                                    plot_restricted_skeleton_with_endnodes=False,
                                     
                                     
                                    angle_threshold = 30,
                                    top_volume_vector = np.array([0,-1,0]),
                                     
                                     spine_density_threshold = 0.00001,
                                    total_skeleton_distance_threshold_multiplier = 0.5,
                                    #apical_width_threshold = 350,
                                     apical_width_threshold = 240,
                                    upward_distance_to_skeletal_distance_ratio_threshold = 0.85,
                                    
                                    verbose=False,
                                    **kwargs):
    """
    Purpose: To identify the branches on the limb that are most likely 
    part of a large upward apical branch
    
    
    Psuedoode:
    0a) Getting the subskeleton region to analyze
    0b) Divided the Restricted Skeleton into components to analyze
    
    For each connected component
    1) Get all the end nodes of the subgraph
    2) Subtract of the closest subgraph node to limb start
    For each end node
    3) Look at the vector between end nodes and closest node 
        (continue if not approximately straight up) and not long enough
    4) Find the branches that contain the two ends of the path

    For all combinations of branches:

    5) Find the shortest path between the two branches on the context network
    6) Get the subskeleton:
    - Analyze for width and spine density (and if too thin or not spiny enough then continue)
    7) If passed all tests then add the branch path as possible candidate
    
    """
    
    
    
    curr_limb = limb_obj
    apical_branches = []
    
    total_skeleton_distance_threshold = total_skeleton_distance_threshold_multiplier*(apical_check_distance_max - apical_check_distance_min)
    
    
    #0a) Getting the subskeleton region to analyze
    
    
    limb_gr = sk.convert_skeleton_to_graph(curr_limb.skeleton)
    st_node = xu.get_graph_node_by_coordinate(limb_gr,curr_limb.current_starting_coordinate)
    nodes_max_distance = xu.find_nodes_within_certain_distance_of_target_node(limb_gr,st_node,apical_check_distance_max)
    nodes_min_distance = xu.find_nodes_within_certain_distance_of_target_node(limb_gr,st_node,apical_check_distance_min)
    nodes_with_distance_range = np.setdiff1d(list(nodes_max_distance),list(nodes_min_distance))


    restricted_limb_gr = limb_gr.subgraph(nodes_with_distance_range)
    restricted_limb_sk = sk.convert_graph_to_skeleton(restricted_limb_gr)
    
    if plot_restricted_skeleton:
        nviz.plot_objects(curr_limb.mesh,
                         skeletons=[restricted_limb_sk])
        
        
    #0b) Divided the Restricted Skeleton into components to analyze
        
    conn_comp = list([np.array(list(k)) for k in nx.connected_components(restricted_limb_gr)])
    conn_comp_closest_nodes = [xu.shortest_path_between_two_sets_of_nodes(limb_gr,[st_node],k)[2]
                               for k in conn_comp]
    
    if plot_restricted_skeleton_with_endnodes:
        nviz.plot_objects(curr_limb.mesh,
                     skeletons=[restricted_limb_sk],
                     scatters=[xu.get_coordinate_by_graph_node(limb_gr,conn_comp_closest_nodes)],
                     scatter_size=1)
        
        
    
    for component_idx in range(len(conn_comp)):
        
        # 1) Get all the end nodes of the subgraph
        curr_cmpnt = conn_comp[component_idx]
        closest_node = conn_comp_closest_nodes[component_idx]
        closest_node_coordinate = xu.get_coordinate_by_graph_node(limb_gr,closest_node)

        c_subgraph = restricted_limb_gr.subgraph(curr_cmpnt)
        endnodes = xu.get_nodes_of_degree_k(c_subgraph,1)

        #2) Subtract of the closest subgraph node to limb start
        filtered_endnodes = np.setdiff1d(endnodes,closest_node)
        filtered_endnodes_coordinates = xu.get_coordinate_by_graph_node(limb_gr,filtered_endnodes)

        if verbose:
            print(f"Filered End nodes for component {component_idx}: {filtered_endnodes}")
            
        
        
        for e_node_idx in range(len(filtered_endnodes)):
            
            #3) Look at the vector between end nodes and closest node 
            e_node = filtered_endnodes[e_node_idx]
            e_node_coordinate = filtered_endnodes_coordinates[e_node_idx]

            # nviz.plot_objects(curr_limb.mesh,
            #                  skeletons=[restricted_limb_sk],
            #                  scatters=[xu.get_coordinate_by_graph_node(limb_gr,[closest_node,e_node])],
            #                  scatter_size=1)

            curr_vector = e_node_coordinate-closest_node_coordinate
            curr_vector_upward_distance = -curr_vector[1]
            curr_vector_len = np.linalg.norm(curr_vector)

            curr_vector_angle = nu.angle_between_vectors(top_volume_vector,curr_vector)

            if verbose:
                print(f"End Node Candidate {e_node_idx} angle = {np.round(curr_vector_angle,2)}"
                      f"\n    Upward distance {np.round(curr_vector_upward_distance,2)}")

            reject_flag = False
            if curr_vector_angle > angle_threshold:
                if verbose:
                    print(f"Rejecting candidate because did not pass angle threshold of ess than {angle_threshold}")
                continue
                
                
                
            #4) Find the branches that contain the two ends of the path
            curr_skeleton_path = sk.convert_graph_to_skeleton(limb_gr.subgraph(nx.shortest_path(limb_gr,closest_node,e_node)))
            curr_skeleton_path_len = sk.calculate_skeleton_distance(curr_skeleton_path)

            e_node_branches = nru.find_branch_with_specific_coordinate(curr_limb,e_node_coordinate)

            closest_node_branches =  nru.find_branch_with_specific_coordinate(curr_limb,closest_node_coordinate)

            #get all possible combinations
            all_branch_pairings = nu.unique_pairings_between_2_arrays(closest_node_branches,
                                                                      e_node_branches
                                                                     )
            if verbose:
                print(f"all_branch_pairings = {all_branch_pairings}")
                
                
                
            
            #for st_branch,end_branch in all_branch_pairings
            #5) Find the shortest path between the two branches on the context network

            for curr_pairing_idx  in range(len(all_branch_pairings)):

                st_branch = all_branch_pairings[curr_pairing_idx][0]
                end_branch = all_branch_pairings[curr_pairing_idx][1]

                try:
                    branch_path = nx.shortest_path(curr_limb.concept_network,st_branch,end_branch)
                except:
                    print(f"Couln't find path between branches")

                #6) Get the subskeleton:
                #- Analyze for width and spine density (and if too thin or not spiny enough then continue)

                #total_skeleton = sk.stack_skeletons([curr_limb[k].skeleton for k in branch_path])
                skeleton_distance_per_branch = np.array([sk.calculate_skeleton_distance(curr_limb[k].skeleton) for k in branch_path])
                branch_widths = np.array([curr_limb[k].width_new["median_mesh_center"] for k in branch_path])
                branch_spines = np.array([curr_limb[k].n_spines for k in branch_path])

                total_skeleton_distance = np.sum(skeleton_distance_per_branch)
                total_spine_density = np.sum(branch_spines)/np.sum(skeleton_distance_per_branch)
                scaled_branch_width = np.sum(skeleton_distance_per_branch*branch_widths)/(total_skeleton_distance)
                curr_vector_upward_distance
                upward_to_skeletal_length_ratio = curr_vector_upward_distance/curr_skeleton_path_len


                if verbose:
                    print(f"total_spine_density = {total_spine_density}")
                    print(f"scaled_branch_width = {scaled_branch_width}")
                    print(f"curr_skeleton_path_len = {curr_skeleton_path_len}")
                    print(f"curr_vector_upward_distance = {curr_vector_upward_distance}")
                    print(f"upward ratio to length = {upward_to_skeletal_length_ratio}")
                    
                # Apply the restrictions
                if ((total_spine_density > spine_density_threshold) and
                    (total_skeleton_distance > total_skeleton_distance_threshold) and 
                    (scaled_branch_width > apical_width_threshold) and
                    (upward_to_skeletal_length_ratio > upward_distance_to_skeletal_distance_ratio_threshold)):
                    
                    #print(f"Adding the following branch path as a apical pathway: {branch_path}")
                    apical_branches += list(branch_path)
                else:
                    print("Did not pass final filters to continuing")
                    continue
    
    return np.unique(apical_branches)
            

def apical_classification(neuron_obj,
                          
                        skip_splitting=True,
                          
                          apical_soma_angle_threshold=40,
                          plot_viable_limbs = False,
                          label_neuron_branches=True,
                          plot_apical=True,
                          verbose=False,
                         **kwargs):
    """
    Will compute a limb branch dict of all 
    the branches that are part of a probably 
    long reaching apical branch
    
    Pseudocode: 
    1) Split the neuron and take the first neuron obj (assume only some in neuron)
    2) Check only 1 soma 
    3) Filter the limbs for viable aplical limbs based on the soma angle
    4) Iterate through the viable limbs to find the apical branches on each limb
    
    Ex:
    apical_classification(neuron_obj,
                          apical_soma_angle_threshold=40,
                          plot_viable_limbs = False,
                          label_neuron_branches=True,
                          plot_apical=True,
                          verbose=False)
    """
    
    split_time = time.time()
    
    if not skip_splitting:
        neuron_obj_list = pru.split_neuron(neuron_obj,
                                          plot_seperated_neurons=False,
                                          verbose=verbose)

        if verbose:
            print(f"Total time for split = {time.time() - split_time}")

        if len(neuron_obj_list)==0:
            raise Exception(f"Split Neurons not just one: {neuron_obj_list}")
            
        curr_neuron_obj = neuron_obj_list[0]
        
    
    else:
        curr_neuron_obj = neuron_obj
    
    
    viable_limbs = nru.viable_axon_limbs_by_starting_angle(curr_neuron_obj,
                                       soma_angle_threshold=apical_soma_angle_threshold,
                                       above_threshold=False,
                                       verbose=verbose)

    if verbose:
        print(f"viable_limbs = {viable_limbs}")
        
        
    if plot_viable_limbs:
        ret_col = nviz.visualize_neuron(curr_neuron_obj,
                     visualize_type=["mesh","skeleton"],
                     limb_branch_dict={f"L{k}":"all" for k in viable_limbs},
                     return_color_dict=True)
        
    
    apical_limb_branch_dict = dict()
    
    for limb_idx in viable_limbs:
        
        curr_limb = curr_neuron_obj[limb_idx]

        if verbose:
            print(f"Working on limb {limb_idx}")
        
        curr_limb_apical_branches = apical_branch_candidates_on_limb(curr_limb,
                                         verbose=verbose,
                                         **kwargs)
        if len(curr_limb_apical_branches) > 0:
            apical_limb_branch_dict.update({f"L{limb_idx}":curr_limb_apical_branches})
        
    
    if plot_apical:
        if len(apical_limb_branch_dict) > 0:
            nviz.visualize_neuron(curr_neuron_obj,
                                 visualize_type=["mesh"],
                                 limb_branch_dict=apical_limb_branch_dict,
                                 mesh_color="blue",
                                 mesh_whole_neuron=True,
                                 mesh_color_alpha=1)
        else:
            print("NO APICAL BRANCHES TO PLOT")
        
    if label_neuron_branches:
        nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=apical_limb_branch_dict,
                    labels="apical")
        
    return apical_limb_branch_dict


        
        
# ---------- For inhibitory and excitatory classification ---------- #
def contains_excitatory_apical(neuron_obj,
                             plot_apical=False,
                               return_n_apicals=False,
                            **kwargs):
    apical_limb_branch_dict = clu.apical_classification(neuron_obj,
                                                    verbose=False,
                                                    plot_apical=plot_apical,
                                               **kwargs)
    if len(apical_limb_branch_dict)>0:
        apical_flag= True
    else:
        apical_flag= False
        
    if return_n_apicals:
        apical_conn_comp = nru.limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict=apical_limb_branch_dict,
                                            use_concept_network_directional=False)
        apical_flag = len(apical_conn_comp)
        
    return apical_flag
    
def contains_excitatory_axon(neuron_obj,
                             plot_axons=False,
                             return_axon_angles=True,
                             return_n_axons=False,
                             label_axon_errors=True,
                             
                             #input arguments that indicate axon classification has already been performed
                             axon_limb_branch_dict=None,
                             axon_angles=None,
                             verbose=False,
                            **kwargs):
    
    if axon_limb_branch_dict is None or (return_axon_angles and axon_angles is None):
        return_value = clu.axon_classification(neuron_obj,
                                                        return_error_labels=False,
                                                        verbose=False,
                                                        plot_axons=plot_axons,
                                                        label_axon_errors=label_axon_errors,
                                                        return_axon_angles=return_axon_angles,
                                                   **kwargs)
        if return_axon_angles:
            axon_limb_branch_dict,axon_angles = return_value
        else:
            axon_limb_branch_dict = return_value
    else:
        if verbose:
            print("Using pre-computed axon classification for contains_excitatory_axon")
            print(f"axon_limb_branch_dict = {axon_limb_branch_dict}")
            print(f"axon_angles = {axon_angles}")
    
    
    if len(axon_limb_branch_dict)>0:
        axon_exist_flag =  True
    else:
        axon_exist_flag =  False
        
        
    if return_n_axons:
        axon_conn_comp = nru.limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict=axon_limb_branch_dict,
                                            use_concept_network_directional=False)
        axon_exist_flag = len(axon_conn_comp)
        
    if return_axon_angles:
        return axon_exist_flag,axon_angles
    else:
        return axon_exist_flag

def spine_level_classifier(neuron_obj,
                           sparsely_spiney_threshold = 0.0001,
                     spine_density_threshold = 0.0003,
                     #min_n_processed_branches=2,
                           min_processed_skeletal_length = 20000,
                           return_spine_statistics=False,
                    verbose=False,
                    **kwargs):
    """
    Purpose: To Calculate the spine density and use it to classify
    a neuron as one of the following categories based on the spine
    density of high interest branches
    
    1) no_spine
    2) sparsely_spine
    3) densely_spine
    
    """
        
    (neuron_spine_density, 
     n_branches_processed, skeletal_length_processed,
     n_branches_in_search_radius,skeletal_length_in_search_radius) = nru.neuron_spine_density(neuron_obj,
                        verbose=verbose,
                        plot_candidate_branches=False,
                        return_branch_processed_info=True,
                            **kwargs)
    
    if verbose:
        print(f"neuron_spine_density = {neuron_spine_density}")
        print(f"skeletal_length_processed = {skeletal_length_processed}")
        print(f"n_branches_processed = {n_branches_processed}")
        print(f"skeletal_length_in_search_radius = {skeletal_length_in_search_radius}")
    
#     if n_branches_processed < min_n_processed_branches:
#         final_label= "no_spined"

    if skeletal_length_processed < min_processed_skeletal_length or neuron_spine_density < sparsely_spiney_threshold:
        final_label="no_spined"
    else:
        if neuron_spine_density > spine_density_threshold:
            final_label= "densely_spined"
        else:
            final_label= "sparsely_spined"
            
    if return_spine_statistics:
        return (final_label,neuron_spine_density,
                n_branches_processed, skeletal_length_processed,
                n_branches_in_search_radius,skeletal_length_in_search_radius)
    else:
        return final_label
        
def inhibitory_excitatory_classifier(neuron_obj,
                                     verbose=False,
                                     return_spine_classification=False,
                                     return_axon_angles=False,
                                     return_n_axons=False,
                                     return_n_apicals=False,
                                     return_spine_statistics=False,
                                     axon_inhibitory_angle = 150,
                                     #axon_inhibitory_width_threshold = 350,
                                     axon_inhibitory_width_threshold = np.inf,
                                     
                                     #precomputed axon classifier info to speed up computation
                                     axon_limb_branch_dict_precomputed=None,
                                     axon_angles_precomputed=None,
                                    **kwargs):
    
    ret_value = clu.spine_level_classifier(neuron_obj,
                                                return_spine_statistics=return_spine_statistics,
                      **kwargs)
    
    if return_spine_statistics:
        (spine_category,neuron_spine_density,
        n_branches_processed, skeletal_length_processed,
                n_branches_in_search_radius,skeletal_length_in_search_radius) = ret_value
    else:
        spine_category = ret_value
    
    
    n_axons = None
    n_apicals=None
    axon_angles = None
    
    inh_exc_category = None
    
    if verbose:
        print(f"spine_category = {spine_category}")
        
    if spine_category == "no_spined":
        if verbose:
            print(f"spine_category was {spine_category} so determined as inhibitory")
        inh_exc_category = "inhibitory"
    elif spine_category == "densely_spined":
        inh_exc_category = "excitatory"
    else:
        
        n_apicals = clu.contains_excitatory_apical(neuron_obj,
                                                     return_n_apicals=return_n_apicals,
                                                     **kwargs)
        
        return_value = clu.contains_excitatory_axon(neuron_obj,
                                                 return_axon_angles=return_axon_angles,
                                                   return_n_axons=return_n_axons, 
                                                    
                                                    axon_limb_branch_dict=axon_limb_branch_dict_precomputed,
                                                    axon_angles=axon_angles_precomputed,
                                                 **kwargs)
        if return_axon_angles:
            n_axons,axon_angles = return_value
        else:
            n_axons = return_value
        
        
        if verbose:
            print(f"n_apicals = {n_apicals}")
            print(f"n_axons = {n_axons}")
            print(f"axon_angles = {axon_angles}")
            
        if n_apicals==1 or n_axons>=1:
            #---------- 1/25 Addition: If have very bottom limb and not axon and above certain width 
                #--> means it is inhibitory
            
            inh_exc_category = "excitatory"
            
            if n_axons == 0:
                nullifying_limbs = nru.viable_axon_limbs_by_starting_angle(neuron_obj,
                                       soma_angle_threshold=axon_inhibitory_angle,
                                       above_threshold=True,
                                       verbose=True)
                
                for n_limb in nullifying_limbs:
                    
                    st_node = neuron_obj[n_limb].current_starting_node
                    st_node_width = neuron_obj[n_limb][st_node].width_new["no_spine_median_mesh_center"]
                    
                    if ( st_node_width>axon_inhibitory_width_threshold):
                        
                        print(f"Classifying as inhibitory because have large downshoot ({st_node_width}) that is not an axon")
                        inh_exc_category = "inhibitory"
                        break
         
        else:
            inh_exc_category = "inhibitory"
        
    
    if (return_axon_angles or return_n_axons) and n_axons is None:
        return_value = clu.contains_excitatory_axon(neuron_obj,
                                                 return_axon_angles=return_axon_angles,
                                                    return_n_axons=return_n_axons, 
                                                    axon_limb_branch_dict=axon_limb_branch_dict_precomputed,
                                                    axon_angles=axon_angles_precomputed,
                                                 **kwargs)
        if return_axon_angles:
            n_axons,axon_angles = return_value
        else:
            n_axons = return_value
            
    if return_n_apicals and n_apicals is None:
        n_apicals = clu.contains_excitatory_apical(neuron_obj,
                                                   return_n_apicals=return_n_apicals,
                                                     **kwargs)
        
    
    
    if (not return_spine_classification and not return_axon_angles 
        and not return_n_apicals and not return_n_axons):
        return inh_exc_category
    
    return_value = [inh_exc_category]
    
    if return_spine_classification:
        return_value.append(spine_category)
        
    if return_axon_angles:
        return_value.append(axon_angles)
        
    if return_n_axons:
        return_value.append(n_axons) 
    
    if return_n_apicals:
        return_value.append(n_apicals) 
        
    if return_spine_statistics:
        return_value += [neuron_spine_density, n_branches_processed, skeletal_length_processed,
                            n_branches_in_search_radius,skeletal_length_in_search_radius]
        
        
    return return_value
    
    

def axon_starting_branch(neuron_obj,
                        axon_limb_name = None,
                        axon_branches = None,
                         verbose=False,
                        ):
    """
    Purpose: Will find the branch that is starting the
    axon according to the concept network
    
    """
    if axon_limb_name is None or axon_branches is None:
        ax_limb_branch_dict = neuron_obj.axon_limb_branch_dict
        
        if len(ax_limb_branch_dict) == 0:
            if verbose:
                print(f"No Axon in the neuron, so returning None as starting coordinate")
            return None
        
        axon_limb_name = list(ax_limb_branch_dict.keys())[0]
        axon_branches = ax_limb_branch_dict[axon_limb_name]

        
    curr_limb = neuron_obj[axon_limb_name]
    starting_branch = curr_limb.current_starting_node
    starting_coordinate = curr_limb.current_starting_coordinate

    #1) Find the axon branch that is closest to the starting 
    shortest_path,_, closest_branch = xu.shortest_path_between_two_sets_of_nodes(curr_limb.concept_network,
                                                                                 [starting_branch],axon_branches)
    if verbose:
        print(f"closest_branch = {closest_branch}")
        
    return closest_branch
        
        
def axon_starting_coordinate(neuron_obj,
                            axon_limb_name = None,
                            axon_branches = None,
                            plot_axon_starting_endpoint=False,
                            verbose=False,):
    """
    Purpose: To find the skeleton endpoint that is closest to the 
    starting node

    Pseudocode: 
    1) Find the axon branch that is closest to the starting 
    node on the concept network
    --> if it is the starting node then just return the current starting coordinate
    2) Find the endpoints of the closest branch
    3) Find the endpoint that is closest to the starting coordinate along the skeleton


    """
    if axon_limb_name is None or axon_branches is None:
        ax_limb_branch_dict = neuron_obj.axon_limb_branch_dict
        
        if len(ax_limb_branch_dict) == 0:
            if verbose:
                print(f"No Axon in the neuron, so returning None as starting coordinate")
            return None
        
        axon_limb_name = list(ax_limb_branch_dict.keys())[0]
        axon_branches = ax_limb_branch_dict[axon_limb_name]

        
    curr_limb = neuron_obj[axon_limb_name]
    starting_branch = curr_limb.current_starting_node
    starting_coordinate = curr_limb.current_starting_coordinate

    #1) Find the axon branch that is closest to the starting 
    shortest_path,_, closest_branch = xu.shortest_path_between_two_sets_of_nodes(curr_limb.concept_network,[starting_branch],axon_branches)
    if verbose:
        print(f"closest_branch = {closest_branch}")

    #--> if it is the starting node then just return the current starting coordinate
    limb_skeleton = None
    if closest_branch == starting_branch:
        axon_starting_endpoint = starting_coordinate
        if verbose:
            print("axon_starting_endpoint was same as starting_coordinate")

    else:
        #2) Find the endpoints of the closest branch
        closest_branch_endpoints = curr_limb[closest_branch].endpoints
        limb_skeleton = curr_limb.skeleton

        if verbose:
            print(f"closest_branch_endpoints= {closest_branch_endpoints}")

        #3) Find the endpoint that is closest to the starting coordinate along the skeleton
        _, axon_starting_endpoint = sk.shortest_path_between_two_sets_of_skeleton_coordiantes(
                        skeleton = limb_skeleton,
                        coordinates_list_1 = [starting_coordinate],
                        coordinates_list_2 = closest_branch_endpoints,
                        return_closest_coordinates = True,
                        plot_closest_coordinates = False)

    if plot_axon_starting_endpoint:
        if limb_skeleton is None:
            limb_skeleton = curr_limb.skeleton
        axon_branches_skeleton = sk.stack_skeletons([curr_limb[k].skeleton for k in axon_branches])
        nviz.plot_objects(main_skeleton=limb_skeleton,
                         skeletons=[axon_branches_skeleton],
                         skeletons_colors="red",
                         scatters=[starting_coordinate,axon_starting_endpoint],
                         scatters_colors=["lime","red"],
                         scatter_size=[0.4,0.3])
        
    return axon_starting_endpoint


# ----------------- 1/24/22: Rewrote the filtering function to be cleaner and have better picking from candidates -----

def filter_axon_candiates(
    neuron_obj,
    axon_subgraph_candidates,

    axon_like_limb_branch_dict = None,

    # adjusting the axon angle thresholds based on the axon composition
    axon_angle_threshold_relaxed = 110,#90,
    axon_angle_threshold = 120,
    relaxation_percentage = 0.85,
    relaxation_axon_length = np.inf,#40_000,

    #parameters for computing skeletal angle
    skeletal_angle_offset = 10000,
    skeletal_angle_comparison_distance = 10000,
    skeletal_angle_buffer = 5000,
    

    #parameters for ais filtering
    min_ais_width=85,#85,
    use_beginning_ais_for_width_filter=True,



    comparison_soma_angle_threshold = 110,
    axon_angle_winning_buffer = 15,
    axon_angle_winning_buffer_backup = 5,
    soma_angle_winning_buffer_backup = 5,
    skeletal_length_winning_buffer = 30_000,
    skeletal_length_winning_min = 10_000,
    tie_breaker_axon_attribute = "soma_plus_axon_angle",#"axon_angle",

    #for last booking on the final winning candidate
    best_axon = False,

    #plotting at end
    plot_winning_candidate = False,
    return_axon_angles = True,
    verbose = False,
    **kwargs):
    
    debug = False

    """
    Purpose: To determine the winning
    axon candidate from a list


    # Pseudocode: 
    # 1) Get the axon-like limb branch dict (to be used for later)

    # Iterate through all of the candidates
    # 2) Get the AIS branch and path back from the soma to branch
    # 3) Get the axon branches on entire limb
    # 4) Get the soma starting angle
    # 5) Calculate percentage of axon branches in the candidate
    # 6) Adjusts the soma axon angle threshold based on the axon percentage
    # 7) Calculate the skeletonstarting angle
    # 8) Filter the candidates based on the ais starting width
    # 9) adding the candidate to be considered----


    # 10) Choose the final candidates with list of queries


    """
    curr_neuron_obj = neuron_obj

    # Global lists

    candidate_filt_dicts = []

    if axon_like_limb_branch_dict is None:
        axon_like_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                functions_list=["matching_label"],
               query="matching_label==True",
               function_kwargs=dict(labels=["axon-like"]),
               )

    final_axon_like_classification = axon_like_limb_branch_dict
    curr_neuron_obj = curr_neuron_obj


    axon_candidates = nru.candidates_from_limb_branch_candidates(curr_neuron_obj,
                                              axon_subgraph_candidates,verbose = True)

    for curr_candidate_idx,axon_cand in enumerate(axon_candidates):

        curr_limb_idx = axon_cand['limb_idx']
        curr_limb = curr_neuron_obj[curr_limb_idx]
        curr_limb_name = nru.get_limb_string_name(curr_limb_idx)

        # 2) Get the AIS branch and path back from the soma to branch
        curr_candidate_subgraph = np.array(axon_cand['branches'])
        candidate_starting_node= axon_cand['start_node']

        current_shortest_path = np.flip(
            nru.branch_path_to_start_node(curr_limb,
                                          candidate_starting_node,
                                         include_branch_idx=True)
        )

        if verbose:
            print(f"\n\n---Working on axon candidate {curr_candidate_idx}: {curr_limb_idx},start_node= {candidate_starting_node}, branches = {curr_candidate_subgraph}")
            print(f"current_shortest_path = {current_shortest_path}")


        # 3) Get the axon branches on entire limb
        if curr_limb_name in final_axon_like_classification.keys():
            axon_branches_on_limb = final_axon_like_classification[curr_limb_name]
        else:
            axon_branches_on_limb = []

        if verbose:
            print(f"axon_branches_on_limb = {axon_branches_on_limb}")

        # 4) Get the soma starting angle
        curr_soma_angle = nst.soma_starting_angle(neuron_obj=neuron_obj,
                                                          limb_obj=curr_limb)
        if verbose:
            print(print(f"curr_soma_angle = {curr_soma_angle}"))

        #5) Calculate percentage of axon branches in the candidate
        axon_branches_on_subgraph = np.intersect1d(axon_branches_on_limb,curr_candidate_subgraph)
        axon_percentage_n_nodes = len(axon_branches_on_subgraph)/len(curr_candidate_subgraph)
        axon_percentage = (np.sum([curr_limb[k].skeletal_length for k in axon_branches_on_subgraph])/
                           np.sum([curr_limb[k].skeletal_length for k in curr_candidate_subgraph]))
        axon_length_over_nodes = np.sum([curr_limb[k].skeletal_length for k in axon_branches_on_subgraph])

        if verbose:
            print(f"{len(axon_branches_on_subgraph)} out of {len(curr_candidate_subgraph)} branches are axons")
            print(f"Axon percentage = {axon_percentage}")
            print(f"axon_percentage_n_nodes = {axon_percentage_n_nodes}")
            print(f"axon_length_over_nodes = {axon_length_over_nodes}")


        #6) Adjusts the soma axon angle threshold based on the axon percentage
        if axon_percentage > relaxation_percentage or axon_length_over_nodes > relaxation_axon_length:
            curr_axon_angle_threshold = axon_angle_threshold_relaxed

        elif len(axon_branches_on_subgraph) == 0:
            if verbose:
                print(f"Not adding candidate no axon branches detected ")
            continue
        else:
            curr_axon_angle_threshold = axon_angle_threshold

        if verbose:
            print(f"curr_axon_angle_threshold = {curr_axon_angle_threshold}")



        #7) Calculate the skeletonstarting angle
        candidate_nodes = np.unique(np.hstack([curr_candidate_subgraph,current_shortest_path]))

        # ----- 1/24: Filtering out the nodes that are on branches that are not axons --------- #
        non_axon_branches_on_subgraph = np.setdiff1d(candidate_nodes,axon_branches_on_limb)

        if verbose:
            print(f"candidate_nodes = {candidate_nodes}")
            print(f"non_axon_branches_on_subgraph = {non_axon_branches_on_subgraph}")




        candidate_angles,restr_skels = clu.candidate_starting_skeletal_angle(limb_obj=curr_limb,
                          candidate_nodes=candidate_nodes,
                              offset = skeletal_angle_offset,
                            axon_sk_direction_comparison_distance = skeletal_angle_comparison_distance,
                            buffer_for_skeleton = skeletal_angle_buffer,
                              top_volume_vector = np.array([0,-1,0]),
                              plot_skeleton_paths_before_restriction=False,
                              plot_skeleton_paths_after_restriction=False,
                                             return_restricted_skeletons=True,
                              verbose=verbose,
                               branches_not_to_consider_for_end_nodes = non_axon_branches_on_subgraph,
                            **kwargs
                             )
        #print(f"candidate_angles,restr_skels = {candidate_angles,restr_skels}")

        if candidate_angles is not None:
            sk_passing_threshold = np.where(candidate_angles>curr_axon_angle_threshold)[0]
        else:
            sk_passing_threshold = []

        if len(sk_passing_threshold) == 0:
            if verbose:
                print(f"Not adding candidate because no angles ({candidate_angles})"
                      f" passed the threhold {curr_axon_angle_threshold} ")
            continue



        #8) Filter the candidates based on the ais starting width
        if candidate_starting_node not in axon_branches_on_limb:
            if verbose:
                print(f"Not adding candidate the first branch was not an axon ")
            continue

        if not use_beginning_ais_for_width_filter:
            ais_width = curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]

        else:
            try:
                ais_width = nst.width_near_branch_endpoint(
                                    limb_obj = curr_limb,
                                    branch_idx = candidate_starting_node,
                                    endpoint = None, # if None then will select most upstream endpoint of branch
                                    verbose = True,
                                )

            except:
                print("Problem with calculating restricted ais so just using overall segment width")
                ais_width = curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]


        if verbose:
            print(f"ais_width  = {ais_width}")
        if (ais_width < min_ais_width):
                if verbose:
                    print(f'Not adding candidate the because AIS width was not higher than threshold ({min_ais_width}): {ais_width} ')
                continue


        # 9) adding the candidate to be considered----

        extra_nodes_to_add = np.intersect1d(axon_branches_on_limb,current_shortest_path[:-1])
        true_axon_branches = np.hstack([curr_candidate_subgraph,extra_nodes_to_add])
        max_axon_angle = np.max(candidate_angles)

        if verbose:
            print(f"Adding the following branches as true axons: {true_axon_branches}\n"
                 f"curr_soma_angle = {curr_soma_angle}\n"
                 f"max_axon_angle = {max_axon_angle}")

        if verbose:
            print(f"Adding axon candidate to be considered for final")
        candidate_filt_dicts.append(dict(
            candidate = axon_cand,
            true_axon_branches = true_axon_branches,
            candidate_idx = curr_candidate_idx,
            soma_angle=curr_soma_angle,
            axon_angle = max_axon_angle,
            soma_plus_axon_angle = curr_soma_angle + max_axon_angle,
            axon_skeletal_length =  nst.skeletal_length_over_candidate(curr_neuron_obj,axon_cand),
            width =  nst.width_over_candidate(curr_neuron_obj,axon_cand),

        ))


    if verbose:
        for i,cand in enumerate(candidate_filt_dicts):
            print(f"filt cand {i}= \n{cand}")
    if len(candidate_filt_dicts) > 0:
        C = flu.Comparator(candidate_filt_dicts,
                          #object_attributes=candidate_filt_dicts_atts
                          )

        C.compute_global_functions(
            attributes_list=["soma_angle",
                             "axon_angle",
                             "axon_skeletal_length",
                            "soma_plus_axon_angle"],
            verbose = True
        )


        objs_filt,df = flu.filter_to_one_by_query(
            C,
            queries = [f"soma_angle >= {comparison_soma_angle_threshold}",
                       f"(axon_skeletal_length > {skeletal_length_winning_min}) and "
                       f"(axon_angle_diff_from_max > -{axon_angle_winning_buffer})",
                        f"(axon_skeletal_length > {skeletal_length_winning_min}) and "
                       f"(axon_angle_diff_from_max > -{axon_angle_winning_buffer_backup}) and"
                       f"(soma_angle_diff_from_max > - {soma_angle_winning_buffer_backup})",
                       
#                        f"(axon_skeletal_length > {skeletal_length_winning_min}) and "
#                        f"(axon_skeletal_length_diff_from_max > -{skeletal_length_winning_buffer})",
                      f"{tie_breaker_axon_attribute} == MAX({tie_breaker_axon_attribute})"],
        return_df_before_query=True,
        verbose=True)
        
        if debug:
            su.compressed_pickle(df,"df_candidates")

        winning_candidate = objs_filt[0]
        winning_limb_idx = winning_candidate["candidate"]["limb_idx"]
        best_axon_candidate_filtered = {winning_limb_idx:winning_candidate["true_axon_branches"]}
        best_axon_candidate_filtered_angles = {winning_limb_idx:
                                               {winning_candidate["candidate_idx"]:winning_candidate["axon_angle"]}}

        if best_axon:
            # --- 2_17: Will add back axon parts that should be accounted for 

            """
            Pseudocode: 
            if there even is an axon
            1) Get the limb name of the best axon
            2) Get the concept network of the limb and the starting node
            3) Get the branches that would be axon
            4) Delete the branches from the concept network
            5) For each connected component in the leftover network
            - if not have starting node and all in axon-like: 
            add to list to add to the true axon

            6) add the new nodes to the axon group
            """


            if len(best_axon_candidate_filtered) > 0:

                #1) Get the limb name of the best axon
                limb_name_of_axon = list(best_axon_candidate_filtered.keys())
                if len(limb_name_of_axon) > 1:
                    raise Excpetion("More than 1 axon key")

                limb_name_of_axon = limb_name_of_axon[0]

                curr_limb = neuron_obj[limb_name_of_axon]
                #2) Get the concept network of the limb and the starting node
                conc_net = nx.Graph(curr_limb.concept_network)
                start_node = curr_limb.current_starting_node

                #3) Get the branches that would be axon
                axon_branches = best_axon_candidate_filtered[limb_name_of_axon]
                axon_like_branches = axon_like_limb_branch_dict[limb_name_of_axon]

                #4) Delete the branches from the concept network
                conc_net.remove_nodes_from(axon_branches)


                #5) For each connected component in the leftover network
                new_axon_branches = []
                for conn_comp in nx.connected_components(conc_net):
                    conn_comp = list(conn_comp)

                    #- if not have starting node and all in axon-like: 
                    #add to list to add to the true axon
                    axon_like_in_conn_comp = np.intersect1d(axon_like_branches,conn_comp)
                    if start_node not in conn_comp and len(axon_like_in_conn_comp) == len(conn_comp):
                        new_axon_branches += conn_comp



                if len(new_axon_branches) > 0:
                    best_axon_candidate_filtered[limb_name_of_axon] = np.array(list(axon_branches) + new_axon_branches)


            if verbose:
                print("Using the best axon approach")
    else:
        best_axon_candidate_filtered = {}
        best_axon_candidate_filtered_angles = {}


    if verbose:
        print(f"best_axon_candidate_filtered= {best_axon_candidate_filtered}")
        print(f"best_axon_candidate_filtered_angles= {best_axon_candidate_filtered_angles}")
    if plot_winning_candidate:
        if len(best_axon_candidate_filtered) > 0:
            print(f"best_axon_candidate_filtered= {best_axon_candidate_filtered}")
            nviz.plot_limb_branch_dict(curr_neuron_obj,best_axon_candidate_filtered)
            
    if return_axon_angles:
        return best_axon_candidate_filtered,best_axon_candidate_filtered_angles
    else:
        return best_axon_candidate_filtered



# ----------------- Parameters ------------------------

global_parameters_dict_default = dict(
    
)

global_parameters_dict_default_axon = dict(
    non_ais_width_axon = 200,
    ais_width_axon = 600,
    max_n_spines_axon = 7,
    max_spine_density_axon = 0.00008,
)

attributes_dict_default = dict(
)    

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01_axon = dict(
    max_spine_density_axon = 0.00008,
    max_n_spines_axon = 7,
    ais_width_axon = 650,
)
global_parameters_dict_h01 = {}




attributes_dict_h01 = {}

# data_type = "default"
# algorithms = None

# modules_to_set = [clu]

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
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import proofreading_utils as pru

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from python_tools ---
from python_tools import filtering_utils as flu
from python_tools import general_utils as gu
from python_tools import ipyvolume_utils as ipvu
from python_tools import matplotlib_utils as mu
from python_tools import module_utils as modu
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu
from python_tools import system_utils as su

from . import classification_utils as clu