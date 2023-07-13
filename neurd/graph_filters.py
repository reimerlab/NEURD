'''






'''
import matplotlib.pyplot as plt
import networkx as nx


def upstream_pair_singular(limb_obj,
                          G=None,
                          upstream_branch=None,
                          downstream_branches=None,
                           plot_starting_branches = False,
                        offset=1000,#1500,
                        comparison_distance = 2000,
                        plot_extracted_skeletons = False,


                        worst_case_sk_angle_match_threshold = 65,

                        width_diff_max = 75,#np.inf,100,
                        width_diff_perc = 0.60,

                        perform_synapse_filter = True,
                        synapse_density_diff_threshold = 0.00015, #was 0.00021
                        n_synapses_diff_threshold = 6,

                        plot_G_local_edge = False,

                        # ----- Phase B.3: parameters for global attributes ---
                        #args for definite pairs
                        perform_global_edge_filter = True,
                        sk_angle_match_threshold = 45,
                        sk_angle_buffer = 27,

                        width_diff_perc_threshold = 0.15,
                        width_diff_perc_buffer = 0.30,

                        # ----- Phase B.4 paraeters for global query ---
                        plot_G_global_edge = False,


                        # ------- For Node Query ----#
                        perform_node_filter = False,
                        use_exclusive_partner = True,
                        plot_G_node_edge = False,
                           
                        kiss_check = False,
                        kiss_check_bbox_longest_side_threshold = 450,

                        # ---- Phase D: Picking the final winner -----
                        plot_final_branch_matches = False,
                        match_method = "all_error_if_not_one_match",# "best_match", #other option is "best_match"

                        verbose = False,
                          ):
    """
    Purpose: To pair the upstream branch
    with a possible match with a downstream branch

    Pseudocode: 
    1) Use local edge filters 
    2) Use global edge filters
    3) Perform node filters if requested
    4) If the upstream and downstream node
    are alone in same component then has pairing
    --> if not return no pairing
    
    
    Ex: 
    from python_tools import networkx_utils as xu
    import matplotlib.pyplot as plt
    import networkx as nx


    ed.upstream_pair_singular(G = G_saved,
    limb_obj = filt_neuron.axon_limb,
    upstream_branch = 65,
    downstream_branches =[30,47],
                          )

    """
    plot_G_local_edge = True
    plot_G_global_edge = True
    plot_G_node_edge= True
    verbose = True

    # find the upstream and downstream branches if None
    if upstream_branch is None:
        upstream_branch = ed.upstream_node_from_G(G)

    if downstream_branches is None:
        downstream_branches = ed.downstream_nodes_from_G(G)

    downstream_branches = np.array(downstream_branches)

    if G is None:

        G_e_2 = nst.compute_edge_attributes_locally_upstream_downstream(
                limb_obj,
                upstream_branch = upstream_branch,
                downstream_branches = downstream_branches,
                offset=offset,
                comparison_distance=comparison_distance,
                plot_extracted_skeletons=plot_extracted_skeletons
        )
    else:
        G_e_2 = G

    if plot_starting_branches:
        nviz.plot_branch_groupings(limb_obj = limb_obj,
        groupings = [[k] for k in G_e_2.nodes],
        verbose = False,
        plot_meshes = True,
        plot_skeletons = True,)


    #2) Filter the edges by local properties
    synapse_query = (f"((synapse_density_diff<{synapse_density_diff_threshold}) or" 
                        f" (n_synapses_diff < {n_synapses_diff_threshold}))")

    branch_match_query = (f"(((width_diff < {width_diff_max}) or (width_diff_percentage < {width_diff_perc}))"
                          f" and (sk_angle < {worst_case_sk_angle_match_threshold}))")

    if perform_synapse_filter:
        branch_match_query += f"and {synapse_query}"

    if verbose:
        print(f"branch_match_query = :\n{branch_match_query}")

    G_edge_filt = xu.query_to_subgraph(G_e_2,
                                      edge_query=branch_match_query,
                                      verbose=verbose)
    if plot_G_local_edge:
        print(f"\n--- Before Local Query ---")
        print(xu.edge_df(G_e_2))
        print("Afer Local query: ")
        print(xu.edge_df(G_edge_filt))
        nx.draw(G_edge_filt,with_labels=True) 
        plt.show()


    G = G_edge_filt
    if len(G_edge_filt.edges()) > 0 and perform_global_edge_filter:
        # ------------- Phase B.2: Looking at global features for query ------- #
        if verbose:
            print(f"Performing global features query")

        # 3) computes the global fetures
        edge_functions_global = dict(definite_partner_sk_delete=dict(function=nst.edges_to_delete_from_threshold_and_buffer,
                                                              arguments=dict(threshold=sk_angle_match_threshold,
                                                                                  buffer= sk_angle_buffer,
                                                                           verbose = False,
                                                                            edge_attribute = "sk_angle")),
                                definite_partner_width_delete=dict(function=nst.edges_to_delete_from_threshold_and_buffer,
                                                              arguments=dict(threshold=width_diff_perc_threshold,
                                                                                  buffer= width_diff_perc_buffer,
                                                                           verbose = False,
                                                                            edge_attribute = "width_diff_percentage"))

                         )

        # 4) Filtering Graph by global properties (applying the definite filter pair)
        G_edge_filt_with_att = nst.compute_edge_attributes_globally(G_edge_filt,
                                             edge_functions_global)
        G_global_1 = xu.query_to_subgraph(G_edge_filt_with_att,
                                          edge_query="(definite_partner_sk_delete == False) or ((definite_partner_sk_delete != True) and (definite_partner_width_delete != True))",
                                          verbose=verbose)

        if plot_G_global_edge:
            print(f"\n--- Before Global Query ---")
            print(xu.edge_df(G_edge_filt_with_att))
            print("Afer Global query: ")
            print(xu.edge_df(G_global_1))
            nx.draw(G_global_1,with_labels=True) 
            plt.show()
            

        G = G_global_1
        if len(G_global_1.edges())>0 and perform_node_filter:
            if verbose:
                print(f"Performing node features query")

            # 5) Computing NOde features (for sfiltering on the upstream node edges)
            edge_functions_node_global = dict(above_threshold_delete=dict(
                                        function=nst.edges_to_delete_on_node_above_threshold_if_one_below,
                                        arguments=dict(threshold=sk_angle_match_threshold,
                                           verbose = False)
                                        )
                             )

            if use_exclusive_partner:
                nodes_to_compute = list(G_global_1.nodes())
            else:
                nodes_to_compute = branch_idx

            G_edge_filt_with_node_att = nst.compute_edge_attributes_around_node(G_global_1,
                                             edge_functions_node_global,
                                                nodes_to_compute=nodes_to_compute,
                                             )

            # 6) Filtering graph based on node features
            G_global_2 = xu.query_to_subgraph(G_edge_filt_with_node_att,
                                          edge_query="above_threshold_delete != True",
                                          verbose=verbose)

            if plot_G_node_edge:
                print(f"\n--- Before Node Query ---")
                print(xu.edge_df(G_edge_filt_with_node_att))
                print("Afer Node query: ")
                print(xu.edge_df(G_global_2))
                nx.draw(G_global_2,with_labels=True) 
                plt.show()

            G = G_global_2

            
    # ------- Phase C: Optional Kiss Filter ------
    """
    ---------- 4/29 Addition: Kiss Filter -----------

    Pseudocode:
    0) Get the offset skeleton coordinates for all nodes in graph
    1) find all the possible partitions of the remaining nodes

    """ 
    if kiss_check:
        if verbose:
            print(f"Attempting to perform Kiss check")
        coordinate = nru.downstream_endpoint(limb_obj,upstream_branch)
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

    # Now check if any partners

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
        elif match_method == "all_error_if_not_one_match":
            error_branches = downstream_branches
            if len(upstream_subgraph) == 2:
                winning_node = upstream_subgraph[upstream_subgraph!=upstream_branch][0]
            else:
                winning_node = None
        else:
            raise Exception(f"Unimplemented match_method : {match_method} ")


        error_branches = downstream_branches[downstream_branches!= winning_node]

        if verbose:
            print(f"for upstream node {upstream_branch}, winning_node = {winning_node}, error_branches = {error_branches}")

    if plot_final_branch_matches:
        nviz.plot_branch_groupings(limb_obj = limb_obj,
        groupings = G,
        verbose = False,
        plot_meshes = True,
        plot_skeletons = True,)
        
    return winning_node,error_branches


def graph_filter_adapter(G,
                        limb_obj,
                         motif,
                         graph_filter_func,
                         attempt_upstream_pair_singular=False,
                         verbose = False,
                         **kwargs):
    """ 
    Purpose: To apply a graph filter to a specific local
    graph by 
    1) Determining if the graph filter should be run on this local graph
    2) If it should be applied, run the graph filter
    and determine if there are any branches that should be errored out
    
    
    Pseudocode:  
    1) Takes motif
    2) Runs motif on the graph
    3) Sends the graph and the limb to 
    the function to get a true false
    4) If it is yes, then maybe run the upstream pair singular

    """    
    motif_matches = dmu.graph_matches(G,motif)
    
    if len(motif_matches) == 0:
        if verbose:
            print(f"Motif not found so returning no errors")
        return []
    
    if verbose:
        print(f"motif_matches = {motif_matches}")


    d_nodes = ed.downstream_nodes_from_G(G)
    upstream_branch = ed.upstream_node_from_G(G)
    
    error_check = graph_filter_func(limb_obj=limb_obj,
                                    upstream_branch=upstream_branch,
                                   downstream_branches=d_nodes,
                                    G = G,
                                   verbose = verbose,
                                   **kwargs)
    
    if verbose:
        print(f"error_check = {error_check}")

    if error_check:
        if attempt_upstream_pair_singular:
            winning_node,error_branches = gf.upstream_pair_singular(limb_obj,G,**kwargs)
            if verbose:
                print(f"In attempt_upstream_pair_singular")
                print(f"winning_node = {winning_node}")
        else:
            error_branches = d_nodes
    else:
        error_branches = []

    if verbose:
        print(f"error_branches = {error_branches}")
        
    return error_branches
    
    


def wide_angle_t_motif(child_width_maximum = 100000,
                       child_width_minimum = 0,
                        parent_width_maximum = 75,
                        child_skeletal_threshold = 10000,
                       child_skeletal_threshold_total = 0,
                        child_angle_max = 40,):
    motif = f"""
    u -> d1
    u -> d2
    d1 -> d2 [sk_angle <= {child_angle_max}]
    
    u.node_type = "upstream"
    u.width_upstream < {parent_width_maximum}
    
    d1.width_downstream < {child_width_maximum}
    d1.width_downstream > {child_width_minimum}
    d1.skeletal_length_downstream > {child_skeletal_threshold}
    d2.skeletal_length_downstream > {skeletal_length_downstream}
    
    d1.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    
    """
    return motif

def axon_webbing_filter(G,
                         limb_obj,
                                
                        #arguments for the motif finding
                        child_width_maximum = None,#75,
                        parent_width_maximum = None,#75,
                        child_skeletal_threshold = None,#3000,#10000
                        child_skeletal_threshold_total = None,#10000,
                        child_angle_max = None,#40,
                                
                        #arguments for checking there is a valid webbing
                        web_size_threshold=None,#120,
                        web_size_type="ray_trace_median",
                        web_above_threshold = None,#True,
                        verbose = False,
                        attempt_upstream_pair_singular = None,#False,
                        error_on_web_none = False,
                       **kwargs):
    """
    Purpose: To find the error branches from the 
    axon webbing filter (no valid webbing if children branches 
    are wide angle t and the parent width is low
    
    Pseudocode: 
    1) Motif checking
    2) Check that downstream branches are connected
    3) Checking the webbing
    4) If Invalid webbing return the error branches 
    
    Ex: 
    from neurd_packages import graph_filters as gf
    gf.axon_webbing_filter(G,
                               limb_obj,
                               verbose = True,
                               child_angle_max=40,
                               child_width_maximum=90,
                                  web_size_threshold = 300)
    """
    
    if child_width_maximum is None:
        child_width_maximum = child_width_maximum_ax_web_ax_web_global
    if parent_width_maximum is None:
        parent_width_maximum = parent_width_maximum_ax_web_global
    if child_skeletal_threshold is None:
        child_skeletal_threshold = child_skeletal_threshold_ax_web_global
    if child_skeletal_threshold_total is None:
        child_skeletal_threshold_total = child_skeletal_threshold_total_ax_web_global
    if child_angle_max is None:
        child_angle_max = child_angle_max_ax_web_global
    if web_size_threshold is None:
        web_size_threshold = web_size_threshold_ax_web_global
    if web_size_type is None:
        web_size_type = web_size_type_ax_web_global
    if web_above_threshold is None:
        web_above_threshold = web_above_threshold_ax_web_global
    
    # ----- 1) Motif checking --------
#     motif = gf.wide_angle_t_motif(child_width_maximum = child_width_maximum,
#                         parent_width_maximum = parent_width_maximum,
#                         child_skeletal_threshold = child_skeletal_threshold,
#                         child_angle_max = child_angle_max,)

    motif = f"""
    u -> d1
    u -> d2
    d1 -> d2 [sk_angle <= {child_angle_max}]
    
    u.node_type = "upstream"
    u.width_upstream < {parent_width_maximum}
    
    d1.width_downstream < {child_width_maximum}
    d2.width_downstream < {child_width_maximum}
    
    d1.skeletal_length_downstream > {child_skeletal_threshold}
    d2.skeletal_length_downstream > {child_skeletal_threshold}
    
    d1.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    d2.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    """
    
    def axon_web_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                      verbose = False,
                     **kwargs):
        mesh_connection = cnu.downstream_nodes_mesh_connected(limb_obj,
                                   upstream_branch,
                                    downstream_branches=downstream_branches,
                                   n_points_of_contact = 2)
        if verbose:
            print(f"mesh_connection = {mesh_connection}")

        if not mesh_connection:
            if verbose:
                print(f"Downstream nodes were not mesh connected so returning no errors")
            return False
        
            # ----- 3) Checking the webbing --------- 
        curr_web = limb_obj[upstream_branch].web

        if curr_web is not None:
            valid_web_result = au.valid_web_for_t(curr_web,
                               size_threshold = web_size_threshold,
                               size_type = web_size_type,
                               above_threshold = web_above_threshold,
                                   verbose=verbose)
        else:
            if error_on_web_none:
                raise Exception("No webbing computed")
            else:
                valid_web_result = True
            
        return not valid_web_result
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=axon_web_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)
    
def axon_spine_at_intersection_filter(G,
                                      limb_obj,
                                    attempt_upstream_pair_singular = True,
                                      upstream_width_threshold = None,#110,
                                            downstream_width_threshold = None,#150,
                                      child_skeletal_threshold_total = None,#10000,
                                    verbose = False,
                                   **kwargs):
    """
    Purpose: Find error branches if there 
    is an axon spine that is downstream of the upstrea branch

    Pseudocode: 
    1) Get all downstsream nodes of the upstream branch
    2) Find the intersection with the limb axon spines
    3) if axon spine is detected
        If attempt_upstream_pair_singular:
            Run upstream_pair_singular and return error branches
        else:
            Return all downstream nodes as errors

    Ex: 
    gf.axon_spine_at_intersection_filter(G,
                                      limb_obj = filt_neuron.axon_limb,
                                    attempt_upstream_pair_singular = True,
                                    verbose = True,
                                   **dict())
    """
    if upstream_width_threshold is None:
        upstream_width_threshold = upstream_width_threshold_ax_spine_at_inters_global
    if downstream_width_threshold is None:
        downstream_width_threshold = downstream_width_threshold_ax_spine_at_inters_global
    if child_skeletal_threshold_total is None:
        child_skeletal_threshold_total = child_skeletal_threshold_total_ax_spine_at_inters_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1
    u -> d2
    u.node_type = "upstream"
    u.width_upstream < {upstream_width_threshold}
    
    d1.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    d2.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    
    d1.width_downstream < {downstream_width_threshold}
    d2.width_downstream < {downstream_width_threshold}
    
    """
    
    def axon_spine_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                      verbose = False,
                     **kwargs):
        axon_spines_on_intersection = np.intersect1d(limb_obj.axon_spines,
                                                 nru.downstream_nodes(limb_obj,upstream_branch))
        if verbose:
            print(f"Current # of axon spines = {len(limb_obj.axon_spines)}")
            print(f"axon_spines_on_intersection = {axon_spines_on_intersection}")

        return len(axon_spines_on_intersection) > 0
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=axon_spine_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)


def min_synapse_dist_to_branch_point_filter(G,
                                      limb_obj,
                                    attempt_upstream_pair_singular = True,
                                      upstream_width_threshold = None,#110,
                                            downstream_width_threshold = None,#150,
                                            min_synape_dist = None,#1300,
                                    verbose = False,
                                   **kwargs):
    """
    Purpose: Find error branches if there 
    is a synapse at the intersection

    Pseudocode: 
    1) Find the min distance of a synapse to the branching point
    3) if min distance of synapse is less than thresshold
        If attempt_upstream_pair_singular:
            Run upstream_pair_singular and return error branches
        else:
            Return all downstream nodes as errors

    Ex: 
    gf.min_synapse_dist_to_branch_point_filter(G,
                                       limb_obj,
                                       verbose=True)
    """
    
    if upstream_width_threshold is None:
        upstream_width_threshold = upstream_width_threshold_min_syn_dist_global
    if downstream_width_threshold is None:
        downstream_width_threshold = downstream_width_threshold_min_syn_dist_global
    if min_synape_dist is None:
        min_synape_dist = min_synape_dist_min_syn_dist_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1
    u -> d2
    u.node_type = "upstream"
    u.width_upstream < {upstream_width_threshold}
    d1.width_downstream < {downstream_width_threshold}
    d2.width_downstream < {downstream_width_threshold}
    
    """
    
    def min_synapse_dist_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                      verbose = False,
                     **kwargs):
        min_distance = nst.min_synapse_dist_to_branch_point(limb_obj,
                                                        downstream_branches=downstream_branches,
                                                    branch_idx = upstream_branch,
                                                    plot_closest_synapse=False,
                                                            synapse_type = "synapses_pre",
                                                    verbose = verbose)
        if verbose:
            print(f"min_distance = {min_distance}")

        return min_distance < min_synape_dist
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=min_synapse_dist_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)

def fork_divergence_filter(G,
                          limb_obj,
                          #arguments for the motif search
                        downstream_width_max = None,#90,
                        upstream_width_max = None,#90,# 145,#90,
                           
                          total_downstream_skeleton_length_threshold = None,#4000,#8000,
                    individual_branch_length_threshold = None,#3000,

                          #for the fork divergence
                          divergence_threshold_mean = None,#160,
                           attempt_upstream_pair_singular = False,
                           comparison_distance = None,
                        verbose = False,
                       **kwargs):
    """
    Purpose: Find error branches if there 
    is a forking that is too close to each other

    Pseudocode: 
    1) Get all downstsream nodes of the upstream branch
    3) if axon spine is detected
        If attempt_upstream_pair_singular:
            Run upstream_pair_singular and return error branches
        else:
            Return all downstream nodes as errors

    Ex: 
    gf.axon_spine_at_intersection_filter(G,
                                      limb_obj = filt_neuron.axon_limb,
                                    attempt_upstream_pair_singular = True,
                                    verbose = True,
                                   **dict())
    """
    
    if downstream_width_max is None:
        downstream_width_max = downstream_width_max_fork_div_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_fork_div_global
    if total_downstream_skeleton_length_threshold is None:
        total_downstream_skeleton_length_threshold = total_downstream_skeleton_length_threshold_fork_div_global
    if individual_branch_length_threshold is None:
        individual_branch_length_threshold = individual_branch_length_threshold_fork_div_global
    if divergence_threshold_mean is None:
        divergence_threshold_mean = divergence_threshold_mean_fork_div_global
    if comparison_distance is None:
        comparison_distance = comparison_distance_fork_div_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1
    u -> d2
    u.node_type = "upstream"
    u.width_upstream < {upstream_width_max}
    
    d1.width_downstream < {downstream_width_max}
    d2.width_downstream < {downstream_width_max}
    
    d1.skeletal_length_downstream > {individual_branch_length_threshold}
    d2.skeletal_length_downstream > {individual_branch_length_threshold}
    
    d1.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    d2.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    """
    
    def fork_div_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                      verbose = False,
                     **kwargs):
        div = nst.fork_divergence(limb_obj,upstream_branch,
                                  downstream_idxs = downstream_branches,
                              total_downstream_skeleton_length_threshold=0,
                              individual_branch_length_threshold = 0,
                                  comparison_distance=comparison_distance,
                                  plot_restrictions=False,
                   verbose = verbose)

        if verbose:
            print(f"div = {div}")

        return div <  divergence_threshold_mean
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=fork_div_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)

def fork_min_skeletal_distance_filter(G,
                          limb_obj,
                          #arguments for the motif search
                        downstream_width_max = None,#90,
                        upstream_width_max = None,#145,#90,
                           
                          total_downstream_skeleton_length_threshold = None,#4000,
                    individual_branch_length_threshold = None,#4000,

                          #for the fork divergence
                          min_distance_threshold = None,#550,#600,#1150,
                           attempt_upstream_pair_singular = False,
                        verbose = False,
                       **kwargs):
    """
    Purpose: Find error branches if there 
    is a forking that is too close to each other

    Pseudocode: 
    1) Get all downstsream nodes of the upstream branch
    3) if axon spine is detected
        If attempt_upstream_pair_singular:
            Run upstream_pair_singular and return error branches
        else:
            Return all downstream nodes as errors

    Ex: 
    gf.axon_spine_at_intersection_filter(G,
                                      limb_obj = filt_neuron.axon_limb,
                                    attempt_upstream_pair_singular = True,
                                    verbose = True,
                                   **dict())
    """
    
    if downstream_width_max is None:
        downstream_width_max = downstream_width_max_fork_min_dist_global
    if upstream_width_max is None:
        upstream_width_max = upstream_width_max_fork_min_dist_global
    if total_downstream_skeleton_length_threshold is None:
        total_downstream_skeleton_length_threshold = total_downstream_skeleton_length_threshold_fork_min_dist_global
    if individual_branch_length_threshold is None:
        individual_branch_length_threshold = individual_branch_length_threshold_fork_min_dist_global
    if min_distance_threshold is None:
        min_distance_threshold = min_distance_threshold_fork_min_dist_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1
    u -> d2
    u.node_type = "upstream"
    u.width_upstream < {upstream_width_max}
    
    d1.width_downstream < {downstream_width_max}
    d2.width_downstream < {downstream_width_max}
    
    d1.skeletal_length_downstream > {individual_branch_length_threshold}
    d2.skeletal_length_downstream > {individual_branch_length_threshold}
    
    d1.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    d2.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    """
    
    def fork_min_skeletal_distance_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                      verbose = False,
                     **kwargs):
        div = nst.fork_min_skeletal_distance(limb_obj,upstream_branch,
                                  downstream_idxs = downstream_branches,
                              total_downstream_skeleton_length_threshold=0,
                              individual_branch_length_threshold = 0,
                                             plot_skeleton_restriction = False,
                                             plot_min_pair=False,
                   verbose = verbose)

        if verbose:
            print(f"div = {div} with threshold set to {min_distance_threshold}")

        return div <  min_distance_threshold
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=fork_min_skeletal_distance_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)


min_double_back_threshold = ed.double_back_threshold_axon_thin
def axon_double_back_filter(G,
                          limb_obj,
                          #arguments for the motif search
                            branch_skeletal_length_min = None,#2000,#4000,
                            total_downstream_skeleton_length_threshold = None,#2000,#10000,
                            upstream_skeletal_length_min = None,#5000,
                            axon_width_threshold_thin = None,#180,#au.axon_ais_threshold
                            axon_width_threshold_thick = None,#180,#au.axon_ais_threshold
                           attempt_upstream_pair_singular = True,
                        verbose = False,
                       **kwargs):
    """
    Purpose: Find errors if branches double back by too much

    Pseudocode: 
    1) 
    """
    
    if branch_skeletal_length_min is None:
        branch_skeletal_length_min = branch_skeletal_length_min_ax_double_b_global
    if total_downstream_skeleton_length_threshold is None:
        total_downstream_skeleton_length_threshold = total_downstream_skeleton_length_threshold_ax_double_b_global
    if upstream_skeletal_length_min is None:
        upstream_skeletal_length_min = upstream_skeletal_length_min_ax_double_b_global
    if axon_width_threshold_thin is None:
        axon_width_threshold_thin = axon_width_threshold_thin_ax_double_b_global
    if axon_width_threshold_thick is None:
        axon_width_threshold_thick = axon_width_threshold_thick_ax_double_b_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1 [sk_angle >= {min_double_back_threshold}]
    u -> d2 
    u.node_type = "upstream"
    u.width_upstream < {axon_width_threshold_thick}
    u.skeletal_length_upstream_total > {upstream_skeletal_length_min}
    
    d1.skeletal_length_downstream > {branch_skeletal_length_min}
    d1.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    
    """
    
    def double_back_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                         G,
                      verbose = False,
                     **kwargs):
        #determine the double back threshold
        downstream_branches = np.array(downstream_branches)
        
        double_back_threshold = ed.double_back_threshold_axon_by_width(
                        width=G.nodes[upstream_branch]["width_upstream"],
                       double_back_threshold_thin = ed.double_back_threshold_axon_thin,
                        double_back_threshold_thick = ed.double_back_threshold_axon_thick,
        )
        
            
        sk_angle_edges = np.array([G[upstream_branch][d]["sk_angle"] for d in downstream_branches])
        double_back_branches = downstream_branches[sk_angle_edges > double_back_threshold]

        if verbose:
            print(f"double_back_threshold = {double_back_threshold}")
            print(f"sk_angle_edges = {sk_angle_edges}")
            print(f"Double back branches: {double_back_branches}")


        return len(double_back_branches) > 0
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=double_back_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)


def thick_t_filter(G,
                    limb_obj,
                    parent_width_maximum = None,#70,
                   min_child_width_max = None,#78,#85,
                   child_skeletal_threshold = None,#3000,#7000,
                   child_skeletal_threshold_total = None,#10000,
                   child_angle_max = None,#40,
                   attempt_upstream_pair_singular = False,
                   verbose = False,
                       **kwargs):
    """
    Purpose: To find the error branches from the 
    axon thick t wide angle children
    
    Example:
    gf.thick_t_filter(G,limb_obj,verbose = True,
               parent_width_maximum = 110,
              child_angle_max=150,
              )
    
    """
    if parent_width_maximum is None:
        parent_width_maximum = parent_width_maximum_thick_t_global
    if min_child_width_max is None:
        min_child_width_max = min_child_width_max_thick_t_global
    if child_skeletal_threshold is None:
        child_skeletal_threshold = child_skeletal_threshold_thick_t_global
    if child_skeletal_threshold_total is None:
        child_skeletal_threshold_total = child_skeletal_threshold_total_thick_t_global
    if child_angle_max is None:
        child_angle_max = child_angle_max_thick_t_global
    
    # ----- 1) Motif checking --------
#     motif = gf.wide_angle_t_motif(child_width_minimum = min_child_width_max,
#                         parent_width_maximum = parent_width_maximum,
#                         child_skeletal_threshold = child_skeletal_threshold,
#                         child_angle_max = child_angle_max,)

    motif = f"""
    u -> d1
    u -> d2
    d1 -> d2 [sk_angle <= {child_angle_max}]
    
    u.node_type = "upstream"
    u.width_upstream < {parent_width_maximum}
    
    d1.width_downstream > {min_child_width_max}
    
    d1.skeletal_length_downstream > {child_skeletal_threshold}
    d2.skeletal_length_downstream > {child_skeletal_threshold}
    
    d1.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    d2.skeletal_length_downstream_total > {child_skeletal_threshold_total}
    """
    
    def thick_t_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                     G,
                      verbose = False,
                     **kwargs):
        if verbose:
            print(f"Downstream angle = {G[downstream_branches[0]][downstream_branches[1]]['sk_angle']}")
            print(f"Downstream 1 widths = {G.nodes[downstream_branches[0]]['width_downstream']}")
            print(f"Downstream 2 widths = {G.nodes[downstream_branches[1]]['width_downstream']}")
            print(f"parent width = {G.nodes[upstream_branch]['width_upstream']}")
        return True
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=thick_t_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)




# ------------------ 7/26 Inhibitory Filters ----------------
min_double_back_threshold_inh = ed.double_back_threshold_axon_thick_inh
def axon_double_back_inh_filter(G,
                          limb_obj,
                          #arguments for the motif search
                            branch_skeletal_length_min = None,#2000,#4000,
                            total_downstream_skeleton_length_threshold = None,#2000,#10000,
                            upstream_skeletal_length_min = None,#5000,
                            axon_width_threshold_thin = None,#au.axon_ais_threshold,
                            axon_width_threshold_thick = None,#au.axon_ais_threshold,
                           attempt_upstream_pair_singular =None,# True,
                        verbose = False,
                       **kwargs):
    """
    Purpose: Find errors if branches double back by too much

    Pseudocode: 
    1) 
    """
    
    if branch_skeletal_length_min is None:
        branch_skeletal_length_min = branch_skeletal_length_min_double_b_axon_inh_global
    if total_downstream_skeleton_length_threshold is None:
        total_downstream_skeleton_length_threshold = total_downstream_skeleton_length_threshold_double_b_axon_inh_global
    if upstream_skeletal_length_min is None:
        upstream_skeletal_length_min = upstream_skeletal_length_min_double_b_axon_inh_global
    if axon_width_threshold_thin is None:
        axon_width_threshold_thin = axon_width_threshold_thin_double_b_axon_inh_global
    if axon_width_threshold_thick is None:
        axon_width_threshold_thick = axon_width_threshold_thick_double_b_axon_inh_global
    if attempt_upstream_pair_singular is None:
        attempt_upstream_pair_singular = attempt_upstream_pair_singular_double_b_axon_inh_global

    # ----- 1) Motif checking --------
    motif = f"""
    u -> d1 [sk_angle >= {min_double_back_threshold_inh}]
    u -> d2 
    u.node_type = "upstream"
    u.width_upstream < {axon_width_threshold_thick}
    u.skeletal_length_upstream_total > {upstream_skeletal_length_min}
    
    d1.skeletal_length_downstream > {branch_skeletal_length_min}
    d1.skeletal_length_downstream_total > {total_downstream_skeleton_length_threshold}
    
    """
    
    def double_back_func(limb_obj,
                     upstream_branch,
                     downstream_branches,
                         G,
                      verbose = False,
                     **kwargs):
        #determine the double back threshold
        downstream_branches = np.array(downstream_branches)
        
        double_back_threshold = ed.double_back_threshold_axon_by_width(
                        width=G.nodes[upstream_branch]["width_upstream"],
                       double_back_threshold_thin = ed.double_back_threshold_axon_thin_inh,
                        double_back_threshold_thick = ed.double_back_threshold_axon_thick_inh,
        )
        
            
        sk_angle_edges = np.array([G[upstream_branch][d]["sk_angle"] for d in downstream_branches])
        double_back_branches = downstream_branches[sk_angle_edges > double_back_threshold]

        if verbose:
            print(f"double_back_threshold = {double_back_threshold}")
            print(f"sk_angle_edges = {sk_angle_edges}")
            print(f"Double back branches: {double_back_branches}")

#         if len(double_back_branches) > 0:
#             print(f"\n{upstream_branch}: sk_angle_edges = {sk_angle_edges[sk_angle_edges > double_back_threshold]}")

        return len(double_back_branches) > 0
    
    return gf.graph_filter_adapter(G=G,
                         limb_obj=limb_obj,
                         motif=motif,
                         graph_filter_func=double_back_func,
                         attempt_upstream_pair_singular=attempt_upstream_pair_singular,
                         verbose = verbose,
                         **kwargs)



    

# ------------- parameters for stats ---------------

global_parameters_dict_default = dict(
    #--- axon web filters ---
    child_width_maximum_ax_web_ax_web = 75,
    parent_width_maximum_ax_web = 75,
    child_skeletal_threshold_ax_web = 3000,
    child_skeletal_threshold_total_ax_web = 10000,
    child_angle_max_ax_web = 40,

    #arguments for checking there is a valid webbing
    web_size_threshold_ax_web = 120,
    web_size_type_ax_web = "ray_trace_median",
    web_above_threshold_ax_web = True,
    
    
    #-- thick_t_filter --
    parent_width_maximum_thick_t = 70,
    min_child_width_max_thick_t = 78,#85,
    child_skeletal_threshold_thick_t = 3000,#7000,
    child_skeletal_threshold_total_thick_t = 10000,
    child_angle_max_thick_t = 40,
    
    # -- axon_double_back_filter-- 
    branch_skeletal_length_min_ax_double_b = 2000,#4000,
    total_downstream_skeleton_length_threshold_ax_double_b = 2000,#10000,
    upstream_skeletal_length_min_ax_double_b = 5000,
    axon_width_threshold_thin_ax_double_b = 180,#au.axon_ais_threshold
    axon_width_threshold_thick_ax_double_b = 180,#au.axon_ais_threshold
    
    
    # -- fork_divergence_filter --
    downstream_width_max_fork_div = 90,
    upstream_width_max_fork_div = 90,# 145,#90,
    total_downstream_skeleton_length_threshold_fork_div = 4000,#8000,
    individual_branch_length_threshold_fork_div = 3000,
    #for the fork divergence
    divergence_threshold_mean_fork_div = 160,
    comparison_distance_fork_div = 400,
    
    
    # --- fork_min_skeletal_distance_filter -- 
    downstream_width_max_fork_min_dist = 90,
    upstream_width_max_fork_min_dist = 145,#90,
    total_downstream_skeleton_length_threshold_fork_min_dist = 4000,
    individual_branch_length_threshold_fork_min_dist = 4000,
    #for the fork divergence
    min_distance_threshold_fork_min_dist = 550,#600,#1150,
    
    
    #--axon_spine_at_intersection_filter--
    upstream_width_threshold_ax_spine_at_inters = 110,
    downstream_width_threshold_ax_spine_at_inters = 150,
    child_skeletal_threshold_total_ax_spine_at_inters = 10000,

    # -- min_synapse_dist_to_branch_point_filter --
    upstream_width_threshold_min_syn_dist = 110,
    downstream_width_threshold_min_syn_dist = 150,
    min_synape_dist_min_syn_dist = 1300,
    
    #  -- axon_double_back_inh_filter ---
    branch_skeletal_length_min_double_b_axon_inh = 2000,#4000,
    total_downstream_skeleton_length_threshold_double_b_axon_inh = 2000,#10000,
    upstream_skeletal_length_min_double_b_axon_inh = 5000,
    axon_width_threshold_thin_double_b_axon_inh = 180,
    axon_width_threshold_thick_double_b_axon_inh = 180,
    attempt_upstream_pair_singular_double_b_axon_inh = True,


)
attributes_dict_default = dict(
    default_graph_filters = [
     axon_webbing_filter,
     thick_t_filter,
     axon_double_back_filter,
     fork_divergence_filter,
     fork_min_skeletal_distance_filter,
     axon_spine_at_intersection_filter,
     min_synapse_dist_to_branch_point_filter,
    ]
)    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01 = dict([(k,1.15*v) if "width" in k else (k,v) for k,v in global_parameters_dict_default.items()])

global_parameters_dict_h01_update = dict(
    branch_skeletal_length_min_ax_double_b = 4000,
    branch_skeletal_length_min_double_b_axon_inh = 4000,
    
    comparison_distance_fork_div = 800,
)

global_parameters_dict_h01.update(global_parameters_dict_h01_update)


attributes_dict_h01 = dict(
    default_graph_filters = [
     axon_webbing_filter,
     thick_t_filter,
     axon_double_back_filter,
     fork_divergence_filter,
     fork_min_skeletal_distance_filter,
     #axon_spine_at_intersection_filter,
     #min_synapse_dist_to_branch_point_filter,
    ]
)



# data_type = "default"
# algorithms = None
# modules_to_set = [gf]

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
from . import concept_network_utils as cnu
from . import error_detection as ed
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz

#--- from python_tools ---
from python_tools import dotmotif_utils as dmu
from python_tools import general_utils as gu
from python_tools import module_utils as modu
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np

from . import graph_filters as gf