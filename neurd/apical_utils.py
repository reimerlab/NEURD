'''



Good neuron to show off for classification

old_seg_id  = 864691135099943968
neuron_obj = du.decomposition_with_spine_recalculation(old_seg_id,0)
from . import apical_utils as apu
apu.apical_classification(neuron_obj,
                          plot_labels=True,
                          verbose = True
                         )



'''

from python_tools import numpy_dep as np
from python_tools import module_utils as modu
from python_tools import general_utils as gu


compartment_colors = dict(
apical = "blue",
apical_shaft = "aqua",
apical_tuft = "purple",
basal = "brown",#"yellow",
axon = "red",
oblique = "green",
#dendrite = "lightsteelblue",
dendrite = "pink",
apical_total = "magenta",
soma = "black"

)

def colors_from_compartments(compartments):
    if "str" in str(type(compartments)):
        return compartment_colors[compartments]
    return [compartment_colors[k] for k in compartments ]

compartment_name_to_int_map = dict(
no_label = 0,
axon = 1,
basal = 2,
apical = 3,
apical_tuft = 4,
apical_shaft = 5,
oblique = 6,
soma = 7,
)

coarse_fine_compartment_map = dict(
no_label = (None,None),
dendrite = ("dendrite",None),
axon = ("axon",None),
basal = ("dendrite","basal"),
apical = ("dendrite","apical"),
apical_tuft = ("dendrite","apical_tuft"),
apical_shaft = ("dendrite","apical_shaft"),
oblique = ("dendrite","oblique"),
soma = ("soma",None)
)

compartment_importance_rankings = list(coarse_fine_compartment_map.keys())[::-1]

def compartment_from_branch(branch_obj):
    curr_label = None
    for s in apu.compartment_importance_rankings:
        if s in branch_obj.labels:
            return s
        
    return curr_label

specific_apicals = ["apical_shaft","apical_tuft","oblique"]
apical_total = ["apical"] + specific_apicals
dendrite_labels = ["dendrite","basal"] + apical_total

default_compartment_order = ["apical_tuft",
              "apical_shaft",
              "oblique",
              "apical",
              "basal",
              "axon",]
def compartments_to_plot(cell_type=None):
    if cell_type == None:
        cell_type = "excitatory"
        
    if cell_type == "excitatory":
        return default_compartment_order
    else:
        return ["axon","dendrite"]
compartments_from_cell_type = compartments_to_plot
    
default_compartment_label = "dendrite"

def dendrite_compartment_labels():
    return [k for k in coarse_fine_compartment_map.keys() if k not in ["axon","no_label","soma"]]

def compartment_labels_for_stats():
    return [k for k in coarse_fine_compartment_map.keys() if k not in ["no_label","soma"]] + ["apical_total"]

def compartment_labels_for_synapses_stats():
    return [k for k in coarse_fine_compartment_map.keys() if k not in ["no_label"]] + ["apical_total"]

def compartment_labels_for_externals():
    return [k for k in coarse_fine_compartment_map.keys() if k not in ["no_label","soma"]]


def coarse_fine_compartment_from_label(label):
    if label is None:
        return (None,None)
    else:
        return coarse_fine_compartment_map[label]


def compartment_label_to_all_labels(label):
    if label == "dendrite":
        return dendrite_compartment_labels()
    elif label == "apical_total":
        return apical_total
    else:
        return [label]
    
def spine_labels_from_compartment(label):
    if label == "axon":
        return ["bouton","non_bouton"]
    elif label == "soma":
        return ["no_label"]
    else:
        return [k for k in spu.spine_labels() if k not in ["bouton","non_bouton"]]
    
def syn_type_from_compartment(label):
    if label in dendrite_compartment_labels():
        return ["postsyn"]
    else:
        return ["presyn","postsyn"]

def apical_shaft_like_limb_branch(neuron_obj,
                                       candidate = None,
                                      limb_branch_dict=None,
                                     limbs_to_process = None,
                                     max_upward_angle = None,#30,
                                       min_upward_length = None,#3000,
                                        min_upward_per_match = None,#0.8,
                                     min_upward_length_backup = None,#20000, 
                                     min_upward_per_match_backup = None,#0.5,
                                         width_min = None,#140,
                                       plot_shaft_branches = False,
                                     verbose = False,
                                        
                                      ):
    """
    Purpose: Will filter the limb branch for those that
    are apical shaft like
    
    """
    if max_upward_angle is None:
        max_upward_angle = max_upward_angle_shaft_like_global
    if min_upward_length is None:
        min_upward_length = min_upward_length_shaft_like_global
    if min_upward_per_match is None:
        min_upward_per_match = min_upward_per_match_shaft_like_global
    if min_upward_length_backup is None:
        min_upward_length_backup = min_upward_length_backup_shaft_like_global
    if min_upward_per_match_backup is None:
        min_upward_per_match_backup = min_upward_per_match_backup_shaft_like_global
    if width_min is None:
        width_min = width_min_shaft_like_global
    
    
    
    
    if candidate is not None:
        limb_branch_dict = nru.limb_branch_from_candidate(candidate)
        
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.dendrite_limb_branch_dict
    
    ap_shaft_limb_branch = ns.query_neuron(neuron_obj,
               functions_list=[ns.skeleton_dist_match_ref_vector,
                               ns.skeleton_perc_match_ref_vector,
                              ns.width_new],
               function_kwargs=dict(reference_vector=mcu.top_of_layer_vector,
                                   max_angle=max_upward_angle),
               query=f"(((skeleton_dist_match_ref_vector > {min_upward_length}) and "
                f" (skeleton_perc_match_ref_vector > {min_upward_per_match})) "
                   f" or ((skeleton_dist_match_ref_vector > {min_upward_length_backup})  and "
                                           f"(skeleton_perc_match_ref_vector > {min_upward_per_match_backup}))) and"
                             f"(width_new > {width_min})",
               limb_branch_dict_restriction=limb_branch_dict,
                limbs_to_process=limbs_to_process,
               plot_limb_branch_dict=plot_shaft_branches,
                                           #return_dataframe = True
               )
    
    if verbose:
        print(f"ap_shaft_limb_branch= {ap_shaft_limb_branch}")
    return ap_shaft_limb_branch


def apical_shaft_classification_old(
    neuron_obj,
    candidate=None,
    limb_branch_dict=None,
    plot_shaft_branches = False,
    
    #for determining the shaft candidates
    max_distance_from_soma_for_start_node = 3000,
    plot_shaft_candidates = False,
    verbose = False,
    
    #for filling in and filtering the candidates
    skip_distance = 5000,
    plot_filtered_candidates = False,
    
    #for picking final shaft
    plot_final_shaft = False,
    
    return_limb_branch_dict = True,
    **kwargs):
    """
    Purpose: Will find apical shaft on apical candidates

    Psueodocode: 
    1) Find the apical shaft query 
    2) If there are shaft branches, divide them into candidates
    3) For each candidate filter into non-branching branch list

    4) Pick the largest candidate as winner
    
    Ex: 
    from neurd_packages import apical_utils as apu
    apu.apical_shaft_classification(neuron_obj,
                               candidate=apical_candidates[0],
                               verbose = True,
                               plot_shaft_branches=True,
                               plot_shaft_candidates=True,
                                plot_filtered_candidates = True,
                               plot_final_shaft=True,

                                skip_distance = 100000,
                                   )


    """    
    apical_branches = []
    
    #1) Find the apical shaft query 
    shaft_limb_branch = apu.apical_shaft_like_limb_branch(neuron_obj,
                                        candidate=candidate,
                                            limb_branch_dict=limb_branch_dict,
                                                             #max_upward_angle=50,
                                        plot_shaft_branches=plot_shaft_branches,
                                        **kwargs)
    if verbose:
        print(f"shaft_limb_branch = {shaft_limb_branch}")

    if len(shaft_limb_branch) == 0:
        if return_limb_branch_dict:
            return {}
        else:
            return apical_branches
        
    #2) If there are shaft branches, divide them into candidates
    shaft_candidates = nru.candidate_groups_from_limb_branch(neuron_obj,
                                                            shaft_limb_branch,
                                                             plot_candidates = plot_shaft_candidates,
                                                             max_distance_from_soma_for_start_node=max_distance_from_soma_for_start_node,
                                                            )
    if verbose:
        print(f"shaft_candidates = {shaft_candidates}")
        
    if len(shaft_candidates) == 0:
        if return_limb_branch_dict:
            return {}
        else:
            return apical_branches
        
    #3) Filter all candidates into non-branching list
    shaft_candidate_limb_branch = []
    for cand in shaft_candidates:
        limb_name = cand["limb_idx"]
        curr_branch_lists = nru.fill_in_and_filter_branch_groups_from_upstream_without_branching(
                    limb_obj =  neuron_obj[limb_name],
                    branches = cand["branches"],
                    start_branch = cand["start_node"],
                    verbose = False,
                    skip_distance = skip_distance,
                    plot_filtered_branches = False
                    )
        shaft_candidate_limb_branch.append({limb_name:curr_branch_lists})
        
    if verbose:
        print(f"shaft_candidate_limb_branch = {shaft_candidate_limb_branch}")
        
    if plot_filtered_candidates:
        print(f"Plotting the filled_in_and_filtered candidates")
        nviz.plot_limb_branch_dict_multiple(neuron_obj,
                                            shaft_candidate_limb_branch,
                                            mesh_color_alpha = 1,
                                           verbose = True)
    
    #4) Pick the largest candidate as winner
    if len(shaft_candidate_limb_branch) > 1:
        sk_lens = [nru.skeletal_length_over_limb_branch(k) for k in shaft_candidate_limb_branch]
        if verbose:
            print(f"Skeletal lengths for candidates = {sk_lens}")
        winning_idx = np.argmax(sk_lens)
    else:
        winning_idx = 0 
        
    winning_limb_branch_dict = shaft_candidate_limb_branch[winning_idx]
    
    if plot_final_shaft:
        print(f"Plotting final shaft: {winning_limb_branch_dict}")
        nviz.plot_limb_branch_dict(neuron_obj,
                                   winning_limb_branch_dict
                                  )
        
    
    if return_limb_branch_dict:
        return winning_limb_branch_dict
    else:
        k = list(winning_limb_branch_dict.keys())
        if len(k) > 1:
            raise Exception("More than one limb for shaft")
        else:
            return winning_limb_branch_dict[k[0]]
    
def filter_apical_candidates(neuron_obj,
                             candidates,
                             min_skeletal_length = None,#10_000,#30000,#50000,
                            min_distance_above_soma = None,#10_000,#30000,#100000,
                             verbose = False,
                             print_node_attributes=False,
                             plot_candidates = False,
                            ):
    
    if min_skeletal_length is None:
        min_skeletal_length = min_skeletal_length_filter_apical_global
    if min_distance_above_soma is None:
        min_distance_above_soma = min_distance_above_soma_filter_apical_global
    
    if verbose:
        print(f"Starting with {len(candidates)} apical shaft candidates")
        
    objs_after_query,node_df  = flu.filter_candidates_by_query(neuron_obj,
                              candidates = candidates,
                              functions_list=[dict(function=nst.skeletal_length_over_candidate,
                                                   name="skeletal_length"),
                                             dict(function=nst.max_layer_distance_above_soma_over_candidate,
                                                 name="distance_above_soma")],
                              query=(f"(skeletal_length > {min_skeletal_length}) "
                                        f"and (distance_above_soma > {min_distance_above_soma})"
                                   ),
                                                      return_df_before_query=True)
    if print_node_attributes:
        print(node_df)
    if verbose:
        print(f"Finish filtering with {len(objs_after_query)} apical shaft candidates")
        
        
    if plot_candidates:
        print(f"Plotting candidates after filter_apical_candidates")
        nviz.plot_candidates(neuron_obj,
                             objs_after_query)
    
    return objs_after_query

def non_upward_skeletal_distance_upstream(neuron_obj,
                                         candidate,
                                         max_angle = 10000,
                                        min_angle = 40,
                                         verbose = False,
                                         **kwargs):
    """
    Purpose: Will find the amount of non-upward facing
    skeletal lengths upstream of a certain candidate
    
    """
    return nst.skeleton_dist_match_ref_vector_sum_over_branches_upstream(
    limb_obj =neuron_obj[candidate["limb_idx"]],
    branches = candidate["branches"],
    max_angle = max_angle,
    min_angle = min_angle,
    verbose = verbose)
    
    
def filter_apical_candidates_to_one(neuron_obj,
                                   candidates,
                                    non_upward_skeletal_distance_upstream_buffer = None,#-10000,
                                    soma_diff_buffer = None,#-50000,
                                    downstream_vector_diff_buffer = None,#-30000,
                                    verbose = False,
                                    default_tie_breaker = None,#"skeletal_length",
                                    plot_final_candidate = False,
                                    print_df_for_filter_to_one = False,
                                   ):
    """
    Purpose: Will filter down the remaining candidates to just one optimal
    """
    if non_upward_skeletal_distance_upstream_buffer is None:
        non_upward_skeletal_distance_upstream_buffer = non_upward_skeletal_distance_upstream_buffer_filter_apical_one_global
    if soma_diff_buffer is None:
        soma_diff_buffer = soma_diff_buffer_filter_apical_one_global
    if downstream_vector_diff_buffer is None:
        downstream_vector_diff_buffer = downstream_vector_diff_buffer_filter_apical_one_global
    if default_tie_breaker is None:
        default_tie_breaker = default_tie_breaker_filter_apical_one_global
    
    
    shaft_candidates_filtered = candidates

    if len(shaft_candidates_filtered) == 0:
        winning_shaft_candidate = None
    elif len(shaft_candidates_filtered) == 1:
        winning_shaft_candidate = shaft_candidates_filtered[0]
    elif len(shaft_candidates_filtered) > 1:
        if verbose:
            print(f"Has to apply graph filters to narrow down multiple candidates")
        winning_candidates, return_win_df  = flu.filter_candidates_to_one_by_query(neuron_obj,
                                             shaft_candidates_filtered,
                                             functions_list=[dict(function=nst.max_layer_distance_above_soma_over_candidate,
                                                         name="distance_above_soma"),
                                                            dict(function=nst.downstream_dist_match_ref_vector_over_candidate,
                                                                name="downstream_dist_match_ref_vector",
                                                                arguments=dict(max_angle=70)),
                                                            dict(function=apu.non_upward_skeletal_distance_upstream)],
                                            functions_list_graph=[dict(function=flu.distance_above_soma_diff_from_max),
                                                                 dict(function=flu.diff_from_max,
                                                                      name="downstream_vector_diff",
                                                                      arguments=dict(attribute_name="downstream_dist_match_ref_vector")),
                                                                  dict(function=flu.diff_from_min,
                                                                      name="upstream_non_upward_diff",
                                                                      arguments=dict(attribute_name="non_upward_skeletal_distance_upstream"))
                                                                     ],
                                             queries=[f"upstream_non_upward_diff > {non_upward_skeletal_distance_upstream_buffer}",
                                                      f"distance_above_soma_diff_from_max > {soma_diff_buffer}",
                                                     f"downstream_vector_diff > {downstream_vector_diff_buffer}"],
                                                                                 return_df_before_query=True,
                                                                                  verbose = False)
        if print_df_for_filter_to_one:
            print(return_win_df)
            
        if len(winning_candidates) == 1:
            winning_shaft_candidate = winning_candidates[0]
        else:
            if verbose:
                print(f"Using default tie breaker method {default_tie_breaker} because still {len(winning_candidates)} candidates")
            
            if default_tie_breaker == "skeletal_length":
                candidate_values = [nru.skeletal_length_over_candidate(neuron_obj,k) for k in winning_candidates]
                winning_idx = np.argmax(candidate_values)
                winning_shaft_candidate = winning_candidates[winning_idx]
                
            else:
                raise Exception(f"Unimplemented Type of tie breaker: {default_tie_breaker}")
            
    
    if verbose:
        print(f"winning_shaft_candidate = {winning_shaft_candidate}")

    if plot_final_candidate:
        print(f"Plotting final candidate")
        if winning_shaft_candidate is None:
            print(f"No winning candidate found")
        else:
            nviz.plot_candidates(neuron_obj,
                            candidates=[winning_shaft_candidate])
        
    return winning_shaft_candidate


def expand_candidate_branches_to_soma(neuron_obj,
                                     candidate,
                                     verbose = False,
                                     plot_candidate = False):
    if candidate is None:
        return None
    candidate = candidate.copy()
    start_node = candidate["start_node"]
    limb_obj = neuron_obj[candidate["limb_idx"]]
    
    path_to_start = nru.branch_path_to_start_node(limb_obj,start_node)
    if verbose:
        print(f"path_to_start = {path_to_start}")
        
    candidate["branches"] = np.union1d(candidate["branches"],path_to_start)
    
    if plot_candidate:
        print(f"Plotting final candidate")
        if candidate is None:
            print(f"No winning candidate found")
        else:
            nviz.plot_candidates(neuron_obj,
                            candidates=[candidate])
    return candidate


def apical_shaft_direct_downstream(neuron_obj,
                                  downstream_buffer_from_soma = 3000,
                                   plot_limb_branch=False,
                                   verbose = False
                                  ):
    """
    Purpose: To find those branches that come directly off the apical shaft
    
    Ex: 
    apu.apical_shaft_direct_downstream(n_test,
                              plot_limb_branch=True)
    """

    apical_starters = ns.query_neuron(neuron_obj,
                                 functions_list=[ns.distance_from_soma,
                                                ns.upstream_node_is_apical_shaft,
                                                 ns.labels_restriction
                                                ],
                                  query=("(upstream_node_is_apical_shaft == True) and "
                                         f"(labels_restriction == True)"),
                                 function_kwargs=dict(not_matching_labels=["apical_shaft"]),
                                 plot_limb_branch_dict=plot_limb_branch)
    if verbose:
        print(f"apical_shaft_direct_downstream = {apical_starters}")
        
    return apical_starters

    

def apical_classification(
    neuron_obj,

    #filtering limbs
    soma_angle_max = None,
    plot_filtered_limbs = False,
    
    multi_apical_height = None,
    
    # shaft-like limb branch
    plot_shaft_like_limb_branch = False,
    
    # finding shaft candidates
    plot_candidates = False,
    candidate_connected_component_radius = None,#5000,
    multi_apical_possible = None,
    
    #for base filtering
    plot_base_filtered_candidates = False,
    plot_base_filtered_df = False,
    verbose = False,
    
    #filter to one candidate
    plot_winning_candidate = False,
    print_df_for_filter_to_one = False,
    
    #expanding winning candidate
    plot_winning_candidate_expanded = False,
    
    # identifying entire apical
    plot_apical_limb_branch = False,
    label_basal = True,
    label_apical_tuft = True,
    label_oblique = True,
    
    
    #plotting the final labels
    plot_labels = False,
    
    apply_synapse_compartment_labels = True,
    
    rotation_function = None,
    unrotation_function = None,
    plot_rotated_function = False,
    
    #
    ):
    
    """
    Purpose: Will identify the limb branch
    that represents the apical shaft 
    
    Pseudocode: 
    1) Filter the limbs that are being considered to have the shaft
    2) Find the shalft like limb branch
    3) Divide the shaft like limb branch into candidates
    4) Filter the shaft candidates for bare minimum requirements
    5) Filter shaft candidates to one winner
    6) If a winner was found --> expand the shaft candidate to connect to soma
    7) Convert the winning candidate into a limb branch dict
    8) Find the limb branch dict of all the apical
    9) Add the apical_shaft and apical labels
    
    Ex: 
    apical_limb_branch_dict = apu.apical_classification(n_test,
                            plot_labels = True,
                           verbose = False)
    """
    if soma_angle_max is None:
        soma_angle_max = soma_angle_to_apical_global
    if multi_apical_height is None:
        multi_apical_height = multi_apical_height_global
    if rotation_function is None:
        rotation_function = rotation_function_axon_alignment
    if unrotation_function is None:
        unrotation_function = unrotation_function_axon_alignment
        
    if candidate_connected_component_radius is None:
        candidate_connected_component_radius = candidate_connected_component_radius_apical_global
        
    if multi_apical_possible is None:
        multi_apical_possible = multi_apical_possible_apical_global
        
    if (rotation_function is not None) and (unrotation_function is not None):
        neuron_obj = rotation_function(neuron_obj)
        if plot_rotated_function:
            nviz.visualize_neuron_lite(neuron_obj)
    
    #1) Filter the limbs that are being considered to have the shaft
    if soma_angle_max is not None:
        possible_apical_limbs = nst.filter_limbs_by_soma_starting_angle(neuron_obj,
                                                soma_angle = soma_angle_max,
                                                angle_less_than = True,
                                                verbose=verbose,
                                               return_int_names=True)
    else:
        possible_apical_limbs = neuron_obj.get_limb_names(return_int=True)
    if verbose: 
        print(f'\nPart 0: possible_apical_limbs = {possible_apical_limbs}')
    
    if plot_filtered_limbs:
        print(f"Plotting filtered apical limbs")
        nviz.visualize_subset_neuron_limbs(neuron_obj,possible_apical_limbs)
        
    soma_layer_height = mcu.coordinates_to_layer_height(neuron_obj["S0"].mesh_center)
    
    if verbose:
        print(f"soma_layer_height = {soma_layer_height}")
    
    if (soma_layer_height > multi_apical_height) and multi_apical_possible:
        if verbose:
            print(f"Doing the multi apical solution")
        apical_limb_branch = apu.apical_classification_high_soma_center(neuron_obj,
                                           verbose = verbose,
                                          plot_final_apical=False)

    else:
        #2) Find the shalft like limb branch
        apical_shaft_limb_branch = apu.apical_shaft_like_limb_branch(neuron_obj,
                                         limbs_to_process=possible_apical_limbs,
                                        plot_shaft_branches=plot_shaft_like_limb_branch,
                                         verbose = verbose,
                                        )

        if verbose:
            print(f"apical_shaft_limb_branch = {apical_shaft_limb_branch}")

        #3) Grouping limb branch into candidates
        shaft_candidates = nru.candidate_groups_from_limb_branch(neuron_obj,
                                                                apical_shaft_limb_branch,
                                                                connected_component_method="local_radius",
                                                                radius = candidate_connected_component_radius,
                                                                 plot_candidates=plot_candidates
                                                                )
        if verbose:
            print(f"shaft_candidates (before filtering): {shaft_candidates}")


        '''
        --------- Old Way of doing the filtering -------------

        shaft_candidates_filtered = apu.filter_candidates(neuron_obj,
                                                      filters = ("skeletal_length",),    
                         candidates=shaft_candidates,
                         verbose = verbose)

        if verbose:
            print(f"\nshaft_candidates_filtered = {shaft_candidates_filtered}")

        '''

        #4) Filter the shaft candidates for bare minimum requirements
        if len(shaft_candidates) > 0:
            shaft_candidates_filtered = apu.filter_apical_candidates(neuron_obj,
                                         shaft_candidates,
                                         verbose = False,
                                        print_node_attributes=plot_base_filtered_df,
                                        plot_candidates = plot_base_filtered_candidates
                                                                    )

            if verbose:
                print(f"shaft_candidates_filtered = {shaft_candidates_filtered}")

            #5) Filter shaft candidates to one winner
            winning_candidate = apu.filter_apical_candidates_to_one(neuron_obj,
                                           shaft_candidates_filtered,
                                           verbose = verbose,
                                           plot_final_candidate=plot_winning_candidate,
                                            print_df_for_filter_to_one=print_df_for_filter_to_one)
        else:
            winning_candidate = None

        if verbose:
            print(f"winning_candidate = {winning_candidate}")


        if winning_candidate is not None:
            #6) If a winner was found --> expand the shaft candidate to connect to soma
            final_shaft_candidate = apu.expand_candidate_branches_to_soma(neuron_obj,
                                      candidate=winning_candidate,
                                      verbose = verbose,
                                      plot_candidate=plot_winning_candidate_expanded,
                                     )

            if verbose:
                print(f"winning candidate expandid = {final_shaft_candidate}")

            #7) Convert the winning candidate into a limb branch dict
            shaft_limb_branch = nru.limb_branch_from_candidate(final_shaft_candidate)
            if verbose:
                print(f"shaft_limb_branch = {shaft_limb_branch}")

            # 7b) Adding the shaft labels
            nru.clear_all_branch_labels(neuron_obj,["apical_shaft","apical"])
            nru.add_branch_label(neuron_obj,
                            limb_branch_dict=shaft_limb_branch,
                            labels=["apical","apical_shaft"])

            #8) Find the limb branch dict of all the apical
            apical_downstream_limb_branch = apu.apical_shaft_direct_downstream(neuron_obj,
                                  plot_limb_branch=False)
            apical_limb_branch = nru.all_donwstream_branches_from_limb_branch(neuron_obj,
                                                                             apical_downstream_limb_branch)
        else:
            apical_limb_branch = {}
        
    # 7b) Adding the shaft labels
    nru.add_branch_label(neuron_obj,
                    limb_branch_dict=apical_limb_branch,
                    labels=["apical"])



    if verbose:
        print(f"apical_limb_branch = {apical_limb_branch}")
        
    if label_basal:
        if verbose:
            print(f"Adding basal labels")
        apu.basal_classfication(neuron_obj,verbose = verbose)
        
    if label_apical_tuft:
        if verbose:
            print(f"Adding apical tuft labels")
        apu.apical_tuft_classification(neuron_obj,
                                       verbose = verbose
                                      #plot_apical_tuft=True
                                      )
        
    if label_oblique:
        if verbose:
            print(f"Adding oblique labels")
        apu.oblique_classification(neuron_obj,
                                   verbose = verbose,
                                  #plot_oblique=True
                                  )
        
    if plot_labels:
        print(f"Plotting compartment classifications")
        nviz.plot_compartments(neuron_obj,
                                  )
        
    if (rotation_function is not None) and (unrotation_function is not None):
        #print(f"neuron_obj.apical_limb_branch_dict = {neuron_obj.apical_limb_branch_dict}")
        neuron_obj = unrotation_function(neuron_obj)
        #print(f"neuron_obj.apical_limb_branch_dict = {neuron_obj.apical_limb_branch_dict}")
        
    if apply_synapse_compartment_labels:
        if verbose:
            print(f"Adding the compartment labels to the synapses")
        apu.set_neuron_synapses_compartment(neuron_obj)
        
    return neuron_obj,apu.apical_total_limb_branch_dict(neuron_obj)
        
def compartment_limb_branch_dict(neuron_obj,
                                compartment_labels,
                                 not_matching_labels = None,
                                match_type = "any",
                                **kwargs):
    return nru.label_limb_branch_dict(neuron_obj,compartment_labels,
                                     match_type=match_type,**kwargs)

def dendrite_limb_branch_dict(neuron_obj):
    return apu.compartment_limb_branch_dict(neuron_obj,dendrite_labels)

def axon_limb_branch_dict(neuron_obj):
    return apu.compartment_limb_branch_dict(neuron_obj,"axon")

def apical_total_limb_branch_dict(neuron_obj):
    return apu.compartment_limb_branch_dict(neuron_obj,apical_total)

def apical_limb_branch_dict(neuron_obj):
    return nru.label_limb_branch_dict(neuron_obj,"apical",
                                     not_matching_labels=specific_apicals)

def apical_shaft_limb_branch_dict(neuron_obj):
    return nru.label_limb_branch_dict(neuron_obj,"apical_shaft")

def apical_tuft_limb_branch_dict(neuron_obj):
    return nru.label_limb_branch_dict(neuron_obj,"apical_tuft")

def basal_limb_branch_dict(neuron_obj):
    return nru.label_limb_branch_dict(neuron_obj,"basal")

def oblique_limb_branch_dict(neuron_obj):
    return nru.label_limb_branch_dict(neuron_obj,"oblique")

def basal_classfication(neuron_obj,plot_basal = False,
                        add_labels = True,
                        clear_prior_labels=True,
                       verbose = False):
    """
    Purpose: To identify and label the basal branches
    """
    dendrite_limb_branch = neuron_obj.dendrite_limb_branch_dict
    apical_limb_branch_dict = apu.apical_total_limb_branch_dict(neuron_obj)
    basal_limb_branch_dict = nru.limb_branch_setdiff([dendrite_limb_branch,apical_limb_branch_dict])
    
    if add_labels:
        if clear_prior_labels:
            nru.clear_all_branch_labels(neuron_obj,["basal"])
        nru.add_branch_label(neuron_obj,
                        limb_branch_dict=basal_limb_branch_dict,
                        labels=["basal"])
        
    if verbose:
        print(f"basal_limb_branch_dict = {basal_limb_branch_dict}")
    if plot_basal:
        nviz.plot_limb_branch_dict(neuron_obj,
                                  basal_limb_branch_dict)
    return basal_limb_branch_dict


def apical_tuft_classification(neuron_obj,
                               plot_apical_tuft = False,
                               add_labels = True,
                                clear_prior_labels=True,
                               add_low_degree_apicals_off_shaft = None,#False,
                               low_degree_apicals_min_angle = None,#0,
                            low_degree_apicals_max_angle = None,#40,
                              verbose = False,
                              label = "apical_tuft"):
    """
    Purpose: To classify the apical tuft branches
    based on previous apical shaft and apical classification

    Pseudocode: 
    1) Get all of the nodes of the apical shaft that have no downstream apical shaft branches
    (assemble them into a limb branch)
    2) make all downstream branches of those the apical tuft
    
    Ex: 
    apu.apical_tuft_classification(neuron_obj,
                               plot_apical_tuft = True,
                               add_labels = True,
                                clear_prior_labels=True,
                              verbose = True)
    """
    if add_low_degree_apicals_off_shaft is None:
        add_low_degree_apicals_off_shaft = add_low_degree_apicals_off_shaft_tuft_global
    if low_degree_apicals_min_angle is None:
        low_degree_apicals_min_angle = low_degree_apicals_min_angle_tuft_global
    if low_degree_apicals_max_angle is None:
        low_degree_apicals_max_angle = low_degree_apicals_max_angle_tuft_global
    
    apical_shaft_ending = ns.query_neuron(neuron_obj,
                   functions_list=[
                                  ns.is_apical_shaft_in_downstream_branches,
                                  ],
                    query = "(is_apical_shaft_in_downstream_branches == False)",
                   #function_kwargs=dict(matching_labels=["apical_shaft"]),
                    limb_branch_dict_restriction=neuron_obj.apical_shaft_limb_branch_dict,

                   )
    apical_tuft_limb_branch = nru.all_donwstream_branches_from_limb_branch(neuron_obj,
                                                limb_branch_dict=apical_shaft_ending,
                                                include_limb_branch_dict=False)
    
    if add_low_degree_apicals_off_shaft:
        low_degree_apicals_limb_branch = apu.oblique_classification(
                            neuron_obj,
                            plot_apical_shaft_direct_downstream = False,
                            #arguments for identifying the start of olbiques
                            min_angle = low_degree_apicals_min_angle,
                            max_angle = low_degree_apicals_max_angle,
                            plot_oblique_start = False,
                            plot_oblique = False,

                            add_labels = False,
                            clear_prior_labels=False,
                            verbose = False,
                            label = "apical_tuft")
        if verbose:
            print(f"low_degree_apicals_limb_branch = {low_degree_apicals_limb_branch}")
        if plot_low_degree_apicals_off_shaft:
            print(f"Plotting low_degree_apicals_off_shaft")
            nviz.plot_limb_branch_dict(neuron_obj,
                                      low_degree_apicals_limb_branch)
        apical_tuft_limb_branch = nru.limb_branch_union([apical_tuft_limb_branch,
                                                        low_degree_apicals_limb_branch])

    if verbose:
        print(f"apical_tuft_limb_branch = {apical_tuft_limb_branch}")

    if plot_apical_tuft:
        print(f"Plotting apical tuft")
        nviz.plot_limb_branch_dict(neuron_obj,apical_tuft_limb_branch)

    curr_label = label

    if add_labels:
        if clear_prior_labels:
            nru.clear_all_branch_labels(neuron_obj,[curr_label])
        nru.add_branch_label(neuron_obj,
                        limb_branch_dict=apical_tuft_limb_branch,
                        labels=[curr_label])
        
    return apical_tuft_limb_branch


def oblique_classification(
    neuron_obj,
    plot_apical_shaft_direct_downstream = False,
    #arguments for identifying the start of olbiques
    min_angle = None,#80,
    max_angle = None,#140,
    per_match_ref_vector_min = None,#0.8,
    dist_match_ref_vector_min = None,#10000,
    plot_oblique_start = False,
    plot_oblique = False,
    
    add_labels = True,
    clear_prior_labels=True,
    verbose = False,
    label = "oblique"):
    """
    Purpose: To find the branches
    that come of the apical shaft at a certain degree

    Pseudocode: 
    1) Get the apical_shaft_direct_downstream
    2) Filter those branches for those with a certain angle
    3) Find all branches downstream of those with a certain 
    angle as the oblique branches
    """

    if min_angle is None:
        min_angle = min_angle_oblique_global
    if max_angle is None:
        max_angle = max_angle_oblique_global
    if per_match_ref_vector_min is None:
        per_match_ref_vector_min = per_match_ref_vector_min_oblique_global
    if dist_match_ref_vector_min is None:
        dist_match_ref_vector_min = dist_match_ref_vector_min_oblique_global



    #1) Get the apical_shaft_direct_downstream
    shaft_direct = apu.apical_shaft_direct_downstream(neuron_obj,
                                                     plot_limb_branch=plot_apical_shaft_direct_downstream)
    shaft_direct_no_tuft = nru.limb_branch_setdiff([shaft_direct,neuron_obj.apical_tuft_limb_branch_dict])
    
    if verbose:
        print(f"shaft_direct = {shaft_direct}")

    #2) Filter those branches for those with a certain angle
    oblique_start_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list=[ns.skeleton_dist_match_ref_vector,
                                  ns.skeleton_perc_match_ref_vector,],
                    query=(f"(skeleton_dist_match_ref_vector > {dist_match_ref_vector_min}) or "
                           f"(skeleton_perc_match_ref_vector > {per_match_ref_vector_min})"
                          ),
                   function_kwargs=dict(max_angle=max_angle,
                                       min_angle=min_angle),
                                               limb_branch_dict_restriction=shaft_direct_no_tuft)

    if verbose:
        print(f"oblique_start_limb_branch = {oblique_start_limb_branch}")

    if plot_oblique_start:
        print(f"Plotting oblique start")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  oblique_start_limb_branch)

    #3) Find all branches downstream of those with a certain 
    oblique_limb_branch = nru.all_donwstream_branches_from_limb_branch(neuron_obj,
                                                 oblique_start_limb_branch)

    if verbose:
        print(f"oblique_limb_branch = {oblique_limb_branch}")

    if plot_oblique:
        print(f"Plotting final oblique")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  oblique_limb_branch)
    
    curr_label = label

    if add_labels:
        if clear_prior_labels:
            nru.clear_all_branch_labels(neuron_obj,[curr_label])
        nru.add_branch_label(neuron_obj,
                        limb_branch_dict=oblique_limb_branch,
                        labels=[curr_label])
        
    return oblique_limb_branch

def apical_classification_high_soma_center(
    neuron_obj,
    
    #determing the possible apical limbs
    possible_apical_limbs = None,
    soma_angle_max=None,
                                          
    plot_filtered_limbs = False,
    width_min = 450,
    distance_from_soma = 80000,
    plot_thick_apical_candidates = False,

    #for filtering the candidate limbs 
    min_thick_near_soma_skeletal_length = 10000,

    plot_final_apical = False,
    verbose = False):
    """
    Purpose: To identify multiple
    possible apicals that are at the 
    top of the soma mesh

    Pseudocode: 
    1) If necessary, filter the limbs
    2) ON the limbs find the number of fat limbs within certain radius of soma
    
    Ex: 
    apu.apical_classification_high_soma_center(n_obj_1,
                                           verbose = True,
                                          plot_final_apical=True)

    """
    if soma_angle_max is None:
        soma_angle_max = soma_angle_to_apical_global
        
    if width_min is None:
        width_min = width_min_apical_high_soma_global
    if distance_from_soma is None:
        distance_from_soma = distance_from_soma_apical_high_soma_global
    if min_thick_near_soma_skeletal_length is None:
        min_thick_near_soma_skeletal_length = min_thick_near_soma_skeletal_length_apical_high_soma_global
    
    #1) Filter the limbs that are being considered to have the shaft
    if possible_apical_limbs is None:
        if soma_angle_max is not None:
            possible_apical_limbs = nst.filter_limbs_by_soma_starting_angle(neuron_obj,
                                                    soma_angle = soma_angle_max,
                                                    angle_less_than = True,
                                                    verbose=verbose,
                                                   return_int_names=True)
        else:
            possible_apical_limbs = neuron_obj.get_limb_names(return_int=True)
        if verbose: 
            print(f'\nPart 0: possible_apical_limbs = {possible_apical_limbs}')

    if plot_filtered_limbs:
        print(f"Plotting filtered POSSIBLE apical limbs")
        nviz.visualize_subset_neuron_limbs(neuron_obj,possible_apical_limbs)


    # query for large fat branches near
    thick_apical_limb_branch = ns.query_neuron(neuron_obj,
                    query=(f"(distance_from_soma < {distance_from_soma}) and "
                          f"(width_new > {width_min})"),
                    functions_list= [ns.distance_from_soma,
                                      ns.width_new],
                   limbs_to_process=possible_apical_limbs,
                   plot_limb_branch_dict=plot_thick_apical_candidates)
    
    if verbose:
        print(f"thick_apical_limb_branch = {thick_apical_limb_branch}")


    final_apical_limbs = []
    for limb_idx in possible_apical_limbs:
        limb_name = nru.get_limb_string_name(limb_idx)
        
        if limb_name in thick_apical_limb_branch.keys():
            curr_sk_length = cnu.sum_feature_over_branches(neuron_obj[limb_name],
                                         thick_apical_limb_branch[limb_name],
                                         feature_name = "skeletal_length")
            if curr_sk_length > min_thick_near_soma_skeletal_length:
                
                final_apical_limbs.append(limb_name)

    final_apical_limb_branch = nru.limb_branch_from_limbs(neuron_obj,
                          final_apical_limbs)

    if verbose:
        print(f"final_apical_limb_branch = {final_apical_limb_branch}")

    if plot_final_apical:
        print(f"Plotting final apical from limb angle approach")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  final_apical_limb_branch)
        
    return final_apical_limb_branch


def compartment_label_from_branch_obj(branch_obj,
                                      label_order = default_compartment_order,
                                      default_label = default_compartment_label,
                                     verbose = False,):
    """
    Purpose: To add compartment
    labels to all the synapses of a branch
    based on the branch labels

    Pseudocode: 
    0) Define order of labels to check for
    For each label
    1) check to see if label is in the labels
    of branch
    2a) if so then add the numbered label to 
    all the synapses in the branch and break
    2b) if not then continue to the next label

    Ex: 
    apu.compartment_label_from_branch_obj(branch_obj = neuron_obj[0][0],
verbose = True)
    """


    curr_label = None
    for l in label_order:
        if l in branch_obj.labels:
            curr_label = l
            break
    if curr_label is None:
        curr_label = default_label

    if verbose:
        print(f"Found label = {curr_label}")

    return curr_label

def set_neuron_synapses_compartment(neuron_obj,**kwargs):
    """
    Purpose: Will set the compartment labels of all synapses
    based on the compartment label of te branch
    
    """
    syu.set_neuron_synapses_compartment(neuron_obj,**kwargs)
    
def compartment_classification_by_cell_type(neuron_obj,
                              cell_type,
                              verbose = False,
                              plot_compartments=False,
                              apply_synapse_compartment_labels = True,
                                **kwargs):
    """
    Purpose: Will label a neuron by the compartments
    based on the cell type
    
    
    """
    
    if cell_type == "excitatory":
        if verbose:
            print(f"Running apical classification because excitatory cell ")
        neuron_obj,_ = apu.apical_classification(neuron_obj,
                                 verbose = verbose,
                                 apply_synapse_compartment_labels = False,
                                 **kwargs)
    elif cell_type == "inhibitory":
        if verbose:
            print(f"Not running any new classification because inhibitory cell")
            
        nru.add_branch_label(neuron_obj,
                        limb_branch_dict=neuron_obj.dendrite_limb_branch_dict,
                        labels=["dendrite"])
        
    else:
        raise Exception(f"Unknown cell type = {cell_type}")
        
    if apply_synapse_compartment_labels:
        if verbose:
            print(f"Adding the compartment labels to the synapses")
        apu.set_neuron_synapses_compartment(neuron_obj)
    
    if plot_compartments: 
        nviz.plot_compartments(neuron_obj)
        
    return neuron_obj


def compartments_stats(neuron_obj,
                     compartment_labels = None,
                     verbose = False):
    """
    Purpose: To compute statistics for all the compartments
    of a neuron_obj
    
    Pseudocode: 
    For each compartment label:
    1) Get the limb branch dict
    2) Get the stats over that limb branch
    3) Add to larger dict with modified names
    
    Ex: 
    apu.compartment_stats(neuron_obj_proof,
                     compartment_labels = None,
                     verbose = True) 
    """
    if compartment_labels is None:  
        compartment_labels = apu.compartment_labels_for_stats()
    
    stats_dict = dict()
    for c in compartment_labels:
        if verbose:
            print(f"\n-- Working on {c}")
            
        func = getattr(apu,f"{c}_limb_branch_dict")
        comp_limb_branch = func(neuron_obj)
        
        local_stats = nst.stats_dict_over_limb_branch(
            neuron_obj = neuron_obj,
            limb_branch_dict = comp_limb_branch)
        
        if verbose:
            print(f"local_stats = {local_stats}")
        stats_dict.update({f"{c}_{k}":v for k,v in local_stats.items()})
        
    return stats_dict

def compartment_feature_over_limb_branch_dict(
    neuron_obj,
    compartment_label,
    feature_for_sum = None,
    feature_func = None,
    verbose = False):
    """
    To compute a certain feature over ONE compartment
    
    """
    c = compartment_label
    func = getattr(apu,f"{c}_limb_branch_dict")
    comp_limb_branch = func(neuron_obj)

    if feature_for_sum is not None:
        local_val = nru.sum_feature_over_limb_branch_dict(neuron_obj,feature_for_sum)
    elif feature_func is not None:
        local_val = feature_func(neuron_obj,comp_limb_branch)
    else:
        raise Exception("")
        
    return local_val

def compartments_feature_over_limb_branch_dict(
    neuron_obj,
    compartment_labels = None,
    feature_for_sum = None,
    feature_func = None,
    feature_name=None,
    verbose = False):
    """
    Purpose: To compute statistics for all the compartments
    of a neuron_obj
    
    Pseudocode: 
    For each compartment label:
    1) Get the limb branch dict
    2) Get the stats over that limb branch
    3) Add to larger dict with modified names
    
    Ex: 
    apu.compartment_stats(neuron_obj_proof,
                     compartment_labels = None,
                     verbose = True) 
    """
    spu.set_soma_synapses_spine_label(neuron_obj)
    
    if compartment_labels is None:  
        compartment_labels = apu.compartment_labels_for_stats()
    
        
    global_dict = dict()
    for c in compartment_labels:
        if verbose:
            print(f"\n-- Working on {c}")
        
        
        
        local_val = apu.compartment_feature_over_limb_branch_dict(
                         neuron_obj,
                         c,
                         feature_for_sum = feature_for_sum,
                         feature_func = feature_func,)
            
        dict_name = f"{c}"
        if feature_name is not None:
            dict_name += f"_{feature_name}"
            
        global_dict.update({dict_name:local_val})
    return global_dict

def compartments_skeleton(neuron_obj,
                         compartment_labels=None,
                         verbose = False):
    
    if compartment_labels is None:  
        compartment_labels = compartment_labels_for_externals()
    
    return compartments_feature_over_limb_branch_dict(neuron_obj,
                     compartment_labels = compartment_labels,
                         feature_func = nru.skeleton_over_limb_branch_dict,
                         feature_name = "skeleton",
                     verbose = False)

def compartment_skeleton(neuron_obj,
                        compartment_label):
    
    return apu.compartment_feature_over_limb_branch_dict(
                         neuron_obj,
                         compartment_label,
                         feature_func = nru.skeleton_over_limb_branch_dict,)
    
    

def compartments_mesh(neuron_obj,
                         compartment_labels=None,
                         verbose = False):
    
    if compartment_labels is None:  
        compartment_labels = compartment_labels_for_externals()
    
    return compartments_feature_over_limb_branch_dict(neuron_obj,
                     compartment_labels = compartment_labels,
                         feature_func = nru.mesh_over_limb_branch_dict,
                         feature_name = "mesh",
                     verbose = False)

def compartment_mesh(neuron_obj,
                        compartment_label):
    
    return apu.compartment_feature_over_limb_branch_dict(
                         neuron_obj,
                         compartment_label,
                         feature_func = nru.mesh_over_limb_branch_dict,)


def compartment_features_from_skeleton_and_soma_center(neuron_obj,
                                                      compartment_label,
                                                       features_to_exclude = ("length","n_branches"),
                                                       soma_label = "S0",
                                                       soma_center = None,
                                                       name_prefix = None,
                                                       include_soma_starting_angles = True,
                                                       neuron_obj_aligned = None,
                                                      **kwargs):
    """
    Purpose: Will compute features about a compartment
    from its skeleton and the skeleton in relation to the soma
    
    Ex: 
    apu.compartment_features_from_skeleton_and_soma_center(neuron_obj_proof,
                                                  compartment_label = "oblique")
    """

    if name_prefix is None:
        name_prefix = compartment_label
    if soma_center is None:
        soma_center = neuron_obj[soma_label].mesh_center
        
    if neuron_obj_aligned is not None:
        skeleton_aligned = neuron_obj_aligned.skeleton
        soma_center = neuron_obj_aligned["S0"].mesh_center
    else:
        skeleton_aligned = None
    compartment_skeleton = apu.compartment_skeleton(neuron_obj,compartment_label)
    return nst.features_from_skeleton_and_soma_center(compartment_skeleton,
                                              soma_center = soma_center,
                                              name_prefix=name_prefix,
                                              features_to_exclude = features_to_exclude,
                                                      skeleton_aligned=skeleton_aligned,
                                              **kwargs)

def print_compartment_features_dict_for_dj_table(comp_feature_dict):
    int_unsigned_group = ["n_branches",
                         "n_short_branches",
                         "n_long_branches",
                         "n_medium_branches"]
    for k in comp_feature_dict:
        int_unsigned_flag = False
        for kk in int_unsigned_group:
            if kk in k:
                int_unsigned_flag = True
                break
                
        if int_unsigned_flag:
            data_type = "int unsigned"
        else:
            data_type = "double"
            
        print(f"{k}=NULL: {data_type}")

def plot_compartment_mesh_and_skeleton(neuron_obj,compartment_label):
    """
    Ex: 
    apu.plot_compartment_mesh_and_skeleton(neuron_obj_proof,"basal")
    """
    comp_mesh = apu.compartment_mesh(neuron_obj,compartment_label)
    comp_sk = apu.compartment_skeleton(neuron_obj,compartment_label)
    
    if len(comp_mesh.faces) > 0 and len(comp_sk) > 0:
        nviz.plot_objects(comp_mesh,comp_sk,
                         meshes=[neuron_obj["S0"].mesh])
    else:
        print(f"No mesh and/or skeleton for this compartment")
    

    
def soma_angle_extrema_from_compartment(
    neuron_obj,
    compartment_label = None,
    compartment_limb_branch = None,
    default_value = None,
    extrema_type = "min",
    verbose = False,
    ):
    """
    Purpose: Find the max or min
    soma starting angle for all limbs with that compartment
    """


    if compartment_limb_branch is None:
        func = getattr(apu,f"{compartment_label}_limb_branch_dict")
        compartment_limb_branch = func(neuron_obj)


    return_value = default_value


    if len(compartment_limb_branch) > 0:
        limb_angles = [nst.soma_starting_angle(neuron_obj=neuron_obj,limb_idx=k)
                      for k in compartment_limb_branch.keys()]

        return_value = getattr(np,extrema_type)(limb_angles)
        if verbose:
            print(f"limb_angles = {limb_angles}")
            print(f"{extrema_type} = {return_value}")
    else:
        if verbose:
            print(f"No {compartment_label} limb branch")

    return return_value

def soma_angle_min_from_compartment(
    neuron_obj,
    compartment_label = None,
    compartment_limb_branch = None,
    default_value = None,
    verbose = False,
    ):
    return soma_angle_extrema_from_compartment(
    neuron_obj,
    compartment_label = compartment_label,
    compartment_limb_branch = compartment_limb_branch,
    default_value = default_value,
    extrema_type = "min",
        verbose = verbose
    )

def soma_angle_max_from_compartment(
    neuron_obj,
    compartment_label = None,
    compartment_limb_branch = None,
    default_value = None,
    verbose = False,
    ):
    return soma_angle_extrema_from_compartment(
    neuron_obj,
    compartment_label = compartment_label,
    compartment_limb_branch = compartment_limb_branch,
    default_value = default_value,
    extrema_type = "max",
        verbose = verbose
    )

def limb_features_from_compartment(
    neuron_obj,
    compartment_label = None,
    compartment_limb_branch=None,
    verbose = False,
    rotation_function = None,
    apply_rotation = True,
    **kwargs):
    """
    Purpose: To compute limb features that depend on alignment of neuron

    Pseudocode: 
    1) Align neuron
    2) Get the compartment limb branch dict
    
    Ex: 
    apu.limb_features_from_compartment(
        neuron_obj,
        compartment_limb_branch=neuron_obj.dendrite_limb_branch_dict,
        compartment_label="axon",
        apply_rotation=True,
    )

    """
    if rotation_function is None:
        rotation_function = rotation_function_axon_alignment
    
    if rotation_function is not None and apply_rotation:
        neuron_obj = rotation_function(neuron_obj)
        
    if compartment_limb_branch is None:
        func = getattr(apu,f"{compartment_label}_limb_branch_dict")
        compartment_limb_branch = func(neuron_obj)

    return_dict = dict(
        n_limbs = len(compartment_limb_branch.keys()),
        soma_angle_max = apu.soma_angle_max_from_compartment(
            neuron_obj,
            compartment_limb_branch=compartment_limb_branch,
            ),
        soma_angle_min = apu.soma_angle_min_from_compartment(
            neuron_obj,
            compartment_limb_branch=compartment_limb_branch,
            )
    )
    
    return return_dict

def limb_features_from_compartment_over_neuron(
    neuron_obj,
    compartments = ("basal","apical_total","axon","dendrite"),
    rotation_function = None,
    verbose = False
    ):
    """
    Purpose: To run limb features for overview compartments
    """
    if rotation_function is None:
        rotation_function = rotation_function_axon_alignment
    
    if rotation_function is not None:
        neuron_obj = rotation_function(neuron_obj)
        
    total_limb_dict = dict()
    for c in compartments:
        if verbose:
            print(f"\nWorking on compartment {c}---")
        if c == "apical_total":
            comp_name = "apical"
        else:
            comp_name = c
            
        curr_dict = apu.limb_features_from_compartment(
            neuron_obj,
            compartment_label = c,
            apply_rotation = False
        )
        
        if verbose:
            print(f"Compartment limb stats = {curr_dict}")
            
        total_limb_dict.update({f"{comp_name}_{k}":v for k,v in curr_dict.items()})
    return total_limb_dict


def compartment_from_face_overlap_with_comp_faces_dict(
    mesh_face_idx,
    comp_faces_dict,
    default_value = None,
    verbose = False,
    ):
    """
    Purpose: Want to find the compartment of a branch
    if we know the faces index for a reference mesh
    and the compartments for a reference mesh

    Pseudocode:
    Iterate through all compartments in compartment dict
    1) Find the overlap between faces and compartment 
    2) if the overlap is greater than 0 and 
    greater than current max then set as compartment
    
    Ex: 
    from neurd_packages import neuron_utils as nru
    neuron_obj = hdju.neuron_objs_from_cell_type_stage(segment_id)

    decimated_mesh = hdju.fetch_segment_id_mesh(segment_id)
    proofread_faces = hdju.fetch_proofread_neuron_faces(segment_id,split_index = split_index)
    limb_branch_dict = None

    limb_branch_face_dict = nru.limb_branch_face_idx_dict_from_neuron_obj_overlap_with_face_idx_on_reference_mesh(
        neuron_obj,
        faces_idx = proofread_faces,
        mesh_reference = decimated_mesh,
        limb_branch_dict = limb_branch_dict,
        verbose = False
    )

    comp_faces_dict = hdju.compartment_faces_dict(segment_id,verbose=False)

    apu.compartment_from_face_overlap_with_comp_faces_dict(
        mesh_face_idx = limb_branch_face_dict["L0"][2],
        comp_faces_dict = comp_faces_dict,
        verbose = True
    )
    """

    max_faces = 0
    curr_comp = default_value
    for comp,comp_faces in comp_faces_dict.items():
        overlap = np.intersect1d(comp_faces,mesh_face_idx)
        if len(overlap) > max_faces:
            if verbose:
                print(f"Changing compartment to {comp} because {len(overlap)}/{len(mesh_face_idx)} faces overlap (greater than current max value of {max_faces})")
            curr_comp = comp

    if verbose:
        print(f"Final compartment = {curr_comp}")

    return curr_comp

def limb_branch_compartment_dict_from_limb_branch_face_and_compartment_faces_dict(
    limb_branch_face_dict,
    compartment_faces_dict,
    verbose = False,
    ):
    limb_branch_compartment_dict = dict()
    for limb_idx,branch_info in limb_branch_face_dict.items():
        if verbose:
            print(f" --> limb {limb_idx}")
        if limb_idx not in limb_branch_compartment_dict:
            limb_branch_compartment_dict[limb_idx] = dict()
        for b_idx in branch_info:
            if verbose:
                print(f" --> branch {b_idx}")
            limb_branch_compartment_dict[limb_idx][b_idx] = apu.compartment_from_face_overlap_with_comp_faces_dict(
                mesh_face_idx = limb_branch_face_dict[limb_idx][b_idx],
                comp_faces_dict = compartment_faces_dict,
                verbose = verbose
            )

    return limb_branch_compartment_dict

def max_height_for_multi_soma():
    return -1*multi_apical_height_global
multi_soma_y = max_height_for_multi_soma


# ------------- parameters for stats ---------------

global_parameters_dict_default_apical = dict(
    #apical parameters
    soma_angle_to_apical = 60,
    multi_apical_height = -460_000,
    
    # apical_shaft_like
    max_upward_angle_shaft_like = 30,
    min_upward_length_shaft_like = 3000,
    min_upward_per_match_shaft_like = 0.8,
    min_upward_length_backup_shaft_like = 20000, 
    min_upward_per_match_backup_shaft_like = 0.5,
    width_min_shaft_like = 140,
    
    # apical_filter
    min_skeletal_length_filter_apical = 10_000,#30000,#50000,
    min_distance_above_soma_filter_apical = 10_000,#30000,#100000,
    
    #apical_classification
    candidate_connected_component_radius_apical = 5000,
    multi_apical_possible_apical = True,
    
    #apical_classification_high_soma_center
    width_min_apical_high_soma = 450,
    distance_from_soma_apical_high_soma = 80000,
    min_thick_near_soma_skeletal_length_apical_high_soma = 10000,
    
    #filter_apical_candidates_to_one
    non_upward_skeletal_distance_upstream_buffer_filter_apical_one = -10000,
    soma_diff_buffer_filter_apical_one = -50000,
    downstream_vector_diff_buffer_filter_apical_one = -30000,
    default_tie_breaker_filter_apical_one = "skeletal_length",
    
    #apical_tuft_classification
    add_low_degree_apicals_off_shaft_tuft = False,
    low_degree_apicals_min_angle_tuft = 0,
    low_degree_apicals_max_angle_tuft = 40,
    
    
)

global_parameters_dict_default_oblique = dict(
    #apical_tuft_classification
    min_angle_oblique = 80,
    max_angle_oblique = 140,
    per_match_ref_vector_min_oblique = 0.8,
    dist_match_ref_vector_min_oblique = 10000,
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_apical,
    global_parameters_dict_default_oblique,
])


attributes_dict_default = dict(
    rotation_function_axon_alignment = None,
    unrotation_function_axon_alignment = None,
)    

# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01_apical = dict(
    multi_apical_possible_apical = False,
)

global_parameters_dict_h01 = gu.merge_dicts([
        global_parameters_dict_h01_apical
])

from . import h01_volume_utils as hvu
attributes_dict_h01 = dict(
    rotation_function_axon_alignment = hvu.data_interface.align_neuron_obj,
    unrotation_function_axon_alignment =  hvu.data_interface.unalign_neuron_obj,
)

# modules_to_set = [apu]
# data_type = "default"
# algorithms = None

# modsetter = modu.ModuleDataTypeSetter(
#     module = modules_to_set,
#     algorithms = algorithms
# )

# set_global_parameters_and_attributes_by_data_type = modsetter.set_global_parameters_and_attributes_by_data_type
# output_global_parameters_and_attributes_from_current_data_type = modsetter.output_global_parameters_and_attributes_from_current_data_type

# set_global_parameters_and_attributes_by_data_type(
#     data_type=data_type,
#     algorithms=algorithms
# )


#--- from neurd_packages ---
from . import concept_network_utils as cnu
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mcu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import spine_utils as spu
from . import synapse_utils as syu

#--- from python_tools ---
from python_tools import filtering_utils as flu
from python_tools import general_utils as gu
from python_tools import ipyvolume_utils as ipvu
from python_tools import module_utils as modu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu

from . import apical_utils as apu