"""
Functions that outline the pipeline of taking
a split neuron all the way to the autoproofreading 
and compartment labeling stage
"""

import time


def cell_type_ax_dendr_stage(
    neuron_obj,
    mesh_decimated,
    store_in_obj = True,
    return_stage_products = False,
    verbose = False,

    plot_initial_neuron = False,

    plot_floating_end_nodes_limb_branch_dict =  False,
    plot_downstream_path_limb_branch = False,
    plot_after_simplification = False,

    filter_low_branch_cluster_dendrite = False,
    plot_limb_branch_filter_away_low_branch = False,

    # -- adding the synapses and spines
    plot_synapses = False,
    segment_id = None,
    synapse_filepath = None,
    plot_spines = False,


    plot_spines_and_sk_filter_for_syn = False,
    plot_spines_and_sk_filter_for_spine = False,

    inh_exc_class_to_use_for_axon = "neurd", #or external

    plot_aligned_neuron_with_syn_sp = False,


    filter_dendrite_on_axon = False,
    plot_initial_axon = False,
    plot_axon_on_dendrite = False,
    plot_high_fidelity_axon = False,
    plot_boutons_web = False,

    plot_axon = False,
    
    ):
    """
    Purpose: to preprocess the split neuron object
    prior to autoproofreading by performing: 
    
    1) Branch simplification (making sure all branches have at least 2 children)
    2) Filtering away large clusters of glia still present 
    3) match neuron to nucleus
    4) Add synapses to neuron 
    5) Divide the spines into head,neck compartments
    6) Perform cell typing based on spine and synapse statistics
    6b) Optional: download cell type from database which may be the cell type you choose
    7) Label the axon
    8) Package up all of the products/statistcs generated
    """
    neuron_obj_pre_filt = neuron_obj
    
    if segment_id is None:
        segment_id = neuron_obj.segment_id
    
    if plot_initial_neuron:
        neuron_obj_rot = vdi.align_neuron_obj(neuron_obj_pre_filt)
        nviz.visualize_neuron(neuron_obj_rot,limb_branch_dict = "all")
        
    
    st = time.time()
    
    
    
    #1) Branch Simplification
    bu.refine_width_array_to_match_skeletal_coordinates(
        neuron_obj_pre_filt,
        verbose = False
    )
    
    # 2) Performs branch simplification so there should always be 2 or more child branches
    neuron_obj_pre_filt_after_simp = nsimp.branching_simplification(
        neuron_obj_pre_filt,
        return_copy = True,

        #floating endpiece arguments
        plot_floating_end_nodes_limb_branch_dict = plot_floating_end_nodes_limb_branch_dict,
        plot_final_neuron_floating_endpoints = False,

        # combine path arguments
        plot_downstream_path_limb_branch = plot_downstream_path_limb_branch,
        plot_final_neuron_path = False,
        plot_after_simplification = False,
        verbose = verbose,
    )
    
    
    # 4) Filter away a large cluster of dendrite if requested SKIPPED

    if filter_low_branch_cluster_dendrite:
        neuron_obj_pre_class, filtering_info_low_branch = pru.apply_proofreading_filters_to_neuron(
            input_neuron = neuron_obj_pre_filt_after_simp,
            filter_list = [pru.low_branch_length_clusters_dendrite_filter],
            plot_limb_branch_filter_with_disconnect_effect=False,
            plot_limb_branch_filter_away=plot_limb_branch_filter_away_low_branch,
            plot_final_neuron=False,

            return_error_info=True,
            verbose=False,
            verbose_outline=verbose)
    else:
        neuron_obj_pre_class = neuron_obj_pre_filt_after_simp
        filtering_info_low_branch = {}
    
    
    # 3) match neuron to nucleus
    nucleus_ids,nucleus_centers = vdi.nuclei_from_segment_id(
        segment_id,
        return_centers=True,
        return_nm=True
        )

    if verbose:
        if nucleus_ids is not None:
            print(f"Number of Corresponding Nuclei = {len(nucleus_ids)}")
            print(f"nucleus_ids = {nucleus_ids}")
            print(f"nucleus_centers = {nucleus_centers}")
            
    winning_nucleus_id, nucleus_info = nru.pair_neuron_obj_to_nuclei(
        neuron_obj_pre_class,
        "S0",
        nucleus_ids,
        nucleus_centers,
        return_matching_info = True,
        verbose=True)

    if verbose:
        print(f"nucleus_info = {nucleus_info}")
        print(f"winning_nucleus_id = {winning_nucleus_id}")
        
        
    # 4) Add synapses to neuron 
    neuron_obj_attr = syu.add_synapses_to_neuron_obj(
        neuron_obj_pre_class,
        segment_id = segment_id,
        synapse_filepath = synapse_filepath,
        verbose  = verbose,
        original_mesh = mesh_decimated,
        plot_valid_error_synapses = False,
        calculate_synapse_soma_distance = False,
        add_valid_synapses = True,
        add_error_synapses=False,
    )

    if plot_synapses:
        syu.plot_synapses(neuron_obj_attr)
    
    
    
    # 5) Divide the spines into head,neck compartments
    neuron_obj_attr = spu.add_head_neck_shaft_spine_objs(
        neuron_obj_attr,
        verbose = verbose
    )

    if plot_spines:
        spu.plot_spines_head_neck(neuron_obj_attr)
        
        
    if plot_aligned_neuron_with_syn_sp:
        print(f"plot_aligned_neuron")
        neuron_obj_rot = vdi.align_neuron_obj(neuron_obj_attr)
        nviz.visualize_neuron(neuron_obj_rot,limb_branch_dict="all")
    
    # 6) Perform cell typing based on spine and synapse statistics
    baylor_e_i,baylor_cell_type_info = ctu.e_i_classification_from_neuron_obj(
        neuron_obj_attr,
        plot_on_model_map=False,
        plot_spines_and_sk_filter_for_syn = plot_spines_and_sk_filter_for_syn,
        plot_spines_and_sk_filter_for_spine = plot_spines_and_sk_filter_for_spine,
        verbose = verbose,
        return_cell_type_info = True
    )

    baylor_cell_type_info["baylor_cell_type"] = baylor_e_i 

    if verbose:
        print(f"baylor_cell_type_info = \n{baylor_cell_type_info}")
    
    
    # 6b) Getting the cell types from the database
    database_cell_type_info = vdi.nuclei_classification_info_from_nucleus_id(
        winning_nucleus_id
    )

    database_e_i_class = database_cell_type_info[f"external_cell_type"]  

    if verbose:
        print(f"database_cell_type_info = {database_cell_type_info}")
        print(f"database_e_i_class = {database_e_i_class}")
    
    #--- Pick the cell type to use
    if (inh_exc_class_to_use_for_axon == "external"
        and database_e_i_class in ["excitatory","inhibitory"]):
        e_i_class  = database_e_i_class
        if verbose:
            print(f"Using external e/i cell type")

        cell_type_used = "external"
    else:
        if verbose:
            print(f"Using neurd e/i cell type")
        e_i_class = baylor_e_i
        cell_type_used = "neurd"

    if verbose:
        print(f"database_e_i_class = {database_e_i_class}")
        print(f"e_i_class = {e_i_class} with cell_type_used = {cell_type_used}")
    
    
    
    
    # 7) Label the axon
    (o_neuron_unalign,
    filtering_info,
    axon_angles_dict,
    G_axon_labeled,)=au.complete_axon_processing(
                neuron_obj_attr,
                cell_type = e_i_class,
                add_synapses_and_head_neck_shaft_spines = False,
                validation = False,
                plot_initial_axon=plot_initial_axon,
                plot_axon_on_dendrite=plot_axon_on_dendrite,
                return_filtering_info = True,
                return_axon_angle_info = True,
                plot_high_fidelity_axon = plot_high_fidelity_axon,
                plot_boutons_web = plot_boutons_web,
                add_synapses_after_high_fidelity_axon = True,
                filter_dendrite_on_axon = filter_dendrite_on_axon,
                return_G_axon_labeled = True,
                verbose = verbose)

    if plot_axon:
        nviz.plot_axon(o_neuron_unalign)
            
        
    #8) Package up all of the products/statistcs generated
    
    # --- h) Get the axon and dendrite stats ----
    dendrite_stats = nst.skeleton_stats_dendrite(o_neuron_unalign,
                    include_centroids=False)
    axon_stats = nst.skeleton_stats_axon(o_neuron_unalign,
                                        include_centroids=False)
    stats_dict = o_neuron_unalign.neuron_stats(stats_to_ignore = [
        "n_not_processed_soma_containing_meshes",
        "n_error_limbs",
        "n_same_soma_multi_touching_limbs",
        "n_multi_soma_touching_limbs",
        "n_somas",
        "spine_density"
        ],
                include_skeletal_stats = False,
                include_centroids= True,
                voxel_adjustment_vector=vdi.voxel_to_nm_scaling,

            )

    #---- i) Calculating the synapse info ------
    syn_dict = syu.n_synapses_analysis_axon_dendrite(
        o_neuron_unalign,
        verbose = verbose
    )

    axon_skeleton = o_neuron_unalign.axon_skeleton

    dendrite_skeleton = o_neuron_unalign.dendrite_skeleton
    G_after_axon = ctcu.G_with_attrs_from_neuron_obj(o_neuron_unalign,plot_G=False)
    
    
    run_time = time.time() - st

    n_dict = dict(
        neuron_graph_axon_labeled = G_axon_labeled,
        neuron_graph_high_fid_axon = G_after_axon,
        axon_skeleton = axon_skeleton,
        dendrite_skeleton = dendrite_skeleton,

        #--- cell types
        external_cell_type = database_e_i_class,
        cell_type = e_i_class,
        cell_type_used=cell_type_used,

        #----- synapses ---
        n_syn_pre = o_neuron_unalign.n_synapses_pre,
        n_syn_post= o_neuron_unalign.n_synapses_post,

        run_time = run_time,

        # statistics for the split
        )


    dicts_for_update = [
    nucleus_info,
    database_cell_type_info,
    filtering_info,
    axon_angles_dict,
    dendrite_stats,
    axon_stats,
    stats_dict,
    baylor_cell_type_info,
    filtering_info_low_branch,
    syn_dict]



    for d in dicts_for_update:
        n_dict.update(d)

    cell_type_products = pipeline.StageProducts(
        n_dict,
    )
    
    if store_in_obj:
        o_neuron_unalign.pipeline_products.set_stage_attrs(
            cell_type_products,
            stage = "cell_type_ax_dendr",
        )
        
        
    if return_stage_products:
        return o_neuron_unalign,cell_type_products
    return o_neuron_unalign

# ----------------------
from . import microns_volume_utils as mvu
attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling,
    vdi = mvu.data_interface
)   
    
from python_tools import pipeline

from . import branch_utils as bu
from . import neuron_simplification as nsimp
from . import neuron_utils as nru
from . import synapse_utils as syu
from . import synapse_utils as syu
from . import spine_utils as spu
from . import cell_type_utils as ctu
from . import axon_utils as au
from . import neuron_statistics as nst
from . import cell_type_conv_utils as ctcu
from . import neuron_visualizations as nviz
