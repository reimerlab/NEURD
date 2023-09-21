"""
Functions that outline the pipeline of taking
a split neuron all the way to the autoproofreading 
and compartment labeling stage
"""

import time
import numpy as np

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
                original_mesh = mesh_decimated,
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

# -------- auto proofreading -----------

def after_auto_proof_stats(
    neuron_obj,
    verbose = False,
    store_in_obj = True,
    ):
    neuron_obj_proof = neuron_obj
    
    dicts_to_update = []

    cell_type = vdi.cell_type(neuron_obj)
    nucleus_id = vdi.nucleus_id(neuron_obj)
    
    filtering_info = neuron_obj.filtering_info

    limb_branch_to_cancel = pru.extract_from_filter_info(
        filtering_info,
        name_to_extract="limb_branch_dict_to_cancel"
    )

    red_blue_suggestions = pru.extract_from_filter_info(
        filtering_info,
        name_to_extract = "red_blue_suggestions"
    )

    split_locations = pru.extract_from_filter_info(
        filtering_info,
        name_to_extract = "split_locations",
        name_must_be_ending = True,)
    split_locations_before_filter = pru.extract_from_filter_info(
        filtering_info,
        name_to_extract = "split_locations_before_filter",
        name_must_be_ending = True
    )

    filter_key = {k:np.round(v,2) for k,v in filtering_info.items() if "area" in k or "length" in k}

    G_after_proof = ctcu.G_with_attrs_from_neuron_obj(
        neuron_obj_proof,
        plot_G=False,
        #neuron_obj_attributes_dict = hdju.neuron_graph_attributes_to_set(segment_id,split_index),
    )

    neuron_objs_dict = dict(
            proof_version = vdi.proof_version,
            limb_branch_to_cancel=limb_branch_to_cancel,
            red_blue_suggestions=red_blue_suggestions,
            split_locations = split_locations,
            split_locations_before_filter=split_locations_before_filter,
            synapse_filepath = vdi.synapse_filepath,
            neuron_graph_after_proof = G_after_proof,
        )

    neuron_objs_dict.update(filter_key)

    dicts_to_update.append(neuron_objs_dict)



    # ---- 5) Neuron Statistcs -----
    if verbose:
        print(f"\n--5a) Neuron basics")


    multiplicity = vdi.multiplicity(neuron_obj)
    soma_x,soma_y,soma_z = nru.soma_centers(neuron_obj_proof,
                                        soma_name="S0",
                                        voxel_adjustment=True)
    basic_cell_dict = dict(multiplicity=multiplicity,
                        soma_x=soma_x,
                        soma_y=soma_y,
                        soma_z=soma_z,
                        cell_type=cell_type,
                        )
    dicts_to_update.append(basic_cell_dict)

    if verbose:
        print(f"\n--5b) Neuron Overall Statistics")
    neuron_stats_dict = nst.neuron_stats(neuron_obj_proof,
                                        cell_type_mode=True)
    dicts_to_update.append(neuron_stats_dict)



    #---- 5b) Synapse Stats
    if verbose:
        print(f"\n--5b) Synapse Stats")
    syn_stats = syu.complete_n_synapses_analysis(neuron_obj_proof)
    dicts_to_update.append(syn_stats)



    # ------- 6) Cell Typing AFter Proofreading --------
    if verbose:
        print(f"\n--6) Cell Typing Info after proofreading")
    baylor_e_i,baylor_cell_type_info = ctu.e_i_classification_from_neuron_obj(
        neuron_obj_proof,
        verbose = False,
        return_cell_type_info = True,
        return_probability = True)

    baylor_cell_type_info["baylor_cell_type"] = baylor_e_i
    baylor_cell_type_info = {f"{k}_after_proof":v for k,v in baylor_cell_type_info.items()}
    dicts_to_update.append(baylor_cell_type_info)


    # -------- 7) Compartment Features
    if verbose:
        print(f"\n--7) Compartment Features --")
    axon_feature_dict = au.axon_features_from_neuron_obj(
        neuron_obj_proof,
        features_to_exclude=("length","n_branches"))

    apical_feature_dict = apu.compartment_features_from_skeleton_and_soma_center(
        neuron_obj_proof,
        compartment_label = "apical_total",
        name_prefix = "apical",
        features_to_exclude=("length","n_branches"),
    )

    basal_feature_dict = apu.compartment_features_from_skeleton_and_soma_center(
        neuron_obj_proof,
        compartment_label = "basal",
        name_prefix = "basal",
        features_to_exclude=("length","n_branches"),
    )

    dendrite_feature_dict = apu.compartment_features_from_skeleton_and_soma_center(
        neuron_obj_proof,
        compartment_label = "dendrite",
        name_prefix = "dendrite",
        features_to_exclude=("length","n_branches"),
    )


    dicts_to_update += [axon_feature_dict,
                        apical_feature_dict,
                        basal_feature_dict,
                        dendrite_feature_dict]


    # -------- 8) Getting the limb alignment features ----
    limb_alignment_dict = apu.limb_features_from_compartment_over_neuron(
        neuron_obj_proof,verbose = True)
    dicts_to_update.append(limb_alignment_dict)


    # -------- 9) Combining Data into One Dict
    if verbose:
        print(f"\n--  Combining Data into One Dict--")

    neuron_proof_dict = dict()

    for d_u in dicts_to_update:
        neuron_proof_dict.update(d_u)
        
    if store_in_obj:
        neuron_obj.pipeline_products.set_stage_attrs(
            neuron_proof_dict,
            stage = "auto_proof",
        )
        
    return neuron_proof_dict


def auto_proof_stage(
    neuron_obj,
    mesh_decimated = None,
    calculate_after_proof_stats = True,
    store_in_obj = True,
    return_stage_products = False,
    
    verbose_outline = False,
    verbose_proofread = False,

    plot_head_neck_shaft_synapses = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    plot_compartments = False,
    plot_valid_synapses = False,
    plot_error_synapses = False,

    debug_time = False,
    **kwargs
    ):

    cell_type = vdi.cell_type(neuron_obj)
    nucleus_id = vdi.nucleus_id(neuron_obj)
        
    st = time.time()
    neuron_obj_proof,filtering_info = pru.proofread_neuron_full(
        neuron_obj,
        original_mesh = mesh_decimated,

        # arguments for processing down in DecompositionCellTypeV7
        cell_type=cell_type,
        add_back_soma_synapses = False,

        proofread_verbose = verbose_proofread,
        verbose_outline = verbose_outline,
        plot_limb_branch_filter_with_disconnect_effect = plot_limb_branch_filter_with_disconnect_effect,
        plot_final_filtered_neuron = False,
        plot_synapses_after_proofread = False,


        plot_compartments = plot_compartments,

        plot_valid_synapses = plot_valid_synapses,
        plot_error_synapses = plot_error_synapses,

        verbose = verbose_outline,
        debug_time = debug_time,

        return_red_blue_splits= True,
        return_split_locations=True,
        **kwargs
    )
    
    neuron_obj_proof.pipeline_products = neuron_obj.pipeline_products
    
    # print(f"stage = {neuron_obj_proof.pipeline_products.stages}")
    
    # print(f"stage orig = {neuron_obj.pipeline_products.stages}")

    run_time = time.time() - st

    auto_proof_products = pipeline.StageProducts(
        filtering_info=filtering_info,
        run_time=run_time,
    )
    
    neuron_obj_proof.pipeline_products.set_stage_attrs(
            auto_proof_products,
            stage = "auto_proof",
    )
    
    if calculate_after_proof_stats:
        after_stats = pipeline.StageProducts(**after_auto_proof_stats(neuron_obj_proof,store_in_obj=False,))
    else:
        after_stats = dict()
        
    if store_in_obj:
        auto_proof_products.update(after_stats)
        
        
    # print(f"stage = {neuron_obj_proof.pipeline_products.stages}")
    
    if return_stage_products:
        return neuron_obj_proof,after_stats
    
    return neuron_obj_proof
    



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


from . import proofreading_utils as pru
from . import apical_utils as apu