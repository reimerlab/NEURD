"""
False somas or messy neuron filters


"""

import datajoint_utils as du

default_min_n_syn_soma = 3
#proof_version = 6
#proof_version = 7
proof_version = None


    
def small_n_syn_soma_errors(min_n_syn_soma = None):
    if min_n_syn_soma is None:
        min_n_syn_soma = default_min_n_syn_soma
    #aprint(f"min_n_syn_soma = {min_n_syn_soma}")
    small_n_syn_soma_errors_table= (neuron_proof_table & f"n_syn_soma <= {min_n_syn_soma}").proj()
    return small_n_syn_soma_errors_table

def false_messy_soma_filter(min_n_syn_soma = None):
    
    """
    Code to verify this restriction: 
    
    import connectome_filtering as conf
    import proofread_verification as pv

    seg_id,sp_idx = conf.false_messy_soma_filter().fetch("segment_id","split_index")

    curr_idx = 15_500
    curr_seg_id = seg_id[curr_idx]
    curr_sp_idx = sp_idx[curr_idx]
    node_name = pv.node_name(curr_seg_id,curr_sp_idx)
    print(f"node_name = {node_name}")
    curr_table = minnie.SomaFeaturesAutoProofreadNeurons6() & dict(segment_id=curr_seg_id,
                                                   split_index = curr_sp_idx)
    dj.U("n_syn_soma","cell_type","dendrite_branch_length_mean") & curr_table

    pv.plot_proofread_neuron(node_name,
                            plot_error_mesh=True
                            )
    
    
    Results of validation: (look for sampling in neuron catalog)
    n_segments filtered away both by this filter and allen: 12694
    n_segments filtered away that were not included in allen non neuron table: 1083 (mostly good)
    n_segments not filtered away that were by the allen non neuron table: 980
        - a lot of partial somas or valid neurons
    
    
    
    """
    if min_n_syn_soma is None:
        min_n_syn_soma = default_min_n_syn_soma
    
    return neuron_proof_table & (small_n_syn_low_dendrite_median_exc_inh.proj() + small_n_syn_soma_errors(min_n_syn_soma).proj()).proj()
    
def neurons_minus_false_messy_soma_filter(min_n_syn_soma=None):
    if min_n_syn_soma is None:
        min_n_syn_soma=default_min_n_syn_soma

#     return  (((du.proofreading_neurons_table() - small_n_syn_low_dendrite_median_exc_inh).proj()
#             - small_n_syn_soma_errors).proj())
    return neuron_proof_table - false_messy_soma_filter(min_n_syn_soma=min_n_syn_soma).proj()

def multi_nuclei_mergers(
    keep_split_neurons = True):
    """
    Purpose: Wnat to filter away the soma soma mergers but 
    couldn't find any really good mesh features that seperaed them, so just
    going to use the number of nuceli within the radius search feature
    
    """
    
    return_table = (neuron_proof_table & "n_nuclei_in_radius >= 2") 
    
    if keep_split_neurons:
        split_table = (conf.neuron_proof_table
                   & "n_nuclei_in_radius >= 2" 
                   & "multiplicity = n_nuclei_in_radius"
                  & "multiplicity = 2"
                  )
        
        return_table = return_table - split_table.proj()
    
    return return_table

def neurons_minus_false_and_multi_soma_mergers(
    add_allen_e_i_filter = False):
    
    curr_table = conf.neurons_minus_false_messy_soma_filter() - multi_nuclei_mergers().proj()
    if add_allen_e_i_filter:
        if proof_version < 7:
            curr_table = curr_table & allen_e_i_filter
    return curr_table

import proofread_verification as pv
def G_minus_false_messy_soma_filter(G=None):
    """
    Current will reduce the graph from (nodes,edges)
    (68001, 2149378) -- > (62013, 2124962)
    """
    if G is None:
        from python_tools import system_utils as su
        G = su.decompress_pickle("/platinum_graph/Data/G_query_v6.pbz2")
        
    sg,sp_idx = neurons_minus_false_messy_soma_filter().fetch("segment_id","split_index")
    n_names = [pv.node_name(k,v) for k,v in zip(sg,sp_idx)]
    
    G_restricted = G.subgraph(n_names).copy()
    return G_restricted

from python_tools import networkx_utils as xu
def G_minus_false_and_multi_soma(G=None,
                                filter_away_baylor_allen_cell_type_mismatch = False,
                                verbose = True):
    """
    Current will reduce the graph from (nodes,edges)
    (68001, 2149378) -- > (62013, 2124962)
    """
    if G is None:
        from python_tools import system_utils as su
        G = su.decompress_pickle("/platinum_graph/Data/G_query_v6.pbz2")
        
    sg,sp_idx = neurons_minus_false_and_multi_soma_mergers().fetch("segment_id","split_index")
    n_names = [pv.node_name(k,v) for k,v in zip(sg,sp_idx)]
    
    G_restricted = G.subgraph(n_names).copy()
    
    if filter_away_baylor_allen_cell_type_mismatch:
        G_restricted = xu.subgraph_from_node_query(G_restricted,"allen_e_i == baylor_e_i").copy()
    return G_restricted

import connectome_filtering as conf


# ------------- Setting up parameters -----------
from python_tools import module_utils as modu 

# data_fetcher = None
# voxel_to_nm_scaling = None

# -- default
import dataInterfaceMinnie65
attributes_dict_default = dict(
    voxel_to_nm_scaling = dataInterfaceMinnie65.voxel_to_nm_scaling,
    data_fetcher = dataInterfaceMinnie65.data_interface
)    
global_parameters_dict_default = dict(
    #max_ais_distance_from_soma = 50_000
)

# -- microns
global_parameters_dict_microns = {}
attributes_dict_microns = {}

#-- h01--
import dataInterfaceH01
attributes_dict_h01 = dict(
    voxel_to_nm_scaling = dataInterfaceH01.voxel_to_nm_scaling,
    data_fetcher = dataInterfaceH01.data_interface
)
global_parameters_dict_h01 = dict()
    
       
data_type = "default"
algorithms = None
modules_to_set = [conf]

def set_global_parameters_and_attributes_by_data_type(data_type,
                                                     algorithms_list = None,
                                                      modules = None,
                                                     set_default_first = True,
                                                      verbose=False):
    if modules is None:
        modules = modules_to_set
    
    modu.set_global_parameters_and_attributes_by_data_type(modules,data_type,
                                                          algorithms=algorithms_list,
                                                          set_default_first = set_default_first,
                                                          verbose = verbose)
    #global neuron_proof_table
    
set_global_parameters_and_attributes_by_data_type(data_type,
                                                   algorithms)

neuron_proof_table = data_fetcher.proofreading_neurons_table#du.proofreading_neurons_table(version = proof_version)

allen_e_i_filter = "(allen_e_i is NULL) or (allen_e_i = baylor_e_i)"

small_n_syn_low_dendrite_median_exc_inh = (neuron_proof_table & 
            "((n_syn_soma <= 17) AND (cell_type = 'excitatory') AND (dendrite_branch_length_mean < 35))" 
            f" OR "
            "((n_syn_soma <= 17) AND (cell_type = 'inhibitory') AND (dendrite_branch_length_mean < 28))" ).proj()

def output_global_parameters_and_attributes_from_current_data_type(
    modules = None,
    algorithms = None,
    verbose = True,
    lowercase = True,
    output_types = ("global_parameters"),
    include_default = True,
    algorithms_only = False,
    **kwargs):
    
    if modules is None:
        modules = modules_to_set
    
    return modu.output_global_parameters_and_attributes_from_current_data_type(
        modules,
        algorithms = algorithms,
        verbose = verbose,
        lowercase = lowercase,
        output_types = output_types,
        include_default = include_default,
        algorithms_only = algorithms_only,
        **kwargs,
        )

        