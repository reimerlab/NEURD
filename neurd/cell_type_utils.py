'''



Interesting website for cell types: 
http://celltypes.brain-map.org/experiment/morphology/474626527




'''
import copy
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn import linear_model
import time
from python_tools import numpy_dep as np
from python_tools import module_utils as modu
from python_tools import general_utils as gu
from python_tools import pathlib_utils as plu

e_i_model = None
e_i_model_features_default = ["syn_density_shaft","spine_density"]
#e_i_model_features_default = ["syn_density_post","spine_density"]
module_path = str(plu.parent_directory(Path(__file__)).absolute())

# use the 
manual_df_path = str(module_path / Path("man_proof_stats_df_for_e_i.csv"))#/meshAfterParty/meshAfterParty/man_proof_stats_df_for_e_i.pbz2"
manual_df_path_backup = "/neurd_packages/meshAfterParty/meshAfterParty/man_proof_stats_df_for_e_i.pbz2"
manual_exc_df = None
manual_inh_df = None

#border_df_path  = "/meshAfterParty/meshAfterParty/border_df_for_e_i_improved.pbz2"
border_df_path = str(module_path / Path("border_df_for_e_i_improved.csv"))
border_df_path_backup = "/neurd_packages/meshAfterParty/meshAfterParty/border_df_for_e_i_improved.pbz2"
border_exc_df = None
border_inh_df = None

e_i_model_logistic_coef = np.array([[ 3.22451485, -4.7]])
e_i_model_logistic_intercept = np.array([-1.2])

nomenclature = dict(
IT = "intratelencephalic",
PT = "pyramidal_tract",
CT = "corticothalamic",
VISp= "primary_visual_cortex",
CF = "corticofugal",
NP = "near_projecting"
    
)
cell_type_fine_names_excitatory = {

 '23P':"layer_2/3_pyramidal", 
 '4P': "layer 4_pyramidal",
 '5P_IT': "layer_5_intratelencephalic",
 '5P_NP': "layer_5_near_projecting", 
 '5P_PT': "layer_5_pyramidal_track",
 '6CT': "layer_6_corticothalamic",
 '6IT': "layer_6_intratelencephalic",
}

cell_type_fine_name_inhibitory = {
 'BC':"basket",
 'BPC':"bipolar",
 'MC':"martinotti",
 'NGC':"neurogliaform"
}

e_i_from_type_dict = gu.merge_dicts([{
'IT_short': "excitatory",
'bc': "inhibitory",
'Martinotti': "inhibitory",
'IT_small_tuft': "excitatory",
'5P_NP': "excitatory",
'sst': "inhibitory",
'VIP': "inhibitory", #basket cell
'ndnf+npy-': "inhibitory",
'Pvalb': "inhibitory", #baske cell
'bpc': "inhibitory",
'IT_big_tuft': "excitatory",
'prox targeting': "inhibitory",
'l1vip': "inhibitory",
'I targeting non-bpc': "unknown",
'ngfc': "inhibitory",
'alfa7': "unknown",
'cb1 basket': "inhibitory",
'np-targeting': "inhibitory",
'small basket': "inhibitory",
'CCK': "inhibitory", #basket cell
'chandelier': "inhibitory",
"PT":"excitatory",
"IT":"excitatory",
"basket":"inhibitory",
"npy":"inhibitory",
"alfa7":"inhibitory",
"chandelier":"inhibitory",
"MC":"inhibitory",
'6P':"excitatory",
"5P_IT":"excitatory",
"5P":"excitatory",
'23P':"excitatory", 
'4P':"excitatory",
'1P':"excitatory",
'INH': "inhibitory",
'Unsure E':"excitatory",
'Unsure I':"inhibitory",
},
    {k:"excitatory" for k in cell_type_fine_names_excitatory.keys()},
    {k:"excitatory" for k in cell_type_fine_names_excitatory.values()},
    
    {k:"inhibitory" for k in cell_type_fine_name_inhibitory.keys()},
    {k:"inhibitory" for k in cell_type_fine_name_inhibitory.values()},])


allen_cell_type_fine_classifier_labels = [
"23P",
"4P",
"5P-IT",
"5P-NP",
#"5P-PT",
"5P-ET",
"6P-CT",
"6P-IT",
"BC",
"BPC",
'MC',
'NGC'
]

cell_type_fine_allen_classifier_labels = allen_cell_type_fine_classifier_labels

allen_cell_type_fine_classifier_labels_exc = [
"23P",
"4P",
"5P-IT",
"5P-NP",
#"5P-PT",
"5P-ET",
"6P-CT",
"6P-IT",
]

allen_cell_type_fine_classifier_labels_inh = [
"BC",
"BPC",
'MC',
'NGC'
]

allen_cell_type_fine_classifier_to_e_i_map = {
"23P":"excitatory",
"4P":"excitatory",
"5P-IT":"excitatory",
"5P-NP":"excitatory",
#"5P-PT":"excitatory",
"5P-ET":"excitatory",
"6P-CT":"excitatory",
"6P-IT":"excitatory",
"BC":"inhibitory",
"BPC":"inhibitory",
'MC':"inhibitory",
'NGC':"inhibitory",
}

cell_type_fine_to_cell_type_coarse_map = allen_cell_type_fine_classifier_to_e_i_map

allen_cell_type_coarse_classifier_labels = ["excitatory","inhibitory"]

publishable_names_map = dict_map={
    "5P-PT":"5P-ET",
    "5P_PT":"5P_ET"
}

def cell_type_fine_mapping_publishable(
    df,
    column="gnn_cell_type_fine",
    dict_map = None):
    if dict_map is None:
        dict_map = publishable_names_map
    return pu.map_column_with_dict(
        df = df,
        column = column,
        dict_map=dict_map,
        use_default_value=False
    )

map_cell_type_fine_publishable = cell_type_fine_mapping_publishable

def classes_from_cell_type_name(cell_type_name):
    """
    Purpose: Returns a list to iterate over depending on 
    the name of the cell type
    
    cell_type_predicted ==> 
    """
    if cell_type_name in ["cell_type_predicted","cell_type_fine"]:
        classes = allen_cell_type_fine_classifier_labels
    else:
        classes = allen_cell_type_coarse_classifier_labels
    return classes

cell_type_fine_classifier_map = {
'IT_short': "5P-IT",
'bc': "BC",
'Martinotti': "MC",
'IT_small_tuft': "5P-IT",
'5P_NP': "5P-NP",
"5P_PT":"5P-PT",
'sst': "SST",
'VIP': "other", #basket cell
'ndnf+npy-': "other",
'Pvalb': "Pvalb", #baske cell
'bpc': "BPC",
'IT_big_tuft': "5P-IT",
'prox targeting': "other",
'l1vip': "other",
'I targeting non-bpc': "other",
'ngfc': "NGC",
'alfa7': "other",
'cb1 basket': "other",
'np-targeting': "other",
'small basket': "BC",
'CCK': "other", #basket cell
'chandelier': "other",
"PT":"other",
"IT":"other",
"basket":"BC",
"npy":"other",
"alfa7":"other",
"chandelier":"other",
"MC":"MC",
'6P':"other",
"5P_IT":"5P-IT",
"5P":"other",
'23P':"23P", 
'4P':"4P",
'1P':"other",
'INH': "other",
'Unsure E':"other",
'Unsure I':"other",
'ndnf+npy_':"other",
'I targeting non_bpc':"other",
'BPC':"BPC",
'SST':"SST",
'BC':"BC",
'np_targeting':"other",
'Unsure':"other",
'NGC':"NGC",
'6P_IT':"6P-IT",
'6P_U':"other",
'6P_CT':"6P-CT",
'WM_P':"other",
'6CT':"6P-CT",
}            

def cell_type_fine_classifier_map_derived(
    cell_type_dict = None,
    e_i_type=None,
    e_i_labels = False,
    default_value = None,
    cell_type_dict_extended = None):
    
    if cell_type_dict is None:
        cell_type_dict = cell_type_fine_classifier_map.copy()
        
    if cell_type_dict_extended is not None:
        cell_type_dict.update(cell_type_dict_extended)
    
    def dummy(g):
        return g
    
    if e_i_labels:
        map_func = ctu.e_i_label_from_cell_type_fine
    else:
        map_func = dummy
    
    return_dict = {}
    for k,v in cell_type_dict.items():
        #print(f"inside cell types")
        if k in ["Unsure E","Unsure I"]:
            #print(f"inside")
            if not e_i_labels:
                if v in ["other","unknown"]:
                    return_dict[k] = default_value
                    continue
                else:
                    return_dict[k] = v
            else:
                if k == "Unsure I":
                    return_dict[k] = "inhibitory"
                else:
                    return_dict[k] = "excitatory"
            continue
            
            
        
        
        if v in ["other","unknown"]:
            return_dict[k] = default_value
            continue
            
        e_i_label = ctu.e_i_label_from_cell_type_fine(k)
        if e_i_type is not None:
            if e_i_label != e_i_type:
                return_dict[k] = default_value
                continue
        
        if e_i_labels:
            return_dict[k] = e_i_label
        else:
            return_dict[k] = v
            
    return return_dict
            
    
        

    
cell_type_fine_classifier_map_classes = list(np.unique(list(cell_type_fine_classifier_map.values())))
cell_type_fine_classifier_map_classes.remove("other")

cell_type_fine_classifier_weights = {
'23P': 0.25,#1294
'4P': 0.3,#890
'5P_IT': 0.5,#465
'6P': 0.8,#342
'6P_IT': 0.8,#263
'5P_PT': 0.8,#224
}
cell_type_fine_color_map_old = {
'23P': "royalblue",#1294
'4P': "slategrey",#890
'5P_IT': "lime",#465
'6P': "lightsteelblue",#342
'6P_IT': "forestgreen",#263
'5P_PT': "Turquoise",#224
'BC': "black",#218
'Martinotti': "brown",#151
'6P_CT': "tan",#132
'SST': "pink",#93
'5P_NP': "blue",#91
'BPC': "red",#88
'Unsure': "yellow",#41
'IT_short': "purple",#39
'IT_small_tuft': "plum",#37
'Pvalb': "orange",#35
'IT_big_tuft': "indigo",#34
'6CT': "tan",#33
'6P_U': "yellowgreen",#27
'WM_P': "teal",#20
'ngfc': "gold",#20
'I targeting non_bpc': "olivedrab",#19
'NGC': "lightcoral",#17
'1P': "springgreen",#15
'VIP': "chocolate",#14
'ndnf+npy_': "tomato",#13
"Unsure E": 'lime',
"Unsure I": "pink"
# 'prox targeting': 4
# 'cb1 basket': 4
# 'l1vip': 3
# 'CCK': 1
# 'alfa7': 1
# 'chandelier': 1
# 'np_targeting': 1
# 'small basket': 1
}

cell_type_fine_color_map_bcm = {
'23P': "darkblue",#1294
'4P': "greenyellow",#890
# '5P_IT': "silver",#465
# '5P-IT': "silver",#465
'5P_IT': "orange",#465
'5P-IT': "orange",#465
'6P': "royalblue",#342
'6P_IT': "yellowgreen",#263
'6P-IT': "yellowgreen",#263
'5P_PT': "Turquoise",#224
'5P-PT': "Turquoise",#224
'5P_ET': "Turquoise",#224
'5P-ET': "Turquoise",#224
'BC': "pink",#218
'Martinotti': "brown",#151
'MC': "brown",#151
'6P_CT': "olivedrab",#132
'6P-CT': "olivedrab",#132
'SST': "orange",#93
'5P_NP': "royalblue",#91
'5P-NP': "royalblue",#91
'BPC': "red",#88
'Unsure': "yellow",#41
'IT_short': "purple",#39
'IT_small_tuft': "plum",#37
'Pvalb': "purple",#35
'IT_big_tuft': "indigo",#34
'6CT': "lightsteelblue",#33
'6P_U': "black",#27
'WM_P': "teal",#20
'ngfc': "coral",#20
'I targeting non_bpc': "rosybrown",#19
'NGC': "black",#17
'1P': "springgreen",#15
'VIP': "chocolate",#14
'ndnf+npy_': "tomato",#13
'prox targeting': "cornsilk",#3
'cb1 basket': "seashell",#3
'l1vip': "yellow",#3
'CCK': "yellow",#1
'alfa7': "yellow",#1
"Unsure E": 'lime',
"Unsure I": "pink",
"Other Exc": 'grey',
"Other Inh": "lime",
'chandelier': "lightcyan",#1
'np_targeting': "ghostwhite",#1
'No Label':"grey",
'small basket': "ghostwhite",#1
}

cell_type_fine_allen_color_map = {
    '23P':'#8268DC',
    '4P':'#647FDC',
    '5P-IT':'#77BCDE',
    '5P-PT':'#87DD90',
    '5P-ET':'#87DD90',
    '5P-NP':'#85DEC9',
    '6P-CT':'#96DD70',
    '6P-IT':'#DCC86E',
    'BC':'#D68C66',
    'BPC':'#D3697C',
    'MC':'#D36BBA',
    'NGC':'#BC6BDB',
    'astrocyte':'#009245',
    'pericyte':'#754C24',
    'microglia':'#006837',
    'oligo':'#998675',
    'OPC':'#8CC63F',
    'error':'#899499'
             
}

cell_type_fine_color_map = cell_type_fine_color_map_bcm

def e_i_label_from_cell_type_fine(
    cell_type,
    verbose = False,
    default_value = "other"):
    
    if cell_type is None:
        return default_value
    #print(f"cell_type = {cell_type}")
    if cell_type.lower() == 'unsure' or "unknown" in cell_type.lower():
        return default_value
    
    candidates = np.array(list(e_i_from_type_dict.keys()))
    cell_type = cell_type.replace("-","_")
    most_match_length = np.array([len(stru.str_overlap(cell_type.lower(),c.lower()))
                            for c in candidates])
    most_match_idx = np.argmax(most_match_length)
    if verbose:
        print(f"most_match_length = {most_match_length} \n({candidates[most_match_idx]}-{most_match_length[most_match_idx]})")
    winning_candid = candidates[most_match_idx]
    if most_match_length[most_match_idx] == 0:
        return default_value
    return e_i_from_type_dict[winning_candid]

def e_i_color_dict(
    excitatory_color="blue",
    inhibitory_color="red",
    other_color = "black",
    ):
    e_i_color_mapping = dict(
        excitatory=excitatory_color,
        inhibitory=inhibitory_color,
        other = other_color,
        unknown = other_color,
        
    )
    
    return {k:e_i_color_mapping[ctu.e_i_label_from_cell_type_fine(k,verbose=False)] for k,v in cell_type_fine_color_map.items()}
    
e_i_raw_color_dict = dict(
    excitatory = "blue",
    inhibitory = "red"
)

cell_type_coarse_color_map = e_i_raw_color_dict

def coarse_cell_type_from_fine(cell_type):
    for k,v in e_i_from_type_dict.items():
        if k in cell_type:
            return v
        
    return None

def coarse_cell_type_from_coarse(cell_type):
    if "exc" in cell_type:
        return "excitatory"
    elif "inh" in cell_type:
        return "inhibitory"
    else:
        return None

def filter_cell_type_df_for_most_complete_duplicates(
    df,
    segment_id_name = "pt_root_id"):
    #filter to keep most complete
    """
    Purpose: To filter so that only one of segment id
    and takes the most filled out one

    Pseudocode: 
    For each unique pt_root_id
    1) Find all of the rows
    2) If more than 1 filter for those not None in cell_type
    3a) if empty then add first of initial
    3b) if not empty then add first of final
    """
    filt_df = []
    for root_id in df[segment_id_name].unique():
        with_nan_df = df.query(f'{segment_id_name} == {root_id}')
        non_nan_df = pu.filter_away_nan_rows(with_nan_df)
        if len(non_nan_df) == 0:
            filt_df.append(with_nan_df.iloc[0].to_dict())
        else:
            filt_df.append(non_nan_df.iloc[0].to_dict())

    filt_df = pd.DataFrame.from_records(filt_df)
    return filt_df
    

cell_type_fine_for_clustering_exc = [
    'IT_short', 'IT_small_tuft', '5P_NP', 'IT_big_tuft'
]

cell_type_fine_for_clustering_inh = ['bc',
 'Martinotti',
 'sst',
 'VIP',
 'ndnf+npy-',
 'Pvalb',
 'bpc',
#  'l1vip',
 'ngfc',
#  'cb1 basket',
#  'small basket',
#  'chandelier'
                                    ]

"""
Location where the allen labels are
ManualCellTypesAllen() & "table_name != 'allen_v1_column_types_slanted'"

"""


#--------parameters ----------


cell_type_fine_names = gu.merge_dicts([cell_type_fine_names_excitatory,
               cell_type_fine_name_inhibitory])
    
def plot_e_i_model_classifier_map(
    data_to_plot = None,
    **kwargs):
    mlu.plot_classifier_map(e_i_model,
                            feature_1_name=e_i_model_features_default[0],
                           feature_2_name=e_i_model_features_default[1],
                           data_to_plot = data_to_plot,
                           **kwargs)


def postsyn_branches_near_soma(neuron_obj,
    perform_axon_classification = False,
    #for the synapses filter
    n_synapses_post_min = 2,
    synapse_post_perc_min = 0.8,
    plot_syn_post_filt = False,

    #for the skeletal length and spine filter
    lower_width_bound = 140,
    upper_width_bound = 520,#380,
    spine_threshold = 2,
    skeletal_distance_threshold = 110000,#30000,
    skeletal_length_threshold = 15000,#10000

    plot_spines_and_sk_filter = False,
    verbose = False):
    """
    Pseudocode: 
    1) Do axon classification without best candidate to eliminate possible axons (filters away)
    2) filter away only branches with a majority postsyns
    3) apply spine and width restrictions

    """

    if perform_axon_classification:
        au.axon_classification_using_synapses(neuron_obj,
                                              return_best_candidate = False,
                                             plot_final_axon=False
                                             )


    non_axon_candidates = ns.query_neuron_by_labels(neuron_obj,
                             not_matching_labels=["axon"])

    if verbose:
        print(f"non_axon_candidates = {non_axon_candidates}")


    syn_post_filt_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list=[ns.n_synapses_post,ns.synapse_post_perc],
                   query=(f"(n_synapses_post>{n_synapses_post_min}) and " 
                         f"synapse_post_perc > {synapse_post_perc_min}"),
                    limb_branch_dict_restriction=non_axon_candidates,
                    plot_limb_branch_dict=plot_syn_post_filt
                   )
    if verbose:
        print(f"syn_post_filt_limb_branch = {syn_post_filt_limb_branch}")

    query_postsyn_filter = (f"(distance_from_soma<{skeletal_distance_threshold})"
                                                       f" and (no_spine_median_mesh_center > {lower_width_bound})"
                                                       f" and (no_spine_median_mesh_center < {upper_width_bound})"
                                                      f" and (n_spines > {spine_threshold})"
                                                       f" and skeletal_length > {skeletal_length_threshold} ")
    if verbose:
        print(f"query_postsyn_filter = {query_postsyn_filter}")
        
    spines_and_sk_filt_limb_branch = ns.query_neuron(neuron_obj,
                                                functions_list=["distance_from_soma","no_spine_median_mesh_center",
                                                                "n_spines","spine_density","skeletal_length"],
                                                query=(query_postsyn_filter
                                                      ),
                                                limb_branch_dict_restriction=syn_post_filt_limb_branch)
    
    if verbose:
        print(f"spines_and_sk_filt_limb_branch = {spines_and_sk_filt_limb_branch}")

    if plot_spines_and_sk_filter:
        print(f"plotting plot_spines_and_sk_filter")
        nviz.plot_limb_branch_dict(neuron_obj,spines_and_sk_filt_limb_branch)
        
    return spines_and_sk_filt_limb_branch

def postsyn_branches_near_soma_for_syn_post_density(neuron_obj,
                                                    plot_spines_and_sk_filter=False,
                                                    
                                                    spine_threshold = None,
                                                    skeletal_length_threshold = None,
                                                    upper_width_bound = None,
                                                    **kwargs):
    """
    Purpose: To restrict the branches close to the soma that will be used 
    for postsynaptic density
    
    Ex: 
    from neurd import cell_type_utils as ctu

    output_limb_branch = ctu.postsyn_branches_near_soma_for_syn_post_density(
                            neuron_obj = neuron_obj_exc_syn_sp,
                           verbose = True)
                           
    from neurd import neuron_visualizations as nviz
    nviz.plot_limb_branch_dict(neuron_obj_exc_syn_sp,
                              output_limb_branch)
    """
    if spine_threshold is None:
        spine_threshold = spine_threshold_syn_density_global
    if skeletal_length_threshold is None:
        skeletal_length_threshold = skeletal_length_threshold_syn_density_global
    if upper_width_bound is None:
        upper_width_bound= upper_width_bound_syn_density_global
    
    return ctu.postsyn_branches_near_soma(
            neuron_obj,
            spine_threshold = spine_threshold,
            skeletal_length_threshold = skeletal_length_threshold,
            upper_width_bound = upper_width_bound,
            plot_spines_and_sk_filter = plot_spines_and_sk_filter,
            **kwargs)
    
    
#     if special_syn_parameters:
#         return ctu.postsyn_branches_near_soma(
#             neuron_obj,
#             spine_threshold = -1,
#             skeletal_length_threshold = 5000,
#             upper_width_bound = 10000,
#             plot_spines_and_sk_filter = plot_spines_and_sk_filter,
#             **kwargs)
#     else:
#         return ctu.postsyn_branches_near_soma(
#             neuron_obj,
#             spine_threshold = -1,
#             skeletal_length_threshold = 5000,
#             plot_spines_and_sk_filter = plot_spines_and_sk_filter,
#             **kwargs)
    


def synapse_density_near_soma(neuron_obj,
                              limb_branch_dict = None,
                              synapse_type = "synapses",
                              verbose = False,
                              multiplier = 1000,
                              return_skeletal_length = False,
                              plot_spines_and_sk_filter = False,
                             **kwargs):
    """
    Application: To be used for cell type (E/I)
    classification
    """
    if limb_branch_dict is None:
        limb_branch_dict = ctu.postsyn_branches_near_soma_for_syn_post_density(
                                neuron_obj = neuron_obj,
                                plot_spines_and_sk_filter=plot_spines_and_sk_filter,
                               verbose = verbose,
                                **kwargs)

    density = syu.synapse_density_over_limb_branch(
        neuron_obj = neuron_obj,
        limb_branch_dict=limb_branch_dict,
        verbose = verbose,
        synapse_type = synapse_type,
        multiplier = multiplier,
        return_skeletal_length = return_skeletal_length)
    return density

def synapse_density_stats(neuron_obj,
                          verbose = False,
                          return_skeletal_length=True,
                         **kwargs):
                          
    """
    Purpose To compute synapse densities that 
    could be used for E/I classification
    """
    
    syn_density_post = ctu.synapse_density_near_soma(neuron_obj,
                                 synapse_type="synapses_post",
                                 verbose = False,**kwargs)

    syn_density_head = ctu.synapse_density_near_soma(neuron_obj,
                                 synapse_type="synapses_head",
                                 verbose = False,**kwargs)

    syn_density_neck = ctu.synapse_density_near_soma(neuron_obj,
                                 synapse_type="synapses_neck",
                                 verbose = False,**kwargs)

    syn_density_shaft,sk_length = ctu.synapse_density_near_soma(neuron_obj,
                                 synapse_type="synapses_shaft",
                                return_skeletal_length = True,
                                 verbose = False,**kwargs)

    if verbose:
        print(f"syn_density_post = {syn_density_post}")
        print(f"syn_density_head = {syn_density_head}")
        print(f"syn_density_neck = {syn_density_neck}")
        print(f"syn_density_shaft = {syn_density_shaft}")
        print(f"sk_length_synapse_density = {sk_length}")
    
    
    
    return_value =  [syn_density_post,syn_density_head,syn_density_neck,syn_density_shaft]
    
    if return_skeletal_length:
        return_value.append(sk_length)
        
    return return_value



def spine_density_near_soma(neuron_obj,
                            limb_branch_dict=None,
                           verbose = True,
                            multiplier = 1000,
                            return_skeletal_length = True,
                            lower_width_bound = None,
                            upper_width_bound = None,
                           **kwargs):
    """
    Purpose: To compute the spine 
    density over branches near the soma

    Application: To be used for 
    cell classification

    Ex: 
    ctu.spine_density_near_soma(neuron_obj = neuron_obj_exc_syn_sp,
        verbose = True,
        multiplier = 1000)
    """
    if lower_width_bound is None:
        lower_width_bound = lower_width_bound_spine_global
    if upper_width_bound is None:
        upper_width_bound = upper_width_bound_spine_global
    
    if limb_branch_dict is None:
        postsyn_limb_branch = ctu.postsyn_branches_near_soma(neuron_obj,
                                                             lower_width_bound=lower_width_bound,
                                                             upper_width_bound=upper_width_bound,
                                                        **kwargs)
    else:
        postsyn_limb_branch = limb_branch_dict
        
    if verbose:
        print(f"postsyn_limb_branch = {postsyn_limb_branch}")

    spine_density,sk_length = spu.spine_density_over_limb_branch(neuron_obj,
                                       limb_branch_dict = postsyn_limb_branch,
                                      verbose = verbose,
                                      multiplier = multiplier,
                                                      return_skeletal_length = True)
    if verbose:
        print(f"spine_density = {spine_density} (multiplier = {multiplier})")
        print(f"sk_length spine density = {sk_length}")
        
    if return_skeletal_length:
        return spine_density,sk_length
    else:
        return spine_density

# ------------ 8/28 E/I Classifier ---------------

def load_manual_exc_inh_df(path=manual_df_path):
    try:
        manual_df = pu.csv_to_df(manual_df_path)
    except: 
        manual_df = pu.csv_to_df(manual_df_path_backup)
    
    global manual_exc_df
    global manual_inh_df
    
    manual_exc_df = manual_df.query("cell_type=='excitatory'")
    manual_inh_df = manual_df.query("cell_type=='inhibitory'")
    manual_inh_df["cell_type_manual"] = "inhibitory"
    manual_exc_df["cell_type_manual"] = "excitatory"


def set_e_i_model_as_kNN(X,
                         y,
                         n_neighbors = 5,
                        plot_map = False,
                        **kwargs):
    """
    To create the kNN
    """
    clf = mlu.kNN_classifier(X,y,
                   n_neighbors=n_neighbors,
                   plot_map = plot_map,
                   verbose = False,
                    map_fill_colors=['orange', 'cyan',],
                    scatter_colors=["darkorange","c"],
                             **kwargs
                  )
    return clf

def set_e_i_model(features= e_i_model_features_default,
                  label = "cell_type",
                  add_features_to_model_obj=True,
                  model_type = "logistic_reg",#"kNN",
                  plot_map = False,
                  return_new_model = False,
                  **kwargs
                 ):
    """
    Purpose: To set the module e/i classifier
    
    Ex: How to specify different features for the classification
    ctu.set_e_i_model(plot_map = True,
                 features= ["spine_density","syn_density_shaft"])
    """
    global e_i_model
    
    if model_type == "kNN":
        global manual_exc_df
        global manual_inh_df

        if manual_exc_df is None or manual_inh_df is None:
            ctu.load_manual_exc_inh_df()

        class_exc = manual_exc_df[label].to_list()
        class_inh = manual_inh_df[label].to_list()

        exc_features = pu.columns_values_from_df(manual_exc_df,features)
        inh_features = pu.columns_values_from_df(manual_inh_df,features)

        y = np.concatenate([class_inh,class_exc])

        total_features = [np.concatenate([i,e]) for i,e 
                                      in zip(inh_features,exc_features)]
        X = np.vstack(total_features).T

        if model_type == "kNN":
            clf = ctu.set_e_i_model_as_kNN(X,
                             y,
                            plot_map = plot_map,
                            **kwargs)
        else:
            raise Exception(f"Unimplemented model type: {model_type}")
                

    elif model_type == "logistic_reg":
        clf = ctu.e_i_model_as_logistic_reg_on_border_df(plot_decision_map = plot_map,**kwargs)
        
    if add_features_to_model_obj:
        clf.features = features

    if not return_new_model:
        e_i_model = clf
    else:
        return clf
        
    
def predict_class_single_datapoint(clf,
                                  data,
                                  verbose = False,
                                  return_probability = False):
    """
    Purpose: To predict the class of a single datapoint
    
    Ex: 
    data = [1,1]
    mlu.predict_class_single_datapoint(clf,data,verbose = True)

    """
    data = np.array(data).reshape(1,-1)
    pred_class = clf.predict(data)[0]
    pred_prob = np.max(clf.predict_proba(data)[0])
    if verbose:
        print(f"Class = {pred_class} for data = {data} with prediction probability {pred_prob}")
        
    if return_probability:
        return pred_class,pred_prob
    else:
        return pred_class
    
def e_i_classification_single(data,
                              features=None,
                              model = None,
                             verbose = False,
                              return_label_name = True,
                              plot_on_model_map = False,
                              return_probability = False,
                             ):
    """
    Right now usually done with 
    'syn_density_shaft', 'spine_density' but can specify other features
    
    ctu.e_i_classification_single([0.5,0.6],
                              features = ["spine_density","syn_density_shaft"],
                              verbose = True)
    
    """
    if model is None: 
        model = e_i_model
        
    if features is not None:
        if features != model.features:
            if verbose:
                print(f"Setting features from previously {model.features} to now {features}")
            model = ctu.set_e_i_model(features=features,return_new_model=True)
    
    if verbose:
        print(f"For model: {model} ")
        print(f"with features: {model.features}")
    
    pred_class,pred_prob = predict_class_single_datapoint(model,data,
                                                   return_probability = True)
    try:
        pred_class_label = model.labels[pred_class]
    except:
        pred_class_label = pred_class
    
    if verbose:
        print(f"pred_class = {pred_class}, pred_class_label = {pred_class_label}")
        
    #print(f"plot_on_model_map = {plot_on_model_map}")
    if plot_on_model_map:
        print(f"******* ATTEMPTING TO PLT ON MODEL MAP****")
        #mlu.plot_classifier_map(model,data_to_plot = data,)
        ctu.plot_classifier_map(model,X = np.array(data).reshape(-1,2),y = [pred_class_label])
    
    if return_label_name:
        return_class = pred_class_label
    else:
        return_class = pred_class
        
    if return_probability:
        return pred_class,pred_prob
    else:
        return pred_class

def e_i_classification_from_neuron_obj_old(neuron_obj,
                                      features = e_i_model_features_default,
                                      verbose = False,
                                      return_cell_type_info = False,
                                       return_dendrite_branch_stats = False,
                                      plot_on_model_map=False,
                                       
                                       #parameters for hand-made rules
                                      apply_hand_made_low_rules = True,
                                      skeletal_length_processed_syn_min = 15_000,
                                      skeletal_length_processed_spine_min = 15_000,
                                       
                                       excitatory_spine_density_min = 0.1,
                                       
                                       #visualization arguments
                                       plot_spines_and_sk_filter_for_syn = False,
                                       plot_spines_and_sk_filter_for_spine = False,
                                       
                                       special_syn_parameters = True,
                                       
                                      ):
    """
    Purpose: To take a neuron object and classify it as 
    excitatory or inhibitory
    """
    st = time.time()
    limb_branch_dict = ctu.postsyn_branches_near_soma_for_syn_post_density(
                                neuron_obj = neuron_obj,
                            plot_spines_and_sk_filter=plot_spines_and_sk_filter_for_syn,
        special_syn_parameters=special_syn_parameters,
                               verbose = False,)
    
    (syn_density_post,
             syn_density_head,
             syn_density_neck,
             syn_density_shaft,
             skeletal_length_processed_syn) = ctu.synapse_density_stats(neuron_obj = neuron_obj,
                          limb_branch_dict = limb_branch_dict,
                                            verbose = verbose)
    
    (spine_density,
     skeletal_length_processed_spine) = ctu.spine_density_near_soma(neuron_obj = neuron_obj,
                                                verbose = verbose,
                                                plot_spines_and_sk_filter=plot_spines_and_sk_filter_for_spine,
                                                multiplier = 1000)
    
    baylor_cell_type_info = dict(
                        syn_density_post = syn_density_post,
                        syn_density_head = syn_density_head,
                        syn_density_neck = syn_density_neck,
                        syn_density_shaft = syn_density_shaft,
                        skeletal_length_processed_syn=skeletal_length_processed_syn,
                        spine_density=spine_density,
                        skeletal_length_processed_spine = skeletal_length_processed_spine
            )
    
    baylor_e_i = None
    if apply_hand_made_low_rules:
        if ((skeletal_length_processed_syn < skeletal_length_processed_syn_min ) or 
            (skeletal_length_processed_spine < skeletal_length_processed_spine_min)):
            if verbose:
                print(f"Classified as Inhibitory because skeletal_length_processed_syn or skeletal_length_processed_spine below min")
            baylor_e_i = "inhibitory"
        elif spine_density < excitatory_spine_density_min:
            if verbose:
                print(f"Classified as Inhibitory because spine_density below min ({excitatory_spine_density_min})")
            baylor_e_i = "inhibitory"
        else:
            if verbose:
                print(f"No manual rules applied")
        
        
    if baylor_e_i is None:
        globs = globals()
        locs = locals()
        baylor_e_i = ctu.e_i_classification_single(data=[eval(k,globs,locs) for k in features],
                                  features=features,
                                 verbose = verbose,
                                  return_label_name = True,
                                    plot_on_model_map=plot_on_model_map,
                                 )
        
    if return_dendrite_branch_stats and return_cell_type_info:
        dendr_stats = ctu.dendrite_branch_stats_near_soma(neuron_obj,
                                                         limb_branch_dict=limb_branch_dict)
        if verbose:
            print(f"dendr_stats:")
            print(f"{dendr_stats}")
        baylor_cell_type_info.update(dendr_stats)

    if verbose:
        print(f"Total time for e/i calculations = {time.time() - st}")
        print(f"baylor_e_i = {baylor_e_i}")
    
    if return_cell_type_info:
        return baylor_e_i,baylor_cell_type_info
    else:
        return baylor_e_i
    
def e_i_classification_from_neuron_obj(neuron_obj,
                                      features = e_i_model_features_default,
                                      verbose = False,
                                      return_cell_type_info = False,
                                       return_dendrite_branch_stats = False,
                                      plot_on_model_map=False,
                                       
                                       #parameters for hand-made rules
                                      apply_hand_made_low_rules = None,
                                      skeletal_length_processed_syn_min = None,
                                      skeletal_length_processed_spine_min = None,
                                       
                                       inhibitory_syn_density_shaft_min = None,
                                       
                                       #visualization arguments
                                       plot_spines_and_sk_filter_for_syn = False,
                                       plot_spines_and_sk_filter_for_spine = False,
                                       
                                       e_i_classification_single = False,
                                       return_probability = True,
                                       
                                       **kwargs
                                       
                                      ):
    """
    Purpose: To take a neuron object and classify it as 
    excitatory or inhibitory
    
    The hand written rules moves the y intercept of the classifier from 
    0.372 to 0.4 
    """
    st = time.time()
    limb_branch_dict = ctu.postsyn_branches_near_soma_for_syn_post_density(
                                neuron_obj = neuron_obj,
                            plot_spines_and_sk_filter=plot_spines_and_sk_filter_for_syn,
                               verbose = False,)
    if verbose:
        print(f"About to do syn_density stats")
    (syn_density_post,
             syn_density_head,
             syn_density_neck,
             syn_density_shaft,
             skeletal_length_processed_syn) = ctu.synapse_density_stats(neuron_obj = neuron_obj,
                          limb_branch_dict = limb_branch_dict,
                                            verbose = verbose)
    
    if verbose:
        print(f"About to do spine_density stats")
    (spine_density,
     skeletal_length_processed_spine) = ctu.spine_density_near_soma(neuron_obj = neuron_obj,
                                                verbose = verbose,
                                                plot_spines_and_sk_filter=plot_spines_and_sk_filter_for_spine,
                                                multiplier = 1000,
                                                                   **kwargs)
    if verbose:
        print(f"Done stats")
    baylor_cell_type_info = dict(
                        syn_density_post = syn_density_post,
                        syn_density_head = syn_density_head,
                        syn_density_neck = syn_density_neck,
                        syn_density_shaft = syn_density_shaft,
                        skeletal_length_processed_syn=skeletal_length_processed_syn,
                        spine_density=spine_density,
                        skeletal_length_processed_spine = skeletal_length_processed_spine
            )
    
    baylor_e_i = None
    if apply_hand_made_low_rules:
        if ((skeletal_length_processed_syn < skeletal_length_processed_syn_min ) or 
            (skeletal_length_processed_spine < skeletal_length_processed_spine_min)):
            if verbose:
                print(f"Classified as Inhibitory because skeletal_length_processed_syn or skeletal_length_processed_spine below min")
            baylor_e_i = "inhibitory"
        elif syn_density_shaft < inhibitory_syn_density_shaft_min:
            if verbose:
                print(f"Classified as excitatory because syn_density_shaft below min ({inhibitory_syn_density_shaft_min})")
            baylor_e_i = "excitatory"
        else:
            if verbose:
                print(f"No manual rules applied")
    else:
        if verbose:
            print(f"Not apply_hand_made_low_rules")
        
        
    if baylor_e_i is None:
        globs = globals()
        locs = locals()
        baylor_e_i,baylor_e_i_prob = ctu.e_i_classification_single(data=[eval(k,globs,locs) for k in features],
                                  features=features,
                                 verbose = verbose,
                                  return_label_name = True,
                                    plot_on_model_map=plot_on_model_map,
                                    return_probability = True,
                                 )
        if return_cell_type_info and return_probability:
            baylor_cell_type_info["baylor_cell_type_exc_probability"] = baylor_e_i_prob
    else:
        globs = globals()
        locs = locals()
        if plot_on_model_map:
            print(f"******* ATTEMPTING TO PLT ON MODEL MAP****")
            #mlu.plot_classifier_map(model,data_to_plot = data,)
            ctu.plot_classifier_map( e_i_model,X = np.array([eval(k,globs,locs) for k in features]).reshape(-1,2),
                                    y = [baylor_e_i])
        
    if return_dendrite_branch_stats and return_cell_type_info:
        dendr_stats = ctu.dendrite_branch_stats_near_soma(neuron_obj,
                                                         limb_branch_dict=limb_branch_dict)
        if verbose:
            print(f"dendr_stats:")
            print(f"{dendr_stats}")
        baylor_cell_type_info.update(dendr_stats)

    if verbose:
        print(f"Total time for e/i calculations = {time.time() - st}")
        print(f"baylor_e_i = {baylor_e_i}")
    
    if return_cell_type_info:
        return baylor_e_i,baylor_cell_type_info
    else:
        if return_probability:
            return baylor_e_i,baylor_e_i_prob
        else:
            return baylor_e_i
    
    
    
def dendrite_branch_stats_near_soma(
    neuron_obj,
    limb_branch_dict = None,
    plot_spines_and_sk_filter = False,
    verbose = False,
    **kwargs
    ):

    """
    Purpose: To get features of the dendrite branches
    from an unclassified neuron

    Applicaiton: Can be used to help with E/I cell typing

    Pseudocode: 
    1) Get the branches near the soma up to certain distance
    
    Ex: 
    ctu.dendrite_branch_stats_near_soma(neuron_obj,)
    """

    # near_branches = ctu.postsyn_branches_near_soma(
    # neuron_obj,
    # plot_syn_post_filt = False,
    # upper_width_bound = 100000,
    # skeletal_distance_threshold = 150_000,
    # skeletal_length_threshold = 2_000,

    # plot_spines_and_sk_filter = True,
    #     verbose = True
    # )

    if limb_branch_dict is None: 
        near_branches = ctu.postsyn_branches_near_soma_for_syn_post_density(
            neuron_obj,
            plot_spines_and_sk_filter = plot_spines_and_sk_filter,
            **kwargs)
    else:
        near_branches = limb_branch_dict


    dendr_branches_dict = nst.branch_stats_over_limb_branch(
        neuron_obj,
        near_branches)

    dendr_branches_dict = {f"dendrite_{k}":v for k,v in dendr_branches_dict.items()}
    
    return dendr_branches_dict


def soma_stats_for_cell_type(neuron_obj):
    """
    Stats we want to include about the soma
    to maybe help cell type

    surface_area
    volume
    sa_to_volume
    ray_trace_percentile_70
    n_syn_soma
    """
    soma_mesh = neuron_obj["S0"].mesh
    soma_dict = dict(
        n_syn_soma = neuron_obj["S0"].n_synapses,
        soma_surface_area = tu.surface_area(soma_mesh)/1_000_000,
        soma_volume = tu.mesh_volume(soma_mesh)/1_000_000_000,
        soma_ray_trace_percentile_70 = tu.mesh_size(soma_mesh,
                                               size_type = "ray_trace_percentile",
                                               percentile=70),
        soma_sa_to_volume = tu.surface_area_to_volume(soma_mesh)
        
    )
    return soma_dict

# ===================== 10/11: improved E/I Classification ================




def load_border_exc_inh_df(path=border_df_path,
                          path_backup = border_df_path_backup):
    try:
        manual_df = pu.csv_to_df(path)
    except: 
        manual_df = pu.csv_to_df(path_backup)
    
    global border_exc_df
    global border_inh_df
    
    border_exc_df = manual_df.query("cell_type_manual=='excitatory'")
    border_inh_df = manual_df.query("cell_type_manual=='inhibitory'")
    
try:
    from machine_learning_tools import visualizations_ml as vml
except:
    vml = None

def e_i_model_as_logistic_reg_on_border_df(
    label="cell_type_manual",
    features= None,
    class_weight = {"excitatory":1, "inhibitory":1.5},
    plot_decision_map = False,
    plot_type ="probability",# "classes"
    use_handmade_params = True,
    **kwargs
    ):
    
    if features is None:
        features = ctu.e_i_model_features_default
    

    ctu.load_border_exc_inh_df()
    combined_df = pu.concat([ctu.border_exc_df,ctu.border_inh_df])

    clf = linear_model.LogisticRegression(class_weight=class_weight,**kwargs)
    X = combined_df[features]
    y = combined_df[label]
    clf.fit(combined_df[features],combined_df[label])

    if use_handmade_params:
        clf.coef_ = ctu.e_i_model_logistic_coef
        clf.intercept_ = ctu.e_i_model_logistic_intercept

    if plot_decision_map:
        if vml:
            vml.contour_map_for_2D_classifier(clf,
                                             #color_type = "probability",
                                             color_type = plot_type,
                                             training_df=combined_df,
                                             training_df_class_name=label,
                                             training_df_feature_names=features,
                                             )
    clf._fit_X = X.to_numpy()
    clf._y = y.to_numpy()
    clf.labels = np.array(["excitatory","inhibitory"])

    return clf



def plot_classifier_map(clf=None,
                       plot_type="probability",
                       X = None,
                       y = None,
                       df=None,
                       df_class_name=None,
                       df_feature_names=None,
                        
                       ):
    
    if clf is None:
        clf = e_i_model
        
    if X is None and df is None:
        X = clf._fit_X
        y = clf._y
        
    if df is None:
        df = ctu.border_training_df()
        cell_type_manual = ""
        
    if df_feature_names is None:
         df_feature_names= e_i_model_features_default#["syn_density_shaft", "spine_density"]
    if vml:
        vml.contour_map_for_2D_classifier(clf,
                                             #color_type = "probability",
                                             color_type = plot_type,
                                             X = X,
                                             y = y,
                                          training_df=df,
                                             training_df_class_name=df_class_name,
                                             training_df_feature_names=df_feature_names,
                                              feature_names = df_feature_names,

                                             )
    
def border_training_df():
    return pu.concat([ctu.border_exc_df,ctu.border_inh_df])
def all_training_df(plot = False):
    ctu.load_manual_exc_inh_df()
    ctu.load_border_exc_inh_df()
    
    train_df =  pu.concat([ctu.manual_exc_df,
           ctu.border_exc_df,
          ctu.manual_inh_df,
          ctu.border_inh_df]).reset_index()
    
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        print(f"Plotting training data")
        sns.jointplot(data=train_df,
                        x="syn_density_post",
                        y="spine_density",
                        hue="cell_type")
        plt.show()
        
    return train_df


cell_type_fine_redundant_mapping = {
        "Unsure I":"Unsure",
        "MC":"Martinotti",
        "bc":"BC",
        "bpc":"BPC",
        "sst":"SST",
    }

def cell_type_fine_name_cleaner(row):
    fine_cell_type_mapping = {
    '6IT':"6P_IT"
    }
    if type(row) == dict or "pandas" in str(type(row)):
        cell_type_fine = row["cell_type_fine"]
    else:
        cell_type_fine = row
        
    if cell_type_fine is None:
        return None
    
    cell_type_fine = cell_type_fine.replace("-","_")
    
    if cell_type_fine in fine_cell_type_mapping:
        cell_type_fine = fine_cell_type_mapping[cell_type_fine]
        
    if cell_type_fine in cell_type_fine_redundant_mapping:
        cell_type_fine = cell_type_fine_redundant_mapping[cell_type_fine]
        
    return cell_type_fine

def clean_cell_type_fine(df):
    df["cell_type_fine"] = pu.new_column_from_row_function(
        df,
        cell_type_fine_name_cleaner
    )
    return df



    
def df_cell_type_fine(df):
    df_fine = df.query("cell_type_fine == cell_type_fine")
    df_fine = ctu.clean_cell_type_fine(df_fine)
    df_fine = df_fine.query("cell_type_fine != 'Unsure E'")
    return df_fine

#feature_df = pu.csv_to_df("/neuron_mesh_tools/Auto_Proofreading/Cell_Type_Classifier/microns_autoproofread_features_v2.csv")

try:
    feature_df = pu.csv_to_df(str(module_path / "microns_autoproofread_features_v2.csv"))
except:
    feature_df = None

def cell_type_from_feature_df(segment_id):
    return feature_df.query(f"id=='{segment_id}'")["cell_type"].to_numpy()[0]


# --------------- For comparing the accuracy of cell types ---------------
def accuracy_df_by_cell_type_fine(
    df,
    verbose = False,
    cell_type_fine_label = "cell_type_fine",
    cell_type_coarse_label = "cell_type_coarse",
    add_overall = True,
    e_i_methods = ["allen","bcm","bcm_gnn"],
    ):
    
    df_fine = df.query("(cell_type_fine not in ['Unsure','Unsure E']) and (cell_type_fine == cell_type_fine)")
    
    cell_type_fine_classes = (df_fine[cell_type_fine_label]).unique()
    print(f"cell_type_fine_classes = {cell_type_fine_classes}")
    
    e_i_methods = [k for k in e_i_methods if f"cell_type_{k}"]
    if verbose:
        print(f"e_i_methods = {e_i_methods}")

    accuracy_results = []
    verbose = False
    for lab in  cell_type_fine_classes:
        df_fine_lab = df_fine.query(f"{cell_type_fine_label} == '{lab}'")
        curr_dict = dict(cell_type_label = lab,n_cells = len(df_fine_lab))
        if verbose:
            print(f"# of cells {lab} = {len(df_fine_lab)}")
        for ct_method in e_i_methods:
            curr_label = df_fine_lab[cell_type_coarse_label].unique()
            if len(curr_label)>1:
                raise Exception("")
            curr_label = curr_label[0]
            n_correct = np.sum(df_fine_lab[cell_type_coarse_label] == df_fine_lab[f"cell_type_{ct_method}"])
            accuracy = n_correct/len(df_fine_lab)
            curr_dict["e_i_type"] = curr_label
            curr_dict[f"e_i_n_correct_{ct_method}"] = n_correct
            curr_dict[f"e_i_accuracy_{ct_method}"] = accuracy

        accuracy_results.append(curr_dict)
    df_fine_results = pd.DataFrame.from_records(accuracy_results)
    df_fine_results = pd.concat([
        df_fine_results.query("e_i_type=='excitatory'"),
        df_fine_results.query("e_i_type=='inhibitory'"),
    ])
    
    if add_overall:
        new_dicts = []
        for overall_type in ["excitatory","inhibitory","total"]:
            if overall_type == "total":
                e_i_type = "exc and inh"
                df_curr = df_fine
            else:
                e_i_type = overall_type
                df_curr = df_fine.query(f"{cell_type_coarse_label} == '{overall_type}'")
                
                
            
            overall_dict = dict(
                cell_type_label = overall_type,
                n_cells = len(df_curr),
                e_i_type = e_i_type)

            for ct_method in e_i_methods:
                n_correct = np.sum(df_curr[cell_type_coarse_label] == df_curr[f"cell_type_{ct_method}"])
                accuracy = n_correct/len(df_curr)
                overall_dict[f"e_i_n_correct_{ct_method}"] = n_correct
                overall_dict[f"e_i_accuracy_{ct_method}"] = accuracy
            
            new_dicts.append(overall_dict)
        new_df = pd.DataFrame.from_records(new_dicts)
        df_fine_results = pd.concat([df_fine_results,new_df])
    
    df_fine_results = df_fine_results.reset_index()
    return df_fine_results

def rename_dict_for_cell_type_fine(
    keep_classes_inh = None,
    keep_classes_exc = None,
    default_name_inh = "Other Inh",
    default_name_exc = "Other Exc",
    verbose = False,
    ):
    """
    Purpose: Generate a renaming
    dictionary based on which exc
    and inhibitory classes you want
    to retain their name and which
    you want to group into a default name
    """
    if keep_classes_inh is None:
        keep_classes_inh= []
        
    if keep_classes_exc is None:
        keep_classes_exc = []
    

    rename_dict_inh = dict([
        (k,k) if k in keep_classes_inh
        else (k,default_name_inh) 
        for k in ctu.allen_cell_type_fine_classifier_labels_inh]
        )

    rename_dict_exc = dict([
        (k,k) if k in keep_classes_exc
        else (k,default_name_exc) 
        for k in ctu.allen_cell_type_fine_classifier_labels_exc]
        )

    rename_dict = rename_dict_inh.copy()
    rename_dict.update(rename_dict_exc)
    if verbose:
        print(f"Renaming Dict:")
        for k,v in rename_dict.items():
            print(f"{k}:{v}")
            
    return rename_dict

def rename_cell_type_fine_column(
    df,
    column = "gnn_cell_type_fine",
    keep_classes_exc = None,
    keep_classes_inh = None,
    in_place = False,
    ):
    
    if not in_place:
        df = df.copy()


    fine_map = ctu.rename_dict_for_cell_type_fine(
        keep_classes_exc = keep_classes_exc,
        keep_classes_inh = keep_classes_inh,
    )

    return pu.map_column_with_dict(
        df,
        column = column,
        dict_map=fine_map,
    )





def plot_cell_type_gnn_embedding(
    df,
    
    column = "cell_type",
    color_map = None,
    trans_cols = ["umap0","umap1"] , 
    nucleus_ids = None,
    
    plot_unlabeld = True,
    unlabeled_color = "grey",
    
    use_labels_as_text_to_plot = False,
    
    figure_width = 7,
    figure_height = 10,
    
    size = 20,
    size_labeled = 23,
    alpha = 0.05,
    alpha_labeled = 1,

    plot_legend = False, 

    title="GNN Classifier - Whole Neuron\nUMAP Embedding",
    axes_fontsize = 35,
    title_fontsize = 35,
    title_append = None,
    xlabel = None,
    ylabel = None,
    ):
    """
    Purpose: to plot certain embeddings
    """
    if color_map is None:
        color_map = ctu.cell_type_fine_color_map
    
    if title_append is not None:
        title = f"{title}\n{title_append}"

    fig,ax = plt.subplots(1,1,figsize=(figure_width,figure_height))

    # Plotting the background data
    if plot_unlabeld:

        if nucleus_ids is not None:
            df_plot = df.query(
                (f"({column} != {column})"
                f"or (nucleus_id not in {list(nucleus_ids)})")
            ).copy()
        else:
            df_plot = df.query(f"({column} != {column})").copy()

        df_plot[column] = None

        ax = vml.plot_df_scatter_classification(
            df = df_plot,
            feature_names = trans_cols,
            target_name = column,
            color = unlabeled_color,
            ndim = len(trans_cols),
            use_labels_as_text_to_plot=False,
            replace_None_with_str_None=True,
            alpha=alpha,
            ax = ax,
            plot_legend = False,
            s = size,
        )


    if nucleus_ids is not None:
        df_plot = df.query(
            (f"({column} == {column})"
            f"and (nucleus_id in {list(nucleus_ids)})")
        ).copy()
    else:
        df_plot = df.query(f"({column} != {column})").copy()



    ax = vml.plot_df_scatter_classification(
        df = df_plot,
        feature_names = trans_cols,
        target_name = column,
        target_to_color=color_map,
        ndim = len(trans_cols),
        use_labels_as_text_to_plot=use_labels_as_text_to_plot,
        replace_None_with_str_None=True,
        alpha=alpha_labeled,
        ax = ax,
        plot_legend = plot_legend,
        s = size_labeled,
    )

    mu.turn_off_axes_tickmarks(ax)

    mu.set_axes_title_size(ax,
        x_fontsize=axes_fontsize,fontsize=axes_fontsize
    )
    ax.set_title(title,fontsize = title_fontsize)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize = axes_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize = axes_fontsize)
    
    return ax

def cell_type_high_probability_df_from_df(
    df,
    cell_type,
    baylor_exc_prob_threshold = 0.65,
    gnn_prob_threshold = 0.65,
    verbose = False,
    return_df = False,
    ):
    """
    Purpose: Table after proofreading of cells
    that are highly likely to be excitatory
    """

    # using the logistic regression model
    df_filt = df.query(
        f"(cell_type == '{cell_type}')" 
        f"and (baylor_cell_type_exc_probability_after_proof >= {baylor_exc_prob_threshold})"
    )

    if verbose:
        print(f"After baylor exc threshold, # of {cell_type} cells = {len(df_filt)}")

    # if there was a gnn classification it needs to be high fidelity
    df_filt= df_filt.query(
        f"((gnn_cell_type_coarse == '{cell_type}') and (gnn_cell_type_fine_prob >= {gnn_prob_threshold}))"
        f"or (gnn_cell_type_coarse != gnn_cell_type_coarse)"
    )

    if verbose:
        print(f"After gnn exc threshold, # of {cell_type} cells = {len(df_filt)}")

    return df_filt
    
def excitatory_high_probability_df_from_df(
    df,
    baylor_exc_prob_threshold = 0.65,
    gnn_prob_threshold = 0.65,
    verbose = False,
    ):

    return cell_type_high_probability_df_from_df(
        df=df,
        cell_type='excitatory',
        baylor_exc_prob_threshold = baylor_exc_prob_threshold,
        gnn_prob_threshold = gnn_prob_threshold,
        verbose = verbose,
    )

# ---------- Setting of parameters ---------- 

attributes_dict_default = dict(
)    

global_parameters_dict_default = dict(
    lower_width_bound_spine = 140,
    upper_width_bound_spine = 520,
    
    apply_hand_made_low_rules = False,
    skeletal_length_processed_syn_min = 15_000,
    skeletal_length_processed_spine_min = 15_000,
    inhibitory_syn_density_shaft_min = 0.4,

    # spine densitys limb branch dict
    spine_threshold_syn_density = -1,
    skeletal_length_threshold_syn_density = 5000,
    upper_width_bound_syn_density = 10000,
)

global_parameters_dict_microns = {}
attributes_dict_microns = {}

attributes_dict_h01 = dict(
)

global_parameters_dict_h01 = dict(
    lower_width_bound_spine = 300,
    upper_width_bound_spine = 800,
)

# data_type = "default"
# algorithms = None
# modules_to_set = [ctu, (spu,"head_neck_shaft")]

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
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import spine_utils as spu
from . import synapse_utils as syu

#--- from machine_learning_tools ---
try:
    from machine_learning_tools import machine_learning_utils as mlu
    from machine_learning_tools import visualizations_ml as vml
except:
    mlu = None
    vml = None

#--- from mesh_tools ---
from mesh_tools import trimesh_utils as tu

#--- from python_tools ---
from python_tools import general_utils as gu
from python_tools import matplotlib_utils as mu
from python_tools import module_utils as modu 
from python_tools import numpy_dep as np
from python_tools import pandas_utils as pu
from python_tools import pathlib_utils as plu
from python_tools import string_utils as stru
from python_tools import system_utils as su

from . import cell_type_utils as ctu

set_e_i_model()