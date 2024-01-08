import numpy as np

def red_blue_split_dict_by_limb_from_red_blue_split_results(
    red_blue_splits,
    ):
    examples = []
    neuroglancer_links = dict()
    for limb_idx,limb_split_info in red_blue_splits.items():
        neuroglancer_links[limb_idx] = []
        for red_blue_split_info in limb_split_info:
            curr_keys = list(red_blue_split_info.keys())
            curr_title = f'{curr_keys[0]} to {curr_keys[1]}'
            
            red_blue_suggestions = red_blue_split_info[curr_keys[0]]
            neuroglancer_links[limb_idx].append(red_blue_suggestions)

    return neuroglancer_links



def limb_red_blue_dict_from_red_blue_splits(
    red_blue_split_results,
    attributes = (
        "valid_points",
        "error_points",
        "coordinate",
    ),
    stack_all_attributes = True,
    ):
    """
    a dictionary data structure that stores for each limb
    - valid points: coordinates that should belong to the existing neuronal process ( a marker of where the valid mesh is). 
    - error points: coordinates that should belong to incorrect neuronal process resulting from merge errors ( a marker of where the error mesh starts)
    - coordinate: locations of split points used in the elimination of soma to soma paths

    The valid and error points can be used as inputs for automatic mesh splitting algorithms in other pipelines (ex: Neuroglancer)
    """
    limb_red_blue_dict = ssu.red_blue_split_dict_by_limb_from_red_blue_split_results(
        red_blue_split_results
    )

    limb_dict = dict()
    for limb_idx,limb_data in limb_red_blue_dict.items():
        limb_dict[limb_idx] = {k:[] for k in attributes}
        for d in limb_data:
            for k,v in d.items():
                for vv in v:
                    for a in attributes:
                        limb_dict[limb_idx][a].append(vv[a])

        if stack_all_attributes:
            for a in attributes:
                limb_dict[limb_idx][a] = np.vstack(limb_dict[limb_idx][a])
    return limb_dict


def plot_red_blue_split_suggestions_per_limb(
    neuron_obj,
    red_blue_splits=None,
    split_results = None,
    plot_cut_paths = True,
    plot_red_blue_points = True,
    plot_skeleton = True,
    valid_color = "blue",
    error_color = "red",
    coordinate_color = "yellow",
    path_color = "green",
    valid_size = 0.3,
    error_size = 0.3,
    coordinate_size = 1.0,
    path_size = 0.3,
    verbose = True,
    plot_somas = True,
    soma_color = "orange",
    **kwargs
    ):
    """
    Purpose: to plot the splits for each limb based on the split results
    
    Pseudocode: 
    1) generate the red
    iterate through the limbs
    a. gather the valid points, error points, coordinates
    b. use plot object to plot the limb
    
    """
    
    if red_blue_splits is None:
        red_blue_splits = getattr(neuron_obj,"red_blue_split_results",None)
        
    if red_blue_splits is None:
        return None
    
    limb_red_blue_dict = limb_red_blue_dict_from_red_blue_splits(
        red_blue_splits
    )
    
    cut_paths_dict = None
    if plot_cut_paths and split_results is not None:
        cut_paths_dict = ssu.path_to_cut_and_coord_dict_from_split_suggestions(
            split_results
        )
    
    for curr_limb_idx,points_info in limb_red_blue_dict.items():
        if verbose:
            print(f"\n\n-------- Suggestions for Limb {curr_limb_idx}------")
        
        mesh = neuron_obj[curr_limb_idx].mesh
        skeleton = neuron_obj[curr_limb_idx].skeleton
        
        valid_points = limb_red_blue_dict[curr_limb_idx]["valid_points"]
        error_points = limb_red_blue_dict[curr_limb_idx]["error_points"]
        coordinate = limb_red_blue_dict[curr_limb_idx]["coordinate"]
        
        scatters = [coordinate]
        scatters_colors = [coordinate_color]
        scatter_size = [coordinate_size]
        
        meshes = []
        meshes_colors = []
        
        if plot_somas:
            somas = nru.all_soma_meshes_from_limb(neuron_obj,curr_limb_idx)
            meshes += somas
            meshes_colors += [soma_color]*len(somas)
        
        if plot_red_blue_points:
            scatters += [valid_points,error_points]
            scatters_colors += [valid_color,error_color]
            scatter_size += [valid_size,error_size]
            
        if cut_paths_dict is not None:
            scatters += [cut_paths_dict[curr_limb_idx]["paths_to_cut"]]
            scatters_colors += [path_color]
            scatter_size += [path_size]
            
        
        
        ipvu.plot_objects(
            mesh,
            skeleton,
            scatters = scatters,
            scatters_colors = scatters_colors,
            scatter_size=scatter_size,
            meshes=meshes,
            meshes_colors=meshes_colors,
            **kwargs
            
        )
        
        
def path_to_cut_and_coord_dict_from_split_suggestions(
    split_results,
    return_total_coordinates = True,
    ):

    limb_dict = dict()
    for limb_idx,limb_info in split_results.items():
        limb_dict[limb_idx] = dict(
            paths_to_cut = [],
            coordinates = []

        )
        for soma_soma_cut_info in limb_info:
            limb_dict[limb_idx]["paths_to_cut"] += soma_soma_cut_info["paths_cut"]
            limb_dict[limb_idx]["coordinates"] += soma_soma_cut_info["coordinate_suggestions"]

        if return_total_coordinates:
            limb_dict[limb_idx]["paths_to_cut"] = np.vstack(limb_dict[limb_idx]["paths_to_cut"]).reshape(-1,3)
            limb_dict[limb_idx]["coordinates"] = np.vstack(limb_dict[limb_idx]["coordinates"]).reshape(-1,3)

    return limb_dict


def calculate_multi_soma_split_suggestions(
    neuron_obj,
    plot = False,
    store_in_obj = True,
    plot_intermediates = False,
    plot_suggestions = False,
    plot_cut_coordinates = False,
    only_multi_soma_paths = False,
    verbose = False,
    **kwargs
    ):
    
    (split_results,
    red_blue_split_results) = pru.multi_soma_split_suggestions(
        neuron_obj,
        plot_intermediates=plot_intermediates,
        plot_suggestions=plot_suggestions,
        plot_cut_coordinates = plot_cut_coordinates,
        verbose = verbose,
        **kwargs
    )
    
    n_paths_cut = pru.get_n_paths_cut(
        split_results,
        verbose = True)
    
    split_products = pipeline.StageProducts(
        split_results=split_results,
        red_blue_split_results=red_blue_split_results,
        n_paths_cut=n_paths_cut,
    )
    
    if store_in_obj:
        neuron_obj.pipeline_products.set_stage_attrs(
            split_products,
            stage = "multi_soma_split_suggestions"
        )
        
    if plot:
        plot_red_blue_split_suggestions_per_limb(
            neuron_obj,
            red_blue_splits=red_blue_split_results,
            
        )
        
    return split_products

from copy import deepcopy
def multi_soma_split_execution(
    neuron_obj,
    split_results = None,
    verbose = False,
    store_in_obj = True,
    ):
    """
    Purpose: to execute the multi-soma
    split suggestions on the neuron (if
    not already generated then generate)
    """

    if split_results is None:
        try:
            split_results = neuron_obj.split_results
        except Exception as e:
            print(e)
            _ = ssu.calculate_multi_soma_split_suggestions(
                neuron_obj,
                store_in_obj = True,
            )

    (neuron_list,
    neuron_list_errored_limbs_area,
    neuron_list_errored_limbs_skeletal_length,
    neuron_list_n_multi_soma_errors,
    neuron_list_n_same_soma_errors) = pru.split_neuron(
        neuron_obj,
        limb_results=split_results,
        verbose=verbose,
        return_error_info=True,
    )


    stage_prods = []
    for idx,k in enumerate(neuron_list):
        k.pipeline_products = deepcopy(
            neuron_obj.pipeline_products
        )
        
        split_products = pipeline.StageProducts(
            split_index = idx,
            multi_soma_errored_limbs_area=neuron_list_errored_limbs_area[idx],
            multi_soma_errored_limbs_skeletal_length=neuron_list_errored_limbs_skeletal_length[idx],
            multi_soma_n_multi_soma_errors=neuron_list_n_multi_soma_errors[idx],
            multi_soma_n_same_soma_errors=neuron_list_n_same_soma_errors[idx],
            multiplicity = len(neuron_list)
        )
        
        if store_in_obj:
            k.pipeline_products.set_stage_attrs(
                split_products,
                stage = "multi_soma_split_execution"
            )
    

    return neuron_list



        
# --- from datasci_tools ---
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import pipeline

# --- from neurd ---
from . import neuron_utils as nru
from . import proofreading_utils as pru

from . import soma_splitting_utils as ssu