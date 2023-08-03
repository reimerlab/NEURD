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
    red_blue_splits,
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
        
# --- from python_tools ---
from python_tools import ipyvolume_utils as ipvu

# --- from neurd ---
from . import neuron_utils as nru

from . import soma_splitting_utils as ssu