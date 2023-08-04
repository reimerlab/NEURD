'''



How this list was easily generated




'''
import datajoint as dj
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from python_tools import numpy_dep as np

neuron_attributes = [
 'multiplicity',
 'cell_type_used',
 'cell_type',
 'nucleus_id',
 'nuclei_distance',
 'n_nuclei_in_radius',
 'n_nuclei_in_bbox',
 'centroid_x',
 'centroid_y',
 'centroid_z',
 'centroid_x_nm',
 'centroid_y_nm',
 'centroid_z_nm',
 'max_soma_n_faces',
 'max_soma_volume',
 'max_soma_area',
 'syn_density_post_after_proof',
 'syn_density_head_after_proof',
 'syn_density_neck_after_proof',
 'syn_density_shaft_after_proof',
 'skeletal_length_processed_syn_after_proof',
 'spine_density_after_proof',
 'skeletal_length_processed_spine_after_proof',
 'baylor_e_i_after_proof',
 'baylor_e_i',
 'allen_e_i',
 'cell_type_used_for_axon',
 'cell_type_for_axon',
 'allen_e_i_n_nuc',
 'allen_cell_type',
 'allen_cell_type_n_nuc',
 'allen_cell_type_e_i',
 'axon_angle_max',
 'axon_angle_min',
 'n_axon_angles',
 'n_vertices',
 'n_faces',
 'n_not_processed_soma_containing_meshes',
 'n_error_limbs',
 'n_same_soma_multi_touching_limbs',
 'n_multi_soma_touching_limbs',
 'n_somas',
 'n_limbs',
 'n_branches',
 'max_limb_n_branches',
 'skeletal_length',
 'max_limb_skeletal_length',
 'median_branch_length',
 'width_median',
 'width_no_spine_median',
 'width_90_perc',
 'width_no_spine_90_perc',
 'n_spines',
 'n_boutons',
 'spine_density',
 'spines_per_branch',
 'skeletal_length_eligible',
 'n_spine_eligible_branches',
 'spine_density_eligible',
 'spines_per_branch_eligible',
 'total_spine_volume',
 'spine_volume_median',
 'spine_volume_density',
 'spine_volume_density_eligible',
 'spine_volume_per_branch_eligible',
 'dendrite_skeletal_length',
 'dendrite_area',
 'dendrite_mesh_volume',
 'dendrite_n_branches',
 'axon_skeletal_length',
 'axon_area',
 'axon_mesh_volume',
 'axon_n_branches',
 'basal_skeletal_length',
 'basal_area',
 'basal_mesh_volume',
 'basal_n_branches',
 'apical_skeletal_length',
 'apical_area',
 'apical_mesh_volume',
 'apical_n_branches',
 'apical_tuft_skeletal_length',
 'apical_tuft_area',
 'apical_tuft_mesh_volume',
 'apical_tuft_n_branches',
 'apical_shaft_skeletal_length',
 'apical_shaft_area',
 'apical_shaft_mesh_volume',
 'apical_shaft_n_branches',
 'oblique_skeletal_length',
 'oblique_area',
 'oblique_mesh_volume',
 'oblique_n_branches',
 'apical_total_skeletal_length',
 'apical_total_area',
 'apical_total_mesh_volume',
 'apical_total_n_branches',
 'n_syn_valid',
 'n_syn_valid_pre',
 'n_syn_valid_post',
 'n_syn_error',
 'n_syn_error_pre',
 'n_syn_error_post',
 'n_syn_presyns_on_dendrite',
 'n_syn_mesh_errored',
 'n_syn_distance_errored',
 'n_syn_no_label',
 'n_syn_head',
 'n_syn_neck',
 'n_syn_shaft',
 'n_syn_no_head',
 'n_syn_bouton',
 'n_syn_non_bouton',
 'n_syn_dendrite',
 'n_syn_axon',
 'n_syn_basal',
 'n_syn_apical',
 'n_syn_apical_tuft',
 'n_syn_apical_shaft',
 'n_syn_oblique',
 'n_syn_soma',
 'n_syn_apical_total',
 'n_syn_dendrite_head_postsyn',
 'n_syn_dendrite_neck_postsyn',
 'n_syn_dendrite_shaft_postsyn',
 'n_syn_dendrite_no_head_postsyn',
 'n_syn_axon_bouton_presyn',
 'n_syn_axon_bouton_postsyn',
 'n_syn_axon_non_bouton_presyn',
 'n_syn_axon_non_bouton_postsyn',
 'n_syn_basal_head_postsyn',
 'n_syn_basal_neck_postsyn',
 'n_syn_basal_shaft_postsyn',
 'n_syn_basal_no_head_postsyn',
 'n_syn_apical_head_postsyn',
 'n_syn_apical_neck_postsyn',
 'n_syn_apical_shaft_postsyn',
 'n_syn_apical_no_head_postsyn',
 'n_syn_apical_tuft_head_postsyn',
 'n_syn_apical_tuft_neck_postsyn',
 'n_syn_apical_tuft_shaft_postsyn',
 'n_syn_apical_tuft_no_head_postsyn',
 'n_syn_apical_shaft_head_postsyn',
 'n_syn_apical_shaft_neck_postsyn',
 'n_syn_apical_shaft_shaft_postsyn',
 'n_syn_apical_shaft_no_head_postsyn',
 'n_syn_oblique_head_postsyn',
 'n_syn_oblique_neck_postsyn',
 'n_syn_oblique_shaft_postsyn',
 'n_syn_oblique_no_head_postsyn',
 'n_syn_soma_no_label_presyn',
 'n_syn_soma_no_label_postsyn',
 'n_syn_apical_total_head_presyn',
 'n_syn_apical_total_head_postsyn',
 'n_syn_apical_total_neck_presyn',
 'n_syn_apical_total_neck_postsyn',
 'n_syn_apical_total_shaft_presyn',
 'n_syn_apical_total_shaft_postsyn',
 'n_syn_apical_total_no_head_presyn',
 'n_syn_apical_total_no_head_postsyn',
 'axon_branch_length_median',
 'axon_branch_length_mean',
 'axon_n_short_branches',
 'axon_n_long_branches',
 'axon_n_medium_branches',
 'axon_bbox_volume',
 'axon_bbox_x_min',
 'axon_bbox_y_min',
 'axon_bbox_z_min',
 'axon_bbox_x_max',
 'axon_bbox_y_max',
 'axon_bbox_z_max',
 'axon_bbox_x_min_soma_relative',
 'axon_bbox_y_min_soma_relative',
 'axon_bbox_z_min_soma_relative',
 'axon_bbox_x_max_soma_relative',
 'axon_bbox_y_max_soma_relative',
 'axon_bbox_z_max_soma_relative',
 'apical_branch_length_median',
 'apical_branch_length_mean',
 'apical_n_short_branches',
 'apical_n_long_branches',
 'apical_n_medium_branches',
 'apical_bbox_volume',
 'apical_bbox_x_min',
 'apical_bbox_y_min',
 'apical_bbox_z_min',
 'apical_bbox_x_max',
 'apical_bbox_y_max',
 'apical_bbox_z_max',
 'apical_bbox_x_min_soma_relative',
 'apical_bbox_y_min_soma_relative',
 'apical_bbox_z_min_soma_relative',
 'apical_bbox_x_max_soma_relative',
 'apical_bbox_y_max_soma_relative',
 'apical_bbox_z_max_soma_relative',
 'basal_branch_length_median',
 'basal_branch_length_mean',
 'basal_n_short_branches',
 'basal_n_long_branches',
 'basal_n_medium_branches',
 'basal_bbox_volume',
 'basal_bbox_x_min',
 'basal_bbox_y_min',
 'basal_bbox_z_min',
 'basal_bbox_x_max',
 'basal_bbox_y_max',
 'basal_bbox_z_max',
 'basal_bbox_x_min_soma_relative',
 'basal_bbox_y_min_soma_relative',
 'basal_bbox_z_min_soma_relative',
 'basal_bbox_x_max_soma_relative',
 'basal_bbox_y_max_soma_relative',
 'basal_bbox_z_max_soma_relative',
 'dendrite_branch_length_median',
 'dendrite_branch_length_mean',
 'dendrite_n_short_branches',
 'dendrite_n_long_branches',
 'dendrite_n_medium_branches',
 'dendrite_bbox_volume',
 'dendrite_bbox_x_min',
 'dendrite_bbox_y_min',
 'dendrite_bbox_z_min',
 'dendrite_bbox_x_max',
 'dendrite_bbox_y_max',
 'dendrite_bbox_z_max',
 'dendrite_bbox_x_min_soma_relative',
 'dendrite_bbox_y_min_soma_relative',
 'dendrite_bbox_z_min_soma_relative',
 'dendrite_bbox_x_max_soma_relative',
 'dendrite_bbox_y_max_soma_relative',
 'dendrite_bbox_z_max_soma_relative',
]

V1_bounds = [
    (280053, 322718, 14850),
    (230308, 322718, 27858),
    (52770, 322718, 27858),
    (52770, 322718, 14896),
    (230308, 60616, 27858),
    (52771, 60616, 27858),
    (280053, 60616, 14850),
    (52771, 60616, 14850),
    (266380, 60616, 18837),
    (266380, 322718, 18837),
    (252036, 60616, 23240),
    (252036, 322718, 23240),
    (239706, 60616, 26108),
    (239706, 322718, 26108)
    ]

RL_bounds1 = [
    (280053, 322718, 14850),
    (280053, 60616, 14850),
    (291053, 60616, 14850),
    (291053, 322718, 14850),
    (304555, 60616, 16084),
    (304555, 322718, 16084),
    (314280, 60616, 18176),
    (314280, 322718, 18176),
    (325895, 60616, 19816),
    (325895, 322718, 19816),
    (341016, 60616, 21822),
    (341016, 322718, 21822),
    (375347, 60616, 27904),
    (375347, 322718, 27904),
    (230308, 60616, 27858),
    (230308, 322718, 27858),
    (266380, 60616, 18837),
    (266380, 322718, 18837),
    (252036, 60616, 23240),
    (252036, 322718, 23240),
    (239706, 60616, 26108),
    (239706, 322718, 26108)
]
RL_bounds2= [
    (341016, 60616, 21822),
    (341016, 322718, 21822),
    (375347, 60616, 27904),
    (375347, 322718, 27904),
    (358451, 60616, 22439),
    (358451, 322718, 22439),
    (375347, 60616, 22722),
    (375347, 322718, 22722),
]

AL_bounds = [
    (307453, 60616, 16516),
    (307453, 322718, 16516),
    (315126, 60616, 14816),
    (315126, 322718, 14816),
    (375347, 60616, 14816),
    (375347, 322718, 14816),
    (314280, 60616, 18176),
    (314280, 322718, 18176),
    (325895, 60616, 19816),
    (325895, 322718, 19816),
    (341016, 60616, 21822),
    (341016, 322718, 21822),
    (358451, 60616, 22439),
    (358451, 322718, 22439),
    (375347, 60616, 22722),
    (375347, 322718, 22722)
]

volume_bound_coordinates = np.vstack([V1_bounds,RL_bounds1,RL_bounds2,AL_bounds])

layer_axis = 1 #the lower the number the higher the cell is in the volume

top_of_layer_vector =  np.array([0,-1,0])

def coordinates_to_layer_height(coordinates,turn_negative=True):
    if coordinates.ndim == 1:
        new_coords = coordinates[layer_axis]
    elif coordinates.ndim == 2:
        new_coords = coordinates[:,layer_axis]
    else:
        raise Exception(f"coordinates.ndim == {coordinates.ndim}")
        
    if turn_negative:
        new_coords = -1*new_coords
    return new_coords
    
def plot_visual_area_xz_projection(
                                  region_names = ["V1","RL","AL"],
                                  region_colors =  ["Blues","Greens","Reds"],
                                verbose=False,):
    """
    Purpose: To plot the triangulation used for the regions of the visual areas
        
    Example:
    import microns_utils as mru
    mru.plot_visual_area_xz_projection(verbose=True)
    
    """
    if verbose:
        for rn,rc in zip(region_names,region_colors):
            print(f"{rn}= {rc}")
            
    region_names = np.array(region_names)
    
    V1_Tri = Delaunay(V1_bounds)
    RL_a_Tri = Delaunay(RL_bounds1)
    RL_b_Tri = Delaunay(RL_bounds2)
    AL_Tri = Delaunay(AL_bounds)
    
    triangulation_list = [V1_Tri,RL_a_Tri,RL_b_Tri,AL_Tri]
    triangulation_points = [np.array(k) for k in [V1_bounds,RL_bounds1,RL_bounds2,AL_bounds]]

    triangulation_color_list = []
    for r_n in ["V1","RL","AL"]:
        curr_color = region_colors[np.where(region_names==r_n)[0][0]]
        
        if r_n == "RL":
            curr_color = [curr_color]*2
        else:
            curr_color = [curr_color]
            
        triangulation_color_list += curr_color
        
    if verbose:
        print(f"triangulation_color_list = {triangulation_color_list}")
    
    fig,ax = plt.subplots()
    for tri_points,tri_col in zip(triangulation_points,triangulation_color_list):
        tri = Delaunay(np.array(tri_points)[:,[0,2]])

        tri_x = tri.points[:,0]
        tri_y = tri.points[:,1]
        facecolors = np.zeros(len(tri.simplices))
        ax.tripcolor(tri_x,
                     tri_y, 
                     tri.simplices, 
                     facecolors=facecolors, 
                     cmap=tri_col,
                     edgecolors='k')

    plt.show()
    
def EM_coordinates_to_visual_areas(coordinates):
    """
    Purpose: To use the boundary points to classify 
    a list of points (usually representing soma centroids)
    in visual area classification (V1,AL,RL)
    
    Ex: 
    centroid_x,centroid_y,centroid_z = minnie.AutoProofreadNeurons3.fetch("centroid_x","centroid_y","centroid_z")
    soma_centers = np.vstack([centroid_x,centroid_y,centroid_z ]).T
    mru.EM_coordinates_to_visual_areas(soma_centers)
    
    """
    point = coordinates
    V1_contains = Delaunay(V1_bounds).find_simplex(point) >= 0 # V1
    RL_contains = np.logical_or(
        Delaunay(RL_bounds1).find_simplex(point)>=0, 
        Delaunay(RL_bounds2).find_simplex(point)>= 0) # RL
    AL_contains = Delaunay(AL_bounds).find_simplex(point) >= 0 # AL

    visual_areas = np.array(["V1","RL","AL"])
    contains_mask = np.vstack([V1_contains,RL_contains,AL_contains]).T
    point_visual_areas = visual_areas[np.argmax(contains_mask,axis=1)]
    return point_visual_areas
    
layer_by_max_height_voxel = {
"LAYER_1":0,
"LAYER_2/3": 100000, 
"LAYER_4":147000,
"LAYER_5": 168500,
"LAYER_6": 224000,
"WHITE_MATTER": 265000
}

layer_by_max_height_nm = {k:v*4 for k,v in layer_by_max_height_voxel.items()}

def EM_coordinates_to_layer(coordinates):
    """
    Purpose: To convert the y value of the EM coordinate(s)
    to the layer in the volume it is located
    
    """
    layer_names = np.array(["LAYER_1","LAYER_2/3","LAYER_4","LAYER_5","LAYER_6","WHITE_MATTER"])
    bins = [100000,147000,168500,224000,265000]
    
    coordinates = coordinates.reshape(-1,3)
    return layer_names[np.digitize(coordinates[:,1],bins)]    


def add_node_attributes_to_proofread_graph(
    G,
    neuron_data_df,
    attributes=None,
    add_visual_area=True,
    debug=False
    ):
    """
    Pseudocode: 
    1) Download all of the attributes want to store in the nodes
    2) Create a dictionar mapping the nuclei to a dict of attribute values
    3) set the attributes of the original graph
    """
    if attributes is None:
        attributes = [
            "spine_category",
            "cell_type_predicted",
            "n_axons",
            "axon_length",
            "n_apicals",
            "n_spines",
            "n_boutons",
            "n_nuclei_in_radius",
            "skeletal_length",

        ] 
        
    if "nucleus_id" not in attributes:
        attributes.append("nucleus_id")
    
    if add_visual_area:
        for s_t in ["centroid_x","centroid_y","centroid_z"]:
            if s_t not in attributes:
                attributes.append(s_t)

    #neuron_data = du.proofreading_neurons_table().fetch(*attributes,as_dict=True)
    
    neuron_data = neuron_data_df[attributes]
    
    if add_visual_area:
        soma_points = np.array([[k["centroid_x"],k["centroid_y"],k["centroid_z"]] for k in neuron_data])
        visual_area_labels = mru.EM_coordinates_to_visual_areas(soma_points)
        layer_labels = mru.EM_coordinates_to_layer(soma_points)
        if debug:
            print(f"soma_points.shape = {soma_points.shape}")
            print(f"visual_area_labels.shape = {visual_area_labels.shape}")
    
    attr_dict = dict()
    for j,k in enumerate(neuron_data):
        curr_dict = {k1:v for k1,v in k.items() if k != "nucleus_id"}
        
        if add_visual_area:
            curr_dict["visual_area"] = visual_area_labels[j]
            curr_dict["layer"] = layer_labels[j]
            
        attr_dict[k["nucleus_id"]] =  curr_dict

    nx.set_node_attributes(G, attr_dict)
    return G

def neuron_soma_layer_height(neuron_obj,soma_name="S0"):
    return mru.coordinates_to_layer_height(neuron_obj["S0"].mesh_center)

voxel_to_nm_scaling = np.array([4,4,40])
def em_voxels_to_nm(data):
    return np.array(data)*voxel_to_nm_scaling
def nm_to_em_voxels(data):
    return np.array(data)/voxel_to_nm_scaling


def visual_area_from_em_centroid_xyz(row):
    soma_points = np.array([row["centroid_x"],row["centroid_y"],row["centroid_z"]])
    visual_area_labels = mru.EM_coordinates_to_visual_areas(soma_points)[0]
    return visual_area_labels


def layer_from_em_centroid_xyz(row):
    soma_points = np.array([row["centroid_x"],row["centroid_y"],row["centroid_z"]])
    layer_labels = mru.EM_coordinates_to_layer(soma_points)[0]
    return layer_labels


microns_volume_coordinates = np.array(V1_bounds + 
                     RL_bounds1 + 
                     RL_bounds2 + 
                     AL_bounds)

def microns_volume_bbox_corners(return_nm=True):
    bbox_corners = nu.bouning_box_corners(microns_volume_coordinates)
    if return_nm:
        return em_voxels_to_nm(bbox_corners)
    return bbox_corners

def microns_volume_bbox_midpoint(return_nm=True):
    bbox_corners = nu.bouning_box_midpoint(microns_volume_coordinates,)
    if return_nm:
        return em_voxels_to_nm(bbox_corners)
    return bbox_corners

def distance_from_microns_volume_bbox_midpoint(coordinates):
    coordinates = np.array(coordinates)
    orig_dim = np.array(coordinates).ndim
    coordinates = coordinates.reshape(-1,3)
    dist_returned = np.linalg.norm(coordinates-microns_volume_bbox_midpoint(),axis=1)
    if orig_dim == 1:
        return dist_returned[0]
    return dist_returned
    
    
def soma_distances_from_microns_volume_bbox_midpoint(neuron_obj,
                                                     return_dict = True):
    """
    Purpose: To return the distances of each some from the middle of the volume
    
    Ex: 
    mru.soma_distances_from_microns_volume_bbox_midpoint(neuron_obj,
                                                    return_dict=False)
    """
    soma_names = neuron_obj.get_soma_node_names()
    soma_distances = {k:mru.distance_from_microns_volume_bbox_midpoint(neuron_obj[k].mesh_center)
                     for k in soma_names}
    
    if return_dict:
        return soma_distances
    else:
        return list(soma_distances.values())
    
def em_alignment_data_raw(
    return_dict = True,
    ):
    """

    """
    m65em = dj.create_virtual_module('minnie_em', 'microns_minnie_em_v2')
    max_alignment = np.max(m65em.EM().fetch("alignment"))
    curr_em_table = m65em.EM() & dict(alignment = max_alignment)
    
    if return_dict:
        return curr_em_table.fetch1()
    else:
        return curr_em
    
def em_alignment_coordinates_info(
    return_nm = True):
    '''
    Purpose: To get the center points and max points
    and all the labels associated

    Pseudocode: 
    1) Get the center,max,min,min anat and max anat
    '''
    align_dict = em_alignment_data_raw(return_dict = True)
    center_pt = np.array([align_dict[f"ctr_pt_{ax}"] for ax in ['x','y','z']])
    min_pt = np.array([align_dict[f"min_pt_{ax}"] for ax in ['x','y','z']])
    min_labels = np.array([align_dict[f"min_pt_{ax}_anat"] for ax in ['x','y','z']])
    max_pt = np.array([align_dict[f"max_pt_{ax}"] for ax in ['x','y','z']])
    max_labels = np.array([align_dict[f"max_pt_{ax}_anat"] for ax in ['x','y','z']])

    curr_dict = dict(
        center_pt=center_pt,
        min_pt=min_pt,
        max_pt=max_pt,
        min_labels=min_labels,
        max_labels=max_labels,
    )

    if return_nm: 
        for k,v in curr_dict.items():
            if "pt" in k:
                curr_dict[k] = v*voxel_to_nm_scaling

    return curr_dict


def align_mesh(
                mesh,
                soma_center=None,
                verbose = False):
    return mesh

def align_skeleton(
                skeleton,
                soma_center=None,
                verbose = False):
    return skeleton

def align_array(
                array,
                soma_center=None,
                verbose = False):
    return array

def align_neuron_obj(neuron_obj,**kwargs):
    return neuron_obj

def unalign_neuron_obj(neuron_obj,**kwargs):
    return neuron_obj

from . import volume_utils
class DataInterface(volume_utils.DataInterface):
    def __init__(self,**kwargs):
        super().__init__(
            **kwargs
        )
    
    def align_array(self,*args,**kwargs):
        return align_array(*args,**kwargs)
    
    def align_mesh(self,*args,**kwargs):
        return align_mesh(*args,**kwargs)
    
    def align_skeleton(self,*args,**kwargs):
        return align_skeleton(*args,**kwargs)
    
    def align_neuron_obj(self,*args,**kwargs):
        return align_neuron_obj(*args,**kwargs)

    def unalign_neuron_obj(self,*args,**kwargs):
        return unalign_neuron_obj(*args,**kwargs) 
    
    def segment_id_to_synapse_dict(
        self,
        segment_id = None,
        synapse_filepath=None,
        **kwargs
        ):
        
        return super().segment_id_to_synapse_dict(
            synapse_filepath=synapse_filepath,
            segment_id = segment_id,
            **kwargs
        )
        
        # if synapse_filepath is None:
        #     raise Exception("")
        # return syu.synapse_dict_from_synapse_csv(
        #     synapse_filepath=synapse_filepath,
        #     segment_id = segment_id,
        #     **kwargs
        # )

data_interface = DataInterface(
    source = "microns",
    voxel_to_nm_scaling = voxel_to_nm_scaling
)

#--- from neurd_packages ---
from . import volume_utils


#--- from python_tools ---
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu

from . import microns_volume_utils as mru
