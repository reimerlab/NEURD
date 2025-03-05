'''



Purpose: To look at the angles and projection
angles of different compartments of neurons




'''
import matplotlib.pyplot as plt
import pandas as pd
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu

def add_xz_angles_to_df(
    df,
    compartments=("axon","dendrite","basal","apical")):
    
    """
    Purpose: To append the xz degree angle for all the vectors
    in all of the compartments
    """
    
    
    df_vec = df
    for comp in compartments:
        df_vec[f"{comp}_limb"] = [int(k.split("_")[0][1:]) if k is not None else None for k in df_vec[f"{comp}_node"].to_numpy()]#[:,0]

        names = [
            f"{comp}_skeleton_vector",
            f"{comp}_skeleton_vector_weighted",
            f"{comp}_soma_vector",
            f"{comp}_soma_vector_weighted"
        ]


        curr_map = df_vec.eval(f"{comp}_node == {comp}_node")
        for name in names:
            angle_name = f"{name}_xz_angle"
            df_vec[angle_name] = None

            df_vec.loc[curr_map,angle_name] = nu.angle_from_xy_vec(df_vec[[f"{name}_x_nm",f"{name}_z_nm"]][curr_map].to_numpy().T)


    return df_vec

def vec_df_from_compartment(
    df,
    compartment,
    verbose = True,
    align_array = True,
    centroid_df = None
    ):
    
    """
    To generate a smaller df that contains all of the vector information (and xz angles)
    for a given compartment. Also aligns the vectors correctly for the given dataset if requested
    
    Pseudocode: 
    1) Filters away nan rows
    2) Aligns the vectors 
    3) computes the xz angles
    """
    
    vector_names = [k for k in df.columns if (("angle" in k) or ("_nm" in k)) and compartment in k]
    #print(f"vector_names = {vector_names}")
    identifier_names = ["segment_id","split_index",f"{compartment}_node",f"{compartment}_n_limbs",f"{compartment}_width"]
    
    
    
    curr_df = df[identifier_names + vector_names]
    curr_df = pu.filter_away_nan_rows(curr_df)
    
    # adjusts all of the vectors using centroid alignment
    if align_array:
        if centroid_df is None:
            centroid_df = vdi.seg_split_centroid_df(nm=True)
        
        centroid_names = ["centroid_x_nm","centroid_y_nm","centroid_z_nm"]
        df_with_centr = pd.merge(curr_df,centroid_df,on=["segment_id","split_index"],how="left")
        vector_names = np.sort([k for k in df.columns if ( ("_nm" in k)) and compartment in k])
        #print(f"vector_names = {vector_names}")
        
        for i in range(len(vector_names)//3):
            curr_names = vector_names[i*3:(i*3)+3]
            
            new_array = []
            for curr_vec_array,centr in zip(df_with_centr[curr_names].to_numpy(),df_with_centr[centroid_names].to_numpy()):
                curr_new_array = vdi.align_array(curr_vec_array,centr)
                new_array.append(curr_new_array)
                
            curr_df[curr_names] = list(new_array)
        
    if verbose:
        print(f"{len(curr_df)} datapoints for {compartment}")
        
    return ngu.add_xz_angles_to_df(curr_df,compartments=[compartment])


def plot_compartment_vector_distribution(
    df,
    n_limbs_min = 4,
    compartment = "basal",
    axes = np.array([0,2]),
    normalize = True,
    
    # -- parameters for 1D histogram distribution
    plot_type="angle_360",
    bins = 100,
    
    title_suffix = None,
    verbose = True,
    ):
    """
    Purpose: To plot the 3D vectors or the 
    360 angle on a 1D histogram for a certain compartment
    over the dataframe

    Pseudocode: 
    1) Restrict the dataframe to only those cells with a certain number 
    of limbs in that compartment
    2) Get the compartment dataframe
    For each vector type
    a) Gets the vectors (restricts them to only certain axes of the vectors)
    b) Normalizes the vector (because sometimes will be less than one if restricting to less than 3 axes)
    c) 
    """
    df_vec_min_limb = df.query(f"({compartment}_n_limbs >= {n_limbs_min})")
    df_vec_curr = ngu.vec_df_from_compartment(df_vec_min_limb,compartment)


    names = [
        f"{compartment}_skeleton_vector",
        f"{compartment}_skeleton_vector_weighted",
        f"{compartment}_soma_vector",
        f"{compartment}_soma_vector_weighted"
    ]

    axes_names = np.array(["X","Y","Z"])
    combined_axes_name = "".join(axes_names[axes])

    
    for name in names:
        if verbose:
            print(f"\n\n----Plotting {name}----\n\n")

        #a) Gets the vectors for that name
        vector_coords = vdi.coordinates_from_df(
            df_vec_curr,
            name=name)[:,axes]

        #b) Normalizes the vectors
        if normalize:
            vector_coords = vector_coords/(np.linalg.norm(vector_coords,axis=1).reshape(-1,1))

        if len(axes) == 3:
            ipvu.plot_scatter(vector_coords)
        else:
            if plot_type=="angle_360":
                #plt.hist(nu.angle_from_xy_vec(vector_coords.T),bins=bins)
                angle_name = f"{name}_xz_angle"
                plt.hist(df_vec_curr[f"{name}_xz_angle"].to_numpy(),bins=bins)
                xlabel = f"{combined_axes_name} Angle (Degrees)"
            else:
                sml.hist2D(vector_coords.T[0],vector_coords.T[1])
                xlabel = f"{combined_axes_name} Vector"

            title = f"{name}"
            if title_suffix is not None:
                title = f"{title}\n{title_suffix}"
            plt.title(title)
            plt.xlabel(xlabel)
            plt.show()


    return df_vec_curr




# ------------- Setting up parameters -----------

# -- default 
attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling,
    vdi = mvu.data_interface
)    
global_parameters_dict_default = dict(
    #max_ais_distance_from_soma = 50_000
)

# -- microns
global_parameters_dict_microns = {}
attributes_dict_microns = {}

#-- h01--
attributes_dict_h01 = dict(
    voxel_to_nm_scaling = hvu.voxel_to_nm_scaling,
    vdi = hvu.data_interface
)
global_parameters_dict_h01 = dict()
    
       
# data_type = "default"
# algorithms = None
# modules_to_set = [ngu]

# def set_global_parameters_and_attributes_by_data_type(data_type,
#                                                      algorithms_list = None,
#                                                       modules = None,
#                                                      set_default_first = True,
#                                                       verbose=False):
#     if modules is None:
#         modules = modules_to_set
    
#     modu.set_global_parameters_and_attributes_by_data_type(modules,data_type,
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
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mvu

#--- from datasci_tools ---
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import module_utils as modu 
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu

from . import neuron_geometry_utils as ngu