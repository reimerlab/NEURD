"""

"""
"""
Purpose: To help analyze the morphology tables:

Applications: 
1) Analyze the vectors
"""

from python_tools import pandas_utils as pu
from python_tools import numpy_utils as nu

default_min_skeletal_length = 20_000
default_axes = (0,2)

n_limbs_min_by_compartment_dict = dict(
    basal = 2,
    oblique = 1,
    apical = 1,
    apical_shaft = 1,
    apical_tuft = 1,
    apical_trunk = 1,
)

default_n_limbs_min = n_limbs_min_by_compartment_dict["basal"]
# ---------------------- 8/24/ Vector/Compartment Analysis
default_vector_names = ("skeleton_vector","soma_vector","leaf_node_vector")

def restrict_df_by_n_limb(
    df,
    n_limbs_min,
    compartment = None,
    verbose = False,):
    """
    Purpose: To restrict a dataframe by the number of 
    limbs
    """
    if compartment is not None:
        df = df.query(f"compartment in {list(nu.array_like(compartment))}")
    
    n_limbs_df = pu.count_unique_column_values(
        df,["segment_id","split_index"],
        count_column_name = "n_limbs").query(
            f"n_limbs >= {n_limbs_min}"
    ).reset_index(drop=True)


    df_restr = pu.intersect_df(
        df,
        n_limbs_df,
        append_restr_columns=True,
    )

    if verbose:
        print(f"After n_limbs restriction ( >= {n_limbs_min}): {len(df_restr)} ")
        
    return df_restr


def subgraph_vector_df_restriction(
    df,
    skeletal_length_min = 10_000,
    compartment = None,#"basal",
    n_limbs_min = None,
    verbose = True,
    restrictions = None,
    ):
    
    """
    Purpose: To do some preprocssing restriction on subgraph
    vector table
    """

    if type(restrictions) == "str":
        restrictions = [restrictions]
    if restrictions is None:
        restrictions = []
    if skeletal_length_min is not None:
        restrictions.append(f"(skeletal_length >= {skeletal_length_min})")

    if compartment is not None:
        compartment = nu.convert_to_array_like(compartment)
        restrictions.append(f"(compartment in {list(compartment)})")

    
        

    df_restr = pu.restrict_df_from_list(
        df,
        restrictions,
        verbose = verbose
    )
    
    if n_limbs_min is not None:
        df_restr= mau.restrict_df_by_n_limb(df_restr,n_limbs_min)

    df_sort = pu.sort_df_by_column(
        df_restr,columns=["segment_id","split_index","compartment","width","skeletal_length"]
    )
    
    if verbose:
        print(f"Origianl length before subgraph restriction: {len(df)}")
        print(f" ----> After: {len(df_sort)}")
    
    return df_sort



def max_width_df(
    df,
    df_sort = None,
    verbose = False,
    n_limbs_min = None, 
    compartment=None,#"basal",
    
    # --- for adding alignment --
    add_angle = False,
    axes = None,
    **kwargs):
    """
    Purpose: Reduce the table to the maximum width table

    Pseudocode:
    1) Sort the table by seg,split,width
    """
    
#     if n_limbs_min is None:
#         n_limbs_min = default_n_limbs_min

    if df_sort is None:
        df_sort = mau.subgraph_vector_df_restriction(
            df,
            verbose = verbose,
            compartment=compartment,
            n_limbs_min=n_limbs_min,
            **kwargs
        )
        
    if n_limbs_min is None:
        if type(compartment) == str:
            n_limbs_min = n_limbs_min_by_compartment_dict[compartment]
        else:
            n_limbs_min = default_n_limbs_min
            
            
    df_sort = mau.restrict_df_by_n_limb(df_sort,n_limbs_min)

    df_max_width = pu.filter_to_first_instance_of_unique_column(
        df_sort,["segment_id","split_index","compartment"])
    
    if verbose:
        print(f"# of single neurons = {len(df_max_width)}")
    
    return_df = df_max_width.reset_index()
    
    if add_angle:
        #return_df = aligned_angle_vector_df(return_df,axes=axes,**kwargs)
        return_df = add_on_angles_2D_xz_and_xyz(return_df,**kwargs)
        
        
        
    return return_df

from python_tools import numpy_utils as nu
import time

def weighted_width_df_from_filtered_df(
    df,
    compartment = None,
    verbose = False,
    ):
    """
    Purpose: To create a grouped weighted
    vector (for maybe certain compartment)
    """
    if verbose:
        st = time.time()

    if compartment is not None:
        compartment = nu.to_list(compartment)
        curr_df = df.query(f"compartment in {compartment}").reset_index(drop=True)
    else:
        curr_df = df

    curr_df_w = pu.weighted_average_df(
        curr_df,
        weight_column = "width",
        columns_to_delete = ["node",],
        group_by_columns = ["segment_id","split_index","compartment"],
        verbose = verbose,
    )


    for vec_name in mau.default_vector_names:
        try:
            curr_df_w = pu.normalize_vector_magnitude(
                df = curr_df_w,
                name = vec_name,
            )
        except:
            if verbose:
                print(f"Did not normalize {vec_name}")

    if verbose:
        print(f"Total time for weighted compartment df = {time.time() - st}")

    return curr_df_w


def weighted_width_df(
    df,
    df_sort = None,
    verbose = False,
    n_limbs_min = None, 
    compartment=None,#"basal",
    
    # --- for adding alignment --
    add_angle = False,
    axes = None,
    **kwargs):
    """
    Purpose: Reduce the table to the maximum width table

    Pseudocode:
    1) Sort the table by seg,split,width
    """
    
#     if n_limbs_min is None:
#         n_limbs_min = default_n_limbs_min

    if df_sort is None:
        df_sort = mau.subgraph_vector_df_restriction(
            df,
            verbose = verbose,
            compartment=compartment,
            n_limbs_min=n_limbs_min,
            **kwargs
        )
        
    if n_limbs_min is None:
        if type(compartment) == str:
            n_limbs_min = n_limbs_min_by_compartment_dict[compartment]
        else:
            n_limbs_min = default_n_limbs_min
            
            
    df_sort = mau.restrict_df_by_n_limb(df_sort,n_limbs_min)

    df_max_width = weighted_width_df_from_filtered_df(
        df_sort,
        verbose = verbose
    )
    
    if verbose:
        print(f"# of single neurons = {len(df_max_width)}")
    
    return_df = df_max_width.reset_index()
    
    if add_angle:
        #return_df = aligned_angle_vector_df(return_df,axes=axes,**kwargs)
        return_df = add_on_angles_2D_xz_and_xyz(return_df,**kwargs)
        
        
    return return_df


# def vector_names_from_df(df):
#     vector_names = [k for k in df.columns if (("angle" in k) or ("_nm" in k))]
    
#     vector_names_final = []
#     for k in df.columns:
#         add_flag = False
#         if "angle" in k:
#             add_flag.

# --- the mean vector df ---
from python_tools import pandas_utils as pu
from python_tools import general_utils as gu

def vector_name_in_df(df,vector_stem,axes = ("x","y","z")):
    result = True
    for v_name in mau.vector_names_from_stem(vector_stem,axes=axes):
        if v_name not in df.columns:
            result = False
            break
            
    return result

def compartment_mean_vector_df(
    df,
    max_y_above = 200_000,
    n_limbs_min = 3,
    compartment = "oblique",
    **kwargs
    ):

    """
    Purpose: To have an average of the 
    oblique angle after filters
    """
    for k in df.columns:
        if type(df[k][0]) != str and k not in ["segment_id","split_index"]:
            df[k] = df[k].astype('float')


    df_restr = mau.subgraph_vector_df_restriction(
        df,
        compartment = compartment,
        n_limbs_min=n_limbs_min,
        restrictions = [f"y_soma_relative > {-1*max_y_above}"],
        **kwargs
    )

    #return df_restr
    """
    Purpose: Want to combined the df through 
    averaging the skeleton vectors
    """
    oblique_grouped = df_restr.groupby(["segment_id","split_index"]).aggregate(
        gu.merge_dicts([{f"{k}":["mean","std"] for k in mau.vector_names_from_stem("skeleton_vector")},
                       {k:"mean" for k in  mau.vector_names_from_stem("centroid")},
                       {"segment_id":"first","split_index":"first"}])
    )

    #oblique_grouped_flat = pu.flatten_column_multi_index(oblique_grouped)

    new_columns = []
    for k in oblique_grouped.columns.to_flat_index():
        split_idx = -5
        new_columns.append(f"{k[0][:split_idx]}_{k[1]}{k[0][split_idx:]}")
    oblique_grouped.columns = new_columns

    
    oblique_grouped_aligned = mau.aligned_angle_vector_df(
        oblique_grouped,
        vector_name=("skeleton_vector_mean","skeleton_vector_std"),
        centroid_stem = "centroid_mean",
        add_new_columns_for_align = True,
    )
    
    return oblique_grouped_aligned

def axes_as_str(axes=None):
    if axes is None:
        axes=(0,2)
    if type(axes[0]) != str:
        axes_names = ["x","y","z"]
        axes = [axes_names[k] for k in axes]
    return axes

def add_on_angles_2D(
    df,
    vector_name=None,
    axes=(0,2),
    suffix = "",
    rad = False,):
    """
    Purpose: To compute the angles of certain vectors in a dataframe
    """
#     if vector_names is None:
#         vector_names= mau.vector_names_from_df(df)
    if vector_name is None:
        vector_name = default_vector_names
        
    if axes is None:
        axes = default_axes
        
    if len(axes) != 2:
        raise Exception("")
    
    axes = axes_as_str(axes)
    
    vector_name = nu.convert_to_array_like(vector_name,include_tuple = True)
    #print(f"vector_name = {vector_name},axes = {axes}")
    for name in vector_name:
        if not vector_name_in_df(df,name,axes = axes):
            print(f"skipping")
            continue
                
        if not rad:
            angle_name = f"{name}_{''.join(axes)}_angle"
        else:
            angle_name = f"{name}_{''.join(axes)}_radians"
        curr_vec = df[[f"{name}_{axes[0]}_nm{suffix}",
                f"{name}_{axes[1]}_nm{suffix}"]].astype('float').to_numpy().T
        #print(f"curr_vec = {curr_vec}")
        df[angle_name] = nu.angle_from_xy_vec(curr_vec)
        if rad:
            df[angle_name] = df[angle_name]/180*np.pi

    return df

def add_on_angle_to_top(
    df,
    vector_name=None,
    vector_to_top = None,
    suffix = "",
    **kwargs
    ):
    
    if vector_to_top is None:
        vector_to_top = hdju.vector_to_top
    if vector_name is None:
        vector_name = default_vector_names
        
    vector_name = nu.convert_to_array_like(vector_name,include_tuple = True)
    for name in vector_name:
        if not vector_name_in_df(df,name):
            continue
        
        angle_name = f"{name}_to_top_angle"
        axes = ["x",'y',"z"]
        curr_vec = df[[f"{name}_{ax}_nm{suffix}" for ax in axes]].astype('float').to_numpy()
        #print(f"curr_vec = {curr_vec}")
        df[angle_name] = [nu.angle_between_vectors(
            vector_to_top,
            k,
        ) for k in curr_vec]
    
    return df

def add_on_angles_2D_xz_and_xyz(
    df,
    vector_name = None,
    **kwargs):
    subgraph_vec_df = mau.add_on_angles_2D(
        df,
        vector_name = vector_name,
        **kwargs
    )

    subgraph_vec_df = mau.add_on_angle_to_top(
        subgraph_vec_df,
        vector_name=vector_name,
        **kwargs
    )
    
    return subgraph_vec_df
    
def vector_names_from_stem(name,axes = ("x","y","z")):
    return [f"{name}_{ax}_nm" for ax in axes]

def aligned_angle_vector_df(
    df,
    align = True,
    in_place = False,
    vector_name = None,
    verbose = False,
    add_new_columns_for_align = False,
    align_suffix = "aligned",
    centroid_stem = "centroid",
    axes=None,
    **kwargs
    ):
    """
    Purpose: To Align vectors and add the angle to different 
    vector types
    """
    
    if not in_place:
        df = df.copy()
        
    if vector_name is None:
        vector_name = default_vector_names
        
    curr_df = pu.filter_away_nan_rows(df)
    vector_name = nu.convert_to_array_like(vector_name,include_tuple = True)
    vector_name = list(vector_name)
    
    if align:
        centroid_names = mau.vector_names_from_stem(centroid_stem)
        new_array_centroid = []
        centroid_names_align = mau.vector_names_from_stem(f"{centroid_stem}_{align_suffix}")
        
        to_add_names=  []
        for j,vn in enumerate(vector_name):
            if not vector_name_in_df(df,vn):
                continue
            
            new_array = []
            curr_names = mau.vector_names_from_stem(vn)
            curr_names_align = mau.vector_names_from_stem(f"{vn}_{align_suffix}")
            
            for curr_vec_array,centr in zip(
                curr_df[curr_names].to_numpy().copy(),
                curr_df[centroid_names].to_numpy().copy()
                ):
                curr_value = hdju.align_array(curr_vec_array,centr)
                #print(f"curr_value = {curr_value} from {curr_vec_array}, center = {centr}")
                new_array.append(curr_value)
                
                if j == 0:#len(vector_name)-1:
                    new_array_centroid.append(hdju.align_array(centr,centr))
            
            if add_new_columns_for_align:
                #print(f"Trying to align:: {f'{vn}_{align_suffix}''}")
                to_add_names.append(f"{vn}_{align_suffix}")
                curr_df[curr_names_align] = list(new_array)
                
                if j == 0:#len(vector_name)-1:
                    curr_df[centroid_names_align] = list(new_array_centroid)
            else:
                curr_df[curr_names] = list(new_array)
                if j == 0:#len(vector_name)-1:
                    curr_df[centroid_names] = list(new_array_centroid)
                
        vector_name += to_add_names
            
    
    return mau.add_on_angles_2D(
        curr_df,
        vector_name=vector_name,
        axes=axes)
            
    
import matplotlib.pyplot as plt
def plot_axes_angle_hist(
    df,
    vector_name = "skeleton_vector",
    angle_name = None,
    bins = 100,
    title = "",
    axes = None,
    figsize = (10,6),
    ):
    """
    Purpose: To plot the vector data as
    a histogram
    """
    axes = axes_as_str(axes)
    if angle_name is None:
        angle_name = f"{vector_name}_{''.join(axes)}_angle"
    fig,ax = plt.subplots(1,1,figsize = figsize)
    ax.hist(df[angle_name].to_numpy(),bins=bins)
    ax.set_title(title)
    ax.set_xlabel(f"{vector_name} {''.join(axes).upper()} Angle (Degrees)")
    return ax

def plot_width_aggr_compartment_aligned_vector(
    df,
    compartment = "basal",
    vector_name = "skeleton_vector_aligned",#"soma_vector_aligned",
    title = f"MICrONS Dataset",
    width_aggr_func = None,
    axes = (0,2),
    bins = 100,
    ):
    """
    Purpose: To plot the aligned compartment
    vector:

    Pseudocode: 
    1) Find the max width df
    2) Align the dataframe
    3) Plot the Compartment Distribution
    """
    if width_aggr_func is None:
        width_ggr_func = mau.max_width_df
    
    if title is None: 
        title = f"{compartment.title()}"
    else:
        title += f"\n{compartment.title()}"
    df_max_width = width_aggr_func(df,compartment=compartment)
    df_max_width_aligned = mau.aligned_angle_vector_df(
        df_max_width,
        align = True,
        add_new_columns_for_align = True,
    )
    
    #return df_max_width_aligned

    #for checking the alignment
    df_max_width_aligned[mau.vector_names_from_stem("centroid") + mau.vector_names_from_stem("centroid_aligned")]

    return mau.plot_axes_angle_hist(
        df = df_max_width_aligned,
        vector_name = vector_name,
        axes = axes,
        bins=bins,
        title = title)

def plot_max_width_compartment_aligned_vector(
    df,
    compartment = "basal",
    vector_name = "skeleton_vector_aligned",#"soma_vector_aligned",
    title = f"MICrONS Dataset",
    axes = (0,2),
    bins = 100,
    ):
    
    return plot_width_aggr_compartment_aligned_vector(
        df,
        compartment = compartment,
        vector_name = vector_name,#"soma_vector_aligned",
        title = title,
        width_aggr_func = mau.max_width_df,
        axes = axes,
        bins = bins,
        )

def plot_weighted_width_compartment_aligned_vector(
    df,
    compartment = "basal",
    vector_name = "skeleton_vector_aligned",#"soma_vector_aligned",
    title = f"MICrONS Dataset",
    axes = (0,2),
    bins = 100,
    ):
    
    return plot_width_aggr_compartment_aligned_vector(
        df,
        compartment = compartment,
        vector_name = vector_name,#"soma_vector_aligned",
        title = title,
        width_aggr_func = mau.weighted_width_df,
        axes = axes,
        bins = bins,
        )

from python_tools import ipyvolume_utils as ipvu
def plot_vector_df_quiver(
    df,
    vector_name="soma_vector",
    color_attribute = "soma_vector_xz_angle",
    size_by_width = True,
    centroid_name = "centroid",
    ):
    
    centers = pu.coordinates_from_df(df,centroid_name)
    vectors = pu.coordinates_from_df(df,vector_name)
    if size_by_width:
        widths = df["width"].to_numpy()/1000
    else:
        widths = None

    if color_attribute is not None:
        color_array = df[color_attribute].to_numpy()
    else:
        color_array = None
    
    ipvu.plot_quiver_with_gradients(
        centers,
        vectors,
        color_array=color_array,
        plot_colormap = True,
        size_array = widths,
        )
    
    
default_angle_match_vector_name = "skeleton_vector_xz_angle"
oblique_angle_feature = "skeleton_vector_to_top_angle"

def similarity_from_start_vector(
    subgraph_dict,
    matching_dict,
    vector_name=None,
    **kwargs
    ):
    if vector_name is None:
        vector_name = default_angle_match_vector_name
    return nu.cdist(matching_dict[vector_name],subgraph_dict[vector_name],rad = False)



def best_matching_dict_from_similarity_func(
    subgraph_dict,
    matching_subgraph_df,
    best_comparator = "min",
    similarity_func = None,#similarity_from_start_vector,scholl_distance_dict_similarity
    similarity_metric_name = None,
    verbose_time = False,
    #in_place = False,
    **kwargs
    ):
#     if not in_place:
#         matching_subgraph_df = matching_subgraph_df.copy(deep=True)
    
    if similarity_metric_name is None:
        similarity_metric_name = default_similarity_metric_name
    
    if similarity_func is None:
        similarity_func = default_similarity_func
        
    st = time.time()
    
    
    sim_df = []
    for j,matching_dict in enumerate(pu.df_to_dicts(matching_subgraph_df)):
        
        return_value = similarity_func(
                subgraph_dict=subgraph_dict,
                matching_dict=matching_dict,
                return_metadata = True,
                verbose_time = verbose_time,
                **kwargs,
            )
        if verbose_time:
            print(f"Time for similarity func # {j}: {time.time() - st}")
            st = time.time()
        
        
        if not nu.is_array_like(return_value,include_tuple = True):
            curr_sim = return_value
            curr_meta = dict()
        elif len(return_value) == 2:
            curr_sim, curr_meta = return_value
        else:
            raise Exception("")
        
        curr_meta = {f"{similarity_metric_name}_{k}":v for k,v in curr_meta.items()}
        
        curr_meta[similarity_metric_name] = curr_sim
        sim_df.append(curr_meta)
            
    sim_df = pd.DataFrame.from_records(sim_df)    
    

    matching_subgraph_df = pd.concat([matching_subgraph_df,sim_df],axis=1)
    curr_query = f"{similarity_metric_name} == {best_comparator.upper()}({similarity_metric_name})"
    #raise Exception("")
    #basal_min_dict = pu.query(matching_subgraph_df,(f"cdist == MIN(cdist)")).iloc[0,:].to_dict()
    basal_min_dict = pu.query(matching_subgraph_df,(curr_query)).iloc[0,:].to_dict()
    
    if verbose_time:
        print(f"Time combining dict # {j}: {time.time() - st}")
        st = time.time()
    
    return basal_min_dict


def plot_matched_dict(
    curr_dict,
    mesh = None,
    mesh_match = None,):
    if "match_leaf_node" in curr_dict:
        nodes = [curr_dict['match_leaf_node'],curr_dict['leaf_node']]
        nodes_as_leaf_nodes = True
    else:
        nodes = [curr_dict['match_node'],curr_dict['node']]
        nodes_as_leaf_nodes = False

    hdju.plot_meshes_node_skeletons_aligned(
        segment_ids = [curr_dict['match_segment_id'],curr_dict['segment_id']],
        split_indexes = [curr_dict['match_split_index'],curr_dict['split_index']],
        nodes = nodes,
        nodes_as_leaf_nodes=nodes_as_leaf_nodes,
        meshes = [mesh_match,mesh]
    )
    
def best_similarity_match_from_matching_df(
    subgraph_dict,
    matching_subgraph_df,
    subgraph_features_to_record =None,
    matching_feature_to_record = None,
    features_to_record = (
        "y_soma_relative",
        "skeletal_length",
        "width",
        "subgraph_idx",
        "compartment",
        "node",
        "segment_id",
        "split_index",
        "cell_type_predicted",
        "cell_type_predicted_prob",
        oblique_angle_feature,
        "leaf_node"
        ),
    #matching_func =best_matching_dict_from_start_vector_cdist,
    similarity_func = None,
    similarity_func_kwargs = None,
    verbose = False,
    verbose_time = False,
    plot = False,
    mesh = None,
    mesh_match = None,
    ):
    """
    Purpose: To get the minimum cdist matching
    branch between one branch data and another
    dataframe of branches
    """
    st = time.time()
    
    if matching_feature_to_record is None:
        matching_feature_to_record = []
    
    
    if subgraph_features_to_record is None:
        subgraph_features_to_record = []
    
    subgraph_features_to_record = list(subgraph_features_to_record) + list(features_to_record)
    matching_feature_to_record = list(matching_feature_to_record) + list(features_to_record)
    
    
    curr_dict = dict()
    curr_dict.update({k:v for k,v in subgraph_dict.items() if k in subgraph_features_to_record} )

    if verbose:
        print(f"--- Working on {curr_dict['segment_id']}_{curr_dict['split_index']} subgraph {curr_dict['subgraph_idx']} (compartment = {curr_dict['compartment']}, node = {curr_dict['node']})")

        
    # Determines the metric for each branch and then finds the dictionary of the best matching branch
    if similarity_func_kwargs is None:
        similarity_func_kwargs= {}
        
    
    basal_min_dict = best_matching_dict_from_similarity_func(
        subgraph_dict,
        matching_subgraph_df,
        similarity_func = similarity_func,
        verbose_time = verbose_time,
        **similarity_func_kwargs
    )
    basal_min_dict_feat = {f"match_{k}":v for k,v in basal_min_dict.items() if k in matching_feature_to_record
                          or default_similarity_metric_name in k}

    curr_dict.update(basal_min_dict_feat)
    
    if verbose:
        print(f"    Found match: {curr_dict['match_segment_id']}_{curr_dict['match_split_index']} subgraph = {curr_dict['match_subgraph_idx']}"
              f", node = {curr_dict['match_node']}, similarity = {curr_dict[f'match_similarity']}")
    
    # can rename the similariy function from the kwargs and the similarity func name
    
    if verbose_time:
        print(f"Time for best_similarity_match_from_matching_df = {time.time() - st}")
    
    if plot:
        mau.plot_matched_dict(
            curr_dict,
            mesh = mesh,
            mesh_match = mesh_match,)
    return curr_dict

import pandas as pd

default_right_angle_buffer = 20
'''
def oblique_right_angle_restr_old(buffer=None):
    """
    Purpose: To filter offshoots to certain angles from 90

    Pseudocode: 1) Restrict the skeletal angle
    to those within a buffer of 90 and 270
    """
    
    if buffer is None:
        buffer = default_right_angle_buffer

    restr = "or".join([f"({k})" for k in [
        "and".join([f"({oblique_angle_feature} >= {90 - buffer})",
        f"({oblique_angle_feature} <= {90 + buffer})"]),
        "and".join([f"({oblique_angle_feature} >= {270 - buffer})",
        f"({oblique_angle_feature} <= {270 + buffer})"]),]
    ])

    return restr
'''

def oblique_right_angle_restr(
    column = None,
    buffer=None,
    match=False):
    """
    Purpose: To filter offshoots to certain angles from 90

    Pseudocode: 1) Restrict the skeletal angle
    to those within a buffer of 90 and 270
    """
    
    if column is None:
        column = oblique_angle_feature
    
    if buffer is None:
        buffer = default_right_angle_buffer

    if match:
        prefix = "match_"
    else:
        prefix = ""
    restr = "and".join([
            f"({prefix}{oblique_angle_feature} >= {90 - buffer})",
            f"({prefix}{oblique_angle_feature} <= {90 + buffer})"
        ])

    return restr


def offshoot_vs_basal_scholl_match(

    ):
    pass
    """
    Purpose: Need to come up with a statistic that measures
    the average school distance
    """
    
def offshoot_vs_basal_df_preprocessing(
    subgraph_vec_df = None,
    min_skeletal_length_basal = None,
    offshoot_compartments = ("oblique","apical"),
    right_angle_restr = True,
    right_angle_buffer = None,
    min_skeletal_length_offshoot = None,
    offshoot_df = None,
    basal_df = None,
    
    plot = False,
    verbose = False,
    verbose_time = False,
    ):
    """
    3) For each offshoot branch (in the whole dataframe)
    - Find the argmin of the circular dist basal branch and record
       and record significant features about the pairing
    """
    if verbose_time:
        st = time.time()
        
    if min_skeletal_length_basal is None:
        min_skeletal_length_basal = default_min_skeletal_length
    
    if min_skeletal_length_offshoot is None:
        min_skeletal_length_offshoot = default_min_skeletal_length
    
    if basal_df is None:
        curr_df = subgraph_vec_df
    else:
        curr_df = basal_df
        
    basal_df = curr_df.query(
        f"(compartment == 'basal')"
        f"and (skeletal_length >= {min_skeletal_length_basal})"
    ).reset_index(drop=True)

    if offshoot_df is None:
        curr_df = subgraph_vec_df
    else:
        curr_df = offshoot_df
        
    offshoot_df = curr_df.query(
        f"(compartment in {offshoot_compartments})"
        f"and (skeletal_length >= {min_skeletal_length_offshoot})"
    ).reset_index(drop=True)
    
    if right_angle_restr and oblique_angle_feature in offshoot_df.columns:
        if verbose:
            print(f"Before right angle restriction, # of offshoots = {len(offshoot_df)}")
        offshoot_df = offshoot_df.query(oblique_right_angle_restr(right_angle_buffer))
        if verbose:
            print(f"After right angle restriction, # of offshoots = {len(offshoot_df)}")

    
    if plot:
        mesh = hdju.fetch_proofread_mesh(
            segment_id=basal_df["segment_id"].unique()[0],
            split_index=basal_df["split_index"].unique()[0]
        )
    else:
        mesh = None
        
    if verbose_time:
        print(f"Time for offshoot_vs_basal_df_preprocessing = {time.time() - st}")
        
    return basal_df,offshoot_df

def offshoot_df_vs_basal_df_best_similarity_match(
    subgraph_vec_df = None,
    min_skeletal_length_basal = None,
    offshoot_compartments = ("oblique","apical"),
    right_angle_restr = False,
    right_angle_buffer = None,
    min_skeletal_length_offshoot = None,
    offshoot_df = None,
    basal_df = None,
    
    similarity_func=None,
    similarity_func_kwargs=None,
    
    add_soma_soma_dist = True,
    add_endpoint_endpoint_dist = True,
    
    filter_away_0_scholl = True,
    
    plot = False,
    mesh = None,
    verbose = False,
    verbose_time = False,
    ):
    """
    3) For each offshoot branch (in the whole dataframe)
    - Find the argmin of the circular dist basal branch and record
       and record significant features about the pairing
    """
    
    basal_df,offshoot_df = mau.offshoot_vs_basal_df_preprocessing(
        subgraph_vec_df = subgraph_vec_df,
        min_skeletal_length_basal = min_skeletal_length_basal,
        offshoot_compartments = offshoot_compartments,
        right_angle_restr = right_angle_restr,
        right_angle_buffer = right_angle_buffer,
        min_skeletal_length_offshoot = min_skeletal_length_offshoot,
        offshoot_df = offshoot_df,
        basal_df = basal_df,
        plot = plot,
        verbose = verbose,
        verbose_time=verbose_time,
        )

    matches_df = []
    if len(basal_df) > 0:
        for offshoot_dict in pu.df_to_dicts(offshoot_df):
            comp_dict = mau.best_similarity_match_from_matching_df(
                subgraph_dict = offshoot_dict,
                matching_subgraph_df = basal_df,
                verbose = verbose,
                plot = plot,
                mesh_match = mesh,

                similarity_func=similarity_func,
                similarity_func_kwargs=similarity_func_kwargs,

                verbose_time=verbose_time,
            )

            basal_dict = basal_df.iloc[0,:].to_dict()
            
            if add_soma_soma_dist:
                comp_dict.update(hdju.soma_soma_xyz_distance_between_segments(
                    segment_id_1 = offshoot_dict["segment_id"],
                    split_index_1 = offshoot_dict["split_index"],
                    segment_id_2 = basal_dict["segment_id"],
                    split_index_2 = basal_dict["split_index"],
                    absolute_value = False,
                    return_dict = True
                ))
            
            # ---- could add on the endpoint dists for convinience
            if add_endpoint_endpoint_dist:
                comp_dict.update(mau.endpoint_xyz_dist_from_dicts(
                    offshoot_dict,
                    basal_dict,
                    absolute_value = False,
                    return_dict = True)
                )
                
            matches_df.append(comp_dict)

    matches_df = pd.DataFrame.from_records(matches_df)
    matches_df["n_basal"] = len(basal_df)
    
    if filter_away_0_scholl:
        if "match_similarity_n_points" in matches_df.columns:
            matches_df = matches_df.query(f"match_similarity_n_points > 0")
    
    
    #print(f"matches_df.columns = {matches_df.columns}")
    return matches_df

import pandas as pd
import numpy as np
def synthetic_subgraph_vectors_df(
    n_samples=50,
    segment_id = 12345,
    split_index = 0,
    node = "L-1",
    compartment = "oblique",
    oblique_angle = 90,
    skeletal_length = None
    ):
    """
    Purpos: To create a synthetic offshoot dataframe
    """
    if skeletal_length is None:
        skeletal_length = default_min_skeletal_length

    synthetic_df = pd.DataFrame.from_records([
        {"segment_id":segment_id,
            "split_index":split_index,
            "node":node,
            "skeletal_length":skeletal_length,
             "compartment":compartment,
             oblique_angle_feature:oblique_angle,
        }
    ])

    synthetic_df = synthetic_df.iloc[np.zeros(n_samples),:]
    synthetic_df[default_angle_match_vector_name] = np.random.rand(n_samples)*360
    synthetic_df["subgraph_idx"] = np.arange(n_samples)
    return synthetic_df

def print_n_compartment_vectors(df):
    if len(df) > 0:
        comps,counts = np.unique(df["compartment"].to_numpy(),return_counts = True)
        for c,count in zip(comps,counts):
            print(f"# of {c}: {count}")
    else:
        print(f"No compartment vectors")

def subgraph_vectors_base_analysis_table(
    min_skeletal_length = None,
    cell_type = "excitatory",
    min_n_basal = 2,
    ):
    """
    Purpose: Create a base table
    for the oblique/basal comparison analysis
    """
    
    if min_skeletal_length is None:
        min_skeletal_length = mau.default_min_skeletal_length

    n_comp_table  = hdju.subgraph_vectors_n_compartment_table(
        hdju.subgraph_vectors_minus_volume_edge_table 
        & f"skeletal_length > {min_skeletal_length}",)

    """
    query would use to restrict the oblique analysis
    """
    base_table = (hdju.proofreading_neurons_with_gnn_cell_type_fine_minus_volume_edge() 
        & dict(gnn_cell_type_coarse =cell_type)
        & (n_comp_table & "n_oblique > 0 " & "n_apical_shaft > 0" & f"n_basal >= {min_n_basal}").proj()
    )

    return base_table

base_analysis_table = subgraph_vectors_base_analysis_table


default_local_radius = 100_000
default_local_type = "bbox"
default_subgraph_table = "offshoot_table_annotated"#"subgraph_vectors_table"
centroid_df = None

def set_centroid_df(table=None):
    if table is None:
        table = subgraph_vectors_base_analysis_table()
    global centroid_df
    centroid_df= hdju.seg_split_centroid_df_with_cell_types(table=table)
    
def local_subgraph_vectors_table(
    segment_id,
    use_centroid_df = True,
    table = None,
    split_index=0,
    radius = None,
    compartments = ("oblique","apical"),
    include_self=False,
    min_skeletal_length = None,
    sample_segment_ids = None,
    local_type = None,
    subgraph_table = None,
    verbose = False,
    verbose_time = False,
    **kwargs
    ):
    """
    Purpose: Generate a local sample of compartment 
    branches based on a local search radius
    """
    if use_centroid_df:
        if centroid_df is None:
            set_centroid_df()
        table = centroid_df
    
    #print(f"len(centroid_df) = {len(centroid_df)}")
    
    st = time.time()
    if local_type is None:
        local_type = default_local_type
    
    if radius is None:
        radius = default_local_radius
        
    if table is None:
        table = subgraph_vectors_base_analysis_table()
        
    if sample_segment_ids is None:
        nearby_table = hdju.local_cell_type_sampling_from_node(
            segment_id,
            split_index=split_index,
            table = table,
            local_type=local_type,
             radius = radius,
            include_self=include_self,
        )
    else:
        sample_segment_ids = nu.array_like(sample_segment_ids)
        if table is None:
            table = self.proofreading_neurons_with_gnn_cell_type_fine
            
        if pu.is_dataframe(table):
            nearby_table = table.query(f"segment_id in {list(sample_segment_ids)}")
        else:
            nearby_table = table & [dict(segment_id=k) for k in sample_segment_ids]

    if subgraph_table is None:
        subgraph_table = getattr(hdju,default_subgraph_table)
        
    if pu.is_dataframe(nearby_table):
        if verbose:
            print(f"# of nearby cells: {len(nearby_table)}")
        random_sample_table= subgraph_table & pu.df_to_dicts(nearby_table[["segment_id","split_index"]])
    else:
        random_sample_table = (subgraph_table
            & nearby_table.proj() 
        )
    
    if verbose_time:
        print(f"Time for random sample table = {time.time() - st}")
            
    
    if verbose:
        print(f"# of nearby cells = {len(nearby_table)}")
        print(f"# of nearby subgraph vectors = {len(random_sample_table)}")
    
    if compartments is not None:
        compartments = nu.array_like(list(compartments))
        
        random_sample_table = (random_sample_table
            & "OR".join([f"(compartment = '{k}')" for k in compartments])
                              )
        
        if verbose:
            print(f"# of nearby subgraph vectors (after compartments = {compartments}) = {len(random_sample_table)}")
            
    if min_skeletal_length is not None:
        random_sample_table = random_sample_table & f"skeletal_length >= {min_skeletal_length}"
        
    if verbose_time:
        print(f"Test download time = {time.time() - st}")
    
    return random_sample_table

import time
def local_subgraph_vectors_df(
    segment_id,
    table = None,
    n_samples = 100,
    split_index=0,
    local_type = None,
    radius = None,
    compartments = ["oblique","apical"],
    include_self=False,
    verbose = False,
    min_skeletal_length = None,
    sample_segment_ids = None,
    verbose_time = False,
    ):
    st = time.time()
    
    import datajoint_utils 
    
    if min_skeletal_length is None:
        min_skeletal_length = default_min_skeletal_length
    
    random_sample_table = local_subgraph_vectors_table(
        segment_id,
        split_index=split_index,
        radius = radius,
        compartments = compartments,
        include_self=include_self,
        verbose = verbose,
        min_skeletal_length = min_skeletal_length,
        sample_segment_ids=sample_segment_ids,
        table=table,
        local_type=local_type,
        )
    
    #print(f"len(random_sample_table) = {len(random_sample_table)}")
    
    if n_samples is not None and n_samples < len(random_sample_table):
        import datajoint_utils as du
        offshoot_df_other = du.sample_table(
            random_sample_table,
            n_samples,
            verbose = False,
        )
    else:
        offshoot_df_other = hdju.df_from_table(random_sample_table)

    if verbose_time:
        print(f"Time for sampling df = {time.time() - st}")
        st = time.time()
    offshoot_df_other = mau.add_on_angles_2D_xz_and_xyz(offshoot_df_other)
    
    if verbose_time:
        print(f"Time for adding on angles = {time.time() - st}")
    
        
    return offshoot_df_other


import matplotlib.pyplot as plt
def offshoot_vs_basal_analysis_with_controls(
    segment_id,
    split_index = 0,
    table = None,
    plot_neuron = False,
    
    # -- For the cdist compartison
    right_angle_restr = False,
    offshoot_compartments = ("oblique","apical"),
    plot_cdist = False,
    
    # -- for controls
    local_control = True,
    sample_segment_ids = None,
    synthetic_control = True,
    n_samples = 40,
    
    plot_hist = False,
    column_for_hist = None,
    verbose = False,
    verbose_inner = False,
    verbose_time = False,
    
    
    
    flip_names = False,
    data_func = None,
    similarity_func = None,
    similarity_func_kwargs = None,
    
    ):
    """
    Purpose: To compute the cdist of a neuron
    plus any controls request and put in dataframe

    Pseudocode: 
    1) Pull down the subgraph vectors of segment
    2) Run the cdist on the neuron and it's own oblique/apical branches
    -- if local control requested --
    3) Get a table of the local neurons
    """
    if data_func is None:
        data_func = offshoot_df_vs_basal_df_best_similarity_match
    
    if table is None:
        table = getattr(hdju,default_subgraph_table)
    
    if plot_neuron:
        hdju.plot_proofread_neuron(
            segment_id,
            split_index
        )

    #1) Pull down the subgraph vectors of segment
    subgraph_vec_df = hdju.subgraph_vectors_df_from_segment(
        segment_id,
        split_index,
        with_angles = True,
        table = table,
    )


    if verbose:
        print(f"For base segment:")
        mau.print_n_compartment_vectors(subgraph_vec_df)

    offshoot_dict = dict(
        same_neuron = None,
    )
    if local_control:
        
        offshoot_df_other= mau.local_subgraph_vectors_df(
            segment_id=segment_id,
            split_index = split_index,
            n_samples=n_samples,
            verbose = verbose,
            min_skeletal_length = default_min_skeletal_length,
            sample_segment_ids=sample_segment_ids,
        )

        offshoot_dict["local_control"] = offshoot_df_other
        #print(f"len(offshoot_df_other) = {len(offshoot_df_other)}")
        #print(f"offshoot_df_other comps = {offshoot_df_other['compartment'].unique()}")

    if synthetic_control:
        synthetic_df = mau.synthetic_subgraph_vectors_df(
            n_samples=n_samples
        )

        offshoot_dict["synthetic_control"] = synthetic_df

    all_dfs = []
    for name,offshoot_df in  offshoot_dict.items():
        #2) Run the cdist on for all of the offshoot dicts
        if verbose:
            print(f"-- Working on {name}---")
            
#         if name == 'local_control':
#             vebose_closest_angle_match = True
#         else:
#             vebose_closest_angle_match = False
            
        other_df = data_func(
            subgraph_vec_df=subgraph_vec_df,
            offshoot_df=offshoot_df,
            offshoot_compartments=offshoot_compartments,
            right_angle_restr=right_angle_restr and name != "synthetic_control",
            plot = plot_cdist,
            verbose = verbose_inner,
            similarity_func = similarity_func,
            similarity_func_kwargs = similarity_func_kwargs,
            verbose_time=verbose_time,
            )
        #print(f"other_df = {other_df}")
        other_df["data_type"] = name
        all_dfs.append(other_df)
        
#         if name == 'local_control':
#             return other_df,offshoot_dict["local_control"]

        if verbose:
            print(f"# of {name} cdist = {len(other_df)}")

    total_df = pu.concat(all_dfs)

    if verbose:
        print(f"Total # of entries = {len(total_df)}")

    if plot_hist:
        from python_tools import matplotlib_utils as mu
        if column_for_hist is None:
            column_for_hist = default_similarity_metric_name
        mu.histograms_overlayed(
            total_df,
            column=f"match_{column_for_hist}",
            hue = "data_type",
            outlier_buffer = None,
        )
        plt.show()

    if flip_names:
        total_df= mau.flip_match_column_names(total_df)
        
    return total_df.reset_index(drop=True)

def flip_match_column_names(df):
    rename_dict = dict([(k,k[6:]) if "match" in k else (k,f"match_{k}")
                  for k in df.columns])
    rename_dict["n_basal"] = "n_basal"
    rename_dict["data_type"] = "data_type"
    
    df = pu.rename_columns(df,rename_dict)
    return df

import morphology_analysis_utils as mau

# --------- More fine scholl analysis ---------------
from python_tools import pandas_utils as pu
def scholl_dicts_combine(
    scholl_dicts,
    verbose = False,
    ):
    """
    Purpose: Combine a list of scholl dictionaries
    into one

    Pseudocode: 
    1) Start off dictionary with first dict copy
    2) For each other dictionary:
    - iterate through the keys 
    a. if key exists: vstack with current value
    b. Add key and value pair

    """
    if pu.is_dataframe(scholl_dicts):
        scholl_dicts = scholl_dicts.scholl_coords.to_list()

    final_dict = scholl_dicts[0].copy()

    for curr_dict in scholl_dicts[1:]:
        for k,v in curr_dict.items():
            if k in final_dict:
                final_dict[k] = np.vstack([final_dict[k],v])
            else:
                final_dict[k] = v
                
    if verbose:
        print(f"Max scholl dist = {list(final_dict.keys())[-1]}")

    return final_dict


from pykdtree.kdtree import KDTree
def scholl_offshoot_cdist_dict(
    basal_scholl_dict=None,
    offshoot_scholl_dict = None,
    offshoot_center=np.array([0,0,0]),
    basal_center=np.array([0,0,0]),
    axes = (0,2),
    basal_df = None,
    offshoot_dict=None,
    verbose = False,
    verbose_cdist = False,
    return_df = False,
    plot = False,
    ):
    """
    Purpose: To calculate the scholl info
    between an offshoot dict and the basal
    scholl coordinates 

    Pseudocode: 
    a. Get the scholl points of offshoot
    For every layer
        b. Get the closest ditances for all oblique scholl
        points to basal scholl points (OF THAT LAYER)
        c. Find the cdist from layer radius and distance
        d. Get the mean and number of datapoints
        e. Save in dictionary with that radius as name
    Have final dictionary of scholl 
    """
    if basal_scholl_dict is None:
        basal_scholl_dict = mau.scholl_dicts_combine(
        basal_df,
        verbose = False) 
        
    if type(basal_scholl_dict) == list:
        basal_scholl_dict = mau.scholl_dicts_combine(
            basal_scholl_dict,
            verbose = False
        )
        

    axes = np.array(axes)

    #a. Get the scholl points of offshoot
    if offshoot_scholl_dict is None:
        offshoot_scholl_dict = offshoot_dict["scholl_coords"]
    if verbose:
        print(f"Max scholl dist for offshoot = {list(offshoot_scholl_dict.keys())[-1]}")
        
    # --- need to now normalize the points based on their soma coordinates ---
    
    #For every layer
    offshoot_results = dict()
    
    #print(f"basal_center= {basal_center}")
    #print(f"offshoot_center = {offshoot_center}")
    
    for radius,off_pts in offshoot_scholl_dict.items():
        if verbose_cdist:
            print(f"-- Working on cdist radius = {radius} --")

        if len(off_pts) == 0:
            continue

        if radius not in basal_scholl_dict:
            continue

        #b. Get the closest ditances for all oblique scholl
        #points to basal scholl points (OF THAT LAYER)

        basal_pts = np.array((basal_scholl_dict[radius]-basal_center)[:,axes]).reshape(-1,len(axes))
        basal_kd = KDTree(basal_pts)

        off_pts_axes = (off_pts - offshoot_center)[:,axes].reshape(-1,len(axes))
        dist,closest_basal_idx = basal_kd.query(off_pts_axes)

        #c. Find the cdist from layer radius and distance
        cdists = nu.angle_from_chord(dist,radius,rad=False)

        #d. Get the mean and number of datapoints
        cdists_mean = np.mean(cdists)
        n_cdists = len(cdists)


        #e. Save in dictionary with that radius as name
        suffix = int(radius)
        local_dict = dict()
        local_dict[f"mean_cdist_{suffix}"] = cdists_mean
        local_dict[f"n_cdist_{suffix}"] = n_cdists
        local_dict[f"basal_scholl_idx_{suffix}"] = closest_basal_idx
    

        if verbose_cdist:
            print(f"   -> mean cdist = {cdists_mean}, n_cdists = {n_cdists}")

        offshoot_results.update(local_dict)

    if return_df:
        pd.DataFrame.from_records([offshoot_results])
    return offshoot_results

from python_tools import pandas_utils as pu
def offshoot_vs_basal_scholl_cdist_df(
    subgraph_vec_df = None,
    min_skeletal_length_basal = None,
    offshoot_compartments = ("oblique","apical"),
    right_angle_restr = False,
    right_angle_buffer = None,
    min_skeletal_length_offshoot = None,
    offshoot_df = None,
    basal_df = None,
    
    subgraph_features_to_record = (
            "y_soma_relative",
            "skeletal_length",
            "width",
            "subgraph_idx",
            "compartment",
            "node",
            "leaf_node",
            "segment_id",
            "split_index",
            "cell_type_predicted",
            "cell_type_predicted_prob",
        ),
    matching_feature_to_record = (
           "segment_id",
           "split_index", 
           "cell_type_predicted",
           "cell_type_predicted_prob",
        ),
    plot = False,
    plot_scholl = False,
    verbose = False,
    verbose_cdist = False,
    ):
    """
    Purpose: To calculate the scholl cdist
    dataframe for a basal and offshoot dataframe

    1) Divide up the subgraph into offshoot df and basal df
    2) Get all the scholl points for the basals
    3) For all the offshoots compute the offshoot_scholl info and save as df
    4) Add on the basal info for what was matched to
    """

    if subgraph_features_to_record is None:
        subgraph_features_to_record = []

    if matching_feature_to_record is None:
        matching_feature_to_record = []

    #1) Divide up the subgraph into offshoot df and basal df
    basal_df,offshoot_df = mau.offshoot_vs_basal_df_preprocessing(
            subgraph_vec_df = subgraph_vec_df,
            min_skeletal_length_basal = min_skeletal_length_basal,
            offshoot_compartments = offshoot_compartments,
            right_angle_restr = right_angle_restr,
            right_angle_buffer = right_angle_buffer,
            min_skeletal_length_offshoot = min_skeletal_length_offshoot,
            offshoot_df = offshoot_df,
            basal_df = basal_df,
            plot = plot,
            verbose = verbose,
    )


    #2) Get all the scholl points for the basals
    if len(basal_df) == 0:
        return pu.empty_df()

    basal_scholl_dict = mau.scholl_dicts_combine(
        basal_df,
        verbose = False) 
    
    basal_center = hdju.soma_nm_coordinate(
        basal_df["segment_id"].to_list()[0],
        basal_df["split_index"].to_list()[0],
    )


    #3) For all the offshoots compute the offshoot_scholl info
    offshoot_dicts = pu.df_to_dicts(offshoot_df)
    offshoot_scholl_df = []
    for j,offshoot_dict in enumerate(offshoot_dicts):
        if verbose:
            print(f"--- Working on offshoot dict {j}---")
        
        offshoot_center = hdju.soma_nm_coordinate(
            offshoot_dict["segment_id"],
            offshoot_dict["split_index"]
        )
        
        sch_dict = mau.scholl_offshoot_cdist_dict(
            offshoot_dict = offshoot_dict,
            basal_scholl_dict = basal_scholl_dict,
            offshoot_center=offshoot_center,
            basal_center=basal_center,
            verbose = verbose,
            verbose_cdist = verbose_cdist,
        )

        # -- need to add any more information here -- 
        curr_dict = {k:v for k,v in offshoot_dict.items()
                        if k in subgraph_features_to_record}
        curr_dict.update(sch_dict)
        offshoot_scholl_df.append(curr_dict)

    offshoot_scholl_df = pd.DataFrame.from_records(offshoot_scholl_df)

    # 4) Adding on the basal match information
    basal_dict = basal_df.iloc[0].to_dict()
    for k in matching_feature_to_record:
        if k in basal_dict:
            offshoot_scholl_df[f"match_{k}"] = basal_dict[k]

    offshoot_scholl_df["n_basal"] = len(basal_df)
    
    return offshoot_scholl_df

def scholl_order(
    df,
    high_order_keywords = ["segment_id","split_index","subgraph_idx","node"],
    radius_keywords = ["mean_cdist","n_cdist","basal_scholl_idx"],
    columns_to_ignore = None,
    verbose = False,
    return_radii = False,
    ):
    """
    Purpose: Want to order the columns of the dataframe
    (especially the radius ones)
    """
    if columns_to_ignore is None:
        columns_to_ignore = []
    
    high_order_names = []
    other_names = []
    radius_names = dict()
    
    if pu.is_dataframe(df):
        columns = df.columns
    elif type(df) == dict:
        columns = list(df.keys())
    else:
        raise Exception("")
        
    for k in columns:
        matched_flag = False
        for cc in columns_to_ignore:
            if cc in k:
                matched_flag = True
                break
                
        if matched_flag:
            continue
        
        
        for cc in high_order_keywords:
            if cc in k:
                high_order_names.append(k)
                if verbose:
                    print(f"matched {k} to high importance")
                matched_flag = True

        if matched_flag:
            continue

        for r in radius_keywords:
            if r in k:
                #print(f"matched {k}")
                # get the radius number
                radius = int(k.split("_")[-1])
                if radius not in radius_names:
                    radius_names[radius] = []
                radius_names[radius].append(k)
                matched_flag = True
                if verbose:
                    print(f"matched {k} to radius")
                break

        if matched_flag:
            continue

        other_names.append(k)

    # -- order the radius names
    final_order = high_order_names + other_names
    radius_ints = np.sort(list(radius_names.keys()))
    for ri in radius_ints:
        final_order += list(np.sort(radius_names[ri]))

    if pu.is_dataframe(df):
        return_df = df[final_order]
    elif type(df) == dict:
        return_df = {k:df[k] for k in final_order}
    else:
        raise Exception("")
        
    if return_radii:
        return return_df,radius_ints
    else:
        return return_df
    
def radii_after_radius_restriction(radii,radius_restriction=None):
    if radius_restriction is not None:
        radius_restriction = [10_000*k if k < 1_000 else k for k in radius_restriction]
        radii = [k for k in radii if k in radius_restriction]
    return radii
def radii_from_df(df,radius_restriction=None):
    _, radii = scholl_order(df,return_radii = True)
    return radii_after_radius_restriction(radii,radius_restriction=radius_restriction,)

def radii_from_dict(scholl_dict,radius_restriction=None):
    radii = [int(k) for k,v in scholl_dict.items() if len(v) > 0]
    return radii_after_radius_restriction(radii,radius_restriction=radius_restriction)

def radii_complement_from_df(
    df,
    radius_restriction=None,
    verbose = False):
    all_radii = mau.radii_from_df(df,radius_restriction=None)
    to_delete_radii = np.setdiff1d(all_radii,radius_restriction)

    if verbose:
        print(f"radii_complement = {to_delete_radii}")

    return to_delete_radii

def columns_from_radii(df,radii,verbose = False):
    to_delete_cols = []
    for k in df.columns:
        for rad in radii:
            curr_str = f"_{rad}"
            #print(f"{k[-len(curr_str):]}")
            if k[-len(curr_str):] == curr_str:
                to_delete_cols.append(k)


    if verbose:
        print(f"radii columns = {to_delete_cols}")
        
    return to_delete_cols

def offshoot_vs_basal_scholl_analysis_with_controls(
    segment_id,
    split_index = 0,
    plot_neuron = False,

    # -- For the cdist compartison
    right_angle_restr = False,
    offshoot_compartments = ("oblique","apical"),


    # -- for controls
    local_control = True,
    sample_segment_ids = None,
    n_samples = 60,

    plot_cdist = False,
    verbose = False,
    verbose_inner = False,
    
    flip_names = False,
    
    columns_to_ignore = None,#("basal_scholl_idx",),
    **kwargs
    
    ):
    
    return_df = mau.offshoot_vs_basal_analysis_with_controls(
        segment_id=segment_id,
        split_index = split_index,
        plot_neuron = plot_neuron,

        # -- For the cdist compartison
        right_angle_restr = right_angle_restr,
        offshoot_compartments = offshoot_compartments,

        # -- for controls
        local_control = local_control,
        sample_segment_ids=sample_segment_ids,
        synthetic_control = False,
        n_samples = n_samples,

        plot_cdist = plot_cdist,
        verbose = verbose,
        verbose_inner = verbose_inner,
        flip_names = flip_names,
        data_func = mau.offshoot_vs_basal_scholl_cdist_df,
    )

    return scholl_df_order_columns(return_df,columns_to_ignore=columns_to_ignore)


# ----- for help with plotting the scholl analysis ----
def basal_scholl_dict_from_segment(
    segment_id,
    split_index=0,
    ):
    """
    Purpose: To get the basal scholl coordinates
    for a segment id

    Pseudocode: 
    1) get the subgraph vector
    2) Divide into basal
    3) Extract the scholl coordinates
    """

    subgraph_vec_df = hdju.subgraph_vectors_df_from_segment(
            segment_id,
            split_index,
            with_angles = False,
    )


    basal_df,_ = mau.offshoot_vs_basal_df_preprocessing(
        subgraph_vec_df = subgraph_vec_df)

    basal_scholl_dict = mau.scholl_dicts_combine(
            basal_df,
            verbose = False) 

    return basal_scholl_dict

def combine_scholl_dicts(scholl_dicts):
    """
    Purpose: want to combine scholl dicts into one

    Pseudocode: 
    """
    if type(scholl_dicts) == dict:
        return scholl_dicts
    curr_dict = dict()
    for curr_scholl in scholl_dicts:
        for k,v in curr_scholl.items():
            if len(v) == 0:
                continue
            if k not in curr_dict:
                curr_dict[k] = v
            else:
                curr_dict[k] = np.vstack([curr_dict[k],v])
    return curr_dict

def scholl_dict_from_segment(
    segment_id,
    subgraph_idx = None,
    split_index = 0,
    compartment=None,
    adjusted = False,
    **kwargs
    ):
    """
    Purpose: To get the scholl dict from a 
    certain subgraph 

    Pseudocode: 
    1) 
    """
    key = dict(
            segment_id=segment_id,
            split_index = split_index,
    )
    curr_table = hdju.subgraph_vectors_table & key
    if subgraph_idx is not None:
        curr_table = curr_table & [dict(subgraph_idx = k) 
                               for k in nu.to_list(subgraph_idx)]
        
    if compartment is not None:
        curr_table = curr_table & [dict(compartment = k) 
                               for k in nu.to_list(compartment)]
        
    if adjusted:
        feature = "scholl_coords_adjusted"
    else:
        feature = "scholl_coords"
    restr_scholl = (curr_table).fetch(feature)
    return combine_scholl_dicts(restr_scholl)
#     if subgraph_idx is not None:
#         return (hdju.subgraph_vectors_table & dict(
#             segment_id=segment_id,
#             split_index = split_index,
#             subgraph_idx = subgraph_idx,
#         )).fetch1("scholl_coords")
#     else:
#         scholl_dicts = (hdju.subgraph_vectors_table & dict(
#             segment_id=segment_id,
#             split_index = split_index,
#         )).fetch("scholl_coords")
#         return combine_scholl_dicts(scholl_dicts)

def subgraph_endpoint_coords_from_segment(
    segment_id,
    subgraph_idx = None,
    split_index = 0,
    compartment=None,
    endpoint_type = "upstream",
    **kwargs
    ):
    
    key = dict(
            segment_id=segment_id,
            split_index = split_index,
    )
    curr_table = hdju.subgraph_vectors_table & key
    if subgraph_idx is not None:
        curr_table = curr_table & [dict(subgraph_idx = k) 
                               for k in nu.to_list(subgraph_idx)]
        
    if compartment is not None:
        curr_table = curr_table & [dict(compartment = k) 
                               for k in nu.to_list(compartment)]
        
    feature = f"endpoint_{endpoint_type}"
    features = [f"{feature}_{ax}_nm" for ax in ['x','y','z']]
    coords = np.vstack((curr_table).fetch(*features)).T
    return coords

def node_from_subgraph_idx(
    segment_id,
    subgraph_idx,
    split_index = 0,
    df = None,
    verbose = False,
    
    ):
    if df is not None:
        node = df.query(
            f"(segment_id == {segment_id})"
            f"and (split_index == {split_index})"
            f"and (subgraph_idx == {subgraph_idx})"
        )["node"].to_list()[0]

    else:
        node = hdju.subgraph_vectors_table & dict(
            segment_id=segment_id,
            split_index=split_index,
            subgraph_idx=subgraph_idx
        ).fetch1("node")

    if verbose:
        print(f"Node = {node}")

    return node

def basal_scholl_idx(
    segment_id,
    subgraph_idx,
    split_index=0,
    df=None,
    filter_away_nan = True,
    ):
    """
    Purpose: To extract the basal match
    idx from the dataframe (put into a dict)

    Pseudocode: 
    1) Restrict the analysis
    """
    if df is None:
        raise Exception("Not implemented")
    else:
        curr_dict = pu.df_to_dicts(df.query(
            f"(segment_id == {segment_id})"
            f"and (split_index == {split_index})"
            f"and (subgraph_idx == {subgraph_idx})"
        ))[0]

    curr_dict = {k:v for k,v in curr_dict.items() if "basal_scholl" in k}

    if filter_away_nan:
        curr_dict = {k:v for k,v in curr_dict.items() if
                    not np.any(np.isnan(v))}

    return curr_dict

def restrict_df_to_subgraph_idx(
    df,
    segment_id,
    subgraph_idx,
    split_index = 0,
    return_singular_dict = False):
    
    return_df =  df.query(
            f"(segment_id == {segment_id})"
            f"and (split_index == {split_index})"
            f"and (subgraph_idx == {subgraph_idx})"
    )

    if return_singular_dict:
        return return_df.iloc[0,:].to_dict()
    else:
        return return_df
    
    
def basal_offshoot_scholl_dicts(
    offshoot_segment_id,
    basal_segment_id,
    offshoot_subgraph_idx,
    offshoot_split_index = 0,    
    basal_split_index = 0,
    scholl_analysis_df = None,#mau.flip_match_column_names(total_df)
    radius = None,
    return_scatter = False,
    verbose = False,
    verbose_cdist = False,
    ):
    """
    Purpose: To get the scholl dicts
    of an offshoot and the basal 
    wit the basal optionally restricted by an analysis
    df that shows the alignment

    Pseudocode: 
    1) Get the basal scholl dict
    2) Get the scholl dict of the subgraph
    3) if a matching df is given then restrict the basal scholl
    dict to only those vertices
    4) Convert the scholl dicts to scatters if requested

    """

    #1) Get the basal scholl dict
    basal_scholl_dict = mau.basal_scholl_dict_from_segment(
        segment_id = basal_segment_id,
        split_index = basal_split_index,   
    )

    if verbose:
        print(f"Possible basal radii: {list(basal_scholl_dict.keys())}")
    
    #2) Get the scholl dict of the subgraph
    subgrah_scholl_dict = mau.scholl_dict_from_segment(
        segment_id = offshoot_segment_id,
        split_index = offshoot_split_index,
        subgraph_idx = offshoot_subgraph_idx,
    )
    
    if verbose:
        print(f"Possible offshoot radii: {list(subgrah_scholl_dict.keys())}")

    #3) if a matching df is given then restrict the basal scholl
    #dict to only those vertices
    if scholl_analysis_df is not None:
        scholl_idx = mau.basal_scholl_idx(
            segment_id = offshoot_segment_id,
            split_index = offshoot_split_index,
            subgraph_idx = offshoot_subgraph_idx,
            df = scholl_analysis_df
        )
        #print(f"scholl_idx = {scholl_idx}")
        
        basal_scholl_dict_new = dict()
        for k,v in scholl_idx.items():
            curr_radius = int(k.split("_")[-1])
            if len(v) > 0 and not np.any(np.isnan(v)):
                basal_scholl_dict_new[curr_radius] = basal_scholl_dict[curr_radius][v]

        basal_scholl_dict = basal_scholl_dict_new

    #4) Convert the scholl dicts to scatters
    if radius is not None:
        radius= nu.array_like(radius)
        if verbose:
            print(f"Restricting to radius = {radius}")
        basal_scholl_dict = {k:basal_scholl_dict[k] for k in radius if k in basal_scholl_dict}
        
        subgrah_scholl_dict = {k:subgrah_scholl_dict[k] for k in radius if k in subgrah_scholl_dict}
        

    if scholl_analysis_df is not None and verbose_cdist:
        if radius is None:
            curr_radius = mau.radii_from_df(scholl_analysis_df)
        else:
            curr_radius = radius
            
        restr_dict = restrict_df_to_subgraph_idx(
            df = scholl_analysis_df,
            segment_id = offshoot_segment_id,
            split_index = offshoot_split_index,
            subgraph_idx = offshoot_subgraph_idx,
            return_singular_dict = True
        )
        for rad in curr_radius:
            print(f"Radius {rad} cdist = {restr_dict[f'mean_cdist_{rad}']}")

    if return_scatter:
        offshoot_scatters = [k for k in subgrah_scholl_dict.values() if len(k) > 0]
        basal_scatters = [k for k in basal_scholl_dict.values() if len(k) > 0]

        if len(offshoot_scatters) > 0:
            offshoot_scatters = np.vstack(offshoot_scatters)
        else:
            offshoot_scatters = None

        if len(basal_scatters) > 0:
            basal_scatters = np.vstack(basal_scatters)
        else:
            basal_scatters = None

        return basal_scatters,offshoot_scatters
    else:
        return basal_scholl_dict,subgrah_scholl_dict
    
def example_scholl_coordinates_not_normalized_for_mesh_center():
    """
    Purpose: To see if the scholl points stored correspond to the mesh
    """
    offshoot_segment_id = 864691134884750842
    mesh = hdju.fetch_proofread_mesh(offshoot_segment_id)
    scholl_dict = mau.scholl_dict_from_segment(offshoot_segment_id,subgraph_idx=4)
    
    import neuron_visualizations as nviz
    nviz.plot_objects(
        meshes = [mesh],
        scatters=list(scholl_dict.values())
    )
    
from python_tools import ipyvolume_utils as ipvu
def plot_offshoot_basal_scholl_coordinates(
    offshoot_segment_id,
    offshoot_subgraph_idx,
    basal_segment_id,
    basal_split_index = 0,
    offshoot_split_index = 0,
    scholl_analysis_df = None,#mau.flip_match_column_names(total_df),
    radius = None,
    verbose = False,
    axis_box_off = False,
    offshoot_mesh = None,
    basal_mesh = None,
    verbose_cdist = True,
    ):
    """
    Purpose: Want to plot the scholl points
    that matched up between an offshoot and
    the target basal (can either do all layers or just one)

    Pseudocode: 
    1) Get the basal and offshoot scatter points
    2) Get the offshoot node to plot
    3) Plot the segments, scatters and nodes
    
    Ex -----
    

    # -- worked for the same segmnet offshoot
    offshoot_segment_id = 864691134884750842
    offshoot_subgraph_idx = 10
    offshoot_split_index = 0

    # offshoot_segment_id = 864691136272948414
    # offshoot_subgraph_idx = 1

    # offshoot_segment_id = 864691136309632090
    # offshoot_subgraph_idx = 12


    basal_segment_id = 864691134884750842
    basal_split_index = 0

    basal_mesh = hdju.fetch_proofread_mesh(basal_segment_id,basal_split_index)
    offshoot_mesh = hdju.fetch_proofread_mesh(offshoot_segment_id,offshoot_split_index)

    mau.plot_offshoot_basal_scholl_coordinates(
        offshoot_segment_id = offshoot_segment_id,
        offshoot_subgraph_idx = offshoot_subgraph_idx,
        offshoot_split_index = offshoot_split_index,

        basal_segment_id = basal_segment_id,
        basal_split_index = basal_split_index,

        scholl_analysis_df = mau.flip_match_column_names(total_df),
        #radius = [5,6],
        verbose = True,

        axis_box_off = False,
        offshoot_mesh = offshoot_mesh,
        basal_mesh = basal_mesh,

        )


    """

    if radius is not None:
        radius = nu.array_like(radius)
        radius = [k*10_000 if k < 1000 else k for k in radius]
    
    
    
    basal_scatter,offshoot_scatter = mau.basal_offshoot_scholl_dicts(
        offshoot_segment_id = offshoot_segment_id,
        basal_segment_id = basal_segment_id,
        offshoot_subgraph_idx = offshoot_subgraph_idx,
        offshoot_split_index = offshoot_split_index,
        basal_split_index=basal_split_index,
        scholl_analysis_df = scholl_analysis_df,
        radius = radius,
        return_scatter = True,
        verbose = verbose,
        verbose_cdist = verbose_cdist,
        )


    #5) Get the node information from the 
    offshoot_node = mau.node_from_subgraph_idx(
        segment_id = offshoot_segment_id,
        subgraph_idx = offshoot_subgraph_idx,
        split_index = offshoot_split_index,
        df = scholl_analysis_df
    ) 

    if verbose:
        print(f"offshoot_node = {offshoot_node}")

    hdju.plot_meshes_node_skeletons_aligned(
        segment_ids=[basal_segment_id,offshoot_segment_id,],
        nodes=[None,offshoot_node],
        scatters=[basal_scatter,offshoot_scatter],
        split_indexes=[basal_split_index,offshoot_split_index],
        meshes = [basal_mesh,offshoot_mesh],
        axis_box_off=axis_box_off
    )
    
    ipvu.top_down()
    
    
# ------------- cdist analysis after data computed -------------------
def df_cdist_columns_from_df(
    df,
    attribute = ("mean",),
    match_segment = True,
    verbose = False,
    return_cdist_columns = False,
    radius_restriction = None,
    ):
    """
    Purpose: Return the offshoot info and 
    the mean cdist
    """
    attribute = nu.array_like(attribute,include_tuple=True)

    if match_segment:
        prefix = "match_"
    else:
        prefix = ""

    radii_in_df = mau.radii_from_df(df,radius_restriction=radius_restriction)

    columns = list(np.concatenate([[f"{prefix}{k}_cdist_{radii}" for k in attribute] for radii in radii_in_df]))
    final_columns = [f"{prefix}segment_id",f"{prefix}split_index",f"{prefix}subgraph_idx"] + columns

    if verbose:
        print(f"final_columns = {final_columns}")

    if return_cdist_columns:
        return df[final_columns],columns
    else:
        return df[final_columns]
    
import copy
def mean_cdist_mean_over_radii(
    df,
    radius_restriction= None,
    match_segment = True,
    mean_column_name = "cdist_mean",#"radii"
    in_place = False,
    delete_radii_columns_not_used = True,
    ):
    """
    Purpose: Calculate the mean cdist over a radii range
    and add onto a dataframe
    """
    if not in_place:
        df = copy.deepcopy(df)
    if match_segment:
        prefix = "match_"
    else:
        prefix = ""

    radii_in_df = mau.radii_from_df(df,radius_restriction=radius_restriction)

    columns = [f"{prefix}mean_cdist_{radii}" for radii in radii_in_df]
    
    if mean_column_name == "radii":
        mean_column_name = f"cdist_mean_radii_{'_'.join([f'{k/10_000}' for k in radii_in_df])}"
        print(f"mean_column_name = {mean_column_name}")
    df[mean_column_name] = df[columns].mean(axis=1)
    
    if delete_radii_columns_not_used:
        radii_comp = radii_complement_from_df(df,radii_in_df)
        radii_columns = columns_from_radii(df,radii_comp)
        
        df = pu.delete_columns(
            df,radii_columns
        )
    
    df = pu.filter_away_rows_with_nan_in_columns(
        df,
        [mean_column_name]
    )
    
    return df


from neuron_morphology_tools import neuron_nx_utils as nxu
def n_scholl_dict_from_segment(
    segment_id,
    subgraph_idx,
    split_index = 0,
    return_named_dict = True,
    **kwargs
    ):
    """
    Purpose: Return the number of scholl points for a subgraph
    """
    scholl_dict = nxu.scholl_dict_from_segment(
        segment_id=segment_id,
        subgraph_idx=subgraph_idx,
        split_index = split_index, 
    )
    
    return n_scholl_dict_from_scholl_dict(
    scholl_dict,
    return_named_dict = return_named_dict,
    )
    
    
# ---------- working on the offshoot analysis with leaf nodes -----
def radius_restriction_scholl_dict(
    scholl_dict,
    radius_restriction = None,
    verbose = False,
    ):
    
    if radius_restriction is not None:
        radius_restriction= nu.array_like(radius_restriction)
        if verbose:
            print(f"Restricting to radius = {radius_restriction}")
        radius_restriction = mau.radii_from_dict(scholl_dict,radius_restriction=radius_restriction)
        scholl_dict = {k:scholl_dict[float(k)] for k in radius_restriction}
        
    return scholl_dict

def scatter_from_scholl_dict(
    scholl_dict,
    radius_restriction = None,
    default_value = None,
    ):
    
    scholl_dict = radius_restriction_scholl_dict(
    scholl_dict,
    radius_restriction = radius_restriction,
    )
    
    scatters = [k for k in scholl_dict.values() if len(k) > 0]
    if len(scatters) > 0:
        scatters = np.vstack(scatters)
    else:
        scatters = default_value
        
    return scatters

import numpy as np
def coordinate_type_from_dict(
    data,
    coordinate_name):
    return np.array([data[f"{coordinate_name}_{k}_nm"] for k in ["x","y","z"]])

def endpoint_upstream_from_dict(data):
    return coordinate_type_from_dict(
    data,
    coordinate_name="endpoint_upstream")

def endpoint_xyz_dist_from_dicts(
    dict1,
    dict2,
    return_dict = True,
    absolute_value = False,
    ):
    endpt1 = mau.endpoint_upstream_from_dict(dict1)
    endpt2 = mau.endpoint_upstream_from_dict(dict2)
    return_value = endpt1 - endpt2
    
    if absolute_value:
        return_value = np.abs(return_value)
    
    if return_dict:
        return {f"endpoint_endpoint_dist_{ax}_nm":val for ax,val in zip(["x","y","z"],return_value)}
    else:
        return return_value
        

def scholl_distance_dict_from_scholl_dicts_same_radius(
    offshoot_scholl_dict,
    basal_scholl_dict,
    offshoot_center,
    basal_center,
    adjusted = False,
    offshoot_upstream_endpoint=None,
    basal_upstream_endpoint = None,
    axes = (0,2),
    verbose = False,
    verbose_dist= True,
    verbose_time = False,
    mesh = None,
    plot = False,
    debug = False,
    ):
    # ------ how currently computes the distance ----------------
    """
    Purpose: To compute the distance between coordinates
    along the same scholl ring (that also accounts for 
    different soma centers and different offshoot angles)

    Ex: 
    mau.scholl_distance_dict_between_basal_offshoot(
        basal_leaf_node = "L5_0",
        offshoot_leaf_node = "L1_1",

        basal_df = basal_df,
        offshoot_df = offshoot_df,

        scholl_adjusted = True,

        axes = (1,),
        verbose = True,
        plot = False,
        plot_pair_for_dist=False,
        mesh = mesh,
        debug = True,
    )
    """
    if adjusted and (offshoot_upstream_endpoint is None or
                    basal_upstream_endpoint is None):
        raise Exception("")

    if len(axes) > 2 and not adjusted:
        raise Exception("")
        
    offshoot_results = dict()
    for radius,off_pts in offshoot_scholl_dict.items():
        if verbose_dist:
            print(f"-- Working on distance for radius = {radius} --")

        if len(off_pts) == 0:
            continue

        if radius not in basal_scholl_dict:
            continue

        #b. Get the closest ditances for all oblique scholl
        #points to basal scholl points (OF THAT LAYER)
        basal_pts = basal_scholl_dict[radius]
        st = time.time()
        if adjusted:
            #print(f"offshoot_center = {offshoot_center}")
            #print(f"offshoot_upstream_endpoint = {offshoot_upstream_endpoint}")
            off_pts = nxu.adjusted_scholl_coordinates(
                    off_pts,
                    soma_coordinate = offshoot_center,
                    upstream_endpoint=offshoot_upstream_endpoint,
                    verbose = False)

            basal_pts = nxu.adjusted_scholl_coordinates(
                    basal_pts,
                    soma_coordinate = basal_center,
                    upstream_endpoint=basal_upstream_endpoint,
                    verbose = False)
            if verbose_time:
                print(f"time for adjustment = {time.time() - st}")
                st = time.time()

        if debug:
            print(f"basal_pts = {basal_pts}, off_pts = {off_pts}")
            print(f"axes = {axes}")
        if plot:
            import neuron_visualizations as nviz
            print(f"Plotting radius = {radius}")
            nviz.plot_objects(
                mesh,
                scatters =  [basal_pts,off_pts],
                scatters_colors = ["red","black"],
            )
            
        basal_pts = np.array((basal_pts-basal_center)[:,axes]).reshape(-1,len(axes))
        basal_kd = KDTree(basal_pts)

        off_pts_axes = (off_pts - offshoot_center)[:,axes].reshape(-1,len(axes))
        
        if verbose_time:
            print(f"time for creating pts and kdtree = {time.time() - st}")
            st = time.time()
        
        
        
        dist,closest_basal_idx = basal_kd.query(off_pts_axes)
#         dist = [np.min(np.linalg.norm(basal_pts-k.reshape(-1,3),axis=1)) 
#                for k in off_pts_axes]

        #d. Get the mean and number of datapoints
        dists_mean = np.mean(dist)
        
        if verbose_time:
            print(f"time for kdtree query = {time.time() - st}")
            st = time.time()

        offshoot_results[radius] = dists_mean

    return offshoot_results

def scholl_distance_dict_from_scholl_dicts(
    offshoot_scholl_dict,
    basal_scholl_dict,
    offshoot_center,
    basal_center,
    adjusted = False,
    offshoot_upstream_endpoint=None,
    basal_upstream_endpoint = None,
    axes = (0,2),
    verbose = False,
    verbose_dist= True,
    verbose_time = False,
    mesh = None,
    plot = False,
    debug = False,
    default_value = np.inf,
    ):
    # ------ how currently computes the distance ----------------
    """
    Purpose: To compute the average closest distance for the offshoot
    scholl ring coordinates 
    
    Ex: 
    mau.scholl_distance_dict_between_basal_offshoot(
        basal_leaf_node = "L5_0",
        offshoot_leaf_node = "L1_1",

        basal_df = basal_df,
        offshoot_df = offshoot_df,

        scholl_adjusted = True,

        axes = (0,2,),
        verbose = True,
        plot = False,
        plot_pair_for_dist=False,
        mesh = mesh,
        debug = True,
    )
    """
    st = time.time()
    if adjusted and (offshoot_upstream_endpoint is None or
                    basal_upstream_endpoint is None):
        raise Exception("")

    if len(axes) > 2 and not adjusted:
        raise Exception("")
        
    
    basal_scholl_coordinates = mau.scatter_from_scholl_dict(
        basal_scholl_dict,
        default_value = np.array([]).reshape(-1,3)
    )
    
    if verbose_time:
        print(f"time for scatter_from_scholl_dict = {time.time() - st}")
        st = time.time()
        
        
    offshoot_results = dict()
    for radius,off_pts in offshoot_scholl_dict.items():
        if verbose_dist:
            print(f"-- Working on distance for offshoot radius = {radius} --")

        if len(off_pts) == 0:
            continue


        #b. Get the closest ditances for all oblique scholl
        #points to basal scholl points (OF THAT LAYER)
        basal_pts = basal_scholl_coordinates.copy()
        
        if verbose_time:
            print(f"time for copying basal_scholl_coordinates = {time.time() - st}")
            st = time.time()
        
        if len(basal_pts) == 0:
            offshoot_results[radius] = default_value
            continue
            
        if adjusted:
            #print(f"offshoot_center = {offshoot_center}")
            #print(f"offshoot_upstream_endpoint = {offshoot_upstream_endpoint}")
            off_pts = nxu.adjusted_scholl_coordinates(
                    off_pts,
                    soma_coordinate = offshoot_center,
                    upstream_endpoint=offshoot_upstream_endpoint,
                    verbose = False)

            basal_pts = nxu.adjusted_scholl_coordinates(
                    basal_pts,
                    soma_coordinate = basal_center,
                    upstream_endpoint=basal_upstream_endpoint,
                    verbose = False)
            
            if verbose_time:
                print(f"time for adjustment = {time.time() - st}")
                st = time.time()

        if debug:
            print(f"basal_pts = {basal_pts}, off_pts = {off_pts}")
            print(f"axes = {axes}")
        if plot:
            import neuron_visualizations as nviz
            print(f"Plotting radius = {radius}")
            nviz.plot_objects(
                mesh,
                scatters =  [basal_pts,off_pts],
                scatters_colors = ["red","black"],
            )
            
        basal_pts = np.array((basal_pts-basal_center)[:,axes]).reshape(-1,len(axes))
        basal_kd = KDTree(basal_pts)

        off_pts_axes = (off_pts - offshoot_center)[:,axes].reshape(-1,len(axes))
        
        if verbose_time:
            print(f"time for creating pts and kdtree = {time.time() - st}")
            st = time.time()
        
        
        
        #dist,closest_basal_idx = basal_kd.query(off_pts_axes)
        dist = [np.min(np.linalg.norm(basal_pts-np.expand_dims(k,axis=0),axis=1)) 
               for k in off_pts_axes]

        #d. Get the mean and number of datapoints
        dists_mean = np.mean(dist)

        offshoot_results[radius] = dists_mean
        
        if verbose_time:
            print(f"time for kdtree query = {time.time() - st}")
            st = time.time()

    return offshoot_results


from python_tools import ipyvolume_utils as ipvu

def scholl_distance_dict_between_basal_offshoot(
    basal_dict = None,
    offshoot_dict = None,
    
    offshoot_center=None,
    basal_center=None,
    
    basal_leaf_node = None,
    offshoot_leaf_node = None,
    
    basal_df = None,
    offshoot_df = None,
    
    basal_scholl_dict = None,
    offshoot_scholl_dict = None,
    
    # for adjusting the parameters that will affect the dist
    axes = (0,2),
    scholl_adjusted = False,
    angle = False,
    
    verbose = False,
    verbose_dist = False,
    verbose_time = False,
    
    # for plotting the pairs in which the distance is calculated
    plot = False,
    plot_pair_for_dist=False,
    mesh = None,
    debug = False,
    **kwargs
    ):
    """
    Purpose: To compute the distance dict
    for two scholl coord dists, and return the distances
    corresponding to the coordinated diffed

    Applicaiton: Can combine this information inot a 
    metric for how well they match

    Pseudocode: 
    1) Get the basal and offshoot dicts if not recieved
    2) Get the scholl coordinates adjusted or not
    3) Find the distance betwen all school coordinates (that overlap)
    and record the radii of all (so get number and distance)

    """
    st = time.time()

    if basal_leaf_node is not None:
        basal_dict = basal_df.query(f"leaf_node == '{basal_leaf_node}'").iloc[0,:].to_dict()

    if offshoot_leaf_node is not None:
        offshoot_dict = offshoot_df.query(f"leaf_node == '{offshoot_leaf_node}'").iloc[0,:].to_dict()


    if offshoot_center is None:
        offshoot_center = hdju.soma_coordinate_nm(
            offshoot_dict["segment_id"],
            offshoot_dict["split_index"]
        )


    if basal_center is None:
        basal_center = hdju.soma_coordinate_nm(
            basal_dict["segment_id"],
            basal_dict["split_index"]
        )


    if scholl_adjusted:
        scholl_name = "scholl_coords_adjusted"
    else:
        scholl_name = "scholl_coords"

    if basal_scholl_dict is None:
        basal_scholl_dict = basal_dict[scholl_name]
    if offshoot_scholl_dict is None:
        offshoot_scholl_dict = offshoot_dict[scholl_name]
    if verbose:
        print(f"Max scholl dist for basal ({basal_dict['leaf_node']}) = {list(basal_scholl_dict.keys())[-1]}")
        print(f"Max scholl dist for offshoot ({offshoot_dict['leaf_node']}) = {list(offshoot_scholl_dict.keys())[-1]}")

    offshoot_endpoint = mau.endpoint_upstream_from_dict(offshoot_dict)
    basal_endpoint = mau.endpoint_upstream_from_dict(basal_dict)
    if plot and (basal_leaf_node is not None or offshoot_leaf_node is not None):
        hdju.plot_meshes_node_skeletons_aligned(
            segment_ids=[basal_dict["segment_id"],offshoot_dict["segment_id"]],
            nodes=[basal_leaf_node,offshoot_leaf_node],
            scatters=[mau.scatter_from_scholl_dict(k) for k in [basal_scholl_dict,offshoot_scholl_dict]],
            nodes_as_leaf_nodes=True,
            show_at_end = False,
        )

        ipvu.plot_multi_scatters(
            [offshoot_endpoint-offshoot_center,basal_endpoint-basal_center],
            color = ["black","red"],
            size=0.3,
            new_figure=False,
            flip_y=True,

        )
        
    if verbose_time:
        print(f"Time for prepping scholl dict = {time.time() - st}")
        st = time.time()

    # -- calculate the scholl distance dict   
    scholl_dist_dict = mau.scholl_distance_dict_from_scholl_dicts(
        offshoot_scholl_dict,
        basal_scholl_dict,
        offshoot_center,
        basal_center,
        adjusted = scholl_adjusted,
        offshoot_upstream_endpoint = offshoot_endpoint,
        basal_upstream_endpoint = basal_endpoint,
        axes = axes,
        verbose = verbose,
        verbose_dist= verbose_dist,
        verbose_time=verbose_time,
        debug = debug,
        plot = plot_pair_for_dist,
        mesh = mesh,
        )
    
    if verbose_time:
        print(f"Time for scholl_distance_dict_from_scholl_dicts = {time.time() - st}")
        st = time.time()

    if angle:
        if len(axes) != 2:
            raise Exception("")
        scholl_dist_dict = {k:nu.angle_from_chord(v,radius = k,rad = False)
                           for k,v in scholl_dist_dict.items()}
        
        
    return scholl_dist_dict



def mean_distance_plus_linear_n_pts_penalty_loss(
    scholl_dist_dict,
    mean_dist_penalty_coeff=1/1_000,
    n_points_penalty_coeff=200,
    default_value = np.inf,
    verbose = False,
    return_metadata = False,
    **kwargs
    ):
    """
    Purpose: To calculate a similarity score from the
    scholl dist dict
    
    Pseudocode:
    1) Calculate the mean dist
    2) Calculate the inverse number of points
    3) Weight them
    """
    meta_dict = dict()
    n_points = len(scholl_dist_dict)
    
    if n_points == 0:
        if return_metadata:
            return default_value,meta_dict
        else:
            return default_value
    
    #1) Calculate the mean dist
    mean_dist = np.sum(list(scholl_dist_dict.values()))/n_points
    
    #2) Calculate the inverse of the number of points
    n_pts_inv = 1/n_points
    
    #3) Calculate the total loss
    mean_dist_weight = mean_dist_penalty_coeff*mean_dist
    n_pts_inv_weight = n_points_penalty_coeff*n_pts_inv
    loss = mean_dist_weight + n_pts_inv_weight
    
    meta_dict["mean_dist"] = mean_dist
    
    if verbose:
        print(f"n_points= {n_points}, mean_dist = {mean_dist:.2f}, n_pts_inv = {n_pts_inv:2f}")
        print(f"mean_dist_weight = {mean_dist_weight}, n_pts_inv_weight = {n_pts_inv_weight}")
        print(f"loss = {loss}")
        
    if return_metadata:
            return loss,meta_dict
    else:
        return loss

def mean_distance_with_exponent_n_pts_loss(
    scholl_dist_dict,
    exponent_n_pts = 2,
    default_value = np.inf,
    divisor = 1_000,
    verbose = False,
    return_metadata = False,
    **kwargs
    ):
    """
    Purpose: To calculate a similarity score from the
    scholl dist dict
    
    Pseudocode:
    1) Calculate the mean dist
    2) Calculate the inverse number of points
    3) Weight them
    """
    n_points = len(scholl_dist_dict)
    
    meta_dict = dict()
    
    if n_points == 0:
        if return_metadata:
            return default_value,meta_dict
        else:
            return default_value
    
    #1) Calculate the mean dist
    mean_dist = np.sum(list(scholl_dist_dict.values()))/(n_points)
    exponent_n_pts = (1/n_points)**(exponent_n_pts-1)
    loss  = mean_dist*exponent_n_pts/divisor
    
    meta_dict["mean_dist"] = mean_dist
    
    if verbose:
        print(f"n_points= {n_points}, mean_dist = {mean_dist:.2f}, exponent_n_pts = {exponent_n_pts:2f}")
        print(f"loss = {loss}")
        
    if return_metadata:
        return loss,meta_dict
    else:
        return loss
    
def mean_distance_loss(
    scholl_dist_dict,
    default_value = np.inf,
    divisor = 1,
    verbose = False,
    return_metadata = False,
    **kwargs
    ):
    
    #print(f"in mean_distance_loss")
    n_points = len(scholl_dist_dict)
    
    meta_dict = dict()
    
    if n_points == 0:
        if return_metadata:
            return default_value,meta_dict
        else:
            return default_value
        
    loss = np.sum(list(scholl_dist_dict.values()))/(n_points)/divisor
    
    if verbose:
        print(f"n_points= {n_points}, mean_dist = {loss:.2f}")
    
    if return_metadata:
        return loss,meta_dict
    else:
        return loss
    


#default_scholl_dist_dict_similarity_func = mean_distance_plus_linear_n_pts_penalty_loss
default_scholl_dist_dict_similarity_func = mean_distance_loss


def scholl_distance_dict_similarity(
    subgraph_dict=None,
    matching_dict=None,
    
    scholl_similarity_func = None,
    
    
    # -- parameters for how the distance is computed
    axes=(0, 2),
    scholl_adjusted=False,
    angle=False,
    
    return_metadata = False,
    verbose_time = False,
    **kwargs
    ):
    st = time.time()
    
    if scholl_similarity_func is None:
        scholl_similarity_func = default_scholl_dist_dict_similarity_func
        
    
    scholl_dict = mau.scholl_distance_dict_between_basal_offshoot(
        basal_dict=matching_dict,
        offshoot_dict=subgraph_dict,
        
        # --- parameters for 
        axes=axes,
        scholl_adjusted=scholl_adjusted,
        angle=angle,
        verbose_time=verbose_time,
        **kwargs
    )
    
    if verbose_time:
        print(f"Time for scholl_distance_dict_between_basal_offshoot = {time.time() - st}")
        st = time.time()
    
    loss,meta_dict = scholl_similarity_func(
        scholl_dict,
        return_metadata = True,
        **kwargs)
    
    if verbose_time:
        print(f"Time for scholl_similarity_func ({scholl_similarity_func}) = {time.time() - st}")
        st = time.time()
    
    meta_dict["n_points"] = len(scholl_dict)
    if meta_dict["n_points"] > 0:
        meta_dict["max_scholl_point"] = np.max(np.array(list(scholl_dict.keys()))/10_000).astype('int')
        
    if return_metadata:
        return loss,meta_dict
    else:
        return meta_dict
    
    
def best_similarity_match_between_two_segments(
    offshoot_segment_id,
    basal_segment_id,
    offshoot_split_index = 0,
    basal_split_index = 0,
    offshoot_leaf_node = None,#"L0_8",
    basal_leaf_node = None,# "L4_2",#"L3_3"
    
    right_angle_restr = False,
    
    plot = False,
    **kwargs
    ):
    """
    Purpose: To get the similarity match between two segments
    (and possibly only subset of leaf nodes on those segments)
    
    Ex: 
    mau.best_similarity_match_between_two_segments(
        offshoot_segment_id = 864691136388413559,
        offshoot_split_index = 0,
        offshoot_leaf_node = "L0_8",

        basal_segment_id = 864691134884756474,
        basal_split_index = 0,
        basal_leaf_node = "L4_2",#"L3_3"

        right_angle_restr = False,

        plot = False,
    )
    """

    basal_df = hdju.subgraph_vectors_df_from_segment(
        segment_id = basal_segment_id,
        split_index = basal_split_index,
        leaf_nodes = basal_leaf_node,    
    )


    offshoot_df = hdju.subgraph_vectors_df_from_segment(
        segment_id = offshoot_segment_id,
        split_index = offshoot_split_index,
        leaf_nodes = offshoot_leaf_node,
    )


    return mau.offshoot_df_vs_basal_df_best_similarity_match(
        offshoot_df = offshoot_df,
        basal_df = basal_df,
        right_angle_restr= right_angle_restr,
        plot = plot,
        **kwargs
    )   


def restriction_str_from_scholl_n_limb(
    scholl_radii= (3,4,5,6,7),
    
    compartment = "basal",
    min_n_scholl = 5,
    max_n_scholl = 10,
    min_n_comp = 3,
    max_n_comp = 5,

    min_n_oblique= 3,
    restrictions_append = None,

    table_type="dj",
    verbose = False,
    ):
    """
    Purpose: Create a restriction that will get a neuron
    with particular limb number and scholl points
    number ()
    """

    min_restrictions = [f"n_{compartment}_scholl_{int(k*10_000)} >= {min_n_scholl}" for k in  scholl_radii]
    max_restrictions = [f"n_{compartment}_scholl_{int(k*10_000)} <= {max_n_scholl}" for k in  scholl_radii]
    n_comp_restriction = [
        f"n_{compartment} >= {min_n_comp}",
        f"n_{compartment} <= {max_n_comp}"
    ]
    
    restrictions = min_restrictions + max_restrictions + n_comp_restriction
    
    if min_n_oblique is not None:
        restrictions.append(
            f"n_oblique >= {min_n_oblique}",
        )
        
    if restrictions_append is not None:
        restrictions += restrictions_append
        
    q =  pu.query_str_from_list(
        restrictions,
        table_type=table_type
    )
    
    if verbose:
        print(f"scholl_n_limbs query = {q}")
        
    return q

def subgraph_vectors_table_after_restricion_from_scholl_n_limbs(**kwargs):
    return hdju.subgraph_vectors_table_after_restricion_from_scholl_n_limbs(**kwargs)

def offshoot_vs_basal_analysis_with_controls_by_hyper_params(
    segment_id,
    split_index,
    plot_hist = False,
    verbose = True,
    verbose_analysis = True,
    similarity_func_kwargs_list= None,
    flip_names = False,
    **kwargs
    ):
    """
    Purpose: To run the offshoot vs 
    basal analysis with different hyperparameters
    and compile into one dictionary
    """
    
    if similarity_func_kwargs_list is None:
        similarity_func_kwargs_list = [
            dict(scholl_adjusted = False,
                axes = (0,2)
                ),
            dict(scholl_adjusted = True,
                axes = (0,2)
                ),
            dict(scholl_adjusted = True,
                axes = (0,1,2)
                ),
        ]
        
    total_comp_dfs = []
    for similarity_func_kwargs in similarity_func_kwargs_list:
        if verbose:
            print(f"\n\n-->     similarity_func_kwargs = {similarity_func_kwargs}")

        synthetic_control= False
        similarity_func = mau.scholl_distance_dict_similarity


        comp_df = mau.offshoot_vs_basal_analysis_with_controls(
            segment_id = segment_id,
            split_index = split_index,
            similarity_func = similarity_func,
            similarity_func_kwargs= similarity_func_kwargs,
            right_angle_restr=False,
            plot_hist = plot_hist,
            column_for_hist = "similarity",
            synthetic_control = False,
            local_control=True,
            verbose = verbose_analysis,
            verbose_time = False,
            flip_names = flip_names,
            **kwargs

        )

        comp_df["scholl_adjusted"] = similarity_func_kwargs["scholl_adjusted"]
        comp_df["n_axes"] = len(similarity_func_kwargs["axes"])
        total_comp_dfs.append(comp_df)

    return pu.concat(total_comp_dfs,axis=0).reset_index(drop=True)
        
    
def add_data_type_to_leaf_match_df(
    df,
    column = "data_type"
    ):
    """
    Purpose: to label the which data
    segments came from same neuron and which from
    a local control
    """
    df[column] = "local_control"

    df = pu.set_column_subset_value_by_query(
        df,
        query = "(segment_id == match_segment_id) and (split_index == match_split_index)",
        column = column,
        value = "same_neuron",
    )

    return df

def set_leaf_match_df_dtypes(
    df,
    int_types = (
        "segment_id",
        "split_index",
        "subgraph_idx",

    ),
    str_types = (
        "compartment",
        "node",
        "leaf_node",
        "cell_type_predicted",
        "e_i_predicted",
        "data_type"
    ),
    default_type = "float",
    columns = None
    ):
    """
    Purpose: To set the datatypes correctly
    """

    if columns is None:  
        columns = list(df.columns)

    for k in columns:
        replaced_flag = False
        for j in int_types:
            if k == j or k == f"match_{j}":
                df[k] = df[k].astype('int')
                #print(f"Trying to set")
                replaced_flag= True

        if replaced_flag:
            continue

        for j in str_types:
            if k == j or k == f"match_{j}":
                df[k] = df[k].astype('str')
                #print(f"Trying to set")
                replaced_flag= True

        if replaced_flag:
            continue

        df[k] = df[k].astype(default_type)
        
    return df

def preprocess_leaf_match_df(
    df,
    flip_match = True,
    calculate_soma_endpoint_dists = True,
    calculate_leaf_node_to_top_angle = True
    ):
    
    df = mau.add_data_type_to_leaf_match_df(df)
    if flip_match:
        df = mau.flip_match_column_names(df)
    """
    Calculate the actual distances between endpoints and somas
    """
    if calculate_soma_endpoint_dists:
        for vc in ["soma_soma_dist","endpoint_endpoint_dist"]:
            df[vc] = pu.distance_from_vector(
                df,
                vector_column = vc
            )

    if calculate_leaf_node_to_top_angle:
        df["leaf_node_to_top_angle"] = nu.angle_between_matrix_of_vectors_and_vector(
            pu.vector_between_coordinates(
                df,
                coordinate_column_1= "leaf_node_coordinate",
                coordinate_column_2 = "endpoint_upstream"
            ),hdju.vector_to_top
        )

        df["match_leaf_node_to_top_angle"] = nu.angle_between_matrix_of_vectors_and_vector(
            pu.vector_between_coordinates(
                df,
                coordinate_column_1= "match_leaf_node_coordinate",
                coordinate_column_2 = "match_endpoint_upstream"
            ),hdju.vector_to_top
        )
#         curr_data = pu.vector_between_coordinates(
#                 df,
#                 coordinate_column_1= "leaf_node_coordinate",
#                 coordinate_column_2 = "endpoint_upstream"
#             )
#         print(f"curr_data = {curr_data}")
#         df["leaf_node_to_top_angle"] = (curr_data)

#         df["match_leaf_node_to_top_angle"] = (pu.vector_between_coordinates(
#                 df,
#                 coordinate_column_1= "match_leaf_node_coordinate",
#                 coordinate_column_2 = "match_endpoint_upstream"
#             ))

    df = set_leaf_match_df_dtypes(df)
        
    return df

def preprocess_leaf_df(df):
    """
    Purpose: Want to preprocess the offshoot
    table to include vectors

    Pseudocode: 
    1) Calculate the leaf vector
    2) Calculate the leaf vector to top
    """

    curr_vec = pu.vector_between_coordinates(
            df,
            coordinate_column_1= "leaf_node_coordinate",
            coordinate_column_2 = "endpoint_upstream"
    )

    df[["leaf_node_vector_x_nm",
        "leaf_node_vector_y_nm",
        "leaf_node_vector_z_nm"]] = curr_vec

    df["leaf_node_to_top_angle"] = nu.angle_between_matrix_of_vectors_and_vector(
        curr_vec,
        hdju.vector_to_top
    )
    
    return df

# --------- plotting different compartments ---
from neuron_morphology_tools import neuron_nx_utils as nxu
def subgraph_vectors_n_compartment_table_with_cell_typee_minus_edge_volume():
    subgraph_comp_df = hdju.df_from_table(
        ((hdju.subgraph_vectors_n_compartment_table(
            include_n_scholl = True,
            include_branch_skeletal_info = True,
        )*hdju.gnn_embedding_table_latest) & hdju.proofreading_neurons_minus_volume_edge_table().proj()),
        remove_method_features = True

    )
    
    for k in nxu.compartments:
        # compute the branching per skeletal length
        subgraph_comp_df[f"{k}_skeletal_length_per_node"] = (
            subgraph_comp_df[f"{k}_skeletal_length"].astype('float') / subgraph_comp_df[f"n_{k}_nodes"].astype('int')
        )
    
    return subgraph_comp_df

from python_tools import matplotlib_utils as mu
from python_tools import pandas_utils as pu
import matplotlib.pyplot as plt


def plot_compartment_attribute_with_cell_type_overlayed(
    df,
    attribute,
    compartments = None,
    e_i_divisoin = True,
    verbose = True,
    density = True,
    hue = None,
    hue_secondary = None,
    ):
    
    import cell_type_utils as ctu

    if compartments is None:
        compartments= nxu.compartments
    for c in compartments:
        if verbose:
            print(f"--- Working on {c}----")
        
        if "compartment" in c:
            curr_column = attribute.replace("compartment",c)
            curr_df = df
        else:
            curr_df = df.query(f"compartment == '{c}'")
            curr_column = attribute

        curr_df = pu.filter_away_rows_with_nan_in_columns(
            curr_df,
            curr_column
        )
        
        curr_df[curr_column] = curr_df[curr_column].astype("float")

        if "cell_type_predicted" in curr_df:
            cell_type_fine = "cell_type_predicted"
            cell_type_coarse = "e_i_predicted"
        else:
            cell_type_fine = "gnn_cell_type_fine"
            cell_type_coarse = "gnn_cell_type_coarse"
        
        if e_i_divisoin:
            hue = cell_type_coarse
            hue_secondary = cell_type_fine
        else:
            hue = cell_type_fine
            hue_secondary = None
            
        
        if c == "dendrite":
            curr_df = curr_df.query(f"{cell_type_fine} in {list(ctu.allen_cell_type_fine_classifier_labels_inh)}")
        elif c != "axon":
            curr_df = curr_df.query(f"{cell_type_fine} in {list(ctu.allen_cell_type_fine_classifier_labels_exc)}")
        else:
            pass

        #print(f"---{curr_column}---")
        #print(f"hue = {hue},hue_secondary = {hue_secondary}")
        mu.histograms_overlayed(
            curr_df,
            curr_column,
            hue = hue,
            hue_secondary = hue_secondary,
            density = density,

        )
        plt.show()
        
def example_n_compartment_with_cell_type_overlayed():
    plot_compartment_attribute_with_cell_type_overlayed(
        attribute = "n_compartment",
        df = subgraph_vectors_n_compartment_table_with_cell_typee_minus_edge_volume(),
        compartments = ["oblique","basal","dendrite"],
        hue_secondary = None,
        )
    
# ---------------------- 2D histogram of angle bias -------------

def histgram_2d_over_angles_from_aligned_df(
    df,
    x="skeleton_vector_aligned_xz_angle",
    y = "centroid_y_nm",
    n_angle_bins = 10,
    n_bins = 4,
    bins = None,
    verbose = False,
    plot = True,
    return_bins = True,
    normalize_rows=True,
    **kwargs
    ):
    """
    Purpose: Want to do a 2d histogram
    of the max width dataframe
    where the x is the angle
    """
    x_bins = np.linspace(0,360,n_angle_bins + 1)

    #hists, x_bins,y_bins = pu.histogram_2d(
    return pu.histogram_2d(
        df=df,
        x=x,
        y = y,
        n_y_bins = n_bins,
        y_bins = bins,
        x_bins = x_bins,
        verbose = verbose,
        return_bins = return_bins,
        normalize_rows = normalize_rows,
        plot = plot,
        return_df = True,
        **kwargs
    )

def histgram_2d_max_width_from_unaligned_df(
    df,
    compartment = "basal",
    x="skeleton_vector_aligned_xz_angle",
    y = "centroid_y_nm",
    n_angle_bins = 10,
    n_bins = 4,
    row_bins = None,
    verbose = False,
    plot = False,
    return_bins = True,
    ):
    """
    Purpose: To get the 2d histogram
    of an angle value and another value
    for a certain compartment
    """

    df_max_width = mau.max_width_df(df,compartment=compartment)
    df_max_width_aligned = mau.aligned_angle_vector_df(
            df_max_width,
            align = True,
            add_new_columns_for_align = True,
        )


    hist,x_bins,y_bins = histgram_2d_over_angles_from_aligned_df(
        df=df_max_width_aligned,
        x=x,
        y = y,
        n_angle_bins = n_angle_bins,
        n_bins = n_bins,
        bins = row_bins,
        verbose = verbose,
        plot = plot,
        return_bins = True,
        )

    if return_bins:
        return hist,x_bins,y_bins
    else:
        return hist
    
def n_limbs_max_from_compartment(compartment,k):
    if compartment in ['axon','apical_shaft']:
        return 1
    else:
        return k
def seg_split_top_k_attribute_from_subgraph_vector_df(
    df,
    k = 2,
    column = "width",
    compartment = None,
    n_limbs_min = 5,
    skeletal_length_min = 10_000,
    add_diff_from_max = True,
    append_compartment_to_column_name = True,
    verbose = True,
    **kwargs
    ):
    """
    Purpose: Generate a dataframe mapping 
    segment/split to the k largest attribute
    and difference from max
    """
    k = n_limbs_max_from_compartment(compartment,k)

    group_columns = ['segment_id','split_index']
    if n_limbs_min is None:
        n_limbs_min = mau.n_limbs_min_by_compartment_dict[compartment]

    if verbose:
        print(f"n_limbs_min = {n_limbs_min}")
    df_sort = mau.subgraph_vector_df_restriction(
        df,
        verbose = verbose,
        compartment=compartment,
        n_limbs_min=n_limbs_min,
        skeletal_length_min=skeletal_length_min,
        **kwargs
    )

    df = pu.top_k_extrema_attributes_as_columns_by_group(
        df_sort,
        column=column,
        group_columns=group_columns,
        k = k
    )
    
    cols = [k for k in df.columns if column in k]
    if verbose:
        print(f"New cols = {cols}")
    if add_diff_from_max and compartment not in ['axon','apical_shaft']:
        df[[f"max_minus_{k}" for k in cols]] = -1*df[cols].to_numpy() + np.max(df[cols].to_numpy(),axis=1).reshape(-1,1)
        
    if append_compartment_to_column_name:
        df.columns = [f"{compartment}_{k}" if column in k else k for k in df.columns]
        
    return df

import seaborn as sns
from python_tools import matplotlib_utils as mu
def plot_histogram_2d_max_width_angle(
    df,
    x="skeleton_vector_aligned_xz_angle",
    y = "depth",
    cmap = "Blues",
    n_angle_bins = 18,
    n_bins = 10, # the number of rows you want
    bins = None,#explictely the bins for the rows
    axes_fontsize = 30,
    axes_tick_fontsize = 30,
    colorbar_tick_fontsize = 25,
    title_fontsize = 30,
    x_tick_labels = None,
    y_tick_labels = None,
    rows_to_skip= None,
    
    title = f"Angle of Thickest Basal",
    source = None,
    title_pad = 10,
    #ylabel = f"Depth ($\mu m$)",
    ylabel = f"Depth (mm)",
    xlabel = "Angle",
    colobar_title = "Histogram Density\n Across Row",
    return_intervals = False,
    ax = None,
    depth_divisor = 1000,
    ):
    """
    Purpose: To plot the stacked row histograms
    of the angles 
    """
    if source is not None:
        title = f"{title} ({source})"
    
    #ybins are the bins including the ends (so if n =10, y_bins.shape = 11)
    hist_df,x_bins,y_bins = mau.histgram_2d_max_width_from_unaligned_df(
        df,
        x=x,
        y = y,
        n_angle_bins = n_angle_bins,
        n_bins = n_bins,
        row_bins=bins,

    )
    
    intervals = np.vstack([y_bins[:-1],y_bins[1:]]).T


    if rows_to_skip is not None:
        rows_to_keep = np.delete(np.arange(len(hist_df)),rows_to_skip)
        hist_df = hist_df.iloc[rows_to_keep,:]
        intervals = intervals[rows_to_keep]
    intervals_mid = np.mean(intervals,axis= 1)

    if depth_divisor is not None:
        hist_df.index = hist_df.index/depth_divisor
        intervals_mid = intervals_mid/depth_divisor

    ax = sns.heatmap(
        data = hist_df,
        cmap = cmap,
        vmin = 0,
        vmax = 0.12,
        ax = ax,
    )

    ax.tick_params(left=False, bottom=False)
    # for tick in ax.xaxis.get_majorticklabels():
    #     tick.set_horizontalalignment("right")
    # for tick in ax.yaxis.get_majorticklabels():
    #     tick.set_verticalalignment("bottom")

    if x_tick_labels is None:
        x_tick_labels = [f"{int(k)}$\circ$" if i%3 == 0 else "" 
               for i,k in enumerate(x_bins[:-1])]

    if y_tick_labels is None:
        y_tick_labels = [f"{float(k/1000):.2f}" if i%2 == 0 else "" 
               for i,k in enumerate(intervals_mid)]
    # ylabels = ["","550","","1100","","1650","","2200","","2750"]

    mu.set_axes_ticklabels(
        ax,
        xlabels=x_tick_labels,
        ylabels=y_tick_labels,
    )

    
    mu.set_axes_tick_font_size(ax,fontsize=axes_tick_fontsize,x_rotation=45)
    ax.set_xlabel(xlabel,fontsize = axes_fontsize)
    ax.set_ylabel(ylabel,fontsize = axes_fontsize)
    ax.set_title(title,fontsize=  title_fontsize,pad=title_pad)

    # here set the labelsize by 20
    mu.set_colorbar_tick_fontsize(ax,colorbar_tick_fontsize)
    mu.set_colorbar_title(
        ax,
        title = colobar_title,
        fontsize = 30,
        labelpad = 60
    )
    
    if return_intervals:
        return ax,intervals
    return ax

def plot_width_diff_from_max_vertical(
    df,
    column = "depth",
    stat_column = "basal_max_minus_width_idx_1",
    n_bins = 10,
    intervals = None,
    bins = None,
    ax = None,
    figsize = (5,10),
    depth_divisor = 10**6,
    xlabel = "Largest vs. Second\nLargest Width Diff (nm)",
    axes_fontsize = 20,
    title_fontsize = 30,
    axes_tick_fontsize = 20,
    xlim = (0,900),
    xlabelpad = 0,
    ):
    """
    Purpose: To plot the largest
    vs second largest basal width difference
    over depth using predefined bins
    """

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize = figsize)


    if bins is None:
        if intervals is not None:
            bins = np.array([k[0] for k in intervals] + [intervals[-1][-1]])

    stats,bins,data_len = pu.bin_df_by_column_stat(
        df = df,
        column = column,
        bins=bins,
        func = stat_column,
        return_bins_mid = True,
        equal_depth_bins=False,
        n_bins=n_bins,   
        return_df_len = True,
        return_std = False,
    )

    bins = bins/depth_divisor
    #ax.plot(bins,stats,label="Exc",orientation="vertical")
    ax.plot(stats,bins,label="Exc")
    #ax.scatter(bins,stats,orientation="vertical")
    ax.scatter(stats,bins)


    ax.set_ylabel(f"",fontsize=1)
    ax.set_xlabel(f"{xlabel}",fontsize = axes_fontsize,labelpad=xlabelpad)
    #ax.set_title(f"Exc Source Percentage vs.\nAxon Distance to Soma",fontsize = title_fontsize)
    ax.set_xlim(list(xlim))
    mu.set_axes_tick_font_size(ax,axes_tick_fontsize)
    mu.flip_ylim(ax)
    mu.hide_y_tick_labels(ax,)
    return ax

# ---- looking at basal orientation ----
def coordinates_from_scholl_dict(
    scholl_dict,
    default_value = np.array([])
    ):
    offshoot_scatters = [k for k in scholl_dict.values() if len(k) > 0]
    if len(offshoot_scatters) > 0:
        offshoot_scatters = np.vstack(offshoot_scatters)
    else:
        offshoot_scatters = default_value
        
    return offshoot_scatters

def print_n_scholl_from_scholl_dict(scholl_dict):
    print(f"--- n scholl coordinates --")
    for k,v in scholl_dict.items():
        print(f"{k}:{len(v)}")
        
def scholl_coordinates_from_segment(
    segment_id,
    subgraph_idx=None,
    split_index=0,
    radius_restriction = None,
    return_dict = False,
    verbose=False,
    compartment=None,
    adjusted = False,
    ):
    scholl_dict = scholl_dict_from_segment(
        segment_id,
        subgraph_idx=subgraph_idx,
        split_index = split_index,
        compartment=compartment,
        adjusted=adjusted,
        )
    
    scholl_dict_restr = radius_restriction_scholl_dict(
        scholl_dict,radius_restriction,
    )
    
    if verbose:
        print_n_scholl_from_scholl_dict(scholl_dict_restr)
        
    if return_dict:
        return scholl_dict_restr
    else:
        return coordinates_from_scholl_dict(
            scholl_dict_restr,
        )

import neuron_visualizations as nviz
def plot_scholl_coordinates(
    segment_id,
    split_index = 0,
    subgraph_idx = None,
    radius_restriction=None,
    verbose = True,
    verbose_colors = True,
    separate_by_radius = True,
    scatter_size = 1,
    plot_axes = True,
    compartment=None,
    plot_mesh = True,
    append_figure = False,
    show_at_end = True,
    adjusted = False,
    ):
    """
    Purpose: Plot the scholl coordinates for a certain
    segment
    """
    if plot_mesh:
        mesh = hdju.fetch_proofread_mesh(segment_id,split_index=split_index)
    else:
        mesh = None
    scholl_coords = mau.scholl_coordinates_from_segment(
        segment_id,
        split_index=split_index,
        subgraph_idx=subgraph_idx,
        verbose = False,
        return_dict = separate_by_radius,
        radius_restriction = radius_restriction,
        compartment=compartment,
        adjusted=adjusted,
    )
    
    print(f"scholl_coords = {scholl_coords}")

    if not separate_by_radius:
        scholl_coords = {"No separation":scholl_coords}

    curr_keys = list(scholl_coords.keys())
    scatters = [scholl_coords[k] for k in curr_keys]
    scatters_colors = mu.generate_non_randon_named_color_list(len(curr_keys))

    if verbose_colors and separate_by_radius:
        for k,col,sc in zip(curr_keys,scatters_colors,scatters):
            print(f"{k}:{col} ({len(sc)})")

    nviz.plot_objects(
        mesh,
        scatters=scatters,
        scatters_colors=scatters_colors,
        scatter_size=scatter_size,
        axis_box_off = not plot_axes,
        append_figure = append_figure,
        show_at_end = show_at_end,
    )
    
def example_plot_scholl_coordinates():
    mau.plot_scholl_coordinates(
        segment_id = 864691135462974270,
        compartment = "basal"
    )
    
from python_tools.tqdm_utils import tqdm

def expand_scholl_coords_dict_in_df(
    df,
    verbose = False,
    add_adjusted_scholl=False):
    """
    Purpose: Want to expand the scholl coordinates to separate lines with the radius and the coordinate

    Pseudocode: 
    1) export all rows dictionaries
    For each dictionary 
    a) Create a dummy dictionary with seg,split_subgraph
    b) 
    """
    st = time.time()
    scholl_dicts_rev = []
    scholl_dicts = pu.df_to_dicts(df)
    
    scholl_names = {"scholl_coords":False}
    if add_adjusted_scholl:
        scholl_names["scholl_coords_adjusted"] = True
    
    for sd in scholl_dicts:
#         id_dict = dict(
#             segment_id = sd['segment_id'],
#             split_index = sd['split_index'],
#             subgraph_idx = sd['subgraph_idx'],
#         )
        
        id_dict = {k:v for k,v in sd.items() if 'scholl_coords' not in k}

        for name,adj_bool in scholl_names.items():
            all_coord_dicts = []
            for dist,coords in sd[name].items():
                all_coord_dicts += [
                    dict(
                        id_dict,
                        radius = dist,
                        scholl_x_nm = c[0],
                        scholl_y_nm = c[1],
                        scholl_z_nm = c[2],
                        adjusted = adj_bool
                    )
                for c in coords]

            scholl_dicts_rev += all_coord_dicts
    scholl_exp_df = pu.dicts_to_df(scholl_dicts_rev)
    
    if verbose:
        print(f"Total time for scholl  df expansion = {time.time() - st}")
        
    return scholl_exp_df

def coordinate_soma_vectors_from_df(
    df,
    coordinate_name,
    centroid_column = "centroid",
    axes = ["x","y","z"],
    suffix = "nm",
    normalize = True,
    centroid_df = None,
    return_df  = True,
    in_place = False,
    append_to_df = False,
    vector_name = None,
    group_opp_dir_vectors = False,
    ):
    """
    Purpose: Generate the vectors from the soma
    to the coordinates

    Pseudocode: 
    1) Add on the centroids
    2) Calculate the vector from the 
    centroids to the coordinates
    """

    if centroid_df is None:
        centroid_df = hdju.seg_split_centroid_df()

    #if centroid_column == "centroid":
    df = hdju.add_soma_centroid_to_df(
        df,
        centroid_df = centroid_df
    )
        
    #print(f"centroid_column = {centroid_column}")

    vectors = (
        pu.coordinates_from_df(df,coordinate_name,axes=axes,suffix=suffix) - 
        pu.coordinates_from_df(df,centroid_column,axes=axes,suffix=suffix)
    )

    if normalize:
        vectors = vectors/(np.linalg.norm(vectors,axis = 1).reshape(-1,1))   

    if group_opp_dir_vectors:
        curr_map = vectors[:,0] < 0
        vectors[curr_map,:] = vectors[curr_map,:]*-1
        
    if append_to_df or return_df:
        if vector_name is None:
            vector_name = f"{coordinate_name}_soma_vector"

        df[[f"{vector_name}_{ax}_{suffix}" for ax in axes]] = vectors
    
    
    if return_df:
        return df
    else:
        return vectors
    
default_similarity_func = scholl_distance_dict_similarity#similarity_from_start_vector
default_similarity_metric_name = "similarity"
# ------------- Setting up parameters -----------
from python_tools import module_utils as modu 

# data_fetcher = None
# voxel_to_nm_scaling = None

# -- default
import dataInterfaceMinnie65
attributes_dict_default = dict(
    voxel_to_nm_scaling = dataInterfaceMinnie65.voxel_to_nm_scaling,
    hdju = dataInterfaceMinnie65.data_interface
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
    hdju = dataInterfaceH01.data_interface
)
global_parameters_dict_h01 = dict()
    
       
data_type = "default"
algorithms = None
modules_to_set = [mau]

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
    
set_global_parameters_and_attributes_by_data_type(data_type,
                                                   algorithms)

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


