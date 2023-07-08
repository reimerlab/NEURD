import pandas as pd
from python_tools import pandas_utils as pu
import numpy as np


from python_tools import numpy_utils as nu
def func_ori_from_angle(
    df,
    func_ori,
    column = "scholl_soma_vector_xz_angle",
    add_n_branches = True,
    ):
    
    directions = df[column].to_numpy()
    
    if type(func_ori) != dict:
        func_ori = nu.to_list(func_ori)
        func_ori = {k.__name__:k for k in func_ori}
        
    data_dict = dict()
    for func_name,func in func_ori.items():
        data_dict[func_name] = [func(directions)]
        
    if add_n_branches:
        data_dict["n_vec"] = len(df)
    return pd.DataFrame.from_dict(
        data_dict
    )

def osi_dsi_from_xy_angle(
    df,
    column = "scholl_soma_vector_xz_angle",
    add_n_branches = True,
    ):
    
    return func_ori_from_angle(
    df,
    func_ori = dict(
        osi = nu.osi_from_directions,
        dsi = nu.dsi_from_directions,
    ),
    column = column,
    add_n_branches = add_n_branches,
    )

def sum_vectors(
    df,
    #vector_name = "leaf_node_coordinate_soma_vector",
    vector_name = "scholl_soma_vector",
    weight_column =None,# "skeletal_length",
    axes = ["x","z"],
    suffix = "nm",
    scaled_by_dsi = False,
    angle_column = "scholl_soma_vector_xz_angle",
    ):
    """
    Purpose: Want to consolidate the vectors of a group
    into a single group

    Pseudocode: 
    1) Group the dataframe
    2) extract the vectors
    3) scale the vectors by a weight
    4) Sum the vectors
    """
    #print(f"scaled_by_dsi = {scaled_by_dsi}")
    vectors = pu.coordinates_from_df(
        df,
        vector_name,
        axes=axes,
        suffix = suffix,
    )
    
    #print(f"axes = {axes}")
    #print(f"Normalizing ({vectors.shape})")
    vectors = vectors/(np.linalg.norm(vectors,axis = 1).reshape(-1,1))
    
    #print(f"scale_by_dsi min func = {scaled_by_dsi}")
    if scaled_by_dsi:
        dsi = func_ori_from_angle(
            df,
            func_ori = dict(
                dsi = nu.dsi_from_directions,
            ),
            column = angle_column,
            add_n_branches = True,
        )
        
    
    if weight_column is not None:
        vectors = vectors*(df[weight_column].to_numpy().reshape(-1,1))

    vector_sum = vectors.sum(axis = 0)
    vector_sum = vector_sum/(np.linalg.norm(vector_sum))
    vector_sum = np.array(vector_sum).astype('float')
    
    df = pd.DataFrame(vector_sum).T
    curr_columns = pu.coordinate_columns(vector_name,axes=axes,suffix = suffix)
    df.columns = curr_columns
    df["dummy"] = "Hello"
    
    if scaled_by_dsi:
        df = pu.concat([df,dsi],axis = 1)
        df[[f"scaled_{k}" for k in curr_columns ]] = df[curr_columns].to_numpy()*df["dsi"].to_numpy()

    #print(f"df.columns = {df.columns}")
    return df#.reset_index(drop=True)



from python_tools import pandas_utils as pu
from python_tools import statistics_utils as stu
import seaborn as sns
import matplotlib.pyplot as plt
import morphology_analysis_utils as mau

def filter_df_by_n_vec_min(
    df,
    n_vec_min,
    verbose = False,
    ):
    if verbose:
        print(f"--Inside filter_df_by_n_vec_min--")
    scholl_df_counts = pu.count_unique_column_values(df,['segment_id','split_index'])
    #return scholl_df_counts
    scholl_df_counts_filt = scholl_df_counts.query(f"unique_counts >= {n_vec_min}")
    if verbose:
        print(f"# of seg before n_vec_min = {len(scholl_df_counts)}")
        print(f"# of seg AFTER n_vec_min = {len(scholl_df_counts_filt)}")
    return pu.intersect_df(
        df,
        scholl_df_counts_filt,
        append_restr_columns=False,
        verbose = verbose,
    )

def mean_compartment_vector(
    df,
    compartments = ['basal'],
    radius = 30_000,
    n_vec_min = 3,
    coordinate_name = "scholl",
    weight_column = None,
    axes = ['x','y','z'],
    axes_vector = ['x','z'],
    bin_width = 100_000,
    scaled_by_dsi=True,
    group_method = "bin",#"segment"
    group_columns = None,
    adjusted = False,
    verbose = True,
    verbose_n_vec_min = False,
    centroid_column = "endpoint_upstream",
    **kwargs
    ):
    
    restrictions = [
        f"compartment in {compartments}",
        f"adjusted == {adjusted}"
    ]
    if radius is not None:
        restrictions.append(f"radius == {radius}")

    scholl_df_restr = pu.query_table_from_list(
        df,
        restrictions
    ).reset_index(drop=True)
    
    if n_vec_min is not None:
        scholl_df_restr = filter_df_by_n_vec_min(
            scholl_df_restr,
            n_vec_min = n_vec_min,
            verbose = verbose_n_vec_min
        )
    #print(f"In mean vector, axes_vector = {axes_vector}")
    soma_vector_name = f"{coordinate_name}_soma_vector"
    def sum_vectors_args(df):
        #print(f"scale_by_dsi in func = {scaled_by_dsi}")
        return mou.sum_vectors(
            df,
            weight_column = weight_column,
            vector_name = soma_vector_name,
            axes=axes_vector,
            scaled_by_dsi=scaled_by_dsi,
        )

    # adding the soma coordinates 
    df_with_vec = mau.coordinate_soma_vectors_from_df(
        df = scholl_df_restr,
        coordinate_name = coordinate_name,
        group_opp_dir_vectors = False,
        centroid_column=centroid_column,
    )

    #adding the xz angles to the subgraphs
    df_with_vec_angle = mau.add_on_angles_2D(
            df_with_vec,
            vector_name = f"{coordinate_name}_soma_vector",
            axes = ['x','z'],   
    )
    
    #print(f"{df_with_vec_angle['scholl_soma_vector_xz_angle'].to_numpy()}")

    #bin the sugraphs by the axes
    df_binned = pu.bin_array_column(
        df = df_with_vec_angle,
        column = "centroid",
        suffix = "nm",
        verbose = verbose,
        bin_width=bin_width
    )

    # computing the mean vector for all neurons
    if group_columns is None:
        if group_method == "bin":
            group_columns = [
                f"centroid_bin_mid_{k}" for k in axes
            ]
        elif 'segment_id' in group_method:
            group_columns = ["segment_id","split_index"]
        else:
            raise Exception("")
            
    df_mean_vec = pu.apply_func_to_grouped_df(
        df = df_binned,
        group_columns = group_columns,
        func = sum_vectors_args,
        verbose = verbose,   
        )
    
    df_mean_vec = pu.sort_df_by_column(
        df_mean_vec,
        columns=group_columns,
        ascending=True,
    )
    
    return df_mean_vec

import matplotlib.pyplot as plt
from python_tools import matplotlib_utils as mu
def plot_2D_mean_vector(
    df,
    axes_2d = ['x','z'],
    column = "scholl_soma_vector",
    buffer = 100_000,
    plots_width = 2,
    skip_first = True,
    skip_last = True,
    single_plot_figsize = (4,3),
    divisor = 1000,
    plot_visual_boundary = True,
    title = None,
    radius = None,
    compartments = None,
    normalize_vectors = True,
    scale_factor = None,
    suptitle_y = 1.05,
    flip_z_axis = True,
    **kwargs
    ):

    color_map = hdju.visual_area_color_map(split_RL=True)
    color_map["RL"] = color_map["RL_1"]
    scatter_dict = hdju.visual_area_boundary_coordinates()


    if "centroid_bin_mid_y" in df.columns:
        n_plots = len(df["centroid_bin_mid_y"].unique())
        
        n_plots = n_plots - int(skip_first) - int(skip_last)

        n_rows = np.ceil(n_plots/plots_width).astype("int")
        figsize = np.array(single_plot_figsize)*np.array([plots_width,n_rows])
        fig,axes = plt.subplots(
            n_rows,
            plots_width,
            figsize = list(figsize))
        axes_rav = axes.ravel()
        df_currs,df_labels = pu.divide_df_by_column(
            df,
            "centroid_bin_mid_y",
            return_names = True
        )
    else: 
        n_plots = 1
        skip_first = False
        skip_last = False
        
        fig,ax = plt.subplots(
            1,
            1,
            figsize = list(single_plot_figsize)
        )
        
        axes_rav = [ax]
        df_labels,df_currs = [0],[df]

    counter = 0
    for j,(lab,df_curr) in enumerate(zip(df_labels,df_currs)):
        if j == 0 and skip_first:
            continue
        if counter >= len(axes_rav):
            continue
        if counter >= len(axes_rav) - 1 and skip_last:
            continue

        ax = axes_rav[counter]
        counter += 1

    #     print(f"-- Working on {lab} depth ({len(df_curr)} datapoints) --")
        centers_2d = pu.coordinates_from_df(
            df_curr,
            name="centroid_bin_mid",
            axes=axes_2d,
            suffix = None,
        )

        #print(f"column = {column}")
        vectors = pu.coordinates_from_df(
            df_curr,
            name=column,
            suffix = "nm",
            axes = axes_2d,
        )

        if normalize_vectors:
            vectors = vectors/(np.linalg.norm(vectors,axis = 1).reshape(-1,1))
            
        if scale_factor is not None:
            vectors = vectors*scale_factor

        ax.quiver(
            centers_2d[:,0]/divisor,
            centers_2d[:,1]/divisor,
            vectors[:,0]/divisor,
            vectors[:,1]/divisor,
        )
        ax.set_xlabel(f"{axes_2d[0]}")
        ax.set_ylabel(f"{axes_2d[1]}")
        ax.set_xlim([centers_2d[:,0].min()/divisor - buffer/divisor,
                     centers_2d[:,0].max()/divisor + buffer/divisor])
        ax.set_ylim([centers_2d[:,1].min()/divisor - buffer/divisor,
                     centers_2d[:,1].max()/divisor + buffer/divisor])

        if plot_visual_boundary: 
            for v_area,coords in scatter_dict.items():
                coords = coords#*hdju.voxel_to_nm_scaling
                #print(f"coords = {coords[:,0].min(),coords[:,0].max()}")
                ax.scatter(
                    coords[:,0]/divisor,
                    coords[:,2]/divisor,
                    c = color_map[v_area],
                    label = v_area,
                )
        if n_plots > 1:
            ax.set_title(f"Mean Depth {hdju.y_coordinate_from_centroid_y_nm(lab):.0f}")
            
        if flip_z_axis: 
            ax = mu.flip_ylim(ax)
        #ax.legend()
        #mu.set_le
    plt.tight_layout()
    if title is None:
        title = f"{column}\n Radius = {radius}\n comp = {compartments}"
    fig.suptitle(f"{title}",y=suptitle_y)
    plt.show()
    return axes_rav
    
    
def compute_and_plot_2d_mean_vector(
    df,
    compartments = ['oblique',],
    collapse_y = True,
    
    # --- for computing dataframe
    adjusted = False,
    n_vec_min = 3,
    bin_width=50_000,
    scaled_by_dsi = True,
    radius = 50_000,
    
    # -- plotting args --
    buffer = 100_000,
    plots_width = 2, #number of plots per row
    single_plot_figsize = (6,3),
    suptitle_y = 1.3,
    skip_first = True,
    skip_last = True,
    scale_factor = 3,
    flip_z_axis = True,
    **kwargs
    ):

    """
    Purpose: Want to compute and plot the 
    mean vector collapsed across all y 
    bins or over slices of the y bin
    """

    if collapse_y:
        axes = ['x','z']
    else:
        axes = ["x",'y','z']

    n = mou.mean_compartment_vector(
        df,
        n_vec_min=n_vec_min,
        axes = axes,
        radius = radius,
        compartments = compartments,
        bin_width=bin_width,
        scaled_by_dsi = scaled_by_dsi,
        adjusted=adjusted,
        **kwargs
    )

    if not scaled_by_dsi:
        column = "scholl_soma_vector"
    else:
        column = "scaled_scholl_soma_vector"

    return mou.plot_2D_mean_vector(
        n,
        axes_2d = ['x','z'],
        column=column,
        buffer = buffer,
        plots_width = plots_width,
        single_plot_figsize = single_plot_figsize,
        radius = radius,
        compartments = compartments,
        normalize_vectors = not scaled_by_dsi,
        scale_factor = scale_factor,
        suptitle_y = suptitle_y,
        skip_first = skip_first,
        skip_last = skip_last,
        flip_z_axis = flip_z_axis,
        **kwargs
    )
    
    
    
    
# def plot_2D_mean_vector_collapsed(
#     df,
#     axes_2d = ['x','z'],
#     column = "scholl_soma_vector",
#     buffer = 100_000,
#     plots_width = 2,
#     skip_first = True,
#     skip_last = True,
#     single_plot_figsize = (4,3),
#     divisor = 1000,
#     plot_visual_boundary = True,
#     title = None,
#     radius = None,
#     compartments = None,
#     normalize_vectors = True,
#     scale_factor = None,
#     ):

#     color_map = hdju.visual_area_color_map(split_RL=True)
#     color_map["RL"] = color_map["RL_1"]
#     scatter_dict = hdju.visual_area_boundary_coordinates()

#     figsize = single_plot_figsize
#     fig,ax = plt.subplots(
#         1,
#         1,
#         figsize = list(figsize))
    
#     counter = 0
#     df_curr = df
#     lab = 1


# #     print(f"-- Working on {lab} depth ({len(df_curr)} datapoints) --")
#     centers_2d = pu.coordinates_from_df(
#         df_curr,
#         name="centroid_bin_mid",
#         axes=axes_2d,
#         suffix = None,
#     )

#     vectors = pu.coordinates_from_df(
#         df_curr,
#         name=column,
#         suffix = "nm",
#         axes = axes_2d,
#     )

#     if normalize_vectors:
#         vectors = vectors/(np.linalg.norm(vectors,axis = 1).reshape(-1,1))

#     if scale_factor is not None:
#         vectors = vectors*scale_factor

#     ax.quiver(
#         centers_2d[:,0]/divisor,
#         centers_2d[:,1]/divisor,
#         vectors[:,0]/divisor,
#         vectors[:,1]/divisor,
#     )
#     ax.set_xlabel(f"{axes_2d[0]}")
#     ax.set_ylabel(f"{axes_2d[1]}")
#     ax.set_xlim([centers_2d[:,0].min()/divisor - buffer/divisor,
#                  centers_2d[:,0].max()/divisor + buffer/divisor])
#     ax.set_ylim([centers_2d[:,1].min()/divisor - buffer/divisor,
#                  centers_2d[:,1].max()/divisor + buffer/divisor])

#     if plot_visual_boundary: 
#         for v_area,coords in scatter_dict.items():
#             coords = coords#*hdju.voxel_to_nm_scaling
#             #print(f"coords = {coords[:,0].min(),coords[:,0].max()}")
#             ax.scatter(
#                 coords[:,0]/divisor,
#                 coords[:,2]/divisor,
#                 c = color_map[v_area],
#                 label = v_area,
#             )
#     ax.set_title(f"Mean Depth {hdju.y_coordinate_from_centroid_y_nm(lab):.0f}")
#     #ax.legend()
#     #mu.set_le
    
#     plt.tight_layout()
#     if title is None:
#         title = f"{column}\n Radius = {radius}\n comp = {compartments}"
#     fig.suptitle(f"{title}",y=1.2)
#     plt.show()
    
    
from python_tools import ipyvolume_utils as ipvu
import neuron_visualizations as nviz

def plot_mean_vector_for_segment_from_dict(
    vector_dict,
    axes = ["x","z"],
    show_at_end=True,
    new_figure=True,
    plot_mesh = True,
    size = None,
    plot_scholl_coordinates = True,
    radius_restriction = None,
    compartment = None,
    adjusted = False,
    verbose = False,
    axis_box_off = True,
    
    plot_start_coordinates = True,
    start_coordinates_color = "yellow",
    ):
    """
    Purpose: To plot the mean vector
    from a dictionary of the vectors
    """
    curr_dict = vector_dict
    segment_id = curr_dict["segment_id"]
    split_index = curr_dict["split_index"]
    
    if verbose:
        print(f"segment_id = {segment_id}_{split_index}")
    
    scholl_soma_vec = np.array([curr_dict[f"scholl_soma_vector_{ax}_nm"] if ax in axes else 0 for ax in ['x','y','z']])
    scholl_soma_vec_scaled = np.array([curr_dict[f"scaled_scholl_soma_vector_{ax}_nm"] if ax in axes else 0 for ax in ['x','y','z']])
    scholl_soma_vec_scaled

    soma_coord = hdju.soma_coordinate_nm(segment_id,split_index)
    soma_coord

    vectors = np.vstack([
        scholl_soma_vec,
        scholl_soma_vec_scaled
    ])
    centers = np.vstack([soma_coord]*len(vectors))
    ipvu.plot_quiver(
        centers = centers,
        vectors = vectors,
        show_at_end=show_at_end and not plot_mesh,
        new_figure=new_figure,
        size_array = size,
        
    )
    
    if plot_mesh is not None:
        if plot_scholl_coordinates:
            mau.plot_scholl_coordinates(
                segment_id = segment_id,
                split_index = split_index,
                radius_restriction = radius_restriction,
                verbose = False,
                verbose_colors = False,
                compartment=compartment,
                plot_mesh = False,
                append_figure = True,
                show_at_end = False,
                adjusted=adjusted,
            )
            
        if plot_start_coordinates:
            curr_coords = mau.subgraph_endpoint_coords_from_segment(
                segment_id,
                split_index = split_index,
                compartment = compartment,
            )
            nviz.plot_objects(
                scatters=[curr_coords],
                append_figure = True,
                show_at_end = False,
                scatters_colors= start_coordinates_color,
            )
        hdju.plot_proofread_neuron(
            segment_id,
            split_index = split_index,
            just_ipv_figure = True,
            append_figure = True,
            axis_box_off=axis_box_off,
            show_at_end = show_at_end,
            
        )
        
import seaborn as sns
import matplotlib.pyplot as plt
"""
How to compute circular orientation: 
https://docs.astropy.org/en/stable/api/astropy.stats.circcorrcoef.html

"""
import numpy as np
def func_struc_pref_dir_analysis(
    df,
    df_func,
    compartments = ['apical','oblique'],
    radius = 20_000,
    n_vec_min = 1,
    bin_width = [50_000,100_000,50_000],
    adjusted = True,
    plot = True,
    verbose = True,
    dsi_min_perc = 70,
    ):
    """
    Purpose: Try the correlation with preferred
    direction and geometry preferred direction
    for different compartments
    """
    from astropy.stats import circcorrcoef
    from astropy import units as u

    print(f"\n-- Working on compartments = {compartments}")
    df_mean_vec_seg = mou.mean_compartment_vector(
        df,
        n_vec_min=n_vec_min,
        radius = radius,
        compartments = compartments,
        scaled_by_dsi = True,
        bin_width=bin_width,
        group_method = "segment_id",
        adjusted = adjusted,
    )
    
    coordinate_name = "scholl"
    df_mean_vec_seg_angle = mau.add_on_angles_2D(
        df_mean_vec_seg,
        vector_name = f"{coordinate_name}_soma_vector",
        axes = ['x','z'],   
        rad = True,
    )
    
    df_mean_vec_seg_angle = hdju.add_nuc_to_df(df_mean_vec_seg_angle)
    
    if dsi_min_perc is not None:
        if verbose:
            print(f"Applying Structural DSI Filter:")
        df_mean_vec_seg_angle = pu.filter_df_by_column_percentile(
            df_mean_vec_seg_angle,
            columns = "dsi",
            percentile_lower = dsi_min_perc,
            percentile_upper = 100,
            verbose = verbose
        )
        if verbose:
            print(f"Applying Functional DSI Filter:")
        df_func = pu.filter_df_by_column_percentile(
            df_func,
            columns = "dsi_func",
            percentile_lower = dsi_min_perc,
            percentile_upper = 100,
            verbose = verbose
        )
        
    
    df_mean_vec_func = pu.intersect_df(
        df_mean_vec_seg_angle,
        df_func[['nucleus_id','pref_ori_func','pref_dir_func','dsi_func']],
        append_restr_columns=True,
    )
    
    
    if verbose:
        print(f"# of Neurons in Pref Ori Analysis = {len(df_mean_vec_func)}")
    
    
    x = "scholl_soma_vector_xz_radians"
    y = "pref_dir_func"
    data = df_mean_vec_func
    
    if plot:
        sns.jointplot(
            data = data,
            x = x,
            y = y,
            kind="hist",
            #colorbar = True
        )
        plt.show()
        
    circ_corr = circcorrcoef(
        data[x].to_numpy()*u.rad,
        data[y].to_numpy()*u.rad,
    )
    
    if verbose:
        print(f"circ_corr = {circ_corr}")
        
    return circ_corr

    
    
    
import morphology_ori_utils as mou
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
modules_to_set = [mou]

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


    
