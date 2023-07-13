'''



Purpose: funtionality for converting a neuron object to 
a graph representation that can be converted to a 2D/3D ativation maps






Ex 1: HOw to change between ravel and index

from python_tools import numpy_utils as nu
curr_act_map[nu.ravel_index([5,4,9],array_size)]






'''
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
from python_tools import numpy_dep as np

branch_attrs_for_G = [
'area', #surface area of branch
"compartment",
'axon_compartment', #whether it is axon or dendrite
"boutons_cdfs",
"boutons_volume",
"labels",
"mesh_center",
"endpoint_upstream",
"endpoint_downstream",
"mesh_volume",
"n_boutons",
"n_spines",
"n_synapses",
'n_synapses_head',
'n_synapses_neck',
'n_synapses_no_head',
'n_synapses_post',
'n_synapses_pre',
'n_synapses_shaft',
'n_synapses_spine',
"skeletal_length",
"spine_density",
"spine_volume_density",
"spine_volume_median",
"synapse_density",
'synapse_density_post',
'synapse_density_pre',
"total_spine_volume",
"width",
"width_new",
'soma_distance_euclidean',
'soma_distance_skeletal',
"skeleton_vector_upstream",
"skeleton_vector_downstream",
"width_upstream",
"width_downstream",
"min_dist_synapses_pre_upstream",
"min_dist_synapses_post_upstream",
"min_dist_synapses_pre_downstream",
"min_dist_synapses_post_downstream",
]

branch_attrs_limb_based_for_G = [
    "parent_skeletal_angle",
    "siblings_skeletal_angle_max",
    "siblings_skeletal_angle_min",
    "children_skeletal_angle_max",
    "children_skeletal_angle_min",
]



soma_attrs_for_G = [
    "area",
    "compartment",
    "mesh_center",
    ["mesh_center","endpoint_upstream"],
    "n_synapses",
    'n_synapses_post',
     'n_synapses_pre',
     'sdf',
    'side_length_ratios',
    'volume_ratio',
    ["volume","mesh_volume"],
]

features_to_output_default = [
#"area",
#"n_synapses_post",
#"n_synapses_pre",
"axon",
"dendrite",
"n_boutons",
#"mesh_volume",
#"n_spines",
"n_synapses_head",
#"n_synapses_neck",
"n_synapses_shaft",
#"n_synapses_no_head",
"skeletal_length",
#"spine_density",
"spine_volume_density",
"synapse_density",
"width_median_mesh_center"
]

soma_attrs_mapping_for_G = dict(
    mesh_volume = "volume",
)

neuorn_obj_attributes = [
    "description",
    "nucleus_id",
    "segment_id",
    "split_index",
]



graph_path = Path("/mnt/dj-stor01/platinum/minnie65/02/graphs")

def save_G_with_attrs(G,segment_id,
                      split_index = 0,
                      file_append="",
                      file_path = graph_path,
                      return_filepath=True,
                     ):
    """
    To save a Graph after processing
    
    Ex: 
    ctcu.save_G_with_attrs(G,segment_id=segment_id,split_index=split_index)
    """
    file_name = f"{segment_id}_{split_index}_neuron_graph"
    if len(file_append) > 0:
        file_name = file_name + file_append
        
    filepath = Path(file_path) / Path(file_name)
    f = su.compressed_pickle(G,filepath,return_filepath=True)
    
    if return_filepath:
        return f

def load_G_with_attrs(filepath):
    return su.decompress_pickle(filepath)

soma_name = "S0"

def soma_branch_df_split(df):
    return df.query(f"node == '{soma_name}'"),df.query("node != 'S0'")

def attr_value_by_node(df,node_name,attr):
    return df.query(f"node == '{node_name}'")[attr].to_list()[0]

def attr_value_soma(df,attr):
    """
    Ex: 
    ctcu.attr_value_soma(df_idx,"n_synapses")
    """
    return attr_value_by_node(df,soma_name,attr)


def mesh_center_xyz(center):
    return dict(x=center[0],y=center[1],z=center[2])

def width_new_extract(width_new):
    return {f"width_{k}":v for k,v in width_new.items()}
def labels_extract(labels):
    labels_to_check = ["basal","apical","soma","axon","dendrite"]
    return {k:int(k in "".join(labels)) for k in labels_to_check}
    
def axon_compartment_extract(axon_comp):
    return dict(axon=int(axon_comp == "axon"),
           dendrite=int(axon_comp == "dendrite"))

def boutons_volume_extract(bouton_volumes):
    return dict(bouton_volumes_median = np.median(bouton_volumes))

def boutons_cdfs_extract(bouton_cdfs):
    return dict(bouton_cdfs_median = np.median(bouton_cdfs))

special_params = dict(
    upstream_endpoint = mesh_center_xyz,
    labels = labels_extract,
    axon_compartment = axon_compartment_extract,
    boutons_volume= boutons_volume_extract,
    boutons_cdfs=boutons_cdfs_extract,
    width_new=width_new_extract,
)

def G_with_attrs_from_neuron_obj(
    neuron_obj,
    verbose = False,
    soma_attributes=soma_attrs_for_G,
    branch_attributes = branch_attrs_for_G,
    include_branch_dynamics = True,
    plot_G  = False,
    neuron_obj_attributes_dict = None,
    ):
    """
    To convert a neuron object to 
    a graph object with attributes stored
    
    Pseudocode: 
    1) Generate the total graph
    2) Assign the node attributes
    3) Assign the soma attributes
    """
    spu.set_neuron_head_neck_shaft_idx(neuron_obj)
    spu.set_neuron_synapses_head_neck_shaft(neuron_obj)
    spu.set_soma_synapses_spine_label(neuron_obj)
    syu.set_limb_branch_idx_to_synapses(neuron_obj)
    
    nst.soma_distance_skeletal_branch_set(neuron_obj)
    nst.soma_distance_euclidean_branch_set(neuron_obj)
    nst.upstream_endpoint_branch_set(neuron_obj)
    bu.set_branches_endpoints_upstream_downstream_idx(neuron_obj)
    
    if verbose:
            print(f"--whole_neuron_branch_concept_network")
    G_total = nru.whole_neuron_branch_concept_network(neuron_obj,
        directional= True,
        print_flag = verbose,
        with_data_in_nodes = False,
    )
    
    update_dict = dict()
    
    for s_name in neuron_obj.get_soma_node_names():
        if verbose:
            print(f"--Working on Soma {s_name}")
        s_dict = nru.branch_attr_dict_from_node(
                    neuron_obj[s_name],
                    node_name = s_name,
                    #attr_list=branch_attributes_global,
                    attr_list = soma_attributes,
                    include_node_name_as_top_key=True)
        
        update_dict.update(s_dict)
        
    #computing the limb attributes   
    limb_df = nst.stats_df(neuron_obj,
        functions_list=[eval(f"lu.{k}_{ns.limb_function_append_name}") 
                        for k in branch_attrs_limb_based_for_G])
    
    for limb_name in neuron_obj.get_limb_node_names():
        if verbose:
            print(f"-- Working on Limb {limb_name}")
        limb_obj = neuron_obj[limb_name]
        soma_start_vec = nst.soma_starting_vector(limb_obj,soma_center = neuron_obj["S0"].mesh_center)
        soma_start_angle = nst.soma_starting_angle(limb_obj,soma_center = neuron_obj["S0"].mesh_center)
        for branch_name in limb_obj.get_branch_names():
            if verbose:
                print(f"  --Branch {branch_name}")
            s_dict = nru.branch_attr_dict_from_node(
                    limb_obj[branch_name],
                    node_name = branch_name,
                    attr_list=branch_attributes,
                    #attr_list = soma_attributes_global,
                    include_node_name_as_top_key=False,
                    include_branch_dynamics = include_branch_dynamics)
            
            s_dict.update(nst.branch_stats_dict_from_df(limb_df,limb_name,branch_name))
            if branch_name == limb_obj.current_starting_node:
                s_dict["soma_start_vec"] = soma_start_vec
                s_dict["soma_start_angle"] = soma_start_angle
            update_dict.update({f"{limb_name}_{branch_name}":s_dict})

    
    xu.set_node_attributes_dict(G_total,update_dict)
    
    if plot_G:
        nx.draw(nx.Graph(G_total),with_labels = True)
        plt.show()
    
    for k in neuorn_obj_attributes:
        xu.set_graph_attr(G_total,k,getattr(neuron_obj,k))
        
    if neuron_obj_attributes_dict is not None:
        for k,v in neuron_obj_attributes_dict.items():
            xu.set_graph_attr(G_total,k,v)
    
    return G_total




def soma_center_from_df(df,col_suffix = ""):
    return df.query(f"node == '{soma_name}'")[[f"x{col_suffix}",
                                     f"y{col_suffix}",
                                     f"z{col_suffix}"]].to_numpy().reshape(3)


# stats_func_list = []
# def generate_new_stats(
#     G,
#     stats_func_list=
#     ):


def stats_df_from_G(
    G,
    no_attribute_default = 0,
    None_default = 0,
    attr_to_skip = ("side_length_ratios","sdf","mesh_center"),
    fix_presyns_on_dendrites = True,
    center_xyz_at_soma = True,
    ):
    """
    Purpose: To convert the data stored in a graph into a dataframe 
    where all columns are scalar values

    Things to figure out: 
    - Null value: 0
    - How to 1 hot encode things

    """
    
    df_dicts = []



    for n in G.nodes():
        curr_dict = {"node":n}
        attr_dict = xu.get_node_attribute_dict(G,node=n)

        if n == "S0":
            print(attr_dict)
        for k,v in attr_dict.items():
            if k in attr_to_skip:
                    continue
            #print(f"k = {k}")
            if v is None:
                curr_dict[k] = None_default
                #print(f"None activated")
                continue

            if special_params is not None:
                if k in special_params.keys():
                    curr_dict.update(special_params[k](v))
                    continue

                curr_dict[k] = v

        df_dicts.append(curr_dict)


    df = pu.fillna(pd.DataFrame.from_records(df_dicts),no_attribute_default)
    
    """
    Purpose: Will make sure no presyn information 
    appears on the dendritic branches
    
    """
    if fix_presyns_on_dendrites:
        idx_to_change = df.query("axon == 0").index
        df.loc[idx_to_change,"synapse_density"] = df.loc[idx_to_change,"synapse_density"]*df.loc[idx_to_change,"n_synapses_post"] / df.loc[idx_to_change,"n_synapses"]
        df.loc[idx_to_change,"n_synapses"] =  df.loc[idx_to_change,"n_synapses_post"]
        df.loc[idx_to_change,"n_synapses_pre"]  =  0
        df.loc[idx_to_change,"synapse_density_pre"]  =  0
    
    if center_xyz_at_soma:
        
        soma_center = soma_center_from_df(df)
        df.loc[:,"x"] = df.loc[:,"x"] - soma_center[0]
        df.loc[:,"y"] = df.loc[:,"y"] - soma_center[1]
        df.loc[:,"z"] = df.loc[:,"z"] - soma_center[2]
        
    
    return df


window_default = dict(
        x=[-np.inf,np.inf],
        y=[-np.inf,np.inf],
        z=[-np.inf,np.inf])

def symmetric_window(size=None,x=None,y=None,z=None):
    """
    Purpose: To Create a dict that will act like a window: 
    
    Ex: 
    ctcu.symmetric_window(x=100,y=200,z = 300)
    """
    curr_dict = dict()
    for ax,ax_val in zip(['x','y','z'],[x,y,z]):
        if ax_val is None:
            ax_val = size
        
        if ax_val is None:
            raise Exception("")
        
        curr_dict[ax] = [-ax_val,ax_val]
    return curr_dict

def plot_df_xyz(df,branch_size = 1,soma_size = 4,
               soma_color = "blue",branch_color = "red",
               col_suffix = "",
                flip_y = True,
                **kwargs,):
    soma_center = soma_center_from_df(df)
    all_points = df[[f"x{col_suffix}",f"y{col_suffix}",f"z{col_suffix}"]].to_numpy().reshape(-1,3)
    s_center = soma_center_from_df(df,col_suffix=col_suffix)
    nviz.plot_objects(scatters=[all_points,s_center.reshape(-1,3)],
                      scatters_colors=[branch_color,soma_color],
                     scatter_size=[branch_size,soma_size],
                      flip_y = flip_y,
                      axis_box_off = False,
                     **kwargs)
        

def filter_df_by_xyz_window(df,window=window_default,
                           verbose = True,
                           plot_xyz = False):
    """
    To restrict the rows to only those located at certain points:
    
    ctcu.filter_df_by_xyz_window(df,window = 50000,plot_xyz=True)
    
    """
    if window is None:
        window = window_default
    if type(window) in [float,int]:
        window = symmetric_window(window)

    if verbose:
        print(f"Before window apppilcation, len(df) = {len(df)}")
    df_result = df
    for col_name in window.keys():
        df_result = df_result.query(f"(({col_name} >= {window[col_name][0]}) and "
                                   f"({col_name} <= {window[col_name][1]}))")
        
    if verbose:
        print(f"AFTER window apppilcation, len(df) = {len(df_result)}")
        
    if plot_xyz:
        plot_df_xyz(df_result)
    
    return df_result

max_distance_soma_filtering = 50_000
def filter_df_by_soma_distance(
    df,
    max_distance = max_distance_soma_filtering,
    distance_type = "soma_distance_skeletal",
    verbose = True,
    plot_xyz = False,
    ):
    """
    Purpose: Will filter nodes that are only a 
    maximum distance away from the soma
    """
    if verbose:
        print(f"Before filtering , len(df) = {len(df)}")
    df_result = df.query(f"{distance_type} <= {max_distance}")
    if verbose:
        print(f"AFTER window apppilcation, len(df) = {len(df_result)}")
        
    if plot_xyz:
        plot_df_xyz(df_result)
    
    return df_result

def filter_df_by_axon_dendrite(
    df,
    dendrite=True,
    verbose = True,
    plot_xyz = False,
    cell_type = "inhibitory",
    ):
    """
    Purpose: Will filter nodes that are only a 
    maximum distance away from the soma
    """
    if verbose:
        print(f"Before filtering , len(df) = {len(df)}")
    if dendrite:
        col_name = "dendrite"
    else:
        col_name = "axon"
    df_result = df.query(f"({col_name} == 1) or (node == 'S0')")
    
    if cell_type == "excitatory" and dendrite:
        df_result = df.query(f"({' + '.join(['basal','apical','dendrite'])}   >= 1) or (node == 'S0')")
    else:
        df_result = df.query(f"({col_name} == 1) or (node == 'S0')")
        
    if verbose:
        print(f"AFTER AXON_DENDRTIE apppilcation, len(df) = {len(df_result)}")
        
    if plot_xyz:
        plot_df_xyz(df_result)
    
    return df_result


def filter_df_by_skeletal_length(
    df,
    min_skeletal_length = 10_000,
    dendrite=True,
    verbose = True,
    plot_xyz = False,
    ):
    """
    Purpose: Will filter nodes that are only a 
    maximum distance away from the soma
    """
    if verbose:
        print(f"Before filtering , len(df) = {len(df)}")

    df_result = df.query(f"(skeletal_length > {min_skeletal_length}) or (node == 'S0')")
    if verbose:
        print(f"AFTER skeletal length apppilcation, len(df) = {len(df_result)}")
        
    if plot_xyz:
        plot_df_xyz(df_result)
        
    return df_result
    


# ------------ Finding out the right mapping of idx -----------#
def axes_limits_from_df(
    df,
    all_axes_same_scale = False,
    neg_positive_same_scale = True,
    min_absolute_value = 5_000,
    global_scale = None,
    verbose = False
    ):

    axes_limits = dict(x=[],y=[],z=[])
    for col_name in axes_limits:
        axes_limits[col_name] =  [min(df[col_name].min(),-min_absolute_value),
                                  max(df[col_name].max(),min_absolute_value)]

    if neg_positive_same_scale:
        for col_name in axes_limits:
            abs_vals = np.abs(axes_limits[col_name])
            axes_limits[col_name] = [-np.max(abs_vals),np.max(abs_vals)]

    if all_axes_same_scale:
        max_val = np.max(np.abs(np.array(list(axes_limits.values())).ravel()))
        axes_limits = {k:[-max_val,max_val] for k in axes_limits}
    
    if global_scale is not None:
        max_val = global_scale
        axes_limits = {k:[-max_val,max_val] for k in axes_limits}

    if verbose:
        print(f"axes_limits = \n{axes_limits}\n")
        
    return axes_limits

def array_shape_from_radius(radius):
    edge_length = radius*2
    array_shape = (edge_length,edge_length,edge_length)
    return array_shape

def axes_limits_coordinates(axes_limits,array_shape=None,radius = None):
    if array_shape is None:
        array_shape = ctcu.array_shape_from_radius(radius)
        
    axes_limits_array = np.vstack([axes_limits[k] for k in ["x","y","z"]])

    limits_coords_by_axis = []
    for idx in range(len(axes_limits_array)):
        stacked_midpoints = []
        for j,ax_lim in enumerate(axes_limits_array[idx]):
            if j == 0:
                midpoints = np.linspace(ax_lim,0,int(array_shape[idx]/2 + 1))
            else:
                midpoints = np.linspace(0,ax_lim,int(array_shape[idx]/2 + 1))

            midpoints = (midpoints[1:] + midpoints[:-1])/2
            stacked_midpoints.append(midpoints)
        limits_coords_by_axis.append(np.concatenate(stacked_midpoints))

    return limits_coords_by_axis


def idx_for_col(
    val,
    col,
    axes_limits,
    nbins=20,
    verbose = False,
    no_soma_reservation=True):
    """
    Purpose: To find out the adjusted idx 
    for a mapping of a datapoint
    
    Pseudocode: 
    a) Figure out if positive or negative (assign -1 or 1 value)
    b) Get the right threshold (need axes_limits)
    c) Bin the value (need number of bins)
    d) Find the right index for the value
    
    
    Ex: 
    col = "x"
    verbose = True
    val = df.loc[20,col]
    nbins = 40

    col = "y"
    ctcu.idx_for_col(df.loc[100,col],col,
                axes_limits=axes_limits,
                verbose = True)
    """
    if verbose:
        print(f"Working on {col}, value = {val}, nbins = {nbins}")

    if val >= 0:
        sgn = 1
    else:
        sgn = -1

    lim = axes_limits[col][int((1+sgn)/2)]

    if verbose:
        print(f"sgn = {sgn}, lim = {lim}")

    bins = np.linspace(0,lim*sgn,nbins+1)
    idx = np.digitize(sgn*val,bins,right = True)
    
    #adds in correction if too large
    if idx == len(bins):
        idx = len(bins) -1

    idx_corrected = nbins + sgn*idx
    
    if no_soma_reservation:
        if sgn > 0:
            idx_corrected =idx_corrected- 1
    if verbose:
        print(f"idx= {idx}, idx_corrected = {idx_corrected}")
    
    return idx_corrected



def idx_xyz_to_df(
    df,
    
    #arguments for the axes scaling
    all_axes_same_scale = False,
    neg_positive_same_scale = True,
    global_scale = None,
    axes_limits =  None,
    
    radius = 10,
    verbose= True,
    plot_idx=False,
    ):

    """
    Purpose: To find the index of the data point based on the relative mesh center

    Pseudocode: 
    0) Determine the axes limits


    For each x,y,z column:
    For each datapoint:
    a) Figure out if positive or negative (assign -1 or 1 value)
    b) Get the right threshold (need axes_limits)
    c) Bin the value (need number of bins)
    d) Find the right index for the value
    """
    nbins = radius

    if axes_limits is None:
        axes_limits = ctcu.axes_limits_from_df(
            df,
            all_axes_same_scale = all_axes_same_scale,
            neg_positive_same_scale = neg_positive_same_scale,
            global_scale=global_scale,
            verbose = verbose)

    def y_idx(row):
        col = "y"
        return ctcu.idx_for_col(
            row[col],
            col,
            axes_limits=axes_limits,
            nbins=nbins,
        )

    def x_idx(row):
        col = "x"
        return ctcu.idx_for_col(
            row[col],
            col,
            nbins=nbins,
            axes_limits=axes_limits,
        )

    def z_idx(row):
        col = "z"
        return ctcu.idx_for_col(
            row[col],
            col,
            nbins=nbins,
            axes_limits=axes_limits,
        )

    df["x_idx"] = pu.new_column_from_row_function(df,x_idx)
    df["y_idx"] = pu.new_column_from_row_function(df,y_idx)
    df["z_idx"] = pu.new_column_from_row_function(df,z_idx)

    if plot_idx:
        ctcu.plot_df_xyz(
            df,
            branch_size = 1,
            soma_size = 4,
            col_suffix="_idx",
            buffer = nbins*0.25,)
        
    return df


def closest_node_idx_to_sample_idx(
    df,
    axes_limits,
    array_shape,
    verbose = False):
    """
    Purpose: To get the index of the closest node
    point to a coordinate in the sampling

    """
    limits_coords_by_axis = ctcu.axes_limits_coordinates(axes_limits,array_shape = array_shape)

    from pykdtree.kdtree import KDTree

    xi,yi,zi = np.meshgrid(*limits_coords_by_axis,indexing="ij")
    limits_coords = np.vstack([k.ravel() for k in [xi,yi,zi]]).T
    
#     if verbose:
#         print(f"limits_coords=\n{limits_coords}")

    limits_coords_kd = KDTree(df[["x","y","z"]].to_numpy())
    dist,closest_nodes = limits_coords_kd.query(limits_coords)
    if verbose:
        print(f"closest_nodes = {closest_nodes}")
        print(f"dist = {dist}")
    return closest_nodes


def attr_activation_map(
    df,
    attr,
    array_shape,
    return_vector = True,
    soma_at_end = False,
    exclude_soma_node = True,
    return_as_df = True,
    fill_zeros_with_closest_value = True,
    axes_limits=None,
    
    
    ):
    """
    To generate an activation map and to export it
    (as a multidimensional array or as a vector)
    
    Ex: 
    edge_length = radius*2
    array_size = (edge_length,edge_length,edge_length)

    attr = "mesh_volume"
    ctcu.attr_activation_map(df_idx,attr,array_shape = array_size,)
    """
    
    if soma_at_end and not return_vector:
        raise Exception("Must return vector for soma to be at end")

    curr_act_map = np.zeros(array_shape)
    placement_map = np.zeros(array_shape)
    
    if soma_at_end or exclude_soma_node:
        df_branch = ctcu.soma_branch_df_split(df)[1]
    else:
        df_branch = df
        
    attr_vals = df_branch[attr].to_numpy()
    attr_idx = df_branch[["x_idx","y_idx","z_idx"]].to_numpy(dtype='int')

    for j,(v,v_idx) in enumerate(zip(attr_vals,attr_idx)):
        placement_map[tuple(v_idx)] = 1
        if curr_act_map[tuple(v_idx)] < v:
            curr_act_map[tuple(v_idx)] = v 

    if return_vector:
        curr_act_map = curr_act_map.ravel()
        placement_map = placement_map.ravel()
        
    """
    -- 12/3 How to fill in the non-zero values
    
    """
    if fill_zeros_with_closest_value:
        if axes_limits is None:
            raise Exception("Need axes limits in order to fill in zeros")
            
#         print(f"len(df) = {len(df)}")
#         print(f"len(attr_vals) = {len(attr_vals)}")
        closest_branch_idx = ctcu.closest_node_idx_to_sample_idx(
            df_branch,
            axes_limits,
            array_shape,
            verbose = False
        )
        
        
        curr_act_map[placement_map <= 0] = attr_vals[closest_branch_idx[placement_map <= 0]]
        
        
    
    if soma_at_end:
        #print(len(curr_act_map))
        curr_act_map = np.concatenate([curr_act_map,[ctcu.attr_value_soma(df,attr)]])
        #print(len(curr_act_map))
        
    
    
    if return_as_df:
        if not return_vector:
            raise Exception("")
        col_names = [f"{attr}_"+"_".join([str(k) for k in np.unravel_index(k, array_shape)] )
                        for k in range(len(curr_act_map))]
        
        if soma_at_end:
            col_names.append(f"{attr}_soma")
        df_curr = pd.DataFrame([curr_act_map])
        df_curr.columns = col_names
        return df_curr
    else:
        return curr_act_map
    
    
def feature_map_df(df_idx,
                   array_shape,
                  features_to_output=features_to_output_default,
                   segment_id=12345,
                   split_index = 0,
                   axes_limits = None,
                   exclude_soma_node = True,
                   fill_zeros_with_closest_value = True,
                  ):
    """
    Will turn a dataframe with the indices of where to map the branch objects
    into a dataframe with the vector unraveled
    """
    
    f_maps = [ctcu.attr_activation_map(df_idx,
                                       attr,
                                       array_shape = array_shape,
                                      axes_limits = axes_limits,
                                      exclude_soma_node=exclude_soma_node,
                                      fill_zeros_with_closest_value = fill_zeros_with_closest_value) 
          for attr in features_to_output]
    f_df = pd.concat(f_maps,axis = 1)
    f_df["segment_id"] = segment_id
    f_df["split_index"] = split_index
    order_at_front = ["segment_id","split_index"]
    return f_df[order_at_front + 
                            [k for k in f_df.columns if k not in order_at_front]]



def no_spatial_df_from_df_filtered(df):
    new_data = dict()
    summed_values = [k for k in df.columns if k[:2] == "n_"] + ["skeletal_length","total_spine_volume"]
    weighted_attributes =[k for k in df.columns if "density" in k] + ["width_median_mesh_center",
                                                                      "width_no_spine_median_mesh_center",
                                                                      "spine_volume_median",
                                                                     "width","bouton_cdfs_median"]
    for v in summed_values:
        try:
            new_data[v] = df[v].sum()
        except:
            pass

    for v in weighted_attributes:
        try:
            new_data[v] = (df[v]*df["skeletal_length"]).sum()/new_data["skeletal_length"]
        except:
            pass

    syn_density_to_comp = ["shaft","head","neck","no_head"]
    for k in syn_density_to_comp:
        new_data[f"synapse_{k}_density"] = new_data[f"n_synapses_{k}"]/new_data["skeletal_length"]
                                                                       
                                                                       
    new_data["soma_volume"] = df.loc[0,"mesh_volume"]
    new_data["soma_n_synapses"] = df.loc[0,"n_synapses_post"]
    new_data["soma_synapse_density"] = df.loc[0,"n_synapses_post"]/df.loc[0,"area"]
    return pd.DataFrame.from_records([new_data])


#--- from neurd_packages ---
from . import branch_utils as bu
from . import limb_utils as lu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import spine_utils as spu
from . import synapse_utils as syu

#--- from python_tools ---
from python_tools import ipyvolume_utils as ipvu
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import pandas_utils as pu
from python_tools import system_utils as su

from . import cell_type_conv_utils as ctcu