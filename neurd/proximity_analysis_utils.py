
import datajoint as dj
import matplotlib.pyplot as plt
import pandas as pd
from pykdtree.kdtree import KDTree
import seaborn as sns
import time
from datasci_tools import numpy_dep as np

from datasci_tools import module_utils as modu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu


def conversion_rate_by_attribute_and_cell_type_pairs(
    df,
    hue = "e_i_predicted",
    hue_pre = None,
    hue_post = None,
    hue_pre_array = None,
    hue_post_array = None,
    attribute = "proximity_dist",
    attribute_n_intervals = 10,
    attribute_intervals=None,
    restrictions = None,
    presyn_width_max = 100,
    verbose = False,
    
    plot = False,
    ):

    """
    Purpose: To get the different conversion
    ratios for different cell types
    based on the contacts table
    """
    if restrictions is None:
        restrictions = []

    if presyn_width_max is not None:
        restrictions += [f"presyn_width < {presyn_width_max}",
                        ]
    
    restrictions_str = " and ".join([f"({k})" 
                                     for k in restrictions])
    
    if verbose:
        print(f"restrictions_str = {restrictions_str}")

    if attribute_intervals is None:
        attribute_intervals=np.linspace(0,df[attribute].max(),
                                        attribute_n_intervals+1)

    if attribute_intervals[0] == 0:
        attribute_intervals = attribute_intervals[1:]

    if hue_pre is None:
        hue_pre = hue
    if hue_post is None:
        hue_post = hue
    if hue_post not in df.columns or hue_post == hue_pre:
            hue_post = f"{hue_post}_post"
        
    if hue_pre_array is None:
        hue_pre_array = df[hue_pre].unique()
        
    if hue_post_array is None:
        hue_post_array = df[hue_post].unique()
    else:
        hue_post_array = hue_post
        
    syn_query = "NOT (synapse_id is NULL)"

    results_df = []
    for mdist in attribute_intervals:
        if verbose:
            print(f"Working on {attribute} < {mdist}")
        for c1 in hue_pre_array:
            for c2 in hue_post_array:
                curr_restr = (
                    f"({hue_pre}=='{c1}')"
                    f" and ({hue_post}=='{c2}')"
                    f" and ({attribute} < {mdist})"
                    )
                if len(restrictions_str)>0:
                    curr_restr += f"and {restrictions_str}"
                    
                curr_df = df.query(curr_restr)
                prox = len(curr_df)
                syn = len(curr_df.query("n_synapses > 0"))
                if prox > 0:
                    conversion = syn/prox*100
                else:
                    conversion = 0
                curr_dict = {
                    attribute: mdist,
                    hue_pre:c1,
                    hue_post:c2,
                    "n_prox": prox,
                    "n_syn":syn,
                    "conversion":conversion
                }
                results_df.append(curr_dict)
                if verbose:
                    
                    print(f"    {c1} -> {c2}: {conversion:.5f} % ({syn}/{prox})")

    results_df = pd.DataFrame.from_records(results_df)

    results_df["category"] = results_df[hue_pre] + "_to_" + results_df[hue_post]
    
    if plot: 
        ax = sns.lineplot(
            data = results_df,#ex_df,#.query(f"category == '{k}'"),
            x = attribute,
            y = "conversion",
            hue="category",
        )
    
    return results_df


def example_basal_conversion_rate(df=None,**kwargs):
    if df is None:
        prox_with_ct = vdi.proximity_with_gnn_cell_type_fine
        prox_with_ct

        prox_ct_table = vdi.df_from_table(
            prox_with_ct,
            features=[
                "e_i_predicted",
                "e_i_predicted_post",
                "n_synapses",
                "proximity_dist",
                "presyn_width",
                "postsyn_width",
                "postsyn_compartment"]
        )
    else:
        prox_ct_table = df

    apical_labels = ["apical_shaft","apical_tuft","apical","oblique","axon"]

    
    pxa.conversion_rate_by_attribute_and_cell_type_pairs(
        df = prox_ct_table.query(f"postsyn_compartment not in {apical_labels}"),
        attribute_n_intervals = 20,
        verbose = True,
        plot = True,
        **kwargs
    )
    
def conversion_df_from_proximity_df(
    df,
    in_place = False,
    verbose = False,
    sum_aggregation = True,):
    """
    Purpose: to turn a table with individual
    proximity entries into a table with the 
    source and target and the number of synapses
    and number of proximities

    Psuedocode: 
    """
    st = time.time() 
    if not in_place:
        df = df.copy()
        
    df["n_proximity"] = 1
    df = pu.replace_None_with_default(df,0)
    df = pu.replace_nan_with_default(df,0)
    df["n_synapses"] = df["n_synapses"].astype('int')
    
    agg_dict = {
        "n_synapses":"sum",
        "n_proximity":"sum",
        "proximity_dist":"mean",
        "presyn_width":"mean",
        "postsyn_compartment":pd.Series.mode,
        "e_i_predicted":pd.Series.mode,
        "e_i_predicted_post":pd.Series.mode,
        "cell_type_predicted":pd.Series.mode,
        "cell_type_predicted_post":pd.Series.mode}
    
    if sum_aggregation:
        df_conv = df.groupby(
            ["segment_id","split_index","segment_id_post","split_index_post"]
            ).sum().reset_index()
    else:
        agg_dict = {k:v for k,v in agg_dict.items() if k in df.columns}
        df_conv = df.groupby(
            ["segment_id","split_index","segment_id_post","split_index_post"]
            ).agg(agg_dict).reset_index()
    
    

    df_conv["conversion"] = df_conv["n_synapses"]/df_conv["n_proximity"]
    
    if verbose:
        print(f"# of pairs = {len(df_conv)} (time = {time.time() - st})")
        
    return df_conv
    



# -------------- For the functional proximity analysis ---------

#------ data fetching --------

# ------------ Postsyn side ----------------------

# ---- presyn side ----

# --- pairwise analysis 



def pairwise_presyn_proximity_onto_postsyn(
    segment_id,
    split_index=0,
    plot_postsyn = False,
    verbose = False,

    # --- for the postsyn side ---
    subgraph_type = "node",
    
    # --- for the presyn side ---
    presyn_table_restriction = None,
    proximity_restrictions = (
        "postsyn_compartment != 'soma'",
        ),
    ):
    """
    Purpose: To end up with a table that has the following

    For every pairwise proximitiy located on the same graph
    1) Skeletal/Euclidean Distance
    2) Pre 1 seg/split/prox_id, Pre 2 seg/split/prox id(Can use later to match the functional)
    3) Pre 1 n_synapses, Pre 2 n_synapses
    4) Can find anything else you want about the proximity using prox id

    Pseudocode: 

    -- getting the postsyn side --
    1) Download the postsyn graph object
    3) Create the graphs for all subgraphs and iterate through the graphs creating arrays of:
        a) generate the graph
        b) export the coordinates
        c) add to the graph idx vector and node idx vector
    4) Build a KD tree of all the coordinates
    -- Have the data we will match to 

    -- getting the presyn side --
    1) Get table of all proximities onto table
    2) Filter the table for only certain presyns (like functional matches)
    3) Filter the table by any other restrictions (like no soma synapses)
    3) Get segment_id,split_index,prox_id and postsyn_x_nm for all proximities
    4) Run the KDTree on all of the postsyn_x_nm of the proximities to find the nearest postsyn point
    5) Use closest point vector to index to create new vectors for each proximity of 
        i) graph_idx vector and node_idx vector,coordinate
        (put these in table)

    # -- To find pairwise only want to do inside the same graph ---
    For each unique graph (that made it in the graph_idx):
    1) Filter the whole df to only that graph (if 1 or less then return)
    2) For each node, find the graph distance between between all other node idxs not on row
    2b) Find the euclidean distance between node coordinate an all other node coordinates
    3) Store the result in the table as:
    - pre 1 seg/split/prox_id, pre 2 seg/split/prox_id, graph distance, euclidean dist

    """
    st = time.time()
    
    empty_df = pu.empty_df(
        columns = ['pre_1_segment_id', 'pre_1_split_index', 'pre_1_prox_id',
           'pre_1_n_synapses', 'pre_2_segment_id', 'pre_2_split_index',
           'pre_2_prox_id', 'pre_2_n_synapses', 'skeletal_distance',
           'euclidean_distance', 'segment_id', 'split_index'],
        )
    
    segment_id_post = segment_id
    split_index_post = split_index
    
    G = vdi.graph_obj_from_proof_stage(
        segment_id_post,
        split_index_post
    )

    if plot_postsyn:
        #nxu.plot(G)
        vdi.plot_proofread_neuron(
            segment_id_post,
            split_index_post,
            )
        
    # ----- postsyn side of data ---------
    """
    3) Create the graphs for all subgraphs and iterate through the graphs creating arrays of:
        a) generate the skeleton graph
        b) export the coordinates
        c) add to the graph idx vector and node idx vector
    """
    if subgraph_type == "limb":
        G_subs = nxu.all_limb_graphs_off_soma(G)
    elif subgraph_type == "node":
        G_subs = nxu.all_node_graphs(G)
    else:
        raise Exception(f"Unknown subgraph type = {subgraph_type}")

    if verbose:
        print(f"# of subgraphs = {len(G_subs)}")
        
    if len(G_subs) == 0:
        if verbose:
            print(f"No node graphs")
            
        return empty_df
        
        
    skeleton_graphs = []
    df_graphs = []
    for g_idx,gsub in enumerate(G_subs):
        gsub = G_subs[g_idx]

        #1) Generate the skeleton graph
        g_sk = nxu.skeleton_graph(gsub)
        skeleton_graphs.append(g_sk)

        #nx.draw(g_sk,with_labels = True)
        curr_df = xu.node_df(g_sk,node_id = "node_idx")
        curr_df["graph_idx"]  = g_idx
        df_graphs.append(curr_df)

    df_graphs = pu.concat(df_graphs).reset_index(drop=True)

    postsyn_coords = np.vstack(df_graphs["coordinates"].to_numpy())
    post_kd = KDTree(postsyn_coords)
    
    # ---------------- Presyn/Proximity Preprocessing -------------------
    """
    -- getting the presyn side --
    1) Get table of all proximities onto table
    2) Filter the table for only certain presyns (like functional matches)
    3) Filter the table by any other restrictions (like no soma synapses)
    3) Get segment_id,split_index,prox_id and postsyn_x_nm for all proximities
    4) Run the KDTree on all of the postsyn_x_nm of the proximities to find the nearest postsyn point
    5) Use closest point vector to index to create new vectors for each proximity of 
        i) graph_idx vector and node_idx vector,coordinate
        (put these in table)
    """
    
    # ---------------- Presyn/Proximity Preprocessing -------------------
    """
    -- getting the presyn side --
    1) Get table of all proximities onto table
    2) Filter the table for only certain presyns (like functional matches)
    3) Filter the table by any other restrictions (like no soma synapses)
    3) Get segment_id,split_index,prox_id and postsyn_x_nm for all proximities
    4) Run the KDTree on all of the postsyn_x_nm of the proximities to find the nearest postsyn point
    5) Use closest point vector to index to create new vectors for each proximity of 
        i) graph_idx vector and node_idx vector,coordinate
        (put these in table)
    """

    #1) Get table of all proximities onto table
    prox_postsyn_table = vdi.proximity_table & dict(segment_id_post=segment_id_post,
                                                    split_index_post = split_index_post)
    if verbose:
        print(f"# of proximities = {len(prox_postsyn_table)}")

    #2) Filter the table for only certain presyns (like functional matches)
    if presyn_table_restriction is not None:
        prox_postsyn_table = prox_postsyn_table & (dj.U('segment_id','split_index') & presyn_table_restriction)

    if verbose:
        print(f"After presyn segment id restrictions, # of proximities = {len(prox_postsyn_table)}")

    #3) Filter the table by any other restrictions (like no soma synapses)
    prox_postsyn_table = vdi.restrict_table_from_list(
        prox_postsyn_table,
        proximity_restrictions,
    )

    if verbose:
        print(f"After restrictions, # of proximities = {len(prox_postsyn_table)}")

    
    # ***** add in loop where checks 
    if len(prox_postsyn_table) <= 0:
        if verbose:
            print(f"No Proximities")
        return empty_df

    prox_df = vdi.df_from_table(
        prox_postsyn_table,
        features=[
            "segment_id",
            "split_index",
            "prox_id",
            "postsyn_proximity_x_nm",
            "postsyn_proximity_y_nm",
            "postsyn_proximity_z_nm",
            "postsyn_compartment",
            "n_synapses"]
    )

    prox_df_coords = vdi.coordinates_from_df(
        prox_df,
        name="postsyn_proximity"
    )

    dist,clos_idx = post_kd.query(
        prox_df_coords,
    )

    """
    5) Use closest point vector to index to create new vectors for each proximity of 
        i) graph_idx vector and node_idx vector,coordinate
        (put these in table)
    """
    df_pre_prox = pd.concat([df_graphs.iloc[clos_idx,:].reset_index(drop=True),prox_df],axis=1)
    

    # ----------------- Doing Pairwise Analysis ------------------------------
    """
    # -- To find pairwise only want to do inside the same graph ---
    For each unique graph (that made it in the graph_idx):
    1) Filter the whole df to only that graph (if 1 or less then return)
    2) For each node, find the graph distance between between all other node idxs not on row
    2b) Find the euclidean distance between node coordinate an all other node coordinates
    3) Store the result in the table as:
    - pre 1 seg/split/prox_id, pre 2 seg/split/prox_id, graph distance, euclidean dist
    """

    unique_graph_idx = df_pre_prox["graph_idx"].unique()

    pairwise_dfs = []

    for graph_idx in tqdm(unique_graph_idx):
        curr_graph_df = pu.sort_df_by_column(df_pre_prox.query(f"graph_idx == {graph_idx}").reset_index(drop=True),
        [
            "segment_id","split_index","prox_id"
        ],ascending=True)


        if len(curr_graph_df) <= 1:
            continue

        sk_graph = skeleton_graphs[graph_idx]

        #2) For each node, find the graph distance between between all other node idxs not on row
        node_dicts = pu.df_to_dicts(curr_graph_df)
        for j,curr_dict in enumerate(node_dicts):

            # get pre_1 info
            curr_node = curr_dict["node_idx"]
            curr_coord = curr_dict["coordinates"]
            pre_1_segment_id = curr_dict["segment_id"]
            pre_1_split_index = curr_dict["split_index"]
            pre_1_prox_id = curr_dict["prox_id"]
            pre_1_n_synapses = curr_dict["n_synapses"]

            # getting all pre_2 info
            mask = np.ones(len(curr_graph_df)).astype('bool')
            mask[np.arange(j+1)] = False
            other_df = curr_graph_df.iloc[mask,:].reset_index(drop=True)

            graph_distances = []
            euclidean_distances = []

            for other_dict in pu.df_to_dicts(other_df):
                graph_distances.append(xu.shortest_path_length(
                    sk_graph,
                    curr_node,
                    other_dict["node_idx"],
                    weight = "weight"
                ))

                euclidean_distances.append(np.linalg.norm(curr_coord-other_dict["coordinates"]))

            # assemble the dataframe to save off the pairwise computation
            pre_2_df = pu.rename_columns(
                other_df[["segment_id","split_index","prox_id","n_synapses"]],
                dict(segment_id="pre_2_segment_id",
                    split_index = "pre_2_split_index",
                    prox_id = "pre_2_prox_id",
                    n_synapses="pre_2_n_synapses"))
            pre_2_df["pre_1_segment_id"] = pre_1_segment_id
            pre_2_df["pre_1_split_index"] = pre_1_split_index
            pre_2_df["pre_1_prox_id"] = pre_1_prox_id
            pre_2_df["pre_1_n_synapses"] = pre_1_n_synapses

            pre_2_df = pre_2_df[[k for k in pre_2_df.columns if "_1_" in k] +
                                [k for k in pre_2_df.columns if "_2_" in k]]

            # adding on the distances computed
            pre_2_df["skeletal_distance"] = graph_distances
            pre_2_df["euclidean_distance"] = euclidean_distances

            pairwise_dfs.append(pre_2_df)


    pairwise_dfs = pu.concat(pairwise_dfs)

    if verbose:
        print(f"After pairwise analysis (before same node filtering) = {len(pairwise_dfs)}")
    # getting rid of the same segment_ids
    pairwise_dfs = pairwise_dfs.query(f"not ((pre_1_segment_id == pre_2_segment_id) and (pre_1_split_index == pre_2_split_index))")

    if verbose:
        print(f"After pairwise analysis (AFTER same node filtering) = {len(pairwise_dfs)}")

    if len(pairwise_dfs.query(f"(pre_1_segment_id == pre_2_segment_id) and (pre_1_split_index == pre_2_split_index) and (pre_1_prox_id == pre_2_prox_id) ")) > 0:
        raise Exception("Repeat proximity")

    pairwise_dfs[["segment_id","split_index"]] = segment_id_post,split_index_post
    

    print(f"\n\n\n ------------ Total Time for Run: {time.time() - st}")
    
    return pairwise_dfs

def example_pairwise_postsyn_analysis():
    seg_df = vdi.df_from_table(
        (vdi.proofreading_neurons_table & "dendrite_skeletal_length > 100000"),
        features=["segment_id","split_index","cell_type"],
    )

    curr_df = pu.randomly_sample_df(
        seg_df,
        n_samples=1000
    )

    for segment_id,split_index in zip(curr_df["segment_id"].to_numpy(),
                                     curr_df["split_index"].to_numpy()):
        print(f"\n\n\n----- Working on {segment_id}_{split_index}")
        pxa.pairwis_df = pairwise_presyn_proximity_onto_postsyn(
            segment_id,
            split_index,
            plot_postsyn = False,
            verbose = True,

            # --- for the postsyn side ---
            subgraph_type = "node",

            # --- for the presyn side ---
            presyn_table_restriction = vdi.functional_tuning_table_raw,
        )
        
def add_euclidean_dist_to_prox_df(
    df,
    centroid_df=None,
    in_place = False,
    add_signed_single_axes_dists = True,
    add_depth_dist = False,
    ):
    """
    Purpose: To add the pre and post euclidean
    distances to an edge dataframe with the proximities
    """

    if centroid_df is not None:
        df_with_centr = pu.append_df_to_source_target(
            df,
            centroid_df,
            source_name="presyn",
            target_name="postsyn",
            on = 'node',
            in_place=in_place,
        )
    else:
        if not in_place:
            df = df.copy()
        
        df_with_centr = df
        
    if add_depth_dist:
        df_with_centr = vdi.add_depth_to_df(
            df_with_centr,
            coordinate_base_name = [
                "presyn_centroid",
                "postsyn_centroid",
                "presyn_proximity",
                "postsyn_proximity",
            ],
            verbose = True,
        )

    axes = ["x","y","z"]
    for t in ['presyn','postsyn']:
        df_with_centr[f"{t}_euclidean_distance"] = pu.distance_between_coordinates(
            df_with_centr,
            coordinate_column_1 = f"{t}_centroid",
            coordinate_column_2= f"{t}_proximity",
            axes = axes,
        )

        for axes_opt in nu.choose_k_combinations(axes,2):
            name_suffix = "".join(axes_opt)
            df_with_centr[f"{t}_euclidean_distance_{name_suffix}"] = pu.distance_between_coordinates(
                df_with_centr,
                coordinate_column_1 = f"{t}_centroid",
                coordinate_column_2= f"{t}_proximity",
                axes = axes_opt,
            )
            
        if add_signed_single_axes_dists:
            if add_depth_dist:
                curr_axes = axes + ["depth"]
            else:
                curr_axes = axes
            for ax in curr_axes:
                df_with_centr[f"{t}_distance_{ax}"] = (
                    df_with_centr[f"{t}_proximity_{ax}_nm"] - 
                    df_with_centr[f"{t}_centroid_{ax}_nm"] 
                )

    return df_with_centr


def conversion_rate(df):
    if len(df) == 0:
        return np.nan
    return df['n_synapses'].sum()/len(df) 

def plot_prox_func_vs_attribute_from_edge_df(
    edge_df,
    source = "Exc",
    targets = ["Exc","Inh"],

    column = "presyn_skeletal_distance_to_soma",

    divisor = 1000,
    hue = "connection_type",

    percentile_upper = 99,
    percentile_lower= 0,
    func = conversion_rate,
    bins = None,
    n_bins = 10,
    equal_depth_bins = True,

    # ploting parameters
    data_source = "H01",
    axes_fontsize = 35,
    title_fontsize = 40,
    tick_fontsize = 30,
    legend_fontsize = 25,
    title_pad = 15,
    legend_title = "Connection Type",
    linewidth = 3,
    xlabel = r"Axon Distance to Soma ($\mu m$)",
    ylabel = f"Mean Conversion Rate",
    title = f"Conversion Rate vs. Axon\n Distance to Soma",
    ax = None,
    figsize = (8,7),
    add_scatter = True,
    scatter_size = 100,
    verbose = False,
    verbose_bin_df = False,
    legend_label=None,
    
    bins_mid_type = "weighted",
    return_n_dict = False,
    ):
    """
    Purpose: To plot the a function of the proximity 
    df as a function of an attribute of the proximities
    (like presyn distance)
    """
    targets = nu.to_list(targets)

    if source is not None:
        title = f"{title} ({data_source})"

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize = figsize)

    edge_df_filt = pu.filter_df_by_column_percentile(
        edge_df,
        columns=column,
        percentile_lower=percentile_lower,
        percentile_upper=percentile_upper,
    )

    edge_df_filt[column] = edge_df_filt[column]/divisor

    n_dict = dict()
    source_list = nu.to_list(source)
    for source in source_list:
        for target in targets:
            label = f'{source} Onto {target}'
            df_to_plot = edge_df_filt.query(f"{hue} == '{label}'")


            exc_post_stats,exc_post_bins,exc_post_len = pu.bin_df_by_column_stat(
                df = df_to_plot,
                func = func,
                column = column,
                return_bins_mid = True,
                bins_mid_type= bins_mid_type,
                bins=bins,
                equal_depth_bins=equal_depth_bins,
                n_bins=n_bins,   
                return_df_len = True,
                return_std = False,
                verbose_bins = verbose_bin_df,
            )
            
            
            if verbose:
                print(f"For {label}:")
                print(f"    bins mid = {exc_post_bins}")
                print(f"    bins length = {exc_post_len}")
            #print(f"bin len = {exc_post_len}")
            
            n_dict[label] = dict(
                datapoint = np.array(exc_post_bins),
                n_prox = np.array(exc_post_len),
            )


            if legend_label is not None:
                label = f"{label} ({legend_label})"
            color =  npp.exc_inh_combination_palette.get(label,None)
            ax.plot(
                exc_post_bins,
                exc_post_stats,
                label=label,
                c = color,
                linewidth = 3,
            )

            if add_scatter:
                ax.scatter(
                    exc_post_bins,
                    exc_post_stats,
                    #label=label,
                    c = color,
                    s = scatter_size
                )

    ax.set_xlabel(xlabel,fontsize=axes_fontsize)
    ax.set_ylabel(ylabel,fontsize = axes_fontsize)
    ax.set_title(
        title,
        fontsize = title_fontsize,
        pad = title_pad
    )
    ax.legend()
    #mu.set_axes_outside_seaborn(ax)
    mu.set_legend_title(ax,legend_title)
    mu.set_legend_fontsizes(ax,legend_fontsize)
    mu.set_axes_tick_font_size(ax,tick_fontsize)
    
    if return_n_dict:
        return ax,n_dict
    else:
        return axs
    
def print_n_dict(
    n_dict,
    category_joiner = "\n",
    verbose = True):
    """
    Purpose: To print out the category and n_proximity dict
    from the optional returned n_dict datastructure from
    plot_prox_func_vs_attribute_from_edge_df

    Pseudocode:
    1) Iterate through all keys
        a. get the datapoints (round to 2 decimal places)
        b. Get proximity
        c. Create str for all datapoints as 
            xi (n_prox = pi),
    2) Append all category strings
    """

    categories_strs = []
    for c,d in n_dict.items():
        cat_str = f"{c}: " + ", ".join([f"{np.round(xi,2)} (n_prox = {pi})" 
                             for xi,pi in zip(d['datapoint'],d['n_prox'])])
        categories_strs.append(cat_str)

    total_str = category_joiner.join(categories_strs)

    if verbose:
        print(f"total_str = \n{total_str}")
        
    return total_str

def conversion_df(
    proximity_df,
    presyn_column = "presyn",
    postsyn_column = "postsyn",
    separate_compartments = True,
    separate_by_neuron_pairs = False,
    verbose = False
    ):
    """
    Purpose: given a proximity table,
    will calculate the conversion df (
    for potentially different compartments
    )
    """
    st = time.time()
    df = proximity_df.copy()
    
    # --- preprocessing dataframe ---
    df["n_proximity"] = 1
    df = pu.replace_None_with_default(df,0)
    df = pu.replace_nan_with_default(df,0)
    df["n_synapses"] = df["n_synapses"].astype('int')
    
    
    agg_dict = {
        "n_synapses":"sum",
        "n_proximity":"sum",
        "proximity_dist":"mean",
        "presyn_width":"mean",
        #"postsyn_compartment":pd.Series.mode,
    }
    
    agg_dict = {k:v for k,v in agg_dict.items() if k in df.columns}
    
    if separate_by_neuron_pairs:
        id_columns = nu.to_list(presyn_column) + \
                     nu.to_list(postsyn_column)
    else:
        id_columns = []
    
    if separate_compartments:
        id_columns += ['postsyn_compartment']
    
    df_conv = df.groupby(
            id_columns
            ).agg(agg_dict).reset_index()
    
    df_conv["conversion"] = df_conv["n_synapses"]/df_conv["n_proximity"]
    
    if verbose:
        print(f"# of pairs = {len(df_conv)} (time = {time.time() - st:.3f})")
        
    return df_conv


def str_of_n_prox_n_syn(
    df,
    category_column = "category",
    independent_axis = "proximity_dist",
    prox_column = "n_prox",
    syn_column = "n_syn",
    category_str_joiner = "\n",
    verbose= False
    ):
    """
    Purpose: Printing the n_prox and n_syn for each datapoint
    in a conversion by x plot

    Pseudocode: 
    1) Iterate through each category
        a. Get the proximity dist (round to 2 decimal places)
        b. Get the n_prox
        c. Get the n_syn

        d. cretae a string concatenated for all prox dist of
            prox_dist (n_prox = x, n_syn = y),
    2) Concatenate all category strings with \n
    """
    
    categories = set(df[category_column].to_list())
    category_strs =[] 
    for c in categories:
        df_sub = df.query(f"{category_column} == '{c}'")
        x = df_sub[independent_axis].to_list()
        x = np.round(x,2)
        p = df_sub[prox_column].to_list()
        s = df_sub[syn_column].to_list()
        
        cat_str = f"{c}: " + ", ".join([f"{xi} (n_prox={pi}, n_syn = {si})"
                                       for xi,pi,si in zip(x,p,s)])
        category_strs.append(cat_str)

    total_str = category_str_joiner.join(category_strs)
    if verbose:
        print(f"total_str = \n{total_str}")
        
    return total_str

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
# modules_to_set = [pxa]

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

#--- from neuron_morphology_tools ---


#--- from neurd_packages ---
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mvu
from . import nature_paper_plotting as npp

#--- from neuron_morphology_tools ---
from neuron_morphology_tools import neuron_nx_utils as nxu

#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu
from datasci_tools import module_utils as modu 
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools.tqdm_utils import tqdm

from . import proximity_analysis_utils as pxa