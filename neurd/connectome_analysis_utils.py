
import matplotlib.pyplot as plt
import pandas as pd
from python_tools import numpy_dep as np

def restrict_edge_df_by_types_compartment(
    edge_df,
    synapse_type = "postsyn",
    cell_type = None,
    cell_type_attribute = "gnn_cell_type",
    cell_type_fine = None,
    cell_type_fine_attribute = "gnn_cell_type_fine",
    compartment = None,
    restriction_dict = None,
    
    #for the max skeletal length
    presyn_skeletal_distance_to_soma_max = None,
    presyn_skeletal_distance_to_soma_min = None,
    
    postsyn_skeletal_distance_to_soma_max = None,
    postsyn_skeletal_distance_to_soma_min = None,
    
    #for the min/max euclidean length
    presyn_soma_euclid_dist_max = None,
    presyn_soma_euclid_dist_min = None,
    
    postsyn_soma_euclid_dist_max = None,
    postsyn_soma_euclid_dist_min = None,
    
    verbose = True,
    
    return_name = False,
    add_number_of_cells_to_name = True
    ):

    if restriction_dict is None:
        restriction_dict = dict()
        
        
    restriction_dict.update({
        cell_type_attribute:cell_type,
        cell_type_fine_attribute:cell_type_fine,
        "compartment":compartment,
    })
    
    name = f"{synapse_type}"
    
    if verbose:
        print(f"restriction_dict = {restriction_dict}")
        
    edge_df_restr = edge_df
        
    for att_name,att_value in restriction_dict.items():
        if att_value is not None:
            if verbose:
                print(f"Before {att_name} restrition: {len(edge_df_restr)}")

            if att_name == "compartment":
                edge_df_restr = edge_df_restr[edge_df_restr[f"{synapse_type}_{att_name}"].str.contains(att_value)]
            else:
                edge_df_restr = edge_df_restr.query(f"{synapse_type}_{att_name} == '{att_value}'")
            name += f"_{att_value}"

            if verbose:
                print(f"After {att_name} restrition: {len(edge_df_restr)}")
                
        
    dist_dict = dict(
        presyn_skeletal_distance_to_soma_max = presyn_skeletal_distance_to_soma_max,
        presyn_skeletal_distance_to_soma_min = presyn_skeletal_distance_to_soma_min,

        postsyn_skeletal_distance_to_soma_max = postsyn_skeletal_distance_to_soma_max,
        postsyn_skeletal_distance_to_soma_min = postsyn_skeletal_distance_to_soma_min,

        #for the min/max euclidean length
        presyn_soma_euclid_dist_max = presyn_soma_euclid_dist_max,
        presyn_soma_euclid_dist_min = presyn_soma_euclid_dist_min,

        postsyn_soma_euclid_dist_max = postsyn_soma_euclid_dist_max,
        postsyn_soma_euclid_dist_min = postsyn_soma_euclid_dist_min,
    )
    
    for att_name,att_value in dist_dict.items():
        if att_value is not None:
            if "max" in att_name:
                comparison_str = "<="
            else:
                comparison_str = ">="
                
            col_name = att_name[:-4]
                
            name += f", {col_name} {comparison_str} {att_value}"
            
            if verbose:
                print(f"Before {col_name} restrition: {len(edge_df_restr)}")

            edge_df_restr = edge_df_restr.query(f"{col_name} {comparison_str} {att_value}")

            if verbose:
                print(f"After {col_name} restrition: {len(edge_df_restr)}")
            
    if add_number_of_cells_to_name:
        try:
            name += f"\nNumber of Presyn Cells = {len(edge_df_restr['source'].unique())}"
            name += f"\nNumber of Postsyn Cells = {len(edge_df_restr['target'].unique())}"
        except:
            name += f"\nNumber of Presyn Cells = {len(edge_df_restr['presyn'].unique())}"
            name += f"\nNumber of Postsyn Cells = {len(edge_df_restr['postsyn'].unique())}"
            
            
    if return_name:
        return edge_df_restr,name
    else:
        return edge_df_restr
    
    

cell_type_fine_labels_global = ['23P', '4P', '5P-IT', '5P-NP', '5P-PT','6P-CT','6P-IT','BC',"BPC","MC","NGC","Pvalb","SST"]

def cell_count(df,synapse_type,return_str = True,):
    if synapse_type == "presyn":
        id_name = "source"
    else:
        id_name = "target"
    counts = {k:len(df.query(f"{synapse_type}_gnn_cell_type_fine=='{k}'")[id_name].unique()) for k in cell_type_fine_labels_global}
    if return_str:
        ret_str = f"\n{synapse_type.capitalize()}s:"
        for j,(k,v) in enumerate(counts.items()):
            ret_str += f"{k} ({v}), " 
            if (j%4 == 3) and j != len(counts)-1:
                ret_str+="\n"
        return ret_str
    else:
        return counts

# def plot_histogram_discrete_labels_old(
#     edge_df,
#     restrictions_dicts= None,
#     compartment_labels = cell_type_fine_labels_global,#["apical"]
#     cell_type_fine_labels = None,
#     synapse_type = "postsyn",
#     histogram_attribute = "presyn_skeletal_distance_to_soma",
#     twin_color = "blue",
#     normalize=True,
#     cumulative=True,
#     verbose = True,
#     labels = None,
#     y_label = None,
#     x_label = None,
#     title = None,
#     add_cell_counts_to_title = True,
#     fontsize_title = None,
#     fontsize_axes = 16,
#     nbins = 100,
#     legend = False,
#     **kwargs
#     ):

    
#     if labels is None:
#         labels = cell_type_fine_labels_global

#     if synapse_type == "postsyn":
#         discrete_label = "presyn_gnn_cell_type_fine"
#         opposite_synapse_type = "presyn"
#     else:
#         discrete_label = "postsyn_gnn_cell_type_fine"
#         opposite_synapse_type = "postsyn"
        
    

#     if restrictions_dicts is None:
#     #for the discrete histogramming
#         restrictions_dicts = []
#         if cell_type_fine_labels is not None:
#             restrictions_dicts = [dict(cell_type_fine=k) for k in cell_type_fine_labels]

#         if compartment_labels is not None:
#             if len(restrictions_dicts) == 0:
#                 restrictions_dicts = [dict(compartment=k) for k in compartment_labels]
#             else:
#                 for d in restrictions_dicts:
#                     for k in compartment_labels:
#                         d["compartment"] = k
        
                

#     for rd in restrictions_dicts:
#         restr_df,name = cona.restrict_edge_df_by_types_compartment(
#             edge_df,
#             verbose = False,
#             synapse_type = synapse_type,
#             return_name=True,
#             **rd,
#             **kwargs
#         )

#         if verbose:
#             print(f"\n\n\n--- Working on {name}: # of edges = {len(restr_df)} ---")

#         if len(restr_df) == 0:
#             continue

#         df_counts = pu.histogram_of_discrete_labels(
#             restr_df,
#             y=discrete_label,
#             normalize=normalize,
#             cumulative=cumulative,
#             x = histogram_attribute,
#             #x_steps = np.linspace(0,1000,100),
#             plot = False,
#             nbins=nbins,
#             )


#         #print(f"df_counts = {len(df_counts)}")

#         ax,ax2 = mu.stacked_bar_graph(
#             df_counts,
#             color_dict = ctu.cell_type_fine_color_map,
#             set_legend_outside_plot = True,
#             plot_twin_counts = True,
#             twin_color=twin_color,
#             labels = labels,
#             fontsize_axes = fontsize_axes,
#             x_multiplier = 1000,
#         )

        
#         if y_label is None:
#             if synapse_type == "presyn":
#                 y_label = "Targets Cell Type"
#             elif synapse_type == "postsyn":
#                 y_label = "Source Cell Type"
#             else:
#                 y_label = f"Cumulative Percentage of \n{discrete_label.replace('_',' ').title()} Connections"
            
#             print(f"y_label = {y_label}")
        
#         ax.set_ylabel(y_label,color = twin_color,fontsize=fontsize_axes)
            
#         if x_label is None:
#             if histogram_attribute == "presyn_skeletal_distance_to_soma":
#                 x_label = "Skeletal Length of Source Axon (um)"
#             elif histogram_attribute == "presyn_skeletal_distance_to_soma":
#                 x_label = "Skeletal Length of Target Dendrite (um)"
#             else:
#                 x_label = f"{histogram_attribute.replace('_',' ').title()}"
#         ax.set_xlabel(x_label,fontsize=fontsize_axes)
#         ax2.set_xlabel(x_label,fontsize=fontsize_axes)
            
#         if title is None:
#             if synapse_type == "presyn":
#                 title_start = "Targets of"
#             else:
#                 title_start = "Sources of Input to"
#             title = f"{title_start} {name.replace('_','s:  ').capitalize()}"
            
#         if add_cell_counts_to_title:
#             title += f"{cell_count(restr_df,synapse_type=opposite_synapse_type)}"
            
#         ax.set_title(f"{title}",fontsize=fontsize_title)
#         return ax,ax2
#         plt.show()
        
        
def plot_histogram_discrete_labels(
    edge_df,
    restrictions_dicts= None,
    compartment_labels = None,# cell_type_fine_labels_global,#["apical"]
    cell_type_fine_labels = None,
    synapse_type = "postsyn",
    histogram_attribute = "presyn_skeletal_distance_to_soma",
    twin_color = "blue",
    normalize=True,
    cumulative=True,
    verbose = True,
    labels = None,
    y_label = None,
    x_label = None,
    title = None,
    add_cell_counts_to_title = True,
    
    #--- sizes of plots ---
    fontsize_title = None,
    figsize = (8,5),
    fontsize_axes = 16,
    fontsize_tick = 20,
    nbins = 100,
    legend = False,
    **kwargs
    ):
    
    
    if labels is None:
        labels = cona.cell_type_fine_labels_global

    if synapse_type == "postsyn":
        discrete_label = "presyn_gnn_cell_type_fine"
        opposite_synapse_type = "presyn"
    else:
        discrete_label = "postsyn_gnn_cell_type_fine"
        opposite_synapse_type = "postsyn"

    if restrictions_dicts is None:
    #for the discrete histogramming
        restrictions_dicts = []
        if cell_type_fine_labels is not None:
            restrictions_dicts = [dict(cell_type_fine=k) for k in cell_type_fine_labels]

        if compartment_labels is not None:
            if len(restrictions_dicts) == 0:
                restrictions_dicts = [dict(compartment=k) for k in compartment_labels]
            else:
                for d in restrictions_dicts:
                    for k in compartment_labels:
                        d["compartment"] = k
                        
        if len(restrictions_dicts) == 0:
            restrictions_dicts.append(None)

    print(f"restrictions_dicts = {restrictions_dicts}")

    for rd in restrictions_dicts:
        if rd is not None:
            restr_df,name = cona.restrict_edge_df_by_types_compartment(
                edge_df,
                verbose = False,
                synapse_type = synapse_type,
                return_name=True,
                **rd,
                **kwargs
            )
        else:
            restr_df = edge_df
            name = "No restrictions"
            

        if verbose:
            print(f"\n\n\n--- Working on {name}: # of edges = {len(restr_df)} ---")

        if len(restr_df) == 0:
            continue

        df_counts = pu.histogram_of_discrete_labels(
            restr_df,
            y=discrete_label,
            normalize=normalize,
            cumulative=cumulative,
            x = histogram_attribute,
            #x_steps = np.linspace(0,1000,100),
            plot = False,
            nbins=nbins,
            )


        #print(f"df_counts = {len(df_counts)}")

        ax,ax2 = mu.stacked_bar_graph(
            df_counts,
            color_dict = ctu.cell_type_fine_color_map,
            set_legend_outside_plot = True,
            plot_twin_counts = True,
            twin_color=twin_color,
            labels = labels,
            fontsize_axes = fontsize_axes,
            x_multiplier = 0.0001,
            legend = legend
        )


        if y_label is None:
            if synapse_type == "presyn":
                y_label = "Targets Cell Type"
            elif synapse_type == "postsyn":
                y_label = "Source Cell Type"
            else:
                y_label = f"Cumulative Percentage of \n{discrete_label.replace('_',' ').title()} Connections"
            
        ax.set_ylabel(
            y_label,
            #color = twin_color,
            fontsize=fontsize_axes)

        if x_label is None:
            if histogram_attribute == "presyn_skeletal_distance_to_soma":
                x_label = "Skeletal Length of Source Axon (um)"
            elif histogram_attribute == "presyn_skeletal_distance_to_soma":
                x_label = "Skeletal Length of Target Dendrite (um)"
            else:
                x_label = f"{histogram_attribute.replace('_',' ').title()}"
        ax.set_xlabel(x_label,fontsize=fontsize_axes)
        ax2.set_xlabel(x_label,fontsize=fontsize_axes)

        if title is None:
            if synapse_type == "presyn":
                title_start = "Targets of"
            else:
                title_start = "Sources of Input to"
            title = f"{title_start} {name.replace('_','s:  ').capitalize()}"

        if add_cell_counts_to_title:
            title += f"{conu.cell_count(restr_df,synapse_type=opposite_synapse_type)}"

        ax.set_title(f"{title}",fontsize=fontsize_title)

    return ax
        
        
def plot_cell_type_edge_stat(
    edge_df,
    cell_type_feature = "cell_type",
    presyn_cell_type_feature = None,
    postsyn_cell_type_feature = None,
    add_presyn_postsyn_to_name = True,
    verbose = True,
    stat_to_plot = "postsyn_skeletal_distance_to_soma",
    density = True,
    filter_away_0 = False,
    maximum_percentile = 98,
    alpha = 0.3,
    bins = 100,
    figsize = None,
    axes_height = 3,
    axes_width = 8,
    title_suffix = ""
    ):
    
    if presyn_cell_type_feature is None:
        presyn_cell_type_feature = cell_type_feature
        if add_presyn_postsyn_to_name:
            presyn_cell_type_feature = f"presyn_{presyn_cell_type_feature}"
    if postsyn_cell_type_feature is None:
        postsyn_cell_type_feature = cell_type_feature
        if add_presyn_postsyn_to_name:
            postsyn_cell_type_feature = f"postsyn_{postsyn_cell_type_feature}"
            
    print(f"postsyn_cell_type_feature = {postsyn_cell_type_feature}")
    print(f"presyn_cell_type_feature= {presyn_cell_type_feature}")
    edge_df["connection_type"] = (edge_df[presyn_cell_type_feature].astype('str') + "_onto_" + 
                                  edge_df[postsyn_cell_type_feature].astype('str'))


    unique_labels = np.sort(edge_df["connection_type"].unique())
    colors = mu.generate_non_randon_named_color_list(len(unique_labels))
    
    if figsize is None:
        figsize = (axes_width,axes_height*len(unique_labels))

    fig,axes = plt.subplots(
        len(unique_labels),
        1,
        figsize = figsize,
        sharex=True)

    for j,(ax,ct_conn) in enumerate(zip(axes,unique_labels)):
        #if j == 0:

        if verbose:
            print(f"Working on {ct_conn}")

        query_str = (f"(connection_type == '{ct_conn}')")
        df_rest = edge_df.query(query_str)
        data = df_rest[stat_to_plot].to_numpy()/1000

        if filter_away_0:
            data = data[data > 0]

        if maximum_percentile is not None:
            data = data[data < np.percentile(data,maximum_percentile)]

        ax.hist(data,
                label=ct_conn,
                density=density,
               alpha = alpha,
                bins = bins,)

        ax.set_title(f"{ct_conn} ({len(data)} datapoints) (mean = {np.mean(data):3f})")
        ax.legend()
        if density:
            ax.set_ylabel(f"Density")
        else:
            ax.set_ylabel(f"Count")

    fig.suptitle(f"{' '.join([k.title() for k in stat_to_plot.split('_')])} \nFor Different Cell Type Connections\n{title_suffix}" )
    ax.set_xlabel(stat_to_plot + " (um)")
    
    return fig,axes


#--- from neurd_packages ---
from . import cell_type_utils as ctu

#--- from python_tools ---
from python_tools import matplotlib_utils as mu
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import pandas_utils as pu

from . import connectome_analysis_utils as cona