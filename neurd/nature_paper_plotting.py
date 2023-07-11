
import matplotlib.pyplot as plt
import seaborn as sns

exc_inh_combination_palette = {
    'Exc Onto Exc':mu.seaborn_blue,
    'Exc Onto Inh':mu.seaborn_brown,
    'Inh Onto Exc':mu.seaborn_green,
    'Inh Onto Inh':mu.seaborn_red,
}

exc_inh_palette = {
    'excitatory':mu.colorblind_orange,
    'inhibitory':mu.colorblind_blue,
}


def plot_edit_labels_subset(
    edits_df,
    edit_labels,
    x = "x",
    y = "y",

    fontsize_axes = 40,
    fontsize_ticks = 25,
    bins = 100,

    color = mu.colorblind_blue,
    isotropic = False,
    ):
    
    xlabel = f"{x.title()} (um)"
    ylabel = f"{y.title()} (um)"


    curr_df = edits_df.query(f"edit_type in {list(edit_labels)}")



    ax = mu.plot_jointplot_from_df_coordinates_with_labels(
        curr_df,
        x = x,
        y = y,
        labels_column = None,
    #     color_dict=dict(
    #         axon = mu.colorblind_blue,
    #         dendrite = mu.colorblind_orange,
    #     ),
        color = color,
        bins = bins,
        xlabel = xlabel,
        ylabel = ylabel,
        fontsize_ticks=fontsize_ticks,
        fontsize_axes = fontsize_axes,
        no_tickmarks= False,
    )
    
    if isotropic:
        ax = mu.scale_axes_lim_isotropic(ax)
    
    return ax

def example_kde_plot(spine_df):
    figsize = (10,10)

    keep_classes_exc = ["5P-ET","5P-NP",]
    keep_classes_inh = ["MC"]
    column = "gnn_cell_type_fine"
    fontsize_title = 30
    fontsize_legend = 12

    spine_df_renamed =ctu.rename_cell_type_fine_column(
        spine_df,
        column = column,
        keep_classes_exc=keep_classes_exc,
        keep_classes_inh=keep_classes_inh,
    )

    spine_df_renamed["gnn_cell_type_fine"]

    features_to_plot = [
        "sp_spine_max_head_sp_vol",
        "sp_spine_max_head_syn_size",
    ]

    new_feature = "'sp_spine_max_head_syn_size_resized'"
    spine_df_renamed[new_feature] = (
        spine_df_renamed['sp_spine_max_head_syn_size']*hdju.syn_size_to_um_vol_scaling
    )
    #fig,ax = plt.subplots(1,1,figsize)
    #ax = None
    #for k in spine_df_renamed[column].unique():
    ax = sns.kdeplot(
        data = spine_df_renamed,#.query(f"{column}=='{k}'"),
        x = features_to_plot[0],
        y = new_feature,#features_to_plot[1],
        hue = column,
        common_norm = False,
        thresh=0.2,
        levels = 4,
        palette=ctu.cell_type_fine_color_map,
        #ax = ax,

    )

    fontsize = 20

    ax.set_xlim([0.025,0.16])
    ax.set_ylim([0,20_000*hdju.syn_size_to_um_vol_scaling])
    ax.set_ylabel(
        "Max Spine Head Syn\n Cleft Volume ($\mu m^3$)",
        fontsize = fontsize
    )
    ax.set_xlabel(
        "Spine Head Volume ($\mu m^3$)",
        fontsize = fontsize
    )
    ax.set_title("Max Spine Syn Size vs.\nSpine Head Volume",fontsize = fontsize_title)

    mu.set_legend_title(ax,"Cell Type")
    mu.set_legend_fontsizes(ax,fontsize_legend)
    #ax.get_legend()._loc='upper left'
    #plt.legend(loc='upper left', )
    mu.move_legend_location_seaborn(ax, "upper left",)
    

def example_histogram_nice(spine_df):
    

    spine_df_ct_rename = ctu.rename_cell_type_fine_column(
        spine_df,
        keep_classes_exc=["4P","5P-NP",],
        keep_classes_inh = ["MC"],
    )

    spine_df_ct_rename["gnn_cell_type_fine"].hist()

    column = "sp_neck_skeletal_length"
    hue = "gnn_cell_type_fine"
    fontsize_axes = 25
    fontsize_title = 25
    bins = 35

    fig,ax = plt.subplots(
        1,1,
        #figsize = (20,12)
    )

    data = spine_df_ct_rename.copy().query(f"gnn_cell_type_coarse == 'excitatory'")
    data = pu.filter_df_by_column_percentile(
        df = data,
        columns=column,
        percentile_lower = 0,
        percentile_upper = 99.9,
    )

    data[column] = data[column]/1000

    palette=ctu.cell_type_fine_color_map
    ax = sns.histplot(
            data = data,
            x = column,
            hue = hue,
            fill=True,
            #bins = "auto",
            palette=palette,
            bins = bins,
            alpha = 0.5,
            common_norm = False,
            stat = "density",
            common_bins = True,
            legend = True,
           linewidth = 0.05,
    )

    ax.legend_.set_title(None)

    ax.set_xlabel(f"Spine Neck Length ($\mu m$)",fontsize = fontsize_axes)
    ax.set_title("Spine Neck Length ",fontsize = fontsize_title)
    ax.set_ylabel("Density",fontsize = fontsize_axes)
    mu.set_axes_tick_font_size(ax,20)
    mu.set_legend_fontsizes(ax,fontsize = 15)
    

#--- from neurd_packages ---
from . import cell_type_utils as ctu

#--- from python_tools ---
from python_tools import matplotlib_utils as mu
from python_tools import pandas_utils as pu

from . import nature_paper_plotting as npp