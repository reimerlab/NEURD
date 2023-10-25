'''



To help query the graph object and do visualizations



'''

from datasci_tools import module_utils as modu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu

# default_node_df_path = "/platinum_graph/Data/G_query_v6_filtered_node_df.csv"
# default_edge_df_path = "/platinum_graph/Data/G_query_v6_filtered_edge_and_node_df.gzip"
# default_graph_path = "/platinum_graph/Data/G_query_v6_filtered.pbz2"

# def load_node_df(filepath = None):
#     if filepath is None:
#         filepath = default_node_df_path
        
#     return su.decompress_pickle(filepath)

# def load_edge_df(filepath = None):
#     if filepath is None:
#         filepath = default_edge_df_path
        
#     return pu.gzip_to_df(filepath)

# ----------------- Node querying ------------------------- #
def soma_centers_from_node_df(node_df):
    return node_df[["centroid_x_nm","centroid_y_nm","centroid_z_nm"]].to_numpy()

def soma_centers_from_node_query(query,
                                G=None,
                                node_df = None,
                                verbose = False,
                                return_query_df=False):
    """
    Purpose: To query the nodes of the graph and return the soma centers

    Pseudocode: 
    1) apply query to the node df
    2) export the soma centers of the query
    3) return the queried table if requested
    
    Ex: 
    conq.soma_centers_from_node_query(
    query = "cell_type == 'inhibitory'",
    #G = G,
    node_df = node_df,
    verbose = True,
    return_query_df = False,
)
    """

    if node_df is None:
        node_df = xu.node_df(G)

    sub_df = node_df.query(query)

    sub_df_centers = conq.soma_centers_from_node_df(sub_df)
    if verbose:
        print(f"# of cells in query = {len(sub_df_centers)}")

    if return_query_df:
        return sub_df_centers,sub_df
    else:
        return sub_df_centers
    
def node_df_from_attribute_value(
    attribute_type=None,
    attribute_value = None,
    query = None,
    G=None,
    node_df = None,
    **kwargs
    ):
    
    if query is None:
        if type(attribute_value) == str:
            query = f"{attribute_type} == '{attribute_value}'"
        else:
            query = f"{attribute_type} == {attribute_value}"

        
    return conq.node_df_from_query(query = query,
                                 G=G,
                                 node_df = node_df,
                                 **kwargs)
    
def node_df_from_query(
    query,
    G=None,
    node_df = None,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: Will return the number of 
    """
    if node_df is None:
        node_df = xu.node_df(G)
        
    curr_df = node_df.query(query)
    
    if verbose:
        print(f"# of nodes: {len(curr_df)}")
    return curr_df

def excitatory_cells_node_df(
    G=None,
    node_df = None,
    **kwargs
    ):
    
    return conq.node_df_from_attribute_value(
    attribute_type="cell_type",
    attribute_value = "excitatory",
    G=G,
    node_df = node_df,
    **kwargs
    )

def inhibitory_cells_node_df(
    G=None,
    node_df = None,
    **kwargs
    ):
    
    return conq.node_df_from_attribute_value(
    attribute_type="cell_type",
    attribute_value = "inhibitory",
    G=G,
    node_df = node_df,
    **kwargs
    )

def n_excitatory_n_inhibitory_nodes(G=None,
                                   node_df = None,
                                   verbose = False):
    n_excitatory = len(conq.excitatory_cells_node_df(G=G,node_df=node_df))
    n_inhibitory = len(conq.inhibitory_cells_node_df(G=G,node_df=node_df))
    
    if verbose:
        print(f"n_excitatory = {n_excitatory},n_inhibitory = {n_inhibitory} ")
    return n_excitatory,n_inhibitory




# ----------------- Helper functions for 3D analysis ------------- #

# -- default
attributes_dict_default = dict(
    #voxel_to_nm_scaling = microns_volume_utils.voxel_to_nm_scaling,
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
    #voxel_to_nm_scaling = h01_volume_utils.voxel_to_nm_scaling,
    vdi = hvu.data_interface
)
global_parameters_dict_h01 = dict()
    
       
# data_type = "default"
# algorithms = None
# modules_to_set = [conq]

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
from datasci_tools import module_utils as modu 
from datasci_tools import networkx_utils as xu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su

from . import connectome_query_utils as conq