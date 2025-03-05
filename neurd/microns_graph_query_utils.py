'''



To help query the graph object and do visualizations



'''


default_node_df_path = "/platinum_graph/Data/G_query_v6_filtered_node_df.csv"
default_edge_df_path = "/platinum_graph/Data/G_query_v6_filtered_edge_and_node_df.gzip"
default_graph_path = "/platinum_graph/Data/G_query_v6_filtered.pbz2"

def load_node_df(filepath = None):
    if filepath is None:
        filepath = default_node_df_path
        
    return su.decompress_pickle(filepath)

def load_edge_df(filepath = None):
    if filepath is None:
        filepath = default_edge_df_path
        
    return pu.gzip_to_df(filepath)

# ----------------- Node querying ------------------------- #
def soma_centers_from_node_df(node_df):
    return node_df[["soma_x_nm","soma_y_nm","soma_z_nm"]].to_numpy()

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
    mqu.soma_centers_from_node_query(
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

    sub_df_centers = mqu.soma_centers_from_node_df(sub_df)
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

        
    return mqu.node_df_from_query(query = query,
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
    
    return mqu.node_df_from_attribute_value(
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
    
    return mqu.node_df_from_attribute_value(
    attribute_type="cell_type",
    attribute_value = "inhibitory",
    G=G,
    node_df = node_df,
    **kwargs
    )

def n_excitatory_n_inhibitory_nodes(G=None,
                                   node_df = None,
                                   verbose = False):
    n_excitatory = len(mqu.excitatory_cells_node_df(G=G,node_df=node_df))
    n_inhibitory = len(mqu.inhibitory_cells_node_df(G=G,node_df=node_df))
    
    if verbose:
        print(f"n_excitatory = {n_excitatory},n_inhibitory = {n_inhibitory} ")
    return n_excitatory,n_inhibitory


#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su

from . import microns_graph_query_utils as mqu