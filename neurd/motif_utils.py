'''


To help analyze the motifs found using the dotmotif package 
from a connectome dataset


'''
import networkx as nx
import pandas as pd
import re
import time


motif_key = "motif"
identifier_name_global = "identifier"
edge_pattern = r"([A-Z])[ ]*->[ ]*([A-Z])"

min_gnn_probability_global = 0.6
cell_type_fine_exclude_global = ['SST','NGC',]

def edges_from_str(
    string,
    verbose = False,
    return_edge_str = False
    ):
    pattern = re.compile(edge_pattern)
    s_find = [k for k in pattern.finditer(string)]
    if return_edge_str:
        from python_tools import regex_utils as reu
        edges = [reu.substr_from_match_obj(k) for k in s_find]
    else:
        edges = [(g.groups()[0],
              g.groups()[1])
            for g in s_find]
    if verbose:
        pritn(f"edges = {edges}")
        
    return edges

def nodes_from_str(string):
    edges_str = mfu.edges_from_str(string,return_edge_str = False)
    return list(np.unique(np.hstack(edges_str)))

def motif_nodes_from_motif(
    motif,
    only_upper = True,
    verbose = False,
    return_n_nodes = False,
    ):
    """
    Purpose: Determine the number of nodes (and what their names are )
    from a motif string

    Pseudocode: 
    1) Look for all upper case letters where there is other
    words before or after
    2) Order the pairs found
    3) Can return the length of the dictionary or just the number
    """
    s = motif
    if only_upper:
        pattern = re.compile(r"\W*([A-Z])\W*")
    else:
        pattern = re.compile(r"\W*([A-Za-z])\W*")

    s_find = pattern.finditer(s)
    s_find = [k.groups()[0] for k in s_find]
    found_letters = np.sort(np.unique(s_find))

    if verbose:
        print(f"found_letters = {found_letters}")

    if return_n_nodes:
        return len(found_letters)
    else:
        return {i:k for i,k in enumerate(found_letters)}
    
def n_nodes_from_motif(
    motif,
    only_upper = True,
    verbose = False,
    ):
    
    return mfu.motif_nodes_from_motif(
    motif,
    only_upper = only_upper,
    verbose = verbose,
    return_n_nodes = True,
    )

def nodes_mapping_from_G(
    G,
    ):
    """
    Purpose: Get the node mapping
    """
    identifiers = []
    names = []
    for n in G.nodes():
        names.append(n)
        identifiers.append(G.nodes[n][identifier_name_global])
        
    arg_idx = np.argsort(identifiers)
    mapping = dict([(identifiers[k],
                    names[k]) for k in arg_idx])
    
    return mapping
    
    
def nodes_from_motif_dict(
    motif_dict,
    return_dict = True,
    verbose = False):
    """
    Purpose: To extract the node names
    from the motif dict
    
    Pseudocode: 
    1) get all of the keys with segment id in them
    2) sort them
    3) iterate and get the segment id and split index and put into dict
    """
    seg_names = [k for k in motif_dict if k[2:] == "segment_id"]
    r_dict = dict()
    if len(seg_names) > 0:
#     if verbose:
#         print(f"segment_id names = {seg_names}")
        
        for s_id in seg_names:
            letter = s_id[0]
            node_name = (f"{motif_dict[f'{letter}_segment_id']}_"
                         f"{motif_dict[f'{letter}_split_index']}"
                        )
            r_dict[letter.upper()] = node_name
    else:
        seg_names = [k for k in motif_dict if k[2:] == "name"]
        for s_id in seg_names:
            letter = s_id[0]
            node_name = motif_dict[s_id]
            r_dict[letter.upper()] = node_name
        
    if verbose:
        print(f"Node names = {r_dict}")
        
    if return_dict:
        return r_dict
    else:
        return list(r_dict.values())
    
def edges_from_motif_dict(
    motif_dict,
    return_dict = False,
    return_node_mapping = False,
    verbose = True,
    ):
    """
    Purpose: To get a list of the edges represented by the motif

    Pseudocode: 
    1) Get a mapping of the nodes
    2) Query the dotmotif for the edge definitions
    3) For each of the groups found substitute in the node name

    Ex: 
    from python_tools import networkx_utils as xu
    import networkx as nx

    G = hdju.G_auto_DiGraph
    motif_info = motif_dicts[20000]

    edges = mfu.edges_from_motif_dict(
        motif_info,
        return_dict=False,
        verbose = True,)

    sub_G = xu.subgraph_from_edges(G,edges)
    nx.draw(sub_G,with_labels = True)
    
    motif_nodes_from_motif
    """

    node_mapping = mfu.nodes_from_motif_dict(motif_dict)
    s = motif_dict[motif_key]

    if verbose:
        print(f"node_mapping = {node_mapping}")

    edges_identifiers = mfu.edges_from_str(s)
    if verbose:
        print(f"# of edges found = {len(edges_identifiers)}")
    
    edges = [(node_mapping[g[0]],
              node_mapping[g[1]])
            for g in edges_identifiers]
    
    
    if verbose:
        print(f"Edges = {edges}")

    if return_dict:
        from python_tools import regex_utils as reu
        s_find = mfu.edges_from_str(s,return_edge_str=True)
        
#         return_dict = {reu.substr_from_match_obj(k):v for k,v in
#                        zip(s_find,edges)}

        return_dict = {k:v for k,v in
                       zip(s_find,edges)}
        
        return_value= return_dict
    else:
        return_value= edges
        
    if return_node_mapping:
        return return_value,node_mapping
    else:
        return return_value
    
    

def subgraph_from_motif_dict(
    G,
    motif_dict,
    verbose = False,
    identifier_name = None,
    plot = False,
    ):
    
    if identifier_name is None:
        identifier_name = identifier_name_global

    if verbose:
        print(f"motif_dict = {motif_dict}")
    edges,node_mapping = mfu.edges_from_motif_dict(
        motif_dict,
        return_dict=False,
        verbose = verbose,
        return_node_mapping=True)
    
    

    sub_G = xu.subgraph_from_edges(G,edges)
    
    for ident,node_name in node_mapping.items():
        sub_G.nodes[node_name]["identifier"] = ident
    
    if plot:
        nx.draw(sub_G,with_labels = True)
        
    return sub_G


def motif_segment_df_from_motifs(
    motifs,
    return_df = True,
    motif = None,
    graph_type = "DiGraph"
    ):
    """
    Purpose: Turn the motif results
    (where motif results are in the form of
    a dictionary A:"segment_split",B:"segment_splits")
    into a dataframe or dictionaries
    
    and returns dictionary or dataframe we
    have keys like a_segment_id,a_split_index,b_segment_id....
    """

    keys_to_write = []
    for j,m in enumerate(motifs):
        local_keys = {}
        for k,v in m.items():

            segment_id,split_index = hdju.segment_id_and_split_index(v)
            local_keys.update({f"{k.lower()}_segment_id":segment_id})
            local_keys.update({f"{k.lower()}_split_index":split_index})
        
        if motif is not None:
            local_keys["motif"] = motif
            
        if graph_type is not None:
            local_keys["graph_type"] = graph_type
        
        keys_to_write.append(local_keys)


    if return_df:
        return pd.DataFrame.from_records(keys_to_write)
    return keys_to_write


def motif_data(
    G,
    motif_dict,
    
    # for edge attributes
    cell_type_kind = "gnn_cell_type_fine",
    include_layer = True,
    include_visual_area = True,
    include_node_identifier = True,

    # for edge attrbutes
    include_edges_in_name = True,
    include_compartment = True,
    edge_attributes = ("presyn_soma_postsyn_soma_euclid_dist",
                       "presyn_soma_postsyn_soma_skeletal_dist",
                       "presyn_skeletal_distance_to_soma",
                       "presyn_soma_euclid_dist",
                       "postsyn_skeletal_distance_to_soma",
                       "postsyn_soma_euclid_dist",
                        "synapse_id",
                      ),
    node_attributes = ("skeletal_length",
                       "external_manual_proofread",
                      "gnn_cell_type_fine_prob",
                      "gnn_cell_type"),
    
    node_attributes_additional = None,
                       

    return_str = False,
    
    verbose = True,
    ):
    """
    Purpose: Convert a graph into a string representation
    to be indexed (used as an identifier)

    2 possible representations: 
    1) list all cell types, then all downstream compartmnets
    2) List presyn_cell_type, downstream cell type, compartment

    Pseudocode: 
    1) Get node mapping and presyns associated
    2) Get all of the edges in the graph
    3) Construct a list of a identifier, identifier_2, compartment
    4) Make name cell type(id), cell type 2 (id2)....: id1id2(comp)....
    """
    
    #print(f"node_attributes before = {node_attributes}")
    if node_attributes_additional is not None:
        node_attributes = list(np.union1d(node_attributes,node_attributes_additional))
        
    #print(f"node_attributes after = {node_attributes}")
    
    st = time.time()
    motif_data_dict = dict()
    edges,node_mapping = mfu.edges_from_motif_dict(
        motif_dict,
        verbose = verbose,
        return_dict = True,
        return_node_mapping = True)

    #ct_mapping = dict()
    cell_type_str = ""

    for j,(ident,node) in enumerate(node_mapping.items()):
        node_dict = xu.get_node_attribute_dict(G,node)
        ct = node_dict[cell_type_kind]
        cell_type_str += f"{ct}"
        
        motif_data_dict[f"{ident}_name"] = node
        motif_data_dict[f"{ident}_gnn_cell_type_fine"] = node_dict["gnn_cell_type_fine"]
        motif_data_dict[f"{ident}_cell_type"] = node_dict["cell_type"]
        motif_data_dict[f"{ident}_layer"] = node_dict['external_layer']
        motif_data_dict[f"{ident}_area"] = node_dict['external_visual_area']
        
        for k in node_attributes:
            motif_data_dict[f"{ident}_{k}"] = node_dict[k]
        
        if include_layer:
            cell_type_str += f"/{node_dict['external_layer']}"
        if include_visual_area:
            cell_type_str += f"/{node_dict['external_visual_area']}"
        if include_node_identifier:
            cell_type_str += f"({ident})"

        if j != len(node_mapping)-1:
            cell_type_str += ", "

    if verbose:
        print(f"cell_type_str = {cell_type_str}")

    if include_edges_in_name:
        cell_type_str += " : "
    for j,(edge_str,node_pair) in enumerate(edges.items()):
        if include_edges_in_name:
            cell_type_str += f"{edge_str}"

        edge_dict = G[node_pair[0]][node_pair[1]]
        compartment = edge_dict["postsyn_compartment_fine"]
        if not type(compartment) == str:
            compartment = edge_dict["postsyn_compartment_coarse"]

        motif_data_dict[f"{edge_str}_postsyn_compartment"] = compartment
        
        if edge_attributes is not None:
            for ea in edge_attributes:
                motif_data_dict[f"{edge_str}_{ea}"] = edge_dict[ea]
        
        if include_edges_in_name and include_compartment:
            cell_type_str += f"({compartment})"

        if j != len(edges) - 1:
            cell_type_str += f", "

    if verbose:
        print(f"cell_type_str (AFTER EDGES) = {cell_type_str}")
                      
    if verbose:
        print(f"Total time = {time.time() - st}")
        
        
    motif_data_dict["motif_str"] = cell_type_str
    
    #copying old attributes over:
    for k in ["motif","graph_type"]:
        motif_data_dict[k] = motif_dict[k]
    
    if return_str:
        return cell_type_str
    else:
        return motif_data_dict
    
def filter_G_attributes(
    G,
    node_attributes = (
        "gnn_cell_type_fine",
        "cell_type",
        "external_layer",
        "external_visual_area",
        "manual_cell_type_fine",
        "identifier",
    ),
    edge_attributes = (
        "postsyn_compartment_coarse",
        "postsyn_compartment_fine",
        "presyn_skeletal_distance_to_soma",
        "postsyn_skeletal_distance_to_soma",
    ),
    ):
    
    # filter the node attributes
    sub_G = xu.filter_down_node_attributes(
        G,
        attributes = node_attributes,
        )
    
    # filter the edge attributes
    sub_G = xu.filter_down_edge_attributes(
        sub_G,
        attributes = edge_attributes,
    )
    
    return sub_G
    
    
def motif_G(
    G,
    motif_dict,
    plot = False,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To form a graph data structure
    representing the motif
    
    Pseudocode: 
    1) Restrict the graph to a subgraph based on the motif
    2) Filter the node attributes and edge attributes to only
    those specified
    
    Ex: 
    curr_G = motif_G(
    G,
    motif_info,
    plot = True)
    
    """
    
    sub_G = mfu.subgraph_from_motif_dict(
        G,motif_dict,
        verbose=verbose,
        plot=plot)
    
    sub_G = mfu.filter_G_attributes(
        sub_G,
        **kwargs
    )
    
    if verbose:
        print(f"Setting graph attributes")
    xu.set_graph_attr(
        sub_G,
        "motif",
        motif_dict["motif"],
    )
    
    try:
        mfu.set_compartment_flat(sub_G)
    except:
        pass
    
    return sub_G

def node_attributes_from_G(
    G,
    features = None,
    features_to_ignore = (
        xu.upstream_name,
        identifier_name_global
    ),
    features_order = (
     "gnn_cell_type_fine",
     "external_layer",
     "external_visual_area",
    ),
    ):
    
    if features_order is None:
        features_order = []
    
    node_df = xu.node_df(G)
    features_list = list(node_df.columns)
    if (features_to_ignore is not None 
        and len(features_to_ignore) > 0):
        features_list = np.setdiff1d(features_list,
                               features_to_ignore)
        
    priority_features = []
    for f in features_order:
        if f in features_list:
            priority_features.append(f)
            
    non_priority_features = list(np.setdiff1d(
        features_list,
        priority_features))
    
    final_features = priority_features + non_priority_features
    
    if features is not None:
        final_features = [k for k in final_features if k in features]
        
    return final_features


def set_compartment_flat(G):
    
    def comp_flat(key):
        if not type(key["postsyn_compartment_fine"]) == str:
            return key["postsyn_compartment_coarse"]
        else:
            return key["postsyn_compartment_fine"]

    xu.derived_edge_attribute_from_func(
        G,
        "postsyn_compartment",
        comp_flat 
        )
    
    return G

# ----------- conversions to str --------------
def str_from_G_motif(
    G,
    node_attributes = None,
    edge_attributes = ("postsyn_compartment_flat",),
    verbose = False,
    joining_str = "/",
    include_edges_in_name = True
    ):

    """
    Purpose: To convert a graph to a string representation
    to be used as an identifier

    Pseudocode: 
    1) Gather the node attributes for each of the 
    nodes (order by identifier and order the attributes)

    2) Gather the edge attributes
    
    Ex: 
    mfu.set_compartment_flat(curr_G)
    mfu.str_from_G_motif(
        curr_G,
        node_attributes = ("gnn_cell_type_fine",),
        edge_attributes=["postsyn_compartment_flat",],
        )
    """

    node_mapping = mfu.nodes_mapping_from_G(G)

    if node_attributes is None:
        node_attributes = mfu.node_attributes_from_G(G)
        if verbose:
            print(f"node_attributes = {node_attributes}")

    G_str = ""
    for j,(ident,name) in enumerate(node_mapping.items()):
        attributes = [str(G.nodes[name][f]) for f in node_attributes]
        attr_str = joining_str.join(attributes)
        G_str += f"{attr_str} ({ident})"

        if j != len(node_mapping) - 1:
            G_str += ", "

    """
    want to add the edge attributes to the motif

    1) 

    pseudocode: 
    1) 
    """
    if include_edges_in_name:
        edges_str = " : "
        
        if edge_attributes is None:
            edge_attributes = []

        curr_motif = xu.get_graph_attr(G,"motif")
        curr_edges = mfu.edges_from_str(curr_motif)

        for j,(id1,id2) in enumerate(curr_edges):
            edges_str += f"{id1}->{id2}"
            if len(edge_attributes) > 0:
                edges_str += f"({', '.join([str(G[node_mapping[id1]][node_mapping[id2]][f]) for f in edge_attributes])})"
            if j != len(curr_edges) - 1:
                edges_str += ", "

        if verbose:
            print(f"edges_str = {edges_str}")

        G_str += edges_str

    if verbose:
        print(f"G_str = {G_str}")
        
    return G_str


    
def dotmotif_str_from_G_motif(
    G,
    node_attributes = None,
    edge_attributes = ("postsyn_compartment",),
    verbose = False,
    ):

    """
    Purpose: To convert a graph to a string representation
    to be used as an identifier

    Pseudocode: 
    1) Gather the node attributes for each of the 
    nodes (order by identifier and order the attributes)

    2) Gather the edge attributes
    
    Ex: 
    mfu.set_compartment_flat(curr_G)
    mfu.str_from_G_motif(
        curr_G,
        node_attributes = ("gnn_cell_type_fine",),
        edge_attributes=["postsyn_compartment_flat",],
        )
        
    Ex: 
    dotmotif_str_from_G_motif(
    curr_G,
    node_attributes = ("gnn_cell_type_fine",))
    """

    G_str = ""
    
    node_mapping = mfu.nodes_mapping_from_G(G)
    
    if edge_attributes is None:
        edge_attributes = []

    curr_motif = xu.get_graph_attr(G,"motif")
    curr_edges = mfu.edges_from_str(curr_motif)

    edges_str = ""
    for j,(id1,id2) in enumerate(curr_edges):
        edges_str += f"{id1}->{id2}"
        if len(edge_attributes) > 0:
            edges_str_curr = ', '.join([f"{f} = {G[node_mapping[id1]][node_mapping[id2]][f]}" for f in edge_attributes])
            edges_str += f"[{edges_str_curr}]"
        if j != len(curr_edges) - 1:
            edges_str += ";\n"

    if verbose:
        print(f"edges_str = {edges_str}")

    G_str += edges_str
    

    if node_attributes is None:
        node_attributes = mfu.node_attributes_from_G(G)
        if verbose:
            print(f"node_attributes for default = {node_attributes}")

    G_str += "\n"
    for j,(ident,name) in enumerate(node_mapping.items()):
        for f in node_attributes:
            try:
                att_value = str(G.nodes[name][f])
            except:
                continue
            G_str += f"{ident}.{f} = {att_value}\n"
            
    if verbose:
        print(f"\n---Final Dotmotif str:--- \n{G_str}")
        
    return G_str



def node_attributes_strs(
    G,
    joining_str = "/",
    node_attributes= None,
    verbose = False,
    
    ):
    """
    Purpose: To get a list of strings
    representing the node attributes 
    (that could then be used as a set for comparisons)

    Pseudocode: 
    1) Get the node attributes you want to output
    """
    st = time.time()


    if node_attributes is None:
        node_attributes = mfu.node_attributes_from_G(G)
        if verbose:
            print(f"node_attributes = {node_attributes}")

    total_nodes = []
    for j,name in enumerate(G.nodes()):
        attributes = []
        for f in node_attributes:
            try:
                curr_attr = str(G.nodes[name][f])
                attributes.append(curr_attr)
            except:
                continue
        curr_str = joining_str.join(attributes)
        if verbose:
            print(f"Node {j}: {curr_str}")
        total_nodes.append(curr_str)
        
    if verbose:
        print(f"Total time for node attributes strs: {time.time() - st}")

    return total_nodes

#edges_str = xu.get_graph_attr(curr_G,"motif")



def motif_column_mapping(
    df,
    mapping):
    
    """
    Purpose: Want to rename certain columns
    to different characters so everything matches

    Columns want to rename are very constrained: 
    option 1:
    [name]_....
    [name]->[name]....


    Pseudocode: 

    """
    
    column_mapping = dict()
    for k in df.columns:
        if "_" == k[1] and k[0] in list(mapping.keys()):
            column_mapping[k] = f"{mapping[k[0]]}{k[1:]}"
        elif (("->" == k[1:3])
              and (k[0] in list(mapping.keys()))
              and (k[3] in list(mapping.keys()))
                  ):
            column_mapping[k] = f"{mapping[k[0]]}->{mapping[k[3]]}{k[4:]}"
        else:
            pass
    
    return column_mapping 

def unique_motif_reduction(
    G,
    df,
    column = "motif_str",
    node_attributes = None,
    edge_attributes=None,
    #new_column = None,
    verbose = False,
    debug_time = False,
    relabel_columns = True,
    ):
    """

    Pseudocode: 
    1) Create a dictionary mapping the non-redundant str to dotmotif
    2) Find all unique str options
    3) For each str option: 
    a. Find one occurance of str
    b. conert it to a graph object

    c. Iterate through all non-reundance keys and do dot motif search
        i) if not found --> continue down list
        ii) if found (make this the non-redundant name and add to dict)
    4) Use non redundant dict to create new columns
    5) Find the count of all non-redundanct and sort from greatest to least
    6) plot the first x number of motifs

    """
    #if new_column is None:
    new_column = f"{column}_unique"

    motif_cell_type_df = df

    
    #motif_str_unique = motif_cell_type_df["motif_str"].unique()
    first_inst_df = pu.filter_to_first_instance_of_unique_column(
        motif_cell_type_df,
        "motif_str"
    ).reset_index()

    if verbose:
        print(f"# of unique motif str (including redundancy) = {len(first_inst_df)}")
        global_time = time.time()

    unique_map = dict()
    unique_dotmotif_map = dict()

    for j,m in tqdm(enumerate(pu.df_to_dicts(first_inst_df))):
        #b. convert it to a graph object
        curr_motif_str = m["motif_str"]

    #     if verbose:
    #         print(f"\n\n--- Working on {j} motif: {curr_motif_str}-----")

        if debug_time:
            st = time.time()
        curr_motif_G = mfu.motif_G(
            G,
            m,
            plot = False,
            )
        
        nodes_mapping = mfu.nodes_mapping_from_G(curr_motif_G)
        reverse_mapping = {v:k for k,v in mfu.nodes_mapping_from_G(curr_motif_G).items()}

        curr_node_strs = set(mfu.node_attributes_strs(
                curr_motif_G,
                verbose = False,
                node_attributes = node_attributes,
        ))

        if debug_time:
            print(f"motif_G generation: {time.time() - st}")
            st = time.time()

        found = False
        for k,data_dict in unique_dotmotif_map.items():
            node_strs = data_dict["node_strs"]

            if node_strs != curr_node_strs:
                continue

            dotmotif_str = data_dict["dotmotif"]
            matches = dmu.graph_matches(
                curr_motif_G,
                dotmotif_str,
                convert_characters = True)

            if debug_time:
                print(f"n_graph_matches: {time.time() - st}")
                st = time.time()

            if len(matches) >= 1:
                if verbose:
                    print(f"{curr_motif_str}\n    matched to \n{k}")
                found = True
                unique_map[curr_motif_str] = dict(
                    match = k,
                    mapping = {reverse_mapping[v]:k for k,v in matches[0].items()})
                
                break


        # c) If no match was found in the unique str
        if not found:
            dotmotif_str = mfu.dotmotif_str_from_G_motif(
                curr_motif_G,
                edge_attributes=edge_attributes,
                node_attributes=node_attributes,

            )


            if debug_time:
                print(f"dotmotif_str_from_G_motif: {time.time() - st}")
                st = time.time()

    #         if verbose:
    #             print(f"Adding {curr_motif_str} to non redundant list")

            unique_dotmotif_map[curr_motif_str] = dict(
                dotmotif=dotmotif_str,
                node_strs = curr_node_strs)
            unique_map[curr_motif_str] = dict(
                match = curr_motif_str,
                mapping = {k:k for k in nodes_mapping})
                

            
    if not relabel_columns:
        motif_cell_type_df[new_column] = pu.new_column_from_dict_mapping(
            motif_cell_type_df,
            {k:v["match"] for k,v in unique_map.items()},
            column_name=column
        )
    else:
        motif_cell_type_df_list = []
        for k,v_data in unique_map.items():
            
            curr_df = motif_cell_type_df.query(f"{column} == '{k}'")
            
            #raise Exception("")
            motif_map = motif_column_mapping(curr_df,mapping = v_data["mapping"])
            curr_df = pu.rename_columns(curr_df,motif_map)
            
            curr_df[new_column] = v_data["match"]
            motif_cell_type_df_list.append(curr_df)
            
            if len(curr_df) > 1:
                pass
                #break
                #raise Exception("")
        
        motif_cell_type_df = pu.concat(motif_cell_type_df_list)
    
    if verbose:
        print(f"Total time for reduction = {time.time() - global_time}")
        
    motif_cell_type_df = pu.delete_columns(motif_cell_type_df,column)
    motif_cell_type_df = pu.rename_columns(motif_cell_type_df,{new_column:column})

    return motif_cell_type_df



def motif_dicts_from_motif_from_database(
    motif,
    ):

    motif_table = hdju.motif_table_from_motif(motif)
    motif_table_df = hdju.df_from_table(motif_table)
    motif_dicts = pu.df_to_dicts(motif_table_df)
    
    return motif_dicts

def annotated_motif_df(
    G,
    motif,
    node_attributes = (
        "external_layer",
        "external_visual_area",
        "gnn_cell_type_fine",
        "gnn_cell_type_fine_prob",
        "gnn_cell_type",
        "skeletal_length"
    ),
    
    edge_attributes = (
        "postsyn_compartment",
    ),
    n_samples = None,
    verbose = False,
    filter_df = True,
    motif_reduction  = True,
    add_counts = True,
    motif_dicts= None,
    matches = None,
    additional_node_attributes = None,
    ):
    """
    Purpose: To add all of the features to the motifs
    
    Ex: 
    from neurd_packages import motif_utils as mfu

    G = hdju.G_auto_DiGraph

    mfu.annotated_motif_df(
        motif = "A->B;B->A",
        G = hdju.G_auto_DiGraph,
        n_samples = None,
        verbose = False
    )
    """
    if additional_node_attributes is not None:
        node_attributes = list(node_attributes) + nu.array_like(additional_node_attributes)
        
    
    global_time = time.time()
    
    if matches is not None:
        motif_dicts = motif_segment_df_from_motifs(
            matches,
            motif=motif,
            return_df=False
        )
        
    if motif_dicts is None:
        motif_dicts = motif_dicts_from_motif_from_database(motif)
#         motif_table = hdju.motif_table_from_motif(motif)
#         motif_table_df = hdju.df_from_table(motif_table)
#         motif_dicts = pu.df_to_dicts(motif_table_df)
        

    
    # --- creating the data dicts ----
    idx = np.arange(len(motif_dicts))
    np.random.seed(1000)
    np.random.shuffle(idx)
    
    column = "motif_str"
    
    if n_samples is None:
        n = n_samples
    else:
        n = np.min([n_samples,len(motif_dicts)])

    cell_type_list = []
    for i in tqdm(idx[:n]):
        cell_type_list.append(mfu.motif_data(
            G,
            motif_dict=motif_dicts[i],

            # for edge attributes
            include_layer = "external_layer" in node_attributes,
            include_visual_area = "external_visual_area" in node_attributes,
            include_node_identifier = True,

            # for edge attrbutes
            include_edges_in_name = True,
            include_compartment = "postsyn_compartment" in edge_attributes,

            return_str = False,
            verbose = verbose,
            node_attributes_additional = node_attributes,
            )

    )
        
    motif_cell_type_df = pd.DataFrame.from_records(cell_type_list)
    
    if motif_reduction:
        unique_df = mfu.unique_motif_reduction(
            G,
            motif_cell_type_df,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            column = column,
            verbose = verbose,
            debug_time = False,
            relabel_columns = True
            )
    else:
        unique_df = motif_cell_type_df
        
        
    sorting_columns = [column]
    if add_counts:
        count_column_name = "n_motifs"
        unique_df = pu.unique_row_counts(
            df = unique_df,
            columns = column,
            count_column_name = count_column_name,
            add_to_df = True,
            verbose = False,
        )

        sorting_columns = [count_column_name] + sorting_columns

    unique_df = pu.sort_df_by_column(
        unique_df,
        sorting_columns,
        ascending = False,
    )
    
    if verbose:
        print(f"Total time for annotated df = {time.time() - global_time}")
        
    if filter_df:
        unique_df = mfu.filter_motif_df(
            unique_df,
            verbose = verbose
        )

    return unique_df



def query_with_edge_col(
    df,
    query,
    edge_delimiter = "->"
    ):
    """
    Purpose: To do an edge query that will
    1) Rename the column values
    2) Rename the query

    so that it is valid with pandas querying
    """

    edge_delimiter = "->"

    edge_delimiter_new = "arrow"

    replace_dict = {edge_delimiter:edge_delimiter_new}

    query_new = reu.multiple_replace(query,replace_dict)
    new_column_dict = {k:reu.multiple_replace(k,replace_dict) for k in df.columns}
    new_column_dict_reverse = {v:k for k,v in new_column_dict.items()}
    df_new = pu.rename_columns(df,new_column_dict)

    df_filt = df_new.query(query_new)
    df_filt = pu.rename_columns(df_filt,new_column_dict_reverse)

    return df_filt


def filter_motif_df(
    df,
    node_filters = None,
    min_gnn_probability = None, #gives about a 90% on inhibitory cells
    edges_filters = None,
    single_edge_motif = False,
    cell_type_fine_exclude = None,
    verbose = False,
    ):
    """
    Purpose: To restrict a motif with node
    and edge requirements
    
    Ex: 
    from neurd_packages import motif_utils as mfu

    G = hdju.G_auto_DiGraph

    unique_df = mfu.annotated_motif_df(
        motif = "A->B;B->A",
        G = hdju.G_auto_DiGraph,
        n_samples = None,
        verbose = False
    )
    
    
    mfu.filter_motif_df(
        unique_df,
        min_gnn_probability = 0.5,
        edges_filters = [
                "edge_postsyn_compartment == 'soma'",
            ]
    )
    """
    if min_gnn_probability is None:
        min_gnn_probability = min_gnn_probability_global
        
    if cell_type_fine_exclude is None:
        cell_type_fine_exclude = cell_type_fine_exclude_global
        
    
    
    if verbose:
        print(f"***Filtering motif df***")

    if node_filters is None:
        node_filters = [
            "node_gnn_cell_type_fine == node_gnn_cell_type_fine",
            "node_gnn_cell_type_fine != 'None'",
            f"node_gnn_cell_type_fine_prob > {min_gnn_probability}",
            f"node_gnn_cell_type_fine not in {cell_type_fine_exclude}",
            f"node_cell_type == node_gnn_cell_type",

        ]
        
#         if single_edge_motif:
#             node_filters = [f"node_{k}" for k in node_filters]
#             node_filters[0] = "node_gnn_cell_type_fine == node_gnn_cell_type_fine"
            

    if edges_filters is None:
        edges_filters = [
            #"edge_postsyn_compartment == 'soma'",
        ]


    if not single_edge_motif:
        curr_str = df.iloc[0,:]["motif_str"]
        edges_str = mfu.edges_from_str(curr_str,return_edge_str = True)
        nodes_str = mfu.nodes_from_str(curr_str)
        #print('inside motif str')
    else:
        nodes_str = ["presyn","postsyn"]


    query_str = []

    for nf in node_filters:
        if "node" in nf:
            curr_query_str = " and ".join([f"( {reu.multiple_replace(nf,dict(node=k))} )" 
                                               for k in nodes_str])
        else:
            curr_query_str = nf
            
        curr_query_str = f"({curr_query_str})"
        query_str.append(curr_query_str)


    
    for ef in edges_filters:
        if "edge" in ef:
            curr_query_str = " and ".join([f"( {reu.multiple_replace(ef,dict(edge=k))} )" 
                                               for k in edges_str])
        else:
            curr_query_str = ef
            
        curr_query_str = f"({curr_query_str})"
        query_str.append(curr_query_str)

    if verbose:
        print(f"query_str =")
        for k in query_str:
            print(f"   {k}")

    total_query = " and ".join(query_str)
    filt_df = mfu.query_with_edge_col(df,total_query)
    return filt_df


def counts_df_from_motif_df(
    motif_df,
    motif_column = "motif_str"):
    
    motif_counts = pu.filter_to_first_instance_of_unique_column(
        motif_df,
        motif_column
    )

    motif_counts = pu.sort_df_by_column(
        motif_counts,
        ["n_motifs","motif_str"])
    
    return motif_counts

def visualize_graph_connections(
    G,
    key,
    verbose = True,
    verbose_visualize = False,
    restrict_to_synapse_ids = True,
    method="neuroglancer",
    **kwargs
    ):
    
    """
    Purpose: To visualize the motif connection
    from an entry in a motif dataframe

    Pseudocode: 
    1) Turn entry into dict if not
    2) Get the node names for the motif
    3) Get the synapse ids
    4) Plot the connections

    """

    if type(key) != dict:
        key = key.to_dict()

    motif_str = key["motif_str"]
    if verbose:
        print(f"motif_str = {motif_str}")

    #2) Get the node names for the motif
    node_names = [key[f'{k}_name']
                  for k in mfu.nodes_from_str(motif_str)]
    if verbose:
        print(f"node_names= {node_names}")

    #3) Get the synapse ids
    if restrict_to_synapse_ids:
        edges = mfu.edges_from_str(motif_str,return_edge_str=True)
        synapse_ids = [key[f"{k}_synapse_id"] for k in edges]
    else:
        synapse_ids = None

    if verbose:
        print(f"synapse_ids= {synapse_ids}")

    from neurd_packages import connectome_utils as conu
    return conu.visualize_graph_connections_by_method(
        G,
        segment_ids=node_names,
        method=method,
        verbose = verbose_visualize,  
    )


motif_Gs_for_n_nodes = xu.motif_Gs_for_n_nodes



# ----------------- Helper functions for 3D analysis ------------- #

# -- default
attributes_dict_default = dict(
    #voxel_to_nm_scaling = microns_volume_utils.voxel_to_nm_scaling,
    hdju = mvu.data_interface
)    
global_parameters_dict_default = dict(
    #max_ais_distance_from_soma = 50_000
)

# -- microns
global_parameters_dict_microns = {}
attributes_dict_microns = {}

#-- h01--
from . import h01_volume_utils as hvu
attributes_dict_h01 = dict(
    #voxel_to_nm_scaling = h01_volume_utils.voxel_to_nm_scaling,
    hdju = hvu.data_interface
)
global_parameters_dict_h01 = dict()
    
       
# data_type = "default"
# algorithms = None
# modules_to_set = [mfu]

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
from . import microns_volume_utils as mvu

#--- from python_tools ---
from python_tools import module_utils as modu 
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu
from python_tools import pandas_utils as pu
from python_tools import regex_utils as reu
from python_tools.tqdm_utils import tqdm

from . import motif_utils as mfu
from python_tools import dotmotif_utils as dmu