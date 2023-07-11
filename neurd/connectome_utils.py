'''


Purpose: To provide helpful functions for analyzing the microns 
grpah



'''
import copy
import h
import matplotlib.pyplot as plt
import pandas as pd
import time


return_nm_default = False

def synapse_coordinate_from_seg_split_syn_id(G,
                                             presyn_id,
                                            postsyn_id,
                                            synapse_id,
                                            return_nm = return_nm_default):
    """
    Will return a syanpse coordinate based on presyn,postsyn id and synapse id
    
    Ex: 
    conu.synapse_coordinate_from_seg_split_syn_id(
    G,
        pre_seg,
        post_seg,
        synapse_id,
        return_nm = True
    )
    """
    
    e_dict = dict(G[presyn_id][postsyn_id])
    
    if xu.is_multigraph(G):
        syn_coord = [[j[f"synapse_{k}"] for k in ["x","y","z"]] for j in e_dict.values() if 
     j["synapse_id"] == synapse_id]
    else:
        syn_coord = [[j[f"synapse_{k}"] for k in ["x","y","z"]] for j in [e_dict] if 
     j["synapse_id"] == synapse_id]
    
    if len(syn_coord) != 1:
        raise Exception("Not just one synapse")
        
    syn_coord = np.array(syn_coord[0])
    
    if return_nm:
        return syn_coord*hdju.voxel_to_nm_scaling
    else:
        return syn_coord


def synapse_ids_and_coord_from_segment_ids_edge(G,segment_id_1,segment_id_2,
                                                    return_nm=return_nm_default,
                                                    verbose = False):
    synapse_ids = []
    synapse_coordinates = []
    
    if segment_id_2 not in G[segment_id_1].keys():
        pass
    else:
        edges_dict = dict(G[segment_id_1][segment_id_2])
        if xu.is_multigraph(G):
            for e_idx,e_dict in edges_dict.items():
                synapse_ids.append(e_dict["synapse_id"])
                synapse_coordinates.append([e_dict[f"synapse_{k}"] for k in ["x","y","z"]])
        else:
            e_dict = edges_dict
            synapse_ids.append(e_dict["synapse_id"])
            synapse_coordinates.append([e_dict[f"synapse_{k}"] for k in ["x","y","z"]])
            
    synapse_ids = np.array(synapse_ids)
    synapse_coordinates = np.array(synapse_coordinates).reshape(-1,3)
    
    if return_nm: 
        synapse_coordinates = synapse_coordinates*hdju.voxel_to_nm_scaling
        
    if verbose:
        print(f"\nFor {segment_id_1} --> {segment_id_2}:")
        print(f"synapse_ids = {synapse_ids}")
        print(f"synapse_coordinates = {synapse_coordinates}\n")
    return synapse_ids,synapse_coordinates

def pre_post_node_names_from_synapse_id(
    G,
    synapse_id,
    node_names = None,#node names to help restrict the search and make it quicker
    return_one = True):
    """
    Purpose: To go from synapse ids to the 
    presyn postsyn segments associated with them
    by a graph lookup
    
    Example: 
    conu.pre_post_node_names_from_synapse_id(
    G,
    node_names = segment_ids,
    synapse_id = 649684,
    return_one = True,

    )
    """

    if node_names is None:
        node_names = list(G.nodes())

    pre_post_pairs = xu.edge_df_from_G(G.subgraph(node_names)).query(f"synapse_id == {synapse_id}")[["source","target"]].to_numpy()

    if return_one:
        if len(pre_post_pairs) > 1:
            raise Exception("")
        pre_post_pairs = pre_post_pairs[0]

    return list(pre_post_pairs)

def segment_ids_from_synapse_ids(
    G,
    synapse_ids,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To get all segment ids involved
    with one synapse
    
    Ex:
    segment_ids = conu.segment_ids_from_synapse_ids(
            G,
            synapse_ids=16551759,
            verbose = verbose,
            )
    
    """
    
    synapse_ids= nu.convert_to_array_like(synapse_ids)
    seg_ids = np.concatenate(
    [
        conu.pre_post_node_names_from_synapse_id(
            G,
            synapse_id = k,
            return_one = True,
            **kwargs
        ) for k in synapse_ids
    ]
    )
    
    seg_ids = list(np.unique(seg_ids))
    
    if verbose:
        print(f"Segment Ids from synapse [{synapse_ids}] = {seg_ids}")
        
    return seg_ids


def synapses_from_segment_id_edges(G,
                                  segment_id_edges=None,
                                  segment_ids =None,
                                  synapse_ids=None,
                                  return_synapse_coordinates = True,
                                   return_synapse_ids = False,
                                  return_nm = return_nm_default,
                                   return_in_dict = False,
                                  verbose = False):
    """
    Purpose: For all segment_ids get the 
    synapses  or synapse coordinates for 
    the edges between them from the graph

    Ex: 
    seg_split_ids = ["864691136388279671_0",
                "864691135403726574_0",
                "864691136194013910_0"]

    conu.synapses_from_segment_id_edges(G,segment_ids = seg_split_ids,
                                  return_nm=True)
    """
    if synapse_ids is not None: 
        synapse_ids = nu.convert_to_array_like(synapse_ids)
    

    if segment_id_edges is None and segment_ids is not None:
        segment_id_edges = nu.all_directed_choose_2_combinations(segment_ids)
    elif segment_id_edges is None:
        raise Exception("")
    else:
        pass

    if verbose:
        print(f"segment_id_edges = {segment_id_edges}")

    synapses_dict = dict()
    synapses_coord_dict = dict()
    for seg_1,seg_2 in segment_id_edges:
        syn_ids,syn_coords = conu.synapse_ids_and_coord_from_segment_ids_edge(G,
                                                    seg_1,
                                                    seg_2,
                                                    return_nm =return_nm,
                                                    verbose = verbose)

        if synapse_ids is not None and len(syn_ids) > 0 and synapse_ids[0] is not None :
            syn_ids,syn_ids_idx,_ = np.intersect1d(syn_ids,synapse_ids,return_indices=True)
            syn_coords = syn_coords[syn_ids_idx]

            if verbose:
                print(f"After synapse restriction: syn_ids= {syn_ids}")
                print(f"syn_coords= {syn_coords}")

        if len(syn_ids) > 0:
            if seg_1 not in synapses_dict.keys():
                synapses_dict[seg_1] = dict()
                synapses_coord_dict[seg_1] = dict()

            synapses_dict[seg_1][seg_2] = syn_ids
            synapses_coord_dict[seg_1][seg_2] = syn_coords
            
    if not return_in_dict and len(synapses_dict) > 0:
        synapses_coord_non_dict = []
        synapses_non_dict = []
        for seg_1 in synapses_dict.keys():
            for seg_2 in synapses_dict[seg_1].keys():
                synapses_coord_non_dict.append(synapses_coord_dict[seg_1][seg_2])
                synapses_non_dict.append(synapses_dict[seg_1][seg_2])
                
        synapses_coord_dict = np.concatenate(synapses_coord_non_dict)
        synapses_dict = np.concatenate(synapses_non_dict)
        
    
    if return_synapse_coordinates:
        if return_synapse_ids:
            return synapses_coord_dict,synapses_dict
        else:
            return synapses_coord_dict
    else:
        return synapses_dict
    
    
def soma_center_from_segment_id(
    G,
    segment_id,
    return_nm = return_nm_default
    ):
    """
    Purpose: To get the soma center of a segment id
    from the Graph

    Ex: 
    conu.soma_center_from_segment_id(G,
                                segment_id = "864691136388279671_0",
    return_nm = True)
    """


    node_dict = G.nodes[segment_id]
    
    if return_nm:
        soma_center = np.array([node_dict[f"centroid_{x}_nm"] for x in ["x","y","z"]])
    else:
        soma_center = np.array([node_dict[f"centroid_{x}"] for x in ["x","y","z"]])
        #soma_center = soma_center * hdju.voxel_to_nm_scaling

    return soma_center

def soma_centers_from_segment_ids(G,
                                 segment_ids,
                                 return_nm = return_nm_default):
    return np.array([conu.soma_center_from_segment_id(G,s,return_nm=return_nm)
                    for s in segment_ids]).reshape(-1,3)

def segment_id_from_seg_split_id(seg_split_id):
    return int(seg_split_id.split("_")[0])



'''
def visualize_graph_connections_neuroglancer(G,
                                             segment_ids,
                                            segment_ids_colors = None,
                                            synapse_color = "yellow",
                                            plot_soma_centers = True,
                                            transparency = 0.3,
                                            output_type="html",
                                            verbose = False,
                                            ):
    """
    Purpose: To visualize neurons and their synapses
    in neuroglancer

    Psuedocode: 
    1) Get colors for every segment
    2) Get the synapses for all the segment pairs
    3) 

    Ex: 
    segment_ids = ["864691136388279671_0",
                    "864691135403726574_0",
                    "864691136194013910_0"]
                    
    conu.visualize_graph_connections_neuroglancer(G,segment_ids = ["864691136388279671_0",
                    "864691135403726574_0",
                    "864691136194013910_0"])
    """


    if segment_ids_colors is None:
        segment_ids_colors = mu.generate_non_randon_named_color_list(len(segment_ids))
    else:
        segment_ids_colors = nu.convert_to_array_like(segment_ids_colors)

    if verbose:
        print(f"segment_ids_colors = {segment_ids_colors}")


    #2) Get the synapses for all the segment pairs
    syn_coords = conu.synapses_from_segment_id_edges(G,
                                                    segment_ids=segment_ids,
                                                   return_nm = False,
                                                   verbose = verbose)
    if verbose:
        print(f"syn_coords = {syn_coords}")

    annotations_info = dict(presyn=dict(color=synapse_color,
                                    coordinates=list(syn_coords)))

    #3) Get the soma centers if requested
    if plot_soma_centers:
        soma_centers = conu.soma_centers_from_segment_ids(G,segment_ids,return_nm = False)

        if verbose:
            print(f"soma_centers = {soma_centers}")

        for s_idx,(sc,col) in enumerate(zip(soma_centers,segment_ids_colors)):
            annotations_info[f"neuron_{s_idx}"] = dict(color = col,
                                                      coordinates = list(sc.reshape(-1,3)))

    if verbose:
        print(f"annotations_info= {annotations_info}")


    #4) Get the regular int names for segment_ids
    seg_ids_int = [conu.segment_id_from_seg_split_id(k) for k in segment_ids]
    if verbose:
        print(f"seg_ids_int = {seg_ids_int}")


    return alu.coordinate_group_dict_to_neuroglancer(seg_ids_int[0],
                                          annotations_info,
                                                    output_type=output_type,
                                                    fixed_ids = seg_ids_int,
                                                    fixed_id_colors = segment_ids_colors,
                                                    transparency=transparency)

'''

def visualize_graph_connections_by_method(
    G,
    segment_ids=None,
    method = 'meshafterparty',#'neuroglancer'
    synapse_ids = None, #the synapse ids we should be restricting to
    segment_ids_colors = None,
    synapse_color = "red",
    plot_soma_centers = True,
    verbose = False,
    verbose_cell_type = True,
    
    
    plot_synapse_skeletal_paths = True,
    plot_proofread_skeleton = False,

    #arguments for the synapse path information
    synapse_path_presyn_color = "aqua",
    synapse_path_postsyn_color = "orange",
    synapse_path_donwsample_factor = None,

    #arguments for neuroglancer
    transparency = 0.9,
    output_type="server",

    #arguments for meshAfterParty
    plot_compartments = False,
    plot_error_mesh = False,
    synapse_scatter_size = 0.2,#0.2
    synapse_path_scatter_size = 0.1,
    
    debug_time = False,
    
    plot_gnn = True,
    gnn_embedding_df = None,
    ):

    """
    Purpose: A generic function that will 
    prepare the visualization information 
    for either plotting in neuroglancer or meshAfterParty


    Pseudocode: 
    0) Determine if whether should return things in nm
    1) Get the segment id colors
    2) Get the synapses for all the segment pairs
    3) Get the soma centers if requested
    4) Get the regular int names for segment_ids (if plotting in neuroglancer)
    
    Ex: 
    from neurd_packages import connectome_utils as conu
    conu.visualize_graph_connections_by_method(
        G,
        ["864691136023767609_0","864691135617737103_0"],
        method = "neuroglancer"
    )
    """
    
    
    if synapse_ids is not None:
        synapse_ids = nu.convert_to_array_like(synapse_ids)
    if segment_ids is None:
        segment_ids = conu.segment_ids_from_synapse_ids(
            G,
            synapse_ids,
            verbose = verbose,
            )
        
    segment_ids= [k if type(k) == str else f"{k}_0" for k in segment_ids]
    
    #Pre work: Setting up the synapse scatter sizes
    scatters = []
    scatter_size = []
    scatters_colors = []
    
    method = method.lower()

    #0) Determine if whether should return things in nm
    if method == "neuroglancer":
        import allen_utils as alu
        alu.initialize_client()
        alu.set_version_to_latest()
        return_nm = False
    elif method == "meshafterparty":
        return_nm = True
    else:
        raise Exception("")

    #1) Get the segment id colors
    if segment_ids_colors is None:
        segment_ids_colors = mu.generate_non_randon_named_color_list(len(segment_ids))
    else:
        segment_ids_colors = nu.convert_to_array_like(segment_ids_colors)

    #if verbose:
    #print(f"segment_ids_colors = {segment_ids_colors}")


    #2) Get the synapses for all the segment pairs
    syn_coords,syn_ids = conu.synapses_from_segment_id_edges(G,
                                                    segment_ids=segment_ids,
                                                    synapse_ids=synapse_ids,
                                                   return_nm = return_nm,
                                                    return_synapse_ids=True,
                                                   verbose = verbose)

    if len(syn_coords) == 0:
        raise Exception("No synapses to plot")

    annotations_info = dict(presyn=dict(color=synapse_color,
                                        coordinates=list(syn_coords)))

    if verbose:
        print(f"syn_coords = {syn_coords}")
        print(f"syn_ids = {syn_ids}")


    #3) Get the soma centers if requested
    if plot_soma_centers:
        soma_centers = conu.soma_centers_from_segment_ids(G,
                                                         segment_ids,
                                                         return_nm = return_nm)

        if verbose:
            print(f"soma_centers = {soma_centers}")

        for s_idx,(sc,col) in enumerate(zip(soma_centers,segment_ids_colors)):
            annotations_info[f"neuron_{stru.number_to_letter(s_idx).upper()}"] = dict(color = col,
                                                      coordinates = list(sc.reshape(-1,3)))
            
    if plot_synapse_skeletal_paths:
        for synapse_id in syn_ids:
            pre_name,post_name = conu.pre_post_node_names_from_synapse_id(G,synapse_id,node_names = segment_ids)
            pre_idx = [k for k,seg in enumerate(segment_ids)  if pre_name == seg][0]
            post_idx = [k for k,seg in enumerate(segment_ids)  if post_name == seg][0]
            presyn_path,postsyn_path = conu.presyn_postsyn_skeletal_path_from_synapse_id(
                                            G,
                                            synapse_id = synapse_id,
                                            segment_ids = segment_ids,
                                            return_nm = return_nm,
                                            verbose = verbose,
                                            plot_skeletal_paths=False,
                debug_time=debug_time

                                        )
            #annotations_info[f"pre_{synapse_id}"] = dict(
            
            unique_flag = False
            curr_name = f"{stru.number_to_letter(pre_idx).upper()}_to_{stru.number_to_letter(post_idx).upper()}"
            curr_idx = 0
            while not unique_flag:
                if f"pre_{curr_name}_idx{curr_idx}" in annotations_info:
                    curr_idx += 1
                else:
                    unique_flag = True
                    
            curr_name = f"{curr_name}_idx{curr_idx}"
                
                
            annotations_info[f"pre_{curr_name}"] = dict(
                color = synapse_path_presyn_color,
                coordinates = list(presyn_path)
            )
            
            #annotations_info[f"post_{synapse_id}"] = dict(
            annotations_info[f"post_{curr_name}"] = dict(
                color = synapse_path_postsyn_color,
                coordinates = list(postsyn_path)
            )

    if not return_nm: 
        #4) Get the regular int names for segment_ids
        seg_ids_int = [conu.segment_id_from_seg_split_id(k) for k in segment_ids]
        if verbose:
            print(f"seg_ids_int = {seg_ids_int}")

            
    if plot_gnn and not return_nm:
        hdju.plot_gnn_embedding(
            node_name = segment_ids,
            df = gnn_embedding_df,
        )

    if not return_nm:
        return_value = alu.coordinate_group_dict_to_neuroglancer(seg_ids_int[0],
                                              annotations_info,
                                                        output_type=output_type,
                                                        fixed_ids = seg_ids_int,
                                                        fixed_id_colors = segment_ids_colors,
                                                        transparency=transparency)
        return_value =  alu.neuroglancer_output_to_link(return_value)
        
    else:

        """
        Pseudocode on how to plot in meshAfterParty:

        For each group in annotations_info without "neuron" in name:
        1) Add the coordinates,color and size to the scatters list

        2) Arguments to set: 
        proofread_mesh_color= segment_ids_colors
        plot_nucleus = True
        plot_synapses = False
        plot_error_mesh = plot_error_mesh
        compartments=compartments

        """
        #print(f"annotations_info.keys() = {annotations_info.keys()}")
        
        for k,syn_dict in annotations_info.items():
            if "neuron" in k:
                continue
                
            if "path" in k or "_to_" in k:
                curr_syn_size = synapse_path_scatter_size
            else:
                curr_syn_size= synapse_scatter_size
                
            scatters.append(np.array(syn_dict["coordinates"]).reshape(-1,3))
            scatter_size.append(curr_syn_size)
            scatters_colors.append(syn_dict["color"])

            
        print(f"scatter_size = {scatter_size}")
        if plot_compartments:
            compartments = None
        else:
            compartments = []

        hdju.plot_multiple_proofread_neuron(
            segment_ids = segment_ids,
            plot_proofread_skeleton = plot_proofread_skeleton,
            proofread_mesh_color = segment_ids_colors,
            proofread_mesh_alpha = transparency,
            proofread_skeleton_color = segment_ids_colors,
            plot_nucleus = plot_soma_centers,
            plot_synapses = False,
            compartments = compartments,
            plot_error_mesh = plot_error_mesh,
            

            scatters = scatters,
            scatter_size = scatter_size,
            scatters_colors = scatters_colors,
            plot_gnn=plot_gnn,
            gnn_embedding_df=gnn_embedding_df,
            align = False,

        )
        
        return_value = None
        
    if verbose_cell_type:
        for s in segment_ids:
            print(f'{s}:{dict([(k,v) for k,v in hdju.cell_info_from_name(s).items() if k in ["cell_type","external_cell_type_fine","external_cell_type"]])}')
            gnn_str = hdju.gnn_cell_type_info(
                segment_id = s,
                return_str=True)
            print(f"   ->{gnn_str}")
            
    return return_value
        

def presyn_postsyn_skeletal_path_from_synapse_id(
    G,
    synapse_id,
    synapse_coordinate = None,
    segment_ids = None,
    return_nm = return_nm_default,
    
    #arguments for plotting the paths
    plot_skeletal_paths = False,
    synapse_color = "red",
    synapse_scatter_size = 0.2,#,
    path_presyn_color = "yellow",
    path_postsyn_color = "blue",
    path_scatter_size = 0.05,#0.2,
    plot_meshes = True,
    mesh_presyn_color = "orange",
    mesh_postsyn_color = "aqua",
    verbose = False,
    debug_time = False,
    
    segment_length = 2000,
    
    remove_soma_synanpse_nodes = True,
    ):
    """
    Purpose: To develop a skeletal path coordinates between 
    two segments ids that go through a soma 

    Application: Can then be sent to a plotting function

    Pseudocode: 
    1) Get the segment ids paired with that synapse id 
    (get synapse coordinate if not precomputed)
    2) Get the proofread skeletons associated with the segment_ids
    3) Get the soma center coordinates to determine paths
    4) Get the closest skeleton node to the coordinate
    5) Find the skeletal path in coordinates
    6) Plot the skeletal paths
    
    
    Ex: 
    G["864691136830770542_0"]["864691136881594990_0"]
    synapse_id = 299949435
    segment_ids = ["864691136830770542_0","864691136881594990_0"]

    conu.presyn_postsyn_skeletal_path_from_synapse_id(
        G,
        synapse_id = synapse_id,
        synapse_coordinate = None,
        segment_ids = segment_ids,
        return_nm = True,
        verbose = True,
        plot_skeletal_paths=True,
        path_scatter_size = 0.04,

    )
    """
    if debug_time: 
        st = time.time()
    #2) Get the proofread skeletons associated with the segment_ids
    pre_seg,post_seg = conu.pre_post_node_names_from_synapse_id(
        G,
        node_names = segment_ids,
        synapse_id = synapse_id,
        return_one = True,

    )
    
    if debug_time: 
        print(f"Time for getting pre post from node names = {time.time() - st}")
        st = time.time()
    
    

    if synapse_coordinate is None:
        synapse_coordinate= conu.synapse_coordinate_from_seg_split_syn_id(
        G,
            pre_seg,
            post_seg,
            synapse_id,
            return_nm = True
        )

    if verbose:
        print(f"syn_id = {synapse_id}: pre_seg = {pre_seg}, post_seg = {post_seg}")
        print(f"synapse_coordinate = {synapse_coordinate}")
        
    if debug_time: 
        print(f"Time for finding synapse_coordinate = {time.time() - st}")
        st = time.time()
    

    segment_ids = [pre_seg,post_seg]
    seg_type = ["presyn","postsyn"]


    #2) Get the proofread skeletons associated with the segment_ids
    seg_sks = [hdju.fetch_proofread_skeleton(*hdju.segment_id_and_split_index(k))
               for k in segment_ids]
    
    if debug_time: 
        print(f"Time for fetch_proofread_skeleton = {time.time() - st}")
        st = time.time()



    #3) Get the soma center coordinates to determine paths
    soma_coordinates = [conu.soma_center_from_segment_id(
                            G,s,return_nm = True) for s in segment_ids]
    
    if debug_time: 
        print(f"Time for soma_center_from_segment_id = {time.time() - st}")
        st = time.time()

    soma_coordinates_closest = [sk.closest_skeleton_coordinate(
                            curr_sk,soma_c) for curr_sk,soma_c in 
                         zip(seg_sks,soma_coordinates)] 
    
    if debug_time: 
        print(f"Time for closest_skeleton_coordinate = {time.time() - st}")
        st = time.time()
    
    

    if verbose:
        print(f"\n Soma Information:")
        print(f"soma_coordinates = {soma_coordinates}")
        print(f"soma_coordinates_closest = {soma_coordinates_closest}")


    #4) Get the closest skeleton node to the coordinate
    closest_sk_coords = [sk.closest_skeleton_coordinate(
                            curr_sk,synapse_coordinate) for curr_sk in 
                         seg_sks]
    
    if debug_time: 
        print(f"Time for closest_skeleton_coordinate for syn = {time.time() - st}")
        st = time.time()
    
    

    #5) Find the skeletal path in coordinates
    skeletal_coord_paths = []
    for idx in range(len(seg_sks)):
        curr_sk = sk.skeleton_path_between_skeleton_coordinates(
            starting_coordinate = soma_coordinates_closest[idx],
            destination_coordinate = closest_sk_coords[idx],
            skeleton = seg_sks[idx],
            plot_skeleton_path = False,
            return_singular_node_path_if_no_path = True
            )
        if segment_length is not None and len(curr_sk) > 2:
            curr_sk = sk.resize_skeleton_with_branching(
                curr_sk,segment_length)
            
        curr_sk = np.array(sk.convert_skeleton_to_nodes(curr_sk)).reshape(-1,3)
        skeletal_coord_paths.append(curr_sk)
    
#     skeletal_coord_paths = [np.array(sk.convert_skeleton_to_nodes(
#         sk.skeleton_path_between_skeleton_coordinates(
#             starting_coordinate = soma_coordinates_closest[idx],
#             destination_coordinate = closest_sk_coords[idx],
#             skeleton = seg_sks[idx],
#             plot_skeleton_path = False,
#             return_singular_node_path_if_no_path = True
#             )).reshape(-1,3)) 
#         for idx in range(len(seg_sks))]

    if verbose:
        print(f"Path lengths = {[len(k) for k in skeletal_coord_paths]}")
        
    if debug_time: 
        print(f"Time for skeletal paths = {time.time() - st}")
        st = time.time()

    
    if remove_soma_synanpse_nodes:
        skeletal_coord_paths_revised =[]
        for soma_c,syn_c,curr_path in zip(soma_coordinates_closest,
                                    closest_sk_coords,
                                    skeletal_coord_paths):
            if len(curr_path) > 1:
                skeletal_coord_paths_revised.append(
                    nu.setdiff2d(curr_path,np.array([soma_c,syn_c]))
                )
            else:
                skeletal_coord_paths_revised.append(np.mean(np.vstack([soma_c,syn_c]),axis=0).reshape(-1,3))
                
        skeletal_coord_paths = skeletal_coord_paths_revised
        
        if debug_time: 
            print(f"Time for removing soma from path = {time.time() - st}")
            st = time.time()
        
    if plot_skeletal_paths:
        print(f"Plotting synapse paths")

        scatters = [synapse_coordinate.reshape(-1,3)]
        scatter_size = [synapse_scatter_size]
        scatters_colors = [synapse_color]

        path_colors = [path_presyn_color,path_postsyn_color]

        for p_sc,p_col in zip(skeletal_coord_paths,
                              path_colors):
            scatters.append(p_sc)
            scatter_size.append(path_scatter_size)
            scatters_colors.append(p_col)

        meshes = []
        meshes_colors = [mesh_presyn_color,mesh_postsyn_color,]

        if plot_meshes:
            meshes = [hdju.fetch_proofread_mesh(k) for k in segment_ids]

        nviz.plot_objects(meshes=meshes,
                          meshes_colors=meshes_colors,
                          skeletons=seg_sks,
                          skeletons_colors=meshes_colors,
                         scatters=scatters,
                         scatter_size=scatter_size,
                         scatters_colors=scatters_colors)
    if not return_nm:
        skeletal_coord_paths = [k/hdju.voxel_to_nm_scaling
                                if len(k) > 0 else k for k in skeletal_coord_paths]
    
    return skeletal_coord_paths





# ----- for computing edge functions over the connectome ----------
def compute_edge_statistic(
    G,
    edge_func,
    verbose = False,
    verbose_loop = False,
    ):
    st = time.time()
    
    for segment_id_1 in tqdm(list(G.nodes())):
        for segment_id_2 in dict(G[segment_id_1]).keys():
            G = edge_func(
                G,
                segment_id_1 = segment_id_1,
                segment_id_2 = segment_id_2,
                verbose = verbose_loop,
            )
    if verbose:
        print(f"Total time for adding {edge_func.__name__} = {time.time() - st} ")
    return G

    
    
def presyn_postsyn_walk_euclidean_skeletal_dist(
    G,
    segment_id_1,
    segment_id_2,
    verbose = False,
    ):
    
    pre_center,post_center = conu.soma_centers_from_segment_ids(G,
                                                        [segment_id_1,
                                                         segment_id_2],
                                                        return_nm=True)
    
    soma_soma_distance = np.linalg.norm(pre_center-post_center)

    if verbose:
        print(f"soma_soma_distance= {soma_soma_distance}")

    
    edges_dict = dict(G[segment_id_1][segment_id_2])
    if not xu.is_multigraph(G):
        edges_dict = {None:edges_dict}
    for e_idx,e_dict in edges_dict.items():
        #print(f"e_dict = {e_dict}")
        syn_coord = syn_coordinate_from_edge_dict(e_dict)
        
        xu.set_edge_attribute(G,segment_id_1,segment_id_2,
                              "presyn_soma_euclid_dist",np.linalg.norm(pre_center-syn_coord),edge_idx = e_idx)
        xu.set_edge_attribute(G,segment_id_1,segment_id_2,
                              "postsyn_soma_euclid_dist",np.linalg.norm(post_center-syn_coord),edge_idx = e_idx)
        xu.set_edge_attribute(G,segment_id_1,segment_id_2,
                              "presyn_soma_postsyn_soma_euclid_dist",soma_soma_distance,edge_idx = e_idx)
    
#         G[segment_id_1][segment_id_2][e_idx]["presyn_soma_euclid_dist"] = np.linalg.norm(pre_center-syn_coord)
#         G[segment_id_1][segment_id_2][e_idx]["postsyn_soma_euclid_dist"] = np.linalg.norm(post_center-syn_coord)
#         G[segment_id_1][segment_id_2][e_idx]["presyn_soma_postsyn_soma_euclid_dist"] = soma_soma_distance

        soma_walks = np.array([
            get_edge_attribute(G,segment_id_1,segment_id_2,k,edge_idx = e_idx)
            for k in ["postsyn_skeletal_distance_to_soma","presyn_skeletal_distance_to_soma"]])
        soma_walks[soma_walks<0] = 0
    
        xu.set_edge_attribute(G,segment_id_1,segment_id_2,
                              "presyn_soma_postsyn_soma_skeletal_dist",
                              np.sum(soma_walks),
                              edge_idx = e_idx)
#         G[segment_id_1][segment_id_2][e_idx]["presyn_soma_postsyn_soma_skeletal_dist"] = np.sum(soma_walks)
    
    return G
    
    
def attribute_from_edge_dict(edge_dict,attribute):
    edge_dict = dict(edge_dict)
    try:
        return edge_dict[attribute]
    except:
        return np.array([e_dict[attribute] for e_idx,e_dict in edge_dict.items()])
    
def postsyn_skeletal_distance_to_soma_from_edge_dict(edge_dict):
    return attribute_from_edge_dict(
        edge_dict,
        attribute = "postsyn_skeletal_distance_to_soma"
    )
    
def syn_coordinate_from_edge_dict(edge_dict):
    try:
        return  np.array([edge_dict[f"synapse_{k}"] for k in ["x","y","z"]])*hdju.voxel_to_nm_scaling
    except:
        edge_dict = dict(edge_dict)
        return  np.array([[e_dict[f"synapse_{k}"] for k in ["x","y","z"]] for e_idx,e_dict in edge_dict.items()]).reshape(-1,3)*hdju.voxel_to_nm_scaling

def compute_presyn_postsyn_walk_euclidean_skeletal_dist(
    G,
    verbose = False,
    verbose_loop = False,
    ):
    
    return conu.compute_edge_statistic(
    G,
    edge_func=conu.presyn_postsyn_walk_euclidean_skeletal_dist,
    verbose = verbose,
    verbose_loop = verbose_loop,
    )

def presyn_postsyn_soma_relative_synapse_coordinate(
    G,
    segment_id_1,
    segment_id_2,
    verbose = False,
    ):
    
    pre_center,post_center = conu.soma_centers_from_segment_ids(G,
                                                        [segment_id_1,
                                                         segment_id_2],
                                                        return_nm=True)
    
    edges_dict = dict(G[segment_id_1][segment_id_2])
    if not xu.is_multigraph(G):
        edges_dict = {None:edges_dict}
        
    for e_idx,e_dict in edges_dict.items():
        syn_coord = conu.syn_coordinate_from_edge_dict(e_dict)
        for syn_type,center in zip(
            ["presyn","postsyn"],
            [pre_center,post_center]):
            relative_coord = (syn_coord - center)/hdju.voxel_to_nm_scaling
            for c,v in zip(["x","y","z"],relative_coord):
                xu.set_edge_attribute(
                    G,segment_id_1,segment_id_2,
                    f"{syn_type}_soma_relative_synapse_{c}",
                    np.round(v,2),
                    edge_idx = e_idx
                )
    return G

def add_presyn_postsyn_syn_dist_signed_to_edge_df(
    df,
    centroid_name = "centroid",
    ):

    for syn_type in ["presyn","postsyn"]:
        for j,ax in enumerate(["x","y","z"]):
            df[f"{syn_type}_syn_to_soma_euclid_dist_{ax}_nm_signed"] = (
                df[f"synapse_{ax}"]*hdju.voxel_to_nm_scaling[j] - df[f"{syn_type}_{centroid_name}_{ax}_nm"]
            )
            
    return df
    


def computed_presyn_postsyn_soma_relative_synapse_coordinate(
    G,
    verbose = False,
    verbose_loop = False,
    ):
    
    return conu.compute_edge_statistic(
    G,
    edge_func=conu.presyn_postsyn_soma_relative_synapse_coordinate,
    verbose = verbose,
    verbose_loop = verbose_loop,
    )


def presyn_soma_postsyn_soma_euclid_dist_axis(
    G,
    segment_id_1,
    segment_id_2,
    verbose = False,
    ):
    
    pre_center,post_center = conu.soma_centers_from_segment_ids(G,
                                                        [segment_id_1,
                                                         segment_id_2],
                                                        return_nm=True)
    
    dist_dict = dict()
    for j,axis_name in enumerate(["x","y","z"]):
        dist_dict[f"presyn_soma_postsyn_soma_euclid_dist_{axis_name}"] = np.abs(pre_center[j] - post_center[j])
        dist_dict[f"presyn_soma_postsyn_soma_euclid_dist_{axis_name}_signed"] = (pre_center[j] - post_center[j])
        
    dist_dict[f"presyn_soma_postsyn_soma_euclid_dist_xy"] = np.linalg.norm(pre_center[[0,1]] - post_center[[0,1]] )
    dist_dict[f"presyn_soma_postsyn_soma_euclid_dist_yz"] = np.linalg.norm(pre_center[[1,2]] - post_center[[1,2]] )
    dist_dict[f"presyn_soma_postsyn_soma_euclid_dist_xz"] = np.linalg.norm(pre_center[[0,2]] - post_center[[0,2]] )
    
    if verbose:
        print(f"dist_dict:")
        for k,v in dist_dict.items():
            print(f"{k}:{np.round(v,2)}")

    
    edges_dict = dict(G[segment_id_1][segment_id_2])
    if not xu.is_multigraph(G):
        edges_dict = {None:edges_dict}
    for e_idx,e_dict in edges_dict.items():
        for k,v in dist_dict.items():
            xu.set_edge_attribute(
                G,segment_id_1,segment_id_2,
                k,
                np.round(v,2),
                edge_idx = e_idx
            )
    
    return G

def compute_presyn_soma_postsyn_soma_euclid_dist_axis(
    G,
    verbose = False,
    verbose_loop = False,
    ):
    
    return conu.compute_edge_statistic(
    G,
    edge_func=conu.presyn_soma_postsyn_soma_euclid_dist_axis,
    verbose = verbose,
    verbose_loop = verbose_loop,
    )
    
    
def soma_centers_from_node_df(node_df,return_nm = True):
    if return_nm:
        append = "_nm"
    else:
        append = ""
    return node_df[[
        f"centroid_x{append}",
        f"centroid_y{append}",
        f"centroid_z{append}",
        ]].to_numpy()    



def plot_3D_distribution_attribute(
    attribute,
    df = None,
    G = None,
    discrete = False,
    
    # -- for continuous
    density = False,
    n_bins_attribute = 20,
    n_intervals = 10,
    n_bins_intervals = 20,
    
    hue = "gnn_cell_type_fine",
    color_dict = ctu.cell_type_fine_color_map,
    
    verbose = False,
    scatter_size = 0.4,
    axis_box_off = False,
    
    ):
    """
    Purpose: To plot a discrete or continuous value in 3D

    Pseudocode: 
    1) Get the soma centers
    2) Get the current value for all of the nodes
    --> decide if this is discrete of continue

    2a) If continuous:
    --> send to the heat map 3D

    2b) If discrete:
    i) Generate a color list for all the unique values
    ii) Generate a color list for all of the points
    iii) Plot a legend of that for color scale
    iv) Plot the y coordinate of each
    v) Plot the 3D values of each
    
    
    Ex: 
    conu.plot_3D_distribution_attribute(
        #"gnn_cell_type_fine",
        "axon_soma_angle_max",
        df = df_to_plot,
        verbose = True
    )
    """

    if df is None:
        df = xu.node_df(G)

    #df["y_coordinate"] = np.abs(-1*df["centroid_y_nm"].to_numpy())/1000
    soma_centers = conu.soma_centers_from_node_df(df)

    df_filt = df.query(f"{attribute} == {attribute}")

    if verbose:
        print(f"{len(df_filt)}/{len(df)} were not None for {attribute}")

    att_values = df_filt[attribute].to_numpy()
    if "str" in str(type(att_values[0])):
        discrete = True


    if not discrete:
        print(f"Overall distribution")
        mu.histograms_overlayed(
            df_filt,
            attribute,
            color_dict=color_dict,
            density=density,
        )
        plt.show()
        
        
        mu.histograms_over_intervals(
            df = df_filt,
            attribute = attribute,
            interval_attribute = "y_coordinate",
            verbose = True,
            outlier_buffer = 1,
            intervals = None,
            n_intervals = n_intervals,
            overlap = 0.1,
            figsize = (8,4), 
            bins = n_bins_intervals,
            hue = hue,
            color_dict=color_dict,
            density=density,
            )
        
        # plotting in 
        sviz.heatmap_3D(
            values=att_values,
            coordinates=conu.soma_centers_from_node_df(df_filt).reshape(-1,3),
            feature_name = attribute,
            n_bins = n_bins_attribute,
            scatter_size = scatter_size,
            axis_box_off = axis_box_off,
        )
    else:
        if verbose:
            print(f"Plotting Discrete Distribution")
        att_values_unique = df_filt[attribute].unique()

        if verbose:
            print(f"Unique {attribute}: {att_values_unique}")

        #ii) Generate a color list for all of the points
        if color_dict is None:
            color_dict = {k:v for k,v in zip(att_values_unique,
                                            mu.generate_non_randon_named_color_list(len(att_values_unique)))}
            if verbose:
                print(f"Generated color_dict: \n{color_dict}")

        #iii) Plot a legend of that for color scale
        mu.histograms_overlayed(
            df_filt,
            column="y_coordinate",
            hue=attribute,
            color_dict=color_dict,
            )
        plt.show()

        #iv) Plot the 3D distribution
        soma_centers = []
        colors = []
        for cat,c in color_dict.items():
            curr_df = df_filt.query(f"{attribute}=='{cat}'")
            if len(curr_df) == 0:
                curr_df = df_filt.query(f"{attribute}=={cat}")
            soma_centers.append(conu.soma_centers_from_node_df(curr_df).reshape(-1,3))
            colors.append(c)

        from neurd_packages import neuron_visualizations as nviz
        nviz.plot_objects(
            scatters=soma_centers,
            scatters_colors=colors,
            scatter_size=scatter_size,
            axis_box_off=axis_box_off,
        )

def plot_3d_attribute(
    df,
    attribute,
    
    G = None,
    discrete = False,
    
    # -- for continuous
    density = False,
    n_bins_attribute = 20,
    n_intervals = 10,
    n_bins_intervals = 20,
    
    hue = "gnn_cell_type_fine",
    color_dict = ctu.cell_type_fine_color_map,
    
    verbose = False,
    scatter_size = 0.4,
    axis_box_off = False,
    
    plot_visual_area = False,
    
    ):
    
    if df is None:
        df = xu.node_df(G)
    
    df_filt = df.query(f"{attribute} == {attribute}")
    
    att_values = df_filt[attribute].to_numpy()
    if "str" in str(type(att_values[0])):
        discrete = True
    
    if plot_visual_area:
        meshes,meshes_colors = hdju.visual_area_boundaries_plotting()
    else:
        meshes,meshes_colors = [],[]
    
    # plotting in 
    if not discrete:
        coords = conu.soma_centers_from_node_df(df_filt).reshape(-1,3)
        print(f"Done computing coords")
        print(f"att_values = {att_values}")
        sviz.heatmap_3D(
            values=att_values,
            coordinates=coords,
            feature_name = attribute,
            n_bins = n_bins_attribute,
            scatter_size = scatter_size,
            axis_box_off = axis_box_off,
            meshes=meshes,
            meshes_colors=meshes_colors,
        )
        
    else:
        att_values_unique = df_filt[attribute].unique()
        
        if verbose:
            print(f"Unique {attribute}: {att_values_unique}")

        #ii) Generate a color list for all of the points
        if color_dict is None:
            color_dict = {k:v for k,v in zip(att_values_unique,
                                            mu.generate_non_randon_named_color_list(len(att_values_unique)))}
            if verbose:
                print(f"Generated color_dict: \n{color_dict}")
                
        soma_centers = []
        colors = []
        for cat,c in color_dict.items():
            curr_df = df_filt.query(f"{attribute}=='{cat}'")
            if len(curr_df) == 0:
                curr_df = df_filt.query(f"{attribute}=={cat}")
            soma_centers.append(conu.soma_centers_from_node_df(curr_df).reshape(-1,3))
            colors.append(c)

        from neurd_packages import neuron_visualizations as nviz
        nviz.plot_objects(
            scatters=soma_centers,
            scatters_colors=colors,
            scatter_size=scatter_size,
            axis_box_off=axis_box_off,
            meshes=meshes,
            meshes_colors=meshes_colors,
        )
        
def exc_to_exc_edge_df(
    G,
    min_skeletal_length = 100_000,
    verbose = False,
    filter_presyns_with_soma_postsyn = True,
    keep_manual_proofread_nodes = True,
    presyn_name_in_G = "u"
    ):
    """
    Purpose: Produce a filtered edge df
    for excitatory to excitatory connections
    """
    man_proofread_nodes = hdju.manual_proofread_segment_ids_in_auto_proof_nodes
    
    e_labels = list(ctu.allen_cell_type_fine_classifier_labels_exc)

    node_filters = [
        f"skeletal_length > {min_skeletal_length}", #filtering for min skeletal length
        f"gnn_cell_type_fine in {e_labels}",# Filtering for just E to E connections
    ]
    
    node_query = pu.query_str_from_list(
        node_filters,
        table_type = "pandas"
    )
    
    if keep_manual_proofread_nodes:
        node_query= f"({node_query}) or ({presyn_name_in_G} in {list(man_proofread_nodes)})"
        
    #print(f"node_query = {node_query}")

    G_sub = xu.subgraph_from_node_query(G,node_query)
    edge_df = xu.edge_df_optimized(G_sub)

    # 2) Filtering for only neurons without a soma postsyn
    if filter_presyns_with_soma_postsyn:
        soma_presyns = list(edge_df.query(f"postsyn_compartment_coarse == 'soma'")["source"].unique())
        if keep_manual_proofread_nodes:
            soma_presyns = list(np.setdiff1d(soma_presyns,man_proofread_nodes))
        edge_no_soma_df = edge_df.query(f"not (source in {soma_presyns})").reset_index(drop=True)
    else:
        edge_no_soma_df = edge_df
    
    if verbose:
        print(f"# of edges = {len(edge_no_soma_df)}")
    return edge_no_soma_df

def presyns_with_soma_postsyns(
    edge_df,
    keep_manual_proofread_nodes = True,
    man_proofread_nodes=None,
    filter_away_from_df=False):
    
    soma_presyns = list(edge_df.query(f"postsyn_compartment_coarse == 'soma'")["source"].unique())
    
    if keep_manual_proofread_nodes:
        if man_proofread_nodes is None:
            man_proofread_nodes = hdju.manual_proofread_segment_ids_in_auto_proof_nodes
            
        soma_presyns = list(np.setdiff1d(soma_presyns,man_proofread_nodes))
        
    if filter_away_from_df:
        return edge_df.query(f"not (source in {soma_presyns})").reset_index(drop=True)
    else:
        return soma_presyns
    
def filter_away_presyns_with_soma_postsyns(
    edge_df,
    keep_manual_proofread_nodes = True,
    man_proofread_nodes=None,
    ):
    
    return presyns_with_soma_postsyns(
        edge_df,
        keep_manual_proofread_nodes = keep_manual_proofread_nodes,
        man_proofread_nodes=man_proofread_nodes,
        filter_away_from_df=True
    )
    
    

def add_synapse_xyz_to_edge_df(
    edge_df,
    node_df=None,
    G = None,
    ):

    """
    Purpose: To add on the synapses centers onto an edge df
    """
    if node_df is None:
        node_df = xu.node_df(G)

    node_df_lite = node_df[["u","centroid_x_nm","centroid_y_nm","centroid_z_nm"]]
    edge_df_with_centroids = pu.merge_df_to_source_target(
        edge_df,
        node_df_lite,
        on = "u",
        append_type = "prefix",
    )

    edge_df_with_centroids[["synapse_x_nm","synapse_y_nm","synapse_z_nm"]] = edge_df_with_centroids[["synapse_x","synapse_y","synapse_z"]]*hdju.voxel_to_nm_scaling
    return edge_df_with_centroids


def set_edge_attribute_from_node_attribute(
    G,
    attribute,
    verbose = True,
    ):
    return xu.set_edge_attribute_from_node_attribute(
        G,
        attribute = attribute,
        upstream_prefix = "presyn",
        downstream_prefix = "postsyn",
        verbose = verbose
    )     

def set_edge_presyn_postsyn_centroid(
    G,
    verbose = True,
    ):
    return xu.set_edge_attribute_from_node_attribute(
        G,
        attribute = [
            "centroid_x_nm",
            "centroid_y_nm",
            "centroid_z_nm",
        ],
        upstream_prefix = "presyn",
        downstream_prefix = "postsyn",
        verbose = verbose
    )     

def add_axes_subset_soma_to_syn_euclidean_dist_to_edge_df(
    edge_df,
    syn_type = ("presyn","postsyn"),
    axes = "xz",
    ):
    """
    Purpose: To add the distance measure
    from synapse to presyn or postsyn
    """
    syn_type = nu.convert_to_array_like(syn_type,include_tuple=True)
    axes = nu.convert_to_array_like(axes,include_tuple=True)
    
    edge_df[["synapse_x_nm","synapse_y_nm","synapse_z_nm"]] = edge_df[["synapse_x","synapse_y","synapse_z"]]*hdju.voxel_to_nm_scaling

    for s in syn_type:
        for axes_comb in axes:
            #print(f"axes_comb = {axes_comb}")
            ax = (axes_comb[0],axes_comb[1])
            edge_df[f"{s}_soma_euclid_dist_{axes_comb}"] = np.linalg.norm(
                pu.coordinates_from_df(edge_df,name="synapse",axes = ax) - 
                pu.coordinates_from_df(edge_df,name=f"{s}_centroid",
                                       axes = ax),axis=1
            )

    return edge_df
 
    
def radius_cell_type_sampling(
    df=None,
    center=None,
    radius = None,
    cell_type_coarse = None,
    cell_type_fine = None,
    randomize = True,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To find cells within a certain
    query (radius is a query)

    queryable: 
    1) Radius (center)
    2) Cell Type

    Pseudocode: 
    1) Get the seg_split_centroid table
    2) Apply the radius restriction

    """
    if df is None:
        df = hdju.seg_split_centroid(
            table = hdju.proofreading_neurons_with_gnn_cell_type_fine,
            features = [
                "gnn_cell_type_coarse",
                "gnn_cell_type_fine",
                "cell_type"
            ],
            return_df = True,
        )

    if cell_type_coarse is not None:
        cell_type_coarse = nu.convert_to_array_like(cell_type_coarse)
        df = df.query(f"gnn_cell_type_coarse in {cell_type_coarse}")

    if cell_type_fine is not None:
        cell_type_fine = nu.convert_to_array_like(cell_type_fine)
        df = df.query(f"gnn_cell_type_fine in {cell_type_fine}")


    if radius is not None and center is not None:
        df = pu.restrict_df_to_coordinates_within_radius(
            df,
            name = "centroid",
            center = center,
            radius = radius
        )

    if randomize:
        df = pu.shuffle_df(df)
        

    return df




def plot_soma_dist_distr(df):
    x = "presyn_soma_postsyn_soma_euclid_dist"
    y = "postsyn_skeletal_distance_to_soma"
    sml.hist2D(
        df[x],
        df[y],
    )

    sml.scatter_2D(
        df[x],
        df[y],
        x,
        y
    )

def plotting_edge_df_delta_ori(
    edge_df,
    title_suffix,
    plot_soma_dist = False,
    
    # for plotting histogram
    attribute = "delta_ori_rad",
    interval_attribute = "postsyn_skeletal_distance_to_soma",
    n_bins = 10,
    
    plot_scatter = True,
    plot_histograms_over_intervals = True,
    verbose = True,
    ):

    curr_edge_df = edge_df
    
    if plot_soma_dist:
        plot_soma_dist_distr(curr_edge_df)
        
    if verbose:
        cols = ["presyn_proofreading_method","postsyn_proofreading_method"]
        manual_cell_counts = pu.count_unique_column_values(curr_edge_df[cols],cols)
        display(manual_cell_counts)

        cols = ["presyn_functional_method","postsyn_functional_method"]
        manual_cell_counts = pu.count_unique_column_values(curr_edge_df[cols],cols)
        display(manual_cell_counts)
    
    #print(f"interval_attribute = {interval_attribute}")
    #print(f"attribute = {attribute}")
    # -- doing the actual plotting
    
    if plot_scatter:
        sml.hist2D(
            curr_edge_df[interval_attribute],
            curr_edge_df[attribute],
            #hue="presyn_manual_cell"
        )

    df = curr_edge_df
    column = interval_attribute

    values,bins,n_datapoints,std = pu.bin_df_by_column_stat(
        df,
        column=column,
        func = attribute,
        bins = None,
        n_bins=n_bins,
        equal_depth_bins = True,
        plot=False,
    )
    

    x = (bins[1:] + bins[:-1])/2
    
    
    
    # ------- plotting the prettier graph ---
    

    x = (bins[1:] + bins[:-1])/2

    for func in ["errorbar"]:
        fig,ax = plt.subplots(1,1,figsize=(10,7))
        if func == "errorbar":
            getattr(ax,func)(x/1000,np.array(values),std/np.sqrt(n_datapoints))
        else:
            getattr(ax,func)(x/1000,np.array(values))
        ax.set_title(
            f"{attribute} vs {interval_attribute} " + title_suffix
           )
        ax.set_xlabel(f"{interval_attribute} (um) ")
        ax.set_ylabel(f"Mean {attribute}")
        plt.show()
    #ax.bar(x,values,widths,)
    
    
    # ---- plotting the histogram ----
    if plot_histograms_over_intervals:
        sum_stats,std_stats,bin_stats,n_samples = mu.histograms_over_intervals(
            curr_edge_df,
            attribute = attribute,
            #attribute = "delta_dir_rad",
            #interval_attribute="presyn_soma_postsyn_soma_euclid_dist_xz",
            #interval_attribute="presyn_soma_postsyn_soma_euclid_dist",
            #interval_attribute="presyn_soma_postsyn_soma_skeletal_dist",
            interval_attribute=interval_attribute,
            overlap=0,
            intervals = np.vstack([bins[:-1],bins[1:]]).T,

        )

def restrict_edge_df(
    df,
    postsyn_compartments = None,
    spine_categories = None,
    cell_types = None,
    presyn_cell_types = None,
    postsyn_cell_types = None,
    layers = None,
    presyn_layers = None,
    postsyn_layers = None,
    functional_methods = None,
    proofreading_methods = None,
    presyn_proofreading_methods = None,
    postsyn_proofreading_methods = None,
    visual_areas = None,
    presyn_visual_areas = None,
    postsyn_visual_areas = None,
    
    return_title_suffix = True,
    title_suffix_from_non_None = False,
    
    verbose = False,
    ):
    
    restrictions_non_None = []
    
    if postsyn_compartments is None:
        postsyn_compartments= list(df.postsyn_compartment_fine.unique())
    else:
        restrictions_non_None.append(f"(postsyn_compartment_fine in {list(nu.array_like(postsyn_compartments))})")
        
    if spine_categories is None:
        spine_categories = list(df.postsyn_spine_bouton.unique())
    else:
        restrictions_non_None.append(f"(postsyn_spine_bouton in {list(nu.array_like(spine_categories))})")
        
    if cell_types is None:
        cell_types= list(set(
                list(df.presyn_gnn_cell_type_fine.unique())).union(
                list(df.postsyn_gnn_cell_type_fine.unique()),
        ))
    else:
        restrictions_non_None.append(f"(cell_types in {list(nu.array_like(cell_types))})")
        
    if presyn_cell_types is None:
        presyn_cell_types = cell_types
    else:
        restrictions_non_None.append(f"(presyn_cell_types in {list(nu.array_like(presyn_cell_types))})")
        
    if postsyn_cell_types is None:
        postsyn_cell_types = cell_types
    else:
        restrictions_non_None.append(f"(postsyn_cell_types in {list(nu.array_like(postsyn_cell_types))})")
        
    if layers is None:
        layers= np.union1d(
                list(df.presyn_external_layer.unique()),
                list(df.postsyn_external_layer.unique())
        )
    else:
        restrictions_non_None.append(f"(layers in {list(nu.array_like(layers))})")
        
    if presyn_layers is None:
        presyn_layers= layers
    else:
        restrictions_non_None.append(f"(presyn_layers in {list(nu.array_like(presyn_layers))})")
        
    if postsyn_layers is None:
        postsyn_layers = layers
    else:
        restrictions_non_None.append(f"(postsyn_layers in {list(nu.array_like(postsyn_layers))})")
        
    if functional_methods is None:
        functional_methods= np.union1d(
                list(df.presyn_functional_method.unique()),
                list(df.postsyn_functional_method.unique())
        )
    else:
        restrictions_non_None.append(f"(functional_methods in {list(nu.array_like(functional_methods))})")
        
    if proofreading_methods is None:
        proofreading_methods= np.union1d(
                list(df.presyn_proofreading_method.unique()),
                list(df.postsyn_proofreading_method.unique())
        )
    else:
        restrictions_non_None.append(f"(proofreading_methods in {list(nu.array_like(proofreading_methods))})")
        
    if presyn_proofreading_methods is None:
        presyn_proofreading_methods= proofreading_methods
    else:
        restrictions_non_None.append(f"(presyn_proofreading_methods in {list(nu.array_like(presyn_proofreading_methods))})")
        
    if postsyn_proofreading_methods is None:
        postsyn_proofreading_methods = proofreading_methods
    else:
        restrictions_non_None.append(f"(postsyn_proofreading_methods in {list(nu.array_like(postsyn_proofreading_methods))})")
        
        
    if visual_areas is None:
        visual_areas= np.union1d(
                list(df.presyn_external_visual_area.unique()),
                list(df.postsyn_external_visual_area.unique())
        )
    else:
        restrictions_non_None.append(f"(visual_areas in {list(nu.array_like(visual_areas))})")
        
    if presyn_visual_areas is None:
        presyn_visual_areas= visual_areas
    else:
        restrictions_non_None.append(f"(presyn_visual_areas in {list(nu.array_like(presyn_visual_areas))})")
        
    if postsyn_visual_areas is None:
        postsyn_visual_areas = visual_areas
    else:
        restrictions_non_None.append(f"(postsyn_visual_areas in {list(nu.array_like(postsyn_visual_areas))})")
            

    restrictions = [
        f"(postsyn_compartment_coarse == 'dendrite')",
        f"(postsyn_compartment_fine in {list(nu.array_like(postsyn_compartments))})",
        
        f"(postsyn_spine_bouton in {list(nu.array_like(spine_categories))})",
        
        f"(presyn_gnn_cell_type_fine in {list(nu.array_like(presyn_cell_types))})",
        f"(postsyn_gnn_cell_type_fine in {list(nu.array_like(postsyn_cell_types))})",
        
        f"(presyn_external_layer in {list(nu.array_like(presyn_layers))})",
        f"(postsyn_external_layer in {list(nu.array_like(postsyn_layers))})",

        f"(presyn_functional_method in {list(nu.array_like(functional_methods))})",
        f"(postsyn_functional_method in {list(nu.array_like(functional_methods))})",
        
        f"(presyn_proofreading_method in {list(nu.array_like(presyn_proofreading_methods))})",
        f"(postsyn_proofreading_method in {list(nu.array_like(postsyn_proofreading_methods))})",
        
        f"(presyn_external_visual_area in {list(nu.array_like(presyn_visual_areas))})",
        f"(postsyn_external_visual_area in {list(nu.array_like(postsyn_visual_areas))})",
    ]
    
    curr_edge_df = df.copy()
    for restr in restrictions:
        try:
            curr_edge_df = curr_edge_df.query(restr)
        except:
            if verbose:
                print(f"Failed to apply restriction = {restr}")
        
    
#     restriction_query = pu.restriction_str_from_list(
#         restrictions = restrictions,
#         table_type="pandas"
#     )
    
#     curr_edge_df = df.query(
#         restriction_query
#     )
    
    if title_suffix_from_non_None:
        title_suffix = "\n".join(restrictions_non_None)
    else:
        title_suffix = "\n".join(restrictions)
    #print(f"title_suffix = {title_suffix}")
    if len(title_suffix) > 0:
        title_suffix = f"\n{title_suffix}"

    if verbose:
        print(title_suffix)
        display(curr_edge_df)
    
    if return_title_suffix:
        return curr_edge_df,title_suffix
    else:
        return curr_edge_df
    
def restrict_edge_df_by_cell_type_and_layer(
    df,
    cell_types = None,
    presyn_cell_types = None,
    postsyn_cell_types = None,
    layers = None,
    presyn_layers = None,
    postsyn_layers = None,
    **kwargs
    ):
    """
    Purpose: To plot the delta ori histogram
    for a certain presyn/postsyn type in a dataframe
    """

    return conu.restrict_edge_df(
        df,
        cell_types = cell_types,
        presyn_cell_types = presyn_cell_types,
        postsyn_cell_types = postsyn_cell_types,
        layers = layers,
        presyn_layers = presyn_layers,
        postsyn_layers = postsyn_layers,
        title_suffix_from_non_None = True,
    )
    
def plot_restricted_edge_df_delta_ori(
    edge_df,
    
    # -- for restrictions ---
    postsyn_compartments = None,
    spine_categories = None,
    cell_types = None,
    presyn_cell_types = None,
    postsyn_cell_types = None,
    layers = None,
    presyn_layers = None,
    postsyn_layers = None,
    functional_methods = None,
    proofreading_methods = None,
    presyn_proofreading_methods = None,
    postsyn_proofreading_methods = None,
    visual_areas = None,
    presyn_visual_areas = None,
    postsyn_visual_areas = None,
    
    
    plot_soma_dist = True,
    
    # for plotting histogram
    attribute = "delta_ori_rad",
    interval_attribute = "postsyn_skeletal_distance_to_soma",
    n_bins = 10,
    
    plot_scatter = True,
    plot_histograms_over_intervals = True,
    verbose = True,
    **kwargs
    ):
    """
    Purpose: To do the delta ori 
    analysis given certain requirements
    """
    df = edge_df
    
    curr_edge_df,title_suffix = restrict_edge_df(
        df,
        postsyn_compartments = postsyn_compartments,
        spine_categories = spine_categories,
        cell_types = cell_types,
        presyn_cell_types = presyn_cell_types,
        postsyn_cell_types = postsyn_cell_types,
        layers = layers,
        presyn_layers = presyn_layers,
        postsyn_layers = postsyn_layers,
        functional_methods = functional_methods,
        proofreading_methods = proofreading_methods,
        presyn_proofreading_methods = presyn_proofreading_methods,
        postsyn_proofreading_methods = postsyn_proofreading_methods,
        visual_areas = visual_areas,
        presyn_visual_areas = presyn_visual_areas,
        postsyn_visual_areas = postsyn_visual_areas,
        verbose=verbose,
    )

    plotting_edge_df_delta_ori(
        curr_edge_df,
        title_suffix = title_suffix,
        interval_attribute=interval_attribute,
        plot_histograms_over_intervals=plot_histograms_over_intervals,
        verbose = verbose,
        n_bins=n_bins,
        **kwargs
        )
    
    return curr_edge_df


def plot_functional_connection_from_df(
    df,
    G,
    idx = 0,
    method = "meshafterparty",
    ori_min = None,
    ori_max = None,
    pre_ori_min = None,
    pre_ori_max = None,
    post_ori_min = None,
    post_ori_max = None,
    delta_ori_min = None,
    delta_ori_max = None,
    features_to_print = [
        "synapse_id",
        "postsyn_spine_bouton",
        "presyn_soma_postsyn_soma_euclid_dist_xz",
        "presyn_soma_postsyn_soma_euclid_dist_y_signed",
        "presyn_skeletal_distance_to_soma",
        "postsyn_skeletal_distance_to_soma",
        "presyn_external_layer",
        "postsyn_external_layer",
        "presyn_external_visual_area",
        "postsyn_external_visual_area",
        "presyn_gnn_cell_type_fine",
        "postsyn_gnn_cell_type_fine",

    ],

    functional_features_to_print = [
        "presyn_ori_rad",
        "postsyn_ori_rad",
        "delta_ori_rad",
    ],
    verbose = True,
    ):
    """
    Purpose: Want to visualiz the connections and their 
    delta orientation

    Pseudocode: 
    0) Restrict to orientation range
    1) Restrict to delta range
    2) Use the curent idx to get the current row
    3) Plot the connection
    """
    if verbose:
        print(f"idx = {idx}")


    if pre_ori_min is None:
        pre_ori_min = np.min(df.presyn_ori_rad)

    if pre_ori_max is None:
        pre_ori_max = np.max(df.presyn_ori_rad)

    if post_ori_min is None:
        post_ori_min = np.min(df.postsyn_ori_rad)

    if post_ori_max is None:
        post_ori_max = np.max(df.postsyn_ori_rad)

    if delta_ori_min is None:
        delta_ori_min = np.min(df.delta_ori_rad)

    if delta_ori_max is None:
        delta_ori_max = np.max(df.delta_ori_rad)

    curr_df = pu.restrict_df_from_list(
        df,
        [
            f"presyn_ori_rad >= {pre_ori_min}",
            f"presyn_ori_rad <= {pre_ori_max}",
            f"postsyn_ori_rad >= {post_ori_min}",
            f"postsyn_ori_rad <= {post_ori_max}",
            f"delta_ori_rad >= {delta_ori_min}",
            f"delta_ori_rad <= {delta_ori_max}"

        ]
    )

    curr_dict = curr_df.iloc[idx,:].to_dict()

    # -- printing characteristics we may care about --
    if features_to_print is None:
        features_to_print = []

    if verbose:
        print(f"Edge Attributes:")
        for f in features_to_print:
            print(f"{f}:{curr_dict[f]}")

        print(f"\nFunctional Attributes:")
        for f in functional_features_to_print:
            print(f"{f}:{curr_dict[f]}")

        print(f"\n")

    # -- plotting the connection ---
    return conu.visualize_graph_connections_by_method(
        G,
        segment_ids=[curr_dict["presyn"],curr_dict["postsyn"]],
        #synapse_ids=[curr_dict["synapse_id"]],
        method = method,
        plot_gnn=False,   
    )



def plot_cell_type_pre_post_attribute_hist(
    df,
    cell_type_pairs,
    attribute = "delta_ori_rad",
    
    # for the synthetic control: 
    n_synthetic_control = 5,
    n_samples = None,
    seed = None,
    
    # -- for the plotting 
    bins = 40, 
    verbose = False,
    return_dfs = False,
    ):
    """
    Purpose: To look at the histogram of 
    attributes for different presyn-postsyn
    cell type pairs
    """
    if type(cell_type_pairs) == dict:
        cell_type_pairs= [cell_type_pairs]
        
    dfs_to_return = []

    for restr_dict in cell_type_pairs:
        if verbose:
            print(f"working on {restr_dict}")
        possible_kwargs = ["cell_type","presyn_cell_type","postsyn_cell_type","layer","presyn_layer","postsyn_layer"]
        new_dict = dict()
        for k in restr_dict:
            if k in possible_kwargs:
                new_dict[f"{k}s"] = restr_dict[k]
            else:
                new_dict[k] = restr_dict[k]
        curr_df,title_str = conu.restrict_edge_df_by_cell_type_and_layer(
            df,
            **new_dict
        )
        
        dfs_to_return.append(curr_df.copy())
        
        #return curr_df

        dfs = [curr_df]
        title_suffix = ["",]

        if seed is None:
            seed = np.random.randint(100)

        # could then run a random sampling of the dataframe for a control
        for i in range(n_synthetic_control):


            if n_samples is None:
                n_samples = len(curr_df)

            seed = seed + i

            syn_df = pu.randomly_sample_source_target_df(
                curr_df,
                n_samples = n_samples,
                seed = seed,
            )
            syn_df = ftu.add_on_delta_to_df(
                syn_df,
                ori_name_1='presyn_ori_rad',
                ori_name_2='postsyn_ori_rad',
                dir_name_1='presyn_dir_rad',
                dir_name_2='postsyn_dir_rad',
            )

            dfs.append(syn_df)

            syn_title = f"\nSynthetic Sampling {i}"
            title_suffix.append(syn_title)

        for df_curr,tit_suf in zip(dfs,title_suffix):
            attributes_values = df_curr[attribute].to_numpy()
            plt.hist(
                attributes_values.astype('float'),
                bins=bins
            )
            if len(attributes_values) > 0:
                mean_attribute,std_attribute = np.mean(attributes_values),np.std(attributes_values)
            else:
                mean_attribute,std_attribute = np.nan,np.nan
                
            plt.title(f"Histogram of {attribute}" + title_str + f"\nMean = {mean_attribute:.3f}, std = {std_attribute:.3f}, n_data = {len(attributes_values)}{tit_suf}")
            plt.show()

        print(f"\n\n\n")
        
#     for dd in dfs_to_return:
#         print(dd.presyn.unique().shape)
        
    if return_dfs:
        return dfs_to_return
        
def add_delta_ori_edge_features(
    edge_df,
    ):

    sampled_edge_df = ftu.add_on_delta_to_df(
                    edge_df,
                    ori_name_1='presyn_ori_rad',
                    ori_name_2='postsyn_ori_rad',
                    dir_name_1='presyn_dir_rad',
                    dir_name_2='postsyn_dir_rad',
                )

    dummy_dict = dict(
        postsyn_compartment_coarse = "dendrite",
        postsyn_compartment_fine = "basal",
        presyn_spine_boutuon = "bouton",
        postsyn_spine_bouton = "spine",
    )

    for k,v in dummy_dict.items():
        if k not in sampled_edge_df.columns:
            sampled_edge_df[k] = v

    sampled_edge_df = hdju.add_proofreading_method_labels_to_df(
        sampled_edge_df
    )

    return sampled_edge_df

def neuroglancer_df_from_edge_df(
    G,
    df,
    columns_at_front=(
        "presyn_segment_id",
         "presyn_gnn_cell_type_fine",
        "presyn_external_layer",
        "presyn_external_visual_area",
         "postsyn_segment_id",
         "postsyn_gnn_cell_type_fine",
        "postsyn_external_layer",
        "postsyn_external_visual_area",
        "postsyn_spine_bouton",
         "synapse_id",
        "synapse_x",
        "synapse_y",
        "synapse_z",
    ),
    neuroglancer_column = "neuroglancer",
    verbose = False,
    verbose_cell_type=False,
    suppress_errors = True,
    ):
    """
    Purpose: From a dataframe that is an edge df
    want to generate a spreadsheet with all of the edge features
    and a neuroglancer link for the connection

    Psedocode: 
    1) Turn the dataframe into dictionaries and for each dictionary
    a. generate the neuroglancer link
    b. Add list of dictionaries

    Convert list of dictionaries to dataframe
    """
    st = time.time()
    
    columns_at_front = list(columns_at_front)
    columns_at_front = [neuroglancer_column] + columns_at_front


    df_dicts = pu.df_to_dicts(df)
    for curr_dict in tqdm(df_dicts):
        try:
            ng_link = conu.visualize_graph_connections_by_method(
                G,
                method= "neuroglancer",
                segment_ids = [curr_dict["source"],curr_dict["target"],],
                synapse_ids = [curr_dict["synapse_id"]],
                plot_gnn = False,
                verbose_cell_type=verbose_cell_type,

            )
        except:
            if suppress_errors:
                continue
            else:
                raise Exception("")
        curr_dict["neuroglancer"] = ng_link
        curr_dict["presyn_segment_id"] = hdju.segment_id_and_split_index(curr_dict["source"])[0]
        curr_dict["postsyn_segment_id"] = hdju.segment_id_and_split_index(curr_dict["target"])[0]

    new_df = pd.DataFrame.from_records(df_dicts)

    new_df = pu.order_columns(
        new_df,
        list(columns_at_front),
    )

    new_df = pu.delete_columns(
        new_df,
        ["index"]
    )
    
    if verbose:
        print(f"Total time = {time.time() - st}")
    
    return new_df



def mean_axon_skeletal_length(
    G,
    nodes = None,
    ):
    
    from graph_tools import graph_statistics as gstat
    
    return gstat.node_attribute_mean(
        G = G,
        attribute = "axon_skeletal_length",
        nodes = nodes,
        verbose = False,
    )

def mean_dendrite_skeletal_length(
    G,
    nodes = None,
    ):
    
    from graph_tools import graph_statistics as gstat
    
    return gstat.node_attribute_mean(
        G = G,
        attribute = "dendrite_skeletal_length",
        nodes = nodes,
        verbose = False,
    )

def excitatory_nodes(
    G,
    attriubte = "cell_type",
    verbose = False,
    ):
    return xu.get_nodes_with_attribute_value(
        G,
        attriubte,
        value = "excitatory",
        verbose = verbose
    )

def inhibitory_nodes(
    G,
    attriubte = "cell_type",
    verbose = False,
    ):
    return xu.get_nodes_with_attribute_value(
        G,
        attriubte,
        value = "inhibitory",
        verbose = verbose,
    )


def basic_connectivity_axon_dendrite_stats_from_G(
    G,
    G_lite = None,
    verbose = True,
    verbose_time = True,
    n_samples = 300,
    n_samples_exc = 300,
    graph_functions_kwargs = None,
    graph_functions_G = None,
    graph_functions = None,
    ):
    """
    Psueodocode: Want to compute statistics
    on the graph for the excitatory and inhibitory

    Pseudocode: 
    1) Get the excitatory and inhibitory nodes
    2) 
    """
    from graph_tools import graph_path_utils as gpu
    from graph_tools import graph_statistics as gstat
    
    if G_lite is None:
        G_lite = xu.nodes_edges_only_G(G)

    nodes = list(G.nodes())
    exc_nodes = conu.excitatory_nodes(G,verbose = verbose)
    inh_nodes = conu.inhibitory_nodes(G,verbose=verbose)
    return_no_path_perc = True

    if graph_functions_G is None:
        graph_functions_G = dict(
            shortest_path_exc = G_lite,
            shortest_path_exc_undirected = G_lite
        )
    
    if graph_functions_kwargs is None:
        graph_functions_kwargs = dict(
            shortest_path_distance_samples_mean_from_source = dict(
                n_samples = n_samples,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),

            shortest_path_distance_samples_perc_95_from_source = dict(
                n_samples = n_samples,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),

            shortest_path_distance_samples_mean_from_source_undirected = dict(
                n_samples = n_samples,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),

            shortest_path_distance_samples_perc_95_from_source_undirected = dict(
                n_samples = n_samples,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),

            shortest_path_exc = dict(
                n_samples = n_samples_exc,
                path_nodes = exc_nodes,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),

            shortest_path_exc_undirected = dict(
                n_samples = n_samples_exc,
                path_nodes = exc_nodes,
                verbose = verbose,
                return_no_path_perc=return_no_path_perc,
            ),


        )



    if graph_functions is None:
        graph_functions = [
            xu.n_nodes,
            #xu.n_edges,
            xu.n_edges_in,
            xu.n_edges_out,
            gstat.in_degree_mean,
            gstat.out_degree_mean,
            conu.mean_axon_skeletal_length,
            conu.mean_dendrite_skeletal_length,
            gpu.shortest_path_distance_samples_mean_from_source,
            gpu.shortest_path_distance_samples_mean_from_source_undirected,
            gpu.shortest_path_distance_samples_perc_95_from_source,
            gpu.shortest_path_distance_samples_perc_95_from_source_undirected,
            (gpu.shortest_path_distance_samples_mean_from_source,"shortest_path_exc"),
            (gpu.shortest_path_distance_samples_mean_from_source_undirected,"shortest_path_exc_undirected"),
        ]


    graph_dict = dict()
    for n,name in zip(
        [nodes,exc_nodes,inh_nodes],
        ["all","excitatory","inhibitory"]):
        local_dict = dict()
        for f in graph_functions:
            st = time.time()
            if nu.is_array_like(f,include_tuple=True):
                f,f_name = f
            else:
                f_name = f.__name__

            if verbose:
                print(f"--- Working on {f_name} ----")
            curr_kwargs = graph_functions_kwargs.get(f_name,dict()).copy()
            curr_nodes = curr_kwargs.pop("nodes",n)

            curr_G = graph_functions_G.get(f_name,G)
            local_dict[f_name] = f(curr_G,nodes=curr_nodes,**curr_kwargs)

            if verbose_time:
                print(f"    time for {f_name} = {time.time() - st}")


        graph_dict.update({f"{name}_{k}":v for k,v in local_dict.items()})

    return graph_dict

def add_compartment_syn_flag_columns_to_edge_df(
    df,
    return_columns = False,
    ):
    
    if "compartment" not in df.columns:
        df["compartment"] = pu.combine_columns_as_str(
            df,
            ["postsyn_compartment_coarse","postsyn_compartment_fine"]
        )
    
    non_fine_comps = ["soma","axon","dendrite"]
    comp_dict = dict([(f"n_{comp}_syn",f"(compartment == 'dendrite_{comp}')") if comp not in non_fine_comps else 
        (f"n_{comp}_syn",f"(postsyn_compartment_coarse == '{comp}') and (postsyn_compartment_fine != postsyn_compartment_fine)"
    ) for comp in apu.compartments_to_plot() + ["soma","dendrite"]])

    edge_df_with_comp_cols = pu.new_column_from_name_to_str_func_dict(
        df,
        comp_dict
    )
    
    if return_columns:
        return edge_df_with_comp_cols,list(comp_dict.keys())
    else:
        return edge_df_with_comp_cols


def n_compartment_syn_from_edge_df(
    df
    ):
    """
    Purpose: to get a dataframe
    that maps the source,target edges
    to the number of compartment synapses

    Application: Can be used to append
    to another dataframe
    """

    edge_df_with_comp_cols,flag_cols = add_compartment_syn_flag_columns_to_edge_df(df,return_columns = True,)
    edge_df_with_comp_cols = pu.flatten_row_multi_index(edge_df_with_comp_cols.groupby(["source",'target',])[flag_cols].sum())
    edge_df_with_comp_cols["n_apical_total_syn"] = edge_df_with_comp_cols[[
        f"n_{comp}_syn" for comp in apu.apical_total
    ]].sum(axis = 1)
    return edge_df_with_comp_cols

    
def add_spine_syn_flag_columns_to_edge_df(
    df,
    return_columns = False,
    ):

    comp_dict = dict([(f"n_{sp}_spine_syn",f"(spine_compartment == '{sp}')") 
                      for sp in ['head', 'shaft', 'no_head', 'neck']])
    comp_dict["non_processed"] = f"(spine_compartment != spine_compartment)"

    edge_df_with_comp_cols = pu.new_column_from_name_to_str_func_dict(
        df,
        comp_dict
    )
    
    if return_columns:
        return edge_df_with_comp_cols,list(comp_dict.keys())
    else:
        return edge_df_with_comp_cols

    
def n_spine_syn_from_edge_df(
    df
    ):
    """
    Purpose: to get a dataframe
    that maps the source,target edges
    to the number of compartment synapses

    Application: Can be used to append
    to another dataframe
    """

    edge_df_with_comp_cols,flag_cols = add_spine_syn_flag_columns_to_edge_df(df,return_columns = True,)
    edge_df_with_comp_cols = pu.flatten_row_multi_index(edge_df_with_comp_cols.groupby(["source",'target',])[flag_cols].sum())
    return edge_df_with_comp_cols


        


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
    
       
data_type = "default"
algorithms = None
modules_to_set = [conu]

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






#--- from neurd_packages ---
from . import apical_utils as apu
from . import cell_type_utils as ctu
from . import functional_tuning_utils as ftu
from . import functional_tuning_utils as ftu 
from . import h
from . import microns_volume_utils as mvu
from . import neuron_visualizations as nviz

#--- from machine_learning_tools ---
from machine_learning_tools import seaborn_ml as sml

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk

#--- from python_tools ---
from python_tools import matplotlib_utils as mu
from python_tools import module_utils as modu 
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu
from python_tools import pandas_utils as pu
from python_tools import statistics_visualizations as sviz
from python_tools import string_utils as stru
from python_tools.tqdm_utils import tqdm

from . import connectome_utils as conu