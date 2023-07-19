'''



Notes on proximities: 
- There are some undercounting of n_synapses in the proximity counting if a lot of synapses
because the cancellation distance 5000, but the search for synapses is only 3000 so could have missed some
in that cancellation range and search range

--> but the 





'''
import datajoint as dj
import pandas as pd
from pykdtree.kdtree import KDTree
import time
from python_tools import numpy_dep as np
from python_tools import module_utils as modu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu

def synapse_coordinates_from_df(df):
    return df[
        ["synapse_x_nm","synapse_y_nm","synapse_z_nm"]].to_numpy().astype('float')

def proximity_search_neurons_from_bounding_box(
    segment_id,
    split_index,
    verbose = False,
    buffer = 7_000,
    min_dendrite_skeletal_length = 1_000_000,
    return_dict = False,
    
    ):
    
    prox_table = (
        hdju.proofreading_neurons_minus_volume_edge_table()
        & f"dendrite_skeletal_length > {min_dendrite_skeletal_length}"
    )
    
    nodes = hdju.bbox_intersect_neurons(
            segment_id = segment_id,
            split_index = split_index,
            verbose = verbose,
            return_nodes=True,
            compartment_source="axon",
            compartment_target=None,
            table = prox_table,
            buffer = buffer
        )
    
    if return_dict:
        return dict(
            n_targets = len(nodes[0]),
            targets_segment_id = nodes[0],
            targets_split_index = nodes[1])
    else:
        return nodes

def proximity_search_neurons_from_database(
    segment_id,
    split_index = 0,
    ):
    key = dict(
        segment_id=segment_id,
        split_index = split_index
    )
    return hdju.proximity_search_nodes_from_segment(
        **key,
        verbose = True,
        return_node_names = True
    )

def presyn_proximity_data(
    segment_id,
    split_index,
    plot = False,
    verbose = False,
    ):
    """
    Purpose: Get the presyn proximity information
    before pairwise proximities are computed
    """
    st = time.time()
    #1) computes the synapse presyn side of the connectome
    synapse_pre_df = hdju.segment_id_to_synapse_table_optimized_connectome(
        segment_id=segment_id,
        split_index=split_index,
        syn_type = "presyn",
        coordinates_nm = True,
        return_df=True)
    
    synapse_pre_raw_df = hdju.segment_id_to_synapse_table_optimized(
        segment_id=segment_id,
        synapse_type = "presyn",
        coordinates_nm = True,
        return_df=True)
    
    synapse_pre_raw_coords = pxu.synapse_coordinates_from_df(synapse_pre_raw_df)
    
    synapse_pre_proof_df = hdju.segment_id_to_synapse_table_optimized_proofread(
        segment_id=segment_id,
        synapse_type = "presyn",
        coordinates_nm = True,
        return_df=True)
    
    synapse_pre_proof_coords = pxu.synapse_coordinates_from_df(synapse_pre_proof_df)


    #2) Gets the mesh for later plotting
    if plot:
        mesh = hdju.fetch_proofread_mesh(
            segment_id,
            split_index)
    else:
        mesh = None

    #3) Gets some coordinate for later euclidean distance calculation
    presyn_soma_coord = hdju.soma_nm_coordinate(
        segment_id,
        split_index)

    #4) Fetches neuron object  for calculations
    G_presyn = hdju.graph_obj_from_proof_stage(segment_id,split_index)


    #5) Calculating all of the width,compartments and skeletons
    (presyn_skeleton_array,
    presyn_width_array,
    presyn_comp_array) = nxu.skeleton_width_compartment_arrays_from_G(
        G_presyn,
        compartments=["axon"],
    )

    #5) Fetches the skeleton and starting coordinate of axon 
    #   to be used later for skeletal distance to soma

    presyn_starting_coordinate = nxu.most_upstream_node_on_axon_limb(
        G_presyn,
        return_endpoint_upstream = True
    )

    if plot:
        print(f"Plotting: Presyn axon skeleton")
    (pre_sk_vert,
    pre_sk_edges)= nxu.axon_skeleton(
        G = G_presyn,
        include_path_to_soma = True,
        plot=plot,
        mesh = mesh,

    )
    
    #G_presyn_sk = sk.convert_skeleton_to_graph(pre_sk_vert[pre_sk_edges])
    G_presyn_sk = xu.graph_from_non_unique_vertices_edges(pre_sk_vert,pre_sk_edges)

    if plot:
        print(f"Plotting: Presyn skeleton and starting coordinate")
        nviz.plot_objects(
            scatters=[pre_sk_vert,presyn_starting_coordinate],
            scatters_colors=["red","blue"],
            scatter_size=[0.2,2]
        )
        
    if verbose:
        print(f"Total time for presyn proximity data = {time.time() - st}")

    return dict(
        # for the synapses to be checked if in vicinity
        synapse_pre_df = synapse_pre_df,

        # for attributes for computing proximities
        presyn_skeleton_array=presyn_skeleton_array,
        presyn_width_array=presyn_width_array,
        presyn_comp_array=presyn_comp_array,


        # for skeletal walk distance
        presyn_starting_coordinate=presyn_starting_coordinate,
        pre_sk_vert=pre_sk_vert,
        pre_sk_edges=pre_sk_edges,
        G_presyn_sk=G_presyn_sk,

        # for computing euclidean distance
        presyn_soma_coord =presyn_soma_coord,
        
        # for more features to compute close to
        synapse_pre_raw_coords=synapse_pre_raw_coords,
        synapse_pre_proof_coords=synapse_pre_proof_coords,
        
        #for ploting 
        mesh = mesh
    )


def postsyn_proximity_data(
    segment_id,
    split_index,
    plot = False,
    verbose = False,
    check_starting_coord_match_skeleton = False,
    
    ):

    """
    Purpose: Get the postsyn proximity information
    before pairwise proximities are computed
    """
    segment_id_target = segment_id
    split_index_target = split_index

    st = time.time()
    #2) Gets the mesh for later plotting
    if plot:
        mesh_post = hdju.fetch_proofread_mesh(
            segment_id_target,
            split_index_target
        )
    else:
        mesh_post = None

    #3) Gets some coordinate for later euclidean distance calculation
    postsyn_soma_coord = hdju.soma_nm_coordinate(
        segment_id_target,
        split_index=split_index_target)

    #4) Fetches neuron object  for calculations
    G_postsyn = hdju.graph_obj_from_proof_stage(segment_id_target,split_index_target)

    #4) Calculating all the skeleton arrays, widths, compartments
    (postsyn_skeleton_array,
    postsyn_width_array,
    postsyn_comp_array) = nxu.skeleton_width_compartment_arrays_from_G(
        G_postsyn,
        plot=False
    )

    postsyn_soma_mesh = hdju.fetch_soma_mesh(
        segment_id_target,
        split_index_target,
        plot_soma = False)

    soma_array = np.array(postsyn_soma_mesh.vertices)
    postsyn_skeleton_array = np.vstack([postsyn_skeleton_array,postsyn_soma_mesh.vertices])
    postsyn_width_array = np.hstack([postsyn_width_array,np.repeat([0],len(soma_array))])
    postsyn_comp_array = np.hstack([postsyn_comp_array,np.repeat(["soma"],len(soma_array))])

    if plot:
        print(f"Plotting: postsyn skeleton array")
        ipvu.plot_mesh_with_scatter(
            mesh_post,
            scatter=postsyn_skeleton_array,
        )

    #5) Fetches the skeleton and starting coordinate of axon 
    #   to be used later for skeletal distance to soma

    if plot:
        print(f"Plotting: postsyn skeleton")
    (post_sk_vert,
    post_sk_edges)= nxu.skeleton(
        G = G_postsyn,
        plot=plot,
        mesh = mesh_post,
        )  

    postsyn_starting_coords= nxu.starting_coordinates_all_limbs(
                G_postsyn,
                verbose = False)


    if check_starting_coord_match_skeleton:
        for k in postsyn_starting_coords.reshape(-1,3):
            min_dist = nu.closest_dist_between_coordinates(
                post_sk_vert.reshape(-1,3),
                k.reshape(-1,3),
                return_min=True
            )

            if min_dist > 0:
                raise Exception("")

    #6) Postsyn and spine data
    synapse_post_df = hdju.segment_id_to_synapse_table_optimized_proofread(
        segment_id=segment_id_target,
        split_index=split_index_target,
        syn_type = "postsyn",
        coordinates_nm = True,
        return_df=True)

    synapse_post_coord = synapse_post_df[
        ["synapse_x_nm","synapse_y_nm","synapse_z_nm"]].to_numpy().astype('float')

    if plot:
        print(f"Plotting: Spine Coordinates")
    spine_shaft_coords = nxu.spine_shaft_coordinates(
        G_postsyn,
        verbose = False,
        plot = plot,
        mesh = mesh_post,
    )

    if verbose:
        print(f"Total time for presyn proximity data = {time.time() - st}")

    return_dict = dict(
        # for attributes for computing proximities
        postsyn_skeleton_array = postsyn_skeleton_array,
        postsyn_width_array = postsyn_width_array,
        postsyn_comp_array = postsyn_comp_array,

        # for skeletal walk distance
        post_sk_vert=post_sk_vert,
        post_sk_edges=post_sk_edges,
        postsyn_starting_coords=postsyn_starting_coords,

        # for computing euclidean distance
        postsyn_soma_coord=postsyn_soma_coord,

        # attributes to be collected in the vicinity of contact
        synapse_post_coord=synapse_post_coord,
        spine_shaft_coords=spine_shaft_coords,
        
        #for ploting 
        mesh_post = mesh_post,
        G_postsyn_sk = None,

    )
    
    return return_dict

def proximity_pre_post(
    segment_id_pre,
    segment_id_post,
    split_index_pre=0,
    split_index_post=0,
    
    presyn_prox_data = None,
    postsyn_prox_data = None,
    
    # -- proximity parameters once pre/post data
    max_proximity_dist = 5_000,
    presyn_coordinate_cancel_dist = 10_000,
    max_attribute_dist = 3_000,
    subtract_width_from_euclidean_dist = True,

    plot = False,
    plot_attributes_under_threshold = False,
    plot_proximities = True,
    
    verbose = True,
    verbose_time = False,
    return_df = False,
    
    ):

    """
    Purpose: Will compute the proximity dictionaries
    for a source and target pair of neurons

    Pseudocode: 
    1) Get the presyn information
    2) Get the postsyn information
    3) Run the contact finding loop and save off the 
    results
    
    Example: 
    pxu.example_proximity()
    """
    segment_id = segment_id_pre
    split_index = split_index_pre
    segment_id_target = segment_id_post
    split_index_target = split_index_post
    
    global_time = time.time()
    if presyn_prox_data is None:
        presyn_prox_data = pxu.presyn_proximity_data(
            segment_id = segment_id,
            split_index = split_index,
            plot = plot,
            verbose = verbose
        )

    if postsyn_prox_data is None:
        postsyn_prox_data  = pxu.postsyn_proximity_data(
        segment_id=segment_id_target,
        split_index=split_index_target,
        plot = plot,
        verbose = verbose,
        )
        
    # --- unpacking the data --------
    # presyn data
    mesh = presyn_prox_data['mesh']
    synapse_pre_df = presyn_prox_data['synapse_pre_df']
    presyn_skeleton_array = presyn_prox_data['presyn_skeleton_array']
    presyn_width_array = presyn_prox_data['presyn_width_array']
    presyn_comp_array = presyn_prox_data['presyn_comp_array']
    presyn_starting_coordinate = presyn_prox_data['presyn_starting_coordinate']
    pre_sk_vert = presyn_prox_data['pre_sk_vert']
    pre_sk_edges = presyn_prox_data['pre_sk_edges']
    presyn_soma_coord = presyn_prox_data['presyn_soma_coord']
    G_presyn_sk = presyn_prox_data["G_presyn_sk"]
    
    synapse_pre_raw_coords = presyn_prox_data['synapse_pre_raw_coords']
    synapse_pre_proof_coords = presyn_prox_data["synapse_pre_proof_coords"]
    

    #postsyn data
    mesh_post = postsyn_prox_data['mesh_post']
    postsyn_skeleton_array = postsyn_prox_data['postsyn_skeleton_array']
    postsyn_width_array = postsyn_prox_data['postsyn_width_array']
    postsyn_comp_array = postsyn_prox_data['postsyn_comp_array']
    post_sk_vert = postsyn_prox_data['post_sk_vert']
    post_sk_edges = postsyn_prox_data['post_sk_edges']
    postsyn_starting_coords = postsyn_prox_data['postsyn_starting_coords']
    postsyn_soma_coord = postsyn_prox_data['postsyn_soma_coord']
    synapse_post_coord = postsyn_prox_data['synapse_post_coord']
    spine_shaft_coords = postsyn_prox_data['spine_shaft_coords']
    G_postsyn_sk = postsyn_prox_data["G_postsyn_sk"]
    
    
    
    # ------- The main proximity loop
    """
    Purpose: Find the locations of the axon on dendrite contacts

    Pseudocode:
    Loop until breaks:
    a) query KDTree to find closest dendrite point to axons
    b) If there is one closer than threshold distance save off:
    - coordinate, distance, compartment
    b2) If not then break
    c) Nullify axon pts within a certain distance of closest point
    d) Find if any synase within certain radius of contacts
    """
    
    st = time.time()
    
    dt = time.time()
    synapse_pre_post_ids,synapse_pre_post_coords = hdju.pre_post_synapse_ids_coords_from_connectome(
        segment_id_pre=segment_id_pre,
        segment_id_post=segment_id_post,
        split_index_pre=split_index_pre,
        split_index_post=split_index_post,
        synapse_pre_df=synapse_pre_df,
    )
    
    if verbose_time:
        print(f"  time for pre-post coords: {time.time() - dt}")
        dt = time.time()
        
    
    postsyn_kd = KDTree(postsyn_skeleton_array)

    if verbose_time:
        print(f"  time for kd tree: {time.time() - dt}")
        dt = time.time()

    presyn_skeleton_array_cp = presyn_skeleton_array.copy()
    presyn_index = np.arange(len(presyn_skeleton_array_cp))
    proximities = []

    counter = 0


    if verbose_time:
        print(f"  time for right before loop: {time.time() - dt}")
        dt = time.time()
        
    exceed_max_proximity = False
    while not exceed_max_proximity:
        if verbose:
            print(f"-- Working on iteration {counter} ---")
        #query KDTree to find closest dendrite point to axons
        dist,closest_face = postsyn_kd.query(presyn_skeleton_array_cp)
        
        if verbose_time:
            print(f"  time for postsyn kd query: {time.time() - dt}")
            dt = time.time()

#         if verbose:
#             print(f"Before subtracting width array, proximity closest_dist = {np.min(dist)}")
            
        if subtract_width_from_euclidean_dist:
            dist = dist - postsyn_width_array[closest_face]

        closest_idx = np.argmin(dist)
        closest_dist = dist[closest_idx]
        
#         if verbose:
#             print(f"proximity closest_dist = {closest_dist}")
#             print(f"postsyn_width_array[closest_face] = {np.max(np.abs(postsyn_width_array[closest_face]))}")

        if closest_dist > max_proximity_dist:
            if verbose:
                print(f"Breaking on iteration {counter} because closest distance = {closest_dist} (max {max_proximity_dist})")
            exceed_max_proximity = True
            break

        proximity_presyn = presyn_skeleton_array_cp[closest_idx]

        postsyn_idx = closest_face[closest_idx]
        proximity_postsyn = postsyn_skeleton_array[closest_face[closest_idx]]
        
        if verbose_time:
            print(f"  time closest faces: {time.time() - dt}")
            dt = time.time()
            
        postsyn_compartment = postsyn_comp_array[postsyn_idx]


        # calculating the skeletal walk
        if G_presyn_sk is None:
            #G_presyn_sk = sk.convert_skeleton_to_graph(pre_sk_vert[pre_sk_edges])
            G_presyn_sk = xu.graph_from_non_unique_vertices_edges(pre_sk_vert,pre_sk_edges)
            
        if verbose_time:
            print(f"  time for getting graph obj presyn: {time.time() - dt}")
            dt = time.time()
            
        if (G_postsyn_sk is None) and (postsyn_compartment != "soma"):
            #G_postsyn_sk = sk.convert_skeleton_to_graph(post_sk_vert[post_sk_edges])
            G_postsyn_sk = xu.graph_from_non_unique_vertices_edges(post_sk_vert,post_sk_edges)
            
        
            
        if verbose_time:
            print(f"  time for getting graph obj postsyn: {time.time() - dt}")
            dt = time.time() 
            
        

        presyn_skeletal_walk_dist = sk.shortest_path_between_two_sets_of_skeleton_coordiantes(
            skeleton = None,
            G = G_presyn_sk,
            coordinates_list_1 = proximity_presyn,
            coordinates_list_2 = presyn_starting_coordinate,
            return_path_distance = True
        )
        
        if verbose_time:
            print(f"  time for pre skeletal_walk distance: {time.time() - dt}")
            dt = time.time()

        if postsyn_compartment == "soma":
            postsyn_skeletal_walk_dist = 0
        else:
            postsyn_skeletal_walk_dist = sk.shortest_path_between_two_sets_of_skeleton_coordiantes(
                skeleton = None,
                G = G_postsyn_sk,
                coordinates_list_1 = proximity_postsyn,
                coordinates_list_2 = postsyn_starting_coords,
                return_path_distance = True
            )
        
        if verbose_time:
            print(f"  time for post skeletal walk dists: {time.time() - dt}")
            dt = time.time()

        # --- compute the number of synapses in the vicinity ----
        contact_coord = (proximity_presyn + proximity_postsyn)/2
        if len(synapse_pre_post_coords) > 0:
            syn_pre_post_dists = np.linalg.norm(synapse_pre_post_coords - contact_coord,axis=1)

            syn_pre_post_mask = syn_pre_post_dists <= max_attribute_dist 
            syn_pre_post_dists_under_thresh = syn_pre_post_dists[syn_pre_post_mask]
            syn_pre_post_ids_under_thresh = synapse_pre_post_ids[syn_pre_post_mask]

            if len(syn_pre_post_dists_under_thresh) > 0:
                ord_idx = np.argsort(syn_pre_post_dists_under_thresh)
                syn_pre_post_dists_under_thresh = syn_pre_post_dists_under_thresh[ord_idx]
                syn_pre_post_ids_under_thresh = syn_pre_post_ids_under_thresh[ord_idx]

                synapse_id = syn_pre_post_ids_under_thresh[0]
                synapse_id_dist = syn_pre_post_dists_under_thresh[0]
            else:
                synapse_id = None
                synapse_id_dist = None
                syn_pre_post_ids_under_thresh = []
        else:
            synapse_id = None
            synapse_id_dist = None
            syn_pre_post_ids_under_thresh = []
            
        if verbose_time:
            print(f"  time for number of synapses in vicinity: {time.time() - dt}")
            dt = time.time()

        attr_dict = dict()
        
        for att_name,att_coords in zip(["synapse_post","spine_post","synapse_pre_raw","synapse_pre_proof"],
                                      [synapse_post_coord,spine_shaft_coords,synapse_pre_raw_coords,synapse_pre_proof_coords]):
            if len(att_coords) > 0:
                spine_coord_under_threshold_mask = np.linalg.norm(att_coords - contact_coord,axis=1) <= max_attribute_dist
                spine_coord_under_threshold = att_coords[spine_coord_under_threshold_mask]

                if plot_attributes_under_threshold:
                    print(f"{att_name} under the threshold ({max_attribute_dist}): {len(spine_coord_under_threshold)}")
                    ipvu.plot_mesh_with_scatter(
                        mesh = mesh_post + mesh,
                        scatter = spine_coord_under_threshold,
                        flip_y = True
                    )

                n_spines_post_under_threshold = len(spine_coord_under_threshold)
            else:
                n_spines_post_under_threshold = 0

            attr_dict[f"n_{att_name}"] = n_spines_post_under_threshold
            
            if verbose_time:
                print(f"  time for attribute under threhsold {att_name}: {time.time() - dt}")
                dt = time.time()


        # -- widths --
        width_presyn = presyn_width_array[presyn_index[closest_idx]]
        width_postsyn = postsyn_width_array[postsyn_idx]
        # if found a proximity then compute the statistics of the proximity
        prox_dict = dict(
            prox_id = counter+1,
            #proximity distane
            proximity_dist = closest_dist,
            proximity_dist_non_adjusted = closest_dist + width_postsyn,

            # the coordinates of the proximity
            presyn_proximity_x_nm = proximity_presyn[0],
            presyn_proximity_y_nm = proximity_presyn[1],
            presyn_proximity_z_nm = proximity_presyn[2],

            postsyn_proximity_x_nm = proximity_postsyn[0],
            postsyn_proximity_y_nm = proximity_postsyn[1],
            postsyn_proximity_z_nm = proximity_postsyn[2],

            #compartment
            postsyn_compartment = postsyn_compartment,

            #widths
            presyn_width = width_presyn,
            postsyn_width = width_postsyn,

            # euclidean distance
            presyn_euclidean_distance_to_soma = np.linalg.norm(proximity_presyn-presyn_soma_coord),
            postsyn_euclidean_distance_to_soma = np.linalg.norm(proximity_postsyn-postsyn_soma_coord),

            #skeletal distance
            presyn_skeletal_distance_to_soma = presyn_skeletal_walk_dist,
            postsyn_skeletal_distance_to_soma = postsyn_skeletal_walk_dist,

            # n synapses in the vicinity
            synapse_id = synapse_id,
            synapse_id_dist = synapse_id_dist,
            n_synapses = len(syn_pre_post_ids_under_thresh),

        )

        # get the n postsyn synapes and spines in the vicinity
        prox_dict.update(attr_dict)
        proximities.append(prox_dict)


        #deleting the presyns from the current array
        dist_from_contact = np.linalg.norm(presyn_skeleton_array_cp - proximity_presyn,axis=1)
        keep_mask = dist_from_contact >= presyn_coordinate_cancel_dist
        presyn_skeleton_array_cp = presyn_skeleton_array_cp[keep_mask]
        presyn_index = presyn_index[keep_mask]

        counter += 1

    if verbose:
        print(f"Time for proximity loop: {time.time() - st}")
    columns = ['prox_id',
            'proximity_dist',
            'proximity_dist_non_adjusted',
            'presyn_proximity_x_nm',
            'presyn_proximity_y_nm',
            'presyn_proximity_z_nm',
            'postsyn_proximity_x_nm',
            'postsyn_proximity_y_nm',
            'postsyn_proximity_z_nm',
            'postsyn_compartment',
            'presyn_width',
            'postsyn_width',
            'presyn_euclidean_distance_to_soma',
            'postsyn_euclidean_distance_to_soma',
            'presyn_skeletal_distance_to_soma',
            'postsyn_skeletal_distance_to_soma',
            'synapse_id',
            'synapse_id_dist',
            'n_synapses',
            'n_synapse_post',
            'n_spine_post',]
    
    if verbose:
        print(f"Time for whole proximity func = {time.time() - global_time}")
    
    if plot_proximities:
        if mesh is None:
            mesh = hdju.fetch_proofread_mesh(
                segment_id,
                split_index
            )

        if mesh_post is None:
            mesh_post = hdju.fetch_proofread_mesh(
                segment_id_target,
                split_index_target,
            )

        proximity_coords = np.array([np.array([k["presyn_proximity_x_nm"],k["presyn_proximity_y_nm"],k["presyn_proximity_z_nm"]])
                                           for k in proximities])
        nviz.plot_objects(
            meshes=[mesh,mesh_post],
            meshes_colors=["red","blue"],
            scatters=[proximity_coords],
            scatter_size=0.3
        )
        
    if verbose_time:
        print(f"  Plotting proximities: {time.time() - dt}")
        dt = time.time()
        
    if return_df:
        if len(proximities) == 0:
            df = pd.DataFrame(columns=columns)
            #df.columns = columns
        else:
            df = pd.DataFrame.from_records(proximities)
            
        return df
    else:
        return proximities
    
def example_proximity(
    verbose = True,
    plot = True,
    return_df = True):
    segment_id_pre = 864691136723442173
    split_index_pre = 0

    segment_id_post = 864691136422863407
    split_index_post = 0
    
    

    presyn_prox_data = pxu.presyn_proximity_data(
        segment_id = segment_id_pre,
        split_index = split_index_pre,
        plot = plot,
        verbose = verbose
    )

    return pxu.proximity_pre_post(
        segment_id_pre = segment_id_pre,
        split_index_pre = split_index_pre,

        segment_id_post = segment_id_post,
        split_index_post = split_index_post,

        presyn_prox_data = presyn_prox_data,
        postsyn_prox_data = None,

        plot_proximities = plot,
        verbose = True,
        return_df = return_df
    )

# ------------- Setting up parameters -----------

# -- default
attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling,
    hdju = mvu.data_interface
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
    hdju = hvu.data_interface
)
global_parameters_dict_h01 = dict()
    
       
# data_type = "default"
# algorithms = None
# modules_to_set = [pxu]

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
from . import neuron_visualizations as nviz

#--- from neuron_morphology_tools ---
from neuron_morphology_tools import neuron_nx_utils as nxu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk

#--- from python_tools ---
from python_tools import ipyvolume_utils as ipvu
from python_tools import module_utils as modu 
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu

from . import proximity_utils as pxu