
import copy
from copy import deepcopy
import itertools
from importlib import reload
import matplotlib.pyplot as plt
from meshparty import trimesh_io

import networkx as nx
from pykdtree.kdtree import KDTree
import time
import trimesh
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from datasci_tools import data_struct_utils as dsu

#importing at the bottom so don't get any conflicts

#for meshparty preprocessing





process_version = 10 #no skeleton jumping hopefully

#from neuron_utils import *



# else:
#     combine_close_skeleton_nodes_threshold_meshparty_axon = 1300
#     filter_end_node_length_meshparty_axon = 1500
#     filter_end_node_length_axon = 1500
#     invalidation_d_axon = 2000
#     smooth_neighborhood_axon = 1
    
    



min_distance_threshold = 0.00001

#--------------- default arguments to use ----------#
def plot_correspondence(
    mesh,
    correspondence,
    idx_to_show = None,
    submesh_from_face_idx = True,
    verbose = True,
    ):
    """
    Purpose: Want to plot mesh correspondence first pass

    Pseudocode: 
    For each entry:
    1) Plot mesh (from idx)
    2) plot skeleton
    """

    if idx_to_show is None:
        idx_to_show = list(correspondence.keys())

    idx_to_show = nu.array_like(idx_to_show)

    for k,v in correspondence.items():
        if k not in idx_to_show:
            continue

        try:
            submesh_idx = v["correspondence_face_idx"]
        except:
            submesh_idx = v["branch_face_idx"]
        subskeleton = v["branch_skeleton"]
        if verbose:
            print(f"For branch {k}: # of mesh faces = {len(submesh_idx)}, skeleton length = {sk.calculate_skeleton_distance(subskeleton)}")
        if submesh_from_face_idx:
            submesh = mesh.submesh([submesh_idx],append=True)
        else:
            try:
                submesh = v["correspondence_mesh"]
            except:
                submesh = v["branch_mesh"]

        nviz.plot_objects(
            mesh,
            subskeleton,
            meshes = [submesh],
            meshes_colors = ["red"],
            buffer = 0,
        )
        

def plot_correspondence_on_single_mesh(
    mesh,
    correspondence,
    ):
    """
    Purpose: To plot the correspondence dict
    once a 1 to 1 was generated

    """

    try:
        meshes = [k["branch_mesh"] for k in correspondence.values()]
    except:
        meshes = [k["correspondence_mesh"] for k in correspondence.values()]
        
    skeletons = [k["branch_skeleton"] for k in correspondence.values()]
    colors = mu.generate_non_randon_named_color_list(len(meshes))

    nviz.plot_objects(
        meshes = meshes,
        meshes_colors=colors,
        skeletons=skeletons,
        skeletons_colors=colors,
        buffer = 0,
    )
    
    
def mesh_correspondence_first_pass(
    mesh,
    skeleton=None,
    skeleton_branches=None,
    distance_by_mesh_center=True,
    remove_inside_pieces_threshold = 0,
    skeleton_segment_width = 1000,
    initial_distance_threshold = 3000,
    skeletal_buffer = 100,
    backup_distance_threshold = 6000,
    backup_skeletal_buffer = 300,
    connectivity="edges",
    plot = False,
    ):
    """
    Will come up with the mesh correspondences for all of the skeleton
    branches: where there can be overlaps and empty faces
    
    """
    curr_limb_mesh = mesh
    curr_limb_sk = skeleton
    
    if remove_inside_pieces_threshold > 0:
        curr_limb_mesh_indices = tu.remove_mesh_interior(curr_limb_mesh,
                                                 size_threshold_to_remove=remove_inside_pieces_threshold,
                                                 try_hole_close=False,
                                                 return_face_indices=True,
                                                )
        curr_limb_mesh = curr_limb_mesh.submesh([curr_limb_mesh_indices],append=True,repair=False)
    else:
        curr_limb_mesh_indices = np.arange(len(curr_limb_mesh.faces))
    
    if skeleton_branches is None:
        if skeleton is None:
            raise Exception("Both skeleton and skeleton_branches is None")
        curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches
    else:
        curr_limb_branches_sk_uneven = skeleton_branches 

    #Doing the limb correspondence for all of the branches of the skeleton
    local_correspondence = dict()
    for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
        local_correspondence[j] = dict()

        
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    distance_threshold = initial_distance_threshold,
                                    buffer = skeletal_buffer,
                                                                connectivity=connectivity)
        if len(returned_data) == 0:
            print("Got nothing from first pass so expanding the mesh correspondnece parameters ")
            returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    buffer=backup_skeletal_buffer,
                                     distance_threshold=backup_distance_threshold,
                                    return_closest_face_on_empty=True,
                                        connectivity=connectivity)
            
        # Need to just pick the closest face is still didn't get anything
        
        # ------ 12/3 Addition: Account for correspondence that does not work so just picking the closest face
        curr_branch_face_correspondence, width_from_skeleton = returned_data
        
            
#             print(f"curr_branch_sk.shape = {curr_branch_sk.shape}")
#             np.savez("saved_skeleton_branch.npz",curr_branch_sk=curr_branch_sk)
#             tu.write_neuron_off(curr_limb_mesh,"curr_limb_mesh.off")
#             #print(f"returned_data = {returned_data}")
#             raise Exception(f"The output from mesh_correspondence_adaptive_distance was nothing: curr_branch_face_correspondence")


        if len(curr_branch_face_correspondence) > 0:
            curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True,repair=False)
        else:
            curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))


        local_correspondence[j]["branch_skeleton"] = curr_branch_sk
        local_correspondence[j]["correspondence_mesh"] = curr_submesh
        local_correspondence[j]["correspondence_face_idx"] = curr_limb_mesh_indices[curr_branch_face_correspondence]
        local_correspondence[j]["width_from_skeleton"] = width_from_skeleton
        
        
    if plot:
        plot_correspondence(mesh,local_correspondence)
    return local_correspondence



def check_skeletonization_and_decomp(
    skeleton,
    local_correspondence):
    """
    Purpose: To check that the decomposition and skeletonization went well
    
    
    """
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    cleaned_branch = skeleton
    empty_submeshes = []

    print(f"Limb decomposed into {len(local_correspondence)} branches")

    #get all of the skeletons and make sure that they from a connected component
    divided_branches = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence]
    divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                    sk.stack_skeletons(divided_branches))

    divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

    cleaned_limb_skeleton = cleaned_branch
    print(f"divided_skeleton_graph_recovered = {divided_skeleton_graph_recovered.shape} and \n"
          f"current_mesh_data[0]['branch_skeletons_cleaned'].shape = {cleaned_limb_skeleton.shape}\n")
    if divided_skeleton_graph_recovered.shape != cleaned_limb_skeleton.shape:
        print(f"****divided_skeleton_graph_recovered and cleaned_limb_skeleton shapes not match: "
                        f"{divided_skeleton_graph_recovered.shape} vs. {cleaned_limb_skeleton.shape} *****")


    #check that it is all one component
    divided_skeleton_graph_n_comp = nx.number_connected_components(divided_skeleton_graph)
    print(f"Number of connected components in deocmposed recovered graph = {divided_skeleton_graph_n_comp}")

    cleaned_limb_skeleton_graph = sk.convert_skeleton_to_graph(cleaned_limb_skeleton)
    cleaned_limb_skeleton_graph_n_comp = nx.number_connected_components(cleaned_limb_skeleton_graph)
    print(f"Number of connected components in cleaned skeleton graph= {cleaned_limb_skeleton_graph_n_comp}")

    if divided_skeleton_graph_n_comp > 1 or cleaned_limb_skeleton_graph_n_comp > 1:
        raise Exception(f"One of the decompose_skeletons or cleaned skeletons was not just one component : {divided_skeleton_graph_n_comp,cleaned_limb_skeleton_graph_n_comp}")

    #check that when we downsample it is not one component:
    curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


    con_comp = list(nx.connected_components(curr_sk_graph_debug))
    if len(con_comp) > 1:
        raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")
    else:
        print(f"The downsampled branches number of connected components = {len(con_comp)}")


    for j,v in local_correspondence.items():
        if len(v["correspondence_mesh"].faces) == 0:
            empty_submeshes.append(j)

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        

        
        
def correspondence_1_to_1(
    mesh,
    local_correspondence,
    curr_limb_endpoints_must_keep=None,
    curr_soma_to_piece_touching_vertices=None,
    must_keep_labels=dict(),
    fill_to_soma_border=True,
    plot = False,
                    ):
    """
    Will Fix the 1-to-1 Correspondence of the mesh
    correspondence for the limbs and make sure that the
    endpoints that are designated as touching the soma then 
    make sure the mesh correspondnece reaches the soma limb border
    
    has an optional argument must_keep_labels that will allow you to specify some labels that are a must keep
    
    """
    
    if len(tu.split(mesh)[0])>1:
        su.compressed_pickle(mesh,"mesh")
        raise Exception("Mesh passed to correspondence_1_to_1 is not just one mesh")
    
    mesh_start_time = time.time()
    print(f"\n\n--- Working on 1-to-1 correspondence-----")

    #geting the current limb mesh

    no_missing_labels = list(local_correspondence.keys()) #counts the number of divided branches which should be the total number of labels
    curr_limb_mesh = mesh

    #set up the face dictionary
    face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

    for j,branch_piece in local_correspondence.items():
        curr_faces_corresponded = branch_piece["correspondence_face_idx"]

        for c in curr_faces_corresponded:
            face_lookup[c].append(j)

    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
    print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")

    if len(original_labels) != len(no_missing_labels):
        raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

    if max(original_labels) + 1 > len(original_labels):
        raise Exception("There are some missing labels in the initial labeling")



    #here is where can call the function that resolves the face labels
    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(
                     curr_limb_mesh = curr_limb_mesh,
                     face_lookup=face_lookup,
                     no_missing_labels = list(original_labels),
                    must_keep_labels=must_keep_labels,
                    branch_skeletons = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence.keys()],
    )

    """  9/17 Addition: Will make sure that the desired starting node is touching the soma border """
    """
    Pseudocode:
    For each soma it is touching
    0) Get the soma border
    1) Find the label_to_expand based on the starting coordinate
    a. Get the starting coordinate

    soma_to_piece_touching_vertices=None
    endpoints_must_keep

    """

    #curr_limb_endpoints_must_keep --> stores the endpoints that should be connected to the soma
    #curr_soma_to_piece_touching_vertices --> maps soma to  a list of grouped touching vertices

    if fill_to_soma_border:
        if (not curr_limb_endpoints_must_keep is None) and (not curr_soma_to_piece_touching_vertices is None):
            for sm,soma_border_list in curr_soma_to_piece_touching_vertices.items():
                for curr_soma_border,st_coord in zip(soma_border_list,curr_limb_endpoints_must_keep[sm]):

                    #1) Find the label_to_expand based on the starting coordinate
                    divided_branches = [v["branch_skeleton"] for v in local_correspondence.values()]
                    #print(f"st_coord = {st_coord}")
                    label_to_expand = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_branches,
                                                                                       current_coordinate=st_coord)[0]


                    face_coloring_copy = cu.waterfill_starting_label_to_soma_border(curr_limb_mesh,
                                                       border_vertices=curr_soma_border,
                                                        label_to_expand=label_to_expand,
                                                       total_face_labels=face_coloring_copy,
                                                       print_flag=True)


    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

    #-- check that all the split mesh pieces are one component --#
    local_correspondence_revised = deepcopy(local_correspondence)
    #save off the new data as branch mesh
    for k in local_correspondence_revised.keys():
        local_correspondence_revised[k]["branch_mesh"] = divided_submeshes[k]
        local_correspondence_revised[k]["branch_face_idx"] = divided_submeshes_idx[k]

        #clean the limb correspondence that we do not need
        del local_correspondence_revised[k]["correspondence_mesh"]
        del local_correspondence_revised[k]["correspondence_face_idx"]
    
    if plot:
        plot_correspondence_on_single_mesh(mesh,local_correspondence_revised)
    
    return local_correspondence_revised



def filter_soma_touching_vertices_dict_by_mesh(mesh,
                                              curr_piece_to_soma_touching_vertices,
                                              verbose=True):
    """
    Purpose: Will take the soma to touching vertics
    and filter it for only those that touch the particular mesh piece

    Pseudocode:
    1) Build a KDTree of the mesh
    2) Create an output dictionary to store the filtered soma touching vertices
    For the original soma touching vertices, iterating through all the somas
        For each soma_touching list:
            Query the mesh KDTree and only keep the coordinates whose distance is equal to 0

    If empty dictionary then return None? (have option for this)
    
    Ex: 
    return_value = filter_soma_touching_vertices_dict_by_mesh(
    mesh = mesh_pieces_for_MAP[0],
    curr_piece_to_soma_touching_vertices = piece_to_soma_touching_vertices[1]
    )

    """
    
    if curr_piece_to_soma_touching_vertices is None:
        if verbose:
            print("In filter_soma_touching_vertices_dict_by_mesh: curr_piece_to_soma_touching_vertices was None so returning none")
        return None

    #1) Build a KDTree of the mesh
    curr_mesh_tree = KDTree(mesh.vertices)

    #2) Create an output dictionary to store the filtered soma touching vertices
    output_soma_touching_vertices = dict()

    for sm_idx,border_verts_list in curr_piece_to_soma_touching_vertices.items():
        for b_verts in border_verts_list:
            dist,closest_nodes = curr_mesh_tree.query(b_verts)
            match_verts = b_verts[dist==0]
            if len(match_verts)>0:
                if sm_idx not in output_soma_touching_vertices.keys():
                    output_soma_touching_vertices[sm_idx] = []
                output_soma_touching_vertices[sm_idx].append(match_verts)
    if len(output_soma_touching_vertices) == 0:
        return None
    else:
        return output_soma_touching_vertices
    
    
    
# ----------------- When refactoring the limb decomposition function ------ #

def find_if_stitch_point_on_end_or_branch(matched_branches_skeletons,
                                                              stitch_coordinate,
                                                              verbose=False):
                    
                    
                        # Step A: Find if stitch point is on endpt/branch point or in middle
                        stitch_point_on_end_or_branch = False
                        if len(matched_branches_skeletons) == 0:
                            raise Exception("No matching branches found for soma extending point")
                        elif len(matched_branches_skeletons)>1:
                            if verbose:
                                print(f"Multiple Branches for MP soma extending connection point {matched_branches_skeletons.shape}")
                            stitch_point_on_end_or_branch = True
                        else:# len(match_sk_branches)==1:
                            if verbose:
                                print(f"Only one Branch for MP soma Extending connection point {matched_branches_skeletons.shape}")
                            if len(nu.matching_rows(sk.find_branch_endpoints(matched_branches_skeletons[0]),
                                                    stitch_coordinate))>0:
                                stitch_point_on_end_or_branch =True

                        return stitch_point_on_end_or_branch
                    

                    




def closest_dist_from_floating_mesh_to_skeleton(
    skeleton,
    floating_mesh,
    verbose = True,
    plot_floating_mesh = False,
    plot_closest_coordinate = False,
    ):
    
    """
    Purpose: To see what the closest distance
    for a floating mesh would be for a given skeleton
    """
    
    curr_limb = pre.preprocess_limb(
        mesh = floating_mesh,
        return_concept_network=False,
        error_on_no_starting_coordinates=False,
    )


    float_sk = sk.stack_skeletons([k["branch_skeleton"] 
                        for k in curr_limb.values()])
    float_sk_endpts = sk.find_skeleton_endpoint_coordinates(float_sk)
    
    if plot_floating_mesh:
        print(f"plot_floating_mesh decomp")
        nviz.plot_objects(
            floating_mesh,
            skeletons=[float_sk],
            scatters=[float_sk_endpts],
            scatter_size=1
        )

    main_sk_coordinates = skeleton.reshape(-1,3)
    limb_kd = KDTree(main_sk_coordinates)
    dist,idx = limb_kd.query(float_sk_endpts)
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]
    closest_coord = main_sk_coordinates[idx[min_idx]]

    if verbose:
        print(f"min_dist = {min_dist}, closest_coord = {closest_coord}")
        
    if plot_closest_coordinate:
        print(f"")
        nviz.plot_objects(
            floating_mesh,
            skeletons=[skeleton],
            scatters=[float_sk_endpts[min_idx],
                     closest_coord],
            scatter_size=1,
            scatters_colors=["red","blue"]
        )
        
    return min_dist

def attach_floating_pieces_to_limb_correspondence(
        limb_correspondence,
        floating_meshes,
        floating_piece_face_threshold = None,
        max_stitch_distance=None,
        distance_to_move_point_threshold = 4000,
    
        verbose = False,
        excluded_node_coordinates=np.array([]),
        filter_end_node_length = None,
        filter_end_node_length_meshparty = 1000,
        use_adaptive_invalidation_d = None,
        axon_width_preprocess_limb_max = None,
        limb_remove_mesh_interior_face_threshold = None,
        error_on_bad_cgal_return = False,
        max_stitch_distance_CGAL=None,
        size_threshold_MAP_stitch = None,
        #invalidation_d = 2000,
    
        mp_only_revised_invalidation_d = None,
        
    **kwargs):
    
    if max_stitch_distance_CGAL is None:
        max_stitch_distance_CGAL = max_stitch_distance_CGAL_global
        
    if max_stitch_distance is None:
        max_stitch_distance = max_stitch_distance_global
        
    if floating_piece_face_threshold is None:
        floating_piece_face_threshold = floating_piece_face_threshold_global
        
    if filter_end_node_length is None:
        filter_end_node_length = filter_end_node_length_global
        
    if axon_width_preprocess_limb_max is None:
        axon_width_preprocess_limb_max = axon_width_preprocess_limb_max_global
        
    if limb_remove_mesh_interior_face_threshold is None:
        limb_remove_mesh_interior_face_threshold = limb_remove_mesh_interior_face_threshold_global
        
    if max_stitch_distance_CGAL is None:
        max_stitch_distance_CGAL = max_stitch_distance_CGAL_global
        
    if size_threshold_MAP_stitch is None:
        size_threshold_MAP_stitch = size_threshold_MAP_stitch_global
        
    if use_adaptive_invalidation_d is None:
        use_adaptive_invalidation_d  = use_adaptive_invalidation_d_floating_global
        
    if mp_only_revised_invalidation_d is None:
        mp_only_revised_invalidation_d = mp_only_revised_invalidation_d_global
    
    
    """
    Purpose: To take a limb correspondence and add on the floating pieces
    that are significant and close enough to a limb

    Pseudocode:
    0) Filter the floating pieces for only those above certain face count
    1) Run all significant floating pieces through preprocess_limb
    2) Get all full skeleton endpoints (degree 1) for all floating pieces


    Start loop until all floating pieces have been added
    a) Get full skeletons of limbs for all limbs in limb correspondence
    b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
    endpoints and all skeleton points of limbs
    c) Find the floating piece that has the closest distance
    --> winning piece

    For the winning piece
    d) Get the closest coordinate on the matching limb
    e) Try and move closest coordinate to an endpoint or high degree node
    f) Find the branch on the main limb that corresponds to the stitch point
    g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
    AKA stitch_point_on_end_or_branch
    h) Find the branch on the floating limb where the closest end point is

    At this point have
    - main limb stitch point and branches (and whether not splitting will be required)  [like MAP]
    - floating limb stitch point and branch [like MP]

    Stitching process:
    i) if not stitch_point_on_end_or_branch
    - cut the main limb branch where stitch is
    - do mesh correspondence with the new stitches
    - (just give the both the same old width)
    - replace the old entry in the limb corresondence with one of the new skeleton cuts
    and add on the other skeletons cuts to the end

    j) Add a skeletal segment from floating limb stitch point to main limb stitch point
    k) Add the floating limb branches to the end of the limb correspondence
    l) Marks the floating piece as processed


    """
    limb_correspondence_cp = limb_correspondence
    non_soma_touching_meshes = floating_meshes
    floating_limbs_above_threshold = [k for k in non_soma_touching_meshes if len(k.faces)>floating_piece_face_threshold]

    #1) Run all significant floating pieces through preprocess_limb
    
    debug_corr = True
    if debug_corr:
        print(f"Starting the floating pieces preprocessing")
    
    
    #with su.suppress_stdout_stderr():
    with su.suppress_stdout_stderr() if not debug_corr else su.dummy_context_mgr():
        
        floating_limbs_correspondence = []
        for j,k in enumerate(floating_limbs_above_threshold):
            if debug_corr:
                print(f"Floating {j}: {k}")
                st_time = time.time()
            
            curr_corr = preprocess_limb(mesh=k,
                           soma_touching_vertices_dict = None,
                           return_concept_network = False, 
                            error_on_no_starting_coordinates=False,
                            filter_end_node_length = filter_end_node_length,
                            filter_end_node_length_meshparty = filter_end_node_length_meshparty,
                            use_adaptive_invalidation_d = use_adaptive_invalidation_d,
                            axon_width_preprocess_limb_max = axon_width_preprocess_limb_max,
                            remove_mesh_interior_face_threshold = limb_remove_mesh_interior_face_threshold,
                            error_on_bad_cgal_return=error_on_bad_cgal_return,
                            max_stitch_distance_CGAL=max_stitch_distance_CGAL,
                            size_threshold_MAP = size_threshold_MAP_stitch,
                            #invalidation_d=invalidation_d,
                            
                            mp_only_revised_invalidation_d=mp_only_revised_invalidation_d,
                            **kwargs,
                           )
                
            floating_limbs_correspondence.append(curr_corr)
            
            if debug_corr:
                #if len(k.faces) == 129010:
                nviz.plot_limb_correspondence(curr_corr)
            
            if debug_corr:
                print(f"--> time = {time.time() - st_time}")
                st_time = time.time()
            
#         floating_limbs_correspondence = [ preprocess_limb(mesh=k,
#                            soma_touching_vertices_dict = None,
#                            return_concept_network = False, 
#                             error_on_no_starting_coordinates=False,
#                                                           **kwargs,
#                            )  for k in floating_limbs_above_threshold]

    #2) Get all full skeleton endpoints (degree 1) for all floating pieces
    floating_limbs_skeleton = [sk.stack_skeletons([k["branch_skeleton"] for k in l_c.values()]) for l_c in floating_limbs_correspondence]
    floating_limbs_skeleton_endpoints = [sk.find_skeleton_endpoint_coordinates(k) for k in floating_limbs_skeleton]
    
#     su.compressed_pickle(floating_limbs_skeleton,"floating_limbs_skeleton")
#     su.compressed_pickle(floating_limbs_skeleton_endpoints,"floating_limbs_skeleton_endpoints")
#     raise Exception("")
    

#     nviz.plot_objects(skeletons=floating_limb_skeletons,
#                      scatters=floating_limbs_skeleton_endpoints,
#                      scatter_size=1)

    floating_limbs_to_process = np.arange(0,len(floating_limbs_skeleton))

    #Start loop until all floating pieces have been added
    while len(floating_limbs_to_process)>0:

        #a) Get full skeletons of limbs for all limbs in limb correspondence
        main_limb_skeletons = []
        for main_idx in np.sort(list(limb_correspondence_cp.keys())):
            main_limb_skeletons.append(sk.stack_skeletons([k["branch_skeleton"] for k in limb_correspondence_cp[main_idx].values()]))

        #b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
        #endpoints and all skeleton points of limbs 
        floating_piece_min_distance_all_main_limbs = dict([(float_idx,[]) for float_idx in floating_limbs_to_process])
        for main_idx,main_limb_sk in enumerate(main_limb_skeletons):

            main_skeleton_coordinates = sk.skeleton_unique_coordinates(main_limb_sk)
            main_kdtree = KDTree(main_skeleton_coordinates)

            for float_idx in floating_piece_min_distance_all_main_limbs.keys():

                dist,closest_node = main_kdtree.query(floating_limbs_skeleton_endpoints[float_idx])
                min_dist_idx = np.argmin(dist)
                min_dist = dist[min_dist_idx]
                min_dist_closest_node = main_skeleton_coordinates[closest_node[min_dist_idx]]
                floating_piece_min_distance_all_main_limbs[float_idx].append([min_dist,min_dist_closest_node,floating_limbs_skeleton_endpoints[float_idx][min_dist_idx]])



        winning_float = -1
        winning_float_match_main_limb = -1
        main_limb_stitch_point = None
        floating_limb_stitch_point = None
        winning_float_dist = np.inf


        #c) Find the floating piece that has the closest distance
        #--> winning piece

        #For the winning piece
        #d) Get the closest coordinate on the matching limb

        for f_idx,dist_data in floating_piece_min_distance_all_main_limbs.items():

            dist_data_array = np.array(dist_data)
            closest_main_limb = np.argmin(dist_data_array[:,0])
            closest_main_dist = dist_data_array[closest_main_limb][0]

            if closest_main_dist < winning_float_dist:

                winning_float = f_idx
                winning_float_match_main_limb = closest_main_limb
                winning_float_dist = closest_main_dist
                main_limb_stitch_point = dist_data_array[closest_main_limb][1]
                floating_limb_stitch_point = dist_data_array[closest_main_limb][2]

        winning_main_skeleton = main_limb_skeletons[winning_float_match_main_limb]
        winning_floating_correspondence = floating_limbs_correspondence[winning_float]

        if verbose:
            print(f"winning_float = {winning_float}")
            print(f"winning_float_match_main_limb = {winning_float_match_main_limb}")
            print(f"winning_float_dist = {winning_float_dist}")
            print(f"main_limb_stitch_point = {main_limb_stitch_point}")
            print(f"floating_limb_stitch_point = {floating_limb_stitch_point}")
            
        #print(f"winning_floating_correspondence = {winning_floating_correspondence}")
        
        #print(f"winning_floating_correspondence = {winning_floating_correspondence['branch_mesh']}")

        
        if winning_float_dist > max_stitch_distance:
            print(f"The closest float distance was {winning_float_dist} which was greater than the maximum stitch distance {max_stitch_distance}\n"
                 " --> so ending the floating mesh stitch processs")
            
#             su.compressed_pickle(main_limb_skeletons,"main_limb_skeletons")
#             su.compressed_pickle(floating_limbs_skeleton_endpoints,"floating_limbs_skeleton_endpoints")
#             su.compressed_pickle(floating_piece_min_distance_all_main_limbs,"floating_piece_min_distance_all_main_limbs")
#             su.compressed_pickle(limb_correspondence_cp,'limb_correspondence_cp')
#             raise Exception("Done stitching")
            
            return limb_correspondence_cp
        else:
            
            
            #e) Try and move closest coordinate to an endpoint or high degree node

            main_limb_stitch_point,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                                skeleton=winning_main_skeleton,
                                                                coordinate=main_limb_stitch_point,
                                                                distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                                verbose=verbose,
                                                                consider_high_degree_nodes=True,
                                                                excluded_node_coordinates=excluded_node_coordinates

                                                                )
            if verbose:
                print(f"Status of Main limb stitch point moved = {change_status}")

        #     #checking that match was right
        #     nviz.plot_objects(meshes=[floating_limbs_above_threshold[winning_float],current_mesh_data[0]["branch_meshes"][winning_float_match_main_limb]],
        #                   meshes_colors=["red","aqua"],
        #                 skeletons=[floating_limbs_skeleton[winning_float],main_limb_skeletons[winning_float_match_main_limb]],
        #                  skeletons_colors=["red","aqua"],
        #                  scatters=[floating_limb_stitch_point.reshape(-1,3),main_limb_stitch_point.reshape(-1,3)],
        #                  scatters_colors=["red","aqua"])


            #f) Find the branch on the main limb that corresponds to the stitch point
            main_limb_branches = np.array([k["branch_skeleton"] for k in limb_correspondence_cp[winning_float_match_main_limb].values()])
            match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(main_limb_branches,
                                current_coordinate=main_limb_stitch_point)

            #g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
            #AKA stitch_point_on_end_or_branch
            stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                                                                    matched_branches_skeletons= main_limb_branches[match_sk_branches],
                                                                     stitch_coordinate=main_limb_stitch_point,
                                                                      verbose=False)

            #h) Find the branch on the floating limb where the closest end point is
            winning_float_branches = np.array([k["branch_skeleton"] for k in winning_floating_correspondence.values()])
            match_float_branches = sk.find_branch_skeleton_with_specific_coordinate(winning_float_branches,
                                current_coordinate=floating_limb_stitch_point)

            if len(match_float_branches) > 1:
                raise Exception("len(match_float_branches) was greater than 1 in the floating pieces stitch")

            if verbose:
                print("\n\n")
                print(f"match_sk_branches = {match_sk_branches}")
                print(f"match_float_branches = {match_float_branches}")
                print(f"stitch_point_on_end_or_branch = {stitch_point_on_end_or_branch}")


            """
            Stitching process:
            i) if not stitch_point_on_end_or_branch
               1. cut the main limb branch where stitch is
               2. do mesh correspondence with the new stitches
               3. (just give the both the same old width)
               4. replace the old entry in the limb corresondence with one of the new skeleton cuts
                  and add on the other skeletons cuts to the end

            j) Add a skeletal segment from floating limb stitch point to main limb stitch point
            k) Add the floating limb branches to the end of the limb correspondence
            l) Marks the floating piece as processed

            """

            # ---------- Begin stitching process ---------------
            if not stitch_point_on_end_or_branch:
                main_branch = match_sk_branches[0]
                #1. cut the main limb branch where stitch is
                matching_branch_sk = sk.cut_skeleton_at_coordinate(skeleton=main_limb_branches[main_branch],
                                                                           cut_coordinate = main_limb_stitch_point)
                #2. do mesh correspondence with the new stitchess
                stitch_mesh = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["branch_mesh"]

                local_correspondnece = mesh_correspondence_first_pass(mesh=stitch_mesh,
                                                          skeleton_branches=matching_branch_sk)

                local_correspondence_revised = correspondence_1_to_1(mesh=stitch_mesh,
                                                            local_correspondence=local_correspondnece)

                #3. (just give the both the same old width)
                old_width = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["width_from_skeleton"]
                for branch_idx in local_correspondence_revised.keys():
                    local_correspondence_revised[branch_idx]["width_from_skeleton"] = old_width

                #4. replace the old entry in the limb corresondence with one of the new skeleton cuts
                #and add on the other skeletons cuts to the end
                print(f"main_branch = {main_branch}")
                del limb_correspondence_cp[winning_float_match_main_limb][main_branch]


                limb_correspondence_cp[winning_float_match_main_limb][main_branch] = local_correspondence_revised[0]
                limb_correspondence_cp[winning_float_match_main_limb][np.max(list(limb_correspondence_cp[winning_float_match_main_limb].keys()))+1] = local_correspondence_revised[1]
                limb_correspondence_cp[winning_float_match_main_limb] = gu.order_dict_by_keys(limb_correspondence_cp[winning_float_match_main_limb])



            #j) Add a skeletal segment from floating limb stitch point to main limb stitch point
            skeleton = winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"]
            adjusted_floating_sk_branch = sk.stack_skeletons([skeleton,np.array([floating_limb_stitch_point,main_limb_stitch_point])])
            
#             adjusted_floating_sk_branch = sk.add_and_smooth_segment_to_branch(skeleton,new_seg=np.array([floating_limb_stitch_point,main_limb_stitch_point]),
#                                                                              resize_mult=0.2,n_resized_cutoff=3)

            winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"] = adjusted_floating_sk_branch

            #k) Add the floating limb branches to the end of the limb correspondence
            curr_limb_key_len = np.max(list(limb_correspondence_cp[winning_float_match_main_limb].keys()))
            for float_idx,flaot_data in winning_floating_correspondence.items():
                limb_correspondence_cp[winning_float_match_main_limb][curr_limb_key_len + 1 + float_idx] = flaot_data
        



        #l) Marks the floating piece as processed
        floating_limbs_to_process = np.setdiff1d(floating_limbs_to_process,[winning_float])
        
    return limb_correspondence_cp



def calculate_limb_concept_networks(limb_correspondence,
                                    network_starting_info,
                                   run_concept_network_checks=True,
                                   verbose=False):
    """
    Can take a limb correspondence and the starting vertices and endpoints
    and create a list of concept networks organized by 
    [soma_idx] --> list of concept networks 
                    (because could possibly have mulitple starting points on the same soma)
    
    """
    
    limb_correspondence_individual = limb_correspondence
    divided_skeletons = np.array([limb_correspondence_individual[k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_individual.keys()))])



    # -------------- Part 18: Getting Concept Networks  [soma_idx] --> list of concept networks -------#

    """
    Concept Network Pseudocode:

    Step 0: Compile the Limb correspondence into final form

    Make sure these have the same list
    limb_to_soma_touching_vertices_list,limb_to_endpoints_must_keep_list

    For every dictionary in zip(limb_to_soma_touching_vertices_list,
                            limb_to_endpoints_must_keep_list):

        #make sure the dicts have same keys
        For every key (represents the soma) in the dictionary:

            #make sure the lists have the same sizes
            For every item in the list (which would be a list of endpoints or list of groups of vertexes):
                #At this point have the soma, the endpoint and the touching vertices

                1) find the branch with the endoint that must keep
                    --> if multiple endpoints then error
                2) Call the branches_to_concept_network with the
                   a. divided skeletons
                   b. closest endpoint
                   c. endpoints of branch (from the branch found)
                   d. touching soma vertices

                3) Run the checks on the concept network

    """

    

    limb_to_soma_concept_networks = dict()

    for soma_idx in network_starting_info.keys():
        
        if soma_idx not in list(limb_to_soma_concept_networks.keys()):
            limb_to_soma_concept_networks[soma_idx] = []
        for soma_group_idx,st_dict in network_starting_info[soma_idx].items():
            t_verts = st_dict["touching_verts"]
            endpt = st_dict["endpoint"]
            if verbose:
                print(f"\n\n---------Working on soma_idx = {soma_idx}, soma_group_idx {soma_group_idx}, endpt = {endpt}---------")



            #1) find the branch with the endoint that must keep
            # ---------------- 11/17 Addition: If the endpoint does not match a skeleton point anymore then just get the closest endpoint of mesh that has touching vertices

            start_branch = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_skeletons,
                                                            current_coordinate=endpt)[0]



            #print(f"Starting_branch = {start_branch}")
            #print(f"Start endpt = {endpt}")
            start_branch_endpoints = sk.find_branch_endpoints(divided_skeletons[start_branch])
            #print(f"Starting_branch endpoints = {start_branch_endpoints}")

            #2) Call the branches_to_concept_network with the
            curr_limb_concept_network = nru.branches_to_concept_network(curr_branch_skeletons=divided_skeletons,
                                                                  starting_coordinate=endpt,
                                                                  starting_edge=start_branch_endpoints,
                                                                  touching_soma_vertices=t_verts,
                                                                       soma_group_idx=soma_group_idx,
                                                                       verbose=verbose)
            if verbose:
                print("Done generating concept network \n\n")


            if run_concept_network_checks:
                #3) Run the checks on the concept network
                #3.1: check to make sure the starting coordinate was recovered

                recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=endpt))

                if verbose:
                    print(f"recovered_touching_piece = {recovered_touching_piece}")
                if recovered_touching_piece[0] != start_branch:
                    raise Exception(f"For limb and soma {soma_idx} the recovered_touching and original touching do not match\n"
                                   f"recovered_touching_piece = {recovered_touching_piece}, original_touching_pieces = {start_branch}")


                #3.2: Check number of nodes match the number of divided skeletons
                if len(curr_limb_concept_network.nodes()) != len(divided_skeletons):
                    raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                                  f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(divided_skeletons)}")

                #3.3: Check that concept network is a connected component
                if nx.number_connected_components(curr_limb_concept_network) > 1:
                    raise Exception("There was more than 1 connected components in the concept network")


                #3.4 Make sure the oriiginal divided skeleton endpoints match the concept map endpoints
                for j,un_resized_b in enumerate(divided_skeletons):
                    """
                    Pseudocode: 
                    1) get the endpoints of the current branch
                    2) get the endpoints in the concept map
                    3) compare
                    - if not equalt then break
                    """
                    #1) get the endpoints of the current branch
                    b_endpoints = neuron.Branch(un_resized_b).endpoints
                    #2) get the endpoints in the concept map
                    graph_endpoints = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",node_list=[j])[0]
                    #print(f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")
                    if not xu.compare_endpoints(b_endpoints,graph_endpoints):
                        raise Exception(f"The node {j} in concept graph endpoints do not match the endpoints of the original branch\n"
                                       f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")

            limb_to_soma_concept_networks[soma_idx].append(curr_limb_concept_network)

    return limb_to_soma_concept_networks                    



def filter_limb_correspondence_for_end_nodes(limb_correspondence,
                                             mesh,
                                             starting_info=None,
                                             filter_end_node_length=4000,
                                             error_on_no_starting_coordinates = True,
                                             plot_new_correspondence = False,
                                             error_on_starting_coordinates_not_endnodes = True,
                                            verbose = True,
                                             
                                             
                                            ):
    """
    Pseudocode:
    1) Get all of the starting coordinates
    2) Assemble the entire skeleton and run the skeleton cleaning process
    3) Decompose skeleton into branches and find out mappin gof old branches to new ones
    4) Assemble the new width and mesh face idx for all new branches
    - width: do weighted average by skeletal length
    - face_idx: concatenate
    5) Make face_lookup and Run waterfilling algorithm to fill in rest
    6) Get the divided meshes and face idx from waterfilling
    7) Store everything back inside a correspondence dictionary
    """

    limb_correspondence_individual=limb_correspondence
    limb_mesh_mparty = mesh
    network_starting_info_revised_cleaned = starting_info


    lc_skeletons = [v["branch_skeleton"] for v in limb_correspondence_individual.values()]
    lc_branch_meshes = [v["branch_mesh"] for v in limb_correspondence_individual.values()]
    lc_branch_face_idx = [v["branch_face_idx"] for v in limb_correspondence_individual.values()]
    lc_width_from_skeletons = [v["width_from_skeleton"] for v in limb_correspondence_individual.values()]

    #1) Get all of the starting coordinates
#     all_starting_coords = []
#     if not starting_info is None:
#         for soma_idx,soma_v in network_starting_info_revised_cleaned.items():
#             for soma_group_idx,soma_group_v in soma_v.items():
#                 all_starting_coords.append(soma_group_v["endpoint"])
                
    if not starting_info is None:
        all_starting_coords = nru.all_soma_connnecting_endpionts_from_starting_info(network_starting_info_revised_cleaned)
    else:
        all_starting_coords = []
                
    
    if error_on_no_starting_coordinates:
        if len(all_starting_coords) == 0:
            raise Exception(f"No starting coordinates found: network_starting_info_revised_cleaned = {network_starting_info_revised_cleaned} ")

    # ---------- 1/5/2021: Will check all starting points as end nodes and if not all degree 1 then error --------#
    

    #2) Assemble the entire skeleton and run the skeleton cleaning process
    curr_limb_sk_cleaned,rem_branches = sk.clean_skeleton(sk.stack_skeletons(lc_skeletons),
                         distance_func=sk.skeletal_distance,
                         min_distance_to_junction=filter_end_node_length,
                        endpoints_must_keep=all_starting_coords,
                         return_skeleton=True,
                         print_flag=False,
                        return_removed_skeletons=True,
                        error_if_endpoints_must_keep_not_endnode=error_on_starting_coordinates_not_endnodes)

    if verbose:
        print(f"Removed {len(rem_branches)} skeletal branches")

    #3) Decompose skeleton into branches and find out mappin gof old branches to new ones
    cleaned_branches = sk.decompose_skeleton_to_branches(curr_limb_sk_cleaned)
    
    if len(cleaned_branches) == 0:
        print("There were no branches after cleaning limb correspondence")
        return limb_correspondence
    
    original_br_mapping = sk.map_between_branches_lists(lc_skeletons,cleaned_branches)

    # 4) Assemble the new width and mesh face idx for all new branches
    # - width: do weighted average by skeletal length
    # - face_idx: concatenate

    new_width_from_skeletons = []
    new_branch_face_idx = []

    for j,cl_b in enumerate(cleaned_branches):
        or_idx = np.where(original_br_mapping==j)[0]

        #doing the width
        total_skeletal_length = 0
        weighted_width = 0
        if len(or_idx) > 1:
            #print(f"\n\nAverageing widths for {j}:")
            for oi in or_idx:
                curr_sk_len = sk.calculate_skeleton_distance(lc_skeletons[oi])
                #print(f"curr_sk_len = {curr_sk_len}, curr_width = {lc_width_from_skeletons[oi]}")
                weighted_width += curr_sk_len*lc_width_from_skeletons[oi]
                total_skeletal_length+=curr_sk_len
            final_width = weighted_width/total_skeletal_length
            #print(f"Final width = {final_width}")
            new_width_from_skeletons.append(final_width)
        else:
            new_width_from_skeletons.append(lc_width_from_skeletons[or_idx[0]])


        #doing the face_idx
        new_branch_face_idx.append(np.concatenate([lc_branch_face_idx[k] for k in or_idx]))


    #5) Make face_lookup and Run waterfilling algorithm to fill in rest

    face_lookup = {j:[] for j in range(0,len(limb_mesh_mparty.faces))}
    face_lookup_marked = gu.invert_mapping(new_branch_face_idx)
    ky = list(face_lookup.keys())
    if verbose:
        print(f"For marked faces: {print(np.max(ky),len(ky))}")
    face_lookup.update(face_lookup_marked)





    original_labels = np.arange(0,len(cleaned_branches))

    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = limb_mesh_mparty,
                                                                    face_lookup=face_lookup,
                                                                    no_missing_labels = list(original_labels))



    #6) Get the divided meshes and face idx from waterfilling
    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(limb_mesh_mparty,face_coloring_copy,
                                                                            return_dict=False)


    #7) Store everything back inside a correspondence dictionary
    limb_correspondence_individual_filtered = dict()
    for j,curr_sk in enumerate(cleaned_branches):
        local_dict = dict(branch_skeleton=curr_sk,
                          width_from_skeleton=new_width_from_skeletons[j],
                         branch_mesh=divided_submeshes[j],
                         branch_face_idx=divided_submeshes_idx[j])
        limb_correspondence_individual_filtered[j] = local_dict


    if plot_new_correspondence:
        plot_limb_correspondence(limb_correspondence_individual_filtered)

    return limb_correspondence_individual_filtered




def preprocess_limb(mesh,
                   soma_touching_vertices_dict = None,
                   distance_by_mesh_center=True, #how the distance is calculated for mesh correspondence
                    meshparty_segment_size = 100,
                   meshparty_n_surface_downsampling = 2,
                    combine_close_skeleton_nodes=True,
                    combine_close_skeleton_nodes_threshold=700,
                    filter_end_node_length=None,
                    use_meshafterparty=True,
                    perform_cleaning_checks = True,
                    
                    #for controlling the pieces processed by MAP
                    width_threshold_MAP = None,
                    size_threshold_MAP = None,
                    
                    #parameters for MP skeletonization,
                    
                    #Parameters for setting how the MAP skeletonization takes place
                    use_surface_after_CGAL=True,
                    surface_reconstruction_size = None,#500,
                    
                    #parametrers for stitching the MAP and MP pieces together
                    move_MAP_stitch_to_end_or_branch = True,
                    distance_to_move_point_threshold=500,
                    
                    #concept_network parameters
                    run_concept_network_checks = True,
                    return_concept_network = True,
                    return_concept_network_starting_info=False,
                    
                    #printing controls
                    verbose = True,
                    print_fusion_steps=True,
                    
                    check_correspondence_branches = True,
                    filter_end_nodes_from_correspondence=True,
                    error_on_no_starting_coordinates=True,
                    
                    prevent_MP_starter_branch_stitches = False, #will control if a MP soma extending branch is able to be stitched to
                    combine_close_skeleton_nodes_threshold_meshparty = None,
                    filter_end_node_length_meshparty=None,
                    invalidation_d=None,
                    smooth_neighborhood = 1,
                    
                    use_adaptive_invalidation_d = None,
                    axon_width_preprocess_limb_max = None,
                    
                    remove_mesh_interior_face_threshold = None,
                    error_on_bad_cgal_return=False,
                    max_stitch_distance_CGAL = None,
                    
                    # ----- 4/17/25 revision
                    mp_only_revised_invalidation_d = None,
                    mp_only_invalidation_d_axon_buffer = None,
                    mp_only_revised_invalidation_d_reference = None,
                    mp_only_revised_width_reference = None,
                    
                   ):
    if filter_end_node_length is None:
        filter_end_node_length = filter_end_node_length_global
    if width_threshold_MAP is None:
        width_threshold_MAP = width_threshold_MAP_global
    if size_threshold_MAP is None:
        size_threshold_MAP = size_threshold_MAP_global
    if invalidation_d is None:
        invalidation_d = invalidation_d_global
    if axon_width_preprocess_limb_max is None:
        axon_width_preprocess_limb_max = axon_width_preprocess_limb_max_global
    if remove_mesh_interior_face_threshold is None:
        remove_mesh_interior_face_threshold = remove_mesh_interior_face_threshold_global
    if surface_reconstruction_size is None:
        surface_reconstruction_size = surface_reconstruction_size_global
    if max_stitch_distance_CGAL is None:
        max_stitch_distance_CGAL = max_stitch_distance_CGAL_global
        
    if use_adaptive_invalidation_d is None:
        use_adaptive_invalidation_d = use_adaptive_invalidation_d_global
        
    if mp_only_revised_invalidation_d is None:
        mp_only_revised_invalidation_d=mp_only_revised_invalidation_d_global
    if mp_only_invalidation_d_axon_buffer is None:
        mp_only_invalidation_d_axon_buffer=mp_only_invalidation_d_axon_buffer_global
    if mp_only_revised_invalidation_d_reference is None:
        mp_only_revised_invalidation_d_reference=mp_only_revised_invalidation_d_reference_global
    if mp_only_revised_width_reference is None:
        mp_only_revised_width_reference=mp_only_revised_width_reference_global
    
    
    print(f"invalidation_d = {invalidation_d}")
    print(f"use_adaptive_invalidation_d= {use_adaptive_invalidation_d}")
    print(f"axon_width_preprocess_limb_max = {axon_width_preprocess_limb_max}")
        
        
    if combine_close_skeleton_nodes_threshold_meshparty is None:
        combine_close_skeleton_nodes_threshold_meshparty = combine_close_skeleton_nodes_threshold
    if filter_end_node_length_meshparty is None:
        filter_end_node_length_meshparty = filter_end_node_length
        
    print(f"filter_end_node_length= {filter_end_node_length}")
    print(f"filter_end_node_length_meshparty = {filter_end_node_length_meshparty}")
    print(f"invalidation_d = {invalidation_d}")
    
    #print(f"soma_touching_vertices_dict = {soma_touching_vertices_dict}")
    #print(f"error_on_no_starting_coordinates = {error_on_no_starting_coordinates}")
    curr_limb_time = time.time()

    limb_mesh_mparty = mesh


    #will store a list of all the endpoints tha tmust be kept:
    limb_to_endpoints_must_keep_list = []
    limb_to_soma_touching_vertices_list = []

    # --------------- Part 1 and 2: Getting Border Vertices and Setting the Root------------- #
    fusion_time = time.time()
    #will eventually get the current root from soma_to_piece_touching_vertices[i]
    if not soma_touching_vertices_dict is None:
        root_curr = soma_touching_vertices_dict[list(soma_touching_vertices_dict.keys())[0]][0][0]
    else:
        root_curr = None
        
    print(f"root_curr = {root_curr}")

    if print_fusion_steps:
        print(f"Time for preparing soma vertices and root: {time.time() - fusion_time }")
        fusion_time = time.time()

    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    for ii in range(0,2):
        sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(
            limb_mesh_mparty,
            root=root_curr,
            invalidation_d=invalidation_d,
            smooth_neighborhood=smooth_neighborhood,
            filter_mesh=False
        )

        print(f"meshparty_segment_size = {meshparty_segment_size}")

        if print_fusion_steps:
            print(f"Time for 1st pass MP skeletonization: {time.time() - fusion_time }")
            fusion_time = time.time()

        
            
            
        (segment_branches, #skeleton branches
        divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
        segment_widths_median) = m_sk.skeleton_obj_to_branches(
                                    sk_meshparty_obj,
                                    mesh = limb_mesh_mparty,
                                    meshparty_segment_size=meshparty_segment_size,
                                    combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold_meshparty,
                                    filter_end_node_length=filter_end_node_length_meshparty
                                    )
        
        debug_meshparty = False
        if debug_meshparty:
            su.compressed_pickle(segment_branches,"segment_branches")
            su.compressed_pickle(divided_submeshes,"divided_submeshes")
            su.compressed_pickle(divided_submeshes_idx,"divided_submeshes_idx")
            su.compressed_pickle(segment_widths_median,"segment_widths_median")
        
            raise Exception("")
        
        
        if not use_adaptive_invalidation_d:
            break
            
        width_median = m_sk.width_median_weighted(segment_widths_median,segment_branches)
        pieces_above_threshold = np.where(segment_widths_median>width_threshold_MAP)[0]
        #width_median = nu.weighted_average(segment_widths_median,[sk.calculate_skeleton_distance(k) for k in segment_branches])
        
        
        if True:#verbose:
            print(f"width_median= {width_median}")
            print(f"segment_widths_median = {segment_widths_median}")
            print(f"# pieces_above_threshold = {len(pieces_above_threshold)}")
        
        if ii == 1:
            break
        
        if width_median <= axon_width_preprocess_limb_max:
            if verbose:
                print(f"Using the axon parameters")
            combine_close_skeleton_nodes_threshold_meshparty = combine_close_skeleton_nodes_threshold_meshparty_axon_global
            filter_end_node_length_meshparty = filter_end_node_length_meshparty_axon_global
            filter_end_node_length= filter_end_node_length_axon_global
            invalidation_d= invalidation_d_axon_global
            smooth_neighborhood = smooth_neighborhood_axon_global
            continue
        elif len(pieces_above_threshold) == 0 and mp_only_revised_invalidation_d and invalidation_d != invalidation_d_axon_global:
            print(f"Using MP only revised invalidation")
            """
            -- 4/17 change --
            if there are no MAP pieces then we want to do a meshparty processing with a slightly finer 
            meshparty processing 
            """
            def new_invalidation_d(
                width_median,
                max_invalidation_d = invalidation_d,
                verbose = False
                ):
                """
                Purpose: to compute the new invalidation_d based on the width 
                (linear interpolation between parameters)
                
                equation (just computes the slope and bases reference on invalidation ref)
                --------
                slope = (mp_only_inv_d_ref - (inv_d_axon + buffer)) / (mp_only_width_ref - ax_width)
                new_inv_d = slope*(width - mp_only_width_ref) + mp_only_inv_d_ref
                """
                lowest_value = invalidation_d_axon_global + mp_only_invalidation_d_axon_buffer
                change_y = (mp_only_revised_invalidation_d_reference - lowest_value)
                change_x = (mp_only_revised_width_reference - axon_width_preprocess_limb_max)

                if change_x == 0:
                    return invalidation_d

                slope = change_y/change_x

                delta_x = width_median - mp_only_revised_width_reference
                new_d = slope*delta_x + mp_only_revised_invalidation_d_reference
                final_d = max(min(new_d,max_invalidation_d),lowest_value)

                if verbose:
                    print(f"new_invalidation_d = {new_d} (max = {max_invalidation_d}), final_invalidation_d = {final_d}")

                return final_d
            
            invalidation_d=new_invalidation_d(width_median,verbose = True)
            continue
         
        else:
            break
            

    if print_fusion_steps:
        print(f"Decomposing first pass: {time.time() - fusion_time }")
        fusion_time = time.time()


    if use_meshafterparty:
        print("Attempting to use MeshAfterParty Skeletonization and Mesh Correspondence")
        # --------------- Part 4: Find Individual Branches that could be MAP processed because of width ------------- #
        #gettin the branches that should be passed through MAP skeletonization
        pieces_above_threshold = np.where(segment_widths_median>width_threshold_MAP)[0]

        #getting the correspondnece info for those MAP qualifying
        width_large = segment_widths_median[pieces_above_threshold]
        sk_large = [segment_branches[k] for k in pieces_above_threshold]
        mesh_large_idx = [divided_submeshes_idx[k] for k in pieces_above_threshold]
    else:
        print("Only Using MeshParty Skeletonization and Mesh Correspondence")
        mesh_large_idx = []
        width_large = []
        sk_large = []


    print("Another print")
    mesh_pieces_for_MAP = []
    mesh_pieces_for_MAP_face_idx = []


    if len(mesh_large_idx) > 0: #will only continue processing if found MAP candidates

        # --------------- Part 5: Find mesh connectivity and group MAP branch candidates into MAP sublimbs ------------- #
        print(f"Found len(mesh_large_idx) MAP candidates: {[len(k) for k in mesh_large_idx]}")

        #finds the connectivity edges of all the MAP candidates
        mesh_large_connectivity = tu.mesh_list_connectivity(meshes = mesh_large_idx,
                                                            connectivity="edges",
                                main_mesh = limb_mesh_mparty,
                                print_flag = False)
        
        """ 1/3/21s
        Big Conclusion from debugging: the large mesh pieces themselves (before combining into map pieces)
        themselves aren't totally connected by edges (can be split)

        - so even if large pieces do have a shared edge and you combine them together,
        they can still be split by the edges into multiple pieces because the original pieces
        could be split into multiple pieces

        

        """
        
        
        if print_fusion_steps:
            print(f"mesh_large_connectivity: {time.time() - fusion_time }")
            fusion_time = time.time()
        """
        --------------- Grouping MAP candidates ----------------
        Purpose: Will see what mesh pieces should be grouped together
        to pass through CGAL skeletonization


        Pseudocode: 
        1) build a networkx graph with all nodes for mesh_large_idx indexes
        2) Add the edges
        3) Find the connected components
        4) Find sizes of connected components
        5) For all those connected components that are of a large enough size, 
        add the mesh branches and skeletons to the final list


        """
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(mesh_large_idx)))
        G.add_edges_from(mesh_large_connectivity)
        conn_comp = list(nx.connected_components(G))

        filtered_pieces = []

        sk_large_size_filt = []
        mesh_large_idx_size_filt = []
        width_large_size_filt = []

        for cc in conn_comp:
            total_cc_size = np.sum([len(mesh_large_idx[k]) for k in cc])
            if total_cc_size>size_threshold_MAP:
                #print(f"cc ({cc}) passed the size threshold because size was {total_cc_size}")
                filtered_pieces.append(pieces_above_threshold[list(cc)])

        if print_fusion_steps:
            print(f"Finding MAP candidates connected components: {time.time() - fusion_time }")
            fusion_time = time.time()

        #filtered_pieces: will have the indexes of all the branch candidates that should  be 
        #grouped together and passed through MAP skeletonization

        if len(filtered_pieces) > 0:
            # --------------- Part 6: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            print(f"len(filtered_pieces) = {len(filtered_pieces)}")
            #all the pieces that will require MAP mesh correspondence and skeletonization
            #(already organized into their components)
            mesh_pieces_for_MAP = [limb_mesh_mparty.submesh([np.concatenate(divided_submeshes_idx[k])],append=True,repair=False) for k in filtered_pieces]
            mesh_pieces_for_MAP_face_idx = [np.concatenate(divided_submeshes_idx[k]) for k in filtered_pieces]



            """
            Old Way: Finding connectivity of pieces through
            mesh_idx_MP = [divided_submeshes_idx[k] for k in pieces_idx_MP]

            mesh_large_connectivity_MP = tu.mesh_list_connectivity(meshes = mesh_idx_MP,
                                    main_mesh = limb_mesh_mparty,
                                    print_flag = False)

            New Way: going to use skeleton connectivity to determine
            connectivity of pieces

            Pseudocode: 
            1)

            """
            # --------------- Part 7: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            # ********* if there are no pieces leftover then will automatically make all the lists below just empty (don't need to if.. else.. the case)****
            pieces_idx_MP = np.delete(np.arange(len(divided_submeshes_idx)),np.concatenate(filtered_pieces))

            skeleton_MP = [segment_branches[k] for k in pieces_idx_MP]
            skeleton_connectivity_MP = sk.skeleton_list_connectivity(
                                            skeletons=skeleton_MP
                                            )
            if print_fusion_steps:
                print(f"skeleton_connectivity_MP : {time.time() - fusion_time }")
                fusion_time = time.time()

            G = nx.Graph()
            G.add_nodes_from(np.arange(len(skeleton_MP)))
            G.add_edges_from(skeleton_connectivity_MP)
            sublimbs_MP = list(nx.connected_components(G))
            sublimbs_MP_orig_idx = [pieces_idx_MP[list(k)] for k in sublimbs_MP]


            #concatenate into sublimbs the skeletons and meshes
            sublimb_mesh_idx_branches_MP = [divided_submeshes_idx[k] for k in sublimbs_MP_orig_idx]
            sublimb_mesh_branches_MP = [[limb_mesh_mparty.submesh([ki],append=True,repair=False)
                                        for ki in k] for k in sublimb_mesh_idx_branches_MP]
            sublimb_meshes_MP = [limb_mesh_mparty.submesh([np.concatenate(k)],append=True,repair=False)
                                                         for k in sublimb_mesh_idx_branches_MP]
            sublimb_meshes_MP_face_idx = [np.concatenate(k)
                                                         for k in sublimb_mesh_idx_branches_MP]
            sublimb_skeleton_branches = [segment_branches[k] for k in sublimbs_MP_orig_idx]
            widths_MP = [segment_widths_median[k] for k in sublimbs_MP_orig_idx]

            if print_fusion_steps:
                print(f"Grouping MP Sublimbs by Graph: {time.time() - fusion_time }")
                fusion_time = time.time()


    # else: #if no pieces were determine to need MAP processing
    #     print("No MAP processing needed: just returning the Meshparty skeletonization and mesh correspondence")
    #     raise Exception("Returning MP correspondence")


    # nviz.plot_objects(main_mesh=tu.combine_meshes([limb_mesh_mparty,current_neuron["S0"].mesh]),
    #                   main_mesh_color="green",
    #     skeletons=sk_large_size_filt,
    #      meshes=[limb_mesh_mparty.submesh([k],append=True) for k in mesh_large_idx_size_filt],
    #       meshes_colors="red")








    # --------------- Part 8: If No MAP sublimbs found, set the MP sublimb lists to just the whole MP branch decomposition ------------- #

    #if no sublimbs need to be decomposed with MAP then just reassign all of the previous MP processing to the sublimb_MPs
    
    
    
    if len(mesh_pieces_for_MAP) == 0:
        print('no MAP pieces')
        sublimb_meshes_MP = [limb_mesh_mparty] #trimesh pieces that have already been passed through MP skeletonization (may not need)
        # -- the decomposition information ---
        sublimb_mesh_branches_MP = [divided_submeshes] #the mesh branches for all the disconnected sublimbs
        sublimb_mesh_idx_branches_MP = [divided_submeshes_idx] #The mesh branches idx that have already passed through MP skeletonization
        sublimb_skeleton_branches = [segment_branches]#the skeleton bnraches for all the sublimbs
        widths_MP = [segment_widths_median] #the mesh branches widths for all the disconnected groups

        MAP_flag = False
    else:
        MAP_flag = True



    mesh_pieces_for_MAP #trimesh pieces that should go through CGAL skeletonization
    sublimb_meshes_MP #trimesh pieces that have already been passed through MP skeletonization (may not need)
    
    # su.save_object(mesh_pieces_for_MAP,"mesh_pieces_for_MAP")
    # su.save_object(sublimb_meshes_MP,"sublimb_meshes_MP")
    # raise Exception("")

    # -- the decomposition information ---
    sublimb_mesh_branches_MP #the mesh branches for all the disconnected sublimbs
    sublimb_mesh_idx_branches_MP #The mesh branches idx that have already passed through MP skeletonization
    sublimb_skeleton_branches #the skeleton bnraches for all the sublimbs
    widths_MP #the mesh branches widths for all the disconnected groups

    if print_fusion_steps:
        print(f"Divinding into MP and MAP pieces: {time.time() - fusion_time }")
        fusion_time = time.time()



    # ------------------- At this point have the correct division between MAP and MP ------------------------

    # -------------- Part 9: Doing the MAP decomposition ------------------ #
    global_start_time = time.time()
    endpoints_must_keep = dict()



    limb_correspondence_MAP = dict()

    for sublimb_idx,(mesh,mesh_idx) in enumerate(zip(mesh_pieces_for_MAP,mesh_pieces_for_MAP_face_idx)):
        print(f"--- Working on MAP piece {sublimb_idx}---")
        #print(f"soma_touching_vertices_dict = {soma_touching_vertices_dict}")
        mesh_start_time = time.time()
        curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
        mesh = mesh,
        curr_piece_to_soma_touching_vertices = soma_touching_vertices_dict
        )

        if print_fusion_steps:
            print(f"MAP Filtering Soma Pieces: {time.time() - fusion_time }")
            fusion_time = time.time()

        # ---- 0) Generating the Clean skeletons  -------------------------------------------#
        if not curr_soma_to_piece_touching_vertices is None:
            curr_total_border_vertices = dict([(k,np.vstack(v)) for k,v in curr_soma_to_piece_touching_vertices.items()])
        else:
            curr_total_border_vertices = None


        cleaned_branch,curr_limb_endpoints_must_keep = sk.skeletonize_and_clean_connected_branch_CGAL(
            mesh=mesh,
            curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,
            total_border_vertices=curr_total_border_vertices,
            filter_end_node_length=filter_end_node_length,
            perform_cleaning_checks=perform_cleaning_checks,
            combine_close_skeleton_nodes = combine_close_skeleton_nodes,
            combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
        use_surface_after_CGAL=use_surface_after_CGAL,
        surface_reconstruction_size=surface_reconstruction_size,
        remove_mesh_interior_face_threshold=remove_mesh_interior_face_threshold,
        error_on_bad_cgal_return=error_on_bad_cgal_return,
        max_stitch_distance = max_stitch_distance_CGAL)

        if not curr_limb_endpoints_must_keep is None:
            limb_to_endpoints_must_keep_list.append(curr_limb_endpoints_must_keep)
            limb_to_soma_touching_vertices_list.append(curr_soma_to_piece_touching_vertices)
        else:
            print("Inside MAP decomposition and curr_limb_endpoints_must_keep was None")

        if len(cleaned_branch) == 0:
            raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")

        if print_fusion_steps:
            print(f"skeletonize_and_clean_connected_branch_CGAL: {time.time() - fusion_time }")
            fusion_time = time.time()

        # ---- 1) Generating Initial Mesh Correspondence -------------------------------------------#
        start_time = time.time()

        print(f"Working on limb correspondence for #{sublimb_idx} MAP piece")
        local_correspondence = mesh_correspondence_first_pass(mesh=mesh,
                                                             skeleton=cleaned_branch,
                                                             distance_by_mesh_center=distance_by_mesh_center,
                                                             connectivity="edges",
                                                             remove_inside_pieces_threshold=100)


        print(f"Total time for decomposition = {time.time() - start_time}")
        if print_fusion_steps:
            print(f"mesh_correspondence_first_pass: {time.time() - fusion_time }")
            fusion_time = time.time()


        #------------- 2) Doing Some checks on the initial corespondence -------- #


        if perform_cleaning_checks:
            check_skeletonization_and_decomp(skeleton=cleaned_branch,
                                            local_correspondence=local_correspondence)

        # -------3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
        local_correspondence_revised = correspondence_1_to_1(mesh=mesh,
                                        local_correspondence=local_correspondence,
                                        curr_limb_endpoints_must_keep=curr_limb_endpoints_must_keep,
                                        curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices)

        # -------3b) Fixing the mesh indices to correspond to the larger mesh as a whole
        for k,v in local_correspondence_revised.items():
            local_correspondence_revised[k]["branch_face_idx"] = mesh_idx[local_correspondence_revised[k]["branch_face_idx"]]

        print(f"Total time for MAP sublimb #{sublimb_idx} mesh processing = {time.time() - mesh_start_time}")

        if print_fusion_steps:
            print(f"correspondence_1_to_1: {time.time() - fusion_time }")
            fusion_time = time.time()

        limb_correspondence_MAP[sublimb_idx] = local_correspondence_revised

    print(f"Total time for MAP sublimb processing {time.time() - global_start_time}")





    # ----------------- Part 10: Doing the MP Decomposition ---------------------- #




    sublimb_meshes_MP #trimesh pieces that have already been passed through MP skeletonization (may not need)
    # -- the decomposition information ---
    sublimb_mesh_branches_MP #the mesh branches for all the disconnected sublimbs
    sublimb_mesh_idx_branches_MP #The mesh branches idx that have already passed through MP skeletonization
    sublimb_skeleton_branches #the skeleton bnraches for all the sublimbs
    widths_MP #the mesh branches widths for all the disconnected groups

    limb_correspondence_MP = dict()

    for sublimb_idx,mesh in enumerate(sublimb_meshes_MP):
        print(f"---- Working on MP Decomposition #{sublimb_idx} ----")
        mesh_start_time = time.time()

        if len(sublimb_meshes_MP) == 1 and MAP_flag == False:
            print("Using Quicker soma_to_piece_touching_vertices because no MAP and only one sublimb_mesh piece ")
            curr_soma_to_piece_touching_vertices = soma_touching_vertices_dict
        else:
            if not soma_touching_vertices_dict is None:
                print("Computing the current soma touching verts dict manually")
                curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
                                                    mesh = mesh,
                                                    curr_piece_to_soma_touching_vertices = soma_touching_vertices_dict
                                                    )
            else:
                curr_soma_to_piece_touching_vertices = None

        if print_fusion_steps:
            print(f"MP filtering soma verts: {time.time() - fusion_time }")
            fusion_time = time.time()

        #creating all of the sublimb groups
        segment_branches = np.array(sublimb_skeleton_branches[sublimb_idx])
        whole_sk_MP = sk.stack_skeletons(segment_branches)
        branch = mesh
        divided_submeshes = np.array(sublimb_mesh_branches_MP[sublimb_idx])
        divided_submeshes_idx = sublimb_mesh_idx_branches_MP[sublimb_idx]
        segment_widths_median = widths_MP[sublimb_idx]


        if curr_soma_to_piece_touching_vertices is None:
            print(f"Do Not Need to Fix MP Decomposition {sublimb_idx} so just continuing")

        else:

            # ------- 11/9 addition: Fixing error where creating soma touching branch on mesh that doesn't touch border ------------------- #
            print(f"Fixing Possible Soma Extension Branch for Sublimb {sublimb_idx}")
            no_soma_extension_add = True 

            endpts_total = dict()
            curr_soma_to_piece_touching_vertices_total = dict()
            for sm_idx,sm_bord_verts_list in curr_soma_to_piece_touching_vertices.items():
                #will be used for later
                endpts_total[sm_idx] = []
                curr_soma_to_piece_touching_vertices_total[sm_idx] = []

                for sm_bord_verts in sm_bord_verts_list:
                    #1) Get the mesh pieces that are touching the border
                    matching_mesh_idx = tu.filter_meshes_by_containing_coordinates(mesh_list=divided_submeshes,
                                               nullifying_points=sm_bord_verts,
                                                filter_away=False,
                                               distance_threshold=min_distance_threshold,
                                               return_indices=True)
                    #2) concatenate all meshes and skeletons that are touching
                    if len(matching_mesh_idx) <= 0:
                        raise Exception("None of branches were touching the border vertices when fixing MP pieces")

                    touch_mesh = tu.combine_meshes(divided_submeshes[matching_mesh_idx])
                    touch_sk = sk.stack_skeletons(segment_branches[matching_mesh_idx])

                    local_curr_soma_to_piece_touching_vertices = {sm_idx:[sm_bord_verts]}
                    new_sk,endpts,new_branch_info = sk.create_soma_extending_branches(current_skeleton=touch_sk,
                                          skeleton_mesh=touch_mesh,
                                          soma_to_piece_touching_vertices=local_curr_soma_to_piece_touching_vertices,
                                          return_endpoints_must_keep=True,
                                          return_created_branch_info=True,
                                          check_connected_skeleton=False)
                    
                    # ---- 12/30 Addition Check if the endpoint found is an endnode or not and if not then manually add branch ---
                    curr_endnode = endpts[sm_idx][0]
                    match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                        current_coordinate=curr_endnode)

                    print(f"match_sk_branches = {match_sk_branches}")
                    if len(match_sk_branches) > 1:
                        border_average_coordinate = np.mean(sm_bord_verts,axis=0)
                        new_branch_sk = np.vstack([curr_endnode,border_average_coordinate]).reshape(-1,2,3)
                        br_info = dict(new_branch = new_branch_sk,border_verts=sm_bord_verts)
                        endpts_total[sm_idx].append(border_average_coordinate)
                    else:
                        
                        br_info = new_branch_info[sm_idx][0]
                        endpts_total[sm_idx].append(endpts[sm_idx][0])
                    # -------------------- End of 12/30 Addition ------------------

                    #3) Add the info to the new running lists
                    
                    curr_soma_to_piece_touching_vertices_total[sm_idx].append(sm_bord_verts)


                    #4) Skip if no new branch was added
                    if br_info is None:
                        print("The new branch info was none so skipping \n")
                        continue

                    #4 If new branch was made then 
                    no_soma_extension_add=False

                    #1) Get the newly added branch (and the original vertex which is the first row)
                    br_new,sm_bord_verts = br_info["new_branch"],br_info["border_verts"] #this will hold the new branch and the border vertices corresponding to it

                    curr_soma_to_piece_touching_vertices_MP = {sm_idx:[sm_bord_verts]}
                    endpoints_must_keep_MP = {sm_idx:[br_new[0][1]]}


                    orig_vertex = br_new[0][0]
                    print(f"orig_vertex = {orig_vertex}")

                    #2) Find the branches that have that coordinate (could be multiple)
                    match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                        current_coordinate=orig_vertex)

                    print(f"match_sk_branches = {match_sk_branches}")



                    """ ******************* THIS NEEDS TO BE FIXED WITH THE SAME METHOD OF STITCHING ********************  """
                    """
                    Pseudocode:
                    1) Find if branch point will require split or not
                    2) If does require split then split the skeleton
                    3) Gather mesh pieces for correspondence and the skeletons
                    4) Run the mesh correspondence
                    - this case calculate the new widths after run 
                    5) Replace the old branch parts with the new ones



                    """

                    stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                                                            matched_branches_skeletons= segment_branches[match_sk_branches],
                                                             stitch_coordinate=orig_vertex,
                                                              verbose=False)


                    if not stitch_point_on_end_or_branch:
                        matching_branch_sk = sk.cut_skeleton_at_coordinate(skeleton=segment_branches[match_sk_branches][0],
                                                                          cut_coordinate = orig_vertex)
                    else:
                        matching_branch_sk = segment_branches[match_sk_branches]


                    #3) Find the mesh and skeleton of the winning branch
                    matching_branch_meshes = np.array(divided_submeshes)[match_sk_branches]
                    matching_branch_mesh_idx = np.array(divided_submeshes_idx)[match_sk_branches]
                    extend_soma_mesh_idx = np.concatenate(matching_branch_mesh_idx)
                    extend_soma_mesh = limb_mesh_mparty.submesh([extend_soma_mesh_idx ],append=True,repair=False)

                    #4) Add newly created branch to skeleton and divide the skeleton into branches (could make 2 or 3)
                    #extended_skeleton_to_soma = sk.stack_skeletons([list(matching_branch_sk),br_new])

                    sk.check_skeleton_connected_component(sk.stack_skeletons(list(matching_branch_sk) + [br_new]))

                    #5) Run Adaptive mesh correspondnece using branches and mesh
                    local_correspondnece_MP = mesh_correspondence_first_pass(mesh=extend_soma_mesh,
                                                                             skeleton_branches = list(matching_branch_sk) + [br_new]
                                                  #skeleton=extended_skeleton_to_soma
                                                                            )

                    # GETTING MESHES THAT ARE NOT FULLY CONNECTED!!
                    local_correspondence_revised = correspondence_1_to_1(mesh=extend_soma_mesh,
                                                                local_correspondence=local_correspondnece_MP,
                                                                curr_limb_endpoints_must_keep=endpoints_must_keep_MP,
                                                                curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices_MP)

                    # All the things that should be revised:
                #     segment_branches, #skeleton branches
                #     divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
                #     segment_widths_median


                    new_submeshes = [k["branch_mesh"] for k in local_correspondence_revised.values()]
                    new_submeshes_idx = [extend_soma_mesh_idx[k["branch_face_idx"]] for k in local_correspondence_revised.values()]
                    new_skeletal_branches = [k["branch_skeleton"] for k in local_correspondence_revised.values()]

                    #calculate the new width
                    ray_inter = tu.ray_pyembree.RayMeshIntersector(limb_mesh_mparty)
                    new_widths = []
                    for new_s_idx in new_submeshes_idx:
                        curr_ray_distance = tu.ray_trace_distance(mesh=limb_mesh_mparty, 
                                            face_inds=new_s_idx,
                                           ray_inter=ray_inter)
                        curr_width_median = np.median(curr_ray_distance[curr_ray_distance!=0])
                        print(f"curr_width_median = {curr_width_median}")
                        if (not np.isnan(curr_width_median)) and (curr_width_median > 0):
                            new_widths.append(curr_width_median)
                        else:
                            print(f"USING A DEFAULT WIDTH BECAUSE THE NEWLY COMPUTED ONE WAS {curr_width_median}: {segment_widths_median[match_sk_branches[0]]}")
                            new_widths.append(segment_widths_median[match_sk_branches[0]])


                    #6) Remove the original branch and mesh correspondence and replace with the multiples
    #                     print(f"match_sk_branches BEFORE = {match_sk_branches}")
    #                     print(f"segment_branches BEFORE = {segment_branches}")
    #                     print(f"len(new_skeletal_branches) = {len(new_skeletal_branches)}")
    #                     print(f"new_skeletal_branches BEFORE= {new_skeletal_branches}")


                    #segment_branches = np.delete(segment_branches,match_sk_branches,axis=0)
                    #segment_branches = np.append(segment_branches,new_skeletal_branches,axis=0)

                    segment_branches = np.array([k for i,k in enumerate(segment_branches) if i not in match_sk_branches] + new_skeletal_branches)


                    divided_submeshes = np.delete(divided_submeshes,match_sk_branches,axis=0)
                    divided_submeshes = np.append(divided_submeshes,new_submeshes,axis=0)


                    #divided_submeshes_idx = np.delete(divided_submeshes_idx,match_sk_branches,axis=0)
                    #divided_submeshes_idx = np.append(divided_submeshes_idx,new_submeshes_idx,axis=0)
                    divided_submeshes_idx = np.array([k for i,k in enumerate(divided_submeshes_idx) if i not in match_sk_branches] + new_submeshes_idx)

                    segment_widths_median = np.delete(segment_widths_median,match_sk_branches,axis=0)
                    segment_widths_median = np.append(segment_widths_median,new_widths,axis=0)

                    try:
                        debug = False
                        if debug:
                            print(f"segment_branches.shape = {segment_branches.shape}")
                            print(f"segment_branches = {segment_branches}")
                            print(f"new_skeletal_branches = {new_skeletal_branches}")
                        sk.check_skeleton_connected_component(sk.stack_skeletons(segment_branches))
                    except:
                        su.compressed_pickle(local_correspondence_revised,"local_correspondence_revised")
                    print("checked segment branches after soma add on")
                    return_find = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                                                 orig_vertex)



                    """ ******************* END OF HOW CAN DO STITCHING ********************  """



            limb_to_endpoints_must_keep_list.append(endpts_total)
            limb_to_soma_touching_vertices_list.append(curr_soma_to_piece_touching_vertices_total)
            
            #print(f"limb_to_endpoints_must_keep_list = {limb_to_endpoints_must_keep_list}")
            #print(f"limb_to_soma_touching_vertices_list = {limb_to_soma_touching_vertices_list}")

            # ------------------- 11/9 addition ------------------- #

            if no_soma_extension_add:
                print("No soma extending branch was added for this sublimb even though it had a soma border (means they already existed)")

            if print_fusion_steps:
                print(f"MP (because soma touching verts) soma extension add: {time.time() - fusion_time }")
                fusion_time = time.time()

        #building the limb correspondence
        limb_correspondence_MP[sublimb_idx] = dict()

        for zz,b_sk in enumerate(segment_branches):
            limb_correspondence_MP[sublimb_idx][zz] = dict(
                branch_skeleton = b_sk,
                width_from_skeleton = segment_widths_median[zz],
                branch_mesh = divided_submeshes[zz],
                branch_face_idx = divided_submeshes_idx[zz]
                )



    #limb_correspondence_MP_saved = copy.deepcopy(limb_correspondence_MP)
    #limb_correspondence_MAP_saved = copy.deepcopy(limb_correspondence_MAP)

    # ------------------------------------- Part C: Will make sure the correspondences can all be stitched together --------------- #

    
    
#     su.compressed_pickle(limb_correspondence_MAP,"limb_correspondence_MAP_before_stitch")
#     su.compressed_pickle(limb_correspondence_MP,"limb_correspondence_MP_before_stitch")

    
    if check_correspondence_branches:
        sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MAP)
        sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MP)
        
    #total_keep_endpoints = np.concatenate([np.array(list(v.values())).reshape(-1,3) for v in limb_to_endpoints_must_keep_list])
    total_keep_endpoints = []
    for entry in limb_to_endpoints_must_keep_list:
        for k,v in entry.items():
            total_keep_endpoints.append(v)
            
    if len(total_keep_endpoints)>0:
        total_keep_endpoints = np.vstack(total_keep_endpoints)
        
    total_keep_endpoints = np.array(total_keep_endpoints)
    
    # Only want to perform this step if both MP and MAP pieces
    if len(limb_correspondence_MAP)>0 and len(limb_correspondence_MP)>0:

        # -------------- Part 11: Getting Sublimb Mesh and Skeletons and Gets connectivitiy by Mesh -------#
        # -------------(filtering connections to only MP to MAP edges)--------------- #

        # ---- Doing the mesh connectivity ---------#
        sublimb_meshes_MP = []
        sublimb_skeletons_MP = []

        for sublimb_key,sublimb_v in limb_correspondence_MP.items():
            sublimb_meshes_MP.append(tu.combine_meshes([branch_v["branch_mesh"] for branch_v in sublimb_v.values()]))
            sublimb_skeletons_MP.append(sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in sublimb_v.values()]))



        sublimb_meshes_MAP = []
        sublimb_skeletons_MAP = []


        for sublimb_key,sublimb_v in limb_correspondence_MAP.items():
            sublimb_meshes_MAP.append(tu.combine_meshes([branch_v["branch_mesh"] for branch_v in sublimb_v.values()]))
            sublimb_skeletons_MAP.append(sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in sublimb_v.values()]))

        sublimb_skeletons_MP_saved = copy.deepcopy(sublimb_skeletons_MP)
        sublimb_skeletons_MAP_saved = copy.deepcopy(sublimb_skeletons_MAP)

        connectivity_type = "edges"
        for i in range(0,2):
            mesh_conn,mesh_conn_vertex_groups = tu.mesh_list_connectivity(meshes = sublimb_meshes_MP + sublimb_meshes_MAP,
                                                main_mesh = limb_mesh_mparty,
                                                connectivity=connectivity_type,
                                                min_common_vertices=1,
                                                return_vertex_connection_groups=True,
                                                return_largest_vertex_connection_group=True,
                                                print_flag = False)
            mesh_conn_old = copy.deepcopy(mesh_conn)



            #check that every MAP piece mapped to a MP piece
            mesh_conn_filt = []
            mesh_conn_vertex_groups_filt = []
            for j,(m1,m2) in enumerate(mesh_conn):
                if m1 < len(sublimb_meshes_MP) and m2 >=len(sublimb_meshes_MP):
                    mesh_conn_filt.append([m1,m2])
                    mesh_conn_vertex_groups_filt.append(mesh_conn_vertex_groups[j])
                else:
                    print(f"Edge {(m1,m2)} was not kept")
            mesh_conn_filt = np.array(mesh_conn_filt)

            mesh_conn = mesh_conn_filt
            mesh_conn_vertex_groups = mesh_conn_vertex_groups_filt

            #check that the mapping should create only one connected component
            G = nx.from_edgelist(mesh_conn)



            try:
                if len(G) != len(sublimb_meshes_MP) + len(sublimb_meshes_MAP):
                    raise Exception("Number of nodes in mesh connectivity graph is not equal to number of  MAP and MP sublimbs")

                connect_comp = list(nx.connected_components(G))
                if len(connect_comp)>1:
                    raise Exception(f"Mesh connectivity was not one component, instead it was ({len(connect_comp)}): {connect_comp} ")
            except:
                
                if connectivity_type == "vertices":
                    print(f"mesh_conn_filt = {mesh_conn_filt}")
                    print(f"mesh_conn_old = {mesh_conn_old}")
                    mesh_conn_adjusted = np.vstack([mesh_conn[:,0],mesh_conn[:,1]-len(sublimb_meshes_MP)]).T
                    print(f"mesh_conn_adjusted = {mesh_conn_adjusted}")
                    print(f"len(sublimb_meshes_MP) = {len(sublimb_meshes_MP)}")
                    print(f"len(sublimb_meshes_MAP) = {len(sublimb_meshes_MAP)}")
                    meshes = sublimb_meshes_MP + sublimb_meshes_MAP
                    #su.compressed_pickle(meshes,"meshes")
                    su.compressed_pickle(sublimb_meshes_MP,"sublimb_meshes_MP")
                    su.compressed_pickle(sublimb_meshes_MAP,"sublimb_meshes_MAP")
                    su.compressed_pickle(limb_mesh_mparty,"limb_mesh_mparty")
                    su.compressed_pickle(sublimb_skeletons_MP,"sublimb_skeletons_MP")
                    su.compressed_pickle(sublimb_skeletons_MAP,"sublimb_skeletons_MAP")




                    raise Exception("Something went wrong in the connectivity")
                else:
                    print(f"Failed on connection type {connectivity_type} ")
                    connectivity_type = "vertices"
                    print(f"so changing type to {connectivity_type}")
            else:
                print(f"Successful mesh connectivity with type {connectivity_type}")
                break


        #adjust the connection indices for MP and MAP indices
        mesh_conn_adjusted = np.vstack([mesh_conn[:,0],mesh_conn[:,1]-len(sublimb_meshes_MP)]).T






        """
        Pseudocode:
        For each connection edge:
            For each vertex connection group:
                1) Get the endpoint vertices of the MP skeleton
                2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)
                3) Find the closest skeletal point on MAP pairing (MAP stitch) 
                4) Find the branches that have that MAP stitch point:
                5A) If the number of branches corresponding to stitch point is multipled
                    --> then we are stitching at a branching oint
                    i) Just add the skeletal segment from MP_stitch to MAP stitch to the MP skeletal segment
                    ii) 

        """



        # -------------- STITCHING PHASE -------#
        stitch_counter = 0
        all_map_stitch_points = []
        for (MP_idx,MAP_idx),v_g in zip(mesh_conn_adjusted,mesh_conn_vertex_groups):
            print(f"\n---- Working on {(MP_idx,MAP_idx)} connection-----")

            """
            This old way of getting the endpoints was not good because could possibly just need
            a stitching done between original branch junction

            skeleton_MP_graph = sk.convert_skeleton_to_graph(curr_skeleton_MP)
            endpoint_nodes = xu.get_nodes_of_degree_k(skeleton_MP_graph,1)
            endpoint_nodes_coordinates = xu.get_node_attributes(skeleton_MP_graph,node_list=endpoint_nodes)
            """


            # -------------- Part 12: Find the MP and MAP stitching point and branches that contain the stitching point-------#

            """  OLD WAY THAT ALLOWED STITICHING POINTS TO NOT BE CONNECTED AT THE CONNECTING BRANCHES
            #getting the skeletons that should be stitched
            curr_skeleton_MP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MP[MP_idx].values()])
            curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])

            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_MP[MP_idx].keys()))]
            endpoint_nodes_coordinates = np.array([sk.find_branch_endpoints(k) for k in curr_MP_branch_skeletons])
            endpoint_nodes_coordinates = np.unique(endpoint_nodes_coordinates.reshape(-1,3),axis=0)

            #2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)
            av_vert = np.mean(v_g,axis=0)
            winning_vertex = endpoint_nodes_coordinates[np.argmin(np.linalg.norm(endpoint_nodes_coordinates-av_vert,axis=1))]
            print(f"winning_vertex = {winning_vertex}")


            #2b) Find the branch points where the winning vertex is located
            MP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MP_branch_skeletons,
                current_coordinate = winning_vertex
            )
            print(f"MP_branches_with_stitch_point = {MP_branches_with_stitch_point}")


            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP.reshape(-1,3),axis=0)
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-winning_vertex,axis=1))]


            #3b) Consider if the stitch point is close enough to end or branch node in skeleton:
            # and if so then reassign
            if move_MAP_stitch_to_end_or_branch:
                MAP_stitch_point_new,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=curr_skeleton_MAP,
                                                        coordinate=MAP_stitch_point,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=True

                                                        )
                MAP_stitch_point=MAP_stitch_point_new


            #4) Find the branches that have that MAP stitch point:
            curr_MAP_branch_skeletons = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"]
                                             for k in np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))]

            MAP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MAP_branch_skeletons,
                current_coordinate = MAP_stitch_point
            )



            MAP_stitch_point_on_end_or_branch = False
            if len(MAP_branches_with_stitch_point)>1:
                MAP_stitch_point_on_end_or_branch = True
            elif len(MAP_branches_with_stitch_point)==1:
                if len(nu.matching_rows(sk.find_branch_endpoints(curr_MAP_branch_skeletons[MAP_branches_with_stitch_point[0]]),
                                        MAP_stitch_point))>0:
                    MAP_stitch_point_on_end_or_branch=True
            else:
                raise Exception("No matching MAP values")

        """

            #*****should only get branches that are touching....****

            #getting the skeletons that should be stitched
            curr_skeleton_MP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MP[MP_idx].values()])
            curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])


            av_vert = np.mean(v_g,axis=0)

            # ---------------- Doing the MAP part first -------------- #
            """
            The previous way did not ensure that the MAP point found will have a branch mesh that is touching the border vertices

            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP.reshape(-1,3),axis=0)

            #this does not guarentee that the MAP branch associated with the MAP stitch point is touching the border group
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]
            """

            # -------------- 11/9 NEW METHOD FOR FINDING MAP STITCH POINT ------------ #
            o_keys = np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))
            curr_MAP_branch_meshes = np.array([limb_correspondence_MAP[MAP_idx][k]["branch_mesh"]
                                             for k in o_keys])
            curr_MAP_branch_skeletons = np.array([limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"]
                                             for k in o_keys])

            MAP_pieces_idx_touching_border = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MAP_branch_meshes,
                                           nullifying_points=v_g,
                                            filter_away=False,
                                           distance_threshold=min_distance_threshold,
                                           return_indices=True)

            MAP_branches_considered = curr_MAP_branch_skeletons[MAP_pieces_idx_touching_border]
            curr_skeleton_MAP_for_stitch = sk.stack_skeletons(MAP_branches_considered)

            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP_for_stitch.reshape(-1,3),axis=0)

            """
            #------- OLD WAY: this does not guarentee that the MAP branch associated with the MAP stitch point is touching the border group
            #MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]

            # ------- 1/1/21 Change to make sure never stitches to soma connecting point ----
            Pseudocode: 
            1) Get all the closest coordinates and sort in order of distance
            2) Iterate through the top coordinates:
            - check if not in the endpoints
            a. if not --> make that the winning MAP stitch point
            b. if not --> continue to next

            3) if get to end and dont have winning coordinate then error
            closest_MAP_coords = MAP_skeleton_coords[np.argsort(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]

            MAP_stitch_point = None
            for c_map in closest_MAP_coords:

                if len(nu.matching_rows(total_keep_endpoints,c_map))==0:
                    MAP_stitch_point = c_map
                    break


            if MAP_stitch_point is None:
                raise Exception('Could not find a MAP_stitch_point that was not a keep_endpoint that was touching the soma')

            
            
            # --------- 1/2/21: this functionality is now taken care of inside move point
            """

            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]

            # --------- 11/13: Making so could possibly stitch to another point that was already stitched to
            curr_br_endpts = np.array([sk.find_branch_endpoints(k) for k in MAP_branches_considered]).reshape(-1,3)
            curr_br_endpts_unique = np.unique(curr_br_endpts,axis=0)



            #3b) Consider if the stitch point is close enough to end or branch node in skeleton:
            # and if so then reassign
            if move_MAP_stitch_to_end_or_branch:
                MAP_stitch_point_new,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=curr_skeleton_MAP,
                                                        coordinate=MAP_stitch_point,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=False,
                                                        possible_node_coordinates=curr_br_endpts_unique,
                                                        excluded_node_coordinates=total_keep_endpoints,
                                                        )
                MAP_stitch_point=MAP_stitch_point_new


            #4) Find the branches that have that MAP stitch point:

            MAP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MAP_branch_skeletons,
                current_coordinate = MAP_stitch_point
            )



            MAP_stitch_point_on_end_or_branch = False
            if len(MAP_branches_with_stitch_point)>1:
                MAP_stitch_point_on_end_or_branch = True
            elif len(MAP_branches_with_stitch_point)==1:
                if len(nu.matching_rows(sk.find_branch_endpoints(curr_MAP_branch_skeletons[MAP_branches_with_stitch_point[0]]),
                                        MAP_stitch_point))>0:
                    MAP_stitch_point_on_end_or_branch=True
            else:
                raise Exception("No matching MAP values")

            #add the map stitch point to the history
            all_map_stitch_points.append(MAP_stitch_point)

            # ---------------- Doing the MP Part --------------------- #



            ord_keys = np.sort(list(limb_correspondence_MP[MP_idx].keys()))
            curr_MP_branch_meshes = [limb_correspondence_MP[MP_idx][k]["branch_mesh"] for k in ord_keys]



            """ old way of filtering MP pieces just to those touching the MAP, but just want the ones touching the connection group

            MAP_meshes_with_stitch_point = tu.combine_meshes([limb_correspondence_MAP[MAP_idx][k]["branch_mesh"] for k in MAP_branches_with_stitch_point])

            conn = tu.mesh_pieces_connectivity(main_mesh=limb_mesh_mparty,
                                       central_piece=MAP_meshes_with_stitch_point,
                                       periphery_pieces=curr_MP_branch_meshes)
            """
            # 11/9 Addition: New way that filters meshes by their touching of the vertex connection group (this could possibly be an empty group)
            conn = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MP_branch_meshes,
                                           nullifying_points=v_g,
                                            filter_away=False,
                                           distance_threshold=min_distance_threshold,
                                           return_indices=True)

            if len(conn) == 0:
                print("Connectivity was 0 for the MP mesh groups touching the vertex group so not restricting by that anymore")
                sk_conn = np.arange(0,len(curr_MP_branch_meshes))
            else:
                sk_conn = conn

            print(f"sk_conn = {sk_conn}")
            print(f"conn = {conn}")


            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in sk_conn]
            endpoint_nodes_coordinates = np.array([sk.find_branch_endpoints(k) for k in curr_MP_branch_skeletons])
            endpoint_nodes_coordinates = np.unique(endpoint_nodes_coordinates.reshape(-1,3),axis=0)
            
            """ ---------- 1 /5: Take out the possible endpoints --------------------"""
            if prevent_MP_starter_branch_stitches:
                endpoint_nodes_coordinates = nu.setdiff2d(endpoint_nodes_coordinates,total_keep_endpoints)


            #2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)

            winning_vertex = endpoint_nodes_coordinates[np.argmin(np.linalg.norm(endpoint_nodes_coordinates-av_vert,axis=1))]
            print(f"winning_vertex = {winning_vertex}")


            #2b) Find the branch points where the winning vertex is located
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_MP[MP_idx].keys()))]
            MP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MP_branch_skeletons,
                current_coordinate = winning_vertex
            )
            print(f"MP_branches_with_stitch_point = {MP_branches_with_stitch_point}")



            print(f"MAP_branches_with_stitch_point = {MAP_branches_with_stitch_point}")
            print(f"MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


            # -------- 11/13 addition: Will see if the MP stitch point was already a MAP stitch point ---- #
            if len(nu.matching_rows(np.array(all_map_stitch_points),winning_vertex)) > 0:
                keep_MP_stitch_static = True
            else:
                keep_MP_stitch_static = False





            # ------------------------- This part does the stitching -------------------- #


            """
            Pseudocode:
            1) For all MP branches
                a) Get neighbor coordinates to MP stitch points
                b) Delete the MP Stitch points on each 
                c) Add skeleton segment from neighbor to MAP stitch point
            2) Get skeletons and meshes from MP and MAP pieces
            3) Run mesh correspondence to get new meshes and mesh_idx and widths
            4a) If MAP_stitch_point_on_end_or_branch is False
            - Delete the old MAP branch parts and replace with new MAP ones
            4b) Revise the meshes,  mesh_idx, and widths of the MAP pieces
            5) Revise the meshes,  mesh_idx, and widths of the MP pieces


            """

            # -------------- Part 13: Will Adjust the MP branches that have the stitch point so extends to the MAP stitch point -------#
            curr_MP_sk = []
            for b_idx in MP_branches_with_stitch_point:
                if not keep_MP_stitch_static:
                    #a) Get neighbor coordinates to MP stitch points
                    MP_stitch_branch_graph = sk.convert_skeleton_to_graph(curr_MP_branch_skeletons[b_idx])
                    stitch_node = xu.get_nodes_with_attributes_dict(MP_stitch_branch_graph,dict(coordinates=winning_vertex))[0]
                    stitch_neighbors = xu.get_neighbors(MP_stitch_branch_graph,stitch_node)

                    if len(stitch_neighbors) != 1:
                        raise Exception("Not just one neighbor for stitch point of MP branch")
                    keep_neighbor = stitch_neighbors[0]  
                    keep_neighbor_coordinates = xu.get_node_attributes(MP_stitch_branch_graph,node_list=[keep_neighbor])[0]

                    #b) Delete the MP Stitch points on each 
                    MP_stitch_branch_graph.remove_node(stitch_node)

                    """ Old way that does not do smoothing

                    #c) Add skeleton segment from neighbor to MAP stitch point
                    new_node_name = np.max(MP_stitch_branch_graph.nodes())+1

                    MP_stitch_branch_graph.add_nodes_from([(int(new_node_name),{"coordinates":MAP_stitch_point})])
                    MP_stitch_branch_graph.add_weighted_edges_from([(keep_neighbor,new_node_name,np.linalg.norm(MAP_stitch_point - keep_neighbor_coordinates))])

                    new_MP_skeleton = sk.convert_graph_to_skeleton(MP_stitch_branch_graph)

                    """
                    try:
                        if len(MP_stitch_branch_graph)>1:
                            new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                                            skeleton_stitch_point=keep_neighbor_coordinates,
                                                             new_stitch_point=MAP_stitch_point)
                        else:
                            print("Not even attempting smoothing segment because once keep_neighbor_coordinates")
                            new_MP_skeleton = np.vstack([keep_neighbor_coordinates,MAP_stitch_point]).reshape(-1,2,3)
                    except:
                        su.compressed_pickle(MP_stitch_branch_graph,"MP_stitch_branch_graph")
                        su.compressed_pickle(keep_neighbor_coordinates,"keep_neighbor_coordinates")
                        su.compressed_pickle(MAP_stitch_point,"MAP_stitch_point")


                        raise Exception("Something went wrong with add_and_smooth_segment_to_branch")





                    #smooth over the new skeleton
                    new_MP_skeleton_smooth = sk.resize_skeleton_branch(new_MP_skeleton,
                                                                      segment_width=meshparty_segment_size)

                    curr_MP_sk.append(new_MP_skeleton_smooth)
                else:
                    print(f"Not adjusting MP skeletons because keep_MP_stitch_static = {keep_MP_stitch_static}")
                    curr_MP_sk.append(curr_MP_branch_skeletons[b_idx])



            #2) Get skeletons and meshes from MP and MAP pieces
            curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_branches_with_stitch_point]

            #2.1) Going to break up the MAP skeleton if need be
            """
            Pseudocode:
            a) check to see if it needs to be broken up
            If it does:
            b) Convert the skeleton into a graph
            c) Find the node of the MAP stitch point (where need to do the breaking)
            d) Find the degree one nodes
            e) For each degree one node:
            - Find shortest path from stitch node to end node
            - get a subgraph from that path
            - convert graph to a skeleton and save as new skeletons

            """
            # -------------- Part 14: Breaks Up MAP skeleton into 2 pieces if Needs (because MAP stitch point not on endpoint or branch point)  -------#

            #a) check to see if it needs to be broken up
            cut_flag = False
            if not MAP_stitch_point_on_end_or_branch:
                if len(curr_MAP_sk) > 1:
                    raise Exception(f"There was more than one skeleton for MAP skeletons even though MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


                skeleton_to_cut = curr_MAP_sk[0]
                curr_MAP_sk = sk.cut_skeleton_at_coordinate(skeleton=skeleton_to_cut,
                                                            cut_coordinate=MAP_stitch_point)
                cut_flag=True


            # ------ 11/13 Addition: need to adjust the MAP points if have to keep MP static
            if keep_MP_stitch_static:
                curr_MAP_sk_final = []
                for map_skel in curr_MAP_sk:
                    #a) Get neighbor coordinates to MP stitch points
                    MP_stitch_branch_graph = sk.convert_skeleton_to_graph(map_skel)
                    stitch_node = xu.get_nodes_with_attributes_dict(MP_stitch_branch_graph,dict(coordinates=MAP_stitch_point))[0]
                    stitch_neighbors = xu.get_neighbors(MP_stitch_branch_graph,stitch_node)

                    if len(stitch_neighbors) != 1:
                        raise Exception("Not just one neighbor for stitch point of MP branch")
                    keep_neighbor = stitch_neighbors[0]  
                    keep_neighbor_coordinates = xu.get_node_attributes(MP_stitch_branch_graph,node_list=[keep_neighbor])[0]

                    #b) Delete the MP Stitch points on each 
                    MP_stitch_branch_graph.remove_node(stitch_node)

                    """ Old way that does not do smoothing

                    #c) Add skeleton segment from neighbor to MAP stitch point
                    new_node_name = np.max(MP_stitch_branch_graph.nodes())+1

                    MP_stitch_branch_graph.add_nodes_from([(int(new_node_name),{"coordinates":MAP_stitch_point})])
                    MP_stitch_branch_graph.add_weighted_edges_from([(keep_neighbor,new_node_name,np.linalg.norm(MAP_stitch_point - keep_neighbor_coordinates))])

                    new_MP_skeleton = sk.convert_graph_to_skeleton(MP_stitch_branch_graph)

                    """
                    try:
                        if len(MP_stitch_branch_graph)>1:
                            new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                                            skeleton_stitch_point=keep_neighbor_coordinates,
                                                             new_stitch_point=winning_vertex)
                        else:
                            print("Not even attempting smoothing segment because once keep_neighbor_coordinates")
                            new_MP_skeleton = np.vstack([keep_neighbor_coordinates,MAP_stitch_point]).reshape(-1,2,3)
                    except:
                        su.compressed_pickle(MP_stitch_branch_graph,"MP_stitch_branch_graph")
                        su.compressed_pickle(keep_neighbor_coordinates,"keep_neighbor_coordinates")
                        su.compressed_pickle(winning_vertex,"winning_vertex")


                        raise Exception("Something went wrong with add_and_smooth_segment_to_branch")





                    #smooth over the new skeleton
                    new_MP_skeleton_smooth = sk.resize_skeleton_branch(new_MP_skeleton,
                                                                      segment_width=meshparty_segment_size)

                    curr_MAP_sk_final.append(new_MP_skeleton_smooth)
                curr_MAP_sk = copy.deepcopy(curr_MAP_sk_final)



            # -------------- Part 15: Gets all of the skeletons and Mesh to divide u and does mesh correspondence -------#
            # ------------- revise IDX so still references the whole limb mesh -----------#

            # -------------- 11/10 Addition accounting for not all MAP pieces always touching each other --------------------#
            if len(MAP_branches_with_stitch_point) > 1:
                print("\nRevising the MAP pieces index:")
                print(f"MAP_pieces_idx_touching_border = {MAP_pieces_idx_touching_border}, MAP_branches_with_stitch_point = {MAP_branches_with_stitch_point}")
                MAP_pieces_for_correspondence = nu.intersect1d(MAP_pieces_idx_touching_border,MAP_branches_with_stitch_point)
                print(f"MAP_pieces_for_correspondence = {MAP_pieces_for_correspondence}")
                curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_pieces_for_correspondence]
            else:
                MAP_pieces_for_correspondence = MAP_branches_with_stitch_point

            curr_MAP_meshes_idx = [limb_correspondence_MAP[MAP_idx][k]["branch_face_idx"] for k in MAP_pieces_for_correspondence]

            # Have to adjust based on if the skeleton were split

            if cut_flag:
                #Then it was cut and have to do mesh correspondence to find what label to cut
                if len(curr_MAP_meshes_idx) > 1:
                    raise Exception("MAP_pieces_for_correspondence was longer than 1 and cut flag was set")
                pre_stitch_mesh_idx = curr_MAP_meshes_idx[0]
                pre_stitch_mesh = limb_mesh_mparty.submesh([pre_stitch_mesh_idx],append=True,repair=False)
                local_correspondnece_stitch = mesh_correspondence_first_pass(mesh=pre_stitch_mesh,
                                          skeleton_branches=curr_MAP_sk)
                local_correspondence_stitch_revised_MAP = correspondence_1_to_1(mesh=pre_stitch_mesh,
                                                            local_correspondence=local_correspondnece_stitch,
                                                            curr_limb_endpoints_must_keep=None,
                                                            curr_soma_to_piece_touching_vertices=None)

#                 curr_MAP_meshes_idx = [pre_stitch_mesh_idx[local_correspondence_stitch_revised_MAP[nn]["branch_face_idx"]] for 
#                                                nn in local_correspondence_stitch_revised_MAP.keys()]
                
                #Need to readjust the mesh correspondence idx
                for k,v in local_correspondence_stitch_revised_MAP.items():
                    local_correspondence_stitch_revised_MAP[k]["branch_face_idx"] = pre_stitch_mesh_idx[local_correspondence_stitch_revised_MAP[k]["branch_face_idx"]]
                    
                curr_MAP_meshes_idx = [v["branch_face_idx"] for v in local_correspondence_stitch_revised_MAP.values()]
            else:
                local_correspondence_stitch_revised_MAP = dict([(gg,limb_correspondence_MAP[MAP_idx][kk]) for gg,kk in enumerate(MAP_pieces_for_correspondence)])
                
                for gg,kk in enumerate(MAP_pieces_for_correspondence):
                    local_correspondence_stitch_revised_MAP[gg]["branch_skeleton"] = curr_MAP_sk[gg]
                    


            #To make sure that the MAP never gives up ground on the labels
            must_keep_labels_MAP = dict()
            must_keep_counter = 0
            for kk,b_idx in enumerate(curr_MAP_meshes_idx):
                #must_keep_labels_MAP.update(dict([(ii,kk) for ii in range(must_keep_counter,must_keep_counter+len(b_idx))]))
                must_keep_labels_MAP[kk] = np.arange(must_keep_counter,must_keep_counter+len(b_idx))
                must_keep_counter += len(b_idx)



            #this is where should send only the MP that apply
            MP_branches_for_correspondence,conn_idx,MP_branches_with_stitch_point_idx = nu.intersect1d(conn,MP_branches_with_stitch_point,return_indices=True)

            curr_MP_meshes_idx = [limb_correspondence_MP[MP_idx][k]["branch_face_idx"] for k in MP_branches_for_correspondence]
            curr_MP_sk_for_correspondence = [curr_MP_sk[zz] for zz in MP_branches_with_stitch_point_idx]

            stitching_mesh_idx = np.concatenate(curr_MAP_meshes_idx + curr_MP_meshes_idx)
            stitching_mesh = limb_mesh_mparty.submesh([stitching_mesh_idx],append=True,repair=False)
            stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk_for_correspondence
            """

            ****** NEED TO GET THE RIGHT MESH TO RUN HE IDX ON SO GETS A GOOD MESH (CAN'T BE LIMB_MESH_MPARTY)
            BUT MUST BE THE ORIGINAL MAP MESH

            mesh_pieces_for_MAP
            sublimb_meshes_MP

            mesh_pieces_for_MAP_face_idx
            sublimb_meshes_MP_face_idx

            stitching_mesh = tu.combine_meshes(curr_MAP_meshes + curr_MP_meshes)
            stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk

            """
            
            # ******************************** this is where should do thing about no mesh correspondence ***************** #

            # -------- 12/22: Trying to do the re-correspondence but if doesn't work then just resort to old one --------- #

            try:
                
                #3) Run mesh correspondence to get new meshes and mesh_idx and widths
                local_correspondnece_stitch = mesh_correspondence_first_pass(mesh=stitching_mesh,
                                              skeleton_branches=stitching_skeleton_branches)

                local_correspondence_stitch_revised = correspondence_1_to_1(mesh=stitching_mesh,
                                                            local_correspondence=local_correspondnece_stitch,
                                                            curr_limb_endpoints_must_keep=None,
                                                            curr_soma_to_piece_touching_vertices=None,
                                                            must_keep_labels=must_keep_labels_MAP)
                
                #Need to readjust the mesh correspondence idx
                for k,v in local_correspondence_stitch_revised.items():
                    local_correspondence_stitch_revised[k]["branch_face_idx"] = stitching_mesh_idx[local_correspondence_stitch_revised[k]["branch_face_idx"]]
            except:
                print("Errored in 1 to 1 correspondence in stitching so just reverting to the original mesh assignments")
                # Setting the correspondence manually because the adaptive way did not work
                local_counter = 0
                local_correspondence_stitch_revised = dict()
                
                # setting the MAP parts (the new skeletons have already been adjusted)
                for k in local_correspondence_stitch_revised_MAP:
                    local_correspondence_stitch_revised[local_counter] = local_correspondence_stitch_revised_MAP[k]
                    local_counter += 1

                # setting the MP parts (the new skeletons have not been adjusted yet so adjusting them here)
                for mp_idx, k in enumerate(MP_branches_for_correspondence):
                    local_correspondence_stitch_revised[local_counter] = limb_correspondence_MP[MP_idx][k] 
                    local_correspondence_stitch_revised[local_counter]["branch_skeleton"] = curr_MP_sk[mp_idx]
                    local_counter += 1
                
                
#                 su.compressed_pickle(stitching_skeleton_branches,"stitching_skeleton_branches")
#                 su.compressed_pickle(stitching_mesh,"stitching_mesh")
#                 su.compressed_pickle(local_correspondnece_stitch,"local_correspondnece_stitch")
#                 su.compressed_pickle(must_keep_labels_MAP,"must_keep_labels_MAP")
                
#                 raise Exception("Something went wrong with 1 to 1 correspondence")


            




            # -------------- Part 16: Overwrite old branch entries (and add on one new to MAP if required a split) -------#


            #4a) If MAP_stitch_point_on_end_or_branch is False
            #- Delete the old MAP branch parts and replace with new MAP ones
            if not MAP_stitch_point_on_end_or_branch:
                print("Deleting branches from dictionary")
                del limb_correspondence_MAP[MAP_idx][MAP_branches_with_stitch_point[0]]
                #adding the two new branches created from the stitching
                limb_correspondence_MAP[MAP_idx][MAP_branches_with_stitch_point[0]] = local_correspondence_stitch_revised[0]
                limb_correspondence_MAP[MAP_idx][np.max(list(limb_correspondence_MAP[MAP_idx].keys()))+1] = local_correspondence_stitch_revised[1]

                #have to reorder the keys
                #limb_correspondence_MAP[MAP_idx] = dict([(k,limb_correspondence_MAP[MAP_idx][k]) for k in np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))])
                limb_correspondence_MAP[MAP_idx] = gu.order_dict_by_keys(limb_correspondence_MAP[MAP_idx])

            else: #4b) Revise the meshes,  mesh_idx, and widths of the MAP pieces if weren't broken up
                for j,curr_MAP_idx_fixed in enumerate(MAP_pieces_for_correspondence): 
                    limb_correspondence_MAP[MAP_idx][curr_MAP_idx_fixed] = local_correspondence_stitch_revised[j]
                #want to update all of the skeletons just in case was altered by keep_MP_stitch_static and not included in correspondence
                if keep_MP_stitch_static:
                    if len(MAP_branches_with_stitch_point) != len(curr_MAP_sk_final):
                        raise Exception("MAP_branches_with_stitch_point not same size as curr_MAP_sk_final")
                    for gg,map_idx_curr in enumerate(MAP_branches_with_stitch_point):
                        limb_correspondence_MAP[MAP_idx][map_idx_curr]["branch_skeleton"] = curr_MAP_sk_final[gg]


            for j,curr_MP_idx_fixed in enumerate(MP_branches_for_correspondence): #************** right here just need to make only the ones that applied
                limb_correspondence_MP[MP_idx][curr_MP_idx_fixed] = local_correspondence_stitch_revised[j+len(curr_MAP_sk)]


            #5b) Fixing the branch skeletons that were not included in the correspondence
            MP_leftover,MP_leftover_idx = nu.setdiff1d(MP_branches_with_stitch_point,MP_branches_for_correspondence)
            print(f"MP_branches_with_stitch_point= {MP_branches_with_stitch_point}")
            print(f"MP_branches_for_correspondence = {MP_branches_for_correspondence}")
            print(f"MP_leftover = {MP_leftover}, MP_leftover_idx = {MP_leftover_idx}")

            for curr_MP_leftover,curr_MP_leftover_idx in zip(MP_leftover,MP_leftover_idx):
                limb_correspondence_MP[MP_idx][curr_MP_leftover]["branch_skeleton"] = curr_MP_sk[curr_MP_leftover_idx]


            print(f" Finished with {(MP_idx,MAP_idx)} \n\n\n")
            stitch_counter += 1
    #         if cut_flag:
    #             raise Exception("Cut flag was activated")

            if check_correspondence_branches:
                sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MAP[MAP_idx])
                sk.check_correspondence_branches_have_2_endpoints(limb_correspondence_MP[MP_idx])
                
#             su.compressed_pickle(limb_correspondence_MAP,f"limb_correspondence_MAP_{MAP_idx}_{MP_idx}")
#             su.compressed_pickle(limb_correspondence_MP,f"limb_correspondence_MP_{MAP_idx}_{MP_idx}")


    else:
        print("There were not both MAP and MP pieces so skipping the stitch resolving phase")

    print(f"Time for decomp of Limb = {time.time() - curr_limb_time}")
    #     # ------------- Saving the MAP and MP Decompositions ---------------- #
    #     proper_limb_mesh_correspondence_MAP[curr_limb_idx] = limb_correspondence_MAP
    #     proper_limb_mesh_correspondence_MP[curr_limb_idx] = limb_correspondence_MP






    # -------------- Part 17: Grouping the MP and MAP Correspondence into one correspondence dictionary -------#
    limb_correspondence_individual = dict()
    counter = 0

    for sublimb_idx,sublimb_branches in limb_correspondence_MAP.items():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter]= branch_dict
            counter += 1
    for sublimb_idx,sublimb_branches in limb_correspondence_MP.items():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter]= branch_dict
            counter += 1


    #info that may be used for concept networks
    network_starting_info = dict(
                touching_verts_list = limb_to_soma_touching_vertices_list,
                endpoints_must_keep = limb_to_endpoints_must_keep_list
    )

    
    
    
    
    
    # -------------- Part 18: 11-17 Addition that filters the network starting info into a more clean presentation ------------ #
    """
    Pseudocode: 
    1) Rearrange the network starting info into a ditionary mapping
      soma_idx --> branch_broder_group --> list of dict(touching_vertices,endpoint)

    2) iterate through all the somas and border vertex groups
    a. filter to only those with an endpoint that is on a branch of the skeleton
    b1: If 1 --> then keep that one
    b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
    b3: If 0 --> find the best available soma extending branch endpoint

    """

    # Part 1: Rearrange network info


    t_verts_list_total,enpts_list_total = network_starting_info.values()
    network_starting_info_revised = dict()
    for j,(v_list_dict,enpts_list_dict) in enumerate(zip(t_verts_list_total,enpts_list_total)):
        #print(f"---- Working on {j} -----")
    #     print(v_list_dict)
    #     print(enpts_list_dict)
        if set(list(v_list_dict.keys())) != set(list(enpts_list_dict)):
            raise Exception("Soma keys not match for touching vertices and endpoints")
        for sm_idx in v_list_dict.keys():
            v_list_soma = v_list_dict[sm_idx]
            endpt_soma = enpts_list_dict[sm_idx]
            if len(v_list_soma) != len(endpt_soma):
                raise Exception(f"touching vertices list and endpoint list not match size for soma {sm_idx}")

            all_border_vertex_groups = soma_touching_vertices_dict[sm_idx]

            for v_l,endpt in zip(v_list_soma,endpt_soma):

                matching_border_group  = []
                for i,curr_border_group in enumerate(all_border_vertex_groups):
                    if nu.test_matching_vertices_in_lists(curr_border_group,v_l,verbose=True):
                        matching_border_group.append(i)

                if len(matching_border_group) == 0 or len(matching_border_group)>1:
                    raise Exception(f"Matching border groups was not exactly 1: {matching_border_group}")

                winning_border_group = matching_border_group[0]

                if sm_idx not in network_starting_info_revised.keys():
                    network_starting_info_revised[sm_idx] = dict()

                if winning_border_group not in network_starting_info_revised[sm_idx].keys():
                    network_starting_info_revised[sm_idx][winning_border_group] = []
                network_starting_info_revised[sm_idx][winning_border_group].append(dict(touching_verts=v_l,endpoint=endpt))

    
    # Part 2 Filter
    """
    2) iterate through all the somas and border vertex groups
    a. filter to only those with an endpoint that is on a branch of the skeleton
    b1: If 1 --> then keep that one
    b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
    b3: If 0 --> find the best available soma extending branch endpoint

    Pseudocode for b3:
    i) get all meshes that touch the vertex group (and keep the vertices that overlap)
    --> error if none
    ii) Get all of the endpoints of all matching branches
    iii) Filter the endpoints to only those that are degree 1 in the overall skeleton
    --> if none then just keep all endpoints (AND THIS WILL CAUSE AN ERROR)
    iv) Find the closest viable endpoint to the mean of the boundary group
    v) save the overlap vertices and the winning endpoint as a dictionary

    """
    
    sorted_keys = np.sort(list(limb_correspondence_individual.keys()))
    curr_branches = [limb_correspondence_individual[k]["branch_skeleton"] for k in sorted_keys]
    curr_meshes = [limb_correspondence_individual[k]["branch_mesh"] for k in sorted_keys]

    network_starting_info_revised_cleaned = dict()
    for soma_idx in network_starting_info_revised.keys():
        network_starting_info_revised_cleaned[soma_idx] = dict()
        for bound_g_idx,endpoint_list in network_starting_info_revised[soma_idx].items():
            endpoint_list = np.array(endpoint_list)

            filter_on_skeleton_list = []
            for zz,endpt_dict in enumerate(endpoint_list):
                #a. filter to only those with an endpoint that is on a branch of the skeleton
                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=curr_branches,
                                                                            current_coordinate=endpt_dict["endpoint"])
                if len(sk_indices) > 0:
                    filter_on_skeleton_list.append(zz)

            endpoint_list_filt = endpoint_list[filter_on_skeleton_list]



            curr_border_group_coordinates = soma_touching_vertices_dict[soma_idx][bound_g_idx]
            boundary_mean = np.mean(curr_border_group_coordinates,axis=0)

            if len(endpoint_list_filt) == 1:
                print("Only one endpoint after filtering away the endpoints that are not on the skeleton")
                winning_dict = endpoint_list_filt[0]
            #b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
            elif len(endpoint_list_filt) > 1:
                print(f"MORE THAN one endpoint after filtering away the endpoints that are not on the skeleton: {len(endpoint_list_filt)}")
                viable_endpoints = [endpt_dict["endpoint"] for endpt_dict in endpoint_list_filt]


                distanes_from_mean = np.linalg.norm(viable_endpoints-boundary_mean,axis=1)
                winning_endpoint_idx = np.argmin(distanes_from_mean)
                winning_dict = endpoint_list_filt[winning_endpoint_idx]

            #if there was no clear winner
            else:
                """
                Pseudocode for no viable options:
                i) get all meshes that touch the vertex group (and keep the vertices that overlap)
                --> error if none
                ii) Get all of the endpoints of all matching branches
                iii) Filter the endpoints to only those that are degree 1 in the overall skeleton
                --> if none then just keep all endpoints
                iv) Find the closest viable endpoint to the mean of the boundary group
                v) save the overlap vertices and the winning endpoint as a dictionary


                """
                print("Having to find a new branch point")
                #i) get all meshes that touch the vertex group (and keep the vertices that overlap)
                mesh_indices_on_border = tu.filter_meshes_by_containing_coordinates(curr_meshes,
                                              nullifying_points=curr_border_group_coordinates,
                                              filter_away=False,
                                              distance_threshold=min_distance_threshold,
                                              return_indices=True)
                if len(mesh_indices_on_border) == 0:
                    raise Exception("There were no meshes that were touching the boundary group")

                total_skeleton_graph = sk.convert_skeleton_to_graph(sk.stack_skeletons(curr_branches))
                skeleton_branches_on_border = [k for n,k in enumerate(curr_branches) if n in mesh_indices_on_border]
                skeleton_branches_on_border_endpoints = np.array([sk.find_branch_endpoints(k) for k in skeleton_branches_on_border])



                viable_endpoints = []
                for enpt in skeleton_branches_on_border_endpoints.reshape(-1,3):
                    curr_enpt_node = xu.get_graph_node_by_coordinate(total_skeleton_graph,enpt,return_single_value=True)
                    curr_enpt_degree = xu.get_node_degree(total_skeleton_graph,curr_enpt_node)
                    #print(f"curr_enpt_degree = {curr_enpt_degree}")
                    if curr_enpt_degree == 1:
                        viable_endpoints.append(enpt)

                if len(viable_endpoints) == 0:
                    print("No branch endpoints were degree 1 so just using all endpoints")
                    viable_endpoints = skeleton_branches_on_border_endpoints.reshape(-1,3)

                distanes_from_mean = np.linalg.norm(viable_endpoints-boundary_mean,axis=1)
                winning_endpoint = viable_endpoints[np.argmin(distanes_from_mean)]


                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=curr_branches,
                                                                                        current_coordinate=winning_endpoint)

                winning_branch = np.intersect1d(mesh_indices_on_border,sk_indices)
                if len(winning_branch) == 0:
                    raise Exception("There was no winning branch for the creation of a new soma extending branch")
                else:
                    winning_branch_single = winning_branch[0]


                winning_touching_vertices = tu.filter_vertices_by_mesh(curr_meshes[winning_branch_single],curr_border_group_coordinates)
                winning_dict = dict(touching_verts=winning_touching_vertices,endpoint=winning_endpoint)








            network_starting_info_revised_cleaned[soma_idx][bound_g_idx] = winning_dict


    # -------------- Part 18: Filter the limb correspondence for any short stubs ------------ #
    if filter_end_nodes_from_correspondence:
        limb_correspondence_individual = pre.filter_limb_correspondence_for_end_nodes(limb_correspondence=limb_correspondence_individual,
                                                     mesh=limb_mesh_mparty,
                                                     starting_info=network_starting_info_revised_cleaned,
                                                    filter_end_node_length=filter_end_node_length,
                                                    error_on_no_starting_coordinates=error_on_no_starting_coordinates,
                                                    error_on_starting_coordinates_not_endnodes= prevent_MP_starter_branch_stitches

                                                    )

    
    
    
    
    
    
    
    
    
    
    
    
    if not return_concept_network:
        if return_concept_network_starting_info: #because may want to calculate the concept networks later
            return limb_correspondence_individual,network_starting_info_revised_cleaned
        else:
            return limb_correspondence_individual
    else:
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_individual,
                                                                        network_starting_info_revised_cleaned,
                                                                        run_concept_network_checks=run_concept_network_checks,
                                                                       )




    return limb_correspondence_individual,limb_to_soma_concept_networks


'''



def preprocess_neuron(
                mesh=None,
                mesh_file=None,
                segment_id=None,
                 description=None,
                sig_th_initial_split=100, #for significant splitting meshes in the intial mesh split
                limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
    
                filter_end_node_length=4000, #used in cleaning the skeleton during skeletonizations
                return_no_somas = False, #whether to error or to return an empty list for somas
    
                decomposition_type="meshafterparty",
                distance_by_mesh_center=True,
                meshparty_segment_size =100,
    
                meshparty_n_surface_downsampling = 2,

                somas=None, #the precomputed somas
                combine_close_skeleton_nodes = True,
                combine_close_skeleton_nodes_threshold=700,

                use_meshafterparty=True):
    pre_branch_connectivity = "edges"
    print(f"use_meshafterparty = {use_meshafterparty}")
    
    whole_processing_tiempo = time.time()


    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    This method includes the fusion

    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"


    if mesh is None:
        if mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = tu.load_mesh_no_processing(mesh_file)
    else:
        current_neuron = mesh
        
        
        
        
        
        
    # -------- Phase 1: Doing Soma Detection (if Not already done) ---------- #
    if somas is None:
        soma_mesh_list,run_time,total_soma_list_sdf = sm.extract_soma_center(segment_id,
                                                 current_neuron.vertices,
                                                 current_neuron.faces)
    else:
        soma_mesh_list,run_time,total_soma_list_sdf = somas
        print(f"Using pre-computed somas: soma_mesh_list = {soma_mesh_list}")

    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return_value= soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
        
        
        
        
        
    
    #--- Phase 2: getting the soma submeshes that are connected to each soma and identifiying those that aren't 
    # ------------------ (and eliminating any mesh pieces inside the soma) ------------------------

    # -------- 11/13 Addition: Will remove the inside nucleus --------- #
    interior_time = time.time()
    main_mesh_total,inside_nucleus_pieces = tu.remove_mesh_interior(current_neuron,return_removed_pieces=True,
                                                                   try_hole_close=False)
    print(f"Total time for removing interior = {time.time() - interior_time}")


    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_time = time.time()
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False,
                                connectivity=pre_branch_connectivity)
    print(f"Total time for splitting mesh = {time.time() - split_time}")

    print(f"# total split meshes = {len(split_meshes)}")

    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes)

    # filtering away any of the inside floating pieces: 
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]

    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        """
        *** want to save these pieces that are inside of the soma***
        """

        non_soma_touching_meshes,inside_pieces = sm.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split,
                                        return_inside_pieces = True)
    
    else:
        non_soma_touching_meshes = []
        inside_pieces=[]
    
    #adding in the nuclei center to the inside pieces
    inside_pieces += inside_nucleus_pieces


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    #     print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    #     print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"\n-----Before filtering away multiple disconneted soma pieces-----")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")
    
    # ------ 11/15 Addition: Part 2.b 

    """
    Pseudocode: 
    1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    2) Save all other meshes not the largest in 
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf


    """
    #1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    soma_containing_meshes_keys = np.array(list(soma_containing_meshes.keys()))
    soma_touching_meshes = np.array([split_meshes[k] for k in soma_containing_meshes_keys])
    largest_soma_touching_mesh_idx = soma_containing_meshes_keys[np.argmax([len(kk.faces) for kk in soma_touching_meshes])]

    #2) Save all other meshes not the largest in 
    not_processed_soma_containing_meshes_idx = np.setdiff1d(soma_containing_meshes_keys,[largest_soma_touching_mesh_idx])
    not_processed_soma_containing_meshes = [split_meshes[k] for k in not_processed_soma_containing_meshes_idx]
    print(f"Number of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)}")

    """
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf

    """

    somas_idx_to_process = soma_containing_meshes[largest_soma_touching_mesh_idx]
    soma_mesh_list = [soma_mesh_list[k] for k in somas_idx_to_process]

    soma_containing_meshes = {largest_soma_touching_mesh_idx:list(np.arange(0,len(soma_mesh_list)))}

    soma_touching_meshes = [split_meshes[largest_soma_touching_mesh_idx]]

    total_soma_list_sdf = total_soma_list_sdf[somas_idx_to_process]

    print(f"\n-----After filtering away multiple disconneted soma pieces-----")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")
    
    
    
    
    
    #--- Phase 3:  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces

    """
    for each soma touching mesh get the following:
    1) original soma meshes
    2) significant mesh pieces touching these somas
    3) The soma connectivity to each of the significant mesh pieces
    -- later will just translate the 


    Process: 

    1) Final all soma faces (through soma extraction and then soma original faces function)
    2) Subtact all soma faces from original mesh
    3) Find all significant mesh pieces
    4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all
       the available somas
    Conclusion: Will have connectivity map


    """

    soma_touching_mesh_data = dict()

    for z,(mesh_idx, soma_idxes) in enumerate(soma_containing_meshes.items()):
        soma_touching_mesh_data[z] = dict()
        print(f"\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx]

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        mesh_pieces_without_soma = sm.subtract_soma(current_soma_mesh_list,current_mesh,
                                                    significance_threshold=250,
                                                   connectivity=pre_branch_connectivity)
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)

        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        soma_meshes = current_mesh.submesh([soma_faces],append=True,repair=False)

        # finding the non-soma original faces
        non_soma_faces = tu.original_mesh_faces_map(current_mesh,soma_meshes,matching=False)
        non_soma_stacked_mesh = current_mesh.submesh([non_soma_faces],append=True,repair=False)

        print(f"Total time for Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False,
                                                            connectivity=pre_branch_connectivity)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes
        
        
        
        
        # 3) Find all significant mesh pieces
        """
        Pseudocode: 
        a) Iterate through all of the somas and get the pieces that are connected
        b) Concatenate all the results into one list and order
        c) Filter away the mesh pieces that aren't touching and add to the floating pieces
        
        """
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True,
                                                                             connectivity=pre_branch_connectivity)
        
        # a) Filter these down to only those touching the somas
        all_conneted_non_soma_pieces = []
        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            all_conneted_non_soma_pieces.append(connected_mesh_pieces)
        
        #b) Iterate through all of the somas and get the pieces that are connected
        t_non_soma_pieces = np.concatenate(all_conneted_non_soma_pieces)
        
        #c) Filter away the mesh pieces that aren't touching and add to the floating pieces
        sig_non_soma_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh in t_non_soma_pieces]
        new_floating_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh not in t_non_soma_pieces]
        
        print(f"new_floating_pieces = {new_floating_pieces}")
        
        non_soma_touching_meshes += new_floating_pieces
        
        

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces
        
        
        
        
        

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        soma_to_piece_touching_vertices = dict()
        soma_to_piece_touching_vertices_idx = dict()
        limb_root_nodes = dict()

        m_vert_graph = tu.mesh_vertex_graph(current_mesh)

        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces

            soma_to_piece_touching_vertices[i] = dict()
            for piece_index,piece_idx in enumerate(connected_mesh_pieces):
                limb_root_nodes[piece_idx] = connected_mesh_pieces_vertices[piece_index][0]

                """ Old way of finding vertex connected components on a mesh without trimesh function
                #find the number of touching groups and save those 
                soma_touching_graph = m_vert_graph.subgraph(connected_mesh_pieces_vertices_idx[piece_index])
                soma_con_comp = [current_mesh.vertices[np.array(list(k)).astype("int")] for k in list(nx.connected_components(soma_touching_graph))]
                soma_to_piece_touching_vertices[i][piece_idx] = soma_con_comp
                """

                soma_to_piece_touching_vertices[i][piece_idx] = tu.split_vertex_list_into_connected_components(
                                                    vertex_indices_list=connected_mesh_pieces_vertices_idx[piece_index],
                                                    mesh=current_mesh, 
                                                    vertex_graph=m_vert_graph, 
                                                    return_coordinates=True
                                                   )





    #         border_debug = False
    #         if border_debug:
    #             print(f"soma_to_piece_connectivity = {soma_to_piece_connectivity}")
    #             print(f"soma_to_piece_touching_vertices = {soma_to_piece_touching_vertices}")


        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    print(f"# of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)} with trimesh : {not_processed_soma_containing_meshes}")
    



    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")

    current_mesh_data = soma_touching_mesh_data
    soma_containing_idx = 0

    #doing inversion of the connectivity and touching vertices
    piece_to_soma_touching_vertices = gu.flip_key_orders_for_dict(soma_to_piece_touching_vertices)
    
    
    
    
    
    
    # Phase 4: Skeletonization, Mesh Correspondence,  

    proper_time = time.time()

    #The containers that will hold the final data for the preprocessed neuron
    limb_correspondence=dict()
    limb_network_stating_info = dict()

    # ---------- Part A: skeletonization and mesh decomposition --------- #
    skeleton_time = time.time()

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):

        #Arguments to pass to the specific function (when working with a limb)
        soma_touching_vertices_dict = piece_to_soma_touching_vertices[curr_limb_idx]

    #     if curr_limb_idx != 10:
    #         continue

        curr_limb_time = time.time()
        print(f"\n\n----- Working on Proper Limb # {curr_limb_idx} ---------")

        print(f"meshparty_segment_size = {meshparty_segment_size}")
        limb_correspondence_individual,network_starting_info = preprocess_limb(mesh=limb_mesh_mparty,
                       soma_touching_vertices_dict = soma_touching_vertices_dict,
                       return_concept_network = False, 
                       return_concept_network_starting_info=True,
                       width_threshold_MAP=500,
                       size_threshold_MAP=2000,
                       surface_reconstruction_size=1000,  

                       #arguments added from the big preprocessing step                                                            
                       distance_by_mesh_center=distance_by_mesh_center,
                       meshparty_segment_size=meshparty_segment_size,
                       meshparty_n_surface_downsampling = meshparty_n_surface_downsampling,
                                                                               
                        use_meshafterparty=use_meshafterparty,
                        error_on_no_starting_coordinates=True

                       )
        #Storing all of the data to be sent to 

        limb_correspondence[curr_limb_idx] = limb_correspondence_individual
        limb_network_stating_info[curr_limb_idx] = network_starting_info
        
    print(f"Total time for Skeletonization and Mesh Correspondence = {time.time() - skeleton_time}")
        
        
        
    # ---------- Part B: Stitching on floating pieces --------- #
    print("\n\n ----- Working on Stitching ----------")
    
#     # --- Get the soma connecting points that don't want to stitch to ---- #
#     excluded_node_coordinates = []
#     for limb_idx,limb_start_v in limb_network_stating_info.items():
#         for soma_idx,soma_v in limb_start_v.items():
#             for soma_group_idx,group_v in soma_v.items():
#                 excluded_node_coordinates.append(group_v["endpoint"])

    excluded_node_coordinates = nru.all_soma_connnecting_endpionts_from_starting_info(limb_network_stating_info)
    
    

    floating_stitching_time = time.time()
    
    if len(limb_correspondence) > 0:
        non_soma_touching_meshes_to_stitch = tu.check_meshes_outside_multiple_mesh_bbox(seperate_soma_meshes,non_soma_touching_meshes,
                                 return_indices=False)
        
        limb_correspondence_with_floating_pieces = attach_floating_pieces_to_limb_correspondence(
                limb_correspondence,
                floating_meshes=non_soma_touching_meshes_to_stitch,
                floating_piece_face_threshold = 600,
                max_stitch_distance=8000,
                distance_to_move_point_threshold = 4000,
                verbose = False,
                excluded_node_coordinates = excluded_node_coordinates)
    else:
        limb_correspondence_with_floating_pieces = limb_correspondence
        



    print(f"Total time for stitching floating pieces = {time.time() - floating_stitching_time}")





    # ---------- Part C: Computing Concept Networks --------- #
    concept_network_time = time.time()

    limb_concept_networks=dict()
    limb_labels=dict()

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_with_floating_pieces[curr_limb_idx],
                                                                        limb_network_stating_info[curr_limb_idx],
                                                                        run_concept_network_checks=True,
                                                                           )   



        limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
        limb_labels[curr_limb_idx]= "Unlabeled"

    print(f"Total time for Concept Networks = {time.time() - concept_network_time}")



    #------ 1/11/ getting the glia faces --------------
    if glia_meshes is not None and len(glia_meshes)>0:
            glia_faces = tu.original_mesh_faces_map(current_neuron,tu.combine_meshes(glia_pieces))
    else:
        glia_faces = np.array([])

    preprocessed_data= dict(
        soma_meshes = current_mesh_data[0]["soma_meshes"],
        soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
        soma_sdfs = total_soma_list_sdf,
        insignificant_limbs=insignificant_limbs,
        not_processed_soma_containing_meshes=not_processed_soma_containing_meshes,
        glia_faces = glia_faces,
        non_soma_touching_meshes=non_soma_touching_meshes,
        inside_pieces=inside_pieces,
        limb_correspondence=limb_correspondence_with_floating_pieces,
        limb_concept_networks=limb_concept_networks,
        limb_network_stating_info=limb_network_stating_info,
        limb_labels=limb_labels,
        limb_meshes=current_mesh_data[0]["branch_meshes"],
        )



    print(f"Total time for all mesh and skeletonization decomp = {time.time() - proper_time}")
    
    return preprocessed_data'''
    
    
    




def preprocess_neuron(
    mesh=None,
    mesh_file=None,
    segment_id=None,
     description=None,
    
    
    # ---------------- all the parameters that control preprocessing -------------
    sig_th_initial_split=100, #for significant splitting meshes in the intial mesh split
    limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
    
    
    # ------- 12/29 for the limb expansion --------------
    apply_expansion = None,
    floating_piece_face_threshold_expansion = 500,
    max_distance_threshold_expansion = 2000,
    min_n_faces_on_path_expansion = 5_000,

    filter_end_node_length=None, #used in cleaning the skeleton during skeletonizations
    

    decomposition_type="meshafterparty",
    distance_by_mesh_center=True,
    meshparty_segment_size =100,
    meshparty_n_surface_downsampling = 2,

    
    combine_close_skeleton_nodes = True,
    combine_close_skeleton_nodes_threshold=700,
    
    #parameters for prprocess limb
    width_threshold_MAP=None,
    size_threshold_MAP=None,
    surface_reconstruction_size=None,#1000,
    
    floating_piece_face_threshold = None,#600,
    max_stitch_distance=None,#8000,
    distance_to_move_point_threshold = 4000,
    
    

    
    # --------- non-parameter flags -------------------------
    glia_faces=None,
    nuclei_faces=None, 
    somas=None, #the precomputed somas
    return_no_somas = False, #whether to error or to return an empty list for somas
    verbose = True,
    
    use_adaptive_invalidation_d = None,
    use_adaptive_invalidation_d_floating = None,
    axon_width_preprocess_limb_max = None,
    limb_remove_mesh_interior_face_threshold = None,
    
    
    error_on_bad_cgal_return=False,
    max_stitch_distance_CGAL = None

    ):
    
    if width_threshold_MAP is None:
        width_threshold_MAP = width_threshold_MAP_global
    if size_threshold_MAP is None:
        size_threshold_MAP = size_threshold_MAP_global
    if apply_expansion is None:
        apply_expansion = apply_expansion_global
    if max_stitch_distance is None:
        max_stitch_distance = max_stitch_distance_global
    if max_stitch_distance_CGAL is None:
        max_stitch_distance_CGAL = max_stitch_distance_CGAL_global
    if filter_end_node_length is None:
        filter_end_node_length = filter_end_node_length_global
    if axon_width_preprocess_limb_max is None:
        axon_width_preprocess_limb_max = axon_width_preprocess_limb_max_global
    if limb_remove_mesh_interior_face_threshold is None:
        limb_remove_mesh_interior_face_threshold = limb_remove_mesh_interior_face_threshold_global
    if surface_reconstruction_size is None:
        surface_reconstruction_size = surface_reconstruction_size_global
        
    if floating_piece_face_threshold is None:
        floating_piece_face_threshold = floating_piece_face_threshold_global
    
    
    
    """
    Purpose: to return preprocess dict of a neuron
    
    """
    
    
    
    print(f"limb_remove_mesh_interior_face_threshold = {limb_remove_mesh_interior_face_threshold}")
    
    pre_branch_connectivity = "edges"
    
    if "meshafterparty" in decomposition_type.lower():
        use_meshafterparty = True
    else:
        use_meshafterparty = False
    
    
    print(f"use_meshafterparty = {use_meshafterparty}")
    
    whole_processing_tiempo = time.time()


    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    This method includes the fusion

    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"


    if mesh is None:
        if mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = tu.load_mesh_no_processing(mesh_file)
    else:
        current_neuron = mesh
        
        
        
        
        
        
    # -------- Phase 1: Doing Soma Detection (if Not already done) ---------- #
    if somas is None:
        (soma_mesh_list, 
         run_time, 
         total_soma_list_sdf,
         glia_pieces,
         nuclei_pieces) = sm.extract_soma_center(segment_id,
                                                 current_neuron.vertices,
                                                 current_neuron.faces)
        
        if len(glia_pieces)>0:
            glia_faces = tu.original_mesh_faces_map(current_neuron,tu.combine_meshes(glia_pieces))
            n_glia_faces = len(glia_faces)
        else:
            glia_faces = []
            n_glia_faces = 0
            
        if len(nuclei_pieces)>0:
            nuclei_faces = tu.original_mesh_faces_map(current_neuron,tu.combine_meshes(nuclei_pieces))
            n_nuclei_faces = len(nuclei_faces)
        else:
            nuclei_faces = []
            n_nuclei_faces = 0
    else:
        soma_mesh_list,run_time,total_soma_list_sdf = somas
        print(f"Using pre-computed somas: soma_mesh_list = {soma_mesh_list}")

    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return_value= soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
        
        
        
        
        
    
    print_optimize = True
    #--- Phase 2: getting the soma submeshes that are connected to each soma and identifiying those that aren't 
    # ------------------ (and eliminating any mesh pieces inside the soma) ------------------------

    # -------- 11/13 Addition: Will remove the inside nucleus --------- #

    optimize_time = time.time()

    glia_faces,nuclei_faces

    if glia_faces is None or nuclei_faces is None:
        main_mesh_total,glia_meshes,nuclei_meshes = sm.remove_nuclei_and_glia_meshes(current_neuron,
                                                                       verbose=True)
        print("Using pre-computed glia and nuclei pieces")
        if len(glia_meshes) > 0 or len(nuclei_meshes) > 0:
            main_mesh_total = tu.subtract_mesh(current_neuron,glia_meshes + nuclei_meshes)
        else:
            main_mesh_total = current_neuron
    else:
        if len(nuclei_faces) > 0:
            nuclei_meshes = current_neuron.submesh([nuclei_faces],append=True,repair=False)
            nuclei_meshes = [nuclei_meshes]
        else:
            nuclei_meshes = []

        total_eliminated_faces = list(glia_faces) + list(nuclei_faces)
        if len(total_eliminated_faces)>0:
            faces_to_keep = np.delete(np.arange(len(current_neuron.faces)),total_eliminated_faces)
            main_mesh_total = current_neuron.submesh([faces_to_keep],append=True,repair=False)
        else:
            main_mesh_total = current_neuron


    if print_optimize:
        print(f"Getting Glia and Nuclei Pieces Subtracted Away {time.time()-optimize_time}")
    optimize_time = time.time()


    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces



    optimize_time = time.time()


    split_meshes,split_meshes_face_idx = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False,
                                return_face_indices=True,
                                connectivity=pre_branch_connectivity)


    if print_optimize:
        print(f" Splitting mesh after soma cancellation {time.time()-optimize_time}")
    optimize_time = time.time()

    print(f"# of split_meshes = {len(split_meshes)}")

    """  Newer slower way of doing it    

    tu.two_mesh_list_connectivity(soma_mesh_list,split_meshes_face_idx,main_mesh_total)
    """
    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes)

    if print_optimize:
        print(f" Containing Mesh Indices {time.time()-optimize_time}")
        print(f"containing_mesh_indices = {containing_mesh_indices}")
    optimize_time = time.time()
    

    # filtering away any of the inside floating pieces: 
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in set(list(containing_mesh_indices.values()))]

    if print_optimize:
        print(f" non_soma_touching_meshes {time.time()-optimize_time}")
    optimize_time = time.time()

    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        """
        *** want to save these pieces that are inside of the soma***
        """

        non_soma_touching_meshes,inside_pieces = sm.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split,
                                        return_inside_pieces = True)

    else:
        non_soma_touching_meshes = []
        inside_pieces=[]

    if print_optimize:
        print(f" Finding inside pieces and non_soma_touching meshes {time.time()-optimize_time}")
    optimize_time = time.time()

    # --------------------- 1/10 Change ---------------- #

    #adding in the nuclei center to the inside pieces
    inside_pieces += nuclei_meshes


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    if verbose:
        print(f"soma_containing_meshes = {soma_containing_meshes}")

    #     print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    #     print(f"# of inside pieces = {len(inside_pieces)}")
    

    
    '''  1/18 Addition that combines all soma meshes
    
    # ------ 11/15 Addition: Part 2.b 

    """
    Pseudocode: 
    1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    2) Save all other meshes not the largest in 
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf


    """
    #1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    soma_containing_meshes_keys = np.array(list(soma_containing_meshes.keys()))
    soma_touching_meshes = np.array([split_meshes[k] for k in soma_containing_meshes_keys])
    largest_soma_touching_mesh_idx = soma_containing_meshes_keys[np.argmax([len(kk.faces) for kk in soma_touching_meshes])]

    #2) Save all other meshes not the largest in 
    not_processed_soma_containing_meshes_idx = np.setdiff1d(soma_containing_meshes_keys,[largest_soma_touching_mesh_idx])
    not_processed_soma_containing_meshes = [split_meshes[k] for k in not_processed_soma_containing_meshes_idx]
    print(f"Number of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)}")

    """
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf

    """

    somas_idx_to_process = soma_containing_meshes[largest_soma_touching_mesh_idx]
    soma_mesh_list = [soma_mesh_list[k] for k in somas_idx_to_process]

    soma_containing_meshes = {largest_soma_touching_mesh_idx:list(np.arange(0,len(soma_mesh_list)))}

    soma_touching_meshes = [split_meshes[largest_soma_touching_mesh_idx]]

    total_soma_list_sdf = total_soma_list_sdf[somas_idx_to_process]

    print(f"\n-----After filtering away multiple disconneted soma pieces-----")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")


    if print_optimize:
        print(f" Filtering Away Disconnected Soma Pieces {time.time()-optimize_time}")
    optimize_time = time.time()
    
    '''
    print(f"\n-----Before combining multiple mesh pieces-----")
    print(f"soma_containing_meshes = {soma_containing_meshes}")
    
    soma_containing_meshes_keys = np.array(list(soma_containing_meshes.keys()))
    combined_soma_containing_mesh = tu.combine_meshes([split_meshes[k] for k in soma_containing_meshes_keys])
    not_processed_soma_containing_meshes = []
    
    #rewriting the 
    soma_containing_meshes = {0:list(np.arange(0,len(soma_mesh_list)))}
    soma_containing_meshes_keys = np.array(list(soma_containing_meshes.keys()))
    
    print(f"\n-----After combining multiple mesh pieces-----")
    print(f"soma_containing_meshes = {soma_containing_meshes}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        #--- Phase 3:  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces

    """
    for each soma touching mesh get the following:
    1) original soma meshes
    2) significant mesh pieces touching these somas
    3) The soma connectivity to each of the significant mesh pieces
    -- later will just translate the 


    Process: 

    1) Final all soma faces (through soma extraction and then soma original faces function)
    2) Subtact all soma faces from original mesh
    3) Find all significant mesh pieces
    4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all
       the available somas
    Conclusion: Will have connectivity map


    """

    soma_touching_mesh_data = dict()

    for z,(mesh_idx, soma_idxes) in enumerate(soma_containing_meshes.items()):
        soma_touching_mesh_data[z] = dict()
        print(f"\n\n----Working on soma-containing mesh piece {z}----")
        current_time = time.time()

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = combined_soma_containing_mesh

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]



        current_time = time.time()
        
        """Old Way
        seperate_soma_meshes = current_soma_mesh_list
        non_soma_stacked_mesh_face_idx = tu.original_mesh_faces_map(original_mesh=current_mesh,
                                      submesh=tu.combine_meshes(current_soma_mesh_list),
                                      exact_match = True,
                                    matching=False)

        non_soma_stacked_mesh = current_mesh.submesh([non_soma_stacked_mesh_face_idx],append=True,repair=False)
        """
        
        seperate_soma_meshes_idx = [tu.original_mesh_faces_map(original_mesh=current_mesh,
                                  submesh=k,
                                  exact_match = False,
                                matching=True) for k in current_soma_mesh_list]
        seperate_soma_meshes = [current_mesh.submesh([k],append=True,repair=False) for k in seperate_soma_meshes_idx]

        non_soma_stacked_mesh = current_mesh.submesh([np.delete(np.arange(len(current_mesh.faces)),np.concatenate(seperate_soma_meshes_idx))],append=True,repair=False)

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes

        print(f"Total time for Subtract Soma and Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()



        # 3) Find all significant mesh pieces
        """
        Pseudocode: 
        a) Iterate through all of the somas and get the pieces that are connected
        b) Concatenate all the results into one list and order
        c) Filter away the mesh pieces that aren't touching and add to the floating pieces

        """
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True,
                                                                             connectivity=pre_branch_connectivity)

        # a) Filter these down to only those touching the somas
        all_conneted_non_soma_pieces = []
        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            all_conneted_non_soma_pieces.append(connected_mesh_pieces)

        #b) Iterate through all of the somas and get the pieces that are connected
        t_non_soma_pieces = np.concatenate(all_conneted_non_soma_pieces)

        #c) Filter away the mesh pieces that aren't touching and add to the floating pieces
        sig_non_soma_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh in t_non_soma_pieces]
        new_floating_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh not in t_non_soma_pieces]

        print(f"new_floating_pieces = {new_floating_pieces}")

        non_soma_touching_meshes += new_floating_pieces
        
#         non_soma_touching_meshes_to_stitch = tu.check_meshes_outside_multiple_mesh_bbox(seperate_soma_meshes,non_soma_touching_meshes,
#                                  return_indices=False)
        
#         su.compressed_pickle(non_soma_touching_meshes_to_stitch,"non_soma_touching_meshes_to_stitch")
#         raise Exception("")



        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        


        # -------- 12/29: Part that will expand the limbs that need to be processed ------
        
#         su.compressed_pickle(non_soma_touching_meshes,"non_soma_touching_meshes")
#         su.compressed_pickle(insignificant_limbs,"insignificant_limbs")
#         su.compressed_pickle(seperate_soma_meshes,"seperate_soma_meshes")
#         print(f"floating_piece_face_threshold_expansion = {floating_piece_face_threshold_expansion}")
#         print(f"max_distance_threshold_expansion = {max_distance_threshold_expansion}")
#         print(f"min_n_faces_on_path_expansion = {min_n_faces_on_path_expansion}")
#         raise Exception("")
        
        if apply_expansion:
            new_limbs_nst,new_limbs_il,nst_still_meshes,il_still_meshes= pre.limb_meshes_expansion(
                non_soma_touching_meshes,
                insignificant_limbs,
                seperate_soma_meshes,

                #Step 1: Filering
                plot_filtered_pieces = False,
                non_soma_touching_meshes_face_min = floating_piece_face_threshold_expansion,
                insignificant_limbs_face_min = floating_piece_face_threshold_expansion,

                #Step 2: Distance Graph Structure
                plot_distance_G = False,
                plot_distance_G_thresholded = False,
                max_distance_threshold = max_distance_threshold_expansion,

                #Step 3: 
                min_n_faces_on_path = min_n_faces_on_path_expansion,
                plot_final_limbs = False,
                plot_not_added_limbs = False,

                verbose = verbose
                )

            non_soma_touching_meshes = list(nst_still_meshes)
            insignificant_limbs = list(il_still_meshes)
            sig_non_soma_pieces += list(new_limbs_il)
        else:
            if verbose:
                print(f"Not applying expansions")
            new_limbs_nst = []
            
        



        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        soma_to_piece_touching_vertices = dict()
        soma_to_piece_touching_vertices_idx = dict()
        limb_root_nodes = dict()

        #m_vert_graph = tu.mesh_vertex_graph(current_mesh)

        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces
            
            #add on the connectivity of th non-soma touching pieces
            

            soma_to_piece_touching_vertices[i] = dict()
            for piece_index,piece_idx in enumerate(connected_mesh_pieces):
                limb_root_nodes[piece_idx] = connected_mesh_pieces_vertices[piece_index][0]

                """ Old way of finding vertex connected components on a mesh without trimesh function
                #find the number of touching groups and save those 
                soma_touching_graph = m_vert_graph.subgraph(connected_mesh_pieces_vertices_idx[piece_index])
                soma_con_comp = [current_mesh.vertices[np.array(list(k)).astype("int")] for k in list(nx.connected_components(soma_touching_graph))]
                soma_to_piece_touching_vertices[i][piece_idx] = soma_con_comp
                """

                soma_to_piece_touching_vertices[i][piece_idx] = tu.split_vertex_list_into_connected_components(
                                                    vertex_indices_list=connected_mesh_pieces_vertices_idx[piece_index],
                                                    mesh=current_mesh, 
                                                    #vertex_graph=m_vert_graph, 
                                                    return_coordinates=True
                                                   )
                
                #add in the soma to piece touching vertices for non-soma touching pieces

        
        #-----12/29: Adding the non-soma touching pieces that don't currently touch ------
        nst_n_vertices = 1
    
        for nst_piece_idx,piece in enumerate(new_limbs_nst):
            closest_idx,closest_vertex = tu.closest_mesh_to_mesh(
                mesh=piece,
                meshes=seperate_soma_meshes,
                verbose = True,
                return_closest_distance = False,
                return_closest_vertex_on_mesh = True,
                )
            if nst_n_vertices == 1:
                touching_vertices = [np.array(closest_vertex).reshape(-1,3)]
            else:
                raise Exception("Unimplemented nst_n_vertices != 1")
                
            curr_idx = len(sig_non_soma_pieces) + nst_piece_idx
            soma_to_piece_connectivity[closest_idx].append(curr_idx)
            soma_to_piece_touching_vertices[closest_idx][curr_idx] = touching_vertices
            limb_root_nodes[curr_idx] = np.array(closest_vertex).reshape(-1)
            
            


    #         border_debug = False
    #         if border_debug:
    #             print(f"soma_to_piece_connectivity = {soma_to_piece_connectivity}")
    #             print(f"soma_to_piece_touching_vertices = {soma_to_piece_touching_vertices}")


        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    print(f"# of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)} with trimesh : {not_processed_soma_containing_meshes}")

    
    
    #adds on the meshes

    soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces + list(new_limbs_nst)

    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")

    current_mesh_data = soma_touching_mesh_data
    soma_containing_idx = 0

    #doing inversion of the connectivity and touching vertices
    piece_to_soma_touching_vertices = gu.flip_key_orders_for_dict(soma_to_piece_touching_vertices)


    
    
    
    
    
    
    # Phase 4: Skeletonization, Mesh Correspondence,  

    proper_time = time.time()

    #The containers that will hold the final data for the preprocessed neuron
    limb_correspondence=dict()
    limb_network_stating_info = dict()

    # ---------- Part A: skeletonization and mesh decomposition --------- #
    skeleton_time = time.time()
    
    #raise Exception("")

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):
        use_meshafterparty_current = copy.copy(use_meshafterparty)

        #Arguments to pass to the specific function (when working with a limb)
        soma_touching_vertices_dict = piece_to_soma_touching_vertices[curr_limb_idx]

#         if curr_limb_idx != 1:
#             continue
        
        verbose = True


#         su.compressed_pickle(limb_mesh_mparty,"limb_mesh_mparty")
#         su.compressed_pickle(soma_touching_vertices_dict,"soma_touching_vertices_dict")
#         su.compressed_pickle(current_mesh_data[0]["branch_meshes"],"limb_meshes")
#         su.compressed_pickle(piece_to_soma_touching_vertices,"piece_to_soma_touching_vertices")
#         raise Exception("")
        curr_limb_time = time.time()
        for jj in range(0,2):
            try:
                print(f"\n\n----- Working on Proper Limb # {curr_limb_idx} ---------")

#                 su.compressed_pickle(limb_mesh_mparty,f"limb_mesh_mparty_{curr_limb_idx}")
#                 su.compressed_pickle(soma_touching_vertices_dict,f"soma_touching_vertices_dict_{curr_limb_idx}")
                
                print(f"meshparty_segment_size = {meshparty_segment_size}")
                
                limb_correspondence_individual,network_starting_info = preprocess_limb(mesh=limb_mesh_mparty,
                               soma_touching_vertices_dict = soma_touching_vertices_dict,
                               return_concept_network = False, 
                               return_concept_network_starting_info=True,
                               width_threshold_MAP=width_threshold_MAP,
                               size_threshold_MAP=size_threshold_MAP,
                               surface_reconstruction_size=surface_reconstruction_size,  

                               #arguments added from the big preprocessing step                                                            
                               distance_by_mesh_center=distance_by_mesh_center,
                               meshparty_segment_size=meshparty_segment_size,
                               meshparty_n_surface_downsampling = meshparty_n_surface_downsampling,

                                use_meshafterparty=use_meshafterparty_current,
                                error_on_no_starting_coordinates=True,
                                                                                       
                                use_adaptive_invalidation_d = use_adaptive_invalidation_d,
                                axon_width_preprocess_limb_max = axon_width_preprocess_limb_max,
                                remove_mesh_interior_face_threshold=limb_remove_mesh_interior_face_threshold,
                                                                                       
                                error_on_bad_cgal_return=error_on_bad_cgal_return,
                                max_stitch_distance_CGAL = max_stitch_distance_CGAL

                               )
                #Storing all of the data to be sent to 

                limb_correspondence[curr_limb_idx] = limb_correspondence_individual
                limb_network_stating_info[curr_limb_idx] = network_starting_info
            except Exception as e:
                if jj == 0:
                    use_meshafterparty_current = False
                else:
                    raise Exception(f"{e}")
            else:
                print(f"Successful Limb Decomposition")
                break
        
    print(f"Total time for Skeletonization and Mesh Correspondence = {time.time() - skeleton_time}")
        
        
        
    # ---------- Part B: Stitching on floating pieces --------- #
    print("\n\n ----- Working on Stitching ----------")
    
#     # --- Get the soma connecting points that don't want to stitch to ---- #
#     excluded_node_coordinates = []
#     for limb_idx,limb_start_v in limb_network_stating_info.items():
#         for soma_idx,soma_v in limb_start_v.items():
#             for soma_group_idx,group_v in soma_v.items():
#                 excluded_node_coordinates.append(group_v["endpoint"])

    excluded_node_coordinates = nru.all_soma_connnecting_endpionts_from_starting_info(limb_network_stating_info)
    
    

    floating_stitching_time = time.time()
    
    if len(limb_correspondence) > 0:
#         if verbose:
#             print(f"BEFORE filtering floating points, {len(non_soma_touching_meshes)} floating pieces: "
#                   f"{non_soma_touching_meshes[:np.min([10,len(non_soma_touching_meshes)])]}")
            
        outside_perc_threshold = 80
        non_soma_touching_meshes_to_stitch = [k for k in non_soma_touching_meshes if tu.n_vertices_outside_mesh_bbox(
            k,seperate_soma_meshes,return_percentage=True) > outside_perc_threshold]    
        
#         non_soma_touching_meshes_to_stitch = tu.check_meshes_outside_multiple_mesh_bbox(seperate_soma_meshes,non_soma_touching_meshes,
#                                  return_indices=False)
        
#         if verbose:
#             print(f"AFTER filtering floating points, {len(non_soma_touching_meshes_to_stitch)} floating pieces: "
#                   f"{non_soma_touching_meshes_to_stitch[:np.min([10,len(non_soma_touching_meshes_to_stitch)])]}")
        
        # add in a check for distance
        """
        NO NEED TO FILTER FOR STITCH DISTANCE BECAUSE ALREADY DONE IN STEP ABOVE LOOKING FOR EXPANSION
        
        """
        
        
        
        limb_correspondence_with_floating_pieces = attach_floating_pieces_to_limb_correspondence(
                limb_correspondence,
                floating_meshes=non_soma_touching_meshes_to_stitch,
                floating_piece_face_threshold = floating_piece_face_threshold,#600,
                max_stitch_distance=max_stitch_distance,#8000,
                distance_to_move_point_threshold = distance_to_move_point_threshold,
                verbose = False,
                excluded_node_coordinates = excluded_node_coordinates,
                use_adaptive_invalidation_d = use_adaptive_invalidation_d_floating,
                axon_width_preprocess_limb_max = axon_width_preprocess_limb_max,
                )
    else:
        limb_correspondence_with_floating_pieces = limb_correspondence
        



    print(f"Total time for stitching floating pieces = {time.time() - floating_stitching_time}")





    # ---------- Part C: Computing Concept Networks --------- #
    concept_network_time = time.time()

    limb_concept_networks=dict()
    limb_labels=dict()

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_with_floating_pieces[curr_limb_idx],
                                                                        limb_network_stating_info[curr_limb_idx],
                                                                        run_concept_network_checks=True,
                                                                           )   



        limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
        limb_labels[curr_limb_idx]= "Unlabeled"

    print(f"Total time for Concept Networks = {time.time() - concept_network_time}")

    soma_meshes = current_mesh_data[0]["soma_meshes"]
    preprocessed_data= dict(
        soma_meshes = soma_meshes,
        soma_volumes = [tu.mesh_volume(k) for k in soma_meshes],
        soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
        soma_sdfs = total_soma_list_sdf,
        insignificant_limbs=insignificant_limbs,
        not_processed_soma_containing_meshes=not_processed_soma_containing_meshes,
        glia_faces = glia_faces,
        non_soma_touching_meshes=non_soma_touching_meshes,
        inside_pieces=inside_pieces,
        limb_correspondence=limb_correspondence_with_floating_pieces,
        limb_concept_networks=limb_concept_networks,
        limb_network_stating_info=limb_network_stating_info,
        limb_labels=limb_labels,
        limb_meshes=current_mesh_data[0]["branch_meshes"],
        )



    print(f"Total time for all mesh and skeletonization decomp = {time.time() - proper_time}")
    
    return preprocessed_data    
    
    
'''
def high_fidelity_axon_decomposition_old(neuron_obj,
                                     plot_new_axon_limb_correspondence=False,
                                     plot_connecting_skeleton_fix = False,
                                     plot_final_limb_correspondence=False,
                                     return_starting_info=True,
                                verbose = True,
    ):
    
    """
    Purpose: To get the decomposition of the 
    axon with the a finer skeletonization
    
    Returns: a limb correspondence of the revised branches
    
    Pseudocode:
    1) Get the starting information for decomposition
    2) Split the axon mesh into just one connected mesh (aka filtering away the disconnected parts)
    3) Run the limb preprocessing
    4) Retriveing starting info from concept network
    5) Adjust the axon decomposition to connect to an upstream piece if there was one
    6) Return limb correspondence and starting information (IF WE REVISED THE STARTING INFO)
    
    --- Add back the floating mesh pieces using the stitching process --
    """
    
    #---1) Get the starting information for decomposition ---
    
    
    axon_limb_name = neuron_obj.axon_limb_name
    curr_limb = neuron_obj[axon_limb_name]
    axon_starting_branch = neuron_obj.axon_starting_branch
    axon_starting_branch_mesh = curr_limb[axon_starting_branch].mesh
    axon_starting_coordinate = neuron_obj.axon_starting_coordinate

    upstream_node_to_axon_starting_branch = None

    # ---- 2/28: MOVE THE AXON BRANCHES BACK ENOUGH SO THE DENDRITES WONT BE ATTACHED
    
    
    
    if axon_starting_branch != curr_limb.current_starting_node:
        if verbose:
            print(f"Starting axon branch was not the limb starting node so generating border vertices and computing upstream branch")
        border_vertices_for_axon = tu.largest_border_to_coordinate(
            mesh = axon_starting_branch_mesh,
            coordinate =axon_starting_coordinate,
            distance_threshold = 1000,
            plot_border_vertices = False,
            error_on_no_border = True,
            plot_winning_border = False,
            verbose = False)

        upstream_node_to_axon_starting_branch = xu.upstream_node(curr_limb.concept_network_directional,
                                                                 axon_starting_branch
                                                                   )

    else:
        if verbose:
            print(f"Starting axon branch was the starting node so using border vertices and NOT computing upstream branch")
        border_vertices_for_axon = curr_limb.current_touching_soma_vertices
        upstream_node_to_axon_starting_branch = None

    soma_touching_vertices_dict = {0:[border_vertices_for_axon]}

    if verbose:
        print(f"upstream_node_to_axon_starting_branch = {upstream_node_to_axon_starting_branch}")
        
        
        
    #---2) Split the axon mesh into just one connected mesh (aka filtering away the disconnected parts)---
        
    # divide the meshes into the only big continuous one (because if disconnected probably)
    diff_meshes = tu.split_by_vertices(neuron_obj.axon_mesh)
    axon_mesh_filtered = tu.filter_meshes_by_containing_coordinates(diff_meshes,
                                              nullifying_points=border_vertices_for_axon[0],
                                               method="distance",
                                               distance_threshold = 0,
                                              filter_away=False)[0]
    if verbose:
        if len(diff_meshes) > 1:
            print(f"More than 1 seperated mesh (taking the top one): {diff_meshes}")
            
            
    #---3) Run the limb preprocessing---
    limb_correspondence_individual,concept_network = pre.preprocess_limb(axon_mesh_filtered,
                       soma_touching_vertices_dict = soma_touching_vertices_dict,
                        meshparty_segment_size = 100,
                        combine_close_skeleton_nodes=True,
                        #combine_close_skeleton_nodes_threshold=1200,
                        combine_close_skeleton_nodes_threshold_meshparty = 1300,
                        filter_end_node_length_meshparty = 1500,
                        filter_end_node_length = 1500,

                        use_meshafterparty=False,
                        perform_cleaning_checks = True,


                        #concept_network parameters
                        run_concept_network_checks = True,
                        return_concept_network = True,
                        return_concept_network_starting_info=False,

                        #printing controls
                        verbose = True,
                        print_fusion_steps=True,

                        check_correspondence_branches = True,
                        filter_end_nodes_from_correspondence=True,
                        error_on_no_starting_coordinates=True,
                        invalidation_d=2000,


                       )
    
    #-- 4) Retriveing starting info from concept network --
    curr_concept_network = concept_network[0][0]

    starting_node = xu.get_starting_node(curr_concept_network,only_one=False)[0]

    starting_coordinate = curr_concept_network.nodes[starting_node]["starting_coordinate"]
    touching_soma_vertices = curr_concept_network.nodes[starting_node]["touching_soma_vertices"]
    
    limb_network_stating_info = {0:{0:{"touching_verts":touching_soma_vertices,
                                      "endpoint":starting_coordinate}}}

    if plot_new_axon_limb_correspondence:
        nviz.plot_limb_correspondence(limb_correspondence_individual,
                                     scatters = [starting_coordinate],
                                     scatter_size=1)
        
        
    
    # -- 5) Adjust the axon decomposition to connect to an upstream piece if there was one --
    """
    Psuedocode: 
    1) Get the starting axon branch and the upstream node
    2) Find the common endpoint
    3) Add a skeletal branch from starting_coordinate (new)
    and the common endpoint to the starting node of new
    axon decomposition

    """
    

    #1) Get the starting axon branch and the upstream node
    if upstream_node_to_axon_starting_branch is not None:
        print("Readjusting starting axon branch skeleton")

        #1) Get the starting axon branch and the upstream node
        axon_starting_branch,upstream_node_to_axon_starting_branch

        #2) Find the common endpoint
    #     shared_endpoint = shared_skeleton_endpoints_for_connected_branches(neuron_obj[axon_limb_name],
    #                                                 axon_starting_branch,
    #                                                      upstream_node_to_axon_starting_branch,
    #                                                 verbose=False)
        shared_endpoint = neuron_obj.axon_starting_coordinate

        if shared_endpoint.ndim > 1:
            shared_endpoint = shared_endpoint[0]

        #3) Add a skeletal branch from starting_coordinate (new)
        #and the common endpoint to the starting node of new
        #axon decomposition

        curr_endpoints = sk.find_skeleton_endpoint_coordinates(limb_correspondence_individual[starting_node]["branch_skeleton"])
        if len(nu.matching_rows(curr_endpoints,axon_starting_coordinate)) == 0:
            print("Fixing the axon starting branch endpoint to align with upstream branch")

            skeleton_pre_fix = limb_correspondence_individual[starting_node]["branch_skeleton"]
            new_skeleton_segment = np.array([starting_coordinate,shared_endpoint]).reshape(-1,2,3)

            limb_correspondence_individual[starting_node]["branch_skeleton"] = sk.stack_skeletons([skeleton_pre_fix,
                                                                                                   new_skeleton_segment
                                                                                                 ])
            if verbose:
                print(f"Starting Branch {starting_node} skeleton before fix: {skeleton_pre_fix.shape}")
                print(f"Starting Branch {starting_node} skeleton AFTER fix: {limb_correspondence_individual[starting_node]['branch_skeleton'].shape}")
        else:
            if verbose:
                print("Not attempting to fix the limb correspondence because the axon_starting_coordainte was already an endpoint")

        
        if plot_connecting_skeleton_fix:
            upstream_branch = neuron_obj[axon_limb_name][upstream_node_to_axon_starting_branch]
            meshes,skeletons = nviz.limb_correspondence_plottable(limb_correspondence_individual)
            nviz.plot_objects(meshes=meshes + [upstream_branch.mesh],
                              meshes_colors="random",
                              skeletons=skeletons +  [upstream_branch.skeleton],
                              skeletons_colors="random",
                                          scatters=[axon_starting_coordinate],
                                         scatter_size=0.3)
    else:
        if verbose:
            print(f"Upstream node was None so don't have to adjust")
            
            
    if plot_final_limb_correspondence:
        nviz.plot_limb_correspondence(limb_correspondence_individual,
                             scatters=[limb_network_stating_info[0][0]["endpoint"],
                                      limb_network_stating_info[0][0]["touching_verts"]],
                              skeleton_colors=["red","blue"],
                             scatter_size=[0.2,0.07])
    
    if return_starting_info:
        return limb_correspondence_individual,limb_network_stating_info
    else:
        return limb_correspondence_individual'''


# --- 3/24 Addition ----
def high_fidelity_axon_decomposition(neuron_obj,
                                     plot_new_axon_limb_correspondence=False,
                                     plot_connecting_skeleton_fix = False,
                                     plot_final_limb_correspondence=False,
                                     return_starting_info=True,
                                    verbose = True,
                                         
                                # new parameters for stitching
                                stitch_floating_axon_pieces = None,#True,
                                filter_away_floating_pieces_inside_soma_bbox = True,
                                soma_bbox_multiplier = 2,
                                floating_piece_face_threshold = None,#12,
                                max_stitch_distance = None,#13000,#np.inf,
                                plot_new_axon_limb_correspondence_after_stitch = False,
                                
                                mp_only_revised_invalidation_d = False,
    ):
    """
    Purpose: To get the decomposition of the 
    axon with the a finer skeletonization (THIS VERSION NOW STITCHES PIECES OF THE AXON)

    Returns: a limb correspondence of the revised branches

    Pseudocode:
    1) Get the starting information for decomposition
    2) Split the axon mesh into just one connected mesh (aka filtering away the disconnected parts)
    3) Run the limb preprocessing
    4) Retriveing starting info from concept network
    5) Adjust the axon decomposition to connect to an upstream piece if there was one
    6) Return limb correspondence and starting information (IF WE REVISED THE STARTING INFO)

    --- Add back the floating mesh pieces using the stitching process --
    """
    if floating_piece_face_threshold is None:
        floating_piece_face_threshold = floating_piece_face_threshold_high_fid_axon_global
    
    if stitch_floating_axon_pieces is None:
        stitch_floating_axon_pieces = stitch_floating_axon_pieces_global
        
    if max_stitch_distance is None:
        max_stitch_distance = max_stitch_distance_high_fid_axon_global
    
    #---1) Get the starting information for decomposition ---


    axon_limb_name = neuron_obj.axon_limb_name
    curr_limb = neuron_obj[axon_limb_name]
    axon_starting_branch = neuron_obj.axon_starting_branch
    axon_starting_branch_mesh = curr_limb[axon_starting_branch].mesh
    axon_starting_coordinate = neuron_obj.axon_starting_coordinate

    upstream_node_to_axon_starting_branch = None

    # ---- 2/28: MOVE THE AXON BRANCHES BACK ENOUGH SO THE DENDRITES WONT BE ATTACHED



    if axon_starting_branch != curr_limb.current_starting_node:
        if verbose:
            print(f"Starting axon branch was not the limb starting node so generating border vertices and computing upstream branch")
        try:
            border_vertices_for_axon = tu.largest_border_to_coordinate(
                mesh = axon_starting_branch_mesh,
                coordinate =axon_starting_coordinate,
                distance_threshold = 1000,
                plot_border_vertices = False,
                error_on_no_border = True,
                plot_winning_border = False,
                verbose = False)
        except:
            """
            If no border vertices could be found then just going to assign closest vertices
            """
            n_closest_vertices = 1
            border_vertices_for_axon = tu.closest_n_attributes_to_coordinate(
                axon_starting_branch_mesh,
                coordinate = axon_starting_coordinate,
                attribute = "vertices",
                n = n_closest_vertices)
            
            
            

        upstream_node_to_axon_starting_branch = xu.upstream_node(curr_limb.concept_network_directional,
                                                                 axon_starting_branch
                                                                   )

    else:
        if verbose:
            print(f"Starting axon branch was the starting node so using border vertices and NOT computing upstream branch")
        border_vertices_for_axon = curr_limb.current_touching_soma_vertices
        upstream_node_to_axon_starting_branch = None

    soma_touching_vertices_dict = {0:[border_vertices_for_axon]}

    if verbose:
        print(f"upstream_node_to_axon_starting_branch = {upstream_node_to_axon_starting_branch}")



    #---2) Split the axon mesh into just one connected mesh (aka filtering away the disconnected parts)---

    # divide the meshes into the only big continuous one (because if disconnected probably)
    diff_meshes = tu.split_by_vertices(neuron_obj.axon_mesh)
    
    #-------- 5/27 change -------- 
    try:
        axon_mesh_filtered_idx = tu.filter_meshes_by_containing_coordinates(diff_meshes,
                                              nullifying_points=border_vertices_for_axon[0],
                                               method="distance",
                                               distance_threshold = min_distance_threshold,
                                              filter_away=False,
                                            return_indices=True)[0]
    except:
        raise Exception("Bad Axon")
        axon_mesh_filtered_idx = tu.filter_meshes_by_containing_coordinates(diff_meshes,
                                                  nullifying_points=border_vertices_for_axon,
                                                   method="distance",
                                                   distance_threshold = min_distance_threshold,
                                                  filter_away=False,
                                                return_indices=True)[0]
        
    axon_mesh_filtered = diff_meshes[axon_mesh_filtered_idx]
    meshes_to_stitch = [diff_meshes[k] for k in range(len(diff_meshes)) if k!= axon_mesh_filtered_idx]

    if verbose:
        if len(diff_meshes) > 1:
            print(f"More than 1 seperated mesh (taking the top one): {diff_meshes}")

    #---3) Run the limb preprocessing---
    limb_correspondence_individual,concept_network = pre.preprocess_limb(
        axon_mesh_filtered,
        soma_touching_vertices_dict = soma_touching_vertices_dict,
        error_on_no_starting_coordinates=True,            
        verbose = True,
        return_concept_network = True,
        meshparty_segment_size= meshparty_segment_size_axon_global,
        combine_close_skeleton_nodes_threshold_meshparty = combine_close_skeleton_nodes_threshold_meshparty_axon_global,
        filter_end_node_length_meshparty = filter_end_node_length_meshparty_axon_global,
        filter_end_node_length = filter_end_node_length_axon_global,
        invalidation_d=invalidation_d_axon_global,
        smooth_neighborhood=smooth_neighborhood_axon_global,
        
        mp_only_revised_invalidation_d=mp_only_revised_invalidation_d,
        **preprocessing_args,
        )

    #-- 4) Retriveing starting info from concept network --
    curr_concept_network = concept_network[0][0]

    starting_node = xu.get_starting_node(curr_concept_network,only_one=False)[0]

    starting_coordinate = curr_concept_network.nodes[starting_node]["starting_coordinate"]
    touching_soma_vertices = curr_concept_network.nodes[starting_node]["touching_soma_vertices"]

    limb_network_stating_info = {0:{0:{"touching_verts":touching_soma_vertices,
                                      "endpoint":starting_coordinate}}}

    if plot_new_axon_limb_correspondence:
        nviz.plot_limb_correspondence(limb_correspondence_individual,
                                     scatters = [starting_coordinate],
                                     scatter_size=1)



    # ---- 3/24 Addition: Stitching the leftover meshes

    if verbose:
        print(f"Limb Correspondence before stitching = {len(limb_correspondence_individual)}")

        
        
    if stitch_floating_axon_pieces:
        #need to specify want high fidelity skeletons
        floating_meshes_non_incorporated = nru.non_soma_touching_meshes_not_stitched(neuron_obj)
        
        
        if filter_away_floating_pieces_inside_soma_bbox: 
            
            
            if verbose:
                print(f"Filtering away non soma floating pieces near the soma")
                print(f"Before filter # of pieces = {len(floating_meshes_non_incorporated)}")
            floating_meshes_non_incorporated = tu.check_meshes_inside_mesh_bbox(neuron_obj["S0"].mesh,
                                floating_meshes_non_incorporated,
                                 return_inside=False,
                                bbox_multiply_ratio=soma_bbox_multiplier)
            
            if verbose:
                print(f"AFTER filter # of pieces = {len(floating_meshes_non_incorporated)}")
                
            
            

        if len(floating_meshes_non_incorporated) > 0:
            meshes_to_stitch = list(meshes_to_stitch) + list(floating_meshes_non_incorporated)
            
    if len(meshes_to_stitch) > 0:

        
#         su.compressed_pickle(meshes_to_stitch,"meshes_to_stitch")
#         su.compressed_pickle(np.array(starting_coordinate).reshape(-1,3),"excluded_node_coordinates")
#         print(f"max_stitch_distance = {max_stitch_distance}")
#         su.compressed_pickle({0:limb_correspondence_individual},"limb_correspondence_before")
        print(f"attemptin to stitch the following = {meshes_to_stitch}")
        
        limb_correspondence_with_floating_pieces = pre.attach_floating_pieces_to_limb_correspondence(
                        {0:limb_correspondence_individual},
                        floating_meshes= meshes_to_stitch,
                        #stitch_floating_axon_pieces = stitch_floating_axon_pieces,
                        max_stitch_distance=max_stitch_distance,
                        distance_to_move_point_threshold = 4000,
                        verbose = False,
                        excluded_node_coordinates = np.array(starting_coordinate).reshape(-1,3),
                        floating_piece_face_threshold = floating_piece_face_threshold,
                        meshparty_segment_size= meshparty_segment_size_axon_global,
                        combine_close_skeleton_nodes_threshold_meshparty = combine_close_skeleton_nodes_threshold_meshparty_axon_global,
                        filter_end_node_length_meshparty = filter_end_node_length_meshparty_axon_global,
                        filter_end_node_length = filter_end_node_length_axon_global,
                        invalidation_d=invalidation_d_axon_global,
                        smooth_neighborhood=smooth_neighborhood_axon_global,
                        use_adaptive_invalidation_d = False,
                        
                        mp_only_revised_invalidation_d=mp_only_revised_invalidation_d,
                        **preprocessing_args
        )
        
#         su.compressed_pickle(limb_correspondence_with_floating_pieces,"limb_correspondence_with_floating_pieces")
#         raise Exception("")
        
        limb_correspondence_individual_stitch = limb_correspondence_with_floating_pieces[0]
        #need to regenerate the concept network
        concept_network_floating_pieces = pre.calculate_limb_concept_networks(limb_correspondence_individual_stitch,
                                                                            limb_network_stating_info,
                                                                            run_concept_network_checks=True,
                                                                               )[0][0]

        starting_node_floating_pieces = xu.get_starting_node(concept_network_floating_pieces,
                                                             only_one=False)[0]

        limb_correspondence_individual = limb_correspondence_individual_stitch
        curr_concept_network = concept_network_floating_pieces
        starting_node = starting_node_floating_pieces

        
        if plot_new_axon_limb_correspondence_after_stitch:
            print(f"\n\nLimb correspondence after stitching")
            nviz.plot_limb_correspondence(limb_correspondence_individual,
                                         scatters = [starting_coordinate],
                                         scatter_size=1)
        

#     if verbose:
#         print(f"Limb Correspondence AFTER stitching = {len(limb_correspondence_individual_stitch)}")
        
    verbose = False

    # -- 5) Adjust the axon decomposition to connect to an upstream piece if there was one --
    """
    Psuedocode: 
    1) Get the starting axon branch and the upstream node
    2) Find the common endpoint
    3) Add a skeletal branch from starting_coordinate (new)
    and the common endpoint to the starting node of new
    axon decomposition

    """

    #1) Get the starting axon branch and the upstream node
    if upstream_node_to_axon_starting_branch is not None:
        print("Readjusting starting axon branch skeleton")

        #1) Get the starting axon branch and the upstream node
        axon_starting_branch,upstream_node_to_axon_starting_branch

        #2) Find the common endpoint
    #     shared_endpoint = shared_skeleton_endpoints_for_connected_branches(neuron_obj[axon_limb_name],
    #                                                 axon_starting_branch,
    #                                                      upstream_node_to_axon_starting_branch,
    #                                                 verbose=False)
        shared_endpoint = neuron_obj.axon_starting_coordinate

        if shared_endpoint.ndim > 1:
            shared_endpoint = shared_endpoint[0]

        #3) Add a skeletal branch from starting_coordinate (new)
        #and the common endpoint to the starting node of new
        #axon decomposition

        curr_endpoints = sk.find_skeleton_endpoint_coordinates(limb_correspondence_individual[starting_node]["branch_skeleton"])
        if len(nu.matching_rows(curr_endpoints,axon_starting_coordinate)) == 0:
            print("Fixing the axon starting branch endpoint to align with upstream branch")

            skeleton_pre_fix = limb_correspondence_individual[starting_node]["branch_skeleton"]
            new_skeleton_segment = np.array([starting_coordinate,shared_endpoint]).reshape(-1,2,3)

            limb_correspondence_individual[starting_node]["branch_skeleton"] = sk.stack_skeletons([skeleton_pre_fix,
                                                                                                   new_skeleton_segment
                                                                                                 ])
            if verbose:
                print(f"Starting Branch {starting_node} skeleton before fix: {skeleton_pre_fix.shape}")
                print(f"Starting Branch {starting_node} skeleton AFTER fix: {limb_correspondence_individual[starting_node]['branch_skeleton'].shape}")
        else:
            if verbose:
                print("Not attempting to fix the limb correspondence because the axon_starting_coordainte was already an endpoint")


        if plot_connecting_skeleton_fix:
            upstream_branch = neuron_obj[axon_limb_name][upstream_node_to_axon_starting_branch]
            meshes,skeletons = nviz.limb_correspondence_plottable(limb_correspondence_individual)
            nviz.plot_objects(meshes=meshes + [upstream_branch.mesh],
                              meshes_colors="random",
                              skeletons=skeletons +  [upstream_branch.skeleton],
                              skeletons_colors="random",
                                          scatters=[axon_starting_coordinate],
                                         scatter_size=0.3)
    else:
        if verbose:
            print(f"Upstream node was None so don't have to adjust")


    if plot_final_limb_correspondence:
        nviz.plot_limb_correspondence(limb_correspondence_individual,
                             scatters=[limb_network_stating_info[0][0]["endpoint"],
                                      limb_network_stating_info[0][0]["touching_verts"]],
                              skeleton_colors=["red","blue"],
                             scatter_size=[0.2,0.07])

    if return_starting_info:
        return limb_correspondence_individual,limb_network_stating_info
    else:
        return limb_correspondence_individual
    
    
# ----- 12/29: Helps with the human data --------------

floating_piece_face_threshold_expansion = 500
def limb_meshes_expansion(
    non_soma_touching_meshes,
    insignificant_limbs,
    soma_meshes,
    
    #Step 1: Filering
    plot_filtered_pieces = False,
    non_soma_touching_meshes_face_min = floating_piece_face_threshold_expansion,
    insignificant_limbs_face_min = floating_piece_face_threshold_expansion,
    
    #Step 2: Distance Graph Structure
    plot_distance_G = False,
    plot_distance_G_thresholded = False,
    max_distance_threshold = 500,
    
    #Step 3: 
    min_n_faces_on_path = 5_000,
    
    plot_final_limbs = False,
    plot_not_added_limbs = False,
    return_meshes_divided = True,
    
    
    verbose = False,
    
    ):
    """
    Purpose: To find the objects that should be made into 
    significant limbs for decomposition 
    (out of the non_soma_touching_meshes and insignificant_limbs )

    Pseudocode: 
    1) Filter the non-soma pieces and insignificant meshes
    2) Find distances between all of the significant pieces and form a graph structure
    3) Determine the meshes that should be made significant limbs
    a) find all paths from NST
    b) filter for those paths with a certain fae total
    3) fin all of the nodes right before the soma 
    and the unique set of those will be significant limbs
    
    
    
    Ex: 
    floating_piece_face_threshold_expansion = 500
    new_limbs_nst,il_still_idx,nst_still_meshes,il_still_meshes = pre.limb_meshes_expansion(
        neuron_obj_comb.non_soma_touching_meshes,
        neuron_obj_comb.insignificant_limbs,
        neuron_obj_comb["S0"].mesh,

        #Step 1: Filering
        plot_filtered_pieces = True,
    #     non_soma_touching_meshes_face_min = floating_piece_face_threshold_expansion,
    #     insignificant_limbs_face_min = floating_piece_face_threshold_expansion,

        #Step 2: Distance Graph Structure
        plot_distance_G = True,
        plot_distance_G_thresholded = True,
        max_distance_threshold = 500,

        #Step 3: 
        min_n_faces_on_path = 5_000,
        plot_final_limbs = True,
        plot_not_added_limbs = True,

        verbose = True
        )
    """


    soma_name = "Soma"
    #1) Filter the non_soma_touching and insignificant limbs for a size
    non_soma_touching_meshes= np.array(non_soma_touching_meshes)
    insignificant_limbs = np.array(insignificant_limbs)

    non_soma_touching_meshes_filt_idx = tu.filter_meshes_by_size(non_soma_touching_meshes,
                                                            non_soma_touching_meshes_face_min,
                                                             return_indices = True)
    non_soma_touching_meshes_filt = non_soma_touching_meshes[non_soma_touching_meshes_filt_idx]


    insignificant_limbs_filt_idx = tu.filter_meshes_by_size(insignificant_limbs,
                                                        insignificant_limbs_face_min,
                                                       return_indices = True)
    insignificant_limbs_filt = insignificant_limbs[insignificant_limbs_filt_idx]

    soma_mesh = tu.combine_meshes(soma_meshes)


    if verbose:
        print(f"len(non_soma_touching_meshes_filt) = {len(non_soma_touching_meshes_filt)}")
        print(f"len(insignificant_limbs_filt) = {len(insignificant_limbs_filt)}")


    if plot_filtered_pieces:
        soma_color = "red"
        non_soma_touch_color = "black"
        insignificant_limb_color = "green"

        nviz.plot_objects(
            soma_mesh,
            main_mesh_color="red",
            meshes=list(non_soma_touching_meshes_filt)+ list(insignificant_limbs_filt),
            meshes_colors=[non_soma_touch_color]*len(non_soma_touching_meshes_filt) + [insignificant_limb_color]*len(insignificant_limbs_filt),
        )

    if len(non_soma_touching_meshes_filt) <= 0:
        if return_meshes_divided:
            return np.array([]),np.array([]),np.array([]),np.array([])
        else:
            return np.array([])





    #2) Find distances between all of the significant pieces and form a graph structure
    meshes = [soma_mesh] + list(non_soma_touching_meshes_filt) + list(insignificant_limbs_filt)
    meshes_names = np.array([soma_name] + [
        f"nst_{k}" for k in range(len(non_soma_touching_meshes_filt))] + [
        f"il_{k}" for k in range(len(insignificant_limbs_filt))
    ])

    mesh_lookup = {k:v for k,v in zip(meshes_names,meshes)}

    mesh_edges = tu.mesh_list_distance_connectivity(
        meshes,
        return_G = False,
        verbose = False
        )


    edges_names_fixed = [list(k) + [v] for k,v in zip(meshes_names[mesh_edges[:,:2].astype('int')],mesh_edges[:,2])]
    G = nx.Graph()
    G.add_weighted_edges_from(edges_names_fixed)

    if plot_distance_G:
        print(f"plot_distance_G")
        nx.draw(G,with_labels = True)
        plt.show()

    G_sub =xu.query_to_subgraph(G,f"weight < {max_distance_threshold}")

    if plot_distance_G_thresholded:
        print(f"plot_distance_G_thresholded")
        nx.draw(G_sub,with_labels = True,)
        plt.show()


    #3) Determine the meshes that should be made significant limbs
    """
    a) find all paths from NST
    b) filter for those paths with a certain fae total
    3) fin all of the nodes right before the soma 
    and the unique set of those will be significant limbs

    """
    nst_nodes= [k for k in G_sub.nodes() if "nst" in k]

    if len(nst_nodes) <= 0:
        if return_meshes_divided:
            return np.array([]),np.array([]),np.array([]),np.array([])
        else:
            return np.array([])

    nst_shortest_paths = []
    for nst_n in nst_nodes:
        try:
            curr_path = xu.shortest_path(G_sub,nst_n,soma_name,weight="weight")
        except:
            if verbose:
                print(f"No path from {nst_n} to {soma_name}")
        else:
            nst_shortest_paths.append(curr_path)

    if verbose:
        print(f"nst_shortest_paths = {nst_shortest_paths}")

#     if len(nst_shortest_paths) <= 0:
#         if return_meshes_divided:
#             return np.array([]),np.array([]),np.array([]),np.array([])
#         else:
#             return np.array([])

    #b) filter for those paths with a certain fae total
    face_totals_for_paths = np.array([np.sum([len(mesh_lookup[k].faces) for k in z if k != soma_name]) for z in nst_shortest_paths])


    face_paths_idx = np.where(face_totals_for_paths>min_n_faces_on_path)[0]


    new_non_sig_limbs = np.unique([nst_shortest_paths[k][-2] for k in face_paths_idx])

    if verbose:
        print(f"face_totals_for_paths = {face_totals_for_paths}")
        print(f"face_paths_idx = {face_paths_idx}")
        print(f"new_non_sig_limbs = {new_non_sig_limbs}")


    
    nst_idx = [int(k.split("_")[-1]) for k in new_non_sig_limbs if "nst" in k]
    il_idx = [int(k.split("_")[-1]) for k in new_non_sig_limbs if "il" in k]

    new_limbs_nst = non_soma_touching_meshes_filt[nst_idx]
    new_limbs_il = insignificant_limbs_filt[il_idx]
    
    if plot_final_limbs:
        
        if len(list(new_limbs_nst) + list(new_limbs_il)) > 0:
            print(f"plot_final_limbs")
            nviz.plot_objects(
                soma_mesh,
                main_mesh_color="red",
                meshes=list(new_limbs_nst) + list(new_limbs_il),
                meshes_colors=mu.generate_non_randon_named_color_list(len(new_non_sig_limbs),colors_to_omit=["red"]),
            )
        else:
            print(f"No new limb meshes to plot")
        
        
    nst_still_idx = np.delete(np.arange(len(non_soma_touching_meshes)),non_soma_touching_meshes_filt_idx[nst_idx])
    il_still_idx = np.delete(np.arange(len(insignificant_limbs)),insignificant_limbs_filt_idx[il_idx])

    nst_still_meshes = non_soma_touching_meshes[nst_still_idx]
    il_still_meshes = insignificant_limbs[il_still_idx]
    
    if plot_not_added_limbs:
        print(f"plot_not_added_limbs")
        nviz.plot_objects(
            soma_mesh,
            main_mesh_color="red",
            meshes=list(nst_still_meshes) + list(il_still_meshes),
            meshes_colors=mu.generate_non_randon_named_color_list(len(nst_still_meshes) + len(il_still_meshes)
                                                                  ,colors_to_omit=["red"]),
        )
        
    if return_meshes_divided:
        if verbose:
            print(f"len(new_limbs_nst) = {len(new_limbs_nst)}")
            print(f"len(new_limbs_il) = {len(new_limbs_il)}")
            print(f"len(nst_still_meshes) = {len(nst_still_meshes)}")
            print(f"len(il_still_meshes) = {len(il_still_meshes)}")

        return new_limbs_nst,new_limbs_il,nst_still_meshes,il_still_meshes

    else:
        return new_non_sig_limbs
    
    
# ------------- parameters for stats ---------------

global_parameters_dict_default_decomp = dict(
        width_threshold_MAP = 500,
        size_threshold_MAP = 2000,
        size_threshold_MAP_stitch = 2000,
        apply_expansion = False,
        max_stitch_distance = 8000,
        max_stitch_distance_CGAL = 5000,
        filter_end_node_length = 4000,
        use_adaptive_invalidation_d = False,
        use_adaptive_invalidation_d_floating = True,
        axon_width_preprocess_limb_max = 200,
        limb_remove_mesh_interior_face_threshold = 0,
        surface_reconstruction_size = 1000,
        floating_piece_face_threshold = 50,
        invalidation_d = 12000,
        remove_mesh_interior_face_threshold = 0,
        
        mp_only_revised_invalidation_d = False,
        mp_only_invalidation_d_axon_buffer = None,
        mp_only_revised_invalidation_d_reference = None,
        mp_only_revised_width_reference = None,
)

global_parameters_dict_default_axon_decomp = dsu.DictType(
    combine_close_skeleton_nodes_threshold_meshparty_axon = 1300,
    filter_end_node_length_meshparty_axon = 1150,
    filter_end_node_length_axon = 1150,
    invalidation_d_axon = 1500,
    smooth_neighborhood_axon = (0,"tinyint unisgned"),
    meshparty_segment_size_axon = 100,
    stitch_floating_axon_pieces = True,
    max_stitch_distance_high_fid_axon = 5000,#2000,
    floating_piece_face_threshold_high_fid_axon = 50,
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_decomp,
    global_parameters_dict_default_axon_decomp
])

attributes_dict_default = dict(
)    

global_parameters_dict_microns = {}
attributes_dict_microns = {}
global_parameters_dict_microns_axon_decomp = {}


attributes_dict_h01 = dict(
)

global_parameters_dict_h01_decomp = dict(
    width_threshold_MAP = 1000,
    size_threshold_MAP = 10_000,
    size_threshold_MAP_stitch = 14_000,
    apply_expansion = True,
    max_stitch_distance = 13000,#5000,
    max_stitch_distance_CGAL = 13000,
    use_adaptive_invalidation_d = True,
    use_adaptive_invalidation_d_floating = True,
    axon_width_preprocess_limb_max = 350,
    
    limb_remove_mesh_interior_face_threshold = 150,
    
    floating_piece_face_threshold = 500,
)

global_parameters_dict_h01_axon_decomp = dict(
    invalidation_d_axon = 2500,
    stitch_floating_axon_pieces = False,
    combine_close_skeleton_nodes_threshold_meshparty_axon = 1700,
    max_stitch_distance_high_fid_axon = 8000,#5000,
    floating_piece_face_threshold_high_fid_axon = 450,
    
)

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_decomp,
    global_parameters_dict_h01_axon_decomp
])

# data_type = "default"
# algorithms = None

# modules_to_set = [pre,spu,nst]

# def set_global_parameters_and_attributes_by_data_type(dt,
#                                                      algorithms_list = None,
#                                                       modules = None,
#                                                      set_default_first = True,
#                                                       verbose=False):
#     if modules is None:
#         modules = modules_to_set
    
#     modu.set_global_parameters_and_attributes_by_data_type(modules,dt,
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


preprocessing_args = dict(
#                     meshparty_segment_size = meshparty_segment_size_axon_global,
                    combine_close_skeleton_nodes=True,
                    #combine_close_skeleton_nodes_threshold=1200,
#                     combine_close_skeleton_nodes_threshold_meshparty = combine_close_skeleton_nodes_threshold_meshparty_axon_global,
#                     filter_end_node_length_meshparty = filter_end_node_length_meshparty_axon_global,
#                     filter_end_node_length = filter_end_node_length_axon_global,

                    use_meshafterparty=False,
                    perform_cleaning_checks = True,


                    #concept_network parameters
                    run_concept_network_checks = True,

                    return_concept_network_starting_info=False,

                    #printing controls
                    print_fusion_steps=True,

                    check_correspondence_branches = True,
                    filter_end_nodes_from_correspondence=True,

#                     invalidation_d=invalidation_d_axon_global,
#                     smooth_neighborhood=smooth_neighborhood_axon_global,
                         )


#--- from neurd_packages ---
from . import neuron
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import soma_extraction_utils as sm
from . import spine_utils as spu
from . import axon_utils as au

#--- from mesh_tools ---
from mesh_tools import compartment_utils as cu
from mesh_tools import meshparty_skeletonize as m_sk
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import data_struct_utils as dsu
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm


from . import preprocess_neuron as pre