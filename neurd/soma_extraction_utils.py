from copy import deepcopy
import datajoint as dj
import os
from pathlib import Path
from pykdtree.kdtree import KDTree
import random
import time
import trimesh
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from datasci_tools import data_struct_utils as dsu

try:
    import cgal_Segmentation_Module as csm
except:
    pass

#from mesh_tools.trimesh_utils import split_significant_pieces,split
#from datasci_tools import numpy_utils as np

save_mesh_intermediates = False

soma_connectivity="edges"
"""
Checking the new validation checks
"""
def side_length_ratios(current_mesh):
    """
    Will compute the ratios of the bounding box sides
    To be later used to see if there is skewness
    """

    # bbox = current_mesh.bounding_box_oriented.vertices
    bbox = current_mesh.bounding_box_oriented.vertices
    x_axis_unique = np.unique(bbox[:,0])
    y_axis_unique = np.unique(bbox[:,1])
    z_axis_unique = np.unique(bbox[:,2])
    x_length = (np.max(x_axis_unique) - np.min(x_axis_unique)).astype("float")
    y_length = (np.max(y_axis_unique) - np.min(y_axis_unique)).astype("float")
    z_length = (np.max(z_axis_unique) - np.min(z_axis_unique)).astype("float")
    #print(x_length,y_length,z_length)
    #compute the ratios:
    xy_ratio = float(x_length/y_length)
    xz_ratio = float(x_length/z_length)
    yz_ratio = float(y_length/z_length)
    side_ratios = [xy_ratio,xz_ratio,yz_ratio]
    flipped_side_ratios = []
    for z in side_ratios:
        if z < 1:
            flipped_side_ratios.append(1/z)
        else:
            flipped_side_ratios.append(z)
    return flipped_side_ratios

def side_length_check(current_mesh,side_length_ratio_threshold=3):
    side_length_ratio_names = ["xy","xz","yz"]
    side_ratios = side_length_ratios(current_mesh)
    pass_threshold = [(k <= side_length_ratio_threshold) and
                      (k >= 1/side_length_ratio_threshold) for k in side_ratios]
    for i,(rt,truth) in enumerate(zip(side_ratios,pass_threshold)):
        if not truth:
            print(f"{side_length_ratio_names[i]} = {rt} ratio was beyong {side_length_ratio_threshold} multiplier")

    if False in pass_threshold:
        return False
    else:
        return True


def largest_mesh_piece(msh):
    mesh_splits_inner = msh.split(only_watertight=False)
    total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
    ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]
    return ordered_mesh_splits_inner[0]

def soma_volume_ratio(current_mesh,
                     #watertight_method="fill_holes,
                     watertight_method="poisson",
                      max_value = 1000,
                     ):
    """
    bounding_box_oriented: rotates the box to be less volume
    bounding_box : does not rotate the box and makes it axis aligned
    
    ** checks to see if closed mesh and if not then make closed **
    """
    poisson_temp_folder = Path.cwd() / "Poisson_temp"
    poisson_temp_folder.mkdir(parents=True,exist_ok=True)
    with meshlab.Poisson(poisson_temp_folder,overwrite=True) as Poisson_obj_temp:

        #get the largest piece
        lrg_mesh = largest_mesh_piece(current_mesh)
        if not lrg_mesh.is_watertight:
            #lrg_mesh.export("lrg_mesh_in_soma_volume.off")
            if watertight_method == "poisson":
                print("Using Poisson Surface Reconstruction for watertightness in soma_volume_ratio")
                #run the Poisson Surface reconstruction and get the largest piece
                new_mesh_inner,poisson_file_obj = Poisson_obj_temp(vertices=lrg_mesh.vertices,
                       faces=lrg_mesh.faces,
                       return_mesh=True,
                       delete_temp_files=True,
                       segment_id=random.randint(0,999999))
                lrg_mesh = largest_mesh_piece(new_mesh_inner)
            elif watertight_method == "fill_holes":
                print("Using the close holes feature for watertightness in soma_volume_ratio")
                fill_hole_obj = meshlab.FillHoles(max_hole_size=2000,
                                                 self_itersect_faces=False)

                mesh_filled_holes,fillholes_file_obj = fill_hole_obj(   
                                                    vertices=lrg_mesh.vertices,
                                                     faces=lrg_mesh.faces,
                                                     return_mesh=True,
                                                     delete_temp_files=True,
                                                    )
                lrg_mesh = largest_mesh_piece(mesh_filled_holes)
                
                    
            else:
                raise Exception(f"Unimplemented watertight_method requested: {watertight_method}")

        #turn the mesh into a closed mesh based on 
        print(f"mesh.is_watertight = {lrg_mesh.is_watertight}")
        
        if not lrg_mesh.is_watertight:
            lrg_mesh = lrg_mesh.convex_hull
        
        ratio_val = lrg_mesh.bounding_box.volume/lrg_mesh.volume
    #     if ratio_val < 1:
    #         raise Exception("Less than 1 value in volume ratio computation")
    if ratio_val < max_value:
        return ratio_val
    else:
        return max_value

def soma_volume_check(current_mesh,multiplier=8,verbose = True):
    ratio_val= soma_volume_ratio(current_mesh)
    if verbose:
        print("Inside sphere validater: ratio_val = " + str(ratio_val))
    if np.abs(ratio_val) > multiplier:
        return False
    return True



# -------------- Function that will extract the soma ------- #


def filter_away_inside_soma_pieces(
                            main_mesh_total,
                            pieces_to_test,
                            significance_threshold=2000,
                            n_sample_points=3,
                            required_outside_percentage = 0.9,
                            print_flag = False,
                            return_inside_pieces=False,
                            ):
    if not nu.is_array_like(main_mesh_total):
        main_mesh_total = [main_mesh_total]
    
    if not nu.is_array_like(pieces_to_test):
        pieces_to_test = [pieces_to_test]
        
    if len(pieces_to_test) == 0:
        print("pieces_to_test was empty so returning empty list or pieces")
        return pieces_to_test
    
    significant_pieces = [m for m in pieces_to_test if len(m.faces) >= significance_threshold]
    
    print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        return []
    
    final_mesh_pieces = []
    inside_pieces = []
    

    
    for i,mesh in enumerate(significant_pieces):
        outside_flag = True
        for j,main_mesh in enumerate(main_mesh_total):
            #gets the number of samples on the mesh to test (only the indexes)
            idx = np.random.choice(len(mesh.vertices),n_sample_points , replace=False)
            
            
            #gets the sample's vertices
            points = mesh.vertices[idx,:]

            start_time = time.time()

            #find the signed distance from the sampled vertices to the main mesh
            # Points outside the mesh will be negative
            # Points inside the mesh will be positive
            signed_distance = trimesh.proximity.signed_distance(main_mesh,points)

            #gets the 
            outside_percentage = sum(signed_distance <= 0)/n_sample_points
            
            if print_flag:
                print(f"Mesh piece {i} has outside_percentage {outside_percentage}, idx = {idx}")
            
            if outside_percentage < required_outside_percentage:
                if print_flag: 
                    print(f"Mesh piece {i} ({mesh}) inside mesh {j} :( ")
                outside_flag = False
                inside_pieces.append(mesh)
                break
        if outside_flag:
            if print_flag: 
                print(f"Mesh piece {i} OUTSIDE all meshes (corrected)")
            final_mesh_pieces.append(mesh)
        
    if return_inside_pieces:
        return final_mesh_pieces,inside_pieces
    else:
        return final_mesh_pieces


# subtacting the soma

''' old version 
def subtract_soma(current_soma,main_mesh,
                 significance_threshold=200,
                 distance_threshold = 550):
    print("\ninside Soma subtraction")
    start_time = time.time()
    face_midpoints_soma = np.mean(current_soma.vertices[current_soma.faces],axis=1)


    curr_mesh_bbox_restriction,faces_bbox_inclusion = (
                    tu.bbox_mesh_restriction(main_mesh,
                                             current_soma.bounds,
                                            mult_ratio=1.3)
    )

    face_midpoints_neuron = np.mean(curr_mesh_bbox_restriction.vertices[curr_mesh_bbox_restriction.faces],axis=1)

    soma_kdtree = KDTree(face_midpoints_soma)

    distances,closest_node = soma_kdtree.query(face_midpoints_neuron)

    distance_passed_faces  = distances<distance_threshold
    
    """ Older way of doing difference
    
    
    faces_to_keep = np.array(list(set(np.arange(0,len(main_mesh.faces))).difference(set(faces_bbox_inclusion[distance_passed_faces]))))
    """
    
    #newer way: using numpy functions
    faces_to_keep = np.delete(np.arange(len(main_mesh.faces)),
                                    faces_bbox_inclusion[distance_passed_faces])

    """
    #didn't work
    distance_passed_faces  = distances>=distance_threshold
    faces_to_keep = faces_bbox_inclusion[distance_passed_faces]
    
    """
    
    
    without_soma_mesh = main_mesh.submesh([faces_to_keep],append=True)
    
    
    

    #get the significant mesh pieces
    mesh_pieces = split_significant_pieces(without_soma_mesh,significance_threshold=significance_threshold)
    print(f"mesh pieces in subtact soma BEFORE the filtering inside pieces = {mesh_pieces}")
    
    mesh_pieces = filter_away_inside_soma_pieces(current_soma,mesh_pieces,
                                                           significance_threshold=significance_threshold)
    print(f"mesh pieces in subtact soma AFTER the filtering inside pieces = {mesh_pieces}")
    print(f"Total Time for soma mesh cancellation = {np.round(time.time() - start_time,3)}")
    return mesh_pieces
'''

def subtract_soma(current_soma_list,main_mesh,
                 significance_threshold=200,
                 distance_threshold = 1500,
                  connectivity="edges",
                 ):
    if type(current_soma_list) == type(trimesh.Trimesh()):
        current_soma_list = [current_soma_list]
    
    if type(current_soma_list) != list:
        raise Exception("Subtract soma was not passed a trimesh object or list for it's soma parameter")

        
    print("\ninside Soma subtraction")
    start_time = time.time()
    current_soma = tu.combine_meshes(current_soma_list)
    face_midpoints_soma = current_soma.triangles_center

    all_bounds = [k.bounds for k in  current_soma_list]
    

    curr_mesh_bbox_restriction,faces_bbox_inclusion = (
                    tu.bbox_mesh_restriction(main_mesh,
                                            all_bounds ,
                                            mult_ratio=1.3)
    )

    face_midpoints_neuron = curr_mesh_bbox_restriction.triangles_center

    soma_kdtree = KDTree(face_midpoints_soma)

    distances,closest_node = soma_kdtree.query(face_midpoints_neuron)

    distance_passed_faces  = distances<distance_threshold

    """ Older way of doing difference


    faces_to_keep = np.array(list(set(np.arange(0,len(main_mesh.faces))).difference(set(faces_bbox_inclusion[distance_passed_faces]))))
    """

    #newer way: using numpy functions
    faces_to_keep = np.delete(np.arange(len(main_mesh.faces)),
                                    faces_bbox_inclusion[distance_passed_faces])

    """
    #didn't work
    distance_passed_faces  = distances>=distance_threshold
    faces_to_keep = faces_bbox_inclusion[distance_passed_faces]

    """


    without_soma_mesh = main_mesh.submesh([faces_to_keep],append=True)




    #get the significant mesh pieces
    mesh_pieces = tu.split_significant_pieces(without_soma_mesh,significance_threshold=significance_threshold,
                                             connectivity=connectivity)
    
    # ----- 11/22 turns out weren't even using this part --------- #
#     print(f"mesh pieces in subtact soma BEFORE the filtering inside pieces = {mesh_pieces}")

#     current_mesh_pieces = filter_away_inside_soma_pieces(current_soma,mesh_pieces,
#                                          significance_threshold=significance_threshold,
#                                                         n_sample_points=5,
#                                                         required_outside_percentage=0.9)
#     print(f"mesh pieces in subtact soma AFTER the filtering inside pieces = {mesh_pieces}")
    print(f"Total Time for soma mesh cancellation = {np.round(time.time() - start_time,3)}")
    
    
    return mesh_pieces

def find_soma_centroids(soma_mesh_list):
    """
    Will return a list of soma centers if given one mesh or list of meshes
    the center is just found by averaging the vertices
    """
    if not nu.is_array_like(soma_mesh_list):
        soma_mesh_list = [soma_mesh_list]
    soma_mesh_list_centers = [np.array(np.mean(k.vertices,axis=0)).astype("float")
                           for k in soma_mesh_list]
    return soma_mesh_list_centers


def find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes,
                                        verbose=False):
    """
    Purpose: Will find the mesh piece that most likely has the 
    soma that was found by the poisson soma finding process
    
    """
    containing_mesh_indices=dict([(i,[]) for i,sm_c in enumerate(soma_mesh_list)])
    for k,sm_mesh in enumerate(soma_mesh_list):
        sm_center = tu.mesh_center_vertex_average(sm_mesh)
        viable_meshes = np.array([j for j,m in enumerate(split_meshes) 
                 if trimesh.bounds.contains(m.bounds,sm_center.reshape(-1,3))
                        ])
        if verbose:
            print(f"viable_meshes = {viable_meshes}")
        if len(viable_meshes) == 0:
            raise Exception(f"The Soma {k} with mesh {sm_center} was not contained in any of the boundying boxes")
        elif len(viable_meshes) == 1:
            containing_mesh_indices[k] = viable_meshes[0]
        else:
            #find which mesh is closer to the soma midpoint (NOT ACTUALLY WHAT WE WANT)
            min_distances_to_soma = []
            dist_min_to_soma = []
            for v_i in viable_meshes:
                # build the KD Tree
                viable_neuron_kdtree = KDTree(split_meshes[v_i].vertices)
                distances,closest_node = viable_neuron_kdtree.query(sm_mesh.vertices.reshape(-1,3))
                min_distances_to_soma.append(np.sum(distances))
                dist_min_to_soma.append(np.min(distances))
            if verbose:
                print(f"min_distances_to_soma = {min_distances_to_soma}")
                print(f"dist_min_to_soma = {dist_min_to_soma}")
            containing_mesh_indices[k] = viable_meshes[np.argmin(min_distances_to_soma)]

    return containing_mesh_indices
    
def grouping_containing_mesh_indices(containing_mesh_indices):
    """
    Purpose: To take a dictionary that maps the soma indiece to the 
             mesh piece containing the indices: {0: 0, 1: 0}
             
             and to rearrange that to a dictionary that maps the mesh piece
             to a list of all the somas contained inside of it 
             
    Pseudocode: 
    1) get all the unique mesh pieces and create a dictionary with an empty list
    2) iterate through the containing_mesh_indices dictionary and add each
       soma index to the list of the containing mesh index
    3) check that none of the lists are empty or else something has failed
             
    """
    
    unique_meshes = np.unique(list(containing_mesh_indices.values()))
    mesh_groupings = dict([(i,[]) for i in unique_meshes])
    
    #2) iterate through the containing_mesh_indices dictionary and add each
    #   soma index to the list of the containing mesh index
    
    for soma_idx, mesh_idx in containing_mesh_indices.items():
        mesh_groupings[mesh_idx].append(soma_idx)
    
    #3) check that none of the lists are empty or else something has failed
    len_lists = [len(k) for k in mesh_groupings.values()]
    
    if 0 in len_lists:
        raise Exception("One of the lists is empty when grouping somas lists")
        
    return mesh_groupings



""" ---------- 9/23: Addition to help filter away false somas"""


def original_mesh_soma(
    mesh,
    original_mesh,
    bbox_restriction_multiplying_ratio = 1.7,
    match_distance_threshold = 1500,
    mesh_significance_threshold = 1000,
    return_inside_pieces = True,
    return_multiple_pieces_above_threshold = True,
    soma_size_threshold = 8000,
    verbose = False,
    ):
    """
    Purpose: To take an approximation of the soma mesh (usually from a poisson surface reconstruction)
    and map it to faces on the original mesh

    Pseudocode: 
    1) restrict the larger mesh with a bounding box or current
    2) Remove all interior pieces
    3) Save the interior pieces if asked for a pass-back
    4) Split the Main Mesh
    5) Find the Meshes that contain the soma
    6) Map to the original with a high distance threshold
    7) Split the new mesh and take the largest
    """

    soma_mesh_list = [mesh]

    #1) restrict the larger mesh with a bounding box or current
    restricted_big_mesh,_ = tu.bbox_mesh_restriction(original_mesh,mesh,mult_ratio=bbox_restriction_multiplying_ratio)

    # -- Split into largest piece -- #
#     restr_split = tu.split_significant_pieces(restricted_big_mesh,
#                                               significance_threshold=mesh_significance_threshold,connectivity="edges")
#     restr_mesh_to_test = restr_split[0]

    restr_mesh_to_test = restricted_big_mesh


    #nviz.plot_objects(restricted_big_mesh)

    #2) Remove all interior pieces
    orig_mesh_to_map,inside_pieces = tu.remove_mesh_interior(restr_mesh_to_test,return_removed_pieces=True,size_threshold_to_remove=300)

    """ Old way that did a lot of splits: but not doing splits anymore
#     #nviz.plot_objects(restr_without_interior)
#     split_meshes = tu.split_significant_pieces(restr_without_interior,
#                                                significance_threshold=mesh_significance_threshold,connectivity="edges")

#     split_meshes = [restr_without_interior]

#     #nviz.plot_objects(meshes=split_meshes)

#     #5) Find the Meshes that contain the soma
#     containing_mesh_indices = find_soma_centroid_containing_meshes(soma_mesh_list,
#                                                 split_meshes,
#                                                 verbose=True)

#     soma_containing_meshes = grouping_containing_mesh_indices(containing_mesh_indices)

#     soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]

#     if len(soma_touching_meshes) != 1:
#         raise Exception(f"soma_touching_meshes not of size 1 {soma_touching_meshes}")

#     orig_mesh_to_map = soma_touching_meshes[0]
    """


    #6) Map to the original with a high distance threshold
    prelim_soma_mesh = tu.original_mesh_faces_map(original_mesh=orig_mesh_to_map,
                              submesh=mesh,
                              matching=True,
                              exact_match=False,
                              match_threshold = match_distance_threshold,
                              return_mesh=True)

    #nviz.plot_objects(prelim_soma_mesh)

    #7) Split the new mesh and take the largest
    split_meshes_after_backtrack = tu.split_significant_pieces(prelim_soma_mesh,
                                significance_threshold=mesh_significance_threshold,
                                                              connectivity=soma_connectivity,)
    if verbose:
        print(f"split_meshes_after_backtrack = {split_meshes_after_backtrack}")
        print(f"soma_size_threshold = {soma_size_threshold}")
        
    if save_mesh_intermediates:
        su.compressed_pickle(split_meshes_after_backtrack,"split_meshes_after_backtrack_2")

    if len(split_meshes_after_backtrack) == 0:
        return_mesh = None
    else:
        if return_multiple_pieces_above_threshold:
            return_mesh = [k for k in split_meshes_after_backtrack if len(k.faces)>soma_size_threshold]

    if return_inside_pieces:
        return return_mesh,inside_pieces
    else:
        return return_mesh


def original_mesh_soma_old(
    mesh,
    soma_meshes,
    sig_th_initial_split=100,
    subtract_soma_distance_threshold=550,
    split_meshes=None):
    
    """
    Purpose: Will help backtrack the Poisson surface reconstruction soma 
    to the soma of the actual mesh
    
    Application: By backtracking to mesh it will help with figuring
    out false somas from neural 3D junk
    
    Ex: 
    
    multi_soma_seg_ids = np.unique(multi_soma_seg_ids)
    seg_id_idx = -2
    seg_id = multi_soma_seg_ids[seg_id_idx]

    dec_mesh = get_decimated_mesh(seg_id)
    curr_soma_meshes = get_seg_extracted_somas(seg_id)
    curr_soma_mesh_list = get_soma_mesh_list(seg_id)

    from mesh_tools import skeleton_utils as sk
    sk.graph_skeleton_and_mesh(main_mesh_verts=dec_mesh.vertices,
                               main_mesh_faces=dec_mesh.faces,
                            other_meshes=curr_soma_meshes,
                              other_meshes_colors="red")
    

    soma_meshes_new = original_mesh_soma(
        mesh = dec_mesh,
        soma_meshes=curr_soma_meshes,
        sig_th_initial_split=15)
    
    
    """
    
    if not nu.is_array_like(soma_meshes):
        soma_meshes = [soma_meshes]
    
    main_mesh_total = mesh
    soma_mesh_list_centers = [tu.mesh_center_vertex_average(k) for k in soma_meshes]
    soma_mesh_list=soma_meshes
    
    

    #--- 2) getting the soma submeshes that are connected to each soma and identifiying those that aren't (and eliminating any mesh pieces inside the soma)

    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    
    if split_meshes is None:
        split_meshes = tu.split_significant_pieces(
                                    main_mesh_total,
                                    significance_threshold=sig_th_initial_split,
                                    connectivity=soma_connectivity,
                                    print_flag=False)

    print(f"# total split meshes = {len(split_meshes)}")


    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes,
                                            verbose=True)

    # filtering away any of the inside floating pieces: 
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]


    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        """
        *** want to save these pieces that are inside of the soma***
        """

        non_soma_touching_meshes,inside_pieces = filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split,
                                        return_inside_pieces = True)                                                      


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}") #Ex: {0: [0, 1]}







    #--- 3)  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces


    """
    for each soma touching mesh get the following:
    1) original soma meshes

    """


    soma_meshes_new = [None]*len(soma_meshes)

    
    for z,(mesh_idx, soma_idxes) in enumerate(soma_containing_meshes.items()):
        print(f"\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx] #gets the current soma containing mesh

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        print(f"current_soma_mesh_list = {current_soma_mesh_list}")
        print(f"current_mesh = {current_mesh}")
        mesh_pieces_without_soma = subtract_soma(current_soma_mesh_list,current_mesh,
                                                 distance_threshold=subtract_soma_distance_threshold,
                                                    significance_threshold=250)
        
        print(f"mesh_pieces_without_soma = {mesh_pieces_without_soma}")
        
        if len(mesh_pieces_without_soma) == 0:
            return None
        
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        debug = False
        
        
        
        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)
        if debug:
            print(f"mesh_pieces_without_soma_stacked = {mesh_pieces_without_soma_stacked}")
        
        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        
        if debug:
            print(f"soma_faces = {soma_faces}")
        soma_meshes = current_mesh.submesh([soma_faces],append=True,repair=False)
        if debug:
            print(f"soma_meshes = {soma_meshes}")
        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False)
        #take the top largest ones depending how many were originally in the soma list
        
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        
        soma_face_components = soma_face_components[:len(soma_mesh_list)]


        
        #storing the new somas
        for zz,s_idx in enumerate(soma_idxes):
            soma_meshes_new[s_idx] = seperate_soma_meshes[zz]
            
        
    return soma_meshes_new
    
def extract_soma_center(
    segment_id=12345,
    current_mesh_verts=None,
    current_mesh_faces=None,
    mesh = None,

    outer_decimation_ratio= None,
    large_mesh_threshold = None,#60000,
    large_mesh_threshold_inner = None, #was changed so dont filter away som somas
    soma_width_threshold = None,
    soma_size_threshold = None, #changed this to smaller so didn't filter some somas away
    inner_decimation_ratio = None,

    segmentation_clusters =3,
    segmentation_smoothness = 0.2,

    volume_mulitplier=None,
    #side_length_ratio_threshold=3
    side_length_ratio_threshold=None,
    soma_size_threshold_max=None,#240000,#192000, #this puts at 12000 once decimated, another possible is 256000
    delete_files=True,
    backtrack_soma_mesh_to_original=None, #should either be None or 
    boundary_vertices_threshold=None,#700 the previous threshold used
    poisson_backtrack_distance_threshold=None,#1500 the previous threshold used
    close_holes=None,

    #------- 11/12 Additions --------------- #

    #these arguments are for removing inside pieces
    remove_inside_pieces = None,
    size_threshold_to_remove=None, #size accounting for the decimation


    pymeshfix_clean=None,
    check_holes_before_pymeshfix=None,
    second_poisson=None,
    segmentation_at_end=None,
    last_size_threshold = None,#1300,

    largest_hole_threshold = None,
    max_fail_loops = None,#np.inf,
    perform_pairing = None,
    verbose=False,

    return_glia_nuclei_pieces = True,
    backtrack_soma_size_threshold=None,
    backtrack_match_distance_threshold=1500,

    filter_inside_meshes_after_glia_removal = False,
    max_mesh_sized_filtered_away = 90000,

    filter_inside_somas=True,
    backtrack_segmentation_on_fail = True,

    #arguments for the glia pieces
    glia_pieces = None,
    nuclei_pieces = None,
    glia_volume_threshold_in_um = None, #minimum volume of glia mesh
    glia_n_faces_threshold = None, #minimum number of faes for glia mesh if volume is not defined
    glia_n_faces_min = None, #minimum number of faes for glia mesh if glia volume defined
    nucleus_min = None, #minimum number of faces for nuclei mesh peices
    nucleus_max = None, #maximum number of faces for nuclie mesh pieces if glia volume is not defined

    second_pass_size_threshold = None,

    **kwargs
    ):
    
    # ------- Nuclei parameters -------------
    if nucleus_min is None:
        nucleus_min = nucleus_min_global
        
    if nucleus_max is None:
        nucleus_max = nucleus_max_global
        
    # ------ Glia parameters ------------
    if glia_volume_threshold_in_um is None:
        glia_volume_threshold_in_um = glia_volume_threshold_in_um_global
        
    if glia_n_faces_threshold is None:
        glia_n_faces_threshold = glia_n_faces_threshold_global
        
    if glia_n_faces_min is None:
        glia_n_faces_min = glia_n_faces_min_global
        
    # -------- Soma parameters -----------
    if outer_decimation_ratio is None:
        outer_decimation_ratio = outer_decimation_ratio_global
        
    if large_mesh_threshold is None:
        large_mesh_threshold = large_mesh_threshold_global
    
    if large_mesh_threshold_inner is None:
        large_mesh_threshold_inner = large_mesh_threshold_inner_global
    
    if inner_decimation_ratio is None:
        inner_decimation_ratio = inner_decimation_ratio_global
        
        
    if max_fail_loops is None:
        max_fail_loops = max_fail_loops_global
    if remove_inside_pieces is None:
        remove_inside_pieces = remove_inside_pieces_global
    if size_threshold_to_remove is None:
        size_threshold_to_remove = size_threshold_to_remove_global
    if pymeshfix_clean is None:
        pymeshfix_clean = pymeshfix_clean_global
    if check_holes_before_pymeshfix is None:
        check_holes_before_pymeshfix = check_holes_before_pymeshfix_global
    if second_poisson is None:
        second_poisson = second_poisson_global
    if soma_width_threshold is None:
        soma_width_threshold = soma_width_threshold_global
    if soma_size_threshold is None:
        soma_size_threshold = soma_size_threshold_global
    if soma_size_threshold_max is None:
        soma_size_threshold_max = soma_size_threshold_max_global
    if volume_mulitplier is None:
        volume_mulitplier = volume_mulitplier_global
    if side_length_ratio_threshold is None:
        side_length_ratio_threshold = side_length_ratio_threshold_global
    if perform_pairing is None:
        perform_pairing = perform_pairing_global
    if backtrack_soma_mesh_to_original is None:
        backtrack_soma_mesh_to_original = backtrack_soma_mesh_to_original_global
    if backtrack_soma_size_threshold is None:
        backtrack_soma_size_threshold = backtrack_soma_size_threshold_global
    if poisson_backtrack_distance_threshold is None:
        poisson_backtrack_distance_threshold = poisson_backtrack_distance_threshold_global
    if close_holes is None:
        close_holes = close_holes_global
    if boundary_vertices_threshold is None:
        boundary_vertices_threshold = boundary_vertices_threshold_global
    if last_size_threshold is None:
        last_size_threshold = last_size_threshold_global
    if segmentation_at_end is None:
        segmentation_at_end = segmentation_at_end_global
    if largest_hole_threshold is None:
        largest_hole_threshold = largest_hole_threshold_global
    if second_pass_size_threshold is None:
        second_pass_size_threshold = second_pass_size_threshold_global
        
    
    
    """
    Purpose: To extract the glia/neurons and soma fetures
    
    Argumetns for glia/nucleus extraction: 
    glia_volume_threshold_in_um = 2500 #minimum volume of glia mesh
    glia_n_faces_threshold = 400000 #minimum number of faes for glia mesh if volume is not defined
    glia_n_faces_min = 100000 #minimum number of faes for glia mesh if glia volume defined
    nucleus_min = 700 #minimum number of faces for nuclei mesh peices
    nucleus_max = None #maximum number of faces for nuclie mesh pieces if glia volume is not defined
    
    
    
    
    Arguments for soma extraction: 
    # **Note: All of the thresholds get scaled by the decimation ratios applied before

    outer_decimation_ratio= 0.25 #decimation ratio for 1st round of decimation
    large_mesh_threshold = 20000 #minimum face count threshold after 1st round of decimation
    large_mesh_threshold_inner = 13000 #minimum face count threshold after poisson reconstruction and split
    inner_decimation_ratio = 0.25 #decimation ratio for 2nd round of decimation after poisson reconstruction
    max_fail_loops = 10 #number of times soma finding can fail in finding an invalid soma in the outer/inner decimation loop

    # other cleaning methods to run on segments after first decimation
    remove_inside_pieces = True #whether to remove inside pieces before processing
    size_threshold_to_remove=1000, #minium number of faces of mesh piece to remove
    pymeshfix_clean=False
    check_holes_before_pymeshfix=False
    second_poisson=False

    #after 2nd round of decimation and mesh segmentation is applied
    soma_width_threshold = 0.32 # minimuum sdf values of mesh segments
    soma_size_threshold = 9000 # minimum face count threshold of mesh segments
    soma_size_threshold_max=1_200_000# maximum face count threshold of mesh segments

    #parameters for checking possible viable somas after mesh segmentation
    volume_mulitplier=8 #for soma_volume_check funcion that makes sure not highly skewed bounding box volume to mesh volume 
    side_length_ratio_threshold=6 #for side_length_check function making sure not highly swkewed x,y,z side length ratios
    delete_files=True #deletes all files in temp folder

    #whether to group certain poisson representations of somas together before backtrack to mesh
    perform_pairing = False


    backtrack_soma_mesh_to_original=True #will backtrack the poisson reconstruction soma to the original mesh
    backtrack_soma_size_threshold=8000 #minimum number of faces for a backtracked mesh


    #for filtering away somas once get backtracked to original mesh with certain filters
    poisson_backtrack_distance_threshold=None #maximum distance between poisson reconstruction and backtracked soma
    close_holes=False #whether to close holes when doing poisson backtrack

    boundary_vertices_threshold=None #filter away somas if too many boundary vertices


    #filters at the very end for meshes that made it thorugh
    last_size_threshold = 2000 #min faces count

    segmentation_at_end=True #whether to attempt to split meshes at end
    largest_hole_threshold = 17000 #maximum hole length for a soma to allow the segmentaiton split at end

    

    """

    global_start_time = time.time()

    #Adjusting the thresholds based on the decimations
    large_mesh_threshold = large_mesh_threshold*outer_decimation_ratio
    large_mesh_threshold_inner = large_mesh_threshold_inner*outer_decimation_ratio
    soma_size_threshold = soma_size_threshold*outer_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*outer_decimation_ratio
    max_mesh_sized_filtered_away = max_mesh_sized_filtered_away*outer_decimation_ratio

    #adjusting for inner decimation
    soma_size_threshold = soma_size_threshold*inner_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*inner_decimation_ratio
    
    print(f"Current Arguments Using (adjusted for decimation):\n large_mesh_threshold= {large_mesh_threshold}"
                 f" \nlarge_mesh_threshold_inner = {large_mesh_threshold_inner}"
                  f" \nsoma_size_threshold = {soma_size_threshold}"
                 f" \nsoma_size_threshold_max = {soma_size_threshold_max}"
                 f"\nouter_decimation_ratio = {outer_decimation_ratio}"
                 f"\ninner_decimation_ratio = {inner_decimation_ratio}"
         f"\nmax_mesh_sized_filtered_away = {max_mesh_sized_filtered_away}")


    # ------------------------------


    temp_folder = f"./{segment_id}"
    temp_object = Path(temp_folder)
    #make the temp folder if it doesn't exist
    temp_object.mkdir(parents=True,exist_ok=True)

    #making the decimation and poisson objections
    Dec_outer = meshlab.Decimator(outer_decimation_ratio,temp_folder,overwrite=True)
    Dec_inner = meshlab.Decimator(inner_decimation_ratio,temp_folder,overwrite=True)
    Poisson_obj = meshlab.Poisson(temp_folder,overwrite=True)

    if mesh is None:
        recov_orig_mesh = trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)
    else:
        recov_orig_mesh = mesh
    
    
    if glia_pieces is not None and nuclei_pieces is not None:
        recov_orig_mesh_no_interior = tu.subtract_mesh(recov_orig_mesh,[glia_pieces,nuclei_pieces])
    else:
        try:
            recov_orig_mesh_no_interior, glia_pieces, nuclei_pieces  = sm.remove_nuclei_and_glia_meshes(
                recov_orig_mesh,
                verbose=True,
                glia_volume_threshold_in_um = glia_volume_threshold_in_um,
                glia_n_faces_threshold = glia_n_faces_threshold,
                glia_n_faces_min = glia_n_faces_min,
                nucleus_min= nucleus_min,
                nucleus_max=nucleus_max,                                                                                

                )
        except:
            print("**Unable to remove_nuclei_and_glia_meshes: so just continuing without doing so **")
            recov_orig_mesh_no_interior = recov_orig_mesh
            glia_pieces = []
            nuclei_pieces = []
    #recov_orig_mesh_no_interior = tu.remove_mesh_interior(recov_orig_mesh)

    

    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Dec_outer(vertices=recov_orig_mesh_no_interior.vertices,
             faces=recov_orig_mesh_no_interior.faces,
             segment_id=segment_id,
             return_mesh=True,
             delete_temp_files=False)

    # if remove_inside_pieces:
    #     print("removing mesh interior after decimation")
    #     new_mesh = tu.remove_mesh_interior(new_mesh,size_threshold_to_remove=size_threshold_to_remove)

    #preforming the splits of the decimated mesh

    mesh_splits = new_mesh.split(only_watertight=False)

    #get the largest mesh
    mesh_lengths = np.array([len(split.faces) for split in mesh_splits])


    total_mesh_split_lengths = [len(k.faces) for k in mesh_splits]
    ordered_mesh_splits = mesh_splits[np.flip(np.argsort(total_mesh_split_lengths))]
    list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold]

    print(f"Total found significant pieces before Poisson = {list_of_largest_mesh}")

    # --------- 1/11 Addition: Filtering away large meshes that are inside another --------- #
    if filter_inside_meshes_after_glia_removal:
        print(f"Filtering away larger meshes that are inside others, before # of meshes = {len(list_of_largest_mesh)}")
        list_of_largest_mesh = tu.filter_away_inside_meshes(list_of_largest_mesh,verbose=True,return_meshes=True,
                                                           max_mesh_sized_filtered_away=max_mesh_sized_filtered_away)
        print(f"After # of meshes = {len(list_of_largest_mesh)}")


    #if no significant pieces were found then will use smaller threshold
    if len(list_of_largest_mesh)<=0:
        print(f"Using smaller large_mesh_threshold because no significant pieces found with {large_mesh_threshold}")
        list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold/2]

    total_soma_list = []
    total_poisson_list = []
    total_soma_list_sdf = []

    filtered_soma_list_components = []

    for i in range(2):
        if len(filtered_soma_list_components)>0 or (i==1 and second_pass_size_threshold is None):
            if verbose:
                print(f"Not need to do a second pass because already found a soma")
                
            if delete_files:
                #now erase all of the files used
                from shutil import rmtree

                #remove the directory with the meshes
                rmtree(str(temp_object.absolute()))

                #removing the temporary files
                temp_folder = Path("./temp")
                temp_files = [x for x in temp_folder.glob('**/*')]
                seg_temp_files = [x for x in temp_files if str(segment_id) in str(x)]

                for f in seg_temp_files:
                    f.unlink()
            break
            
        if i == 1:
            if verbose:
                print(f"Using backup size thresholds")

            soma_size_threshold = second_pass_size_threshold
            last_size_threshold = second_pass_size_threshold
            backtrack_soma_size_threshold = second_pass_size_threshold


        
        #start iterating through where go through all pieces before the poisson reconstruction
        no_somas_found_in_big_loop = 0
        for i,largest_mesh in enumerate(list_of_largest_mesh):
            print(f"----- working on large mesh #{i}: {largest_mesh}")

            if remove_inside_pieces:
                print("remove_inside_pieces requested ")
                try:
                    largest_mesh = tu.remove_mesh_interior(largest_mesh,
                                                           size_threshold_to_remove=size_threshold_to_remove,
                                                          try_hole_close=False)
                except:
                    print("Unable to remove inside pieces in list_of_largest_mesh")
                    largest_mesh = largest_mesh


            if pymeshfix_clean:
                print("Requested pymeshfix_clean")
                """
                Don't have to check if manifold anymore actually just have to plug the holes
                """
                hole_groups = tu.find_border_face_groups(largest_mesh)
                if len(hole_groups) > 0:
                    largest_mesh_filled_holes = tu.fill_holes(largest_mesh,max_hole_size = 10000)
                else:
                    largest_mesh_filled_holes = largest_mesh

                if check_holes_before_pymeshfix:
                    hole_groups = tu.find_border_face_groups(largest_mesh_filled_holes)
                else:
                    print("Not checking if there are still existing holes before pymeshfix")
                    hole_groups = []

                if len(hole_groups) > 0:
                    #segmentation_at_end = False
                    print(f"*** COULD NOT FILL HOLES WITH MAX SIZE OF {np.max([len(k) for k in hole_groups])} so not applying pymeshfix and segmentation_at_end = {segmentation_at_end}")

        #                 tu.write_neuron_off(largest_mesh_filled_holes,"largest_mesh_filled_holes")
        #                 raise Exception()
                else:
                    print("Applying pymeshfix_clean because no more holes")
                    largest_mesh = tu.pymeshfix_clean(largest_mesh_filled_holes,verbose=True)

            if second_poisson:
                print("Applying second poisson run")
                current_neuron_poisson = tu.poisson_surface_reconstruction(largest_mesh)
                largest_mesh = tu.split_significant_pieces(current_neuron_poisson,connectivity=soma_connectivity)[0]

            somas_found_in_big_loop = False

            largest_file_name = str(output_obj.stem) + "_largest_piece.off"
            pre_largest_mesh_path = temp_object / Path(str(output_obj.stem) + "_largest_piece.off")
            pre_largest_mesh_path = pre_largest_mesh_path.absolute()
            print(f"pre_largest_mesh_path = {pre_largest_mesh_path}")
            # ******* This ERRORED AND CALLED OUR NERUON NONE: 77697401493989254 *********
            new_mesh_inner,poisson_file_obj = Poisson_obj(vertices=largest_mesh.vertices,
                       faces=largest_mesh.faces,
                       return_mesh=True,
                       mesh_filename=largest_file_name,
                       delete_temp_files=False)


            #splitting the Poisson into the largest pieces and ordering them
            mesh_splits_inner = new_mesh_inner.split(only_watertight=False)
            total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
            ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]

            list_of_largest_mesh_inner = [k for k in ordered_mesh_splits_inner if len(k.faces) > large_mesh_threshold_inner]
            print(f"Total found significant pieces AFTER Poisson = {list_of_largest_mesh_inner}")

            n_failed_inner_soma_loops = 0
            for j, largest_mesh_inner in enumerate(list_of_largest_mesh_inner):
                to_add_list = []
                to_add_list_sdf = []

                print(f"----- working on mesh after poisson #{j}: {largest_mesh_inner}")

                largest_mesh_path_inner = str(poisson_file_obj.stem) + "_largest_inner.off"

                #Decimate the inner poisson piece
                largest_mesh_path_inner_decimated,output_obj_inner = Dec_inner(
                                    vertices=largest_mesh_inner.vertices,
                                     faces=largest_mesh_inner.faces,
                                    mesh_filename=largest_mesh_path_inner,
                                     return_mesh=True,
                                     delete_temp_files=False)

                dec_splits = tu.split_significant_pieces(largest_mesh_path_inner_decimated,significance_threshold=15,
                                                        connectivity=soma_connectivity,)
                print(f"\n-------Splits after inner decimation len = {len(dec_splits)}--------\n")

                if len(dec_splits) == 0:
                    print("There were no signifcant splits after inner decimation")
                    n_failed_inner_soma_loops += 1
                else:
                    print(f"done exporting decimated mesh: {largest_mesh_path_inner}")

                    largest_mesh_path_inner_decimated_clean = dec_splits[0]
                    if save_mesh_intermediates:
                        su.compressed_pickle(largest_mesh_path_inner_decimated_clean,f"largest_mesh_path_inner_decimated_clean_{j}")
                    """ # ----------- 1/12: Addition that does the segmentation again -------------------- #"""

                    for ii in range(3):
                        print(f"\n    --- On segmentation loop {ii} --")
                        print(f"largest_mesh_path_inner_decimated_clean = {largest_mesh_path_inner_decimated_clean}\n")

                        '''
                        faces = np.array(largest_mesh_path_inner_decimated_clean.faces)
                        verts = np.array(largest_mesh_path_inner_decimated_clean.vertices)

                        # may need to do some processing


                        segment_id_new = int(str(segment_id) + f"{i}{j}")
                        #print(f"Before the classifier the pymeshfix_clean = {pymeshfix_clean}")
                        verts_labels, faces_labels, soma_value,classifier = wcda.extract_branches_whole_neuron(
                                                import_Off_Flag=False,
                                                segment_id=segment_id_new,
                                                vertices=verts,
                                                 triangles=faces,
                                                pymeshfix_Flag=False,
                                                 import_CGAL_Flag=False,
                                                 return_Only_Labels=True,
                                                 clusters=3,
                                                 smoothness=0.2,
                                                soma_only=True,
                                                return_classifier = True
                                                )
                        print(f"soma_sdf_value = {soma_value}")


                        #total_poisson_list.append(largest_mesh_path_inner_decimated)

                        # Save all of the portions that resemble a soma
                        median_values = np.array([v["median"] for k,v in classifier.sdf_final_dict.items()])
                        segmentation = np.array([k for k,v in classifier.sdf_final_dict.items()])

                        #order the compartments by greatest to smallest
                        sorted_medians = np.flip(np.argsort(median_values))
                        print(f"segmentation[sorted_medians],median_values[sorted_medians] = {(segmentation[sorted_medians],median_values[sorted_medians])}")
                        print(f"Sizes = {[classifier.sdf_final_dict[g]['n_faces'] for g in segmentation[sorted_medians]]}")
                        print(f"soma_size_threshold = {soma_size_threshold}")
                        print(f"soma_size_threshold_max={soma_size_threshold_max}")

                        valid_soma_segments_width = [g for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]
                        valid_soma_segments_sdf = [h for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                            and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]

                        print("valid_soma_segments_width")
                        '''

                        """# ----------- 1/14 Addition that will just use the trimesh segmentation function ------------"""
                        
                        print(f"largest_mesh_path_inner_decimated_clean = {largest_mesh_path_inner_decimated_clean}")
                        print(f"soma_size_threshold = {soma_size_threshold}")
                        print(f"soma_size_threshold_max = {soma_size_threshold_max}")
                        print(f"soma_width_threshold = {soma_width_threshold}")
                        
                        divided_meshes,divided_meshes_sdf = tu.mesh_segmentation(mesh=largest_mesh_path_inner_decimated_clean,clusters=segmentation_clusters,
                                                                                 smoothness=segmentation_smoothness)

                        divided_meshes_len = np.array([len(k.faces) for k in divided_meshes])
                        valid_indexes = np.where((divided_meshes_len > soma_size_threshold) & 
                                                 (divided_meshes_len < soma_size_threshold_max) & 
                                                (np.array(divided_meshes_sdf) > soma_width_threshold))[0]

                        valid_soma_meshes = [divided_meshes[k] for k in valid_indexes]
                        valid_soma_segments_width = [divided_meshes_sdf[k] for k in valid_indexes]
                        
                        if save_mesh_intermediates:
                            su.compressed_pickle(valid_soma_meshes,f"valid_soma_meshes_{j}")
                            su.compressed_pickle(valid_soma_meshes,f"valid_soma_segments_width_{j}")



                        """# =------------- 1/12: Addition that will repeat this loop --------------"""
                        if len(valid_soma_meshes) > 0:
                            break
                        else:
                            """
                            Pseudocode: 
                            Get the largest mesh segment

                            Old:
                            new_mesh_try_faces = np.where(classifier.labels_list == nu.mode_1d(classifier.labels_list))[0]
                            largest_mesh_path_inner_decimated_clean = largest_mesh_path_inner_decimated_clean.submesh([new_mesh_try_faces],append=True)
                            """
                            largest_mesh_path_inner_decimated_clean = divided_meshes[0]

                    if len(valid_soma_segments_width) > 0:
                        print(f"      ------ Found {len(valid_soma_segments_width)} viable somas: {valid_soma_segments_width}")
                        somas_found_in_big_loop = True
                        #get the meshes only if signfiicant length

                        for soma_mesh,sdf in zip(valid_soma_meshes,valid_soma_segments_width):


                            # ---------- No longer doing the extra checks in here --------- #


                            curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
                            curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)


                            if curr_side_len_check and curr_volume_check:
                                #check if we can split this into two
                                to_add_list.append(soma_mesh)
                                to_add_list_sdf.append(sdf)

                                """  -----------Removed 1/12: When trying to force a split between them 
                                possible_smoothness = [0.2,0.05,0.01]
                                for smooth_value in possible_smoothness:
                                    #1) Run th esegmentation algorithm again to segment the mesh (had to run the higher smoothing to seperate some)
                                    mesh_extra, mesh_extra_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=smooth_value,verbose=True)
                                    mesh_extra = np.array(mesh_extra)

                                    #2) Filter out meshes by sizs and sdf threshold
                                    mesh_extra_lens = np.array([len(kk.faces) for kk in mesh_extra])
                                    filtered_meshes_idx = np.where((mesh_extra_lens >= soma_size_threshold) & (mesh_extra_lens <= soma_size_threshold_max) & (mesh_extra_sdf>soma_width_threshold))[0]

                                    if len(filtered_meshes_idx) >= 2:
                                        if verbose:
                                            print(f"Breakin on smoothness: {smooth_value}")
                                        break

                                if len(filtered_meshes_idx) >= 2:
                                    filtered_meshes = mesh_extra[filtered_meshes_idx]
                                    filtered_meshes_sdf = mesh_extra_sdf[filtered_meshes_idx]

                                    to_add_list_retry = []
                                    to_add_list_sdf_retry = []

                                    for f_m,f_m_sdf in zip(filtered_meshes,filtered_meshes_sdf):
                                        curr_side_len_check_retry = side_length_check(f_m,side_length_ratio_threshold)
                                        curr_volume_check_retry = soma_volume_check(f_m,volume_mulitplier)

                                        if curr_side_len_check_retry and curr_volume_check_retry:
                                            to_add_list_retry.append(f_m)
                                            to_add_list_sdf_retry.append(f_m_sdf)

                                    if len(to_add_list_retry)>1:
                                        if verbose:
                                            print("Using the new feature that split the soma further into more groups")
                                        to_add_list += to_add_list_retry
                                        to_add_list_sdf += to_add_list_sdf_retry

                                    else:
                                        to_add_list.append(soma_mesh)
                                        to_add_list_sdf.append(sdf)


                                else:
                                    to_add_list.append(soma_mesh)
                                    to_add_list_sdf.append(sdf)
                                """

                            else:
                                # ---------- 1/7 Addition: Trying one more additional cgal segmentation to see if there is actually a soma ---
                                """
                                Pseudocode: 
                                1) Run th esegmentation algorithm again to segment the mesh
                                2) Filter out meshes by sizs and sdf threshold
                                3) If there are any remaining meshes, pick the largest sdf mesh and test for volume and side length check
                                --> if matches then adds
                                """

                                print(f"->Attempting retry of soma because failed first checks: "
                                         f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                                #1) Run th esegmentation algorithm again to segment the mesh
                                mesh_extra, mesh_extra_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=0.2,verbose=True)
                                mesh_extra = np.array(mesh_extra)

                                #2) Filter out meshes by sizs and sdf threshold
                                mesh_extra_lens = np.array([len(kk.faces) for kk in mesh_extra])
                                filtered_meshes_idx = np.where((mesh_extra_lens >= soma_size_threshold) & (mesh_extra_lens <= soma_size_threshold_max) & (mesh_extra_sdf>soma_width_threshold))[0]


                                if len(filtered_meshes_idx) > 0:
                                    filtered_meshes = mesh_extra[filtered_meshes_idx]
                                    filtered_meshes_sdf = mesh_extra_sdf[filtered_meshes_idx]

                                    sdf_winning_index = np.argmax(filtered_meshes_sdf)
                                    soma_mesh_retry = filtered_meshes[sdf_winning_index]
                                    sdf_retry = filtered_meshes_sdf[sdf_winning_index]

                                    curr_side_len_check_retry = side_length_check(soma_mesh_retry,side_length_ratio_threshold)
                                    curr_volume_check_retry = soma_volume_check(soma_mesh_retry,volume_mulitplier)

                                    if curr_side_len_check_retry and curr_volume_check_retry:
                                        to_add_list.append(soma_mesh_retry)
                                        to_add_list_sdf.append(sdf_retry)
                                    else:
                                        print(f"--->This soma mesh was not added because failed retry of sphere validation:\n "
                                             f"soma_mesh = {soma_mesh_retry}, curr_side_len_check = {curr_side_len_check_retry}, curr_volume_check = {curr_volume_check_retry}")
                                        continue
                                else:
                                    print(f"Could not find valid soma mesh in retry")
                                    continue


                        n_failed_inner_soma_loops = 0

                    else:
                        n_failed_inner_soma_loops += 1

                total_soma_list_sdf += to_add_list_sdf
                total_soma_list += to_add_list

                # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
                if n_failed_inner_soma_loops >= max_fail_loops:
                    print(f"breaking inner loop because {max_fail_loops} soma fails in a row")
                    break


            # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
            if somas_found_in_big_loop == False:
                no_somas_found_in_big_loop += 1
                if no_somas_found_in_big_loop >= max_fail_loops:
                    print(f"breaking because {max_fail_loops} fails in a row in big loop")
                    break

            else:
                no_somas_found_in_big_loop = 0





        """ IF THERE ARE MULTIPLE SOMAS THAT ARE WITHIN A CERTAIN DISTANCE OF EACH OTHER THEN JUST COMBINE THEM INTO ONE"""
        pairings = []

        if perform_pairing:
            for y,soma_1 in enumerate(total_soma_list):
                for z,soma_2 in enumerate(total_soma_list):
                    if y<z:
                        mesh_tree = KDTree(soma_1.vertices)
                        distances,closest_node = mesh_tree.query(soma_2.vertices)

                        if np.min(distances) < 4000:
                            pairings.append([y,z])


        #creating the combined meshes from the list
        total_soma_list_revised = []
        total_soma_list_revised_sdf = []
        if len(pairings) > 0:
            """
            Pseudocode: 
            Use a network function to find components

            """


            import networkx as nx
            new_graph = nx.Graph()
            new_graph.add_edges_from(pairings)
            grouped_somas = list(nx.connected_components(new_graph))

            somas_being_combined = []
            print(f"There were soma pairings: Connected components in = {grouped_somas} ")
            for comp in grouped_somas:
                comp = list(comp)
                somas_being_combined += list(comp)
                current_mesh = total_soma_list[comp[0]]
                for i in range(1,len(comp)):
                    current_mesh += total_soma_list[comp[i]] #just combining the actual meshes

                total_soma_list_revised.append(current_mesh)
                #where can average all of the sdf values
                total_soma_list_revised_sdf.append(np.min(np.array(total_soma_list_sdf)[comp]))

            #add those that weren't combined to total_soma_list_revised
            leftover_somas = [total_soma_list[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
            leftover_somas_sdfs = [total_soma_list_sdf[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
            if len(leftover_somas) > 0:
                total_soma_list_revised += leftover_somas
                total_soma_list_revised_sdf += leftover_somas_sdfs

            print(f"Final total_soma_list_revised = {total_soma_list_revised}")
            print(f"Final total_soma_list_revised_sdf = {total_soma_list_revised_sdf}")


        if len(total_soma_list_revised) == 0:
            total_soma_list_revised = total_soma_list
            total_soma_list_revised_sdf = total_soma_list_sdf

        run_time = time.time() - global_start_time

        print(f"\n\n\n Total time for run = {time.time() - global_start_time}")
        print(f"Before Filtering the number of somas found = {len(total_soma_list_revised)}")

        if save_mesh_intermediates:
            su.compressed_pickle(total_soma_list_revised,"total_soma_list_revised_1")

        #     from datasci_tools import system_utils as su
        #     su.compressed_pickle(total_soma_list_revised,"total_soma_list_revised")
        #     su.compressed_pickle(new_mesh,"original_mesh")

        #need to erase all of the temporary files ******
        #import shutil
        #shutil.rmtree(directory)

        """
        Running the extra tests that depend on
        - border vertices
        - how well the poisson matches the backtracked soma to the real mesh
        - other size checks

        """
        filtered_soma_list = []
        filtered_soma_list_sdf = []

        for yy,(soma_mesh,curr_soma_sdf) in enumerate(zip(total_soma_list_revised,total_soma_list_revised_sdf)):
            add_inside_pieces_flag = False
            if backtrack_soma_mesh_to_original:
                if verbose:
                    print(f"\n---Performing Soma Mesh Backtracking to original mesh for poisson soma {yy}")
                soma_mesh_poisson = deepcopy(soma_mesh)
                    #print("About to find original mesh")

                try:
                    if verbose:
                        print(f"backtrack_soma_size_threshold = {backtrack_soma_size_threshold}")
                    soma_mesh_list,soma_mesh_inside_pieces = sm.original_mesh_soma(
                                                    original_mesh = recov_orig_mesh_no_interior,
                                                    mesh=soma_mesh_poisson,
                                                    soma_size_threshold=backtrack_soma_size_threshold,
                                                    match_distance_threshold=backtrack_match_distance_threshold,
                                                    verbose = verbose)

                except:
                    import traceback
                    traceback.print_exc()
                    print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                    continue

                if soma_mesh_list is None:
                    print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                    continue



                if verbose: 
                    print(f"After backtrack the found {len(soma_mesh_list)} possible somas: {soma_mesh_list} ")

                for rr,soma_mesh in enumerate(soma_mesh_list):
                    if verbose:
                        print(f"\n--- working on backtrack soma {rr}: {soma_mesh}")

                    print(f"poisson_backtrack_distance_threshold = {poisson_backtrack_distance_threshold}")
                    #do the check that tests if there is a max distance between poisson and backtrack:
                    if not poisson_backtrack_distance_threshold is None and poisson_backtrack_distance_threshold > 0:

                        #soma_mesh.export("soma_mesh.off")
                        if close_holes: 
                            print("Using the close holes feature")
                            fill_hole_obj = meshlab.FillHoles(max_hole_size=2000,
                                                             self_itersect_faces=False)

                            soma_mesh_filled_holes,output_subprocess_obj = fill_hole_obj(   
                                                                vertices=soma_mesh.vertices,
                                                                 faces=soma_mesh.faces,
                                                                 return_mesh=True,
                                                                 delete_temp_files=True,
                                                                )
                        else:
                            soma_mesh_filled_holes = soma_mesh


                        #soma_mesh_filled_holes.export("soma_mesh_filled_holes.off")



                        print("APPLYING poisson_backtrack_distance_threshold CHECKS")
                        mesh_1 = soma_mesh_filled_holes
                        mesh_2 = soma_mesh_poisson

                        poisson_max_distance = tu.max_distance_betwee_mesh_vertices(mesh_1,mesh_2,
                                                                          verbose=True)
                        print(f"poisson_max_distance = {poisson_max_distance}")
                        if poisson_max_distance > poisson_backtrack_distance_threshold:
                            print(f"--->This soma mesh was not added because it did not pass the poisson_backtrack_distance check:\n"
                              f" poisson_max_distance = {poisson_max_distance}")
                            continue


                    #do the boundary check:
                    if not boundary_vertices_threshold is None:
                        print("USING boundary_vertices_threshold CHECK")
                        soma_boundary_groups_sizes = np.array([len(k) for k in tu.find_border_face_groups(soma_mesh)])
                        print(f"soma_boundary_groups_sizes = {soma_boundary_groups_sizes}")
                        large_boundary_groups = soma_boundary_groups_sizes[soma_boundary_groups_sizes>boundary_vertices_threshold]
                        print(f"large_boundary_groups = {large_boundary_groups} with boundary_vertices_threshold = {boundary_vertices_threshold}")
                        if len(large_boundary_groups)>0:
                            print(f"--->This soma mesh was not added because it did not pass the boundary vertices validation:\n"
                                  f" large_boundary_groups = {large_boundary_groups}")
                            continue

                    curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
                    curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)


                    # -------- 1/12 Addition: Does a second round of segmentation after to see if can split somas at all ---- #
                    if (not curr_side_len_check) or (not curr_volume_check):

                        if backtrack_segmentation_on_fail:
                            print("Trying backtrack segmentation")
                            mesh_tests,mesh_tests_sdf = tu.mesh_segmentation(soma_mesh,clusters=3,smoothness=0.2)

                            soma_mesh_filtered = []
                            soma_mesh_sdf_filtered = []

                            for m_test,m_test_sdf in zip(mesh_tests,mesh_tests_sdf):

                                if len(m_test.faces) >= backtrack_soma_size_threshold and m_test_sdf >=soma_width_threshold:

                                    if side_length_check(m_test,side_length_ratio_threshold) and soma_volume_check(m_test,volume_mulitplier):

                                        soma_mesh_filtered.append(m_test)
                                        soma_mesh_sdf_filtered.append(m_test_sdf)

                            if len(soma_mesh_filtered) == 0:
                                print(f"--->This soma mesh was not added because it did not pass the sphere validation EVEN AFTER SEGMENTATION:\n "
                                 f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                                continue


                        else:
                            print(f"--->This soma mesh was not added because it did not pass the sphere validation:\n "
                             f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                            continue
                    else:


                        soma_mesh_filtered = [soma_mesh]
                        soma_mesh_sdf_filtered = [curr_soma_sdf]



                    #tu.write_neuron_off(soma_mesh_poisson,"original_poisson.off")
                    #If made it through all the checks then add to final list
                    filtered_soma_list += soma_mesh_filtered
                    filtered_soma_list_sdf += soma_mesh_sdf_filtered
                    add_inside_pieces_flag = True #setting flag so will add inside pieces


            if len(soma_mesh_inside_pieces) > 0 and add_inside_pieces_flag:
                print(f"About to add the following inside nuclei pieces after soma backtrack: {nuclei_pieces}")
                nuclei_pieces +=soma_mesh_inside_pieces


        """
        Need to delete all files in the temp folder *****
        """

        # ----------- 11 /11 Addition that does a last step segmentation of the soma --------- #
        #return total_soma_list, run_time
        #return total_soma_list_revised,run_time,total_soma_list_revised_sdf

        """
        Things we should ask about the segmentation:

        Advantages: 
        1) could help filter away negatives

        Disadvantages:
        1) Can actually cut up the soma and then filter away the soma (not what we want)
        2) Could introduce a big hole (don't think can guard against this)
        """


        #filtered_soma_list_saved = copy.deepcopy(filtered_soma_list)

        if len(filtered_soma_list) > 0:
            filtered_soma_list_revised = []
            filtered_soma_list_sdf_revised = []
            for f_soma,f_soma_sdf in zip(filtered_soma_list,filtered_soma_list_sdf):

                print("Skipping the segmentatio filter at end")
                if not (len(f_soma.faces) >= last_size_threshold and f_soma_sdf >= soma_width_threshold):
                    print(f"Soma (size = {len(f_soma.faces)}, width={soma_width_threshold}) did not pass thresholds (size threshold={last_size_threshold}, width threshold = {soma_width_threshold}) ")
                    continue


                if segmentation_at_end:


                    if remove_inside_pieces:
                        print("removing mesh interior before segmentation")
                        f_soma = tu.remove_mesh_interior(f_soma,size_threshold_to_remove=size_threshold_to_remove)

                    print("Doing the soma segmentation filter at end")

                    meshes_split,meshes_split_sdf = tu.mesh_segmentation(
                        mesh = f_soma,
                        smoothness=0.5
                    )
        #                 print(f"meshes_split = {meshes_split}")
        #                 print(f"meshes_split_sdf = {meshes_split_sdf}")

                    #applying the soma width and the soma size threshold
                    above_width_threshold_mask = meshes_split_sdf>=soma_width_threshold
                    meshes_split_sizes = np.array([len(k.faces) for k in meshes_split])
                    above_size_threshold_mask = meshes_split_sizes >= last_size_threshold

                    above_width_threshold_idx = np.where(above_width_threshold_mask & above_size_threshold_mask)[0]
                    if len(above_width_threshold_idx) == 0:
                        print(f"No split meshes were above the width threshold ({soma_width_threshold}) and size threshold ({last_size_threshold}) so continuing")
                        print(f"So just going with old somas")

                        f_soma_final = f_soma
                        f_soma_sdf_final = f_soma_sdf


                    else:
                        meshes_split = np.array(meshes_split)
                        meshes_split_sdf = np.array(meshes_split_sdf)

                        meshes_split_filtered = meshes_split[above_width_threshold_idx]
                        meshes_split_sdf_filtered = meshes_split_sdf[above_width_threshold_idx]

                        soma_width_threshold
                        #way to choose the index of the top candidate
                        top_candidate = 0


                        largest_hole_before_seg = tu.largest_hole_length(f_soma)
                        largest_hole_after_seg = tu.largest_hole_length(meshes_split_filtered[top_candidate])

                        print(f"Largest hole before segmentation = {largest_hole_before_seg}, after = {largest_hole_after_seg},")
                        if largest_hole_before_seg > 0:
                              print(f"\nratio = {largest_hole_after_seg/largest_hole_before_seg}, difference = {largest_hole_after_seg - largest_hole_before_seg}")

                        if largest_hole_after_seg < largest_hole_threshold:
                            f_soma_final = meshes_split_filtered[top_candidate]
                            f_soma_sdf_final = meshes_split_sdf_filtered[top_candidate]
                        else:
                            f_soma_final = f_soma
                            f_soma_sdf_final = f_soma_sdf

                else:
                    f_soma_final = f_soma
                    f_soma_sdf_final = f_soma_sdf


                filtered_soma_list_revised.append(f_soma_final)
                filtered_soma_list_sdf_revised.append(f_soma_sdf_final)




            filtered_soma_list = np.array(filtered_soma_list_revised)
            filtered_soma_list_sdf = np.array(filtered_soma_list_sdf_revised)

            """
            # ----------- 1/7/21 ---------------#
            Now was to stitch the somas together if they are touching

            """
            if len(filtered_soma_list)>1:
                connected_meshes_components = tu.mesh_list_connectivity(meshes=filtered_soma_list,
                                         main_mesh=recov_orig_mesh_no_interior,
                                                            return_connected_components=True)

                filtered_soma_list_components = np.array([tu.combine_meshes(filtered_soma_list[k]) for k in connected_meshes_components])
                filtered_soma_list_sdf_components = np.array([np.mean(filtered_soma_list_sdf[k]) for k in connected_meshes_components])
            elif len(filtered_soma_list)==1:
                filtered_soma_list_components = filtered_soma_list
                filtered_soma_list_sdf_components = filtered_soma_list_sdf
            else:
                filtered_soma_list_components = []
                filtered_soma_list_sdf_components = np.array([])
        else:
            filtered_soma_list_components = []
            filtered_soma_list_sdf_components = np.array([])



        #----------- 1/9 Addition: Final Size Threshold ------------- #
        if verbose:
            print(f"filtered_soma_list_components = {filtered_soma_list_components}")
        if save_mesh_intermediates:
            su.compressed_pickle(filtered_soma_list_components,"filtered_soma_list_components")
        if backtrack_soma_mesh_to_original:

            filtered_soma_list_components_new = []
            filtered_soma_list_sdf_components_new = []

            for soma_mesh, soma_mesh_sdf in zip(filtered_soma_list_components,filtered_soma_list_sdf_components):
                # --------- 1/9: Extra Size Threshold For Somas ------------- #
                if len(soma_mesh.faces) < backtrack_soma_size_threshold:
                    print(f"--->This soma mesh with size {len(soma_mesh.faces)} was not bigger than the threshold {backtrack_soma_size_threshold}")
                    continue
                else:
                    filtered_soma_list_components_new.append(soma_mesh)
                    filtered_soma_list_sdf_components_new.append(soma_mesh_sdf)

            filtered_soma_list_components = filtered_soma_list_components_new
            filtered_soma_list_sdf_components = np.array(filtered_soma_list_sdf_components_new)

        if filter_inside_somas:
            if len(filtered_soma_list_components)>1:
                keep_indices = tu.filter_away_inside_meshes(mesh_list = filtered_soma_list_components,
                                            distance_type="shortest_vertex_distance",
                                            distance_threshold = 2000,
                                            inside_percentage_threshold = 0.20,
                                            verbose = False,
                                            return_meshes = False,
                                            )
            else:
                keep_indices = np.arange(len(filtered_soma_list_components))

            filtered_soma_list_components = [k for i,k in enumerate(filtered_soma_list_components) if i in keep_indices]
            filtered_soma_list_sdf_components = filtered_soma_list_sdf_components[keep_indices]

    

    
    if return_glia_nuclei_pieces:
        return list(filtered_soma_list_components),run_time,filtered_soma_list_sdf_components,glia_pieces, nuclei_pieces
    else:
        return list(filtered_soma_list_components),run_time,filtered_soma_list_sdf_components
    
    
def remove_nuclei_and_glia_meshes(
    mesh,
    glia_volume_threshold_in_um = None, #the volume of a large soma mesh is 800 billion
    glia_n_faces_threshold = 400000,
    glia_n_faces_min = 100000,
    nucleus_min = None,
    nucleus_max = None,
    connectivity="edges",
    try_hole_close=False,
    verbose=False,
    return_glia_nucleus_pieces=True,
    **kwargs
    ):
    
    """
    Will remove interior faces of a mesh with a certain significant size
    
    Pseudocode: 
    1) Run the mesh interior
    2) Divide all of the interior meshes into glia (those above the threshold) and nuclei (those below)
    For Glia:
    3) Do all of the removal process and get resulting neuron
    
    For nuclei:
    4) Do removal process from mesh that already has glia removed (use subtraction with exact_match = False )
    
    
    
    
    """
    # ------- Nuclei Parameters ---------
    if nucleus_min is None:
        nucleus_min = nucleus_min_global
    
    if nucleus_max is None:
        nucleus_max = nucleus_max_global
        
    # --------- Glia Parameters ----------
    if glia_volume_threshold_in_um is None:
        glia_volume_threshold_in_um= glia_volume_threshold_in_um_global
    
    
    glia_volume_threshold = glia_volume_threshold_in_um*1000000000
    
    st = time.time()
    original_mesh_size = len(mesh.faces)
    
    #1) Run the mesh interior
    curr_interior_mesh_non_split = tu.mesh_interior(mesh,return_interior=True,
                                       try_hole_close=False,
                                       **kwargs)
    
    curr_interior_mesh = tu.split_significant_pieces(curr_interior_mesh_non_split,significance_threshold=nucleus_min,
                                            connectivity=connectivity)
    
    #su.compressed_pickle(curr_interior_mesh,"curr_interior_mesh")
    
    curr_interior_mesh = np.array(curr_interior_mesh)
    curr_interior_mesh_len = np.array([len(k.faces) for k in curr_interior_mesh])
    
    
    
    if verbose:
        print(f"There were {len(curr_interior_mesh)} total interior meshes")
    
    #2) Divide all of the interior meshes into glia (those above the threshold) and nuclei (those below)
    if glia_volume_threshold is None:
        if nucleus_max is None:
            nucleus_max = glia_n_faces_threshold
        
        

        nucleus_pieces = curr_interior_mesh[(curr_interior_mesh_len >= nucleus_min) & (curr_interior_mesh_len< nucleus_max)]
        glia_pieces = curr_interior_mesh[ (curr_interior_mesh_len >= glia_n_faces_threshold)]
        
        if verbose:
            print(f"Pieces satisfying glia requirements (n_faces) (x >= {glia_n_faces_threshold}): {len(glia_pieces)}")
            print(f"Pieces satisfying nuclie requirements (n_faces) ({nucleus_min} <= x < {nucleus_max}): {len(nucleus_pieces)}")
    else:
        """
        Psuedocode:
        """
        curr_interior_mesh_volume = np.array([k.convex_hull.volume for k in curr_interior_mesh])
        
        glia_pieces = curr_interior_mesh[ (curr_interior_mesh_volume >= glia_volume_threshold) &
                                           (curr_interior_mesh_len >= glia_n_faces_min)]
        nucleus_pieces = curr_interior_mesh[(curr_interior_mesh_len >= nucleus_min) & 
                                            (curr_interior_mesh_volume < glia_volume_threshold)]
        
        if verbose:
            print(f"Pieces satisfying glia requirements (volume) (x >= {glia_volume_threshold}): {len(glia_pieces)}")
            print(f"Pieces satisfying nuclie requirements: n_faces ({nucleus_min} <= x)"
                  f" and volume (x < {glia_volume_threshold}) : {len(nucleus_pieces)}")
        
        
    #3) Do the Glia removal process
    if len(glia_pieces) > 0:
        mesh_removed_glia,glia_submesh = tu.find_large_dense_submesh(mesh,
                                                                        glia_pieces=glia_pieces,
                                                                        connectivity=connectivity,
                                                                            verbose=verbose,
                                                                            **kwargs)
        if glia_submesh is None:
            glia_submesh = []
        else:
            glia_submesh = [glia_submesh]
    else:
        mesh_removed_glia = mesh
        glia_submesh = []
        
    
    #4) Do the Nucleus removal process
    
    if len(nucleus_pieces) > 0:
        main_mesh_total,inside_nucleus_pieces = tu.remove_mesh_interior(mesh_removed_glia,
                                                                        inside_pieces=nucleus_pieces,
                                                                        return_removed_pieces=True,
                                                                        connectivity=connectivity,
                                                                        size_threshold_to_remove = nucleus_min,
                                                                   try_hole_close=False,
                                                                        verbose=verbose,
                                                                       **kwargs)
    else:
        main_mesh_total = mesh_removed_glia
        inside_nucleus_pieces = []
    
    
    if verbose:
        print(f"\n\nOriginal Mesh size: {original_mesh_size}, Final mesh size: {len(main_mesh_total.faces)}")
        print(f"Total time = {time.time() - st}")
    
    if return_glia_nucleus_pieces:
        return main_mesh_total,glia_submesh,inside_nucleus_pieces
    else:
        return main_mesh_total
    
    
def glia_nuclei_faces_from_mesh(
    mesh,
    glia_meshes,
    nuclei_meshes,
    return_n_faces = False,
    verbose = False,
    ):
    """
    Purpose: To map the glia and nuclei
    meshes to the 
    """

    if len(glia_meshes)>0:
        glia_faces = tu.original_mesh_faces_map(mesh,tu.combine_meshes(glia_meshes))
        n_glia_faces = len(glia_faces)
    else:
        glia_faces = []
        n_glia_faces = 0

    if len(nuclei_meshes)>0:
        nuclei_faces = tu.original_mesh_faces_map(mesh,tu.combine_meshes(nuclei_meshes))
        n_nuclei_faces = len(nuclei_faces)
    else:
        nuclei_faces = []
        n_nuclei_faces = 0

    if verbose:
        print(f"n_glia_faces = {n_glia_faces}, n_nuclei_faces = {n_nuclei_faces}")

    if return_n_faces:
        return glia_faces,nuclei_faces,n_glia_faces,n_nuclei_faces
    else:
        return glia_faces,nuclei_faces
    
def plot_soma_products(
    mesh,
    soma_products,
    verbose = True):
    
    nviz.plot_soma_extraction_meshes(
            mesh = mesh,
            soma_meshes = soma_products.soma_meshes,
            glia_meshes = soma_products.glia_meshes,
            nuclei_meshes = soma_products.nuclei_meshes,
            verbose = verbose,
    )

def soma_indentification(
    mesh_decimated,
    verbose=False,
    plot = False,
    **soma_extraction_parameters
    ):

    (total_soma_list, 
     run_time, 
     total_soma_list_sdf,
     glia_pieces,
     nuclei_pieces) = sm.extract_soma_center(
        mesh = mesh_decimated,
        return_glia_nuclei_pieces=True,
        verbose = verbose,
        **soma_extraction_parameters
    )

    soma_products = pipeline.StageProducts(
        soma_extraction_parameters = soma_extraction_parameters,
        soma_meshes=total_soma_list,
        soma_run_time=run_time, 
        soma_sdfs=total_soma_list_sdf,
        glia_meshes=glia_pieces,
        nuclei_meshes=nuclei_pieces,
    )
    
    if plot:
        plot_soma_products(
            mesh_decimated,
            soma_products = soma_products,
        )


    return soma_products
    
    
# ------------------------- Parameters ---------------------
# ------------- Setting up parameters -----------

# -- default
attributes_dict_default = dict(
)    



global_parameters_dict_default_nuclei = dsu.DictType(
    nucleus_min = 700, #minimum number of faces for nuclei mesh peices
    nucleus_max = (None,"int unsigned"), #maximum number of faces for nuclie mesh pieces if glia volume is not defined
)

global_parameters_dict_default_glia = dsu.DictType(
    glia_volume_threshold_in_um = 2500, #minimum volume of glia mesh
    glia_n_faces_threshold = 400_000, #minimum number of faes for glia mesh if volume is not defined
    glia_n_faces_min = 100_000, #minimum number of faes for glia mesh if glia volume defined
)

global_parameters_dict_default_soma  = dsu.DictType(
    outer_decimation_ratio=0.25 , #decimation ratio for 1st round of decimation
    large_mesh_threshold = 20_000, #minimum face count threshold after 1st round of decimation
    large_mesh_threshold_inner = 13_000, #minimum face count threshold after poisson reconstruction and split
    inner_decimation_ratio = 0.25, #decimation ratio for 2nd round of decimation after poisson reconstruction
    max_fail_loops = 10, #number of times soma finding can fail in finding an invalid soma in the outer/inner decimation loop

    # other cleaning methods to run on segments after first decimation
    remove_inside_pieces = True, #whether to remove inside pieces before processing
    size_threshold_to_remove=1_000, #minium number of faces of mesh piece to remove
    pymeshfix_clean=False,
    check_holes_before_pymeshfix=False,
    second_poisson=False,

    #after 2nd round of decimation and mesh segmentation is applied
    soma_width_threshold = 0.32, # minimuum sdf values of mesh segments
    soma_size_threshold = 9_000, # minimum face count threshold of mesh segments
    soma_size_threshold_max=1_200_000,# maximum face count threshold of mesh segments

    #parameters for checking possible viable somas after mesh segmentation
    volume_mulitplier=8, #for soma_volume_check funcion that makes sure not highly skewed bounding box volume to mesh volume 
    side_length_ratio_threshold=6, #for side_length_check function making sure not highly swkewed x,y,z side length ratios

    #whether to group certain poisson representations of somas together before backtrack to mesh
    perform_pairing = False,


    backtrack_soma_mesh_to_original=True, #will backtrack the poisson reconstruction soma to the original mesh
    backtrack_soma_size_threshold=8000, #minimum number of faces for a backtracked mesh


    #for filtering away somas once get backtracked to original mesh with certain filters
    poisson_backtrack_distance_threshold=(None,"int unsigned"), #maximum distance between poisson reconstruction and backtracked soma
    close_holes=False, #whether to close holes when doing poisson backtrack

    boundary_vertices_threshold=(None,"int unsigned"), #filter away somas if too many boundary vertices


    #filters at the very end for meshes that made it thorugh
    last_size_threshold = 2000, #min faces count

    segmentation_at_end=True, #whether to attempt to split meshes at end
    largest_hole_threshold = 17000, #maximum hole length for a soma to allow the segmentaiton split at end

    second_pass_size_threshold = (None,"int unsigned"),
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_nuclei,
    global_parameters_dict_default_glia,
    global_parameters_dict_default_soma,
])

# -- microns--
global_parameters_dict_microns = {}
attributes_dict_microns = {}

#-- h01--
global_parameters_dict_h01_glia = dict(
    glia_n_faces_threshold=3000000,
    glia_n_faces_min=3000000,
)

global_parameters_dict_h01_nuclei= dict(
)

global_parameters_dict_h01_soma = dict(
    
    large_mesh_threshold=40000,
    large_mesh_threshold_inner=20000,
    soma_size_threshold=15000,
    soma_size_threshold_max=2000000,
    backtrack_soma_size_threshold=15000,
    last_size_threshold=15000,
    second_pass_size_threshold=5000,
)

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_glia,
    global_parameters_dict_h01_nuclei,
    global_parameters_dict_h01_soma
])

attributes_dict_h01 = dict()

# data_type = "default"
# algorithms = None
# modules_to_set = [sm]

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


# def output_global_parameters_nuclei(**kwargs):
#     return modu.output_global_parameters_and_attributes_from_current_data_type(
#     algorithms = ["nuclei"],
#     include_default = True,
#     algorithms_only = True,
#         **kwargs
#     )

# def output_global_parameters_soma(**kwargs):
#     return modu.output_global_parameters_and_attributes_from_current_data_type(
#     algorithms = ["soma"],
#     include_default = True,
#     algorithms_only = True,
#         **kwargs
#     )



#--- from mesh_tools ---
from mesh_tools import meshlab
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import data_struct_utils as dsu
from datasci_tools import general_utils as gu
from datasci_tools import module_utils as modu 
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools import pipeline

from . import soma_extraction_utils as sm
from . import neuron_visualizations as nviz
from . import parameter_utils as paru
def output_global_parameters_glia(**kwargs):
    return paru.category_param_from_module(
        module = sm,
        category = "glia",
    )
    
    
    # return modu.output_global_parameters_and_attributes_from_current_data_type(
    #     [sm],
    #     algorithms = ["glia"],
    #     include_default = True,
    #     algorithms_only = True,
    #         **kwargs
    # )
    
def output_global_parameters_nuclei(**kwargs):
    return paru.category_param_from_module(
        module = sm,
        category = "nuclei",
    )
    
    
    # return modu.output_global_parameters_and_attributes_from_current_data_type(
    #     [sm],
    #     algorithms = ["nuclei"],
    #     include_default = True,
    #     algorithms_only = True,
    #         **kwargs
    # )
    
def output_global_parameters_soma(**kwargs):
    return paru.category_param_from_module(
        module = sm,
        category = "soma",
    )
    
    
    # return modu.output_global_parameters_and_attributes_from_current_data_type(
    #     [sm],
    #     algorithms = ["soma"],
    #     include_default = True,
    #     algorithms_only = True,
    #         **kwargs
    # )
