
from mesh_tools import skeleton_utils as sk
import soma_extraction_utils as sm
from mesh_tools import trimesh_utils as tu
import trimesh
from python_tools import numpy_utils as nu
import numpy as np
from importlib import reload
import networkx as nx
import time
from mesh_tools import compartment_utils as cu
from python_tools import networkx_utils as xu
from python_tools import matplotlib_utils as mu

#importing at the bottom so don't get any conflicts
import itertools
from python_tools.tqdm_utils import tqdm

#for meshparty preprocessing
from mesh_tools import meshparty_skeletonize as m_sk
from python_tools import general_utils as gu
from mesh_tools import compartment_utils as cu
from meshparty import trimesh_io
from copy import deepcopy

#from neuron_utils import *




def further_mesh_correspondence_processing_from_skeleton(soma_touching_mesh_data,        
                                                        combine_close_skeleton_nodes = True,
                                                        combine_close_skeleton_nodes_threshold=700,
                                                         distance_by_mesh_center=True,
                                                        branch_skeleton_data=None,
                                                        soma_to_piece_touching_vertices=None,
                                                        endpoints_must_keep=None):
    
    current_mesh_data = soma_touching_mesh_data



    # *************** Phase B *****************

    print("\n\n\n\n\n****** Phase B ***************\n\n\n\n\n")




    # visualizing the original neuron
#     current_neuron = trimesh.load_mesh(current_mesh_file)
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                               main_mesh_faces=current_neuron.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )


    # visualizing the somas that were extracted
#     soma_meshes = tu.combine_meshes(current_mesh_data[0]["soma_meshes"])
#     sk.graph_skeleton_and_mesh(main_mesh_verts=soma_meshes.vertices,
#                               main_mesh_faces=soma_meshes.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )


    # # Visualize the extracted branches
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons"],
    #                           other_skeletons_colors="random")








    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing

    
    if branch_skeleton_data is None:
        
        total_cleaned = []
        for j,curr_skeleton_to_clean in enumerate(current_mesh_data[0]["branch_skeletons"]):
            
            print(f"\n---- Working on Limb {j} ----")
            start_time = time.time()
            print(f"before cleaning limb size of skeleton = {curr_skeleton_to_clean.shape}")
            
#             """ 9/16 Edit: Now send the border vertices and don't want to clean anyy end nodes that are within certain distance of border"""
#             *** WE DIDN'T END UP NEEDING TO DO THE SKELETON CLEANING AGAIN BECAUSE ALREADY RAN IT
#             if not soma_to_piece_touching_vertices is None:
#                 total_border_vertices = []
#                 for k in soma_to_piece_touching_vertices.keys():
#                     if j in soma_to_piece_touching_vertices[k].keys():
#                         total_border_vertices.append(soma_to_piece_touching_vertices[k][j])

#                 if len(total_border_vertices) > 0:
#                     total_border_vertices = np.concatenate(total_border_vertices)
#             else:
#                 total_border_vertices=None

            
#             skelton_cleaning_threshold = 4001
#             distance_cleaned_skeleton = sk.clean_skeleton(
#                                                         curr_skeleton_to_clean,
#                                                         distance_func=sk.skeletal_distance,
#                                                         min_distance_to_junction = skelton_cleaning_threshold,
#                                                         soma_border_vertices = total_border_vertices,
#                                                         skeleton_mesh=current_mesh_data[0]["branch_meshes"][j],
#                                                         return_skeleton=True,
                
#                                                         print_flag=False)
            
            distance_cleaned_skeleton = curr_skeleton_to_clean
            
            #make sure still connected componet
            distance_cleaned_skeleton_components = nx.number_connected_components(sk.convert_skeleton_to_graph(distance_cleaned_skeleton))
            if distance_cleaned_skeleton_components > 1:
                raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

            print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")
            cleaned_branch = sk.clean_skeleton_with_decompose(distance_cleaned_skeleton)

            cleaned_branch_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cleaned_branch))
            if cleaned_branch_components > 1:
                raise Exception(f"cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")

            #do the cleanin ghtat removes loops from branches
            print(f"After DECOMPOSITION cleaning limb size of skeleton = {cleaned_branch.shape}")
            print(f"Total time = {time.time() - start_time}")
            total_cleaned.append(cleaned_branch)

        current_mesh_data[0]["branch_skeletons_cleaned"] = total_cleaned
    else:
        print("****Skipping skeleton cleaning and USING THE PRE-COMPUTED SKELETONS ****")
        current_mesh_data[0]["branch_skeletons_cleaned"] =branch_skeleton_data

    
#     sk_debug = True
#     if sk_debug:
#         from python_tools import system_utils as su
#         su.compressed_pickle(deepcopy(current_mesh_data[0]["branch_skeletons_cleaned"]),
#                             "second_branch_skeletons_cleaned")
    
    # checking all cleaned branches are connected components

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Original limb {k} was not a single component: it was actually {n_components} components")

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons_cleaned"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Cleaned limb {k} was not a single component: it was actually {n_components} components")


    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons_cleaned"],
    #                           other_skeletons_colors="random",
    #                           mesh_alpha=0.15,
    #                           html_path=f"{segment_id}_limb_skeleton.html")


    if combine_close_skeleton_nodes:
        print(f"********COMBINING CLOSE SKELETON NODES WITHIN {combine_close_skeleton_nodes_threshold} DISTANCE**********")
        current_mesh_data[0]["branch_skeletons_cleaned"] = [sk.combine_close_branch_points(curr_limb_sk,
                                                            combine_threshold = combine_close_skeleton_nodes_threshold,
                                                            print_flag=True) for curr_limb_sk in current_mesh_data[0]["branch_skeletons_cleaned"]]

    
#     if sk_debug:
#         from python_tools import system_utils as su
#         su.compressed_pickle(deepcopy(current_mesh_data[0]["branch_skeletons_cleaned"]),
#                             "third_combining_skeleton_nodes")
    
    save_clean_skeleton = False
    if save_clean_skeleton:
        from python_tools import system_utils as su
        su.compressed_pickle(current_mesh_data[0]["branch_skeletons_cleaned"],"branch_skeletons_cleaned")





    # --- 2) Decomposing of limbs into branches and finding mesh correspondence (using adaptive mesh correspondence followed by a water fill for conflict and empty faces), checking that it went well with no empty meshes and all connected component graph (even when downsampling the skeleton) when constructed from branches, plus visualization at end



    start_time = time.time()

    limb_correspondence = dict()
    soma_containing_idx= 0

    

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"Working on limb #{limb_idx}")
            limb_correspondence[limb_idx] = dict()
            curr_limb_sk = current_mesh_data[soma_containing_idx]["branch_skeletons_cleaned"][limb_idx]
            curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches
            
            """ 9/17 Edit: Want to Not do mesh adaptive correspondence on the """

            for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
                limb_correspondence[limb_idx][j] = dict()

                try:
                    returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                                  curr_limb_mesh,
                                                 skeleton_segment_width = 1000,
                                                 distance_by_mesh_center=distance_by_mesh_center)
                    curr_branch_face_correspondence, width_from_skeleton = returned_data
                except:
                    print(f"curr_branch_sk.shape = {curr_branch_sk.shape}")
                    np.savez("saved_skeleton_branch.npz",curr_branch_sk=curr_branch_sk)
                    tu.write_neuron_off(curr_limb_mesh,"curr_limb_mesh.off")
                    print(f"returned_data = {returned_data}")
                    raise Exception(f"The output from mesh_correspondence_adaptive_distance was nothing: curr_branch_face_correspondence={curr_branch_face_correspondence}, width_from_skeleton={width_from_skeleton}")


                if len(curr_branch_face_correspondence) > 0:
                    curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True,repair=False)
                else:
                    curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))

                limb_correspondence[limb_idx][j]["branch_skeleton"] = curr_branch_sk
                limb_correspondence[limb_idx][j]["correspondence_mesh"] = curr_submesh
                limb_correspondence[limb_idx][j]["correspondence_face_idx"] = curr_branch_face_correspondence
                limb_correspondence[limb_idx][j]["width_from_skeleton"] = width_from_skeleton


    print(f"Total time for decomposition = {time.time() - start_time}")


    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes

    empty_submeshes = []

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"\n---- checking limb {limb_idx}---")
            print(f"Limb {limb_idx} decomposed into {len(limb_correspondence[limb_idx])} branches")

            #get all of the skeletons and make sure that they from a connected component
            divided_branches = [limb_correspondence[limb_idx][k]["branch_skeleton"] for k in limb_correspondence[limb_idx]]
            divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                            sk.stack_skeletons(divided_branches))

            divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

            cleaned_limb_skeleton = current_mesh_data[0]['branch_skeletons_cleaned'][limb_idx]
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


            for j in limb_correspondence[limb_idx].keys():
                if len(limb_correspondence[limb_idx][j]["correspondence_mesh"].faces) == 0:
                    empty_submeshes.append(dict(limb_idx=limb_idx,branch_idx = j))

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")



    # from python_tools import matplotlib_utils as mu

    # sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
    #                           other_meshes_colors="random",
    #                            other_skeletons=total_branch_skeletons,
    #                            other_skeletons_colors="random"
    #                           )















    # ---3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece

    #--- this is the function that will clean up a limb piece so have 1-1 correspondence

    #things to prep for visualizing the axons
#     total_widths = []
#     total_branch_skeletons = []
#     total_branch_meshes = []



    for limb_idx in limb_correspondence.keys():
        mesh_start_time = time.time()
        #clear out the mesh correspondence if already in limb_correspondecne
        for k in limb_correspondence[limb_idx].keys():
            if "branch_mesh" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_mesh"]
            if "branch_face_idx" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_face_idx"]
        #geting the current limb mesh
        print(f"\n\nWorking on limb_correspondence for #{limb_idx}")
        no_missing_labels = list(limb_correspondence[limb_idx].keys()) #counts the number of divided branches which should be the total number of labels
        curr_limb_mesh = current_mesh_data[soma_containing_idx]["branch_meshes"][limb_idx]

        #set up the face dictionary
        face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

        for j,branch_piece in limb_correspondence[limb_idx].items():
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
                         no_missing_labels = list(original_labels)
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
        if (not endpoints_must_keep is None) and (not soma_to_piece_touching_vertices is None):
            for sm,limb_list in soma_to_piece_touching_vertices.items():
                if limb_idx not in limb_list:
                    continue
                #0) Get the soma border
                curr_soma_border = soma_to_piece_touching_vertices[sm][limb_idx]
                #1) Find the label_to_expand based on the starting coordinate
                st_coord = endpoints_must_keep[limb_idx][sm]
                divided_branches = [limb_correspondence[limb_idx][k]["branch_skeleton"] for k in limb_correspondence[limb_idx]]
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

        #save off the new data as branch mesh
        for k in limb_correspondence[limb_idx].keys():
            limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
            limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]

            #clean the limb correspondence that we do not need
            del limb_correspondence[limb_idx][k]["correspondence_mesh"]
            del limb_correspondence[limb_idx][k]["correspondence_face_idx"]
#             total_widths.append(limb_correspondence[limb_idx][k]["width_from_skeleton"])
#             total_branch_skeletons.append(limb_correspondence[limb_idx][k]["branch_skeleton"])
#             total_branch_meshes.append(limb_correspondence[limb_idx][k]["branch_mesh"])

        print(f"Total time for limb mesh processing = {time.time() - mesh_start_time}")
    
    return limb_correspondence



# ===================== For helping to split limbs at base ===================== #
from importlib import reload
import os
from pathlib import Path

import neuron_utils as nru
import neuron
import neuron_visualizations as nviz
import time
from python_tools import system_utils as su
from mesh_tools import skeleton_utils as sk
from pykdtree.kdtree import KDTree
from mesh_tools import trimesh_utils as tu
import numpy as np
from python_tools import networkx_utils as xu
from mesh_tools import compartment_utils as cu
import networkx as nx
from python_tools import numpy_utils as nu
import copy
from python_tools import general_utils as gu
from python_tools import system_utils as su

def check_if_branch_needs_splitting(curr_limb,soma_idx,curr_soma_mesh,
                                    significant_skeleton_threshold=30000,
                                   print_flag=False):
    """
    Purpose: Checks to see if a certian limb needs splitting
    """
    
    
    #1) get the starting node:
    curr_starting_branch_idx = curr_limb.get_starting_branch_by_soma(soma_idx)
    #a. Get the staring branch skeleton and find the closest skeleton point to the soma border
    curr_branch = curr_limb[curr_starting_branch_idx]
    curr_branch_sk = curr_branch.skeleton
    #b. find the closest skeleton point to the soma border
    curr_soma_border = curr_limb.get_concept_network_data_by_soma(soma_idx)["touching_soma_vertices"]
    unique_skeleton_nodes = np.unique(curr_branch_sk.reshape(-1,3),axis=0)
    
    curr_soma_border_kdtree = KDTree(curr_soma_border)
    distances,closest_node = curr_soma_border_kdtree.query(unique_skeleton_nodes)
    cut_coordinate = unique_skeleton_nodes[np.argmin(distances),:]
    
    #c. cut the limb skeleton at that point
    curr_limb_sk_graph = sk.convert_skeleton_to_graph(curr_limb.skeleton)

    node_to_cut = xu.get_nodes_with_attributes_dict(curr_limb_sk_graph,dict(coordinates=cut_coordinate))

    if len(node_to_cut) != 1:
        raise Exception("Node to cut was not of length 1")

    node_to_cut = node_to_cut[0]
    
    #c. Seperate the graph into 2 components, If there are 2 connected components after cut, are the connected components both significant
    curr_limb_sk_graph.remove_node(node_to_cut)

    seperated_components = list(nx.connected_components(curr_limb_sk_graph))

    if len(seperated_components) <= 1:
        return None
        #raise Exception(f"Continue to next limb because number of seperated_components = {len(seperated_components)}")

    #c1. Seperate the graph into subgraph based on components and output the skeletons from each
    seperated_skeletons = [sk.convert_graph_to_skeleton(curr_limb_sk_graph.subgraph(k)) for k in seperated_components]
    skeleton_lengths = np.array([sk.calculate_skeleton_distance(k) for k in seperated_skeletons])


    n_significant_skeletons = np.sum(skeleton_lengths>significant_skeleton_threshold)
    if print_flag:
        print(f"n_significant_skeletons={n_significant_skeletons}")
        print(f"skeleton_lengths = {skeleton_lengths}")

    if n_significant_skeletons < 2:
        return None
    else:
        return cut_coordinate
        #raise Exception(f"Continue to next limb because n_significant_skeletons = {n_significant_skeletons} with lengths {skeleton_lengths}")

        
def split_limb_on_soma(curr_limb,soma_idx,curr_soma_mesh,
                       current_neuron_mesh,
                       soma_meshes,
                       cut_coordinate=None,
                      print_flag=False):
    """
    Purpose: Will actually do the limb splitting
    
    """
    #1) get the starting node:
    curr_starting_branch_idx = curr_limb.get_starting_branch_by_soma(soma_idx)
    #a. Get the staring branch skeleton and find the closest skeleton point to the soma border
    curr_branch = curr_limb[curr_starting_branch_idx]
    curr_branch_sk = curr_branch.skeleton
    
    if cut_coordinate is None:
        if print_flag:
            print("Having to recalculate the cut coordinate")
        
        #b. find the closest skeleton point to the soma border
        curr_soma_border = curr_limb.get_concept_network_data_by_soma(soma_idx)["touching_soma_vertices"]
        unique_skeleton_nodes = np.unique(curr_branch_sk.reshape(-1,3),axis=0)

        curr_soma_border_kdtree = KDTree(curr_soma_border)
        distances,closest_node = curr_soma_border_kdtree.query(unique_skeleton_nodes)
        cut_coordinate = unique_skeleton_nodes[np.argmin(distances),:]
    
    
    
    # --------------- Part A: Finding the new split skeletons ------------------------- #
    # Finding the node to cut on the BRANCH skeleton
    curr_branch_sk_graph = sk.convert_skeleton_to_graph(curr_branch_sk)

    if print_flag:
        print(f"cut_coordinate={cut_coordinate}")
    node_to_cut = xu.get_nodes_with_attributes_dict(curr_branch_sk_graph,dict(coordinates=cut_coordinate))
    if len(node_to_cut) != 1:
        raise Exception("Node to cut was not of length 1")

    node_to_cut = node_to_cut[0]
    
    G = curr_branch_sk_graph
    endpoint_nodes = xu.get_nodes_of_degree_k(G,1)
    endpoint_nodes_coord = xu.get_node_attributes(curr_branch_sk_graph,node_list=endpoint_nodes)
    paths_to_endpt = [nx.dijkstra_path(G,node_to_cut,k) for k in endpoint_nodes]
    path_lengths = np.array([len(k) for k in paths_to_endpt])
    closest_endpoint = np.argmin(path_lengths)
    farthest_endpoint = 1 - closest_endpoint
    closest_endpoint_len = path_lengths[closest_endpoint]

    if closest_endpoint_len <= 1:
        #need to readjust the node_to_cut and paths
        print("Having to readjust endpoint")
        node_to_cut = paths_to_endpt[farthest_endpoint][1]
        paths_to_endpt = [nx.dijkstra_path(G,node_to_cut,k) for k in endpoint_nodes]

    #generate the subnode in each graph
    paths_to_endpt[farthest_endpoint].remove(node_to_cut)

    subgraph_list = [G.subgraph(k) for k in paths_to_endpt]

    starting_endpoints = xu.get_node_attributes(curr_branch_sk_graph,node_list=[k[0] for k in paths_to_endpt])


    #export the skeletons of the subgraphs
    exported_skeletons = [sk.convert_graph_to_skeleton(s) for s in subgraph_list]
    endpoint_nodes_coord #will have the endpoints belonging to each split
    
    
    # --------------- Part B: Getting Initial Mesh Correspondence ------------------------- #
    """
    3) Do Mesh correspondnece to get new branch meshes for the split skeleton
    - where have to do face resolving as well
    4) Check that both of the meshes are touching the soma
    5) If one of them is not touching the soma
    --> do water growing algorithm until it is

    """

    div_st_branch_face_corr = []
    div_st_branch_width = []
    for sub_sk in exported_skeletons:
        curr_branch_face_correspondence, width_from_skeleton = cu.mesh_correspondence_adaptive_distance(sub_sk,
                                                  curr_branch.mesh,
                                                 skeleton_segment_width = 1000,
                                                 distance_by_mesh_center=True)
        div_st_branch_face_corr.append(curr_branch_face_correspondence)
        div_st_branch_width.append(width_from_skeleton)
    
    
    divided_submeshes,divided_submeshes_idx = cu.groups_of_labels_to_resolved_labels(current_mesh = curr_branch.mesh,
                                          face_correspondence_lists=div_st_branch_face_corr)
    
    
#     # ------------ Intermediate part where intentionally messing up --------------- #    
#     label_to_expand = 1

#     #0) Turn the mesh into a graph
#     total_mesh_graph = nx.from_edgelist(curr_branch.mesh.face_adjacency)

#     #1) Get the nodes that represent the border
#     border_vertices =  curr_limb.get_concept_network_data_by_soma(soma_idx)["touching_soma_vertices"]
#     border_faces = tu.vertices_coordinates_to_faces(curr_branch.mesh,border_vertices)

#     label_face_idx = divided_submeshes_idx[label_to_expand]

#     final_faces = label_face_idx.copy()

#     for i in range(0,40):
#         final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))

#     other_mesh_faces = np.delete(np.arange(0,len(curr_branch.mesh.faces)),final_faces)
    
#     divided_submeshes_idx = [other_mesh_faces,final_faces]
#     divided_submeshes = [curr_branch.mesh.submesh([k],append=True) for k in divided_submeshes_idx]
    
#     sk.graph_skeleton_and_mesh(other_meshes=list(divided_submeshes),
#                            other_meshes_colors=["black","red"],
#                           other_skeletons=exported_skeletons,
#                           other_skeletons_colors=["black","red"],)
    
    
    
    # ---------------- Part C: Checking that both pieces are touching the soma ------------- #
    touching_pieces,touching_pieces_verts = tu.mesh_pieces_connectivity(main_mesh=tu.combine_meshes([curr_branch.mesh,curr_soma_mesh]),
                           central_piece=curr_soma_mesh,
                           periphery_pieces=divided_submeshes,
                           merge_vertices=True,
                            return_vertices=True,
                           print_flag=False)
    if print_flag:
        print(f"touching_pieces = {touching_pieces}")
        
        
        
    # --------------- Part D: Doing Waterfilling Unitl Both Pieces are Touching Soma ------------- #
    if len(touching_pieces) == 0:
        raise Exception("There were none of the new meshes that were touching the soma")
    if len(touching_pieces) < 2:
        #find which piece was not touching
        label_to_expand = 1 - touching_pieces[0]
        print(f"new_mesh {label_to_expand} was not touching the mesh so need to expand until touches soma")

        #0) Turn the mesh into a graph
        total_mesh_graph = nx.from_edgelist(curr_branch.mesh.face_adjacency)

        #1) Get the nodes that represent the border
        border_vertices =  curr_limb.get_concept_network_data_by_soma(soma_idx)["touching_soma_vertices"]
        border_faces = set(tu.vertices_coordinates_to_faces(curr_branch.mesh,border_vertices))

        label_face_idx = divided_submeshes_idx[label_to_expand]

        final_faces = label_face_idx.copy()

        n_touching_soma = 0
        counter = 0
        while n_touching_soma < 10:
            final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))
            n_touching_soma = len(border_faces.intersection(set(final_faces)))
            counter+= 1


        other_mesh_faces = np.delete(np.arange(0,len(curr_branch.mesh.faces)),final_faces)

        


        print(f"Took {counter} iterations to expand the label back")

        divided_submeshes_idx[label_to_expand] = final_faces
        divided_submeshes_idx[touching_pieces[0]] = other_mesh_faces

        #Need to fix the labels one more time to make sure the expansion did not cut off one of the labels
        print(f"divided_submeshes_idx = {divided_submeshes_idx}")
        divided_submeshes,divided_submeshes_idx = cu.groups_of_labels_to_resolved_labels(curr_branch.mesh,divided_submeshes_idx)

        print(f"divided_submeshes_idx = {divided_submeshes_idx}")

        divided_submeshes = [curr_branch.mesh.submesh([k],append=True) for k in divided_submeshes_idx]

        #recalculate the border vertices and the list should be 2
        touching_pieces,touching_pieces_verts = tu.mesh_pieces_connectivity(main_mesh=tu.combine_meshes([curr_branch.mesh,curr_soma_mesh]),
                               central_piece=curr_soma_mesh,
                               periphery_pieces=divided_submeshes,
                               merge_vertices=True,
                                return_vertices=True,
                               print_flag=False)
        if len(touching_pieces) != 2:
            raise Exception(f"Number of touching pieces not equal to 2 even after correction: {touching_pieces}")

    soma_border_verts = touching_pieces_verts

#     sk.graph_skeleton_and_mesh(other_meshes=list(divided_submeshes),
#                                other_meshes_colors=["black","red"],
#                               other_skeletons=exported_skeletons,
#                               other_skeletons_colors=["black","red"],
#                               other_scatter=[endpoint_nodes_coord[0].reshape(-1,3),endpoint_nodes_coord[1].reshape(-1,3)],
#                                other_scatter_colors=["black","red"],
#                               scatter_size=1)


    
    # ----------------- Part E: Check that the mesh can't be split ----------------- #
    
    # check that the mesh can't be split
    for j,sub in enumerate(divided_submeshes):
        c_mesh,c_indic = tu.split(sub)
        if len(c_mesh) > 1:
            raise Exception(f"New Mesh {j} had {len(c_mesh)} pieces after split")

            
    # ----------------- Part F: Reorganize the Concept Network ----------------- #
    neighbors_to_starting_node = xu.get_neighbors(curr_limb.concept_network,curr_starting_branch_idx)
    """
    sk.graph_skeleton_and_mesh(other_meshes=[curr_limb[k].mesh for k in neighbors_to_starting_node + [curr_starting_branch_idx]],
                              other_meshes_colors="random")
    """
    
    match=dict([(k,[]) for k in neighbors_to_starting_node])
    for ex_neighbor in neighbors_to_starting_node:
        ex_neighbor_branch = curr_limb[ex_neighbor]
        for j,endpt in enumerate(endpoint_nodes_coord):
            if len(nu.matching_rows(ex_neighbor_branch.endpoints,endpt))>0:
                match[ex_neighbor].append(j)

    #make sure that there was only one match
    for k,v in match.items():
        if len(v) != 1:
            raise Exception(f"Neighbor {k} did not have one matching but instead had {v}")
            
   
    concept_network_copy = copy.deepcopy(curr_limb.concept_network)
    concept_network_copy.remove_node(curr_starting_branch_idx)
    concept_conn_comp = list(nx.connected_components(concept_network_copy))

    #divide up the connected components into the groups they belong to
    new_branch_groups = [[],[]]
    for c in concept_conn_comp:
        #find the matching neighbor in that
        matching_neighbor = c.intersection(set(neighbors_to_starting_node))
        if len(matching_neighbor) != 1:
            raise Exception(f"matching_neighbor was not size 1 : {matching_neighbor}")
        matching_neighbor = list(matching_neighbor)[0]
        new_branch_groups[match[matching_neighbor][0]].extend(list(c))

#     #check that the lists are not empty (DON'T ACTUALLY NEED THIS CHECK)
#     for i,g in enumerate(new_branch_groups):
#         if len(g) == 0:
#             raise Exception(f"New branch group {i} was empty after dividing the rest of the nodes")

    if print_flag:
        print(f"new_branch_groups = {new_branch_groups}")
        
        
    
    # Visualize that correctly split
#     divided_neighbor_meshes = [tu.combine_meshes([curr_limb[k].mesh for k in curr_group]) for curr_group in new_branch_groups]
#     divided_neighbor_meshes_with_original = [tu.combine_meshes([k,v]) for k,v in zip(divided_neighbor_meshes,divided_submeshes)]
#     #sk.graph_skeleton_and_mesh(other_meshes=)
#     sk.graph_skeleton_and_mesh(other_meshes=divided_neighbor_meshes_with_original,
#                               other_meshes_colors=["black","red"],
#                               other_skeletons=exported_skeletons,
#                               other_skeletons_colors=["black","red"],
#                               other_scatter=[endpoint_nodes_coord[0].reshape(-1,3),endpoint_nodes_coord[1].reshape(-1,3)],
#                                other_scatter_colors=["black","red"],)

    
    # ----------------- Part G: Put Everything Back into a Limb Object ----------------- #
    new_limbs = []
    for curr_new_branch_idx in range(len(new_branch_groups)):
        print(f"\n--- Working on new limb {curr_new_branch_idx} -------")
        
        #new_limb_dict[curr_new_branch_idx]["soma_border_verts"] = soma_border_verts[curr_new_branch_idx]

        # a) Creating the new concept network
        curr_limb_divided_skeletons =  [curr_limb[k].skeleton for k in new_branch_groups[curr_new_branch_idx]] + [exported_skeletons[curr_new_branch_idx]]
        closest_endpoint = starting_endpoints[curr_new_branch_idx]
        endpoints = neuron.Branch(exported_skeletons[curr_new_branch_idx]).endpoints
        curr_limb_concept_network = nru.branches_to_concept_network(curr_limb_divided_skeletons,closest_endpoint,np.array(endpoints).reshape(-1,3),
                                            touching_soma_vertices= soma_border_verts[curr_new_branch_idx])

        #Run some checks on the new concept network developed
        curr_starting_branch_idx= nru.check_concept_network(curr_limb_concept_network,closest_endpoint = closest_endpoint,
                                  curr_limb_divided_skeletons=curr_limb_divided_skeletons,print_flag=True)[0]

        # b) Creating the new mesh

        """Old way: 
        remaining_meshes_faces_idx =  [curr_limb[k].mesh_face_idx for k in new_branch_groups[curr_new_branch_idx]]
        remaining_meshes_faces_idx.append(np.array(curr_branch.mesh_face_idx[divided_submeshes_idx[curr_new_branch_idx]]))
        """

        new_limb_branch_face_idx = []
        remaining_meshes_face_idx = []
        total_face_count = 0
        for k in new_branch_groups[curr_new_branch_idx]:
            curr_face_idx  = curr_limb[k].mesh_face_idx
            remaining_meshes_face_idx.append(curr_face_idx)
            new_limb_branch_face_idx.append(np.arange(total_face_count,total_face_count+len(curr_face_idx)))
            total_face_count += len(curr_face_idx)

        last_face_idx = np.array(curr_branch.mesh_face_idx[divided_submeshes_idx[curr_new_branch_idx]])
        remaining_meshes_face_idx.append(last_face_idx)
        new_limb_branch_face_idx.append(np.arange(total_face_count,total_face_count+len(last_face_idx)))


        final_remaining_faces = np.concatenate(remaining_meshes_face_idx)                         
        curr_new_limb_mesh = curr_limb.mesh.submesh([final_remaining_faces],append=True,repair=False)

        """
        Checking that it went well:
        reovered_mesh = curr_new_limb_mesh.submesh([new_limb_branch_face_idx[2]],append=True,repair=False)
        original_mesh = curr_limb[new_branch_groups[curr_new_branch_idx][2]].mesh
        reovered_mesh,original_mesh
        """

        curr_limb_correspondence = dict()
        for j,neighb in enumerate(new_branch_groups[curr_new_branch_idx]):
            #calculate the new mesh correspondence
            curr_limb_correspondence[j] = dict(branch_skeleton = curr_limb[neighb].skeleton,
                                              width_from_skeleton = curr_limb[neighb].width,
                                              branch_mesh=curr_limb[neighb].mesh,
                                              branch_face_idx=new_limb_branch_face_idx[j])
        #add on the new mesh
        curr_limb_correspondence[len(new_branch_groups[curr_new_branch_idx])] = dict(
                        branch_skeleton = exported_skeletons[curr_new_branch_idx],
                        width_from_skeleton = div_st_branch_width[curr_new_branch_idx],
                        branch_mesh=divided_submeshes[curr_new_branch_idx],
                        branch_face_idx=new_limb_branch_face_idx[-1])

        # curr_limb_concept_network_dicts = [dict(starting_endpoints=endpoints,
        #                                        starting_node=curr_starting_branch_idx,
        #                                        starting_soma=soma_idx,
        #                                        starting_coordinate=closest_endpoint)]
        curr_limb_concept_network_dicts = {soma_idx:curr_limb_concept_network}

        new_limb_obj = neuron.Limb(mesh = curr_new_limb_mesh,
                                   curr_limb_correspondence=curr_limb_correspondence,
                                   concept_network_dict=curr_limb_concept_network_dicts)
        new_limb_obj.all_concept_network_data = nru.compute_all_concept_network_data_from_limb(new_limb_obj,
                                                                                               current_neuron_mesh=current_neuron_mesh,
                                                                                              soma_meshes=soma_meshes)

        new_limbs.append(new_limb_obj)
        #new_limb_dict[curr_new_branch_idx]["curr_starting_branch_idx"] = new_limb_obj.current_starting_node
        
    return new_limbs


def recursive_limb_splitting(curr_limb,soma_meshes,current_neuron_mesh,significant_skeleton_threshold=30000,
                            print_flag=False):
    """
    Purpose: To split the a limb as many times as needed if connected at the soma
    
    Pseudocode:
    1) Get all the somas that the limb is attached to (from all_concept_network_data)
    2) For each soma it is attached to, check if it needs to be split:
    
    If yes:
    a. Split the limb into its parts for that soma
    b. Compute the all_concept_network_data for all of the split limbs
    c. Start loop where send all of the limb objects through function and collect results
    d. concatenate results and return
    
    if No: 
    - continue to next soma
    
    if No and the last soma
    - return the limb object
    
    Arguments:
    1) Limb
    2) Soma
    
    Example: 
    ex_limb = current_neuron[2]
    split_limbs = recursive_limb_splitting(current_neuron,ex_limb)

    color_choices = ["red","black"]
    sk.graph_skeleton_and_mesh(other_meshes=[split_limbs[0].mesh,split_limbs[1].mesh],
                               other_meshes_colors=color_choices,
                               other_skeletons=[split_limbs[0].skeleton,split_limbs[1].skeleton],
                               other_skeletons_colors=color_choices)
    """

    #1) Get all the somas that the limb is attached to (from all_concept_network_data)
    total_somas_idx = curr_limb.touching_somas()
    total_soams_meshes = [soma_meshes[k] for k in total_somas_idx]
    
    if print_flag:
        print(f"total_somas_idx = {total_somas_idx}")
        print(f"total_soams_meshes = {total_soams_meshes}")
    
    #2) For each soma it is attached to, check if it needs to be split:
    for soma_idx,curr_soma_mesh in zip(total_somas_idx,total_soams_meshes):
        
        cut_coordinate = check_if_branch_needs_splitting(curr_limb,soma_idx,curr_soma_mesh,
                                   significant_skeleton_threshold=significant_skeleton_threshold,
                                   print_flag=print_flag)
        if print_flag:
            print(f"cut_coordinate = {cut_coordinate}")
        
        # If No then continue to next soma
        if cut_coordinate is None:
            continue
            
        #If yes:
        #a. Split the limb into its parts for that soma and
        #b. Compute the all_concept_network_data for all of the split limbs
        
        if print_flag:
            split_limb_objs = split_limb_on_soma(curr_limb,soma_idx,curr_soma_mesh,
                                                 current_neuron_mesh = current_neuron_mesh,
                                                 soma_meshes=soma_meshes,
                                                 cut_coordinate=cut_coordinate,
                                                print_flag=print_flag)
        else:
            with su.suppress_stdout_stderr():
                split_limb_objs = split_limb_on_soma(curr_limb,soma_idx,curr_soma_mesh,
                                                     current_neuron_mesh = current_neuron_mesh,
                                                     soma_meshes=soma_meshes,
                                                 cut_coordinate=cut_coordinate,
                                                print_flag=print_flag)
        
        if print_flag:
            print(f"split_limb_objs = {split_limb_objs}")
        
        total_split_limbs = []
        for split_limb in split_limb_objs:
            curr_results = recursive_limb_splitting(curr_limb=split_limb,
                                                    soma_meshes=soma_meshes,
                                                    current_neuron_mesh = current_neuron_mesh,
                                     significant_skeleton_threshold=significant_skeleton_threshold,
                                    print_flag=print_flag)
            total_split_limbs = total_split_limbs + curr_results
        return total_split_limbs
        
    #If Did not need to split any of then return the current limb
    if print_flag:
        print("Hit Recursive return point and returning limb")
    return [curr_limb]


nru = reload(nru)

from copy import deepcopy
def limb_split(limbs,soma_meshes,current_neuron_mesh,print_flag=False):
    """
    Purpose: Will end up giving new limb correspondence
    and other information that corresponds to limbs that have been split
    
    Example:
    current_file = "/notebooks/test_neurons/meshafterparty_processed/12345_double_soma_meshafterparty"
    neuron_obj = nru.decompress_neuron(filepath=current_file,
                                      original_mesh=current_file,
                                      minimal_output=True)

    
    limbs = [current_neuron[k] for k in current_neuron.get_limb_node_names(return_int=True)]
    soma_meshes = [current_neuron.concept_network.nodes[nru.soma_label(k)]["data"].mesh for k in [0,1]]
    current_neuron_mesh = current_neuron.mesh

    (new_limb_correspondence,
     new_soma_to_piece_connectivity,
     new_limb_meshes,
     new_limb_concept_networks,
     new_limb_labels) = limb_split(limbs,soma_meshes,current_neuron_mesh)
    
    
    """
    
    """
    will map the [limb_idx AS A NUMBER][branch_idx] to 
    dict_keys(['branch_skeleton', 'width_from_skeleton', 'branch_mesh', 'branch_face_idx'])
    """
    new_limb_correspondence = dict() 
    """
    Maps Soma to who they are connected to
    Ex: {0: [0, 1, 3, 4, 5, 9], 1: [1, 2, 6, 7, 8]}
    """
    new_soma_to_piece_connectivity = dict([(k,[]) for k,v in enumerate(soma_meshes)])

    """
    Just a list that will hold all of the meshes
    """
    new_limb_meshes = []

    """
    a dictionary that maps the limb_idx to a dictionary mapping the soma_idx to the concept map
    {0:{0:Graph},
     1:{0:Graph,1:Graph},
     2:{1:Graph}....}

    ** can easily get this from the limb property concept_network_data_by_soma
    """
    new_limb_concept_networks = dict()

    """
    Labels for the limbs
    """
    new_limb_labels = []



    """
    Pseudocode: 
    Iterate through each split limb
    1) Get all of the split limbs from that one limb
    For each limb :
    -look at the current length of the new_limb_meshes to get the current index for limb
    a) Add a new entry in the new_limb_correspondence by iterating over the branches
    b) get the somas that touch the limb and add CURRENT INDEX them to the new_soma_to_piece_connectivity dictionary
    c) Add the limb mesh to new_limb_meshes
    d) Use the concept_network_data_by_soma attribute to get the concept_network dictionary and add to 
        new_limb_concept_networks
    e) make new merge labels based on the number of connections in the concept_network_data
    """



    for limb_idx,curr_limb in enumerate(limbs):
        print(f"\n----- Working on Limb {limb_idx}--------")
        split_limbs = recursive_limb_splitting(curr_limb,soma_meshes,
                                              current_neuron_mesh=current_neuron_mesh,
                                              print_flag=print_flag)

        print(f"Found {len(split_limbs)} limbs after limb split")

        for sp_limb in split_limbs:
            #-look at the current length of the new_limb_meshes to get the current index for limb
            new_limb_idx = len(new_limb_meshes)
            #a) Add a new entry in the new_limb_correspondence by iterating over the branches
            new_limb_correspondence[new_limb_idx] = dict()
            for curr_branch_idx in sp_limb.get_branch_names():
                curr_branch = sp_limb[curr_branch_idx]
                new_limb_correspondence[new_limb_idx][curr_branch_idx] = dict(
                                                        branch_skeleton = curr_branch.skeleton,
                                                        width_from_skeleton=curr_branch.width,
                                                        branch_mesh=curr_branch.mesh,
                                                        branch_face_idx=curr_branch.mesh_face_idx)
            #b) get the somas that touch the limb and add CURRENT INDEX them to the new_soma_to_piece_connectivity dictionary
            touching_somas = sp_limb.touching_somas()
            for s in touching_somas:
                new_soma_to_piece_connectivity[s].append(new_limb_idx)

            #c) Add the limb mesh to new_limb_meshes
            new_limb_meshes.append(sp_limb.mesh)

            #d) Use the concept_network_data_by_soma attribute to get the concept_network dictionary and add to 
            #new_limb_concept_networks
            concept_network_dict = dict()
            for s in sp_limb.touching_somas():
                sp_limb.set_concept_network_directional(starting_soma=s)
                concept_network_dict[s] = deepcopy(sp_limb.concept_network)
            
            
            new_limb_concept_networks[new_limb_idx] = concept_network_dict

            #e) make new merge labels based on the number of connections in the concept_network_data
            # OPtions: (['Normal'], ['MergeError'])
            if len(sp_limb.concept_network_data_by_soma.keys()) > 1:
                new_limb_labels.append("MergeError")
            else:
                new_limb_labels.append("Normal")
    
    return new_limb_correspondence,new_soma_to_piece_connectivity,new_limb_meshes,new_limb_concept_networks,new_limb_labels
            
            



# ------------------------ For the preprocessing ----------------------- #

def preprocess_neuron(mesh=None,
                     mesh_file=None,
                     segment_id=None,
                     description=None,
                     sig_th_initial_split=15, #for significant splitting meshes in the intial mesh split
                     limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
                      filter_end_node_length=4001, #used in cleaning the skeleton during skeletonizations
                      return_no_somas = False,
                      decomposition_type="meshafterparty",
                      mesh_correspondence = "meshparty", # USE "meshafterparty_adaptive" for the adaptive manner
                      distance_by_mesh_center=True,
                      meshparty_segment_size = 0,
                      meshparty_n_surface_downsampling = 0,
                      somas=None,
                      branch_skeleton_data=None,
                      combine_close_skeleton_nodes = True,
                    combine_close_skeleton_nodes_threshold=700,
                     ):
    
    print("inside preproces neuron")
    whole_processing_tiempo = time.time()
    
    
    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"

    
    if mesh is None:
        if current_mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = trimesh.load_mesh(current_mesh_file)
    else:
        current_neuron = mesh
        
    # ************************ Phase A ********************************
    
    print("\n\n\n\n\n****** Phase A ***************\n\n\n\n\n")
    
    
    
    
    
    # --- 1) Doing the soma detection
    if somas is None:
        print("\n\nUsing the glia soma extract!!\n\n")
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
    
    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
    
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                           main_mesh_faces=current_neuron.faces,
#                            main_mesh_color = [0.,1.,0.,0.8]
#                           )

    # ********At this point assume that there are somas (if not would just skip to the limb skeleton stuff) *******
    
    
    
    
    
    
    
    
    #--- 2) getting the soma submeshes that are connected to each soma and identifiying those that aren't (and eliminating any mesh pieces inside the soma)
    
    main_mesh_total = current_neuron
    

    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False)

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


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


#     print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
#     print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")

   
    

    
    
    
    #--- 3)  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces
    
#     sk.graph_skeleton_and_mesh(other_meshes=[soma_meshes])

    

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
                                                    significance_threshold=250)
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

        # 3) Find all significant mesh pieces
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True)

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        soma_to_piece_touching_vertices = dict()
        limb_root_nodes = dict()
        for i,curr_soma in enumerate(seperate_soma_meshes):
            connected_mesh_pieces,connected_mesh_pieces_vertices  = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces
            
            soma_to_piece_touching_vertices[i] = dict()
            for piece_index,piece_idx in enumerate(connected_mesh_pieces):
                limb_root_nodes[piece_idx] = connected_mesh_pieces_vertices[piece_index][0]
                soma_to_piece_touching_vertices[i][piece_idx] = connected_mesh_pieces_vertices[piece_index]
        
#         border_debug = False
#         if border_debug:
#             print(f"soma_to_piece_connectivity = {soma_to_piece_connectivity}")
#             print(f"soma_to_piece_touching_vertices = {soma_to_piece_touching_vertices}")
            
       
        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    
    
    
    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")
        
    current_mesh_data = soma_touching_mesh_data
    soma_containing_idx = 0
    
    
    # ****Soma Touching mesh Data has the branches and the connectivity (So this is where you end up skipping if you don't have somas)***
    
    
    
    
    
    
    
    
    
# --------------------------------------------------- 8/28 starting with MESHAFTERPARY (for skeletonization and mesh correspondence---------------------
    
    if decomposition_type.lower() == "meshafterparty":
        print("Using DECOMPOSITION TYPE: meshAfterParty")
        # ---5) Working on the Actual skeleton of all of the branches


        global_start_time = time.time()
        

        endpoints_must_keep = dict()
        if branch_skeleton_data is None:
            for j,(soma_containing_mesh_idx,mesh_data) in enumerate(soma_touching_mesh_data.items()):
                
#                 sk_debug = True
#                 if sk_debug:
#                     from python_tools import system_utils as su
#                     su.compressed_pickle(mesh_data["branch_meshes"],
#                                         "ordered_branch_meshes")
#                 raise Exception("Done exporting branches")
                
                print(f"\n-- Working on Soma Continaing Mesh {j}--")
                current_branches = mesh_data["branch_meshes"]

                #skeletonize each of the branches
                total_skeletons = []

                for z,branch in enumerate(current_branches):
                    print(f"\n    -- Working on branch {z}--")
#                     if z != 2:
#                         continue
                    clean_time = time.time()
                    current_skeleton = sk.skeletonize_connected_branch(branch)
                    

#                     sk_debug = True
#                     if sk_debug:
#                         from python_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(branch,
#                                             "curr_branch_saved")
#                     if sk_debug:
#                         from python_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(current_skeleton,
#                                             "current_skeleton")
                    
                    print(f"    Total time for skeletonizing branch {z}: {time.time() - clean_time}")
                    clean_time = time.time()
                    """  --------- 9/17 Addition: Will make sure all branches have endpoint extending towards soma -------- """
                    #build the soma to piece touching vertices dictionary for this neuorn
                    curr_soma_to_piece_touching_vertices = dict()
                    for s_index,v in soma_to_piece_touching_vertices.items():
                        if z not in v:
                            continue
                        curr_soma_to_piece_touching_vertices[s_index] = soma_to_piece_touching_vertices[s_index][z]
                    
                    current_skeleton, curr_limb_endpoints_must_keep = sk.create_soma_extending_branches(
                                    current_skeleton=current_skeleton, #current skeleton that was created
                                    skeleton_mesh=branch, #mesh that was skeletonized
                                    soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,#dictionary mapping a soma it is touching to the border vertices,
                                    return_endpoints_must_keep=True,
                                                                    )
                    
                    endpoints_must_keep[z] = curr_limb_endpoints_must_keep
                            
                    

                    print(f"    Total time for Fixing Skeleton Soma Endpoint Extension {z}: {time.time() - clean_time}")
                    """  --------- END OF 9/17 Addition:  -------- """

                    #                     sk_debug = True
                    #                     if sk_debug:
                    #                         from python_tools import system_utils as su
                    #                         print("**Saving the skeletons**")
                    #                         su.compressed_pickle(current_skeleton,
                    #                                             "current_skeleton_after_addition")
                    
                    
                    
                        # --------  Doing the cleaning ------- #
                    clean_time = time.time()
                    print(f"filter_end_node_length = {filter_end_node_length}")
                    
                    """ 9/16 Edit: Now send the border vertices and don't want to clean anyy end nodes that are within certain distance of border"""
                    
                    total_border_vertices = dict()
                    for k in soma_to_piece_touching_vertices.keys():
                        if z in soma_to_piece_touching_vertices[k].keys():
                            total_border_vertices[k] = soma_to_piece_touching_vertices[k][z]
                            
                    #soma_border_vertices = total_border_vertices,
                    #skeleton_mesh=branch,
                    new_cleaned_skeleton = sk.clean_skeleton(current_skeleton,
                                            distance_func=sk.skeletal_distance,
                                      min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                                      return_skeleton=True,
                                        soma_border_vertices = total_border_vertices,
                                        skeleton_mesh=branch,
                                        endpoints_must_keep = curr_limb_endpoints_must_keep,
                                      print_flag=False)
                    
#                     sk_debug = True
#                     if sk_debug:
#                         from python_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(new_cleaned_skeleton,
#                                             "new_cleaned_skeleton")
                    
                    
                    
                    
                    print(f"    Total time for cleaning of branch {z}: {time.time() - clean_time}")
                    if len(new_cleaned_skeleton) == 0:
                        raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")
                    total_skeletons.append(new_cleaned_skeleton)
                    

                soma_touching_mesh_data[j]["branch_skeletons"] = total_skeletons

            print(f"Total time for skeletonization = {time.time() - global_start_time}")
        else:
            print("***** Skipping INITIAL skeletonization because recieved pre-computed skeleton *****")
            soma_touching_mesh_data[0]["branch_skeletons"] = branch_skeleton_data
            
            
#         sk_debug = True
#         if sk_debug:
#             from python_tools import system_utils as su
#             from copy import deepcopy
#             print("**Saving the skeletons**")
#             su.compressed_pickle(deepcopy(soma_touching_mesh_data[j]["branch_skeletons"]),
#                                 "initial_cleaned_skeletons")
        
        
        limb_correspondence = further_mesh_correspondence_processing_from_skeleton(soma_touching_mesh_data,
                                                                                  branch_skeleton_data=branch_skeleton_data,
                                                                                  combine_close_skeleton_nodes = combine_close_skeleton_nodes,
                                                                                   distance_by_mesh_center=distance_by_mesh_center,
                                                                                combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
                                                                                  soma_to_piece_touching_vertices=soma_to_piece_touching_vertices,
                                                                                  endpoints_must_keep=endpoints_must_keep)
        
        

    
# --------------------------------- WHERE FINISH WITH meshAFTERpary option -------------------------------------------- #

# --------------------------------- where START with meshparty option ------------------------------------------------ #
    elif decomposition_type.lower() == "meshparty":
        print("Using DECOMPOSITION TYPE: meshparty")
        """
        The things that we need to have by the end of the meshparty part:

        current_mesh_data[0]["branch_meshes"]

        Using:
        curr_limb_mesh = current_mesh_data[soma_containing_idx]["branch_meshes"][limb_idx]

        limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
        limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]

        limb_correspondence[limb_idx][j]["branch_skeleton"] = curr_branch_sk
        limb_correspondence[limb_idx][j]["width_from_skeleton"] = width_from_skeleton
        
        Things to help you: 
        -- helps with the connectivity (maps the somas limbs and the )
        soma_to_piece_connectivity[i] = connected_mesh_pieces
        
        
        -- the actual meshes of the limbs
        current_mesh_data[0]["branch_meshes"]
        
        """
        
        
        limb_correspondence = dict()
        endpoints_must_keep = None

        total_skeletons =  []
        for soma_containing_idx in current_mesh_data.keys():
            for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
                print(f"------------Working on limb #{limb_idx}-------------")
                limb_correspondence[limb_idx] = dict()
                
                
                limb_mesh_mparty =current_mesh_data[0]["branch_meshes"][limb_idx]
                
                #get a vertex to serve as the root node
                root = limb_root_nodes[limb_idx]
                print(f"Root for limb {limb_idx} = {root}")
                
                # Step 1: Getting the skeleton
                
                limb_obj_tr_io  = trimesh_io.Mesh(vertices = limb_mesh_mparty.vertices,
                                                       faces = limb_mesh_mparty.faces,
                                                       normals=limb_mesh_mparty.face_normals)
                
                
        

                meshparty_time = time.time()
                print("\nStep 1: Starting Skeletonization")
                sk_meshparty_obj, v = m_sk.skeletonize_mesh(limb_obj_tr_io,
                                      soma_pt = root,
                                      soma_radius = 0,
                                      collapse_soma = False,
                                      invalidation_d=12000,
                                      smooth_vertices=True,
                                       smooth_neighborhood = 1,
                                      compute_radius = True, #Need the pyembree list
                                      compute_original_index=True,
                                      verbose=True)
                print(f"Total time for meshParty skeletonization = {time.time() - meshparty_time}")
                
                sk_meshparty = sk_meshparty_obj.vertices[sk_meshparty_obj.edges]
                total_skeletons.append(sk_meshparty)
                
                
                
                if mesh_correspondence != "meshafterparty_adaptive":
                    
                    #Step 2: Getting the branches
                    print("\nStep 2: Decomposing Branches")
                    meshparty_time = time.time()

                    segments, segment_maps = m_sk.compute_segments(sk_meshparty_obj)
                    # getting the skeletons that go with them
                    segment_branches = [sk_meshparty_obj.vertices[np.vstack([k[:-1],k[1:]]).T] for k in segments]
                    
                    
                    #------------ Add in the downsampling and resizing ----------------- #
                    
                    
                    if meshparty_n_surface_downsampling > 0:
                        print(f"Going to downsample the meshparty segments {meshparty_n_surface_downsampling}")
                        for j,s in enumerate(segment_branches):
                            for i in range(n_surface_downsampling):
                                s = downsample_skeleton(s)
                            segment_branches[j] = s
                    
                    if meshparty_segment_size > 0:
                        print(f"Resizing meshparty skeletal segments to length {meshparty_segment_size} nm")
                        for j,s in enumerate(segment_branches):
                            segment_branches[j] = sk.resize_skeleton_branch(s,segment_width = meshparty_segment_size)
                    
                    #------------ END OF downsampling and resizing ----------------- #


                    print(f"Total time for meshParty decomposition = {time.time() - meshparty_time}")


                    # -- Step 3: Creating the mesh correspondence --
                    print("\nStep 3: Mesh correspondence")
                    meshparty_time = time.time()

                    sk_vertices_to_mesh_vertices = gu.invert_mapping(sk_meshparty_obj.mesh_to_skel_map)
                    #getting a list of all the original vertices that belong to each segment
                    segment_mesh_vertices = [np.unique(np.concatenate([sk_vertices_to_mesh_vertices[k] for k in segment_list])) for segment_list in segments]
                    #getting a list of all the original vertices that belong to each segment
                    segment_mesh_faces = [np.unique(limb_mesh_mparty.vertex_faces[k]) for k in segment_mesh_vertices]
                    segment_mesh_faces = [k[k>=0] for k in segment_mesh_faces]

                    face_lookup = gu.invert_mapping(segment_mesh_faces)

                    curr_limb_mesh = limb_mesh_mparty


                    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
                    print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")

                    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = curr_limb_mesh,
                                                                                face_lookup=face_lookup,
                                                                                no_missing_labels = list(original_labels))


                    # -- splitting the mesh pieces into individual pieces
                    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

                    #print(f"divided_submeshes = {divided_submeshes}")
                    #print(f"divided_submeshes_idx = {divided_submeshes_idx}")

                    print(f"Total time for meshParty mesh correspondence = {time.time() - meshparty_time}")

                    # -- Step 4: Getting the Widths ---
                    print("\nStep 4: Retrieving Widths")
                    meshparty_time = time.time()

                    segment_width_measurements = [sk_meshparty_obj.vertex_properties["rs"][k] for k in segments]
                    segment_widths_median = [np.median(k) for k in segment_width_measurements]

                    print(f"Total time for meshParty Retrieving Widths = {time.time() - meshparty_time}")
                    # ---- Our Final Products -----

                    segment_branches #skeleton branches

                    divided_submeshes, divided_submeshes_idx #mesh correspondence (mesh and indices)

                    segment_widths_median #widths



                    output_data_flag = False
                    for i,b in enumerate(segment_branches):
                        if b.shape[0] == 0:
                            print(f"Branch {i} has 0 length")
                            output_data_flag = True

                    print(f"\nbranch segment sizes = {dict([(i,b.shape) for i,b in enumerate(segment_branches)])}")

                    from python_tools import system_utils as su
                    if output_data_flag:
                        print("******* exporting the data for debugging *************")
                        limb_mesh_mparty.export(f"limb_{limb_idx}_mesh.off")
                        su.save_object(sk_meshparty_obj,f"limb_{limb_idx}_sk_meshparty_obj")

                    for k,(sk_b,width_b) in enumerate(zip(segment_branches,
                                                          segment_widths_median)):
                        limb_correspondence[limb_idx][k] = dict()
                        limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
                        limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]

                        limb_correspondence[limb_idx][k]["branch_skeleton"] = sk_b
                        limb_correspondence[limb_idx][k]["width_from_skeleton"] = width_b
                        
    #-------------- Starting the adaptive mesh correspondence for the meshparty option (COPIED CODE FROM ABOVE) --------------------------------- #
        if mesh_correspondence == "meshafterparty_adaptive":
            print("****************** Using the adaptive mesh correspondence in the meshparty option ***************************")
            soma_touching_mesh_data[0]["branch_skeletons"] = total_skeletons
            limb_correspondence = further_mesh_correspondence_processing_from_skeleton(soma_touching_mesh_data,
                                                                                      combine_close_skeleton_nodes = combine_close_skeleton_nodes,
                                                                                       distance_by_mesh_center=distance_by_mesh_center,
                                                                                        combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold)
            
            
            
            
    #---------------------------------------------- STOPPING the adaptive mesh correspondence for the meshparty option --------------------------------- #

# --------------------------------- where END with meshparty option ------------------------------------------------ #


    else:
        raise Exception(f"Invalid decomposition type chosen: {decomposition_type.lower()}")
    
    # Visualizing the results of getting the mesh to skeletal segment correspondence completely 1-to-1
    
#     from matplotlib import pyplot as plt
#     fig,ax = plt.subplots(1,1)
#     bins = plt.hist(np.array(total_widths),bins=100)
#     ax.set_xlabel("Width measurement of mesh branch (nm)")
#     ax.set_ylabel("frequency")
#     ax.set_title("Width measurement of mesh branch frequency")
#     plt.show()
    
#     sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
#                           other_meshes_colors="random",
#                           other_skeletons=total_branch_skeletons,
#                           other_skeletons_colors="random",
#                           #html_path="two_soma_mesh_skeleton_decomp.html"
#                           )

    
#     sk.graph_skeleton_and_mesh(other_meshes=[total_branch_meshes[47]],
#                               other_meshes_colors="random",
#                               other_skeletons=[total_branch_skeletons[47]],
#                               other_skeletons_colors="random",
#                               html_path="two_soma_mesh_skeleton_decomp.html")
    
    
    
    
    
    
    
    
    
    
    
    # ********************   Phase C ***************************************
    # PART 3: LAST PART OF ANALYSIS WHERE MAKES CONCEPT GRAPHS
    
    
    print("\n\n\n\n\n****** Phase C ***************\n\n\n\n\n")
    
    
    
    
    
    # ---1) Making concept graphs:

    limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
        limb_correspondence = limb_correspondence,
        #limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        #limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        
        soma_meshes=current_mesh_data[0]["soma_meshes"],
        soma_idx_connectivity=current_mesh_data[0]["soma_to_piece_connectivity"],
        limb_to_soma_starting_endpoints = endpoints_must_keep,
        #soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        #soma_idx_connectivity = soma_idx_connectivity,
        
        current_neuron=current_neuron,
        return_limb_labels=True
        )

#     #Before go and get concept maps:
#     print("Sizes of dictionaries sent")
#     for curr_limb in limb_idx_to_branch_skeletons_dict.keys():
#         print((len(limb_idx_to_branch_skeletons_dict[curr_limb]),len(limb_idx_to_branch_meshes_dict[curr_limb])))


#     print("\n\n Sizes of concept maps gotten back")
#     for curr_idx in limb_concept_networks.keys():
#         for soma_idx,concept_network in limb_concept_networks[curr_idx].items():
#             print(len(np.unique(list(concept_network.nodes()))))
            
    
    
    
    
    
    
    
    
    

    
    # ---2) Packaging the data into a dictionary that can be sent to the Neuron class to create the object
    
    #Preparing the data structure to save or use for Neuron class construction

#     sk_debug = True
#     if sk_debug:
#         from python_tools import system_utils as su
#         su.compressed_pickle(limb_correspondence,
#                             "fourth_original_limb_correspondence")
    
    
    """ 9/17 Addition: No longer doing limb split because do not require it"""
    
    perform_limb_split = False
    
    if not perform_limb_split:
        #Old way of getting the processed data
        print("NOT USING THE LIMB SPLITTING ALGORITHM")
        preprocessed_data= dict(
                                soma_meshes = current_mesh_data[0]["soma_meshes"],
                                soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
                                soma_sdfs = total_soma_list_sdf,
                                insignificant_limbs=insignificant_limbs,
                                non_soma_touching_meshes=non_soma_touching_meshes,
                                inside_pieces=inside_pieces,
                                limb_correspondence=limb_correspondence,
                                limb_concept_networks=limb_concept_networks,
                                limb_labels=limb_labels,
                                limb_meshes=current_mesh_data[0]["branch_meshes"],
                                )
    else:
    
        print("USING THE LIMB SPLITTING ALGORITHM")

        # ------------- Post-Processing: Will now do the limb splitting if need ----------------- #
        limb_concept_networks=limb_concept_networks
        limb_meshes=current_mesh_data[0]["branch_meshes"]

        #make the limb objects
        new_limb_objs = []
        for curr_limb_idx,curr_limb_mesh in enumerate(current_mesh_data[0]["branch_meshes"]):
            new_curr_limb = neuron.Limb(mesh = curr_limb_mesh,
                                           curr_limb_correspondence=limb_correspondence[curr_limb_idx],
                                           concept_network_dict=limb_concept_networks[curr_limb_idx])
            new_limb_objs.append(new_curr_limb)


        #need to run the function
        soma_meshes = current_mesh_data[0]["soma_meshes"]
        current_neuron_mesh = current_neuron


    #     from python_tools import system_utils as su
    #     su.compressed_pickle(new_limb_objs,
    #                         filename="new_limb_objs")
    #     su.compressed_pickle(soma_meshes,
    #                         filename="soma_meshes")
    #     su.compressed_pickle(current_neuron,
    #                         filename="current_neuron")
    #     su.compressed_pickle(limb_concept_networks,
    #                         filename="limb_concept_networks")
    #     su.compressed_pickle(limb_concept_networks,
    #                         filename="limb_concept_networks")


        #send the new data to the function
        (new_limb_correspondence,
         new_soma_to_piece_connectivity,
         new_limb_meshes,
         new_limb_concept_networks,
         new_limb_labels) = limb_split(limbs=new_limb_objs,
                                          soma_meshes=soma_meshes,
                                          current_neuron_mesh=current_neuron)

    #     if sk_debug:
    #         from python_tools import system_utils as su
    #         su.compressed_pickle(new_limb_correspondence,
    #                             "fifth_new_limb_correspondence")
        preprocessed_data = dict(
                                soma_meshes = current_mesh_data[0]["soma_meshes"],
                                soma_to_piece_connectivity = new_soma_to_piece_connectivity,
                                soma_sdfs = total_soma_list_sdf,
                                insignificant_limbs=insignificant_limbs,
                                non_soma_touching_meshes=non_soma_touching_meshes,
                                inside_pieces=inside_pieces,
                                limb_correspondence=new_limb_correspondence,
                                limb_concept_networks=new_limb_concept_networks,
                                limb_labels=new_limb_labels,
                                limb_meshes=new_limb_meshes
                                )

    


    
    
    
    print(f"\n\n\n Total processing time = {time.time() - whole_processing_tiempo}")
    
    #print(f"returning preprocessed_data = {preprocessed_data}")
    return preprocessed_data

'''
def preprocess_neuron_OLD(mesh=None,
                     mesh_file=None,
                     segment_id=None,
                     description=None,
                     sig_th_initial_split=15, #for significant splitting meshes in the intial mesh split
                     limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
                      filter_end_node_length=5000, #used in cleaning the skeleton during skeletonizations
                      return_no_somas = False
                     ):
    
    
    whole_processing_tiempo = time.time()
    
    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"

    
    if mesh is None:
        if current_mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = trimesh.load_mesh(current_mesh_file)
    else:
        current_neuron = mesh
        
    # ************************ Phase A ********************************
    
    print("\n\n\n\n\n****** Phase A ***************\n\n\n\n\n")
    
    
    
    
    
    # --- 1) Doing the soma detection
    
    soma_mesh_list,run_time,total_soma_list_sdf = sm.extract_soma_center(segment_id,
                                             current_neuron.vertices,
                                             current_neuron.faces)
    
    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
    
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                           main_mesh_faces=current_neuron.faces,
#                            main_mesh_color = [0.,1.,0.,0.8]
#                           )

    # ********At this point assume that there are somas (if not would just skip to the limb skeleton stuff) *******
    
    
    
    
    
    
    
    
    #--- 2) getting the soma submeshes that are connected to each soma and identifiying those that aren't (and eliminating any mesh pieces inside the soma)
    
    main_mesh_total = current_neuron
    

    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False)

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


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")

   
    

    
    
    
    #--- 3)  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces
    
#     sk.graph_skeleton_and_mesh(other_meshes=[soma_meshes])

    

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
        print("\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx]

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        mesh_pieces_without_soma = sm.subtract_soma(current_soma_mesh_list,current_mesh,
                                                    significance_threshold=250)
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)

        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        soma_meshes = current_mesh.submesh([soma_faces],append=True)

        # finding the non-soma original faces
        non_soma_faces = tu.original_mesh_faces_map(current_mesh,soma_meshes,matching=False)
        non_soma_stacked_mesh = current_mesh.submesh([non_soma_faces],append=True)

        print(f"Total time for Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()

        # 3) Find all significant mesh pieces
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True)

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        for i,curr_soma in enumerate(seperate_soma_meshes):
            connected_mesh_pieces,connected_mesh_pieces_vertices  = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces

        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    
    
    
    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")
    
    
    # ****Soma Touching mesh Data has the branches and the connectivity (So this is where you end up skipping if you don't have somas)***
    
    
    
    
    
    
    
    
    
    
    
    
    # ---5) Working on the Actual skeleton of all of the branches

    
    global_start_time = time.time()

    for j,(soma_containing_mesh_idx,mesh_data) in enumerate(soma_touching_mesh_data.items()):
        print(f"\n-- Working on Soma Continaing Mesh {j}--")
        current_branches = mesh_data["branch_meshes"]

        #skeletonize each of the branches
        total_skeletons = []

        for z,branch in enumerate(current_branches):
            print(f"\n    -- Working on branch {z}--")
            curren_skeleton = sk.skeletonize_connected_branch(branch)
            #clean the skeleton
                # --------  Doing the cleaning ------- #
            clean_time = time.time()
            
            new_cleaned_skeleton = sk.clean_skeleton(curren_skeleton,
                                    distance_func=sk.skeletal_distance,
                              min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                              return_skeleton=True,
                              print_flag=False)
            print(f"    Total time for skeleton and cleaning of branch {z}: {time.time() - clean_time}")
            if len(new_cleaned_skeleton) == 0:
                raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")
            total_skeletons.append(new_cleaned_skeleton)

        soma_touching_mesh_data[j]["branch_skeletons"] = total_skeletons

    print(f"Total time for skeletonization = {time.time() - global_start_time}")
    
    
    
    
    
    
    
    
    
    
    
    
    # *************** Phase B *****************
    
    print("\n\n\n\n\n****** Phase B ***************\n\n\n\n\n")
    
    current_mesh_data = soma_touching_mesh_data
    
    
    # visualizing the original neuron
#     current_neuron = trimesh.load_mesh(current_mesh_file)
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                               main_mesh_faces=current_neuron.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )
    
    
    # visualizing the somas that were extracted
#     soma_meshes = tu.combine_meshes(current_mesh_data[0]["soma_meshes"])
#     sk.graph_skeleton_and_mesh(main_mesh_verts=soma_meshes.vertices,
#                               main_mesh_faces=soma_meshes.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )


    # # Visualize the extracted branches
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons"],
    #                           other_skeletons_colors="random")
    
    
    
    
    
    
    
    
    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing
    
    skelton_cleaning_threshold = 4001
    total_cleaned = []
    for j,curr_skeleton_to_clean in enumerate(current_mesh_data[0]["branch_skeletons"]):
        print(f"\n---- Working on Limb {j} ----")
        start_time = time.time()
        print(f"before cleaning limb size of skeleton = {curr_skeleton_to_clean.shape}")
        distance_cleaned_skeleton = sk.clean_skeleton(
                                                    curr_skeleton_to_clean,
                                                    distance_func=sk.skeletal_distance,
                                                    min_distance_to_junction = skelton_cleaning_threshold,
                                                    return_skeleton=True,
                                                    print_flag=False) 
        #make sure still connected componet
        distance_cleaned_skeleton_components = nx.number_connected_components(sk.convert_skeleton_to_graph(distance_cleaned_skeleton))
        if distance_cleaned_skeleton_components > 1:
            raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

        print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")
        cleaned_branch = sk.clean_skeleton_with_decompose(distance_cleaned_skeleton)

        cleaned_branch_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cleaned_branch))
        if cleaned_branch_components > 1:
            raise Exception(f"cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")

        #do the cleanin ghtat removes loops from branches
        print(f"After DECOMPOSITION cleaning limb size of skeleton = {cleaned_branch.shape}")
        print(f"Total time = {time.time() - start_time}")
        total_cleaned.append(cleaned_branch)

    current_mesh_data[0]["branch_skeletons_cleaned"] = total_cleaned
    
    
    
    # checking all cleaned branches are connected components

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Original limb {k} was not a single component: it was actually {n_components} components")

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons_cleaned"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Cleaned limb {k} was not a single component: it was actually {n_components} components")
            
    
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons_cleaned"],
    #                           other_skeletons_colors="random",
    #                           mesh_alpha=0.15,
    #                           html_path=f"{segment_id}_limb_skeleton.html")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # --- 2) Decomposing of limbs into branches and finding mesh correspondence (using adaptive mesh correspondence followed by a water fill for conflict and empty faces), checking that it went well with no empty meshes and all connected component graph (even when downsampling the skeleton) when constructed from branches, plus visualization at end
    
    

    start_time = time.time()

    limb_correspondence = dict()
    soma_containing_idx= 0

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"Working on limb #{limb_idx}")
            limb_correspondence[limb_idx] = dict()
            curr_limb_sk = current_mesh_data[soma_containing_idx]["branch_skeletons_cleaned"][limb_idx]
            curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches

            for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
                limb_correspondence[limb_idx][j] = dict()


                curr_branch_face_correspondence, width_from_skeleton = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                              curr_limb_mesh,
                                             skeleton_segment_width = 1000)



                if len(curr_branch_face_correspondence) > 0:
                    curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True)
                else:
                    curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))

                limb_correspondence[limb_idx][j]["branch_skeleton"] = curr_branch_sk
                limb_correspondence[limb_idx][j]["correspondence_mesh"] = curr_submesh
                limb_correspondence[limb_idx][j]["correspondence_face_idx"] = curr_branch_face_correspondence
                limb_correspondence[limb_idx][j]["width_from_skeleton"] = width_from_skeleton


    print(f"Total time for decomposition = {time.time() - start_time}")
    
    
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    
    empty_submeshes = []

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"\n---- checking limb {limb_idx}---")
            print(f"Limb {limb_idx} decomposed into {len(limb_correspondence[limb_idx])} branches")

            #get all of the skeletons and make sure that they from a connected component
            divided_branches = [limb_correspondence[limb_idx][k]["branch_skeleton"] for k in limb_correspondence[limb_idx]]
            divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                            sk.stack_skeletons(divided_branches))

            divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

            cleaned_limb_skeleton = current_mesh_data[0]['branch_skeletons_cleaned'][limb_idx]
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


            for j in limb_correspondence[limb_idx].keys():
                if len(limb_correspondence[limb_idx][j]["correspondence_mesh"].faces) == 0:
                    empty_submeshes.append(dict(limb_idx=limb_idx,branch_idx = j))

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        
        

    # from python_tools import matplotlib_utils as mu

    # sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
    #                           other_meshes_colors="random",
    #                            other_skeletons=total_branch_skeletons,
    #                            other_skeletons_colors="random"
    #                           )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ---3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
    
    #--- this is the function that will clean up a limb piece so have 1-1 correspondence

    #things to prep for visualizing the axons
#     total_widths = []
#     total_branch_skeletons = []
#     total_branch_meshes = []

    soma_containing_idx = 0

    for limb_idx in limb_correspondence.keys():
        mesh_start_time = time.time()
        #clear out the mesh correspondence if already in limb_correspondecne
        for k in limb_correspondence[limb_idx].keys():
            if "branch_mesh" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_mesh"]
            if "branch_face_idx" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_face_idx"]
        #geting the current limb mesh
        print(f"\n\nWorking on limb_correspondence for #{limb_idx}")
        no_missing_labels = list(limb_correspondence[limb_idx].keys()) #counts the number of divided branches which should be the total number of labels
        curr_limb_mesh = current_mesh_data[soma_containing_idx]["branch_meshes"][limb_idx]

        #set up the face dictionary
        face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

        for j,branch_piece in limb_correspondence[limb_idx].items():
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
                         no_missing_labels = list(original_labels)
        )


        # -- splitting the mesh pieces into individual pieces
        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

        #-- check that all the split mesh pieces are one component --#

        #save off the new data as branch mesh
        for k in limb_correspondence[limb_idx].keys():
            limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
            limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]
            
            #clean the limb correspondence that we do not need
            del limb_correspondence[limb_idx][k]["correspondence_mesh"]
            del limb_correspondence[limb_idx][k]["correspondence_face_idx"]
#             total_widths.append(limb_correspondence[limb_idx][k]["width_from_skeleton"])
#             total_branch_skeletons.append(limb_correspondence[limb_idx][k]["branch_skeleton"])
#             total_branch_meshes.append(limb_correspondence[limb_idx][k]["branch_mesh"])

        print(f"Total time for limb mesh processing = {time.time() - mesh_start_time}")
    
    
    
    
    
    # Visualizing the results of getting the mesh to skeletal segment correspondence completely 1-to-1
    
#     from matplotlib import pyplot as plt
#     fig,ax = plt.subplots(1,1)
#     bins = plt.hist(np.array(total_widths),bins=100)
#     ax.set_xlabel("Width measurement of mesh branch (nm)")
#     ax.set_ylabel("frequency")
#     ax.set_title("Width measurement of mesh branch frequency")
#     plt.show()
    
#     sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
#                           other_meshes_colors="random",
#                           other_skeletons=total_branch_skeletons,
#                           other_skeletons_colors="random",
#                           #html_path="two_soma_mesh_skeleton_decomp.html"
#                           )

    
#     sk.graph_skeleton_and_mesh(other_meshes=[total_branch_meshes[47]],
#                               other_meshes_colors="random",
#                               other_skeletons=[total_branch_skeletons[47]],
#                               other_skeletons_colors="random",
#                               html_path="two_soma_mesh_skeleton_decomp.html")
    
    
    
    
    
    
    
    
    
    
    
    # ********************   Phase C ***************************************
    # PART 3: LAST PART OF ANALYSIS WHERE MAKES CONCEPT GRAPHS
    
    
    print("\n\n\n\n\n****** Phase C ***************\n\n\n\n\n")
    
    
    
    
    
    # ---1) Making concept graphs:

    limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
        limb_correspondence = limb_correspondence,
        #limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        #limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        
        soma_meshes=current_mesh_data[0]["soma_meshes"],
        soma_idx_connectivity=current_mesh_data[0]["soma_to_piece_connectivity"],
        #soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        #soma_idx_connectivity = soma_idx_connectivity,
        
        current_neuron=current_neuron,
        return_limb_labels=True
        )

#     #Before go and get concept maps:
#     print("Sizes of dictionaries sent")
#     for curr_limb in limb_idx_to_branch_skeletons_dict.keys():
#         print((len(limb_idx_to_branch_skeletons_dict[curr_limb]),len(limb_idx_to_branch_meshes_dict[curr_limb])))


#     print("\n\n Sizes of concept maps gotten back")
#     for curr_idx in limb_concept_networks.keys():
#         for soma_idx,concept_network in limb_concept_networks[curr_idx].items():
#             print(len(np.unique(list(concept_network.nodes()))))
            
    
    
    
    
    
    
    
    
    

    
    # ---2) Packaging the data into a dictionary that can be sent to the Neuron class to create the object
    
    #Preparing the data structure to save or use for Neuron class construction

    
    
    preprocessed_data = dict(
                            soma_meshes = current_mesh_data[0]["soma_meshes"],
                            soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
                            soma_sdfs = total_soma_list_sdf,
                            insignificant_limbs=insignificant_limbs,
                            non_soma_touching_meshes=non_soma_touching_meshes,
                            inside_pieces=inside_pieces,
                            limb_correspondence=limb_correspondence,
                            limb_concept_networks=limb_concept_networks,
                            limb_labels=limb_labels,
                            limb_meshes=current_mesh_data[0]["branch_meshes"],
                            )

    
    
    print(f"\n\n\n Total processing time = {time.time() - whole_processing_tiempo}")
    
    print(f"returning preprocessed_data = {preprocessed_data}")
    return preprocessed_data
    
'''
    
    