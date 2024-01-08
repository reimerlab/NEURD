
import copy
from copy import deepcopy
import ipyvolume as ipv
import itertools
from importlib import reload
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import trimesh
from datasci_tools import numpy_dep as np
from datasci_tools import general_utils as gu
import ipyvolume as ipv

soma_color = "red"
glia_color = "aqua"
nuclei_color = "black"

def plot_soma_limb_concept_network(neuron_obj,
                                  soma_color="red",
                                  limb_color="aqua",
                                   multi_touch_color = "brown",
                                  node_size=500,
                                  font_color="black",
                                  node_colors=dict(),
                                  **kwargs):
    """
    Purpose: To plot the connectivity of the soma and the meshes in the neuron

    How it was developed: 

    from datasci_tools import networkx_utils as xu
    xu = reload(xu)
    node_list = xu.get_node_list(my_neuron.concept_network)
    node_list_colors = ["red" if "S" in n else "blue" for n in node_list]
    nx.draw(my_neuron.concept_network,with_labels=True,node_color=node_list_colors,
           font_color="white",node_size=500)

    """

    node_list = xu.get_node_list(neuron_obj.concept_network)
    multi_touch_nodes = neuron_obj.same_soma_multi_touching_limbs
    node_list_colors = []
    for n in node_list:
        if n in list(node_colors.keys()):
            curr_color = node_colors[n]
        else:
            if "S" in n:
                curr_color = soma_color
            else:
                if int(n[1:]) in multi_touch_nodes:
                    curr_color = multi_touch_color
                else:
                    curr_color = limb_color
        node_list_colors.append(curr_color)
    
    #print(f"font_color = {font_color}")
    nx.draw(neuron_obj.concept_network,with_labels=True,node_color=node_list_colors,
           font_color=font_color,node_size=node_size)
    plt.show()
    
def plot_limb_concept_network_2D(neuron_obj,
                                 node_colors=dict(),
                                 limb_name=None,
                                 somas=None,
                                 starting_soma=None,
                                 starting_soma_group=None,
                                 default_color = "green",
                                  node_size=2000,
                                  font_color="white",
                                 font_size=30,
                                 directional=True,
                                 print_flag=False,
                                 plot_somas=True,
                                 soma_color="red",
                                 pos=None,
                                 pos_width = 3,
                                 width_min = 0.3,
                                 width_noise_ampl=0.2,
                                 pos_vertical_gap=0.05,
                                 fig_width=40,
                                 fig_height=20,
                                 suppress_disconnected_errors=True,
                                  **kwargs):
    """
    Purpose: To plot the concept network as a 2D networkx graph
    
    Pseudocode: 
    0) If passed a neuron object then use the limb name to get the limb object
    - make copy of limb object
    1) Get the somas that will be used for concept network
    2) Assemble the network by concatenating (directional or undirectional)
    3) Assemble the color list to be used for the coloring of the nodes. Will take:
    a. dictionary
    b. List
    c. Scalar value for all nodes
    
    4) Add on the soma to the graphs if asked for it
    5) Generate a hierarchical positioning for graph if position argument not specified
    
    for all the starting somas
    4) Use the nx.draw function
    
    Ex: 
    nviz = reload(nviz)
    xu = reload(xu)
    limb_idx = "L3"
    nviz.plot_limb_concept_network_2D(neuron_obj=uncompressed_neuron,
                                     limb_name=limb_idx,
                                     node_colors=color_dictionary)
    """
    
    #0) If passed a neuron object then use the limb name to get the limb object
    #- make copy of limb object
    
    if limb_name is None and len(node_colors)>0:
        #just strip the L name of the first key that is not the soma
        limb_name = [k for k in node_colors.keys() if "S" not in k][0].split("_")[0]
        print(f"No limb name was given so using {limb_name} because was the limb in the first key")
        
    if str(type(neuron_obj)) == str(neuron.Neuron):
        if not limb_name is None:
            limb_obj = deepcopy(neuron_obj.concept_network.nodes[limb_name]["data"])
        else:
            raise Exception("Neuron object recieved but no limb name specified")
    elif str(type(neuron_obj)) == str(neuron.Limb):
        limb_obj = deepcopy(neuron_obj)
    else:
        raise Exception(f"Non Limb or Neuron object recieved: {type(neuron_obj)}")
    
    #1) Get the somas that will be used for concept network
    if somas is None:
        somas = limb_obj.touching_somas()
        somas = [somas[0]]
    
    #2) Assemble the network by concatenating (directional or undirectional)
    # (COULD NOT END UP CONCATENATING AND JUST USE ONE SOMA AS STARTING POINT)
    if directional:
        graph_list = []
        if starting_soma is not None or starting_soma_group is not None:
            limb_obj.set_concept_network_directional(starting_soma=starting_soma,
                                                     soma_group_idx = starting_soma_group,
                                                     suppress_disconnected_errors=suppress_disconnected_errors)
            full_concept_network = limb_obj.concept_network_directional
        else:
            for s in somas:
                limb_obj.set_concept_network_directional(starting_soma=s,suppress_disconnected_errors=suppress_disconnected_errors)
                graph_list.append(limb_obj.concept_network_directional)
            full_concept_network = xu.combine_graphs(graph_list)
    else:
        full_concept_network = limb_obj.concept_network

    
    #3) Assemble the color list to be used for the coloring of the nodes. Will take:
    #a. dictionary
    #b. List
    #c. Scalar value for all nodes
    color_list = []
    node_list = xu.get_node_list(full_concept_network)
    
    if type(node_colors) == dict:
        #check to see if it is a limb_branch_dict
        L_check = np.any(["L" in k for k in node_colors.keys()])
        
        if L_check:
            if limb_name is None:
                raise Exception("Limb_branch dictionary given for node_colors but no limb name given to specify color mappings")
            node_colors = dict([(int(float(k.split("_")[-1])),v) for k,v in node_colors.items() if limb_name in k])
        
        if set(list(node_colors.keys())) != set(node_list):
            if print_flag:
                print(f"Node_colors dictionary does not have all of the same keys so using default color ({default_color}) for missing nodes")
        for n in node_list:
            if n in node_colors.keys():
                color_list.append(node_colors[n])
            else:
                color_list.append(default_color)
    elif type(node_colors) == list:
        if len(node_list) != len(node_colors):
            raise Exception(f"List of node_colors {(len(node_colors))} passed does not match list of ndoes in limb graph {(len(node_list))}")
        else:
            color_list = node_colors
    elif type(node_colors) == str:
        color_list = [node_colors]*len(node_list)
    else:
        raise Exception(f"Recieved invalid node_list type of {type(node_colors)}")
    
    #4) Add on the soma to the graphs if asked for it
    if plot_somas:
        #adding the new edges
        new_edge_list = []
        for k in limb_obj.all_concept_network_data:
            curr_soma = k["starting_soma"]
            curr_soma_group = k["soma_group_idx"]
            sm_name = f'S{k["starting_soma"]}_{k["soma_group_idx"]}'
            if curr_soma == limb_obj.current_starting_soma and curr_soma_group == limb_obj.current_soma_group_idx:
                new_edge_list.append((sm_name,k["starting_node"]))
            else:
                new_edge_list.append((k["starting_node"],sm_name))
        #new_edge_list = [(f'S{k["starting_soma"]}',k["starting_node"]) for k in limb_obj.all_concept_network_data]
        full_concept_network.add_edges_from(new_edge_list)
        #adding the new colors
        color_list += [soma_color]*len(new_edge_list)
    
    #print(f"full_concept_network.nodes = {full_concept_network.nodes}")
    #5) Generate a hierarchical positioning for graph if position argument not specified
    if pos is None:
        sm_name = f'S{limb_obj.current_starting_soma}_{limb_obj.current_soma_group_idx}'
        if plot_somas:
            starting_hierarchical_node = sm_name
        else:
            starting_hierarchical_node = {limb_obj.current_starting_node}
        #print(f"full_concept_network.nodes() = {full_concept_network.nodes()}")
        pos = xu.hierarchy_pos(full_concept_network,starting_hierarchical_node,
                              width=pos_width,width_min=width_min,width_noise_ampl=width_noise_ampl, vert_gap = pos_vertical_gap, vert_loc = 0, xcenter = 0.5)    
        #print(f"pos = {pos}")
    
    if print_flag:
        print(f"node_colors = {node_colors}")
        
    #6) Use the nx.draw function
    #print(f"pos={pos}")
    
    
    plt.figure(1,figsize=(fig_width,fig_height))
    nx.draw(full_concept_network,
            pos=pos,
            with_labels=True,
            node_color=color_list,
           font_color=font_color,
            node_size=node_size,
            font_size=font_size,
           **kwargs)
    plt.show()
    


def plot_concept_network(curr_concept_network,
                            arrow_size = 0.5,
                            arrow_color = "maroon",
                            edge_color = "black",
                            node_color = "red",
                            scatter_size = 0.1,
                            starting_node_color="pink",
                            show_at_end=True,
                            append_figure=False,
                            highlight_starting_node=True,
                            starting_node_size=-1,
                                 flip_y=True,
                        suppress_disconnected_errors=False):
    
    if starting_node_size == -1:
        starting_node_size = scatter_size*3
    
    """
    Purpose: 3D embedding plot of concept graph
    
    
    Pseudocode: 

    Pseudocode for visualizing direction concept graphs
    1) Get a dictionary of the node locations
    2) Get the edges of the graph
    3) Compute the mipoints and directions of all of the edges
    4) Plot a quiver plot using the midpoints and directions for the arrows
    5) Plot the nodes and edges of the graph

    
    Example of how to use with background plot of neuron:
    
    my_neuron #this is the curent neuron object
    plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                        show_at_end=False,
                        append_figure=False)

    # Just graphing the normal edges without

    curr_neuron_mesh =  my_neuron.mesh
    curr_limb_mesh =  my_neuron.concept_network.nodes[f"L{curr_limb_idx}"]["data"].mesh

    sk.graph_skeleton_and_mesh(other_meshes=[curr_neuron_mesh,curr_limb_mesh],
                              other_meshes_colors=["olive","brown"],
                              show_at_end=True,
                              append_figure=True)
                              
                              
    Another example wen testing: 
    from neurd import neuron_visualizations as nviz
    nviz = reload(nviz)
    nru = reload(nru)
    sk = reload(sk)

    nviz.plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                            scatter_size=0.3,
                            show_at_end=True,
                            append_figure=False)
    
    """
    
    
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))
        ipvu.hide_legend_panel()
    
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])

    node_edges = np.array(list(curr_concept_network.edges))



    if type(curr_concept_network) == type(nx.DiGraph()):
        #print("plotting a directional concept graph")
        #getting the midpoints then the directions of arrows for the quiver
        midpoints = []
        directions = []
        for n1,n2 in curr_concept_network.edges:
            difference = node_locations[n2] - node_locations[n1]
            directions.append(difference)
            midpoints.append(node_locations[n1] + difference/2)
        directions = np.array(directions)
        midpoints = np.array(midpoints)



        ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                        directions[:,0],directions[:,1],directions[:,2],
                        size=arrow_size,
                        size_selected=20,
                        color = arrow_color)

    #graphing the nodes

    # nodes_mesh = ipv.pylab.scatter(node_locations_array[:,0], 
    #                                 node_locations_array[:,1], 
    #                                 node_locations_array[:,2],
    #                                 size = 0.01,
    #                                 marker = "sphere")

    node_locations_array = np.array([v for v in node_locations.values()])
    #print(f"node_locations_array = {node_locations_array}")

    
    
    if highlight_starting_node:
        starting_node_num = xu.get_starting_node(curr_concept_network,only_one=False)
        starting_node_num_coord = [curr_concept_network.nodes[k]["data"].mesh_center for k in starting_node_num]
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        
        for k in starting_node_num_coord:
            sk.graph_skeleton_and_mesh(
                                       other_scatter=[k],
                                       other_scatter_colors=starting_node_color,
                                       scatter_size=starting_node_size,
                                       show_at_end=False,
                                       append_figure=True
                                      )
    
    #print(f"Current scatter size = {scatter_size}")
    concept_network_skeleton = nru.convert_concept_network_to_skeleton(curr_concept_network)
    sk.graph_skeleton_and_mesh(other_skeletons=[concept_network_skeleton],
                              other_skeletons_colors=edge_color,
                               other_scatter=[node_locations_array.reshape(-1,3)],
                               other_scatter_colors=node_color,
                               scatter_size=scatter_size,
                               show_at_end=False,
                               append_figure=True,
                               flip_y=flip_y,
                              )
    
    

    
    
    
    if show_at_end:
        ipv.show()
        
def visualize_concept_map(curr_concept_network,
                            node_color="red",
                            #node_color="black",
                            node_alpha = 0.5,
                            edge_color="black",
                            node_size=0.1,

                            starting_node=True,
                            starting_node_size = 0.3,
                            starting_node_color= "pink",
                            starting_node_alpha = 0.8,

                            arrow_color = "brown",
                            arrow_alpha = 0.8,
                            arrow_size = 0.5,

                            arrow_color_reciprocal = "brown",
                            arrow_alpha_reciprocal = 0.8,
                            arrow_size_reciprocal = 0.5,
                          
                            show_at_end=True,
                            append_figure=False,
                         print_flag=False,
                         flip_y=True):

    
    """
    Purpose: To plot a concept network with more
    parameters than previous plot_concept_network
    
    Ex: 
    
    neuron = reload(neuron)
    recovered_neuron = neuron.Neuron(recovered_neuron)
    nru = reload(nru)
    nviz = reload(nviz)
    returned_network = nru.whole_neuron_branch_concept_network(recovered_neuron,
                                      directional=True,
                                     limb_soma_touch_dictionary = "all",
                                     print_flag = False)
    
    nviz.visualize_concept_map(returned_network,
                          #starting_node_size = 10,
                          arrow_color = "green")
    """
    
    if flip_y:
        curr_concept_network = deepcopy(curr_concept_network)
        for k in curr_concept_network.nodes():
            curr_concept_network.nodes[k]["data"].mesh_center[...,1] = -curr_concept_network.nodes[k]["data"].mesh_center[...,1]
    
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))
        ipvu.hide_legend_panel()
    
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])
    
    node_edges = np.array(list(curr_concept_network.edges))


    #Adding the arrows for a directional graph
    if type(curr_concept_network) == type(nx.DiGraph()):
        #getting the midpoints then the directions of arrows for the quiver
        midpoints = []
        directions = []
        
        reciprocal_edges = xu.find_reciprocal_connections(curr_concept_network,redundant=True)
        
        for n1,n2 in curr_concept_network.edges:
            #going to skip reciprocal connections because will do them later
            if len(nu.matching_rows_old(reciprocal_edges,[n1,n2])) > 0:
                continue
            difference = node_locations[n2] - node_locations[n1]
            directions.append(difference)
            midpoints.append(node_locations[n1] + difference/2)
        directions = np.array(directions)
        midpoints = np.array(midpoints)

        arrow_rgba = mu.color_to_rgba(arrow_color,arrow_alpha)

        ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                        directions[:,0],directions[:,1],directions[:,2],
                        size=arrow_size,
                        color = arrow_rgba)
        
        
        if len(reciprocal_edges) > 0:
            #getting the midpoints then the directions of arrows for the quiver
            midpoints = []
            directions = []

            for n1,n2 in reciprocal_edges:
                #going to skip reciprocal connections because will do them later
                difference = node_locations[n2] - node_locations[n1]
                directions.append(difference)
                midpoints.append(node_locations[n1] + difference/2)
            directions = np.array(directions)
            midpoints = np.array(midpoints)

            arrow_rgba = mu.color_to_rgba(arrow_color_reciprocal,
                                          arrow_alpha_reciprocal)
            
            ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                            directions[:,0],directions[:,1],directions[:,2],
                            size=arrow_size_reciprocal,
                            color = arrow_rgba)

            
    if starting_node:
        starting_node_num = xu.get_starting_node(curr_concept_network,only_one=False)
        starting_node_num_coord = [curr_concept_network.nodes[k]["data"].mesh_center for k in starting_node_num]
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        for k in starting_node_num_coord:
#             print(f"mu.color_to_rgba(starting_node_color,starting_node_alpha) = {mu.color_to_rgba(starting_node_color,starting_node_alpha)}")
#             print(f"[k] = {[k]}")
#             print(f"scatter_size = {node_size}")
            sk.graph_skeleton_and_mesh(
                                       other_scatter=[k],
                                       other_scatter_colors=[mu.color_to_rgba(starting_node_color,starting_node_alpha)],
                                       scatter_size=starting_node_size,
                                       show_at_end=False,
                                       append_figure=True,
                                        flip_y=False
                
                                   )
    
    #print("************ Done plotting the starting nodes *******************")
    #plot all of the data points using the colors
    if type(node_color) != dict:
        color_list = mu.process_non_dict_color_input(node_color)
        #now go through and add the alpha levels to those that don't have it
        color_list_alpha_fixed = mu.apply_alpha_to_color_list(color_list,alpha=node_alpha)
        color_list_correct_size = mu.generate_color_list_no_alpha_change(user_colors=color_list_alpha_fixed,
                                                                         n_colors=len(curr_concept_network.nodes()))
        node_locations_array = [v for v in node_locations.values()]
    else:
        #if dictionary then check that all the color dictionary keys match
        node_names = list(curr_concept_network.nodes())
        if set(list(node_color.keys())) != set(node_names):
            raise Exception(f"The node_color dictionary ({node_color}) did not match the nodes in the concept network ({curr_concept_network})")
        
        #assemble the color list and the 
        color_list_correct_size = [node_color[k] for k in node_names]
        node_locations_array = [node_locations[k] for k in node_names]
        
#     print(f"node_locations = {node_locations}")
#     print(f"\n\nnode_locations_array = {node_locations_array}")
    #print("***** About to do all the other scatter points ***********")
    
    #print(f"Current scatter size = {scatter_size}")
    if print_flag:
        print(f"edge_color = {edge_color} IN SKELETON")
    concept_network_skeleton = nru.convert_concept_network_to_skeleton(curr_concept_network)
    
    plot_ipv_skeleton(concept_network_skeleton,edge_color,flip_y=False)
    sk.graph_skeleton_and_mesh(
                              #other_skeletons=[concept_network_skeleton],
                              #other_skeletons_colors=[edge_color],
                               other_scatter=node_locations_array,
                               other_scatter_colors=color_list_correct_size,
                               scatter_size=node_size,
                               show_at_end=False,
                               append_figure=True,
                                flip_y=False
                              )

    if show_at_end:
        ipv.show()
        
        
        
    
    



def plot_branch_pieces(neuron_network,
                       node_to_branch_dict,
                      background_mesh=None,
                      **kwargs):
    if background_mesh is None:
        background_mesh = trimesh.Trimesh(vertices = np.array([]),
                                         faces = np.array([]))
        
    total_branch_meshes = []
    
    for curr_limb,limb_branches in node_to_branch_dict.items():
        meshes_to_plot = [neuron_network.nodes[curr_limb]["data"].concept_network.nodes[k]["data"].mesh for k in limb_branches]
        total_branch_meshes += meshes_to_plot

    if len(total_branch_meshes) == 0:
        print("**** Warning: There were no branch meshes to visualize *******")
        return
    
    sk.graph_skeleton_and_mesh(main_mesh_verts=background_mesh.vertices,
                              main_mesh_faces=background_mesh.faces,
                              other_meshes=total_branch_meshes,
                              other_meshes_colors="red",
                              **kwargs)
    
    
    
######  Don't think need general configurations because would like for mesh, skeleton and concept_network to have different defaults
#     #the general configurations      
#     configuration_dict=None,
#     limb_branch_dict=None
#     resolution=default_resolution,
#     color_grouping="branch",
#     color="random",
#     color_alpha=default_alpha,
#     soma=False,
#     soma_color="red",
#     soma_alpha=default_alpha,
#     whole_neuron=False,
#     whole_neuron_color="grey",
#     whole_neuron_alpha=default_alpha,
    
    

def plot_ipv_mesh(elephant_mesh_sub,color=[1.,0.,0.,0.2],
                 flip_y=True):
    if elephant_mesh_sub is None:
        return 
    if len(elephant_mesh_sub.vertices) == 0:
        return
    
    if flip_y:
        elephant_mesh_sub = elephant_mesh_sub.copy()
        elephant_mesh_sub.vertices[...,1] = -elephant_mesh_sub.vertices[...,1]
    
    #check if the color is a dictionary
    if type(color) == dict:
        #get the type of values stored in there
        labels = list(color.items())
        
        #if the labels were stored as just numbers/decimals
        if type(labels[0]) == int or type(labels[0]) == float:
            #get all of the possible labels
            unique_labels = np.unique(labels)
            #get random colors for all of the labels
            colors_list =  mu.generate_color_list(n_colors)
            for lab,curr_color in zip(unique_labels,colors_list):
                #find the faces that correspond to that label
                faces_to_keep = [k for k,v in color.items() if v == lab]
                #draw the mesh with that color
                curr_mesh = elephant_mesh_sub.submesh([faces_to_keep],append=True)
                
                mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                               elephant_mesh_sub.vertices[:,1],
                               elephant_mesh_sub.vertices[:,2],
                               triangles=elephant_mesh_sub.faces)
                mesh4.color = curr_color
                mesh4.material.transparent = True
    else:          
        mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                                   elephant_mesh_sub.vertices[:,1],
                                   elephant_mesh_sub.vertices[:,2],
                                   triangles=elephant_mesh_sub.faces)

        mesh4.color = color
        mesh4.material.transparent = True
        
def plot_ipv_skeleton(edge_coordinates,color=[0,0.,1,1],
                     flip_y=True):
    if len(edge_coordinates) == 0:
        #print("Edge coordinates in plot_ipv_skeleton were of 0 length so returning")
        return []
    
    if flip_y:
        edge_coordinates = edge_coordinates.copy()
        edge_coordinates[...,1] = -edge_coordinates[...,1] 
    
    unique_skeleton_verts_final,edges_final = sk.convert_skeleton_to_nodes_edges_optimized(edge_coordinates)
    mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                            unique_skeleton_verts_final[:,1], 
                            unique_skeleton_verts_final[:,2], 
                            lines=edges_final)
    #print(f"color in ipv_skeleton = {color}")
    mesh2.color = color 
    mesh2.material.transparent = True
    
    #print(f"Color in skeleton ipv plot local = {color}")
    
    if flip_y:
        unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]

    return unique_skeleton_verts_final

def plot_ipv_scatter(scatter_points,scatter_color=[1.,0.,0.,0.5],
                    scatter_size=0.4,
                    flip_y=True):
    scatter_points = np.array(scatter_points).reshape(-1,3).astype("float")
    if flip_y:
        scatter_points = scatter_points.copy()
        scatter_points[...,1] = -scatter_points[...,1]
        
    if len(scatter_points) <= 0:
        print("No scatter points to plot")
        return

    mesh_5 = ipv.scatter(
            scatter_points[:,0], 
            scatter_points[:,1],
            scatter_points[:,2], 
            size=scatter_size, 
            color=scatter_color,
            marker="sphere")
    mesh_5.material.transparent = True    


current_module = sys.modules[__name__]


def visualize_neuron(
    #the neuron we want to visualize
    input_neuron,
    
    #the categories that will be visualized
    visualize_type=["mesh","skeleton"],
    limb_branch_dict=dict(L0=[]),
    #limb_branch_dict=dict(L0=[]),
    
    #for the mesh type:
    mesh_configuration_dict=dict(),
    mesh_limb_branch_dict=None,
    mesh_resolution="branch",
    mesh_color_grouping="branch",
    mesh_color="random",
    mesh_fill_color="brown",
    mesh_color_alpha=0.2,
    mesh_soma=True,
    mesh_soma_color="red",
    mesh_soma_alpha=0.2,
    mesh_whole_neuron=False,
    mesh_whole_neuron_color="green",
    mesh_whole_neuron_alpha=0.2,
    subtract_from_main_mesh=True,
    
    mesh_spines = False,
    mesh_spines_color = "red",
    mesh_spines_alpha = 0.8,
    
    mesh_boutons = False,
    mesh_boutons_color = "aqua",
    mesh_boutons_alpha = 0.8,
    
    mesh_web = False,
    mesh_web_color = "pink",
    mesh_web_alpha = 0.8,
            
    
    #for the skeleton type:
    skeleton_configuration_dict=dict(),
    skeleton_limb_branch_dict=None,
    skeleton_resolution="branch",
    skeleton_color_grouping="branch",
    skeleton_color="random",
    skeleton_color_alpha=1,
    skeleton_soma=True,
    skeleton_fill_color = "green",
    skeleton_soma_color="red",
    skeleton_soma_alpha=1,
    skeleton_whole_neuron=False,
    skeleton_whole_neuron_color="blue",
    skeleton_whole_neuron_alpha=1,
    
    #for concept_network 
    network_configuration_dict=dict(),
    network_limb_branch_dict=None,
    network_resolution="branch",
    network_color_grouping="branch",
    network_color="random",
    network_color_alpha=0.5,
    network_soma=True,
    network_fill_color = "brown",
    network_soma_color="red",
    network_soma_alpha=0.5,
    network_whole_neuron=False,
    network_whole_neuron_color="black",
    network_whole_neuron_alpha=0.5,
    network_whole_neuron_node_size=0.15,
    
    # ------ specific arguments for the concept_network -----
    network_directional=True,
    limb_to_starting_soma="all",
    
    edge_color = "black",
    node_size = 0.15,
    
    starting_node=True,
    starting_node_size=0.3,
    starting_node_color= "pink",
    starting_node_alpha=0.5,
    
    arrow_color = "brown",
    arrow_alpha = 0.8,
    arrow_size = 0.3,
    
    arrow_color_reciprocal = "pink",#"brown",
    arrow_alpha_reciprocal = 1,#0.8,
    arrow_size_reciprocal = 0.7,#0.3,
    
    # arguments for plotting other meshes associated with neuron #
    
    inside_pieces = False,
    inside_pieces_color = "red",
    inside_pieces_alpha = 1,
    
    insignificant_limbs = False,
    insignificant_limbs_color = "red",
    insignificant_limbs_alpha = 1,
    
    non_soma_touching_meshes = False, #whether to graph the inside pieces
    non_soma_touching_meshes_color = "red",
    non_soma_touching_meshes_alpha = 1,
    
    
    
    # arguments for how to display/save ipyvolume fig
    buffer=1000,
    axis_box_off=True,
    html_path="",
    show_at_end=True,
    append_figure=False,
    
    # arguments that will help with random colorization:
    colors_to_omit= [],
    
    #whether to return the color dictionary in order to help
    #locate certain colors
    return_color_dict = False, #if return this then can use directly with plot_color_dict to visualize the colors of certain branches
    
    
    print_flag = False,
    print_time = False,
    flip_y=True,
    
    #arguments for scatter
    scatters=[],
    scatters_colors=[],
    scatter_size=0.3,
    main_scatter_color = "red",
    
    soma_border_vertices = False,
    soma_border_vertices_size = 0.3,
    soma_border_vertices_color="random",
    
    verbose=True,
    subtract_glia = True,
    
    zoom_coordinate=None,
    zoom_radius = None,
    zoom_radius_xyz = None,
    
    # --- 6/9 parameters for synapses --- #
    total_synapses = False,
    total_synapses_size = None,
    
    
    limb_branch_synapses = False,
    limb_branch_synapse_type = "synapses",
    distance_errored_synapses = False,
    mesh_errored_synapses = False,
    soma_synapses = False,
    
    limb_branch_size = None,
    distance_errored_size = None,
    mesh_errored_size = None,
    soma_size = None,

    
    
    ):
    
    """
    ** tried to optimize for speed but did not find anything that really sped it up**
    ipv.serialize.performance = 0/1/2 was the only thing I really found but this didn't help
    (most of the time is spent on compiling the visualization and not on the python,
    can see this by turning on print_time=True, which only shows about 2 seconds for runtime
    but is really 45 seconds for large mesh)
    
    How to plot the spines:
    nviz.visualize_neuron(uncompressed_neuron,
                      limb_branch_dict = dict(),
                     #mesh_spines=True,
                      mesh_whole_neuron=True,
                      mesh_whole_neuron_alpha = 0.1,
                      
                    mesh_spines = True,
                    mesh_spines_color = "red",
                    mesh_spines_alpha = 0.8,
                      
                     )
    Examples: 
    How to do a concept_network graphing: 
    nviz=reload(nviz)
    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_whole_neuron=True,
                                                network_whole_neuron_node_size=1,
                                                network_whole_neuron_alpha=0.2,
                                                network_directional=True,

                                                #network_soma=["S1","S0"],
                                                #network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1=[11,15]),
                                                network_color=["pink","green"],
                                                network_color_alpha=1,
                                                node_size = 5,
                                                arrow_size = 1,
                                                return_color_dict=True)
    
    
    
    Cool facts: 
    1) Can specify the soma names and not just say true so will
    only do certain somas
    
    Ex: 
    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                     visualize_type=["network"],
                     network_resolution="limb",
                                            network_soma=["S0"],
                    network_soma_color = ["red","black"],       
                     limb_branch_dict=dict(L1=[],L2=[]),
                     node_size = 5,
                     return_color_dict=True)
    
    2) Can put "all" for limb_branch_dict or can put "all"
    for the lists of each branch
    
    3) Can specify the somas you want to graph and their colors
    by sending lists
    
    
    Ex 3: How to specifically color just one branch and fill color the rest of limb
    limb_idx = "L0"
    ex_limb = uncompressed_neuron.concept_network.nodes[limb_idx]["data"]
    branch_idx = 3
    ex_branch = ex_limb.concept_network.nodes[2]["data"]

    nviz.visualize_neuron(double_neuron_processed,
                          visualize_type=["mesh"],
                         limb_branch_dict=dict(L0="all"),
                          mesh_color=dict(L1={3:"red"}),
                          mesh_fill_color="green"

                         )
    
    
    """
    reload(nviz)
    if total_synapses_size is None:
        total_synapses_size = syu.default_synapse_size
    if limb_branch_size is None:
        limb_branch_size = syu.default_synapse_size
    if distance_errored_size is None:
        distance_errored_size = syu.default_synapse_size
    if mesh_errored_size is None:
        mesh_errored_size = syu.default_synapse_size
    if soma_size is None:
        soma_size = syu.default_synapse_size
    
    
    if limb_branch_dict == "axon":
        ax_name = input_neuron.axon_limb_name
        if ax_name is None:
            raise Exception("No axon to plot")
        limb_branch_dict = {ax_name:"all"}
        
    if limb_branch_dict is None:
        limb_branch_dict=dict(L0=[])
        
    
    
    
    total_time = time.time()
    #print(f"print_time = {print_time}")
    
    
    current_neuron = neuron.Neuron(input_neuron)
    
    local_time = time.time()
    #To uncomment for full graphing
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))
        ipvu.hide_legend_panel()

    if print_time:
        print(f"Time for setting up figure = {time.time() - local_time}")
        local_time = time.time()
        
    main_vertices = []
    
    #do the mesh visualization type
    for viz_type in visualize_type:
        local_time = time.time()
        if verbose:
            print(f"\n Working on visualization type: {viz_type}")
        if viz_type=="mesh":
            current_type = "mesh"
            
            
            #configuring the parameters
            configuration_dict = mesh_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",mesh_limb_branch_dict)
            configuration_dict.setdefault("resolution",mesh_resolution)
            configuration_dict.setdefault("color_grouping",mesh_color_grouping)
            configuration_dict.setdefault("color",mesh_color)
            configuration_dict.setdefault("fill_color",mesh_fill_color)
            configuration_dict.setdefault("color_alpha",mesh_color_alpha)
            configuration_dict.setdefault("soma",mesh_soma)
            configuration_dict.setdefault("soma_color",mesh_soma_color)
            configuration_dict.setdefault("soma_alpha",mesh_soma_alpha)
            configuration_dict.setdefault("whole_neuron",mesh_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",mesh_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",mesh_whole_neuron_alpha)
            
            configuration_dict.setdefault("mesh_spines",mesh_spines)
            configuration_dict.setdefault("mesh_spines_color",mesh_spines_color)
            configuration_dict.setdefault("mesh_spines_alpha",mesh_spines_alpha)
            
            configuration_dict.setdefault("mesh_boutons",mesh_boutons)
            configuration_dict.setdefault("mesh_boutons_color",mesh_boutons_color)
            configuration_dict.setdefault("mesh_boutons_alpha",mesh_boutons_alpha)
            
            configuration_dict.setdefault("mesh_web",mesh_web)
            configuration_dict.setdefault("mesh_web_color",mesh_web_color)
            configuration_dict.setdefault("mesh_web_alpha",mesh_web_alpha)
            
        elif viz_type == "skeleton":
            current_type="skeleton"
            
            #configuring the parameters
            configuration_dict = skeleton_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",skeleton_limb_branch_dict)
            configuration_dict.setdefault("resolution",skeleton_resolution)
            configuration_dict.setdefault("color_grouping",skeleton_color_grouping)
            configuration_dict.setdefault("color",skeleton_color)
            configuration_dict.setdefault("fill_color",skeleton_fill_color)
            configuration_dict.setdefault("color_alpha",skeleton_color_alpha)
            configuration_dict.setdefault("soma",skeleton_soma)
            configuration_dict.setdefault("soma_color",skeleton_soma_color)
            configuration_dict.setdefault("soma_alpha",skeleton_soma_alpha)
            configuration_dict.setdefault("whole_neuron",skeleton_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",skeleton_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",skeleton_whole_neuron_alpha)
            
        elif viz_type == "network":
            current_type="mesh_center"
            
            #configuring the parameters
            configuration_dict = network_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",network_limb_branch_dict)
            configuration_dict.setdefault("resolution",network_resolution)
            configuration_dict.setdefault("color_grouping",network_color_grouping)
            configuration_dict.setdefault("color",network_color)
            configuration_dict.setdefault("fill_color",network_fill_color)
            configuration_dict.setdefault("color_alpha",network_color_alpha)
            configuration_dict.setdefault("soma",network_soma)
            configuration_dict.setdefault("soma_color",network_soma_color)
            configuration_dict.setdefault("soma_alpha",network_soma_alpha)
            configuration_dict.setdefault("whole_neuron",network_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",network_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",network_whole_neuron_alpha)
            configuration_dict.setdefault("whole_neuron_node_size",network_whole_neuron_node_size)
            
            # ------ specific arguments for the concept_network -----
            configuration_dict.setdefault("network_directional",network_directional)
            configuration_dict.setdefault("limb_to_starting_soma",limb_to_starting_soma)
            
            configuration_dict.setdefault("node_size",node_size)
            configuration_dict.setdefault("edge_color",edge_color)
            
            
            configuration_dict.setdefault("starting_node",starting_node)
            configuration_dict.setdefault("starting_node_size",starting_node_size)
            configuration_dict.setdefault("starting_node_color",starting_node_color)
            configuration_dict.setdefault("starting_node_alpha",starting_node_alpha)
            
            configuration_dict.setdefault("arrow_color",arrow_color)
            configuration_dict.setdefault("arrow_alpha",arrow_alpha)
            configuration_dict.setdefault("arrow_size",arrow_size)
            
            configuration_dict.setdefault("arrow_color_reciprocal",arrow_color_reciprocal)
            configuration_dict.setdefault("arrow_alpha_reciprocal",arrow_alpha_reciprocal)
            configuration_dict.setdefault("arrow_size_reciprocal",arrow_size_reciprocal)
            
            
            
        else:
            raise Exception(f"Recieved invalid visualization type: {viz_type}")
        
        
        #sets the limb branch dict specially  (uses overall one if none assigned)

        #print(f"current_type = {current_type}")
        
        #handle if the limb_branch_dict is "all"
        #print(f'configuration_dict["limb_branch_dict"] = {configuration_dict["limb_branch_dict"]}')
        if configuration_dict["limb_branch_dict"] is None:
            #print("limb_branch_dict was None")
            configuration_dict["limb_branch_dict"] = limb_branch_dict
        
        if configuration_dict["limb_branch_dict"] == "all":
            configuration_dict["limb_branch_dict"] = dict([(k,"all") for k in current_neuron.get_limb_node_names()])
            
        configuration_dict["limb_branch_dict"] = {k:v for k,v in 
            configuration_dict["limb_branch_dict"].items() 
            if k in current_neuron.get_limb_node_names()}
        #print(f'configuration_dict["limb_branch_dict"] = {configuration_dict["limb_branch_dict"]}')
            
        #print(f'configuration_dict["limb_branch_dict"] = {configuration_dict["limb_branch_dict"]}')
        
        if print_time:
            print(f"Extracting Dictionary = {time.time() - local_time}")
            local_time = time.time()
        
        #------------------------- Done with collecting the parameters ------------------------
        
        if print_flag:
            for k,v in configuration_dict.items():
                print(k,v)
            
        #get the list of items specific
        limbs_to_plot = sorted(list(configuration_dict["limb_branch_dict"].keys()))
        plot_items = []
        plot_items_order = []
        if configuration_dict["resolution"] == "limb":
            if current_type == "mesh":
                plot_items = [nru.limb_mesh_from_branches(current_neuron[li]) for li in limbs_to_plot]
            else:
                plot_items = [getattr(current_neuron[li],current_type) for li in limbs_to_plot]
            plot_items_order = [[li] for li in limbs_to_plot]
        elif configuration_dict["resolution"] == "branch":
            
            for li in limbs_to_plot:
                curr_limb_obj = current_neuron.concept_network.nodes[li]["data"]
                
                if "empty" in curr_limb_obj.labels or curr_limb_obj.concept_network is None:
                    continue
                    
                #handle if "all" is the key
                if ((configuration_dict["limb_branch_dict"][li] == "all") or 
                   ("all" in configuration_dict["limb_branch_dict"][li])):
                    #gather all of the branches: 
                    plot_items += [getattr(curr_limb_obj.concept_network.nodes[k]["data"],current_type) for k in sorted(curr_limb_obj.concept_network.nodes())]
                    plot_items_order += [[li,k] for k in sorted(curr_limb_obj.concept_network.nodes())]
                else:
                    for branch_idx in sorted(configuration_dict["limb_branch_dict"][li]):
                        plot_items.append(getattr(curr_limb_obj.concept_network.nodes[branch_idx]["data"],current_type))
                        plot_items_order.append([li,branch_idx])
        else:
            raise Exception("The resolution specified was neither branch nore limb")
            
        
        #getting the min and max of the plot items to set the zoom later (could be empty)
        
        
        if print_time:
            print(f"Creating Plot Items = {time.time() - local_time}")
            local_time = time.time()
       
        
#         print(f"plot_items_order= {plot_items_order}")
#         print(f"plot_items= {plot_items}")
        
     
        # Now need to build the colors dictionary
        """
        Pseudocode:
        if color is a dictionary then that is perfect and what we want:
        
        -if color grouping or resolution at limb then this dictionary should be limb --> color
        -if resolution adn color grouping aat branch should be limb --> branch --> color
        
        if not then generate a dictionary like that where 
        a) if color is random: generate list of random colors for length needed and then store in dict
        b) if given one color or list of colors:
        - make sure it is a list of colors
        - convert all of the strings into rgb colors
        - for all the colors in the list that do not have an alpha value fill it in with the default alpha
        - repeat the list enough times to give every item a color
        - assign the colors to the limb or limb--> branch dictionary

        """
        
        #need to do preprocessing of colors if not a dictionary
        if type(configuration_dict["color"]) != dict:
            color_list = mu.process_non_dict_color_input(configuration_dict["color"])
        else:
            #if there was a dictionary given then compile a color list and fill everything not specified with mesh_fill_color
            color_list = []
            for dict_keys in plot_items_order:
                #print(f"dict_keys = {dict_keys}")
                first_key = dict_keys[0]
                if first_key not in configuration_dict["color"]:
                    color_list.append(mu.color_to_rgb(configuration_dict["fill_color"]))
                    continue
                if len(dict_keys) == 1:
                    color_list.append(mu.color_to_rgb(configuration_dict["color"][first_key]))
                elif len(dict_keys) == 2:
                    second_key = dict_keys[1]
                    if second_key not in configuration_dict["color"][first_key]:
                        color_list.append(mu.color_to_rgb(configuration_dict["fill_color"]))
                        continue
                    else:
                        color_list.append(mu.color_to_rgb(configuration_dict["color"][first_key][second_key]))
                else:
                    raise Exception(f"plot_items_order item is greater than size 2: {dict_keys}")
        
        if print_flag:
            print(f"color_list = {color_list}")
            

            
        #now go through and add the alpha levels to those that don't have it
        color_list_alpha_fixed = mu.apply_alpha_to_color_list(color_list,alpha=configuration_dict["color_alpha"])
        
        color_list_correct_size = mu.generate_color_list_no_alpha_change(user_colors=color_list_alpha_fixed,
                                                                         n_colors=len(plot_items),
                                                                        colors_to_omit=colors_to_omit)
        
        if print_flag:
            print(f"color_list_correct_size = {color_list_correct_size}")
            print(f"plot_items = {plot_items}")
            print(f"plot_items_order = {plot_items_order}")
            
            
        if print_time:
            print(f"Creating Colors list = {time.time() - local_time}")
            local_time = time.time()
        #------at this point have a list of colors for all the things to plot -------
        
        
        
        #4) If soma is requested then get the some items
        
        
        soma_names = current_neuron.get_soma_node_names()
        if nu.is_array_like(configuration_dict["soma"]):
            soma_names = [k for k in soma_names if k in configuration_dict["soma"]]
            
        if viz_type == "mesh":
            local_time = time.time()
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                min_max_vertices = np.array([[np.min(k.vertices,axis=0),np.max(k.vertices,axis=0)] for k in  plot_items]).reshape(-1,3)
                min_vertices = np.min(min_max_vertices,axis=0)
                max_vertices = np.max(min_max_vertices,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Collecting vertices for mesh = {time.time() - local_time}")
                local_time = time.time()
                
            
            #Can plot the meshes now
            for curr_mesh,curr_mesh_color in zip(plot_items,color_list_correct_size):
                plot_ipv_mesh(curr_mesh,color=curr_mesh_color,flip_y=flip_y)
            
            if print_time:
                print(f"Plotting mesh pieces= {time.time() - local_time}")
                local_time = time.time()
                
            #Plot the soma if asked for it
            if configuration_dict["soma"]:
                """
                Pseudocode: 
                1) Get the soma meshes
                2) for the color specified: 
                - if string --> convert to rgba
                - if numpy array --> 
                
                configuration_dict.setdefault("soma",mesh_soma)
                configuration_dict.setdefault("soma_color",mesh_soma_color)
                configuration_dict.setdefault("soma_alpha",mesh_soma_alpha)

                """
                
                soma_meshes = [current_neuron.concept_network.nodes[k]["data"].mesh for k in soma_names]
                
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_meshes))
                soma_names,soma_colors_list_alpha_fixed_size
                for curr_soma_mesh,curr_soma_color in zip(soma_meshes,soma_colors_list_alpha_fixed_size):
                    plot_ipv_mesh(curr_soma_mesh,color=curr_soma_color,flip_y=flip_y)
                    main_vertices.append(curr_soma_mesh.vertices)
                
                if print_time:
                    print(f"plotting mesh somas= {time.time() - local_time}")
                    local_time = time.time()
                    
            #will add the background mesh if requested
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                if subtract_glia:
                    if (current_neuron.glia_faces is not None) and (len(current_neuron.glia_faces) > 0):
                        whole_mesh = current_neuron.mesh.submesh([np.delete(np.arange(len(current_neuron.mesh.faces)),
                                                                                 current_neuron.glia_faces)],append=True,repair=False)
                    else:
                        whole_mesh = current_neuron.mesh
                else:
                    whole_mesh = current_neuron.mesh
                    
                    
                # Will do the erroring of the mesh
                if (subtract_from_main_mesh and (len(plot_items)>0)):
                    main_mesh_to_plot = tu.subtract_mesh(original_mesh=whole_mesh,
                                              subtract_mesh=plot_items)
                else:
                    main_mesh_to_plot = whole_mesh
                    
                
        
                
                # will do the plotting
                plot_ipv_mesh(main_mesh_to_plot,color=whole_neuron_colors_list_alpha[0],flip_y=flip_y)
                main_vertices.append([np.min(main_mesh_to_plot.vertices,axis=0),
                                      np.max(main_mesh_to_plot.vertices,axis=0)])
                
                if print_time:
                    print(f"Plotting mesh whole neuron = {time.time() - local_time}")
                    local_time = time.time()
                
                
            
#             # plotting the boutons:
#             if configuration_dict["mesh_boutons"]:
#                 bouton_meshes = []
                
#                 for limb_names in current_neuron.neuron_obj.get_limb_names():
#                     curr_limb_obj = current_neuron.concept_network.nodes[limb_names]["data"]
#                     for branch_name in curr_limb_obj.get_branch_names():
#                         if hasattr(curr_limb_obj[branch_name],boutons) and curr_limb_obj[branch_name].boutons is not None :
#                             bouton_meshes += curr_limb_obj[branch_name].boutons
                
#                 boutons_color_list = mu.process_non_dict_color_input(configuration_dict["mesh_boutons_color"])
#                 boutons_color_list_alpha = mu.apply_alpha_to_color_list(boutons_color_list,alpha=configuration_dict["mesh_boutons_alpha"])
                
                
#                 if len(boutons_color_list_alpha) == 1:
#                     combined_bouton_meshes = tu.combine_meshes(bouton_meshes)
#                     plot_ipv_mesh(combined_bouton_meshes,color=boutons_color_list_alpha[0],flip_y=flip_y)
#                     main_vertices.append(combined_bouton_meshes.vertices)
                    
#                 else:
#                     boutons_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(boutons_color_list_alpha,
#                                                                                               n_colors=len(bouton_meshes))


#                     for curr_bouton_mesh,curr_bouton_color in zip(bouton_meshes,boutons_colors_list_alpha_fixed_size):
#                         plot_ipv_mesh(curr_bouton_mesh,color=curr_bouton_color,flip_y=flip_y)
#                         main_vertices.append(curr_bouton_mesh.vertices)
#                 if print_time:
#                     print(f"Plotting mesh boutons= {time.time() - local_time}")
#                     local_time = time.time()
                    
            # -- 4/1 Addition:    
            if configuration_dict["mesh_boutons"]:
                bouton_verts = plot_branch_mesh_attribute(current_neuron,
                              mesh_attribute="boutons",
                              mesh_color=configuration_dict["mesh_boutons_color"],
                              mesh_alpha=configuration_dict["mesh_boutons_alpha"],
                               return_vertices = True,
                               flip_y=flip_y,
                               plot_at_end=False,
                              verbose = print_time)
                if len(bouton_verts) > 0:
                    main_vertices += bouton_verts
                    
            
            if configuration_dict["mesh_web"]:
                web_verts = plot_branch_mesh_attribute(current_neuron,
                              mesh_attribute="web",
                              mesh_color=configuration_dict["mesh_web_color"],
                              mesh_alpha=configuration_dict["mesh_web_alpha"],
                               return_vertices = True,
                               flip_y=flip_y,
                               plot_at_end=False,
                              verbose = print_time)
                if len(web_verts) > 0:
                    main_vertices += web_verts
                            
                
            #plotting the spines
            if configuration_dict["mesh_spines"]:
                #plotting the spines
                spine_meshes = []
                
                for limb_names in current_neuron.get_limb_node_names():
                    #iterate through all of the branches
                    curr_limb_obj = current_neuron.concept_network.nodes[limb_names]["data"]
                    for branch_name in curr_limb_obj.concept_network.nodes():
                        curr_spines = curr_limb_obj.concept_network.nodes[branch_name]["data"].spines
                        if not curr_spines is None:
                            spine_meshes += curr_spines
                
                spines_color_list = mu.process_non_dict_color_input(configuration_dict["mesh_spines_color"])
                
                
                
                spines_color_list_alpha = mu.apply_alpha_to_color_list(spines_color_list,alpha=configuration_dict["mesh_spines_alpha"])
                #print(f"spines_color_list_alpha = {spines_color_list_alpha}")
                if len(spines_color_list_alpha) == 1:
                    #concatenate the meshes
                    #print("Inside spine meshes combined")
                    combined_spine_meshes = tu.combine_meshes(spine_meshes)
                    plot_ipv_mesh(combined_spine_meshes,color=spines_color_list_alpha[0],flip_y=flip_y)
                    main_vertices.append(combined_spine_meshes.vertices)
                    
                else:
                    spines_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(spines_color_list_alpha,
                                                                                              n_colors=len(spine_meshes))


                    for curr_spine_mesh,curr_spine_color in zip(spine_meshes,spines_colors_list_alpha_fixed_size):
                        plot_ipv_mesh(curr_spine_mesh,color=curr_spine_color,flip_y=flip_y)
                        main_vertices.append(curr_spine_mesh.vertices)
                if print_time:
                    print(f"Plotting mesh spines= {time.time() - local_time}")
                    local_time = time.time()
                
            
        elif viz_type == "skeleton":
            local_time = time.time()
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Gathering vertices for skeleton= {time.time() - local_time}")
                local_time = time.time()
            
            #Can plot the meshes now
            for curr_skeleton,curr_skeleton_color in zip(plot_items,color_list_correct_size):
                plot_ipv_skeleton(curr_skeleton,color=curr_skeleton_color,flip_y=flip_y)
            
            if print_time:
                print(f"Plotting skeleton pieces = {time.time() - local_time}")
                local_time = time.time()
            
            
            if configuration_dict["soma"]:
                    
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_names))
                #get the somas associated with the neurons
                soma_skeletons = [nru.get_soma_skeleton(current_neuron,k) for k in soma_names]
                
                for curr_soma_sk,curr_soma_sk_color in zip(soma_skeletons,soma_colors_list_alpha_fixed_size):
                    sk_vertices = plot_ipv_skeleton(curr_soma_sk,color=curr_soma_sk_color,flip_y=flip_y)
                    main_vertices.append(sk_vertices) #adding the vertices
                
                if print_time:
                    print(f"Plotting skeleton somas = {time.time() - local_time}")
                    local_time = time.time()
            
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                #graph
                sk_vertices = plot_ipv_skeleton(current_neuron.skeleton,color=whole_neuron_colors_list_alpha[0],flip_y=flip_y)
                main_vertices.append([np.min(sk_vertices,axis=0),
                                      np.max(sk_vertices,axis=0)])
                
                if print_time:
                    print(f"Plotting skeleton whole neuron = {time.time() - local_time}")
                    local_time = time.time()
                
                
        elif viz_type == "network":
            local_time = time.time()
            """
            Pseudocode: 
            0) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            1) get the current concept network (limb or branch) based on the resolution
            - if branch level then use the function that assembles
            2) get a list of all the nodes in the plot_items_order and assemble into a dictionary (have to fix the name)
            3) For all the somas to be added, add them to the dictionary of label to color (and add vertices to main vertices)
            4) Use that dictionary to send to the visualize_concept_map function and call the function
            with all the other parameters in the configuration dict
            
            5) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            
            """
            
            
            #0) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Gathering vertices for network = {time.time() - local_time}")
                local_time = time.time()
                
            
            #1) get the current concept network (limb or branch) based on the resolution
            #- if branch level then use the function that assembles
            if configuration_dict["resolution"] == "branch":

                curr_concept_network = nru.whole_neuron_branch_concept_network(current_neuron,
                                                          directional= configuration_dict["network_directional"],
                                                         limb_soma_touch_dictionary = configuration_dict["limb_to_starting_soma"],
                                                         print_flag = False)

                    
                
                
                #2) get a list of all the nodes in the plot_items_order and assemble into a dictionary for colors (have to fix the name)
                item_to_color_dict = dict([(f"{name[0]}_{name[1]}",col) for name,col in zip(plot_items_order,color_list_correct_size)])
            else:
                #2) get a list of all the nodes in the plot_items_order and assemble into a dictionary for colors (have to fix the name)
                curr_concept_network = current_neuron.concept_network
                item_to_color_dict = dict([(f"{name[0]}",col) for name,col in zip(plot_items_order,color_list_correct_size)])
                
            if print_time:
                print(f"Getting whole concept network and colors = {time.time() - local_time}")
                local_time = time.time()
            
            
            #3) For all the somas to be added, add them to the dictionary of label to color
            if soma_names:
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_names))
                
                for s_name,s_color in zip(soma_names,soma_colors_list_alpha_fixed_size):
                    item_to_color_dict[s_name] = s_color
                    main_vertices.append(current_neuron.concept_network.nodes[s_name]["data"].mesh_center)
                    
                if print_time:
                    print(f"Adding soma items to network plotting = {time.time() - local_time}")
                    local_time = time.time()
                    
            #print(f"plot_items_order = {plot_items_order}")
            #print(f"item_to_color_dict = {item_to_color_dict}")
                
            curr_concept_network_subgraph = nx.subgraph(curr_concept_network,list(item_to_color_dict.keys()))
            
            
            
            if print_time:
                print(f"Getting Subgraph of concept network = {time.time() - local_time}")
                local_time = time.time()
            
            #4) Use that dictionary to send to the visualize_concept_map function and call the function
            #with all the other parameters in the configuration dict
            
            visualize_concept_map(curr_concept_network_subgraph,
                            node_color=item_to_color_dict,
                            edge_color=configuration_dict["edge_color"],
                            node_size=configuration_dict["node_size"],

                            starting_node=configuration_dict["starting_node"],
                            starting_node_size = configuration_dict["starting_node_size"],
                            starting_node_color= configuration_dict["starting_node_color"],
                            starting_node_alpha = configuration_dict["starting_node_alpha"],

                            arrow_color = configuration_dict["arrow_color"] ,
                            arrow_alpha = configuration_dict["arrow_alpha"],
                            arrow_size = configuration_dict["arrow_size"],

                            arrow_color_reciprocal = configuration_dict["arrow_color_reciprocal"] ,
                            arrow_alpha_reciprocal = configuration_dict["arrow_alpha_reciprocal"],
                            arrow_size_reciprocal = configuration_dict["arrow_size_reciprocal"],
                          
                            show_at_end=False,
                            append_figure=True)
            
            if print_time:
                print(f"Graphing concept network pieces = {time.time() - local_time}")
                local_time = time.time()
            
            # plot the entire thing if asked for it
            if configuration_dict["whole_neuron"]:
                #compute the new color
                whole_neuron_network_color = mu.color_to_rgba(configuration_dict["whole_neuron_color"],
                                                             configuration_dict["whole_neuron_alpha"])
                
                whole_neuron_network_edge_color = mu.color_to_rgba(configuration_dict["edge_color"],
                                                             configuration_dict["whole_neuron_alpha"])
                print(f"whole_neuron_network_edge_color = {whole_neuron_network_edge_color}")
                
                visualize_concept_map(curr_concept_network,
                            node_color=whole_neuron_network_color,
                            edge_color=whole_neuron_network_edge_color,
                            node_size=configuration_dict["whole_neuron_node_size"],

                            starting_node=configuration_dict["starting_node"],
                            starting_node_size = configuration_dict["starting_node_size"],
                            starting_node_color= configuration_dict["starting_node_color"],
                            starting_node_alpha = configuration_dict["starting_node_alpha"],

                            arrow_color = configuration_dict["arrow_color"] ,
                            arrow_alpha = configuration_dict["arrow_alpha"],
                            arrow_size = configuration_dict["arrow_size"],

                            arrow_color_reciprocal = configuration_dict["arrow_color_reciprocal"] ,
                            arrow_alpha_reciprocal = configuration_dict["arrow_alpha_reciprocal"],
                            arrow_size_reciprocal = configuration_dict["arrow_size_reciprocal"],
                          
                            show_at_end=False,
                            append_figure=True)
                if print_time:
                    print(f"Graphing whole neuron concept network = {time.time() - local_time}")
                    local_time = time.time()
            
        else:
            raise Exception("Invalid viz_type")
        
        
    # -------------- plotting the insignificant meshes, floating meshes and non-significant limbs ----- #
    """
    Pseudocode: for [inside_piece,insignificant_limbs,non_soma_touching_meshes]
    
    1) get whether the argument was True/False or a list
    2) If True or list, assemble the color
    3) for each mesh plot it with the color

    """
    local_time = time.time()
    
    other_mesh_dict = dict(
        inside_pieces=inside_pieces,
        inside_pieces_color=inside_pieces_color,
        inside_pieces_alpha=inside_pieces_alpha,
        
        insignificant_limbs=insignificant_limbs,
        insignificant_limbs_color=insignificant_limbs_color,
        insignificant_limbs_alpha=insignificant_limbs_alpha,
        
        non_soma_touching_meshes=non_soma_touching_meshes,
        non_soma_touching_meshes_color=non_soma_touching_meshes_color,
        non_soma_touching_meshes_alpha=non_soma_touching_meshes_alpha
    
    
    )

    
    other_mesh_types = ["inside_pieces","insignificant_limbs","non_soma_touching_meshes"]
    
    for m_type in other_mesh_types:
        if other_mesh_dict[m_type]:
            if type(other_mesh_dict[m_type]) is bool:
                current_mesh_list = getattr(current_neuron,m_type)
            elif "all" in other_mesh_dict[m_type]:
                current_mesh_list = getattr(current_neuron,m_type)
            else:
                total_mesh_list = getattr(current_neuron,m_type)
                current_mesh_list = [k for i,k in enumerate(total_mesh_list) if i in other_mesh_dict[m_type]]
                
            #get the color
            curr_mesh_colors_list = mu.process_non_dict_color_input(other_mesh_dict[m_type + "_color"])
            curr_mesh_colors_list_alpha = mu.apply_alpha_to_color_list(curr_mesh_colors_list,alpha=other_mesh_dict[m_type + "_alpha"])

            #graph
            for curr_mesh in current_mesh_list:
                plot_ipv_mesh(curr_mesh,color=curr_mesh_colors_list_alpha,flip_y=flip_y)
                main_vertices.append(curr_mesh.vertices)
        if print_time:
            print(f"Plotting mesh pieces of {m_type} = {time.time() - local_time}")
            local_time = time.time()
    
    # ----- doing any extra scatter plotting you may need ---- #
    """
    scatters=[],
    scatters_colors=[],
    scatter_size=0.3,
    main_scatter_color="red"
    
    soma_border_vertices
    soma_border_vertices_color
    """
    
    if soma_border_vertices:
        if len(plot_items_order) > 0:
            if verbose:
                print("working on soma border vertices")
            unique_limb_names = np.unique([k[0] for k in plot_items_order])
            all_soma_verts = [[k["touching_soma_vertices"] for k in 
                                        input_neuron[curr_limb_idx].all_concept_network_data] for curr_limb_idx in unique_limb_names]

            
            new_borders = list(itertools.chain.from_iterable(all_soma_verts))
            if soma_border_vertices_color != "random":
                new_borders_colors = [soma_border_vertices_color]*len(new_borders)
            else:
                new_borders_colors = mu.generate_color_list(n_colors=len(new_borders),alpha_level=1)
            
            for curr_scatter,curr_color in zip(new_borders,new_borders_colors):
                plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                            scatter_size=soma_border_vertices_size,flip_y=flip_y)
                main_vertices.append(curr_scatter)

    if type(scatters_colors) == str:
        scatters_colors = [scatters_colors]
        
    
    if len(scatters) > 0 and len(scatters_colors) == 0:
        scatters_colors = [main_scatter_color]*len(scatters)

#     print(f"scatters = {scatters}")
#     print(f"scatters_colors = {scatters_colors}")
    
    
    for curr_scatter,curr_color in zip(scatters,scatters_colors):
        
#         print(f"curr_scatter = {curr_scatter}")
#         print(f"curr_color = {curr_color}")
#         print(f"scatter_size= {scatter_size}")
        
        plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                    scatter_size=scatter_size,flip_y=flip_y)
        
        main_vertices.append(curr_scatter)   
        
        
        
    # ---------- 6/9: Applying the Synapse Scatters -------#
    

    syu.append_synapses_to_plot(input_neuron,
                               total_synapses = total_synapses,
                                limb_branch_dict = nru.limb_branch_dict_valid(input_neuron,
                                                                              limb_branch_dict),
                                total_synapses_size = total_synapses_size,
                                limb_branch_synapses = limb_branch_synapses,
                                limb_branch_synapse_type = limb_branch_synapse_type,
                                distance_errored_synapses = distance_errored_synapses,
                                mesh_errored_synapses = mesh_errored_synapses,
                                soma_synapses = soma_synapses,
                                limb_branch_size = limb_branch_size,
                                distance_errored_size = distance_errored_size,
                                mesh_errored_size = mesh_errored_size,
                                soma_size = soma_size,)
    

        
    #To uncomment for full graphing
    
    #create the main mesh vertices for setting the bounding box
    if len(main_vertices) == 0:
        raise Exception("No vertices plotted in the entire function")
    elif len(main_vertices) == 1:
        main_vertices = main_vertices[0]
    else:
        #get rid of all empty ones
        main_vertices = np.vstack([np.array(k).reshape(-1,3) for k in main_vertices if len(k)>0])
    
    if len(main_vertices) == 0:
        raise Exception("No vertices plotted in the entire function (after took out empty vertices)")
    
    main_vertices = np.array(main_vertices).reshape(-1,3)
    
    if flip_y:
        main_vertices = main_vertices.copy()
        main_vertices[...,1] = -main_vertices[...,1]
        
    volume_max = np.max(main_vertices.reshape(-1,3),axis=0)
    volume_min = np.min(main_vertices.reshape(-1,3),axis=0)
    
    if print_time:
        print(f"Getting volume min/max = {time.time() - local_time}")
        local_time = time.time()
        
    #setting the min/max of the plots
    ranges = volume_max - volume_min
    index = [0,1,2]
    max_index = np.argmax(ranges)
    min_limits = [0,0,0]
    max_limits = [0,0,0]


    for i in index:
        if i == max_index:
            min_limits[i] = volume_min[i] - buffer
            max_limits[i] = volume_max[i] + buffer 
            continue
        else:
            difference = ranges[max_index] - ranges[i]
            min_limits[i] = volume_min[i] - difference/2  - buffer
            max_limits[i] = volume_max[i] + difference/2 + buffer
    
    if print_time:
        print(f"calculating max limits = {time.time() - local_time}")
        local_time = time.time()
    
    ipv.xlim(min_limits[0],max_limits[0])
    ipv.ylim(min_limits[1],max_limits[1])
    ipv.zlim(min_limits[2],max_limits[2])
    
    if print_time:
        print(f"setting ipyvolume max limits = {time.time() - local_time}")
        local_time = time.time()
    
    ipv.style.set_style_light()
    if axis_box_off:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
    
    if print_time:
        print(f"Setting axis and box on/off = {time.time() - local_time}")
        local_time = time.time()
    
    if show_at_end:
        ipv.show()
    
    if print_time:
        print(f"ipv.show= {time.time() - local_time}")
        local_time = time.time()
    
    if html_path != "":
        ipv.pylab.save(html_path)
    
    if print_time:
        print(f"saving html = {time.time() - local_time}")
        local_time = time.time()
    
        


    
    if return_color_dict:
        #build the color dictionary
        if len(plot_items_order) == 0 or len(color_list_correct_size)==0:
            print("No color dictionary to return because plot_items_order or color_list_correct_size empty")
            if print_time:
                print(f"Total time for run = {time.time() - total_time}")
            return dict()
        
        if len(plot_items_order[0]) == 1:
            color_dict_to_return = dict([(k[0],v) for k,v in zip(plot_items_order,color_list_correct_size)])
        elif len(plot_items_order[0]) == 2:
            color_dict_to_return = dict([(f"{k[0]}_{k[1]}",v) for k,v in zip(plot_items_order,color_list_correct_size)])
        else:
            raise Exception("Length of first element in plot_items order is greater than 2 elements")
        
        #whether to add soma mappings to the list of colors:
        #soma_names, soma_colors_list_alpha_fixed_size
        try: 
            color_dict_to_return_soma = dict([(k,v) for k,v in zip(soma_names,soma_colors_list_alpha_fixed_size)])
            color_dict_to_return.update(color_dict_to_return_soma)
        except:
            pass
        
        if print_time:
            print(f"Preparing color dictionary = {time.time() - local_time}")
            local_time = time.time()
        if print_time:
            print(f"Total time for run = {time.time() - total_time}")
        
        return color_dict_to_return
        
        
    if zoom_coordinate is not None:
        nviz.set_zoom(zoom_coordinate,
                          radius = zoom_radius,
                          radius_xyz = zoom_radius_xyz)
        
    
    ipvu.hide_legend_panel()
    
    if print_time:
        print(f"Total time for run = {time.time() - total_time}")
    return


def plot_spines(current_neuron,
                mesh_whole_neuron_alpha=0.1,
                mesh_whole_neuron_color="green",
                mesh_spines_alpha=0.8,
                spine_color="aqua",
                flip_y=True,
               **kwargs):
    visualize_neuron(current_neuron,
                          limb_branch_dict = dict(),
                          mesh_whole_neuron=True,
                          mesh_whole_neuron_alpha = mesh_whole_neuron_alpha,
                            mesh_whole_neuron_color=mesh_whole_neuron_color,
                        mesh_spines = True,
                        mesh_spines_color = spine_color,
                        mesh_spines_alpha = mesh_spines_alpha,
                     flip_y=flip_y,
                     **kwargs

                         )
    
def plot_boutons(current_neuron,
                mesh_whole_neuron_alpha=0.1,
                mesh_whole_neuron_color="green",
                boutons_color="red",
                mesh_boutons_alpha=0.8,
                flip_y=True,
                 plot_web = False,
                 web_color = "aqua",
                 mesh_web_alpha = 0.8,
               **kwargs):
    visualize_neuron(current_neuron,
                          limb_branch_dict = dict(),
                          mesh_whole_neuron=True,
                          mesh_whole_neuron_alpha = mesh_whole_neuron_alpha,
                            mesh_whole_neuron_color=mesh_whole_neuron_color,
                        mesh_boutons = True,
                        mesh_boutons_color = boutons_color,
                        mesh_boutons_alpha = mesh_boutons_alpha,
                         mesh_web = plot_web,
                        mesh_web_color = web_color,
                        mesh_web_alpha = mesh_web_alpha,
                     flip_y=flip_y,
                     **kwargs

                         )
# -------  9/24: Wrapper for the sk.graph function that is nicer to interface with ----#
"""
def graph_skeleton_and_mesh(main_mesh_verts=[],
                            main_mesh_faces=[],
                            unique_skeleton_verts_final=[],
                            edges_final=[],
                            edge_coordinates=[],
                            other_meshes=[],
                            other_meshes_colors =  [],
                            mesh_alpha=0.2,
                            other_meshes_face_components = [],
                            other_skeletons = [],
                            other_skeletons_colors =  [],
                            return_other_colors = False,
                            main_mesh_color = [0.,1.,0.,0.2],
                            main_skeleton_color = [0,0.,1,1],
                            main_mesh_face_coloring = [],
                            other_scatter=[],
                            scatter_size = 0.3,
                            other_scatter_colors=[],
                            main_scatter_color=[1.,0.,0.,0.5],
                            buffer=1000,
                           axis_box_off=True,
                           html_path="",
                           show_at_end=True,
                           append_figure=False):
                           
things that need to be changed:
1) main_mesh combined
2) edge_coordinates is just the main_skeleton
other_scatter --> scatters
3) change all the other_[]_colors names


*if other inputs aren't list then make them list
                           
"""

'''
def plot_objects(main_mesh=None,
                 main_skeleton=None,
                 main_mesh_color = [0.,1.,0.,0.2],
                 main_mesh_alpha = None,
                 
                main_skeleton_color = [0,0.,1,1],
                meshes=[],
                meshes_colors =  [],
                mesh_alpha=0.2,
                            
                skeletons = [],
                skeletons_colors =  [],
                            
                scatters=[],
                scatters_colors=[],
                scatter_size = 0.3,
                main_scatter_color="red",#[1.,0.,0.,0.5],
                scatter_with_widgets = True,
                 
                buffer=0,#1000,
                axis_box_off=True,
                html_path="",
                show_at_end=True,
                append_figure=False,
                flip_y=True,
                
                subtract_from_main_mesh=True,
                set_zoom = True, #used for the skeleton graph
                zoom_coordinate=None,
                zoom_radius = None,
                zoom_radius_xyz = None,
                adaptive_min_max_limits = True):
    #from neurd import neuron_visualizations as nviz
    #nviz = reload(nviz)
    
    if (main_mesh is None 
        and main_skeleton is None 
        and len(meshes) == 0
        and len(skeletons) == 0
        and len(scatters) == 0):
        print("Nothing to plot so returning")
        return 
    
    
        
    if main_skeleton is None:
        edge_coordinates = []
    else:
        edge_coordinates=main_skeleton
        
        
    convert_to_list_vars = [meshes,meshes_colors,skeletons,
                            skeletons_colors,scatters,scatters_colors,scatter_size]
    
    def convert_to_list(curr_item):
        if type(curr_item) != list:
            if nu.is_array_like(curr_item):
                return list(curr_item)
            else:
                return [curr_item]
        else:
            return curr_item
    
    meshes =  convert_to_list(meshes)
    meshes_colors =  convert_to_list(meshes_colors)
    skeletons =  convert_to_list(skeletons)
    skeletons_colors =  convert_to_list(skeletons_colors)
    scatters =  convert_to_list(scatters)
    scatters_colors =  convert_to_list(scatters_colors)
    
    # --- 6/10: Making sure all scatters are numpy arrays ---
    scatters = [np.array(s).reshape(-1,3) for s in scatters]
    
    
    if (subtract_from_main_mesh and (not main_mesh is None) and (len(meshes)>0)):
        main_mesh = tu.subtract_mesh(original_mesh=main_mesh,
                                  subtract_mesh=meshes,exact_match=False)
    
    
    if main_mesh is None or nu.is_array_like(main_mesh):
        main_mesh_verts = []
        main_mesh_faces= []
    else:
        main_mesh_verts = main_mesh.vertices
        main_mesh_faces= main_mesh.faces
        
    if main_mesh_alpha is None:
        main_mesh_alpha = 0.2
    else: 
        if nu.is_array_like(main_mesh_color):
            main_mesh_color = list(main_mesh_color)
            if len(main_mesh_color) == 4:
                main_mesh_color[3] = main_mesh_alpha
            else:
                main_mesh_color.append(main_mesh_alpha)
            
    
    if type(main_mesh_color) == str:
        main_mesh_color = mu.color_to_rgba(main_mesh_color,alpha=main_mesh_alpha)
    
    
    #print(f"scatters = {scatters}")
    return_value = sk.graph_skeleton_and_mesh(main_mesh_verts=main_mesh_verts,
                           main_mesh_faces=main_mesh_faces,
                           edge_coordinates=edge_coordinates,
                           other_meshes=meshes,
                                      mesh_alpha=mesh_alpha,
                            other_meshes_colors=meshes_colors,
                            other_skeletons=skeletons,
                            other_skeletons_colors=skeletons_colors,
                            other_scatter=scatters,
                            other_scatter_colors=scatters_colors,
                            scatter_size=scatter_size,
                            main_scatter_color=main_scatter_color,
                            scatter_with_widgets=scatter_with_widgets,             
                                              
                            buffer=buffer,
                            axis_box_off=axis_box_off,
                            html_path=html_path,
                            show_at_end=show_at_end,
                            append_figure=append_figure,
                            flip_y=flip_y,
                                      main_mesh_color=main_mesh_color,
                                    main_skeleton_color = main_skeleton_color,
                                              
                            set_zoom=set_zoom,
                            adaptive_min_max_limits=adaptive_min_max_limits,
                           )
    #return 
    
    if zoom_coordinate is not None:
        print(f"Trying to set zoom")
        nviz.set_zoom(zoom_coordinate,
                          radius = zoom_radius,
                          radius_xyz = zoom_radius_xyz)
        
        
    return return_value
'''
            
        
def plot_branch_spines(curr_branch,plot_skeletons=True,**kwargs):
    from mesh_tools import trimesh_utils as tu
    """
    Purpose: To plot a branch with certain spines
    """
    if curr_branch.spines is None:
        curr_spines = [tu.empty_mesh()]
    else:
        curr_spines = curr_branch.spines
        
    shaft_mesh = tu.subtract_mesh(curr_branch.mesh,curr_spines,exact_match=False)
    if plot_skeletons:
        skeletons = [curr_branch.skeleton]
    else:
        skeletons = None
    nviz.plot_objects(main_mesh=shaft_mesh,
                     meshes=curr_spines,
                      skeletons=skeletons,
                      meshes_colors="red",
                     mesh_alpha=1,
                     **kwargs)
    
    
def plot_split_suggestions_per_limb(
    neuron_obj,
    limb_results,
    #red_blue_splits=None,
    scatter_color = "red",
    scatter_alpha = 0.3,
    scatter_size=0.3,
    mesh_color_alpha=0.2,
    add_components_colors=True,
    component_colors = "random",
    ):

    """
    
    
    """
    for curr_limb_idx,path_cut_info in limb_results.items():
        component_colors_cp = copy.copy(component_colors)
        print(f"\n\n-------- Suggestions for Limb {curr_limb_idx}------")
        
        curr_scatters = []
        for path_i in path_cut_info:
            if len(path_i["coordinate_suggestions"])>0:
                curr_scatters.append(np.concatenate(path_i["coordinate_suggestions"]).reshape(-1,3))
                
        if len(curr_scatters) == 0:
            print("\n\n No suggested cuts for this limb!!")
            
            nviz.visualize_neuron(neuron_obj,
                             visualize_type=["mesh","skeleton"],
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color="green",
                             skeleton_color="blue",
                             )
            continue
            
        curr_scatters = np.vstack(curr_scatters)
        scatter_color_list = [mu.color_to_rgba(scatter_color,scatter_alpha)]*len(curr_scatters)
        
        # will create a dictionary that will show all of the disconnected components in different colors
        if add_components_colors:
            curr_limb = pru.cut_limb_network_by_suggestions(copy.deepcopy(neuron_obj[curr_limb_idx]),
                                                      path_cut_info)
            limb_nx = curr_limb.concept_network
            
#             for cut in path_cut_info:
#                 limb_nx.remove_edges_from(cut["edges_to_cut"])
#                 limb_nx.add_edges_from(cut["edges_to_add"])
            
            conn_comp= list(nx.connected_components(limb_nx))
            
            if component_colors_cp == "random":
                component_colors_cp = mu.generate_color_list(n_colors = len(conn_comp))
            elif type(component_colors_cp) == list:
                component_colors_cp = component_colors_cp*np.ceil(len(conn_comp)/len(component_colors_cp)).astype("int")
            else:
                component_colors_cp = ["green"]*len(conn_comp)

            color_dict = dict()
            for groud_ids,c in zip(conn_comp,component_colors_cp):
                for i in groud_ids:
                    color_dict[i] = c
                    
            mesh_component_colors = color_dict
            skeleton_component_colors = color_dict
            #print(f"skeleton_component_colors = {color_dict}")
        else:
            mesh_component_colors = "green"
            skeleton_component_colors = "blue"
            
        #at this point have all of the scatters we want
        nviz.visualize_neuron(neuron_obj,
                             visualize_type=["mesh","skeleton"],
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color={f"L{curr_limb_idx}":mesh_component_colors},
                             mesh_color_alpha=mesh_color_alpha,
                             skeleton_color={f"L{curr_limb_idx}":skeleton_component_colors},
                             scatters=[curr_scatters],
                             scatters_colors=scatter_color_list,
                             scatter_size=scatter_size,
                              mesh_soma_alpha=1,
        )
        

        
        
def visualize_neuron_path(neuron_obj,
                          limb_idx,
                          path,
                          path_mesh_color="red",
                          path_skeleton_color = "red",
                          mesh_fill_color="green",
                          skeleton_fill_color="green",
                         visualize_type=["mesh","skeleton"],
                         scatters=[],
                         scatter_color_list=[],
                         scatter_size=0.3,
                         **kwargs):
    
    curr_limb_idx = nru.limb_idx(limb_idx)
    

    mesh_component_colors = dict([(k,path_mesh_color) for k in path])
    skeleton_component_colors = dict([(k,path_skeleton_color) for k in path])
    
    nviz.visualize_neuron(neuron_obj,
                             visualize_type=visualize_type,
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color={f"L{curr_limb_idx}":mesh_component_colors},
                              mesh_fill_color=mesh_fill_color,
                          
                             skeleton_color={f"L{curr_limb_idx}":skeleton_component_colors},
                          skeleton_fill_color=skeleton_fill_color,
                             scatters=scatters,
                             scatters_colors=scatter_color_list,
                             scatter_size=scatter_size,
                             **kwargs)

def limb_correspondence_plottable(limb_correspondence,
                                  mesh_name="branch_mesh",
                                 combine = False):
    """
    Extracts the meshes and skeleton parts from limb correspondence so can be plotted
    
    """
    keys = list(limb_correspondence.keys())
    if list(limb_correspondence[keys[0]].keys())[0] == 0:
        # then we have a limb correspondence with multiple objects
        meshes=gu.combine_list_of_lists([[k[mesh_name] for k in ki.values()] for ki in limb_correspondence.values()])
        skeletons=gu.combine_list_of_lists([[k["branch_skeleton"] for k in ki.values()] for ki in limb_correspondence.values()])
    else:
        meshes=[k[mesh_name] for k in limb_correspondence.values()]
        skeletons=[k["branch_skeleton"] for k in limb_correspondence.values()]
    
    if combine:
        meshes = tu.combine_meshes(meshes)
        skeletons = sk.stack_skeletons(skeletons)
        
    return meshes,skeletons

def plot_limb_correspondence(limb_correspondence,
                            meshes_colors="random",
                            skeleton_colors="random",
                            mesh_name="branch_mesh",
                            scatters=[],
                            scatter_size=0.3,
                            **kwargs):
    meshes,skeletons = limb_correspondence_plottable(limb_correspondence,mesh_name=mesh_name)
        
    nviz.plot_objects(
                      meshes=meshes,
                     meshes_colors=meshes_colors,
                     skeletons=skeletons,
                     skeletons_colors=skeleton_colors,
        scatters=scatters,
        scatter_size = scatter_size,
        **kwargs
                     )
    
    
def plot_limb_path(limb_obj,path,**kwargs):
    """
    Purpose: To highlight the nodes on a path
    with just given a limb object
    
    Pseudocode: 
    1) Get the entire limb mesh will be the main mesh
    2) Get the meshes corresponding to the path
    3) Get all of the skeletons
    4) plot
    
    """
    
    nviz.plot_objects(main_mesh = limb_obj.mesh,
                        meshes=[limb_obj[k].mesh for k in path],
                      meshes_colors="red",
                     skeletons=[limb_obj[k].skeleton for k in path],
                     **kwargs)
    
    
    
# ----------- For plotting classifications ------------------ #
def plot_labeled_limb_branch_dicts(neuron_obj,
                           labels,
                           colors="red",
                          skeleton=False,
                           mesh_alpha = 1,
                                   print_color_map= True,
                          **kwargs):
    """
    Purpose: Will plot the limb branches for certain labels
    
    Ex: 
    nviz.plot_labeled_limb_branch_dicts(n_test,
                                   ["apical","apical_shaft","axon"],
                                   ["blue","aqua","red"],
                                   )
    """
    if not nu.is_array_like(labels):
        labels = [labels]
    if not nu.is_array_like(colors):
        colors = [colors]*len(labels)
        
    limb_branch_dicts = [ns.query_neuron_by_labels(neuron_obj,
                             matching_labels = [k],
                             ) for k in labels]
    if skeleton:
        visualize_type = ["mesh","skeleton"]
    else: 
        visualize_type = ["mesh"]
        
    
    nviz.plot_limb_branch_dict_multiple(neuron_obj,
                                       limb_branch_dicts,
                                       color_list = colors,
                                       visualize_type=visualize_type,
                                       mesh_color_alpha=mesh_alpha,
                                       **kwargs)
    if print_color_map:
        for l,c in zip(labels,colors):
            print(f"{l}:{c}")
        print(f"\n")
    
def plot_axon(
    neuron_obj,
    skeleton=False,
    plot_synapses = False,
    **kwargs
    ):
    
    axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                             matching_labels = ["axon"],
                             )
    axon_skeleton = neuron_obj.axon_skeleton
    nviz.visualize_neuron(neuron_obj,
                         visualize_type=["mesh","skeleton"],
                         limb_branch_dict=axon_limb_branch_dict,
                         mesh_color="red",
                         mesh_whole_neuron=True,
                         skeleton_color="black",
                          total_synapses=plot_synapses,
                         )
def plot_dendrite_and_synapses(neuron_obj,**kwargs):
    nviz.visualize_neuron(neuron_obj,
                     limb_branch_dict=neuron_obj.dendrite_limb_branch_dict,
                     mesh_color="green",
                     skeleton_color="blue",
                     limb_branch_synapses=True)
    
def plot_axon_merge_errors(neuron_obj):
    error_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                             matching_labels = ["axon-like"],
                             not_matching_labels = ["axon"]
                             )
    nviz.visualize_neuron(neuron_obj,
                         visualize_type=["mesh"],
                         limb_branch_dict=error_limb_branch_dict,
                         mesh_color="red",
                         mesh_whole_neuron=True)
    
    


def plot_branch_with_boutons_old(branch_obj,
                            bouton_color = "red",
                            non_bouton_color = "aqua",
                             main_mesh_color = "green",
                            non_bouton_size_filter = 80,
                            non_bouton_filtered_away_color = "random",
                            verbose=False):
    """
    To visualize a branch object with the bouton information 
    plotted
    
    """
    if hasattr(branch_obj,"non_boutons"):
        non_bouton_flag = True
    else:
        non_bouton_flag = False
        
    if hasattr(branch_obj,"boutons"):
        b_meshes = branch_obj.boutons
        
        if non_bouton_flag:
            non_b_meshes = branch_obj.non_boutons

            # splitting the non significant non_bouton meshes
            if non_bouton_size_filter is not None:
                non_b_kept_idx = tu.filter_meshes_by_size(non_b_meshes,
                    size_threshold = non_bouton_size_filter,
                    return_indices = True)
                non_b_filtered_idx = np.delete(np.arange(len(non_b_meshes)),non_b_kept_idx)
                if verbose:
                    print(f"Applying filtering away:")
                    print(f"non_b_kept_idx = {non_b_kept_idx}")
                    print(f"non_b_filtered_idx = {non_b_filtered_idx}")

                non_b_meshes_filtered = list(np.array(non_b_meshes)[non_b_filtered_idx])
                non_b_meshes = list(np.array(non_b_meshes)[non_b_kept_idx])

            else:
                non_b_kept_idx = np.arange(len(non_b_meshes))
                non_b_meshes_filtered = []
        
  
        non_bouton_mean = np.mean(branch_obj.width_array["non_bouton"][non_b_kept_idx])
        non_bouton_median = np.median(branch_obj.width_array["non_bouton"][non_b_kept_idx])

        width_info_to_print = dict(non_bouton_mean=non_bouton_mean,
                                  non_bouton_median=non_bouton_median,
                                  )
        width_types = ["bouton_mean",
          "bouton_median",
          "non_bouton_mean",
          "non_bouton_median",
                       "no_spine_median_mesh_center"
                      ]
        
        for w_t in width_types:
            if w_t in width_info_to_print.keys():
                curr_width = width_info_to_print[w_t]
            else:
                curr_width = branch_obj.width_new[w_t]
            print(f"{w_t}:{curr_width}")
            
        if non_bouton_filtered_away_color == "random":
            non_b_meshes_too_small_colors = mu.generate_unique_random_color_list(len(non_b_meshes_filtered),
                                        colors_to_omit=[bouton_color,non_bouton_color])
        else:
            non_b_meshes_too_small_colors = [non_bouton_filtered_away_color]*len(non_b_meshes_filtered)

        color_list = ([bouton_color]*len(b_meshes) + 
                      [non_bouton_color]*len(non_b_meshes) + 
                      non_b_meshes_too_small_colors)
                      
                      
        meshes_list = b_meshes + non_b_meshes + non_b_meshes_filtered

        nviz.plot_objects(main_mesh = branch_obj.mesh,
                          main_mesh_color=main_mesh_color,
                          main_mesh_alpha = 1,
                         meshes=meshes_list,
                         meshes_colors=color_list,
                     mesh_alpha=1)
    else:
        print(f"No bouton processing done")
    

def plot_branches_with_mesh_attribute(branches,
                                      mesh_attribute,
                                      plot_skeletons=True,
                             verbose=True):
    """
    To plot the branch meshes and their spines
    with information about them
    
    """
    
    
    if not nu.is_array_like(branches):
        branches = [branches]
        
    if plot_skeletons:
        skeletons = [k.skeleton for k in branches]
    else:
        skeletons = []
        
    total_mesh = tu.combine_meshes([k.mesh for k in branches])

    
    all_spine_list = []
    for k in branches:
        if (hasattr(k,mesh_attribute) and  
            getattr(k,mesh_attribute) is not None and
            len(getattr(k,mesh_attribute)) > 0):
            all_spine_list.append(getattr(k,mesh_attribute))
    
    if len(all_spine_list) > 0:
        total_spines = np.concatenate(all_spine_list)
    else:
        print(f"No {mesh_attribute} to plot")
        total_spines = []
        
    if verbose:
        for curr_branch in branches:
            print(f"")
            print(f"width = {curr_branch.width_new}, \nn_{mesh_attribute} = {getattr(curr_branch,f'n_{mesh_attribute}')},")
            if mesh_attribute == "spines":
                  print(f" spine_density = {curr_branch.spine_density}\n spine_volume_density = {curr_branch.spine_volume_density}")
            
            print(f"skeleton_length (in microns) = {sk.calculate_skeleton_distance(curr_branch.skeleton)/1000}\n"
                 f"area = {curr_branch.area}")
            print(f"n_synapses = {curr_branch.n_synapses}, 85% width = {tu.mesh_size(curr_branch.mesh,'ray_trace_percentile',85)}")
            print(f"n_faces = {len(curr_branch.mesh.faces)}")
            
    nviz.plot_objects(total_mesh,
                     meshes=total_spines,
                     meshes_colors="red",
                      mesh_alpha=1,
                     skeletons=skeletons)


def plot_branches_with_spines(branches,plot_skeletons=True,
                             verbose=True):
    """
    To plot the branch meshes and their spines
    with information about them
    
    """
    plot_branches_with_mesh_attribute(branches,
                                      mesh_attribute="spines",
                                      plot_skeletons=plot_skeletons,
                             verbose=verbose)
    
def plot_branches_with_boutons(branches,plot_skeletons=True,
                             verbose=True):
    """
    To plot the branch meshes and their spines
    with information about them
    
    """
    plot_branches_with_mesh_attribute(branches,
                                      mesh_attribute="boutons",
                                      plot_skeletons=plot_skeletons,
                             verbose=verbose)
    
'''def plot_branches_with_spines(branches,plot_skeletons=True,
                             verbose=True):
    """
    To plot the branch meshes and their spines
    with information about them
    
    """
    
    
    if not nu.is_array_like(branches):
        branches = [branches]
        
    if plot_skeletons:
        skeletons = [k.skeleton for k in branches]
    else:
        skeletons = []
        
    total_mesh = tu.combine_meshes([k.mesh for k in branches])
    
    all_spine_list = []
    for k in branches:
        if k.spines is not None and len(k.spines) > 0:
            all_spine_list.append(k.spines)
    
    if len(all_spine_list) > 0:
        total_spines = np.concatenate(all_spine_list)
    else:
        print("No spines to plot")
        total_spines = []
        
    if print_spine_info:
        for curr_branch in branches:
            print(f"width = {curr_branch.width_new}, \nn_spines = {curr_branch.n_spines}, spine_density = {curr_branch.spine_density}\n spine_volume_density = {curr_branch.spine_volume_density}"
                 f"\nskeleton_length (in microns) = {sk.calculate_skeleton_distance(curr_branch.skeleton)/1000}\n"
                 f"area = {curr_branch.area}")
            
    nviz.plot_objects(total_mesh,
                     meshes=total_spines,
                     meshes_colors="red",
                      mesh_alpha=1,
                     skeletons=skeletons)'''
def plot_branch_on_whole_mesh(neuron_obj,
                             limb_idx,
                             branch_idx,
                              visualize_type=None,
                              alpha = 1,
                             color="red",
                             **kwargs):
    """
    Will plot one branch with the background of whole neuron
    """
    if visualize_type is None:
        visualize_type = ["mesh"]
    limb_name = nru.get_limb_string_name(limb_idx)
    ret_col = nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                         limb_branch_dict={limb_name:[branch_idx]},
                         mesh_color=color,
                          skeleton_color=color,
                          mesh_color_alpha=alpha,
                         mesh_whole_neuron=True,
                          return_color_dict=False,
                         **kwargs)
    
def plot_limb_branch_dict(neuron_obj,
                         limb_branch_dict,
                          visualize_type=["mesh"],
                          plot_random_color_map = False,
                         color="red",
                         alpha=1,
                          dont_plot_if_empty = True,
                         **kwargs):
    """
    How to plot the color map along with: 
    nviz.plot_limb_branch_dict(filt_neuron,
                          limb_branch_dict_to_cancel,
                          plot_random_color_map=True)
    
    """
    if len(limb_branch_dict) == 0 and dont_plot_if_empty:
        print(f"limb_branch_dict empty ")
        return 
    
    if plot_random_color_map:
        color = "random"
        
    ret_col = nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                         limb_branch_dict=limb_branch_dict,
                         mesh_color=color,
                          skeleton_color=color,
                          mesh_color_alpha=alpha,
                         mesh_whole_neuron=True,
                          return_color_dict=True,
                         **kwargs)
    if plot_random_color_map:
        mu.plot_color_dict(ret_col)
        
"""def plot_limb_branch_dicts(neuron_obj,
                           limb_branch_dicts,
                           colors = None,
                           visualize_type=["mesh"],
                           alpha=1,
                           dont_plot_if_empty = True,
                           verbose = True,
                           **kwargs
                          ):
    if len(limb_branch_dicts) == 0 and dont_plot_if_empty:
        print(f"limb_branch_dicts empty ")
        return 
    
    if colors is None:
        colors = mu.generate_non_randon_named_color_list(len(limb_branch_dicts))
    
    for j,(lb,lb_color) in enumerate(zip(limb_branch_dicts,colors)):
        if verbose:
            print(f"limb branch {j} ({lb_color}): {lb}")
        
        show_at_end = False
        append_figure = True
        mesh_whole_neuron = False
        if j == 0:
            mesh_whole_neuron = True
            append_figure = False
        
        if j == len(limb_branch_dicts):
            show_at_end = True
            

        ret_col = nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                         limb_branch_dict=lb,
                         mesh_color=lb_color,
                          skeleton_color=lb_color,
                          mesh_color_alpha=alpha,
                         mesh_whole_neuron=mesh_whole_neuron,
                        show_at_end = show_at_end,  
                        append_figure=append_figure,
                          return_color_dict=True,
                         **kwargs)"""

    
def visualize_neuron_lite(neuron_obj,
                         **kwargs):
    nviz.visualize_neuron(neuron_obj,
                     visualize_type=["mesh"],
                     mesh_whole_neuron=True,
                         **kwargs)    

    
def visualize_neuron_limbs(neuron_obj,limbs_to_plot=None,
                          plot_soma_limb_network = True):
    if limbs_to_plot is not None:
        limbs_to_plot = [nru.get_limb_string_name(k) for k in limbs_to_plot]
    else: 
        limbs_to_plot = neuron_obj.get_limb_names()
        
    limb_branch = {k:"all" for k in limbs_to_plot}
    ret_col = nviz.visualize_neuron(neuron_obj,
                      visualize_type=["mesh"],
                     #limb_branch_dict="all",
                                    limb_branch_dict = limb_branch,
                      mesh_resolution="limb",
                                mesh_color_alpha=1,
                      return_color_dict=True
                     )
    if plot_soma_limb_network:
        nviz.plot_soma_limb_concept_network(neuron_obj,
                                   node_colors=ret_col)
def visualize_subset_neuron_limbs(neuron_obj,
                                 limbs_to_plot,):
    """
    Purpose: Will just plot some of the limbs
    """
    nviz.visualize_neuron_limbs(neuron_obj,limbs_to_plot,
                               plot_soma_limb_network = False)
    
def visualize_neuron_specific_limb(neuron_obj,
                                   limb_idx = None,
                                  limb_name=None,
                                   mesh_color_alpha=1,
                                  ):
    if limb_idx == "axon" or limb_name == "axon":
        limb_name = neuron_obj.axon_limb_name
    if limb_name is None:
        if type(limb_idx) == str:
            limb_name = limb_idx
        else:
            limb_name = f"L{limb_idx}"
        
    print(f"limb_name = {limb_name}")
    ret_col = nviz.visualize_neuron(neuron_obj,
                     limb_branch_dict={limb_name:"all"},
                      return_color_dict=True,
                      mesh_color_alpha=mesh_color_alpha,         
                     )
    
    nviz.plot_limb_concept_network_2D(neuron_obj,
                                     ret_col)
    
plot_limb_idx = visualize_neuron_specific_limb
plot_limb = visualize_neuron_specific_limb

def plot_valid_error_synapses(neuron_obj,
                              synapse_dict,
                             synapse_scatter_size=0.2,
                              valid_presyn_color="yellow",
                              valid_postsyn_color="aqua",
                              error_presyn_color="black",
                              error_postsyn_color="orange",
                              error_presyn_non_axon_color = "brown",
                             meshes=None,
                             meshes_colors=None,
                              
                              scatter_size=None,
                              scatters = None,
                              scatters_colors = None,
                              plot_error_synapses = False,
                              
                              mesh_alpha=0.2,
                              main_mesh_alpha=0.2,
                              
                             **kwargs):
    """
    Plot Neuron along with the presyn and postsyn
    errored synapses
    
    synapse_dict must have the following keys:
    valid_syn_centers_presyn
    errored_syn_centers_presyn
    valid_syn_centers_postsyn
    errored_syn_centers_postsyn
    
    """
    debug=False
    
    
    filtered_neuron = neuron_obj 
    synapse_center_coordinates = synapse_dict
    
    if debug:
        print(f"synapse_center_coordinates = {synapse_center_coordinates}")
    
    if scatter_size is None:
        scatter_size = []
        
    if scatters is None:
        scatters = []
        
    if scatters_colors is None:
        scatters_colors = []
    
    
    if neuron_obj is None:
        main_mesh = None
    elif type(neuron_obj) == type(trimesh.Trimesh()):
        main_mesh = neuron_obj
        if meshes is None:
            meshes = []
            meshes_colors=[]
    else:
        main_mesh=filtered_neuron.mesh_from_branches
        meshes=filtered_neuron.get_soma_meshes()
        meshes_colors=["red"]
    
    
    synapse_list = ["valid_syn_centers_presyn",
                    "valid_syn_centers_postsyn"]
                   
    synapse_scatters_colors=[valid_presyn_color,valid_postsyn_color]
    
    if plot_error_synapses:
        synapse_list += ["errored_syn_centers_presyn",
                   "errored_syn_centers_postsyn",
                        "errored_syn_centers_presyn_non_axon"]
        
        # will not double plot the non axon centers
        if "errored_syn_centers_presyn_non_axon" in synapse_center_coordinates.keys():
            synapse_center_coordinates["errored_syn_centers_presyn"] = nu.setdiff2d(
                synapse_center_coordinates["errored_syn_centers_presyn"],
                synapse_center_coordinates["errored_syn_centers_presyn_non_axon"]
            )
    
        synapse_scatters_colors += [error_presyn_color,error_postsyn_color]
        
    
    synapse_scatters = [synapse_center_coordinates.get(k,np.array([])) for k in synapse_list]
    
    
    if debug:
        print(f"synapse_scatters = {synapse_scatters}")
    
    if not nu.is_array_like(synapse_scatter_size ):
        synapse_scatter_size = [synapse_scatter_size]
        
    
    
    
    synapse_scatter_size = synapse_scatter_size*len(synapse_scatters)
    
    
#     synapse_scatters = [synapse_center_coordinates["valid_syn_centers_presyn"],
#                                synapse_center_coordinates["errored_syn_centers_presyn"],
#                       synapse_center_coordinates["valid_syn_centers_postsyn"],
#                                synapse_center_coordinates["errored_syn_centers_postsyn"],
#                      ]
    
    if False:
        for j,m in enumerate(meshes):
            print(f"Mesh {j} with center: {tu.mesh_center_vertex_average(m)}")
            
    
    scatter_size_to_use=scatter_size + synapse_scatter_size
    
    
        
    nviz.plot_objects(main_mesh=main_mesh,
                  meshes=meshes,
                  meshes_colors=meshes_colors,
                  mesh_alpha=mesh_alpha,
                      main_mesh_alpha=main_mesh_alpha,
            scatters=scatters+synapse_scatters,
             scatters_colors=scatters_colors+synapse_scatters_colors,
                     scatter_size=scatter_size_to_use,
                     **kwargs)

def visualize_limb_obj(limb_obj,                 
    meshes_colors = "random",
    skeletons_colors = "random",
    plot_soma_vertices = True,
    soma_vertices_size = 0.3,
    plot_starting_coordinate = False,
    starting_coordinate_size = 1,):
    """
    purpose: To visualize just a limb object


    """
    


    all_scatters = []
    all_scatters_sizes = []

    if plot_soma_vertices:
        all_scatters.append(limb_obj.current_touching_soma_vertices)
        all_scatters_sizes.append(soma_vertices_size)

    if plot_starting_coordinate:
        all_scatters.append(limb_obj.current_starting_coordinate)
        all_scatters_sizes.append(starting_coordinate_size)


    meshes,skeletons = nviz.limb_correspondence_plottable(limb_obj.limb_correspondence)
    nviz.plot_objects(meshes=meshes,
                     meshes_colors=meshes_colors,
                     skeletons=skeletons,
                     skeletons_colors=skeletons_colors,
                     scatters=all_scatters,
                     scatter_size=all_scatters_sizes)
    
def visualize_axon_dendrite(neuron_obj,
                           axon_color = "black",
                            dendrite_color = "aqua",
                            plot_mesh = True,
                            plot_skeleton = True):
    """
    Purpose: To visualize the axon
    and dendrite of a neuron

    Pseudocode:
    1) Get the axon and dendrite limb branches
    2) Construct an overall limb branch using the axon-dnedrite colors
    3) plot neuron
    """


    visualize_type = []
    if plot_mesh:
        visualize_type.append("mesh")
    if plot_skeleton:
        visualize_type.append("skeleton")

    #construct overall limb branch dict
    axon_limb_branch_dict = neuron_obj.axon_limb_branch_dict
    dendrite_limb_branch_dict = neuron_obj.dendrite_limb_branch_dict
    total_limb_branch_dict = nru.limb_branch_union([axon_limb_branch_dict,dendrite_limb_branch_dict])

    #construct the the color limb_brnach dict
    color_dict = dict()

    limb_dicts = [axon_limb_branch_dict,dendrite_limb_branch_dict]
    dict_colors = [axon_color,dendrite_color]

    for l_dict,l_color in zip(limb_dicts,dict_colors):
        for limb_name,branch_list in l_dict.items():
            if limb_name not in color_dict:
                color_dict[limb_name]= dict()
            for b in branch_list:
                color_dict[limb_name][b] = l_color

    nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                          limb_branch_dict=total_limb_branch_dict,
                         mesh_color=color_dict,
                         skeleton_color=color_dict)

    
def limb_branch_dicts_to_combined_color_dict(limb_branch_dict_list,
                                            color_list):
    """
    Purpose: Will combine multiple limb branch dict lists into 
    one color dictionary of limb_name --> branch_name --> color
    
    """
    color_dict = dict()
    
    for limb_branch_dict,c in zip(limb_branch_dict_list,color_list):
        for limb_name,branch_list in limb_branch_dict.items():
            if limb_name not in color_dict:
                color_dict[limb_name] = dict()
            for b in branch_list:
                color_dict[limb_name][b] = c
    return color_dict


def visualize_neuron_axon_dendrite(
    neuron_obj,
    visualize_type=["mesh"],
    axon_color = "aqua",
    dendrite_color="blue",

    mesh_color_alpha = 1,

    mesh_soma_color = "red",
    mesh_soma_alpha = 1,
    **kwargs):
    """
    Purpose: Fast way to visuzlize the axon and dendritic
    parts of a neuron
    
    """

    color_dict = nviz.limb_branch_dicts_to_combined_color_dict([neuron_obj.axon_limb_branch_dict,
                                             neuron_obj.dendrite_limb_branch_dict],
                                            color_list=[axon_color,dendrite_color])

    nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                          limb_branch_dict="all",
                          mesh_color_alpha=mesh_color_alpha,
                          mesh_color=color_dict,
                          mesh_soma_color=mesh_soma_color,
                          mesh_soma_alpha=mesh_soma_alpha,

                          skeleton_color=color_dict,
                          **kwargs
                         )    
    
def visualize_neuron_axon_merge_errors(
    neuron_obj,
    visualize_type=["mesh"],
    axon_error_color = "aqua",

    mesh_color="black",
    mesh_color_alpha = 1,

    mesh_soma_color = "red",
    mesh_soma_alpha = 1,
    **kwargs):
    """
    Purpose: Fast way to visuzlize the axon and dendritic
    parts of a neuron
    
    """
    axon_errors =  ns.query_neuron_by_labels(neuron_obj,matching_labels=["axon-error"])
    non_error_limb_branch_dict = nru.limb_branch_setdiff([neuron_obj.limb_branch_dict,
                       axon_errors])

    color_dict = nviz.limb_branch_dicts_to_combined_color_dict([axon_errors,
                                                                non_error_limb_branch_dict],
                                            color_list=[axon_error_color,mesh_color])

    nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                          limb_branch_dict="all",
                          mesh_color_alpha=mesh_color_alpha,
                          mesh_color=color_dict,
                          mesh_soma_color=mesh_soma_color,
                          mesh_soma_alpha=mesh_soma_alpha,

                          skeleton_color=color_dict,
                          **kwargs
                         )    
    
    
def plot_branch_mesh_attribute(neuron_obj,
                              mesh_attribute,
                              mesh_color,
                              mesh_alpha=0.8,
                               return_vertices = True,
                               flip_y=True,
                               plot_at_end=True,
                              verbose = False):
    """
    Purpose: To take a mesh attribute that is part of a branch object
    inside of a neuron and then to plot all of them
    
    Ex:
    nviz.plot_branch_mesh_attribute(neuron_obj_high_fid_axon,
                              mesh_attribute="boutons",
                              mesh_color="aqua",
                              mesh_alpha=0.8,
                               return_vertices = True,
                                plot_at_end=False,
                                flip_y = True,
                              verbose = True)
    """
    local_time = time.time()
    
    current_neuron = neuron_obj
    
    if plot_at_end:
        ipv.clear()
    
    bouton_meshes = []
                
    for limb_names in current_neuron.get_limb_names():
        curr_limb_obj = current_neuron[limb_names]
        for branch_name in curr_limb_obj.get_branch_names():
            if hasattr(curr_limb_obj[branch_name],mesh_attribute) and getattr(curr_limb_obj[branch_name],mesh_attribute) is not None :
                curr_atr = getattr(curr_limb_obj[branch_name],mesh_attribute)
                
                if type(curr_atr) == list:
                    bouton_meshes += curr_atr
                else:
                    bouton_meshes += [curr_atr]

    if verbose:
        print(f"Number of {mesh_attribute} meshes = {len(bouton_meshes)}")
    boutons_color_list = mu.process_non_dict_color_input(mesh_color)
    boutons_color_list_alpha = mu.apply_alpha_to_color_list(boutons_color_list,alpha=mesh_alpha)

    total_vertices = []
    if len(boutons_color_list_alpha) == 1:
        combined_bouton_meshes = tu.combine_meshes(bouton_meshes)
        plot_ipv_mesh(combined_bouton_meshes,color=boutons_color_list_alpha[0],flip_y=flip_y)
        total_vertices.append(combined_bouton_meshes.vertices)

    else:
        boutons_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(boutons_color_list_alpha,
                                                                                  n_colors=len(bouton_meshes))


        for curr_bouton_mesh,curr_bouton_color in zip(bouton_meshes,boutons_colors_list_alpha_fixed_size):
            plot_ipv_mesh(curr_bouton_mesh,color=curr_bouton_color,flip_y=flip_y)
            total_vertices.append(curr_bouton_mesh.vertices)
            
    if verbose:
        print(f"Plotting mesh {mesh_attribute}= {time.time() - local_time}")
        local_time = time.time()
        
    if plot_at_end:
        ipv.show()

    if return_vertices:
        return total_vertices
    
def plot_web_intersection(neuron_obj,
                          limb_idx,
                         branch_idx,
                          parent_color="yellow",
                          downstream_color = "pink",
                          web_color = "purple",
                          mesh_alpha = 1,
                          
                          print_web_info = True,
                         plot_boutons = True,
                          plot_whole_limb = False,
                          whole_limb_color = "green",
                          whole_limb_alpha = 0.2,
                          
                    mesh_boutons_color = "aqua",
                         verbose=False,
                         **kwargs):
    """
    To plot the webbing of a branch at it's intersection

    Pseudocode: 
    1) Get the downstream nodes of the branch
    2) Assemble the meshes of the parent and downstream branch
    3) If requested, get all of the bouton meshes
    4) Get the web mesh of parent node
    5) Plot

    """
    if verbose:
        print(f"Plotting web intersection for limb_idx {limb_idx}, branch_idx {branch_idx}")
    
    limb_obj = neuron_obj[limb_idx]
    parent_branch_obj = limb_obj[branch_idx]
    
    #1) Get the downstream nodes of the branch
    downstream_branches = xu.downstream_nodes(limb_obj.concept_network_directional,
                                              branch_idx)
    
    #2) Assemble the meshes of the parent and downstream branch
    total_nodes = list(downstream_branches) + [branch_idx]
    
    meshes = []
    meshes_colors = []
    
    meshes.append(parent_branch_obj.mesh)
    meshes_colors.append(parent_color)
    
    meshes += [limb_obj[k].mesh for k in downstream_branches]
    meshes_colors+= [downstream_color]*len(downstream_branches)
    
    if plot_boutons:
        for k in total_nodes:
            curr_boutons = limb_obj[k].boutons
            if curr_boutons is not None and len(curr_boutons)>0:
                meshes += curr_boutons
                meshes_colors += [mesh_boutons_color]*len(curr_boutons)
    
    #4) Get the web mesh of parent node
    try:
        web_mesh = parent_branch_obj.web
        web_cdf = parent_branch_obj.web_cdf
        
        meshes.append(web_mesh)
        meshes_colors.append(web_color)
    
        web_flag = True
    except:
        print(f"No webbing for this branch!!")
        web_flag = False
    
    
    
    
    if print_web_info:
        width_name = "no_bouton_median"
        backup_width = "no_spine_median_mesh_center"
        
        if web_flag:
            web_bbox_rations = tu.bbox_side_length_ratios(web_mesh)
            web_volume_ratio = tu.mesh_volume_ratio(web_mesh)
            print(f"Web Mesh = {web_mesh}, web_cdf = {web_cdf} ")
            print(f"web_bbox_rations = {web_bbox_rations}, web_volume_ratio = {web_volume_ratio}")
        
        print(f"\nParent Node {branch_idx}, n_boutons = {parent_branch_obj.n_boutons}")
        try:
            print(f"Parent_width ({width_name})= {parent_branch_obj.width_new[width_name]}")
            print(f"Parent_width ({backup_width}) = {parent_branch_obj.width_new[backup_width]}")
        except:
            print(f"Parent_width ({backup_width}) = {parent_branch_obj.width_new[backup_width]}")
            
        
        for d in downstream_branches:
            print(f"\nDownstream Branch {d}, n_boutons = {limb_obj[d].n_boutons}")
            print(f"\nDownstream branch {d} width ({backup_width}) = {limb_obj[d].width_new[backup_width]}")
            try:
                print(f"Downstream branch {d} width ({width_name})= {limb_obj[d].width_new[width_name]}")
            except:
                pass
            
            
            child_angle = nru.find_parent_child_skeleton_angle(limb_obj,
                                                              d)
            print(f"child_angle = {child_angle}")
            
        if len(downstream_branches)>0:
            sibling_angles = nru.find_sibling_child_skeleton_angle(limb_obj,
                                                                   downstream_branches[0],
                                         )
            print(f"\nsibling_angles = {sibling_angles}\n")
        else:
            print(f"\n No downstream nodes to show")
        
    
    if plot_whole_limb:
        limb_mesh = limb_obj.mesh
    else:
        limb_mesh= None
        
    nviz.plot_objects(main_mesh = limb_mesh,
                     main_mesh_color=whole_limb_color,
                      main_mesh_alpha=whole_limb_alpha,
                     meshes=meshes,
                     meshes_colors=meshes_colors,
                      mesh_alpha=mesh_alpha,
                      **kwargs
                     )

    
def set_zoom(center_coordinate,
                       radius=None,
                       radius_xyz = None,
                       show_at_end=False,
                      flip_y = True,
                    turn_axis_on=False):
    if radius is None:
        radius = 5000
    
    coord = np.array(center_coordinate)
    
    if flip_y:
        coord[...,1] = -coord[...,1]
        
    if radius_xyz is None:
        radius_xyz = np.array([radius,radius,radius])
    coord_radius = [k if k is not None else radius for k in radius_xyz]
    ipv_function = [ipv.xlim,ipv.ylim,ipv.zlim]
    for c,c_rad,ipvf in zip(coord,coord_radius,ipv_function):
        ipvf(c - c_rad, c + c_rad)

    if turn_axis_on:
        ipv.style.axes_on()
        ipv.style.box_on()
        
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
    if show_at_end:
        ipv.show()    
        
def plot_limb_correspondence_multiple(limb_correspondence_list,
                                      color_list=None,
                                      verbose = False,
                                      **kwargs
                                     ):
    if type(limb_correspondence_list) == dict:
        limb_correspondence_list = [v for k,v in limb_correspondence_list.items()]
    if color_list is None:
        color_list = mu.generate_non_randon_named_color_list(len(limb_correspondence_list))
        if verbose:
            print(f"Colors generated: {color_list}")
    total_meshes = []
    total_skeletons = []
    
    
    
    for k in limb_correspondence_list:
        mesh,skel = nviz.limb_correspondence_plottable(
            k,
            combine=True)
        total_meshes.append(mesh)
        total_skeletons.append(skel)
        
    if verbose:
        print(f"Color_list: {[(k,c) for k,c in enumerate(color_list)]}")
    nviz.plot_objects(meshes = total_meshes,
                     meshes_colors = color_list,
                     skeletons=total_skeletons,
                     skeletons_colors=color_list,
                     **kwargs)
    

def plot_limb_branch_dict_multiple(neuron_obj,
                                   limb_branch_dict_list,
                                   color_list=None,
                                   visualize_type=["mesh"],
                                  scatters_list=[],
                                   scatters_colors=None,
                                   scatter_size=0.1,
                                   mesh_color_alpha = 0.2,
                                   verbose = False,
                                   mesh_whole_neuron = True,
                                  **kwargs):
    """
    Purpose: to plot multiple limb branch dicts
    with scatter points associated with it
    
    """
    if color_list is None:
        color_list = mu.generate_non_randon_named_color_list(len(limb_branch_dict_list))
        if verbose:
            print(f"Colors generated: {color_list}")
    
    limb_branch_total = nru.limb_branch_union(limb_branch_dict_list)
    color_finals = nviz.limb_branch_dicts_to_combined_color_dict(limb_branch_dict_list,color_list)
    
    if scatters_colors is None:
        scatters_colors = color_list
    
    nviz.visualize_neuron(neuron_obj,
                          visualize_type=visualize_type,
                         limb_branch_dict=limb_branch_total,
                         mesh_color=color_finals,
                          mesh_color_alpha=mesh_color_alpha,
                          skeleton_color=color_finals,
                         mesh_whole_neuron=mesh_whole_neuron,
                          scatters=scatters_list,
                          scatters_colors=color_list,
                          scatter_size=scatter_size,
                         **kwargs)
    
        

def visualize_branch_at_downstream_split(neuron_obj,
                                         limb_idx,
                                         branch_idx,
                                        radius = 20000,
                                         turn_axis_on=True,
                                        branch_color = "mediumblue",
                                        downstream_color = "red",
                                        print_axon_border_info = True,
                                        verbose = True,
                                        **kwargs):
    """
    Purpose: To zoom on the point at which a branch splits off
    
    
    Ex:
    axon_limb_name = neuron_obj.axon_limb_name
    curr_idx = 1
    curr_border_idx = border_brnaches[curr_idx]
    nviz.visualize_branch_at_downstream_split(neuron_obj=neuron_obj,
                                             limb_idx=neuron_obj.axon_limb_name,
                                             branch_idx=curr_border_idx,
                                            radius = 20000,
                                            branch_color = "mediumblue",
                                            downstream_color = "red",
                                            print_border_info = True,
                                            verbose = True)
    """
    limb_idx = nru.limb_label(limb_idx)
    
    border_color = branch_color
    axon_limb_name = limb_idx
    curr_border_idx = branch_idx


    shared_skeleton_pt, downstream_branches = nru.skeleton_coordinate_connecting_to_downstream_branches(neuron_obj[axon_limb_name],
                                                                    curr_border_idx,
                                                                            return_downstream_branches=True)

    if verbose:
        print(f"# of downstream targets = {len(downstream_branches)}")


    # plot the neuron at the current branching point

    dict_list = [{axon_limb_name:[curr_border_idx]},
                 {axon_limb_name:downstream_branches}]
    color_list = [border_color,downstream_color]
    color_dict = nviz.limb_branch_dicts_to_combined_color_dict(dict_list,
                                                 color_list)

    nviz.visualize_neuron(neuron_obj,
                      visualize_type=["mesh","skeleton"],
                      limb_branch_dict = {axon_limb_name:list(downstream_branches) + [curr_border_idx]},
                      mesh_color=color_dict,
                      skeleton_color=color_dict,
                      mesh_boutons=True,
                      mesh_web=True,
                      mesh_whole_neuron=True,
                          **kwargs
                                        )


    if print_axon_border_info:

        attr_dict = au.axon_branching_attributes(neuron_obj,
                                    neuron_obj.axon_limb_idx,
                                    curr_border_idx,
                                    verbose=False)
        for k,v in attr_dict.items():
            print(f"{k}:{v}")
    nviz.set_zoom(shared_skeleton_pt,
                      radius=radius,
                 turn_axis_on=turn_axis_on)
        
        
def set_zoom_to_limb_branch(neuron_obj,
                           limb_idx,
                           branch_idx,
                            radius=3000,
                           turn_axis_on=True):
    shared_skeleton_pt, downstream_branches = nru.skeleton_coordinate_connecting_to_downstream_branches(neuron_obj[limb_idx],
                                                            branch_idx,
                                                                            return_downstream_branches=True)
    nviz.set_zoom(shared_skeleton_pt,
                      radius=radius,
                 turn_axis_on=turn_axis_on)

def add_scatter_to_current_plot(scatters,
                               scatters_colors,
                               scatter_size=0.1):
    nviz.plot_objects(scatters=scatters,
                      scatters_colors=scatters_colors,
                      scatter_size=scatter_size,
                      append_figure=True,
                  show_at_end=False,
                  set_zoom=False,
                 #zoom_coordinate=shared_skeleton_pt,
                 #zoom_radius=3000
                 )
    
def plot_original_vs_proofread(original,
    proofread,
    original_color = "red",
    proofread_color = "blue",
    mesh_alpha = 1,
    plot_mesh= True,
    plot_skeleton = False):
    """
    Purpose: To visualize the original version
    and proofread version of a neuron_obj

    Pseudocode:
    1) Turn original neuron and the proofread neuron into meshes
    2) Plot both meshes
    
    Ex: 
    nviz.plot_original_vs_proofread(original = neuron_obj,
        proofread = filtered_neuron,
        original_color = "red",
        proofread_color = "blue",
        mesh_alpha = 0.3,
        plot_mesh= True,
        plot_skeleton = True)

    """


    skeletons = []
    skeletons_colors = []
    if plot_skeleton:
        skeletons.append(original.skeleton)
        skeletons.append(proofread.skeleton)

        skeletons_colors = [original_color,proofread_color]

    if plot_mesh:
        if not tu.is_mesh(original):
            original = original.mesh
        if not tu.is_mesh(proofread):
            proofread = proofread.mesh
    else:
        original= None
        proofread = None


    print(f"original = {original}")
    nviz.plot_objects(main_mesh=original,
                     main_mesh_color=original_color,
                      main_mesh_alpha=mesh_alpha,
                     meshes = [proofread],
                     meshes_colors=[proofread_color],
                     skeletons=skeletons,
                     skeletons_colors=skeletons_colors,
                     mesh_alpha=mesh_alpha)
    
    
def vector_to_scatter_line(vector,
                          start_coordainte,
                          distance_to_plot = 2000,
                        n_points = 20):
    """
    Will turn a vector into a sequence of scatter points to be graphed
    
    """
    scaling = np.linspace(0,distance_to_plot,n_points).reshape(-1,1)
    normal_line = scaling*nu.repeat_vector_down_rows(vector,len(scaling)) + start_coordainte
    return normal_line

def plot_intermediates(limb_obj,
                       branches,
                      verbose = True):
    """
    Purpose: To graph the skeletons
    """
    if len(branches) != 4:
        curr_colors = mu.generate_non_randon_named_color_list(len(branches))
    else:
        curr_colors = ["red","aqua","purple","green"]
    if verbose: 
        print(f"coordinate_branches = {list(branches)}")
        for c,col in zip(branches,curr_colors):
            print(f"{c} = {col}")

    
    nviz.plot_objects(meshes=[limb_obj[k].mesh for k in branches],
                     meshes_colors=curr_colors,
                     skeletons=[limb_obj[k].skeleton for k in branches],
                     skeletons_colors=curr_colors)
    
def plot_branch_groupings(limb_obj,
groupings,
verbose = False,
plot_meshes = True,
plot_skeletons = True,
                         extra_group = None,
                         extra_group_color = None,
                          extra_group_color_name = "skipped",
                         ):
    """
    Purpose: To Plot branch objects all of a certain color
    that are in the same group, and the grouping
    is described with a graph

    Pseudocode: 
    1) Get all the connected components (if a graph is given for the groupings)
    2) Generate a color list for the groups
    3) Based on what attributes are set, compile plottable lists (and add the colors to it)
    4) Plot the branch objects
    
    Ex:
    nviz.plot_branch_groupings(limb_obj = neuron_obj[0],
    groupings = G,
    verbose = False,
    plot_meshes = True,
    plot_skeletons = True)


    """

    if xu.is_graph(groupings):
        groupings = xu.connected_components(groupings)
        if verbose:
            print(f"groupings = {groupings}")

    
    if extra_group is not None and extra_group_color is None:
        groupings_colors = mu.generate_non_randon_named_color_list(len(groupings)+1)
        extra_group_color = groupings_colors[-1]
        groupings_colors = groupings_colors[:-1]
    else:
        groupings_colors = mu.generate_non_randon_named_color_list(len(groupings))

    if extra_group is not None: 
        groupings_colors = list(groupings_colors) + [extra_group_color] 
        groupings = list(groupings) + [extra_group]
        print(f"Last group is {extra_group_color_name}")
        
    meshes = []
    skeletons = []
    color_list = []

    for j,(gp,c) in enumerate(zip(groupings,groupings_colors)):
        print(f"Group {c}: {gp}")
        if plot_meshes:
            meshes += [limb_obj[k].mesh for k in gp]
        if plot_skeletons:
            skeletons += [limb_obj[k].skeleton for k in gp]

        if plot_meshes or plot_skeletons:
            color_list += [c]*len(gp)


    nviz.plot_objects(meshes=meshes,
                      meshes_colors= color_list,
                     skeletons=skeletons,
                      skeletons_colors= color_list,)

    
def plottable_from_branches(limb_obj,branch_list,attributes):
    if not nu.is_array_like(branch_list):
        branch_list = [branch_list]
    
    singular = False
    if not nu.is_array_like(attributes):
        singular = True
        attributes = [attributes]
        
    return_attributes = [[getattr(limb_obj[k],v) for k in branch_list] for v in attributes]
    
    if singular:
        return return_attributes[0]
    else:
        return return_attributes
    
def plottable_meshes(limb_obj,branch_list):
    return plottable_from_branches(limb_obj,branch_list,"mesh")

def plottable_skeletons(limb_obj,branch_list):
    return plottable_from_branches(limb_obj,branch_list,"skeleton")

def plottable_meshes_skeletons(limb_obj,branch_list):
    return plottable_from_branches(limb_obj,branch_list,["mesh","skeleton"])

def plot_branches_with_colors(limb_obj,branch_list,colors = None,verbose = True):
    meshes,skeletons = nviz.plottable_meshes_skeletons(limb_obj,branch_list)
    if colors is None:
        colors = mu.generate_non_randon_named_color_list(len(meshes))
        
    if verbose:
        for c,b_idx in zip(colors,branch_list):
            print(f"{b_idx}:{c}")
    nviz.plot_objects(meshes = meshes,
                     meshes_colors= colors,
                     skeletons = skeletons,
                     skeletons_colors=colors)


def plot_branch_with_neighbors(limb_obj,branch_idx,neighbor_idxs=None,
                               branch_color = "red",
                               neighbors_color = "blue",
                               scatters_colors="yellow",
                               scatter_size = 1,
                               visualize_type = ["mesh","skeleton"],
                               verbose = False,
                               main_skeleton = None,
                               skeletons = None,
                              **kwargs):
    """
    Will plot a main branch and other branches around it
    
    Ex: 
    nviz.plot_branch_with_neighbors(limb_obj,16,nru.downstream_nodes(limb_obj,16),
                                scatters=[nru.downstream_endpoint(limb_obj,16)],
                                verbose = True)
    """
    if neighbor_idxs is None:
        neighbor_idxs = nru.downstream_nodes(limb_obj,branch_idx)
    
    if not nu.is_array_like(neighbor_idxs):
        neighbor_idxs = [neighbor_idxs]
        
    downstream_nodes = [k for k in neighbor_idxs if k != branch_idx]
    
    plot_dict = dict()
    
    if "mesh" in visualize_type:
        plot_dict["main_mesh"] = limb_obj[branch_idx].mesh
        plot_dict["meshes"] = nviz.plottable_meshes(limb_obj,neighbor_idxs)
        
    if "skeleton" in visualize_type:
        if main_skeleton is None:
            plot_dict["main_skeleton"] = limb_obj[branch_idx].skeleton
        else:
            plot_dict["main_skeleton"] = main_skeleton
        
        if skeletons is None:
            plot_dict["skeletons"] = nviz.plottable_skeletons(limb_obj,neighbor_idxs)
        else:
            plot_dict["skeletons"] = skeletons
    
    if verbose:
        print(f"branch_idx ({branch_idx}): {branch_color}\n neighbor nodes ({neighbor_idxs}): {neighbors_color}")

    plot_dict.update(kwargs)
    
    nviz.plot_objects(
                     main_mesh_color=branch_color,
                     meshes_colors=neighbors_color,
                      skeletons_colors=neighbors_color,
                     scatters_colors = scatters_colors,
                     scatter_size=scatter_size,
                    **plot_dict)
    
def plot_candidates(neuron_obj,
                   candidates,
                   color_list=None,
                   mesh_color_alpha=1,
                    visualize_type  = ["mesh"],
                    verbose = False,
                    dont_plot_if_no_candidates = True,
                   **kwargs):
    
    if dont_plot_if_no_candidates and len(candidates) == 0:
        print(f"Not plotting because no candidates")
        return 
    
    if verbose:
        print(f"Plotting candidates")
    
    candidate_lb = [nru.limb_branch_from_candidate(k) for k in candidates]
    nviz.plot_limb_branch_dict_multiple(neuron_obj,
                                       limb_branch_dict_list=candidate_lb,
                                       color_list=color_list,
                                       visualize_type=visualize_type,
                                        mesh_color_alpha=mesh_color_alpha,
                                        **kwargs
                                       )
    
def plot_compartments(neuron_obj,
                     apical_color = "blue",
                     apical_shaft_color = "aqua",
                     apical_tuft_color = "purple",
                     basal_color = "yellow",
                     axon_color = "red",
                     oblique_color = "green"
                     ):
#     nviz.plot_limb_branch_dict_multiple(neuron_obj,
#                               [neuron_obj.label_limb_branch_dict("apical"),
#                               neuron_obj.label_limb_branch_dict("apical_shaft"),
#                               neuron_obj.label_limb_branch_dict("apical_tuft"),
#                               neuron_obj.label_limb_branch_dict("basal"),
#                               neuron_obj.label_limb_branch_dict("axon"),
#                               neuron_obj.label_limb_branch_dict("oblique")],
#                               [ apical_color,
#                                  apical_shaft_color,
#                                  apical_tuft_color,
#                                  basal_color,
#                                  axon_color,
#                                   oblique_color,
                                  
#                               ])
    nviz.plot_labeled_limb_branch_dicts(neuron_obj,
                          ["apical",
                           "apical_shaft",
                           "apical_tuft",
                           "basal",
                           "axon",
                           "oblique"],
                          [ apical_color,
                             apical_shaft_color,
                             apical_tuft_color,
                             basal_color,
                             axon_color,
                              oblique_color,

                          ])
    print(f"Unlabeled: transparent green")
    
    
def plot_mesh_face_idx(mesh,face_idx,meshes_colors = "random",**kwargs):
    """
    To plot a mesh divided up by a face_mesh_idx
    
    Ex: 
    nviz.plot_mesh_face_idx(neuron_obj[0][0].mesh,return_face_idx)
    """
    nviz.plot_objects(meshes = tu.split_mesh_into_face_groups(mesh,
                                  face_idx,
                                  return_dict=False,
                                  return_idx = False),
                          meshes_colors=meshes_colors,
                     **kwargs)
    
def plot_soma_meshes(neuron_obj,
                     meshes_colors = None,
                     verbose = False,
                    **kwargs):
    soma_node_names = neuron_obj.get_soma_node_names()
    if meshes_colors is None:
        meshes_colors = mu.generate_non_randon_named_color_list(len(soma_node_names))
        
    if verbose:
        for s_name,s_col in zip(soma_node_names,meshes_colors):
            print(f"{s_name}: {s_col}")
            
    nviz.plot_objects(meshes=[neuron_obj[k].mesh for k in soma_node_names],
                     meshes_colors=meshes_colors,
                     **kwargs)

def plot_meshes_skeletons(meshes,skeletons,**kwargs):

    cols = mu.generate_non_randon_named_color_list(len(meshes))
    nviz.plot_objects(
        meshes = meshes,
        meshes_colors=cols,
        skeletons = skeletons,
        skeletons_colors=cols,
        **kwargs
    )
    
def plot_soma_extraction_meshes(
    mesh,
    soma_meshes,
    glia_meshes = None,
    nuclei_meshes = None,
    soma_color = soma_color,
    glia_color = glia_color,
    nuclei_color = nuclei_color,
    verbose = False,
    ):
    """
    Purpose: To plot the dataproducts from the
    soma extractio
    """
    
    total_soma_list = nu.to_list(soma_meshes)
    if glia_meshes is None:
        glia_meshes = []
    glia_pieces = nu.to_list(glia_meshes)
    if nuclei_meshes is None:
        nuclei_meshes = []
    nuclei_pieces = nu.to_list(nuclei_meshes)

    if verbose:
        print(f"# of somas = {len(total_soma_list)}")
        print(f"# of glia = {len(glia_pieces)}")
        print(f"# of nuclei = {len(nuclei_pieces)}")

    meshes = total_soma_list + glia_pieces + nuclei_pieces
    meshes_colors = [soma_color]*len(total_soma_list) + [glia_color]*len(glia_pieces) + [nuclei_color]*len(nuclei_pieces)
    ipvu.plot_objects(
        mesh,
        meshes = meshes,
        meshes_colors=meshes_colors
    )

def plot_spines_head_neck(neuron_obj,**kwargs):
    spu.plot_spines_head_neck(neuron_obj,**kwargs)
    
def plot_synapses(neuron_obj,**kwargs):
    syu.plot_synapses(neuron_obj,**kwargs)
    
def plot_branch(
    branch_obj,
    upstream_color = "yellow",
    downstream_color = "aqua",
    verbose = True,
    **kwargs):
    
    if verbose:
        print(f"upstream_color = {upstream_color}")
        print(f"downstream_color = {downstream_color}")
    
    upstream = branch_obj.endpoint_upstream
    downstream = branch_obj.endpoint_downstream
    
    ipvu.plot_objects(
        branch_obj.mesh,
        branch_obj.skeleton,
        scatters=[upstream,downstream],
        scatters_colors=[upstream_color,downstream_color],
        **kwargs
    )

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import ipyvolume_utils as ipvu

#--- from neurd_packages ---
from . import axon_utils as au
from . import neuron
from . import neuron_searching as ns
from . import neuron_utils as nru
from . import proofreading_utils as pru
from . import synapse_utils as syu
from . import spine_utils as spu



plot_objects = ipvu.plot_objects
from . import neuron_visualizations as nviz
