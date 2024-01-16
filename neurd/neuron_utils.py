'''



Purpose of this file: To help the development of the neuron object
1) Concept graph methods
2) Preprocessing pipeline for creating the neuron object from a meshs




'''
import copy
from copy import deepcopy
import itertools
from importlib import reload
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from pykdtree.kdtree import KDTree
from pykdtree.kdtree import KDTree 
import time
import trimesh
from trimesh.ray import ray_pyembree
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from . import microns_volume_utils as mvu
from . import h01_volume_utils as hvu

soma_face_offset = 2


#importing at the bottom so don't get any conflicts

#for meshparty preprocessing

# tools for restricting 

# -------------- 7/22 Help Filter Bad Branches ------------------ #
def classify_error_branch(curr_branch,width_to_face_ratio=5):
    curr_width = curr_branch.width
    curr_face_count = len(curr_branch.mesh.faces)
    if curr_width/curr_face_count > width_to_face_ratio:
        return True
    else:
        return False
    
def classify_endpoint_error_branches_from_limb_concept_network(curr_concept_network,**kwargs):
    """
    Purpose: To identify all endpoints of concept graph where the branch meshes/skeleton
    are likely a result of bad skeletonization or meshing:
    
    Applications: Can get rid of these branches later
    
    Pseudocode: 
    1) Get all of the endpoints of the concept network
    2) Get all of the branch objects for the endpoints
    3) Return the idx's of the branch objects that test positive for being an error branch
    """
    
    #1) Get all of the endpoints of the concept network
    end_nodes = xu.get_nodes_of_degree_k(curr_concept_network,1)
    
    #2) Get all of the branch objects for the endpoints
    end_node_branches = [curr_concept_network.nodes[k]["data"] for k in end_nodes]
    
    #3) Return the idx's of the branch objects that test positive for being an error branch
    total_error_end_node_branches = []
    
    for en_idx,e_branch in zip(end_nodes,end_node_branches):
        if classify_error_branch(e_branch):
            total_error_end_node_branches.append(en_idx)
    
    return total_error_end_node_branches
    

# -------------- tools for the concept networks ------------------ #


def whole_neuron_branch_concept_network_old(input_neuron,
                                  directional=True,
                                 limb_soma_touch_dictionary = None,
                                 print_flag = False):
    
    """
    Purpose: To return the entire concept network with all of the limbs and 
    somas connected of an entire neuron
    
    Arguments:
    input_neuron: neuron object
    directional: If want a directional or undirectional concept_network returned
    limb_soma_touch_dictionary: a dictionary mapping the limb to the starting soma
    you want it to start if directional option is set
    Ex:  {"L1":[0,1]})
    
    
    Pseudocode:  
    1) Get the soma subnetwork from the concept network of the neuron
    2) For each limb network:
    - if directional: 
    a) if no specific starting soma picked --> use the soma with the smallest index as starting one
    - if undirectional
    a2) if undirectional then just choose the concept network
    b) Rename all of the nodes to L#_#
    c) Add the network to the soma/total network and add an edge from the soma to the starting node
    (do so for all)

    3) Then take a subgraph of the concept network based on the nodes you want
    4) Send the subgraph to a function that graphs the networkx graph

    
    """

    

    current_neuron = copy.deepcopy(input_neuron)
    
    if limb_soma_touch_dictionary is None:
        limb_soma_touch_dictionary=dict()
    elif type(limb_soma_touch_dictionary) == dict:
        pass
    elif limb_soma_touch_dictionary == "all":
        limb_soma_touch_dictionary = dict([(limb_idx,xu.get_neighbors(current_neuron.concept_network,limb_idx,int_label=False)) for limb_idx in current_neuron.get_limb_node_names()])
    else:
        raise Exception(f"Recieved invalid input for  limb_soma_touch_dictionary: {limb_soma_touch_dictionary}")

    total_network= nx.DiGraph(current_neuron.concept_network.subgraph(current_neuron.get_soma_node_names()))

    for limb_idx in current_neuron.get_limb_node_names():
        if print_flag:
            print(f"Working on Limb: {limb_idx}")


        if limb_idx in limb_soma_touch_dictionary.keys():
            touching_soma = limb_soma_touch_dictionary[limb_idx]
        else:
            touching_soma = []

        curr_limb_obj = current_neuron.concept_network.nodes[limb_label(limb_idx)]["data"]
        curr_network = None
        if not directional:
            curr_network = curr_limb_obj.concept_network
        else:
            if len(touching_soma) > 0:
                """
                For all somas specified: get the network
                1) if this is first one then just copy the network
                2) if not then get the edges and add to existing network
                """
                for starting_soma in touching_soma:
                    # now need to iterate through all touching groups
                    if print_flag:
                        print(f"---Working on soma: {starting_soma}")
                    curr_limb_obj.set_concept_network_directional(starting_soma,soma_group_idx=-1)
                    soma_specific_network = curr_limb_obj.concept_network_directional
                    
                    #Just making sure that curr_network already exists to add things to
                    if curr_network is None:
                        curr_network = copy.deepcopy(soma_specific_network)
                    else:
                        # ---------- Will go through and set the edges and the network data ---------- #
                        
                        #get the edges
                        curr_network.add_edges_from(soma_specific_network.edges())

                        #get the specific starting node for that network and add it to the current one
                        #print(f"For limb_idx {limb_idx}, curr_limb_obj.all_concept_network_data = {curr_limb_obj.all_concept_network_data}")
                        matching_concept_network_data = [k for k in curr_limb_obj.all_concept_network_data if 
                                                         ((soma_label(k["starting_soma"]) == starting_soma) or (["starting_soma"] == starting_soma))]

                        if len(matching_concept_network_data) != 1:
                            raise Exception(f"The concept_network data for the starting soma ({starting_soma}) did not have exactly one match: {matching_concept_network_data}")

                        matching_concept_network_dict = matching_concept_network_data[0]
                        curr_starting_node = matching_concept_network_dict["starting_node"]
                        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]

                        #set the starting coordinate in the concept network
                        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate}}
                        if print_flag:
                            print(f"attrs = {attrs}")
                        xu.set_node_attributes_dict(curr_network,attrs)

            else:
                curr_network = curr_limb_obj.concept_network_directional

        #At this point should have the desired concept network

        #print(curr_network.nodes())
        mapping = dict([(k,f"{limb_label(limb_idx)}_{k}") for k in curr_network.nodes()])
        curr_network = nx.relabel_nodes(curr_network,mapping)
        #print(curr_network.nodes())
#         if print_flag:
#             print(f'current network edges = {curr_network["L0_17"],curr_network["L0_20"]}')


        #need to get all connections from soma to limb:
        soma_to_limb_edges = []
        for soma_connecting_dict in curr_limb_obj.all_concept_network_data:
            soma_to_limb_edges.append((soma_label(soma_connecting_dict["starting_soma"]),
                                      f"{limb_label(limb_idx)}_{soma_connecting_dict['starting_node']}"))

        total_network = nx.compose(total_network,curr_network)
        total_network.add_edges_from(soma_to_limb_edges)
        
        if print_flag:
            print(f'current network edges = {total_network["L0_17"],total_network["L0_20"]}')
        
    if directional:
        return nx.DiGraph(total_network)
    
    return total_network


def whole_neuron_branch_concept_network(input_neuron,
                                    directional=True,
                                    limb_soma_touch_dictionary = "all",
                                with_data_in_nodes = True,
                                    print_flag = True):

    """
    Purpose: To return the entire concept network with all of the limbs and 
    somas connected of an entire neuron

    Arguments:
    input_neuron: neuron object
    directional: If want a directional or undirectional concept_network returned
    limb_soma_touch_dictionary: a dictionary mapping the limb to the starting soma and soma_idx
    you want visualize if directional is chosen

    This will visualize multiple somas and multiple soma touching groups
    Ex:  {1:[{0:[0,1],1:[0]}]})


    Pseudocode:  
    1) Get the soma subnetwork from the concept network of the neuron
    2) For each limb network:
    - if directional: 
    a) if no specific starting soma picked --> use the soma with the smallest index as starting one
    - if undirectional
    a2) if undirectional then just choose the concept network
    b) Rename all of the nodes to L#_#
    c) Add the network to the soma/total network and add an edge from the soma to the starting node
    (do so for all)

    3) Then take a subgraph of the concept network based on the nodes you want
    4) Send the subgraph to a function that graphs the networkx graph


        """



    current_neuron = copy.deepcopy(input_neuron)

    if limb_soma_touch_dictionary is None:
        limb_soma_touch_dictionary=dict()
    elif type(limb_soma_touch_dictionary) == dict:
        #make sure that the limb labels are numbers
        pass
    elif limb_soma_touch_dictionary == "all":
        """
        Pseudocode: 
        Iterate through all of the limbs
            Iterate through all of the soma starting info
                Build the dictionary for all possible touches

        """
        limb_soma_touch_dictionary = limb_to_soma_mapping(current_neuron)
    else:
        raise Exception(f"Recieved invalid input for  limb_soma_touch_dictionary: {limb_soma_touch_dictionary}")

    total_network= nx.DiGraph(current_neuron.concept_network.subgraph(current_neuron.get_soma_node_names()))

    for limb_idx,soma_info_dict in limb_soma_touch_dictionary.items():
        curr_limb = current_neuron[limb_idx]

        curr_network = None
        if not directional:
            curr_network = curr_limb.concept_network
        else:
            for starting_soma,soma_group_info in soma_info_dict.items():
                """
                For all somas specified: get the network
                1) if this is first one then just copy the network
                2) if not then get the edges and add to existing network
                """
                for soma_group_idx in soma_group_info:
                    if print_flag:
                        print(f"---Working on soma: {starting_soma}, group = {soma_group_idx}")
                    curr_limb.set_concept_network_directional(starting_soma,soma_group_idx=soma_group_idx)
                    soma_specific_network = curr_limb.concept_network_directional

                    #Just making sure that curr_network already exists to add things to
                    if curr_network is None:
                        curr_network = copy.deepcopy(soma_specific_network)
                    else:
                        # ---------- Will go through and set the edges and the network data ---------- #

                        #get the edges
                        curr_network.add_edges_from(soma_specific_network.edges())

                        matching_concept_network_dict = curr_limb.get_concept_network_data_by_soma_and_idx(starting_soma,
                                                                                                           soma_group_idx)


                        curr_starting_node = matching_concept_network_dict["starting_node"]
                        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]

                        #set the starting coordinate in the concept network
                        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate}}
                        if print_flag:
                            print(f"attrs = {attrs}")
                        xu.set_node_attributes_dict(curr_network,attrs)
                        
                        


        #At this point should have the desired concept network

        mapping = dict([(k,f"{limb_label(limb_idx)}_{k}") for k in curr_network.nodes()])
        curr_network = nx.relabel_nodes(curr_network,mapping)

        #need to get all connections from soma to limb:
        soma_to_limb_edges = []
        for soma_connecting_dict in curr_limb.all_concept_network_data:
            soma_to_limb_edges.append((soma_label(soma_connecting_dict["starting_soma"]),
                                      f"{limb_label(limb_idx)}_{soma_connecting_dict['starting_node']}"))

        total_network = nx.compose(total_network,curr_network)
        total_network.add_edges_from(soma_to_limb_edges)

#         if print_flag:
#             print(f'current network edges = {total_network["L0_17"],total_network["L0_20"]}')


    if not with_data_in_nodes:
        total_network = xu.copy_G_without_data(total_network)
        
    if directional:
        return nx.DiGraph(total_network)
    

        
    return total_network



def get_limb_names_from_concept_network(concept_network):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    return [k for k in concept_network.nodes() if "L" in k]

def get_soma_names_from_concept_network(concept_network):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    return [k for k in concept_network.nodes() if "S" in k]
    

def return_concept_network(current_neuron):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    if current_neuron.__class__.__name__ == "Neuron":
        curr_concept_network = current_neuron.concept_network
    #elif type(current_neuron) == type(xu.GraphOrderedEdges()):
    elif current_neuron.__class__.__name__ == "GraphOrderedEdges":
        curr_concept_network = current_neuron
    else:
        exception_string = (f"current_neuron not a Neuron object or Graph Ordered Edges instance: {type(current_neuron)}"
                       f"\n {current_neuron.__class__.__name__}"
                       f"\n {xu.GraphOrderedEdges().__class__.__name__}"
                           f"\n {current_neuron.__class__.__name__ == xu.GraphOrderedEdges().__class__.__name__}")
        print(exception_string)
        raise Exception("")
    return curr_concept_network
    

def convert_limb_concept_network_to_neuron_skeleton(curr_concept_network,check_connected_component=True):
    """
    Purpose: To take a concept network that has the branch 
    data within it to the skeleton for that limb
    
    Pseudocode: 
    1) Get the nodes names of the branches 
    2) Order the node names
    3) For each node get the skeletons into an array
    4) Stack the array
    5) Want to check that skeleton is connected component
    
    Example of how to run: 
    full_skeleton = convert_limb_concept_network_to_neuron_skeleton(recovered_neuron.concept_network.nodes["L1"]["data"].concept_network)
    
    """
    sorted_nodes = np.sort(list(curr_concept_network.nodes()))
    #print(f"sorted_nodes = {sorted_nodes}")
    full_skeleton = sk.stack_skeletons([curr_concept_network.nodes[k]["data"].skeleton for k in sorted_nodes])
    if check_connected_component:
        sk.check_skeleton_one_component(full_skeleton)
    return full_skeleton
    
def get_starting_info_from_concept_network(concept_networks):
    """
    Purpose: To turn a dictionary that maps the soma indexes to a concept map
    into just a list of dictionaries with all the staring information
    
    Ex input:
    concept_networks = {0:concept_network, 1:concept_network,}
    
    Ex output:
    [dict(starting_soma=..,starting_node=..
            starting_endpoints=...,starting_coordinate=...,touching_soma_vertices=...)]
    
    Pseudocode: 
    1) get the soma it's connect to
    2) get the node that has the starting coordinate 
    3) get the endpoints and starting coordinate for that nodes
    """
    
    
    output_dicts = []
    for current_soma,curr_concept_network_list in concept_networks.items():
        for curr_concept_network in curr_concept_network_list:
            curr_output_dict = dict()
            # 1) get the soma it's connect to
            curr_output_dict["starting_soma"] = current_soma

            # 2) get the node that has the starting coordinate 
            starting_node = xu.get_starting_node(curr_concept_network)
            curr_output_dict["starting_node"] = starting_node

            endpoints_dict = xu.get_node_attributes(curr_concept_network,attribute_name="endpoints",node_list=[starting_node],
                           return_array=False)

            curr_output_dict["starting_endpoints"] = endpoints_dict[starting_node]

            starting_node_dict = xu.get_node_attributes(curr_concept_network,attribute_name="starting_coordinate",node_list=[starting_node],
                           return_array=False)
            #get the starting coordinate of the starting dict
            curr_output_dict["starting_coordinate"] = starting_node_dict[starting_node]

            if "touching_soma_vertices" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["touching_soma_vertices"] = curr_concept_network.nodes[starting_node]["touching_soma_vertices"]
            else:
                curr_output_dict["touching_soma_vertices"] = None
                
            #soma starting group
            if "soma_group_idx" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["soma_group_idx"] = curr_concept_network.nodes[starting_node]["soma_group_idx"]
            else:
                curr_output_dict["soma_group_idx"] = None
                
            

            curr_output_dict["concept_network"] = curr_concept_network
            output_dicts.append(curr_output_dict)
    
    return output_dicts


def convert_concept_network_to_skeleton(curr_concept_network):
    #get the midpoints
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])
    curr_edges = curr_concept_network.edges()
    graph_nodes_skeleton = np.array([(node_locations[n1],node_locations[n2]) for n1,n2 in curr_edges]).reshape(-1,2,3)
    return graph_nodes_skeleton


def convert_concept_network_to_undirectional(concept_network):
    return nx.Graph(concept_network)

def convert_concept_network_to_directional(concept_network,
                                        node_widths=None,
                                          no_cycles=True,
                                          suppress_disconnected_errors=False,
                                          verbose=False):
    """
    Pseudocode: 
    0) Create a dictionary with the keys as all the nodes and empty list as values
    1) Get the starting node
    2) Find all neighbors of starting node
    2b) Add the starting node to the list of all the nodes it is neighbors to
    3) Add starter node to the "procesed_nodes" so it is not processed again
    4) Add each neighboring node to the "to_be_processed" list

    5) Start loop that will continue until "to_be_processed" is done
    a. Get the next node to be processed
    b. Get all neighbors
    c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
    --> add the curr_node to those neighbor nodes lists
    d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
    ...
    z. when no more nodes in to_be_processed list then reak

    6) if the no_cycles option is selected:
    - for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    7) convert the incoming edges dictionary to edge for a directional graph
    
    Example of how to use: 
    
    example_concept_network = nx.from_edgelist([[1,2],[2,3],[3,4],[4,5],[2,5],[2,6]])
    nx.draw(example_concept_network,with_labels=True)
    plt.show()
    xu.set_node_attributes_dict(example_concept_network,{1:dict(starting_coordinate=np.array([1,2,3]))})

    directional_ex_concept_network = nru.convert_concept_network_to_directional(example_concept_network,no_cycles=True)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()

    node_widths = {1:0.5,2:0.61,3:0.73,4:0.88,5:.9,6:0.4}
    directional_ex_concept_network = nru.convert_concept_network_to_directional(example_concept_network,no_cycles=True,node_widths=node_widths)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()
    """

    curr_limb_concept_network = concept_network
    mesh_widths = node_widths

    #if only one node in concept_network then return
    if len(curr_limb_concept_network.nodes()) <= 1:
        if verbose:
            print("Concept graph size was 1 or less so returning original")
        return nx.DiGraph(curr_limb_concept_network)

    #0) Create a dictionary with the keys as all the nodes and empty list as values
    incoming_edges_to_node = dict([(k,[]) for k in curr_limb_concept_network.nodes()])
    to_be_processed_nodes = []
    processed_nodes = []
    max_iterations = len(curr_limb_concept_network.nodes()) + 100

    #1) Get the starting node 
    starting_node = xu.get_starting_node(curr_limb_concept_network)

    #2) Find all neighbors of starting node
    curr_neighbors = xu.get_neighbors(curr_limb_concept_network,starting_node)

    #2b) Add the starting node to the list of all the nodes it is neighbors to
    for cn in curr_neighbors:
        incoming_edges_to_node[cn].append(starting_node)

    #3) Add starter node to the "procesed_nodes" so it is not processed again
    processed_nodes.append(starting_node)

    #4) Add each neighboring node to the "to_be_processed" list
    to_be_processed_nodes.extend([k for k in curr_neighbors if k not in processed_nodes ])
    # print(f"incoming_edges_to_node AT START= {incoming_edges_to_node}")
    # print(f"processed_nodes_AT_START = {processed_nodes}")
    # print(f"to_be_processed_nodes_AT_START = {to_be_processed_nodes}")

    #5) Start loop that will continue until "to_be_processed" is done
    for i in range(max_iterations):
    #     print("\n")
    #     print(f"processed_nodes = {processed_nodes}")
    #     print(f"to_be_processed_nodes = {to_be_processed_nodes}")

        if len(to_be_processed_nodes) == 0:
            break
        #a. Get the next node to be processed
        
        curr_node = to_be_processed_nodes.pop(0)
        #print(f"curr_node = {curr_node}")
        #b. Get all neighbors
        curr_node_neighbors = xu.get_neighbors(curr_limb_concept_network,curr_node)
        #print(f"curr_node_neighbors = {curr_node_neighbors}")
        #c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
        #--> add the curr_node to those neighbor nodes lists
        for cn in curr_node_neighbors:
            if cn == curr_node:
                raise Exception("found a self connection in network graph")
            if cn not in incoming_edges_to_node[curr_node]:
                incoming_edges_to_node[cn].append(curr_node)

            #d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
            if cn not in to_be_processed_nodes and cn not in processed_nodes:
                to_be_processed_nodes.append(cn)


        # add the nodes to those been processed
        processed_nodes.append(curr_node)


        #z. when no more nodes in to_be_processed list then reak
        

    #print(f"incoming_edges_to_node = {incoming_edges_to_node}")
    #6) if the no_cycles option is selected:
    #- for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    incoming_lengths = [k for k,v in incoming_edges_to_node.items() if len(v) >= 1]
    
    if not suppress_disconnected_errors:
        if len(incoming_lengths) != len(curr_limb_concept_network.nodes())-1:
            raise Exception("after loop in directed concept graph, not all nodes have incoming edges (except starter node)")

    if no_cycles == True:
        if verbose:
            print("checking and resolving cycles")
        #get the nodes with multiple incoming edges
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) >= 2])


        if len(multi_incoming) > 0:
            if verbose:
                print("There are loops to resolve and 'no_cycles' parameters set requires us to fix eliminate them")
            #find the mesh widths of all the incoming edges and the current edge

            #if mesh widths are available then go that route
            if not mesh_widths is None:
                if verbose:
                    print("Using mesh_widths for resolving loops")
                for curr_node,incoming_nodes in multi_incoming.items():
                    curr_node_width = mesh_widths[curr_node]
                    incoming_nodes_width_difference = [np.linalg.norm(mesh_widths[k]- curr_node_width) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmin(incoming_nodes_width_difference).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
            else: #if not mesh widths available then just pick the longest edge
                """
                Get the coordinates of all of the nodes
                """
                node_coordinates_dict = xu.get_node_attributes(curr_limb_concept_network,attribute_name="coordinates",return_array=False)
                if set(list(node_coordinates_dict.keys())) != set(list(incoming_edges_to_node.keys())):
                    if verbose:
                        print("The keys of the concept graph with 'coordinates' do not match the keys of the edge dictionary")
                        print("Just going to use the first incoming edge by default")
                    for curr_node,incoming_nodes in multi_incoming.items():
                        winning_incoming_node = incoming_nodes[0]
                        incoming_edges_to_node[curr_node] = [winning_incoming_node]
                else: #then have coordinate information
                    if verbose:
                        print("Using coordinate distance to pick the winning node")
                    curr_node_coordinate = node_coordinates_dict[curr_node]
                    incoming_nodes_distance = [np.linalg.norm(node_coordinates_dict[k]- curr_node_coordinate) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmax(incoming_nodes_distance).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
        else:
            if verbose:
                print("No cycles to fix")


        #check that all have length of 1
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) == 1])
        
        if not suppress_disconnected_errors:
            if len(multi_incoming) != len(curr_limb_concept_network.nodes()) - 1:
                raise Exception("Inside the no_cycles but at the end all of the nodes only don't have one incoming cycle"
                               f"multi_incoming = {multi_incoming}")

    #7) convert the incoming edges dictionary to edge for a directional graph
    total_edges = []

    if no_cycles:
        for curr_node,incoming_nodes in multi_incoming.items():
            curr_incoming_edges = [(j,curr_node) for j in incoming_nodes]
            total_edges += curr_incoming_edges
    else:
        for curr_node,incoming_nodes in incoming_edges_to_node.items():
            curr_incoming_edges = [(j,curr_node) for j in incoming_nodes]
            total_edges += curr_incoming_edges


    #creating the directional network
    curr_limb_concept_network_directional = nx.DiGraph(nx.create_empty_copy(curr_limb_concept_network,with_data=True))
    curr_limb_concept_network_directional.add_edges_from(total_edges)

    return curr_limb_concept_network_directional


def branches_to_concept_network(curr_branch_skeletons,
                             starting_coordinate,
                              starting_edge,
                                touching_soma_vertices=None,
                                soma_group_idx=None,
                                starting_soma=None,
                             max_iterations= 1000000,
                               verbose=False):
    """
    Will change a list of branches into 
    """
    if verbose:
        print(f"Starting_edge inside branches_to_conept = {starting_edge}")
    
    start_time = time.time()
    processed_nodes = []
    edge_endpoints_to_process = []
    concept_network_edges = []

    """
    If there is only one branch then just pass back a one-node graph 
    with no edges
    """
    if len(curr_branch_skeletons) == 0:
        raise Exception("Passed no branches to be turned into concept network")
    
    if len(curr_branch_skeletons) == 1:
        concept_network = xu.GraphOrderedEdges()
        concept_network.add_node(0)
        
        starting_node = 0
        #print("setting touching_soma_vertices 1")
        attrs = {starting_node:{"starting_coordinate":starting_coordinate,
                                "endpoints":neuron.Branch(starting_edge).endpoints,
                               "touching_soma_vertices":touching_soma_vertices,
                                "soma_group_idx":soma_group_idx,
                               "starting_soma":starting_soma}
                                }
        
        xu.set_node_attributes_dict(concept_network,attrs)
        #print(f"Recovered touching vertices after 1 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
        
        #add the endpoints 
        return concept_network

    # 0) convert each branch to one segment and build a graph from it
    
    
    # 8-29 debug
    #curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in curr_branch_skeletons]
    curr_branch_meshes_downsampled = []
    for i,b in enumerate(curr_branch_skeletons):
        try:
            curr_branch_meshes_downsampled.append(sk.resize_skeleton_branch(b,n_segments=1))
        except:
            if verbose:
                print(f"The following branch {i} could not be downsampled: {b}")
            raise Exception("not downsampled branch")
        
    
    """
    In order to solve the problem that once resized there could be repeat edges
    
    Pseudocode: 
    1) predict the branches that are repeats and then create a map 
    of the non-dom (to be replaced) and dominant (the ones to replace)
    2) Get an arange list of the branch idxs and then delete the non-dominant ones
    3) Run the whole concept map process
    4) At the end for each non-dominant one, at it in (with it's idx) and copy
    the edges of the dominant one that it was mapped to
    
    
    """
    
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    # curr_sk_graph_debug = sk.convert_skeleton_to_graph_old(downsampled_skeleton)
    # nx.draw(curr_sk_graph_debug,with_labels = True)

    #See if touching row matches the original: 
    

    all_skeleton_vertices = downsampled_skeleton.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    
    reshaped_indices = np.sort(indices.reshape(-1,2),axis=1)
    unique_edges,unique_edges_indices = np.unique(reshaped_indices,axis = 0,return_inverse=True)
    from collections import Counter
    multiplicity_edge_counter = dict(Counter(unique_edges_indices))
    #this will give the unique edge that appears multiple times
    duplicate_edge_identifiers = [k for k,v in multiplicity_edge_counter.items() if v > 1] 
    
    #for keeping track of original indexes
    original_idxs = np.arange(0,len(curr_branch_meshes_downsampled))
    
    """
    This will delete any branches that have the same two common endpoints
    """
    if len(duplicate_edge_identifiers) > 0:
        if verbose:
            print(f"There were {len(duplicate_edge_identifiers)} duplication nodes found")
        all_conn_comp = []
        for d in duplicate_edge_identifiers:
            all_conn_comp.append(list(np.where(unique_edges_indices == [d] )[0]))

        domination_map = dict()
        for curr_comp in all_conn_comp:
            dom_node = curr_comp[0]
            non_dom_nodes = curr_comp[1:]
            for n_dom in non_dom_nodes:
                domination_map[n_dom] = dom_node
        if verbose:
            print(f"domination_map = {domination_map}")
        

        to_delete_rows = list(domination_map.keys())

        #delete all of the non dominant rows from the indexes and the skeletons
        original_idxs = np.delete(original_idxs,to_delete_rows,axis=0)
        curr_branch_meshes_downsampled = [k for i,k in enumerate(curr_branch_meshes_downsampled) if i not in to_delete_rows]
    
    #print(f"curr_branch_meshes_downsampled[24] = {curr_branch_meshes_downsampled[24]}")
    curr_stacked_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    #print(f"curr_stacked_skeleton[24] = {curr_stacked_skeleton[24]}")

    branches_graph = sk.convert_skeleton_to_graph(curr_stacked_skeleton) #can recover the original skeleton
#     print(f"len(curr_stacked_skeleton) = {len(curr_stacked_skeleton)}")
#     print(f"len(branches_graph.edges_ordered()) = {len(branches_graph.edges_ordered())}")
#     print(f"(branches_graph.edges_ordered())[24] = {(branches_graph.edges_ordered())[24]}")
#     print(f"coordinates = (branches_graph.edges_ordered())[24] = {xu.get_node_attributes(branches_graph,node_list=(branches_graph.edges_ordered())[24])}")


    #************************ need to just make an edges lookup dictionary*********#


    #1) Identify the starting node on the starting branch
    starting_node = xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=starting_coordinate))
    
    if verbose:
        print(f"At the start, starting_node (in terms of the skeleton, that shouldn't match the starting edge) = {starting_node}")
    if len(starting_node) != 1:
        raise Exception(f"The number of starting nodes found was not exactly one: {starting_node}")
    #1b) Add all edges incident and their other node label to a list to check (add the first node to processed nodes list)
    incident_edges = xu.node_to_edges(branches_graph,starting_node)
    #print(f"incident_edges = {incident_edges}")
    # #incident_edges_idx = edge_to_index(incident_edges)

    # #adding them to the list to be processed (gets the edge and the downstream edge)
    edge_endpoints_to_process = [(edges,edges[edges != starting_node ]) for edges in incident_edges]
    processed_nodes.append(starting_node)

    #need to add all of the newly to look edges and the current edge to the concept_network_edges
    """
    Pseudocode: 
    1) convert starting edge to the node identifiers
    2) iterate through all the edges to process and add the combos where the edge does not match
    """
    edge_coeff= []
    for k in starting_edge:
        edge_coeff.append(xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=k))[0])
    
    
    for curr_edge,edge_enpt in edge_endpoints_to_process:
        if not np.array_equal(np.sort(curr_edge),np.sort(edge_coeff)):
            #add to the concept graph
            concept_network_edges += [(np.array(curr_edge),np.array(edge_coeff))]
        else:
            starting_node_edge = curr_edge
            if verbose:
                print("printing out current edge:")
                print(xu.get_node_attributes(branches_graph,node_list=starting_node_edge))
        
    
    for i in range(max_iterations):
        #print(f"==\n\n On iteration {i}==")
        if len(edge_endpoints_to_process) == 0:
            if verbose:
                print(f"edge_endpoints_to_process was empty so exiting loop after {i} iterations")
            break

        #2) Pop the edge edge number,endpoint of the stack
        edge,endpt = edge_endpoints_to_process.pop(0)
        #print(f"edge,endpt = {(edge,endpt)}")
        #- if edge already been processed then continue
        if endpt in processed_nodes:
            #print(f"Already processed endpt = {endpt} so skipping")
            continue
        #a. Find all edges incident on this node
        incident_edges = xu.node_to_edges(branches_graph,endpt)
        #print(f"incident_edges = {incident_edges}")

        considering_edges = [k for k in incident_edges if not np.array_equal(k,edge) and not np.array_equal(k,np.flip(edge))]
        #print(f"considering_edges = {considering_edges}")
        #b. Create edges from curent edge to those edges incident with it
        concept_network_edges += [(edge,k) for k in considering_edges]

        #c. Add the current node as processed
        processed_nodes.append(endpt)

        #d. For each edge incident add the edge and the other connecting node to the list
        new_edge_processing = [(e,e[e != endpt ]) for e in considering_edges]
        edge_endpoints_to_process = edge_endpoints_to_process + new_edge_processing
        #print(f"edge_endpoints_to_process = {edge_endpoints_to_process}")

    if len(edge_endpoints_to_process)>0:
        raise Exception(f"Reached max_interations of {max_iterations} and the edge_endpoints_to_process not empty")

    #flattening the connections so we can get the indexes of these edges
    flattened_connections = np.array(concept_network_edges).reshape(-1,2)
    
    orders = xu.get_edge_attributes(branches_graph,edge_list=flattened_connections)
    #******
    
    fixed_idx_orders = original_idxs[orders]
    concept_network_edges_fixed = np.array(fixed_idx_orders).reshape(-1,2)

    
    # # edge_endpoints_to_process
    #print(f"concept_network_edges_fixed = {concept_network_edges_fixed}")
    concept_network = xu.GraphOrderedEdges()
    #print("type(concept_network) = {type(concept_network)}")
    concept_network.add_edges_from([k for k in concept_network_edges_fixed])
    
    #add the endpoints as attributes to each of the nodes
    node_endpoints_dict = dict()
    old_ordered_edges = branches_graph.edges_ordered()
    for edge_idx,curr_branch_graph_edge in enumerate(old_ordered_edges):
        new_edge_idx = original_idxs[edge_idx]
        curr_enpoints = np.array(xu.get_node_attributes(branches_graph,node_list=curr_branch_graph_edge)).reshape(-1,3)
        node_endpoints_dict[new_edge_idx] = dict(endpoints=curr_enpoints)
        xu.set_node_attributes_dict(concept_network,node_endpoints_dict)
    
    
    
    
    #add the starting coordinate to the corresponding node
    #print(f"starting_node_edge right before = {starting_node_edge}")
    starting_order = xu.get_edge_attributes(branches_graph,edge_list=[starting_node_edge]) 
    #print(f"starting_order right before = {starting_order}")
    if len(starting_order) != 1:
        raise Exception(f"Only one starting edge index was not found,starting_order={starting_order} ")
    
    starting_edge_index = original_idxs[starting_order[0]]
    if verbose:
        print(f"starting_node in concept map (that should match the starting edge) = {starting_edge_index}")
    #attrs = {starting_node[0]:{"starting_coordinate":starting_coordinate}} #old way that think uses the wrong starting_node
    attrs = {starting_edge_index:{"starting_coordinate":starting_coordinate,"touching_soma_vertices":touching_soma_vertices,"soma_group_idx":soma_group_idx,"starting_soma":starting_soma}} 
    #print("setting touching_soma_vertices 2")
    xu.set_node_attributes_dict(concept_network,attrs)
    #print(f"Recovered touching vertices after 2 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
    
    #want to set all of the edge endpoints on the nodes as well just for a check
    
    
    if verbose:
        print(f"Total time for branches to concept conversion = {time.time() - start_time}\n")
    
    
    # Add back the nodes that were deleted
    if len(duplicate_edge_identifiers) > 0:
        if verbose:
            print("Working on adding back the edges that were duplicates")
        for non_dom,dom in domination_map.items():
            #print(f"Re-adding: {non_dom}")
            #get the endpoints attribute
            # local_node_endpoints_dict
            
            curr_neighbors = xu.get_neighbors(concept_network,dom)  
            new_edges = np.vstack([np.ones(len(curr_neighbors))*non_dom,curr_neighbors]).T
            concept_network.add_edges_from(new_edges)
            
            curr_endpoint = xu.get_node_attributes(concept_network,attribute_name="endpoints",node_list=[dom])[0]
            #print(f"curr_endpoint in add back = {curr_endpoint}")
            add_back_attribute_dict = {non_dom:dict(endpoints=curr_endpoint)}
            #print(f"To add dict = {add_back_attribute_dict}")
            xu.set_node_attributes_dict(concept_network,add_back_attribute_dict)
            
    return concept_network



""" Older function definition
def generate_limb_concept_networks_from_global_connectivity(
    limb_idx_to_branch_meshes_dict,
    limb_idx_to_branch_skeletons_dict,
    soma_idx_to_mesh_dict,
    soma_idx_connectivity,
    current_neuron,
    return_limb_labels=True
    ): 
"""

def check_concept_network(curr_limb_concept_network,closest_endpoint,
                          curr_limb_divided_skeletons,print_flag=False,
                         return_touching_piece=True,
                         verbose=False):
    
    recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=closest_endpoint))
    
    
    if verbose:
        print(f"recovered_touching_piece = {recovered_touching_piece}")
        print(f"After concept mapping size = {len(curr_limb_concept_network.nodes())}")
    if len(curr_limb_concept_network.nodes()) != len(curr_limb_divided_skeletons):
        raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                      f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(curr_limb_divided_skeletons)}")
    if nx.number_connected_components(curr_limb_concept_network) > 1:
        raise Exception("There was more than 1 connected components in the concept network")


    for j,un_resized_b in enumerate(curr_limb_divided_skeletons):
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
    if return_touching_piece:
        return recovered_touching_piece
            
#for finding the closest endpoint
def generate_limb_concept_networks_from_global_connectivity(
        limb_correspondence,
        soma_meshes,
        soma_idx_connectivity,
        current_neuron,
        limb_to_soma_starting_endpoints = None,
        return_limb_labels=True
        ):
    
    
    """
    ****** Could significantly speed this up if better picked the 
    periphery meshes (which now are sending all curr_limb_divided_meshes)
    sent to 
    
    tu.mesh_pieces_connectivity(main_mesh=current_neuron,
                                        central_piece = curr_soma_mesh,
                                    periphery_pieces=curr_limb_divided_meshes)
    *********
    
    
    Purpose: To create concept networks for all of the skeletons
             based on our knowledge of the mesh

        Things it needs: 
        - branch_mehses
        - branch skeletons
        - soma meshes
        - whole neuron
        - soma_to_piece_connectivity

        What it returns:
        - concept networks
        - branch labels
        
    
    Pseudocode: 
    1) Get all of the meshes for that limb (that were decomposed)
    2) Use the entire neuron, the soma meshes and the list of meshes and find out shich one is touching the soma
    3) With the one that is touching the soma, find the enpoints of the skeleton
    4) Find the closest matching endpoint
    5) Send the deocmposed skeleton branches to the branches_to_concept_network function
    6) Graph the concept graph using the mesh centers

    Example of Use: 
    
    from neurd import neuron
    neuron = reload(neuron)

    #getting mesh and skeleton dictionaries
    limb_idx_to_branch_meshes_dict = dict()
    limb_idx_to_branch_skeletons_dict = dict()
    for k in limb_correspondence.keys():
        limb_idx_to_branch_meshes_dict[k] = [limb_correspondence[k][j]["branch_mesh"] for j in limb_correspondence[k].keys()]
        limb_idx_to_branch_skeletons_dict[k] = [limb_correspondence[k][j]["branch_skeleton"] for j in limb_correspondence[k].keys()]      

    #getting the soma dictionaries
    soma_idx_to_mesh_dict = dict()
    for k,v in enumerate(current_mesh_data[0]["soma_meshes"]):
        soma_idx_to_mesh_dict[k] = v

    soma_idx_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"]


    limb_concept_networkx,limb_labels = neuron.generate_limb_concept_networks_from_global_connectivity(
        limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        soma_idx_connectivity = soma_idx_connectivity,
        current_neuron=current_neuron,
        return_limb_labels=True
        )
    

    """
    print("********************************** generate_limb_concept_networks_from_global_connectivity****************************")
    # ------------ 7/17 Added preprocessing Step so can give the function more generic arguments ---------------- #
    #getting mesh and skeleton dictionaries
    limb_idx_to_branch_meshes_dict = dict()
    limb_idx_to_branch_skeletons_dict = dict()
    for k in limb_correspondence.keys():
        limb_idx_to_branch_meshes_dict[k] = [limb_correspondence[k][j]["branch_mesh"] for j in limb_correspondence[k].keys()]
        limb_idx_to_branch_skeletons_dict[k] = [limb_correspondence[k][j]["branch_skeleton"] for j in limb_correspondence[k].keys()]      

    #getting the soma dictionaries
    soma_idx_to_mesh_dict = dict()
    for k,v in enumerate(soma_meshes):
        soma_idx_to_mesh_dict[k] = v




    
    
    
    

    
    
    if set(list(limb_idx_to_branch_meshes_dict.keys())) != set(list(limb_idx_to_branch_skeletons_dict.keys())):
        raise Exception("There was a difference in the keys for the limb_idx_to_branch_meshes_dict and limb_idx_to_branch_skeletons_dict")
        
    global_concept_time = time.time()
    
    total_limb_concept_networks = dict()
    total_limb_labels = dict()
    soma_mesh_faces = dict()
    for limb_idx in limb_idx_to_branch_meshes_dict.keys():
        local_concept_time = time.time()
        print(f"\n\n------Working on limb {limb_idx} -------")
        curr_concept_network = dict()
        
        curr_limb_divided_meshes = limb_idx_to_branch_meshes_dict[limb_idx]
        #curr_limb_divided_meshes_idx = [v["branch_face_idx"] for v in limb_correspondence[limb_idx].values()]
        curr_limb_divided_skeletons = limb_idx_to_branch_skeletons_dict[limb_idx]
        print(f"inside loop len(curr_limb_divided_meshes) = {len(curr_limb_divided_meshes)}"
             f" len(curr_limb_divided_skeletons) = {len(curr_limb_divided_skeletons)}")
        
        #find what mesh piece was touching
        touching_soma_indexes = []
        for k,v in soma_idx_connectivity.items():
            if limb_idx in v:
                touching_soma_indexes.append(k)
        
        if len(touching_soma_indexes) == 0:
            raise Exception("Did not find touching soma index")
        if len(touching_soma_indexes) >= 2:
            print("Merge limb detected")
            
        
        for soma_idx in touching_soma_indexes:
            print(f"--- Working on soma_idx: {soma_idx}----")
            curr_soma_mesh = soma_idx_to_mesh_dict[soma_idx]
            
            
            
            if soma_idx in list(soma_mesh_faces.keys()):
                soma_info = soma_mesh_faces[soma_idx]
            else:
                soma_info = curr_soma_mesh
                
            #filter the periphery pieces
            original_idxs = np.arange(0,len(curr_limb_divided_meshes))
            
            periph_filter_time = time.time()
            distances_periphery_to_soma = np.array([tu.closest_distance_between_meshes(curr_soma_mesh,k) for k in curr_limb_divided_meshes])
            periphery_distance_threshold = 2000
            
            original_idxs = original_idxs[distances_periphery_to_soma<periphery_distance_threshold]
            filtered_periphery_meshes = np.array(curr_limb_divided_meshes)[distances_periphery_to_soma<periphery_distance_threshold]
            
            print(f"Total time for filtering periphery meshes = {time.time() - periph_filter_time}")
            periph_filter_time = time.time()
            
            if len(filtered_periphery_meshes) == 0:
                raise Exception("There were no periphery meshes within a threshold distance of the mesh")
            

            touching_pieces,touching_vertices,central_piece_faces = tu.mesh_pieces_connectivity(main_mesh=current_neuron,
                                        central_piece = soma_info,
                                        periphery_pieces = filtered_periphery_meshes,
                                                         return_vertices=True,
                                                        return_central_faces=True
                                                                                 )
            soma_mesh_faces[soma_idx] = central_piece_faces
            
            #fixing the indexes so come out the same
            touching_pieces = original_idxs[touching_pieces]
            print(f"touching_pieces = {touching_pieces}")
            print(f"Total time for mesh connectivity = {time.time() - periph_filter_time}")
            
            
            if len(touching_pieces) >= 2:
                print("**More than one touching point to soma, touching_pieces = {touching_pieces}**")
                
                """ 9/17: Want to pick the one with the starting endpoint if exists
                Pseudocode: 
                1) Get the starting endpoint if exists
                2) Get the endpoints of all the touching branches
                3) Get the index (if any ) of the branch that has this endpoint in skeleton
                4) Make that the winning index
                
                
                
                """
                if not limb_to_soma_starting_endpoints is None:
                    print("Using new winning piece based on starting coordinate")
                    ideal_starting_coordinate = limb_to_soma_starting_endpoints[limb_idx][soma_idx]
                    touching_pieces_branches = [neuron.Branch(curr_limb_divided_skeletons[k]).endpoints for k in touching_pieces]
                    print("trying to use new find_branch_skeleton_with_specific_coordinate")
                    winning_piece_idx = sk.find_branch_skeleton_with_specific_coordinate(touching_pieces_branches,
                                                                  current_coordinate=ideal_starting_coordinate)[0]
                    
                else:
                    # picking the piece with the most shared vertices
                    len_touch_vertices = [len(k) for k in touching_vertices]
                    winning_piece_idx = np.argmax(len_touch_vertices)
                    
                print(f"winning_piece_idx = {winning_piece_idx}")
                touching_pieces = [touching_pieces[winning_piece_idx]]
                print(f"Winning touching piece = {touching_pieces}")
                touching_pieces_soma_vertices = touching_vertices[winning_piece_idx]
            else:
                touching_pieces_soma_vertices = touching_vertices[0]
            if len(touching_pieces) < 1:
                raise Exception("No touching pieces")
            
            #print out the endpoints of the winning touching piece
            
                
            #3) With the one that is touching the soma, find the enpoints of the skeleton
            print(f"Using touching_pieces[0] = {touching_pieces[0]}")
            touching_branch = neuron.Branch(curr_limb_divided_skeletons[touching_pieces[0]])
            endpoints = touching_branch.endpoints
            
            
            """  OLDER WAY OF FINDING STARTING ENDPOINT WHERE JUST COMPARES TO SOMA CENTER
            print(f"Touching piece endpoints = {endpoints}")
            soma_midpoint = np.mean(curr_soma_mesh.vertices,axis=0)

            #4) Find the closest matching endpoint
            closest_idx = np.argmin([np.linalg.norm(soma_midpoint-k) for k in endpoints])
            closest_endpoint = endpoints[closest_idx]
            
            """
            

            closest_endpoint = None
            if not limb_to_soma_starting_endpoints is None:
                """  # -----------  9/16 -------------- #
                Will pick the starting coordinate that was given if it was on the winning piece
                """
                ideal_starting_coordinate = limb_to_soma_starting_endpoints[limb_idx][soma_idx]
                endpoints_list = endpoints.reshape(-1,3)
                match_result = nu.matching_rows(endpoints_list,ideal_starting_coordinate)
                if len(match_result)>0:
                    closest_endpoint = endpoints_list[match_result[0]]
            if closest_endpoint is None:
                """  # -----------  9/1 -------------- #
                New method for finding 
                1) Build a KDTree of the winning touching piece soma boundary vertices
                2) query the endpoints against the vertices
                3) pick the endpoint that has the closest match
                """
                ex_branch_KDTree = KDTree(touching_pieces_soma_vertices)
                distances,closest_nodes = ex_branch_KDTree.query(endpoints)
                closest_endpoint = endpoints[np.argmin(distances)]
            
            

            
            
            
            print(f"inside inner loop "
             f"len(curr_limb_divided_skeletons) = {len(curr_limb_divided_skeletons)}")
            print(f"closest_endpoint WITH NEW KDTREE METHOD= {closest_endpoint}")
            
            print(f"About to send touching_soma_vertices = {touching_pieces_soma_vertices}")
            curr_limb_concept_network = branches_to_concept_network(curr_limb_divided_skeletons,closest_endpoint,np.array(endpoints).reshape(-1,3),
                                                                   touching_soma_vertices=touching_pieces_soma_vertices)
            
            #print(f"Recovered touching vertices = {xu.get_all_nodes_with_certain_attribute_key(curr_limb_concept_network,'touching_soma_vertices')}")
            curr_concept_network[soma_idx] = curr_limb_concept_network
            
            
            # ----- Some checks that make sure concept mapping went well ------ #
            #get the node that has the starting coordinate:
            recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=closest_endpoint))
            
            print(f"recovered_touching_piece = {recovered_touching_piece}")
            if recovered_touching_piece[0] != touching_pieces[0]:
                raise Exception(f"For limb {limb_idx} and soma {soma_idx} the recovered_touching and original touching do not match\n"
                               f"recovered_touching_piece = {recovered_touching_piece}, original_touching_pieces = {touching_pieces}")
                                                                         

            print(f"After concept mapping size = {len(curr_limb_concept_network.nodes())}")
            
            if len(curr_limb_concept_network.nodes()) != len(curr_limb_divided_skeletons):
                   raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                                  f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(curr_limb_divided_skeletons)}")

            if nx.number_connected_components(curr_limb_concept_network) > 1:
                raise Exception("There was more than 1 connected components in the concept network")
            
#             #for debugging: 
#             endpoints_dict = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",return_array=False)
#             print(f"endpoints_dict = {endpoints_dict}")
#             print(f"{curr_limb_concept_network.nodes()}")
            
            
            #make sure that the original divided_skeletons endpoints match the concept map endpoints
            for j,un_resized_b in enumerate(curr_limb_divided_skeletons):
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
                
                
        total_limb_concept_networks[limb_idx] = curr_concept_network
        
        if len(curr_concept_network) > 1:
            total_limb_labels[limb_idx] = "MergeError"
        else:
            total_limb_labels[limb_idx] = "Normal"
            
        print(f"Local time for concept mapping = {time.time() - local_concept_time}")

    print(f"\n\n ----- Total time for concept mapping = {time.time() - global_concept_time} ----")
        
        
    #returning from the function    
        
    if return_limb_labels:
        return total_limb_concept_networks,total_limb_labels
    else:
        return total_limb_concept_networks

# ----------------------- End of Concept Networks ------------------------------------- #
    
    

# -----------------------  For the compression of a neuron object ---------------------- #
def find_face_idx_and_check_recovery(original_mesh,submesh_list,print_flag=False,check_recovery=True):
    debug = False
    if len(submesh_list) == 0:
        if print_flag:
            print("Nothing in submesh_list sent to find_face_idx_and_check_recovery so just returning empty list")
            return []
    submesh_list_face_idx = []
    for i,sm in enumerate(submesh_list):
        
        sm_faces_idx = tu.original_mesh_faces_map(original_mesh=original_mesh, 
                                   submesh=sm,
                               matching=True,
                               print_flag=False,
                                                 exact_match=True)
        submesh_list_face_idx.append(sm_faces_idx)
        if debug:
            print(f"For submesh {i}: sm.faces.shape = {sm.faces.shape}, sm_faces_idx.shape = {sm_faces_idx.shape}")
        
    if check_recovery:
        recovered_submesh_meshes = [original_mesh.submesh([sm_f],append=True,repair=False) for sm_f in submesh_list_face_idx]
        #return recovered_submesh_meshes

    
        for j,(orig_sm,rec_sm) in enumerate(zip(submesh_list,recovered_submesh_meshes)):
            result = tu.compare_meshes_by_face_midpoints(orig_sm,rec_sm,print_flag=False)
            if not result:
                tu.compare_meshes_by_face_midpoints(orig_sm,rec_sm,print_flag=True)
                raise Exception(f"Submesh {j} was not able to be accurately recovered")
    
    return submesh_list_face_idx
    




def smaller_preprocessed_data(neuron_object,print_flag=False):
    double_soma_obj = neuron_object
    
    total_compression_time = time.time()
    
    
    
    
    
    # doing the soma recovery more streamlined
    compression_time = time.time()
    soma_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["soma_meshes"])
    if print_flag:
        print(f"Total time for soma meshes compression = {time.time() - compression_time }")
    compression_time = time.time()
    #insignificant, non_soma touching and inside pieces just mesh pieces ()
    insignificant_limbs_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["insignificant_limbs"])
    not_processed_soma_containing_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["not_processed_soma_containing_meshes"])

    inside_pieces_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["inside_pieces"])

    non_soma_touching_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["non_soma_touching_meshes"])
    
    if print_flag:
        print(f"Total time for insignificant_limbs,inside_pieces,non_soma_touching_meshes,not_processed_soma_containing_meshes compression = {time.time() - compression_time }")
    compression_time = time.time()
    
    # recover the limb meshes from the original
    #------------------------------- THERE IS SOME DISCONNECTED IN THE MESH THAT IS IN PREPROCESSED DATA AND THE ACTUAL LIMB MESH ------------------ #
    #------------------------------- MAKES SENSE BECAUSE DOING MESH CORRESPONDENCE AFTERWARDS THAT ALTERS THE MESH BRANCHES A BIT, SO JUST PULL FROM THE MESH_FACES_IDX ------------------ #

    limb_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["limb_meshes"])
    if print_flag:
        print(f"Total time for limb_meshes compression = {time.time() - compression_time }")
    compression_time = time.time()    
    
    # limb_correspondence can get rid of branch mesh and just recover from branch_face_idx
    
    
    
    
    """
    Pseudocode: 
    1) Want to keep skeleton and width
    2) Generate new branch_face_idx based on the original mesh
    --> later can recover the branch_mesh from the whole neuron mesh and the new branch_face_idx
    --> regenerate the and branch_face_idx from the recovered limb mesh and he recovered mesh


    """
    if print_flag:
        print(f"    Starting Limb Correspondence Compression")
    new_limb_correspondence = copy.deepcopy(double_soma_obj.preprocessed_data["limb_correspondence"])
    

    for k in new_limb_correspondence:
        for j in tqdm(new_limb_correspondence[k]):
            new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"] = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=[new_limb_correspondence[k][j]["branch_mesh"]])[0]

            if "branch_face_idx" in new_limb_correspondence[k][j].keys():
                del new_limb_correspondence[k][j]["branch_face_idx"]
            if "branch_mesh" in new_limb_correspondence[k][j].keys():
                del new_limb_correspondence[k][j]["branch_mesh"]
                
    if print_flag:
        print(f"Total time for new_limb_correspondence compression = {time.time() - compression_time }")
    compression_time = time.time() 
    
    
    # all of the data will be 
    soma_meshes_face_idx

    # soma_to_piece_connectivity is already small dictionary
    double_soma_obj.preprocessed_data["soma_to_piece_connectivity"]
    double_soma_obj.preprocessed_data["soma_sdfs"]
    

    insignificant_limbs_face_idx
    inside_pieces_face_idx
    non_soma_touching_meshes_face_idx

    limb_meshes_face_idx

    new_limb_correspondence

    double_soma_obj.preprocessed_data['limb_labels']
    double_soma_obj.preprocessed_data['limb_concept_networks']
    
    """
    Algorithm for how to save off the following: 
    1) width_new (the dictionary where keyword maps to scalar) #save as dict of dict
    2) width_array (the dictionary where keyword maps to array)# 
    3) spines(list or none): 
    4) branch_labels (list): dict of dict
    
    How to store the width_new (just a dictionary for all of the widths)
    width_new_key = just anraveled
    
    """

    
    computed_attribute_dict = double_soma_obj.get_computed_attribute_data()
    
    #geting the labels data
    labels_lookup =double_soma_obj.get_attribute_dict("labels")

    if "soma_volume_ratios" not in double_soma_obj.preprocessed_data.keys():
        double_soma_obj.preprocessed_data["soma_volume_ratios"] = [double_soma_obj[ll].volume_ratio for ll in double_soma_obj.get_soma_node_names()]
        
    if hasattr(double_soma_obj,"original_mesh_idx"):
        original_mesh_idx=double_soma_obj.original_mesh_idx
    else:
        original_mesh_idx = None
    
    soma_names = double_soma_obj.get_soma_node_names()
    
    
    pipeline_products = getattr(double_soma_obj,"pipeline_products",None)
    if pipeline_products is not None:
        pipeline_products = pipeline_products.export()
        
        try:
            meshu.clear_all_mesh_cache_in_nested_data_struct(pipeline_products)
        except Exception as e:
            print(e)
    
    compressed_dict = dict(
                          #saving the original number of faces and vertices to make sure reconstruciton doesn't happen with wrong mesh
                          original_mesh_n_faces = len(double_soma_obj.mesh.faces),
                          original_mesh_n_vertices = len(double_soma_obj.mesh.vertices), 
        
                          soma_meshes_face_idx=soma_meshes_face_idx,

                          soma_to_piece_connectivity=double_soma_obj.preprocessed_data["soma_to_piece_connectivity"],
                          soma_volumes=[double_soma_obj[k]._volume for k in soma_names],
                          
                          # --- 6/9 Addition: Synapses stored off
                        
                          soma_synapses = [syu.synapses_to_exports(getattr(double_soma_obj[k],"synapses",[])) for k in soma_names],
                          distance_errored_synapses = syu.synapses_to_exports(getattr(double_soma_obj,"distance_errored_synapses",[])),
                          mesh_errored_synapses = syu.synapses_to_exports(getattr(double_soma_obj,"mesh_errored_synapses",[])),
        
                          
                          soma_sdfs = double_soma_obj.preprocessed_data["soma_sdfs"],
                          soma_volume_ratios=double_soma_obj.preprocessed_data["soma_volume_ratios"],

                          insignificant_limbs_face_idx=insignificant_limbs_face_idx,
                          not_processed_soma_containing_meshes_face_idx = not_processed_soma_containing_meshes_face_idx,
                          glia_faces = double_soma_obj.preprocessed_data["glia_faces"],
                          labels = double_soma_obj.labels,
                          inside_pieces_face_idx=inside_pieces_face_idx,
                          non_soma_touching_meshes_face_idx=non_soma_touching_meshes_face_idx,

                          limb_meshes_face_idx=limb_meshes_face_idx,

                          new_limb_correspondence=new_limb_correspondence,
                            
                          segment_id=double_soma_obj.segment_id,
                          description=double_soma_obj.description,
                          decomposition_type=double_soma_obj.decomposition_type,
            
                          # don't need these any more because will recompute them when decompressing
                          #limb_labels= double_soma_obj.preprocessed_data['limb_labels'],
                          #limb_concept_networks=double_soma_obj.preprocessed_data['limb_concept_networks']
        
                          #new spine/width/labels compression
                          computed_attribute_dict = computed_attribute_dict,
                          
                          #For concept network creation
                          limb_network_stating_info = double_soma_obj.preprocessed_data["limb_network_stating_info"],
        
                          #for storing the faces indexed into the original mesh
                          original_mesh_idx=original_mesh_idx,
        
                          nucleus_id = double_soma_obj.nucleus_id,
                          split_index = double_soma_obj.split_index,
                          
                          pipeline_products = pipeline_products,
                                           
    )
    
    if print_flag:
        print(f"Total time for compression = {time.time() - total_compression_time }")
    
    return compressed_dict


def save_compressed_neuron(
    neuron_object,
    output_folder="./",
    file_name="",
    file_name_append = None,
    return_file_path=False,
    export_mesh=False):
    output_folder = Path(output_folder)
    
    if file_name == "":
        file_name = f"{neuron_object.segment_id}_{neuron_object.description}"
        
    if file_name_append is not None:
        file_name += f"_{file_name_append}"
    
    output_path = output_folder / Path(file_name)
    
    output_path = Path(output_path)
    output_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    
    inhib_object_compressed_preprocessed_data = smaller_preprocessed_data(neuron_object,print_flag=True)
    compressed_size = su.compressed_pickle(inhib_object_compressed_preprocessed_data,output_path,return_size=True)
    
    print(f"\n\n---Finished outputing neuron at location: {output_path.absolute()}---")
    
    if export_mesh:
        neuron_object.mesh.export(str(output_path.absolute()) +".off")
    
    if return_file_path:
        return output_path
    
    

#For decompressing the neuron
def decompress_neuron(filepath,original_mesh,
                     suppress_output=True,
                     debug_time = False,
                      using_original_mesh=True,
                     #error_on_original_mesh_faces_vertices_mismatch=False
                     ):
    if suppress_output:
        print("Decompressing Neuron in minimal output mode...please wait")
    
    with su.suppress_stdout_stderr() if suppress_output else su.dummy_context_mgr():
        
        decompr_time = time.time()
        
        loaded_compression = su.decompress_pickle(filepath)
        print(f"Inside decompress neuron and decomposition_type = {loaded_compression['decomposition_type']}")

        if debug_time:
            print(f"Decompress pickle time = {time.time() - decompr_time}")
            decompr_time = time.time()
        
        #creating dictionary that will be used to construct the new neuron object
        recovered_preprocessed_data = dict()

        """
        a) soma_meshes: use the 
        Data: soma_meshes_face_idx 
        Process: use submesh on the neuron mesh for each

        """
        if type(original_mesh) == type(Path()) or type(original_mesh) == str:
            if str(Path(original_mesh).absolute())[-3:] == '.h5':
                original_mesh = tu.load_mesh_no_processing_h5(original_mesh)
            else:
                original_mesh = tu.load_mesh_no_processing(original_mesh)
        elif type(original_mesh) == type(trimesh.Trimesh()):
            print("Recieved trimesh as orignal mesh")
        else:
            raise Exception(f"Got an unknown type as the original mesh: {original_mesh}")
            
        if debug_time:
            print(f"Getting mesh time = {time.time() - decompr_time}")
            decompr_time = time.time()
            

        # ------- 1/23 Addition: where using a saved mesh face idx to index into an original mesh ------#
        original_mesh_idx = loaded_compression.get("original_mesh_idx",None) 
        
        if using_original_mesh:
            if original_mesh_idx is None:
                print("The flag for using original mesh was set but no original_mesh_faces_idx stored in compressed data")
            else:
                print(f"Original mesh BEFORE using original_mesh_idx = {original_mesh}")
                original_mesh = original_mesh.submesh([original_mesh_idx],append=True,repair=False)
                print(f"Original mesh AFTER using original_mesh_idx = {original_mesh}")
            
            
        error_on_original_mesh_faces_vertices_mismatch=False
        
        if len(original_mesh.faces) != loaded_compression["original_mesh_n_faces"]:
            warning_string = (f"Number of faces in mesh used for compression ({loaded_compression['original_mesh_n_faces']})"
                            f" does not match the number of faces in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.faces)})")
            if error_on_original_mesh_faces_vertices_mismatch:
                raise Exception(warning_string)
            else:
                print(warning_string)
        else:
            print("Passed faces original mesh check")

        if len(original_mesh.vertices) != loaded_compression["original_mesh_n_vertices"]:
            warning_string = (f"Number of vertices in mesh used for compression ({loaded_compression['original_mesh_n_vertices']})"
                            f" does not match the number of vertices in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.vertices)})")
            
            if error_on_original_mesh_faces_vertices_mismatch:
                raise Exception(warning_string)
            else:
                print(warning_string)
        else:
            print("Passed vertices original mesh check")
            
            
        if debug_time:
            print(f"Face and Vertices check time = {time.time() - decompr_time}")
            decompr_time = time.time()


        recovered_preprocessed_data["soma_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["soma_meshes_face_idx"]]
        
        

        """
        b) soma_to_piece_connectivity
        Data: soma_to_piece_connectivity
        Process: None

        c) soma_sdfs
        Data: soma_sdfs
        Process: None
        """
        recovered_preprocessed_data["soma_to_piece_connectivity"] = loaded_compression["soma_to_piece_connectivity"]
        recovered_preprocessed_data["soma_sdfs"] = loaded_compression["soma_sdfs"]
        
        recovered_preprocessed_data["soma_volumes"] = loaded_compression.get("soma_volumes",None)
        recovered_preprocessed_data["soma_synapses"] = loaded_compression.get("soma_synapses",
                                                                    [None]*len(loaded_compression["soma_sdfs"]))
        recovered_preprocessed_data["mesh_errored_synapses"] = loaded_compression.get("mesh_errored_synapses",[])
        recovered_preprocessed_data["distance_errored_synapses"] = loaded_compression.get("distance_errored_synapses",[])
        
                                                
        
        if "soma_volume_ratios" in  loaded_compression.keys():
            print("using precomputed soma_volume_ratios")
            recovered_preprocessed_data["soma_volume_ratios"] = loaded_compression["soma_volume_ratios"]
        else:
            recovered_preprocessed_data["soma_volume_ratios"] = None
            
        if debug_time:
            print(f"Original Soma mesh time = {time.time() - decompr_time}")
            decompr_time = time.time()

        """
        d) insignificant_limbs
        Data: insignificant_limbs_face_idx
        Process: use submesh on the neuron mesh for each

        d) non_soma_touching_meshes
        Data: non_soma_touching_meshes_face_idx
        Process: use submesh on the neuron mesh for each

        d) inside_pieces
        Data: inside_pieces_face_idx
        Process: use submesh on the neuron mesh for each
        """

        recovered_preprocessed_data["insignificant_limbs"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["insignificant_limbs_face_idx"]]
        
        
        
        recovered_preprocessed_data["not_processed_soma_containing_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["not_processed_soma_containing_meshes_face_idx"]]
        
        
        
        
        if "glia_faces" in loaded_compression.keys():
            curr_glia = loaded_compression["glia_faces"]
        else:
            curr_glia = np.array([])
        
        recovered_preprocessed_data["glia_faces"] = curr_glia
        
        
        if "labels" in loaded_compression.keys():
            curr_labels = loaded_compression["labels"]
        else:
            curr_labels = np.array([])
        
        recovered_preprocessed_data["labels"] = curr_labels
        
        
        

        recovered_preprocessed_data["non_soma_touching_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["non_soma_touching_meshes_face_idx"]]

        recovered_preprocessed_data["inside_pieces"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["inside_pieces_face_idx"]]
        
        if debug_time:
            print(f"Insignificant and Not-processed and glia time = {time.time() - decompr_time}")
            decompr_time = time.time()

        """
        e) limb_meshes
        Data: limb_meshes_face_idx
        Process: use submesh on the neuron mesh for each

        """

        recovered_preprocessed_data["limb_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["limb_meshes_face_idx"]]

        if debug_time:
            print(f"Limb meshes time = {time.time() - decompr_time}")
            decompr_time = time.time()
        """

        f) limb_correspondence
        Data: new_limb_correspondence
        Process: 
        -- get branch mesh for each item
        --> later can recover the branch_mesh from the whole neuron mesh and the new branch_face_idx
        -- get branch_face_idx for each itme
        --> regenerate the and branch_face_idx from the recovered limb mesh and he recovered mesh

        """

        new_limb_correspondence = loaded_compression["new_limb_correspondence"]

        for k in new_limb_correspondence:
            print(f"Working on limb {k}")
            comb_meshes = None
            for j in tqdm(new_limb_correspondence[k]):
                print(f"  Working on branch {j}")
                
                new_limb_correspondence[k][j]["branch_mesh"] = original_mesh.submesh([new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"]],append=True,repair=False)
                
                try:
                    
                    new_limb_correspondence[k][j]["branch_face_idx"] = tu.original_mesh_faces_map(original_mesh=recovered_preprocessed_data["limb_meshes"][k], 
                                               submesh=new_limb_correspondence[k][j]["branch_mesh"] ,
                                           matching=True,
                                           print_flag=False,
                                           exact_match=True)
                except:
                    #Then try using the stitched meshes
                    #possible_non_touching_meshes = [c for c in recovered_preprocessed_data["non_soma_touching_meshes"] if len(c.faces) == len(new_limb_correspondence[k][j]["branch_mesh"].faces)]
                    possible_non_touching_meshes = [c for c in recovered_preprocessed_data["non_soma_touching_meshes"] if len(c.faces) >= len(new_limb_correspondence[k][j]["branch_mesh"].faces)]
                    found_match = False
                    for zz,t_mesh in enumerate(possible_non_touching_meshes):
                        try:
                            new_limb_correspondence[k][j]["branch_face_idx"] = tu.original_mesh_faces_map(original_mesh=t_mesh, 
                                                   submesh=new_limb_correspondence[k][j]["branch_mesh"] ,
                                               matching=True,
                                               print_flag=False,
                                               exact_match=True)
                            found_match=True
                            break
                        except:
                            print(f"Viable Non soma touching mesh({zz}): {t_mesh} was not a match")
                    if not found_match:
                        if comb_meshes is None:
                            comb_meshes = [recovered_preprocessed_data["limb_meshes"][k]] + list(recovered_preprocessed_data["non_soma_touching_meshes"])
                            comb_meshes = tu.combine_meshes(comb_meshes)
                        #print(f"comb_meshes = {comb_meshes}")
                        new_limb_correspondence[k][j]["branch_face_idx"] = tu.original_mesh_faces_map(
                                original_mesh=comb_meshes, 
                                                   submesh=new_limb_correspondence[k][j]["branch_mesh"],
                                               matching=True,
                                               print_flag=False,
                                               exact_match=True)
#                         except:
#                             raise Exception(f'Could Not find matching faces on decompression of mesh {new_limb_correspondence[k][j]["branch_mesh"]}')
                    


                if "branch_face_idx_whole_neuron" in new_limb_correspondence[k][j].keys():
                    del new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"]

        recovered_preprocessed_data["limb_correspondence"] = new_limb_correspondence

        if debug_time:
            print(f"Limb Correspondence = {time.time() - decompr_time}")
            decompr_time = time.time()

        # ------------------ This is old way of restoring the limb concept networks but won't work now ------------- #
        
        '''
        """
        g) limb_concept_networks, limb_labels:
        Data: All previous data
        Process: Call the funciton that creates the concept_networks using all the data above
        """

        limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
                limb_correspondence = recovered_preprocessed_data["limb_correspondence"],
                #limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
                #limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,

                soma_meshes=recovered_preprocessed_data["soma_meshes"],
                soma_idx_connectivity=recovered_preprocessed_data["soma_to_piece_connectivity"] ,
                #soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
                #soma_idx_connectivity = soma_idx_connectivity,

                current_neuron=original_mesh,
                return_limb_labels=True
                )
        '''
        
        # ----------------- ------------- #
        """
        Pseudocode for limb concept networks
        
        
        
        """
        
        limb_network_stating_info = loaded_compression["limb_network_stating_info"]
        
        limb_concept_networks=dict()
        limb_labels=dict()

        for curr_limb_idx,new_limb_correspondence_indiv in new_limb_correspondence.items():
            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(new_limb_correspondence_indiv,
                                                                                limb_network_stating_info[curr_limb_idx],
                                                                                run_concept_network_checks=True,
                                                                               )   



            limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
            limb_labels[curr_limb_idx]= None
        
        if debug_time:
            print(f"calculating limb networks = {time.time() - decompr_time}")
            decompr_time = time.time()

        recovered_preprocessed_data["limb_concept_networks"] = limb_concept_networks
        recovered_preprocessed_data["limb_labels"] = limb_labels
        recovered_preprocessed_data["limb_network_stating_info"] = limb_network_stating_info


        """
        h) get the segment ids and the original description

        """
        if "computed_attribute_dict" in loaded_compression.keys():
            computed_attribute_dict = loaded_compression["computed_attribute_dict"]
        else:
            computed_attribute_dict = None
        #return computed_attribute_dict
        
        # ------ 6/11: Adding in the nucleus id 
        recovered_preprocessed_data["nucleus_id"] = loaded_compression.get("nucleus_id",None)
        recovered_preprocessed_data["split_index"] = loaded_compression.get("split_index",None)
        
        pipeline_products = loaded_compression.get("pipeline_products",None)

        # Now create the neuron from preprocessed data
        decompressed_neuron = neuron.Neuron(
            mesh=original_mesh,
            segment_id=loaded_compression["segment_id"],
            description=loaded_compression["description"],
            decomposition_type = loaded_compression["decomposition_type"],
            preprocessed_data=recovered_preprocessed_data,
            computed_attribute_dict = computed_attribute_dict,
            suppress_output=suppress_output,
            calculate_spines=False,
            widths_to_calculate=[],
            original_mesh_idx=original_mesh_idx,
            pipeline_products=pipeline_products,
            )
        if debug_time:
            print(f"Sending to Neuron Object = {time.time() - decompr_time}")
            decompr_time = time.time()
            
            
        
    # if pipeline_products is not None:
    #     decompressed_neuron.pipeline_products = pipeline_products
    
    return decompressed_neuron

# --------------  END OF COMPRESSION OF NEURON ---------------- #

# --------------  7/23 To help with visualizations of neuron ---------------- #

def get_whole_neuron_skeleton(current_neuron,
                             check_connected_component=True,
                             print_flag=False):
    """
    Purpose: To generate the entire skeleton with limbs stitched to the somas
    of a neuron object
    
    Example Use: 
    
    total_neuron_skeleton = nru.get_whole_neuron_skeleton(current_neuron = recovered_neuron)
    sk.graph_skeleton_and_mesh(other_meshes=[current_neuron.mesh],
                              other_skeletons = [total_neuron_skeleton])
                              
    Ex 2: 
    nru = reload(nru)
    returned_skeleton = nru.get_whole_neuron_skeleton(recovered_neuron,print_flag=True)
    sk.graph_skeleton_and_mesh(other_skeletons=[returned_skeleton])
    """
    limb_skeletons_total = []
    for limb_idx in current_neuron.get_limb_node_names():
        if print_flag:
            print(f"\nWorking on limb: {limb_idx}")
        curr_limb_obj = current_neuron.concept_network.nodes[limb_idx]["data"]
        #stack the new skeleton pieces with the current skeleton 
        curr_limb_skeleton = curr_limb_obj.get_skeleton(check_connected_component=True)
        if print_flag:
            print(f"curr_limb_skeleton.shape = {curr_limb_skeleton.shape}")
        
        limb_skeletons_total.append(curr_limb_skeleton)

    #get the soma skeletons
    soma_skeletons_total = []
    for soma_idx in current_neuron.get_soma_node_names():
        if print_flag:
            print(f"\nWorking on soma: {soma_idx}")
        #get the soma skeletons
        curr_soma_skeleton = get_soma_skeleton(current_neuron,soma_name=soma_idx)
        
        if print_flag:
            print(f"for soma {soma_idx}, curr_soma_skeleton.shape = {curr_soma_skeleton.shape}")
        
        soma_skeletons_total.append(curr_soma_skeleton)

    total_neuron_skeleton = sk.stack_skeletons(limb_skeletons_total + soma_skeletons_total)

    if check_connected_component:
        sk.check_skeleton_one_component(total_neuron_skeleton)

    return total_neuron_skeleton

def get_soma_skeleton(current_neuron,soma_name):
    """
    Purpose: to return the skeleton for a soma that goes from the 
    soma center to all of the connecting limb
    
    Pseudocode: 
    1) get all of the limbs connecting to the soma (through the concept network)
    2) get the starting coordinate for that soma
    For all of the limbs connected
    3) Make the soma center to that starting coordinate a segment
    

    
    """
    #1) get all of the limbs connecting to the soma (through the concept network)
    limbs_connected_to_soma = xu.get_neighbors(current_neuron.concept_network,soma_name,int_label=False)
    #2) get the starting coordinate for that soma
    curr_soma_center = current_neuron.concept_network.nodes[soma_name]["data"].mesh_center
    
    #For all of the limbs connected
    #3) Make the soma center to that starting coordinate a segment
    soma_skeleton_pieces = []
    for limb_idx in limbs_connected_to_soma:
        curr_limb_obj = current_neuron.concept_network.nodes[limb_idx]["data"]
        
        curr_starting_coordinate = [cn_data["starting_coordinate"] for cn_data in curr_limb_obj.all_concept_network_data
                                                    if f"S{cn_data['starting_soma']}" == soma_name]
#         if len(curr_starting_coordinate) != 1:
#             raise Exception(f"curr_starting_coordinate not exactly one element: {curr_starting_coordinate}")
        
        for curr_endpoint in curr_starting_coordinate:
            new_skeleton_piece = np.vstack([curr_soma_center,curr_endpoint]).reshape(-1,2,3)
            soma_skeleton_pieces.append(new_skeleton_piece)
    
    return sk.stack_skeletons(soma_skeleton_pieces)


# def get_soma_skeleton_for_limb(current_neuron,limb_idx):
#     """
#     Purpose: To get the extra piece of skeleton
#     associated with that limb for all of those soma it connects to
    

#     """
    
#     #
    
#     soma_to_starting_dict = dict()
#     for cn_data in curr_limb_obj.all_concept_network_data:
#         soma_to_starting_dict[cn_data["starting_soma"]] = cn_data["starting_coordinate"]

#     """
#     will generate the new skeleton stitches


#     """
#     new_skeleton_pieces = []
#     for curr_soma,curr_endpoint in soma_to_starting_dict.items():
#         curr_soma_center = current_neuron.concept_network.nodes[f"S{curr_soma}"]["data"].mesh_center
#         #print(f"curr_soma_center = {curr_soma_center}")
#         new_skeleton_piece = np.vstack([curr_soma_center,curr_endpoint]).reshape(-1,2,3)
#         new_skeleton_pieces.append(new_skeleton_piece)
#         #print(f"new_skeleton_piece = {new_skeleton_piece}")

    
#     return new_skeleton_pieces


def soma_label(name_input, force_int=True):
    if type(name_input) == str:
        return name_input
    if force_int:
        name_input = int(name_input)
    if type(name_input) == int:
        return f"S{name_input}"
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for soma name")

def limb_label(name_input,force_int=True):
    if type(name_input) == str:
        return name_input
    if force_int:
        name_input = int(name_input)
    if type(name_input) == int or type(name_input) == float:
        return f"L{int(name_input)}"
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for limb name")

def limb_idx(name_input):
    if "str" in str(type(name_input)):
        return int(name_input[1:])
    elif type(name_input) == int or type(name_input) == float:
        return int(name_input)
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for limb name")
    
    
# --------------- 8/5 --------------------------#
def branch_mesh_no_spines(branch):
    """
    Purpose: To return the branch mesh without any spines
    """
    original_mesh_flag = False
    if not branch.spines is None:
        if len(branch.spines) > 0:
            ex_branch_no_spines_mesh = tu.original_mesh_faces_map(branch.mesh,
                                    tu.combine_meshes(branch.spines),
                                   matching=False,
                                   print_flag=False,
                                   match_threshold = 0.001,
                                                            return_mesh=True,
                                                                 )
        else:
            original_mesh_flag = True
    else: 
        original_mesh_flag = True
    
    if original_mesh_flag:
        ex_branch_no_spines_mesh = branch.mesh
        
    return ex_branch_no_spines_mesh

#xu.endpoint_connectivity(end_1,end_2)


# ---------------------- 8/31: For querying and axon searching --------------------------- #
def branch_skeletal_distance_from_soma(curr_limb,
                                       branch_idx,
                                    somas = None,
                                      dict_return=True,
                                       use_limb_copy = True,
                                      print_flag=False):
    """
    Purpose: Will find the distance of a branch from the specified somas
    as measured by the skeletal distance
    
    Pseudocode
    1) Make a copy of the current limb
    2) Get all of the somas that will be processed 
    (either specified or by default will )
    3) For each soma, find the skeletal distance from that branch to that soma and save in dictioanry
    4) if not asked to return dictionary then just return the minimum distance
    """
    
    if use_limb_copy:
        curr_limb_copy =  deepcopy(curr_limb)
    else:
        curr_limb_copy = curr_limb
    
    #0) Create dictionary that will store all of the results
    return_dict = dict()
    
    #For each directional concept network
    if somas is None:
        touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    else:
        if not nu.is_array_like(somas):
            somas = [somas]
        touching_somas = somas
        
    if print_flag:
        print(f"touching_somas = {touching_somas}")
    
    for curr_st_data in curr_limb_copy.all_concept_network_data:
        sm_start = curr_st_data["starting_soma"]
        sm_group_start = curr_st_data["soma_group_idx"]
        
        if sm_start not in touching_somas:
            continue
            
        
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start,sm_group_start)
        except:
            raise Exception(f"Limb ({limb_name}) was not connected to soma {sm_start} accordinag to all concept networks")
        
        curr_directional_network = curr_limb_copy.concept_network_directional
        starting_node = curr_limb_copy.current_starting_node
        
        if print_flag:
            print(f"starting_node = {starting_node}")
        
        try:
            curr_shortest_path = nx.shortest_path(curr_directional_network,starting_node,branch_idx)
        except:
            if print_flag:
                print(f"branch_idx {branch_idx} did not have a path to soma {sm}, so making distance np.inf")
            return_dict[sm_start] = np.inf
            continue
            
        path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                           for k in curr_shortest_path[:-1]])
        
        if print_flag:
            print(f"path_length = {path_length}")
        
        return_dict[sm_start] = path_length
    
    #once have the final dictionary either return the dictionary or the minimum path
    if dict_return:
        return return_dict
    else: #return the minimum path length
        return np.min(list(return_dict.values()))
    

# ------------------------------ 9/1 To help with mesh correspondence -----------------------------------------------------#


def apply_adaptive_mesh_correspondence_to_neuron(current_neuron,
                                                apply_sdf_filter=False,
                                                n_std_dev=1):

    
    for limb_idx in np.sort(current_neuron.get_limb_node_names()):
        
        ex_limb = current_neuron.concept_network.nodes[limb_idx]["data"]
        if apply_sdf_filter:
            print("Using SDF filter")
            ray_inter = ray_pyembree.RayMeshIntersector(ex_limb.mesh)
            
        
        segment_mesh_faces = dict()
        for branch_idx in np.sort(ex_limb.concept_network.nodes()):
            print(f"---- Working on limb {limb_idx} branch {branch_idx} ------")
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            #1) get all the neighbors 1 hop away in connectivity
            #2) Assemble a mesh of all the surrounding neighbors
            one_hop_neighbors = xu.get_neighbors(ex_limb.concept_network,branch_idx)
            if len(one_hop_neighbors) > 0:
                two_hop_neighbors = np.concatenate([xu.get_neighbors(ex_limb.concept_network,k) for k in one_hop_neighbors])
                branches_for_surround = np.unique([branch_idx] + list(one_hop_neighbors) + list(two_hop_neighbors))


                surround_mesh_faces = np.concatenate([ex_limb.concept_network.nodes[k]["data"].mesh_face_idx for k in branches_for_surround])
                surrounding_mesh = ex_limb.mesh.submesh([surround_mesh_faces],append=True,repair=False)

                #3) Send the skeleton and the surrounding mesh to the mesh adaptive distance --> gets back indices
                return_value = cu.mesh_correspondence_adaptive_distance(curr_branch_skeleton=ex_branch.skeleton,
                                                     curr_branch_mesh=surrounding_mesh)

                if len(return_value) == 2:
                    remaining_indices, width = return_value
                    final_limb_indices = surround_mesh_faces[remaining_indices]
                else: #if mesh correspondence couldn't be found
                    print("Mesh correspondence couldn't be found so using defaults")
                    final_limb_indices = ex_branch.mesh_face_idx
                    width = ex_branch.width

            else: #if mesh correspondence couldn't be found
                print("Mesh correspondence couldn't be found so using defaults")
                final_limb_indices = ex_branch.mesh_face_idx
                width = ex_branch.width



            """  How we would get the final mesh  
            branch_mesh_filtered = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False) 

            """
            #5b) store the width measurement based back in the mesh object
            ex_branch.width_new["adaptive"] = width

            if apply_sdf_filter:
                #---------- New step:  Further filter the limb indices
                new_branch_mesh = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False)
                new_branch_obj = copy.deepcopy(ex_branch)
                new_branch_obj.mesh = new_branch_mesh
                new_branch_obj.mesh_face_idx = final_limb_indices

                filtered_branch_mesh,filtered_branch_mesh_idx,filtered_branch_sdf_mean= sdf_filter(curr_branch=new_branch_obj,
                                                                                                       curr_limb=ex_limb,
                                                                                                       return_sdf_mean=True,
                                                                                                       ray_inter=ray_inter,
                                                                                                      n_std_dev=n_std_dev)
                final_limb_indices = final_limb_indices[filtered_branch_mesh_idx]
            
            segment_mesh_faces[branch_idx] = final_limb_indices

        #This ends up fixing any conflicts in labeling
        face_lookup = gu.invert_mapping(segment_mesh_faces,total_keys=np.arange(0,len(ex_limb.mesh.faces)))
        #original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
        #original_labels = gu.get_unique_values_dict_of_lists(face_lookup)
        original_labels = np.arange(0,len(ex_limb))

        face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = ex_limb.mesh,
                                                                                        face_lookup=face_lookup,
                                                                                        no_missing_labels = list(original_labels),
                                                                     max_submesh_threshold=50000)

        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(ex_limb.mesh,face_coloring_copy)

        #now reassign the new divided supmeshes
        for branch_idx in ex_limb.concept_network.nodes():
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            ex_branch.mesh = divided_submeshes[branch_idx]
            ex_branch.mesh_face_idx = divided_submeshes_idx[branch_idx]
            ex_branch.mesh_center = tu.mesh_center_vertex_average(ex_branch.mesh)
            
            #need to change the preprocessed_data to reflect the change
            limb_idx_used = int(limb_idx[1:])
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_mesh"] = ex_branch.mesh 
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_face_idx"] = ex_branch.mesh_face_idx
            
            

# --- 9/2: Mesh correspondence that helps deal with the meshparty data  ----
def sdf_filter(curr_branch,curr_limb,size_threshold=20,
               return_sdf_mean=False,
               ray_inter=None,
              n_std_dev = 1):
    """
    Purpose: to eliminate edge parts of meshes that should
    not be on the branch mesh correspondence
    
    Pseudocode
    The filtering step (Have a size threshold for this maybe?):
    1) Calculate the sdf values for all parts of the mesh
    2) Restrict the faces to only thos under mean + 1.5*std_dev
    3) split the mesh and only keep the biggest one

    Example: 
    
    limb_idx = 0
    branch_idx = 20
    branch_idx = 3
    #branch_idx=36
    filtered_branch_mesh, filtered_branch_mesh_idx = sdf_filter(double_neuron_processed[limb_idx][branch_idx],double_neuron_processed[limb_idx],
                                                               n_std_dev=1)
    filtered_branch_mesh.show()

    """
    

    
    #1) Calculate the sdf values for all parts of the mesh
    ray_trace_width_array = tu.ray_trace_distance(curr_limb.mesh,face_inds=curr_branch.mesh_face_idx,ray_inter=ray_inter)
    ray_trace_width_array_mean = np.mean(ray_trace_width_array[ray_trace_width_array>0])
    #apply the size threshold
    if len(curr_branch.mesh.faces)<20:
        if return_sdf_mean:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx)),ray_trace_width_array_mean
        else:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx))
    
    #2) Restrict the faces to only thos under mean + 1.5*std_dev
    ray_trace_mask = ray_trace_width_array < (ray_trace_width_array_mean + n_std_dev*np.std(ray_trace_width_array))
    filtered_mesh = curr_limb.mesh.submesh([curr_branch.mesh_face_idx[ray_trace_mask]],append=True,repair=False)

    
    #3) split the mesh and only keep the biggest one
    filtered_split_meshes, filtered_split_meshes_idx = tu.split(filtered_mesh)
    
    if return_sdf_mean:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0],ray_trace_width_array_mean
    else:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0]
    
    
# --------- 9/9 Helps with splitting the mesh limbs ------------ #
def get_limb_to_soma_border_vertices(current_neuron,print_flag=False):
    """
    Purpose: To create a lookup dictionary indexed by 
    - soma
    - limb name
    The will return the vertex coordinates on the border of the soma and limb

    
    """
    start_time = time.time()

    limb_to_soma_border_by_soma = dict()

    for soma_name in current_neuron.get_soma_node_names():

        soma_idx = int(soma_name[1:])


        curr_soma_mesh = current_neuron[soma_label(soma_idx)].mesh
        touching_limbs = current_neuron.get_limbs_touching_soma(soma_idx)
        touching_limb_objs = [current_neuron[k] for k in touching_limbs]

        touching_limbs_meshes = [k.mesh for k in touching_limb_objs]
        touching_pieces,touching_vertices = tu.mesh_pieces_connectivity(main_mesh=current_neuron.mesh,
                                                central_piece = curr_soma_mesh,
                                                periphery_pieces = touching_limbs_meshes,
                                                                 return_vertices=True,
                                                                return_central_faces=False,
                                                                        print_flag=False
                                                                                         )
        limb_to_soma_border = dict([(k,v) for k,v in zip(np.array(touching_limbs)[touching_pieces],touching_vertices)])
        limb_to_soma_border_by_soma[soma_idx] = limb_to_soma_border
    if print_flag:
        print(time.time() - start_time)
    return limb_to_soma_border_by_soma


        
def compute_all_concept_network_data_from_limb(curr_limb,current_neuron_mesh,soma_meshes,soma_restriction=None,
                                              print_flag=False):
    ex_limb = curr_limb
    
    curr_limb_divided_meshes = [ex_limb[k].mesh for k in ex_limb.get_branch_names()]
    curr_limb_divided_skeletons = [ex_limb[k].skeleton for k in ex_limb.get_branch_names()]


    """ Old way of doing it which required the neuron
    if soma_restriction is None:
        soma_restriction_names = current_neuron.get_soma_node_names()
    else:
        soma_restriction_names = [soma_label(k) for k in soma_restriction]

    soma_restriction_names_ints = [int(k[1:]) for k in soma_restriction_names]
    soma_mesh_list = [current_neuron.concept_network.nodes[k]["data"].mesh for k in soma_restriction_names]
    """
    
    if soma_restriction is None:
        soma_mesh_list = soma_meshes
        soma_restriction_names_ints = list(np.arange(0,len(soma_mesh_list)))
    else:
        soma_mesh_list = [k for i,k in soma_meshes if i in soma_restriction]
        soma_restriction_names_ints = soma_restriction


    derived_concept_network_data = []
    for soma_idx,soma_mesh in zip(soma_restriction_names_ints,soma_mesh_list):
        periph_filter_time = time.time()

        original_idxs = np.arange(0,len(curr_limb_divided_meshes))


        distances_periphery_to_soma = np.array([tu.closest_distance_between_meshes(soma_mesh,k) for k in curr_limb_divided_meshes])
        periphery_distance_threshold = 2000

        original_idxs = original_idxs[distances_periphery_to_soma<periphery_distance_threshold]
        filtered_periphery_meshes = np.array(curr_limb_divided_meshes)[distances_periphery_to_soma<periphery_distance_threshold]


        touching_pieces,touching_vertices,central_piece_faces = tu.mesh_pieces_connectivity(main_mesh=current_neuron_mesh,
                                    central_piece = soma_mesh,
                                    periphery_pieces = filtered_periphery_meshes,
                                                     return_vertices=True,
                                                    return_central_faces=True
                                                                             )
        if print_flag:
            print(f"Total time for mesh connectivity = {time.time() - periph_filter_time}")
        #print(f"touching_pieces = {original_idxs[touching_pieces[0]]}")
        if len(touching_pieces) > 0:
            touching_pieces = original_idxs[touching_pieces]
            if print_flag:
                print(f"touching_pieces = {touching_pieces}")


            if len(touching_pieces) >= 2:
                if print_flag:
                    print("**More than one touching point to soma, touching_pieces = {touching_pieces}**")
                # picking the piece with the most shared vertices
                len_touch_vertices = [len(k) for k in touching_vertices]
                winning_piece_idx = np.argmax(len_touch_vertices)
                if print_flag:
                    print(f"winning_piece_idx = {winning_piece_idx}")
                touching_pieces = [touching_pieces[winning_piece_idx]]
                if print_flag:
                    print(f"Winning touching piece = {touching_pieces}")
                touching_pieces_soma_vertices = touching_vertices[winning_piece_idx]
            else:
                touching_pieces_soma_vertices = touching_vertices[0]
            if len(touching_pieces) < 1:
                raise Exception("No touching pieces")

            #3) With the one that is touching the soma, find the enpoints of the skeleton
            if print_flag:
                print(f"Using touching_pieces[0] = {touching_pieces[0]}")
            touching_branch = neuron.Branch(curr_limb_divided_skeletons[touching_pieces[0]])
            endpoints = touching_branch.endpoints

            """  # -----------  9/1 -------------- #
            New method for finding 
            1) Build a KDTree of the winning touching piece soma boundary vertices
            2) query the endpoints against the vertices
            3) pick the endpoint that has the closest match
            """
            ex_branch_KDTree = KDTree(touching_pieces_soma_vertices)
            distances,closest_nodes = ex_branch_KDTree.query(endpoints)
            closest_endpoint = endpoints[np.argmin(distances)]

            derived_concept_network_data.append(dict(starting_soma=soma_idx,
                                                    starting_node=touching_pieces[0],
                                                     starting_endpoints=endpoints,
                                                     starting_coordinate=closest_endpoint,
                                                    touching_soma_vertices=touching_pieces_soma_vertices
                                               ))
    return derived_concept_network_data

def error_limb_indexes(neuron_obj):
    return np.where(np.array([len(limb.all_concept_network_data) for limb in neuron_obj])>1)[0]

def same_soma_multi_touching_limbs(neuron_obj,return_n_touches=False):
    same_soma_multi_touch_limbs = []

    touch_dict = dict()
    for curr_limb_idx, curr_limb in enumerate(neuron_obj):
        if len(curr_limb.all_concept_network_data) > 0:
            touching_somas = [k["starting_soma"] for k in curr_limb.all_concept_network_data]
            soma_mapping = gu.invert_mapping(touching_somas)

            for soma_idx,touch_idxs in soma_mapping.items():
                if len(touch_idxs) > 1:
                    if curr_limb_idx not in touch_dict.keys():
                        touch_dict[curr_limb_idx] = dict()

                    same_soma_multi_touch_limbs.append(curr_limb_idx)
                    touch_dict[curr_limb_idx][soma_idx] = len(touch_idxs)
                    break
                
    if return_n_touches:
        return touch_dict
    else:
        return np.array(same_soma_multi_touch_limbs)
                     

def multi_soma_touching_limbs(neuron_obj):
    multi_soma_touch_limbs = []

    for curr_limb_idx, curr_limb in enumerate(neuron_obj):
        if len(curr_limb.all_concept_network_data) > 0:
            touching_somas = [k["starting_soma"] for k in curr_limb.all_concept_network_data]
            soma_mapping = gu.invert_mapping(touching_somas)
            if len(soma_mapping.keys()) > 1:
                multi_soma_touch_limbs.append(curr_limb_idx)

    return np.array(multi_soma_touch_limbs)

def error_limbs(neuron_obj):
    """
    Purpose: Will return all of the 
    
    """
    multi_soma_limbs = nru.multi_soma_touching_limbs(neuron_obj)
    multi_touch_limbs = nru.same_soma_multi_touching_limbs(neuron_obj)
    return np.unique(np.concatenate([multi_soma_limbs,multi_touch_limbs])).astype('int')


# ---- 11/20 functions that will help compute statistics of the neuron object -----------



def n_error_limbs(neuron_obj):
    return len(error_limb_indexes(neuron_obj))

def n_somas(neuron_obj):
    return len(neuron_obj.get_soma_node_names())

def n_limbs(neuron_obj):
    return len(neuron_obj.get_limb_node_names())

def n_branches_per_limb(neuron_obj):
    return [len(ex_limb.get_branch_names()) for ex_limb in neuron_obj]

def n_branches(neuron_obj):
    return np.sum(neuron_obj.n_branches_per_limb)

def skeleton_length_per_limb(neuron_obj):
    return [sk.calculate_skeleton_distance(limb.skeleton) for limb in neuron_obj]

def skeletal_length(neuron_obj):
    return np.sum(neuron_obj.skeleton_length_per_limb)


def max_limb_n_branches(neuron_obj):
    if len(neuron_obj.n_branches_per_limb)>0:
        return np.max(neuron_obj.n_branches_per_limb)
    else:
        return None

def max_limb_skeletal_length(neuron_obj):
    if len(neuron_obj.skeleton_length_per_limb) > 0:
        return np.max(neuron_obj.skeleton_length_per_limb)
    else:
        return None

def all_skeletal_lengths(neuron_obj):
    all_skeletal_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
            all_skeletal_lengths.append(curr_branch_sk_len)
    return np.array(all_skeletal_lengths)

def median_branch_length(neuron_obj):
    if len(all_skeletal_lengths(neuron_obj))>0:
        return np.round(np.median(all_skeletal_lengths(neuron_obj)),3)
    else:
        return None
    

# -- width data --
def all_medain_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_width = curr_branch.width_new["median_mesh_center"]
            if curr_width < np.inf:
                all_widths.append(curr_width)
    return np.array(all_widths)

def all_no_spine_median_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_width = curr_branch.width_new["no_spine_median_mesh_center"]
            if curr_width < np.inf:
                all_widths.append(curr_width)
    return np.array(all_widths)

def width_median(neuron_obj):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_medain_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_no_spine_median(neuron_obj):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_no_spine_median_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_perc(neuron_obj,perc=90):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_medain_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None

def width_no_spine_perc(neuron_obj,perc=90):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_no_spine_median_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None



# -- spine data --

def calculate_spines_skeletal_length(neuron_obj):
    if neuron_obj.spines is None:
        return None
    sk_len_array = []
    for k in neuron_obj.spines:
        try:
            curr_skeletal_length = sk.calculate_skeleton_distance(sk.surface_skeleton(k))
        except:
            curr_skeletal_length = 0 

        sk_len_array.append(curr_skeletal_length)
    neuron_obj.spines_skeletal_length = np.array(sk_len_array)
    return sk_len_array
    

def n_spines(neuron_obj,skeletal_length_max=None):
    if skeletal_length_max is None:
        skeletal_length_max = skeletal_length_max_n_spines_global
    if neuron_obj.spines is None:
        return 0
    else:
        if skeletal_length_max is not None:
            if not hasattr(neuron_obj,"spines_skeletal_length")  or len(neuron_obj.spines_skeletal_length) != len(neuron_obj.spines):
#                 sk_len_array = []
#                 for k in neuron_obj.spines:
#                     try:
#                         curr_skeletal_length = sk.calculate_skeleton_distance(sk.surface_skeleton(k))
#                     except:
#                         curr_skeletal_length = 0 
                        
#                     sk_len_array.append(curr_skeletal_length)
#                 neuron_obj.spines_skeletal_length = np.array(sk_len_array)
                
                neuron_obj.spines_skeletal_length = nru.calculate_spines_skeletal_length(neuron_obj)
                                                                            
                #neuron_obj.spines_skeletal_length = np.array([sk.calculate_skeleton_distance(sk.surface_skeleton(k)) for k in neuron_obj.spines])
            valid_spine_idx = np.where(np.array(neuron_obj.spines_skeletal_length)<skeletal_length_max)[0]
            return len(valid_spine_idx)
        else:
            return len(neuron_obj.spines)
    
def n_boutons(neuron_obj):
    if neuron_obj.boutons is None:
        return 0
    else:
        return len(neuron_obj.boutons)
    
def n_web(neuron_obj):
    if neuron_obj.boutons is None:
        return 0
    else:
        return 1
    
def compute_mesh_attribute_volume(branch_obj,
                                  mesh_attribute,
                                 max_hole_size=2000,
                                 self_itersect_faces=False):
    if getattr(branch_obj,mesh_attribute) is None:
        setattr(branch_obj,f"{mesh_attribute}_volume",None)
    else:
        vol_list = [tu.mesh_volume(sp,verbose=False) for sp in
                   getattr(branch_obj,mesh_attribute)]
        setattr(branch_obj,f"{mesh_attribute}_volume",vol_list)

def feature_list_over_object(obj,
                            feature_name):
    """
    Purpose: Will compile a list of all of the 
    """
    obj._index = -1
    total_feature = []
    for b in obj:
        if not getattr(b,feature_name) is None:
            total_feature += list(getattr(b,feature_name))
    return total_feature

def compute_feature_over_object(obj,
                               feature_name):
    obj._index = -1
    for b in obj:
        getattr(b,f"compute_{feature_name}")()
    
def spine_density(neuron_obj):
    skeletal_length = neuron_obj.skeletal_length
    if skeletal_length > 0:
        spine_density = neuron_obj.n_spines/skeletal_length
    else:
        spine_density = 0
    return spine_density

def spines_per_branch(neuron_obj):
    if neuron_obj.n_branches > 0:
        spines_per_branch = neuron_obj.n_spines/neuron_obj.n_branches
    else:
        spines_per_branch = 0
    return spines_per_branch
    
def n_spine_eligible_branches(neuron_obj):
    n_spine_eligible_branches = 0
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                n_spine_eligible_branches += 1
    return n_spine_eligible_branches

def spine_eligible_branch_lengths(neuron_obj):
    spine_eligible_branch_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
                spine_eligible_branch_lengths.append(curr_branch_sk_len)
    return spine_eligible_branch_lengths

def skeletal_length_eligible(neuron_obj):
    return np.round(np.sum(neuron_obj.spine_eligible_branch_lengths),3)

def spine_density_eligible(neuron_obj):
    #spine eligible density and per branch
    if neuron_obj.skeletal_length_eligible > 0:
        spine_density_eligible = neuron_obj.n_spines/neuron_obj.skeletal_length_eligible
    else:
        spine_density_eligible = 0
    
    return spine_density_eligible

def spines_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spine_eligible_branches > 0:
        spines_per_branch_eligible = np.round(neuron_obj.n_spines/neuron_obj.n_spine_eligible_branches,3)
    else:
        spines_per_branch_eligible = 0
    
    return spines_per_branch_eligible


# ------- all the spine volume stuff -----------
def total_spine_volume(neuron_obj):
    if neuron_obj.n_spines > 0:
        spines_vol = np.array(neuron_obj.spines_volume)
        return np.sum(spines_vol)
        
    else:
        return 0

def spine_volume_median(neuron_obj):
    spines_vol = np.array(neuron_obj.spines_volume)
    if neuron_obj.n_spines > 0:
        #spine_volume_median
        valid_spine_vol = spines_vol[spines_vol>0]

        if len(valid_spine_vol) > 0:
            spine_volume_median = np.median(valid_spine_vol)
        else:
            spine_volume_median = 0
        
        return spine_volume_median
        
    else:
        return 0
    
def spine_volume_density(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length_eligible > 0:
            spine_volume_density_eligible = neuron_obj.total_spine_volume/neuron_obj.skeletal_length
        else:
            spine_volume_density_eligible = 0
        
        return spine_volume_density_eligible
        
    else:
        return 0


def spine_volume_density_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length > 0:
            spine_volume_density = neuron_obj.total_spine_volume/neuron_obj.skeletal_length_eligible
        else:
            spine_volume_density = 0
        
        return spine_volume_density
        
    else:
        return 0
    
def spine_volume_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.n_spine_eligible_branches > 0:
            spine_volume_per_branch_eligible = neuron_obj.total_spine_volume/neuron_obj.n_spine_eligible_branches
        else:
            spine_volume_per_branch_eligible = 0
        
        return spine_volume_per_branch_eligible
        
    else:
        return 0
    
    
# -------------- 11 / 26 To help with erroring------------------------------#


def align_and_restrict_branch(base_branch,
                              common_endpoint=None,
                              width_name= "no_spine_median_mesh_center",
                              width_name_backup= "no_spine_median_mesh_center",
                             offset=500,
                             comparison_distance=2000,
                             skeleton_segment_size=1000,
                              verbose=False,
                             ):
    
    if width_name not in base_branch.width_array.keys():
        width_name = width_name_backup
        
    #Now just need to do the resizing (and so the widths calculated will match this)
    base_skeleton_ordered = sk.resize_skeleton_branch(base_branch.skeleton,skeleton_segment_size)

    if not common_endpoint is None:
        #figure out if need to flip or not:
        if np.array_equal(common_endpoint,base_skeleton_ordered[-1][-1]):

            base_width_ordered = np.flip(base_branch.width_array[width_name])
            base_skeleton_ordered = sk.flip_skeleton(base_skeleton_ordered)
            flip_flag = True
            if verbose:
                print("Base needs flipping")
                print(f"Skeleton after flip = {base_skeleton_ordered}")
        elif np.array_equal(common_endpoint,base_skeleton_ordered[0][0]):
            base_width_ordered = base_branch.width_array[width_name]
            flip_flag = False
        else:
            raise Exception("No matching endpoint")
    else:
        base_width_ordered = base_branch.width_array[width_name]
        
    # apply the cutoff distance
    if verbose:
        print(f"Base offset = {offset}")
        
    
    (skeleton_minus_buffer,
     offset_indexes,
     offset_success) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                    offset,
                                                                     subtract_cutoff=True)
   
    
    base_final_skeleton = None
    base_final_indexes = None

    if offset_success:
        
        (skeleton_comparison,
         comparison_indexes,
         comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        
        if comparison_success:
            if verbose:
                print("Base: Long enough for offset and comparison length")
            base_final_skeleton = skeleton_comparison
            base_final_indexes = offset_indexes[comparison_indexes]

        else:
            if verbose:
                print("Base: Passed the offset phase but was not long enough for comparison")
    else:
        if verbose:
            print("Base: Was not long enough for offset")


    if base_final_skeleton is None:
        if verbose:
            print("Base: Not using offset ")
        (base_final_skeleton,
         base_final_indexes,
         _) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        

    
    base_final_widths = base_width_ordered[np.clip(base_final_indexes,0,len(base_width_ordered)-1)]
    base_final_seg_lengths = sk.calculate_skeleton_segment_distances(base_final_skeleton,cumsum=False)
    
    return base_final_skeleton,base_final_widths,base_final_seg_lengths

def branch_boundary_transition_old(curr_limb,
                              edge,
                              width_name= "no_spine_median_mesh_center",
                            width_name_backup = "no_spine_median_mesh_center",
                              offset=500,
                              comparison_distance=2000,
                              skeleton_segment_size=1000,
                              return_skeletons=True,
                              verbose=False):
    """
    Purpose: Will find the boundary skeletons and width average at the boundary
    with some specified boundary skeletal length (with an optional offset)


    """

    base_node = edge[-1]
    upstream_node= edge[0]
    upstream_node_original = upstream_node

    base_branch = curr_limb[base_node]
    upstream_branch = curr_limb[upstream_node]


    # 0) make sure the two nodes are connected in the concept network
    if base_node not in xu.get_neighbors(curr_limb.concept_network,upstream_node):
        raise Exception(f"base_node ({base_node}) and upstream_node ({upstream_node}) are not connected in the concept network")

    # ----- Part 1: Do the processing on the base node -------------- #
    common_endpoint = sk.shared_endpoint(base_branch.skeleton,upstream_branch.skeleton)
    common_endpoint_original = copy.copy(common_endpoint)
    if verbose:
        print(f"common_endpoint = {common_endpoint}")
    
    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(base_branch,
                              common_endpoint=common_endpoint,
                                 width_name=width_name,
                                width_name_backup=width_name_backup,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=verbose,
                             )
    
    
    
    
    
    

    # ----- Part 2: Do the processing on the upstream nodes -------------- #
    upstream_offset = offset
    upstream_comparison = comparison_distance
    upstream_node = edge[0]
    previous_node = edge[1]
    upstream_skeleton = []
    upstream_seg_lengths = []
    upstream_seg_widths = []

    count = 0
    while upstream_comparison > 0:
        """
        Pseudocode:
        1) Get shared endpoint of upstream and previous node
        2) resize the upstream skeleton to get it ordered and right scale of width
        3) Flip the skeleton and width array if needs to be flipped
        4) if current offset is greater than 0, then restrict skeelton to offset:
        5a) if it was not long enough:
            - subtact total length from buffer
        5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break
        6)  change out upstream node and previous node (because at this point haven't broken outside loop)

        """
        if verbose:
            print(f"--- Upstream iteration: {count} -----")
        prev_branch = curr_limb[previous_node]
        upstream_branch = curr_limb[upstream_node]

        #1) Get shared endpoint of upstream and previous node
        common_endpoint = sk.shared_endpoint(prev_branch.skeleton,upstream_branch.skeleton)

        #2) resize the upstream skeleton to get it ordered and right scale of width
        upstream_skeleton_ordered = sk.resize_skeleton_branch(upstream_branch.skeleton,skeleton_segment_size)
        if verbose:
            print(f"upstream_skeleton_ordered {sk.calculate_skeleton_distance(upstream_skeleton_ordered)} = {upstream_skeleton_ordered}")
            
        
          # ----------- 1 /5 : To prevent from erroring when indexing into width
#         #accounting for the fact that the skeleton might be a little longer thn the width array now
#         upstream_width = upstream_branch.width_array[width_name]
#         extra_width_segment = [upstream_width[-1]]*(len(upstream_skeleton_ordered)-len(upstream_width))
#         upstream_width = np.hstack([upstream_width,extra_width_segment])
         

        #3) Flip the skeleton and width array if needs to be flipped
        if np.array_equal(common_endpoint,upstream_skeleton_ordered[-1][-1]):
            upstream_width_ordered = np.flip(upstream_branch.width_array[width_name])
            upstream_skeleton_ordered = sk.flip_skeleton(upstream_skeleton_ordered)
            flip_flag = True
        elif np.array_equal(common_endpoint,upstream_skeleton_ordered[0][0]):
            upstream_width_ordered = upstream_branch.width_array[width_name]
            flip_flag = False
        else:
            raise Exception("No matching endpoint")

            
        if verbose: 
            print(f"flip_flag = {flip_flag}")
            print(f"upstream_offset = {upstream_offset}")

        #4) if current offset is greater than 0, then restrict skeelton to offset:
        if upstream_offset > 0:
            if verbose:
                print("Restricting to offset")
            (skeleton_minus_buffer,
             offset_indexes,
             offset_success) = sk.restrict_skeleton_from_start(upstream_skeleton_ordered,
                                                                            upstream_offset,
                                                                             subtract_cutoff=True)
        else:
            if verbose:
                print("Skipping the upstream offset because 0")
            skeleton_minus_buffer = upstream_skeleton_ordered
            offset_indexes = np.arange(len(upstream_skeleton_ordered))
            offset_success = True
        
        
        #print(f"skeleton_minus_buffer {sk.calculate_skeleton_distance(skeleton_minus_buffer)} = {skeleton_minus_buffer}")

        """
        5a) if it was not long enough:
        - subtact total length from buffer
        """
        if not offset_success:
            upstream_offset -= sk.calculate_skeleton_distance(upstream_skeleton_ordered)
            if verbose:
                print(f"Subtracting the offset was not successful so changing to {upstream_offset} and reiterating")
        else:
            """
            5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break

            """
            #making sure the upstream offset is 0 if we were successful
            upstream_offset = 0
            
            if verbose:
                print(f"After subtracting the offset the length is: {sk.calculate_skeleton_distance(skeleton_minus_buffer)}")

            #- restrit skeleton by comparison distance
            (skeleton_comparison,
             comparison_indexes,
             comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                            upstream_comparison,
                                                                             subtract_cutoff=False)
            #- Add skeleton, width and skeelton lengths to list
            upstream_skeleton.append(skeleton_comparison)
            upstream_seg_lengths.append(sk.calculate_skeleton_segment_distances(skeleton_comparison,cumsum=False))

            
            upstream_indices = offset_indexes[comparison_indexes]
            upstream_seg_widths.append(upstream_width_ordered[np.clip(upstream_indices,0,len(upstream_width_ordered)-1) ])

            # - subtract new distance from comparison distance
            upstream_comparison -= sk.calculate_skeleton_distance(skeleton_comparison)

            if comparison_success:
                if verbose:
                    print(f"Subtracting the comparison was successful and exiting")
                break
            else:
                if verbose:
                    print(f"Subtracting the comparison was not successful so changing to {upstream_comparison} and reiterating")

        #6)  change out upstream node and previous node (because at this point haven't broken outside loop)
        previous_node = upstream_node
        upstream_node = xu.upstream_node(curr_limb.concept_network_directional,upstream_node)

        if verbose:
            print(f"New upstream_node = {upstream_node}")

        if upstream_node is None:
            if verbose:
                print("Breaking because hit None upstream node")
            break

        count += 1

    upstream_final_skeleton = sk.stack_skeletons(upstream_skeleton)
    if verbose:
        print(f"upstream_final_skeleton = {upstream_final_skeleton}")

    # Do a check at the very end and if no skeleton then just take that branches
    if len(upstream_final_skeleton) <= 0:
        print("No upstream skeletons so doing backup")
        resize_sk = sk.resize_skeleton_branch(curr_limb[upstream_node_original].skeleton,
                                                       skeleton_segment_size)
        upstream_skeleton = [resize_sk]
        upstream_seg_lengths = [sk.calculate_skeleton_segment_distances(resize_sk,cumsum=False)]
        upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name]]
        
        (upstream_final_skeleton,
         upstream_final_widths,
        upstream_final_seg_lengths) = nru.align_and_restrict_branch(curr_limb[upstream_node_original],
                                  common_endpoint=common_endpoint_original,
                                width_name=width_name,
                                 offset=offset,
                                 comparison_distance=comparison_distance,
                                 skeleton_segment_size=skeleton_segment_size,
                                  verbose=verbose,
                                 )
    else:
        upstream_final_seg_lengths = np.concatenate(upstream_seg_lengths)
        upstream_final_widths = np.concatenate(upstream_seg_widths)




    #Final results
    base_final_skeleton
    base_final_widths
    base_final_seg_lengths

    upstream_skeleton 
    upstream_seg_lengths 
    upstream_seg_widths

    base_final_skeleton
    

    base_width_average = nu.average_by_weights(weights = base_final_seg_lengths,
                                values = base_final_widths)
    upstream_width_average = nu.average_by_weights(weights = upstream_final_seg_lengths,
                            values = upstream_final_widths)

    if return_skeletons:
        return upstream_width_average,base_width_average,upstream_final_skeleton,base_final_skeleton
    else:
        return upstream_width_average,base_width_average
    
    
def branch_boundary_transition(curr_limb,
                              edge,
                                   upstream_common_endpoint=None,
                                   downstream_common_endpoint=None,
                              width_name= "no_spine_median_mesh_center",
                               width_name_backup = "no_spine_median_mesh_center",
                              offset=500,
                              comparison_distance=2000,
                              skeleton_segment_size=1000,
                              return_skeletons=True,
                                   error_on_no_network_connection=False,
                              verbose=False):
    """
    Purpose: Will find the boundary skeletons and width average at the boundary
    with some specified boundary skeletal length (with an optional offset)


    """

    base_node = edge[-1]
    upstream_node= edge[0]
    upstream_node_original = upstream_node

    base_branch = curr_limb[base_node]
    
    upstream_branch = curr_limb[upstream_node]


    # 0) make sure the two nodes are connected in the concept network
    
    if base_node not in xu.get_neighbors(curr_limb.concept_network,upstream_node):
        warning_string = f"base_node ({base_node}) and upstream_node ({upstream_node}) are not connected in the concept network"
        if error_on_no_network_connection:
            raise Exception(warning_string)
        else:
            if verbose:
                print(warning_string)

    # ----- Part 1: Do the processing on the base node -------------- #
    
    if upstream_common_endpoint is None or downstream_common_endpoint is None:
        common_endpoint = sk.shared_endpoint(base_branch.skeleton,upstream_branch.skeleton)
        common_endpoint_original = copy.copy(common_endpoint)
        if verbose:
            print(f"common_endpoint = {common_endpoint}")
            
        non_common_endpoints_flag = False
    else:
        if verbose:
            print(f"upstream_common_endpoint = {upstream_common_endpoint}")
            print(f"downstream_common_endpoint = {downstream_common_endpoint}")
            
        common_endpoint_original = copy.copy(upstream_common_endpoint)
        non_common_endpoints_flag = True
            
    
    if non_common_endpoints_flag:
        common_endpoint = downstream_common_endpoint
        
        
    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(base_branch,
                              common_endpoint=common_endpoint,
                                 width_name=width_name,
                                 width_name_backup=width_name_backup,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=verbose,
                             )
#     print(f"base_node = {base_node}")
#     su.compressed_pickle(base_branch,"base_branch")
#     su.compressed_pickle(base_final_skeleton,"base_final_skeleton")
    
    
    
    
    
    

    # ----- Part 2: Do the processing on the upstream nodes -------------- #
    upstream_offset = offset
    upstream_comparison = comparison_distance
    upstream_node = edge[0]
    previous_node = edge[1]
    upstream_skeleton = []
    upstream_seg_lengths = []
    upstream_seg_widths = []
    
    
    count = 0
    while upstream_comparison > 0:
        """
        Pseudocode:
        1) Get shared endpoint of upstream and previous node
        2) resize the upstream skeleton to get it ordered and right scale of width
        3) Flip the skeleton and width array if needs to be flipped
        4) if current offset is greater than 0, then restrict skeelton to offset:
        5a) if it was not long enough:
            - subtact total length from buffer
        5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break
        6)  change out upstream node and previous node (because at this point haven't broken outside loop)

        """
        if verbose:
            print(f"--- Upstream iteration: {count} -----")
        prev_branch = curr_limb[previous_node]
        upstream_branch = curr_limb[upstream_node]

        #1) Get shared endpoint of upstream and previous node
        if count == 0 and non_common_endpoints_flag:
            common_endpoint = upstream_common_endpoint
        else:
            common_endpoint = sk.shared_endpoint(prev_branch.skeleton,upstream_branch.skeleton)

        #2) resize the upstream skeleton to get it ordered and right scale of width
        upstream_skeleton_ordered = sk.resize_skeleton_branch(upstream_branch.skeleton,skeleton_segment_size)
        if verbose:
            print(f"upstream_skeleton_ordered {sk.calculate_skeleton_distance(upstream_skeleton_ordered)} = {upstream_skeleton_ordered}")
            
        
          # ----------- 1 /5 : To prevent from erroring when indexing into width
#         #accounting for the fact that the skeleton might be a little longer thn the width array now
#         upstream_width = upstream_branch.width_array[width_name]
#         extra_width_segment = [upstream_width[-1]]*(len(upstream_skeleton_ordered)-len(upstream_width))
#         upstream_width = np.hstack([upstream_width,extra_width_segment])
         

        #3) Flip the skeleton and width array if needs to be flipped
        if np.array_equal(common_endpoint,upstream_skeleton_ordered[-1][-1]):
            try:
                upstream_width_ordered = np.flip(upstream_branch.width_array[width_name])
            except:
                upstream_width_ordered = np.flip(upstream_branch.width_array[width_name_backup])
                
            upstream_skeleton_ordered = sk.flip_skeleton(upstream_skeleton_ordered)
            flip_flag = True
        elif np.array_equal(common_endpoint,upstream_skeleton_ordered[0][0]):
            try:
                upstream_width_ordered = upstream_branch.width_array[width_name]
            except:
                upstream_width_ordered = upstream_branch.width_array[width_name_backup]
                
            flip_flag = False
        else:
            raise Exception("No matching endpoint")

            
        if verbose: 
            print(f"flip_flag = {flip_flag}")
            print(f"upstream_offset = {upstream_offset}")

        #4) if current offset is greater than 0, then restrict skeelton to offset:
        if upstream_offset > 0:
            if verbose:
                print("Restricting to offset")
            (skeleton_minus_buffer,
             offset_indexes,
             offset_success) = sk.restrict_skeleton_from_start(upstream_skeleton_ordered,
                                                                            upstream_offset,
                                                                             subtract_cutoff=True)
        else:
            if verbose:
                print("Skipping the upstream offset because 0")
            skeleton_minus_buffer = upstream_skeleton_ordered
            offset_indexes = np.arange(len(upstream_skeleton_ordered))
            offset_success = True
        
        
        #print(f"skeleton_minus_buffer {sk.calculate_skeleton_distance(skeleton_minus_buffer)} = {skeleton_minus_buffer}")

        """
        5a) if it was not long enough:
        - subtact total length from buffer
        """
        if not offset_success:
            upstream_offset -= sk.calculate_skeleton_distance(upstream_skeleton_ordered)
            if verbose:
                print(f"Subtracting the offset was not successful so changing to {upstream_offset} and reiterating")
        else:
            """
            5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break

            """
            #making sure the upstream offset is 0 if we were successful
            upstream_offset = 0
            
            if verbose:
                print(f"After subtracting the offset the length is: {sk.calculate_skeleton_distance(skeleton_minus_buffer)}")

            #- restrit skeleton by comparison distance
            (skeleton_comparison,
             comparison_indexes,
             comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                            upstream_comparison,
                                                                             subtract_cutoff=False)
            #- Add skeleton, width and skeelton lengths to list
            upstream_skeleton.append(skeleton_comparison)
            upstream_seg_lengths.append(sk.calculate_skeleton_segment_distances(skeleton_comparison,cumsum=False))

            
            upstream_indices = offset_indexes[comparison_indexes]
            upstream_seg_widths.append(upstream_width_ordered[np.clip(upstream_indices,0,len(upstream_width_ordered)-1) ])

            # - subtract new distance from comparison distance
            upstream_comparison -= sk.calculate_skeleton_distance(skeleton_comparison)

            if comparison_success:
                if verbose:
                    print(f"Subtracting the comparison was successful and exiting")
                break
            else:
                if verbose:
                    print(f"Subtracting the comparison was not successful so changing to {upstream_comparison} and reiterating")

        #6)  change out upstream node and previous node (because at this point haven't broken outside loop)
        previous_node = upstream_node
        upstream_node = xu.upstream_node(curr_limb.concept_network_directional,upstream_node)

        if verbose:
            print(f"New upstream_node = {upstream_node}")

        if upstream_node is None:
            if verbose:
                print("Breaking because hit None upstream node")
            break

        count += 1

    upstream_final_skeleton = sk.stack_skeletons(upstream_skeleton)
    if verbose:
        print(f"upstream_final_skeleton = {upstream_final_skeleton}")

    # Do a check at the very end and if no skeleton then just take that branches
    if len(upstream_final_skeleton) <= 0:
        print("No upstream skeletons so doing backup")
        resize_sk = sk.resize_skeleton_branch(curr_limb[upstream_node_original].skeleton,
                                                       skeleton_segment_size)
        upstream_skeleton = [resize_sk]
        upstream_seg_lengths = [sk.calculate_skeleton_segment_distances(resize_sk,cumsum=False)]
        try:
            upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name]]
        except:
            upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name_backup]]
        
        (upstream_final_skeleton,
         upstream_final_widths,
        upstream_final_seg_lengths) = nru.align_and_restrict_branch(curr_limb[upstream_node_original],
                                  common_endpoint=common_endpoint_original,
                                width_name=width_name,
                                 offset=offset,
                                 comparison_distance=comparison_distance,
                                 skeleton_segment_size=skeleton_segment_size,
                                  verbose=verbose,
                                 )
    else:
        upstream_final_seg_lengths = np.concatenate(upstream_seg_lengths)
        upstream_final_widths = np.concatenate(upstream_seg_widths)




    #Final results
    base_final_skeleton
    base_final_widths
    base_final_seg_lengths

    upstream_skeleton 
    upstream_seg_lengths 
    upstream_seg_widths

    base_final_skeleton
    

    base_width_average = nu.average_by_weights(weights = base_final_seg_lengths,
                                values = base_final_widths)
    upstream_width_average = nu.average_by_weights(weights = upstream_final_seg_lengths,
                            values = upstream_final_widths)

    if return_skeletons:
        return upstream_width_average,base_width_average,upstream_final_skeleton,base_final_skeleton
    else:
        return upstream_width_average,base_width_average
    

global_comparison_distance = 3000
def find_parent_child_skeleton_angle(curr_limb_obj,
                            child_node,   
                            parent_node=None,
                           #comparison_distance=3000,
                            comparison_distance=global_comparison_distance,
                            offset=0,
                           verbose=False,
                           check_upstream_network_connectivity=True,
                                    plot_extracted_skeletons=False,
                                    **kwargs):
    
#     print(f"comparison_distance = {comparison_distance}")
#     print(f"offset = {offset}")
    if parent_node is None:
        parent_node = xu.upstream_node(curr_limb_obj.concept_network_directional,child_node)
        
    # -------Doing the parent calculation---------
    parent_child_edge = [parent_node,child_node]

    up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                      edge=parent_child_edge,
                                      comparison_distance = comparison_distance,
                                    offset=offset,
                                    verbose=False,  #check_upstream_network_connectivity=check_upstream_network_connectivity
                                                            )
    up_sk_flipped = sk.flip_skeleton(up_sk)

    up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    parent_child_angle = np.round(nu.angle_between_vectors(up_vec,d_vec_child),2)
    
    if plot_extracted_skeletons:
        bs = [parent_node,child_node]
        parent_color = "red"
        child_color = "blue"
        print(f"Parent ({parent_node}):{parent_color}, child ({child_node}):{child_color}")
        c = [parent_color,child_color]
        nviz.plot_objects(meshes=[curr_limb_obj[k].mesh for k in bs],
                         meshes_colors=c,
                         skeletons =[up_sk_flipped,d_sk],
                         skeletons_colors=c)
        
       

    if verbose:
        print(f"parent_child_angle = {parent_child_angle}")
        
    return parent_child_angle    



def find_sibling_child_skeleton_angle(curr_limb_obj,
                            child_node,
                            parent_node=None,
                           #comparison_distance=3000,
                         comparison_distance=global_comparison_distance,
                            offset=0,
                           verbose=False):
    
    
    # -------Doing the parent calculation---------
    if parent_node is None:
        parent_node = xu.upstream_node(curr_limb_obj.concept_network_directional,child_node)
        
    parent_child_edge = [parent_node,child_node]

    up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                      edge=parent_child_edge,
                                      comparison_distance = comparison_distance,
                                    offset=offset,
                                    verbose=False)
    
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    # -------Doing the child calculation---------
    sibling_nodes = xu.sibling_nodes(curr_limb_obj.concept_network_directional,
                                    child_node)
    
    sibl_angles = dict()
    for s_n in sibling_nodes:
        sibling_child_edge = [parent_node,s_n]

        up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                          edge=sibling_child_edge,
                                          comparison_distance = comparison_distance,
                                        offset=offset,
                                        verbose=False)

        up_vec = up_sk[-1][-1] - up_sk[0][0] 
        d_vec_sibling = d_sk[-1][-1] - d_sk[0][0]

        sibling_child_angle = np.round(nu.angle_between_vectors(d_vec_child,d_vec_sibling),2)
        
        sibl_angles[s_n] = sibling_child_angle
        
    return sibl_angles
    

def all_concept_network_data_to_dict(all_concept_network_data):
    return_dict = dict()
    for st_info in all_concept_network_data:
        curr_soma_idx = st_info["starting_soma"]
        curr_soma_group_idx = st_info["soma_group_idx"]
        curr_endpoint = st_info["starting_coordinate"]
        curr_touching_soma_vertices = st_info["touching_soma_vertices"]
        
        if curr_soma_idx not in return_dict.keys():
            return_dict[curr_soma_idx] = dict()
        
        return_dict[curr_soma_idx][curr_soma_group_idx] = dict(touching_verts=curr_touching_soma_vertices,
                                                         endpoint=curr_endpoint
                                                        )
        

            
    return return_dict
            
    
def limb_to_soma_mapping(current_neuron):
    """
    Purpose: Will create a mapping of 
    limb --> soma_idx --> list of soma touching groups
    
    """
    limb_soma_touch_dictionary = dict()
    for curr_limb_idx,curr_limb in enumerate(current_neuron):
        limb_soma_touch_dictionary[curr_limb_idx] = dict()
        for st_info in curr_limb.all_concept_network_data:
            curr_soma_idx = st_info["starting_soma"]
            curr_soma_group_idx = st_info["soma_group_idx"]
            if curr_soma_idx not in limb_soma_touch_dictionary[curr_limb_idx].keys():
                limb_soma_touch_dictionary[curr_limb_idx][curr_soma_idx] = []
            limb_soma_touch_dictionary[curr_limb_idx][curr_soma_idx].append(curr_soma_group_idx)
            
    return limb_soma_touch_dictionary

    
    
def all_starting_dicts_by_soma(curr_limb,soma_idx):
    return [k for k in curr_limb.all_concept_network_data if k["starting_soma"] == soma_idx]
def all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,attr="starting_node"):
    starting_dicts = all_starting_dicts_by_soma(curr_limb,soma_idx)
    return [k[attr] for k in starting_dicts]

def convert_int_names_to_string_names(limb_names,start_letter="L"):
    return [f"{start_letter}{k}" for k in limb_names]

def convert_string_names_to_int_names(limb_names):
    return [int(k[1:]) for k in limb_names]

def get_limb_string_name(limb_idx,start_letter="L"):
    if limb_idx is None:
        return None
    if type(limb_idx) == int or "int" in str(type(limb_idx)) or "float" in str(type(limb_idx)):
        return f"{start_letter}{limb_idx}" 
    elif type(limb_idx) == str or "str" in str(type(limb_idx)):
        return limb_idx
    else:
        raise Exception("Not int or string input")
        
def get_limb_int_name(limb_name):
    if limb_name is None:
        return None
    if type(limb_name) == int:
        return limb_name
    elif type(limb_name) == str:
        return int(limb_name[1:])
    else:
        raise Exception("Not int or string input")
        
def get_soma_string_name(soma_idx,start_letter="S"):
    limb_idx = soma_idx
    if limb_idx is None:
        return None
    if type(limb_idx) == int or "int" in str(type(limb_idx)) or "float" in str(type(limb_idx)):
        return f"{start_letter}{limb_idx}" 
    elif type(limb_idx) == str or "str" in str(type(limb_idx)):
        return limb_idx
    else:
        raise Exception("Not int or string input")
        
def get_soma_int_name(soma_name):
    limb_name = soma_name
    if limb_name is None:
        return None
    if type(limb_name) == int:
        return limb_name
    elif type(limb_name) == str:
        return int(limb_name[1:])
    else:
        raise Exception("Not int or string input")
        


def filter_limbs_below_soma_percentile(neuron_obj,
                                        above_percentile = 70,
                                         return_string_names=True,
                                       visualize_remianing_neuron=False,
                                        verbose = True):
    """
    Purpose: Will only keep those limbs that have 
    a mean touching vertices lower than the soma faces percentile specified
    
    Pseudocode: 
    1) Get the soma mesh
    2) Get all of the face midpoints
    3) Get only the y coordinates of the face midpoints  and turn negative
    4) Get the x percentile of those y coordinates
    5) Get all those faces above that percentage
    6) Get those faces as a submesh and show

    -- How to cancel out the the limbs

    """
    keep_limb_idx = []
    for curr_limb_idx,curr_limb in enumerate(neuron_obj):

        touching_somas = curr_limb.touching_somas()

        keep_limb = False
        for sm_idx in touching_somas:
            if not keep_limb :
                sm_mesh = neuron_obj[f"S{sm_idx}"].mesh

                tri_centers_y = -sm_mesh.triangles_center[:,1]
                perc_y_position = np.percentile(tri_centers_y,above_percentile)


                """ Don't need this: just for verification that was working with soma
                kept_faces = np.where(tri_centers_y <= perc_y_position)[0]

                soma_top = sm_mesh.submesh([kept_faces],append=True)
                """

                """
                Pseudocode for adding limb as possible:
                1) Get all starting dictionaries for that soma
                For each starting dict:
                a) Get the mean of the touching_soma_vertices (and turn negative)
                b) If mean is less than the perc_y_position then set keep_limb to True and break


                """
                all_soma_starting_dicts = all_starting_dicts_by_soma(curr_limb,sm_idx)
                for j,curr_start_dict in enumerate(all_soma_starting_dicts):
                    if verbose:
                        print(f"Working on touching group {j}")

                    t_verts_mean = -1*np.mean(curr_start_dict["touching_soma_vertices"][:,1])

                    if t_verts_mean <= perc_y_position:
                        if verbose:
                            print("Keeping limb because less than y position")
                        keep_limb = True
                        break

                if keep_limb:
                    break
                    
        #decide whether or not to keep limb
        if keep_limb:
            if verbose:
                print(f"Keeping Limb {curr_limb_idx}")
            
            keep_limb_idx.append(curr_limb_idx)
            
    if visualize_remianing_neuron:
        remaining_limbs = convert_int_names_to_string_names(keep_limb_idx)
        ret_col = nviz.visualize_neuron(neuron_obj,
                     visualize_type=["mesh","skeleton"],
                     limb_branch_dict=dict([(k,"all") for k in remaining_limbs]),
                     return_color_dict=True)
            
    if verbose:
        print(f"\n\nTotal removed Limbs = {np.delete(np.arange(len(neuron_obj.get_limb_node_names())),keep_limb_idx)}")
    if return_string_names:
        return convert_int_names_to_string_names(keep_limb_idx)
    else:
        return keep_limb_idx

def limb_branch_dict_to_faces(neuron_obj,limb_branch_dict):
    """
    Purpose: To return the face indices of the main
    mesh that correspond to the limb/branches indicated by dictionary
    
    Pseudocode: 
    0) Have a final face indices list
    
    Iterate through all of the limbs
        Iterate through all of the branches
            1) Get the original indices of the branch on main mesh
            2) Add to the list
            
    3) Concatenate List and return
    
    ret_val = nru.limb_branch_dict_to_faces(neuron_obj,dict(L1=[0,1,2]))
    """
    final_face_indices = []
    
    for limb_name,branch_names in limb_branch_dict.items():
        
        all_branch_meshes = [neuron_obj[limb_name][k].mesh for k in branch_names]
        
        if len(all_branch_meshes)>0:
            match_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                        all_branch_meshes,
                                                           matching=True,
                                                           print_flag=False)
        else:
            match_faces = []
        
        final_face_indices.append(match_faces)
    
    if len(final_face_indices)>0:
        match_faces_idx = np.concatenate(final_face_indices).astype("int")
    else:
        match_faces_idx = np.array([])
        
    return match_faces_idx
 
    
    
def skeleton_touching_branches(limb_obj,branch_idx,
                              return_endpoint_groupings=True):
    """
    Purpose: Can find all the branch numbers
    that touch a certain branch object based on the skeleton endpoints
    
    """
    curr_short_seg = branch_idx
    curr_limb = limb_obj
    branch_obj = limb_obj[branch_idx]

    network_nodes = np.array(curr_limb.concept_network.nodes())
    network_nodes = network_nodes[network_nodes!= curr_short_seg]

    network_branches = [curr_limb[k].skeleton for k in network_nodes]
    neighbor_branches_by_endpoint = [network_nodes[sk.find_branch_skeleton_with_specific_coordinate(network_branches,e)] for e in branch_obj.endpoints]
    
    
    
    if return_endpoint_groupings:
        return neighbor_branches_by_endpoint,branch_obj.endpoints
    else:
        return np.concatenate(neighbor_branches_by_endpoint)
    
    
def all_soma_connnecting_endpionts_from_starting_info(starting_info):
    all_endpoints = []
    try:
        for limb_idx,limb_start_v in starting_info.items():
            for soma_idx,soma_v in limb_start_v.items():
                for soma_group_idx,group_v in soma_v.items():
                    all_endpoints.append(group_v["endpoint"])
    except:
        for soma_idx,soma_v in starting_info.items():
            for soma_group_idx,group_v in soma_v.items():
                all_endpoints.append(group_v["endpoint"])
        
    if len(all_endpoints) > 0:
        all_endpoints = np.unique(np.vstack(all_endpoints),axis=0)
    return all_endpoints
    
    

def skeleton_points_along_path(limb_obj,branch_path,
                               skeletal_distance_per_coordinate=2000,
                               return_unique=True):
    """
    Purpose: Will give skeleton coordinates for the endpoints of the 
    branches along the specified path
    
    if skeletal_distance_per_coordinate is None then will just endpoints
    """
    if skeletal_distance_per_coordinate is None:
        skeleton_coordinates = np.array([sk.find_branch_endpoints(limb_obj[k].skeleton) for k in branch_path]).reshape(-1,3)
    else:
        skeleton_coordinates = np.concatenate([sk.resize_skeleton_branch(
                                        limb_obj[k].skeleton,
                                        segment_width=skeletal_distance_per_coordinate) for k in branch_path]).reshape(-1,3)
        
    if return_unique:
        return np.unique(skeleton_coordinates,axis=0)
    else:
        return skeleton_coordinates
    
    
def get_matching_concept_network_data(limb_obj,soma_idx=None,soma_group_idx=None,
                                     starting_node=None,
                                     verbose=False):
    
    if type(soma_idx) == str:
        soma_idx = int(soma_idx[1:])
    
    if soma_idx is None and (soma_group_idx is None) and starting_node is None:
        raise Exception("All soma, soma_group and starting node descriptions are None")
        
    matching_concept_network_dicts_idx = np.arange(len(limb_obj.all_concept_network_data))
  
    if soma_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_soma"] == soma_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if soma_group_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["soma_group_idx"] == soma_group_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if starting_node is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_node"] == starting_node])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if verbose:
        print(f"matching_concept_network_dicts_idx = {matching_concept_network_dicts_idx}")
        
    return [limb_obj.all_concept_network_data[k] for k in matching_concept_network_dicts_idx]
    
    
    
# ----------- 1/15: For Automatic Axon and Apical Classification ---------------#
def add_branch_label(neuron_obj,limb_branch_dict,
                    labels):
    """
    Purpose: Will go through and apply a label to the branches
    specified
    
    """
    if not nu.is_array_like(labels):
        labels = [labels]
    
    for limb_name ,branch_array in limb_branch_dict.items():
        for b in branch_array:
            branch_obj = neuron_obj[limb_name][b]
            
            for l in labels:
                if l not in branch_obj.labels:
                    branch_obj.labels.append(l)
                    
def clear_all_branch_labels(neuron_obj,labels_to_clear="all",limb_branch_dict=None):
    if labels_to_clear != "all" and not nu.is_array_like(labels_to_clear):
        labels_to_clear = [labels_to_clear]
        
    for l_name in neuron_obj.get_limb_node_names():
        
        if limb_branch_dict is not None and l_name not in limb_branch_dict.keys():
            continue
        else:
            l = neuron_obj[l_name]
        
        for b_name in l.get_branch_names():
            if limb_branch_dict is not None and b_name not in limb_branch_dict[l_name]:
                continue
            else:
                b = neuron_obj[l_name][b_name]
            
            if labels_to_clear == "all":
                b.labels=[]
            else:
                b.labels = list(np.setdiff1d(b.labels,labels_to_clear))

def clear_certain_branch_labels(neuron_obj,labels_to_clear,limb_branch_dict=None):
    return clear_all_branch_labels(neuron_obj,
                                   labels_to_clear=labels_to_clear,
                                   limb_branch_dict=limb_branch_dict)
            
def viable_axon_limbs_by_starting_angle_old(neuron_obj,
                                       axon_soma_angle_threshold=70,
                                       return_starting_angles=False):
    """
    This is method that does not use neuron querying (becuase just simple iterating through limbs)
    """
    
    possible_axon_limbs = []
    # Find the limb find the soma angle AND Filter away all limbs with a soma starting angle above threshold
    limb_to_starting_angle = dict()
    for curr_limb_idx,curr_limb in enumerate(curr_neuron_obj):
        curr_soma_angle = nst.soma_starting_angle(curr_neuron_obj,curr_limb_idx)
        limb_to_starting_angle[curr_limb_idx] = curr_soma_angle

        if curr_soma_angle > axon_soma_angle_threshold:
            possible_axon_limbs.append(curr_limb_idx)
    
    if return_starting_angles:
        return possible_axon_limbs,limb_to_starting_angle
    else:
        return possible_axon_limbs
    
def viable_axon_limbs_by_starting_angle(neuron_obj,
                                       soma_angle_threshold,
                                        above_threshold=False,
                                        soma_name="S0",
                                        return_int_name=True,
                                       verbose=False):
    
    curr_neuron_obj = neuron_obj
    soma_center = curr_neuron_obj[soma_name].mesh_center

    if above_threshold:
        curr_query = f"soma_starting_angle>{soma_angle_threshold}"
    else:
        curr_query = f"soma_starting_angle<{soma_angle_threshold}"
    
    possible_axon_limbs_dict = ns.query_neuron(curr_neuron_obj,
                        query=curr_query,
                       functions_list=["soma_starting_angle"],
                       function_kwargs=dict(soma_center=soma_center,
                                           verbose=verbose))

    possible_axon_limbs = list(possible_axon_limbs_dict.keys())
    if return_int_name:
        return [nru.get_limb_int_name(k) for k in possible_axon_limbs]
    else:
        return possible_axon_limbs
    
def get_limb_starting_angle_dict(neuron_obj):
    """
    Purpose: To return a dictionary mapping
    limb_idx --> soma_idx --> soma_group --> starting angle

    Psuedocode: 
    1) Iterate through all of the limbs
    2) Iterate through all of the starting dict information
    3) compute the staritng angle
    4) Save in a dictionary

    """


    starting_angle_dict = dict()
    for limb_name in neuron_obj.get_limb_names(return_int=True):
        limb_obj = neuron_obj[limb_name]
        for st_dict in limb_obj.all_concept_network_data:
            st_soma = st_dict["starting_soma"]
            st_soma_group = st_dict["soma_group_idx"]

            st_angle = nst.soma_starting_angle(limb_obj,
                            neuron_obj=neuron_obj,
                           soma_idx=st_soma,
                           soma_group_idx=st_soma_group)

            if limb_name not in starting_angle_dict.keys():
                starting_angle_dict[limb_name] = dict()
            if st_soma not in starting_angle_dict[limb_name].keys():
                starting_angle_dict[limb_name][st_soma] = dict()


            starting_angle_dict[limb_name][st_soma][st_soma_group] = st_angle

    return starting_angle_dict
    
    
def skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    branches = None,
                    **kwargs
                            
    ):

    """
    Purpose: To determine the skeletal distance away from 
    a soma a branch piece is
    
    Pseudocode: 
    0) Create dictionary that will store all of the results
    For each directional concept network
    1) Find the starting node
    For each node: 
    1)find the shortest path from starting node to that node
    2) convert the path into skeletal distance of each node 
    and then add up
    3) Map of each of distances to the node in a dictionary and return
    - replace a previous one if smaller
    
    Example: 
    skeletal_distance_from_soma(
                    limb_name = "L1"
                    curr_limb = uncompressed_neuron.concept_network.nodes[limb_name]["data"]
                    print_flag = True
                    #soma_list=None
                    somas = [0,1]
                    check_all_nodes_in_return=True
    )

    """
    if print_flag:
        print(f"\n\n------Working on Limb ({limb_name})-------")
        print(f"Starting nodes BEFORE copy = {xu.get_starting_node(curr_limb.concept_network,only_one=False)}")

    curr_limb_copy =  deepcopy(curr_limb)
    
    if print_flag:
        print(f"Starting nodes after copy = {xu.get_starting_node(curr_limb_copy.concept_network,only_one=False)}")

    #0) Create dictionary that will store all of the results
    return_dict = dict()

    #For each directional concept network
    if somas is None:
        touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    else:
        if not nu.is_array_like(somas):
            somas = [somas]
        touching_somas = somas

    if print_flag:
        print(f"Performing analysis for somas: {touching_somas}")

    
    nodes_to_process = curr_limb_copy.get_branch_names()
        
    for sm_start in touching_somas:
        #1) Find the starting node
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start)
        except:
            if print_flag:
                print(f"Limb ({limb_name}) was not connected to soma {sm_start} accordinag to all concept networks")
            continue
        curr_directional_network = curr_limb_copy.concept_network_directional
        starting_node = curr_limb_copy.current_starting_node
        
        if branches is not None:
            nodes_to_process = branches
        else:
            nodes_to_process = curr_directional_network.nodes()

        #For each node: 
        for n in nodes_to_process:
            #1)find the shortest path from starting node to that node
            #( could potentially not be there because it is directional)
            try:
                curr_shortest_path = nx.shortest_path(curr_directional_network,starting_node,n)
            except:
                #return_dict[n] = np.inf
                continue
            #2) convert the path into skeletal distance of each node and then add up
            if not include_node_skeleton_dist:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path[:-1]])
            else:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path])


            #3) Map of each of distances to the node in a dictionary and return
            #- replace a previous one if smaller

            if n in return_dict.keys():
                if path_length < return_dict[n]:
                    return_dict[n] = path_length
            else:
                return_dict[n] = path_length
    if print_flag:
        print(f"\nBefore Doing the dictionary correction, return_dict={return_dict}\n")
    #check that the return dict has all of the nodes
    for n in nodes_to_process:
        if n not in return_dict.keys():
            return_dict[n] = np.inf
   
    if error_if_all_nodes_not_return:
        #if set(list(return_dict.keys())) != set(list(curr_limb_copy.concept_network.nodes())):
        if set(list(return_dict.keys())) != set(list(nodes_to_process)):
            raise Exception("return_dict keys do not exactly match the curr limb nodes")
            
    return return_dict
    

def find_branch_with_specific_coordinate(limb_obj,
                                        coordinates):
    """
    Purpose: To find all branch idxs whos skeleton contains a certain coordinate
    
    """
    
    coordinates = np.array(coordinates).reshape(-1,3)
    
    network_branches = [k.skeleton for k in limb_obj]
    
    final_branch_idxs = []
    for e in coordinates:
        curr_branch_idx = sk.find_branch_skeleton_with_specific_coordinate(network_branches,e)
        if len(curr_branch_idx) > 0:
            final_branch_idxs.append(curr_branch_idx)
    
    if len(final_branch_idxs) > 0:
        final_branch_idxs = np.concatenate(final_branch_idxs)
    
    if len(final_branch_idxs)>0:
        final_branch_idxs = np.sort(final_branch_idxs)
    
    return final_branch_idxs

def find_branch_with_specific_endpoint(limb_obj,
                                        coordinates):
    """
    Purpose: To find all branch idxs whos skeleton contains a certain coordinate
    
    """
    
    coordinates = np.array(coordinates).reshape(-1,3)
    
    network_branches = [k.endpoints.reshape(-1,2,3) for k in limb_obj]
    
    final_branch_idxs = []
    for e in coordinates:
        curr_branch_idx = sk.find_branch_skeleton_with_specific_coordinate(network_branches,e)
        if len(curr_branch_idx) > 0:
            final_branch_idxs.append(curr_branch_idx)
    
    if len(final_branch_idxs) > 0:
        final_branch_idxs = np.concatenate(final_branch_idxs)
    
    if len(final_branch_idxs)>0:
        final_branch_idxs = np.sort(final_branch_idxs)
    
    return final_branch_idxs



def neuron_spine_density(neuron_obj,
                        lower_width_bound = 140,
                        upper_width_bound = 520,#380,
                        spine_threshold = 2,
                        skeletal_distance_threshold = 110000,#30000,
                        skeletal_length_threshold = 15000,#10000
                        verbose=False,
                        plot_candidate_branches=False,
                        return_branch_processed_info=True,
                        **kwargs):
    """
    Purpose: To Calculate the spine density used to classify
    a neuron as one of the following categories based on the spine
    density of high interest branches
    
    1) no_spine
    2) sparsely_spine
    3) densely_spine
    
    
    """
    curr_neuron_obj= neuron_obj
    
    
    
    
    if plot_candidate_branches:
        return_dataframe=False
        close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","no_spine_median_mesh_center",
                                                            "n_spines","spine_density","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                   f" and (no_spine_median_mesh_center > {lower_width_bound})"
                                                   f" and (no_spine_median_mesh_center < {upper_width_bound})"
                                                  f" and (n_spines > {spine_threshold})"
                                                   f" and skeletal_length > {skeletal_length_threshold} "
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
        
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=close_limb_branch_dict,
                              mesh_color="red",
                              mesh_whole_neuron=True)
        
        
    
    
    return_dataframe = True
    close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","no_spine_median_mesh_center",
                                                            "n_spines","spine_density","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                   f" and (no_spine_median_mesh_center > {lower_width_bound})"
                                                   f" and (no_spine_median_mesh_center < {upper_width_bound})"
                                                  f" and (n_spines > {spine_threshold})"
                                                   f" and skeletal_length > {skeletal_length_threshold} "
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
    
    total_branches_in_search_radius = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
    
    
    # ---- 1/24: Calculating the skeletal length of the viable branches --- #
    if len(total_branches_in_search_radius)>0:
        total_skeletal_length_in_search_radius = np.sum(total_branches_in_search_radius["skeletal_length"].to_numpy())
        processed_skeletal_length = np.sum(close_limb_branch_dict["skeletal_length"].to_numpy())

        if len(close_limb_branch_dict)>0:
            median_spine_density = np.median(close_limb_branch_dict["spine_density"].to_numpy())
        else:
            median_spine_density = 0

        if verbose:
            print(f'median spine density = {median_spine_density}')
            print(f"Number of branches = {len(close_limb_branch_dict)}")
            print(f"Number of branches in radius = {len(total_branches_in_search_radius)}")
            print(f"processed_skeletal_length = {processed_skeletal_length}")
            print(f"total_skeletal_length_in_search_radius = {total_skeletal_length_in_search_radius}")
    else:
        total_skeletal_length_in_search_radius = 0
        processed_skeletal_length = 0
        median_spine_density=0
        
        
    if return_branch_processed_info:
        return (median_spine_density,
                len(close_limb_branch_dict),processed_skeletal_length,
                len(total_branches_in_search_radius),total_skeletal_length_in_search_radius)
    else:
        return median_spine_density
    
def all_concept_network_data_to_limb_network_stating_info(all_concept_network_data):
    """
    Purpose: Will conver the concept network data list of dictionaries into a 
    the dictionary representation of only the limb touching vertices and
    endpoints of the limb_network_stating_info in the preprocessed data
    
    Pseudocode: 
    Iterate through all of the network dicts and store as
    soma--> soma_group_idx --> dict(touching_verts,
                                    endpoint)
                                    
    stored in the concept network as 
    touching_soma_vertices
    starting_coordinate
    
    """
    limb_network = dict()
    for k in all_concept_network_data:
        soma_idx = k["starting_soma"]
        soma_group_idx = k["soma_group_idx"]
        
        if soma_idx not in limb_network.keys():
            limb_network[soma_idx] = dict()
            
        limb_network[soma_idx][soma_group_idx] = dict(touching_verts=k["touching_soma_vertices"],
                                                     endpoint = k["starting_coordinate"])
        
    return limb_network
    

def clean_all_concept_network_data(all_concept_network_data,
                                  verbose=False):
    
    """
    Purpose: To make sure that there are
    no duplicate entries of that starting nodes
    and either to combine the soma touching points
    or just keep the largest one

    Pseudocode: 
    1) Start with an empty dictionary
    For all the dictionaries:
    2)  store the result
    indexed by starting soma and starting node
    3) If an entry already existent --> then either add the soma touching
    vertices (and unique) to the list or replace it if longer

    4) Turn the one dictionary into a list of dictionaries
    like the all_concept_network_data attribute

    5) Replace the all_concept_network_data


    """

    new_network_data = dict()

    for n_dict in all_concept_network_data:
        starting_soma = n_dict["starting_soma"]
        starting_node = n_dict["starting_node"]

        if starting_soma not in new_network_data.keys():
            new_network_data[starting_soma] = dict()

        if starting_node in new_network_data[starting_soma].keys():
            if (len(new_network_data[starting_soma][starting_node]["touching_soma_vertices"]) < 
                len(n_dict["touching_soma_vertices"])):
                if verbose:
                    print(f"Replacing the Soma_{starting_soma}_Node_{starting_node} dictionary")
                new_network_data[starting_soma][starting_node] = n_dict
            else:
                if verbose:
                    print(f"Skipping the Soma_{starting_soma}_Node_{starting_node} dictionary because smaller")
        else:
            new_network_data[starting_soma][starting_node] = n_dict

    #4) Turn the one dictionary into a list of dictionaries
    #like the all_concept_network_data attribute

    new_network_list = []
    for soma_idx,soma_info in new_network_data.items():
        for idx,(starting_node,node_info) in enumerate(soma_info.items()):
            node_info["soma_group_idx"] = idx
            new_network_list.append(node_info)

    return new_network_list



def clean_neuron_all_concept_network_data(neuron_obj,verbose=False):
    """
    Will go through and clean all of the concept network data
    in all the limbs of a Neuron
    """
    for j,curr_limb in enumerate(neuron_obj):
        if verbose:
            print(f"\n\n---- Working on Limb {j} ----")
            
            
        cleaned_network = nru.clean_all_concept_network_data(curr_limb.all_concept_network_data,
                                                                          verbose=verbose)
        
        if verbose:
            print(f"cleaned_network = {cleaned_network}\n\n")
        
        curr_limb.all_concept_network_data = cleaned_network
        
        #setting the concept network
        st_soma = curr_limb.all_concept_network_data[0]["starting_soma"]
        st_node = curr_limb.all_concept_network_data[0]["starting_node"]
        curr_limb.set_concept_network_directional(starting_soma=st_soma,
                                                 starting_node=st_node)
        
        # --------- 1/24: Cleaning the preprocessed data as well -----------#
        if verbose:
            print(f"cleaned_network = {cleaned_network}")
            
        new_limb_network = nru.all_concept_network_data_to_limb_network_stating_info(cleaned_network)
        
        if verbose:
            print(f"\n---------\nnew_limb_network = {new_limb_network}\n---------\n")
        neuron_obj.preprocessed_data["limb_network_stating_info"][j] = new_limb_network
        
        if verbose:
            print(f"curr_limb.all_concept_network_data = {curr_limb.all_concept_network_data}\n\n")
            
#         neuron_obj[j] = curr_limb
    
#     return neuron_obj

def connected_components_from_branches(
    limb_obj,
    branches,
    use_concept_network_directional=False,
    verbose = False
    ):
    """
    Purpose: to find the connected components on a branch
    """
    
    if use_concept_network_directional:
        curr_network = nx.Graph(limb_obj.concept_network_directional)
    else:
        curr_network = nx.Graph(limb_obj.concept_network_directional)
        #curr_network = limb_obj.concept_network

    axon_subgraph = curr_network.subgraph(branches)
    conn_comp = [np.array(list(k)) for k in nx.connected_components(axon_subgraph)]
    if verbose:
        print(f"conn_comp = {conn_comp}")
    return conn_comp

def limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict,
            use_concept_network_directional=False):
    """
    Purpose: To turn the limb branch dict into a
    list of all the connected components described by the
    limb branch dict
    
    """
    
    axon_connected_comps = []
    for limb_name, axon_branches in limb_branch_dict.items():
        limb_obj = neuron_obj[limb_name]
        conn_comp_pre = nru.connected_components_from_branches(
            limb_obj,axon_branches,
            use_concept_network_directional=use_concept_network_directional)
        conn_comp = [(limb_name,k) for k in conn_comp_pre]
        axon_connected_comps += conn_comp

    return axon_connected_comps


        
def empty_limb_object(labels=["empty"]):
    curr_limb = neuron.Limb(mesh=None,
                        curr_limb_correspondence=dict(),
                         concept_network_dict=dict(),
                        labels=labels)
    curr_limb.concept_network = nx.Graph()
    curr_limb.concept_network_directional = nx.DiGraph()
    return curr_limb

def limb_true_false_dict_to_limb_branch_dict(neuron_obj,
                                       limb_true_false_dict):
    """
    To convert a dictionary that has limb_idx --> branch --> True or False
    
    Pseudocode: 
    For each limb
    1) Make sure that the true false dict lenght matches the number of branches
    Iterate through all the branches
        2) if true then add to local list
    3) store the local list in new limb branch dict
    
    """
    limb_branch_dict = dict()
    
    if len(limb_true_false_dict) != neuron_obj.n_limbs:
        raise Exception(f"limb_true_false_dict ({len(limb_true_false_dict)}) not match neuron_obj.n_limbs ({neuron_obj.n_limbs})")
        
    for limb_idx,true_false_dict in limb_true_false_dict.items():
        
        if len(true_false_dict) != len(neuron_obj[limb_idx]):
            raise Exception(f"True False Dict length ({len(true_false_dict)}) not match len(neuron_obj[limb_idx]) ({len(neuron_obj[limb_idx])})")
            
        local_list = np.array([k for k,v in true_false_dict.items() if v]).astype("int")
        
        if len(local_list) > 0:
            limb_branch_dict[limb_idx] = local_list
        
    return limb_branch_dict

def limb_branch_dict_to_limb_true_false_dict(neuron_obj,
                                            limb_branch_dict):
    """
    To convert limb branch dict to a dictionary of:
        limb_idx --> branch --> True or False
    
    Pseudocode: 
    1) Iterate through the neuron limbs
        a) 
        if the limb is not in limb branch dict: 
            make the limb list empty
        else:
            get limb list
            
        b) Get the branch node names from neuron
        c) Get a diff of the list to find the false values
        d) Iterate through limb_list and make true,
        e) Iterate through diff list and make false
        f) store the local dictionary in the true_false dict for return
        
    """
    true_false_dict = dict()
    for limb_name in neuron_obj.get_limb_node_names():
        
        limb_list = limb_branch_dict.get(limb_name,[])
        branch_names = np.array(neuron_obj[limb_name].get_branch_names())
        
        true_false_list = np.zeros(len(branch_names)).astype("bool")
        true_false_list[limb_list] = True
        
        output_dict = {k:v for k,v in enumerate(true_false_list)}
        true_false_dict[limb_name] = output_dict
        
        
#         false_list = np.setdiff1d(branch_names,limb_list)
#         true_dict = {int(k):True for k in limb_list}
#         false_dict = {int(k):False for k in false_list}
#         true_dict.update(false_dict)
#          true_false_dict[limb_name] = true_dict
        
    return true_false_dict
        
    
    
def concatenate_feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature,
                                     feature_function=None,):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    """
    
    feature_total = []
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            feature_value = getattr(neuron_obj[limb_name][b],feature)
            if feature_function is not None:
                feature_value = feature_function(feature_value)
                
            
            feature_total += feature_value
            
    return feature_total        

def sum_feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature=None,
                                      branch_func_instead_of_feature = None,
                                     feature_function=None):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    """
    
    feature_total = 0
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            if feature == "n_branches":
                feature_value = 1
            elif branch_func_instead_of_feature is not None:
                feature_value = branch_func_instead_of_feature(neuron_obj[limb_name][b])
            else:
                feature_value = getattr(neuron_obj[limb_name][b],feature)
            if feature_function is not None:
                feature_value = feature_function(feature_value)
            feature_total += feature_value
            
    return feature_total

def feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature=None,
                                     feature_function=None,
                                  feature_from_fuction = None,
                                  feature_from_fuction_kwargs = None,
                                  keep_seperate=False,
                                  branch_func_instead_of_feature = None,
                                 skip_None=True):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    
    
    """
    
    feature_total = []
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            if feature_from_fuction is not None:
                if feature_from_fuction_kwargs is None:
                    feature_from_fuction_kwargs = dict()
                feature_value = feature_from_fuction(neuron_obj[limb_name][b],**feature_from_fuction_kwargs)
            else:
                if branch_func_instead_of_feature is not None:
                    feature_value = branch_func_instead_of_feature(neuron_obj[limb_name][b])
                else:
                    feature_value = getattr(neuron_obj[limb_name][b],feature)
                if feature_function is not None:
                    feature_value = feature_function(feature_value)
            
            if skip_None and feature_value is None:
                continue
            
            if keep_seperate:
                feature_total.append(feature_value)
            else:
                if nu.is_array_like(feature_value):
                    feature_total+= list(feature_value)
                else:
                    feature_total.append(feature_value)
            
    return feature_total


def limb_branch_removed_after_limb_branch_removal(neuron_obj,
                                      limb_branch_dict,
                             return_removed_limb_branch = False,
                             verbose=False
                            ):
    """
    Purpose: To take a branches that should be deleted from
    different limbs in a limb branch dict then to determine all of the
    branches that were removed from this deletion due to 
    disconnecting from starting branch
    """
    return limb_branch_after_limb_branch_removal(neuron_obj,
                                      limb_branch_dict,
                             return_removed_limb_branch = True,
                             verbose=verbose
                            )

def limb_branch_after_limb_branch_removal(neuron_obj,
                                      limb_branch_dict,
                             return_removed_limb_branch = False,
                             verbose=False
                            ):

    """
    Purpose: To take a branches that should be deleted from
    different limbs in a limb branch dict then to determine the leftover branches
    of each limb that are still connected to the starting node



    Pseudocode:
    For each starting node
    1) Get the starting node
    2) Get the directional conept network and turn it undirected
    3) Find the total branches that will be deleted and kept
    once the desired branches are removed (only keeping the ones 
    still connected to the starting branch)
    4) add the removed and kept branches to the running limb branch dict

    """
    

    limb_branch_dict_kept = dict()
    limb_branch_dict_removed = dict()

    for limb_name in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_name]
        branch_names = limb_obj.get_branch_names()

        if limb_name not in limb_branch_dict.keys():
            limb_branch_dict_kept[limb_name] = branch_names
            continue



        nodes_to_remove = limb_branch_dict[limb_name]

        G = nx.Graph(limb_obj.concept_network_directional)
        nodes_to_keep = limb_obj.current_starting_node

        kept_branches,removed_branches = xu.nodes_in_kept_groups_after_deletion(G,
                                            nodes_to_keep,
                                               nodes_to_remove=nodes_to_remove,
                                            return_removed_nodes = True
                                               ) 
        if len(kept_branches)>0:
            limb_branch_dict_kept[limb_name] = np.array(kept_branches)
        if len(removed_branches) > 0:
            limb_branch_dict_removed[limb_name] = np.array(removed_branches)

    if return_removed_limb_branch:
        return limb_branch_dict_removed
    else:
        return limb_branch_dict_kept
    
# ------ 2/1: Utils for a lot of the edge functions ----------- #
def limb_edge_dict_with_function(neuron_obj,
                                edge_function,
                                verbose=False,
                                **kwargs):

    """
    Purpose: To create a limb_edge dictionary
    based on a function that generates cuts for a certain limb
    
    Funciton must pass back: edges_to_create,edges_to_delete

    Pseudocode: 
    Iterate through all of the limbs of a neuron
    a. Get the cuts that should be created and deleted
    b. If either is non-empty then add to the limb_edge dictionary

    return limb_edge dictionary
    """


    limb_edge_dict = dict()
    for limb_name in neuron_obj.get_limb_node_names():

        limb_obj = neuron_obj[limb_name]

        edges_to_create,edges_to_delete = edge_function(limb_obj,
                                          verbose=verbose,**kwargs)

        if verbose:
            print(f"\n--- Working on Limb {limb_name} ---\n"
                 f"edges_to_create = {edges_to_create}\n"
                 f"edges_to_create = {edges_to_create}")

        edges_to_create = list(nu.unique_rows(edges_to_create))
        edges_to_delete = list(nu.unique_rows(edges_to_delete))
        
        if len(edges_to_create)>0 or len(edges_to_delete):
            limb_edge_dict[limb_name] = dict(edges_to_create=edges_to_create,
                                            edges_to_delete=edges_to_delete)
        
            

    return limb_edge_dict    



def branches_on_limb_after_edges_deleted_and_created(
    limb_obj,
    edges_to_delete=None,
    edges_to_create=None,
    return_removed_branches = False,
    verbose=False):                       

    """
    Purpose: To take a edges of concept network that should 
    be created or destroyed and then returning the branches that still remain
    and those that were deleted
    """

    original_branches = list(limb_obj.concept_network.nodes())
    
    if edges_to_delete is not None:
        edges_to_delete = list(edges_to_delete)
    
    if edges_to_create is not None:
        edges_to_create = list(edges_to_create)
        

    new_concept_network = xu.create_and_delete_edges(limb_obj.concept_network,
        edges_to_delete=edges_to_delete,
        edges_to_create=edges_to_create,
        return_copy=True,
            )

    kept_branches,removed_branches = xu.nodes_in_kept_group(new_concept_network,
                                                nodes_to_keep=limb_obj.current_starting_node,
                                                return_removed_nodes = True,
                                                            verbose=False,
                                                   )
    if verbose:
        print("After edges deleted and created: ")
        print(f"kept_branches= {kept_branches}")
        print(f"removed_branches = {removed_branches}")
        
    if return_removed_branches:
        return kept_branches,removed_branches
    else:
        return kept_branches    
    
def limb_branch_after_limb_edge_removal(neuron_obj,
                                      limb_edge_dict,
                             return_removed_limb_branch = False,
                             verbose=False
                            ):

    """
    Purpose: To take a branches that should be deleted from
    different limbs in a limb branch dict then to determine the leftover branches
    of each limb that are still connected to the starting node



    Pseudocode:
    For each starting node
    1) Get the starting node
    2) Get the directional conept network and turn it undirected
    3) Find the total branches that will be deleted and kept
    once the desired branches are removed (only keeping the ones 
    still connected to the starting branch)
    4) add the removed and kept branches to the running limb branch dict

    """
    

    limb_branch_dict_kept = dict()
    limb_branch_dict_removed = dict()

    for limb_name in neuron_obj.get_limb_node_names():
        if verbose:
            print(f"\n--- Working on Limb {limb_name} ---\n")
                  
        limb_obj = neuron_obj[limb_name]
        branch_names = limb_obj.get_branch_names()

        if limb_name not in limb_edge_dict.keys():
            limb_branch_dict_kept[limb_name] = branch_names
            if verbose:
                print("skipping because was not in the limb_edge dict")
            continue



        edges_to_delete = limb_edge_dict[limb_name].get("edges_to_delete",[])
        edges_to_create = limb_edge_dict[limb_name].get("edges_to_create",[])


        kept_branches,removed_branches = branches_on_limb_after_edges_deleted_and_created(
                                            limb_obj,
                                            edges_to_delete=edges_to_delete,
                                            edges_to_create=edges_to_create,
                                            return_removed_branches = True,
                                            verbose=verbose)
        if len(kept_branches)>0:
            limb_branch_dict_kept[limb_name] = kept_branches
        if len(removed_branches) > 0:
            limb_branch_dict_removed[limb_name] = removed_branches

    if return_removed_limb_branch:
        return limb_branch_dict_removed
    else:
        return limb_branch_dict_kept
    
def limb_branch_from_edge_function(neuron_obj,
                                   edge_function,
                                   verbose=False,
                                    **kwargs):
    """
    Purpose: To generate a limb branch dict of nodes
    from a function that generates cuts for a neuron_limb
    
    Pseudocode:
    1) Generate a limb_edge dictionary
    2) Generate a limb branch dictionary and return that
    """
    
    limb_edge_dict = nru.limb_edge_dict_with_function(neuron_obj,
                                                    edge_function,
                                                      verbose=verbose,
                                                     **kwargs)
    
    limb_branch_dict = nru.limb_branch_after_limb_edge_removal(neuron_obj,
                                       limb_edge_dict=limb_edge_dict,
                                        return_removed_limb_branch=True,
                                        verbose=verbose,
                                       )
    return limb_branch_dict
    

def branches_within_skeletal_distance(limb_obj,
                                    start_branch,
                                    max_distance_from_start,
                                    verbose = False,
                                    include_start_branch_length = False,
                                    include_node_branch_length = False,
                                    only_consider_downstream = False):

    """
    Purpose: to find nodes within a cetain skeletal distance of a certain 
    node (can be restricted to only those downstream)

    Pseudocode: 
    1) Get the directed concept grpah
    2) Get all of the downstream nodes of the node
    3) convert directed concept graph into an undirected one
    4) Get a subgraph using all of the downstream nodes
    5) For each node: 
    - get the shortest path from the node to the starting node
    - add up the skeleton distance (have options for including each endpoint)
    - if below the max distance then add
    6) Return nodes


    Ex: 
    start_branch = 53
        
    viable_downstream_nodes = nru.branches_within_skeletal_distance(limb_obj = current_neuron[6],
                                start_branch = start_branch,
                                max_distance_from_start = 50000,
                                verbose = False,
                                include_start_branch_length = False,
                                include_node_branch_length = False,
                                only_consider_downstream = True)

    limb_branch_dict=dict(L6=viable_downstream_nodes+[start_branch])

    nviz.plot_limb_branch_dict(current_neuron,
                              limb_branch_dict)

    """

    curr_limb = limb_obj



    viable_downstream_nodes = []

    dir_nx = curr_limb.concept_network_directional

    #2) Get all of the downstream nodes of the node

    if only_consider_downstream:
        all_downstream_nodes = list(xu.all_downstream_nodes(dir_nx,start_branch))
    else:
        all_downstream_nodes = list(dir_nx.nodes())
        all_downstream_nodes.remove(start_branch)

    if len(all_downstream_nodes) == 0:
        if verbose:
            print(f"No downstream nodes to test")

        return []

    if verbose:
        print(f"Number of downstream nodes = {all_downstream_nodes}")

    #3) convert directed concept graph into an undirected one
    G_whole = nx.Graph(dir_nx)

    #4) Get a subgraph using all of the downstream nodes
    G = G_whole.subgraph(all_downstream_nodes + [start_branch])

    for n in all_downstream_nodes:

        #- get the shortest path from the node to the starting node
        try:
            curr_shortest_path = nx.shortest_path(G,start_branch,n)
        except:
            if verbose:
                print(f"Continuing because No path between start node ({start_branch}) and node {n}")
            continue 


        if not include_node_branch_length:
            curr_shortest_path = curr_shortest_path[:-1]

        if not include_start_branch_length:
            curr_shortest_path = curr_shortest_path[1:]

        total_sk_length_of_path = np.sum([curr_limb[k].skeletal_length for k in curr_shortest_path])

        if total_sk_length_of_path <= max_distance_from_start:
            viable_downstream_nodes.append(n)
        else:
            if verbose:
                print(f"Branch {n} was too far from the start node : {total_sk_length_of_path} (threshold = {max_distance_from_start})")

    return viable_downstream_nodes

def low_branch_length_clusters(neuron_obj,
                              max_skeletal_length = 8000,
                                min_n_nodes_in_cluster = 4,
                               width_max = None,
                               skeletal_distance_from_soma_min = None,
                               use_axon_like_restriction = False,
                               verbose=False,
                               remove_starting_node = True,
                               limb_branch_dict_restriction = None,
                               plot = False,
                               
                               **kwargs
                                ):

    """
    Purpose: To find parts of neurons with lots of nodes
    close together on concept network with low branch length
    
    Pseudocode:
    1) Get the concept graph of a limb 
    2) Eliminate all of the nodes that are too long skeletal length
    3) Divide the remaining axon into connected components
    - if too many nodes are in the connected component then it is
    an axon mess and should delete all those nodes
    
    Application: Helps filter away axon mess

    """
    
    if verbose:
        print(f"max_skeletal_length = {max_skeletal_length}")
        print(f"min_n_nodes_in_cluster= {min_n_nodes_in_cluster}")
        print(f"limb_branch_dict_restriction = {limb_branch_dict_restriction}")
    
    use_deletion=False
    
    curr_neuron_obj=neuron_obj

    if width_max is None:
        width_max = np.inf
        
    if skeletal_distance_from_soma_min is None:
        skeletal_distance_from_soma_min = -1

    limb_branch_dict = dict()
    
    
    
    # ---------- Getting the restriction that we will check over ---- #
    if use_axon_like_restriction:
        axon_limb_branch_dict = clu.axon_like_limb_branch_dict(curr_neuron_obj)
    else:
        axon_limb_branch_dict = limb_branch_dict_restriction
        
    if verbose:
        print(f"limb_branch_dict_restriction before query = {limb_branch_dict_restriction}")
    

    if not use_deletion:
        limb_branch_restriction = ns.query_neuron(curr_neuron_obj,
                        functions_list=["skeletal_length",
                                        "median_mesh_center",
                                       "skeletal_distance_from_soma"],
                       query = ( f" (skeletal_length < {max_skeletal_length}) and "
                               f" (median_mesh_center < {width_max}) "
                               f" and (skeletal_distance_from_soma > {skeletal_distance_from_soma_min})"),
                       limb_branch_dict_restriction=axon_limb_branch_dict)
        if verbose:
            print(f"limb_branch_restriction = {limb_branch_restriction}")
    else:
        limb_branch_restriction = nru.neuron_limb_branch_dict(curr_neuron_obj)


    for limb_name,nodes_to_keep in limb_branch_restriction.items():
        curr_limb = curr_neuron_obj[limb_name]
        curr_starting_node = curr_neuron_obj[limb_name].current_starting_node
        
        if verbose:
            print(f"--- Working on Limb {limb_name} ---")

        if use_deletion:
        #1) Get the branches that are below a certain threshold
            nodes_to_delete = [jj for jj,branch in enumerate(curr_limb) 
                               if ((curr_limb[jj].skeletal_length > max_skeletal_length ))]

            if verbose:
                print(f"nodes_to_delete = {nodes_to_delete}")

            #2) Elimnate the nodes from the concept graph
            G_short = nx.Graph(curr_limb.concept_network)
            G_short.remove_nodes_from(nodes_to_delete)
        
        else:
            #2) Elimnate the nodes from the concept graph
            G= nx.Graph(curr_limb.concept_network)
            G_short = G.subgraph(nodes_to_keep)

            if verbose:
                print(f"nodes_to_keep = {nodes_to_keep}")

        
        #3) Divide the remaining graph into connected components
        conn_comp = [list(k) for k in nx.connected_components(G_short)]

        potential_error_branches = []

        for c in conn_comp:
            if remove_starting_node:
                c = np.array(c)
                c = c[c != curr_starting_node]
                
            if len(c) > min_n_nodes_in_cluster:
                potential_error_branches += list(c)

        #4)  If found any error nodes then add to limb branch dict
        if len(potential_error_branches) > 0:
            limb_branch_dict[limb_name] = potential_error_branches

            
    if plot:
        print(f"Plotting final low_branch clusters = {limb_branch_dict}")
        if len(limb_branch_dict) > 0:
            nviz.plot_limb_branch_dict(neuron_obj,limb_branch_dict)
        else:
            print(f"---- Nothing to plot -------")
    return limb_branch_dict

def neuron_limb_branch_dict(neuron_obj):
    """
    Purpose: To develop a limb branch dict represnetation
    of the limbs and branchs of a neuron
    
    """
    limb_branch_dict_new = dict()
    
    if neuron_obj.__class__.__name__ == "Neuron":
        for limb_name in neuron_obj.get_limb_node_names():
            limb_branch_dict_new[limb_name] = neuron_obj[limb_name].get_branch_names()
    else:
        net = neuron_obj
        curr_limb_names = [k for k in net.nodes() if "L" in k]
        for limb_name in curr_limb_names:
            limb_branch_dict_new[limb_name] = np.array(list(net.nodes[limb_name]["data"].concept_network.nodes()))

        
    return limb_branch_dict_new

def limb_branch_invert(neuron_obj,
                           limb_branch_dict,
                           verbose=False):
    """
    Purpose: To get every node that is not in limb branch dict
    
    Ex: 
    invert_limb_branch_dict(curr_neuron_obj,limb_branch_return,
                       verbose=True)
    """
    
    limb_branch_dict_new = dict()
    for j,curr_limb in enumerate(neuron_obj):
        
        limb_name = f"L{j}"
        
        if verbose:
            print(f"\n--- Working on limb {limb_name}")
        
        if limb_name in limb_branch_dict:
            curr_branches = limb_branch_dict[limb_name]
        else:
            curr_branches = []
            
        
            
        leftover_branches = np.setdiff1d(curr_limb.get_branch_names(),curr_branches)
        if verbose:
            print(f"curr_branches = {curr_branches}")
            print(f"leftover_branches = {leftover_branches}")
            print(f"total combined branches = {len(curr_branches) +len(leftover_branches) }, len(limb) = {len(curr_limb)}")
        if len(leftover_branches)>0:
            limb_branch_dict_new[limb_name] = leftover_branches
            
    return limb_branch_dict_new

def limb_branch_combining(
                           limb_branch_dict_list,
                           combining_function,
                           verbose=False):
    """
    Purpose: To get every node that is not in limb branch dict
    
    Ex: 
    invert_limb_branch_dict(curr_neuron_obj,limb_branch_return,
                       verbose=True)
    """
    if len(limb_branch_dict_list) == 0:
        return dict()
    all_keys = nu.union1d_multi_list([list(k.keys()) for k in limb_branch_dict_list])
    
    
    limb_branch_dict_new = dict()
    for limb_name in all_keys:
        
        if verbose:
            print(f"\n--- Working on limb {limb_name}")
        
        curr_branches = [k.get(limb_name,[]) for k in limb_branch_dict_list]
        
        leftover_branches = nu.function_over_multi_lists(curr_branches,combining_function)
        
        if verbose:
            print(f"combining_function = {combining_function}")
            print(f"curr_branches = {curr_branches}")
            print(f"leftover_branches = {leftover_branches}")
            
        if len(leftover_branches)>0:
            limb_branch_dict_new[limb_name] = leftover_branches
            
    return limb_branch_dict_new

def limb_branch_setdiff(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.setdiff1d,
                           verbose=False)

def limb_branch_union(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.union1d,
                           verbose=False)

def limb_branch_intersection(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.intersect1d,
                           verbose=False
    )

def limb_branch_dict_valid(neuron_obj,
                          limb_branch_dict):
    """
    Will convert a limb branch dict input with shortcuts
    (like "axon" or "all") into a valid limb branch dict
    
    Ex: 
    limb_branch_dict_valid(neuron_obj,
                      limb_branch_dict = dict(L2="all",L3=[3,4,5]))
    """
    if limb_branch_dict == "axon":
        ax_name = neuron_obj.axon_limb_name
        if ax_name is None:
            limb_branch_dict = {}
        else:
            limb_branch_dict = {ax_name:"all"}
    
    if limb_branch_dict == "all":
        return neuron_obj.limb_branch_dict
    
    final_limb_branch_dict = dict()
    for limb_name,branch_list in limb_branch_dict.items():
        if branch_list == "all":
            branch_list = neuron_obj[limb_name].get_branch_names()
        final_limb_branch_dict[limb_name] = branch_list
    
    return final_limb_branch_dict

def limb_branch_get(limb_branch_dict,limb_name):
    """
    Will get the branches associated with a certain limb idx or limb name 
    (with checks for it not being there)
    
    Ex: 
    limb_idx = 0
    short_thick_limb_branch = au.short_thick_branches_limb_branch_dict(neuron_obj_exc_syn_sp,
                                            plot_limb_branch_dict = False)
    nodes_to_exclude = nru.limb_branch_get(short_thick_limb_branch,limb_idx)
    nodes_to_exclude
    """
    limb_name = nru.get_limb_string_name(limb_name)
    
    if limb_name in limb_branch_dict.keys():
        return limb_branch_dict[limb_name]
    else:
        return np.array([])

def in_limb_branch_dict(limb_branch_dict,limb_idx,branch_idx=None,verbose = False):
    """
    Will return true or false if limb and branch in limb branch dict
    """
    limb_in = False
    branch_in = False
    
    limb_name = nru.get_limb_string_name(limb_idx)
    
    limb_in = limb_name in limb_branch_dict.keys()
    
    if branch_idx is None:
        return limb_in
    
    if limb_in:
        branches = limb_branch_dict[limb_name]
        branch_in = branch_idx in branches
        
    if verbose:
        print(f"For limb = {limb_idx}, branch_idx = {branch_idx}")
        print(f"limb_in = {limb_in}, branch_in = {branch_in}")
    
    return branch_in and limb_in
# ----------- For rules with doubling back, width jumps, high degree nodes, train track crossings -------- #

def high_degree_branching_coordinates_on_limb(limb_obj,
                                              min_degree_to_find=5,
                                             exactly_equal=False,
                                             verbose=False):
    """
    Purpose: To find high degree coordinates on a limb
    
    """
    
    
    #1) Get the limb skeleton
    #2) Convert the limb skeleton to a graph
    #limb_sk_gr = sk.convert_skeleton_to_graph(limb.skeleton)
    
    return sk.high_degree_coordinates_on_skeleton(limb_obj.skeleton,
                                                 min_degree_to_find=min_degree_to_find,
                                                  exactly_equal=exactly_equal,
                                                 verbose=verbose)

    

def branches_at_high_degree_coordinates(limb_obj,
                                        min_degree_to_find=5,
                                        
                                        **kwargs
                                       ):
    """
    Purpose: To identify branches groups that are touching 
    skeleton nodes that have nax_degree or more branches touching them


    Pseudocode: 
    1) Find the coordinates wtih max_degree
    For each coordinate
    2) Find branches that correspond to that coordinate and store as group
    """
    print(f"min_degree_to_find = {min_degree_to_find}")
    
    curr_high_degree_coordinates = nru.high_degree_branching_coordinates_on_limb(limb_obj,
                                                                min_degree_to_find = min_degree_to_find,
                                                                **kwargs)

    high_degree_groups = [list(nru.find_branch_with_specific_coordinate(limb_obj,c_coord)) 
                                  for c_coord in curr_high_degree_coordinates]
    return high_degree_groups
    
    
def high_degree_branching_coordinates_on_neuron(neuron_obj,
                                 min_degree_to_find = 5,
                                      exactly_equal = False,
                                 verbose = False):
    """
    Purpose: To find coordinate where high degree branching coordinates occur
    
    
    """

    limb_high_degree_coordinates = []

    for i,limb in enumerate(neuron_obj):

        if verbose:
            print(f"--- Working on Limb {i} ---")

        curr_high_degree_coordinates = high_degree_branching_coordinates_on_limb(limb,
                                              min_degree_to_find=min_degree_to_find,
                                             exactly_equal=exactly_equal,
                                             verbose=verbose)
        if len(curr_high_degree_coordinates)>0:
            
            limb_high_degree_coordinates += list(curr_high_degree_coordinates)

    return limb_high_degree_coordinates
    
    
def ordered_endpoints_on_branch_path(limb_obj,
            path,
            starting_endpoint_coordinate):

    """
    Purpose: To get the ordered endpoints of the skeletons 
    of a path of branches starting at one endpoint

    """


    branch_skeletons = [limb_obj[k].skeleton for k in path]

    ordered_endpoints = sk.order_skeletons_connecting_endpoints(branch_skeletons,
    starting_endpoint_coordinate=starting_endpoint_coordinate)

    return ordered_endpoints


def axon_only_group(limb_obj,
                   branches,
                   use_axon_like=True,
                   verbose=False):
    """
    checks group or branches and returns true if all are axon
    or axon-dependent
    """
    labels_to_check = ["axon"]
    if use_axon_like:
        labels_to_check.append("axon-like")
        
    return_value = True
    
    for b in branches:
        if len(np.intersect1d(limb_obj[b].labels,labels_to_check)) == 0:
            if verbose:
                print(f"branch {b} did not have one of the following in their labels : {labels_to_check}")
            return_value=False
            break
            
    return return_value

def max_soma_volume(neuron_obj,
                    divisor = 1_000_000_000):
    """
    Will find the largest number of faces out of all the somas
    
    """
    #soma_volumes = [neuron_obj[k].volume/divisor for k in neuron_obj.get_soma_node_names()] 
    
    try:
        soma_volumes = [tu.mesh_volume(neuron_obj[k].mesh)/divisor for k in neuron_obj.get_soma_node_names()] 
    except:
        return 0
    
    largest_volume = np.max(soma_volumes)
    return largest_volume

def max_soma_n_faces(neuron_obj):
    """
    Will find the largest number of faces out of all the somas
    
    """
    soma_areas = [len(neuron_obj[k].mesh.faces) for k in neuron_obj.get_soma_node_names()] 
    largest_soma_area = np.max(soma_areas)
    return largest_soma_area

def max_soma_area(neuron_obj):
    """
    Will find the largest number of faces out of all the somas
    
    """
    soma_n_faces = [neuron_obj[k].area for k in neuron_obj.get_soma_node_names()] 
    largest_n_faces = np.max(soma_n_faces)
    return largest_n_faces



def soma_centers(neuron_obj,
                 soma_name=None,
                voxel_adjustment=False,
                 voxel_adjustment_vector = None,
                 return_int_form=True,
                return_single=True):
    """
    Will come up with the centers predicted for each of the somas in the neuron
    """
    if voxel_adjustment_vector is None:
        voxel_adjustment_vector = voxel_to_nm_scaling
    
    if soma_name is None:
#         current_soma_means = np.array([tu.mesh_center_vertex_average(neuron_obj[s_name].mesh) 
#                                for s_name in neuron_obj.get_soma_node_names()])
        current_soma_means = np.array([neuron_obj[soma_name].mesh_center
                               for s_name in neuron_obj.get_soma_node_names()])
    else:
        current_soma_means = np.array([neuron_obj[soma_name].mesh_center]).reshape(-1,3)
    
    if voxel_adjustment and voxel_adjustment_vector is not None:
        
        current_soma_means = current_soma_means/voxel_adjustment_vector
        
    if return_int_form:
        current_soma_means = current_soma_means.astype("int")
        
    if return_single:
        if len(current_soma_means) != 1:
            raise Exception(f"Not just one soma center: {current_soma_means}")
        current_soma_means=current_soma_means[0]
        
    return current_soma_means
    
def check_points_inside_soma_bbox(neuron_obj,
                             coordinates,
                            soma_name="S0",
                            voxel_adjustment=False,
                             verbose=False
                            ):
    """
    Purpose: Test if points are inside soma bounding box
    
    """
    if voxel_adjustment:
        divisor = voxel_to_nm_scaling
    else:
        divisor = [1,1,1]
    inside_point_idxs = tu.check_coordinates_inside_bounding_box(neuron_obj[soma_name].mesh,
                                         coordinates=coordinates,
                                          bbox_coordinate_divisor=divisor,
                                         verbose=verbose)
    return inside_point_idxs

def pair_neuron_obj_to_nuclei(
    neuron_obj,
    soma_name,
    nucleus_ids,
    nucleus_centers,
    nuclei_distance_threshold = 15000,
    return_matching_info = True,
    return_id_0_if_no_matches=True,
    return_nuclei_within_radius=False,
    return_inside_nuclei=False,
    verbose=False,
    default_nuclei_id = None,
    ):

    """
    Pseudocode: 
    1) Get the Soma Center
    2) Get all Nuclei within a certain distance of the Soma Center
    3) If any Nuclei Found, Get the closest one and the distance
    4) Get the number of nuceli within the bouding box:
    -if No Nuclei were found and one was found within the bounding
    box then use that one

    """
    
    if nucleus_ids is None:
        
        winning_nuclei = default_nuclei_id
        nuclei_distance=None
        n_nuclei_in_radius=None
        n_nuclei_in_bbox=None
        
        matching_info = dict(nucleus_id=winning_nuclei,
                        nuclei_distance=None,
                        n_nuclei_in_radius=n_nuclei_in_radius,
                        n_nuclei_in_bbox=n_nuclei_in_bbox)

        

    else:
        #1) Get the Soma Center
        soma_center = nru.soma_centers(neuron_obj,soma_name)

        if verbose:
            print(f"soma_center = {soma_center}")
            print(f"nucleus_centers= {nucleus_centers}")



        #2) Get all Nuclei within a certain distance of the Soma Center


        nuclei_distances = np.linalg.norm(nucleus_centers-soma_center,axis=1)
        if verbose:
            print(f"nuclei_distances = {nuclei_distances}")
        
        nuclei_within_radius_idx = np.where(nuclei_distances<nuclei_distance_threshold)[0]
        nuclei_within_radius = [nucleus_ids[k] for k in nuclei_within_radius_idx]
        nuclei_within_radius_distance = nuclei_distances[nuclei_within_radius_idx]

        if verbose:
            print(f"nuclei_within_radius = {nuclei_within_radius}")
            print(f"nuclei_within_radius_distance = {nuclei_within_radius_distance}")


        n_nuclei_in_radius = len(nuclei_within_radius)

        winning_nuclei = None
        winning_nuclei_distance = None


        #3) If any Nuclei Found, Get the closest one and the distance
        if len(nuclei_within_radius)>0:

            winning_nuclei_idx = np.argmin(nuclei_within_radius_distance)
            winning_nuclei = nuclei_within_radius[winning_nuclei_idx]
            winning_nuclei_distance = nuclei_within_radius_distance[winning_nuclei_idx]

            if verbose:
                print(f"\nThere were {n_nuclei_in_radius} nuclei found within the radius of {nuclei_distance_threshold} nm")
                print(f"winning_nuclei = {winning_nuclei}")
                print(f"winning_nuclei_distance = {winning_nuclei_distance}")


        #4) Get the number of nuceli within the bouding box:
        inside_nuclei_idx = nru.check_points_inside_soma_bbox(neuron_obj,
                                    coordinates=nucleus_centers,
                                    soma_name="S0",
                                    )

        inside_nuclei = [nucleus_ids[k] for k in inside_nuclei_idx]
        inside_nuclei_distance = nuclei_distances[inside_nuclei_idx]

        n_nuclei_in_bbox = len(inside_nuclei)

        if verbose:
            print("\n For Bounding Box Search:")
            print(f"inside_nuclei = {inside_nuclei}")


        if winning_nuclei is None and len(inside_nuclei)>0:
            winning_nuclei_idx = np.argmin(inside_nuclei_distance)
            winning_nuclei = inside_nuclei[winning_nuclei_idx]
            winning_nuclei_distance = inside_nuclei_distance[winning_nuclei_idx]


            if verbose:
                print(f"\nUsed the Bounding Box to find the winning Nuclei")
                print(f"winning_nuclei = {winning_nuclei}")
                print(f"winning_nuclei_distance = {winning_nuclei_distance}")

        if return_id_0_if_no_matches:
            if winning_nuclei is None:
                winning_nuclei = 0
            if winning_nuclei_distance is None:
                try:
                    winning_nuclei_distance = np.min(nuclei_distances)
                except:
                    winning_nuclei_distance = -1
                
        if verbose:
            print(f"\n\nAt End: using return_id_0_if_no_matches = {return_id_0_if_no_matches}")
            
            print(f"winning_nuclei = {winning_nuclei}")
            print(f"winning_nuclei_distance = {winning_nuclei_distance}")
            print(f"n_nuclei_in_radius = {n_nuclei_in_radius}")
            print(f"n_nuclei_in_bbox = {n_nuclei_in_bbox}")

        matching_info = dict(nucleus_id=winning_nuclei,
                            nuclei_distance=np.round(winning_nuclei_distance,2),
                            n_nuclei_in_radius=n_nuclei_in_radius,
                            n_nuclei_in_bbox=n_nuclei_in_bbox)
        
        
        
    if not (return_matching_info + return_nuclei_within_radius + return_inside_nuclei):
        return winning_nuclei
    
    return_value = [winning_nuclei]
    
    
    if return_matching_info:
        return_value.append(matching_info)
        
    if return_nuclei_within_radius:
        return_value.append(nuclei_within_radius)
    if return_inside_nuclei:
        return_value.append(inside_nuclei)
        
    return return_value
    

# ---- 2/15: For helping with backtracking synapses back to the somas -------- #

def original_mesh_face_to_limb_branch(neuron_obj,
                                      original_mesh=None,
                                      original_mesh_kdtree=None,
                                      add_soma_label = True,
                                     verbose = False
                                     ):
    """
    Pupose: To create a mapping from the original mesh faces
    to which limb and branch it corresponds to 
    
    
    Ex:
    original_mesh_face_idx_to_limb_branch = nru.original_mesh_face_to_limb_branch(neuron_obj,
                                 original_mesh)
    
    matching_faces  = np.where((original_mesh_face_idx_to_limb_branch[:,0]==3) & 
            (original_mesh_face_idx_to_limb_branch[:,1]== 2))[0]
    nviz.plot_objects(original_mesh.submesh([matching_faces],append=True))
    
    
    """
    if original_mesh is None:
        original_mesh = neuron_obj.mesh
    if original_mesh_kdtree is None:
        if verbose:
            print(f"Having ot generate KDTree from scratch")
        original_mesh_kdtree = KDTree(original_mesh.triangles_center)

    original_mesh_face_idx_to_limb_branch = np.ones((len(original_mesh.faces),2))*np.nan    
    
    t_time = time.time()
    for limb_idx,limb_obj in enumerate(neuron_obj):
        if verbose:
            print(f"\nStarting Limb {limb_idx}")
        for branch_idx,branch_obj in enumerate(limb_obj):
            st = time.time() 
            #1) Can create a mapping of the original mesh face
            original_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                    branch_obj.mesh,
                                                    exact_match=True,
                                                    original_mesh_kdtree=original_mesh_kdtree)



            #2) Map original mesh face --> [limb,branch]
            original_mesh_face_idx_to_limb_branch[original_mesh_faces] = [limb_idx,branch_idx]


            if verbose:
                print(f"-- Time for L{limb_idx}_B{branch_idx}: {time.time() - st} ---")
    
    if add_soma_label:
        offset = nru.soma_face_offset
        soma_idxs = nru.convert_string_names_to_int_names(neuron_obj.get_soma_node_names())
        for s_idx in soma_idxs:
            curr_label = -1*(s_idx + offset)
            if verbose:
                print(f"\nApplying Soma Label {s_idx}")
                
            st = time.time() 
            #1) Can create a mapping of the original mesh face
            original_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                    neuron_obj[f"S{s_idx}"].mesh,
                                                    exact_match=True,
                                                    original_mesh_kdtree=original_mesh_kdtree)



            #2) Map original mesh face --> [limb,branch]
            original_mesh_face_idx_to_limb_branch[original_mesh_faces] = [curr_label,curr_label]

            if verbose:
                print(f"-- Time for Soma Label {s_idx} ---")
                
        
    if verbose:
        print(f"\n\nTotal time for mapping: {time.time() - t_time}")
        
    return original_mesh_face_idx_to_limb_branch
        
    
    

def distance_to_soma_from_coordinate_close_to_branch(neuron_obj,
                                                     coordinate,
                                                    limb_idx,
                                                    branch_idx,
                                                    limb_graph=None,
                                                    destination_node=None):
    """
    Purpose: To find the distance traced along the skeleton to the soma
    of a coordinate close to a specific branch on a limb of a neuron
    
    """
    curr_limb = neuron_obj[limb_idx]
    if limb_graph is None:
        limb_graph = sk.convert_skeleton_to_graph(curr_limb.skeleton)
    
    if destination_node is None:
        curr_starting_coordinate = curr_limb.current_starting_coordinate
        destination_node = xu.get_graph_node_by_coordinate(curr_limb_graph,
                                    curr_starting_coordinate,
                                    return_single_value=True)
        
    curr_branch_skeleton = curr_limb[branch_idx].skeleton

    syn_coord_sk = sk.closest_skeleton_coordinate(curr_branch_skeleton,
                            coordinate)

    output_distance = sk.skeleton_path_between_skeleton_coordinates(
                starting_coordinate = syn_coord_sk,
                skeleton_graph = limb_graph,
                destination_node = destination_node,
                only_skeleton_distance = True,)
    
    return output_distance



def synapse_skeletal_distances_to_soma(neuron_obj,
                                       synapse_coordinates,
                                       original_mesh = None,
                                       original_mesh_kdtree = None,
                                       verbose = False,
                                       scale="um"
                                       
                                      ):
    """
    Purpose: To calculate the distance of synapses to the soma

    Pseudocode: 
    A) Create the mapping of original face idx to (limb,branch)
    B) Map Synapses to the original face idx to get
        synapse --> (limb,branch)
    C) Calculate the limb skeleton graphs before hand
    D) For each synapse coordinate:
    1) Calculate the closest skeleton point on the (limb,branch)
    2) Calculate distance from skeleton point to the starting coordinate of branch
    
    ** The soma distances that are -1 are usually the ones that are errored or 
    are on the actual soma *****
    """
    if original_mesh is None:
        original_mesh = neuron_obj.mesh
    if original_mesh_kdtree is None:
        if verbose:
            print(f"Having ot generate KDTree from scratch")
        original_mesh_kdtree = KDTree(original_mesh.triangles_center)
    
    if len(synapse_coordinates) == 0:
        return []

    filtered_neuron = neuron_obj
    all_syn_coord = synapse_coordinates

    # Part A: Get Mapping from original mesh faces to limb/branch
    face_to_limb_branch = nru.original_mesh_face_to_limb_branch(filtered_neuron,
                                                               original_mesh,
                                                               original_mesh_kdtree)

    # Part B: Map Synapses to the original faces
    dist,closest_face = original_mesh_kdtree.query(all_syn_coord)
    coord_limb_branch_map = face_to_limb_branch[closest_face]


    #Part C: Calculate the limb skeletons as graph

    unique_limbs = np.unique(coord_limb_branch_map[:,0])
    unique_limbs = unique_limbs[~np.isnan(unique_limbs)].astype("int")

    limb_graphs = dict()
    limb_destination_nodes = dict()

    for k in unique_limbs:

        curr_limb = filtered_neuron[k]
        curr_limb_graph = sk.convert_skeleton_to_graph(curr_limb.skeleton)
        limb_graphs[k] = curr_limb_graph

        curr_starting_coordinate = curr_limb.current_starting_coordinate
        limb_destination_nodes[k] = xu.get_graph_node_by_coordinate(curr_limb_graph,
                                    curr_starting_coordinate,
                                    return_single_value=True)

    # Part D: get closest skeleton point to coordinate and then distance to starting coordinate
    """
    Pseudocode:
    1) Turn the synapse coordinate into closest branch skeleton coordinate
    2) Find the skeleton distance between starting coordinate and closest skeleton coordinate

    """


    synapse_to_soma_distance = []

    for syn_idx,syn_coord in enumerate(all_syn_coord):
        limb_idx,branch_idx = coord_limb_branch_map[syn_idx]
        if np.isnan(limb_idx) or np.isnan(branch_idx):
            synapse_to_soma_distance.append(-1)
            continue
            
        # --------- 6/4: Accounts for the labels given to the somas --------
        if limb_idx < 0 or branch_idx < 0:
            synapse_to_soma_distance.append(limb_idx)

        limb_idx = int(limb_idx)
        branch_idx = int(branch_idx)

        output_distance = distance_to_soma_from_coordinate_close_to_branch(
                                                    filtered_neuron,
                                                     coordinate=syn_coord,
                                                    limb_idx=limb_idx,
                                                    branch_idx=branch_idx,
                                            limb_graph=limb_graphs[limb_idx],
                                destination_node=limb_destination_nodes[limb_idx])
        
        if scale == "um":
            output_distance = output_distance/1000

        if verbose:
            print(f"Synapse {syn_idx} distance: {output_distance}")

        synapse_to_soma_distance.append(output_distance)
        
    return np.array(synapse_to_soma_distance)

def axon_length(neuron_obj,
                units="um"):
    
    axon_limb_branch_dict = clu.axon_limb_branch_dict(neuron_obj)
    
    axon_skeletal_length = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=axon_limb_branch_dict,
                                     feature="skeletal_length")
    if units == "um":
        axon_skeletal_length = axon_skeletal_length/1000
        
    return axon_skeletal_length

def axon_area(neuron_obj,
                units="um"):
    
    axon_limb_branch_dict = clu.axon_limb_branch_dict(neuron_obj)
    
    axon_mesh_area = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=axon_limb_branch_dict,
                                         feature="area")
    if units == "nm":
        axon_mesh_area = axon_mesh_area*1_000_000
        
    return axon_mesh_area


def axon_mesh(neuron_obj):
    axon_limb_branch_dict = clu.axon_limb_branch_dict(neuron_obj)
    axon_meshes = nru.feature_over_limb_branch_dict(neuron_obj,axon_limb_branch_dict,
                                     feature="mesh")
    return tu.combine_meshes(axon_meshes)

def dendrite_mesh(neuron_obj):
    limb_branch_dict = clu.dendrite_limb_branch_dict(neuron_obj)
    meshes = nru.feature_over_limb_branch_dict(neuron_obj,limb_branch_dict,
                                     feature="mesh")
    return tu.combine_meshes(meshes)
    

def axon_skeleton(neuron_obj):
    axon_limb_branch_dict = clu.axon_limb_branch_dict(neuron_obj)
    axon_meshes = nru.feature_over_limb_branch_dict(neuron_obj,axon_limb_branch_dict,
                                     feature="skeleton")
    return sk.stack_skeletons(axon_meshes)
    
    
def shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                    branch_1,
                                                    branch_2,
                                                   verbose=False,
                                                    check_concept_network_connectivity=True):
    """
    Purpose: To find the shared skeleton endpoint of 
    branches that are connected in the concept network
    
    Ex:
    nru.shared_skeleton_endpoints_for_connected_branches(neuron_obj[5],
                                                0,1,
                                                verbose=True)
    
    """
    curr_concept_network = limb_obj.concept_network
    if check_concept_network_connectivity:
        if branch_1 not in xu.get_neighbors(curr_concept_network,branch_2):
            raise Exception(f"Branches {branch_1} and {branch_2} are not connected in the concept network")

    shared_endpoints = sk.shared_endpoint(limb_obj[branch_1].skeleton,
                                          limb_obj[branch_2].skeleton,
                                          return_possibly_two=True)
    if verbose:
        print(f"shared_endpoints = {shared_endpoints}")
        
    return shared_endpoints


def closest_branch_endpoint_to_limb_starting_coordinate(limb_obj,
                                                        branches,
                                                       ):
    """
    Purpose: Will get the closest endpoints
    out of all the branches to the starting coordinate
    of limb

    Pseudocode:
    1) Get the limb graph and starting coordinate
    2) Get the endpoints of all of the branches
    3) Find the closest endpoint to the starting coordinate
    using the skeleton search function
    
    
    Ex:
    
    axon_limb_dict = neuron_obj.axon_limb_branch_dict
    axon_limb_name = list(axon_limb_dict.keys())[0]

    limb_obj = neuron_obj[axon_limb_name]
    branches = axon_limb_dict[axon_limb_name]

    nru.closest_branch_endpoint_to_limb_starting_coordinate(limb_obj,
                                                        branches,
                                                           )
    
    """
    

    #1) Get the limb graph and starting coordinate
    curr_limb_sk = limb_obj.skeleton
    curr_limb_st_coord = limb_obj.current_starting_coordinate

    #2) Get the endpoints of all of the branches
    all_endpoints = np.array([limb_obj[b].endpoints for b in branches]).reshape(-1,3)

    #3) Find the closest endpoint to the starting coordinate
    #using the skeleton search function
    coord_1,coord_2 = sk.shortest_path_between_two_sets_of_skeleton_coordiantes(curr_limb_sk,
                                                            [curr_limb_st_coord],
                                                             all_endpoints,
                                                             return_closest_coordinates=True)
    return coord_2

def neuron_limb_overwrite(neuron_obj,limb_name,limb_obj):
    """
    Purpose: to overwrite the limb object in a neuron
    with another limb object
    
    """
    neuron_obj.concept_network.nodes[limb_name]["data"] = limb_obj
    

def limb_branch_dict_to_skeleton(neuron_obj,limb_branch_dict):
    """
    Purpose: turn a limb_branch_dict into
    the corresponding skeleton of branches
    stacked together
    
    Pseudocode:
    1) Get the skeletons over the limb branch dict
    2) Stack the skeletons
    
    """
    branch_skeletons = nru.feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature="skeleton",
                                  keep_seperate=True,
                                 skip_None=True)
    return np.array(sk.stack_skeletons(branch_skeletons))

def axon_skeleton(neuron_obj):
    axon_limb_branch_dict = neuron_obj.axon_limb_branch_dict
    return nru.limb_branch_dict_to_skeleton(neuron_obj,axon_limb_branch_dict)

def dendrite_skeleton(neuron_obj):
    dendrite_limb_branch_dict = neuron_obj.dendrite_limb_branch_dict
    return nru.limb_branch_dict_to_skeleton(neuron_obj,dendrite_limb_branch_dict)


def mesh_without_mesh_attribute(obj,
                               mesh_attribute):
    """
    Purpose: To return the branch mesh without any spines
    """
    original_mesh_flag = False
    if hasattr(obj,mesh_attribute) and getattr(obj,mesh_attribute) is not None:
        if len(getattr(obj,mesh_attribute)) > 0:
            ex_branch_no_spines_mesh = tu.original_mesh_faces_map(obj.mesh,
                                    tu.combine_meshes(getattr(obj,mesh_attribute)),
                                   matching=False,
                                   print_flag=False,
                                   match_threshold = 0.001,
                                                            return_mesh=True,
                                                                 )
        else:
            original_mesh_flag = True
    else: 
        original_mesh_flag = True
    
    if original_mesh_flag:
        ex_branch_no_spines_mesh = obj.mesh
        
    return ex_branch_no_spines_mesh

def mesh_without_boutons(obj):
    return mesh_without_mesh_attribute(obj,
                                      mesh_attribute="boutons")


def coordinates_to_closest_limb_branch(neuron_obj,
                                     coordinates,
                                      original_mesh = None,
                                      original_mesh_kdtree = None,
                                       return_distances_to_limb_branch=False,
                                       return_closest_faces = False,
                                       verbose = False,):
    """
    Purpose: To map a coordinate to the closest limb branch idx
    of a neuron object
    
    Pseudocode: 
    A) Create the mapping of original face idx to (limb,branch)
    B) Map Coordinate to the original face idx to get
    c) Find mapping of Coordinate to --> (limb,branch)
    """
    
    if len(coordinates) == 0:
        coord_limb_branch_map = np.array([],dtype="int").reshape(-1,2)
        distance_to_limb_branch = np.array([])
        closest_faces = np.array([],dtype="int")
        if not return_closest_faces and not return_distances_to_limb_branch:
            return coord_limb_branch_map
        
        return_value = [coord_limb_branch_map]
        if return_distances_to_limb_branch:
            return_value.append(distance_to_limb_branch)
        if return_closest_faces:
            return_value.append(closest_faces)
        
        return return_value
    
    if original_mesh is None:
        original_mesh = neuron_obj.mesh_from_branches
    if original_mesh_kdtree is None:
        if verbose:
            print(f"Having ot generate KDTree from scratch")
        original_mesh_kdtree = KDTree(original_mesh.triangles_center)
        
    # Part A: Get Mapping from original mesh faces to limb/branch
    face_to_limb_branch = nru.original_mesh_face_to_limb_branch(neuron_obj,
                                                               original_mesh,
                                                               original_mesh_kdtree)
    
    
    
    all_syn_coord = np.array(coordinates).reshape(-1,3)
    
    # Part B: Map Synapses to the original faces
    dist,closest_face = original_mesh_kdtree.query(all_syn_coord)
    
    if verbose:
        print(f"dist = {dist}")
        print(f"closest_face = {closest_face}")
    
    #c) Find mapping of Coordinate to --> (limb,branch)
    coord_limb_branch_map = face_to_limb_branch[closest_face]
    
    if not return_closest_faces and not return_distances_to_limb_branch:
        return coord_limb_branch_map
    return_value = [coord_limb_branch_map]
    if return_distances_to_limb_branch:
        return_value.append(dist)
    if return_closest_faces:
        return_value.append(closest_face)
        
    return return_value

closest_branch_to_coordinates = coordinates_to_closest_limb_branch

def limb_branch_list_to_limb_branch_dict(limb_branch_list,
                                        verbose=False):
    limb_branch_dict = dict()
    for l_b in limb_branch_list:
        
        
        if np.isnan(l_b[0]):
            continue
        limb_name = nru.limb_label(l_b[0])
        if verbose:
            print(f"l_b = {l_b}")
            print(f"limb_name = {limb_name}")
        branch_idx = l_b[1]
        if limb_name not in limb_branch_dict.keys():
            limb_branch_dict[limb_name] = []
        if branch_idx not in limb_branch_dict[limb_name]:
            limb_branch_dict[limb_name].append(int(branch_idx))
        
        if verbose:
            print(f"limb_branch_dict = {limb_branch_dict}")
        
    limb_branch_dict_final = dict()
    for k in np.sort(list(limb_branch_dict.keys())):
        limb_branch_dict_final[k] = np.array(limb_branch_dict[k])
    return limb_branch_dict_final

def filter_limb_branch_dict_by_limb(limb_branch_dict,
                                    limb_names,
                                   verbose=False):
    """
    To filter a limb branch dict to only those limbs specified
    in the limb name
    """
    if not nu.is_array_like(limb_names):
        limb_names = [limb_names]
        
    limb_names = [nru.limb_label(k) for k in limb_names]
    if verbose:
        print(f"limb_names = {limb_names}")
    return {k:v for k,v in limb_branch_dict.items() if k in limb_names}

def boutons_above_thresholds(branch_obj,return_idx=False,
                               **kwargs):
    """
    To filter the boutons using some measurement
    
    Example:
    ns.n_boutons_above_thresholds(neuron_obj_with_boutons[axon_limb_name][5],
                             faces=100,
                            ray_trace_percentile=200)
                            
    threshodls to set: 
    "faces","ray_trace_percentile"
    """
    if branch_obj.n_boutons == 0:
        return []
    
    boutons_idx = np.arange(branch_obj.n_boutons)
    
    size_params = ["faces","ray_trace_percentile"]
    
    for p in size_params:
        if p in kwargs.keys():
            if p == "ray_trace_percentile":
                bouton_n_faces = np.array(branch_obj.boutons_cdfs)
            else:
                bouton_n_faces = np.array([tu.mesh_size(k,p) for k in branch_obj.boutons])
            
            bouton_n_faces_idx = np.where(bouton_n_faces >= kwargs[p])
            boutons_idx = np.intersect1d(boutons_idx,bouton_n_faces_idx)
    
    if return_idx:
        return boutons_idx
    else:
        return [k for i,k in enumerate(branch_obj.boutons) if i in boutons_idx]

    
def skeleton_coordinate_connecting_to_downstream_branches(limb_obj,
                                                         branch_idx,
                                                          return_downstream_branches=False,
                                                         verbose=False):
    """
    Psuedocode:
    1) Will find the skeleton point that connects 
    the current branch to the downstream branches
    
    """
    limb_obj_nx = limb_obj.concept_network_directional
    downstream_branches = xu.downstream_nodes(limb_obj_nx,branch_idx)
    
    if verbose:
        print(f"downstream_branches = {downstream_branches}")
        
    if len(downstream_branches) == 0:
        raise Exception("No downstream branches")
        
    shared_skeleton_pt = nru.shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                                branch_idx,         
                                                    downstream_branches[0])
    if verbose:
        print(f"shared_skeleton_pt= {shared_skeleton_pt}")
        
    if return_downstream_branches:
        return shared_skeleton_pt,downstream_branches
    else:
        return shared_skeleton_pt
    
def filter_branches_by_restriction_mesh(limb_obj,
                                       restriction_mesh,
                                       percentage_threshold=0.6,
                                       size_measure="faces",
                                       match_threshold = 0.001,
                                       verbose = False):
    """
    Purpose: To Find the branches that overlap with a restriction mesh
    up to a certain percentage
    
    Purpose: To select a group of meshes
    from one other mesh based on matching threshold

    Pseudocode: 

    0) Build a KDTree of the error mesh
    Iterate through all of the branches in that limb
    1) Get the mesh of the branch
    2) Map the branch mesh to the error mesh
    3) Compute the percent match of faces
    4) If above certain threshold then add to list
    
    """
    mesh_list_names =np.array(limb_obj.get_branch_names())
    mesh_list = [limb_obj[k].mesh for k in mesh_list_names]

    mesh_idx_match = tu.restrict_mesh_list_by_mesh(mesh_list,
                                  restriction_mesh,
                                   size_measure = size_measure,
                                match_threshold = match_threshold,
                                    verbose=verbose,
                                 percentage_threshold=percentage_threshold)
    
    return mesh_list_names[mesh_idx_match]

def limb_mesh_from_branches(
    limb_obj,
    plot = False,):
    
    neuron_mesh_list = []
    for branch_obj in limb_obj:
        neuron_mesh_list.append(branch_obj.mesh)
        
    neuron_mesh_from_branches = tu.combine_meshes(neuron_mesh_list)
    if plot:
        nviz.plot_objects(neuron_mesh_from_branches)
    return neuron_mesh_from_branches
    
def neuron_mesh_from_branches(neuron_obj,
                             plot_mesh=False):
    """
    Purpose: To reconstruct the mesh of neuron
    from all of the branch obejcts

    Pseudocode:
    Iterate through all the limbs:
        iterate through all the branches
            add to big list

    Add some to big list

    concatenate list into mesh
    """
    from mesh_tools import trimesh_utils as tu

    neuron_mesh_list = []
    for limb_obj in neuron_obj:
        for branch_obj in limb_obj:
            neuron_mesh_list.append(branch_obj.mesh)
    neuron_mesh_list += neuron_obj.get_soma_meshes()

    neuron_mesh_from_branches = tu.combine_meshes(neuron_mesh_list)
    if plot_mesh:
        nviz.plot_objects(neuron_mesh_from_branches)
    return neuron_mesh_from_branches

def non_soma_touching_meshes_not_stitched(neuron_obj,
                                          return_meshes=True):
                                          
    """
    Purpose: Find floating meshes not used

    Pseudocode: 
    1) construct the neuron mesh from branches
    2) restrict the non_soma touching pieces by the neuron_mesh
    3) Return either the meshes or indexes
    """

    #1) construct the neuron mesh from branches
    n_mesh = nru.neuron_mesh_from_branches(neuron_obj)

    #2) restrict the non_soma touching pieces by the neuron_mesh
    floating_meshes_non_incorporated = tu.restrict_mesh_list_by_mesh(neuron_obj.non_soma_touching_meshes,
                                  restriction_mesh=n_mesh,
                                  percentage_threshold=0.2,
                                  match_threshold = 5,
                                  return_under_threshold=True,
            return_meshes=return_meshes)
    return floating_meshes_non_incorporated

def non_soma_touching_meshes_stitched(neuron_obj,
                                          return_meshes=True):
                                          
    """
    Purpose: Find floating meshes not used

    Pseudocode: 
    1) construct the neuron mesh from branches
    2) restrict the non_soma touching pieces by the neuron_mesh
    3) Return either the meshes or indexes
    """

    #1) construct the neuron mesh from branches
    n_mesh = nru.neuron_mesh_from_branches(neuron_obj)

    #2) restrict the non_soma touching pieces by the neuron_mesh
    floating_meshes_incorporated = tu.restrict_mesh_list_by_mesh(neuron_obj.non_soma_touching_meshes,
                                  restriction_mesh=n_mesh,
                                  percentage_threshold=0.2,
                                  match_threshold = 5,
                                  return_under_threshold=False,
            return_meshes=return_meshes)
    return floating_meshes_incorporated

def all_downstream_branches(limb_obj,
                           branch_idx):
    """
    Will return all of the branches that are downstream
    of the branch_idx
    
    """
    return xu.all_downstream_nodes(limb_obj.concept_network_directional,branch_idx)

def feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=None,
                                  feature_function=None,
                          use_limb_obj_and_branch_idx = False,
                         combining_function=None,
                         verbose = False,
                         **kwargs):
    """
    To calculate a certain feature over 
    all the branches in a list
    """
    feature_list = []
    for b in branch_list:
        branch_obj = limb_obj[b]
        if feature_name is not None:
            curr_val = getattr(branch_obj,feature_name)
        elif feature_function is not None:
            if not use_limb_obj_and_branch_idx:
                curr_val = feature_function(branch_obj,**kwargs)
            else:
                curr_val = feature_function(limb_obj=limb_obj,branch_idx=b,**kwargs)
        else:
            raise Exception("Need to set either feature_name or feature function")
            
        feature_list.append(curr_val)
        
    if verbose:
        print(f"feature_list (before combining) = {feature_list}")
    if combining_function is not None:
        if len(feature_list)>0:
            feature_list = combining_function(feature_list)
        else:
            feature_list = 0
    
    if verbose:
        print(f"feature_list (after combining) = {feature_list}")
    
    return feature_list

def sum_feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=None,
                                  feature_function=None,
                         verbose = False):
    return feature_over_branches(limb_obj,
                                 branch_list,
                                 feature_name=feature_name,
                                  feature_function=feature_function,
                                 combining_function = np.sum,
                         verbose = verbose)

def skeletal_length_over_downstream_branches(limb_obj,
                                            branch_idx,
                                            combining_function = np.sum,
                                             include_branch_skeletal_length=True,
                                             nodes_to_exclude = None,
                                            verbose=False):
    """
    Will compute how much skeleton there is downstream of a certain node
    
    """
    downstream_branches = list(nru.all_downstream_branches(limb_obj,branch_idx))
    
    if nodes_to_exclude is not None:
        downstream_branches = list(np.setdiff1d(downstream_branches,nodes_to_exclude))
    
    if include_branch_skeletal_length:
        downstream_branches.append(branch_idx)
        
    sk_sum =nru.feature_over_branches(limb_obj,
                              branch_list = downstream_branches,
                              feature_name="skeletal_length",
                              combining_function = combining_function,
                              verbose = verbose
                             )
    if verbose:
        print(f"Total skeleton downstream = {sk_sum}")
        
    return sk_sum

def classify_upstream_downsream(limb_obj,
                               branch_list,
                                verbose = False):
    """
    Psuedocode: Given a list of branches that are all touching a certain coordinate,
    determine which of the branches are the upstream and which are the downstream

    Pseudocode: 
    1) Pick the first branch
    2) Get the sibling nodes
    3) Get overlap, if no overlap between sibling nodes and rest of the group
    yes --> it is upstream --> get downstream by filtering out upstream
    no --> it is downstream --> get upstream by filtering out all of siblings and sel
    """


    branch_list = np.array(branch_list)
    #1) Pick the first branch
    test_node = branch_list[0]

    #2) Get the sibling nodes
    sib_nodes = xu.sibling_nodes(limb_obj.concept_network_directional,test_node)
    if verbose:
        print(f"For test node {test_node}, sibling nodes were: {sib_nodes}")

    #3) Getting overlap
    overlap = np.intersect1d(branch_list,sib_nodes)
    if verbose:
        print(f"overlap = {overlap}")

    if len(overlap) == 0:
        upstream_node = test_node
        downstream_nodes = branch_list[branch_list != test_node]
        if verbose:
            print(f"With test node equal to the upstream node")
            print(f"upstream = {upstream_node}, downstream_nodes = {downstream_nodes}")
    else:
        downstream_nodes = np.hstack([sib_nodes,[test_node]])
        upstream_node = np.setdiff1d(branch_list,downstream_nodes)[0]
        if verbose:
            print(f"With test node equal to the downstream node")
            print(f"upstream = {upstream_node}, downstream_nodes = {downstream_nodes}")
            
    downstream_nodes = np.array(downstream_nodes)
    #xu.downstream_nodes(limb_obj.concept_network_directional,upstream_node)
    return upstream_node,downstream_nodes


def is_limb_obj(obj):
    """
    Determines if the object is a limb object
    """
    return str(type(obj)) == str(neuron.Limb)

def is_neuron_obj(obj):
    """
    Determines if the object is a limb object
    """
    return str(type(obj)) == str(neuron.Neuron)

def is_branch_obj(obj):
    """
    Determines if the object is a limb object
    """
    return str(type(obj)) == str(neuron.Branch)

def branches_combined_mesh(limb_obj,branches,
                          plot_mesh=False):
    """
    To combine the mesh objects of branch indexes
    
    Ex:
    branches_combined_mesh(limb_obj,branches=[45, 58, 61,66],
                          plot_mesh=True)
    """
    meshes = [limb_obj[k].mesh for k in branches]
    mesh_inter = tu.combine_meshes(meshes)
    
    if plot_mesh:
        nviz.plot_objects(mesh_inter)
    return mesh_inter

def coordinate_to_offset_skeletons(limb_obj,
                                  coordinate,
                                   branches= None,
                                   offset=1500,
                                    comparison_distance = 2000,
                                   plot_offset_skeletons=False,
                                   verbose = False,
                                   return_skeleton_endpoints = False,
                                  ):
    """
    Will return the offset skeletons of branches
    that all intersect at a coordinate
    """
    if branches is None:
        branches = nru.find_branch_with_specific_coordinate(limb_obj,coordinate)
        
    if verbose:
        print(f"branches = {branches}")
    
    skeletons = dict([(k,limb_obj[k].skeleton) for k in branches])
    upstream_branch = branches[0]
    unique_comb = [[upstream_branch,k] for k in branches[1:]]

    if verbose:
        print(f"unique_comb = {unique_comb}")
    
    aligned_skeletons = []
    for pair in unique_comb:
        edge_skeletons = [skeletons[pair[0]],skeletons[pair[1]]]
        aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                                 offset=offset,
                                                                                 comparison_distance=comparison_distance,
                                                                                 common_endpoint=coordinate)
        aligned_skeletons.append(aligned_sk_parts)

    offset_skeletons = [aligned_skeletons[0][0]] + [k[1] for k in aligned_skeletons]
    
    skeleton_offset_end_points = [k[-1][-1] for k in offset_skeletons]

    if plot_offset_skeletons:
        colors = mu.generate_non_randon_named_color_list(len(offset_skeletons),
                                                         user_colors=["purple","aqua","green","red",])
        for b,c in zip(branches,colors):
            print(f"{b}:{c}")
            
        meshes = [limb_obj[k].mesh for k in branches]
        nviz.plot_objects(meshes=meshes,
                          meshes_colors=colors,
                          skeletons=offset_skeletons,
                          skeletons_colors=colors,
                         scatters=skeleton_offset_end_points,
                         scatters_colors=colors)
        
    if return_skeleton_endpoints:
        return offset_skeletons,skeleton_offset_end_points
    else:
        return offset_skeletons
    
    
def upstream_node(limb_obj,branch):
    return xu.upstream_node(limb_obj.concept_network_directional,branch)

'''
def upstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False):
    """
    Pseudocode: 
    1) Find the upsream node
    2a) if upstream node is None then use the current current starting node
    2b) if upstream node, find the common skeleton point between the 2

    Ex: 
    
    limb_obj = neuron_obj[2]
    for branch_idx in limb_obj.get_branch_names():
        k = nru.upstream_endpoint(limb_obj = limb_obj,
        branch_idx = branch_idx,
        verbose = True,
                             return_endpoint_index = True)
        total_dist = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        print(f"k = {k}")
        print(f"total upstream dist = {total_dist}\n")
    """

    upstream_node = nru.upstream_node(limb_obj,branch_idx)
    if verbose:
        print(f"upstream_node for {branch_idx}: {upstream_node}")

    if upstream_node is None:
        common_endpoint = limb_obj.current_starting_coordinate
    else:
        shared_endooints = nru.shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                        branch_idx,
                                                        upstream_node,
                                                        check_concept_network_connectivity=False).reshape(-1,3)
        common_endpoint = shared_endooints[0]

        if verbose:
            print(f"shared_endooints = {shared_endooints}")

    if verbose:
        print(f"common_endpoint = {common_endpoint}")
        
    if return_endpoint_index:
        endpoint_index = nu.matching_row_index(limb_obj[branch_idx].endpoints,common_endpoint)
        if verbose:
            print(f"endpoints = {limb_obj[branch_idx].endpoints}")
            print(f"return_endpoint_index = {endpoint_index}")
        return endpoint_index
        
        
    return common_endpoint
'''

def neighbor_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False,
                     neighbor_type="upstream"):
    """
    Pseudocode: 
    1) Find the upsream node
    2a) if upstream node is None then use the current current starting node
    2b) if upstream node, find the common skeleton point between the 2

    Ex: 
    
    limb_obj = neuron_obj[2]
    for branch_idx in limb_obj.get_branch_names():
        k = nru.upstream_endpoint(limb_obj = limb_obj,
        branch_idx = branch_idx,
        verbose = True,
                             return_endpoint_index = True)
        total_dist = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        print(f"k = {k}")
        print(f"total upstream dist = {total_dist}\n")
    """

    upstream_node = nru.upstream_node(limb_obj,branch_idx)
    if verbose:
        print(f"upstream_node for {branch_idx}: {upstream_node}")

    if upstream_node is None:
        common_endpoint = limb_obj.current_starting_coordinate
    else:
        shared_endooints = nru.shared_skeleton_endpoints_for_connected_branches(limb_obj,
                                                        branch_idx,
                                                        upstream_node,
                                                        check_concept_network_connectivity=False).reshape(-1,3)
        common_endpoint = shared_endooints[0]

        if verbose:
            print(f"shared_endooints = {shared_endooints}")

    
        
    if verbose:
        print(f"common_endpoint = {common_endpoint}")
        
    if neighbor_type == "upstream":
        pass
    elif neighbor_type == "downstream":
        endpoint_index = nu.matching_row_index(limb_obj[branch_idx].endpoints,common_endpoint)
        common_endpoint = limb_obj[branch_idx].endpoints[1-endpoint_index]
        if verbose:
            print(f"The donwsream endpoint is {common_endpoint}")
    else:
        raise Exception(f"Unknown neighbor_type: {neighbor_type}")
        
    if return_endpoint_index:
        endpoint_index = nu.matching_row_index(limb_obj[branch_idx].endpoints,common_endpoint)
        if verbose:
            print(f"endpoints = {limb_obj[branch_idx].endpoints}")
            print(f"return_endpoint_index = {endpoint_index}")
        return endpoint_index
        
        
    return common_endpoint


def upstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False):
    """
    Purpose: To get the coordinate of the part of the 
    skeleton connecting to the upstream branch
    
    branch_idx = 263
    nviz.plot_objects(main_mesh = neuron_obj[0].mesh,
                     skeletons=[neuron_obj[0][branch_idx].skeleton],
                      scatters=[nru.upstream_endpoint(neuron_obj[0],
                         branch_idx),
                               nru.downstream_endpoint(neuron_obj[0],
                         branch_idx)],
                      scatters_colors=["red","blue"]
                     )
    
    """
    return neighbor_endpoint(limb_obj,
                     branch_idx,
                     verbose,
                     return_endpoint_index,
                     neighbor_type="upstream")

def downstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = False,
                     return_endpoint_index=False):
    return neighbor_endpoint(limb_obj,
                     branch_idx,
                     verbose,
                     return_endpoint_index,
                     neighbor_type="downstream")

def upstream_downstream_endpoint_idx(
    limb_obj,
    branch_idx,
    verbose = False,
    ):
    """
    To get the upstream and downstream endpoint idx
    returned (upstream_idx,downstream_idx)
    """
    
    up_idx = upstream_endpoint(limb_obj,
                     branch_idx,
                     verbose = verbose,
                     return_endpoint_index=True)
    return (up_idx,1-up_idx)
    
    

def downstream_nodes(limb_obj,branch):
    return xu.downstream_nodes(limb_obj.concept_network_directional,branch)

def n_downstream_nodes(limb_obj,branch):
    return len(nru.downstream_nodes(limb_obj,branch))
    
def width(branch_obj,
         axon_flag=None,
         width_name_backup="no_spine_median_mesh_center",
         width_name_backup_2 = "median_mesh_center",
         verbose = False):
    """
    Will extract the width from a branch that
    tries different width types
    """
    if axon_flag is None:
        if "axon" or "axon-like" in branch_obj.labels:
            axon_flag = True
        else:
            axon_flag = False
    
    if axon_flag:
        width_name = "no_bouton_median"
    else:
        width_name = "no_spine_median_mesh_center"
    
    widths_to_try = [width_name,width_name_backup,width_name_backup_2]
    for w_name in widths_to_try:
        if w_name in branch_obj.width_new.keys():
            if verbose:
                print(f"Using width {w_name}")
            return branch_obj.width_new[w_name]
    
    if verbose:
        print(f"Using generic width because None of the widths found: {widths_to_try}")
    return branch_obj.width
    
    
def branch_path_to_node(limb_obj,
                             start_idx,
                            destination_idx,
                             include_branch_idx = False,
                              include_last_branch_idx = True,
                            skeletal_length_min = None,
                            verbose = False,
                            reverse_di_graph=True,
                            starting_soma_for_di_graph = None,
                       ):
    """
    Purpose: Will find the branch objects on the path
    from current branch to the starting coordinate

    Application: Will know what width objects to compare to
    for width jump

    Pseudocode: 
    1) Get the starting coordinate of brnach
    2) Find the shortest path from branch_idx to starting branch
    3) Have option to include starting branch or not
    3) If skeletal length threshold is set then:
    a. get skeletal length of all branches on path
    b. Filter out all branches that are not above the skeletal length threshold

    Example: 
    for k in limb_obj.get_branch_names():
        nru.branch_path_to_start_node(limb_obj = neuron_obj[0],
        branch_idx = k,
        include_branch_idx = False,
        skeletal_length_min = 2000,
        verbose = False)
        
        
    Ex: How to find path from one branch to another after starting
    from a certain soma
    
    nru.branch_path_to_node(neuron_obj[0],
                       start_idx = 109,
                       destination_idx = 164,
                       starting_soma_for_di_graph = "S0",
                       include_branch_idx = True)
    """
    if starting_soma_for_di_graph is not None:
        limb_obj.set_concept_network_directional(soma_group_idx=starting_soma_for_di_graph,
                                                suppress_disconnected_errors=True)
        G = limb_obj.concept_network_directional
    elif reverse_di_graph:
        G = xu.reverse_DiGraph(limb_obj.concept_network_directional)
    else:
        G = limb_obj.concept_network_directional

    #2) Find the shortest path from branch_idx to starting branch
    try:
        shortest_path = nx.shortest_path(G,
                                         start_idx,destination_idx)
    except:
        shortest_path = None
        
    if verbose:
        print(f"shortest_path = {shortest_path}")
        
    if shortest_path is None:
        if verbose:
            print(f"No path between Nodes so returning None")
        return shortest_path

    if not include_branch_idx:
        shortest_path = shortest_path[1:]

    shortest_path = np.array(shortest_path)
    
    if not include_last_branch_idx:
        shortest_path = shortest_path[:-1]

    if skeletal_length_min is not None:
        sk_len = np.array([limb_obj[k].skeletal_length for k in shortest_path])
        shortest_path = shortest_path[sk_len>skeletal_length_min]
        if verbose:
            print(f"shortest_path AFTER skeletal length filtering above"
                  f" skeletal_length_min ({skeletal_length_min}): \n   {shortest_path}")

    return shortest_path

def branch_path_to_start_node(limb_obj,
                             branch_idx,
                             include_branch_idx = False,
                              include_last_branch_idx = True,
                            skeletal_length_min = None,
                            verbose = False,):
    start_idx = limb_obj.current_starting_node
    if verbose:
        print(f"start_idx = {start_idx}")
    
    return nru.branch_path_to_node(limb_obj,
                             branch_idx,
                            destination_idx=start_idx,
                             include_branch_idx = include_branch_idx,
                              include_last_branch_idx = include_last_branch_idx,
                            skeletal_length_min = skeletal_length_min,
                            verbose = verbose)

def branch_path_to_soma(limb_obj,branch_idx,plot = False,):
    path = xu.shortest_path(nx.Graph(limb_obj.concept_network_directional),limb_obj.current_starting_node,branch_idx)
    if plot:
        nviz.plot_limb_path(
            limb_obj,path, 
        )
    return path
    
def min_width_upstream(limb_obj,
                      branch_idx,
                      skeletal_length_min = 2000,
                       default_value = 10000,
                        verbose = False,
                      remove_first_branch=True,
                      remove_zeros=True):
    """
    Purpose: Find the width jump from 
    the minimum of all of the branches proceeding

    Pseudocode: 
    1) Get all of the nodes that proceed the branch
    2) Find the minimum of these branches
    3) Subtrack the minimum from the current branch width
    """
    
    path_to_start = nru.branch_path_to_start_node(limb_obj = limb_obj,
        branch_idx = branch_idx,
        include_branch_idx = False,
        skeletal_length_min = skeletal_length_min,
        include_last_branch_idx = not remove_first_branch,
        verbose = False)

    if verbose:
        print(f"path_to_start = {path_to_start}")

    path_widths = np.array([nru.width(limb_obj[k]) for k in path_to_start])

    if verbose:
        print(f"path_widths = {path_widths}")
        
    if remove_zeros:
        path_widths = list(path_widths[path_widths>0])
        
        if verbose:
            print(f"path_widths AFTER REMOVING ZEROS= {path_widths}")
    
    path_widths.append(default_value)
    min_path_width = np.min(path_widths)

    if verbose:
        print(f"min_path_width = {min_path_width}")

    return min_path_width

def pair_branch_connected_components(limb_obj,
        branches = None,
        conn_comp = None,
        plot_conn_comp_before_combining = False,
        pair_method = "skeleton_angle",

        #for the skeleton angle pairing
        match_threshold = 70,
        thick_width_threshold = 200,
        comparison_distance_thick = 3000,
        offset_thick = 1000,
        comparison_distance_thin = 1500,
        offset_thin = 0,
        plot_intermediates = False,
                                    verbose = False,
                                    **kwargs):

    """
    Purpose: To pair branches of a subgraph
    together if they match skeleton angles 
    or some other criteria

    Application: for grouping 
    red/blue splits together

    Arguments: 
    1) Limb object
    2) branches to check for connectivity (or the connected components precomputed)

    Pseudocode: 
    0) Compute the connected components if not already done
    1) For each connected component: 
    a. Find path of connected component back to the starting node
    b. If path only of size 1 then just return either error branches or connected components
    c. Get the border error branch and the border parent branch from the path
    d) add the border error branch to a dictionary mapping parent to border branch and the conn comp it belongs to

    2) For each parent branch:
    - if list is longer than 1
      a. match the border error branches to each other to see if should be connected
      ( have argument to set the function to use for this)
      b. If have any matches then add the pairings to a list of lists, else add to a seperate list

    3. Use the pairings to create new connected components if any should be combined


    Example: 
    nru.pair_branch_connected_components(limb_obj=neuron_obj[1],
    branches = limb_branch_dict["L1"],
    conn_comp = None,
    plot_conn_comp_before_combining = False,
    pair_method = "pair_all",
                                verbose = True)
                                
                                
    Example 2: 
    nru.pair_branch_connected_components(limb_obj=neuron_obj[1],
    #branches = limb_branch_dict["L1"],
    conn_comp = xu.connected_components(limb_obj.concept_network_directional.subgraph(limb_branch_dict["L1"]),
                       ),
    plot_conn_comp_before_combining = False,
                                verbose = True)
    """


    G = nx.Graph(limb_obj.concept_network_directional)
    #0) Compute the connected components if not already done
    if conn_comp is None:
        conn_comp = xu.connected_components(G.subgraph(branches))
    else:
        if len(conn_comp)>0:
            branches = nu.concatenate_lists(conn_comp)
        else:
            branches = []

    if verbose:
        print(f"conn_comp = {conn_comp}")
        print(f"branches = {branches}")

    if plot_conn_comp_before_combining:
        print(f"plot connected components before combined")
        nx.draw(G.subgraph(branches),with_labels=True)
        plt.show()

    """    
    1) For each connected component: 
    a. Find path of connected component back to the starting node
    b. If path only of size 1 then just return either error branches or connected components
    c. Get the border error branch and the border parent branch from the path
    d) add the border error branch to a dictionary mapping parent to border branch and the conn comp it belongs to
    """
    parent_to_branch_borders = dict()

    for j,c in enumerate(conn_comp):
        short_path,st,end = xu.shortest_path_between_two_sets_of_nodes(G,c,[limb_obj.current_starting_node])
    #     if verbose:
    #         print(f"short_path,st,end= {short_path,st,end}")

        if len(short_path)<2:
            continue

        parent_border = short_path[1]
        branch_border = st

        if parent_border not in parent_to_branch_borders:
            parent_to_branch_borders[parent_border] = dict(branch_idx = [branch_border],
                                                         conn_comp_idx = [j],
                                                          branch_to_comp_map = {branch_border:j})
        else:
            parent_to_branch_borders[parent_border]["branch_idx"].append(branch_border)
            parent_to_branch_borders[parent_border]["conn_comp_idx"].append(j)
            parent_to_branch_borders[parent_border]["branch_to_comp_map"][branch_border] = j


    if verbose:
        print(f"parent_to_branch_borders = {parent_to_branch_borders}")

    """
    2) For each parent branch:
    - if list is longer than 1
      a. match the border error branches to each other to see if should be connected
      ( have argument to set the function to use for this)
      b. If have any matches then add the pairings to a list of lists, else add to a seperate list
    """
    conn_comp_pairings = []
    for parent_idx,border_info in parent_to_branch_borders.items():
        if len(border_info["branch_idx"])<2:
            continue

        #a. match the border error branches to each other to see if should be connected
        if pair_method == "pair_all":
            conn_comp_pairings.append(border_info["conn_comp_idx"])
            #conn_comp_pairings.append(border_info["branch_idx"])
        elif pair_method == "skeleton_angle":
            parent_width = nru.width(limb_obj[parent_idx])

            if parent_width > thick_width_threshold:
                comparison_distance = comparison_distance_thick
                offset = offset_thick
            else:
                comparison_distance = comparison_distance_thin
                offset = offset_thin


            curr_branches = border_info["branch_idx"]
            try:
                matched_edges, matched_edges_angles = ed.matched_branches_by_angle(limb_obj,
                                                               branches = border_info["branch_idx"],
                                                            offset=offset,
                                                            comparison_distance = comparison_distance,
                                                            match_threshold = match_threshold,
                                                            verbose = False,
                                                            plot_intermediates = plot_intermediates,
                                                            plot_match_intermediates = False,
                                                            less_than_threshold = True
                                                            )
            except Exception as e:
                if verbose:
                    print(f"Hit an error when doing matched_branches_by_angle: {str(e)}")
                matched_edges = []

            if len(matched_edges)>0:
                match_G = xu.edges_and_weights_to_graph(matched_edges)
                match_G_conn_comp = xu.connected_components(match_G)
                for m_comp in match_G_conn_comp:
                    m_comp_conn_idx = [border_info["branch_to_comp_map"][k] for k in m_comp]

                    conn_comp_pairings.append(m_comp_conn_idx)


            if verbose:
                print(f"parent_width = {parent_width}, comparison_distance = {comparison_distance}, offset = {offset}")
                print(f"matched_edges = {matched_edges}")
        else:
            raise Exception(f"Unimplemented pair_method = {pair_method}")

    if verbose:
        print(f"conn_comp_pairings = {conn_comp_pairings}")

    """
    3. Use the pairings to create new connected components if any should be combined

    """
    all_conn_comp_idx = np.arange(len(conn_comp))
    conn_comp_ar = np.array(conn_comp)
    final_comp_comp_from_pair = [list(np.concatenate(conn_comp_ar[k]))
                                 for k in conn_comp_pairings]
    if len(final_comp_comp_from_pair)>0:
        conn_comp_idx_from_pairs = np.hstack(conn_comp_pairings)
    else:
        conn_comp_idx_from_pairs = []

    conn_comp_idx_leftover = np.delete(all_conn_comp_idx,conn_comp_idx_from_pairs)
    conn_comp_from_leftover = [conn_comp[k] for k in conn_comp_idx_leftover]

    conn_comp_combined = final_comp_comp_from_pair + conn_comp_from_leftover


    if verbose:
        print(f"final_comp_comp_from_pair = {final_comp_comp_from_pair}")
        print(f"conn_comp_idx_leftover = {conn_comp_idx_leftover}")
        print(f"conn_comp_from_leftover = {conn_comp_from_leftover}")
        print(f"conn_comp_combined = {conn_comp_combined}")

    return conn_comp_combined

'''def restrict_skeleton_from_start_plus_offset_upstream_old(
    limb_obj,
    branch_idx,
    starting_endpoint=None,
    offset=500,
    comparison_distance=2000,
    skeleton_segment_size=100,
    width_name= "no_spine_median_mesh_center",
    width_name_backup = "no_spine_median_mesh_center",
    return_width = False,
    plot_skeleton = False,
    verbose = False):
    
    """
    Purpose: To Get the upstream skeleton (that potentially goes past the current branch itself)
    
    Ex: 
    restrict_skeleton_from_start_plus_offset_upstream(
    limb_obj,
    branch_idx=100,
    starting_endpoint=None,
    verbose = True,
    plot_skeleton=True)
    
    """

    if starting_endpoint is None:
        upstream_common_endpoint = nru.downstream_endpoint(limb_obj,branch_idx)
    else:
        upstream_common_endpoint = starting_endpoint

    # ----- Part 2: Do the processing on the upstream nodes -------------- #
    upstream_offset = offset
    upstream_comparison = comparison_distance
    upstream_node = branch_idx
    upstream_skeleton = []
    upstream_seg_lengths = []
    upstream_seg_widths = []
    curr_limb = limb_obj
    previous_node = None

    count = 0
    while upstream_comparison > 0:
        """
        Pseudocode:
        1) Get shared endpoint of upstream and previous node
        2) resize the upstream skeleton to get it ordered and right scale of width
        3) Flip the skeleton and width array if needs to be flipped
        4) if current offset is greater than 0, then restrict skeelton to offset:
        5a) if it was not long enough:
            - subtact total length from buffer
        5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break
        6)  change out upstream node and previous node (because at this point haven't broken outside loop)

        """
        if verbose:
            print(f"--- Upstream iteration: {count} -----")
        upstream_branch = curr_limb[upstream_node]

        #1) Get shared endpoint of upstream and previous node
        if count == 0:
            common_endpoint = upstream_common_endpoint
        else:
            prev_branch = curr_limb[previous_node]
            common_endpoint = sk.shared_endpoint(prev_branch.skeleton,upstream_branch.skeleton)

        #2) resize the upstream skeleton to get it ordered and right scale of width
        upstream_skeleton_ordered = sk.resize_skeleton_branch(upstream_branch.skeleton,skeleton_segment_size)
        if verbose:
            print(f"upstream_skeleton_ordered {sk.calculate_skeleton_distance(upstream_skeleton_ordered)} = {upstream_skeleton_ordered}")


          # ----------- 1 /5 : To prevent from erroring when indexing into width
    #         #accounting for the fact that the skeleton might be a little longer thn the width array now
    #         upstream_width = upstream_branch.width_array[width_name]
    #         extra_width_segment = [upstream_width[-1]]*(len(upstream_skeleton_ordered)-len(upstream_width))
    #         upstream_width = np.hstack([upstream_width,extra_width_segment])

        #3) Flip the skeleton and width array if needs to be flipped
        if np.array_equal(common_endpoint,upstream_skeleton_ordered[-1][-1]):
            if return_width:
                try:
                    upstream_width_ordered = np.flip(upstream_branch.width_array[width_name])
                except:
                    upstream_width_ordered = np.flip(upstream_branch.width_array[width_name_backup])

            upstream_skeleton_ordered = sk.flip_skeleton(upstream_skeleton_ordered)
            flip_flag = True
        elif np.array_equal(common_endpoint,upstream_skeleton_ordered[0][0]):
            if return_width:
                try:
                    upstream_width_ordered = upstream_branch.width_array[width_name]
                except:
                    upstream_width_ordered = upstream_branch.width_array[width_name_backup]

            flip_flag = False
        else:
            raise Exception("No matching endpoint")


        if verbose: 
            print(f"flip_flag = {flip_flag}")
            print(f"upstream_offset = {upstream_offset}")

        #4) if current offset is greater than 0, then restrict skeelton to offset:
        if upstream_offset > 0:
            if verbose:
                print("Restricting to offset")
            (skeleton_minus_buffer,
             offset_indexes,
             offset_success) = sk.restrict_skeleton_from_start(upstream_skeleton_ordered,
                                                                            upstream_offset,
                                                                             subtract_cutoff=True)
        else:
            if verbose:
                print("Skipping the upstream offset because 0")
            skeleton_minus_buffer = upstream_skeleton_ordered
            offset_indexes = np.arange(len(upstream_skeleton_ordered))
            offset_success = True


        #print(f"skeleton_minus_buffer {sk.calculate_skeleton_distance(skeleton_minus_buffer)} = {skeleton_minus_buffer}")

        """
        5a) if it was not long enough:
        - subtact total length from buffer
        """
        if not offset_success:
            upstream_offset -= sk.calculate_skeleton_distance(upstream_skeleton_ordered)
            if verbose:
                print(f"Subtracting the offset was not successful so changing to {upstream_offset} and reiterating")
        else:
            """
            5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break

            """
            #making sure the upstream offset is 0 if we were successful
            upstream_offset = 0

            if verbose:
                print(f"After subtracting the offset the length is: {sk.calculate_skeleton_distance(skeleton_minus_buffer)}")

            #- restrit skeleton by comparison distance
            (skeleton_comparison,
             comparison_indexes,
             comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                            upstream_comparison,
                                                                             subtract_cutoff=False)
            #- Add skeleton, width and skeelton lengths to list
            upstream_skeleton.append(skeleton_comparison)
            upstream_seg_lengths.append(sk.calculate_skeleton_segment_distances(skeleton_comparison,cumsum=False))

            if return_width:
                upstream_indices = offset_indexes[comparison_indexes]
                upstream_seg_widths.append(upstream_width_ordered[np.clip(upstream_indices,0,len(upstream_width_ordered)-1) ])

            # - subtract new distance from comparison distance
            upstream_comparison -= sk.calculate_skeleton_distance(skeleton_comparison)

            if comparison_success:
                if verbose:
                    print(f"Subtracting the comparison was successful and exiting")
                break
            else:
                if verbose:
                    print(f"Subtracting the comparison was not successful so changing to {upstream_comparison} and reiterating")

        #6)  change out upstream node and previous node (because at this point haven't broken outside loop)
        previous_node = upstream_node
        upstream_node = xu.upstream_node(curr_limb.concept_network_directional,upstream_node)

        if verbose:
            print(f"New upstream_node = {upstream_node}")

        if upstream_node is None:
            if verbose:
                print("Breaking because hit None upstream node")
            break

        count += 1

    upstream_final_skeleton = sk.stack_skeletons(upstream_skeleton)
    if verbose:
        print(f"upstream_final_skeleton = {upstream_final_skeleton}")

    # Do a check at the very end and if no skeleton then just take that branches
    if len(upstream_final_skeleton) <= 0:
        if verbose:
            print("No upstream skeletons so doing backup")
        resize_sk = sk.resize_skeleton_branch(curr_limb[upstream_node_original].skeleton,
                                                       skeleton_segment_size)
        upstream_skeleton = [resize_sk]
        upstream_seg_lengths = [sk.calculate_skeleton_segment_distances(resize_sk,cumsum=False)]
        
        if return_width:
            try:
                upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name]]
            except:
                upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name_backup]]

        (upstream_final_skeleton,
         upstream_final_widths,
        upstream_final_seg_lengths) = nru.align_and_restrict_branch(curr_limb[upstream_node_original],
                                  common_endpoint=common_endpoint_original,
                                width_name=width_name,
                                 offset=offset,
                                 comparison_distance=comparison_distance,
                                 skeleton_segment_size=skeleton_segment_size,
                                  verbose=verbose,
                                 )
    else:
        upstream_final_seg_lengths = np.concatenate(upstream_seg_lengths)
        if return_width:
            upstream_final_widths = np.concatenate(upstream_seg_widths)


    if return_width:
        upstream_width_average = nu.average_by_weights(weights = upstream_final_seg_lengths,
                                values = upstream_final_widths)
    

    if plot_skeleton:
        upstream_node = nru.upstream_node(limb_obj,branch_idx)
        if upstream_node is None:
            upstream_node = []
        else:
            upstream_node = [upstream_node]
            
        downstream_nodes = list(nru.downstream_nodes(limb_obj,branch_idx))
        total_branches_to_mesh = upstream_node + downstream_nodes
        
        nviz.plot_objects(main_mesh = limb_obj[branch_idx].mesh,
                         main_mesh_color="red",
                         skeletons=[upstream_final_skeleton],
                         skeletons_colors="red",
                          meshes=[limb_obj[k].mesh for k in total_branches_to_mesh],
                          meshes_colors="blue",
                          scatters=[upstream_common_endpoint],
                          scatters_colors=["red"]
                         )
    
    if return_width:
        return upstream_final_skeleton,upstream_width_average
    else:
        return upstream_final_skeleton'''
    
def restrict_skeleton_from_start_plus_offset_upstream(
    limb_obj,
    branch_idx,
    start_coordinate=None,
    offset=500,
    comparison_distance=2000,
    skeleton_resolution=100,
    min_comparison_distance = 1000,
    plot_skeleton = False,
    nodes_to_exclude = None,
    verbose = False):
    
    """
    Purpose: To get the upstream skeleton using the new subgraph around node function

    Pseudocode: 
    1) Get the upstream subgraph around the node that is a little more than the offset 
    and comparison distance
    2) Get the skeleton of all of the 
    """
    
    if start_coordinate is None:
        start_coordinate = nru.downstream_endpoint(limb_obj,branch_idx)
        
    upstream_node_list = cnu.subgraph_around_branch(limb_obj,
                           branch_idx,
                            downstream_distance = -1,
                           upstream_distance=comparison_distance + offset + 1000,
                          include_branch_in_upstream_dist=True,
                           include_branch_idx = True,
                          return_branch_idxs=True,
                          nodes_to_exclude=nodes_to_exclude,
                          )
    
    if verbose:
        print(f"upstream_node_list = {upstream_node_list}")
    
    upstream_skeleton_total = sk.stack_skeletons([limb_obj[k].skeleton for k in upstream_node_list])
    upstream_final_skeleton = sk.restrict_skeleton_from_start_plus_offset(upstream_skeleton_total,
                                                   offset=offset,
                                                comparison_distance=comparison_distance,
                                                    min_comparison_distance=min_comparison_distance,
                                                verbose=verbose,
                                                 start_coordinate=start_coordinate,
                                                skeleton_resolution = skeleton_resolution
                                                   )
    
    
    if plot_skeleton:
        upstream_node = nru.upstream_node(limb_obj,branch_idx)
        if upstream_node is None:
            upstream_node = []
        else:
            upstream_node = [upstream_node]
            
        downstream_nodes = list(nru.downstream_nodes(limb_obj,branch_idx))
        total_branches_to_mesh = upstream_node + downstream_nodes
        
        nviz.plot_objects(main_mesh = limb_obj[branch_idx].mesh,
                         main_mesh_color="red",
                         skeletons=[upstream_final_skeleton],
                         skeletons_colors="red",
                          meshes=[limb_obj[k].mesh for k in total_branches_to_mesh],
                          meshes_colors="blue",
                          scatters=[start_coordinate],
                          scatters_colors=["red"]
                         )
    return upstream_final_skeleton


def restrict_skeleton_from_start_plus_offset_downstream(
    limb_obj,
    branch_idx,
    start_coordinate=None,
    offset=500,
    comparison_distance=2000,
    skeleton_resolution=100,
    min_comparison_distance = 1000,
    plot_skeleton = False,
    nodes_to_exclude = None,
    verbose = False):
    
    """
    Purpose: To get the upstream skeleton using the new subgraph around node function

    Pseudocode: 
    1) Get the upstream subgraph around the node that is a little more than the offset 
    and comparison distance
    2) Get the skeleton of all of the 
    
    
    Ex: 
    nru.restrict_skeleton_from_start_plus_offset_downstream(limb_obj,97,
                                                      comparison_distance=100000,
                                                     plot_skeleton=True,
                                                       verbose=True)
    """
    
    if start_coordinate is None:
        start_coordinate = nru.upstream_endpoint(limb_obj,branch_idx)
        
    upstream_node_list = cnu.subgraph_around_branch(limb_obj,
                           branch_idx,
                           downstream_distance=comparison_distance + offset + 1000,
                            upstream_distance=-1,
                            only_non_branching_downstream = True,
                          include_branch_in_upstream_dist=True,
                           include_branch_idx = True,
                          return_branch_idxs=True,
                            nodes_to_exclude=nodes_to_exclude,
                                                    verbose = verbose
                          )
    if verbose:
        print(f"downstream_node_list = {upstream_node_list}")
    
    upstream_skeleton_total = sk.stack_skeletons([limb_obj[k].skeleton for k in upstream_node_list])
    
    upstream_final_skeleton = sk.restrict_skeleton_from_start_plus_offset(upstream_skeleton_total,
                                                   offset=offset,
                                                comparison_distance=comparison_distance,
                                                    min_comparison_distance=min_comparison_distance,
                                                verbose=verbose,
                                                 start_coordinate=start_coordinate,
                                                skeleton_resolution = skeleton_resolution
                                                   )
    
    
    if plot_skeleton:
        upstream_node = nru.upstream_node(limb_obj,branch_idx)
        if upstream_node is None:
            upstream_node = []
        else:
            upstream_node = [upstream_node]
            
        downstream_nodes = list(nru.downstream_nodes(limb_obj,branch_idx))
        total_branches_to_mesh = upstream_node + downstream_nodes
        
        nviz.plot_objects(main_mesh = limb_obj[branch_idx].mesh,
                         main_mesh_color="red",
                         skeletons=[upstream_final_skeleton],
                         skeletons_colors="red",
                          meshes=[limb_obj[k].mesh for k in total_branches_to_mesh],
                          meshes_colors="blue",
                          scatters=[start_coordinate],
                          scatters_colors=["red"]
                         )
    return upstream_final_skeleton

def copy_neuron(neuron_obj):
    return neuron.Neuron(neuron_obj)

# --------- 7/28: For the apical classification ------------
def candidate_groups_from_limb_branch(
    neuron_obj,
    limb_branch_dict,
    print_candidates = False,
    # arguments for determining connected component manner
    connected_component_method = "downstream", #other options: "local_radius"
    
    radius = 20000,#5000,
    
    require_connected_components=False,
    plot_candidates = False,
    max_distance_from_soma_for_start_node = None,
    verbose = False,
    return_one = False):
    """
    Purpose: To group a limb branch dict
    into a group of candidates based on 
    upstream connectivity 
    (leader of the group will be the most upstream member)
    
    Ex: 
    apical_candidates = nru.candidate_groups_from_limb_branch(neuron_obj,
                                      {'L0': np.array([14, 11, 5])},
                                      verbose = verbose,
                                    print_candidates=print_candidates,
                                                         require_connected_components = True)

    """
    
    #raise Exception("NEED TO MAKE SURE THERE CAN'T BE GAPS BETWEEN THE CANDIDATES")

    candidates = []
    for l_idx,b_idxs in limb_branch_dict.items():
        limb_obj = neuron_obj[l_idx]
        
        #G = limb_obj.concept_network_directional
        G = cnu.G_weighted_from_limb(limb_obj)

        if connected_component_method == "downstream":
            limb_conn_comp = xu.downstream_conn_comps(G,
            nodes = b_idxs,
            start_node = limb_obj.current_starting_node,
            verbose = False
            )
        elif connected_component_method == "local_radius":
            #print(f"Inside local radius")
            limb_conn_comp = xu.local_radius_conn_comps(G,
                                                       nodes = b_idxs,
                                                        radius = radius,
                                                        return_upstream_dict = True,
                                                       verbose = False)
            
        else:
            raise Exception(f"Unimplemented connected_component_method: {connected_component_method}")

        if verbose:
            print(f"{l_idx} : limb_conn_comp = {limb_conn_comp}")

        limb_candidates = [dict(limb_idx = l_idx,start_node=k,branches=v,) for k,v in limb_conn_comp.items()]
    
    
        if require_connected_components:
            """
            Purpose: Will reduce the branches in the group to only 
            those that are in a connected component with the starter branch

            """
            if verbose:
                print(f"require_connected_components set")
                print(f"limb_candidates before = {limb_candidates}")
            
            new_limb_candidates = []
            for j,c in enumerate(limb_candidates):
                new_d = dict(c)
                new_d["branches"] = xu.connected_component_with_node(node=c["start_node"],
                                                 G= G.subgraph(c["branches"]),
                                                 return_only_one = True,
                                                 verbose=False)
                new_limb_candidates.append(new_d)
            limb_candidates = new_limb_candidates
            
            if verbose:
                print(f"limb_candidates AFTER  = {limb_candidates}")
 
        candidates += limb_candidates

    if verbose:
        print(f"# of candidates = {len(candidates)}")

    if print_candidates:
        print(f"candidates = {candidates}")
        
    if plot_candidates:
        nviz.plot_candidates(neuron_obj,
               candidates)
        
    if max_distance_from_soma_for_start_node is not None:
        """
        Purpose: Filter candidates for only those with starting branches within a certain 
        distance of the soma (BECAUSE WE ARE ASSUMING THE APICAL HAD TO HAVE SPLIT OFF BY THAT POINT)

        """
        if verbose:
            print(f"Filtering canddiates for only those starting less than {max_distance_from_soma_for_start_node} away from soma")
        candidates = [k for k in  candidates if nst.distance_from_soma(neuron_obj[k["limb_idx"]],
                                                                                  branch_idx = k["start_node"],
                                                                                  include_node_skeleton_dist=False,
                                                                                ) <= max_distance_from_soma_for_start_node]
        
        if verbose:
            print(f"\nAfter filtering for starting node distances")
            print(f"# of candidates = {len(candidates)}")

        if print_candidates:
            print(f"candidates = {candidates}\n")

        if plot_candidates:
            print(f"After filtering candidates for starting node distances, candidates are")
            nviz.plot_candidates(neuron_obj,
                   candidates,
                   verbose = False)
            
            
    if return_one:
        if len(candidates) > 0:
            candidates = candidates[0]
        else:
            candidates = None
    
    return candidates
def most_upstream_branch(limb_obj,branches,verbose = False):
    """
    Purpose: To find the most upstream branch in 
    a group of branches
    
    Ex: 
    most_upstream_branch(limb_obj,[ 2,  6, 20, 23, 24, 25, 26, 33])
    """
    return xu.most_upstream_node(limb_obj.concept_network_directional,
                                 branches,
                                 verbose = verbose,
                                )

def limb_branch_from_candidate(candidate):
    return {candidate['limb_idx']:candidate["branches"]}

def candidate_from_branches(limb_obj,
                           branches,
                            limb_idx,
                           ):
    candidate = dict(limb_idx=nru.get_limb_string_name(limb_idx),
                    branches=branches)
    if branches is not None and len(branches)>0:
        most_upstream = nru.most_upstream_branch(limb_obj,branches)
    else:
        most_upstream = None
        
    candidate["start_node"] = most_upstream
    return candidate
    
def candidates_from_limb_branch_candidates(
    neuron_obj,
    limb_branch_candidates,
    verbose = False):
    """
    Purpose: to convert a dictionary of all the candidates into
    a list of candidate dictionaries
    
    Application: 
    --original
    {1: array([list([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 16, 17, 18, 20, 21, 22, 23, 24, 25, 52, 53, 54]),
            list([8, 11, 12, 14, 15, 19, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51])],
           dtype=object),
     2: array([list([1, 5, 6, 7, 8]), list([0, 10, 11, 12]), list([9, 2])],
           dtype=object),
     3: array([[0, 1, 2, 3, 4, 5, 6, 7]]),
     5: array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])}
             
    Output: 
    
    """
    all_candidates = []
    for limb_idx,branch_groups in limb_branch_candidates.items():
        limb_obj = neuron_obj[limb_idx]
        for b in branch_groups:
            all_candidates.append(nru.candidate_from_branches(limb_obj,
                                                                b,
                                                                limb_idx))
            
    if verbose:
        print(f"Total # of candidates = {len(all_candidates)}")
        
    return all_candidates
    

def skeleton_over_limb_branch_dict(neuron_obj,
                                   limb_branch_dict,
                                   stack_skeletons = True,
                                   plot_skeleton = False,
                                  ):
    """
    Purpose: To collect the meshes over a limb branch dict
    
    nru.mesh_over_limb_branch_dict(neuron_obj,
                              nru.limb_branch_from_candidate(apical_candidates[0]),
                              plot_mesh=True)
    """

    individual_sk = nru.feature_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=limb_branch_dict,
                                      keep_seperate = True,
                                     feature="skeleton")
    if stack_skeletons:
        individual_sk = sk.stack_skeletons(individual_sk)
        
    if plot_skeleton:
        if stack_skeletons:
            p_sk = [individual_sk]
        else:
            p_sk = individual_sk
            
        nviz.plot_objects(neuron_obj.mesh,
                          skeletons = p_sk)
        
    return individual_sk


def mesh_over_limb_branch_dict(neuron_obj,
                                   limb_branch_dict,
                                   combine_meshes = True,
                                   plot_mesh = False,
                                  ):
    """
    Purpose: To collect the skeletons over a limb branch dict
    
    nru.skeleton_over_limb_branch_dict(neuron_obj,
                              nru.limb_branch_from_candidate(apical_candidates[0]),
                              plot_skeleton=True)
    """

    individual_mesh = nru.feature_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=limb_branch_dict,
                                      keep_seperate = True,
                                     feature="mesh")
    if combine_meshes:
        individual_mesh = tu.combine_meshes(individual_mesh)
        
    if plot_mesh:
        if combine_meshes:
            p_sk = [individual_mesh]
        else:
            p_sk = individual_mesh
            
        nviz.plot_objects(neuron_obj.mesh,
                          meshes = p_sk,
                         meshes_colors = ["red"])
        
    return individual_mesh

def mesh_over_candidate(neuron_obj,
                       candidate,
                       **kwargs):
    """
    Ex: 
    nru.mesh_over_candidate(neuron_obj,
                        apical_candidates[0],
                       plot_mesh = True)
    
    """
    return nru.mesh_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict = nru.limb_branch_from_candidate(candidate),
                                         **kwargs)

def skeleton_over_candidate(neuron_obj,
                       candidate,
                       **kwargs):
    """
    Ex: 
    nru.skeleton_over_candidate(neuron_obj,
                        apical_candidates[0],
                       plot_skeleton = False)
    """
    return nru.skeleton_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict = nru.limb_branch_from_candidate(candidate),
                                         **kwargs)

def skeletal_length_over_limb_branch(neuron_obj,
                                    limb_branch_dict,
                                    verbose = False):
    """
    Ex: 
    nru.skeletal_length_over_limb_branch(neuron_obj,
                                nru.limb_branch_from_candidate(ap_cand))
    
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                limb_branch_dict,
                                                feature="skeletal_length")

def area_over_limb_branch(neuron_obj,
                                    limb_branch_dict,
                                    verbose = False):
    """
    Ex: 
    nru.skeletal_length_over_limb_branch(neuron_obj,
                                nru.limb_branch_from_candidate(ap_cand))
    
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                limb_branch_dict,
                                                feature="area")

def volume_over_limb_branch(neuron_obj,
                                    limb_branch_dict,
                                    verbose = False):
    """
    Ex: 
    nru.skeletal_length_over_limb_branch(neuron_obj,
                                nru.limb_branch_from_candidate(ap_cand))
    
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                limb_branch_dict,
                                                feature="mesh_volume")

def skeletal_length_over_candidate(neuron_obj,
                                  candidate,
                                  verbose = False):
    return nru.skeletal_length_over_limb_branch(neuron_obj,
                                               nru.limb_branch_from_candidate(candidate),
                                               verbose = verbose)

def all_downstream_branches_from_candidate(neuron_obj,
                                        candidate,
                                        include_candidate_branches=False,
                                        verbose = False):
    """
    Purpose: To get all of the branches downstream of a candidate
    """
    
    '''
    all_downs = np.concatenate([cnu.all_downtream_branches(neuron_obj[candidate["limb_idx"]],
                                           k) for k in candidate["branches"]])
    if include_candidate_branches:
        all_downs = np.concatenate([all_downs,candidate["branches"]])
    else:
        all_downs = np.setdiff1d(all_downs,candidate["branches"])
        
    downstream_nodes = np.unique(all_downs)
    
    if verbose:
        print(f"downstream_nodes = {downstream_nodes}")
        
    return downstream_nodes
    '''
    return cnu.all_downstream_branches_from_branches(neuron_obj[candidate["limb_idx"]],
                                                    branches = candidate["branches"],
                                                    include_original_branches=include_candidate_branches,
                                                    verbose = verbose)



def all_donwstream_branches_from_limb_branch(neuron_obj,
                                            limb_branch_dict,
                                            include_limb_branch_dict=True,
                                             verbose = False,
                                             plot = False,
                                            ):
    disconn_limb_branch = nru.limb_branch_after_limb_branch_removal(neuron_obj,
                                                                   limb_branch_dict,
                                                                   return_removed_limb_branch = True,)
    if not include_limb_branch_dict:
        disconn_limb_branch = nru.limb_branch_setdiff([disconn_limb_branch,limb_branch_dict])
        
    if verbose:
        print(f"downstream limb_branch = {disconn_limb_branch}")
        
    if plot:
        nviz.plot_limb_branch_dict(neuron_obj,disconn_limb_branch)
        
    return disconn_limb_branch

'''
def fill_in_and_filter_branch_groups_from_upstream_without_branching(limb_obj,
                                                  branches,
                                                  start_branch=None,
                                                   skip_distance = 5000,
                                                   plot_filtered_branches = False,
                                                  verbose = False,):
    """
    Purpose: To filter a branches in 
    a candidate for only those
    within a certain distance of the closest
    upstream branch (and optimally not the result of a branching point)

    Psuedocode: 
    1) Compute with the starting branch if not determined
    2) Make the start branch the current branch
    3) For range of the number of branches:
    a. Get the branches that are within certain distance downstream (that also belong to the branches list)
    b. If no branches then break
    c. If yes then make sure on the same path
    
    
    Ex: 
    nru.fill_in_and_filter_branch_groups_from_upstream_without_branching(
    limb_obj =  neuron_obj[6],
    branches = [24, 25, 26],
    start_branch = None,
    verbose = True,
    skip_distance = 5000,
    plot_filtered_branches = True
    )
    """

    if start_branch is None:
        start_branch = xu.most_upstream_node(limb_obj.concept_network_directional,
                                            branches)
        if verbose:
            print(f"Comptued start_branch = {start_branch}")
    
    curr_branch = start_branch
    final_branches = [start_branch]

    G = limb_obj.concept_network_directional

    for i in range(len(branches)):

        down_nodes = cnu.nodes_downstream(limb_obj,
                        curr_branch,distance=skip_distance,
                                          nodes_to_include = branches)

        if verbose:
            print(f"For curr_branch: {curr_branch}")
            print(f"down_nodes = {down_nodes}")


        if len(down_nodes) == 0:
            break

        if not xu.check_downstream_nodes_on_same_path(G,down_nodes,start_node=curr_branch) and skip_distance > 0:
            skip_distance = 0
            down_nodes = cnu.nodes_downstream(limb_obj,
                        curr_branch,distance=skip_distance,
                                              nodes_to_include = branches)
            if verbose:
                print(f"After skip distance changed to 0: down_nodes = {down_nodes}")
                
        if len(down_nodes) == 0:
            break

        if not xu.check_downstream_nodes_on_same_path(G,down_nodes,start_node=curr_branch):
            if verbose:
                print(f"Not all downstream nodes on same path so breaking")
            break

        most_downstream_node = xu.least_downstream_node(G,down_nodes)
        if verbose:
            print(f"most_downstream_node= {most_downstream_node}")

        final_branches = np.union1d(final_branches,
                                    xu.shortest_path(G,curr_branch,most_downstream_node))

        curr_branch = most_downstream_node

    if verbose:
        print(f"final_branches = {final_branches} ")
        
    if plot_filtered_branches:
        print(F"Plotting final branches")
        nviz.plot_limb_path(limb_obj,final_branches)
        
    return final_branches


def candidate_filter_skeletal_length(neuron_obj,
                                    candidate,
                                    candidate_skeletal_length_min=18000,
                                     candidate_skeletal_length_max = None,
                                     verbose = False,
                                     **kwargs
                                    ):
    sk_len = nst.skeletal_length_over_candidate(neuron_obj,
                                  candidate)
    if verbose:
        print(f"skeletal length = {sk_len}")
        
    if  candidate_skeletal_length_min is not None and sk_len < candidate_skeletal_length_min:
        if verbose:
            print(f"Filtered away candidate because skeletal length too small")
        return False
    
    if  candidate_skeletal_length_max is not None and sk_len > candidate_skeletal_length_max:
        if verbose:
            print(f"Filtered away candidate because skeletal length too large")
        return False
    
    return True


    
    
    
def filter_candidates(neuron_obj,
                     candidates,
                      filters = ("skeletal_length",),
                      
                      #-- arguments for the different filters --
                      candidate_skeletal_length_min = 18000,
                      verbose = False,
                      **kwargs
                     ):
    """
    Purpose: To filter the apical shaft candidates 
    for base characteristics:
    a) skeletal length

    -- Not yet implmented --
    Angle and length between the 
    start of the candidate and the soma
    
    Ex: 
    nru.filter_candidates(neuron_obj,
                 candidates=shaft_candidates,
                 verbose = verbose)
    """

    filtered_cand = []

    for j,can in enumerate(candidates):
        if verbose:
            print(f"\nBase Filtering candidate {j}: {can}")

        cand_lb = nru.limb_branch_from_candidate(can)

        #filter 1) minimum skeletal length
        if "skeletal_length" in filters:
            if not nru.candidate_filter_skeletal_length(neuron_obj,
                                                       candidate=can,
                                                       candidate_skeletal_length_min=candidate_skeletal_length_min,
                                                        verbose = verbose,
                                                       **kwargs):
                continue

        # ------- more filters would be implemented here
        if verbose:
            print(f"candidate {j} made it through filtering")
        filtered_cand.append(can)
    return filtered_cand
'''

def upstream_labels(limb_obj,branch_idx,
                   verbose = False):
    """
    Purpose: Will find the labels of the upstream node

    Pseudoode: 
    1) Find the upstream node
    2) return labels of that node

    """
    
    
    upstream_node = nru.upstream_node(limb_obj,branch_idx)
    
    if upstream_node is None:
        labels = []
    else:
        labels = limb_obj[upstream_node].labels
        
    if verbose:
        print(f"upstream_node = {upstream_node} with labels = {labels}")
        
    return labels

def upstream_node_has_label(limb_obj,branch_idx,
                           label,
                           verbose):
    """
    Purpose: To determine if the 
    upstream node has a certain label

    Pseudocode: 
    1) Find the upstream labels
    2) Return boolean if label of interest is in labels
    
    Ex: 
    nru.upstream_node_has_label(limb_obj = n_test[1],
               branch_idx = 9,
                label = "apical",
               verbose = True)
    """
    upstream_labels = nru.upstream_labels(limb_obj,
                                         branch_idx,
                                         verbose = verbose)
    if label in upstream_labels:
        return True
    else:
        return False
    

    
def label_limb_branch_dict(neuron_obj,
                          label,
                           not_matching_labels = None,
                          match_type="all", #ohter option is any
                          ):
    if not nu.is_array_like(label):
        label = [label]
    label_limb_branch = ns.query_neuron_by_labels(
                        neuron_obj,
                        matching_labels=label,
                        not_matching_labels=not_matching_labels,
                        match_type=match_type)
    return label_limb_branch

def downstream_labels(limb_obj,branch_idx,
                      all_downstream_nodes = False,
                     verbose = False):

    """
    Purpose: Get all of the downstream labels
    of a node

    Pseudocode: 
    1) Get all of the downstream nodes (optionally all downstream)
    2) get the labels over all the branches
    3) concatenate the labels

    """
    labels = []
    if all_downstream_nodes:
        d_nodes = nru.all_downstream_branches(limb_obj,branch_idx)
    else:
        d_nodes = nru.downstream_nodes(limb_obj,branch_idx)
        
    if verbose:
        print(f"d_nodes = {d_nodes}")
        
    if len(d_nodes)> 0:
        labels = np.concatenate(nru.feature_over_branches(limb_obj,d_nodes,"labels"))
        labels = np.unique(labels)
        
    if verbose:
        print(f"labels = {labels}")
        
    return list(labels)

def limb_branch_from_limbs(neuron_obj,
                           limbs,
                          ):
    """
    Purpose: To convert list of limbs to limb_branch_dict

    Pseudocode: 
    For each limb
    1) Convert limb to name
    2) Get the branches for the limb and store in dict
    """

    if not nu.is_array_like(limbs):
        limbs = [limbs]

    final_lb = dict()
    for l in limbs:
        l_name = nru.get_limb_string_name(l)
        branches = neuron_obj[l].get_branch_names()
        #print(f"branches = {branches}")
        final_lb[l_name] = branches
    
    return final_lb

def set_branch_attribute_over_neuron(neuron_obj,
                                          branch_func,
                                         verbose = False,
                                       **kwargs):
    """
    Purpose: To set attributes of 
    synapes throughout neuron

    Psueodocde: 
    Iterating through all branches
    1) run the branch func
    
    
    """
    for l in neuron_obj.get_limb_names():
        if verbose:
            print(f"Working on limb {l}")
        for b in neuron_obj[l].get_branch_names():
            branch_obj = neuron_obj[l][b]
            branch_func(branch_obj,**kwargs)
            
def n_branches_over_limb_branch_dict(neuron_obj,
                                     limb_branch_dict
                                    ):
    """
    Purpose: to count up the number of branches in a compartment
    
    nru.n_branches_over_limb_branch_dict(neuron_obj_proof,
                                    apu.oblique_limb_branch_dict(neuron_obj_proof))
    """
    return nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                                limb_branch_dict,
                                                feature = "n_branches")

def all_soma_soma_connections_from_limb(limb_obj,
                                        only_multi_soma_paths=False,
                                       verbose = False):
    """
    Purpose: To return all the soma soma paths on a limb
    
    
    Ex: 
    segment_id = 864691136174988806

    neuron_obj = du.neuron_obj_from_table(    
            segment_id = segment_id,
            table_name = "Decomposition",
            verbose = False
        )

    nru.all_soma_soma_connections_from_limb(neuron_obj[0],
                                            only_multi_soma_paths = True,
                                           verbose = True,
                                           )
    
    """
    
    all_starting_nodes = [f"S{k['starting_soma']}_{k['soma_group_idx']}" for k in limb_obj.all_concept_network_data]

    starting_node_combinations = list(itertools.combinations(all_starting_nodes,2))
    
    starting_node_combinations = [list(k) for k in nu.unique_non_self_pairings(starting_node_combinations)]
    
    if verbose:
        print(f"starting_node_combinations = {starting_node_combinations}")
        
    if only_multi_soma_paths:
        starting_node_combinations = [k for k in starting_node_combinations if
                                         k[0].split("_")[0] != k[1].split("_")[0]]
        if verbose:
            print(f"starting_node_combinations (only_multi_soma_paths) = {starting_node_combinations}")
    return starting_node_combinations
            
def all_soma_names_from_limb(limb_obj):
    return list(np.unique([f'S{k["starting_soma"]}' for k in limb_obj.all_concept_network_data]))

def all_soma_meshes_from_limb(neuron_obj,limb_idx,verbose = False):
    soma_names = all_soma_names_from_limb(neuron_obj[limb_idx])
    if verbose:
        print(f"for limb {limb_idx}, soma_names = {soma_names}")
        
    return [neuron_obj[idx].mesh for idx in soma_names]
    

def soma_idx_and_group_from_name(soma_name):
    split_name = soma_name.split("_")
    return int(split_name[0][1:]),int(split_name[1])
def concept_network_data_from_soma(limb_obj,
                                   soma_name = None,
                                   soma_idx=None,
                                   soma_group_idx=None,
                                  data_name=None):
    if soma_name is not None:
        soma_idx,soma_group_idx = nru.soma_idx_and_group_from_name(soma_name)
        
    concept_data = limb_obj.get_concept_network_data_by_soma_and_idx(soma_idx,soma_group_idx)
    
    if data_name is not None:
        return concept_data[data_name]
    else:
        return concept_data
    
def starting_node_from_soma(limb_obj,
                            soma_name = None,
                           soma_idx=None,
                            soma_group_idx=None,
                          data_name=None):
    """
    Ex: nru.starting_node_from_soma(limb_obj,"S2_0")
    """
    return concept_network_data_from_soma(limb_obj,
                                   soma_idx=soma_idx,
                                          soma_group_idx=soma_group_idx,
                                          soma_name = soma_name,
                                  data_name="starting_node")

def starting_node_combinations_of_limb_sorted_by_microns_midpoint(neuron_obj,
                                                        limb_idx,
                                                        only_multi_soma_paths = False,
                                                        return_soma_names = False,
                                                        verbose = False):
    """
    Purpose: To sort the error
    connections of a limb by the 
    distance of the soma to the midpoint
    of the microns dataset

    Pseudocode: 
    0) Compute the distance of each some to the dataset midpoint
    1) Get all of the possible connection pathways
    2) Construct the distance matrix for the pathways
    3) Order the connection pathways across their rows independent
    4) Order the rows of the connections pathways
    5) Filter for only different soma pathways if requested
    """

    #0) Compute the distance of each some to the dataset midpoint
    limb_obj = neuron_obj[limb_idx]
    midpoint_dist_dict = mru.soma_distances_from_microns_volume_bbox_midpoint(neuron_obj)

    if verbose:
        print(f"midpoint_dist_dict = {midpoint_dist_dict}")
        print(f"All somas attached to limb = {nru.all_soma_names_from_limb(limb_obj)}")


    #1) Get all of the possible connection pathways


    begginning_soma_paths = np.array(nru.all_soma_soma_connections_from_limb(
        limb_obj,
        verbose = False,
        only_multi_soma_paths = only_multi_soma_paths))

    if verbose:
        print(f"begginning_soma_paths= \n{begginning_soma_paths}")

    if len(begginning_soma_paths) > 1: 
        midpoint_dist = np.array([[midpoint_dist_dict[k.split("_")[0]] 
                              for k in v] for v in begginning_soma_paths])
        if verbose:
            print(f"midpoint_dist =\n {midpoint_dist}")


        #3) Order the connection pathways across their rows independent

        row_idx,col_idx = nu.argsort_rows_of_2D_array_independently(midpoint_dist)
        begginning_soma_paths_row_ord = begginning_soma_paths[row_idx,col_idx]
        midpoint_dist_row_ord = midpoint_dist[row_idx,col_idx]

        if verbose:
            print(f"midpoint_dist_row_ord =\n {midpoint_dist_row_ord}")
            print(f"begginning_soma_paths_row_ord =\n {begginning_soma_paths_row_ord}")

        #4) Order the rows of the connections pathways
        row_order = nu.argsort_multidim_array_by_rows(midpoint_dist_row_ord)
        begginning_soma_paths_final = begginning_soma_paths_row_ord[row_order]

        if verbose:
            print(f"begginning_soma_paths_final=\n {begginning_soma_paths_final}")
    else:
        begginning_soma_paths_final= begginning_soma_paths


    if not return_soma_names:
        begginning_soma_paths_final = np.array([[nru.starting_node_from_soma(limb_obj,k) for k in v] for v in begginning_soma_paths_final])
        if verbose:
            print(f"Starting path combinations = \n")
            print(f"{begginning_soma_paths_final}")

    return begginning_soma_paths_final


def shortest_path(limb_obj,start_branch_idx,destiation_branch_idx,
                 plot_path=False):
    shortest_p = np.array(xu.shortest_path(limb_obj.concept_network,start_branch_idx,destiation_branch_idx))
    if plot_path:
        print(f"Plotting path: {shortest_p}")
        nviz.plot_limb_path(limb_obj,shortest_p)
    return shortest_p

def get_soma_meshes(neuron_obj):
    return neuron_obj.get_soma_meshes()
    
    
# ---------- 10/22 -------------
def skeleton_nodes_from_limb_branch(
    neuron_obj,
    limb_branch_dict,
    downsample_size = 1500,
    downsample_factor = None,
    plot_skeletons_before_downsampling = False,
    plot_nodes = False,
    scatter_size = 0.2,
    verbose = False,
    ):
    """
    Purpose: To convert a limb branch dict
    into a list of points from the skeleton
    (and have an option to downsample the number of skeletons)

    downsample_facto

    """

    if type(limb_branch_dict) == dict:
        limb_branch_sks = nru.skeleton_over_limb_branch_dict(
            neuron_obj,
            limb_branch_dict,
            stack_skeletons = False
        )
    elif nu.is_array_like(limb_branch_dict):
        limb_branch_sks = [neuron_obj[k].skeleton for k in limb_branch_dict]

    if plot_skeletons_before_downsampling:
        nviz.plot_objects(
        skeletons = limb_branch_sks,
            skeletons_colors="random"

        )

    if downsample_factor is not None:
        if verbose:
            print(f"Downsampling nodes by a factor of {downsample_factor}")
        nodes = sk.convert_skeleton_to_nodes(
            sk.stack_skeletons(limb_branch_sks)
        )
        nodes = nodes[::downsample_factor]
    elif downsample_size is not None:
        if verbose:
            print(f"Downsampling to segment lengths of {downsample_size}")
        limb_branch_sks_resized = [sk.resize_skeleton_branch(
            k,
            segment_width=downsample_size) for k in limb_branch_sks]
        nodes = sk.convert_skeleton_to_nodes(
            sk.stack_skeletons(limb_branch_sks_resized)
        )
    else:
        if verbose:
            print(f"No downsampling applied")
        nodes = sk.convert_skeleton_to_nodes(
            sk.stack_skeletons(limb_branch_sks)
        )


    if plot_nodes:
        print(f"Plotting nodes extracted from limb branch")
        nviz.plot_objects(
        skeletons=limb_branch_sks,
            scatters=[nodes.reshape(-1,3)],
            scatter_size=scatter_size
        )
    
    return nodes

def skeleton_nodes_from_branches_on_limb(
    limb_obj,
    branches,
    **kwargs
    ):
    """
    Get skeleton nodes just from limb and list of branches
    
    Ex: 
    nru.skeleton_nodes_from_branches_on_limb(neuron_obj[0],[0,1,2],plot_nodes = True)
    
    #checking
    nviz.plot_objects(
        meshes = [neuron_obj[0][k].mesh for k in [0,1,2]],
        skeletons = [neuron_obj[0][k].skeleton for k in [0,1,2]]
    )
    
    """
    
    
    return nru.skeleton_nodes_from_limb_branch(
    limb_obj,
    branches,
    **kwargs
    )

def all_downstream_branches_from_multiple_branhes(
    limb_obj,
    branches_idx,
    include_branches_idx = True,
    verbose = False,
    ):

    """
    Purpose: Get all of the downstream branches of certain 
    other branches that would be removed if those
    branches were deleted

    Ex: 
    all_downstream_branches_from_multiple_branhes(
    neuron_obj[0],
    branches_idx=[20,24],
    )
    """
    branches_idx = np.array(branches_idx)
    branches_idx = branches_idx[branches_idx>=0]
    
    downstream_nodes = nu.union1d_multi_list([nru.all_downstream_branches(
        limb_obj,branch_idx = k) for k in branches_idx])
    if include_branches_idx:
        downstream_nodes = np.union1d(downstream_nodes,branches_idx)
        
    if verbose:
        print(f"Total downstream branches = {downstream_nodes}")
        
    return downstream_nodes


def branch_attr_dict_from_node(
    obj,
    node_name = None,
    attr_list=None,
    include_node_name_as_top_key = False,
    include_branch_dynamics = False,
    verbose = False,):
    """
    Purpose: To output a dictionary of attributes of the node
    attributes
    
    Ex: 
    nru.branch_attr_dict_from_node(
    neuron_obj_proof,
    "S0",
    #attr_list=branch_attributes_global,
    attr_list = soma_attributes_global,
    include_node_name_as_top_key=True)
    """
    curr_obj = obj
    
    if attr_list is None:
        if verbose:
            print(f"Using all attributes as default")
        attr_list = [k for k in dir(curr_obj) if k[0] != "_"]
    
    
    curr_dict = dict([(a,getattr(curr_obj,a)) if type(a) == str else (a[1],getattr(curr_obj,a[0])) for a in attr_list])
    if include_branch_dynamics:
        branch_dyn_dict = bu.branch_dynamics_attr_dict_dynamics_from_node(curr_obj)
        curr_dict.update(branch_dyn_dict)
    
    if include_node_name_as_top_key:
        curr_dict = {node_name: curr_dict}
    return curr_dict

def branch_neighbors(
    limb_obj,
    branch_idx,
    verbose = False,
    include_parent = True,
    include_siblings = False,
    include_children = True):
    """
    Purpose: To get all the neighboring
    branches to current branch

    """
    #neighbors = xu.get_neighbors(limb_obj.concept_network,branch_idx)
    neighbors = []
    
    if include_parent:
        parent_node = nru.parent_node(limb_obj,branch_idx,verbose = verbose)
        if parent_node is not None:
            neighbors.append(parent_node)
            
    if include_siblings:
        sibling_nodes= nru.sibling_nodes(limb_obj,branch_idx,verbose = verbose)
        neighbors += sibling_nodes
        
    if include_children:
        children_nodes = nru.children_nodes(limb_obj,branch_idx, verbose = verbose)
        neighbors += children_nodes
    
    if verbose:
        print(f"neighbors = {neighbors}")
        
    return neighbors

def branch_neighbors_attribute(limb_obj,
                            branch_idx,
                            attr,
                            verbose = False,
                               **kwargs
                           ):
    return [getattr(limb_obj[k],attr)  for k in 
            nru.branch_neighbors(limb_obj,branch_idx,verbose = verbose,**kwargs)]

def branch_neighbors_mesh(limb_obj,
                            branch_idx,
                            verbose = False,
                          **kwargs,
                           ):
    return tu.combine_meshes(nru.branch_neighbors_attribute(
        limb_obj,
        branch_idx,
        "mesh",
        verbose = verbose,
        **kwargs))


def width_average_from_limb_correspondence(
    limb_correspondence,
    verbose = False):
    
    """
    Purpose: To calculate the average width based on a limb correspondence
    dictionary of branch_idx > dict(width, skeleton, mesh)
    
    """

    total_sk_lens = [sk.calculate_skeleton_distance(v["branch_skeleton"]) for k,v in limb_correspondence.items()]
    total_widths = [v["width_from_skeleton"] for k,v in limb_correspondence.items()]
    average_width = nu.weighted_average(total_widths,total_sk_lens)
    if verbose:
        print(f"average_width = {average_width}")
        
    return average_width


# ------- 12/27: 


def combined_somas_neuron_obj(
    neuron_obj,
    inplace = True,
    plot_soma_mesh = False,
    plot_soma_limb_network = False,
    verbose = False
    ):
    """
    Purpose: To combine a neuron object with multiple somas
    into a neuron object with just one soma
    
    Pseudocode: 
    1) Redo the preprocessing data
    
    Inside: preprocessed_data
    soma_meshes:
    - just combine the meshes

    soma_to_piece_connectivity: 
    - just make it a combined dict: 
    Ex: {0: [1, 2, 3, 5, 6, 7, 11], 1: [0, 4, 8], 2: [9, 10]}

    soma_sdfs: just combine as weighted average


    limb_network_stating_info
    - structure: limb_idx > soma_idx > starting_idx > 

    Goal: keep the same but just map to soma_idx = 0
    and reorder the starting idx

    
    
    2) Redo the concept network
    3) Adjust starting info for all limbs
    
    
    Ex: 
    from neurd import neuron_utils as nru
    from neurd import neuron_utils as nru
    neuron_obj = nru.decompress_neuron("./3502576426_somas_seperate.pbz2",original_mesh="./3502576426_0_25.off")

    neuron_obj_comb = nru.combined_somas_neuron_obj(neuron_obj,
                                                    inplace = False,
                                                    verbose = True,
                                                    plot_soma_limb_network = True)

    """
    if len(neuron_obj.get_soma_node_names()) <= 1:
        return neuron_obj
    
    if not inplace:
        neuron_obj = deepcopy(neuron_obj)
        
    
    # 1) --------Redo the preprocessing data--------
    
    preprocessed_data_cp = neuron_obj.preprocessed_data.copy()
    #1) Combine the soma meshes
    preprocessed_data_cp["soma_meshes"] = [tu.combine_meshes(neuron_obj.preprocessed_data["soma_meshes"])]

    if plot_soma_mesh:
        nviz.plot_objects(preprocessed_data_cp["soma_meshes"][0])

    #2) redo the soma to piece connectivity
    preprocessed_data_cp["soma_to_piece_connectivity"] = {0:list(nu.union1d_multi_list(list(
        neuron_obj.preprocessed_data["soma_to_piece_connectivity"].values())).astype('int'))}

    if verbose:
        print(f"New soma_to_piece_connectivity =  {preprocessed_data_cp['soma_to_piece_connectivity']}")

    #3) Combine the Soma sdfs
    preprocessed_data_cp["soma_sdfs"] = np.array([nu.weighted_average(neuron_obj.preprocessed_data["soma_sdfs"],
            [len(k.faces) for k in neuron_obj.preprocessed_data["soma_meshes"]]) ])

    if verbose:
        print(f"New soma_sdfs =  {preprocessed_data_cp['soma_sdfs']}")

    #4) Combine limb_network_stating_info
    new_stating_info = {}
    reverse_mapping = dict()
    for limb_idx,limb_data in neuron_obj.preprocessed_data["limb_network_stating_info"].items():
    #     if verbose:
    #         print(f"Limb: {limb_idx}")
        new_stating_info[limb_idx] = {0:dict()}
        reverse_mapping[limb_idx]  = dict()
        counter = 0
        for soma_idx,soma_data in limb_data.items():
    #         if verbose:
    #             print(f" Soma: {soma_idx}")
            for group_idx,soma_group_data in soma_data.items():
    #             if verbose:
    #                 print(f"   Group: {group_idx}")
                new_stating_info[limb_idx][0][counter] = soma_group_data
                reverse_mapping[limb_idx][(soma_idx,group_idx)] = counter
                counter += 1
    if verbose:
        print(f"reverse_mapping = {reverse_mapping}")

    preprocessed_data_cp["limb_network_stating_info"] = new_stating_info
    neuron_obj.preprocessed_data = preprocessed_data_cp
    
    # --------2) Fixing the concept network--------
    if plot_soma_limb_network:
        print(f"BEFORE reorganization: ")
        nviz.plot_soma_limb_concept_network(neuron_obj)
    
    sdf_comb = nu.weighted_average([neuron_obj[k].sdf for k in neuron_obj.get_soma_node_names()],
            [len(neuron_obj[k].mesh.faces) for k in neuron_obj.get_soma_node_names()])

    if verbose:
        print(f"sdf_comb= {sdf_comb}")

    mesh_face_idx_comb = nu.union1d_multi_list([neuron_obj[k].mesh_face_idx for k in neuron_obj.get_soma_node_names()])
    if verbose:
        print(f"mesh_face_idx_comb.shape = {mesh_face_idx_comb.shape}")

    mesh_comb = tu.combine_meshes([neuron_obj[k].mesh for k in neuron_obj.get_soma_node_names()])

    Soma_obj = neuron.Soma(mesh=mesh_comb,mesh_face_idx=mesh_face_idx_comb,sdf=sdf_comb)
    
    soma_temp_name = "S_new"
    neuron_obj.concept_network.add_nodes_from([soma_temp_name])
    neuron_obj.concept_network.nodes[soma_temp_name]["data"] = Soma_obj
    
    neuron_obj.concept_network.remove_nodes_from([k for k in neuron_obj.get_soma_node_names() if k != soma_temp_name])
    neuron_obj.concept_network.add_edges_from([(soma_temp_name,k) for k in neuron_obj.get_limb_node_names(return_int=False)])
    
    nx.relabel_nodes(neuron_obj.concept_network,dict(S_new="S0"),copy=False)
    if plot_soma_limb_network:
        print(f"AFTER reorganization: ")
        nviz.plot_soma_limb_concept_network(neuron_obj)
    
    
    #3) --------fixing the all_concept_network_data:--------
    for limb_idx in neuron_obj.get_limb_names(return_int=True):
        limb_obj = neuron_obj[limb_idx]
        all_concept_network_data_revised = []
        for d in limb_obj.all_concept_network_data:
            d["soma_group_idx"] = reverse_mapping[limb_idx][(d["starting_soma"],d["soma_group_idx"])]
            d["starting_soma"] = 0

        # Go through and recalculate all of the concept networkx
        limb_obj.set_concept_network_directional(starting_soma = "S0")
        
    return neuron_obj


def mesh_not_in_neuron_branches(neuron_obj,
                               plot=False):
    """
    To figure out what part of the mesh is not 
    incorporated into the branches
    """
    leftover_mesh = tu.subtract_mesh(neuron_obj.mesh,
                nru.neuron_mesh_from_branches(neuron_obj))
    if plot:
        nviz.plot_objects(leftover_mesh)
        
    return leftover_mesh
    
    
def filter_away_neuron_limbs(
    neuron_obj,
    limb_idx_to_filter,
    plot_limbs_to_filter = False,
    verbose = False,
    in_place = False,
    plot_final_neuron= False,
    ):
    """
    Purpose: To filter away limbs 
    specific

    Application: To filter away limbs that
    are below a certain skeletal length

    Pseudocode: 
    1) Find the new mapping of the old limb idx to new limb idx
    2) Create the new preprocessing dict of the neuron

    soma_to_piece_connectivity
    limb_correspondence
    limb_meshes
    limb_mehses_face_idx
    limb_labels
    limb_concept_networks
    limb_network_stating_info


    3) Delete and rename the nodes of the graph
    """


    if in_place == False:
        neuron_obj = copy.deepcopy(neuron_obj)

    limb_idxs = np.array(neuron_obj.get_limb_names(return_int=True))

    if plot_limbs_to_filter:
        print(f"Limbs to filter away: {limb_idx_to_filter}")
        nviz.visualize_neuron(neuron_obj,limb_branch_dict={f"L{k}":"all" for k in limb_idx_to_filter})

    keep_idx = np.setdiff1d(limb_idxs,limb_idx_to_filter)
    map_idx = {k:i for i,k in enumerate(keep_idx) }
    if verbose:
        print(f"map_idx= {map_idx}")


    #2) Create the new preprocessing dict of the neuron
    soma_to_piece_connectivity_new = dict()
    for soma_idx,soma_limbs in neuron_obj.preprocessed_data["soma_to_piece_connectivity"].items():
        soma_to_piece_connectivity_new[soma_idx] = [map_idx[k] for k in soma_limbs if k in keep_idx]

    neuron_obj.preprocessed_data["soma_to_piece_connectivity"] = soma_to_piece_connectivity_new

    list_attr = ["limb_meshes","limb_mehses_face_idx"]
    dict_attr = ["limb_correspondence",     
                "limb_labels",
                "limb_concept_networks",
                "limb_network_stating_info",
                ]
    for l in list_attr:
        neuron_obj.preprocessed_data[l] = [k for i,k in enumerate(neuron_obj.preprocessed_data[l]) if i in keep_idx]
        if verbose:
            print(f"{l}_new = {len(neuron_obj.preprocessed_data[l])}")
    for d in dict_attr:
        neuron_obj.preprocessed_data[d] = {map_idx[k]:v for k,v in 
                               neuron_obj.preprocessed_data[d].items() if k in keep_idx}
        if verbose:
            print(f"{d}_new = {neuron_obj.preprocessed_data[d].keys()}")


    #3) Delete and rename the nodes of the graph
    map_idx_limbs_names = {f"L{k}":f"L{v}" for k,v in map_idx.items()}
    if verbose:
        print(f"map_idx_limbs_names= {map_idx_limbs_names}")

    from datasci_tools import networkx_utils as xu
    neuron_obj.concept_network.remove_nodes_from([f"L{k}" for k in limb_idx_to_filter])
    xu.relabel_node_names(neuron_obj.concept_network,map_idx_limbs_names)
    if verbose:
        print(f"neuron_obj.concept_network.nodes at end = {neuron_obj.concept_network.nodes}")

    if plot_final_neuron:
        nviz.visualize_neuron(neuron_obj,limb_branch_dict="all")
    
    return neuron_obj

def filter_away_neuron_limbs_by_min_skeletal_length(
    neuron_obj,
    min_skeletal_length_limb = 10_000,
    verbose = False,
    
    #arguments for filtering neuron_limbs
    plot_limbs_to_filter = False,
    in_place = False,
    plot_final_neuron= False,
    ):
    
    """
    Purpose: To filter away neuron_limbs if below
    a certain skeletal length
    """
    limb_sk_length = np.array([neuron_obj[l_idx].skeletal_length for l_idx in neuron_obj.get_limb_node_names(return_int=True)])
    limb_idx_to_filter = np.where(limb_sk_length<min_skeletal_length_limb)[0]
    
    if verbose:
        print(f"limb_idx_to_filter = {limb_idx_to_filter}")

    n_obj = nru.filter_away_neuron_limbs(
        neuron_obj,
        limb_idx_to_filter,
        plot_limbs_to_filter = plot_limbs_to_filter,
        verbose = verbose,
        in_place = in_place,
        plot_final_neuron= plot_final_neuron,
        )
    
    return n_obj

def order_branches_by_skeletal_distance_from_soma(
    limb_obj,
    branches,
    verbose  = False,
    closest_to_farthest = True,
    ):
    """
    Purpose: To order branches from 
    most upstream to most downstream
    accroding to skeletal distance from soma

    Pseudocode:
    1) Calculate the skeletal distance from soma for branches
    2) Order and return
    """
    branches = np.array(branches)
    soma_dists = [nst.distance_from_soma(
                    limb_obj,k,
    ) for k in branches]
    
    soma_dist_order = np.argsort(soma_dists)
    
    if not closest_to_farthest:
        soma_dist_order = np.flip(soma_dist_order)
        
    if verbose:
        print(f"for branches ({branches}) soma_dists = {soma_dists}")
        print(f"soma_dist_order = {soma_dist_order}")
        
    return branches[soma_dist_order]


def parent_node(
    limb_obj,
    branch_idx,
    verbose=False):
    """
    Purpose: to get the parent 
    branch of a branch_idx
    
    """
    parent_node = xu.upstream_node(limb_obj.concept_network_directional,branch_idx)
    if verbose:
        print(f"parent_node of branch {branch_idx} = branch {parent_node}")
        
    return parent_node

def sibling_nodes(
    limb_obj,
    branch_idx,
    verbose=False):
    """
    Purpose: to get the parent 
    branch of a branch_idx
    
    """
    sibling_nodes = xu.sibling_nodes(limb_obj.concept_network_directional,branch_idx)
    if verbose:
        print(f"sibling_nodes of branch {branch_idx} = {sibling_nodes}")
        
    return list(sibling_nodes)

def children_nodes(
    limb_obj,
    branch_idx,
    verbose=False):
    """
    Purpose: to get the parent 
    branch of a branch_idx
    
    Ex: 
    nru.children_nodes(limb_obj,7)
    """
    child_nodes = nru.downstream_nodes(limb_obj,branch_idx)
    if verbose:
        print(f"child_nodes of branch {branch_idx} = {child_nodes}")
        
    return list(child_nodes)


def neighborhood_mesh(
    limb_obj,
    branch_idx,
    verbose  = False,
    plot = False,
    neighborhood_color = "red",
    branch_color = "blue",
    ):
    """
    Purpose: To get the branch
    parent,siblings and children
    mesh around a mesh
    
    Ex: 
    neighborhood_mesh(
        limb_obj,
        branch_idx,
        plot = True)
    """
    mesh = nru.branch_neighbors_mesh(
    limb_obj,
    branch_idx,
    include_siblings=True,
    include_parent = True,
    include_children = True,
    verbose = verbose)
    
    if verbose:
        print(f"Neighborhood mesh = {mesh}")
        
    if plot:
        nviz.plot_objects(
            meshes = [limb_obj[branch_idx].mesh,mesh],
            meshes_colors=[branch_color,neighborhood_color]
        )
    
    return mesh

def is_branch_mesh_connected_to_neighborhood(
    limb_obj,
    branch_idx,
    verbose = False,
    plot = False,
    default_value = True,
    ):
    """
    Purpose: Determine if a branch mesh
    has connectiviity to its neighborhood mesh

    Pseudocode: 
    1) Get the neighborhood mesh
    2) Find mesh connectivity
    3) Return True if connected
    
    Ex: 
    limb_idx = 1
    branch_idx = 10
    limb_obj = neuron_obj[limb_idx]
    nru.is_branch_mesh_connected_to_neighborhood(
        limb_obj,
        branch_idx,
        verbose = True
    )
    
    """


    branch_obj = limb_obj[branch_idx]

    n_mesh  = nru.neighborhood_mesh(limb_obj,
                                    branch_idx, 
                                    plot = plot,
                                    verbose = verbose)
    
    if len(n_mesh.faces) == 0:
        return default_value

    mesh_conn = tu.mesh_list_connectivity(
        [n_mesh,branch_obj.mesh],
        main_mesh=limb_obj.mesh,
        verbose = verbose
    )
    
    return len(mesh_conn) > 0


def pair_branch_connected_components_by_common_upstream(
    limb_obj,
    conn_comp,
    verbose = False,
    ):
    """
    Purpose: To group connected components
    of branches by a common upstream branch

    Pseudocode: 
    1) For each connected component find the upstream branch
    and add the connected component to the dictionary book-keeping
    2) combine all the connected components in the dictionary
    
    Ex: 
    nru.pair_branch_connected_components_by_common_upstream(
    neuron_obj[1],
    conn_comp = [[13], [14], [9, 12, 15, 16, 19], [51, 21, 22, 26, 27, 28], [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 20, 23, 24, 25, 31], [32], [48, 33, 45]],
    verbose = True)
    """
    if verbose:
        print(f"Before connected components combined: {conn_comp}")
        
    upstream_dict = dict()
    
    for j,c in enumerate(conn_comp):
        most_up_branch = nru.most_upstream_branch(limb_obj,c)
        parent_b = nru.parent_node(limb_obj,most_up_branch)
        if parent_b is None:
            parent_b = -1
        
        if verbose:
            print(f"comp {j} most upstream branch {most_up_branch} had parent {parent_b}")
            
        if parent_b not in upstream_dict:
            upstream_dict[parent_b] = []
        
        upstream_dict[parent_b].append(j)
        
    final_comp = [np.concatenate([conn_comp[k] for k in v]) for v in upstream_dict.values()]
    
    if verbose:
        print(f"AFTER connected components combined: {final_comp}")
        
    return final_comp

def recalculate_endpoints_and_order_skeletons_for_branch(branch_obj):
    branch_obj.calculate_endpoints()
    branch_obj.order_skeleton_by_smallest_endpoint()
    branch_obj._skeleton_graph = None
    branch_obj._endpoints_nodes = None
    
def recalculate_endpoints_and_order_skeletons_over_neuron(neuron_obj):
    """
    Purpose: Recalculate endpoints
    and order the skeletons
    """

    for limb_idx in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            #branch_obj = limb_obj[branch_idx]
            nru.recalculate_endpoints_and_order_skeletons_for_branch(neuron_obj[limb_idx][branch_idx])
            
def candidate_limb_branch_dict_branch_intersection(
    candidate,
    limb_branch_dict,
    return_candidate = False,
    verbose = False):
    """
    Purpose: To find which branches are in both the candidate and limb branch
    """
    limb_idx = candidate["limb_idx"]
    if limb_idx not in limb_branch_dict:
        return_branches= np.array([])
    else:
        return_branches = np.intersect1d(candidate["branches"],limb_branch_dict[limb_idx])
        
    if return_candidate:
        return dict(limb_idx=limb_idx,branches=return_branches)
    else:
        return return_branches
    
def limb_branch_from_keywords(neuron_obj,limb_branch_dict):
    """
    Purpose: To fill in the branches part of limb branch dict if
    used keywords instead of branches numbers
    
    """
    new_dict = dict()
    for k,v in limb_branch_dict.items():
        if v == "all":
            new_dict[k] = neuron_obj[k].get_branch_names()
        else:
            new_dict[k] = v
    
    return new_dict
    

    
def limb_correspondence_on_limb(
    limb_obj,
    width_name = "width"):
    
    limb_corr = dict()
    #for idx,b in enumerate(self):
    for idx in limb_obj.get_branch_names():
        b = limb_obj[idx]
        
        if width_name == "width":
            curr_width = b.width
        else:
            curr_width = b.width_new[width_name]

        limb_corr[idx] = dict(branch_skeleton=b.skeleton,
                              width_from_skeleton = curr_width,
                             branch_mesh = b.mesh,
                             branch_face_idx = b.mesh_face_idx,
                             )


    return limb_corr

def limb_correspondence_on_neuron(
    neuron_obj,
    **kwargs
    ):
    
    return {limb_idx:nru.limb_correspondence_on_limb(
        neuron_obj[limb_idx],**kwargs) for limb_idx in neuron_obj.get_limb_names(return_int=True)}

def get_starting_node_from_limb_concept_network(limb_obj):    
    return xu.get_starting_node(limb_obj.concept_network)

def set_preprocessed_data_from_limb_no_mesh_change(
    neuron_obj,
    limb_idx,
    limb_obj=None,
    ):
    
    if limb_obj is None:
        limb_obj = neuron_obj[limb_idx]
    
    neuron_obj.preprocessed_data["limb_correspondence"][limb_idx]= nru.limb_correspondence_on_limb(limb_obj)
    neuron_obj.preprocessed_data["limb_network_stating_info"][limb_idx] = nru.all_concept_network_data_to_limb_network_stating_info(
        limb_obj.all_concept_network_data
    )
    neuron_obj.preprocessed_data["limb_concept_networks"][limb_idx] = limb_obj.concept_network
    
    return neuron_obj

def translate_neuron_obj(
    neuron_obj,
    translation = None,
    new_center = None,
    in_place = False,
    verbose = False,
    plot_final_neuron = False,
    align_synapses=True,
    **kwargs):
    """
    Purpose: To rotate all of the meshes
    and skeletons of a neuron object
    
    Ex: 
    neuron_obj_rot = copy.deepcopy(neuron_obj)
    mesh_center = neuron_obj["S0"].mesh_center
    for i in range(0,10):
        neuron_obj_rot = hvu.align_neuron_obj(neuron_obj_rot,
                                             mesh_center=mesh_center,
                                             verbose =True)
    nviz.visualize_neuron(
        neuron_obj_rot,limb_branch_dict = "all")
    
    Ex: 
    neuron_obj_1 = nru.translate_neuron_obj(
        neuron_obj_h01_aligned,
        new_center=neuron_obj_m65["S0"].mesh_center,
        plot_final_neuron = True)
    
    """
    if not in_place:
        neuron_obj = copy.deepcopy(neuron_obj)

    if translation is None:
        translation = new_center - neuron_obj["S0"].mesh_center

    if verbose:
        print(f"translation = {translation}")
        
    for j,limb_obj in enumerate(neuron_obj):
        for branch_obj in limb_obj:
            branch_obj.mesh = tu.translate_mesh(branch_obj.mesh,translation = translation)
            branch_obj.mesh_center = tu.mesh_center_vertex_average(branch_obj.mesh)

            branch_obj.skeleton = branch_obj.skeleton + translation
            branch_obj.endpoints = branch_obj.endpoints + translation
            
            
            if align_synapses:
                for syn in branch_obj.synapses:
                    for att in syu.synapse_coordinate_system_dependent_attributes:
                        setattr(syn,att,getattr(syn,f"{att}") + translation)
                        
            #doing the spine alignment
            if branch_obj.spines is not None:
                branch_obj.spines = [tu.translate_mesh(k,translation = translation) for k in branch_obj.spines]
            
            if branch_obj.spines_obj is not None:
                for s_obj in branch_obj.spines_obj:
                    s_obj.mesh = tu.translate_mesh(s_obj.mesh,translation = translation)
                
                        
        #changing the concept network
        all_concept_network_data = []
        att_to_change = ["starting_endpoints","starting_coordinate","touching_soma_vertices"]
        
        for k in limb_obj.all_concept_network_data:
            new_data = copy.deepcopy(k)
            for att in att_to_change:
                new_data[att] = k[att] + translation
            all_concept_network_data.append(new_data)
            
        for att in att_to_change:
            setattr(limb_obj,f"current_{att}",getattr(limb_obj,f"current_{att}") + translation)

        limb_obj.mesh = tu.translate_mesh(limb_obj.mesh,translation = translation)
        limb_obj.all_concept_network_data = copy.deepcopy(all_concept_network_data)
        limb_obj.set_concept_network_directional()
        
    neuron_obj.mesh = tu.translate_mesh(neuron_obj.mesh,translation)

        
    #finishing soma mesh stuff
    for s_name in neuron_obj.get_soma_node_names():
        neuron_obj[s_name].mesh = tu.translate_mesh(neuron_obj[s_name].mesh,translation)
        
        if align_synapses:
            for syn in neuron_obj[s_name].synapses:
                for att in syu.synapse_coordinate_system_dependent_attributes:
                    setattr(syn,att,getattr(syn,f"{att}") + translation)

        neuron_obj[s_name].mesh_center = neuron_obj[s_name].mesh_center + translation
        
    if plot_final_neuron:
        nviz.visualize_neuron(neuron_obj,limb_branch_dict = "all")
    return neuron_obj

def align_neuron_objs_at_soma(
    neuron_objs,
    center = None,
    plot = False,
    inplace = False,
    verbose = True,
    ):
    """
    Purpose: Align two neuron objects at their soma

    1) Get the mesh centers of both
    2) Find the translation needed
    3) Adjust all attributes by that amount
    """
    if not inplace:
        neuron_objs_trans = [copy.deepcopy(k) for k in neuron_objs]

    if center is None:
        center = neuron_objs[0]["S0"].mesh_center

    if verbose:
        print(f"center = {center}")

    neuron_objs_trans = [ 
        nru.translate_neuron_obj(
        k,
        new_center=center,
        plot_final_neuron = False) 
        for k in neuron_objs_trans]

    if plot:
        nviz.plot_objects(
            meshes = [k.mesh for k in neuron_objs_trans],
            meshes_colors = "random",
        )

    return neuron_objs_trans

def non_axon_like_limb_branch_on_dendrite(
    n_obj,
    plot = False):

    non_axon_like_on_dendrite =  nru.limb_branch_setdiff(
            [n_obj.dendrite_limb_branch_dict,
            ns.query_neuron_by_labels(
                n_obj,
                matching_labels=["axon-like"],
            )]
    )

    if plot:
        nviz.plot_limb_branch_dict(
            n_obj,
            non_axon_like_on_dendrite
        )
        
    return non_axon_like_on_dendrite

def add_limb_branch_combined_name_to_df(
    df,
    limb_column = "limb_idx",
    branch_column = "branch_idx",
    limb_branch_column = "limb_branch",
    ):
    """
    Purpose: To add the limb_branch column to
    a dataframe

    Pseudocode
    """

    limbs = [nru.get_limb_string_name(k) for k in df[limb_column].to_list()]
    branches = [int(k) if not np.isnan(k) else -1 for k in df[branch_column].to_list()]
    df[limb_branch_column] = [f"{k}_{v}" for k,v in zip(limbs,branches)]
    return df
        
def limb_branch_str_names_from_limb_branch_dict(
    limb_branch_dict,
    ):
    """
    Purpos: Creates names like
    
    ['L0_0',
     'L0_1',
     'L0_2',
     'L0_3',
     'L0_4',
     'L0_5',
     'L0_6',
     'L0_7',
    """
    names = []
    for limb_name, branches in limb_branch_dict.items():
        names += [f"{limb_name}_{b}" for b in branches]
        
    return names

def limb_branch_face_idx_dict_from_neuron_obj_overlap_with_face_idx_on_reference_mesh(
    neuron_obj,
    mesh_reference,
    faces_idx=None,
    mesh_reference_kdtree = None,
    limb_branch_dict = None,
    overlap_percentage_threshold = 5,
    return_limb_branch_dict = False,
    verbose = False,
    ):
    """
    Purpose: Want to find a limb branch dict of branches
    that have a certain level of face overlap with given
    faces

    Pseudocode:
    Generate a KDTree for the mesh_reference
    For each branch in limb branch:
        a. Get the faces corresponding to the mesh_reference
        b. Compute the percentage overlap with the faces_idx_list
        c. If above certain threshold then store the limb,branch,face-list 
        in the dictionary

    return either the limb branch dict or limb-branch-facelist dict
    """
    if faces_idx is None:
        faces_idx = np.arange(len(mesh_reference.faces))

    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict

    if mesh_reference_kdtree is None:
        mesh_reference_kdtree = KDTree(mesh_reference.triangles_center)

    output_dict = dict()
    for limb_idx,branches in limb_branch_dict.items():
        if verbose:
            print(f"Working on limb {limb_idx}")
        for b in branches:
            if verbose:
                print(f"   -> Working on branch {b}")
            branch_obj_mesh = neuron_obj[limb_idx][b].mesh
            branch_faces = tu.original_mesh_faces_map(
                original_mesh = mesh_reference,
                submesh = branch_obj_mesh,
                exact_match = True,
                original_mesh_kdtree = mesh_reference_kdtree,
            )

            overlap_faces = np.intersect1d(faces_idx,branch_faces)
            overlap_faces_perc = len(overlap_faces)/len(branch_faces)*100

            if verbose:
                print(f"overlap_faces_perc = {overlap_faces_perc}")
            if overlap_faces_perc > overlap_percentage_threshold:
                if verbose:
                    print(f"Adding branch to final dict")
                if limb_idx not in output_dict:
                    output_dict[limb_idx] = dict()
                output_dict[limb_idx][b] = branch_faces


    if return_limb_branch_dict:
        output_dict = {k:list(v.keys()) for k,v in output_dict}

    return output_dict
    #package where can use the Branches class to help do branch skeleton analysis
    
    
def calculate_decomposition_products(
    neuron_obj,
    store_in_obj = False,
    verbose = False,
    ):
    
    # ---- basic statistics of neuron
    stats_dict = neuron_obj.neuron_stats(stats_to_ignore = [
                    "n_boutons",
                     "axon_length",
                     "axon_area",
                     "max_soma_volume",
                     "max_soma_n_faces",],
        include_skeletal_stats = True,
        include_centroids= True,
        #voxel_adjustment_vector=voxel_adjustment_vector,
    )
    
    # --- generating the skeleton 
    skeleton = neuron_obj.skeleton
    
    # --- skeleton stats
    sk_stats = nst.skeleton_stats_from_neuron_obj(
            neuron_obj,
            include_centroids=True,
            verbose = verbose,
    )
    
    stats_dict.update(sk_stats)
    decomp_products = pipeline.StageProducts(
        skeleton=skeleton,
        **stats_dict,
    )

    if store_in_obj:
        neuron_obj.pipeline_products.set_stage_attrs(
            decomp_products,
            stage = "decomposition"
        )
        
    return decomp_products


# ----------- for aligning neuron ---------
align_attr = "align_matrix"

def align_attribute(obj,attribute_name,
                    soma_center=None,
                    rotation=None,
                   align_matrix = None,):
    setattr(obj,f"{attribute_name}",align_array(
                getattr(obj,f"{attribute_name}"),
        soma_center=soma_center,
        rotation=rotation,
        align_matrix = align_matrix))
def align_array(array,align_matrix = None,**kwargs):
    return nu.align_array(array,align_matrix = align_matrix)

def align_mesh(mesh,align_matrix = None,**kwargs):
    return meshu.align_mesh(mesh,align_matrix=align_matrix)

def align_skeleton(skeleton,align_matrix = None,**kwargs):
    return nu.align_array(array=skeleton,align_matrix = align_matrix)

def align_neuron_obj_from_align_matrix(
    neuron_obj,
    align_matrix=None,
    align_synapses = True,
    verbose = False,
    align_array = align_array,
    align_mesh=align_mesh,
    align_skeleton=align_skeleton,
    in_place = False,
    **kwargs
    ):
    
    
    if align_matrix is None:
        align_matrix = getattr(neuron_obj,align_attr,None)
    
    if align_matrix is None:
        return neuron_obj
    
    if not in_place:
        neuron_obj = copy.deepcopy(neuron_obj)
    
    for j,limb_obj in enumerate(neuron_obj):
        for branch_obj in limb_obj:
            branch_obj.mesh = align_mesh(
                                branch_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
            )

            branch_obj.mesh_center = tu.mesh_center_vertex_average(branch_obj.mesh)

            branch_obj.skeleton = align_skeleton(
                                branch_obj.skeleton,
                                align_matrix=align_matrix,
                                verbose = False
            )
            branch_obj.endpoints = align_array(branch_obj.endpoints,
                                                    align_matrix=align_matrix,)
            
            
            
            if align_synapses:
                for syn in branch_obj.synapses:
                    for att in syu.synapse_coordinate_system_dependent_attributes:
                        align_attribute(syn,att,align_matrix=align_matrix,)
                        
            #doing the spine alignment
            if branch_obj.spines is not None:
                branch_obj.spines = [align_mesh(k,align_matrix=align_matrix) for k in branch_obj.spines]
            
            if branch_obj.spines_obj is not None:
                for s_obj in branch_obj.spines_obj:
                    s_obj.mesh = align_mesh(s_obj.mesh,align_matrix=align_matrix)
                
                        
            
        
        #changing the concept network
        all_concept_network_data = []
        att_to_change = ["starting_endpoints","starting_coordinate","touching_soma_vertices"]
        
        for k in limb_obj.all_concept_network_data:
            new_data = copy.deepcopy(k)
            for att in att_to_change:
                new_data[att] = align_array(k[att],align_matrix=align_matrix,)
            all_concept_network_data.append(new_data)
            
        for att in att_to_change:
            setattr(limb_obj,f"current_{att}",align_array(
                getattr(limb_obj,f"current_{att}"),align_matrix=align_matrix,))

        limb_obj.mesh = align_mesh(
                                limb_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
            )
        limb_obj.all_concept_network_data = copy.deepcopy(all_concept_network_data)
        limb_obj.set_concept_network_directional()
        
    neuron_obj.mesh = align_mesh(
                                neuron_obj.mesh,
                                align_matrix=align_matrix,
                                verbose = False
                                )
        
    #finishing soma mesh stuff
    for s_name in neuron_obj.get_soma_node_names():
        neuron_obj[s_name].mesh = align_mesh(
            neuron_obj[s_name].mesh ,
            align_matrix=align_matrix,
        )
        
        if align_synapses:
            for syn in neuron_obj[s_name].synapses:
                for att in syu.synapse_coordinate_system_dependent_attributes:
                    align_attribute(syn,att,align_matrix=align_matrix,)

        neuron_obj[s_name].mesh_center = tu.mesh_center_vertex_average(neuron_obj[s_name].mesh)
        #print(f"neuron_obj[s_name].mesh_center = {neuron_obj[s_name].mesh_center}")
        
        
    return neuron_obj


def unalign_neuron_obj_from_align_matrix(
    neuron_obj,
    align_matrix=None,
    verbose = False,
    **kwargs
    ):
    
    if align_matrix is None:
        align_matrix = getattr(neuron_obj,align_attr,None)
    
    if align_matrix is None:
        return neuron_obj
    
    align_matrix = np.linalg.inv(align_matrix)
    
    
    curr_neuron =  align_neuron_obj_from_align_matrix(
        neuron_obj,
        align_matrix=align_matrix,
        verbose = verbose,
        **kwargs
        )
    
    setattr(curr_neuron,align_attr,None)
    return curr_neuron
    
    
def most_upstream_conn_comp_node(
    neuron_obj,
    limb_branch_dict=None,
    verbose = False
    ):
    """
    Purpose: Given a limb branch dict, find all of the root branches
    of the subgraphs

    Pseudocode: 
    iterating through all of the limbs of the limb branch
    1) Divide into connected components

        For each connected component:
        a) Find the most upstream node
        b) add to the list for this limb branch

    Ex:
    nru.most_upstream_conn_comp_node_from_limb_branch_dict(
        limb_branch_dict = n_obj_proof.basal_limb_branch_dict,
        neuron_obj = n_obj_proof,
        verbose = True,
    )

    """
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict

    upstream_nodes_dict = dict()
    for limb_name,branches_idx in limb_branch_dict.items():

        upstream_nodes_dict[limb_name] = []
        limb_obj = neuron_obj[limb_name]

        conn_comp = nru.connected_components_from_branches(
            limb_obj,
            branches=branches_idx,
        )

        if verbose:
            print(f"Working on {limb_name}: # of conn comp = {len(conn_comp)}")
            print(f" -- most upstream node --")
        for j,cc in enumerate(conn_comp):
            most_upstream_node = nru.most_upstream_branch(limb_obj,cc)

            if verbose:
                print(f"   conn comp {j}: {most_upstream_node}")

            upstream_nodes_dict[limb_name].append(most_upstream_node)


    return upstream_nodes_dict

def statistic_per_branch(
    neuron_obj,
    stat_func,
    limb_branch_dict=None,
    suppress_errors = False,
    default_value = None,
    ):
    """
    Purpose: Find a statistic for a limb branch dict
    
    Pseudocode: 
    1) 
    """
    
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict
        

    def local_stat(branch_obj):
        try:
            if isinstance(stat_func,str):
                return_value = getattr(branch_obj,stat_func)
            else:
                return_value = stat_func(branch_obj)
        except Exception as e:
            if suppress_errors:
                return_value = default_value
            else:
                raise Exception(e)
        return return_value
        
    stat_dict = dict([(limb_name,[local_stat(neuron_obj[limb_name][k]) for k in branches]) for limb_name,branches in limb_branch_dict.items()])
    
    return stat_dict

def most_upstream_conn_comp_node_stat(
    neuron_obj,
    stat_func,
    limb_branch_dict=None,
    verbose = False,
    return_upstream_conn_comp_nodes = False,
    **kwargs
    ):
    """
    Purpose: calculate the statistic for the most upstream node 
    of every connected component in a limb branch dict
    """
    upstream_limb_branch_dict = nru.most_upstream_conn_comp_node(
        neuron_obj,
        limb_branch_dict=limb_branch_dict,
        verbose = verbose
    )
    
    stats_dict = statistic_per_branch(
        neuron_obj,
        stat_func,
        limb_branch_dict=upstream_limb_branch_dict,
        **kwargs
    )
    
    if return_upstream_conn_comp_nodes:
        return stats_dict,upstream_limb_branch_dict
    else:
        return stats_dict
    
roots_stat = most_upstream_conn_comp_node_stat

def compartment_roots_stat(
    neuron_obj,
    compartment,
    stat_func,
    verbose = False,
    return_root_nodes = False,
    **kwargs
    ):
    """
    Purpose: To compute the statistic for all the
    root nodes of a certain compartment
    """
    return nru.most_upstream_conn_comp_node_stat(
    neuron_obj,
    stat_func,
    limb_branch_dict=nru.label_limb_branch_dict(neuron_obj,compartment),
    verbose = verbose,
    return_upstream_conn_comp_nodes = return_root_nodes,
    **kwargs
    )
    


def compartment_roots_stat_extrema(
    neuron_obj,
    compartment,
    stat_func,
    extrema = "max",
    return_limb_branch_idx=False,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: to compute the extrema of all of the root statistics
    for a certain compartment
    """
    
    if isinstance(extrema,str):
        extrema_func = getattr(np,extrema)
    else:
        extrema_func = extrema
    
    stat_dict,lb = compartment_roots_stat(
        neuron_obj,
        compartment = compartment,
        stat_func=stat_func,
        return_root_nodes = True,
        **kwargs
    )
    
    stats_list = []
    name_list = []
    
    for limb_name in lb:
        stats_list += stat_dict[limb_name]
        name_list += [f"{limb_name}_{k}" for k in lb[limb_name]]
        
    extrema_idx = getattr(np,f"arg{extrema_func.__name__}")(stats_list)
    extrema_value = stats_list[extrema_idx]
    extream_name = name_list[extrema_idx]
    
    if verbose:
        print(f"{extrema_func.__name__} stat = {extrema_value} ({extream_name})")
        
    if return_limb_branch_idx:
        return extrema_value,extream_name
    else:
        return extrema_value
    
# def compartment_roots_stat_max(
#     neuron_obj,
#     compartment,
#     stat_func,
#     return_limb_branch_idx=False,
#     verbose = False,
#     **kwargs
#     ):
    
#     return compartment_roots_stat_extrema(
#     neuron_obj,
#     compartment=compartment,
#     stat_func=stat_func,
#     extrema = "max",
#     return_limb_branch_idx=return_limb_branch_idx,
#     verbose = verbose,
#     **kwargs
#     )

 
# def compartment_roots_stat_min(
#     neuron_obj,
#     compartment,
#     stat_func,
#     return_limb_branch_idx=False,
#     verbose = False,
#     **kwargs
#     ):
    
#     return compartment_roots_stat_extrema(
#     neuron_obj,
#     compartment=compartment,
#     stat_func=stat_func,
#     extrema = "min",
#     return_limb_branch_idx=return_limb_branch_idx,
#     verbose = verbose,
#     **kwargs
#     )

# def compartment_root_skeleton_angle_max(
#     neuron_obj,
#     compartment,
#     return_limb_branch_idx = False,
#     verbose = False,
#     **kwargs
#     ):
    
#     return nru.compartment_roots_stat_max(
#         neuron_obj,
#         compartment = compartment,
#         stat_func = bu.skeleton_angle_from_top,
#         return_limb_branch_idx = return_limb_branch_idx,
#         verbose = verbose,
#         **kwargs
#     )
    
# def compartment_root_skeleton_angle_min(
#     neuron_obj,
#     compartment,
#     return_limb_branch_idx = False,
#     verbose = False,
#     **kwargs
#     ):
    
#     return nru.compartment_roots_stat_min(
#         neuron_obj,
#         compartment = compartment,
#         stat_func = bu.skeleton_angle_from_top,
#         return_limb_branch_idx = return_limb_branch_idx,
#         verbose = verbose,
#         **kwargs
#     )
    
from functools import partial,update_wrapper
from . import branch_utils as bu

compartment_roots_stat_max = update_wrapper(
    partial(compartment_roots_stat_extrema,extrema = "max"),
    compartment_roots_stat_extrema,
)

compartment_roots_stat_min = update_wrapper(
    partial(compartment_roots_stat_extrema,extrema = "min"),
    compartment_roots_stat_extrema,
)

compartment_root_skeleton_angle_max = update_wrapper(
    partial(compartment_roots_stat_max,
            stat_func = bu.skeleton_angle_from_top,),
    compartment_roots_stat_max
)

compartment_root_skeleton_angle_min = update_wrapper(
    partial(compartment_roots_stat_min,
            stat_func = bu.skeleton_angle_from_top,),
    compartment_roots_stat_min
)

compartment_root_width_max = update_wrapper(
    partial(compartment_roots_stat_max,
            stat_func = "width_upstream"),
    compartment_roots_stat_max
    
)

compartment_root_width_min = update_wrapper(
    partial(compartment_roots_stat_min,
            stat_func = "width_upstream",),
    compartment_roots_stat_min
)
    

# ------------- parameters for stats ---------------



global_parameters_dict_default = dict(
    skeletal_length_max_n_spines = 3000,
)

attributes_dict_default = dict(
    voxel_to_nm_scaling = mvu.voxel_to_nm_scaling
)    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01 = dict(
    skeletal_length_max_n_spines = 6_000
)

attributes_dict_h01 = dict(
    voxel_to_nm_scaling = hvu.voxel_to_nm_scaling
)

# data_type = "default"
# algorithms = None
# modules_to_set = [nru]

# modsetter = modu.ModuleDataTypeSetter(
#     module = modules_to_set,
#     algorithms = algorithms
# )

# set_global_parameters_and_attributes_by_data_type = modsetter.set_global_parameters_and_attributes_by_data_type
# set_global_parameters_and_attributes_by_data_type(data_type=data_type,
#                                                    algorithms=algorithms)

# output_global_parameters_and_attributes_from_current_data_type = modsetter.output_global_parameters_and_attributes_from_current_data_type


#--- from neurd_packages ---

from . import classification_utils as clu
from . import concept_network_utils as cnu
from . import error_detection as ed
from . import h01_volume_utils as hvu
from . import microns_volume_utils as mru
from . import microns_volume_utils as mvu
from . import neuron 
from . import neuron
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_visualizations as nviz
from . import preprocess_neuron as pre
from . import proofreading_utils as pru
from . import soma_extraction_utils as sm
from . import synapse_utils as syu
from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import compartment_utils as cu
from mesh_tools import meshparty_skeletonize as m_sk
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm
from datasci_tools import pipeline
from datasci_tools import mesh_utils as meshu

from . import neuron_utils as nru