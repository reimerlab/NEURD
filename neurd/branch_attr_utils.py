'''



Purpose: For help with manipulating and 
calculating qualities of objects stored on branches
- spines
- synapses




'''
import time
from python_tools import numpy_dep as np

def calculate_endpoints_dist(
    branch_obj,
    attr_obj):
    """
    Purpose: Will calculate the endpoint distance for a attribute
    """
    attr_obj.endpoints_dist = [sk.skeleton_path_between_skeleton_coordinates(
                            starting_coordinate = attr_obj.closest_sk_coordinate,
                            destination_node=j,
                            skeleton_graph = branch_obj.skeleton_graph,
                            only_skeleton_distance = True,) for j in branch_obj.endpoints_nodes]
            
def calculate_upstream_downstream_dist(limb_obj,
                                       branch_idx,
                                       attr_obj):
    bu.set_endpoints_upstream_downstream_idx_on_branch(limb_obj,branch_idx)
    down_idx = limb_obj[branch_idx].endpoints_upstream_downstream_idx[1]
    attr_obj.downstream_dist = attr_obj.endpoints_dist[down_idx]
    attr_obj.upstream_dist = attr_obj.endpoints_dist[1-down_idx]
    
def calculate_upstream_downstream_dist_from_down_idx(
    attr_obj,
    down_idx):
    attr_obj.downstream_dist = attr_obj.endpoints_dist[down_idx]
    attr_obj.upstream_dist = attr_obj.endpoints_dist[1-down_idx]
    
def calculate_upstream_downstream_dist_from_up_idx(
    attr_obj,up_idx):
    bau.calculate_upstream_downstream_dist_from_down_idx(attr_obj,1-up_idx)

    
def calculate_branch_attr_soma_distances_on_limb(
    limb_obj,
    branch_attr,
    calculate_endpoints_dist_if_empty=True,
    verbose=False):
    """
    Purpose: To store the distances to the soma 
    for all of the synapses

    Computing the upstream soma distance
    for each branch
    1) calculate the upstream distance
    2) Calcualte the upstream endpoint
        For each synapse:
        3) Soma distance = endpoint_dist
        
    Ex: 
    calculate_limb_synapse_soma_distances(limb_obj = neuron_obj[2],
    verbose = True)

    """
    bu.set_branches_endpoints_upstream_downstream_idx_on_limb(limb_obj)

    for branch_idx in limb_obj.get_branch_names():
        branch_obj = limb_obj[branch_idx]

        #1) Calculate the upstream distance
        upstream_dist = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        #upstream_endpoint_idx = nru.upstream_endpoint(limb_obj,branch_idx,return_endpoint_index=True)
        upstream_endpoint_idx = branch_obj.endpoints_upstream_downstream_idx[0]

        curr_attr_list = getattr(branch_obj,branch_attr)
        if curr_attr_list is not None:
            for attr_obj in curr_attr_list:
                if attr_obj.endpoints_dist is None:
                    recompute_dists = True
                elif attr_obj.endpoints_dist[upstream_endpoint_idx] == -1:
                    recompute_dists = True
                else:
                    recompute_dists = False
                if recompute_dists:
                    if calculate_endpoints_dist_if_empty:
                        bau.calculate_endpoints_dist(branch_obj,attr_obj)
                        bau.calculate_upstream_downstream_dist(limb_obj,branch_idx,attr_obj)
                        #endpoint_dist = attr_obj.endpoints_dist[upstream_endpoint_idx]
                        #endpoint_dist = attr_obj.upstream_dist
                    else:
                        raise Exception("Endpoint distance was not calculated yet and calculate_endpoint_dist_if_empty not set")
                #endpoint_dist = attr_obj.endpoints_dist[upstream_endpoint_idx]
                bau.calculate_upstream_downstream_dist_from_up_idx(attr_obj,upstream_endpoint_idx)
                
                
                attr_obj.soma_distance = attr_obj.upstream_dist + upstream_dist
    return limb_obj
            
def calculate_neuron_soma_distance(
    neuron_obj,
    branch_attr,
    verbose=False,
    **kwargs):
    """
    Purpose: To calculate all of the soma distances for all objects on 
    branches in a neuron
    
    Ex: 
    calculate_neuron_soma_distance(neuron_obj,
                              verbose = True)
    """
    st = time.time()
    for limb_name in neuron_obj.get_limb_names():
        st_loc = time.time()
        
        limb_obj = neuron_obj[limb_name]
        bau.calculate_branch_attr_soma_distances_on_limb(
            limb_obj = limb_obj,
            branch_attr = branch_attr,
            verbose = False)
        if verbose:
            print(f"\n--- Limb {limb_name} soma calculation time = {np.round(time.time() - st_loc,3)}")
    
        
    if verbose:
        print(f"Total soma distance calculation time = {time.time() - st}")
        
def calculate_neuron_soma_distance_euclidean(
    neuron_obj,
    branch_attr,
    verbose=False,
    ):
    """
    Purpose: To calculate all of the soma distances for all the valid synapses
    on limbs
    
    Ex: 
    calculate_neuron_soma_distance(neuron_obj,
                              verbose = True)
    """
    st = time.time()
    soma_center = neuron_obj["S0"].mesh_center
    
    for attr_obj in getattr(neuron_obj,branch_attr):
        attr_obj.soma_distance_euclidean = np.linalg.norm(soma_center - attr_obj.coordinate)
                    
    if verbose:
        print(f"Total soma distance calculation time = {time.time() - st}")
        
def set_limb_branch_idx_to_attr(
    neuron_obj,
    branch_attr):
    """
    Purpose: Will add limb and branch indexes for
    all synapses in a Neuron object
    """
    for limb_idx in neuron_obj.get_limb_names(return_int=True):
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            branch_obj = limb_obj[branch_idx]
            attr_list = getattr(branch_obj,branch_attr)
            if attr_list is not None:
                for s in attr_list:
                    s.limb_idx = limb_idx
                    s.branch_idx = branch_idx


#--- from neurd_packages ---
from . import branch_utils as bu
from . import neuron_statistics as nst
from . import neuron_utils as nru

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk

#--- from python_tools ---
from python_tools import numpy_dep as np

from . import branch_attr_utils as bau