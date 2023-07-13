
import time
from python_tools import numpy_dep as np

default_skeleton_segment_size = 1000

def skeleton_resized_ordered(
    skeleton,
    skeleton_segment_size=default_skeleton_segment_size,
    width_segment_size=None,
    return_skeletal_points = False,
    verbose = False):
    
    #resizes the branch to the desired width (HELPS WITH SMOOTHING OF THE SKELETON)
    ex_branch_skeleton_resized = sk.resize_skeleton_branch(skeleton,segment_width = skeleton_segment_size)

    #The size we want the widths to be calculated at
    if not width_segment_size is None:
        ex_branch_skeleton_resized = sk.resize_skeleton_branch(ex_branch_skeleton_resized,segment_width = width_segment_size)
        
    ex_branch_skeleton_resized = sk.order_skeleton(ex_branch_skeleton_resized)
    
    if verbose:
        print(f"Starting node = {ex_branch_skeleton_resized[0][0]}")
        print(f"# of segments = {len(ex_branch_skeleton_resized[:,0])}")
        
    if return_skeletal_points:
        return sk.skeleton_coordinate_path_from_start(ex_branch_skeleton_resized)
    return ex_branch_skeleton_resized

def calculate_new_width(branch, 
                               skeleton_segment_size=1000,
                                   width_segment_size = None,
                              return_average=False,
                                   distance_by_mesh_center=True,
                              no_spines=True,
                              summary_measure="mean",
                                no_boutons=False,
                              print_flag=False,
                        distance_threshold_as_branch_width = False,
                        distance_threshold = 3000,
                               old_width_calculation=None):
    """
    Purpose: To calculate the overall width 

    Ex: 
    curr_branch_obj = neuron_obj_exc_syn_sp[4][30]

    skeleton_segment_size = 1000
    width_segment_size=None
    width_name = "no_spine_average"
    distance_by_mesh_center= True

    no_spines = True
    summary_measure = "mean"
    current_width_array,current_width = wu.calculate_new_width(curr_branch_obj, 
                                              skeleton_segment_size=skeleton_segment_size,
                                              width_segment_size=width_segment_size, 
                                              distance_by_mesh_center=distance_by_mesh_center,
                                              no_spines=no_spines,
                                              summary_measure=summary_measure,
                                              return_average=True,
                                              print_flag=True,
                                            )
    """
    
    f = getattr(np,summary_measure)
    
    #ges branch mesh without any spines
    if no_spines:
        ex_branch_no_spines_mesh = nru.branch_mesh_no_spines(branch)
    elif no_boutons:
        if hasattr(branch,"boutons"):
            ex_branch_no_spines_mesh = nru.mesh_without_boutons(branch)
        else:
            ex_branch_no_spines_mesh = branch.mesh
    else:
        ex_branch_no_spines_mesh = branch.mesh
        
    if distance_threshold_as_branch_width:
        distance_threshold = branch_width
    else:
        distance_threshold = distance_threshold
        
    
    ex_branch_skeleton_resized = wu.skeleton_resized_ordered(
        skeleton=branch.skeleton,
        skeleton_segment_size=skeleton_segment_size,
        width_segment_size=width_segment_size)

    (total_distances,
     total_distances_std,
     new_submesh,
     unique_faces) = cu.get_skeletal_distance_no_skipping(main_mesh=ex_branch_no_spines_mesh,
                                    edges=ex_branch_skeleton_resized,
                                     buffer=0.01,
                                    bbox_ratio=1.2,
                                   distance_threshold=distance_threshold,
                                    distance_by_mesh_center=distance_by_mesh_center,
                                    print_flag=False,
                                    edge_loop_print=False
                                                         )
    
    total_distances = np.array(total_distances)
    
    if old_width_calculation is None:
        old_width_calculation = branch.width

#     if print_flag: 
#         print(f"total_distances before replacement= {total_distances}")
        
    if len(total_distances[total_distances > 0]) > 0:
        total_distances[total_distances <= 0] = f(total_distances[total_distances > 0])
        
#     if print_flag: 
#         print(f"total_distances After replacement= {total_distances}")
        
    
    branch_width_average = f(total_distances)
    if branch_width_average < 0.0001:
        #just assing the old width
        print("Assigning the old width calculation because no valid new widths")
        branch_width_average = old_width_calculation
        total_distances = np.ones(len(ex_branch_skeleton_resized))*branch_width_average
    else:
        total_distances[total_distances == 0] = branch_width_average #IF RETURNED 0 THEN FILL with 

        if print_flag:
            print(f"Overall {summary_measure} = {branch_width_average}")
            print(f"Total_distances = {total_distances}")

    if return_average:
        return total_distances,branch_width_average
    else:
        return total_distances
        
        
def find_mesh_width_array_border(curr_limb,
                             node_1,
                             node_2,
                            width_name = "no_spine_median_mesh_center",
                            segment_start = 1,
                            segment_end = 4,
                            skeleton_segment_size = None,
                            width_segment_size = None,
                            recalculate_width_array = False, #will automatically recalculate the width array
                            default_segment_size = 1000,
                                 no_spines=True,
                                 summary_measure="mean",
                            print_flag=True,
                            **kwargs
                            ):

    """
    Purpose: To send back an array that 
    represents the widths of curent branches
    at their boundary
    - the widths may be calculated differently than currently
      stored if specified so

    Applications: 
    1) Will help with filtering out false positives
    with the axon detection
    2) For merge detections to help detect
    large width change

    Process: 
    0) make sure the two nodes are connected in the concept network
    1) if the skeleton_segment_size and width_semgent is None then recalculate the width array
    - send the 
    2) calculate the endpoints from the skeletons (to ensure they are in the right order)
    3) find the connectivity of the endpoints
    4) Get the subarrays of the width_arrays according to the start and end specified
    5) return the subarrays

    Example of Use: 
    find_mesh_width_array_border(curr_limb=curr_limb_obj,
                             #node_1=56,
                             #node_2=71,
                             node_1 = 8,
                             node_2 = 5,
                            width_name = "no_spine_average_mesh_center",
                            segment_start = 1,
                            segment_end = 4,
                            skeleton_segment_size = 50,
                            width_segment_size = None,
                            recalculate_width_array = True, #will automatically recalculate the width array
                            default_segment_size = 1000,
                            print_flag=True
                            )

    """

    # 0) make sure the two nodes are connected in the concept network
    if node_2 not in xu.get_neighbors(curr_limb.concept_network,node_1):
        raise Exception(f"Node_1 ({node_1}) and Node_2 ({node_2}) are not connected in the concept network")


    # 0) extract the branch objects
    branch_obj_1 = curr_limb.concept_network.nodes[node_1]["data"]
    branch_obj_2 = curr_limb.concept_network.nodes[node_2]["data"]
    
    branch_obj_1.order_skeleton_by_smallest_endpoint()
    branch_obj_2.order_skeleton_by_smallest_endpoint()
    # 1) if the skeleton_segment_size and width_semgent is then recalculate the width array
    if not skeleton_segment_size is None or recalculate_width_array:

        if "mesh_center" in width_name:
            distance_by_mesh_center = True
        else:
            distance_by_mesh_center = False
            
        if ("no_spine" in width_name) or (no_spines):
            no_spines = True
        else:
            if print_flag:
                print("Using no spines")
            
        if print_flag:
            print(f"distance_by_mesh_center = {distance_by_mesh_center}")

        if skeleton_segment_size is None:
            skeleton_segment_size = default_segment_size

        if not nu.is_array_like(skeleton_segment_size):
            skeleton_segment_size = [skeleton_segment_size]

        if width_segment_size is None:
            width_segment_size = skeleton_segment_size

        if not nu.is_array_like(width_segment_size):
            width_segment_size = [width_segment_size]


        current_width_array_1,current_width_1 = calculate_new_width(branch_obj_1, 
                                          skeleton_segment_size=skeleton_segment_size[0],
                                          width_segment_size=width_segment_size[0], 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                          return_average=True,
                                          print_flag=False,
                                        no_spines=no_spines,
                                                                   summary_measure=summary_measure)

        current_width_array_2,current_width_2 = calculate_new_width(branch_obj_2, 
                                          skeleton_segment_size=skeleton_segment_size[-1],
                                          width_segment_size=width_segment_size[-1], 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                            no_spines=no_spines,
                                          return_average=True,
                                          print_flag=False,
                                            summary_measure=summary_measure)
    else:
        if print_flag:
            print("**Using the default width arrays already stored**")
        current_width_array_1 = branch_obj_1.width_array[width_name]
        current_width_array_2 = branch_obj_2.width_array[width_name]

    if print_flag:
        print(f"skeleton_segment_size = {skeleton_segment_size}")
        print(f"width_segment_size = {width_segment_size}")
        print(f"current_width_array_1 = {current_width_array_1}")
        print(f"current_width_array_2 = {current_width_array_2}")
    
    
    
    
    

    #2) calculate the endpoints from the skeletons (to ensure they are in the right order)
    end_1 = sk.find_branch_endpoints(branch_obj_1.skeleton)
    end_2 = sk.find_branch_endpoints(branch_obj_2.skeleton)
    
    if print_flag:
        print(f"end_1 = {end_1}")
        print(f"end_2 = {end_2}")
    

    #3) find the connectivity of the endpoints
    node_connectivity = xu.endpoint_connectivity(end_1,end_2)

    #4) Get the subarrays of the width_arrays according to the start and end specified
    """
    Pseudocode: 

    What to do if too small? Take whole thing

    """
    if print_flag:
        print(f"node_connectivity = {node_connectivity}")
    
    return_arrays = []
    width_arrays = [current_width_array_1,current_width_array_2]

    for j,current_width_array in enumerate(width_arrays):

        if len(current_width_array)<segment_end:
            if print_flag:
                print(f"The number of segments for current_width_array_{j+1} ({len(current_width_array)}) "
                     " was smaller than the number requested, so just returning the whole width array")

            return_arrays.append(current_width_array)
        else:
            if node_connectivity[j] == 0:
                return_arrays.append(current_width_array[segment_start:segment_end])
            elif node_connectivity[j] == 1:
                return_arrays.append(current_width_array[-segment_end:-segment_start])
            else:
                raise Exception("Node connectivity was not 0 or 1")

    return return_arrays


def new_width_from_mesh_skeleton(skeleton,
                                mesh,
                                skeleton_segment_size=1000,
                                width_segment_size = None,
                                return_average=True,
                                distance_by_mesh_center=True,
                            distance_threshold_as_branch_width = False,
                            distance_threshold = 3000,
                              summary_measure="median",
                              verbose=False,
                               backup_width=None):
    """
    Purpose: To calculate the new width from a
    skeleton and the surounding mesh
    """
    print_flag = verbose
    f = getattr(np,summary_measure)
    
    ex_branch_no_spines_mesh = mesh
    
    #resizes the branch to the desired width (HELPS WITH SMOOTHING OF THE SKELETON)
    
    ex_branch_skeleton_resized = sk.resize_skeleton_with_branching(skeleton,segment_width = skeleton_segment_size)


    #The size we want the widths to be calculated at
    if not width_segment_size is None:
        ex_branch_skeleton_resized = sk.resize_skeleton_with_branching(ex_branch_skeleton_resized,segment_width = width_segment_size)
    
    if distance_threshold_as_branch_width:
        distance_threshold = branch_width
    else:
        distance_threshold = distance_threshold
    
    (total_distances,
     total_distances_std,
     new_submesh,
     unique_faces) = cu.get_skeletal_distance_no_skipping(main_mesh=ex_branch_no_spines_mesh,
                                    edges=ex_branch_skeleton_resized,
                                     buffer=0.01,
                                    bbox_ratio=1.2,
                                   distance_threshold=distance_threshold,
                                    distance_by_mesh_center=distance_by_mesh_center,
                                    print_flag=False,
                                    edge_loop_print=False
                                                         )
    
    total_distances = np.array(total_distances)
    #print(f"total_distances = {total_distances}")
    
    old_width_calculation = backup_width
    
    branch_width_average = f(total_distances)
    if branch_width_average < 0.0001:
        #just assing the old width
        if print_flag:
            print("Assigning the old width calculation because no valid new widths")
        branch_width_average = old_width_calculation
        total_distances = np.ones(len(ex_branch_skeleton_resized))*branch_width_average
    else:
        total_distances[total_distances == 0] = branch_width_average #IF RETURNED 0 THEN FILL with 

        if print_flag:
            print(f"Overall {summary_measure} = {branch_width_average}")
            print(f"Total_distances = {total_distances}")

    if return_average:
        return branch_width_average
    else:
        return total_distances
    
    
def calculate_new_width_for_neuron_obj(neuron_obj,
                          skeleton_segment_size = 1000,
                           width_segment_size=None,
                          width_name = None,
                            distance_by_mesh_center=True,
                           no_spines=True,
                            summary_measure="mean",
                            limb_branch_dict = None,
                            verbose = True,
                            skip_no_spine_width_if_no_spine = True,
                          **kwargs):
        """
        Purpose: To calculate new width definitions based on if
        1) Want to use skeleton center or mesh center
        2) Want to include spines or not
        
        Examples:
        current_neuron.calculate_new_width(no_spines=False,
                                               distance_by_mesh_center=True)
                                               
        current_neuron.calculate_new_width(no_spines=False,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
                                       
        current_neuron.calculate_new_width(no_spines=True,
                                       distance_by_mesh_center=True,
                                       summary_measure="mean")
        
        current_neuron.calculate_new_width(no_spines=True,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
        
        """
        if verbose:
            print(f"width_name BEFORE processing = {width_name}")
        
        
        if width_name is None:
            width_name = str(summary_measure)
        else:
            if "mesh_center" in width_name:
                distance_by_mesh_center = True
            else:
                distance_by_mesh_center = False

            if "no_spine" in width_name:
                no_spines=True
            else:
                no_spines=False
            
            if "mean" in width_name:
                summary_measure = "mean"
            elif "median" in width_name:
                summary_measure = "median"
            else: 
                raise Exception("No summary statistic was specified in the name")
        
        if summary_measure != "mean":
            width_name = width_name.replace("mean",summary_measure)
            if summary_measure not in width_name:
                width_name = f"{width_name}_{summary_measure}"
                
        if ("no_spine" not in width_name) and (no_spines):
            width_name = f"no_spine_{width_name}"
        if ("mesh_center" not in width_name) and (distance_by_mesh_center):
            width_name = f"{width_name}_mesh_center"
        
        if verbose:
            print(f"After processing")
            print(f"width_name = {width_name}, distance_by_mesh_center= {distance_by_mesh_center}, no_spines = {no_spines}, summary_measure= {summary_measure}")
        
        for limb_idx in neuron_obj.get_limb_node_names():
            
            if limb_branch_dict is not None:
                if limb_idx not in limb_branch_dict.keys():
                    continue
                    
            for branch_idx in neuron_obj.get_branch_node_names(limb_idx):
                
                if limb_branch_dict is not None:
                    if branch_idx not in limb_branch_dict[limb_idx]:
                        continue
                
                if verbose:
                    print(f"Working on limb {limb_idx} branch {branch_idx}")
                curr_branch_obj = neuron_obj[limb_idx][branch_idx]
                

                #Add rule that will help skip segment if has no spines
                already_computed = False
                
                if skip_no_spine_width_if_no_spine:
                    if ((curr_branch_obj.spines is None or len(curr_branch_obj.spines) == 0) and no_spines) and "no_spine" in width_name:

                        #see if we can skip
                        new_width_name = width_name.replace("no_spine_","")
                        if new_width_name in curr_branch_obj.width_new.keys():

                            curr_branch_obj.width_new[width_name] = curr_branch_obj.width_new[new_width_name]
                            curr_branch_obj.width_array[width_name] = curr_branch_obj.width_array[new_width_name]
                            

                            if verbose:
                                print(f"    No spines and using precomputed width: {curr_branch_obj.width_new[new_width_name]}")

                            already_computed=True
                
                if not already_computed:
                    current_width_array,current_width = wu.calculate_new_width(curr_branch_obj, 
                                          skeleton_segment_size=skeleton_segment_size,
                                          width_segment_size=width_segment_size, 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                          no_spines=no_spines,
                                          summary_measure=summary_measure,
                                          return_average=True,
                                          print_flag=False,
                                        **kwargs)
                        
                    if verbose:
                        print(f"    current_width= {current_width}")

                    curr_branch_obj.width_new[width_name] = current_width
                    curr_branch_obj.width_array[width_name] = current_width_array
                
                curr_branch_obj.width_array_skeletal_lengths = None
                    
                    

def neuron_width_calculation_standard(
    neuron_obj,
    widths_to_calculate = ("median_mesh_center",
                           "no_spine_median_mesh_center"),
    verbose = True,
    limb_branch_dict=None,
    **kwargs):
    for w in widths_to_calculate:
        st = time.time()
        if verbose:
            print(f"\n\n----Working on width: {w}-----")
        wu.calculate_new_width_for_neuron_obj(
            neuron_obj,
            width_name=w,
            verbose = verbose,
            limb_branch_dict=limb_branch_dict,
            **kwargs)
        if verbose:
            print(f"Time for calculating {w}: {time.time() - st}")
            
    return neuron_obj


#--- from neurd_packages ---
from . import neuron_utils as nru

#--- from mesh_tools ---
from mesh_tools import compartment_utils as cu
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from python_tools ---
from python_tools import networkx_utils as xu
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu

from . import width_utils as wu