
import copy
import networkx as nx
import time
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from datasci_tools import data_struct_utils as dsu
from .h01_volume_utils import data_interface as hvu

axon_version = 7 #version that finds the axon using bare synapse branches
axon_version = 8 #improved synapse filtering for axon
axon_version = 9 #Using Baylor cell types for axon finding
axon_thick_threshold = 100
axon_ais_threshold = 180

# excitatory_axon_ais_max_search_distance = 14_000
# inhibitory_axon_ais_max_search_distance = 70_000

# excitatory_axon_soma_angle_threshold = 70


def axon_width(branch_obj,
              width_name="no_bouton_median",
               width_name_backup="no_spine_median_mesh_center",
               width_name_backup_2 = "median_mesh_center",):
    """
    Computes the widht of the branch (specifically in the axon case)
    
    Ex: 
    branch_obj = neuron_obj["L6"][0]
    au.axon_width(branch_obj)

    nviz.visualize_neuron(neuron_obj,
                         limb_branch_dict=dict(L6=[0]),
                         mesh_color="red",
                         mesh_whole_neuron=True)
    
    """
    curr_branch = branch_obj
    try:
        width_jump = curr_branch.width_new[width_name]
    except:
        try:
            width_jump = curr_branch.width_new[width_name_backup]
        except:
            width_jump = curr_branch.width_new[width_name_backup_2] 
    return width_jump




def filter_axon_limb_false_positive_end_nodes(curr_limb,curr_limb_axon_like_nodes,verbose=False,skeleton_length_threshold=30000):
    """
    Purpose: Will remove end nodes that were accidentally mistaken as axons
    
    """
    final_axon_like_nodes = np.array(copy.copy(curr_limb_axon_like_nodes))

    if len(curr_limb_axon_like_nodes)>0:
        #2) Filter for only those that are end nodes
        axon_node_degrees = np.array(xu.get_node_degree(curr_limb.concept_network,curr_limb_axon_like_nodes))
        end_node_idx = np.where(axon_node_degrees == 1)[0]

        if len(end_node_idx) == 0:
            pass
        else:
            nodes_to_check = curr_limb_axon_like_nodes[end_node_idx]
            for n_name in nodes_to_check:
                curr_sk_length = sk.calculate_skeleton_distance(curr_limb[n_name].skeleton) 
                if curr_sk_length > skeleton_length_threshold:
                    if verbose:
                        print(f"Skipping because skeleton too long: {curr_sk_length}")
                    continue
                
                curr_neighbors = xu.get_neighbors(curr_limb.concept_network,n_name)
                if verbose:
                    print(f"curr_neighbors = {curr_neighbors}")
                if len(curr_neighbors) == 0:
                    if verbose:
                        print("skipping because no neighbors")
                    pass
                elif curr_neighbors[0] in curr_limb_axon_like_nodes:
                    if verbose:
                        print("skipping because neighbor axon")
                    pass
                else:
                    if verbose:
                        print(f"Skipping end node {n_name} because neighbor was dendrite")
                    final_axon_like_nodes = final_axon_like_nodes[final_axon_like_nodes != n_name]
    return final_axon_like_nodes



def filter_axon_neuron_false_positive_end_nodes(neuron_obj,current_axons_dict):
    filtered_axon_dict = dict()
    for limb_name_key,curr_limb_axon_like_nodes in current_axons_dict.items():

        curr_limb_idx = int(limb_name_key[1:])

        curr_limb = neuron_obj[curr_limb_idx]
        filtered_axon_dict[limb_name_key] = filter_axon_limb_false_positive_end_nodes(curr_limb,curr_limb_axon_like_nodes)
    return filtered_axon_dict

def axon_like_segments(neuron_obj,include_ais=False,filter_away_end_false_positives=True,visualize_at_end=False,width_to_use=None,
                       verbose=False):
    current_neuron = neuron_obj
    axon_like_limb_branch_dict = ns.axon_width_like_segments(current_neuron,
                                                        include_ais=include_ais,
                                                             width_to_use=width_to_use,
                                                            verbose=verbose)
    
    
    
    current_functions_list = ["axon_segment"]
    limb_branch_dict_upstream_filter = ns.query_neuron(current_neuron,
                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=3000,
                                                            width_match_threshold=50,
                                                           print_flag=False),
                                       functions_list=current_functions_list)
    
    if filter_away_end_false_positives:
        if verbose:
            print("Using filter_away_end_false_positives")
        limb_branch_dict_upstream_filter = filter_axon_neuron_false_positive_end_nodes(neuron_obj,
                                                                                      limb_branch_dict_upstream_filter)
        
    if visualize_at_end:
        colors_dict_returned = nviz.visualize_neuron(current_neuron,
                                             visualize_type=["mesh"],
                      limb_branch_dict=limb_branch_dict_upstream_filter,
                     mesh_color="red",
                     mesh_color_alpha=1,
                     mesh_whole_neuron=True,
                     return_color_dict=True)
    
    return limb_branch_dict_upstream_filter




def bouton_meshes(mesh,
                 clusters=5,#4,
                 smoothness=0.1,
                  plot_segmentation=False,
                 filter_away_end_meshes=True,
                 cdf_threshold = 0.20,#0.30,# 0.35,
                  plot_boutons=False,
                  verbose=False,
                  skeleton=None,
                  min_size_threshold=None,#50,#None,
                  max_size_threshold = None,#600,350,#185,#None,
                  size_type="faces",
                  plot_boutons_after_size_threshold = False,
                  ray_trace_filter = "ray_trace_percentile", #"ray_trace_median"
                  ray_trace_percentile = 70,
                  ray_trace_threshold = None,#270,
                  return_non_boutons = False,
                  exclude_end_meshes_from_non_boutons = True,
                  return_cdf_widths = False,
                  end_mesh_method = "endpoint_radius",
                  endpoint_radius_threshold = 1000,
                  skeletal_length_max = 2200,
                 ):
    
    if min_size_threshold is None:
        min_size_threshold = min_size_threshold_bouton_global
    if max_size_threshold is None:
        max_size_threshold = max_size_threshold_bouton_global
    if ray_trace_threshold is None:
        ray_trace_threshold = ray_trace_threshold_bouton_global
    """
    Purpose: To find the bouton and non-bouton mesh pieces
    
    
    from mesh_tools import trimesh_utils as tu
    b_meshes,non_b_meshes = au.bouton_meshes(branch_obj,
                     clusters=4,
                     smoothness=0.1,
                      plot_segmentation=True,
                     filter_away_end_meshes=True,
                     cdf_threshold = 0.35,
                     verbose=True,
                     min_size_threshold =50,
                     max_size_threshold=200,
                     size_type = "faces",
                    return_non_boutons = True,
                    ray_trace_filter = "ray_trace_percentile",              
                    ray_trace_percentile = 70,
                      ray_trace_threshold = 270,
                     )
    b_meshes,non_b_meshes
    
    ** Note: Tried using side_length_ratios and volume_ratios as other features
    to filter on but did not seem immediately useful **
    """
    from neurd import neuron
    
    if type(mesh) == neuron.Branch:
        if skeleton is None:
            skeleton = mesh.skeleton
            
        mesh = mesh.mesh
        
        
    
    new_meshes, cgal_info = tu.mesh_segmentation(mesh,
                    clusters=clusters,
                    smoothness=smoothness)
    
    new_meshes = np.array(new_meshes)
    cgal_info = np.array(cgal_info)
    
    if plot_segmentation:
        print(f"Segmentation Info:")
        for j,(m,c) in enumerate(zip(new_meshes,cgal_info)):
            print(f"Mesh {j}: {m} ({c})")
        nviz.plot_objects(meshes=new_meshes,
                         meshes_colors="random")
    
        

    boutons_idx = np.where(cgal_info>cdf_threshold)[0]
    bouton_meshes = new_meshes[boutons_idx]
    bouton_meshes_cdf = cgal_info[boutons_idx]
    
    
    
    if verbose:
        print("\n After CDF Trheshold:")
        print(f"# of bouton meshes = {len(boutons_idx)}:")
        print(f"boutons_idx = {boutons_idx}")
        print(f"bouton_meshes = {bouton_meshes}")
        print(f"bouton_meshes_cdf = {bouton_meshes_cdf}")
        

        
    #calculating the indexes of the end meshes
    if exclude_end_meshes_from_non_boutons or filter_away_end_meshes:
        
        from mesh_tools import skeleton_utils as sk
        end_meshes_idx = []
        for end_coordinate in sk.find_branch_endpoints(skeleton):
            
            if end_mesh_method == "closest_mesh":
                if verbose:
                    print(f"\nUsing closest_mesh method for filter_away_end_meshes")
                    
                closest_index = tu.closest_mesh_to_coordinate(new_meshes,end_coordinate,return_mesh=False,
                                                             distance_method="min_distance_and_bbox_center")
                end_meshes_idx.append(closest_index)
                
            elif end_mesh_method == "endpoint_radius":
                
                if verbose:
                    print(f"\nUsing endpoint_radius with radius {endpoint_radius_threshold} method for filter_away_end_meshes")
                
                if endpoint_radius_threshold is None:
                    raise Exception(f"end_mesh_method = {end_mesh_method} but endpoint_radius_threshold is None")
                mesh_idx_within_raidus = tu.filter_meshes_by_containing_coordinates(new_meshes,
                                                                          nullifying_points=[end_coordinate],
                                                filter_away=False,
                                           method="distance",
                                           distance_threshold=endpoint_radius_threshold,
                                           verbose=False,
                                           return_indices=True)
                end_meshes_idx += list(mesh_idx_within_raidus)
            else:
                raise Exception(f"Unimplemented type: {end_mesh_method}")
                
        
    if filter_away_end_meshes:
        if len(bouton_meshes)>0:
            if verbose:
                print(f"end_meshes_idx = {end_meshes_idx}")

            boutons_idx = np.setdiff1d(boutons_idx,end_meshes_idx)
            bouton_meshes = new_meshes[boutons_idx]
            bouton_meshes_cdf = cgal_info[boutons_idx]
            
            if verbose:
                print(f"\n--After Filtering End Meshes--")
                print(f"# of bouton meshes = {len(boutons_idx)}:")
                print(f"bouton_meshes = {bouton_meshes}")
                print(f"bouton_meshes_cdf = {bouton_meshes_cdf}")
        else:
            if verbose:
                print(f"No boutons so not attmepting to filter away")
                
    original_bouton_idx = boutons_idx.copy()
    
    if min_size_threshold is not None or max_size_threshold is not None:
        if verbose:
            print(f"Applying size threshold of {min_size_threshold}< {size_type} <= {max_size_threshold}")
        """
        if min_size_threshold is not None:
            boutons_idx_min = tu.filter_meshes_by_size(mesh_list=bouton_meshes,
                              size_threshold=min_size_threshold,
                             size_type=size_type,
                              above_threshold=True,
                              return_indices = True,
                             verbose=False)
            if verbose:
                print(f"boutons_idx_min = {boutons_idx_min}")
        else:
            boutons_idx_min = np.arange(len(bouton_meshes))
        
        if min_size_threshold is not None:
            boutons_idx_max = tu.filter_meshes_by_size(mesh_list=bouton_meshes,
                              size_threshold=max_size_threshold,
                             size_type=size_type,
                              above_threshold=False,
                              return_indices = True,
                             verbose=False)
            if verbose:
                print(f"boutons_idx_max = {boutons_idx_max}")
        else:
            boutons_idx_max = np.arange(len(bouton_meshes))
            
        boutons_idx = np.intersect1d(boutons_idx_min,boutons_idx_max)
        """
        boutons_idx = tu.filter_meshes_by_size_min_max(mesh_list=bouton_meshes,
                              min_size_threshold=min_size_threshold,
                            max_size_threshold=max_size_threshold,
                             size_type=size_type,
                              return_indices = True,
                             verbose=False)
        
        if verbose:
            print(f"boutons_idx = {boutons_idx}")
        
        bouton_meshes = bouton_meshes[boutons_idx]
        bouton_meshes_cdf = bouton_meshes_cdf[boutons_idx]
        original_bouton_idx = original_bouton_idx[boutons_idx]
        
        if verbose:
                print(f"\n--After Filtering for size --")
                print(f"# of bouton meshes = {len(boutons_idx)}:")
                print(f"bouton_meshes = {bouton_meshes}")
                print(f"bouton_meshes_cdf = {bouton_meshes_cdf}")
                print(f"original_bouton_idx = {original_bouton_idx}")
                
        if plot_boutons_after_size_threshold:
            print(f"Boutons after size threshold of \n"
                  f"min_size_threshold={min_size_threshold}"
                 f"\nmax_size_threshold = {max_size_threshold}")
            nviz.plot_objects(mesh,
                             meshes=bouton_meshes,
                             meshes_colors="red")
                
                
        
    if ray_trace_filter is not None and ray_trace_threshold is not None:
        if verbose:
            print(f"\n--Applying ray trace threshold = {ray_trace_threshold}--")
            
        boutons_idx = tu.filter_meshes_by_size(mesh_list=bouton_meshes,
                              size_threshold=ray_trace_threshold,
                             size_type=ray_trace_filter,
                              above_threshold=True,
                              return_indices = True,
                             verbose=verbose,
                            percentile = ray_trace_percentile)
        if verbose:
            print(f"boutons_idx = {boutons_idx}")
        
        bouton_meshes = bouton_meshes[boutons_idx]
        bouton_meshes_cdf = bouton_meshes_cdf[boutons_idx]
        original_bouton_idx = original_bouton_idx[boutons_idx]
        
        if verbose:
                print(f"\n--After Filtering for ray trace distance --")
                print(f"# of bouton meshes = {len(boutons_idx)}:")
                print(f"bouton_meshes = {bouton_meshes}")
                print(f"bouton_meshes_cdf = {bouton_meshes_cdf}")
                print(f"original_bouton_idx = {original_bouton_idx}")
                
    
    
    if skeletal_length_max is not None:
        boutons_idx = tu.filter_meshes_by_size(bouton_meshes,
                        size_threshold = skeletal_length_max,
                        size_type="skeleton",
                        above_threshold=False,
                        return_indices=True)
        
        if verbose:
            print(f"boutons_idx = {boutons_idx}")
        
        bouton_meshes = bouton_meshes[boutons_idx]
        bouton_meshes_cdf = bouton_meshes_cdf[boutons_idx]
        original_bouton_idx = original_bouton_idx[boutons_idx]
        
        if verbose:
                print(f"\n--After Skeleton Filtering Under size {skeletal_length_max} nm --")
                print(f"# of bouton meshes = {len(boutons_idx)}:")
                print(f"bouton_meshes = {bouton_meshes}")
                print(f"bouton_meshes_cdf = {bouton_meshes_cdf}")
                print(f"original_bouton_idx = {original_bouton_idx}")
    
    
    
    if plot_boutons:
        if len(bouton_meshes) == 0:
            print(f"No boutons to plot")
        else:
            nviz.plot_objects(mesh,
                            meshes=bouton_meshes,
                          meshes_colors="red")
            
    if return_cdf_widths:
        new_meshes_widths = np.array([mesh_size(m,
                            size_type=ray_trace_filter) for m in new_meshes])
                            
        
        
    if return_non_boutons:
        non_bouton_idx = np.delete(np.arange(len(new_meshes)),original_bouton_idx)
        
        if exclude_end_meshes_from_non_boutons:
            non_bouton_idx = np.setdiff1d(non_bouton_idx,end_meshes_idx)
        
        non_bouton_meshes = new_meshes[non_bouton_idx]
        
        if verbose:
            print(f" # of non_bouton_meshes = {len(non_bouton_meshes)}")
            
        if return_cdf_widths:
            non_bouton_widths = new_meshes_widths[non_bouton_idx]
            boutons_widths = new_meshes_widths[original_bouton_idx]
            return bouton_meshes,non_bouton_meshes,boutons_widths,non_bouton_widths
        else:
            return bouton_meshes,non_bouton_meshes
    else:
        if return_cdf_widths:
            boutons_widths = new_meshes_widths[original_bouton_idx]
            return bouton_meshes,boutons_widths
        else:
            return bouton_meshes
        

def calculate_boutons_over_limb_branch_dict(neuron_obj,
                      limb_branch_dict,
                     width_name = "no_bouton_median",
                    old_width_name = "no_spine_median_mesh_center",
                        calculate_bouton_cdfs=True,
                        catch_bouton_errors = False,
                     verbose = False):
    """
    Psuedocode: Iterate through all of the branch objects and 
    compute the bouton meshes and store as boutons


    """
    
    for l_idx,branch_list in limb_branch_dict.items():
        for b in tqdm(branch_list):
            branch_obj = neuron_obj[l_idx][b]
            
            if catch_bouton_errors:
                try:
                    b_meshes= au.bouton_meshes(branch_obj,
                                        return_non_boutons=False,
                                             return_cdf_widths=False)


                except:
                    b_meshes = []
            
            else:
                b_meshes= au.bouton_meshes(branch_obj,
                                        return_non_boutons=False,
                                             return_cdf_widths=False)
            
            branch_obj.boutons = list(b_meshes)
            
            if calculate_bouton_cdfs:
                branch_obj.boutons_cdfs = [tu.mesh_size(k,
                size_type="ray_trace_percentile") for k in branch_obj.boutons]


            # ----- Doing the new width calculation -------- #

            skeleton_segment_size=1000
            width_segment_size = None
            distance_by_mesh_center = True
            no_spines = False
            summary_measure = "median"

            if len(b_meshes) > 0:
                if verbose:
                    print(f"Calculating new width because had {branch_obj.n_boutons} boutons")
                current_width_array,current_width = wu.calculate_new_width(branch_obj, 
                              skeleton_segment_size=skeleton_segment_size,
                              width_segment_size=width_segment_size, 
                              distance_by_mesh_center=distance_by_mesh_center,
                              no_spines=no_spines,
                              summary_measure=summary_measure,
                              return_average=True,
                              print_flag=False,
                            no_boutons=True,
                                old_width_calculation=branch_obj.width_new[old_width_name])

            else:
                current_width = branch_obj.width_new[old_width_name]
                current_width_array = branch_obj.width_array[old_width_name] 
            #calculating the mean/width width of boutons and non_boutons

            branch_obj.width_new[width_name]  = current_width
            branch_obj.width_array[width_name] = current_width_array
        
    return neuron_obj
    

def calculate_boutons(neuron_obj,
                     max_bouton_width_to_check = None,
                    plot_axon_branches_to_check = False,
                    width_name = "no_bouton_median",
                    old_width_name = "no_spine_median_mesh_center",
                    plot_boutons = False,
                    verbose = False,
                     **kwargs):

    """
    Purpose: To find boutons on axon branches 
    and then to save off the meshes and the widths
    without the boutons

    Pseudocode: 
    1) Restrict the axon to only those branches that should be checked
    for boutons based on their width
    2) Compute the Boutons for the restricted axon branches
    3) Plot the boutons if requested
    """
    
    if max_bouton_width_to_check is None:
        max_bouton_width_to_check = max_bouton_width_to_check_global



    ax_limb_name = neuron_obj.axon_limb_name

    if max_bouton_width_to_check is not None:

        axons_not_checked_for_boutons = ns.query_neuron(neuron_obj,
                        functions_list=["median_mesh_center","axon_label"],
                       query = f"(median_mesh_center >= {max_bouton_width_to_check}) and (axon_label == True)",

                       function_kwargs=dict(limbs_to_process=[ax_limb_name],
                                           ))

        axon_checked_for_boutons = ns.query_neuron(neuron_obj,
                        functions_list=["median_mesh_center","axon_label"],
                       query = f"(median_mesh_center < {max_bouton_width_to_check}) and (axon_label == True)",

                       function_kwargs=dict(limbs_to_process=[ax_limb_name],
                                           ))

        if verbose or plot_axon_branches_to_check:
            print(f"Restricting Axons to search for boutons using max_bouton_width_to_check = {max_bouton_width_to_check}\n"
                  f"axons not checked for boutons (blue) = {axons_not_checked_for_boutons}\n",
                 f"axons checked for boutons (red) = {axon_checked_for_boutons}")

        if plot_axon_branches_to_check:
            color_dict = nviz.limb_branch_dicts_to_combined_color_dict([axons_not_checked_for_boutons,
                                                                       axon_checked_for_boutons,
                                                                       ],
                                                                      color_list = ["blue","red"])
            nviz.visualize_neuron(neuron_obj,
                                      visualize_type=["mesh"],
                                      limb_branch_dict=neuron_obj.axon_limb_branch_dict,
                                      mesh_color=color_dict,
                                     mesh_color_alpha=1)

    else:
        axon_checked_for_boutons = neuron_obj.axon_limb_branch_dict


    # Part 2a: Putting placeholder values for those where boutons were not searched for:
    if ax_limb_name in axon_checked_for_boutons.keys():
        non_bouton_checked_branches = np.setdiff1d(neuron_obj.limb_branch_dict[ax_limb_name],
                                              axon_checked_for_boutons[ax_limb_name])
    else:
        non_bouton_checked_branches = neuron_obj.limb_branch_dict[ax_limb_name]
        
    for b_idx in non_bouton_checked_branches:
        neuron_obj[ax_limb_name][b_idx].boutons = None
        neuron_obj[ax_limb_name][b_idx].boutons_cdfs = None
        neuron_obj[ax_limb_name][b_idx].width_new[width_name] = neuron_obj[ax_limb_name][b_idx].width_new[old_width_name]
        neuron_obj[ax_limb_name][b_idx].width_array[width_name] = neuron_obj[ax_limb_name][b_idx].width_array[old_width_name]
        
        
    # Part 2b: Calculating the Boutons
    
    neuron_obj = calculate_boutons_over_limb_branch_dict(neuron_obj,
                                           limb_branch_dict = axon_checked_for_boutons,
                                           width_name=width_name,
                                            old_width_name = old_width_name,
                                           verbose = verbose)
    
    

    # Part 3: Plot the boutons if requested
    if plot_boutons:
        nviz.plot_boutons(neuron_obj,
                          mesh_whole_neuron_alpha=0.2,
                         )
        
    return neuron_obj
        

def calculate_axon_webbing_on_branch(neuron_obj,
                                 limb_idx,
                                 branch_idx,
                            allow_plotting = True,
                            plot_intersection_mesh = False,
                            plot_intersection_mesh_without_boutons = False,
                            split_significance_threshold = None,
                            plot_split = False,
                            plot_split_closest_mesh = False,
                            plot_segmentation_before_web = False,
                            plot_web = False,
                          verbose = False,
                           upstream_node_color = "red",
                            downstream_node_color = "aqua",
                                     maximum_volume_threshold=None, #in um**2
                                     minimum_volume_threshold = None,
#                                      maximum_volume_threshold=None, #in um**2
#                                      minimum_volume_threshold = None,
                                     smoothness = 0.08,
                                clusters=7,#5,
                                    ):
    """
    Purpose: If branch has been designated to be searched for a webbing,
    then run the webbing finding algorithm
    
    """
    if split_significance_threshold is None:
        split_significance_threshold = split_significance_threshold_web_global
        
    if maximum_volume_threshold is None:
        maximum_volume_threshold = maximum_volume_threshold_web_global
        
    if minimum_volume_threshold is None:
        minimum_volume_threshold = minimum_volume_threshold_web_global
    
    
    plot_idx = allow_plotting
    v = branch_idx
    curr_nx = neuron_obj[limb_idx].concept_network_directional
    ax_limb = neuron_obj[limb_idx]
    
    #a. find the downstream nodes and generate the mesh of combining upstream and downstream nodes
    d_nodes = xu.downstream_nodes(curr_nx,v)

    v_obj = ax_limb[v]
    d_nodes_obj = [ax_limb[d] for d in d_nodes]

    v_mesh_with_boutons = v_obj.mesh
    d_meshes_with_boutons = [k.mesh for k in d_nodes_obj]

    if plot_intersection_mesh and plot_idx:
        print(f"Upstream Node ({upstream_node_color}), Downstream Nodes ({downstream_node_color})")
        nviz.plot_objects(v_mesh_with_boutons,
                          main_mesh_color=upstream_node_color,
                          main_mesh_alpha=1,
                         meshes=d_meshes_with_boutons,
                         meshes_colors=downstream_node_color,
                         mesh_alpha=1)


    v_mesh_without_boutons = nru.mesh_without_boutons(v_obj)
    d_meshes_without_boutons = [nru.mesh_without_boutons(d_obj) for d_obj in d_nodes_obj]
    intersection_mesh = tu.combine_meshes([v_mesh_without_boutons] + d_meshes_without_boutons)

    if plot_intersection_mesh_without_boutons and plot_idx:
        print(f"Upstream Node ({upstream_node_color}), Downstream Nodes ({downstream_node_color})")
        nviz.plot_objects(v_mesh_without_boutons,
                          main_mesh_color=upstream_node_color,
                          main_mesh_alpha=1,
                         meshes=d_meshes_without_boutons,
                         meshes_colors=downstream_node_color,
                         mesh_alpha=1)


    #b. Find skeleton points around the intersection

    all_intersecting_skeletons = [d.skeleton for d in d_nodes_obj] + [v_obj.skeleton]
    joining_endpoint_1 = sk.shared_coordiantes(all_intersecting_skeletons,
                         return_one=True)
    coordinates_of_intersection = [sk.skeleton_coordinate_offset_from_endpoint(k,joining_endpoint_1,500) for k in all_intersecting_skeletons]


    if verbose:
        print(f"joining_endpoint_1 = {joining_endpoint_1}")
        print(f"coordinates_of_intersection = {coordinates_of_intersection}")



    #c. Split the mesh to only include central part (after filtering away boutons)
    potential_webbing_mesh = tu.closest_split_to_coordinate(intersection_mesh,
                              coordinate = coordinates_of_intersection,
                              plot_split=plot_split and plot_idx,
                              plot_closest_mesh=plot_split_closest_mesh and plot_idx,
                              significance_threshold = split_significance_threshold,
                              verbose = False)

    if maximum_volume_threshold is not None or minimum_volume_threshold is not None:
        if maximum_volume_threshold is None:
            maximum_volume_threshold = np.inf
        if minimum_volume_threshold is None:
            minimum_volume_threshold = 0
            
        
        mesh_segs, cgal_info = tu.mesh_segmentation(potential_webbing_mesh,
                            clusters=clusters,
                            smoothness=smoothness)
        
        if plot_segmentation_before_web and plot_idx:
            print(f"Before volume filter")
            tu.plot_segmentation(mesh_segs,cgal_info)
            
            
        if verbose:
            print(f"Number of segmentation before volume filter = {len(mesh_segs)}")
         
        
        mesh_segs_idx = tu.filter_meshes_by_size_min_max(
                            mesh_list=mesh_segs,
                            max_size_threshold = maximum_volume_threshold*1000*1000,
                            min_size_threshold = minimum_volume_threshold*1000*1000,
                            size_type='volume',
                            return_indices=True,
                            verbose=False,
                        )
        
        
#         mesh_segs_sizes = []
#         for k in mesh_segs:
#             try:
#                 curr_volume = int(tu.mesh_volume(k.bounding_box_oriented)/1000/1000)
#             except:
#                 curr_volume = 0
#             mesh_segs_sizes.append(curr_volume)
            
#         mesh_segs_sizes = np.array(mesh_segs_sizes)
        
#         #print(f"mesh_segs_sizes = {mesh_segs_sizes}")
        
#         mesh_segs_idx = np.where((mesh_segs_sizes<maximum_volume_threshold) & 
#                                 (mesh_segs_sizes>=minimum_volume_threshold))[0]
        
        mesh_segs = [mesh_segs[k] for k in mesh_segs_idx]
        cgal_info = [cgal_info[k] for k in mesh_segs_idx]
        if verbose:
            print(f"Number of segmentation AFTER volume filter = {len(mesh_segs)}")
            
        if plot_segmentation_before_web and plot_idx:
            print(f"After volume filter")
            tu.plot_segmentation(mesh_segs,cgal_info)
            plot_segmentation_before_web = False
        
    else:
        mesh_segs = None
        cgal_info = None
    
    #d. Running mesh segmentation to find webbing
    web_mesh,web_cdf = tu.closest_segmentation_to_coordinate(potential_webbing_mesh,
                                  coordinates_of_intersection,
                                smoothness = smoothness,
                                clusters=clusters,
                                  plot_segmentation=plot_segmentation_before_web and plot_idx,
                                  plot_closest_mesh=plot_web and plot_idx,
                                     return_cgal=True,
                                    verbose = False,
                                    mesh_segmentation = mesh_segs,
                                    mesh_segmentation_cdfs = cgal_info)

    if verbose:
        print(f"web_mesh = {web_mesh}, web_cdf = {web_cdf}")
        
    return web_mesh,web_cdf
    
    

def calculate_axon_webbing(neuron_obj,
                           n_downstream_targets_threshold = 2,
                            width_threshold = np.inf,#90,
                            width_name = "no_bouton_median",
                            width_name_backup = "no_spine_median_mesh_center",
                            idx_to_plot = None,
                            plot_intersection_mesh = False,
                            plot_intersection_mesh_without_boutons = False,
                            split_significance_threshold = None,
                            plot_split = False,
                            plot_split_closest_mesh = False,
                            plot_segmentation_before_web = False,
                            plot_web = False,
                           plot_webbing_on_neuron=False,
                          verbose = False,
                           upstream_node_color = "red",
                            downstream_node_color = "aqua"
                          ):

    """
    Purpose: To compute the webbing meshes for a neuron object
    that stores these meshes in the upstream node

    Pseudocode: 
    1) Identify all nodes that have a specific amount 
    of downstream nodes or a minimum number and a certain width

    2) For each Node to check generate the webbing mesh
    a. find the downstream nodes and generate the mesh of combining upstream and downstream nodes
    b. Find skeleton points around the intersection
    c. Split the mesh to only include central part (after filtering away boutons)
    d. Running mesh segmentation to find webbing
    e. Saving the webbing and and webbing cdf in the branch object

    """
    
    if split_significance_threshold is None:
        split_significance_threshold = split_significance_threshold_web_global

    ax_limb_name = neuron_obj.axon_limb_name
    ax_limb_idx = int(ax_limb_name[1:])
    ax_limb = neuron_obj[ax_limb_name]

    # ---- Part 1: Minimum/Certain Downstream Nodes and Width Requirement ---- 

    branches_to_check_for_webbing = ns.query_neuron(neuron_obj,
                    functions_list = ["n_downstream_nodes",
                                     "width_new"],
                    query= (
                    f"n_downstream_nodes>={n_downstream_targets_threshold}"
                    f" and width_new<{width_threshold}"
                    ),
                    limb_branch_dict_restriction = neuron_obj.axon_limb_branch_dict,
                    function_kwargs = dict(width_new_name = width_name,
                                           width_new_name_backup = width_name_backup
                                          )
    )

    if ax_limb_name in branches_to_check_for_webbing.keys():
        viable_nodes = branches_to_check_for_webbing[ax_limb_name]
    else:
        viable_nodes = []
        
    if verbose:
        print(f"branches_to_check_for_webbing = {viable_nodes}")


    #  ---- Part 2) For each Node to check generate the webbing mesh -----
    curr_nx = ax_limb.concept_network_directional
    if idx_to_plot is None:
        idx_to_plot = np.arange(len(viable_nodes))


    
    for j,v in enumerate(viable_nodes):
        if verbose:
            print(f"\n -- Working on Node {j}: Branch {v}")

        plot_idx = j in idx_to_plot


        
            

        """
        If wanted to apply size filter then would be like 
        web_size = tu.mesh_volume(web_mesh.bounding_box_oriented)/1000/1000
        
        A reasonable value for this is tu.mesh_volume(curr_web.bounding_box_oriented)/1000/1000
        """    
        try:
            web_mesh,web_cdf = calculate_axon_webbing_on_branch(neuron_obj,
                                     limb_idx = ax_limb_name,
                                     branch_idx = v,
                                allow_plotting = plot_idx,
                                plot_intersection_mesh = plot_intersection_mesh,
                                plot_intersection_mesh_without_boutons = plot_intersection_mesh_without_boutons,
                                split_significance_threshold = split_significance_threshold,
                                plot_split = plot_split,
                                plot_split_closest_mesh = plot_split_closest_mesh,
                                plot_segmentation_before_web = plot_segmentation_before_web,
                                plot_web = plot_web,
                              verbose = verbose,

                               upstream_node_color = upstream_node_color,
                                downstream_node_color = downstream_node_color)
        except:
            web_mesh = None
            web_cdf = None
            
        #e. Saving the webbing and and webbing cdf in the branch object
        neuron_obj[ax_limb_idx][v].web = web_mesh
        neuron_obj[ax_limb_idx][v].web_cdf = web_cdf

        
    if plot_webbing_on_neuron:
        nviz.plot_boutons(neuron_obj,
                  mesh_whole_neuron_alpha = 0.2,
                 plot_web=True)
        
    return neuron_obj

def axon_branching_attributes(neuron_obj,
                        limb_idx,
                        branch_idx,
                        verbose=False
                        ):
    """
    Purpose: Will compute a lot of statistics about the 
    branching behavior in an axon branching point
    
    """
    n_obj = neuron_obj
    attr_dict = dict()

    if verbose:
            print(f"Plotting web intersection for limb_idx {limb_idx}, branch_idx {branch_idx}")

    limb_obj = n_obj[limb_idx]
    parent_branch_obj = limb_obj[branch_idx]

    #1) Get the downstream nodes of the branch
    downstream_branches = xu.downstream_nodes(limb_obj.concept_network_directional,
                                              branch_idx)
    print(f"downstream_branches = {downstream_branches}")

    #2) Assemble the meshes of the parent and downstream branch
    total_nodes = list(downstream_branches) + [branch_idx]


    #4) Get the web mesh of parent node
    try:
        web_mesh = parent_branch_obj.web
        web_cdf = parent_branch_obj.web_cdf

        if web_mesh is not None:
            web_flag = True
        else:
            web_flag = False
    except:
        if verbose:
            print(f"No webbing for this branch!!")
        web_flag = False


    """
    Pseudocode: 
    1) For each valid and invalid boundary collect the following statistics:
    a. web mesh size (skeletal length, n_faces, ray_trace percentile)
    b. web_cdf
    c. web_bbox_ratio 
    d. web_volume_ratio

    information about boutons
    parent width (no_bouton_median/no_spine_median_mesh_center)
    max/min downstream width (both kinds)
    max/min downstream width differential (both kinds)
    max/min child angle
    max/min siblings

    """
    #1) Attributes about webbing
    width_names = ["no_bouton_median","no_spine_median_mesh_center"]
    size_measures = ["faces","volume","skeleton","ray_trace_percentile"]

    if web_flag:
        for s in size_measures:
            attr_dict[f"web_size_{s}"] = tu.mesh_size(web_mesh,s)

        web_bbox_rations = tu.bbox_side_length_ratios(web_mesh)
        attr_dict["web_bbox_ratios_max"] = np.max(web_bbox_rations)
        attr_dict["web_bbox_ratios_min"] = np.min(web_bbox_rations)

        web_volume_ratio = tu.mesh_volume_ratio(web_mesh)
        attr_dict["web_volume_ratio"] = web_volume_ratio

        attr_dict["web_cdf"] = web_cdf
    else:
        for s in size_measures:
            attr_dict[f"web_size_{s}"] = np.nan
            
        attr_dict["web_bbox_ratios_max"] = np.nan
        attr_dict["web_bbox_ratios_min"] = np.nan
        attr_dict["web_volume_ratio"] = np.nan
        attr_dict["web_cdf"] = np.nan
    """
    information about parent node:
    a) information about boutons
    b) parent width (no_bouton_median/no_spine_median_mesh_center)

    """
    large_bouton_face_threshold = 60
    large_bouton_ray_trace_percentile = 400

    n_large_boutons = len(nru.boutons_above_thresholds(parent_branch_obj,
                                faces=large_bouton_face_threshold,
                ray_trace_percentile=large_bouton_ray_trace_percentile))
    n_boutons = parent_branch_obj.n_boutons

    attr_dict["parent_n_large_boutons"] = n_large_boutons
    attr_dict["parent_n_boutons"] = n_boutons

    for w in width_names:
        try:
            attr_dict[f"parent_{w}"] = parent_branch_obj.width_new[w]
        except:
            attr_dict[f"parent_{w}"] = np.nan

    """
    Information about children:

    max/min downstream width (both kinds)
    max/min downstream width differential (both kinds)
    max/min child angle
    max/min siblings
    """
    downstream_widths = dict([(k,[]) for k in width_names])
    downstream_width_diff = dict([(k,[]) for k in width_names])
    downstream_child_angles = []
    downstream_n_boutons = []
    downstream_n_large_boutons = []

    for d in downstream_branches:
        d_obj = limb_obj[d]

        for w in width_names:
            if w in d_obj.width_new.keys():
                downstream_widths[w] = d_obj.width_new[w]
                downstream_width_diff[w] = d_obj.width_new[w] - parent_branch_obj.width_new[w]

        downstream_child_angles.append(nru.find_parent_child_skeleton_angle(limb_obj,
                                                          d))

        downstream_n_boutons.append(d_obj.n_boutons)

        downstream_n_large_boutons.append(
        len(nru.boutons_above_thresholds(d_obj,
                                faces=large_bouton_face_threshold,
                ray_trace_percentile=large_bouton_ray_trace_percentile))

        )

    # finding the max and min of all the categories
    stat_func_names = ["min","max"]
    prefix = "child"
    for s in stat_func_names:
        def s_func(x):
            curr_func = getattr(np,s)
            try:
                return curr_func(x)
            except:
                return np.nan
            
        for w in width_names:
            attr_dict[f"{prefix}_{w}_{s}"] = s_func(downstream_widths[w])
            attr_dict[f"{prefix}_{w}_diff_{s}"] = s_func(downstream_width_diff[w])

        attr_dict[f"{prefix}_angle_{s}"] = s_func(downstream_child_angles)
        attr_dict[f"{prefix}_n_boutons_{s}"] = s_func(downstream_n_boutons)
        attr_dict[f"{prefix}_n_large_boutons_{s}"] = s_func(downstream_n_large_boutons)


    if len(downstream_branches)>0:
        print(f"downstream_branches[0] = {downstream_branches[0]}")
        sibling_angles = list(nru.find_sibling_child_skeleton_angle(limb_obj,
                                                               downstream_branches[0],
                                     ).values())
    else:
        sibling_angles = []

    if len(sibling_angles)>0:
        attr_dict["sibling_angles_min"] = np.min(sibling_angles)
        attr_dict["sibling_angles_max"] = np.max(sibling_angles)
    else:
        attr_dict["sibling_angles_min"] = np.nan
        attr_dict["sibling_angles_max"] = np.nan
    
    # -- 4/19 Addition: Adding the number of downstream branches
    try:
        attr_dict["n_downstream"] = len(downstream_branches)
    except:
        attr_dict["n_downstream"] = 0
    
    
    return attr_dict


def complete_axon_processing_old(neuron_obj,
                             perform_axon_classification = True,
                            plot_high_fidelity_axon=False,
                            plot_boutons_web=False,
                            verbose=False,
                            add_axon_description = True,
                             return_filtering_info = True,
                            **kwargs):
    """
    To run the following axon classification processes
    1) Initial axon classification
    2) Filtering away dendrite on axon merges
    3) High fidelity axon skeletonization
    4) Bouton Identification
    5) Webbing Identification
    
    """
    
    #1) Initial axon classification
    if perform_axon_classification:
        clu.axon_classification(neuron_obj,
                           plot_axons=False,
                                verbose=verbose,
                               **kwargs)
    
    if neuron_obj.axon_limb_name is None:
        print(f"\n***No axon found for this neuron ***")
        return neuron_obj
    

    #2) Filtering away dendrite on axon merges
    pre_filters = pru.get_exc_filters_high_fidelity_axon_preprocessing()
    o_neuron_pre, filtering_info_pre = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj,
                                            filter_list = pre_filters,
                        plot_limb_branch_filter_with_disconnect_effect=False,
                                            plot_limb_branch_filter_away=False,
                                            plot_final_neuron=False,

                                            return_error_info=True,
                                             verbose=False,
                                            verbose_outline=verbose)
    
    #3) High fidelity axon skeletonization
    try:
        neuron_obj_high_fid_axon = pru.refine_axon_for_high_fidelity_skeleton(o_neuron_pre,
                                                                         **kwargs)
    except:
        raise Exception("Errored in the refine_axon_for_high_fidelity_skeleton")
        #neuron_obj_high_fid_axon = o_neuron_pre
    
    
    if plot_high_fidelity_axon:
        print(f"High Fidelity Axon")
        nviz.plot_axon(neuron_obj_high_fid_axon)
        
    #4) Bouton Identification
    neuron_obj_with_boutons = au.calculate_boutons(#parameters for run
    neuron_obj = neuron_obj_high_fid_axon,
    plot_axon_branches_to_check = False,
    plot_boutons = False,
    verbose = False,
    **kwargs
    )
    
    #5) Webbing Identification
    neuron_obj_with_web = au.calculate_axon_webbing(neuron_obj_with_boutons,
                      idx_to_plot = [],
                plot_intersection_mesh = False,
                plot_intersection_mesh_without_boutons = False,
                plot_split = False,
                plot_split_closest_mesh = False,
                plot_segmentation_before_web = False,
                plot_web = False,
                        plot_webbing_on_neuron = False,
                    verbose = False,
                        )
    
    
    if plot_boutons_web:
        nviz.plot_boutons(neuron_obj_with_web,
                  mesh_whole_neuron_alpha = 0.2,
                 plot_web=True)
        
    if add_axon_description:
        neuron_obj_with_web.description += f"_axon_v{au.axon_version}"
        
    if return_filtering_info:
        return neuron_obj_with_web,filtering_info_pre
    else:
        return neuron_obj_with_web



def wide_angle_t_candidates(neuron_obj,
                            axon_only = True,
                            child_width_maximum = 75,
                            parent_width_maximum = 75,
                            plot_two_downstream_thin_axon_limb_branch = False,
                            plot_wide_angled_children = False,
                            child_skeletal_threshold = 10000,
                            verbose = True):
    """
    To find all of the nodes that thin and wide angle
    t splits in the neuron 
    
    Application: Can be used to identify merge errors when
    there is not a valid web mesh at the location
    """

    #1) Find all of the candidate branches in the axon
    if axon_only:
        axon_branches = ns.query_neuron_by_labels(neuron_obj,
                                             matching_labels = ["axon"])
    else:
        axon_branches = neuron_obj.limb_branch_dict


    two_downstream_thin_axon_limb_branch = ns.query_neuron(neuron_obj,
                   functions_list = ["n_downstream_nodes",
                                     "n_small_children","axon_width"],
                   query = ("(n_small_children == 2) and "
                            f"(axon_width<{parent_width_maximum}) and "
                            f"(n_downstream_nodes == 2)"),
                    function_kwargs=dict(width_maximum=child_width_maximum),
                   return_dataframe=False,
                    limb_branch_dict_restriction=axon_branches)

    if plot_two_downstream_thin_axon_limb_branch:
        print(f"Plotting two_downstream_thin_axon_limb_branch")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  two_downstream_thin_axon_limb_branch)

    # find the ones that have wide angle children
    child_angle_min = 120
    wide_angled_children = ns.query_neuron(neuron_obj,
                   functions_list = ["two_children_angle","children_skeletal_lengths_min"],
                   query = (f"(two_children_angle > {child_angle_min}) & "
                           f"(children_skeletal_lengths_min > {child_skeletal_threshold})"),
                   return_dataframe=False,
                    limb_branch_dict_restriction=two_downstream_thin_axon_limb_branch)

    if plot_wide_angled_children:
        print(f"Plotting wide_angled_children")
        nviz.plot_limb_branch_dict(neuron_obj,
                                  wide_angled_children)
        
    if verbose:
        print(f"two_downstream_thin_axon_limb_branch = {two_downstream_thin_axon_limb_branch}")
        print(f"wide_angled_children= {wide_angled_children}")
        
    return wide_angled_children

def valid_web_for_t(mesh,
                    size_threshold=120,
                size_type="ray_trace_median",
                above_threshold = True,
                   verbose=False):
    """
    Will return if the mesh is a valid 
    
    """
    m_size = tu.mesh_size(mesh,
                         size_type=size_type)
    if verbose:
        print(f"Web size = {m_size}")
    if above_threshold:
        return m_size>size_threshold
    else:
        return m_size<=size_threshold
    
    
def short_thick_branches_limb_branch_dict(neuron_obj,
    width_min_threshold = None,#120,
    skeletal_length_max_threshold = None,#3500,
    ray_trace_threshold = None,#350,
    parent_width_threshold = None,#None,#200,
    plot_limb_branch_dict = False,
    exclude_starting_nodes = None,#True,
    add_zero_width_segments = None,#True,
    width_min_threshold_parent = None,#95,
    width_global_min_threshold_parent = None,#40,
    verbose = False,
    only_axon = True,
    ):
    """
    Purpose: Identify short thick branches a neuron object
    (excluding the starter node)

    Application: Can be filtered away from high_degree_coordinate
    resolution for error detection

    Pseudocode: 
    1) Query the limb or neuron using the 
    - width threshold
    - skeletal length threshold
    - end node threshold (can exclude the starting node)
    """
    if width_min_threshold is None:
        width_min_threshold = width_min_threshold_short_thick_global
    if skeletal_length_max_threshold is None:
        skeletal_length_max_threshold = skeletal_length_max_threshold_short_thick_global
    if ray_trace_threshold is None:
        ray_trace_threshold = ray_trace_threshold_short_thick_global
    if parent_width_threshold is None:
        parent_width_threshold = parent_width_threshold_short_thick_global
    if exclude_starting_nodes is None:
        exclude_starting_nodes = exclude_starting_nodes_short_thick_global
    if add_zero_width_segments is None:
        add_zero_width_segments = add_zero_width_segments_short_thick_global
    if width_min_threshold_parent is None:
        width_min_threshold_parent = width_min_threshold_parent_short_thick_global
    if width_global_min_threshold_parent is None:
        width_global_min_threshold_parent = width_global_min_threshold_parent_short_thick_global
    

    limb_branch_dict = ns.query_neuron(neuron_obj,
                   functions_list=["axon_width","skeletal_length","n_downstream_nodes","ray_trace_perc","parent_width"],
                   query=(f"((axon_width > {width_min_threshold}) or ( ray_trace_perc > {ray_trace_threshold})) and "
                          f"(skeletal_length < {skeletal_length_max_threshold}) and "
                          f"(n_downstream_nodes == 0) "
                          f"and (parent_width > {width_min_threshold_parent}) "
                          #f"and (parent_width > {width_global_min_threshold_parent})"
                         ),
                    return_dataframe=False
                    )
    


    if verbose:
        print(f"limb_branch_dict before starting node removal:\n{limb_branch_dict}")
        
    if add_zero_width_segments:
        limb_branch_dict_zero = ns.query_neuron(neuron_obj,
                   functions_list=["axon_width","width_new"],
                   query = "(axon_width == 0) or (width_new == 0)",
                                          )
        limb_branch_dict = nru.limb_branch_union([limb_branch_dict_zero,limb_branch_dict])
        
        if verbose:
            print(f"limb_branch_dict after zero segment add:\n{limb_branch_dict}")

    if exclude_starting_nodes:
        return_dict = dict()
        for limb_name,branch_list in limb_branch_dict.items():
            branch_list = np.array(branch_list)
            leftover_branches = branch_list[branch_list != neuron_obj[limb_name].current_starting_node ]
            if len(leftover_branches)>0:
                return_dict[limb_name] = leftover_branches
    else:
        return_dict = limb_branch_dict

    if verbose:
        print(f"limb_branch_dict AFTER starting node removal:\n{return_dict}")
        

                                
        
        
    if parent_width_threshold is not None:
        return_dict = ns.query_neuron(neuron_obj,
                   functions_list=["parent_width","skeletal_length","n_downstream_nodes","ray_trace_perc"],
                   query=(f"(parent_width > {parent_width_threshold})"
                         ),
                    return_dataframe=False,
                                           limb_branch_dict_restriction=return_dict
                    )
        if verbose:
            print(f"Filtered only those segments with a parent width of greater than {parent_width_threshold}:")
            print(f"limb_branch_dict = {return_dict}")
            
    if only_axon:
        axon_labels = ns.query_neuron_by_labels(neuron_obj, matching_labels=["axon"])
        return_dict = nru.limb_branch_intersection([return_dict,axon_labels])
        if verbose:
            print(f"Restricting to only axon")
            print(f"limb_branch_dict = {return_dict}")
            
    if plot_limb_branch_dict:
        nviz.plot_limb_branch_dict(neuron_obj,return_dict)

    return return_dict

'''def short_thick_branches_from_limb(limb_obj,
    width_min_threshold = 170,
    width_min_threshold_parent = 95,
    skeletal_length_max_threshold = 6000,
    max_ray_trace_percentile = 500,
    exclude_starting_nodes = True,
    verbose = False,
    ):
    """
    Purpose: Identify short thick branches on a limb object
    (maybe excluding the starter node)

    Pseudocode: 
    1) Query the limb or neuron using the 
    - width threshold
    - skeletal length threshold
    - end node threshold (can exclude the starting node)
    2) Remove the starting node
    """
    
#     more_relaxed_conditions = False
    
#     if more_relaxed_conditions:
#         width_min_threshold = 120
#         skeletal_length_max_threshold = 10000
#         max_ray_trace_percentile = 400
    
    short_thick_branches = []
    for b_idx in limb_obj.get_branch_names():
        b_obj = limb_obj[b_idx]
        if (((ns.axon_width(b_obj) > width_min_threshold) or 
                 (tu.mesh_size(b_obj.mesh,size_type="ray_trace_percentile") > max_ray_trace_percentile)) and
            (ns.skeletal_length(b_obj)<skeletal_length_max_threshold) and 
            (xu.n_downstream_nodes(limb_obj.concept_network_directional,b_idx) == 0)):
            
            if parent_width(limb_obj,b_idx) > width_min_threshold_parent:
                short_thick_branches.append(b_idx)
    
    short_thick_branches = np.array(short_thick_branches)

    if verbose:
        print(f"short_thick_branches before starting node removal:\n{short_thick_branches}")

    if exclude_starting_nodes:
        leftover_branches = short_thick_branches[short_thick_branches != limb_obj.current_starting_node ]
    else:
        leftover_branches = short_thick_branches

    if verbose:
        print(f"leftover_branches AFTER starting node removal:\n{leftover_branches}")

    return leftover_branches'''
    

def axon_angles(neuron_obj,
               verbose = False):
    """
    Purpose: To compute the axon angles for a neuron
    object that already has the axon identified
    
    """
    
    limb_branch_dict_axon = neuron_obj.axon_limb_branch_dict
    if len(limb_branch_dict_axon) == 0:
        return {}
    
    axon_angles = dict()
    for axon_limb_name,candidate_nodes in limb_branch_dict_axon.items():
        local_dict = dict()
        limb_obj = neuron_obj[axon_limb_name]
        sub_graph = limb_obj.concept_network.subgraph(candidate_nodes)
        sub_graph_conn_comp = [list(k) for k in nx.connected_components(sub_graph)]
        
        if not limb_obj.current_starting_node in candidate_nodes:
            sh_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(limb_obj.concept_network_directional,
                                                                                  [limb_obj.current_starting_node],
                                                                                  candidate_nodes,
                                                                                  )
            candidate_nodes = np.hstack([candidate_nodes,sh_path[:-1]])
        
        for j,s_graph in enumerate(sub_graph_conn_comp):
            local_axon_angles = clu.candidate_starting_skeletal_angle(limb_obj,candidate_nodes)
            print(f"local_axon_angles = {local_axon_angles}")
            local_dict[j] = local_axon_angles[0]
        axon_angles[nru.get_limb_int_name(axon_limb_name)] = local_dict
    return axon_angles



def axon_features_from_neuron_obj(neuron_obj,
                                  add_axon_prefix_to_all_keys = True,
                                  features_to_exclude=(),
                                  add_axon_start_features = True,
                                  **kwargs):
#     if neuron_obj.axon_limb_name is None:
#         return dict()

    axon_dict= apu.compartment_features_from_skeleton_and_soma_center(neuron_obj,
                                                  compartment_label = "axon",
                                                      features_to_exclude=features_to_exclude,
                                                        **kwargs)
#     axon_dict = axon_features_from_axon_sk_and_soma_center(neuron_obj.axon_skeleton,
#                                                      neuron_obj["S0"].mesh_center,
#                                                      **kwargs)
    if add_axon_prefix_to_all_keys:
        axon_dict = dict([(k,v) if "axon_" in k else (f"axon_{k}",v) for k,v in axon_dict.items()])
        
    if add_axon_start_features:
        axon_dict["axon_start_distance_from_soma"] = au.axon_start_distance_from_soma(neuron_obj)
        
        
    return axon_dict

"""
def axon_features_from_axon_sk_and_soma_center_old(axon_sk,
                                               soma_center,
                                               short_threshold = 6000,
                                    long_threshold = 100000,
                                    volume_divisor = (10**14),
                                               verbose = False,
                                    in_um = True,):
    
    # Calculating the boudning box
    sk_branches = sk.decompose_skeleton_to_branches(axon_sk)

    sk_branches_dist = np.array([sk.calculate_skeleton_distance(k) for k in sk_branches])

    n_branches = len(sk_branches)
    n_short_branches = np.sum(sk_branches_dist<short_threshold)
    n_long_branches = np.sum(sk_branches_dist>long_threshold)
    n_medium_branches = np.sum((sk_branches_dist<=long_threshold) & 
                              (sk_branches_dist>=short_threshold))

    if verbose:
        print(f"Total Number of Branches = {(n_branches)}")
        print(f"n_short_branches = {n_short_branches}, n_medium_branches = {n_medium_branches}, n_long_branches = {n_long_branches}")

    # calculating the skeletal lengths
    if in_um:
        divisor = 1000
    else:
        divisor = 1

    axon_length = np.sum(sk_branches_dist)/divisor
    axon_branch_length_median = np.median(sk_branches_dist)/divisor
    axon_branch_length_mean = np.mean(sk_branches_dist)/divisor

    if verbose:
        print(f"axon_length = {axon_length}, axon_branch_length_median = {axon_branch_length_median}, axon_branch_length_mean = {axon_branch_length_mean}")

    sk_points = axon_sk.reshape(-1,3)
    bbox = tu.coordinates_to_bounding_box(sk_points)
    bbox_volume = tu.bbox_volume(bbox)/volume_divisor
    bbox_corners = tu.bounding_box_corners(bbox)
    bbox_corners_soma_relative = bbox_corners - soma_center

    if verbose:
        print(f"bbox_volume = {bbox_volume}")
        print(f"bbox_corners = {bbox_corners}")
        print(f"bbox_corners_soma_relative = {bbox_corners_soma_relative}")

    axon_dict = dict(

                    axon_length = axon_length,
                    axon_branch_length_median = axon_branch_length_median,
                    axon_branch_length_mean = axon_branch_length_mean,

                    n_branches = n_branches,
                    n_short_branches = n_short_branches,
                    n_long_branches = n_long_branches,
                    n_medium_branches = n_medium_branches,

                    bbox_volume=bbox_volume,
                    bbox_x_min=bbox_corners[0][0],
                    bbox_y_min=bbox_corners[0][1],
                    bbox_z_min=bbox_corners[0][2],
                    bbox_x_max=bbox_corners[1][0],
                    bbox_y_max=bbox_corners[1][1],
                    bbox_z_max=bbox_corners[1][2],

                    bbox_x_min_soma_relative=bbox_corners_soma_relative[0][0],
                    bbox_y_min_soma_relative=bbox_corners_soma_relative[0][1],
                    bbox_z_min_soma_relative=bbox_corners_soma_relative[0][2],
                    bbox_x_max_soma_relative=bbox_corners_soma_relative[1][0],
                    bbox_y_max_soma_relative=bbox_corners_soma_relative[1][1],
                    bbox_z_max_soma_relative=bbox_corners_soma_relative[1][2],

                    )
    return axon_dict
"""


    
def axon_spines_limb_branch_dict(
    neuron_obj,
    ray_trace_min = None,#270,
    ray_trace_max = None,#1200,
    skeletal_length_min = None,#1000,
    skeletal_length_max = None,#6000,
    n_synapses_pre_min = None,#1,
    n_synapses_pre_max = None,#3,
    n_faces_min = None,#90,
    downstream_upstream_dist_diff = None,#1000,
    downstream_dist_min_over_syn = None,                        
    plot_short_end_nodes_with_syn = False,
    plot_axon_spines_branch_dict = False,
    exclude_starting_nodes = None,#True,
                                
    verbose = False,
    ):
    """
    Purpose: To identify all of the 
    boutons that sprout off 
    that should not cause a high order degree

    Brainstorming:
    end_node
    has one or two synapses
    between certain length: 1000 - 5000


    85% ray trace: above 270 (ray_trace_perc)
    
    Ex: 
    from neurd import axon_utils as au
    au.axon_spines_limb_branch_dict(neuron_obj,
        ray_trace_min = 270,
        ray_trace_max = 1200,
        skeletal_length_min = 1000,
        skeletal_length_max = 10000,
        n_synapses_pre_min = 1,
        n_synapses_pre_max = 3,
        n_faces_min = 150,
        plot_short_end_nodes_with_syn = False,
        plot_axon_spines_branch_dict = False,
        exclude_starting_nodes = True,
        verbose = False,
        )
    """
    
    if ray_trace_min is None:
        ray_trace_min = ray_trace_min_axon_spines_global
    if ray_trace_max is None:
        ray_trace_max = ray_trace_max_axon_spines_global
    if skeletal_length_min is None:
        skeletal_length_min = skeletal_length_min_axon_spines_global
    if skeletal_length_max is None:
        skeletal_length_max = skeletal_length_max_axon_spines_global
    if n_synapses_pre_min is None:
        n_synapses_pre_min = n_synapses_pre_min_axon_spines_global
    if n_synapses_pre_max is None:
        n_synapses_pre_max = n_synapses_pre_max_axon_spines_global
    if n_faces_min is None:
        n_faces_min = n_faces_min_axon_spines_global
    if downstream_upstream_dist_diff is None:
        downstream_upstream_dist_diff = downstream_upstream_dist_diff_axon_spines_global
    if downstream_dist_min_over_syn is None:
        downstream_dist_min_over_syn = downstream_dist_min_over_syn_axon_spines_global
    if exclude_starting_nodes is None:
        exclude_starting_nodes = exclude_starting_nodes_axon_spines_global
    
    short_end_nodes_query = (f"(ray_trace_perc >= {ray_trace_min}) and "
                          f"(ray_trace_perc <= {ray_trace_max}) and "
                          f"(skeletal_length >= {skeletal_length_min}) and "
                          f"(skeletal_length <= {skeletal_length_max}) and "
                          f"(n_synapses_post == {0}) and "
                          f"(n_synapses_pre >= {n_synapses_pre_min}) and "
                          f"(n_synapses_pre <= {n_synapses_pre_max}) and "
                          f"(n_faces_branch >= {n_faces_min}) and "
                          f"(n_downstream_nodes == 0) and "
                          f"((skeletal_length < 2500) or (synapse_closer_to_downstream_endpoint_than_upstream)) and " # part that makes sure synapse is at the end
                          f"(downstream_upstream_diff_of_most_downstream_syn < {downstream_upstream_dist_diff})"
                        f" and (downstream_dist_min_over_syn < {downstream_dist_min_over_syn})"
                         )
    
    if verbose:
        print(f"short_end_nodes_query = {short_end_nodes_query}")
    
    short_end_nodes_with_syn = ns.query_neuron(neuron_obj,
                   functions_list=["ray_trace_perc","skeletal_length","n_downstream_nodes",
                                  "n_synapses_post","n_synapses_pre","n_faces_branch",
                                  "synapse_closer_to_downstream_endpoint_than_upstream",
                                  "downstream_upstream_diff_of_most_downstream_syn",
                                  syu.downstream_dist_min_over_syn],
                   query=short_end_nodes_query,
                                              )
    if plot_short_end_nodes_with_syn:
        print(f"short_end_nodes_with_syn = {short_end_nodes_with_syn}")
        nviz.plot_limb_branch_dict(neuron_obj,short_end_nodes_with_syn)
                             
    
    axon_spines_query = (f"(ray_trace_perc >= {ray_trace_min}) and "
                          f"(ray_trace_perc <= {ray_trace_max}) "
                         )
    limb_branch_dict = ns.query_neuron(neuron_obj,
                   functions_list=["ray_trace_perc"],
                   query=axon_spines_query,
                    return_dataframe=False,
                    limb_branch_dict_restriction=short_end_nodes_with_syn
                    )
    


    if verbose:
        print(f"limb_branch_dict before starting node removal:\n{limb_branch_dict}")

    if exclude_starting_nodes:
        return_dict = dict()
        for limb_name,branch_list in limb_branch_dict.items():
            branch_list = np.array(branch_list)
            leftover_branches = branch_list[branch_list != neuron_obj[limb_name].current_starting_node ]
            if len(leftover_branches)>0:
                return_dict[limb_name] = leftover_branches
    else:
        return_dict = limb_branch_dict

    if verbose:
        print(f"limb_branch_dict AFTER starting node removal:\n{return_dict}")

    if plot_axon_spines_branch_dict:
        print(f"plot_axon_spines_branch_dict = {return_dict}")
        nviz.plot_limb_branch_dict(neuron_obj,return_dict)

    return return_dict

    return axon_dict

# ------------ 7/17: New axon classification ------------ #
def axon_classification_using_synapses(
    neuron_obj,
    axon_soma_angle_threshold = None,#70, # for excitatory

    # neuron_obj = n_obj_syn_inh
    # axon_soma_angle_threshold = None

    #inital query arguments
    ais_syn_density_max = None,#0.00007,#0.0003,
    ais_syn_alternative_max = None,
    ais_n_syn_pre_max = None,
    
    ais_width_min = None,#95,#140,
    ais_width_max = None,#550,
    max_search_distance = None,#inhibitory_axon_ais_max_search_distance, # 20_000 for excitatory
    min_skeletal_length = None,
    plot_filt_branches_without_postsyn_req = False,

    #arguments for postsyn downstream
    n_postsyn_max = None,
    postsyn_distance = None,
    plot_low_postsyn_branches = False,
    
    #arguments for ais filtering
    ais_width_filter = None,
    ais_new_width_min = None,#170,#170,
    ais_new_width_downstream_skeletal_length = None,
    
    # --- New filters added 8/11 -----
    #arguments for ais branch off filtering
    #for inhibitory use au.inhibitory_axon_ais_max_search_distance
    #for excitatory use ais_max_distance_from_soma = au.excitatory_axon_ais_max_search_distance
    ais_max_distance_from_soma = None, 
    
    #arguments for spine filtering
#     ais_spine_density_max = 0.00015,
#     n_spines_max_per_branch = 9,
    n_synapses_spine_offset_endpoint_upstream_max = None,#4,
    
    # arguments if the there is no winningn candidate
    attempt_second_pass = None,
    ais_syn_density_max_backup = None,
    ais_n_syn_pre_max_backup = None,
    max_search_distance_addition_backup = None,

    # for phase 3: picking the winning candidate
    return_best_candidate = None,
    best_candidate_method = None,
    max_skeletal_length_above_threshold_and_buffer_soma_ranges = [10_000,25_000,50_000,75_000,np.inf],
    
    #arguments for max_skeletal_length_above_threshold_and_buffer
    max_skeletal_length_min = None,
    max_skeletal_length_buffer = None,
    
    #arguments for significant_lowest_density option
    significant_lowest_density_min_skeletal_length = None,
    lowest_density_ratio = None,
    backup_best_candidate_method = ["significant_lowest_density","max_skeletal_length"],

    plot_final_axon = False,

    clean_prior_axon_labels = True,
    set_axon_labels = True,
    
    #for labeling the merge errors
    label_merge_errors = True,
    min_distance_from_soma_dendr_on_axon = None,
    
    plot_axon_on_dendrite = False,
    plot_dendrite_on_axon = False,
    
    return_axon_angle_info = True,
    downstream_distance_for_axon_angle = None,
    
    # will default to the one closer to the soma if both have certain length
    

    verbose = False,
    
    axon_classification_without_synapses_if_no_candidate = None,
    axon_classification_without_synapses = None,
    
    candidate_downstream_postsyn_density_max = None,
    ):
    """
    Purpose: To find the axon limb branch for a generic neuron
    
    Pseudocode: 

    Phase 1: Filtering
    0) Optionally restrict limbs by the connection of the soma
    1) Do a query to find branches that
    - low synapse desnity
    - min width
    - have min distance to the soma
    2) Restrict the branches to only those without a lot of downstream postsyns in the near vicitnity

    Phase 2: Gathering into Candidates

    Phase 3: Picking winning Candidate


    Things to improve:
    Can think about looking at insignificant limbs for axon
    
    
    Ex: 
    from neurd import axon_utils as au
    axon_limb_branch_dict,axon_angles_dict = au.axon_classification_using_synapses(neuron_obj_exc_syn_sp,
                                          plot_filt_branches_without_postsyn_req = False,
                                          plot_low_postsyn_branches = False,
                                         plot_final_axon=True,
                                         verbose = True)
    """
    if axon_soma_angle_threshold is None:
        axon_soma_angle_threshold = axon_soma_angle_threshold_global
    if ais_syn_density_max is None:
        ais_syn_density_max = ais_syn_density_max_global
    if ais_syn_alternative_max is None:
        ais_syn_alternative_max = ais_syn_alternative_max_global
    if ais_n_syn_pre_max is None:
        ais_n_syn_pre_max = ais_n_syn_pre_max_global
    if ais_width_min is None:
        ais_width_min = ais_width_min_global
    if ais_width_max is None:
        ais_width_max = ais_width_max_global
    if max_search_distance is None:
        max_search_distance = max_search_distance_global
    if min_skeletal_length is None:
        min_skeletal_length = min_skeletal_length_global
    if n_postsyn_max is None:
        n_postsyn_max = n_postsyn_max_global
    if postsyn_distance is None:
        postsyn_distance = postsyn_distance_global
    if ais_width_filter is None:
        ais_width_filter = ais_width_filter_global
    if ais_new_width_min is None:
        ais_new_width_min = ais_new_width_min_global
    if ais_new_width_downstream_skeletal_length is None:
        ais_new_width_downstream_skeletal_length = ais_new_width_downstream_skeletal_length_global
    if ais_max_distance_from_soma is None:
        ais_max_distance_from_soma = ais_max_distance_from_soma_global
    if n_synapses_spine_offset_endpoint_upstream_max is None:
        n_synapses_spine_offset_endpoint_upstream_max = n_synapses_spine_offset_endpoint_upstream_max_global
    if attempt_second_pass is None:
        attempt_second_pass = attempt_second_pass_global
    if ais_syn_density_max_backup is None:
        ais_syn_density_max_backup = ais_syn_density_max_backup_global
    if ais_n_syn_pre_max_backup is None:
        ais_n_syn_pre_max_backup = ais_n_syn_pre_max_backup_global
    if max_search_distance_addition_backup is None:
        max_search_distance_addition_backup = max_search_distance_addition_backup_global
    if return_best_candidate is None:
        return_best_candidate = return_best_candidate_global
    if best_candidate_method is None:
        best_candidate_method = best_candidate_method_global
    if max_skeletal_length_min is None:
        max_skeletal_length_min = max_skeletal_length_min_global
    if max_skeletal_length_buffer is None:
        max_skeletal_length_buffer = max_skeletal_length_buffer_global
    if significant_lowest_density_min_skeletal_length is None:
        significant_lowest_density_min_skeletal_length = significant_lowest_density_min_skeletal_length_global
    if lowest_density_ratio is None:
        lowest_density_ratio = lowest_density_ratio_global
    if downstream_distance_for_axon_angle is None:
        downstream_distance_for_axon_angle = downstream_distance_for_axon_angle_global
    if axon_classification_without_synapses_if_no_candidate is None:
        axon_classification_without_synapses_if_no_candidate = axon_classification_without_synapses_if_no_candidate_global
    if axon_classification_without_synapses is None:
        axon_classification_without_synapses = axon_classification_without_synapses_global
    if candidate_downstream_postsyn_density_max is None:
        candidate_downstream_postsyn_density_max = candidate_downstream_postsyn_density_max_global
        
        
    if min_distance_from_soma_dendr_on_axon is None:
        min_distance_from_soma_dendr_on_axon = min_distance_from_soma_inhibitory_dendr_on_axon_global
    
    
    
    #print(f"ais_syn_density_max= {ais_syn_density_max}, ais_syn_density_max_backup = {ais_syn_density_max_backup}")
    
    #------------------ 0) Filter Limbs By Starting Angle  ------------------
    if not axon_classification_without_synapses:
        if axon_soma_angle_threshold is not None:
            if verbose:
                print(f"Restricting limbs to those greater than {axon_soma_angle_threshold}")
            soma_center = neuron_obj["S0"].mesh_center

            possible_axon_limbs_dict = ns.query_neuron(neuron_obj,
                            query=f"soma_starting_angle>{axon_soma_angle_threshold}",
                           functions_list=[ns.soma_starting_angle],
                           function_kwargs=dict(soma_center=soma_center,
                                               verbose=verbose))

            possible_axon_limbs = list(possible_axon_limbs_dict.keys())
            possible_axon_limbs = [nru.get_limb_int_name(k) for k in possible_axon_limbs]

        else:
            possible_axon_limbs = neuron_obj.get_limb_names(return_int=True)
        if verbose: 
            print(f'\nPart 0: possible_axon_limbs = {possible_axon_limbs}')


        #---------1) Initial query to filter branches ---------
    #     low_density_limb_branch = ns.query_neuron(neuron_obj,
    #                    functions_list=[ns.synapse_density,ns.width_new,ns.n_spines,ns.n_synapses_post_spine,
    #                                   ns.synapse_density_post_offset_endpoint_upstream,
    #                                   ns.n_synapses_spine_offset_endpoint_upstream],
    #                     query=(f"((synapse_density<{ais_syn_density_max}) or ((synapse_density_post_offset_endpoint_upstream < {ais_syn_density_max_offset}) and (synapse_density < {ais_syn_density_max_backup})))"
    #                            f" and (width_new > {ais_width_min}) "
    #                            f"and (n_synapses_spine_offset_endpoint_upstream < {n_synapses_spine_offset_endpoint_upstream_max})"),  
    #                                               limbs_to_process=possible_axon_limbs,
    #                    )

        for i in range(2):
            curr_query = (f"skeletal_length > {min_skeletal_length} "
                                   f" and ((synapse_density_offset_endpoint_upstream < {ais_syn_density_max}) or (n_synapses_offset_endpoint_upstream <= {ais_syn_alternative_max}))"
                                   f" and (n_synapses_pre_offset_endpoint_upstream <= {ais_n_syn_pre_max})"
                                   f" and (width_new > {ais_width_min}) and (width_new < {ais_width_max}) "
                                   f"and (n_synapses_spine_offset_endpoint_upstream < {n_synapses_spine_offset_endpoint_upstream_max})")
            low_density_limb_branch = ns.query_neuron(neuron_obj,
                           functions_list=[ns.synapse_density,ns.width_new,ns.n_spines,ns.n_synapses_post_spine,
                                          ns.synapse_density_offset_endpoint_upstream,
                                          ns.n_synapses_spine_offset_endpoint_upstream,ns.skeletal_length,ns.n_synapses_offset_endpoint_upstream,
                                           ns.n_synapses_pre_offset_endpoint_upstream,
                                          ],
                            query=curr_query,  
                                                      limbs_to_process=possible_axon_limbs,
                           )
            print(f"branches_without_postsyn_req query = \n{curr_query}")

            low_density_limb_branch_within_dist = ns.query_neuron(neuron_obj,
                    functions_list=[ns.skeletal_distance_from_soma_excluding_node],
                    query=f"skeletal_distance_from_soma_excluding_node < {max_search_distance}",
                           limb_branch_dict_restriction=low_density_limb_branch,
                                                                 limbs_to_process=possible_axon_limbs,)

            if verbose:
                print(f"low_density_limb_branch = {low_density_limb_branch}")
                print(f"low_density_limb_branch_within_dist = {low_density_limb_branch_within_dist}")

            if plot_filt_branches_without_postsyn_req:
                nviz.plot_limb_branch_dict(neuron_obj,
                                          low_density_limb_branch_within_dist)

            # ------ 2) Restrict to Branches without a lot of postsyns downstream nearby -----
            low_postsyn = ns.query_neuron(neuron_obj,
                    functions_list=[ns.n_synapses_post_downstream_within_dist],
                    query=f"n_synapses_post_downstream_within_dist < {n_postsyn_max}",
                                                                  function_kwargs=dict(distance=postsyn_distance),
                           limb_branch_dict_restriction=low_density_limb_branch_within_dist)


            # ------- Phase 2: Gathering into Candidates ----------
            if verbose:
                print(f"low_postsyn = {low_postsyn}")

            if plot_low_postsyn_branches:
                nviz.plot_limb_branch_dict(neuron_obj,
                                          low_postsyn)

        #     axon_candidates = []
        #     for l_idx,b_idxs in low_postsyn.items():
        #         limb_obj = neuron_obj[l_idx]
        #         G = limb_obj.concept_network_directional

        #         limb_conn_comp = xu.downstream_conn_comps(G,
        #         nodes = b_idxs,
        #         start_node = limb_obj.current_starting_node,
        #         verbose = False
        #         )

        #         if verbose:
        #             print(f"{l_idx} : limb_conn_comp = {limb_conn_comp}")

        #         limb_candidates = [dict(limb_idx = l_idx,start_node=k,branches=v,) for k,v in limb_conn_comp.items()]
        #         axon_candidates += limb_candidates

            axon_candidates = nru.candidate_groups_from_limb_branch(
                                neuron_obj,
                                low_postsyn,
                                print_candidates = False,
                                connected_component_method = "local_radius",
                                radius = 5_000,
                                verbose = verbose)
            
            if verbose:
                print(f"Before filtering canddiate, axon_candidates = {axon_candidates}")
            
            axon_candidates = au.filter_candidate_branches_by_downstream_postsyn(
                neuron_obj,
                candidates = axon_candidates,
                postsyn_density_max=candidate_downstream_postsyn_density_max,
            )
            

            if verbose:
                print(f"len(axon_candidates) = {len(axon_candidates)}")
                print(f"axon_candidates = {axon_candidates}")

            if ais_width_filter:
                axon_candidates_ais_width = []
                for a in axon_candidates:
                    limb_obj = neuron_obj[a["limb_idx"]]
                    branch_idx = a["start_node"]
                    downstream_nodes = a["branches"]

                    ais_width_ac = cnu.width_downstream_restricted(limb_obj,
                                                                  branch_idx = branch_idx,
                                downstream_skeletal_length=ais_new_width_downstream_skeletal_length,
                                                                   downstream_nodes=downstream_nodes,
                                                                  )
                    axon_candidates_ais_width.append(ais_width_ac)

                if verbose:
                    print(f"axon_candidates_ais_width = {axon_candidates_ais_width}")

                axon_candidates = [a for a,a_w in zip(axon_candidates,axon_candidates_ais_width) if a_w > ais_new_width_min]

                if verbose:
                    print(f"After AIS width threhold")
                    print(f"len(axon_candidates) = {len(axon_candidates)}")
                    print(f"axon_candidates = {axon_candidates}")

            if ais_max_distance_from_soma is not None:
                """
                --- 8/11 
                    Purpose: Will require that the start of the canddiate is within a certain distance
                    of the soma

                """
                axon_candidates_soma_dist = []
                for a in axon_candidates:
                    limb_obj = neuron_obj[a["limb_idx"]]
                    branch_idx = a["start_node"]
                    downstream_nodes = a["branches"]

                    curr_soma_dist = nst.distance_from_soma(limb_obj,branch_idx)
                    axon_candidates_soma_dist.append(curr_soma_dist)

                if verbose:
                    print(f"axon_candidates_soma_dist = {axon_candidates_soma_dist}")

                axon_candidates = [a for a,a_w in zip(axon_candidates,axon_candidates_soma_dist) if a_w < ais_max_distance_from_soma]

                if verbose:
                    print(f"After AIS distance from soma threshold")
                    print(f"len(axon_candidates) = {len(axon_candidates)}")
                    print(f"axon_candidates = {axon_candidates}")


            # ------- Phase 3: Picking winning Candidate ----------s




            if len(axon_candidates) > 0 or not attempt_second_pass:
                if verbose:
                    print(f"Found at least one candidate so breaking")
                break
            else:
                print(f"\n\n!!No candidates found so trying to find candidate again with new parameters")
                print(f"Changing ais_syn_density_max ({ais_syn_density_max}) to {ais_syn_density_max_backup}")
                print(f"Changing ais_n_syn_pre_max ({ais_n_syn_pre_max}) to {ais_n_syn_pre_max_backup}")

                ais_syn_density_max = ais_syn_density_max_backup
                ais_n_syn_pre_max = ais_n_syn_pre_max_backup
                max_search_distance += max_search_distance + max_search_distance_addition_backup
    else:
        axon_candidates = []

    if (len(axon_candidates) == 0 and axon_classification_without_synapses_if_no_candidate) or axon_classification_without_synapses:
        if verbose:
            print(f"Running axon_classification_without_synapses because no axon candidates found")
        ax_cand_no_syn = au.axon_classification_without_synapses(neuron_obj,
                                     plot_final_axon=False,
                                     verbose = verbose,
                                     return_candidate=True
                                    )
        if ax_cand_no_syn is not None: 
            axon_candidates = [ax_cand_no_syn]
        else:
            axon_candidates = []

    if return_best_candidate and len(axon_candidates) > 1:
        
        found_best_candidate = False
        skeletal_lengths = np.array([np.sum([neuron_obj[a["limb_idx"]][b].skeletal_length for b in a["branches"]]) for a in axon_candidates])

#         if ((best_candidate_method == "large_skeletal_length_within_close_soma_distance") and (not found_best_candidate)):
#             """
#             Added 12/6
#             Purpose: If there are candidate that have a large skeletal length and within a close distance to soma, 
#             chose the largest one of those
            
#             Pseudocode: 
#             1) Find the distance of all candidates to the soma
#             2) Find the lengths of all of the candidates
#             3) Check to see if any meet the soma distance and skeletal length criterion
#             4) If so choose the closest candidate
            
#             """
#             large_skeletal_length_within_close_soma_distance_sk_length_threshold = 35_000
#             large_skeletal_length_within_close_soma_distance_soma_distance_threshold = 20_000
            
#             axon_candidates_soma_dist = np.array([nst.distance_from_soma_candidate(neuron_obj,a) for a in axon_candidates])
            
            
        
        if ((best_candidate_method == "max_skeletal_length_above_threshold_and_buffer") or 
                    (("max_skeletal_length_above_threshold_and_buffer" in backup_best_candidate_method) and (not found_best_candidate))):
            if verbose:
                print(f"Using Method max_skeletal_length_above_threshold_and_buffer")
                
            """
            Purpose: If there is a lot of section that is low density and more than other sections
            --> then make this the axon
            
            Psueodocde: 
            1) Check if any of the skeletal length is above the threshold
            2) Check that the buffer between max skeletal length and others is above the buffer
            """
            if max_skeletal_length_above_threshold_and_buffer_soma_ranges is None:
                max_skeletal_length_above_threshold_and_buffer_soma_ranges = [np.inf]
                
            
            axon_candidates_soma_dist = np.array([nst.distance_from_soma_candidate(neuron_obj,a) for a in axon_candidates])
            
            for sm_dist in max_skeletal_length_above_threshold_and_buffer_soma_ranges:
                if found_best_candidate:
                    break
                    
                above_threshold_len = np.where((skeletal_lengths > max_skeletal_length_min) &
                                               (axon_candidates_soma_dist < sm_dist))[0]

                if verbose:
                    print(f"above_threshold_len (for sm_dist = {sm_dist}) = {above_threshold_len}")

                if len(above_threshold_len) > 0:
                    axon_candidates_local = [axon_candidates[ii] for ii in above_threshold_len].copy()
                    skeletal_lengths_local = np.array([nru.skeletal_length_over_candidate(neuron_obj,a) for a in axon_candidates_local])
                    
                    max_sk_len = np.max(skeletal_lengths_local)
                    below_max_sk_len = skeletal_lengths_local-max_sk_len
                    n_above_buffer = np.sum(below_max_sk_len > (-1*max_skeletal_length_buffer))

                    if verbose:
                        print(f"below_max_sk_len = {below_max_sk_len}")
                        print(f"n_above_buffer = {n_above_buffer}")

                    if  n_above_buffer == 1:
                        winning_idx = np.argmax(skeletal_lengths_local)
                        axon_candidates = [axon_candidates_local[winning_idx]]

                        if verbose:
                            print(f"Adding axon candidate {winning_idx} as winner due to max_skeletal_length_above_threshold_and_buffer")

                        found_best_candidate = True
                
        
        if best_candidate_method == "significant_lowest_density":
            """
            Psuedocode: 
            1) Restrict the candidates to only those of a certain length
            
            """
            viable_lengths = np.where(skeletal_lengths > significant_lowest_density_min_skeletal_length)[0]
            if verbose:
                print(f"viable_lengths = {viable_lengths}")
            
            if len(viable_lengths) > 0:
                axon_candid_limb_branch = [{a["limb_idx"]:a["branches"]} for a in axon_candidates]
                syn_densities = np.array([syu.synapse_density_over_limb_branch(neuron_obj,
                                        axon_candid_limb_branch[k],verbose = False) for k in viable_lengths])
                    
                min_syn_density = np.min(syn_densities)
                syn_density_ratio = syn_densities/min_syn_density
                
                if verbose:
                    print(f"syn_densities = {syn_densities}")
                    print(f"min_syn_density = {min_syn_density}")
                    print(f"syn_density_ratio = {syn_density_ratio}")
                    
                one_ratio = np.where(syn_density_ratio == 1)[0]
                if len(one_ratio) == 1:
                    syn_density_ratio[syn_density_ratio == 1] == np.inf
                    min_syn_density_2nd = np.min(syn_density_ratio)
                    
                    if verbose:
                        print(f"min_syn_density_2nd = {min_syn_density_2nd}")
                    
                    if min_syn_density_2nd > lowest_density_ratio:
                        winning_idx = viable_lengths[one_ratio[0]]
                        axon_candidates = [axon_candidates[winning_idx]]
                        found_best_candidate = True
                        if verbose:
                            print(f"Setting winning axon candidate ({winning_idx}): {axon_candidates} ")

        if (best_candidate_method == "max_skeletal_length") or (("max_skeletal_length" in backup_best_candidate_method) and (not found_best_candidate)):
            winning_idx = np.argmax(skeletal_lengths)
            axon_candidates = [axon_candidates[winning_idx]]

            if verbose:
                print(f"Using max_skeletal_length for setting the axon ")
                print(f"skeletal_lengths = {skeletal_lengths}")
                print(f"winning_idx = {winning_idx}")
                print(f"axon_candidates = {axon_candidates}")
#         else:
#             raise Exception(f"Unimplmented best_candidate_method = {best_candidate_method}")

    # determing all downstream axon branches from the remaining axon candidates
    axon_full_candidates = [{a["limb_idx"]:cnu.all_downtream_branches_including_branch(neuron_obj[a["limb_idx"]],a["start_node"])} for a in axon_candidates]

    full_axon_limb_branch = nru.limb_branch_union(axon_full_candidates)

    if verbose:
        print(f"full_axon_limb_branch = {full_axon_limb_branch}")

    if plot_final_axon:
        print(f"# of final axon groups = {len(axon_full_candidates)}")
        print(f"full_axon_limb_branch = {full_axon_limb_branch}")
        nviz.plot_limb_branch_dict(neuron_obj,full_axon_limb_branch)
        print(f"After plotting plot_limb_branch_dict")


    if set_axon_labels:
        if clean_prior_axon_labels:
            nru.clear_all_branch_labels(neuron_obj,["axon"])

        nru.add_branch_label(neuron_obj,
                        limb_branch_dict=full_axon_limb_branch,
                        labels="axon")

    full_axon_limb_branch = {k:np.array(v) for k,v in full_axon_limb_branch.items()}
    """
    7/22: Will label the merge errors
    
    """
    if label_merge_errors:
        axon_on_dendrite_dict = au.compute_axon_on_dendrite_limb_branch_dict(neuron_obj = neuron_obj,
                                        plot_axon_on_dendrite=plot_axon_on_dendrite,
                                        verbose = verbose)
        if verbose:
            print(f"axon_on_dendrite_dict= {axon_on_dendrite_dict}")

        
        
        dendrite_on_axon_dict = au.compute_dendrite_on_axon_limb_branch_dict(neuron_obj,
                                             verbose = verbose,
                                            min_distance_from_soma = min_distance_from_soma_dendr_on_axon,
                                             plot_final_dendrite_on_axon = plot_dendrite_on_axon)
        if verbose:
            print(f"dendrite_on_axon_dict= {dendrite_on_axon_dict}") 
    
    """
    7/22: Will calculate the trajectory of the axon 
    candidates
    
    """
    if return_axon_angle_info:
        axon_full_candidates_axon_angles = []
        if len(axon_candidates) > 0:
            short_thick_limb_branch = au.short_thick_branches_limb_branch_dict(neuron_obj,
                                            plot_limb_branch_dict = False)

            
            for a,a_full in zip(axon_candidates,axon_full_candidates):
                limb_idx = str(a["limb_idx"])
                max_angle,min_angle,n_angles = nst.trajectory_angle_from_start_branch_and_subtree(limb_obj = neuron_obj[limb_idx],
                                                                                                  start_branch_idx=a["start_node"],
                                                                                                  subtree_branches = a_full[limb_idx],
                                                                                    nodes_to_exclude=nru.limb_branch_get(short_thick_limb_branch,limb_idx),
                                                                                                  return_max_min=True,return_n_angles=True,
                                                                                                 downstream_distance=downstream_distance_for_axon_angle)
                axon_full_candidates_axon_angles.append(dict(axon_angle_max = max_angle,
                                                       axon_angle_min = min_angle,
                                                       n_axon_angles =n_angles ))
        else:
            axon_full_candidates_axon_angles.append(dict(axon_angle_max = -1,
                                                       axon_angle_min = -1,
                                                       n_axon_angles =0 ))
            
        if return_best_candidate:
            axon_full_candidates_axon_angles = axon_full_candidates_axon_angles[0]
        
        return full_axon_limb_branch,axon_full_candidates_axon_angles
    
    return full_axon_limb_branch


def axon_classification_excitatory(
    neuron_obj,
    axon_soma_angle_threshold = None, # for excitatory
    ais_max_distance_from_soma = None,
    axon_classification_without_synapses_if_no_candidate = None,
    axon_classification_without_synapses = None,
    
    #dendrite on soma
    min_distance_from_soma_dendr_on_axon = None,
    
    ais_syn_density_max = None,
    ais_syn_density_max_backup = None, #100000  
    
    **kwargs):
    """
    Purpose: To label the axon on an excitatory neuron
    
    Example: 
    
    segment_id = 864691136333776819
    neuron_obj = du.decomposition_with_spine_recalculation(segment_id,0,
                                                        ignore_DecompositionCellType=True )
    validation=True
    n_obj_exc_1 = syu.add_synapses_to_neuron_obj(neuron_obj,
                                validation = validation,
                                verbose  = True,
                                original_mesh = None,
                                plot_valid_error_synapses = True,
                                calculate_synapse_soma_distance = False,
                                add_valid_synapses = True,
                                  add_error_synapses=False)

    au.axon_classification_excitatory(
        neuron_obj = n_obj_exc_1
    )
    nviz.plot_axon(n_obj_exc_1)
    
    """
    if axon_soma_angle_threshold is None:
        axon_soma_angle_threshold = axon_soma_angle_threshold_excitatory_global
        
    if ais_max_distance_from_soma is None:
        ais_max_distance_from_soma = ais_max_distance_from_soma_excitatory_global
        
    if axon_classification_without_synapses_if_no_candidate is None:
        axon_classification_without_synapses_if_no_candidate = axon_classification_without_synapses_if_no_candidate_excitatory_global
        
    if axon_classification_without_synapses is None:
        axon_classification_without_synapses = axon_classification_without_synapses_excitatory_global
        
        
    if min_distance_from_soma_dendr_on_axon is None:
        min_distance_from_soma_dendr_on_axon = min_distance_from_soma_excitatory_dendr_on_axon_global
        
    if ais_syn_density_max is None:
        ais_syn_density_max = ais_syn_density_max_excitatory_global
    if ais_syn_density_max_backup is None:
        ais_syn_density_max_backup = ais_syn_density_max_backup_excitatory_global
        
    
    return axon_classification_using_synapses(
                neuron_obj,
                axon_soma_angle_threshold = axon_soma_angle_threshold,
                ais_max_distance_from_soma=ais_max_distance_from_soma,
                axon_classification_without_synapses_if_no_candidate = axon_classification_without_synapses_if_no_candidate,
                axon_classification_without_synapses = axon_classification_without_synapses,
                min_distance_from_soma_dendr_on_axon=min_distance_from_soma_dendr_on_axon,
                ais_syn_density_max = ais_syn_density_max,
                ais_syn_density_max_backup = ais_syn_density_max_backup,
                **kwargs
                )

def axon_classification_inhibitory(
    neuron_obj,# for excitatory
    min_distance_from_soma_dendr_on_axon = None,
    ais_new_width_min = None,
    **kwargs):
    
    if min_distance_from_soma_dendr_on_axon is None:
        min_distance_from_soma_dendr_on_axon = min_distance_from_soma_inhibitory_dendr_on_axon_global
        
    if ais_new_width_min is None:
        ais_new_width_min= ais_new_width_min_inhibitory_global
        
    return axon_classification_using_synapses(
                neuron_obj,
                axon_soma_angle_threshold = None,
                min_distance_from_soma_dendr_on_axon=min_distance_from_soma_dendr_on_axon,
                ais_new_width_min=ais_new_width_min,
                **kwargs)


def myelination_limb_branch_dict(
    neuron_obj,
    min_skeletal_length = None,#8_000,
    max_synapse_density = None,#0.00007,
    max_synapse_density_pass_2 = None,# 0.0001,
    min_skeletal_length_pass_2 = None,#20_000,
    max_width = None,#650,
    min_distance_from_soma = None,#10_000,
    min_distance_from_soma_pass_2 = None,
    limb_branch_dict_restriction = None,
    skeletal_length_downstream_min = None,
    n_synapses_post_downstream_max = None,
    verbose = False,
    plot = False
    ):
    """
    Purpose: To find the parts of the axon that are myelinated
    with low postsyn and low width
    
    """
    if min_skeletal_length is None:
        min_skeletal_length = min_skeletal_length_myelin_global
    if max_synapse_density is None:
        max_synapse_density = max_synapse_density_myelin_global
        
        
    if min_skeletal_length_pass_2 is None:
        min_skeletal_length_pass_2 = min_skeletal_length_pass_2_myelin_global
    if max_synapse_density_pass_2 is None:
        max_synapse_density_pass_2 = max_synapse_density_pass_2_myelin_global  
    if min_distance_from_soma_pass_2 is None:
        min_distance_from_soma_pass_2 = min_distance_from_soma_pass_2_myelin_global
    
    if max_width is None:
        max_width = max_width_myelin_global
    if min_distance_from_soma is None:
        min_distance_from_soma = min_distance_from_soma_myelin_global
    if skeletal_length_downstream_min is None:
        skeletal_length_downstream_min = skeletal_length_downstream_min_ax_on_dendr_global
        
    if n_synapses_post_downstream_max is None:
        n_synapses_post_downstream_max = n_synapses_post_downstream_max_myelin_global
        
    total_myelin_lb = []
    for i,(min_sk_len,max_syn_density,min_soma_dist,downstr_post_max) in enumerate(zip([min_skeletal_length/2,min_skeletal_length,min_skeletal_length_pass_2],
               [max_synapse_density,max_synapse_density,max_synapse_density_pass_2],
               [min_distance_from_soma,min_distance_from_soma,min_distance_from_soma_pass_2],
                [n_synapses_post_downstream_max_myelin_global,n_synapses_post_downstream_max_myelin_global+2,100000000])):
        myelination_query = (f"(skeletal_length > {min_sk_len}) "
                    f"and (synapse_density_post < {max_syn_density})"
                    f"and (axon_width < {max_width})"
                    #f"and (is_axon_like == False)"
                    f"and (is_axon == False)"
                    f"and (distance_from_soma > {min_soma_dist})"
                    f" and (skeletal_length_downstream > {skeletal_length_downstream_min})"
                    )
        functions_list=["skeletal_length",
                            ns.synapse_density_post,
                            ns.axon_width,
                            #ns.is_axon_like,
                            ns.is_axon,
                            ns.distance_from_soma,
                            ns.skeletal_length_downstream,
                           ]
        if downstr_post_max < 10000:
            myelination_query += f" and (n_synapses_post_downstream < {downstr_post_max})"
            functions_list += [ ns.n_synapses_post_downstream]
        if verbose:
            print(f"myelination_query {i} = {myelination_query}")

        myelin_dict = ns.query_neuron(
            neuron_obj,
            functions_list=functions_list,
            query = myelination_query,
            plot_limb_branch_dict=plot,
            return_dataframe=False,
            limb_branch_dict_restriction=neuron_obj.dendrite_limb_branch_dict,
        )

        if verbose:
            print(f"myelin_dict_{i} = {myelin_dict}")
        total_myelin_lb.append(myelin_dict)
        
    myelin_dict_total = nru.limb_branch_union(total_myelin_lb)   
    
    if verbose:
        print(f"myelin_dict_total= {myelin_dict_total}")
    
        
    return myelin_dict_total

    


def compute_axon_on_dendrite_limb_branch_dict(
    neuron_obj,
    
    width_max = None,#270,
    n_spines_max = None,#10,
    n_synapses_post_spine_max = None,#3,
    n_synapses_pre_min = None,#1,
    synapse_pre_perc_min = None,#0.6,
    synapse_pre_perc_downstream_min = None,#0.9,
    axon_skeletal_legnth_min = None,#3000,
                                      
    
    skeletal_length_downstream_min = None,#7000
    filter_away_thin_branches = None,#True,
    dendrite_width_min = None,#80,
    thin_axon_skeletal_length_min = None,#4000,
    thin_axon_n_synapses_post_downstream_max = None,#3,
    
    #for the thick myelinated axon
    filter_away_myelination = None,#True
                                              
    #safety nets for insignificant branches
    mesh_area_min = None,#1,
    closest_mesh_skeleton_dist_max = None,#500,
                                      
    plot_axon_on_dendrite = False,
    set_axon_labels = True,
    clean_prior_labels = True,
                                              
    prevent_downstream_axon = True,
                                      
                                      
                
    verbose = False):
    """
    Purpose: To find the dendritic
    branches that are axon-like

    Pseudocode: 
    1) Query for dendritic branhes that are
    a. thin
    b. Have at least one presyn
    c. have a high pre_percentage

    2) Restrict the query to only those
    with a high pre percentage

    """
    
    if width_max is None:
        width_max = width_max_ax_on_dendr_global
    if n_spines_max is None:
        n_spines_max = n_spines_max_ax_on_dendr_global
    if n_synapses_post_spine_max is None:
        n_synapses_post_spine_max = n_synapses_post_spine_max_ax_on_dendr_global
    if n_synapses_pre_min is None:
        n_synapses_pre_min = n_synapses_pre_min_ax_on_dendr_global
    if synapse_pre_perc_min is None:
        synapse_pre_perc_min = synapse_pre_perc_min_ax_on_dendr_global
    if synapse_pre_perc_downstream_min is None:
        synapse_pre_perc_downstream_min = synapse_pre_perc_downstream_min_ax_on_dendr_global
    if axon_skeletal_legnth_min is None:
        axon_skeletal_legnth_min = axon_skeletal_legnth_min_ax_on_dendr_global
    if filter_away_thin_branches is None:
        filter_away_thin_branches = filter_away_thin_branches_ax_on_dendr_global
    if filter_away_myelination is None:
        filter_away_myelination = filter_away_myelination_myelin_global
    if dendrite_width_min is None:
        dendrite_width_min = dendrite_width_min_ax_on_dendr_global
    if thin_axon_skeletal_length_min is None:
        thin_axon_skeletal_length_min = thin_axon_skeletal_length_min_ax_on_dendr_global
    if thin_axon_n_synapses_post_downstream_max is None:
        thin_axon_n_synapses_post_downstream_max = thin_axon_n_synapses_post_downstream_max_ax_on_dendr_global
        
    if mesh_area_min is None:
        mesh_area_min = mesh_area_min_ax_on_dendr_global
    if closest_mesh_skeleton_dist_max is None:
        closest_mesh_skeleton_dist_max = closest_mesh_skeleton_dist_max_ax_on_dendr_global
    
    if skeletal_length_downstream_min is None:
        skeletal_length_downstream_min = skeletal_length_downstream_min_ax_on_dendr_global
    
    pre_limb_branch = neuron_obj.dendrite_limb_branch_dict
    
    
    
    axon_like_dendr_query = (f"(n_synapses_pre >= {n_synapses_pre_min}) and "
                             f"(synapse_pre_perc >= {synapse_pre_perc_min}) and "
                            f"(axon_width <= {width_max}) and "
                            f"(n_spines <= {n_spines_max}) and "
                            f"(n_synapses_post_spine <= {n_synapses_post_spine_max}) and "
                            f"(skeletal_length > {axon_skeletal_legnth_min}) and "
                            f"(area > {mesh_area_min}) and "
                             f"(closest_mesh_skeleton_dist < {closest_mesh_skeleton_dist_max})")
    
    if verbose:
        print(f"axon_like_dendr_query = {axon_like_dendr_query}")

    axon_like_dendr = ns.query_neuron(neuron_obj,
                   functions_list=[ns.n_synapses_pre,
                                   ns.synapse_pre_perc,
                                  ns.axon_width,
                                  ns.n_spines,
                                  ns.n_synapses_post_spine,
                                  ns.skeletal_length,
                                  ns.closest_mesh_skeleton_dist,
                                    ns.area],
                    query = axon_like_dendr_query,
                    limb_branch_dict_restriction=pre_limb_branch
                   )
    if verbose:
        print(f"axon_like_dendr = {axon_like_dendr}")

    pre_down_limb_branch_query =(f"(synapse_pre_perc_downstream >= {synapse_pre_perc_downstream_min}) or "
                             f"(n_synapses_downstream == 0) or (n_synapses_post_downstream < {thin_axon_n_synapses_post_downstream_max})")
    
    if verbose:
        print(f"pre_down_limb_branch_query = {pre_down_limb_branch_query}")
    
    pre_down_limb_branch = ns.query_neuron(neuron_obj,
                                           functions_list=[ns.synapse_pre_perc_downstream,
                                                          ns.n_synapses_downstream,
                                                          ns.n_synapses_post_downstream,],
                                           query = pre_down_limb_branch_query,
                                           limb_branch_dict_restriction = axon_like_dendr
                                                        )
    if verbose:
        print(f"pre_down_limb_branch = {pre_down_limb_branch}")
    
    if filter_away_thin_branches:
        thin_branches_query = (f"(width_new <= {dendrite_width_min}) and "
                                                f"(skeletal_length > {thin_axon_skeletal_length_min}) and "
                                                f"(n_synapses_post_downstream <= {thin_axon_n_synapses_post_downstream_max}) and"
                                                f"(area > {mesh_area_min}) and "
                                                f"(closest_mesh_skeleton_dist < {closest_mesh_skeleton_dist_max})"
                              f" and (skeletal_length_downstream > {skeletal_length_downstream_min})")
        if verbose:
            print(f"thin_branches_query= {thin_branches_query}")
        thin_axon_limb_branch = ns.query_neuron(neuron_obj,
                                               functions_list = [ns.width_new,ns.skeletal_length,
                                                                ns.n_synapses_post_downstream,
                                                              ns.closest_mesh_skeleton_dist,
                                                                 ns.skeletal_length_downstream,
                                                              ns.area],
                                query = thin_branches_query,
                                                   limb_branch_dict_restriction=pre_limb_branch)
        if verbose:
            print(f"filter_away_thin_branches = {filter_away_thin_branches}")
            print(f"thin_axon_limb_branch = {thin_axon_limb_branch}")
            
        pre_down_limb_branch = nru.limb_branch_union([thin_axon_limb_branch,pre_down_limb_branch])
        
    if filter_away_myelination:
        myelin_limb_branch = au.myelination_limb_branch_dict(neuron_obj,
                                                            skeletal_length_downstream_min=skeletal_length_downstream_min,
                                                             verbose = verbose
                                                            )
        if verbose:
            print(f"filter_away_myelination set")
            print(f"myelin_limb_branch = {myelin_limb_branch}")
        
        pre_down_limb_branch = nru.limb_branch_union([myelin_limb_branch,pre_down_limb_branch])
        
        
    if prevent_downstream_axon:
        axon_downstream = ns.query_neuron(neuron_obj,
                                            functions_list = [ns.is_axon_in_downstream_branches,
                                                             ],
                                            query = "is_axon_in_downstream_branches",)
        
        if verbose:
            print(f"axon_downstream= {axon_downstream}")
            
        pre_down_limb_branch = nru.limb_branch_setdiff([pre_down_limb_branch,axon_downstream])

    if verbose:
        print(f"pre_down_limb_branch = {pre_down_limb_branch}")

    if plot_axon_on_dendrite:
        nviz.plot_limb_branch_dict(neuron_obj,
                                  pre_down_limb_branch)
    if set_axon_labels:
        labels = ["axon-like","axon-error"]
        if clean_prior_labels:
            nru.clear_all_branch_labels(neuron_obj,labels)
            
        nru.add_branch_label(neuron_obj,
                            limb_branch_dict=pre_down_limb_branch,
                            labels=labels)
    return pre_down_limb_branch


def compute_dendrite_on_axon_limb_branch_dict(
    neuron_obj,
                                              
    # for filtering away starter ais branches
    min_distance_from_soma = None,
    n_synapses_pre_min = None,
    synapse_post_perc_min = None,
    dendrite_width_min = None,
    dendrite_skeletal_length_min = None,
    spine_density_min = None,#0.00012,
    plot_dendr_like_axon = False,
    
    #--- for a coarser catch for the dendrites -------
    coarse_dendrite_filter = None,
    coarse_dendrite_axon_width_min = None,#330
    coarse_dendrite_synapse_post_perc_min = None,#0.75
    coarse_dendrite_n_synapses_post_min = None,#20
    coarse_dendrite_n_spines_min = None,#10
    coarse_dendrtie_spine_density = None,#0.00015
    
    add_low_branch_cluster_filter = False,
    plot_low_branch_cluster_filter = False,

    synapse_post_perc_downstream_min = None,
    n_synapses_pre_downstream_max = None,

    # arguments for filtering away thick banches
    filter_away_spiney_branches = None,
    n_synapses_post_spine_max = None,
    spine_density_max = None,
    plot_spiney_branches = False,

    plot_final_dendrite_on_axon = False,
    set_axon_labels = True,
    clean_prior_labels = True,
    verbose = False,

    ):
    """
    Purpose: To find the axon branches that
    are dendritic-like

    Previous things used to find dendrites: 

    dendritic_merge_on_axon_query=None,
    dendrite_merge_skeletal_length_min = 20000,
    dendrite_merge_width_min = 100,
    dendritie_spine_density_min = 0.00015,

    dendritic_merge_on_axon_query = (f"labels_restriction == True and "
                        f"(median_mesh_center > {dendrite_merge_width_min}) and  "
                        f"(skeletal_length > {dendrite_merge_skeletal_length_min}) and "
                        f"(spine_density) > {dendritie_spine_density_min}")

    Pseudocode: 
    1) Filter away the safe postsyn group
    - thick
    - has a postsyn

    """
    if min_distance_from_soma is None:
        min_distance_from_soma = min_distance_from_soma_inhibitory_dendr_on_axon_global
    if n_synapses_pre_min is None:
        n_synapses_pre_min = n_synapses_pre_min_dendr_on_axon_global
    if synapse_post_perc_min is None:
        synapse_post_perc_min = synapse_post_perc_min_dendr_on_axon_global
    if spine_density_min is None:
        spine_density_min = spine_density_min_dendr_on_axon_global
    if dendrite_width_min is None:
        dendrite_width_min = dendrite_width_min_dendr_on_axon_global
    if dendrite_skeletal_length_min is None:
        dendrite_skeletal_length_min = dendrite_skeletal_length_min_dendr_on_axon_global
    if synapse_post_perc_downstream_min is None:
        synapse_post_perc_downstream_min = synapse_post_perc_downstream_min_dendr_on_axon_global
    if n_synapses_pre_downstream_max is None:
        n_synapses_pre_downstream_max = n_synapses_pre_downstream_max_dendr_on_axon_global
    if filter_away_spiney_branches is None:
        filter_away_spiney_branches = filter_away_spiney_branches_dendr_on_axon_global
    if n_synapses_post_spine_max is None:
        n_synapses_post_spine_max = n_synapses_post_spine_max_dendr_on_axon_global
    if spine_density_max is None:
        spine_density_max = spine_density_max_dendr_on_axon_global
        
        
    # coarse dendrite filter: 
    if coarse_dendrite_filter is None:
        coarse_dendrite_filter = coarse_dendrite_filter_global
    if coarse_dendrite_axon_width_min is None:
        coarse_dendrite_axon_width_min = coarse_dendrite_axon_width_min_global
    if coarse_dendrite_synapse_post_perc_min is None:
        coarse_dendrite_synapse_post_perc_min = coarse_dendrite_synapse_post_perc_min_global
    if coarse_dendrite_n_synapses_post_min is None:
        coarse_dendrite_n_synapses_post_min = coarse_dendrite_n_synapses_post_min_global
    if coarse_dendrite_n_spines_min is None:
        coarse_dendrite_n_spines_min = coarse_dendrite_n_spines_min_global
    if coarse_dendrtie_spine_density is None:
        coarse_dendrtie_spine_density = coarse_dendrtie_spine_density_global
        
    
    


    pre_limb_branch = neuron_obj.axon_limb_branch_dict
    
    if min_distance_from_soma is not None:
        pre_limb_branch = ns.query_neuron(neuron_obj,
                                         functions_list = [ns.distance_from_soma],
                                         query = f"distance_from_soma > {min_distance_from_soma}",
                                         limb_branch_dict_restriction=pre_limb_branch)
        if verbose:
            print(f"Restricting branches to only those a certain distance away from soma ({min_distance_from_soma})")
            print(f"pre_limb_branch AFTER restriction = {pre_limb_branch}")

    dendr_like_axon_query = (f"(n_synapses_pre >= {n_synapses_pre_min}) and "
                                 f"(synapse_post_perc >= {synapse_post_perc_min}) and "
                                f"(axon_width >= {dendrite_width_min}) and "
                                f"(spine_density >= {spine_density_min}) and "
                                #f"(n_spines <= {n_spines_max}) and "
                                f"(skeletal_length > {dendrite_skeletal_length_min})")
    if verbose:
        print(f"dendr_like_axon_query = {dendr_like_axon_query}")
    
    dendr_like_axon = ns.query_neuron(neuron_obj,
                       functions_list=[ns.n_synapses_pre,
                                       ns.synapse_post_perc,
                                      ns.axon_width,
                                      ns.n_spines,
                                      ns.n_synapses_spine,
                                      ns.skeletal_length,
                                      ns.spine_density],
                        query = dendr_like_axon_query,
                        limb_branch_dict_restriction=pre_limb_branch
                       )
    if verbose:
        print(f"dendr_like_axon = {dendr_like_axon}")

    if plot_dendr_like_axon:
        print(f"Plotting: plot_dendr_like_axon")
        nviz.plot_limb_branch_dict(neuron_obj,dendr_like_axon)
        
        
    if coarse_dendrite_filter:
        if verbose:
            print(f"Using coarse dendrite filter")
            
        coarse_dendrite_query = (f"(axon_width >  {coarse_dendrite_axon_width_min}) and "
                         f"(synapse_post_perc > {coarse_dendrite_synapse_post_perc_min}) and "
                          f"(n_synapses_post > {coarse_dendrite_n_synapses_post_min}) and "
                         f"(n_spines > {coarse_dendrite_n_spines_min}) and "
                          f"(spine_density > {coarse_dendrtie_spine_density})")

        coarse_dendrite_limb_branch = ns.query_neuron(neuron_obj,
                               functions_list=[ns.n_synapses_pre,
                                               ns.n_synapses_post,
                                               ns.synapse_post_perc,
                                              ns.axon_width,
                                              ns.n_spines,
                                              ns.n_synapses_spine,
                                              ns.skeletal_length,
                                              ns.spine_density],
                                query = coarse_dendrite_query,
                                          return_dataframe=False,
                                limb_branch_dict_restriction=pre_limb_branch,
                                             plot_limb_branch_dict=True
                               )
        
        
        dendr_like_axon = nru.limb_branch_union([dendr_like_axon,coarse_dendrite_limb_branch])
        if verbose:
            print(f"coarse_dendrite_query = {coarse_dendrite_query}")
            print(f"coarse_dendrite_limb_branch = {coarse_dendrite_limb_branch}")
            print(f"After COARSE: dendr_like_axon = {dendr_like_axon}")
            
        
        
    post_down_limb_branch_query = (f"(synapse_post_perc_downstream >= {synapse_post_perc_downstream_min}) or "
                                 f"(n_synapses_downstream == 0) or (n_synapses_pre_downstream < {n_synapses_pre_downstream_max})")

    post_down_limb_branch = ns.query_neuron(neuron_obj,
                                           functions_list=[ns.synapse_post_perc_downstream,
                                                              ns.n_synapses_downstream,
                                                              ns.n_synapses_pre_downstream],
                                               query=post_down_limb_branch_query,
                                               limb_branch_dict_restriction = dendr_like_axon
                                                            )
    if verbose:
        print(f"post_down_limb_branch_query= {post_down_limb_branch_query}")
        print(f"post_down_limb_branch = {post_down_limb_branch}")
        
    if add_low_branch_cluster_filter:
        low_branch_cluster_limb_branch = pru.low_branch_length_large_clusters_axon(neuron_obj,
                                                                                  plot = plot_low_branch_cluster_filter)
        if verbose:
            print(f"low_branch_cluster_limb_branch = {low_branch_cluster_limb_branch}")
            
        post_down_limb_branch = nru.limb_branch_union([low_branch_cluster_limb_branch,post_down_limb_branch])

    if filter_away_spiney_branches:
        spiney_limb_branch_query = (f"(axon_width >= {dendrite_width_min}) and "
                                                    f"(skeletal_length > {dendrite_skeletal_length_min}) and "
                                                    f"((n_synapses_post_spine >= {n_synapses_post_spine_max}) or "
                                            f"(spine_density > {spine_density_max}))")
        spiney_limb_branch = ns.query_neuron(neuron_obj,
                                                   functions_list = [ns.axon_width,ns.skeletal_length,
                                                                    ns.n_synapses_post_spine,ns.spine_density],
                                    query = spiney_limb_branch_query,
                                                       limb_branch_dict_restriction=pre_limb_branch)
        if verbose:
            print(f"spiney_limb_branch_query = {spiney_limb_branch_query}")
            print(f"spiney_limb_branch = {spiney_limb_branch}")

        if plot_spiney_branches:
            print(f"Plotting: spiney_limb_branch")
            nviz.plot_limb_branch_dict(neuron_obj,spiney_limb_branch)

        post_down_limb_branch = nru.limb_branch_union([spiney_limb_branch,post_down_limb_branch])

    if verbose:
        print(f"Final dendrite on axon: {post_down_limb_branch}")

    if plot_final_dendrite_on_axon:
        print(f"Plotting: final_dendrite_on_axon")
        nviz.plot_limb_branch_dict(neuron_obj,post_down_limb_branch)

    if set_axon_labels:
        labels = ["dendrite-like","dendrite-error"]
        if clean_prior_labels:
            nru.clear_all_branch_labels(neuron_obj,labels)
        nru.add_branch_label(neuron_obj,
                            limb_branch_dict=post_down_limb_branch,
                            labels=labels)    

    return post_down_limb_branch


def compute_dendrite_on_axon_limb_branch_dict_excitatory(
    neuron_obj,
    min_distance_from_soma = None,
    **kwargs
    ):
    
    if min_distance_from_soma is None:
        min_distance_from_soma = min_distance_from_soma_excitatory_dendr_on_axon_global
    
    return compute_dendrite_on_axon_limb_branch_dict(
        neuron_obj,
        min_distance_from_soma = min_distance_from_soma,
        **kwargss
    )


def compute_dendrite_on_axon_limb_branch_dict_inhibitory(
    neuron_obj,
    min_distance_from_soma = None,
    **kwargs
    ):
    
    if min_distance_from_soma is None:
        min_distance_from_soma = min_distance_from_soma_inhibitory_dendr_on_axon_global
    
    return compute_dendrite_on_axon_limb_branch_dict(
        neuron_obj,
        min_distance_from_soma = min_distance_from_soma,
        **kwargss
    )


def axon_limb_branch_dict(neuron_obj,**kwargs):
    axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         matching_labels=["axon"])
    return axon_limb_branch_dict

def dendrite_limb_branch_dict(neuron_obj,**kwargs):
    dendrite_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         not_matching_labels=["axon"])
    return dendrite_limb_branch_dict

def axon_on_dendrite_limb_branch_dict(neuron_obj,**kwargs):
    axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         matching_labels=["axon-error"])
    return axon_limb_branch_dict

def dendrite_on_axon_limb_branch_dict(neuron_obj,**kwargs):
    axon_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,
                         matching_labels=["dendrite-error"])
    return axon_limb_branch_dict
    
def axon_angles_from_neuron(neuron_obj,
                           return_max_min = True,
                            return_n_angles = True,
                            downstream_distance_for_axon_angle = None,
                            verbose = False,
                            return_dict = True,
                            **kwargs):
    """
    Purpose: to find the axon starting angle
    given an axon has already been classified

    Pseudocode: 
    1) Get the name of the axon limb
    2) if it has a name:
    - get the axon limb branch dict and extract out the branches
    3) send the branches to the trajectory angle function to get the axon angles
    """
    
    if downstream_distance_for_axon_angle is None:
        downstream_distance_for_axon_angle = downstream_distance_for_axon_angle_global

    axon_limb_name = neuron_obj.axon_limb_name

    if verbose:
        print(f"axon_limb_name = {axon_limb_name}")

    if axon_limb_name is None:
        if verbose:
            print(f"axon_limb_name was None so setting angles to -1")
        subtree_angles = np.array([-1])
        if return_max_min:
            max_angle = np.max(subtree_angles)
            min_angle = np.min(subtree_angles)
            if return_n_angles:
                return max_angle,min_angle, 0
            else:
                return max_angle,min_angle
        else:
            return subtree_angles

    else:
        short_thick_limb_branch = au.short_thick_branches_limb_branch_dict(neuron_obj,
                                            plot_limb_branch_dict = False)

        axon_branches = neuron_obj.axon_limb_branch_dict[axon_limb_name]
        return_axon_info =  nst.trajectory_angle_from_start_branch_and_subtree(neuron_obj[axon_limb_name],
                                                                 subtree_branches=axon_branches,
                                                                  nodes_to_exclude=short_thick_limb_branch[axon_limb_name],
                                                                return_max_min=return_max_min,return_n_angles=return_n_angles,
                                                                  downstream_distance=downstream_distance_for_axon_angle,
                                                                               verbose = verbose,
                                                                  **kwargs
                                                                 )
        if return_dict:
            axon_dict = dict()
            if return_max_min:
                axon_dict.update(dict(axon_angle_max = return_axon_info[0],
                                                   axon_angle_min = return_axon_info[1],))
                if return_n_angles:
                    axon_dict.update(dict(n_axon_angles =return_axon_info[2] ))
                    
            else:
                axon_dict = dict(axon_angles = return_axon_info)
            return_axon_info = axon_dict
                                 
        if verbose:
            print(f"return_axon_info = {return_axon_info}")
        
        return return_axon_info
    
def complete_axon_processing(neuron_obj,
                             #arguments for synapses and head neck spine
                             cell_type = None,
                             
                             add_synapses_and_head_neck_shaft_spines = True,
                             validation = False,
                             
                             #arguments for initial axon finding
                             perform_axon_classification = True,
                             plot_initial_axon = False,
                             rotation_function = None,
                             unrotation_function = None,
                             
                             #arguments for axon on dendrite/ dendrite on axon finding
                             label_merge_errors = True,
                             plot_axon_on_dendrite = False,
                             plot_dendrite_on_axon = False,
                             
                            plot_high_fidelity_axon=False,
                            plot_boutons_web=False,
                             
                             #arguments for readding the synapses to the high fidelity axon
                             add_synapses_after_high_fidelity_axon = True,
                             
                            verbose=False,
                            add_axon_description = True,
                             return_filtering_info = True,
                             return_axon_angle_info = True,
                             
                             filter_dendrite_on_axon = True,
                             neuron_simplification = True,
                             
                             return_G_axon_labeled = False,
                             original_mesh = None,
                             
                            **kwargs):
    """
    To run the following axon classification processes
    1) Initial axon classification
    2) Filtering away dendrite on axon merges
    3) High fidelity axon skeletonization
    4) Bouton Identification
    5) Webbing Identification
    
    """
    if rotation_function is None:
        rotation_function = align_neuron_obj
    if unrotation_function is None:
        unrotation_function = unalign_neuron_obj
    
    if add_synapses_and_head_neck_shaft_spines:
        st = time.time()
        if verbose:
            print(f"Adding the synapses and the head_neck_shaft")
        neuron_obj = syu.add_synapses_to_neuron_obj(neuron_obj,
                            validation = validation,
                            verbose  = verbose,
                            original_mesh = None,
                            plot_valid_error_synapses = False,
                            calculate_synapse_soma_distance = False,
                            add_valid_synapses = True,
                              add_error_synapses=False,
                              **kwargs)
        neuron_obj = spu.add_head_neck_shaft_spine_objs(neuron_obj,
                                                           verbose = verbose
                                                          )
        if verbose:
            print(f"Done adding synapses and head_neck_shaft: {time.time() - st}")
    
    if cell_type is None:
        axon_fun = au.axon_classification_using_synapses
    else:
        if cell_type == "excitatory":
            axon_fun = axon_classification_excitatory
        elif cell_type == "inhibitory":
            axon_fun = axon_classification_inhibitory
            
    if verbose:
        print(f"axon_fun = {axon_fun}")
        
        
    if (rotation_function is not None) and (unrotation_function is not None):
        neuron_obj = rotation_function(neuron_obj)
    
    #1) Initial axon classification
    if perform_axon_classification:
        axon_limb_branch_dict,axon_angles_dict = axon_fun(neuron_obj,
                                     plot_final_axon=plot_initial_axon,
                                    label_merge_errors=label_merge_errors,
                                    plot_axon_on_dendrite=plot_axon_on_dendrite,
                                    plot_dendrite_on_axon=plot_dendrite_on_axon,                               
                                     verbose = verbose,
                                     **kwargs)
    else:
        axon_limb_branch_dict = neuron_obj.axon_limb_branch_dict
        axon_angles_dict = au.axon_angles_from_neuron(neuron_obj = neuron_obj,
                        verbose = verbose
                       )
        
    if (rotation_function is not None) and (unrotation_function is not None):
        neuron_obj = unrotation_function(neuron_obj)
        
    
        
    if verbose:
        print(f"# of neuron_obj.synapses_somas = {len(neuron_obj.synapses_somas)}")
        print(f"axon_angles_dict = {axon_angles_dict}")
        print(f"axon_limb_branch_dict = {axon_limb_branch_dict}")
        print(f"neuron_obj.align_matrix = {neuron_obj.align_matrix}")
        

    if return_G_axon_labeled:
        from neurd import neuron_graph_lite_utils as ctcu
        if verbose:
            print(f"Computing the axon labeled graph")
        G_axon_labeled = ctcu.G_with_attrs_from_neuron_obj(neuron_obj,plot_G=False)
            
    
    if neuron_obj.axon_limb_name is None:
        print(f"\n***No axon found for this neuron ***")
        filtering_info_pre = dict(dendrite_on_axon_merges_error_area=0,
                                 dendrite_on_axon_merges_error_length=0)
        
        neuron_obj_with_web= neuron_obj
    
    else:

        # -------- Now do the merge error labels ----------- #


        
        #2) Filtering away dendrite on axon merges
        if filter_dendrite_on_axon:# and len(au.dendrite_on_axon_limb_branch_dict(neuron_obj)) > 0:
            pre_filters = pru.get_exc_filters_high_fidelity_axon_preprocessing()
            o_neuron_pre, filtering_info_pre = pru.apply_proofreading_filters_to_neuron(input_neuron = neuron_obj,
                                                    filter_list = pre_filters,
                                plot_limb_branch_filter_with_disconnect_effect=False,
                                                    plot_limb_branch_filter_away=False,
                                                    plot_final_neuron=False,

                                                    return_error_info=True,
                                                     verbose=False,
                                                    verbose_outline=verbose)
        else:
            filtering_info_pre = dict(dendrite_on_axon_merges_error_area=0,
                                 dendrite_on_axon_merges_error_length=0)
            o_neuron_pre = neuron_obj
            
        if verbose:
            print(f"After pre filtering: # of neuron_obj.synapses_somas = {len(neuron_obj.synapses_somas)}")
            print(f"o_neuron_pre.align_matrix = {o_neuron_pre.align_matrix}")
        
        if o_neuron_pre.axon_limb_name is not None:

            #3) High fidelity axon skeletonization
            
            
            neuron_obj_high_fid_axon = pru.refine_axon_for_high_fidelity_skeleton(o_neuron_pre,
                                                                                 **kwargs)
            
            if verbose:
                print(f"After high fidelity: # of neuron_obj.synapses_somas = {len(neuron_obj.synapses_somas)}")
        
            if plot_high_fidelity_axon:
                print(f"High Fidelity Axon")
                nviz.plot_axon(neuron_obj_high_fid_axon)

            #4) Bouton Identification
            neuron_obj_with_boutons = au.calculate_boutons(#parameters for run
                neuron_obj = neuron_obj_high_fid_axon,
                plot_axon_branches_to_check = False,
                plot_boutons = False,
                verbose = False,
                **kwargs
                )
            
            if verbose:
                print(f"After pre bouton: # of neuron_obj.synapses_somas = {len(neuron_obj.synapses_somas)}")
                print(f"neuron_obj_with_boutons.align_matrix = {neuron_obj_with_boutons.align_matrix}")
        

            #5) Webbing Identification
            neuron_obj_with_web = au.calculate_axon_webbing(neuron_obj_with_boutons,
                idx_to_plot = [],
                plot_intersection_mesh = False,
                plot_intersection_mesh_without_boutons = False,
                plot_split = False,
                plot_split_closest_mesh = False,
                plot_segmentation_before_web = False,
                plot_web = False,
                plot_webbing_on_neuron = False,
                verbose = False,
                )
            
            if verbose:
                print(f"neuron_obj_with_web.align_matrix = {neuron_obj_with_web.align_matrix}")


            if plot_boutons_web:
                nviz.plot_boutons(neuron_obj_with_web,
                          mesh_whole_neuron_alpha = 0.2,
                         plot_web=True)

            if add_axon_description:
                neuron_obj_with_web.description += f"_axon_v{au.axon_version}"

            if add_synapses_after_high_fidelity_axon:
                if verbose:
                    print(f"Readding Synapses to the high fidelity axon after all processing donw")
                neuron_obj_with_web = syu.add_synapses_to_neuron_obj(neuron_obj_with_web,
                        validation = validation,
                        verbose  = verbose,
                        original_mesh = original_mesh,
                        plot_valid_error_synapses = False,
                        calculate_synapse_soma_distance = True,
                        add_valid_synapses = True,
                        add_error_synapses=False,
                        limb_branch_dict_to_add_synapses=neuron_obj_with_web.axon_limb_branch_dict,
                        **kwargs)
                
                if verbose:
                    print(f"After add_synapses_after_high_fidelity_axon: # of neuron_obj.synapses_somas = {len(neuron_obj.synapses_somas)}")
                    print(f"neuron_obj_with_web.align_matrix = {neuron_obj_with_web.align_matrix}")
        else:
            if verbose:
                print(f"No axon after the dendrites filtered away")
            neuron_obj_with_web = o_neuron_pre
    
    if neuron_simplification:
        if verbose:
            print(f"Working on branch simplification after axon finding")
        neuron_obj_with_web = nsimp.branching_simplification(neuron_obj_with_web,verbose = verbose)
        
    #adding the spining info and idx to the newly added axon synapses from the high fidelity axon
    spu.set_neuron_head_neck_shaft_idx(neuron_obj_with_web)
    spu.set_neuron_synapses_head_neck_shaft(neuron_obj_with_web)
    syu.set_limb_branch_idx_to_synapses(neuron_obj_with_web)
    
    
        
    if not return_filtering_info and not return_axon_angle_info and not return_G_axon_labeled:
        return neuron_obj_with_web
    
    return_value = [neuron_obj_with_web]
    
    if return_filtering_info:
        return_value.append(filtering_info_pre)
    
    if return_axon_angle_info:
        return_value.append(axon_angles_dict)
        
    if return_G_axon_labeled:
        return_value.append(G_axon_labeled)
        
    return return_value


def axon_classification_without_synapses(
    neuron_obj,
    plot_axon_like_segments = False,
    axon_soma_angle_threshold = None,
    ais_max_search_distance = None,
    plot_candidates = False,
    plot_final_axon= False,
    verbose = False,
    axon_verbose = True,
    return_candidate = False,
    **kwargs):
    
    if axon_soma_angle_threshold is None:
        axon_soma_angle_threshold = axon_soma_angle_threshold_excitatory_global
        
    if ais_max_search_distance is None:
        ais_max_search_distance = ais_max_distance_from_soma_excitatory_global
    
    """
    Purpose: To run the axon classification developed
    previously without synapses
    
    Ex: 
    axon_candidate = au.axon_classification_without_synapses(n_obj,
                                     plot_final_axon=True,
                                     verbose = True,
                                     return_candidate=True
                                    )
    
    """

    return_value = clu.axon_classification(
                    neuron_obj=neuron_obj,
                    axon_soma_angle_threshold = axon_soma_angle_threshold,
                    ais_threshold = ais_max_search_distance,
                    plot_axon_like_segments=plot_axon_like_segments,
                    plot_candidates = plot_candidates,
                        
                    #Part 4: Filtering Candidates
                    plot_axons = False,
                    plot_axon_errors=False,
                    add_axon_labels = False,
                    clean_prior_axon_labels=False,  
                    label_axon_errors =False,


                    return_axon_labels=True,
                    return_axon_angles=False,

                    return_error_labels=False,
                    best_axon = True,

                    no_dendritic_branches_off_axon=True,


                    verbose = axon_verbose,

                        **kwargs
    )
    
    if verbose:
        print(f"Final axon limb branch = {return_value}")
        
    if plot_final_axon:
        print(f"Plotting final axon after classification with no synapses: {return_value}")
        nviz.plot_limb_branch_dict(neuron_obj,return_value)
        
    if return_candidate:
        return_value = nru.candidate_groups_from_limb_branch(neuron_obj,
                                                            return_value,
                                                             return_one=True,
                                                             #connected_component_method = "local_radius",
                                                             #radius = 5_000
                                                            )
        if verbose:
            print(f"Final axon candidate = {return_value}")
    
    return return_value


def filter_candidates_away_with_downstream_high_postsyn_branches_NOT_USED(
    neuron_obj,
    axon_candidates,
    plot_ais_skeleton_restriction = False,
    max_search_skeletal_length = 100_000,
    skeletal_length_min = 15_000,
    postsyn_density_max = 0.0015,
    verbose = True,
    ):
    """
    Purpose: To outrule a candidate because if the postsyn
    density is high far from the ais then 
    probably not the right candidate

    Pseudocode:
    1) get the branches farther than AIS distance
    For each candidate
    2) Filter the candidates by the limb branch to get remaining branches
    3) Decide if enough skeletal length to work with (if not then continue)
    4) Compute the postsynaptic density
    5) if postsynaptic density is too high then don't add to final candidates
    
    Ex: 
    au.filter_candidates_away_with_downstream_high_postsyn_branches_NOT_USED(
    neuron_obj,
    axon_candidates = [{'limb_idx': 'L0', 'start_node': 5, 'branches': [5]}, {'limb_idx': 'L0', 'start_node': 14, 'branches': [14]}]
    ,verbose = True)
    """


    from neurd import neuron_searching as ns
    restr_limb_b = ns.query_neuron(
        neuron_obj,
        query = (f"(distance_from_soma > {au.max_ais_distance_from_soma})"
                 f" and (distance_from_soma < {max_search_skeletal_length})"),
        plot_limb_branch_dict=plot_ais_skeleton_restriction,
    )

    candidates_remaining = []
    for i,c in enumerate(axon_candidates):
        new_cand = c.copy()
        new_cand["branches"] = nru.all_downstream_branches_from_candidate(neuron_obj,c,include_candidate_branches=True)
        remaining_branches = nru.candidate_limb_branch_dict_branch_intersection(candidate=new_cand,limb_branch_dict=restr_limb_b)
        if verbose:
            print(f"\n--- working on candidate {i}: {c}")
            print(f"remaining_branches = {remaining_branches}")

        if len(remaining_branches) == 0:
            if verbose:
                print(f"Adding canddiate because no remaining branches")
            candidates_remaining.append(c)
            continue

        total_skeletal_length = np.sum([neuron_obj[c["limb_idx"]][k].skeletal_length for k in remaining_branches])
        if verbose:
            print(f"total_skeletal_length = {total_skeletal_length}")

        if total_skeletal_length < skeletal_length_min:
            if verbose:
                print(f"Adding canddiate because not enough skeleton")
            candidates_remaining.append(c)
            continue

        total_postsyns = np.sum([neuron_obj[c["limb_idx"]][k].n_synapses_post for k in remaining_branches])
        postsyn_density = total_postsyns/total_skeletal_length

        if verbose:
            print(f"total_postsyns = {total_postsyns}")
            print(f"postsyn_density = {postsyn_density}")

        if postsyn_density < postsyn_density_max:
            if verbose:
                print(f"Adding canddiate because small postsyn density")
            candidates_remaining.append(c)
            
    return candidates_remaining

def filter_candidate_branches_by_downstream_postsyn(
    neuron_obj,
    candidates,
    max_search_distance = 80_000,
    max_search_distance_downstream = 50_000,
    skeletal_length_min_downstream = 15_000,
    postsyn_density_max = 0.00015,
    filter_away_empty_candidates = True,
    verbose = False,
    ):
    """
    Purpose: Need to refine the candidates so that they don't extend too far up because then the propogation down will be bad
    - want to only to apply to branches close to the soma

    Psueodocode:
    For branches with an endpoint closer than the max split distance
    1) Find the children
    2) find allb ranches within a certain downstream distance of of children
    3) If have a one of them has a significant skeleton that have any axon, then 
    remove the branch from the candidate

    """




    candidate_check_lb = ns.query_neuron(
        neuron_obj,
        functions_list=[ns.distance_from_soma],
        function_kwargs=dict(include_node_skeleton_dist = True),
        query = f"distance_from_soma < {max_search_distance}",
    )

    if verbose:
        print(f"candidate_check_lb = {candidate_check_lb}")


    final_candidates = []
    for i,c in enumerate(candidates):
        nodes_to_check = nru.candidate_limb_branch_dict_branch_intersection(candidate=c,limb_branch_dict=candidate_check_lb)
        if verbose:
            print(f"\n--Candidate {i}: {c}")
            print(f"nodes_to_check = {nodes_to_check}")
        limb_idx = c["limb_idx"]
        limb_obj = neuron_obj[limb_idx]

        branches_to_remove = []
        for n in nodes_to_check:
            if n in branches_to_remove:
                if verbose:
                    print(f"Already processed {n} so continuing")
                continue

            if verbose:
                print(f"--> checking node {n}")
            child_nodes = nru.children_nodes(limb_obj,n)
            if verbose:
                print(f"child_nodes = {child_nodes}")

            ax_children = 0
            branches_to_remove_local = []
            maybe_remove_branches = []
            for ch in child_nodes:
                downstream_branches = nru.branches_within_skeletal_distance(
                    limb_obj,
                    start_branch = ch,
                    max_distance_from_start=max_search_distance_downstream,
                    include_start_branch_length = True,
                    include_node_branch_length=False,
                    only_consider_downstream=True,
                                    )
                
                all_sub_nodes = list(nru.all_downstream_branches(limb_obj,ch)) + [ch,n]

                downstream_branches = list(downstream_branches) + [ch]
                if verbose:
                    print(f"    -> child branch with downstream_branches = {downstream_branches}")

                #get the skeletal length of all the downstream branches
                total_skeletal_length = np.sum([limb_obj[k].skeletal_length for k in downstream_branches])
                if verbose:
                    print(f"total_skeletal_length = {total_skeletal_length}")

                if total_skeletal_length <=  skeletal_length_min_downstream:
                    maybe_remove_branches += all_sub_nodes
                else:
                    if verbose:
                        print(f"Enough skeleton was computed so ")

                    total_postsyns = np.sum([limb_obj[k].n_synapses_post for k in downstream_branches])
                    postsyn_density = total_postsyns/total_skeletal_length

                    if verbose:
                        print(f"postsyn_density={postsyn_density}")

                    if postsyn_density > postsyn_density_max:
                        
                        if verbose:
                            print(f"**Removing the following nodes because the synapse density was too high: {all_sub_nodes}")

                        branches_to_remove_local += all_sub_nodes
                    else:
                        ax_children += 1
            if len(branches_to_remove_local) > 0:
                branches_to_remove_local += maybe_remove_branches
            if ax_children <= 1:
                branches_to_remove += branches_to_remove_local


        c_new = c.copy()
        c_new["branches"] = np.setdiff1d(c_new["branches"],branches_to_remove)
        

        if filter_away_empty_candidates and len(c_new["branches"]) == 0:
            continue
        else:
            c_new["start_node"] = nru.most_upstream_branch(limb_obj,c_new["branches"])
            final_candidates.append(c_new)

    return final_candidates


def axon_start_distance_from_soma(
    neuron_obj,
    default_value = None,
    verbose = False,
    ):
    """
    Purpose: To find the distance of the 
    start of an axon from the soma

    Pseudocode: 
    1) Find the most upstream branch of the axon limb branch
    2) Find the distance of that branch from the soma
    """
    ax_lb = neuron_obj.axon_limb_branch_dict
    return_value = default_value

    if ax_lb is not None:
        if len(ax_lb) > 0:
            ax_limb_name = list(ax_lb.keys())[0]
            if verbose:
                print(f"Axon Limb Name = {ax_limb_name}")

            ax_branches = ax_lb[ax_limb_name]
            limb_obj = neuron_obj[ax_limb_name]
            upstream_branch = nru.most_upstream_branch(
                limb_obj,
                branches = ax_branches,)

            distance_from_soma = nst.distance_from_soma(
                limb_obj,
                upstream_branch
            )

            if verbose:
                print(f"upstream_branch = {upstream_branch}")
                print(f"Axon distance from soma = {distance_from_soma}")

            return_value = distance_from_soma

    return return_value



# ---------- Setting of parameters ---------- 

attributes_dict_default = dict(
    align_neuron_obj = None,
    unalign_neuron_obj = None,
    max_ais_distance_from_soma = 50_000,
)    

global_parameters_dict_default_axon_finding = dsu.DictType(
    #-- cell type specific axon finding --
    # -- excitatory
    axon_soma_angle_threshold_excitatory = 70,
    ais_max_distance_from_soma_excitatory = 14_000,
    axon_classification_without_synapses_excitatory = False,
    axon_classification_without_synapses_if_no_candidate_excitatory = True,

    # -- inhibitory
    ais_max_distance_from_soma_inhibitory = 70_000,

    
    
    # ----- for the axon finding with synapses -------
    axon_soma_angle_threshold= (None,"int unsigned"),
    
    #inital query arguments
    ais_syn_density_max = 0.00015,
    ais_syn_alternative_max = 2,
    ais_n_syn_pre_max = 1,


    ais_width_min = 95,#95,#140,
    ais_width_max = 650,#550,
    max_search_distance = 80_000,
    min_skeletal_length = 10_000,

    #arguments for postsyn downstream
    n_postsyn_max = 15,
    postsyn_distance = 10000,

    #arguments for ais filtering
    ais_width_filter = True,
    ais_new_width_min = 140,#170,#170,
    ais_new_width_min_inhibitory = 140,
    ais_new_width_downstream_skeletal_length = 20_000,

    # --- New filters added 8/11 -----
    #arguments for ais branch off filtering
    #for inhibitory use au.inhibitory_axon_ais_max_search_distance
    #for excitatory use ais_max_distance_from_soma = au.excitatory_axon_ais_max_search_distance
    ais_max_distance_from_soma = (None, "int unsigned"),

    #arguments for spine filtering
    n_synapses_spine_offset_endpoint_upstream_max = 3,#4,

    # arguments if the there is no winningn candidate
    attempt_second_pass = True,
    ais_syn_density_max_backup = 0.0007,
    ais_n_syn_pre_max_backup = 100,
    max_search_distance_addition_backup = 0,

    # for phase 3: picking the winning candidate
    return_best_candidate = True,
    best_candidate_method = "max_skeletal_length_above_threshold_and_buffer",

    #arguments for max_skeletal_length_above_threshold_and_buffer
    max_skeletal_length_min = 50_000,
    max_skeletal_length_buffer = 20_000,

    #arguments for significant_lowest_density option
    significant_lowest_density_min_skeletal_length = 15000,
    lowest_density_ratio = 4,

    #for labeling the merge errors
    downstream_distance_for_axon_angle = 30_000,

    # for backup axon finding if didn't find one
    axon_classification_without_synapses_if_no_candidate = False,
    axon_classification_without_synapses = False,
    
    
    
    #excitatory
    ais_syn_density_max_excitatory = None,
    ais_syn_density_max_backup_excitatory = None,
    
    candidate_downstream_postsyn_density_max = 0.00015,
)

global_parameters_dict_default_dendr_on_axon = dict(
    #base dendrite like restriction for width,synapses and min length
    min_distance_from_soma_inhibitory_dendr_on_axon = 10_000,
    min_distance_from_soma_excitatory_dendr_on_axon = 10_000,
    n_synapses_pre_min_dendr_on_axon = 1,
    synapse_post_perc_min_dendr_on_axon = 0.6,
    spine_density_min_dendr_on_axon = 0.00012,
    dendrite_width_min_dendr_on_axon = 170,
    dendrite_skeletal_length_min_dendr_on_axon = 3000,

    # for the coarse filtering of the dendrites to catch any that might have been missed
    coarse_dendrite_filter = True,
    coarse_dendrite_axon_width_min = 300,
    coarse_dendrite_synapse_post_perc_min = 0.75,
    coarse_dendrite_n_synapses_post_min = 20,
    coarse_dendrite_n_spines_min = 10,
    coarse_dendrtie_spine_density = 0.00015,
    
    # restricting dendrites to only those with synapses downstream
    synapse_post_perc_downstream_min_dendr_on_axon = 0.9,
    n_synapses_pre_downstream_max_dendr_on_axon = 3,

    # arguments for filtering away spiney branches
    filter_away_spiney_branches_dendr_on_axon = False,
    n_synapses_post_spine_max_dendr_on_axon = 4,
    spine_density_max_dendr_on_axon = 0.00015,
)

global_parameters_dict_default_bouton_webbing = dict(
    max_bouton_width_to_check = 120,
    
    #bouton calculation
    min_size_threshold_bouton = 27,
    max_size_threshold_bouton = 350,
    ray_trace_threshold_bouton = 200,
    
    #web calculations
    split_significance_threshold_web = 20,
    maximum_volume_threshold_web=3000, #in um**2
    minimum_volume_threshold_web = 20,
)

global_parameters_dict_default_auto_proof = dsu.DictType(
    
    # ---axon spines ----
    ray_trace_min_axon_spines = 270,
    ray_trace_max_axon_spines = 1200,
    skeletal_length_min_axon_spines = 1000,
    skeletal_length_max_axon_spines = 6000,
    n_synapses_pre_min_axon_spines = 1,
    n_synapses_pre_max_axon_spines = 3,
    n_faces_min_axon_spines = 90,
    downstream_upstream_dist_diff_axon_spines = 1000,
    downstream_dist_min_over_syn_axon_spines = 2000,
    exclude_starting_nodes_axon_spines = True,
    
    # -- short_thick_branches_limb_branch_dict --
    width_min_threshold_short_thick = 120,
    skeletal_length_max_threshold_short_thick = 3500,
    ray_trace_threshold_short_thick = 350,
    parent_width_threshold_short_thick = (None,"int unsigned"),#200,
    exclude_starting_nodes_short_thick = True,
    add_zero_width_segments_short_thick = True,
    width_min_threshold_parent_short_thick = 95,
    width_global_min_threshold_parent_short_thick = 40,
    
)

global_parameters_dict_default_axon_on_dendrite = dict(
    width_max_ax_on_dendr = 270,
    n_spines_max_ax_on_dendr = 10,
    n_synapses_post_spine_max_ax_on_dendr = 3,
    n_synapses_pre_min_ax_on_dendr = 1,
    synapse_pre_perc_min_ax_on_dendr = 0.6,
    synapse_pre_perc_downstream_min_ax_on_dendr = 0.9,
    axon_skeletal_legnth_min_ax_on_dendr = 2500,#3000,
                                      
    filter_away_thin_branches_ax_on_dendr = True,
    dendrite_width_min_ax_on_dendr = 80,
    thin_axon_skeletal_length_min_ax_on_dendr = 2500,#4000,
    thin_axon_n_synapses_post_downstream_max_ax_on_dendr = 3,
    
    mesh_area_min_ax_on_dendr = 1,
    closest_mesh_skeleton_dist_max_ax_on_dendr = 500,
    
    #myelination
    filter_away_myelination_myelin = True,
    min_skeletal_length_myelin = 5_000,
    max_synapse_density_myelin = 0.00007,
    max_synapse_density_pass_2_myelin = 0.0001,
    min_skeletal_length_pass_2_myelin = 25_000,
    min_distance_from_soma_pass_2_myelin = -1,
    max_width_myelin = 650,
    min_distance_from_soma_myelin = 10_000,
    skeletal_length_downstream_min_ax_on_dendr = 7_000,
    n_synapses_post_downstream_max_myelin = 5,
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_axon_finding,
    global_parameters_dict_default_dendr_on_axon,
    global_parameters_dict_default_bouton_webbing,
    global_parameters_dict_default_auto_proof,
    global_parameters_dict_default_axon_on_dendrite
    ])


global_parameters_dict_microns = {}
global_parameters_dict_microns_auto_proof = {}
attributes_dict_microns = {}

attributes_dict_h01 = dict(
    align_neuron_obj = hvu.align_neuron_obj,
    unalign_neuron_obj = hvu.unalign_neuron_obj,
    
    max_ais_distance_from_soma = 50_000,
)

global_parameters_dict_h01_axon_finding = dsu.DictType(
    axon_classification_without_synapses_excitatory = True,
    
    
    #ais_width_max = 1200,
    ais_width_max = 900,
    
    ais_new_width_min = 140,
    
    min_skeletal_length = 10_000,
    #min_skeletal_length = 5000,
    ais_syn_density_max_excitatory = 100000,
    ais_syn_density_max_backup_excitatory = 100000, 
    
    #--- inhibitory parameters ---
    ais_new_width_min_inhibitory = 140,
    
    n_postsyn_max = 100000,
    postsyn_distance = 90_000,
    
    max_search_distance_addition_backup = 10_000,
                                                                 
)

global_parameters_dict_h01_dendr_on_axon = dict(
    dendrite_width_min_dendr_on_axon = 250,
    min_distance_from_soma_excitatory_dendr_on_axon = 60_000,
    min_distance_from_soma_inhibitory_dendr_on_axon = 10_000,
    
    coarse_dendrite_filter = True,
    n_synapses_pre_min_dendr_on_axon = 0,
    
    min_skeletal_length_myelin = 12_000,
)

global_parameters_dict_h01_bouton_webbing = dict(
    max_bouton_width_to_check = 120,
    
    #bouton calculation
    min_size_threshold_bouton = 30,
    max_size_threshold_bouton = 400,
    
    #web calculations
    split_significance_threshold_web = 23,
    maximum_volume_threshold_web=3500, #in um**2
    minimum_volume_threshold_web = 22,
    
    
)

global_parameters_dict_h01_auto_proof = dict(
    skeletal_length_max_axon_spines = 12_000,
)

global_parameters_dict_h01_axon_on_dendrite = dict(
    n_synapses_pre_min_ax_on_dendr = 0,
    synapse_pre_perc_min_ax_on_dendr = 0,
    width_max_ax_on_dendr = 350,
    dendrite_width_min_ax_on_dendr = 200,
)


global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_axon_finding,
    global_parameters_dict_h01_dendr_on_axon,
    global_parameters_dict_h01_bouton_webbing,
    global_parameters_dict_h01_auto_proof,
    global_parameters_dict_h01_axon_on_dendrite
])


# data_type = "default"
# algorithms = None
# modules_to_set = [au,(pre,"axon_decomp"),(clu,"axon"),nst,(pru,"low_branch_clusters")]

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
#                                                    algorithms,)

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
from . import apical_utils as apu
from . import classification_utils as clu
from . import concept_network_utils as cnu
from .h01_volume_utils import data_interface as hvu
from . import neuron_searching as ns
from . import neuron_simplification as nsimp
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import preprocess_neuron as pre
from . import proofreading_utils as pru
from . import spine_utils as spu
from . import synapse_utils as syu
from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import data_struct_utils as dsu
from datasci_tools import general_utils as gu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import module_utils as modu 
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools.tqdm_utils import tqdm

from . import axon_utils as au