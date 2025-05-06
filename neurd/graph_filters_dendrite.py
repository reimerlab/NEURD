from dataclasses import dataclass
from . import (
    #axon_utils as au,
    #neuron_statistics as nst,
    #neuron_searching as ns,
    #concept_network_utils as cnu,
    limb_utils as lu,
    branch_utils as bu,
    graph_filters_refactored as gf,
    axon_utils as au,
    neuron_utils as nru,
)
import numpy as np

## -- Filter 1
"""
Every filter will have the following classes
i.  DownstreamErrorFilter
ii. DownstreamErrorFilterConfig (initializes the filter)
"""
class DoubleBackDownstreamErrorFilter(gf.DownstreamErrorFilter):
    @dataclass(frozen = True)
    class Config:
        skeleton_attribute: str = "skeleton_smooth"
        double_back_threshold: float = 110
        with_offset: bool = True
        max_skeleton_endpoint_dist_threshold = 5000
    
    def parent_angle(
        self,
        limb_obj,
        n,
        with_offset = None,
        default_value=0,
        verbose = False,):
            
        if with_offset is None:
            with_offset = self.config.with_offset
            
        angles = []
        p_angle = lu.parent_skeletal_angle(limb_obj,n,
                                           default_value=default_value,
                                           skeleton_attribute=self.config.skeleton_attribute)
        angles.append(p_angle)
        if with_offset:
            p_angle_offset = lu.parent_skeletal_angle_extra_offset(limb_obj,n,
                                               default_value=default_value,
                                               skeleton_attribute=self.config.skeleton_attribute)
            angles.append(p_angle_offset)
        parent_angle_final = min(*angles)
        if verbose:
            print(f"node {n}: all parent angles (with_offset = {with_offset}) = {angles}, final angle = {parent_angle_final}")
        return parent_angle_final
    

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose = False,angle_verbose = False,**kwargs):
        error_nodes = []
        for n in downstream_nodes:
            p_angle = self.parent_angle(limb_obj,n,verbose=angle_verbose,**kwargs)
            if verbose:
                print(f"downstream node {n} had parent angle: {p_angle}")
            if p_angle > self.config.double_back_threshold:
                
                # check if the skeleton jumps too much to be considered for an error
                if self.config.max_skeleton_endpoint_dist_threshold < np.inf:
                    max_skeleton_endpoint_dist = limb_obj[n].max_skeleton_endpoint_dist
                    if verbose:
                        print(f"max_skeleton_endpoint_dist = {max_skeleton_endpoint_dist}")
                    if max_skeleton_endpoint_dist > self.config.max_skeleton_endpoint_dist_threshold:
                        if verbose:
                            print(f"max_skeleton_endpoint_dist ({max_skeleton_endpoint_dist:.2f}) greater than threshold ({self.config.max_skeleton_endpoint_dist_threshold:.2f}) so skipping")
                        continue
                else:
                    max_skeleton_endpoint_dist = None
                    
                description=f"parent angle ({p_angle:.2f}) with {parent_node} greater than double_back_threshold ({self.config.double_back_threshold:.2f}), max_skeleton_endpoint_dist = {max_skeleton_endpoint_dist:.2f}"
                if verbose:
                    print(f"\t error: {description}")
                error_nodes.append(gf.NodeError(n,description))
        return error_nodes
        
    @property
    def description(self):
        return f"Errors downstream node if parent angle greater than {self.config.double_back_threshold}"
    
## --- filter #2: CrossRoads filter --
from dataclasses import dataclass

# @dataclass
# class CrossRoadsDownstreamErrorFilterConfig:
#     angle_extra_offset: bool = True
#     sibling_angle_min: float = 140
#     width_extra_offset: bool = True
#     width_diff_perc_max: float = 0.2
#     width_diff_max: float = 100

class CrossRoadsDownstreamErrorFilter(gf.DownstreamErrorFilter):
    @dataclass(frozen=True)
    class Config:
        angle_extra_offset: bool = True
        sibling_angle_min: float = 140
        width_extra_offset: bool = True
        width_diff_perc_max: float = 1.5
        width_diff_max: float = 100
        
    # def __init__(
    #     self,
    #     config: Config = None,
    #     **kwargs
    # ):
    #     super().__init__(self,config,**kwargs)

    def sibling_angle(self,limb,n1,n2,verbose = False):
        return lu.sibling_angle_smooth(
            limb,
            n1,
            n2,
            extra_offset=self.config.angle_extra_offset,
            verbose = verbose,
            
        )

    def width_diff_perc(self,limb,n1,n2,verbose=False):
        return bu.width_diff_directional(
            limb_obj = limb,
            branch_1 = n1,
            branch_2 = n2,
            extra_offset = self.config.width_extra_offset,
            verbose = verbose,
            branch_1_dir = "upstream",
            branch_2_dir = "upstream",
            return_percentage = True,
        )

    def width_diff(self,limb,n1,n2,verbose=False):
        return bu.width_diff_directional(
            limb_obj = limb,
            branch_1 = n1,
            branch_2 = n2,
            extra_offset = self.config.width_extra_offset,
            verbose = verbose,
            branch_1_dir = "upstream",
            branch_2_dir = "upstream",
            return_percentage = False,
        )

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose = False,**kwargs):
        error_nodes = []
        for n in downstream_nodes:
            if verbose:
                print(f"-- Working on branch {n}")
            if n in error_nodes:
                if verbose:
                    print(f"\tSkipping because already in downstream_nodes")
                continue
            for sib in downstream_nodes:
                if n >= sib:
                    continue
                sib_angle = self.sibling_angle(limb_obj,n,sib)
                width_diff_perc = self.width_diff_perc(limb_obj,n,sib)
                width_diff = self.width_diff(limb_obj,n,sib)

                stats_str = (f"sib_angle = {sib_angle:.2f}"
                          f", width_diff_perc = {width_diff_perc:.2f}"
                          f", width_diff = {width_diff:.2f}")
                if verbose:
                    print(f"\tFor sibling {sib}: {stats_str}")
                # determine if an error occured
                if (sib_angle > self.config.sibling_angle_min) and (
                    (width_diff_perc < self.config.width_diff_perc_max)
                    or (width_diff < self.config.width_diff_max)
                ):

                    #if error then add both nodes to error nodes
                    if verbose:
                        print(f"\tCross roads error for {n},{sib}")
                    n_node = gf.NodeError(n,f"cross roads with {sib}: {stats_str}")
                    sib_node = gf.NodeError(sib,f"cross roads with {n}: {stats_str}")
                    error_nodes += [n_node,sib_node]
        if verbose:
            print(f"Final error nodes = {error_nodes}")
        return error_nodes

    @property
    def description(self):
        return_str = f"""
            Returns a downstream node as an error if the following
                    conditions are met with another downstream node:
            1. width difference is under width_diff_perc_max ({self.config.width_diff_perc_max})
                or under width_diff_max ({self.config.width_diff_max})
            2. sibling angle > {self.config.sibling_angle_min}
        """
        return return_str
    
## -- Filter 3
class DendriteWidthJumpDownstreamErrorFilter(gf.DownstreamErrorFilter):
    @dataclass(frozen=True)
    class Config:
        upstream_width_jump_threshold: float = 190
        upstream_width_jump_perc_threshold: float = 0.4
        max_skeleton_endpoint_dist_threshold: float = np.inf

        # for computing the nodes to compare the width to
        require_filtered_graph: bool = True
        ignore_skipped_nodes_for_width: bool = True
        min_skeletal_length_for_upstream_path: float = 5000
        remove_first_branch_for_upstream_path: bool = True

    def max_skeleton_endpoint_dist(self,limb_obj,n):
        return limb_obj[n].max_skeleton_endpoint_dist
        
    @property
    def width_func(self):
        return au.axon_width
        
    def upstream_width_min(
        self,
        limb_obj,
        n,
        filtered_graph=None,
        verbose= False,
        nodes_to_ignore = None,
        width_func = None,
        remove_zeros = True,
        default = 1000000,
        return_min_upstream = True):
        """
        
        """
        if self.config.ignore_skipped_nodes_for_width:
            nodes_to_ignore = filtered_graph.skipped_nodes_too_small_skeleton
            if verbose:
                print(f"too_small_skeleton skeleton nodes being ignore = {nodes_to_ignore}")
        else:
            nodes_to_ignore = []

        width_path,branch_path = lu.width_path_to_start(
            limb_obj,
            n,
            nodes_to_ignore = nodes_to_ignore,
            remove_zeros = True,
            verbose = verbose,
            skeletal_length_min = self.config.min_skeletal_length_for_upstream_path,
            remove_start_branch = not self.config.remove_first_branch_for_upstream_path,
            width_func = self.width_func,
            return_branch_path = True,
            )

        if len(width_path) > 0:
            min_idx = np.argmin(width_path)
            width_min = width_path[min_idx]
            min_upstream = branch_path[min_idx]
        else:
            width_min = default
            min_upstream = None

        if verbose:
            print(f"width_min = {width_min:.2f} (min_upstream = {min_upstream}")

        if return_min_upstream:
            return width_min,min_upstream
        else:
            return width_min

    def upstream_width_min_jump(
        self,
        limb_obj,
        n,
        filtered_graph,
        verbose = False,**kwargs):
        up_width_min,min_upstream = self.upstream_width_min(
            limb_obj,
            n,
            filtered_graph=filtered_graph,
            verbose= verbose
        )
        branch_width = self.width_func(limb_obj[n])
        width_jump = branch_width - up_width_min
        width_jump_perc = width_jump/up_width_min
        if verbose:
            print(f"width_jump = {width_jump:.2f} (min_upstream = {min_upstream})")
        return width_jump,width_jump_perc,min_upstream
    
    def __call__(
        self,
        limb_obj,
        downstream_nodes,
        filtered_graph,
        verbose = False,
        verbose_width_jump=False,
        **kwargs
        ):
        """
        Calculate the 
        """
        limb = limb_obj
        error_nodes = []
        for n in downstream_nodes: 
            if verbose:
                print(f"Working on downstream node {n}:")
            
            (upstream_width_jump,
             upstream_width_jump_perc,
             min_upstream) = self.upstream_width_min_jump(
                limb,n,filtered_graph=filtered_graph,verbose = verbose_width_jump,**kwargs
            )
            width_flag = (
                (upstream_width_jump > self.config.upstream_width_jump_threshold)
                and (upstream_width_jump_perc >  self.config.upstream_width_jump_perc_threshold)
            )
                          
            if verbose:
                print(f"\tupstream_width_jump = {upstream_width_jump:.2f}, upstream_width_jump_perc = {upstream_width_jump_perc:.2f}  (min_upstream = {min_upstream}, width_flag = {width_flag})")
            
            sk_endpt_threshold = self.config.max_skeleton_endpoint_dist_threshold
            sk_endpt_flag = True
            if sk_endpt_threshold is not None and sk_endpt_threshold < np.inf:
                skeleton_endpoint_jump_dist = self.max_skeleton_endpoint_dist(limb,n)
                if skeleton_endpoint_jump_dist > sk_endpt_threshold:
                    sk_endpt_flag = False
                if verbose:
                    print(f"\tskeleton_endpoint_jump_dist = {skeleton_endpoint_jump_dist:.2f} (threshold = {sk_endpt_threshold:.2f}, flag = {sk_endpt_flag}")
            else:
                skeleton_endpoint_jump_dist = None
                
            if width_flag and sk_endpt_flag:
                description=f"upstream_width_jump ({upstream_width_jump:.2f}, {upstream_width_jump_perc:.2f}) " \
                f"compared to upstream ({min_upstream}) greater than threshold " \
                f"({self.config.upstream_width_jump_threshold:.2f},{self.config.upstream_width_jump_perc_threshold:.2f})," \
                f"\nsk_endpoint_dist ({skeleton_endpoint_jump_dist:.2f}) less than threshold ({sk_endpt_threshold:.2f})"
                if verbose:
                    print(f"Node {n}: {description}")
                error_nodes.append(gf.NodeError(n,description))
        if verbose:
            print(f"Final error nodes: {error_nodes}")
        return error_nodes
    @property
    def description(self):
        return_str = f"""
        Will error any downstream branches that have too much of a width increase ({self.config.width_threshold})
        compared to the parent branch, because natural dendrite morphohlogy usually decreases in
        width from parent to child branch
        """
        
# -- Filter 4 ---
class InternalBendDownstreamErrorFilter(gf.DownstreamErrorFilter):
    @dataclass(frozen = True)
    class Config:
        """
        double_back_threshold: float
            The maximum angle for an inflection point before an error occurs
        index_perc_buffer: float [0,1]
            Defines the allowable index window for a valid error to occur
            index_window = [index_perc_buffer,1-index_perc_buffer]
        min_distance_from_limb_start: float
            The minimum distance, either euclidean or skeletal distance
            of the internal bend from the start of the branch
        
        # -- skeleton bend angle parameters --
        skeleton_attribute:
            the type of skeleton used for computing bend angles
        bend_angle_window_size:
            The window size for computing the bend angle
        ignore_large_jump_idx:
            whether to ignore skeleton points with large skeletal jumps
        large_jump_window:
            How high to the set the ingore_indices windows around skeleton points with large jumps for computing bend angles
        

        """
        double_back_threshold: float = 80
        index_perc_buffer: float = 0.10
        min_distance_from_limb_start: float = 10_000
        skeleton_attribute: str = "skeleton_smooth"
        bend_angle_window_size: float = 100
        ignore_large_jump_idx: float = False
        large_jump_window: float = 35*3

    def bend_angle(self,limb_obj,n,verbose = False,**kwargs):
        branch = limb_obj[n]
        max_angle,index_perc,coordinate = bu.bend_max_on_branch_skeleton(
            branch,
            skeleton_attribute = self.config.skeleton_attribute,
            window_size = self.config.bend_angle_window_size,
            ignore_large_jump_idx = self.config.ignore_large_jump_idx,
            large_jump_window=self.config.large_jump_window,
            verbose = verbose,
            plot = False,
            return_index_perc=True,
            return_coordinate=True,
            return_dict=False,
        ) 
        return max_angle,coordinate,index_perc
    
    def distance_from_soma(self,limb,n,bend_coordinate,index_perc,verbose=False):
        path = nru.branch_path_to_soma(limb,n)
        skeletal_distance_to_soma = (
            np.sum([limb[k].skeletal_length for k in path[:-1]]) 
            + limb[n].skeletal_length * index_perc
        )

        euclidean_distance_to_soma = np.linalg.norm(
            limb.current_starting_coordinate - bend_coordinate
        )

        if verbose:
            print(f"skeletal_distance_to_soma = {skeletal_distance_to_soma}")
            print(f"euclidean_distance_to_soma = {euclidean_distance_to_soma}")
            
        return skeletal_distance_to_soma,euclidean_distance_to_soma
    

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose,**kwargs):
        error_nodes = []
        #for n in downstream_nodes:
        n = parent_node
        if verbose:
            print(f"--Working on parent node {n}:")
        max_angle,index_perc,coordinate = self.bend_angle(limb_obj,n,verbose=verbose,**kwargs)
        
        if index_perc is not None:
            idx_range = [self.config.index_perc_buffer,1-self.config.index_perc_buffer]

            if ((max_angle > self.config.double_back_threshold) and 
                (index_perc >= idx_range[0] and index_perc <= idx_range[1])):
                
                (skeletal_distance_to_soma,
                euclidean_distance_to_soma) = self.distance_from_soma(
                    limb_obj,
                    n,
                    bend_coordinate=coordinate,
                    index_perc=index_perc,
                    verbose = verbose
                )
                
                if ((skeletal_distance_to_soma < self.config.min_distance_from_limb_start)
                    and (euclidean_distance_to_soma < self.config.min_distance_from_limb_start)):
                        if verbose:
                            print(f"Both skeletal_distance_to_soma ({skeletal_distance_to_soma})"
                                f" and euclidean_distance_to_soma ({euclidean_distance_to_soma})"
                                f"of parent node closer than min_distance_from_limb_start ({self.config.min_distance_from_limb_start})"
                                f" so skipping error classification")
                else:
                    description = (
                        f"Had an internal bend ({max_angle:.2f}) greater than double_back_threshold ({self.config.double_back_threshold:.2f})"
                        f". Index_perc ({index_perc:.2f}) was within allowed range ({idx_range}). Coordinate = {coordinate} (skeletal_distance_to_soma = {skeletal_distance_to_soma}, euclidean_distance_to_soma = {euclidean_distance_to_soma})"
                    )
                    if verbose:
                        print(f"   Error: {description}")
                    error_nodes.append(gf.NodeError(n,description))
            
        if verbose:
            print(f"Final error nodes = {error_nodes}")
        return error_nodes

    @property
    def description(self):
        return f"""
        Looks for an internal bend of an angle greater than ({self.config.double_back_threshold:.2f}) becuase
        this usually indicates that an orphan merge has occured in an end to end fashion without creating
        a branching junction (because otherwise the double back filter would identify). 
        Also constrains filter to only those internal bends a certain percentage ({self.config.index_perc_buffer})
        of skeletal length away from the endpoints so that stitching edges don't accidentally set of filter
        """
    
from . import graph_filters_refactored as gf