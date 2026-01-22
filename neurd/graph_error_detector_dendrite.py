"""
# -- Instructions --

# -- Instruction 1: How to build DownstreamErrorDetector --
Step 0. Create a class inheriting from ged.DownstreamErrorDetector
Step 1. Define a Config subclassclass with the default parameters
Step 2. Define a __call__ function with the following criter
    Arguments:
        limb_obj
        parent_node
        downstream_nodes
    Returns:
        error_nodes: List[ged:NodeError]
Step 3. Create a description property

- Template

class (ged.DownstreamErrorDetector):
    @dataclass(frozen = True)
    class Config:
        '''
        Parameters:
            param1: type
                description
        '''
        
    
        

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose = False,**kwargs):
        error_nodes = []
        
        return error_nodes
        
    @property
    def description(self):
        return f"
    

# -- Instruction 2: How to build an ErrorDetector --
Purpose: These classes will generate an ErrorLimbBranch object
for their call functions using:
a. GraphScreenerConfig object
b. DownstreamErrorDetection object

Step 0. Create a class inheriting from ged.NeuronGraphErrorDetector
Step 1. Define Config subclass with the following items
    # -- DownstreamErrorDetector related --
    error_detector_cls: the DownstreamErrorDetector class you are using
    error_detector_kwargs: dict overrides to detectors default parameters
    # -- NeuronBranchLimbGraphScreenerConfig related --
    neuron_graph_config_kwargs: dict overrides to graph screener config default parameters
    branch_graph_config_kwargs: dict overrides to branch screener config default parameters
    graph_screener_kwargs: any other kwarg used in creation of the 
    # -- call function kwargs --
    generate_error_kwargs: dict for arguments that can be used during __call__
Step 2: NeuronGraphErrorDetector inheritance does the rest

Template:

class ClassName(ged.NeuronGraphErrorDetector):
    @dataclass
    class Config:
        # give it a name (used for logging / pipeline keys)
        name: str = 
        
        # Which downstream detector to use, plus its kwargs
        error_detector_cls: Type[ged.DownstreamErrorDetector] = 
        error_detector_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # instruction for How to build the graph screener
        neuron_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        branch_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # Any extra kwargs you want to pass into the screener build
        graph_screener_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # Any extra kwargs you want to pass when you actually call it
        # (verbose, branch_verbose, debug, etc)
        generate_error_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            verbose=False,
            branch_verbose=False,
            debug=False,
            error_detector_kwargs = dict(

            ),
            
        ))
    
"""

from dataclasses import dataclass
from . import (
    #axon_utils as au,
    #neuron_statistics as nst,
    #neuron_searching as ns,
    #concept_network_utils as cnu,
    limb_utils as lu,
    branch_utils as bu,
    graph_error_detector as ged,
    axon_utils as au,
    neuron_utils as nru,
    proofreading_utils as pru,
    error_detection as ed
)
import numpy as np
from dataclasses import dataclass, field
from typing import Type, Dict, Any

## -- #1 DownstreamErrorDetector 

class DendriteDoubleBackDownstreamErrorDetector(ged.DownstreamErrorDetector):
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
                error_nodes.append(ged.NodeError(n,description))
        return error_nodes
        
    @property
    def description(self):
        return f"Errors downstream node if parent angle greater than {self.config.double_back_threshold}"
    
dendrite_double_back_skip_distance = 9_000
class DendriteDoubleBackErrorDetector(ged.NeuronGraphErrorDetector):
    @dataclass
    class Config:
        # give it a name (used for logging / pipeline keys)
        name: str = "dendrite_double_back"
        
        # Which downstream detector to use, plus its kwargs
        error_detector_cls: Type[ged.DownstreamErrorDetector] = DendriteDoubleBackDownstreamErrorDetector
        error_detector_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            skeleton_attribute="skeleton_smooth",
            double_back_threshold=95,
            with_offset=True,
        ))
        
        # instruction for How to build the graph screener
        neuron_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            limb_branch_dict_func="dendrite_limb_branch_dict",
            filter_short_thick_endnodes=False,
            min_distance_from_soma_mesh=0,
            min_skeletal_length_endpoints=dendrite_double_back_skip_distance,
            min_upstream_skeletal_distance=dendrite_double_back_skip_distance,
        ))
        branch_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            skip_distance=dendrite_double_back_skip_distance,
            min_degree_to_resolve=1,
            width_func=au.axon_width,
            axon_dependent=False,
            width_min=0,
            max_skeleton_endpoint_dist=5_000,
            min_downstream_endnode_skeletal_dist_from_soma=10_000,
        ))
        
        # Any extra kwargs you want to pass into the screener build
        graph_screener_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())
        
        # Any extra kwargs you want to pass when you actually call it
        # (verbose, branch_verbose, debug, etc)
        generate_error_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            verbose=False,
            branch_verbose=False,
            debug=False,
            error_detector_kwargs=dict(
                verbose=False,
                angle_verbose=False,
            ),
        ))

    
## --- #2 DownstreamErrorDetector: CrossRoads  --
from dataclasses import dataclass

# @dataclass
# class CrossRoadsDownstreamErrorDetectorConfig:
#     angle_extra_offset: bool = True
#     sibling_angle_min: float = 140
#     width_extra_offset: bool = True
#     width_diff_perc_max: float = 0.2
#     width_diff_max: float = 100

class DendriteCrossRoadsDownstreamErrorDetector(ged.DownstreamErrorDetector):
    @dataclass(frozen=True)
    class Config:
        angle_extra_offset: bool = True
        sibling_angle_min: float = 140
        width_extra_offset: bool = True
        width_diff_perc_max: float = 1.5
        width_diff_from_min_directional_and_entire: bool = True
        width_diff_max: float = 100
        skeletal_length_downstream_min = 8_500
        
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
        width_diff = bu.width_diff_directional(
            limb_obj = limb,
            branch_1 = n1,
            branch_2 = n2,
            extra_offset = self.config.width_extra_offset,
            verbose = verbose,
            branch_1_dir = "upstream",
            branch_2_dir = "upstream",
            return_percentage = True,
        )
        
        if self.config.width_diff_from_min_directional_and_entire:
            entire_width_diff = bu.width_diff_entire_branch(
                limb_obj = limb,
                branch_1 = n1,
                branch_2 = n2,
                verbose = verbose,
                return_percentage = True
            )
            
            min_width_diff = min(width_diff,entire_width_diff)
            
            if verbose:
                print(f"For directional width_diff_perc ({width_diff:.2f}) and  entire width_diff_perc ({entire_width_diff:.2f}), min = {min_width_diff} ")
            return min_width_diff
        return width_diff

    def width_diff(self,limb,n1,n2,verbose=False):
        width_diff = bu.width_diff_directional(
            limb_obj = limb,
            branch_1 = n1,
            branch_2 = n2,
            extra_offset = self.config.width_extra_offset,
            verbose = verbose,
            branch_1_dir = "upstream",
            branch_2_dir = "upstream",
            return_percentage = False,
        )
        
        if self.config.width_diff_from_min_directional_and_entire:
            entire_width_diff = bu.width_diff_entire_branch(
                limb_obj = limb,
                branch_1 = n1,
                branch_2 = n2,
                verbose = verbose,
                return_percentage = False
            )
            
            min_width_diff = min(width_diff,entire_width_diff)
            
            if verbose:
                print(f"For directional width_diff ({width_diff:.2f}) and  entire width_diff ({entire_width_diff:.2f}), min = {min_width_diff} ")
            return min_width_diff
        return width_diff
            
            
        
        

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose = False,**kwargs):
        error_nodes = []
        sk_len_min = self.config.skeletal_length_downstream_min
        if sk_len_min is not None:
            downstream_nodes_too_small = [k for k in downstream_nodes
                                          if limb_obj[k].skeletal_length  < sk_len_min]
            if verbose:
                print(f"downstream_nodes_too_small = {downstream_nodes_too_small}")
            downstream_nodes = [k for k in downstream_nodes
                                if k not in downstream_nodes_too_small]
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
                    n_node = ged.NodeError(n,f"cross roads with {sib}: {stats_str}")
                    sib_node = ged.NodeError(sib,f"cross roads with {n}: {stats_str}")
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
    
cross_roads_skip_distance = 5_000
class DendriteCrossRoadsErrorDetector(ged.NeuronGraphErrorDetector):
    @dataclass
    class Config:
        # give it a name (used for logging / pipeline keys)
        name: str = "dendrite_cross_roads"
        
        # Which downstream detector to use, plus its kwargs
        error_detector_cls: Type[ged.DownstreamErrorDetector] = DendriteCrossRoadsDownstreamErrorDetector
        error_detector_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            sibling_angle_min = 140,
            width_diff_perc_max = 1.5,
        ))
        
        # instruction for How to build the graph screener
        neuron_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            limb_branch_dict_func = "dendrite_limb_branch_dict",
            filter_short_thick_endnodes = False,
            min_distance_from_soma_mesh = 0,#30_000,
            min_skeletal_length_endpoints = cross_roads_skip_distance,
            min_upstream_skeletal_distance = cross_roads_skip_distance,
        ))
        branch_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            skip_distance = cross_roads_skip_distance,
            min_degree_to_resolve = 2,
            width_func = au.axon_width,

            axon_dependent = False,
            min_downstream_endnode_skeletal_dist_from_soma = 10_000,
            width_min = 0,
        ))
        
        # Any extra kwargs you want to pass into the screener build
        graph_screener_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # Any extra kwargs you want to pass when you actually call it
        # (verbose, branch_verbose, debug, etc)
        generate_error_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            verbose = False,
            branch_verbose = False,
            debug = False,
            error_detector_kwargs = dict(
                verbose = False,
            )
            
        ))
        
    def debug_vectors(
        self,
        limb,
        branches,
        parent_idx = None,
        verbose = True,
        ):
        
        if verbose:
            for b in branches:
                branch = limb[b]
                print(f"{b} vector = {branch.skeleton_smooth_vector_upstream}, skeletal_length = {branch.skeletal_length}")
                
        nviz.plot_cross_roads_vectors(
            limb = limb,
            branches = branches,
            parent_idx = parent_idx,
            verbose = verbose,
            plot_upstream_mesh = True,
        )
    
## -- DownstreamErrorDetector 3
class DendriteWidthJumpDownstreamErrorDetector(ged.DownstreamErrorDetector):
    @dataclass(frozen=True)
    class Config:
        """
        Parameters:
            upstream_width_jump_threshold: float
                minimum raw nm width increase from upstream minimum required for an error
            upstream_width_jump_perc_threshold: float
                minimum percentage width increase from upstream minimum required for an error
            max_skeleton_endpoint_dist_threshold:
                The maximum distance the endpoint of the skeleton can be from the mesh in order for an error to be found (helps filter away false positives)


            require_screened_graph: bool
                Whether DownstreamErrorDetector needs the 
            ignore_skipped_nodes_for_width: bool
                Whether to ignore nodes that were skipped in graph screener step when calculating all the upstream widths
            min_skeletal_length_for_upstream_path: float
                minimum skeletal lenght an upstream node needs to have for width to be considered in determining minimum upstream width
            remove_first_branch_for_upstream_path: bool
                whether the limb starting node should be removed from upstream list that upstream minimum width is determined from
        
        """
        upstream_width_jump_threshold: float = 190
        upstream_width_jump_perc_threshold: float = 0.4
        max_skeleton_endpoint_dist_threshold: float = np.inf

        # for computing the nodes to compare the width to
        require_screened_graph: bool = True
        ignore_skipped_nodes_for_width: bool = True
        min_skeletal_length_for_upstream_path: float = 5000
        max_skeleton_endpoint_dist_for_upstream_path: float = 4500
        remove_first_branch_for_upstream_path: bool = True

    def max_skeleton_endpoint_dist(self,limb_obj,n):
        return limb_obj[n].max_skeleton_endpoint_dist
        
    @property
    def branch_width_func(self):
        return bu.width_min
    
    @property
    def upstream_width_func(self):
        return bu.width_max
        
    def upstream_width_min(
        self,
        limb_obj,
        n,
        screened_graph=None,
        verbose= False,
        nodes_to_ignore = None,
        width_func = None,
        remove_zeros = True,
        default = 1000000,
        return_min_upstream = True):
        """
        
        """
        if width_func is None:
            width_func = self.upstream_width_func
            
        if self.config.ignore_skipped_nodes_for_width:
            nodes_to_ignore = screened_graph.skipped_nodes_too_small_skeleton
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
            width_func = width_func,
            return_branch_path = True,
            )
        
        width_path,branch_path = np.array(width_path),np.array(branch_path)
        
        max_endpt = self.config.max_skeleton_endpoint_dist_for_upstream_path
        if max_endpt is not None and max_endpt < np.inf:
            endpt_dist = np.array([limb_obj[k].max_skeleton_endpoint_dist for k in branch_path])
            endpt_mask = endpt_dist < max_endpt
            width_path,branch_path = width_path[endpt_mask],branch_path[endpt_mask]
            if verbose:
                print(f"After filtering by endpoint_dist: branch_path = {branch_path},width_path = {width_path} ")

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
        screened_graph,
        verbose = False,**kwargs):
        up_width_min,min_upstream = self.upstream_width_min(
            limb_obj,
            n,
            screened_graph=screened_graph,
            verbose= verbose
        )
        branch_width = self.branch_width_func(limb_obj[n])
        width_jump = branch_width - up_width_min
        width_jump_perc = width_jump/up_width_min
        if verbose:
            print(f"width_jump = {width_jump:.2f} (min_upstream = {min_upstream})")
        return width_jump,width_jump_perc,min_upstream
    
    def __call__(
        self,
        limb_obj,
        downstream_nodes,
        screened_graph,
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
                limb,n,screened_graph=screened_graph,verbose = verbose_width_jump,**kwargs
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
                    print(f"\tskeleton_endpoint_jump_dist = {skeleton_endpoint_jump_dist:.2f} (threshold = {sk_endpt_threshold:.2f}, flag = {sk_endpt_flag})")
            else:
                skeleton_endpoint_jump_dist = None
                
            if width_flag and sk_endpt_flag:
                description=f"upstream_width_jump ({upstream_width_jump:.2f}, {upstream_width_jump_perc:.2f}) " \
                f"compared to upstream ({min_upstream}) greater than threshold " \
                f"({self.config.upstream_width_jump_threshold:.2f},{self.config.upstream_width_jump_perc_threshold:.2f})," \
                f"\nsk_endpoint_dist ({skeleton_endpoint_jump_dist:.2f}) less than threshold ({sk_endpt_threshold:.2f})"
                if verbose:
                    print(f"Node {n}: {description}")
                error_nodes.append(ged.NodeError(n,description))
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
        
dend_width_jump_skip_distance = 9_000
class DendriteWidthJumpErrorDetector(ged.NeuronGraphErrorDetector):
    @dataclass
    class Config:
        # give it a name (used for logging / pipeline keys)
        name: str = "dendrite_width_jump"
        
        # Which downstream detector to use, plus its kwargs
        error_detector_cls: Type[ged.DownstreamErrorDetector] = DendriteWidthJumpDownstreamErrorDetector
        error_detector_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            upstream_width_jump_threshold= 190,
            upstream_width_jump_perc_threshold = 0.4,
            max_skeleton_endpoint_dist_threshold= 5000,
            
            ignore_skipped_nodes_for_width = True,
            min_skeletal_length_for_upstream_path = 13000,
        ))
        
        # instruction for How to build the graph screener
        neuron_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            limb_branch_dict_func = "dendrite_limb_branch_dict",
            filter_short_thick_endnodes = False,
            min_distance_from_soma_mesh = 0,
            min_skeletal_length_endpoints = dend_width_jump_skip_distance,
            min_upstream_skeletal_distance = dend_width_jump_skip_distance,
            
        ))
        branch_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            skip_distance = dend_width_jump_skip_distance,
            min_degree_to_resolve = 1,
            width_func = au.axon_width,
            axon_dependent = False,
            width_min = 0,
            max_skeleton_endpoint_dist = np.inf,
            min_downstream_endnode_skeletal_dist_from_soma = 30_000,
        ))
        
        # Any extra kwargs you want to pass into the screener build
        graph_screener_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # Any extra kwargs you want to pass when you actually call it
        # (verbose, branch_verbose, debug, etc)
        generate_error_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            verbose=False,
            branch_verbose=False,
            debug=False,
            error_detector_kwargs = dict(
                verbose = False,
                verbose_width_jump = False,
            ),
            
        ))
    

    def debug_branch_path_stats(self,limb,branch_idx):
        branch_width = self.error_detector.branch_width_func(limb[branch_idx])
        for b_idx in nru.branch_path_to_soma(limb,branch_idx):
            obj = limb[b_idx]
            up_width_min = self.error_detector.upstream_width_func(obj)
            width_jump = branch_width - up_width_min
            width_jump_perc = width_jump/up_width_min
            print(f"{b_idx}: max_endpt_dist = {obj.max_skeleton_endpoint_dist:.2f}, width = {up_width_min:.2f}, sk_length = {obj.skeletal_length:.2f}, width_jump = {width_jump:.2f}, width_jump_perc = {width_jump_perc:.2f}")
    
        
# -- DownstreamDetector 4 ---
class DendriteInternalBendDownstreamErrorDetector(ged.DownstreamErrorDetector):
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
    

    def __call__(self,limb_obj,parent_node,downstream_nodes,verbose=False,**kwargs):
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
                    error_nodes.append(ged.NodeError(n,description))
            
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
        
end_to_end_double_back_skip_distance = 15_000
class DendriteInternalBendErrorDetector(ged.NeuronGraphErrorDetector):
    @dataclass
    class Config:
        # give it a name (used for logging / pipeline keys)
        name: str = "dendrite_internal_bend"
        
        # Which downstream detector to use, plus its kwargs
        error_detector_cls: Type[ged.DownstreamErrorDetector] = DendriteInternalBendDownstreamErrorDetector
        error_detector_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            double_back_threshold = 80,
            index_perc_buffer = 0.1
        ))
        
        # instruction for How to build the graph screener
        neuron_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            limb_branch_dict_func = "dendrite_limb_branch_dict",
            filter_short_thick_endnodes = False,
            min_distance_from_soma_mesh = 0,
            min_skeletal_length_endpoints = dendrite_double_back_skip_distance,
            min_upstream_skeletal_distance = dendrite_double_back_skip_distance,
        ))
        branch_graph_config_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            skip_distance = dendrite_double_back_skip_distance,
            min_degree_to_resolve = 0,
            width_func = au.axon_width,

            axon_dependent = False,
            width_min = 0,

            min_downstream_endnode_skeletal_dist_from_soma = 10_000,
        ))
        
        # Any extra kwargs you want to pass into the screener build
        graph_screener_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            
        ))
        
        # Any extra kwargs you want to pass when you actually call it
        # (verbose, branch_verbose, debug, etc)
        generate_error_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
            verbose=False,
            branch_verbose=False,
            debug=False,
            error_detector_kwargs = dict(
                verbose = True,
            )
            
        ))
        
  
    
# --- original m65 microns filters --
class ExcDendriteWidthJumpErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "exc_dendrite_width_jump"
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        return ed.width_jump_up_dendrite(
            neuron_obj,**kwargs
        )
        
class ExcDendriteDoubleBackErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "exc_dendrite_double_back"
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        return ed.double_back_dendrite(
            neuron_obj,**kwargs
        )
        
class InhDendriteDoubleBackErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "inh_dendrite_double_back"
    
    def error_limb_branch_dict(
        self,
        neuron_obj,
        double_back_threshold = None,
        **kwargs):
        
        if double_back_threshold is None:
            double_back_threshold = pru.double_back_threshold_inh_double_b_global
        
            
        return ed.double_back_dendrite(
            neuron_obj,
            double_back_threshold=double_back_threshold,
            **kwargs
        )
    
from . import (
    graph_error_detector_dendrite as gedd,
    neuron_visualizations as nviz,
    neuron_utils as nru,
)