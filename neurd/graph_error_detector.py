"""
Classes and functions for refactored infrastructure for running
graph filters of different types on a neuron object. Reuse
a lot of the same infrastructure created for high-degree merges
and allow more metadata to be saved off

Application: Use for smarter dendrite filters
"""


import numpy as np
from collections import UserDict     
from dataclasses import dataclass, field
from typing import List,Callable,Any
from copy import deepcopy



from . import (
     axon_utils as au,
    neuron_statistics as nst,
    neuron_searching as ns,
    concept_network_utils as cnu,
    limb_utils as lu,
    branch_utils as bu,
    error_detection as ed,
    graph_error_detector as ged,
    
    #neuron_utils as nru,
)

old_high_degree_pseudocode = """
Pseudocode
----------
#ed.high_degree_branch_errors_dendrite_limb_branch_dict
1. too close to soma filter (too_close): Get branches that are too close to soma
    param: 
        min_distance_from_soma_mesh
    plot: plot_soma_restr,
    - nodes to exclude from search
    
2. too short endpoints (axon_spines): Get endpoints that are too short
    param: 
        min_skeletal_length_endpoints
    plot: plot_endpoints_filtered
    - nodes to exclude from downstream
    
3. filter short thick branches
     filter_short_thick_endnodes (bool)
     - nodes to exclude from downstream

#ed.high_degree_branch_errors_limb_branch_dict
4. Start iterating through limbs and branches to search over:
    a. Save all the nodes to exclude from downstream in the limb object
    b. For each node do the high_degree_upstream_match
        min_upstream_skeletal_distance
        skip_distance

    #ed.high_degree_upstream_match  
    i. preprocesses the limb to determine:
        a. if the parent node should be processed
        b. What the appropriate downstream branches should be

        #ed.high_low_degree_upstream_match_preprocessing
        1. short thick endnodes:
            - skips if parent is in list
            - adds nodes to nodes_to_exclude
        2. axon spines: 
            - skips if parent is in list
            - adds nodes to nodes_to_exclude
        3. if skeletal length was less than threshold then skips parent
            param:
                min_upstream_skeletal_distance
        4. skips if parent is a minimum distance from the soma
            param:
                min_distance_from_soma_for_proof
        5. Returns if already determined parent will be skipped
        6. If skip_distance is None, calculates it:
            #ed.calculate_skip_distance
            param:
                skip_distance
        7. Determines all the downstream branches 
        8. Filters downstream branches:
            Removes short_thick_endnodes 
            Removes axon_spines
        9. Returns if number of branches is below threshold
            param: 
                min_degree_to_resolve
        10. Returns all downstream branches as errors if number of dwonstream
            is greater than thresholdd
            param:
                max_degree_to_resolve_absolute
        11. Width maximum: Skips parent branch if with above maximum
            param:
                upstream_width_max
                width_func
        12. If width is above threshold, increases the max_degree_to_resolve to 
                the wide version
            param:
                max_degree_to_resolve_width_threshold
                max_degree_to_resolve_wide
        13. If number of downstream branches above max_degree_to_resolve
            returns all downstream branches as errors
        14. Calculates the width off the parent and all downstream branches
        15. If all the widths in parent+downstream are greater than threshold,
            skips parent
            param: 
                width_max
        16. axon_dependent (False for dendrite): skips parenet if any 
            downstream branch is not an axon
            param:
                axon_dependent
        17. upstream_wdith_min: if branch width (axon_width) is below threshold then skip

When skipped parent: return_value = [None,np.array([])]
When errored out all downstream: return_value = [None,downstream_branches]
            
                

   --------- Want the preprocessing object to do up to this -----
    ii. If valid parent branch with downstream branches to process
        a. Runs the filter to determine winning branch, error branches
"""

limb_screener_graph_doc = """
Purpose
-------
Create code that will input a limb object and parameters for a certain filter
and then do all the preprocessing necessary so that for each node will determine:
1. If it should be tested for that filter
2. what the corresponding downstream branches should be for the filter

Application
-----------
Can then use this object with different filter classes to perform
different specific types of filters:
1. width jump
2. double back
3. high degree

Goal
----
going to reimplement all of the dendrite filters

Revised Pseudocode
------------------

Object Oriented Solution

What we want to determine:
a. if the parent node should be processed
b. What the appropriate downstream branches should be

Persistent Attributes:
1. parents_to_skip (computed from precomputed list)
2. downstream_to_exclude (computed from precomputed)
3. Mapping of parent to downstream (should have value if want to skip)
    Should include reason of why skipped or why errored all
    3b. Derivative mapping of parent to "should_process"


Methods: 
-- preprocess function --

1. Feature: Determine branches_to_skip:
    a. Find all short_thick_endpoints (if requested)
        param: 
            filter_short_thick_endnodes (bool)
    b. find all axon spines (short thick)
        param:
            min_skeletal_length_endpoints
2. Feature: Determine parents to skip:
    a. Include all branches to skip list
    b. Find all branches too close to the soma 
        param: 
            min_distance_from_soma_mesh
    c. find all branches with skeleton length too low
        param:
            min_upstream_skeletal_distance

For each branch:
    0. Determine if should skip because of parent to skip list
    1. Feature: Calculate skip distance if not set
        param:
            skip_distance
    2. Feature: Determine all downstream branches using skip distance
    - filter away branches_to_skip
    3. Skip parent if len(down) < threshold
        param:
            min_degree_to_resolve
    4. Error all downstream branches is len(down)>max_degree_to_resolve_absolute
        param:
            max_degree_to_resolve_absolute
    5. Feature: Calculate parent width
        param:
            width_func
    5. Skip parent if width too large
        param:
            upstream_width_max
    6. Param change: If width is above threshold, 
        increases the max_degree_to_resolve to the wide version
        param:
            max_degree_to_resolve
            max_degree_to_resolve_width_threshold
            max_degree_to_resolve_wide
    
    7. Error all: If number of downstream branches above max_degree_to_resolve
                returns all downstream branches as errors
        param:
            max_degree_to_resolve
    8. Feature: Calculates the width off the parent and all downstream branches
    9. Skip parent:If all the widths in parent+downstream are greater than threshold
        param:
            width_max
    10: Skip parent: if any downstream branches not axon
        param:
            axon_dependent
    11. Skip parent: if branch_width is below threshold
        param:
            width_min

Final output:
- set branches_to_skip
- set parents_to_skip
- set branch_downstream_object mapping for each node:
    * skipped
    * error all
    * downstream branches
    * reason of determination   
"""


## ---- Classes for preprocessing graph --
from dataclasses import dataclass, field,asdict

# @dataclass
# class NeuronGraphFilterConfig:
#     # what branches to search over
#     limb_branch_dict_func: Callable[..., Any] = None
#     # short_thick_endnodes params
#     filter_short_thick_endnodes: bool = False
#     # too_close_limb_branch params
#     min_distance_from_soma_mesh: int = 0
#     plot_soma_restr: bool = False
#     # spines params
#     min_skeletal_length_endpoints: int = 0
#     plot_endpoints_filtered:bool = False


# from typing import Union, List,Callable,Any
# import numpy as np

# #ListOrArray: TypeAlias = Union[List[int], np.ndarray]

# @dataclass
# class LimbGraphFilterConfig:
#     """
#     Attributes:
#     -----------
#     branches_to_process: list
#         The nodes that will be checked as a parent (derived from limb branch dict)
#     short_thick_endpoint_nodes: list
#         Nodes that were computed using the au.short_thick_branches_limb_branch_dict
#         but only if the filter_short_thick_endnodes flag was set
    
#     """
#     branches_to_process: Union[List[int], np.ndarray] = None
#     short_thick_endpoint_nodes: Union[List[int], np.ndarray]=None
#     too_close_to_soma_nodes: Union[List[int], np.ndarray] = None
#     spine_nodes: Union[List[int], np.ndarray] = None

#     # -- limb scope parameters -- 
#     min_upstream_skeletal_distance:float = 0

# inf = 10000000
# @dataclass
# class BranchGraphFilterConfig:
#     # -- branch scope preprocessing parameters --
#     skip_distance:float = 0
#     min_degree_to_resolve:int = 1
    
#     max_degree_to_resolve_absolute:int = inf
    
#     width_func: Callable[..., Any] = au.axon_width
#     upstream_width_max:float = inf

#     max_degree_to_resolve:int = inf
#     max_degree_to_resolve_width_threshold:float = inf
#     max_degree_to_resolve_wide:int = inf

#     width_max:float = inf

#     axon_dependent:bool = False

#     width_min:float = 0

@dataclass
class NeuronGraphScreenerConfig:
    """
    parameters for whole neuron queries

    Attributes
    ----------
    limb_branch_dict_func: Union[func,str]
        function (or string referring to neuron_obj attr) that will return limb_branch to search
    filter_short_thick_endnodes:
        Whether to filter away downstream nodes returned by `au.short_thick_branches_limb_branch_dict()`.
        Stored in `LimbGraphScreenerConfig.short_thick_endpoint_nodes`
    min_distance_from_soma_mesh: float
        The minimum distance away a parent node needs to be to be considered (measured from it's upstream skeleton endpoint). If you want to constrain by distance from downstream skeleton endpoint then use `branch_graph_config.min_downstream_endnode_skeletal_dist_from_soma`.
        Stored in `LimbGraphScreenerConfig.too_close_to_soma_nodes`
    min_skeletal_length_endpoints:
        The maximum skeletal length of an downstream node that is filtered away (used for axon spines)
        Stored in `LimbGraphScreenerConfig.spine_nodes`
    min_upstream_skeletal_distance:
        The minimum skeletal length a parent node needs to be to be considered 
        Stored in `LimbGraphScreenerConfig.too_small_skeleton`
    """
    # what branches to search over
    limb_branch_dict_func: Callable[..., Any] = None
    # short_thick_endnodes params
    filter_short_thick_endnodes: bool = False
    # too_close_limb_branch params
    min_distance_from_soma_mesh: float = 0
    plot_soma_restr: bool = False
    # spines params
    min_skeletal_length_endpoints: float = 0
    plot_endpoints_filtered:bool = False
    # too_small skeleton params
    min_upstream_skeletal_distance:float = 0

from typing import Union, List,Callable,Any
import numpy as np

#ListOrArray: TypeAlias = Union[List[int], np.ndarray]

@dataclass
class LimbGraphScreenerConfig:
    """
    parameters for controlling parent and downstream nodes considered

    Attributes
    ----------
    branches_to_process : list[int]
        Nodes that will be treated as “parent” candidates, derived
        from the limb’s branch dictionary.
    short_thick_endpoint_nodes : list[int]
        Nodes returned by `au.short_thick_branches_limb_branch_dict()`.
        Only populated if the `filter_short_thick_endnodes` flag is set.
    too_close_to_soma_nodes: list[int]
        Nodes returned by the query for nodes with a path length of `min_distance_from_soma_mesh` of soma
    spine_nodes: list[int]
        Nodes return by query for endnodes with skeletal length less than `min_skeletal_length_endpoints`
    too_small_skeleton: list[int]
        Nodes returns by query for those with skeletal length less than `min_upstream_skeletal_distance`
    """
    
    branches_to_process: Union[List[int], np.ndarray] = None
    short_thick_endpoint_nodes: Union[List[int], np.ndarray]=None
    too_close_to_soma_nodes: Union[List[int], np.ndarray] = None
    spine_nodes: Union[List[int], np.ndarray] = None
    too_small_skeleton: Union[List[int], np.ndarray] = None
    

inf = np.inf
@dataclass
class BranchGraphScreenerConfig:
    """
    parameters controlling which parent nodes are:
    i. skipped
    2. error all: have all downstream nodes 
    3. Have its downstream nodes analyze by an error filter function for a subset being errors

    Attributes
    ----------
    skip_distance : float
        Skeletal length of downstream nodes that are skipped and have their children added as downstream nodes
    min_degree_to_resolve: int
        The minimum number of children a parent node needs to be considered by error_detector (or else skipped)
    max_degree_to_resolve_absolute: int
        The maximum number of children a parent node can have or else all downstream nodes errored (error_all)
    width_func: function
        Function that is called on a branch object to determine its width (typically `au.axon_width`)
    upstream_width_max: float
        The maximum width of a parent branch to be considered by error_detector (or else skipped)
    
    max_degree_to_resolve: int
        The maximum number of children a parent node can have or else all downstream nodes errored (error_all).
        But this number can be shifted if the parent width is high enough
    max_degree_to_resolve_width_threshold: float
        The maximum width of a parent branch before `max_degree_to_resolve` is changed to `max_degree_to_resolve_wide`
    max_degree_to_resolve_wide: float
        The new `max_degree_to_resolve` if the parent width is above `max_degree_to_resolve_width_threshold`

    width_max: float
        If all parent and downstream node widths are greater than `width_max`, then parent is skipped
    axon_dependent: bool
        If all downstream nodes are not an axon, then parent is skipped
    width_min: float
        Minimum width a parent node must be to be considered by error_detector (or else skipped)
        
    max_skeleton_endpoint_dist: float
        Maximum amount of a distane a skeleton endpoint can be from a mesh
        to control for nodes that may be ill-posed
        
    min_downstream_endnode_skeletal_dist_from_soma: float
        the Maximum skeletal distance the downstream endpoint of a branch can be
        for its downstream nodes to be considered
    """
    # -- branch scope preprocessing parameters --
    skip_distance:float = 0
    min_degree_to_resolve:int = 1
    
    max_degree_to_resolve_absolute:int = inf
    
    width_func: Callable[..., Any] = au.axon_width
    upstream_width_max:float = inf

    max_degree_to_resolve:int = inf
    max_degree_to_resolve_width_threshold:float = inf
    max_degree_to_resolve_wide:int = inf

    width_max:float = inf

    axon_dependent:bool = False

    width_min:float = 0
    
    max_skeleton_endpoint_dist: float = inf
    
    min_downstream_endnode_skeletal_dist_from_soma: float = 0
    
class NeuronBranchLimbGraphScreenerConfig:
    """
    Bundle together the three sub-configs needed by NeuronGraphScreener.
    """
    def __init__(self,
                 neuron_graph_config_kwargs = None,
                 branch_graph_config_kwargs= None,
                 limb_graph_config_kwargs=   None,
                 neuron_graph_config = None,
                 branch_graph_config = None,
                 limb_graph_config = None,
                 ):
        # avoid mutable defaults
        ngk = neuron_graph_config_kwargs or {}
        lgk = limb_graph_config_kwargs   or {}
        bgk = branch_graph_config_kwargs or {}
        
        if neuron_graph_config is None:
            neuron_graph_config = NeuronGraphScreenerConfig(**ngk)
        if  limb_graph_config is None:
            limb_graph_config = LimbGraphScreenerConfig(**lgk)
        if branch_graph_config is None:
            branch_graph_config = BranchGraphScreenerConfig(**bgk) 
        self.neuron_graph_config =   neuron_graph_config
        self.limb_graph_config    = limb_graph_config
        self.branch_graph_config  = branch_graph_config

    @property
    def graph_screener_config_dict(self) -> dict:
        return {
            "neuron_config": self.neuron_graph_config,
            "limb_config":   self.limb_graph_config,
            "branch_config": self.branch_graph_config,
        }

    
@dataclass
class BranchNode:
    downstream_nodes: List[int] = field(default_factory=list)
    skipped_downstream_nodes: List[int] = field(default_factory=list)
    removed_downstream_nodes: List[int] = field(default_factory=list)
    skipped: bool = False
    error_all: bool = False
    skipped_reason: str = None
    error_all_reason: str = None

    def is_skipped(self):
        return self.skipped

    def __repr__(self):
        if self.skipped:
            return f"skipped ({self.skipped_reason})"
        elif self.error_all:
            return f"error all ({self.error_all_reason}), errors = {self.downstream_nodes}"
        else:
            return_str = f"{self.downstream_nodes}"
            if len(self.skipped_downstream_nodes)>0:
                return_str += f", skipped nodes={self.skipped_downstream_nodes}"
            if len(self.removed_downstream_nodes)>0:
                return_str += f", removed nodes={self.removed_downstream_nodes}"
            return return_str

import inspect

class NeuronGraphScreener(UserDict):
    def __init__(
        self,
        neuron_obj,
        neuron_config = None,
        limb_config = None,
        branch_config = None,
        verbose_limb = False,
        verbose_branch = False,
        debug_branch = False,
        
    ):
        """
        Purpose
        -------
        Stores all the skipping,erroring results for a neuron of the prescreening
        before any filters are run (aka deciding where the filters on entire neuron should even run on)
        
        Main Use: To convert a neuron object into an object that stores for every limb what branches should be skipped (based on the filtering config)
        and what are the downstream branches for all branches (for the filter to later run on)

        Parameters
        ----------
        neuron_obj : _type_
            _description_
        neuron_config : _type_, optional
            _description_, by default None
        limb_config : _type_, optional
            _description_, by default None
        branch_config : _type_, optional
            _description_, by default None
        verbose_limb : bool, optional
            _description_, by default False
        verbose_branch : bool, optional
            _description_, by default False
        debug_branch : bool, optional
            _description_, by default False
        """
        if neuron_config is None:
            neuron_config = NeuronGraphScreenerConfig()
        self.neuron_config = neuron_config
        self.limb_config = limb_config
        self.branch_config = branch_config
        
        
        config = self.neuron_config
        
        # Determining the maximum sized limb_branch_dict to search over this neuron
        if config.limb_branch_dict_func is None:
            self.limb_branch_dict = neuron_obj.limb_branch_dict
        elif isinstance(config.limb_branch_dict_func,str):
            self.limb_branch_dict = getattr(neuron_obj,config.limb_branch_dict_func)
        else:
            self.limb_branch_dict = config.limb_branch_dict_func(neuron_obj)

        # using the global neuron config for skipping info on the entire neuron that will eventually be unpacked into limb_screener
        # -- find all the neuron wide limb branches to use as filters --
        if config.filter_short_thick_endnodes:
            short_thick_endnodes_to_remove_limb_branch = au.short_thick_branches_limb_branch_dict(
                neuron_obj,
                verbose = False
            )
        else:
            short_thick_endnodes_to_remove_limb_branch = {}
        
        if (config.min_distance_from_soma_mesh is not None 
            and config.min_distance_from_soma_mesh>0):
            limb_branch_too_close_limb_branch = nst.euclidean_distance_close_to_soma_limb_branch(
                neuron_obj,
                distance_threshold=config.min_distance_from_soma_mesh,
                plot = config.plot_soma_restr
            )  
        else:
            limb_branch_too_close_limb_branch = {}
        
        if (config.min_skeletal_length_endpoints is not None 
            and config.min_skeletal_length_endpoints > 0):
            axon_spines_limb_branch = ns.query_neuron(
                neuron_obj,
                query = (
                    f"(skeletal_length < {config.min_skeletal_length_endpoints})"
                    " and (n_downstream_nodes == 0)"),
                limb_branch_dict_restriction=self.limb_branch_dict,
                plot_limb_branch_dict=config.plot_endpoints_filtered
            )
        else:
            axon_spines_limb_branch  = {}
            
        if (config.min_upstream_skeletal_distance is not None 
            and config.min_upstream_skeletal_distance > 0):
            too_small_skeleton_limb_branch = ns.query_neuron(
                neuron_obj,
                query = (
                    f"(skeletal_length < {config.min_upstream_skeletal_distance})"
                ),
                limb_branch_dict_restriction=self.limb_branch_dict,
            )
        else:
            too_small_skeleton_limb_branch  = {}
        
        
        self.short_thick_endnodes_to_remove_limb_branch = short_thick_endnodes_to_remove_limb_branch
        self.limb_branch_too_close_limb_branch = limb_branch_too_close_limb_branch
        self.axon_spines_limb_branch = axon_spines_limb_branch
        self.too_small_skeleton_limb_branch = too_small_skeleton_limb_branch

        if limb_config is None:
            limb_config_default = LimbGraphScreenerConfig()
        else:
            limb_config_default = limb_config
            
        
        self.data = {}
        for limb_idx in self.limb_branch_dict:
            # -- unbacks all the skipping information from the neuron_config and stores in the limb_config
            limb_config = deepcopy(limb_config_default)
                
            if verbose_limb:
                print(f"-- Working on limb {limb_idx} --")
            limb_config.branches_to_process = self.limb_branch_dict[limb_idx]
            limb_config.short_thick_endpoint_nodes = self.short_thick_endnodes_to_remove_limb_branch.get(limb_idx,None)
            limb_config.too_close_to_soma_nodes =self.limb_branch_too_close_limb_branch.get(limb_idx,None)
            limb_config.spine_nodes = self.axon_spines_limb_branch.get(limb_idx,None)
            limb_config.too_small_skeleton = self.too_small_skeleton_limb_branch.get(limb_idx,None)
            
            # -- 
            G = LimbGraphScreener(
                limb = neuron_obj[limb_idx],
                config = limb_config,
                branch_config = branch_config,
                verbose = verbose_limb,
                verbose_branch = verbose_branch,
                debug_branch = debug_branch,
            )
            
            self.data[limb_idx] = G
    def __repr__(self):
        total_str = []
        for limb_idx,limbG in self.data.items():
            total_str.append(f"Limb {limb_idx}")
            for b,b_obj in limbG.items():
                total_str.append(f"\t{b}:{b_obj}")
        return "\n".join(total_str)
            

class LimbGraphScreener(UserDict):
    def __init__(
        self,
        limb,
        config=None,
        branch_config=None,
        verbose = False,
        verbose_branch = False,
        debug_branch = False,
        **kwargs
        ):
            
        self.limb = limb
        if config is None:
            config = LimbGraphScreenerConfig()
        if branch_config is None:
            branch_config = BranchGraphScreenerConfig()
            
        
        self.short_thick_endpoint_nodes = config.short_thick_endpoint_nodes
        self.too_close_to_soma_nodes = config.too_close_to_soma_nodes
        self.spine_nodes = config.spine_nodes
        self.too_small_skeleton = config.too_small_skeleton
            
        self.preprocess_kwargs = kwargs
        self.branches_to_skip_attributes = [
            "short_thick_endpoint_nodes",
            "spine_nodes",
        ]
        self.parents_to_skip_attributes = [
            "too_close_to_soma_nodes",
            "too_small_skeleton",
        ] + self.branches_to_skip_attributes

        if config.branches_to_process is None:
            config.branches_to_process = limb.get_branch_names()
        self.branches_to_process = list(config.branches_to_process)
        self.branch_map = {}
        
        # store the config
        self.config = config
        self.branch_config = branch_config
        self.preprocess(
            verbose = verbose,
            verbose_branch = verbose_branch,
            debug_branch = debug_branch,
            **kwargs)
        self.data = self.branch_map
        
    def __repr__(self):
        total_str = []
        for b,b_obj in self.items():
            total_str.append(f"{b}:{b_obj}")
        return "\n".join(total_str)

    def combine_attr_list(self,attr_list):
        att_vals = []
        for att in attr_list:
            att_value = getattr(self,att,None)
            if att_value is not None:
                att_vals += list(att_value)
        return set(att_vals)
        
    @property
    def branches_to_skip(self):
        return self.combine_attr_list(self.branches_to_skip_attributes)
    @property
    def parents_to_skip(self):
        return self.combine_attr_list(self.parents_to_skip_attributes)

    def preprocess(
        self,
        verbose = False,
        verbose_branch = False,
        debug_branch = False,
        **kwargs):
        """
        Purpose
        -------
        Builds out the self.branch_map with branch objects that indicate for each branch on the limb
        if the branch:
        a. should be skipped 
        b. if not skipped, what are the relevant downstream nodes
        """
        """
        1. Feature: Determine branches_to_skip:
            a. add all short_thick_endpoints (if requested)
                param: 
                    filter_short_thick_endnodes (bool)
            b. find all axon spines (short thick)
                param:
                    min_skeletal_length_endpoints
        """
        config = self.config
        branch_config = self.branch_config
        
        """
        2. Feature: Determine parents to skip:
            a. Include all branches to skip list
            b. Find all branches too close to the soma 
                param: 
                    min_distance_from_soma_mesh
            c. find all branches with skeleton length too low
                param:
                    min_upstream_skeletal_distance
        """
        # Iterates over all the branches of the limb and creates an object for each that informs if branch:
        # a. is skipped (and for what reason)
        # b. what downstream branches should be used in the filter
        for branch_idx in self.limb.get_branch_names():
            self.branch_map[branch_idx] = self.preprocess_branch(
                branch_idx,
                verbose = verbose_branch,
                debug = debug_branch,
                **kwargs)
    
    def preprocess_branch(
        self,
        branch_idx,
        verbose = False,
        debug = False,
        **kwargs
    ):
        """
        determines whether a specific branch should be skipped or not
        and if not skipped then what the downstream branches should be when running the filter

        Parameters
        ----------
        branch_idx : _type_
            _description_
        verbose : bool, optional
            _description_, by default False
        debug : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        
        config = self.branch_config
        config = deepcopy(config)
        if verbose or debug:
            print(f"-- preprocssing branch {branch_idx} --")
        def in_list(mylist,value):
            if mylist is None:
                return False
            else:
                return value in mylist

        obj = BranchNode()
        #0. Determine if should skip because of parent to skip list or limb branch list
        for p_list in self.parents_to_skip_attributes:
            if in_list(getattr(self,p_list),branch_idx):
                obj.skipped = True
                obj.skipped_reason = f"parent skip: {p_list}"
                return obj

        if branch_idx not in self.branches_to_process:
            obj.skipped = True
            obj.skipped_reason = f"parent skip: Not in limb_branch_dict"
            return obj

        #1. Feature: Calculate skip distance if not set
        if config.skip_distance is None:
            config.skip_distance = ed.calculate_skip_distance(self.limb,branch_idx)

        
        #2. Feature: Determine all downstream branches using skip distance
        downstream_nodes,skipped_nodes = cnu.downstream_nodes_with_skip_distance(
            self.limb,
            branch_idx = branch_idx,
            skip_distance = config.skip_distance,
            return_skipped=True
        )

        obj.skipped_downstream_nodes = list(skipped_nodes)

        #2b.filter away branches_to_skip
        obj.downstream_nodes = list(set(downstream_nodes).difference(self.branches_to_skip))
        obj.removed_downstream_nodes = list(set(downstream_nodes).intersection(self.branches_to_skip))
        dnodes =  obj.downstream_nodes  

        if debug:
            print(f"dnodes = {dnodes}")

        #3. Skip parent: if len(downstream nodes) < min_degree_to_resolve
        if config.min_degree_to_resolve is not None:
            if len(dnodes) < config.min_degree_to_resolve:
                obj.skipped = True
                obj.skipped_reason = f"degree ({len(dnodes)}) less than min_degree_to_resolve ({config.min_degree_to_resolve})"
                return obj

        #4. Error all downstream branches is len(down)>max_degree_to_resolve_absolute
        if config.max_degree_to_resolve_absolute is not None:
            if len(dnodes) > config.max_degree_to_resolve_absolute:
                obj.error_all = True
                obj.error_all_reason = f"degree ({len(dnodes)}) greater than max_degree_to_resolve_absolute ({config.max_degree_to_resolve_absolute})"
                return obj
            
        #5. Feature: Calculate parent width
        branch_obj = self.limb[branch_idx]
        parent_width = config.width_func(branch_obj)
        if debug:
            print(f"parent_width = {parent_width}")

        #5b. Skip parent: if width too large:
        if config.upstream_width_max is not None:
            if parent_width > config.upstream_width_max:
                obj.skipped = True
                obj.skipped_reason = f"parent width ({parent_width}) greater than upstream_width_max ({config.upstream_width_max})"
                return obj

        #6. If width is above max_degree_to_resolve_width_threshold,
        #   increases the max_degree_to_resolve to the wide version
        if config.max_degree_to_resolve_wide is not None and config.max_degree_to_resolve_width_threshold is not None:
            if parent_width > config.max_degree_to_resolve_width_threshold:
                config.max_degree_to_resolve = config.max_degree_to_resolve_wide
                if verbose:
                    print(f"Changing max_degree_to_resolve = {config.max_degree_to_resolve_wide} because upstream width was {parent_width} ")

        #Error all: If number of downstream branches above max_degree_to_resolve
        if config.max_degree_to_resolve is not None:
            if len(dnodes) > config.max_degree_to_resolve:
                obj.error_all = True
                obj.error_all_reason = f"degree ({len(dnodes)}) greater than max_degree_to_resolve ({config.max_degree_to_resolve})"
                return obj

        #8. Feature: Calculates the width off the parent and all downstream branches
        widths_in_branches = np.array([config.width_func(self.limb[k]) for k in dnodes] + [parent_width])

        if debug:
            print(f"widths_in_branches = {widths_in_branches}")

        #9. Skip parent:If all the widths in parent+downstream are greater than threshold
        if config.width_max is not None:
            if len(widths_in_branches[widths_in_branches>config.width_max]) >= len(dnodes)+1:
                obj.skipped = True
                obj.skipped_reason = f"all branch widths ({widths_in_branches}) greater than width_max ({config.width_max})"
                return obj

        #10: Skip parent: if any downstream branches not axon
        if config.axon_dependent:
            for b in dnodes:
                if "axon" not in self.limb[b].labels:
                    obj.skipped = True
                    obj.skipped_reason = f"branch {b} was not an axon and axon_dependent = {config.axon_dependent}"
                    return obj

        #11. Skip parent: if branch_width is below threshold
        if config.width_min is not None:
            if parent_width < config.width_min:
                obj.skipped = True
                obj.skipped_reason = f"parent width ({parent_width}) less than width_min ({config.width_min})"
                return obj
            
        #12. If skeleton endpoint jumped too much then skip
        if config.max_skeleton_endpoint_dist is not None:
            if config.max_skeleton_endpoint_dist < inf:
                if branch_obj.max_skeleton_endpoint_dist > config.max_skeleton_endpoint_dist:
                    obj.skipped = True
                    obj.skipped_reason = f"Parent max_skeleton_endpoint_dist ({branch_obj.max_skeleton_endpoint_dist}) was greater than maximum ({config.max_skeleton_endpoint_dist})"
                    return obj
                
        #13. If the downstream skeleton endpoint is too close to soma then skip
        if config.min_downstream_endnode_skeletal_dist_from_soma is not None:
            if config.min_downstream_endnode_skeletal_dist_from_soma > 0:
                
                # compute downstream_skeletal_length_to_soma
                downstream_endnode_skeletal_dist_from_soma = lu.downstream_endnode_skeletal_distance_from_soma(self.limb,branch_idx)
                
                if downstream_endnode_skeletal_dist_from_soma < config.min_downstream_endnode_skeletal_dist_from_soma:
                    obj.skipped = True
                    obj.skipped_reason = f"Parent downstream_endnode_skeletal_dist_from_soma ({downstream_endnode_skeletal_dist_from_soma:.2f}) is less than min_downstream_endnode_skeletal_dist_from_soma ({config.min_downstream_endnode_skeletal_dist_from_soma:.2f})"
                    return obj
                
        if debug and obj.skipped == False and obj.error_all == False:
            print(f"No skips or error alls triggered")
            
        return obj
    
    @property
    def skipped_nodes_too_small_skeleton(self):
        return self.skipped_nodes_filtered(
        substr_filter = "too_small_skeleton",
        return_dict = False,
        verbose = False,
        )
    
    def skipped_nodes_filtered(
        self,
        substr_filter = None,# = "too_small_skeleton",
        return_dict = False,
        verbose = False,
        ):
        """
        Retrieve all nodes that were skipped (filtering by optional substring)
        
        Pseudocode
        ----------
        """
        
        skipped_nodes = []
        
        if verbose:
            if substr_filter is not None:
                print(f"Screenering for substr_filter = {substr_filter}")
        for b,obj in self.items():
            if not obj.skipped:
                if verbose:
                    print(f"Not adding {b} because not skipped")
                continue
            if substr_filter is not None:
                if substr_filter not in obj.skipped_reason:
                    if verbose:
                        print(f"Not adding {b} because did not have substr_filter ({obj.skipped_reason})")
                    continue
            if verbose:
                print(f"adding {b}")
                    
            skipped_nodes.append((b,obj))
        
        if return_dict:
            return_value = dict(skipped_nodes)
        else:
            return_value = [k[0] for k in skipped_nodes]
        
        return return_value

LimbGraphScreener.__doc__ = limb_screener_graph_doc

## ----- DownstreamErrorDetector Class ----
from abc import ABC, abstractmethod
from dataclasses import field

@dataclass
class NodeError:
    node: int = None
    description: str = None

    def __repr__(self):
        return f"{self.node} ({self.description})"
        
    def __eq__(self, other):
        if not isinstance(other, NodeError):
            return NotImplemented
        return self.node == other.node

    def __hash__(self):
        return hash(self.node)
    

class DownstreamErrorDetector(ABC):
    """
    Purpose:
    --------
    The blueprints for how a branching filter
    should be implemented so it can be used
    throughout the whole neuron
    """
    
    def __init__(
        self,
        config=None,
        **kwargs):
        self.config = config
        if self.config is None:
            self.config = self.Config(**kwargs)

        
    @abstractmethod
    def __call__(self,limb_obj,parent_node,downstream_nodes) -> bool:
        """
        Purpose:
        -------
        Will return all of the downstream branches that are errors
        """
        pass

    @property
    @abstractmethod
    def description(self):
        """
        Must include a description of what this particular
        filter does to determine if any downstream nodes are errors 
        """
        pass

from typing import Any, Callable, Iterable, List
class FunctionDownstreamErrorDetector(DownstreamErrorDetector):
    """
    Thin rapper that can turn any function and string into a 
    subclass of DownstreamErrorDetector
    """
    def __init__(self,
                 func: Callable[[Any, Any, Iterable[Any]], List[Any]],
                 desc: str):
        self._func = func
        self._desc = desc

    def __call__(self,
                 limb_obj: Any,
                 parent_node: Any,
                 downstream_nodes: Iterable[Any]
                 ) -> List[Any]:
        return self._func(limb_obj, parent_node, downstream_nodes)

    @property
    def description(self) -> str:
        return self._desc


        
## ------ GraphErrorScreeners -----

from collections import UserDict 
class ErrorLimbBranch(UserDict):
    """
    Purpose
    -------
    A datastructure that stores the limb branches information
    concerning errors on a neuron (initialized from a limb_branch dictionary)

    Parameters
    ----------
    UserDict : _type_
        _description_
    """
    def __init__(self,limb_branch_dict=None,description=None):
        """
        Example 1
        ---------
        limb_branch = dict(L0  =[1,3,4,4,5],L2 = [0,0,4,5],L3 = [14,15])
        test = ged.ErrorLimbBranch(limb_branch)
        test.limb_branch_dict
        """
        if isinstance(self,type(limb_branch_dict)):
            self.data = dict(limb_branch_dict.data)
            self.ensure_unique_branches()
            return
        
        self.data = {}
        
        if limb_branch_dict is None:
            limb_branch_dict = {}
            
        for limb_name,branches in limb_branch_dict.items():
            self.data[limb_name] = [NodeError(b,description) for b in branches]
            
        self.ensure_unique_branches()
            

    @staticmethod
    def unique_branch_list(branch_list):
        seen = set()
        unique_branches = []
        for e in branch_list:
            if e not in seen:
                unique_branches.append(e)
                seen.add(e)
        return unique_branches
    
    def ensure_unique_branches(self):
        for limb_name,branches in self.data.items():
            self.data[limb_name] = self.unique_branch_list(self.data[limb_name])
        
    def add_limb(self,limb_idx,branch_list,filter_to_unique=True):
        if filter_to_unique:
            branch_list = self.unique_branch_list(branch_list)
        self.data[limb_idx] = branch_list
    
    @property
    def limb_branch_dict(self):
        return {limb_idx:np.array([k.node for k in branches])
                for limb_idx,branches in self.data.items()}
        
    def __repr__(self):
        return_str = []
        for k,v in self.data.items():
            return_str.append(f"{k}: {len(v)} errors")
            for b in v:
                return_str.append(f"\t{b}")
        return "\n".join(return_str)
    
class LimbBranchErrorDetector(ABC):
    """
    Baseclass that any function that returns a limb branch error dict
    can be turned into a valid detector class
    """
    def __init__(
        self,
        config = None,
        **kwargs):
        
        if config is None:
            config = self.Config(**kwargs)
        self.config = config
        
    @abstractmethod
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        pass 
    
    def __call__(self,neuron_obj,**kwargs):
        return ErrorLimbBranch(
            self.error_limb_branch_dict(neuron_obj,**kwargs),
            description = self.config.name,
        )
        
import time
class NeuronGraphErrorDetector:
    """
    Purpose
    -------
    Combine a GraphScreener and a DownstreamErrorDetector
    to determine a list of all error nodes on a neuron object
    and return that list as a limb branch dict
    
    Ultimately generates a ErrorLimbBranch as the output of the call function
    
    Attributes:
    ---------
        graph_screener_config: NeuronGraphScreenerConfig
        error_detector: DownstreamErrorDetector
    
    """
    @property
    def debug_detector_kwargs(self):
        kwargs = self._generate_error_kwargs
        new_dict = deepcopy(kwargs)
        for k,v in kwargs.items():
            if type(v) == dict:
                for jk,jv in v.items():
                    new_dict[k][jk] = True
            else:
                new_dict[k] = True
        return new_dict
    
    def debug_run(self,neuron_obj,**kwargs):
        return self(neuron_obj,**self.debug_detector_kwargs,**kwargs)
    
    def __init__(
        self,
        
        error_detector_cls: DownstreamErrorDetector=None,
        error_detector_kwargs = None,
        neuron_graph_config_kwargs = None,
        branch_graph_config_kwargs = None,
        graph_screener_kwargs = None,
        
        graph_screener_config: NeuronBranchLimbGraphScreenerConfig=None,
        error_detector: DownstreamErrorDetector=None,
        
        config = None,
        **kwargs
        ):
        """
        Create a NeuornGraphErrorScreener object that uses
        a certain DownstreamErrorDetector subclass and initializes
        all the graph_screener_config and error_detector automatically
        """
        if hasattr(self,"Config"):
            config = self.Config(**kwargs)
        
        if config is not None:
            if error_detector_cls is None:
                error_detector_cls = config.error_detector_cls
            if error_detector_kwargs is None:
                error_detector_kwargs = config.error_detector_kwargs
            if neuron_graph_config_kwargs is None:
                neuron_graph_config_kwargs = config.neuron_graph_config_kwargs
            if branch_graph_config_kwargs is None:
                branch_graph_config_kwargs = config.branch_graph_config_kwargs
            if graph_screener_kwargs is None:
                graph_screener_kwargs = config.graph_screener_kwargs
                
            if hasattr(config,"generate_error_kwargs"):
                self._generate_error_kwargs = config.generate_error_kwargs
            else:
                self._generate_error_kwargs = {}
        
        if graph_screener_config is None:
            if graph_screener_kwargs is None:
                graph_screener_kwargs = dict()
            if error_detector_kwargs is None:
                error_detector_kwargs = dict()

            # Step 1: graph_screener_config - Instantiating the graph filter config object
            graph_screener_config = NeuronBranchLimbGraphScreenerConfig(
                neuron_graph_config_kwargs=neuron_graph_config_kwargs,
                branch_graph_config_kwargs=branch_graph_config_kwargs,
                **graph_screener_kwargs
            )

        if error_detector is None:
            error_detector = error_detector_cls(
                **error_detector_kwargs
            )
        self.graph_screener_config = graph_screener_config
        self.error_detector = error_detector
        self.config = config
        
    def __call__(
        self,
        neuron_obj,
        verbose = False,
        branch_verbose = False,
        debug = False,
        graph_screener_kwargs = None,
        error_detector_kwargs = None,
        **kwargs
    ):
        """
        Purpose
        -------
        Use both the graph filter and error filter
        to determine a list of all error nodes on a neuron object

        Pseudocode
        ----------
        0. Initialize an empty ErrorLimbBranch
        1. Run the graph filter to determine the executable_graph to operate on
        2. Iterate over the limbs of executable_graph:
            create an empty error list
            For every node in limb graph:
                a. if error_all: add all downstream to error list (with reason)
                b. If skipped: continue
                c. if regular downstream list:
                    Run the DownstreamErrorDetector on the node and all downstream (add to the list)
            d. add the error list and limb idx to an error limb branch dict

        3. Return the ErrorLimbBranch
        """
        #overwritting the generic error kwargs
        kwargs = {**self._generate_error_kwargs, **kwargs}
        
        eobj = ErrorLimbBranch()

        st = time.time()
        
        if graph_screener_kwargs is None:
            graph_screener_kwargs = dict()
        screened_graph = NeuronGraphScreener(
            neuron_obj,
            **self.graph_screener_config.graph_screener_config_dict,
            **graph_screener_kwargs
            )
        self.screened_graph = screened_graph
        
        if debug:
            print(f"Time for graph_screener = {time.time() - st}")
            print(f"screened_graph.limb_branch_dict = {screened_graph.limb_branch_dict}")

        error_detector = self.error_detector
        if error_detector_kwargs is None:
            error_detector_kwargs = dict()
            
        if getattr(error_detector.config,"require_screened_graph",False):
            add_filtered_limb = True
        else:
            add_filtered_limb = False
            
        for limb_idx,limbG in screened_graph.items():
            
            limb = neuron_obj[limb_idx]
            limb_errors = []
        
            if verbose:
                print(f"-- Working on {limb_idx} --")
            for branch_idx,b_node in limbG.items():
                if verbose:
                    print(f"\t-- branch: {branch_idx} --")
                if b_node.skipped:
                    if branch_verbose:
                        print(f"\t\tSkipped: {b_node.skipped_reason}")
                    continue
                if b_node.error_all:
                    if branch_verbose:
                        print(f"\t\tError All: {b_node.error_all_reason})")
                        print(f"\t\tNodes erroring:{b_node.downstream_nodes}")
                    limb_errors += [NodeError(k,f"parent {branch_idx} error all {b_node.skipped_reason}")
                                   for k in b_node.downstream_nodes]
                    continue
            
                downstream_nodes = b_node.downstream_nodes
                if branch_verbose:
                    print(f"\t\t\nRunning DownstreamErrorDetector for parent = {branch_idx} with downstream node = {downstream_nodes}\n")
            
                kwargs_dict = dict(
                    limb_obj=limb,
                    parent_node=branch_idx,
                    downstream_nodes=downstream_nodes,
                    **error_detector_kwargs
                )
                if add_filtered_limb:
                    kwargs_dict['screened_graph'] = limbG
                    
                branch_downstream_errors = error_detector(
                    **kwargs_dict
                )
            
                if branch_verbose:
                    print(f"\t\t\nDownstream errors of {branch_idx} = {branch_downstream_errors}\n")
            
                limb_errors +=branch_downstream_errors

            if verbose:
                print(f"Final limb {limb_idx} errors = {limb_errors}")
            
            eobj.add_limb(limb_idx,limb_errors)
            
        return eobj
    
    error_limb_branch = __call__
  
        
from . import (
    graph_error_detector_dendrite as gedd,
    neuron_visualizations as nviz,
    neuron_utils as nru,
)
    
#from . import graph_filters_refactored as gf