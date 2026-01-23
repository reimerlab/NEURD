from dataclasses import dataclass,asdict
from abc import ABC, abstractmethod
from typing import Any, Dict

from neurd import (
    neuron,
    graph_error_detector as ged,
    graph_error_detector_dendrite as dend,
    graph_error_detector_axon as ax,
)
from neurd.neuron import Neuron
import time


class LimbBranch:
    """
    Purpose: To store the ErrorLimb branch, Neuron Limb branch
    and then be able to compute other types of limb branches
    1. The limb branch remaining
    2. The limb branch after removal
    """
    @staticmethod
    def lb_dict(obj):
        return getattr(obj,"limb_branch_dict",obj)
        
    def __init__(
        self,
        neuron_obj,
        error_limb_branch,
        neuron_limb_branch=None,
        
    ):
        self.error_limb_branch = ged.ErrorLimbBranch(error_limb_branch)
        
        if neuron_limb_branch is None:
            neuron_limb_branch = nru.neuron_limb_branch_dict(neuron_obj)
        self.neuron_limb_branch = neuron_limb_branch

        self._error_limb_branch_downstream = None
        self.set_error_limb_branch_downstream(neuron_obj)
        self.limb_branch_dict = self.lb_dict(self.error_limb_branch)
        
    def __len__(self):
        if self.error_limb_branch is None:
            return 0
        return len([k for k,v in self.error_limb_branch.items()
                    if len(v)>0])
    
    @classmethod
    def from_neuron_and_error_limb_branch(
        cls,
        neuron_obj,
        error_limb_branch
        ):
        
        return cls(neuron_obj,error_limb_branch)
        

    def set_error_limb_branch_downstream(self,neuron_obj):
        """
        Will compute all the final limb branch that is removed after applying the cuts
        at the specified error_limb_branch

        Parameters
        ----------
        neuron_obj : _type_
            _description_
        """
        self._error_limb_branch_downstream = nru.limb_branch_after_limb_branch_removal(
                neuron_obj = neuron_obj,
                limb_branch_dict = self.lb_dict(self.error_limb_branch),
                return_removed_limb_branch = True
        )

    @property
    def error_limb_branch_downstream(self):
        return self._error_limb_branch_downstream
    
    error_limb_branch_downstream_dict = error_limb_branch_downstream
    
    @property
    def error_limb_branch_dict(self):
        return self.error_limb_branch.limb_branch_dict

    @property
    def kept_limb_branch_dict(self):
        return nru.limb_branch_setdiff(
            [self.lb_dict(self.neuron_limb_branch),
            self.lb_dict(self.error_limb_branch_downstream),
            ]
        )            

@dataclass
class RemovedStats:
    """
    Dataclass that will hold all of the skeletal length and area of the
    downstream limb branch. Class function can generate object from neuron obj
    and error limb branch
    
    Example Generation
    ------------------
    stats_obj = FilterStats.from_neuron_and_limb_branch(
        neuron_obj,
        lb
    )
    """
    skeletal_length: float
    area: float
    # …any other numeric summaries…

    @classmethod
    def from_neuron_and_limb_branch(
        cls,
        neuron_obj: Any,
        limb_branch: LimbBranch,
        *,
        verbose: bool = False,
    ) -> "RemovedStats":
        
        if isinstance(limb_branch,dict):
            limb_branch_dict = limb_branch
        else:
            limb_branch_dict = limb_branch.error_limb_branch_downstream
        
            
        # compute skeletal length
        ske_len = nru.sum_feature_over_limb_branch_dict(
            neuron_obj,
            limb_branch_dict,
            "skeletal_length"
        ) or 0.0

        # compute area
        area = nru.sum_feature_over_limb_branch_dict(
            neuron_obj,
            limb_branch_dict,
            "area"
        ) or 0.0

        if verbose:
            print("Downstream Error Limb–branch dict:", limb_branch_dict)
            print(f"  skeletal_length = {ske_len:.2f} nm")
            print(f"  area            = {area:.2f} µm²")

        return cls(
            skeletal_length=ske_len,
            area=area
        )
    def __repr__(self):
        return (f"error skeletal_length = {self.skeletal_length:.2f} nm, error area = {self.area:.2f} µm²")


@dataclass
class RedBlueSuggestions:
    """
    Datastructure to hold the red blue suggestions output.
    Has a class function that allows you to calculate it from the
    neuron object and the downstream limb branch

    The proofreading global functions have _red_blue before them
    """
    @dataclass
    class Config:
        plot_error_graph_before_create_edges: bool = False
        plot_error_branches: bool = False
        created_edges: Any = None
        plot_error_graph_after_create_edges: bool = False
        plot_error_connected_components: bool = False
        include_one_hop_downstream_error_branches: bool = True
        one_hop_downstream_error_branches_max_distance: int = 7000
        offset_distance_for_points_valid: int = 1000#3000
        offset_distance_for_points_error: int = 1000#3000
        n_points: int = 1
        n_red_points: int = 8 #3
        n_blue_points: int = 8 #2
        red_blue_points_method: str = 'closest_mesh_face'
        plot_final_blue_red_points: bool = False
        scatter_size: float = 0.3
        pair_conn_comp_by_common_upstream: bool = True
        pair_conn_comp_errors: bool = True
        group_all_conn_comp_together: bool = False
        only_outermost_branches: bool = True
        min_error_downstream_length_total: int = 5000
        verbose: bool = False
        valid_upstream_branches_restriction: Any = None
        split_red_blue_by_common_upstream: bool = True
        use_undirected_graph: bool = False
        avoid_one_red_or_blue: bool = True
        min_cancel_distance_absolute: int = 1000
        min_cancel_distance_absolute_all_points: Any = None
        add_additional_point_to_no_children_branches: bool = True
        return_error_skeleton_points: bool = True
        return_synapse_points: bool = True
        red_blue_suggestions: Any = None
        
    Config_h01_c2 = Config
        
    @dataclass
    class Config_minnie:
        n_red_points: int = 3
        n_blue_points: int = 2

    def __init__(self,**kwargs):
        
        config = kwargs.pop("config", None)
        red_blue_suggestions = kwargs.pop("red_blue_suggestions")
        
        if config is None:
            config = self.Config(**kwargs)
        self.config = config

        self.red_blue_suggestions = red_blue_suggestions

    def plot(self,neuron_obj):
        pru.plot_limb_to_red_blue_groups(
            neuron_obj,
            self.red_blue_suggestions
        )

    @classmethod
    def from_neuron_and_limb_branch(
        cls,
        neuron_obj,
        limb_branch,
        plot_final_blue_red_points = False,
        dataset = None,
        **kwargs
    ):
        kwargs["plot_final_blue_red_points"] = plot_final_blue_red_points
        if dataset is None:
            config = cls.Config(**kwargs)
        else:
            config = getattr(cls,f"Config_{dataset}")(**kwargs)
            
        red_blue_suggestions = pru.limb_branch_dict_to_cancel_to_red_blue_groups(
            neuron_obj,
            limb_branch_dict_to_cancel=limb_branch.error_limb_branch_downstream,
            #plot_final_blue_red_points = plot_final_blue_red_points,
            **asdict(config)
        )

        return cls(config=config,red_blue_suggestions=red_blue_suggestions)
    
    
@dataclass
class SplitSuggestions:
    """
    Data structure that holds the split suggestion points before and after downstream.
    Has a class method that generates the split suggestion coordinates using the limb_utils
    function given the neuron object and limb branch errors

    Returns
    -------
    _type_
        _description_
    """
    split_locations: Any = None
    split_locations_before_downstream: Any = None


    @classmethod
    def from_neuron_and_limb_branch(
        cls,
        neuron_obj,
        limb_branch,
        verbose = False,
    ):

        split_locations_before_downstream  = lu.most_upstream_endpoints_of_limb_branch(
            neuron_obj,
            limb_branch_dict = limb_branch.error_limb_branch_dict,
            group_by_conn_comp = False,
            verbose = verbose
        )

        split_locations= lu.most_upstream_endpoints_of_limb_branch(
                neuron_obj,
                limb_branch_dict = limb_branch.error_limb_branch_downstream,
                group_by_conn_comp = False,
                verbose = verbose
        )

        return cls(
            split_locations_before_downstream=split_locations_before_downstream,
            split_locations=split_locations
        )
        
    def plot(
        self,
        neuron_obj,
        limb_branch = None,
        split_color = "red",
        scatter_size = 1,
        mesh_color = "black",
    ):
        if limb_branch is not None:
            meshes = nru.feature_over_limb_branch_dict(
                neuron_obj,
                limb_branch.error_limb_branch_dict,
                feature = "mesh",
            )
        else:
            meshes = None
        
        all_split_points = []
        for limb_idx,split_segments in self.split_locations_before_downstream.items():
            all_split_points += [k[0] for k in split_segments]
        
        ipvu.plot_objects(
            neuron_obj.mesh_from_branches,
            meshes=meshes,
            meshes_colors=mesh_color,
            scatters=[all_split_points],
            scatters_colors=split_color,
            scatter_size=scatter_size,
        )
        
# -- Container for all dataproducts

@dataclass
class FilterResults:
    """
    Container for holding all the metadata we want (skeletal length/area, split points, red blue points) of a certain filter given.
    Cotains class method for generating container from the neuron object and error limb branch

    Returns
    -------
    _type_
        _description_
    """
    name: str
    time: float = None
    limb_branch: LimbBranch = None
    stats: RemovedStats = None
    red_blue: RedBlueSuggestions = None
    splits: SplitSuggestions = None
    neuron_obj: neuron.Neuron = None
        
    @classmethod
    def from_neuron_and_error_limb_branch(
        cls,
        neuron_obj,
        error_limb_branch,
        name,
        time=None,
        plot_red_blue = False,
        plot_splits = False,
        dataset = None,
        verbose = False,
        **kwargs
        ):
        
        limb_branch = LimbBranch(neuron_obj,error_limb_branch)
        stats = RemovedStats.from_neuron_and_limb_branch(neuron_obj,limb_branch)
        red_blue = RedBlueSuggestions.from_neuron_and_limb_branch(
            neuron_obj,
            limb_branch,
            dataset=dataset,
            )
        if plot_red_blue:
            red_blue.plot(neuron_obj)
        splits = SplitSuggestions.from_neuron_and_limb_branch(neuron_obj,limb_branch)
        if plot_splits:
            splits.plot(neuron_obj,limb_branch)
        
        return cls(
            name = name,
            time = time,
            limb_branch=limb_branch,
            stats=stats,
            red_blue=red_blue,
            splits=splits
        )
        
        
class NeuronFilter:
    """
    Object that is initialized with an error detector that will then using the run function calculate the limb branch error from an error filter, compute the filter results and then clean the neuron accordingly, and return the new neuron object and results
    
    Responsible for taking a detection method that identifies all error nodes and 
    1. Detect: Running the detedction
    2. Analyze: Generating all of the data products
    3. Clean: running the neuron cleaning
    - then returns results

    What configuratin do we need?
    """
    @dataclass
    class Config:
        name:str=None
        combine_path_branches: bool = True
        save_intermediate_neuron: bool = False
        #start_with_deepcopy = True
        
    def __init__(
        self,
        detector,
        name = None,
        **kwargs):
        
        config = kwargs.pop("config",None)
        kwargs['name'] = name
        
        if config is None:
            config = self.Config(**kwargs)
        
        if config.name is None:
            config.name = detector.config.name
            
        self.detector = detector
        self.config = config

    
    def detect(
        self, 
        neuron_obj: Neuron,
        **kwargs) -> ged.ErrorLimbBranch:
        
        return self.detector(neuron_obj,**kwargs)
        

    def analyze(
        self, 
        neuron_obj:Neuron,
        error_limb_branch: ged.ErrorLimbBranch,
        dataset = None,
        **kwargs) -> FilterResults:

        return FilterResults.from_neuron_and_error_limb_branch(
            neuron_obj=neuron_obj,
            error_limb_branch=error_limb_branch,
            name = self.config.name,
            dataset=dataset,
            **kwargs
        )

    def clean(
        self,
        neuron_obj,
        removed_limb_branch_dict,
        plot_neuron_after_cancellation = False,
        plot_final_neuron = False,
        verbose = False,
        verbose_time = False,
        **kwargs):
        
        if verbose_time:
            st = time.time()
        
        #if self.config.start_with_deepcopy:
        neuron_obj = neuron.Neuron(neuron_obj)
            
        if verbose_time:
            print(f"\t-- time for working on copying neuron: {(time.time() - st):.2f}--")
            st = time.time()
            
        new_neuron = pru.delete_branches_from_neuron(
            neuron_obj,
            limb_branch_dict = removed_limb_branch_dict,
            plot_neuron_after_cancellation = plot_neuron_after_cancellation,
            plot_final_neuron = plot_final_neuron,
            verbose = verbose,
        )
        
        if verbose_time:
            print(f"\t-- time for delete_branches_from_neuron : {(time.time() - st):.2f}--")
            st = time.time()
            
        if self.config.combine_path_branches:
            new_neuron = nsimp.combine_path_branches(
                    new_neuron,
                    return_copy=False,
                    plot_downstream_path_limb_branch = False,
                    verbose = verbose,
            )
            
            if verbose_time:
                print(f"\t-- time for combine_path_branches : {(time.time() - st):.2f}--")
                st = time.time()
            
        return new_neuron

    def run(
        self, 
        neuron_obj: Neuron,
        verbose_time=False,
        detector_verbose = False,
        verbose = False,
        dataset = None,
        **kwargs):
        global_time = time.time()
        
        if verbose_time:
            st = time.time()
            
        #1: Detect
        error  = self.detect(neuron_obj,verbose = detector_verbose,**kwargs)
        
        if verbose_time:
            print(f"\t-- time for detection: {(time.time() - st):.2f}--")
            st = time.time()
        #2: Analyze
        results = self.analyze(
            neuron_obj,
            error,
            dataset=dataset,
            verbose = verbose,
            **kwargs
        )
        
        if verbose_time:
            print(f"\t-- time for analyzing : {(time.time() - st):.2f}--")
            st = time.time()
        #3: Clean
        if len(results.limb_branch) == 0:
            if verbose or verbose_time:
                print(f"Skipping cleaning because limb_branch empty")
            cleaned = neuron_obj
        else:
            
            cleaned = self.clean(
                neuron_obj,
                results.limb_branch.error_limb_branch_downstream,
                verbose = verbose,
                verbose_time=verbose_time,
                **kwargs
            )
            
        if verbose_time:
            print(f"\t-- time for cleaning : {(time.time() - st):.2f}--")
            st = time.time()
        
        if self.config.save_intermediate_neuron:
            results.neuron_obj = cleaned
            
        # storing the time of the filter
        results.time = time.time() - global_time
        
        return cleaned,results

## -- Orchestration class --
class FilterPipeline:
    """
    Object that is initialized with a set of filters and runs the filters one after the other
    and stores the results.
    
    Can export the list of results in the same old format
    """
    
    def __init__(self, filters,**kwargs):
        self.filters = [self.to_filter(k,**kwargs)
                        for k in filters]
        
    @staticmethod
    def to_filter(obj,**kwargs):
        if "detector" in str(type(obj)).lower():
            return NeuronFilter(detector = obj,**kwargs)

    def visualize(self,**kwargs):
        raise NotImplementedError('')
    
    def run(
        self, 
        neuron_obj: Neuron,
        verbose_time=False,
        visualize = False,
        dataset = None):
        results: dict[str, FilterResults] = {}
        current = neuron_obj
        for flt in self.filters:
            if verbose_time:
                st = time.time()
                print(f"\n-- working on {flt.config.name}--")
            cleaned, res = flt.run(current,verbose_time=verbose_time,dataset=dataset)
            if verbose_time:
                print(f"\n\t--total time for {flt.config.name}: {(time.time() - st):.2f}--")
            results[flt.config.name] = res
            if visualize:
                self.visualize(current,res)
            current = cleaned
        return current, results
    
    @staticmethod
    def filter_results_dict_to_old_filtering_info(filter_results_dict):
        """
        Purpose: To turn zour results into a filter_info dictionary just like in old process
        
        Pseudocode
        ----------
        0. Create an empty global_results dictionary
        Iterate through all the run filters:
        1. Do the name translation (this is the prefix
        2. Build the local_results dictionary with {filter_name}_ as prefix
            time
            error_area
            error_length
            
            limb_branch_dict_to_cancel
            created_edges
            red_blue_suggestions
        
            split_locations
            split_locations_before_filter
        3. update the global results with the local
        4. return 
        """
        name_translator = dict(
            axon_on_dendrite = "axon_on_dendrite_merges",
            exc_axon_high_degree = "high_degree_branching",
            inh_axon_high_degree = "high_degree_branching",
            exc_axon_low_degree = "low_degree_branching",
            inh_axon_low_degree = "low_degree_branching",
            exc_axon_width_jump = "width_jump_up_axon",
            dendrite_double_back = "double_back_dendrite",
            exc_dendrite_double_back = "double_back_dendrite",
            inh_dendrite_double_back = "double_back_dendrite",
            dendrite_cross_roads = "high_degree_branching_dendrite",
            dendrite_width_jump = "width_jump_up_dendrite",
            exc_dendrite_width_jump = "width_jump_up_dendrite",
            dendrite_internal_bend = "dendrite_internal_bend",
        )
        
        global_results = dict()
        time = None
        
        for new_filter_name,res in filter_results_dict.items():
            old_filter_name = name_translator[new_filter_name]
        
            local_results = dict(
                time = getattr(res,"time",None),
                error_area = res.stats.area,
                error_length = res.stats.skeletal_length,
                
                limb_branch_dict_to_cancel = res.limb_branch.error_limb_branch_downstream_dict,
                created_edges = None,
                red_blue_suggestions = res.red_blue.red_blue_suggestions,
                
                split_locations = res.splits.split_locations,
                split_locations_before_filter = res.splits.split_locations_before_downstream,
                
            )
        
            local_results = {f"{old_filter_name}_{k}":v for k,v in local_results.items()}
            global_results.update(local_results)
        
        return global_results
    
    
# --- Fully refactored proofread_neuron_full --
filters_for_datasets = dict(
    h01_c2 = dict(
        excitatory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.ExcAxonHighDegreeErrorDetector(),
            ax.ExcAxonLowDegreeErrorDetector(),
            
            dend.DendriteCrossRoadsErrorDetector(),
            dend.DendriteDoubleBackErrorDetector(),
            dend.DendriteWidthJumpErrorDetector(),
            dend.DendriteInternalBendErrorDetector(),

            ax.ExcAxonWidthJumpErrorDetector(),    
            
        ],
        inhibitory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.InhAxonHighDegreeErrorDetector(),
            ax.InhAxonLowDegreeErrorDetector(),
            
            dend.DendriteCrossRoadsErrorDetector(),
            dend.DendriteDoubleBackErrorDetector(),
            dend.DendriteWidthJumpErrorDetector(),
            dend.DendriteInternalBendErrorDetector(),

            ax.ExcAxonWidthJumpErrorDetector(), 
        ]
    ),
    microns = dict(
        excitatory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.ExcAxonHighDegreeErrorDetector(),
            ax.ExcAxonLowDegreeErrorDetector(),
            
            dend.ExcDendriteWidthJumpErrorDetector(),
            ax.ExcAxonWidthJumpErrorDetector(),  
            dend.ExcDendriteDoubleBackErrorDetector(),  
        ],
        inhibitory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.ExcAxonHighDegreeErrorDetector(),
            ax.ExcAxonLowDegreeErrorDetector(),
            
            dend.ExcDendriteWidthJumpErrorDetector(),
            ax.ExcAxonWidthJumpErrorDetector(),  
            dend.InhDendriteDoubleBackErrorDetector(),  
        ]
    ),
    microns_ml = dict(
        excitatory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.ExcAxonHighDegreeErrorDetector(),
            ax.ExcAxonLowDegreeErrorDetector(
                skip_greater_than_max_degree_to_resolve = True,
            ),
            
            dend.ExcDendriteWidthJumpErrorDetector(),
            ax.ExcAxonWidthJumpErrorDetector(),  
            dend.ExcDendriteDoubleBackErrorDetector(),  
        ],
        inhibitory = [
            ax.AxonOnDendriteErrorDetector(),
            ax.ExcAxonHighDegreeErrorDetector(),
            ax.ExcAxonLowDegreeErrorDetector(
                skip_greater_than_max_degree_to_resolve = True,
            ),
            
            dend.ExcDendriteWidthJumpErrorDetector(),
            ax.ExcAxonWidthJumpErrorDetector(),  
            dend.InhDendriteDoubleBackErrorDetector(),  
        ]
    ),
)
# exc_filters_h01_c2 = [
    
# ]

# inh_filters_h01_c2 = [
       
# ]

# exc_filters_microns = [
    
# ]

# inh_filters_microns = [
    
# ]


def filters_from_cell_type(cell_type,dataset = "h01_c2"):
    if dataset in filters_for_datasets:
        filter_dict = filters_for_datasets[dataset]
        if cell_type in filter_dict:
            return filter_dict[cell_type]
        else:
            raise ValueError(f"cell_type ({cell_type}) requested is neiether not in the predefined options in {list(filter_dict.keys())}")
    else:
        raise ValueError(f"dataset ({dataset}) requested is neiether not in the predefined options")

def proofread_neuron_full_refactored(
    neuron_obj,
    cell_type = None,
    verbose = False,
    verbose_time = False,
    store_ml_training_in_limb_branch_dict_to_cancel = True,
    filters_dataset = "h01_c2",
    filters = None,
    **kwargs):
    
    # Step 1: Pick the right filters
    # cell_type_filters = 
    # if cell_type == 'excitatory':
    #     cell_type_filters = exc_filters_v8
    # elif cell_type == 'inhibitory':
    #     cell_type_filters = inh_filters_v8
    # else:
    #     raise ValueError(f"cell type was not excitatory nor inhibitory")
    if filters is not None:
        cell_type_filters = filters
    else:
        cell_type_filters = filters_from_cell_type(cell_type,dataset=filters_dataset)
    
    if verbose:
        print(f"Using the following filters for {cell_type} cell:")
        for k in cell_type_filters:
            print(k)
        print(f"\n")
    
    # Step 2: Instnatiate the filter pipeline object and run the pipeline
    pipeline = FilterPipeline(
        cell_type_filters,
        combine_path_branches = True,
        save_intermediate_neuron = False,
        
    )
    
    cleaned_neuron, results = pipeline.run(
        neuron_obj,
        dataset = filters_dataset,
        verbose_time=verbose
    )
    
    # Step 3: Generate the filtering info
    filtering_info = FilterPipeline.filter_results_dict_to_old_filtering_info(results)
    
    if store_ml_training_in_limb_branch_dict_to_cancel:
        ml_train = ml_training_data_from_neuron_obj_and_cell_type(
            neuron_obj,
            cell_type=cell_type,
            filters=filters,
            dataset=filters_dataset
            )
        filtering_info["limb_branch_dict_to_cancel"] = ml_train
    
    return cleaned_neuron,filtering_info

# -- For exporting error rules
def convert_limb_branch_dict_to_node_names(limb_branch_dict):
    node_names = []
    for limb,branches in limb_branch_dict.items():
        node_names += [f"{limb}_{k}" for k in branches]
    return node_names

def ml_training_data_from_neuron_obj_and_cell_type(
    neuron_obj,
    cell_type=None,
    filters_dataset = "h01_c2",
    filters = None,
    verbose=False,
    return_node_names = True,
    
    **kwargs
    ):
    """
    will run the cell type filters on the particular neuron object 
    
    For each filter will:
    1. Turns each detector into a neuron filter
    2. Runs the detect stage of the filter
    3. Generates the error_limb branch
    4. Adds to the results dictionary - filter name: list of error node names (ex: AxonOnDendriteErrorDetector:['L0_0',
  'L0_1',])
  
    Final Output: dictionary with
    - cell type
    - segment id
    - filter results

    Parameters
    ----------
    neuron_obj : _type_
        _description_
    cell_type : _type_
        _description_
    verbose : bool, optional
        _description_, by default False
    return_node_names : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    if filters is not None:
        cell_type_detectors = filters
    else:
        cell_type_detectors = filters_from_cell_type(
            cell_type=cell_type,
            dataset=filters_dataset
        )
    
    if verbose:  
        print(f"cell_type_detectors = ")
        for k in cell_type_detectors:
            print(k)
    
    results = {}
    for det in cell_type_detectors:
        st = time.time()
        name = det.__class__.__name__
        if verbose:
            print(f"\n--- Working on filter {name} ---")
    
        filter = NeuronFilter(det)
    
        error_results = filter.detect(neuron_obj)
        error_limb_branch = error_results.limb_branch_dict
    
        results[name] = error_limb_branch
        if verbose:
            print(f"error_limb_branch = {error_limb_branch}")
            print(f"Total time = {time.time() - st:.2f}")
    results_node_names = {k:convert_limb_branch_dict_to_node_names(v) for k,v in results.items()}
    if verbose:
        print(f"final classification: {results_node_names}")

    key = dict(
        segment_id=neuron_obj.segment_id,
        cell_type=cell_type
    )
    results_node_names.update(key)
    results.update(key)

    if return_node_names:
        return results_node_names
    else:
        return results

from . import (
    neuron_utils as nru,
    proofreading_utils as pru,
    error_detection as ed,
    limb_utils as lu,
    neuron_simplification as nsimp,
    graph_error_detector as ged,
)

from datasci_tools import (
    ipyvolume_utils as ipvu
)