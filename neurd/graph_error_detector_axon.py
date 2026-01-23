"""
Purpose
-------
Adapt the current axon on axon filters for the detector format

Template
--------
```
class (ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = 
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        
    
```
"""
from neurd import (
    graph_error_detector as ged,
    proofreading_utils as pru,
    axon_utils as au,
    error_detection as ed,
    graph_filters as gf
)
from dataclasses import dataclass

class AxonOnDendriteErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "axon_on_dendrite"
        method: str = "labels"
        
    method_options = (
        "labels", #gets the axon-like classifications from precomputed label tags
        "scratch", #runs the axon-like identification over for axon on dendrite
    )
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        method = self.config.method
        if method == "labels":
            return pru.filter_away_axon_on_dendrite_merges_old_limb_branch_dict(neuron_obj,**kwargs)
        elif method == "scratch":
            return au.compute_axon_on_dendrite_limb_branch_dict(neuron_obj,**kwargs)
        else:
            raise ValueError(f"method {self.method} not one of valid options {method}")
      
    
class ExcAxonHighDegreeErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "exc_axon_high_degree"
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        return ed.high_degree_branch_errors_limb_branch_dict(
            neuron_obj,**kwargs
        )
        
class InhAxonHighDegreeErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "inh_axon_high_degree"
    
    def error_limb_branch_dict(
        self,
        neuron_obj,
        width_max = None,
        upstream_width_max = None,
        **kwargs):
        
        if width_max is None:
            width_max = pru.width_max_high_low_degree_inh_global
        if upstream_width_max is None:
            upstream_width_max = pru.upstream_width_max_high_low_degree_inh_global
            
        return ed.high_degree_branch_errors_limb_branch_dict(
            neuron_obj,
            width_max=width_max,
            upstream_width_max=upstream_width_max,
            **kwargs
        )
        
class ExcAxonLowDegreeErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "exc_axon_low_degree"
        skip_greater_than_max_degree_to_resolve: bool = False
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):

        skip_greater_than_max_degree_to_resolve = getattr(self.config,"skip_greater_than_max_degree_to_resolve",False)
        print(f"inside ExcAxonLowDegreeErrorDetector, skip_greater_than_max_degree_to_resolve = {skip_greater_than_max_degree_to_resolve}")
        
        return ed.low_degree_branch_errors_limb_branch_dict(
            neuron_obj,
            skip_greater_than_max_degree_to_resolve=skip_greater_than_max_degree_to_resolve,
            **kwargs
        )
        
        return ed.low_degree_branch_errors_limb_branch_dict(
            neuron_obj,**kwargs
        )
        
class InhAxonLowDegreeErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "inh_axon_low_degree"
        skip_greater_than_max_degree_to_resolve: bool = False
    
    def error_limb_branch_dict(
        self,
        neuron_obj,
        width_max = None,
        upstream_width_max = None,
        max_degree_to_resolve_absolute = None,
        filters_to_run = None,
        **kwargs
        ):

        skip_greater_than_max_degree_to_resolve = getattr(self.config,"skip_greater_than_max_degree_to_resolve",False)
        print(f"inside ExcAxonLowDegreeErrorDetector, skip_greater_than_max_degree_to_resolve = {skip_greater_than_max_degree_to_resolve}")
        
        if width_max is None:
            width_max = pru.width_max_high_low_degree_inh_global
        if upstream_width_max is None:
            upstream_width_max = pru.upstream_width_max_high_low_degree_inh_global
        if max_degree_to_resolve_absolute is None:
            max_degree_to_resolve_absolute = pru.max_degree_to_resolve_absolute_low_degree_inh_global
            
        
        if filters_to_run is None:
            filters_to_run = [
                        gf.axon_webbing_filter,
                        gf.thick_t_filter,
                        #gf.axon_double_back_filter,
                        gf.axon_double_back_inh_filter,
                        gf.fork_divergence_filter,
                        gf.fork_min_skeletal_distance_filter,

                    ]
        
        
        return ed.low_degree_branch_errors_limb_branch_dict(
            neuron_obj,
            width_max = width_max,
            upstream_width_max = upstream_width_max,
            max_degree_to_resolve_absolute = max_degree_to_resolve_absolute,
            filters_to_run=filters_to_run,
            skip_greater_than_max_degree_to_resolve=skip_greater_than_max_degree_to_resolve,
            **kwargs
        )
    
    
class ExcAxonWidthJumpErrorDetector(ged.LimbBranchErrorDetector):
    @dataclass
    class Config:
        name: str = "exc_axon_width_jump"
    
    def error_limb_branch_dict(self,neuron_obj,**kwargs):
        return ed.width_jump_up_axon(
            neuron_obj,**kwargs
        )
    
