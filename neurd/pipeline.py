import copy
from python_tools import function_utils as funcu

class StageInfo:
    def __init__(
        self,
        *args,
        **kwargs):
        if len(args) > 0:
            if type(args[0]) == type(self):
                kwargs.update(args[0].export()) 
            elif type(args[0]) == dict:
                kwargs.update(args[0])
            elif type(args[0]) == None:
                pass
        for k,v in kwargs.items():
            setattr(self,k,v)
            
    def export(self):
        export_dict = {k:getattr(self,k) for k in dir(self)
                       if k[:2] != "__" and "bound method" not in str(getattr(self,k))
                       #and not funcu.isfunction(getattr(self,k))
                       }
        return export_dict
    
    def update(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
            
class StageProducts(StageInfo):
    pass
            
class PipelineInfo:
    def __init__(
        self,
        *args,
        **kwargs
        ):
        
        if len(args) > 0:
            if args[0] is None:
                pass
            elif type(args[0]) == type(self):
                kwargs.update(args[0].export())
            
        self.products = {k:StageProducts(v) for k,v in kwargs.items()}
        # for k,v in self.products.items():
        #     if type(v) != StageProducts:
        #         raise Exception(f"{k} is not a StageProducts object")
            
    def __getattr__(self, name):
        if name in self.products:
            return self.products[name]
        else:
            return self.__getattribute__(name)
        
    
    def export(self):
        return {k:v.export() for k,v in self.products.items()}
    
    
    def stage_names(self):
        return list(self.products.keys())
    
    def set_stage_products(
        self,
        stage_name,
        attr_dict = None,
        clean_write = True,
        **kwargs
        ):
        
        if attr_dict is None:
            attr_dict = dict()
            
        attr_dict.update(kwargs)
        
        if clean_write or stage_name not in products:
            self.products[stage_name] = StageProducts(attr_dict)
        else:
            self.products[stage_name].update(**attr_dict)
        
    def get_stage_attr(
        self,
        attribute_name,
        stages = None,
        verbose = False,
        default_value = None):
        """
        Purpose: Want to get a stage attribute but not 
        necessarily know what the stage name is called

        """
        
        if stages is not None:
            stages = self.stages_names
            
        for st in stages:
            v = self.products[st]
            if hasattr(v,attribute_name):
                if verbose:
                    print(f"Found {attribute_name} in stage {st}")
                return getattr(v,attribute_name)
        
        if verbose:
            print(f"{attribute_name} was not found in any stage")
            
        return default_value
    
# class PipelineProducts(PipelineInfo):
#     pass
    
# class Pipeline():
#     def __init__(
#         self,
#         mesh = None,
#         **kwargs):
        
#         self._mesh = None
#         self._mesh_file = None
        
        
#     @property
    
    
#     def load    
 