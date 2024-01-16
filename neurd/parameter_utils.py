from pathlib import Path
import numpy as np
from os import sys
import time


modes_default = (
        "default",
        "h01",
        "microns",
)

att_types_default = ("global_parameters","attributes")
default_category = "no_category"
global_param_suffix = "_global"
suffixes_to_ignore_default = (
    "_global",
)



#---- code used to read in legacy parameters stored inside module into their own json file ----


def injest_nested_dict(
    data,
    filter_away_suffixes = True,
    suffixes_to_ignore = None,
    **kwargs):
    """
    Purpose: To remove any suffixes from a diction
    """
    if suffixes_to_ignore is None:
        suffixes_to_ignore = suffixes_to_ignore_default
        
    data = gu.flatten_nested_dict(data)
    data = gu.remove_dict_suffixes(
        data,
        suffixes = suffixes_to_ignore
    )
    
    return data
    
class Parameters:
    def __init__(
        self,
        data = None,
        filepath = None,
        **kwargs):
        # need to have a much better parsing function
        if isinstance(data,self.__class__):
            data = data._dict.copy()
            
        if data is None:
            data = jsu.json_to_dict(filepath)
        self._dict = injest_nested_dict(data,**kwargs)
        
    def json_dict(self):
        return jsonable_dict(self._dict)
    
    # def __getattr__(self,k):
    #     if k[:2] == "__":
    #         raise AttributeError(k)
    #     try:
    #         return self._dict[k]
    #     except:
    #         return getattr(self._dict,k)
        
    # def __setattr__(self,k,v):
    #     print(f"inside setattr small param")
    #     if k in self._dict:
    #         self._dict[k] = v
    #     else:
    #         self.__dict__[k] = v
            
    def __getattr__(self,k):
        if k[:2] == "__" or k == "_dict":
            raise AttributeError(k)
        try:
            return self._dict[k]
        except:
            return getattr(self._dict,k)
        
    def __setattr__(self,k,v):
        if hasattr(self,"_dict") and k in self._dict:
            self._dict[k] = v
        else:
            self.__dict__[k] = v
        
    def __contains__(self, item):
        return item in self._dict

    @property
    def dict(self):
        return self._dict
    
    def __getitem__(self,k):
        return self._dict[k]
    
    def __setitem__(self,k,v):
        self._dict[k] = v
        
    def update(self,data):
        if not isinstance(data,self.__class__):
            data = self.__class__(data)
        self._dict.update(data._dict)
        
    def __str__(self):
        try:
            return str(jsu.dict_to_json(self._dict))
        except:
            return str(self._dict)
    
    def attr_map(
        self,
        attr_list=None,
        suffixes_to_ignore = None,
        plus_unused = False,
        return_used_params = False,
        return_unused_params = False,
        verbose = False,):
        """
        Example: 
        p_obj['apical_utils'].attr_map(
            ["multi_apical_height","candidate_connected_component_radius_apical"]
        )
        """
        if attr_list is None:
            attr_list = list(self._dict.keys())
        
        if suffixes_to_ignore is None:
            suffixes_to_ignore = suffixes_to_ignore_default
        
        att_dict = dict()
        attr_list = nu.to_list(attr_list)
        
        param_used = []
        
        for k in attr_list:
            success = False
            if k not in self._dict:
                for suf in suffixes_to_ignore:
                    new_name = k.replace(suf,"")
                    if  new_name in self._dict:
                        att_dict[k] = self._dict[new_name]
                        success = True
                        param_used.append(new_name)
                        break
            else:
                att_dict[k] = self._dict[k]
                param_used.append(k)
                success = True
                
            if verbose and not success:
                print(f"Could not locate: {k}")
            
        if plus_unused or return_unused_params:
            attr_unused = np.setdiff1d(
                list(self._dict.keys()),
                     param_used,
            )
            
        if plus_unused:
            att_dict.update({k:self[k] for k in attr_unused})
            
        if not return_unused_params and not return_used_params:
            return att_dict
        else:
            return_value = [att_dict]

        if return_used_params:
            return_value.append(param_used)
            
        if return_unused_params:
            return_value.append(attr_unused)
        
        return return_value
    
import copy
class PackageParameters:
    def __init__(
        self,
        data = None,
        filepath = None,
        shared_data = None, # for any parameters shared acroos modules
        ):
        if isinstance(filepath,self.__class__):
            self.data = copy.deepcopy(filepath._data)
            self.filepath = filepath.filepath
            return
        
        self.filepath = filepath
        if data is None:
            if filepath is None:
                data = {}
            else:
                data = jsu.json_to_dict(filepath)
            
        self._data = {mod_name:Parameters(data)
                      for mod_name,data in data.items()}
        
        if shared_data is None:
            shared_data = {}
            
        for k,v in shared_data.items():
            setattr(self,k,v)
        
    def __str__(self):
        curr_str = ""
        for k,v in self._data.items():
            curr_str+=(f"\n       ---{k}.py---\n")
            curr_str+=(str(v))
        return curr_str 
    
    def __getitem__(self,k):
        return self._data[k]   
    def __setitem__(self,k,v):
        self._data[k] = v 
    def __contains__(self, item):
        return item in self._data 
        
    def module_attr_map(
        self,
        module_name=None,
        attr_list = None,
        module = None,
        plus_unused = False,
        **kwargs):
        
        if module_name is None:
            module_name = module.__name__.split(".")[-1]
            
        if attr_list is None:
            if module is not None:
                attr_list = parameter_list_from_module(
                    module
                )
                
        if module_name in self._data:
            return self._data[module_name].attr_map(
                attr_list = attr_list,
                plus_unused=plus_unused,
                **kwargs
            )
        else:
            return {}
        
    def module_attr_map_requested(
        self,
        module,
        plus_unused = True,
        ):
        
        return self.module_attr_map(
            module = module,
            plus_unused=plus_unused,
        )
        
    def update(self,other_obj):
        other_obj = self.__class__(other_obj)
        for k in other_obj.keys():
            if k in self:
                self[k].update(other_obj[k])
            else:
                self._data.update({k:Parameters(other_obj[k])})
            
    def __getattr__(self,k):
        if k[:2] == "__" or k == "_data":
            raise AttributeError(k)
        try:
            return self._data[k]
        except:
            return getattr(self._data,k)
        
    def __setattr__(self,k,v):
        if hasattr(self,"_data") and k in self._data:
            self._data[k] = v
        else:
            self.__dict__[k] = v
            

def parameter_list_from_module(
    module,
    verbose = False,
    clean_dict = False,
    att_types = None,
    add_global_suffix = True,
    ):
    """
    Purpose: Know what parameters a modules needs to set

    Pseudocode: 
    1) Get the default dictionary of parameters and attributes
    2) export the keys as a list
    
    Ex: 
    from neurd import connectome_utils as conu
    parameter_list_from_module(
        conu,
        verbose = False
    )
    """

    module_params = gu.flatten_nested_dict(
        modes_global_param_and_attributes_dict_from_module(
            module,
            modes = "default",
            clean_dict = clean_dict,
            att_types=att_types,
            add_global_suffix=add_global_suffix,
        )
    )

    params_to_request = list(module_params.keys())
    if verbose:
        print(f"Required parameters for {module.__name__} ({len(params_to_request)})")
        for k in params_to_request:
            print(f"{k}")
            
    return params_to_request   
        

def jsonable_dict(data):
    if isinstance(data,dsu.DictType):
        data = data.asdict()
    return {k:v for k,v in data.items()
            if jsu.is_jsonable(v)}

def clean_modules_dict(data):
    for module,module_dict in data.items():
        for param_type,param_dict in data.items():
            for cat,cat_dict in param_dict.items():
                param_dict[cat] = jsonable_dict(cat_dict)
                        
    return data

def add_global_name_to_dict(mydict):
    return {f"{k}_global":v for k,v in 
        mydict.items() 
    }
    
def modes_global_param_and_attributes_dict_from_module(
    module,
    verbose = False,
    modes = None,
    att_types = None,
    default_name = "no_category",
    add_global_suffix = False,
    clean_dict = True,
    ):
    """
    Purpose: To read in parameter and attribute dictionaries,
    add it to a bigger dictionary and then be able to export the
    dictionary
    """
    
    if modes is None:
        modes = modes_default
        
    modes = nu.to_list(modes)
        
    if att_types is None:
        att_types = att_types_default
        
    att_types = nu.to_list(att_types)
    
    mod_name = module.__name__.split(".")[-1]

    if verbose:
        print(f"mod_name = {mod_name}")

    mode_jsons = {}
    
    

    for mode in modes:
        mode_dict = {mod_name:dict()}
        for att_type in att_types:
            local_dict = dict()
            att_param_name = f"{att_type}_dict_{mode}"

            att_dicts = [k for k in dir(module)
                                  if k[:len(att_param_name)] == att_param_name]
            
            #print(f"att_param_name = {att_param_name}")
            #print(f"att_dicts = {att_dicts}")
            
            cat_name = None
            if len(att_dicts) == 0:
                if verbose:
                    print(f"No {att_type} dicts found in {mod_name}")
            if len(att_dicts) > 1:
                """
                Addition: extracted 
                """

                att_dicts.remove(att_param_name)
                for ad in att_dicts:
                    cat_name = ad.replace(f"{att_param_name}_","")
                    local_dict[cat_name] = getattr(module,ad)
                    
                        
                # adding back parameters from original
                combined_dict = gu.merge_dicts(list(local_dict.values()))
                
                #print(f"combined_dict = {combined_dict}")
                
                leftover_dict = {k:v for k,v 
                                 in getattr(module,att_param_name).items()
                                 if k not in combined_dict}
                
                if len(leftover_dict) > 0:
                    local_dict[default_name] = leftover_dict
                    
                #print(f"\n\nlocal_dict[default_name] = {local_dict}")
                
                
                if add_global_suffix and att_type == 'global_parameters':
                    for cat_name in local_dict:
                        local_dict[cat_name] = add_global_name_to_dict(local_dict[cat_name])
                        
                #print(f"\n\nlocal_dict[default_name] AFTER= {local_dict}")
                
            elif len(att_dicts) == 1:
                cat_name = default_name
                local_dict[cat_name] = getattr(module,att_dicts[0])
                
                if add_global_suffix and att_type == 'global_parameters':
                    local_dict[cat_name] = add_global_name_to_dict(getattr(module,att_dicts[0]))
                else:
                    local_dict[cat_name] = getattr(module,att_dicts[0])
            else:
                pass


            if len(local_dict) > 0:
                mode_dict[mod_name][att_type] = local_dict

        if len(mode_dict[mod_name]) > 0:
            if clean_dict:
                mode_dict = clean_modules_dict(mode_dict)
            mode_jsons[mode] = mode_dict
            
    return mode_jsons


def modes_global_param_and_attributes_dict_all_modules(
    directory,
    verbose = False,
    clean_dict = True,
    ):
    """
    Purpose: to generate the nested dictionary
    for all of the modules in the neurd folder

    Pseudocode: 
    1) Load all of the modules in a directory 
    (and get references to them)

    2) For each module: generate the nested dictionary
    
    3) update the larger dictionary
    """

    modes = paru.modes_default
    final_dict = {k:dict() for k in modes}

    mod_objs = modu.load_modules_in_directory(
        directory,
        return_objects=True,
    )

    for mod in mod_objs:
        return_dict = paru.modes_global_param_and_attributes_dict_from_module(
        module = mod,
        verbose = verbose,
        clean_dict=clean_dict,
        )

        for k,v in return_dict.items():
            final_dict[k].update(v)

    return final_dict

def global_param_and_attributes_dict_to_separate_mode_jsons(
    data,
    filepath = f"./",
    filename = f"[mode_name]_config.json",
    filename_mode_placeholder = "[mode_name]",
    indent = None,
    verbose = False,
    modes = None
    ):

    """
    Purpose: To dump the dictionaries
    generated from modules into a json format

    Pseudocode: 
    For each mode: 
    1) Get the dictionary
    2) convert dictionary into a json file
    """
    if modes is None:
        modes = modes_default
    
    modes = nu.to_list(modes)
    
    filepath = Path(filepath)
    filepath.mkdir(exist_ok = True)
    
    saved_name = str(filename)

    for mode in modes:
        if filename_mode_placeholder in saved_name:
            new_name = saved_name.replace(filename_mode_placeholder,mode)
        else:
            new_name = saved_name

        if new_name[-5:] != ".json":
            new_name = f"{new_name}.json"

        mode_dict = data[mode]
        #mode_dict = clean_mode_dict(mode_dict)

        total_path = Path(filepath) / Path(new_name)

        if verbose:
            print(f"Writing mode = {mode} to:\n   {str(total_path.absolute())}")

        jsu.dict_to_json_file(
            data = mode_dict,
            filepath = total_path,
            indent = indent,
        )
        

parameter_config_folder_name = "parameter_configs"
def parameter_config_folder(return_str = True):
    return_path = Path(__file__).parents[0]  / Path(f"{parameter_config_folder_name}")
    
    if return_str:
        return_path = str(return_path.absolute())
        
    return return_path


def parameters_from_filepath(
    filename = None,#"parameters_config_default.py",
    dict_name = "parameters",
    directory = None,
    filepath = None,
    return_dict = False,
    ):
    
    """
    Purpose: To import the parameter dictionary
    from a python file
    """

    if isinstance(filepath,dict):
        return_value = filepath
    else:
        if filepath is not None:
            filepath = Path(filepath)
            filename = filepath.stem
            directory = str(plu.parent_directory(filepath).absolute())
            
        module_name = filename.replace(".py","")
        
        if directory is None:
            directory = parameter_config_folder()

        if directory not in sys.path:
            sys.path.append(directory)

        exec(f"import {module_name}; from {module_name} import {dict_name}")
        return_value = eval(dict_name)
        
    if not return_dict:
        return_value =  PackageParameters(
            return_value
        )
        
    return return_value


def parameter_dict_from_module_and_obj(
    module,
    obj,
    parameters_obj_name = "parameters_obj",
    plus_unused = False,
    error_on_no_attr = True,
    verbose = False,
    ):
    """
    Purpose: using an object 
    (with a potentially PackageParameters attribute)
    , a dictionary of attributes to set based on a modules
    attributes and global parameters

    Pseudocode: 
    1) Get the params to set for module
    2) Use the list to get a dictionary from 
    the obj.PackageParameters
    3) Find diff between return dict and list
    4) Goes and gets different from the attribute of object

    """

    params_to_set = paru.parameter_list_from_module(module)

    par_obj = getattr(obj,parameters_obj_name,None)

    if par_obj is not None:
        param_dict = par_obj.module_attr_map(
            module = module,
            attr_list = params_to_set,
            plus_unused=plus_unused,
        )

    else:
        print(f"Warning: Parameter instance is not contained within object")
        param_dict = {}
        

    if verbose:
        print(f"---param_dict before obj namepsace---")
        for k,v in param_dict.items():
            print(f"{k}:{v}")

    params_to_find = np.setdiff1d(
        params_to_set,
        list(param_dict.keys())
    )

    if verbose:
        print(f"\nparams_to_find = {params_to_find}")

    for k in params_to_find:
        if hasattr(obj,k):
            param_dict[k] = getattr(obj,k)
        else:
            not_find_str = f"Could not find parameter {k} for {modu.module_name_no_prefix(module)}"
            if verbose:
                print(f"{not_find_str}")
            if error_on_no_attr:
                raise Exception(not_find_str)


    return param_dict

def this_directory():
    return str(Path(__file__).parents[0].absolute())

config_directory_name = "parameter_configs"
def config_directory():
    return str((
        Path(__file__).parents[0] / 
        config_directory_name
        ).absolute())
    

def set_parameters_for_directory_modules_from_obj(
    obj,
    directory = None,
    verbose_loop = False,
    from_package = "neurd",
    # -- for setting the parameters ---
    parameters_obj_name = "parameters_obj",
    verbose_param = False,
    error_on_no_attr = False,
    modules = None,
    ):
    """
    Purpose: to set attributes of all 
    modules in a directory using an object 
    (with a potentially PackageParameters attribute)

    Pseudocode: 
    For all modules in the directory
    1) Try importing the module
        -> if can't then skip
    2) Use the module and object to get
    a dictionary of all attributes to set
    3) Set the attributes

    """
    
    if directory is None:
        directory = this_directory()

    if modules is None:
        p = Path(directory)
        modules = pku.module_names_from_directories(
            p
        )

    for i in range(0,2):
        for k in modules:
            try:
                if verbose_loop:
                    print(f"--Working on module {k}--")

                imp_str = f"import {k}"
                if from_package is not None:
                    imp_str = f"from {from_package} {imp_str}"
                exec(imp_str)
                if verbose_loop:
                    print(f"Accomplished import")
            except Exception as e:

                if verbose_loop:
                    print(f"Failed import: {e} ")
                continue

            module = eval(k)
            p_dict = paru.parameter_dict_from_module_and_obj(
                module = module,
                obj = obj,
                parameters_obj_name = parameters_obj_name,
                plus_unused = i == 1,
                verbose = verbose_param,
                error_on_no_attr=error_on_no_attr,
            )
            
            #print(f"{k} p_dict = {p_dict}")
            

            for k,v in p_dict.items():
                setattr(module,k,v)


def export_package_param_dict_to_file(
    package_directory = None,
    mode = "default",
    clean_dict = False,
    export_filepath = None,
    export_folder = None,
    export_filename = None,
    return_dict = False,
    ):
    """
    Purpose: To export the parameters for a certain
    mode to a file
    """
    if package_directory is None:
        package_directory = str(Path(__file__).parents[0].absolute())

    nest_dict = paru.modes_global_param_and_attributes_dict_all_modules(
        package_directory,
        clean_dict = clean_dict,
        )[mode]

    if export_filepath is None:
        if export_folder is None:
            export_folder = Path("./")
        if export_filename is None:
            export_filename = Path(f"parameters_config_{mode}.py")
            
        export_filepath = str((Path(export_folder) / Path(export_filename)).absolute())
        
        
    export_filepath = str(Path(export_filepath).absolute())
        
    gu.print_nested_dict(
        nest_dict,
        filepath =export_filepath,
        overwrite = True
    )
    
    if return_dict:
        return nest_dict


def category_param_from_module(
    module,
    category = "no_category",
    verbose = False,
    ):
    """
    Purpose: Want to export parameters belonging 
    to a specific category in a module

    Psuedocode:
    1) 
    """
    return_dict = modes_global_param_and_attributes_dict_from_module(
        module,
        clean_dict=False,
        modes="default"
    )["default"][modu.module_name_no_prefix(module)]

    output_dict = dict()
    for k,v in return_dict.items():
        if category in v:
            curr_dict = v[category]
            if k == "global_parameters":
                curr_dict = add_global_name_to_dict(
                    curr_dict
                )
            output_dict.update(curr_dict)
            
    output_dict = {
        k:getattr(module,k) for k in output_dict.keys()
    }

    return output_dict



#--- from python-tools
from datasci_tools import package_utils as pku
from datasci_tools import module_utils as modu
from datasci_tools import data_struct_utils as dsu
from datasci_tools import json_utils as jsu
from datasci_tools import numpy_utils as nu
from datasci_tools import general_utils as gu
from datasci_tools import pathlib_utils as plu



from . import parameter_utils as paru