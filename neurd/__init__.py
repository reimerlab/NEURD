from python_tools import module_utils as modu
from pathlib import Path

default_data_type = "microns"

def set_volume_params(
    volume = default_data_type,
    verbose = False,
    verbose_loop = False):
    
    
    directory = Path(f"{__file__}").parents[0] 
    #print(f"Tried to set modules, directory = {directory}")
    modu.all_modules_set_global_parameters_and_attributes(
        data_type = volume,
        directory=directory,
        verbose = verbose,
        verbose_loop = verbose_loop,
        from_package = f"neurd"
    )
    
import os
from python_tools import package_utils as pku


pku.load_all_modules_in_package(
    package_directory = os.path.dirname(__file__),
    reload_after_load = True
)

set_volume_params()





