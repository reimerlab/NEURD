from python_tools import module_utils as modu
from pathlib import Path

default_data_type = "microns"

def set_volume_params(
    volume = default_data_type,
    verbose = False,
    verbose_loop = False):
    
    directory = Path(f"{__file__}").parents[0]
    modu.all_modules_set_global_parameters_and_attributes(
        data_type = volume,
        directory=directory,
        verbose = verbose,
        verbose_loop = verbose_loop,
        from_package = f"neurd"
    )
    
set_volume_params()
    