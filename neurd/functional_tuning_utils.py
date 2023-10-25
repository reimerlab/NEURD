

from datasci_tools import numpy_dep as np

def cdiff(alpha, beta, period=np.pi,rad = True):    
    return nu.cdiff(alpha=alpha,beta=beta,period=period,rad=rad)

def cdist(alpha, beta, period=np.pi,rad=True):
    return nu.cdist(alpha, beta, period=period,rad=rad)


def add_on_delta_to_df(
    df,
    ori_name_1 = "ori_rad",
    ori_name_2 = "ori_rad_post",
    dir_name_1 = "dir_rad",
    dir_name_2 = "dir_rad_post"):
    df["delta_ori_rad"] = ftu.cdist(
        df[ori_name_1],
        df[ori_name_2],
    )

    df["delta_dir_rad"] = ftu.cdist(
        df[dir_name_1],
        df[dir_name_2],
    )
    
    return df


#--- from datasci_tools ---
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu

from . import functional_tuning_utils as ftu