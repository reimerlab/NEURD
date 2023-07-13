

from python_tools import numpy_dep as np

def ais_count_bins_dict(
    ais_distance_min = 0,
    ais_distance_max = 50_000,
    interval_dist = None,
    n_intervals = 20,
    verbose = False,
    name_method = "upper",
    ais_distance_name = "ais_distance",
    prefix="n_ais"
    ):
    """
    Purpose: To generate a dictionary that creates projections
    to discretely count what bins the ais synapses are in

    Pseudocode: 
    1) using the min, max and step size compute the intervals
    2) develop a dictionary mapping a name to the dj query for that interval
    """
    if interval_dist is None:
        interval_dist = (ais_distance_max-ais_distance_min)/n_intervals

    if verbose:
        print(f"interval_dist = {interval_dist}")

    bins = np.arange(ais_distance_min,ais_distance_max+1,interval_dist)
    upper_bins = bins[1:]
    lower_bins = bins[:-1]

    restr_dicts = dict()
    for j,(up,low) in enumerate(zip(upper_bins,lower_bins)):
        if name_method == "upper":
            curr_name = f"{int(up)}"
        elif name_method == "range":
            curr_name = f"{int(low)}_to_{int(up)}"
        else:
            raise Exception("")

        curr_name = f"{prefix}_{curr_name}"

        if j == len(upper_bins) -1:
            upper_inequality = "<="
        else:
            upper_inequality = "<"
        restr_dicts[curr_name] = f"SUM(({ais_distance_name} >= {low})*({ais_distance_name} {upper_inequality} {up}))"

    return restr_dicts



def filter_ais_df_cell_type_splits_by_n_ais_perc(
    df,
    column = "total_ais_postsyn",
    percentile = 99,
    category_columns = "cell_type",
    verbose = True,
    ):
    """
    Purpose: Want to filter the excitatory and inhibitory cells
    for a certain percentile
    """

    return pu.filter_df_splits_by_column_percentile(
        df,
        column = column,
        split_columns = category_columns,
        percentile_upper = percentile,
        percentile_lower = 0,
        verbose = verbose,
    )

n_ais_prefix = "n_ais_"
def n_ais_columns(
    df,
    min_dist = 0,
    max_dist = np.inf,
    verbose = False):
    
    cols = np.array([k for k in df.columns if k[:len(n_ais_prefix)] == n_ais_prefix])
    num = np.array([float(k.replace(n_ais_prefix,"")) for k in cols])
    
    if verbose:
        print(f"num = {num}")
        
    return list(cols[(num>min_dist) & (num<=max_dist) ])


def n_ais_sum_from_min_max_dist(
    df,
    min_dist = 10_000,
    max_dist = 40_000,
    column = None,
    verbose = False,):
    """
    Purpose: return sum sum of 
    a range of min dist to max dist

    Pseudocode: 1) Get all of the column in range
    """

    cols = n_ais_columns(df,min_dist=min_dist,max_dist = max_dist,verbose=verbose,)
    value = df[cols].sum(axis = 1)

    if column is not None:
        df[column] = value
    else:
        return value


#--- from python_tools ---
from python_tools import numpy_dep as np
from python_tools import numpy_utils as nu
from python_tools import pandas_utils as pu
