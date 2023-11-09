try:
    import cloudvolume
    from caveclient import CAVEclient
except:
    print(f"Must install cloudvolume and caveclinet")
    print(f"pip3 install cloud-volume --upgrade")
    print("pip3 install caveclient --upgrade")
    
    
try:
    from code_structure_tools.decorators import default_args,add_func
except:
    print(f"Must install code_structure_utils")
    print(f"https://github.com/bacelii/code_structure_tools")
    
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
from datasci_tools import json_utils as jsu
import trimesh
    
""" tables in public release
'synapses_pni_2',
 'baylor_gnn_cell_type_fine_model_v2',
 'nucleus_alternative_points',
 'connectivity_groups_v507',
 'proofreading_status_public_release',
 'allen_column_mtypes_v1',
 'allen_v1_column_types_slanted_ref',
 'aibs_column_nonneuronal_ref',
 'nucleus_ref_neuron_svm',
 'aibs_soma_nuc_exc_mtype_preds_v117',
 'baylor_log_reg_cell_type_coarse_v1',
 'apl_functional_coreg_forward_v5',
 'nucleus_detection_v0',
 'aibs_soma_nuc_metamodel_preds_v117',
 'coregistration_manual_v3'
"""

voxel_to_nm_scaling = np.array([4,4,40])

release_tables = {
    "minnie65_public_v117": dict(
        neuron_non_neuron= 'nucleus_neuron_svm',
        synapse = 'synapses_pni_2',
        nucleus = "nucleus_detection_v0",
        release_info = "proofreading_status_public_release",
    ),
    "minnie65_public": dict(
        synapse = 'synapses_pni_2',
        nucleus = "nucleus_detection_v0",
        gnn_cell_type_fine = "baylor_gnn_cell_type_fine_model_v2",
        neuron_non_neuron="nucleus_ref_neuron_svm",
        cell_type = "allen_column_mtypes_v1",
        release_info = "proofreading_status_public_release",
    )
    
}

env_filepath = "./.env"
release_name = "minnie65_public"
cloudvolume_secrets_path = Path("/root/.cloudvolume/secrets")
#release_name = "minnie65_public"

def table_dict_from_release_name(
    release_name,
    return_default = True,):
    table_dict = release_tables.get(
            release_name,
            None,
    )
    
    if return_default and table_dict is None:
        try:
            table_dict = release_tables[release_name]
        except:
            first_key = list(release_tables.keys())[0]
            table_dict = release_tables[first_key]
    
    return table_dict

def table_name_from_table_str(table_str):
    return f"{table_str}_table_name"

@default_args()
def set_global_table_names(
    release_name = None,
    verbose = False
    ):
    
    table_dict = table_dict_from_release_name(release_name)
    
    for k,v in table_dict.items():
        table_name = table_name_from_table_str(k)
        if verbose:
            print(f"Creating {table_name} = {v}")
        exec(f"global {table_name};{table_name}='{v}'",globals(),locals())
        
set_global_table_names()


@default_args()
def load_cave_token(env_filepath=None):
    try:
        load_dotenv(env_filepath)
    except Exception as e:
        print(f"{e}")
        return None
    cave_token = os.getenv('CAVE_TOKEN')
    return cave_token
        
@default_args()
def save_cave_token_to_cloudvolume_secrets(
    cave_token,
    cloudvolume_secrets_path = None,
    **kwargs
    ):
    
    cloudvolume_secrets_path.mkdir(exist_ok = True)
    
    data = {"token": cave_token}
    
    json_file = jsu.dict_to_json_file(
        data,
        cloudvolume_secrets_path / Path("cave-secret.json")
    )
    
    return json_file

@default_args()
def set_cave_auth(
    client = None,
    cave_token=None,
    env_filepath = None,
    set_global_token = True,
    **kwargs
    ):
    if cave_token is None:
        cave_token = load_cave_token(env_filepath=env_filepath)
        
    if cave_token is None:
        return
    
    client.auth.token = cave_token
    
    if set_global_token:
        try:
            client.auth.save_token(token=cave_token)
        except:
            client.auth.save_token(token=cave_token,overwrite = True)
            
    save_cave_token_to_cloudvolume_secrets(
        cave_token=cave_token,
        **kwargs
    )

def release_name_from_release(
    release,
    prefix = "minnie65_public"
    ):
    
    return f'{prefix}_{release}'

def release_from_release_name(release_name):
    return int(release_name.split("_")[-1])

@default_args()
def init_cave_client(
    release_name = None,
    env_filepath = None,
    cave_token = None,
    release = None,
    ):
    
    if release_name is None:
        release_name = release_name_from_release(release)
    
    client = CAVEclient(release_name)
    return client
    
def release_name_from_client(client):
    return client.datastack_name


# --------- functions for helping access client

def neuron_nucleus_df(
    client,
    neuron_non_neuron_table_name=None,
    verbose = False):
    nuc_df = client.materialize.query_table(
        neuron_non_neuron_table_name,
        filter_equal_dict={'cell_type':'neuron'},
        filter_out_dict={'pt_root_id':[0]}
    ) 
    
    if verbose:
        print(f"# of nuclei = {len(verbose)}")
    
    return pd.DataFrame(nuc_df)

def segment_ids_with_nucleus(
    client,
    neuron_non_neuron_table_name=None,
    verbose=False):
    nuc_df = neuron_nucleus_df(
        client,
        neuron_non_neuron_table_name=neuron_non_neuron_table_name,
        )
    return_list = np.unique(nuc_df["pt_root_id"].to_numpy())
    if verbose:
        print(f"# of segments with at least one nucleus: {len(return_list)}")
    
    return return_list

def table_size(self,client,table_name):
    return self.client.materialize.get_annotation_count(
        table_name
    ) 
    
# pick a segment you selected from neuroglancer to enter here
def presyn_df_from_seg_id(
    seg_id,
    client,
    verbose = False
    ):
    
    df = client.materialize.synapse_query(pre_ids=seg_id)
    if verbose:
        print(f"# of pre = {len(df)}")
    return pd.DataFrame(df)

def postsyn_df_from_seg_id(
    seg_id,
    client,
    verbose = False
    ):
    
    df = client.materialize.synapse_query(post_ids=seg_id)
    if verbose:
        print(f"# of post = {len(df)}")
    return pd.DataFrame(df)
    
@default_args()
def prepost_syn_df_from_cave_syn_df(
    syn_df,
    seg_id,
    columns = (
        "segment_id",
        "segment_id_secondary",
        "synapse_id",
        "prepost",
        "synapse_x",
        "synapse_y",
        "synapse_z",
        "synapse_size",
    ),
    voxel_to_nm_scaling = None,
    
    ):
    """
    Purpose: Want to reformat the synapse dataframe from the CAVE table
    to the standard synapse format

    --- old columns ---
    "pre_pt_root_id"
    "post_pt_root_id"
    "size"
    "id"
    "prepost"

    --- new columns ---
    segment_id,
    segment_id_secondary,
    synapse_id,
    prepost,
    synapse_x,
    synapse_y,
    synapse_z,
    synapse_size,

    ctr_pt_position

    Pseudocode: 
    For presyn/postsyn:
    1) Restrict the dataframe to the current segment_id
    
    Example: 
    from neurd import cave_client_utils as ccu
    ccu.prepost_syn_df_from_cave_syn_df(
        syn_df,
        seg_id=seg_id,
    )
    """
    new_dfs = []
    syn_types = ["pre","post"]
    columns = list(columns)
    
    #print(f"Inside prepost syn voxel_to_nm_scaling = {voxel_to_nm_scaling}")
    size_scaling = np.prod(voxel_to_nm_scaling)

    for i in range(len(syn_types)):

        prepost = syn_types[i]
        other = syn_types[1-i]

        prepost_col = f"{prepost}_pt_root_id"
        other_col = f"{other}_pt_root_id"

        prepost_df = syn_df.query(f"({prepost_col} == {seg_id})").reset_index(drop=True)
        center_positions = np.vstack(prepost_df["ctr_pt_position"].to_numpy())
        prepost_df["synapse_x"] = center_positions[:,0]*voxel_to_nm_scaling[0].astype('int')
        prepost_df["synapse_y"] = center_positions[:,1]*voxel_to_nm_scaling[1].astype('int')
        prepost_df["synapse_z"] = center_positions[:,2]*voxel_to_nm_scaling[2].astype('int')
        prepost_df["synapse_size"] = prepost_df["size"]*size_scaling.astype('int')

        prepost_df_renamed = pu.rename_columns(
            prepost_df,
            {
                prepost_col:"segment_id",
                other_col:'segment_id_secondary',
                "id":"synapse_id",
            },
        )
        prepost_df_renamed["prepost"] = prepost

        prepost_df_restr = prepost_df_renamed[columns]
        new_dfs.append(prepost_df_restr)

    new_syn_df = pd.concat(new_dfs,axis = 0).reset_index(drop=True)
    
    return new_syn_df

def pre_post_df_from_seg_id(
    seg_id,
    client,
    concat = True,
    verbose = False,):
    """
    Example:
    seg_id = 864691137197197121
    """

    # inputs/outputs
    pre_df = presyn_df_from_seg_id(
        seg_id=seg_id,
        client = client)
    post_df = postsyn_df_from_seg_id(
        seg_id=seg_id,
        client = client)
    if verbose:
        print(f"# of pre = {len(pre_df)}")
        print(f"# of post = {len(post_df)}")
        
    if concat:
        return pd.concat([pre_df,post_df],axis=0).reset_index(drop=True)
    
    return pre_df,post_df

@default_args()
def synapse_df_from_seg_id(
    seg_id,
    client,
    verbose = False,
    voxel_to_nm_scaling=None,
    ):
    
    #print(f"voxel_to_nm_scaling inside synapse_df_from_seg_id = {voxel_to_nm_scaling}")
    new_syn_df = pre_post_df_from_seg_id(
        seg_id,
        client,
        verbose = verbose,
    )
    
    new_syn_df=prepost_syn_df_from_cave_syn_df(
        new_syn_df,
        seg_id=seg_id,
        voxel_to_nm_scaling = voxel_to_nm_scaling,
    )
    
    return new_syn_df

def get_tables(client):
    return client.materialize.get_tables()


def mesh_from_seg_id(
    seg_id,
    client,
    use_https = True,
    progress=False,
    return_trimesh = True,
    verbose = False):
    """
    Purpose: To fetch a mesh from the cave
    table using cloudvolume
    
    example_cell_id = 864691137197197121
    """
    # to access the minnie65 public release dataset
    # you initialize the client like this
    cv = cloudvolume.CloudVolume(
        client.info.segmentation_source(),
        progress=progress,
        use_https = use_https,
    )
    # which, given a root_id, can be used to get a mesh
    # cloud volume returns a dictionary with the neuron segment id as the key 
    # and the mesh as the value
    
    mesh = cv.mesh.get(seg_id)[seg_id]
    
    if return_trimesh:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
        )
        
    return mesh

# ------------------- wrapper for interface -------
@add_func(
    func_to_add_unpacked = [
        neuron_nucleus_df,
        segment_ids_with_nucleus,
        set_cave_auth,
        load_cave_token,
        table_size,
        presyn_df_from_seg_id,
        postsyn_df_from_seg_id,
        pre_post_df_from_seg_id,
        get_tables,
        mesh_from_seg_id,
        synapse_df_from_seg_id,
        mesh_from_seg_id,
        
    ],
)
class CaveInterface():
    @default_args()
    def __init__(
        self,
        release_name = None,
        env_filepath = None,
        cave_token = None,
        client = None,
        release = None,
        ):
        
        if client is None:
            client = init_cave_client(
                release = release,
                env_filepath = env_filepath,
                release_name = release_name,
                cave_token = cave_token,
            )
        
        self.client = client
        self.release_name = release_name_from_client(client)
        #self.release = release_from_release_name(self.release_name)
        set_cave_auth(
            self.client,
            cave_token=cave_token,
            env_filepath = env_filepath
        )
    
        self.set_table_names()
        
        
    def set_table_names(self,table_dict=None):
        if table_dict is None:
            table_dict = table_dict_from_release_name(
                self.release_name
            )
        for k,v in table_dict.items():
            setattr(
                self,
                table_name_from_table_str(k),
                v
            )
        
    @property
    def release_info(self):
        return  self.client.materialize.get_table_metadata(
            self.release_info_table_name
        )
        
    
    @property
    def voxel_to_nm_scaling(self):
        return np.array(
            self.release_info['voxel_resolution']
        )
        

    def __getattr__(self,k):
        if k[:2] == "__":
            raise AttributeError(k)
        try:
            return getattr(self.client,k)
        except:
            return self.__getattribute__(k)
    
    


from datasci_tools import pandas_utils as pu

import numpy as np
import pandas as pd


        
