'''



To Do: Want to add how close a spine is to upstream and downstream endpoint



'''
import copy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
from pykdtree.kdtree import KDTree
import seaborn as sns
import time
import time 
from datasci_tools import numpy_dep as np
from datasci_tools import module_utils as modu
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu

try:
    import cgal_Segmentation_Module as csm
except:
    pass


volume_divisor = 1000_000_000

spine_attributes = ["mesh_face_idx",
                      "mesh",
                      "neck_face_idx",
                      "head_face_idx",
                      "neck_sdf",
                      "head_sdf",
                      "head_width",
                      "neck_width",
                      "volume",
                    "spine_id",
                    "sdf",
                    #features to help with electrical properties
                     "endpoints_dist",
                     "upstream_dist",
                     "downstream_dist",
                     "coordinate",
                    "coordinate_border_verts",
                     "closest_sk_coordinate",
                     "closest_branch_face_idx",
                     "closest_face_dist",
                      "closest_face_coordinate",
                      "soma_distance",
                      "soma_distance_euclidean",
                      "compartment",
                      "limb_idx",
                      "branch_idx",
                        "skeleton",
                        "skeletal_length",
                    
                    #attributes for more spine features
                    "bbox_oriented_side_lengths",
                    "head_bbox_oriented_side_lengths",
                    "neck_bbox_oriented_side_lengths",
                    "head_mesh_splits",
                    "head_mesh_splits_face_idx",
                    
                    # attributes from the branch_obj
                    "compartment",
                    "branch_width_overall",
                    "branch_skeletal_length",
                    "branch_width_at_base",
                     ]
head_neck_shaft_dict = dict(no_label=-1,
                            head=0,
                           neck=1,
                           shaft=2,
                           no_head=3,
                           bouton=4,
                           non_bouton=5)

def spine_labels(include_no_label = False):
    if include_no_label:
        return list(head_neck_shaft_dict.keys())
    else:
        return [k for k in head_neck_shaft_dict.keys() if k != "no_label"]
    

    

head_neck_shaft_dict_inverted = gu.invert_mapping(head_neck_shaft_dict,one_to_one=True)

def decode_head_neck_shaft_idx(array):
    return [head_neck_shaft_dict_inverted.get(k,"no_label")
            for k in array
    ]

neck_color_default = "gold"#"yellow"#,"pink"#,"yellow"# "aqua"
head_color_default = "red"
no_head_color_default = "black"
shaft_color_default = "lime"
bouton_color_default = "orange"
base_color_default = "blue"
center_color_default = "orange"

spine_bouton_labels_colors_dict = dict(
head = "red",
neck = "aqua",
shaft = "plum",
no_head = "limegreen",
bouton = "greenyellow",
non_bouton = "aqua",
no_label = "royalblue"
)

# ---------- queries for differnt types of spines -----------
spine_volume_to_spine_area_min = 0.01
# slab_query_1_dj = "((shaft_border_area > 0.18) AND (spine_volume < 0.05)) OR (shaft_border_area is NULL)"
# slab_query_2_dj = f"((NOT (spine_volume is NULL)) AND (spine_volume_to_spine_area < {spine_volume_to_spine_area_min}))"

slab_query_1 = "((shaft_border_area > 0.18) and (spine_volume < 0.05)) or (shaft_border_area != shaft_border_area)"
slab_query_2 = f"((spine_volume == spine_volume) and (spine_volume_to_spine_area < {spine_volume_to_spine_area_min}))"
slab_query_3 = f"(shaft_border_area > 0.5)"
slab_query_4 = f"(spine_to_shaft_border_area <= 2)"
slab_query_5 = f"(spine_area_per_faces) < 0.005"
long_merge = f"(head_skeletal_length > 2000)"
too_fat_of_neck = f"(neck_width_ray_80_perc > 400) and (n_heads > 0)"


measurable_features = ("spine_area",
            "spine_n_faces",
            "spine_volume",
            "spine_skeletal_length",
            "spine_width_ray",
            "spine_width_ray_80_perc",
                      )

n_heads_max = 1
min_distance_from_endpt = 2000
default_spine_restrictions = [
    f"downstream_dist > {min_distance_from_endpt}",
    f"upstream_dist > {min_distance_from_endpt}",
    f"n_heads <= {n_heads_max}",
]

spine_synapse_rename_dict  =dict(
            spine_compartment = "spine_compartment",
            syn_spine_volume = "spine_volume",
            syn_spine_area = "spine_area",
            syn_spine_width_ray_80_perc = "spine_width_ray_80_perc"
        )

def false_positive_queries(
    table_type = "pandas",
    invert = False,
    include_axons = True):
    return_queries = [
        slab_query_1,
        slab_query_2,
        slab_query_3,
        slab_query_4,
        slab_query_5,
        long_merge,
        #too_fat_of_neck
    ]
    if include_axons:
        return_queries.append("compartment == 'axon'")
    if invert:
        return_queries = [f"not ({k})" for k in return_queries]
    if table_type == "dj":
        return_queries  = [dju.dj_query_from_pandas_query(k) for k in return_queries]

    return return_queries

def spine_table_restriction_high_confidence(
    table_type = "pandas",
    include_default_restrictions=True,
    return_query_str = False,):
    return_value = false_positive_queries(table_type,invert = True)
    if include_default_restrictions:
        additional_queries = default_spine_restrictions.copy()
        if table_type == "dj":
            additional_queries = [dju.dj_query_from_pandas_query(k) for k in additional_queries]
        return_value += additional_queries
        
    if return_query_str:
        return_value = pu.query_str_from_list(return_value,table_type = table_type)
        
    return return_value
    
def filter_for_high_confidence_df(
    df,
    apply_default_restrictions = True,
    verbose = False):
    return_df = spu.filter_away_fp_from_df(df,verbose = verbose)
    if apply_default_restrictions:
        return_df =  return_df.query(spine_table_restriction_high_confidence(table_type='pandas',return_query_str=True))
    return return_df

def filter_away_fp_from_df(
    df,
    fp_queries = None,
    verbose = False,
    eta = 0.000001
    ):
    if verbose:
        print(f"Attempting to filter away fp")
    
    if fp_queries is None:
        fp_queries = false_positive_queries()
        
    df["spine_volume_to_spine_area"] = df["spine_volume"] / (df["spine_area"] + eta)
    df["spine_to_shaft_border_area"] = df["spine_area"] / (df["shaft_border_area"] + eta)
    df["spine_area_per_faces"] = df["spine_area"]/df["spine_n_faces"]
    
    return_df = pu.query_table_from_list(
        df,
        restrictions = [f"not ({k})" for k in fp_queries],
        verbose_filtering = verbose, 
    )
    
    return return_df

def colors_from_spine_bouton_labels(spine_bouton_labels):
    return [spine_bouton_labels_colors_dict[k] for k in spine_bouton_labels]

def spine_bouton_labels_to_plot():
    return list(spine_bouton_labels_colors_dict.keys())

class Spine:
    """
    Classs that will hold information about a spine extracted from a neuron

    Attributes
    ----------
    mesh_face_idx: a list of face indices of the branch that belong to the spine mesh
    mesh: the submesh of the branch that represent the spine (mesh_face_idx indexed into the branch mesh
    neck_face_idx: a list of face indices of the spine’s mesh that were classified as the neck (can be empty if not detected)
    head_face_idx:  list of face indices of the spine’s mesh that were classified as the head (can be empty if not detected)
    neck_sdf: the sdf value of the neck submesh from the clustering algorithm used to segment the head from the neck 
    head_sdf: the sdf value of the head submesh from the clustering algorithm used to segment the head from the neck 
    head_width: a width approximation using ray tracing of the head submesh
    neck_width:  a width approximation using ray tracing of the head submesh
    volume: volume of entire mesh
    spine_id: unique identifier for spine
    sdf:  the sdf value of the spine submesh from the clustering algorithm used to segment the spine from the branch mesh
    endpoints_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the branch skeletal endpoints
    upstream_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the upstream branch skeletal endpoint
    downstream_dist: skeletal walk distance of the skeletal point closest to the start of the spine protrusion to the downstream branch skeletal endpoint
    coordinate_border_verts: 
    coordinate: one coordinate of the border vertices to be used for spine locations
    bbox_oriented_side_lengths
    head_bbox_oriented_side_lengths
    neck_bbox_oriented_side_lengths
    head_mesh_splits
    head_mesh_splits_face_idx
    branch_width_overall
    branch_skeletal_length
    branch_width_at_base
    skeleton: surface skeleton over the spine mesh
    skeletal_length: length of spine skeleton

    # -- attributes similar to those of spine attribute
    closest_branch_face_idx
    closest_sk_coordinate: 3D location in space of closest skeletal point on branch for which spine is located
    closest_face_coordinate: center coordinate of closest mesh face  on branch for which spine is located
    closest_face_dist: distance from synapse coordinate to closest_face_coordinate
    soma_distance: skeletal walk distance from synapse to soma
    soma_distance_euclidean: straight path distance from synapse to soma center
    compartment: the compartment of the branch that the spine is located on
    limb_idx: the limb identifier that the spine is located on
    branch_idx: the branch identifier that the spine is located on

    """
    def __init__(
        self,
        mesh,
        calculate_spine_attributes=False,
        branch_obj = None,
        **kwargs):
        if type(mesh) == spu.Spine:
            for k,v in mesh.export().items():
                #setattr(self,k,v)
                try:
                    setattr(self,k,v)
                except:
                    setattr(self,f"_{k}",v)
            return 
            
        for a in spine_attributes:
            
            try:
                setattr(self,a,None)
            except:
                setattr(self,f"_{a}",None)
                
        self.mesh = mesh
        #if synapse_dict is not None:
        for k,v in kwargs.items():
            if k in spine_attributes:
                try:
                    setattr(self,k,v)
                except:
                    setattr(self,f"_{k}",v)
                
        if calculate_spine_attributes:
            self.calculate_spine_attributes(branch_obj=branch_obj)
            
                
    def calculate_spine_attributes(self,branch_obj=None):
        spu.calculate_spine_attributes(self,branch_obj=branch_obj)
        
        
    def export(
        self,  
        attributes_to_skip = None,
        attributes_to_add = None,
        **kwargs):
        return spu.export(
            self,
            attributes_to_skip=attributes_to_skip,
            attributes_to_add=attributes_to_add,
            **kwargs
            )
        
        
    @property
    def base_coordinate(self):
        return self.coordinate
    @property
    def base_coordinate_x_nm(self):
        return self.base_coordinate[0]
    @property
    def base_coordinate_y_nm(self):
        return self.base_coordinate[1]
    @property
    def base_coordinate_z_nm(self):
        return self.base_coordinate[2]
    
    @property
    def bbox_oriented_side_lengths(self):
        if self._bbox_oriented_side_lengths is None:
            self._bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.mesh)
        return self._bbox_oriented_side_lengths
    
    @property
    def head_bbox_oriented_side_lengths(self):
        if self.head_mesh is None:
            return [None,None,None]
        if len(self.head_mesh.faces) == 0:
            return [0,0,0]
        if self._head_bbox_oriented_side_lengths is None:
            self._head_bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.head_mesh)
        return self._head_bbox_oriented_side_lengths
    
    @property
    def neck_bbox_oriented_side_lengths(self):
        if self.neck_mesh is None:
            return [None,None,None]
        if len(self.neck_mesh.faces) == 0:
            return [0,0,0]
        if self._neck_bbox_oriented_side_lengths is None:
            self._neck_bbox_oriented_side_lengths = meshu.bounding_box_side_lengths_sorted(self.neck_mesh)
        return self._neck_bbox_oriented_side_lengths
    
    @property
    def spine_bbox_oriented_side_max(self):
        return self.bbox_oriented_side_lengths[0]
    bbox_oriented_side_max = spine_bbox_oriented_side_max
    @property
    def spine_bbox_oriented_side_middle(self):
        return self.bbox_oriented_side_lengths[1]
    bbox_oriented_side_middle = spine_bbox_oriented_side_middle
    @property
    def spine_bbox_oriented_side_min(self):
        return self.bbox_oriented_side_lengths[2]
    bbox_oriented_side_min = spine_bbox_oriented_side_min
    
    @property
    def head_bbox_oriented_side_max(self):
        return self.head_bbox_oriented_side_lengths[0]
    @property
    def head_bbox_oriented_side_middle(self):
        return self.head_bbox_oriented_side_lengths[1]
    @property
    def head_bbox_oriented_side_min(self):
        return self.head_bbox_oriented_side_lengths[2]
    
    @property
    def neck_bbox_oriented_side_max(self):
        return self.neck_bbox_oriented_side_lengths[0]
    @property
    def neck_bbox_oriented_side_middle(self):
        return self.neck_bbox_oriented_side_lengths[1]
    @property
    def neck_bbox_oriented_side_min(self):
        return self.neck_bbox_oriented_side_lengths[2]
    
    
    
    @property
    def head_mesh(self):
        return spu.head_mesh(self)
    
#     def head_mesh_splits(
#         self,
#         n_faces_threshold = 10):
#         if self.n_faces_head == 0:
#             return []
#         else:
#             return tu.split_significant_pieces(
#                 self.head_mesh,
#                 significance_threshold=n_faces_threshold,
#                 connectivity="vertices"
#             )

    @property
    def head_mesh_splits(self):
        if self.head_mesh is None:
            return None
        if self._head_mesh_splits is None:
            self._head_mesh_splits,self._head_mesh_splits_face_idx = spu.split_head_mesh(
                self,
            )
        return self._head_mesh_splits 
    
    @property
    def head_mesh_splits_face_idx(self):
        if self.head_mesh is None:
            return None
        if self._head_mesh_splits_face_idx is None:
            self._head_mesh_splits,self._head_mesh_splits_face_idx = spu.split_head_mesh(
                self,
            )
            
        return self._head_mesh_splits_face_idx
    
    @property
    def n_heads(self):
        if self.head_mesh_splits is None:
            return None
        return len(self.head_mesh_splits)
    
    def head_mesh_splits_from_index(self,index):
        return spu.head_mesh_splits_from_index(self,index=index)
    
    @property
    def head_mesh_splits_max(self):
        return spu.head_mesh_splits_from_index(self,index=0)
    
    @property
    def head_mesh_splits_min(self):
        return spu.head_mesh_splits_from_index(self,index=self.n_heads-1)
    
    @property
    def n_faces(self):
        return len(self.mesh.faces)
    
    @property
    def n_faces_head(self):
        if self.head_face_idx is None:
            return None
        return len(self.head_face_idx)
    
    @property
    def n_faces_neck(self):
        if self.neck_face_idx is None:
            return None
        return len(self.neck_face_idx)
    
    @property
    def neck_mesh(self):
        return spu.neck_mesh(self)
    
    @property
    def area(self):
        return self.mesh.area
    
    @property
    def no_head_mesh(self):
        return spu.no_head_mesh(self)
    
    @property
    def head_exist(self):
        return spu.head_exist(self)
    
    @property
    def no_head_face_idx(self):
        if self.head_exist:
            return np.array([])
        return self.neck_face_idx
    
    @property
    def skeletal_length(self):
        if self.skeleton is None:
            self.calculate_skeleton()
        return self._skeletal_length
    
    @property
    def skeleton(self):
        if self._skeleton is None:
            self.calculate_skeleton()
        return self._skeleton
    
    def calculate_head_neck(self,**kwargs):
        (self.head_face_idx,
         self.neck_face_idx,
         self.head_sdf,
         self.neck_sdf,
         self.head_width,
         self.neck_width) = spine_head_neck(
            mesh=self.mesh,
            return_meshes = False,
            no_head_coordinates = self.coordinate_border_verts,
            return_sdf = True,
            **kwargs)
        self._head_mesh_splits = None
    def calculate_face_idx(self,
                      original_mesh=None,
                    original_mesh_kdtree=None,
                          **kwargs):
        self.mesh_face_idx = spu.calculate_face_idx(self,
                                                   original_mesh=original_mesh,
                                                   original_mesh_kdtree=original_mesh_kdtree,
                                                   **kwargs)
        
        
    def plot_head_neck(self,**kwargs):
        spu.plot_head_neck(self)
        
    def calculate_closest_mesh_sk_coordinates(self,branch_obj,**kwargs):
        spu.calculate_spine_obj_mesh_skeleton_coordinates(branch_obj,self,**kwargs)
    
    def calculate_volume(self):
        if self.volume is None:
            self.volume = spu.volume_from_spine(self)
    
    def calculate_skeleton(self):
        self._skeleton = spu.skeleton_from_spine(self)
        self._skeletal_length = sk.calculate_skeleton_distance(self.skeleton)
        
    @property
    def mesh_center(self):
        return self.mesh.centroid
    
    @property
    def mesh_center_x_nm(self):
        return self.mesh_center[0]
    @property
    def mesh_center_y_nm(self):
        return self.mesh_center[1]
    @property
    def mesh_center_z_nm(self):
        return self.mesh_center[2]
    
    @property
    def sdf_mean(self):
        return np.mean(self.sdf)
    
    @property
    def sdf_median(self):
        return np.median_sdf(self.sdf)
    
    @property
    def sdf_90_perc(self):
        return np.percentile(self.sdf,90)
    
    @property
    def sdf_70_perc(self):
        return np.percentile(self.sdf,70)
    
    @property
    def endpoint_dist_0(self):
        if self.endpoints_dist is None:
            return None
        else:
            return self.endpoints_dist[0]
    @property   
    def endpoint_dist_1(self):
        if self.endpoints_dist is None:
            return None
        else:
            return self.endpoints_dist[1]
        
    @property
    def area_of_border_verts(self):
        return area_of_border_verts(self)
    coordinate_border_verts_area = area_of_border_verts
    border_area = area_of_border_verts
    shaft_border_area = area_of_border_verts
    
    #--- Mesh Attribute Functions for spine -----
    @property
    def spine_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'spine',**kwargs)

    @property
    def spine_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'spine',**kwargs)

    #--- Mesh Attribute Functions for head -----
    @property
    def head_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'head',**kwargs)

    @property
    def head_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'head',**kwargs)

    #--- Mesh Attribute Functions for neck -----
    @property
    def neck_width_ray(self,**kwargs):
        return width_ray_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_width_ray_80_perc(self,**kwargs):
        return width_ray_80_perc_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_area(self,**kwargs):
        return area_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_volume(self,**kwargs):
        return volume_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_skeletal_length(self,**kwargs):
        return skeletal_length_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_n_faces(self,**kwargs):
        return n_faces(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_x_nm(self,**kwargs):
        return bbox_min_x_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_y_nm(self,**kwargs):
        return bbox_min_y_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_min_z_nm(self,**kwargs):
        return bbox_min_z_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_x_nm(self,**kwargs):
        return bbox_max_x_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_y_nm(self,**kwargs):
        return bbox_max_y_nm_from_compartment(self,compartment = 'neck',**kwargs)

    @property
    def neck_bbox_max_z_nm(self,**kwargs):
        return bbox_max_z_nm_from_compartment(self,compartment = 'neck',**kwargs)
    
    
    
computed_attributes_to_export = (
        "area",
        "sdf_mean",
        "n_faces",
        "n_faces_head",
        "n_faces_neck",
        "mesh_center",
        "sdf_mean",
        "sdf_90_perc",
        "sdf_70_perc",
        "bbox_oriented_side_max",
        "bbox_oriented_side_middle",
        "bbox_oriented_side_min",
        "n_heads",
        "endpoint_dist_0",
        "endpoint_dist_1"
)
    
    
def calculate_soma_distance_euclidean(spine_obj,soma_center=None):
    if soma_center is None:
        return
    else:
        spine_obj.soma_distance_euclidean = np.linalg.norm(soma_center - spine_obj.mesh_center)
        
def calculate_soma_distance_skeletal(spine_obj,upstream_skeletal_length=None):
    if upstream_skeletal_length is None or spine_obj.upstream_dist is None:
        return
    spine_obj.soma_distance = spine_obj.upstream_dist + upstream_skeletal_length 
    
    

def calculate_branch_width_at_base(spine_obj,branch_obj):
    spine_obj.branch_width_at_base = bu.width_array_value_closest_to_coordinate(
                branch_obj,
                spine_obj.base_coordinate
    )
    
branch_overall_features_for_spine = ["width_overall","skeletal_length"]#branch_width_at_base
def calculate_branch_overall_features(
    spine_obj,
    branch_obj,
    branch_features = None):
    if branch_features is None:
        branch_features = branch_overall_features_for_spine

    branch_features = nu.to_list(branch_features)

    for feat in branch_features:
        if "str" in str(type(feat)):
            curr_value = getattr(branch_obj,feat)
        else:
            curr_value = feat(branch_obj)

        setattr(spine_obj,f"branch_{feat}",curr_value)
        
    
        
    
def calculate_spine_attributes(
    spine_obj,
    branch_obj=None,
    calculate_coordinates = True,
    calculate_head_neck = False,
    branch_shaft_mesh_face_idx=None,
    soma_center = None,
    upstream_skeletal_length = None,
    branch_features = None,
    verbose_time=False,
    mesh = None,
    **kwargs):
    
    if verbose_time:
        st = time.time()
    spine_obj.skeleton
    
    if verbose_time:
        print(f"Time for skeleton = {time.time() - st}")
        st = time.time()
    spine_obj.calculate_volume()
    if verbose_time:
        print(f"Time for volume = {time.time() - st}")
        st = time.time()
    spine_obj.bbox_oriented_side_lengths
    if verbose_time:
        print(f"Time for bbox = {time.time() - st}")
        st = time.time()
    
    if calculate_coordinates:
        if branch_obj is not None:
            spine_obj.calculate_closest_mesh_sk_coordinates(
                branch_obj,
                verbose_time=verbose_time,
                branch_shaft_mesh_face_idx=branch_shaft_mesh_face_idx,
            )
            if verbose_time:
                print(f"Time for closest_coordinate = {time.time() - st}")
                st = time.time()
            spu.calculate_endpoints_dist(branch_obj,spine_obj)
            if verbose_time:
                print(f"Time for calculate_endpoints_dist = {time.time() - st}")
                st = time.time()
            if branch_obj.endpoints_upstream_downstream_idx is not None:
                spu.calculate_upstream_downstream_dist_from_up_idx(
                    spine_obj,
                    branch_obj.endpoints_upstream_downstream_idx[0]
                )
                if verbose_time:
                    print(f"Time for calculate_upstream_downstream_dist_from_up_idx = {time.time() - st}")
                    st = time.time()
                    
                calculate_branch_width_at_base(spine_obj,branch_obj)
            
            
    
    if calculate_head_neck:
        if mesh is None:
            mesh = branch_obj.mesh
        spine_obj = spu.calculate_spine_obj_mesh_skeleton_coordinates(
            spine_obj=spine_obj,
            mesh = mesh,
        )
        spine_obj.calculate_head_neck(**kwargs)
        if verbose_time:
            print(f"Time for calculate_head_neck = {time.time() - st}")
            st = time.time()
        spine_obj.head_mesh_splits
        if verbose_time:
            print(f"Time for head_mesh_splits = {time.time() - st}")
            st = time.time()
            
    
    calculate_soma_distance_euclidean(spine_obj,soma_center)
    calculate_soma_distance_skeletal(spine_obj,upstream_skeletal_length)
    if branch_obj is not None:
        calculate_branch_overall_features(spine_obj,branch_obj,branch_features=branch_features,)
    
    return spine_obj

def mesh_minus_spine_objs(
    spine_objs,
    mesh = None,
    branch_obj=None,
    return_idx = False,):
    """
    Purpose: To get the shaft mesh of a branch
    given a list of spines
    """
    if mesh is None:
        mesh = branch_obj.mesh
    idx = np.delete(
        np.arange(len(mesh.faces)),np.hstack([k.mesh_face_idx for k in spine_objs]
                                      ).astype('int')
    )
    
    if return_idx:
        return idx
    else:
        return mesh.submesh([idx],append=True)

def calculate_spine_attributes_for_list(
    spine_objs,
    branch_obj = None,
    calculate_coordinates = True,
    calculate_head_neck=False,
    verbose_time=False,
    mesh = None,
    **kwargs
    ):
    if calculate_coordinates and len(spine_objs) > 0:
        branch_shaft_mesh_face_idx = mesh_minus_spine_objs(
            spine_objs,
            branch_obj=branch_obj,
            mesh = mesh,
            return_idx = True,
        )
    else:
        branch_shaft_mesh_face_idx = None
    return [spu.calculate_spine_attributes(
        k,
        branch_obj = branch_obj,
        calculate_coordinates=calculate_coordinates,
        calculate_head_neck=calculate_head_neck,
        verbose_time=verbose_time,
        branch_shaft_mesh_face_idx=branch_shaft_mesh_face_idx,
        mesh=mesh,
        **kwargs
    ) for k in spine_objs]
    
        
def is_spine_obj(obj):
    #return str(type(obj)) == str(spu.Spine)
    return obj.__class__ == spu.Spine
        
def calculate_face_idx(spine_obj,
                      original_mesh=None,
                    original_mesh_kdtree=None,
                      **kwargs):
    """
    Purpose: To calculate the original faces 
    of the spine to a reference mesh
    """
    if original_mesh is None and original_mesh_kdtree is None:
        raise Exception("Either original_mesh or original_mesh_kdtree needs to be non None")
    
    return tu.original_mesh_faces_map(original_mesh = original_mesh,
                                     original_mesh_kdtree=original_mesh_kdtree,
                                     submesh = spine_obj.mesh,
                                      exact_match=True,
                                     **kwargs)

def plot_head_neck(spine_obj,
                   neck_color = neck_color_default,
                    head_color = head_color_default,
                   no_head_color = no_head_color_default,
                  verbose = True):
    neck_mesh =spine_obj.neck_mesh
    head_mesh = spine_obj.head_mesh
    no_head_mesh = spine_obj.no_head_mesh
    
    if verbose:
        print(f"head_mesh ({head_color}): {head_mesh}")
        print(f"neck_mesh ({neck_color}): {neck_mesh}")
        print(f"no_head_mesh ({no_head_color}): {no_head_mesh}")

    nviz.plot_objects(meshes = [neck_mesh,head_mesh,no_head_mesh],
                     meshes_colors=[neck_color,head_color,no_head_color])
    
    
def mesh_from_name_or_idx(
    spine_obj,
    name=None,
    idx=None,
    largest_component = False):
    if idx is None:
        if name is not None:
            if "face_idx" not in name:
                name = f"{name}_face_idx"
        idx = getattr(spine_obj,name,None)
    
    if idx is None:
        return None
    elif len(idx) == 0:
        return tu.empty_mesh()
    else:
        mesh =  spine_obj.mesh.submesh([idx],append=True)
        if largest_component:
            mesh = tu.largest_conn_comp(mesh)
            
        return mesh
    
def head_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="head")
def neck_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="neck",largest_component=False)
def no_head_mesh(spine_obj):
    return mesh_from_name_or_idx(spine_obj,name="no_head")
def head_mesh_splits_face_idx_by_index(spine_obj,index):
    if spine_obj.head_mesh is None:
        return None
    return np.array(spine_obj.head_face_idx)[np.where(spine_obj.head_mesh_splits_face_idx == index)[0]]
def head_mesh_splits_from_index(spine_obj,index):
    return mesh_from_name_or_idx(spine_obj,idx = head_mesh_splits_face_idx_by_index(spine_obj,index))
"""
def head_mesh(spine_obj):
    if spine_obj.head_face_idx is None or len(spine_obj.head_face_idx) == 0:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.head_face_idx],append=True)
    
def neck_mesh(spine_obj):
    if spine_obj.neck_face_idx is None:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.neck_face_idx],append=True)
    
def no_head_mesh(spine_obj):
    if spine_obj.no_head_face_idx is None or len(spine_obj.no_head_face_idx) == 0:
        return tu.empty_mesh()
    else:
        return spine_obj.mesh.submesh([spine_obj.no_head_face_idx],append=True)
"""
    
def head_exist(spine_obj):
    if spine_obj.head_face_idx is None or len(spine_obj.head_face_idx) == 0:
        return False
    return True



def export(
    spine_obj,
    attributes_to_skip = None,
    attributes_to_add = None,
    suppress_errors = True,
    attributes = None,
    default_value = None):
    
    if attributes is None:
        curr_attributes = spine_attributes
        if attributes_to_skip is not None:
            attributes_to_skip= nu.to_list(attributes_to_skip)
            curr_attributes = np.setdiff1d(curr_attributes,attributes_to_skip)

        if attributes_to_add is not None:
            attributes_to_add=nu.to_list(attributes_to_add)
            curr_attributes = np.union1d(curr_attributes,attributes_to_add)
    else:
        curr_attributes = attributes

    return_dict = {}
    for k in curr_attributes:
        try:
            curr_value = getattr(spine_obj,k,default_value)
        except:
            if suppress_errors:
                curr_value = default_value
            else:
                raise Exception("")
                
        return_dict[k] = curr_value
    return return_dict
        
    #return {k:getattr(spine_obj,k,None) for k in curr_attributes}
    




def set_branch_spines_obj(branch_obj,
                               calculate_mesh_face_idx = True,
                                verbose = False):
    """
    Purpose: To set the spine 0bj
    attribute for a branch

    Pseudocode: 
    1) store the spine mesh and the volume
    2) calculate the neck face idx and sdf
    2) optional: calculate the mesh face_idx
    """
    curr_spines = branch_obj.spines
    curr_spines_vol = branch_obj.spines_volume
    
    if curr_spines is None:
        branch_obj.spines_obj = None
        return 

    spines_obj = []
    branch_kd = tu.mesh_kdtree_face(branch_obj.mesh)

    if not verbose:
        tqu.turn_off_tqdm()
    else:
        tqu.turn_on_tqdm()

    for s,vol in tqu.tqdm(zip(curr_spines,curr_spines_vol)):
        sp_obj = spu.Spine(mesh=s,volume=vol)
        sp_obj.calculate_head_neck()

        if calculate_mesh_face_idx:
            sp_obj.calculate_face_idx(original_mesh=branch_obj.mesh,
                                      original_mesh_kdtree=branch_kd)
        spines_obj.append(sp_obj)

    branch_obj.spines_obj = spines_obj
    
def set_neuron_spine_attribute(neuron_obj,func,
                              verbose = False):
    """
    Purpose: To set the spines obj
    for all branches in the neuron obj
    """
    for limb_name in neuron_obj.get_limb_names():
        if verbose:
            print(f"Working on Limb {limb_name}")
        limb_obj = neuron_obj[limb_name]
        for b_idx in limb_obj.get_branch_names():
            if verbose:
                print(f"--- branch {b_idx}")
            func(branch_obj = limb_obj[b_idx])
    
def set_neuron_spines_obj(neuron_obj,
                         verbose = False):
    spu.set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_spines_obj,
                              verbose=verbose)

            
def set_branch_head_neck_shaft_idx(branch_obj,
                                  plot_face_idx=False,
                                    add_no_head_label = True,
                                  verbose = False):
    branch_obj.head_neck_shaft_idx = spu.head_neck_shaft_idx_from_branch(
                    branch_obj,
                    plot_face_idx  = plot_face_idx,
                    add_no_head_label = add_no_head_label,
                    verbose = verbose)
    
def set_neuron_head_neck_shaft_idx(neuron_obj,
                                   add_no_head_label= True,
                         verbose = False):
    set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_head_neck_shaft_idx,
                              verbose=verbose)
    
def set_branch_synapses_head_neck_shaft(branch_obj,
                                        verbose = False
                                       ):
    """
    Purpose: To use the head_neck_shaft_idx
    of the branch objects to give the 
    synapses of a branch the head_neck_shaft label

    Pseudocode: 
    If the branch has any 
    1) Build a KDTree of the branch mesh
    2) find which faces are the closest for the coordinates of all the synapses
    3) Assign the closest face and the head_neck_shaft to the synapse objects
    """
    spines_obj = branch_obj.spines_obj
    if branch_obj.synapses is not None and len(branch_obj.synapses)>0:
        mesh_kd = tu.mesh_kdtree_face(branch_obj.mesh)

        synapse_coords = np.array(syu.synapses_to_coordinates(branch_obj.synapses)).reshape(-1,3)

        dist,closest_face = mesh_kd.query(synapse_coords)
 
        for j,s in enumerate(branch_obj.synapses):
            s.closest_branch_face_idx = closest_face[j]
            s.head_neck_shaft = branch_obj.head_neck_shaft_idx[s.closest_branch_face_idx]
            
def set_neuron_synapses_head_neck_shaft(neuron_obj,
                                       verbose=False):
    set_neuron_spine_attribute(neuron_obj,
                               func = spu.set_branch_synapses_head_neck_shaft,
                              verbose=verbose)

def add_head_neck_shaft_spine_objs(neuron_obj,
                                   add_synapse_labels=True,
                                   filter_spines_for_size = True,
                                   add_distance_attributes = True,
                                  verbose = False):
    """
    Will do the additionaly processing that
    adds the spine objects to a neuron and then
    creates the head_neck_shaft_idx for the branches 
    
    Application: Can be used later to map synapses
    to the accurate label
    """
    st = time.time()
    
    if filter_spines_for_size:
        
        neuron_obj= spu.filter_spines_by_size(neuron_obj,
                                verbose = verbose)
        if verbose:
            print(f"Total time for fitlering spines by size {time.time() - st}")
            st = time.time()
    
    spu.set_neuron_spines_obj(neuron_obj)
    if verbose:
        print(f"Total time for set_neuron_spines_obj {time.time() - st}")
        st = time.time()
    
    spu.set_neuron_head_neck_shaft_idx(neuron_obj)
    if verbose:
        print(f"Total time for set_neuron_head_neck_shaft_idx {time.time() - st}")
        st = time.time()
    if add_synapse_labels:
        spu.set_neuron_synapses_head_neck_shaft(neuron_obj)
        if verbose:
            print(f"Total time for set_neuron_synapses_head_neck_shaft {time.time() - st}")
            st = time.time()
            
    if add_distance_attributes:
        neuron_obj= spu.calculate_spine_obj_attr_for_neuron(
                neuron_obj,
                verbose = verbose)
        if verbose:
            print(f"Total time for calculate_spine_obj_attr_for_neuron {time.time() - st}")
            st = time.time()
    
    return neuron_obj
            
    
def spines_head_meshes(obj):
    return [k.head_mesh for k in obj.spines_obj]
def spines_neck_meshes(obj):
    return [k.neck_mesh for k in obj.spines_obj]
def spines_no_head_meshes(obj):
    return [k.no_head_mesh for k in obj.spines_obj]

def spines(neuron_obj):
    if type(neuron_obj) != list:
        curr_spines = neuron_obj.spines_obj
    else:
        curr_spines = neuron_obj
    return curr_spines

def spines_head(neuron_obj):
    curr_spines = spu.spines(neuron_obj)
    return [k for k in curr_spines if k.head_exist]
def spines_no_head(neuron_obj):
    curr_spines = spu.spines(neuron_obj)
    return [k for k in curr_spines if not k.head_exist]
def spines_neck(neuron_obj):
    return spu.spines_head(neuron_obj)

def n_spines(neuron_obj):
    return len(spu.spines(neuron_obj))
def n_spines_head(neuron_obj):
    return len(spu.spines_head(neuron_obj))
def n_spines_no_head(neuron_obj):
    return len(spu.spines_no_head(neuron_obj))
def n_spines_neck(neuron_obj):
    return len(spu.spines_neck(neuron_obj))


def plot_spines_head_neck(neuron_obj,
                         head_color = head_color_default,
                         neck_color = neck_color_default,
                          no_head_color = no_head_color_default,
                          bouton_color = bouton_color_default,
                          mesh_alpha = 0.5,
                          verbose=False,
                          show_at_end=True,
                          combine_meshes = True,
                         ):
    
    nviz.visualize_neuron_lite(neuron_obj,
                              show_at_end=False)
    meshes_colors = []
    
    spine_heads = spu.spines_head_meshes(neuron_obj)
    meshes_colors += [head_color]*len(spine_heads)
    
    spine_necks = spu.spines_neck_meshes(neuron_obj)
    meshes_colors += [neck_color]*len(spine_necks)
    
    spine_no_heads = spu.spines_no_head_meshes(neuron_obj)
    meshes_colors += [no_head_color]*len(spine_no_heads)
    
    boutons = neuron_obj.boutons
    meshes_colors += [bouton_color]*len(boutons)
    
    if verbose:
        print(f"# of spine_heads = {len(spine_heads)}, # of spine_necks= {len(spine_necks)-len(spine_no_heads)}, # of spine_no_heads = {len(spine_no_heads)}")
    
    if combine_meshes:
        meshes_colors = [head_color,neck_color,no_head_color,bouton_color]
        spine_heads = [tu.combine_meshes(spine_heads)]
        spine_necks = [tu.combine_meshes(spine_necks)]
        spine_no_heads = [tu.combine_meshes(spine_no_heads)]
        boutons = [tu.combine_meshes(boutons)]
        meshes = spine_heads+spine_necks+spine_no_heads+boutons
        
    
    nviz.plot_objects(meshes = meshes,
                     meshes_colors=meshes_colors,
                     append_figure=True,
                     mesh_alpha=mesh_alpha,
                     show_at_end=show_at_end,)
    
# -------------- End of Spines Obj -------------- #
    
    
connectivity = "edges"

"""   DON'T NEED THIS FUNCTION ANYMORE BECAUSE REPLACED BY TRIMESH_UTILS MESH_SEGMENTATION
def cgal_segmentation(written_file_location,
                      clusters=2,
                      smoothness=0.03,
                      return_sdf=True,
                     print_flag=False,
                     delete_temp_file=True):
    
    if written_file_location[-4:] == ".off":
        cgal_mesh_file = written_file_location[:-4]
    else:
        cgal_mesh_file = written_file_location
    if print_flag:
        print(f"Going to run cgal segmentation with:"
             f"\nFile: {cgal_mesh_file} \nclusters:{clusters} \nsmoothness:{smoothness}")

    csm.cgal_segmentation(cgal_mesh_file,clusters,smoothness)

    #read in the csv file
    cgal_output_file = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
    cgal_output_file_sdf = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

    cgal_data = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
    cgal_sdf_data = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')
    
    if delete_temp_file:
        cgal_output_file.unlink()
        cgal_output_file_sdf.unlink()
        
    
    if return_sdf:
        return cgal_data,cgal_sdf_data
    else:
        return cgal_data"""

def split_mesh_into_spines_shaft_old(current_mesh,
                           segment_name="",
                           clusters=None,
                          smoothness=None,
                          cgal_folder = Path("./cgal_temp"),
                          delete_temp_file=True,
                          shaft_threshold = None,
                                 return_sdf = True,
                                print_flag = True,
                                plot_segmentation = False,**kwargs):
    
    """
    if not cgal_folder.exists():
        cgal_folder.mkdir(parents=True,exist_ok=False)

    file_to_write = cgal_folder / Path(f"segment_{segment_name}.off")
    
    
    
    # ------- 1/14 Additon: Going to make sure mesh has no degenerate faces --- #
    if filter_away_degenerate_faces:
        mesh_to_segment,faces_kept = tu.connected_nondegenerate_mesh(current_mesh,
                                                                     return_kept_faces_idx=True,
                                                                     return_removed_faces_idx=False)


        written_file_location = tu.write_neuron_off(mesh_to_segment,file_to_write)
    else:
        written_file_location = tu.write_neuron_off(current_mesh,file_to_write)
    
    cgal_data_pre_filt,cgal_sdf_data_pre_filt = cgal_segmentation(written_file_location,
                                             clusters,
                                             smoothness,
                                             return_sdf=True,
                                               delete_temp_file=delete_temp_file)
    
    if filter_away_degenerate_faces:
        cgal_data = np.ones(len(current_mesh.faces))*(np.max(cgal_data_pre_filt)+1)
        cgal_data[faces_kept] = cgal_data_pre_filt

        cgal_sdf_data = np.zeros(len(current_mesh.faces))
        cgal_sdf_data[faces_kept] = cgal_sdf_data_pre_filt
    else:
        cgal_data = cgal_data_pre_filt
        cgal_sdf_data = cgal_sdf_data_pre_filt
        
    #print(f"file_to_write = {file_to_write.absolute()}")
    if delete_temp_file:
        #print("attempting to delete file")
        file_to_write.unlink()
    """
    
    if clusters is None:
        clusters = clusters_threshold_global
    
    if smoothness is None:
        smoothness = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
    
    

    #print(f"plot_segmentation= {plot_segmentation}")
    cgal_data,cgal_sdf_data = tu.mesh_segmentation(current_mesh,
                                                  cgal_folder=cgal_folder,
                                                   clusters=clusters,
                                                   smoothness=smoothness,
                                                   return_sdf=True,
                                                   delete_temp_files=delete_temp_file,
                                                   return_meshes=False,
                                                   return_ordered_by_size=False,
                                                   plot_segmentation = plot_segmentation,
                                                  )
    
    """ 1/14: Need to adjust for the degenerate faces removed
    """

    
    #get a look at how many groups and what distribution:
    from collections import Counter
    if print_flag:
        print(f"Counter of data = {Counter(cgal_data)}")

    #gets the meshes that are split using the cgal labels
    split_meshes,split_meshes_idx = tu.split_mesh_into_face_groups(current_mesh,cgal_data,return_idx=True,
                                   check_connect_comp = False)
    
    
    
    split_meshes,split_meshes_idx
    
    
    if len(split_meshes.keys()) <= 1:
        print("There was only one mesh found from the spine process and mesh split, returning empty array")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
        
    
#     # How to identify just one shaft
#     shaft_index = -1
#     shaft_total = -1
#     for k,v in split_meshes.items():
#         curr_length = len(v.faces)
#         if  curr_length > shaft_total:
#             shaft_index = k
#             shaft_total = curr_length
    
#     shaft_mesh = split_meshes.pop(shaft_index)
#     shaft_mesh_idx = split_meshes_idx.pop(shaft_index)
    
#     print(f"shaft_index = {shaft_index}")
    
    shaft_meshes = []
    shaft_meshes_idx = []
    
    spine_meshes = []
    spine_meshes_idx = []
    
    #Applying a length threshold to get all other possible shaft meshes
    for spine_id,spine_mesh in split_meshes.items():
        if len(spine_mesh.faces) < shaft_threshold:
            spine_meshes.append(spine_mesh)
            spine_meshes_idx.append(split_meshes_idx[spine_id])
        else:
            shaft_meshes.append(spine_mesh)
            shaft_meshes_idx.append(split_meshes_idx[spine_id])
 
    if len(shaft_meshes) == 0:
        if print_flag:
            print("No shaft meshes detected")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
 
    if len(spine_meshes) == 0:
        if print_flag:
            print("No spine meshes detected")
            

    if return_sdf:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx,cgal_sdf_data
    else:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx
    
    
def get_spine_meshes_unfiltered_from_mesh(
    current_mesh,
    segment_name=None,
    clusters=None,
    smoothness=None,
    shaft_expansion_method = "path_to_all_shaft_mesh",
    cgal_folder = Path("./cgal_temp"),
    delete_temp_file=True,
    return_sdf=False,
    return_mesh_idx = False,
    print_flag=False,
    shaft_threshold=None,
    ensure_mesh_conn_comp = False,
    plot_segmentation = False,
    plot_shaft = False,
    plot = False,
    ):
    
    if clusters is None:
        clusters = clusters_threshold_global
    
    if smoothness is None:
        smoothness = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
    
    
    if segment_name is None:
        segment_name = f"{np.random.randint(10,1000)}_{np.random.randint(10,1000)}"
    
    #print(f"segment_name before cgal = {segment_name}")
    (spine_meshes,
     spine_meshes_idx,
     shaft_meshes,
     shaft_meshes_idx,
    cgal_sdf_data) = spine_data_returned= split_mesh_into_spines_shaft(current_mesh,
                               segment_name=segment_name,
                               clusters=clusters,
                              smoothness=smoothness,
                              cgal_folder = cgal_folder,
                              delete_temp_file=delete_temp_file,
                              return_sdf = True,
                              print_flag=print_flag,
                              shaft_threshold = shaft_threshold,
                              plot_segmentation = plot_segmentation,
                            plot_shaft=plot_shaft,
                                                                      )
    if len(spine_meshes) == 0:
        
        if return_sdf and return_mesh_idx:
            return  [],[],[]
        elif return_sdf or return_mesh_idx:
            return [],[]
        else:
            return []
    else:
        spine_mesh_names = [f"s{i}" for i,mesh in enumerate(spine_meshes)]
        shaft_mesh_names = [f"b{i}" for i,mesh in enumerate(shaft_meshes)]

        total_meshes = spine_meshes + shaft_meshes
        total_meshes_idx = spine_meshes_idx + shaft_meshes_idx
        total_names = spine_mesh_names + shaft_mesh_names

        total_edges = []
        for j,(curr_mesh,curr_mesh_idx) in enumerate(zip(total_meshes,total_meshes_idx)):
            touching_meshes = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_mesh_idx,
                            periphery_pieces=total_meshes_idx,
                            connectivity = "vertices")
            try:
                touching_meshes.remove(j)
            except:
                print(f"j = {j}")
                su.compressed_pickle(current_mesh,"current_mesh")
                su.compressed_pickle(curr_mesh_idx,"curr_mesh_idx")
                su.compressed_pickle(total_meshes_idx,"total_meshes_idx")
                su.compressed_pickle(total_names,"total_names")
                raise Exception("didn't do remove")
                
            #construct the edges
            curr_edges = [[total_names[j],total_names[h]] for h in touching_meshes]
            total_edges += curr_edges

        spine_graph = xu.remove_selfloops(nx.from_edgelist(total_edges))
        #nx.draw(spine_graph,with_labels=True)


        """
        How to determine whih parts that are the shaft
        1) start with biggest shaft
        2) Find the shoftest paths to all shaft parts
        3) add all the nodes that aren't already in the shaft category to the shaft category
        """

        #find the biggest shaft
        
        biggest_shaft = f"b{np.argmax([len(k.faces) for k in shaft_meshes])}"
        non_biggest_shaft = [k for k in shaft_mesh_names if k != biggest_shaft]

        final_shaft_mesh_names = shaft_mesh_names.copy()
        if len(non_biggest_shaft) > 0:
            #find all shortest paths from biggest shaft to non_biggest_shaft
            
            #shaft_shortest_paths = [nx.shortest_path(spine_graph,
            #source=biggest_shaft,target=curr_shaft) for curr_shaft in non_biggest_shaft]
            
            shaft_shortest_paths = []
            if shaft_expansion_method == "path_to_largest_shaft_mesh":
                for curr_shaft in non_biggest_shaft:
                    try:
                        c_path = nx.shortest_path(spine_graph,
                                                         source=biggest_shaft,target=curr_shaft) 
                    except:
                        #print(f"Mesh {curr_shaft} Seems to be not connected to mesh")
                        shaft_shortest_paths.append([curr_shaft])
                    else:
                        shaft_shortest_paths.append(c_path)
            elif shaft_expansion_method == "path_to_all_shaft_mesh":
                for shaft_1 in shaft_mesh_names:
                    for shaft_2 in shaft_mesh_names:
                        if shaft_1 != shaft_2:
                            try:
                                c_path = nx.shortest_path(spine_graph,
                                        source=shaft_1,target=shaft_2) 
                            except:
                                #print(f"{shaft_1} and {shaft_2} not connected on the same mesh")
                                shaft_shortest_paths.append([shaft_1,shaft_2])
                            else:
                                shaft_shortest_paths.append(c_path)
            else:
                raise Exception("")
                            
            

            new_shaft_meshes = [int(k[1:]) for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            #print(f"new_shaft_meshes = {new_shaft_meshes}")
            final_shaft_mesh_names += [k for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            final_shaft_meshes = shaft_meshes + [spine_meshes[k] for k in new_shaft_meshes]
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx + [spine_meshes_idx[k] for k in new_shaft_meshes]))
        else:
            final_shaft_meshes = shaft_meshes
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx))

        final_shaft_mesh_names = np.unique(final_shaft_mesh_names)

        final_spine_faces_idx = np.delete(np.arange(0,len(current_mesh.faces)), np.array(final_shaft_meshes_idx).astype('int'))

        """
        #Old way of getting all of the spines: by just dividing the mesh using disconnected components
        #after subtracting the shaft mesh

        spine_submesh = current_mesh.submesh([final_spine_faces_idx],append=True)
        spine_submesh_split = spine_submesh.split(only_watertight=False)

        """

        """
        #New way of extracting the spines using graphical methods

        Pseudocode:
        1) remove the shaft meshes from the graph
        2) get the connected components
        3) assemble the connected components total face_idx:
        a. get the sdf values that correspond to those
        b. get the submesh that corresponds to those

        """
        spine_graph.remove_nodes_from(final_shaft_mesh_names)

        spine_submesh_split=[]
        spine_submesh_split_idx = []
        spine_submesh_split_sdf = []
        for sp_list in list(nx.connected_components(spine_graph)):
            curr_spine_face_idx_split = np.concatenate([spine_meshes_idx[int(sp[1:])] for sp in sp_list ])
            spine_submesh_split_sdf.append(cgal_sdf_data[curr_spine_face_idx_split])
            spine_submesh_split_idx.append(curr_spine_face_idx_split)
            spine_submesh_split.append(current_mesh.submesh([curr_spine_face_idx_split],append=True))


        if print_flag or plot:
            print(f"\n\nTotal Number of Spines Found = {len(spine_submesh_split)}")


        #sort the list by size
        spine_length_orders = [len(k.faces) for k in spine_submesh_split]
        greatest_to_least = np.flip(np.argsort(spine_length_orders))
        spines_greatest_to_least =  np.array(spine_submesh_split)[greatest_to_least]
        spines_sdf_greatest_to_least = np.array(spine_submesh_split_sdf)[greatest_to_least]
        spine_submesh_split_idx = np.array(spine_submesh_split_idx)[greatest_to_least]
        
        if plot:
            print(f"--- plotting shaft meshes after graph fix -- ")
            if len(spines_greatest_to_least) == 0:
                meshes = None
            else:
                meshes = spines_greatest_to_least
            
            nviz.plot_objects(
                current_mesh,
                meshes = list(spines_greatest_to_least),
                meshes_colors = "red",
                buffer = 0,
            )
        
        if ensure_mesh_conn_comp:
            spines_greatest_to_least_new = []
            spine_submesh_split_idx_new = []
            sdf_new = []
            for k,k_idx,k_sdf in zip(
                spines_greatest_to_least,
                spine_submesh_split_idx,
                spines_sdf_greatest_to_least):
                sp_mesh,sp_idx = tu.largest_conn_comp(k,return_face_indices=True)
                spines_greatest_to_least_new.append(sp_mesh)
                spine_submesh_split_idx_new.append(k_idx[sp_idx])
                sdf_new.append(k_sdf[sp_idx])
                
            spines_greatest_to_least = np.array(spines_greatest_to_least_new)
            spine_submesh_split_idx = np.array(spine_submesh_split_idx_new)
            spines_sdf_greatest_to_least = np.array(sdf_new)
            
                
            
        if not return_sdf and not return_mesh_idx:
            return spines_greatest_to_least
        return_value = [spines_greatest_to_least]
        if return_sdf:
            return_value.append(spines_sdf_greatest_to_least)
        if return_mesh_idx:
            return_value.append(spine_submesh_split_idx)
            
        return return_value


def get_spine_meshes_unfiltered(current_neuron,
                 limb_idx,
                branch_idx,
                clusters=3,#2,
                smoothness=0.1,#0.05,
                cgal_folder = Path("./cgal_temp"),
                delete_temp_file=True,
                return_sdf=False,
                print_flag=False,
                shaft_threshold=300,
                               mesh=None):
    
    
    current_mesh = current_neuron.concept_network.nodes[nru.limb_label(limb_idx)]["data"].concept_network.nodes[branch_idx]["data"].mesh
    
    return get_spine_meshes_unfiltered_from_mesh(current_mesh,
                                        segment_name=f"{limb_idx}_{branch_idx}",
                                        clusters=clusters,
                                        smoothness=smoothness,
                                        cgal_folder = cgal_folder,
                                        delete_temp_file=delete_temp_file,
                                        return_sdf=return_sdf,
                                        print_flag=print_flag,
                                        shaft_threshold=shaft_threshold)
    
        
        
"""
These filters didn't seem to work very well...

"""
def sdf_median_mean_difference(sdf_values):
    return np.abs(np.median(sdf_values) - np.mean(sdf_values)) 

def apply_sdf_filter(sdf_values,sdf_median_mean_difference_threshold = 0.025,
                    return_not_passed=False):
    pass_filter = []
    not_pass_filter = []
    for j,curr_sdf in enumerate(sdf_values):
        if sdf_median_mean_difference(curr_sdf)< sdf_median_mean_difference_threshold:
            pass_filter.append(j)
        else:
            not_pass_filter.append(j)
    if return_not_passed:
        return not_pass_filter
    else:
        return pass_filter

def surface_area_to_volume(current_mesh):
    """
    Method to try and differentiate false from true spines
    conclusion: didn't work
    
    Even when dividing by the number of faces
    """
    return current_mesh.bounding_box_oriented.volume/current_mesh.area


def filter_spine_meshes(spine_meshes,
                        spine_n_face_threshold=None,
                       spine_sk_length_threshold=None,
                       verbose = False):
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_global
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    keep_idx = np.arange(len(spine_meshes))
    if spine_n_face_threshold is not None:
        spine_n_faces = np.array([len(k.faces) for k in spine_meshes])
        keep_idx = np.intersect1d(keep_idx,np.where(spine_n_faces >= spine_n_face_threshold)[0])
        if verbose:
            print(f"After face threshold {spine_n_face_threshold}, keep_idx = {keep_idx}")
    if spine_sk_length_threshold is not None:
        spine_lens = np.array([spu.spine_length(k) for k in spine_meshes])
        keep_idx = np.intersect1d(keep_idx,np.where(spine_lens >= spine_sk_length_threshold)[0])
        if verbose:
            print(f"After sk length threshold {spine_sk_length_threshold}, keep_idx = {keep_idx}")
    return [k for i,k in enumerate(spine_meshes) if i in keep_idx]


#------------ 9/23 Addition -------------- #
def filter_out_border_spines(mesh,spine_submeshes,
                            border_percentage_threshold=None,                                                    
                             check_spine_border_perc=None,
                            verbose=False,
                            return_idx = False):
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if check_spine_border_perc is None:
        check_spine_border_perc = check_spine_border_perc_global
    return tu.filter_away_border_touching_submeshes_by_group(mesh,spine_submeshes,
                                                             border_percentage_threshold=border_percentage_threshold,
                                                             inverse_border_percentage_threshold=check_spine_border_perc,
                                                             verbose = verbose,
                                                             return_meshes = not return_idx,
                                                            )

def filter_out_soma_touching_spines(
    spine_submeshes,
    soma_vertices=None,
    soma_kdtree=None,
    verbose=False,
    return_idx = False,
    ):
    """
    Purpose: To filter the spines that are touching the somae
    Because those are generally false positives picked up 
    by cgal segmentation
    
    Pseudocode
    1) Create a KDTree from the soma vertices
    2) For each spine:
    a) Do a query against the KDTree with vertices
    b) If any of the vertices have - distance then nullify


    """
    if soma_kdtree is None and not soma_vertices is None:
        soma_kdtree = KDTree(soma_vertices)
    if soma_kdtree is None and soma_verties is None:
        raise Exception("Neither a soma kdtree or soma vertices were given")

    if verbose:
        print(f"Number of spines before soma border filtering = {len(spine_submeshes)}")
    final_spines = []
    final_spines_idx = []
    for j,sp_mesh in enumerate(spine_submeshes):
        sp_dist,sp_closest = soma_kdtree.query(sp_mesh.vertices)
        n_match_vertices = np.sum(sp_dist==0)
        
        if n_match_vertices == 0:
            final_spines.append(sp_mesh)
            final_spines_idx.append(j)
        else:
            if verbose:
                print(f"Spine {j} was removed because had {n_match_vertices} border vertices")
    if verbose:
        print(f"Number of spines before soma border filtering = {len(final_spines)}")
    
    if return_idx:
        return final_spines_idx
    else:
        return final_spines


def spine_head_neck(
    mesh,
    cluster_options = (2,3,4),
    smoothness = None,#0.15,
    plot_segmentation = False,
    head_ray_trace_min = None,#240,
    head_face_min = None,#10,
    default_head_face_idx = np.array([]).astype("int"),
    default_head_sdf = -1,
    
    #for filtering away head meshes that have the following coordinate
    stop_segmentation_after_first_success = False,
    no_head_coordinates = None,
    only_allow_one_connected_component_neck = None,
    
    plot_head_neck = False,
    return_meshes = False,
    return_sdf = True,
    return_width=True,
    verbose = False,
    ):
    """
    Purpose: To determine the head and neck
    face idx of a mesh representing a spine

    Pseudocode: 
    for clusters [2,3]:
    1) Run the segemntation algorithm
    2) Filter meshes for all those with a face count above threshold
    and a sdf count above threshold
    3) Store the meshes not face as neck and store sdf value as weighted average
    4a) If none then continue
    4b) If at least one, concatenate the faces of all of the spine heads
    into one array (and do weighted average of the sdf)
    5) Break if found a head

    6) Optionally plot the spine neck and head


    Can optionally return:
    1) Meshes instead of face idx
    
    Ex: 
    curr_idx = 38
    sp_mesh = curr_branch.spines[curr_idx]
    spu.spine_head_neck(sp_mesh,
                        cluster_options=(2,3,4),
                        smoothness=0.15,
                        verbose = True,
                        plot_segmentation=True,
                       plot_head_neck=True)

    """
    if head_ray_trace_min is None:
        head_ray_trace_min = head_ray_trace_min_global
        
    if smoothness is None:
        smoothness = head_smoothness_global
        
    if head_face_min is None:
        head_face_min = head_face_min_global
        
    if only_allow_one_connected_component_neck is None:
        only_allow_one_connected_component_neck = only_allow_one_connected_component_neck_global
    

    neck_face_idx = None
    head_face_idx = default_head_face_idx

    neck_sdf = None
    head_sdf = default_head_sdf
    head_width = default_head_sdf

    winning_n_heads = 0
    neck_face_idx_win = neck_sdf_win = neck_width_win = None
    head_face_idx_win = np.array([])
    head_sdf_win = default_head_sdf
    head_width_win = default_head_sdf
    for c in cluster_options:
        if verbose:
            print(f"Using clusters {c}")

        meshes,sdfs,mesh_idx = tu.mesh_segmentation(mesh,
                                                    clusters = c,
                                                    smoothness = smoothness,
                                           return_meshes = True,
                                           return_sdf=True,
                                            plot_segmentation=plot_segmentation,
                                          return_ordered_by_size=False,
                                          return_mesh_idx = True)
        ray_trace_perc = np.array([tu.mesh_size(k,"ray_trace_percentile") for k in meshes])
        mesh_sizes = np.array([len(k.faces) for k in meshes])

        if verbose:
            print(f"sdfs={sdfs}, ray_trace_perc = {ray_trace_perc}, mesh_sizes = {mesh_sizes}")
            print(f"Thresholds: head_ray_trace_min = {head_ray_trace_min}, head_face_min = {head_face_min}")

        head_obj_idx = np.where((ray_trace_perc > head_ray_trace_min) & (mesh_sizes > head_face_min))[0]
        
        if verbose:
            print(f"head_obj_idx = {head_obj_idx}")
        
        # Want to delete any meshes that have the border coordinates
        if no_head_coordinates is not None:
            head_obj_idx_final = []
            for hm in head_obj_idx:
                curr_mesh = meshes[hm]
                closest_dist = tu.closest_mesh_distance_to_coordinates_fast(
                    curr_mesh,
                    coordinates =no_head_coordinates,
                    attribute = "vertices",
                    stop_after_0_dist = True,
                )
                
                if closest_dist > 0:
                    head_obj_idx_final.append(hm)
                    
            head_obj_idx = head_obj_idx_final
            if verbose:
                print(f"After filtering for no_head_coordinates: head_obj_idx = {head_obj_idx}")
        
        neck_obj_idx = np.delete(np.arange(len(meshes)),np.array(head_obj_idx).astype('int'))

        #makes sure there always has to be a neck idx
        if len(neck_obj_idx) == 0:
            neck_obj_idx = np.arange(len(meshes))
            head_obj_idx = np.array([]).astype('int')
        elif only_allow_one_connected_component_neck:
            n_neck_components = len(tu.connected_components_from_face_idx(
                mesh,face_idx = np.concatenate(mesh_idx[neck_obj_idx]),return_meshes = False))
            if verbose:
                print(f"n_neck_components = {n_neck_components}")
            if  n_neck_components> 1:
                if verbose:
                    print(f"More than one neck connected component so not valid")
                neck_obj_idx = np.arange(len(meshes))
                head_obj_idx = np.array([]).astype('int')
        else:
            pass

        if verbose:
            print(f"head_obj_idx = {head_obj_idx}")
            print(f"neck_obj_idx= {neck_obj_idx}")

        # save off the neck face idx
        neck_face_idx = np.concatenate(mesh_idx[neck_obj_idx])
        neck_sdf = nu.weighted_average(sdfs[neck_obj_idx],mesh_sizes[neck_obj_idx])
        neck_width = nu.weighted_average(ray_trace_perc[neck_obj_idx],mesh_sizes[neck_obj_idx])

        if verbose:
            print(f"neck_face_idx= {neck_face_idx}")
            print(f"neck_sdf = {neck_sdf}")
            print(f"neck_width = {neck_width}")

        if len(head_obj_idx) == 0:
            if neck_face_idx_win is None:
                neck_face_idx_win = neck_face_idx
                neck_sdf_win = neck_sdf
                neck_width_win = neck_width
            continue

        head_face_idx = np.concatenate(mesh_idx[head_obj_idx])
        head_sdf = nu.weighted_average(sdfs[head_obj_idx],mesh_sizes[head_obj_idx])
        head_width = nu.weighted_average(ray_trace_perc[head_obj_idx],mesh_sizes[head_obj_idx])

        if verbose:
            print(f"head_face_idx= {head_face_idx}")
            print(f"head_sdf = {head_sdf}")
            print(f"head_width = {head_width}")

        n_heads = len(tu.connected_components_from_face_idx(
            mesh,face_idx = head_face_idx,return_meshes = False)
        )
        if n_heads > winning_n_heads:
            winning_n_heads = n_heads
            
            neck_face_idx_win = neck_face_idx
            neck_sdf_win = neck_sdf
            neck_width_win = neck_width
            
            head_face_idx_win = head_face_idx
            head_sdf_win = head_sdf
            head_width_win = head_width
            
            if verbose:
                print(f"New winning number of heads = {winning_n_heads}")
            
            if stop_segmentation_after_first_success:
                break

    # Filtering away the heads that are 
#     if no_head_coordinates is not None:
#         if len(head_face_idx) > 0:
#             if verbose:
#                 print(f"Attempting to filter head synapses close to coordinates")
#             head_face_idx = tu.filter_away_connected_comp_in_face_idx_with_minimum_vertex_distance_to_coordinates(
#                 mesh = mesh,
#                 face_idx = head_face_idx,
#                 coordinates = no_head_coordinates,
#                 verbose = verbose,
#                 plot = False,
#             )
            
#             if len(head_face_idx) > 0:
#                 head_face_idx = np.hstack(head_face_idx)
                
#             neck_face_idx = np.delete(np.arange(len(mesh.faces)),head_face_idx)
            
    neck_face_idx = neck_face_idx_win
    neck_sdf = neck_sdf_win
    neck_width = neck_width_win

    head_face_idx = head_face_idx_win
    head_sdf = head_sdf_win
    head_width = head_width_win       
            
    neck_mesh = mesh.submesh([neck_face_idx],append=True)
    if len(head_face_idx) > 0:
        head_mesh = mesh.submesh([head_face_idx],append=True)
    else:
        head_mesh = tu.empty_mesh()
        
    
    
    if plot_head_neck:
        neck_color = "green"
        head_color = "red"
        print(f"head_mesh ({head_color}): {head_mesh}")
        print(f"neck_mesh ({neck_color}): {neck_mesh}")

        nviz.plot_objects(meshes = [neck_mesh,head_mesh],
                         meshes_colors=[neck_color,head_color])
        
    if return_meshes:
        return_value= [head_mesh,neck_mesh]
    else:
        return_value = [head_face_idx.astype('int'),neck_face_idx.astype('int')]
        
    if return_sdf:
        return_value += [head_sdf,neck_sdf]
        
    if return_width:
        return_value += [head_width,neck_width]
        
    return return_value



def bouton_non_bouton_idx_from_branch(branch_obj,
                                     plot_branch_boutons=False,
                                     plot_face_idx=False,
                                     verbose = False):
    """
    Purpose: To add axon labels to the branch

    Ex: 
    return_face_idx = spu.bouton_non_bouton_idx_from_branch(branch_obj = neuron_obj[0][0],
    plot_branch_boutons = False,
    verbose = True,
    plot_face_idx = True,
    )

    """
    if plot_branch_boutons:
        nviz.plot_branches_with_boutons(branch_obj)

    head_neck_shaft_idx = np.ones(branch_obj.mesh_face_idx.shape)*spu.head_neck_shaft_dict["non_bouton"]

    boutons = branch_obj.boutons
    if boutons is not None and len(boutons) > 0:
        boutons_idx = np.concatenate(tu.convert_meshes_to_face_idxes(boutons,branch_obj.mesh))

        if verbose:
            print(f"{len(boutons_idx)} bouton faces")

        head_neck_shaft_idx[boutons_idx] = spu.head_neck_shaft_dict["bouton"]


    if plot_face_idx:
        nviz.plot_mesh_face_idx(branch_obj.mesh,head_neck_shaft_idx)
    return head_neck_shaft_idx

def head_neck_shaft_idx_from_branch(branch_obj,
    plot_face_idx  = False,
    add_no_head_label = True,
    verbose = False,
    process_axon_branches=True):
    """
    Purpose: To create an array
    mapping the mesh face idx of the branch to a
    label of head/neck/shaft

    Pseudocode: 
    1) Create an array the size of branch mesh 
    initialized to shaft
    2) iterate through all of the spine objects of the branch
        a) set all head index to head
        b) set all neck index to neck
        
    Ex: 
    spu.head_neck_shaft_idx_from_branch(branch_obj = neuron_obj_exc_syn[0][6],
    plot_face_idx  = True,
    verbose = True,)
    """

    if process_axon_branches and "axon" in branch_obj.labels:
        """
        Purpose: To add axon labels to the branch
        
        Psuedocode: 
        1) 
        """
        head_neck_shaft_idx = spu.bouton_non_bouton_idx_from_branch(branch_obj)
        if verbose:
            print(f'# of bouton faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["bouton"])}')
            print(f'# of non_bouton faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["non_bouton"])}')
    else:
        head_neck_shaft_idx = np.ones(branch_obj.mesh_face_idx.shape)*spu.head_neck_shaft_dict["shaft"]

        if branch_obj.spines_obj is not None:
            for sp_obj in branch_obj.spines_obj:
                if add_no_head_label and (not sp_obj.head_exist):
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.neck_face_idx]] = spu.head_neck_shaft_dict["no_head"]
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.head_face_idx]] = spu.head_neck_shaft_dict["head"]
                else:
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.head_face_idx]] = spu.head_neck_shaft_dict["head"]
                    head_neck_shaft_idx[sp_obj.mesh_face_idx[sp_obj.neck_face_idx]] = spu.head_neck_shaft_dict["neck"]

        if verbose:
            print(f'# of shaft faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["shaft"])}')
            print(f'# of head faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["head"])}')
            print(f'# of neck faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["neck"])}')
            print(f'# of no_head faces = {np.sum(head_neck_shaft_idx == spu.head_neck_shaft_dict["no_head"])}')

    if plot_face_idx:
        nviz.plot_objects(meshes = tu.split_mesh_into_face_groups(branch_obj.mesh,
                                  head_neck_shaft_idx,
                                  return_dict=False,
                                  return_idx = False),
                          meshes_colors="random")
    return head_neck_shaft_idx

def spine_density(obj,um = True):
    """
    n_spine / skeletal length (um)
    """
    skeletal_length = obj.skeletal_length
    if um:
        skeletal_length = skeletal_length/1000
        
    if skeletal_length == 0:
        return 0
    return len(obj.spines_obj)/skeletal_length
    
def spine_volume_density(obj,um = True):
    """
    sum spine volume (um**3) / skeletal length (um)
    """
    skeletal_length = obj.skeletal_length
    if um:
        skeletal_length = skeletal_length/1000
    
    if skeletal_length == 0:
        return 0
    return np.sum([k.volume for k in obj.spines_obj])/skeletal_length


def spine_density_over_limb_branch(neuron_obj,
                                     limb_branch_dict,
                                    synapse_type = "synapses",
                                    multiplier = 1,
                                     verbose = False,
                                   return_skeletal_length = False,
                                    ):
    """
    Purpose: To calculate the 
    spine density over lmb branch

    Application: To be used for cell type (E/I)
    classification

    Pseudocode: 
    1) Restrict the neuron branches to be processed
    for spine density
    2) Calculate the skeletal length over the limb branch
    3) Find the number of spines over limb branch
    4) Compute postsynaptic density
    
    Ex: 

    """
    sk_length = n_synapses = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict,
                                         feature="skeletal_length")

    n_spines = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict,
                                         feature="n_spines")
    if sk_length != 0:
        density = n_spines/sk_length
    else:
        density = 0

    density = density*multiplier

    if verbose:
        print(f"sk_length = {sk_length}")
        print(f"# of spines = {n_spines}")
        print(f"Density = {density}")

    if return_skeletal_length:
        return density,sk_length
    else:
        return density
    
def update_spines_obj(neuron_obj):
    """
    Will update all of the spine objects in a neuron
    """
    for l in neuron_obj:
        for b in l:
            if b.spines_obj is not None:
                b.spines_obj = [spu.Spine(k) for k in b.spines_obj]
                
def spine_str_label(spine_label):
    """
    spu.spine_str_label(-2)
    """
    if spine_label is None:
        return "no_label"
    if type(spine_label) != str:
        spine_label =  head_neck_shaft_dict_inverted[spine_label]
    return spine_label

def spine_int_label(spine_label):
    if type(spine_label) == str:
        spine_label =  head_neck_shaft_dict[spine_label]
    return spine_label

def set_soma_synapses_spine_label(neuron_obj,
                                soma_spine_label = "no_label"):
    for s in neuron_obj.get_soma_node_names():
        for syn in neuron_obj[s].synapses:
            syn.head_neck_shaft = spu.spine_int_label(soma_spine_label)


# -------------- 12/6: Doing the spine calculation ----------------
def calculate_spines_on_branch(
    branch,
    
    # arguments for the intial segmentation
    clusters_threshold=None,#2,
    smoothness_threshold=None,
    shaft_threshold = None,
    plot_spines_before_filter = False,
    
    spine_n_face_threshold=None,
    spine_sk_length_threshold = None,#1350,
    plot_spines_after_face_threshold = False,
    
    
    filter_by_bounding_box_longest_side_length=None,
    side_length_threshold = None,
    plot_spines_after_bbox_threshold = False,
    
    
    filter_out_border_spines=None, #this seemed to cause a lot of misses
    border_percentage_threshold=None,
    check_spine_border_perc=None,
    plot_spines_after_border_filter = False,
    
    
    skeleton_endpoint_nullification=None,
    skeleton_endpoint_nullification_distance = None,
    plot_spines_after_skeleton_endpt_nullification = False,
    
    
    soma_vertex_nullification = None,
    soma_verts = None,
    soma_kdtree = None,
    plot_spines_after_soma_nullification = False,
    

    #-------1/20 Addition --------
    filter_by_volume = None,
    calculate_spine_volume=None,
    filter_by_volume_threshold = None, #calculated from experiments
    plot_spines_after_volume_filter = False,
    
    
    print_flag = False,
    plot_segmentation = False,
    **kwargs,
    ):
    
    
    """
    Purpose: Will calculate the spines on a branch object
    
    Pseudocode: 
    1) Initial segmentation to get unfiltered spines
    2) Run the following filters: 
        a) face length
        b) By Bounding Box max length
        c) Border spines
        d) skeleton endpoint nullification
        e) soma nullification
        f) spine volume filter
    4) Return spines and volume
    
    
    Ex: 
    curr_limb = neuron_obj[2]
    soma_verts = np.concatenate([neuron_obj[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])

    branch = neuron_obj[2][7]
    sp_filt,sp_vol, spine_submesh_split_filtered_not = calculate_spines_on_branch(
        branch,

        shaft_threshold = 500,
        smoothness_threshold = 0.08,


        plot_spines_before_filter = False,
        plot_spines_after_face_threshold=False,

        plot_spines_after_bbox_threshold = True,
        plot_spines_after_border_filter = True,

        soma_verts = soma_verts,
        plot_spines_after_skeleton_endpt_nullification = True,
        plot_spines_after_soma_nullification = True,
        plot_spines_after_volume_filter = True,

        print_flag=True,

    )
    """
    
    if clusters_threshold is None:
        clusters_threshold = clusters_threshold_global
        
    if smoothness_threshold is None:
        smoothness_threshold = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
        
    if calculate_spine_volume is None:
        calculate_spine_volume = calculate_spine_volume_global
    
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_global
        
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    if filter_by_bounding_box_longest_side_length is None:
        filter_by_bounding_box_longest_side_length = filter_by_bounding_box_longest_side_length_global
        
    if side_length_threshold is None:
        side_length_threshold = side_length_threshold_global
        
    if filter_out_border_spines is None:
        filter_out_border_spines = filter_out_border_spines_global
        
    if skeleton_endpoint_nullification is None:
        skeleton_endpoint_nullification = skeleton_endpoint_nullification_global
        
    if skeleton_endpoint_nullification_distance is None:
        skeleton_endpoint_nullification_distance = skeleton_endpoint_nullification_distance_global
        
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global
        
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if check_spine_border_perc is None:
        check_spine_border_perc = check_spine_border_perc_global
        
    if filter_by_volume is None:
        filter_by_volume = filter_by_volume_global
        
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_global
    
    cgal_path=Path("./cgal_temp")
    
    # Step 1: Initial segmentation to get unfiltered spines
    spine_submesh_split= spu.get_spine_meshes_unfiltered_from_mesh(branch.mesh,
                                                                   segment_name="no_name",
                                                                clusters=clusters_threshold,
                                                                smoothness=smoothness_threshold,
                                                                cgal_folder = cgal_path,
                                                                delete_temp_file=True,
                                                                return_sdf=False,
                                                                print_flag=False,
                                                                shaft_threshold=shaft_threshold,
                                                                  plot_segmentation=plot_segmentation)
    
    
    if print_flag:
        print(f"--> n_spines found before filtering = {len(spine_submesh_split)}")
        
    if plot_spines_before_filter:
        print(f"plot_spines_before_filter: {len(spine_submesh_split)}")
        nviz.plot_objects(branch.mesh,
                         meshes = spine_submesh_split,
                         meshes_colors="red")
        
        
    # Step 2: Filter Spines by Face length
    if spine_n_face_threshold > 0 or spine_sk_length_threshold > 0:
        print(f"Filtering away by face and skeletal length")
        spine_submesh_split_filtered = spu.filter_spine_meshes(spine_submesh_split,
                                                            spine_n_face_threshold=spine_n_face_threshold,
                                                          spine_sk_length_threshold=spine_sk_length_threshold)
    else:
        spine_submesh_split_filtered = spine_submesh_split
    
    
    if plot_spines_after_face_threshold:
        print(f"plot_spines_after_face_threshold and sk_length_threshold: {len(spine_submesh_split_filtered)}")
        nviz.plot_objects(branch.mesh,
                         meshes = spine_submesh_split_filtered,
                         meshes_colors="red")
        
        
    # Step 3: Filter Spines by Bounding Box
    if filter_by_bounding_box_longest_side_length:
        old_length = len(spine_submesh_split_filtered)
        spine_submesh_split_filtered = tu.filter_meshes_by_bounding_box_longest_side(spine_submesh_split_filtered,
                                                                                 side_length_threshold=side_length_threshold)
    
        if plot_spines_after_bbox_threshold:
            print(f"plot_spines_after_bbox_threshold: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(branch.mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
        
        
    if filter_out_border_spines:
        if print_flag:
            print("Using the filter_out_border_spines option")
        spine_submesh_split_filtered = spu.filter_out_border_spines(branch.mesh,
                                                                    spine_submesh_split_filtered,
                                                                    border_percentage_threshold=border_percentage_threshold,
                                                                    check_spine_border_perc=check_spine_border_perc,
                                                                    verbose=print_flag
                                                                   )
        
        if plot_spines_after_border_filter:
            print(f"plot_spines_after_bbox_threshold: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(branch.mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
            
    if skeleton_endpoint_nullification:
        if print_flag:
            print("Using the skeleton_endpoint_nullification option")


        curr_branch_end_coords = sk.find_skeleton_endpoint_coordinates(branch.skeleton)
        spine_submesh_split_filtered = tu.filter_meshes_by_containing_coordinates(spine_submesh_split_filtered,
                                                    curr_branch_end_coords,
                                                    distance_threshold=skeleton_endpoint_nullification_distance)
        
        if plot_spines_after_skeleton_endpt_nullification:
            print(f"plot_spines_after_skeleton_endpt_nullification: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(branch.mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
            
    
    
    if soma_vertex_nullification:
        if soma_verts is not None:
            soma_kdtree = KDTree(soma_verts)
        if soma_kdtree is None:
            raise Exception("Requested endpoint nullification but no soma information given")
            
        if print_flag:
            print("Using the soma_vertex_nullification option")

        spine_submesh_split_filtered = spu.filter_out_soma_touching_spines(spine_submesh_split_filtered,
                                                    soma_kdtree=soma_kdtree)
        
        if plot_spines_after_soma_nullification:
            print(f"plot_spines_after_soma_nullification: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(branch.mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
            
            
    if calculate_spine_volume or filter_by_volume: 
        spine_volumes = np.array([tu.mesh_volume(k) for k in spine_submesh_split_filtered])
        
    if filter_by_volume:
        """
        Pseudocode: 
        1) Calculate the volumes of all the spines
        2) Filter those spines for only those above the volume

        """
        if len(spine_submesh_split_filtered) > 0:
            volume_kept_idx = np.where(spine_volumes > filter_by_volume_threshold)[0]
            if print_flag:
                print(f"Number of spines filtered away by volume = {len(spine_volumes) - len(volume_kept_idx)}")
            #spine_submesh_split_filtered_not = [spine_submesh_split_filtered[k] for k in range(len(spine_volumes)) if k not in volume_kept_idx]
            spine_submesh_split_filtered = [spine_submesh_split_filtered[k] for k in volume_kept_idx]
            
            spine_volumes = spine_volumes[volume_kept_idx]
            
        if plot_spines_after_volume_filter:
            print(f"plot_spines_after_volume_filter: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(branch.mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
        
        
    if calculate_spine_volume:
        return spine_submesh_split_filtered,spine_volumes
    else:
        return spine_submesh_split_filtered
    
def calculate_spines_on_neuron(
    neuron_obj,
    limb_branch_dict=None,
    #---arguments for the query restriction -----
    #query="width > 400 and n_faces_branch>100",
    #query="median_mesh_center > 140 and n_faces_branch>100",#previous used median_mesh_center > 140
    query=None,#previous used median_mesh_center > 140
    plot_query = False,
    
    # limb specific arguments for spine calculation:
    soma_vertex_nullification = None,
    
    #---- arguments for the spine calculation on a branch ----
    
    #-- arguments for volume -- 
    calculate_spine_volume = None,
    
    print_flag=False,
    limb_branch_dict_exclude = None,
    **kwargs):
    
    if query is None:
        query = query_global
    
    if calculate_spine_volume is None:
        calculate_spine_volume = calculate_spine_volume_global
    
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global
    
    """
    Purpose: Will recalculate spines over a neuron object
    
    Pseudocode: 
    1) Recalculates the median mesh center if it was requested for a filter 
    but not already calculated
    
    
    Ex: 
    spu.calculate_spines_on_neuron(
    recovered_neuron,
    plot_query = False,
    print_flag = True,
    )
    
    nviz.plot_spines(recovered_neuron)
    
    """
    if query is None:
        query = query_global
    
    # --- Step 1: Applies query to see which branches to calculate spines over ---
    if limb_branch_dict is None:
        print(f"query = {query}")
        if type(query) == dict():
            functions_list = query["functions_list"]
            current_query = query["query"]
        else:
            functions_list = ["median_mesh_center","n_faces_branch"]
            current_query = query


        #check that have calculated the median mesh center if required
        if "median_mesh_center" in functions_list:
            if len(neuron_obj.get_limb_node_names())>0 and "median_mesh_center" not in neuron_obj[0][0].width_new.keys():
                print("The median_mesh_center was requested but has not already been calculated so calculating now.... ")

                wu.calculate_new_width_for_neuron_obj(neuron_obj,
                                                      no_spines=False,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
            else:
                print("The median_mesh_center was requested and HAS already been calculated")

        limb_branch_dict = ns.query_neuron(neuron_obj,
                           functions_list=functions_list,
                           query=current_query,
                                         plot_limb_branch_dict=plot_query)
    else:
        if plot_query:
            nviz.plot_limb_branch_dict(neuron_obj,limb_branch_dict)
            
    if print_flag:
        print(f"limb_branch_dict = {limb_branch_dict}")
        
        
    # --- Step 2: Calculating the Spines over ---
        
    for limb_idx in limb_branch_dict.keys():
        curr_limb = neuron_obj[limb_idx]
        
        if soma_vertex_nullification:
            soma_verts = np.concatenate([neuron_obj[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])
            soma_kdtree = KDTree(soma_verts)
        else:
            soma_kdtree = None
            
        for branch_idx in limb_branch_dict[limb_idx]:
            if limb_branch_dict_exclude is not None:
                if limb_idx in limb_branch_dict_exclude:
                    if branch_idx in limb_branch_dict_exclude[limb_idx]:
                        if print_flag:
                            print(f"Skipping because in limb_branch exclude")
                        continue
            
            curr_branch = curr_limb[branch_idx]
            
            if print_flag:
                print(f"Working on limb {limb_idx} branch {branch_idx}")
                
                
            if calculate_spine_volume:
                branch_spines,branch_spines_vol  = calculate_spines_on_branch(
                    curr_branch,
                    soma_kdtree = soma_kdtree,
                    print_flag=print_flag,
                    calculate_spine_volume =calculate_spine_volume,
                    **kwargs
                )
                curr_branch.spines_volume = list(branch_spines_vol)
                already_calculated_volumes = True
            else:
                branch_spines  = calculate_spines_on_branch(
                    curr_branch,
                    soma_kdtree = soma_kdtree,
                    print_flag=print_flag,
                    calculate_spine_volume =calculate_spine_volume,
                    **kwargs
                )
                already_calculated_volumes = False
            
            curr_branch.spines = list(branch_spines)
            
            
            if calculate_spine_volume and not already_calculated_volumes:
                curr_branch.compute_spines_volume()
                
                
# --------------- for filtering spines: 1/25 ------------
def print_filter_spine_thresholds():
    print(f"spine_n_face_threshold_global = {spine_n_face_threshold_global}")
    print(f"filter_by_volume_threshold_global = {filter_by_volume_threshold_global}")
    print(f"spine_sk_length_threshold_global = {spine_sk_length_threshold_global}")
    
def filter_spines_by_size_branch(
    branch_obj,
    spine_n_face_threshold=None,
    filter_by_volume_threshold = None,
    spine_sk_length_threshold= None,
    verbose = False,
    assign_back_to_obj = True,
    calculate_spines_length_on_whole_neuron = True,
    ):
    """
    Purpose: To filter away any of the 
    spines according to the size thresholds

    """
    if spine_n_face_threshold is None:
        spine_n_face_threshold= spine_n_face_threshold_global
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_global
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_global
        
    if verbose:
        print(f"Spines before filtering away: {len(branch_obj.spines_obj)}")
        
    if branch_obj.spines_obj is not None and len(branch_obj.spines_obj) > 0:
        s_objs_keep = [s_obj for s_obj in branch_obj.spines_obj
                              if ((len(s_obj.mesh.faces) > spine_n_face_threshold)
                               and (s_obj.volume > filter_by_volume_threshold)
                                 and (s_obj.skeletal_length > spine_sk_length_threshold))
                              ]
        if verbose:
            print(f"Spines before filtering away: {len(s_objs_keep)}")

        if assign_back_to_obj:
            branch_obj.spines_obj = s_objs_keep
            branch_obj.spines = [s.mesh for s in s_objs_keep]
            branch_obj.spines_volume = [s.volume for s in s_objs_keep]

        return_value = s_objs_keep
    elif branch_obj.spines is not None:
        new_spines = []
        new_spines_volume = []
        for spine_mesh,spine_vol in zip(branch_obj.spines,branch_obj.spines_volume):
            if ((len(spine_mesh.faces) >= spine_n_face_threshold) 
                and (spine_vol >= filter_by_volume_threshold)
               and (spine_length(spine_mesh) > spine_sk_length_threshold)):
                new_spines.append(spine_mesh)
                new_spines_volume.append(spine_vol)
                
        if assign_back_to_obj:
            branch_obj.spines = new_spines
            branch_obj.spines_volume = new_spines_volume
        
        return_value = new_spines
    else:
        return_value = None
        
    if calculate_spines_length_on_whole_neuron:
        nru.calculate_spines_skeletal_length(branch_obj)
        
    return return_value
    
def filter_spines_by_size(
    neuron_obj,
    spine_n_face_threshold=None,
    filter_by_volume_threshold = None,
    verbose = False,
    **kwargs
    ):
    st = time.time()
    for limb_obj in neuron_obj:
        for branch_obj in limb_obj:
            spu.filter_spines_by_size_branch(branch_obj,
                                             **kwargs
                                )
    if verbose:
        print(f"Total time for spine filtering: {time.time() - st}")
        
    return neuron_obj


# ------------- 2/7 adjustments -----------------
def adjust_obj_with_face_offset(
    spine_obj,
    face_offset,
    verbose = False,
    ):
    """
    Purpose: To adjust the spine properties that
    would be affected by a different face idx
    
    Ex: 
    b_test = neuron_obj[0][18]
    sp_obj = b_test.spines_obj[0]
    sp_obj.export()

    spu.adjust_spine_obj_with_face_offset(
        sp_obj,
        face_offset = face_offset,
        verbose = True
    ).export()
    """
    new_obj = copy.deepcopy(spine_obj)
    for k,v in spine_obj.export().items():
        if "face_idx" not in k:
            continue
        
        if v is None:
            continue
            
        if k in ["head_face_idx","neck_face_idx"]:
            continue
            
        if verbose:
            print(f"Adjusting {k} because face_idx and not None")
            
        try:
            setattr(new_obj,k,v + face_offset)
        except:
            setattr(new_obj,f"_{k}",v + face_offset)
        
    return new_obj

def spine_length(
    spine_mesh,
    verbose = False,
    surface_skeleton_method = "slower",
    plot = False):
    if is_spine_obj(spine_mesh):
        spine_mesh = spine_mesh.mesh
        
    #spine_mesh = tu.largest_conn_comp(spine_mesh)
    
    if surface_skeleton_method == "meshparty":
        curr_sk = sk.surface_skeleton(spine_mesh)
    else:
        curr_sk = sk.generate_surface_skeleton_slower(spine_mesh)
        
    curr_sk_length = sk.calculate_skeleton_distance(curr_sk)
    if verbose:
        print(f"skeletal length = {curr_sk_length}")
    if plot:
        nviz.plot_objects(spine_mesh,curr_sk)
    return curr_sk_length

def complete_spine_processing(
    neuron_obj,
    compute_initial_spines = True,
    compute_no_spine_width = True,
    compute_spine_objs = True,
    limb_branch_dict_exclude = None,
    verbose = False,
    plot= False,
    ):
    """
    Will redo all of the spine processing

    Pseudocode: 
    1) Redo the spines
    2) Redo the spine widthing
    3) Redo the spine calculation
    
    Ex: 
    import time
    spu.set_global_parameters_and_attributes_by_data_type(data_type)
    spu.complete_spine_processing(
        neuron_obj,
        verbose = True)

    """
    global_time = time.time()
    
    if compute_initial_spines: 
        st = time.time()
        spu.calculate_spines_on_neuron(
            neuron_obj,
            limb_branch_dict_exclude = neuron_obj.axon_limb_branch_dict)
        
        if verbose:
            print(f"Time for compute_initial_spines = {time.time() - st}")
            
    if compute_no_spine_width: 
        from neurd import width_utils as wu
        st = time.time()
        widths_to_calculate=["no_spine_median_mesh_center"]
        
        for w in widths_to_calculate:
            wu.calculate_new_width_for_neuron_obj(neuron_obj,width_name=w)
            
        if verbose:
            print(f"Time for compute_no_spine_width = {time.time() - st}")
            
    if compute_spine_objs: 
        st = time.time()
        neuron_obj = spu.add_head_neck_shaft_spine_objs(neuron_obj,
                                                        verbose = verbose
                                                                      )
        
        if verbose:
            print(f"Time for compute_spine_objs = {time.time() - st}")
            
    if plot:
        spu.plot_spines_head_neck(neuron_obj)
        
    return neuron_obj

# -------------- for other properties per spine obj -------------

def calculate_spine_obj_mesh_skeleton_coordinates_for_branch(branch_obj):
    if branch_obj.spines_obj is None:
        return branch_obj
    branch_obj.spines_obj = [spu.calculate_spine_obj_mesh_skeleton_coordinates(branch_obj,k)
                            for k in branch_obj.spines_obj] 
    return branch_obj

def calculate_spine_obj_mesh_skeleton_coordinates(
    branch_obj=None,
    spine_obj=None,
    coordinate_method = 'first_coordinate',#"mean",
    plot_intersecting_vertices = False,
    plot_closest_skeleton_coordinate = False,
    spine_objs = None,
    branch_shaft_mesh_face_idx=None,
    verbose = False,
    verbose_time = False,
    mesh = None,
    skeleton = None,
    **kwargs
    ):


    """
    Will compute a lot of the properties of 
    spine objects that are equivalent to those
    computed in syu.add_valid_synapses_to_neuron_obj

    The attributes include

    "endpoints_dist",
    "upstream_dist",
    "downstream_dist",
    "coordinate",
    "closest_sk_coordinate",
    "closest_face_idx",
    "closest_branch_face_idx",
    "closest_face_dist",
    "closest_face_coordinate",

    Pseudocode: 
    1) Make sure the branches have upstream and downstream set
    2) Find intersection of vertices between branch and shaft
    3) Find average vertices that make up the coordinate
    4) Find the closest mesh coordinate
    5) Find the closest skeeleton point

    """
    
    ''' Old method
    try:
        
        overlap_verts= tu.find_border_vertices(
            spine_obj.mesh,
            return_coordinates=True)
    except:
        branch_obj_minus_spine = tu.subtract_mesh(
            branch_obj.mesh,
            spine_obj.mesh,
        )
        overlap_verts = nu.intersect2d(
            spine_obj.mesh.vertices,
            branch_obj_minus_spine.vertices
        )
        


    if len(overlap_verts) == 0:
#         overlap_verts = spine_obj.mesh.vertices[0].reshape(-1,3)
#         if verbose:
#             print(f"Using first spine vertex as coordinate because no overlapping")
        overlap_verts = tu.closest_mesh_vertex_to_other_mesh(
            spine_obj.mesh,
            branch_obj.mesh_shaft,
            plot = False,
            verbose = False,
        ).reshape(-1,3)



    #3) Find average vertices that make up the coordinate    
    if coordinate_method == "mean":
        coordinate = np.mean(overlap_verts,axis = 0)
    else:
        coordinate = overlap_verts[0]
    if verbose:
        print(f"coordinate = {coordinate}")
    '''
    attr_to_set = [
        "coordinate",
        "closest_branch_face_idx",
        "closest_face_dist",
        "closest_face_coordinate",
        "coordinate_border_verts"
    ]
    
    if branch_obj is not None and mesh is None:
        mesh = branch_obj.mesh
    if branch_obj is not None and skeleton is None:
        skeleton = branch_obj.skeleton
    
    
    if spine_objs is not None:
        meshes_to_minus = [k.mesh for k in spine_objs]
    else:
        meshes_to_minus = None
        
    # find the overlapping vertices
    if branch_shaft_mesh_face_idx is not None:
        overlapping_vertices = tu.overlapping_vertices_from_face_lists(
            mesh,
            face_lists = [branch_shaft_mesh_face_idx,spine_obj.mesh_face_idx],
            return_idx = False
        )
    else:
        overlapping_vertices = None
        
    #print(f"overlapping_vertices = {overlapping_vertices}")
    
    try:
        coordinate,coordinate_border_verts = tu.coordinate_on_mesh_mesh_border(
            mesh=spine_obj.mesh,
            mesh_border=mesh,
            meshes_to_minus = meshes_to_minus,
            coordinate_method = coordinate_method,#"mean",
            overlapping_vertices=overlapping_vertices,
            verbose = verbose,
            verbose_time=verbose_time,
            return_winning_coordinate_group = True,
            plot=False,
        )
    except:
        coordinate = tu.closest_mesh_coordinate_to_other_mesh(
            spine_obj.mesh,
            mesh
        )
        
        coordinate_border_verts = coordinate
    

    if plot_intersecting_vertices:
        print(f"plot_intersecting_vertices")
        nviz.plot_objects(mesh,
                         meshes=[spine_obj.mesh],
                          meshes_colors="red",
                         scatters=[overlap_verts,coordinate],
                          scatters_colors=["orange",'blue'],
                         scatter_size=0.5)

    #4) Find the closest mesh coordinate
    closest_branch_face_idx = tu.closest_face_to_coordinate(mesh,coordinate)
    closest_face_coordinate= mesh.triangles_center[closest_branch_face_idx]
    closest_face_dist = np.linalg.norm(coordinate - closest_face_coordinate)

    if verbose:
        print(f"closest_branch_face_idx = {closest_branch_face_idx}")
        print(f"closest_face_coordinate = {closest_face_coordinate}")
        print(f"closest_face_dist = {closest_face_dist}")

    #5) Find the closest skeeleton point
    closest_sk_coordinate = None
    if skeleton is not None:
        closest_sk_coordinate = sk.closest_skeleton_coordinate(skeleton,
                                    closest_face_coordinate)
        attr_to_set.append("closest_sk_coordinate")
    if verbose:
        print(f"closest_sk_coordinate= {closest_sk_coordinate}")

    if plot_closest_skeleton_coordinate:
        print(f"plot_closest_skeleton_coordinate")
        nviz.plot_objects(mesh,skeleton,
                         meshes=[spine_obj.mesh],
                          meshes_colors="red",
                         scatters=[closest_sk_coordinate],
                          scatters_colors=['blue'],
                         scatter_size=0.5)

    
    for k in attr_to_set:
        setattr(spine_obj,k,eval(k))
        
    return spine_obj

def id_from_idx(
    limb_idx,
    branch_idx,
    spine_idx,
    ):
    
    limb_idx = nru.get_limb_int_name(limb_idx)
    name_as_string = f"{str(limb_idx).zfill(2)}{str(branch_idx).zfill(3)}{str(spine_idx).zfill(3)}"
    return int(name_as_string)
    
def calculate_spine_obj_attr_for_neuron(
    neuron_obj,
    verbose = False,
    create_id = True,
    **kwargs
    ):
    """
    Purpose: To set all of the
    neuron_obj spine attributes

    Pseudocode: 
    for limbs
        for branches
            1) calculate the mesh and skeleton info
            2) calculate_branch_attr_soma_distances_on_limb

    calculate the soma distance
    
    Ex: neuron_obj= spu.calculate_spine_obj_attr_for_neuron(neuron_obj,verbose = True)

    """
    
    for limb_idx in neuron_obj.get_limb_names():
        if verbose:
            print(f"--- Working on Limb {limb_idx}")
        limb_obj = neuron_obj[limb_idx]
        for branch_idx in limb_obj.get_branch_names():
            if verbose:
                print(f"     Branch {branch_idx}")
            branch_obj = limb_obj[branch_idx]
            #1) calculate the mesh and skeleton info
            limb_obj[branch_idx] = spu.calculate_spine_obj_mesh_skeleton_coordinates_for_branch(branch_obj)
            
            if create_id and branch_obj.spines_obj is not None:
                for s_idx,sp_ogj in enumerate(branch_obj.spines_obj):
                    branch_obj.spine_id = spu.id_from_idx(limb_idx,branch_idx,s_idx)

        #2) calculate_branch_attr_soma_distances_on_limb
        limb_obj = bau.calculate_branch_attr_soma_distances_on_limb(
            limb_obj,
            branch_attr="spines_obj",
            calculate_endpoints_dist_if_empty=True
        )
        neuron_obj[limb_idx] = limb_obj
        
        

#     if verbose:
#         print(f"Working on calculate_neuron_soma_distance")
#     bau.calculate_neuron_soma_distance(neuron_obj,branch_attr="spines_obj")   
    if verbose:
        print(f"Working on calculate_neuron_soma_distance_euclidean")
    bau.calculate_neuron_soma_distance_euclidean(neuron_obj,branch_attr="spines_obj")
    bau.set_limb_branch_idx_to_attr(neuron_obj,branch_attr="spines_obj")
    
    
    return neuron_obj

def calculate_endpoints_dist(branch_obj,spine_obj):
    bau.calculate_endpoints_dist(branch_obj,spine_obj)
    
def calculate_upstream_downstream_dist_from_up_idx(spine_obj,up_idx):
    bau.calculate_upstream_downstream_dist_from_up_idx(spine_obj,up_idx=up_idx)
    
    
# ------- for calculating properties of spines ----
def skeleton_from_spine(spine,plot=False):
    if "trimesh" in str(type(spine)):
        mesh = spine
    else:
        mesh = spine.mesh
        
    spine_sk = sk.surface_skeleton(mesh,plot=plot)
    return spine_sk

def skeletal_length_from_spine(spine,plot=False):
    return sk.calculate_skeleton_distance(spu.skeleton_from_spine(spine))

def volume_from_spine(spine,default_value = 0):
    try:
        return tu.mesh_volume(spine.mesh)
    except:
        return default_value


# ---------------------- 11/3: Calculating the spines with all of the information  -----
def spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj(
    branch_obj = None,
    plot_segmentation = False,
    ensure_mesh_conn_comp = True,
    plot_spines_before_filter = False,
    
    
    filter_out_border_spines = None,
    border_percentage_threshold = None,
    plot_spines_after_border_filter = False,

    skeleton_endpoint_nullification = False,
    skeleton_endpoint_nullification_distance = None,
    plot_spines_after_skeleton_endpt_nullification = False,

    soma_vertex_nullification = None,
    soma_verts = None,
    soma_kdtree = None,
    plot_spines_after_soma_nullification = None,
    
    plot = False,
    verbose = False,
    mesh = None,
    skeleton = None,
    **kwargs
    ):
    
    if filter_out_border_spines is None:
        filter_out_border_spines = filter_out_border_spines_global
        
    if border_percentage_threshold is None:
        border_percentage_threshold = border_percentage_threshold_global
        
    if skeleton_endpoint_nullification is None:
        skeleton_endpoint_nullification = skeleton_endpoint_nullification_global
        
    if skeleton_endpoint_nullification_distance is None:
        skeleton_endpoint_nullification_distance = skeleton_endpoint_nullification_distance_global
        
    if soma_vertex_nullification is None:
        soma_vertex_nullification = soma_vertex_nullification_global

    
    if mesh is None:
        mesh = branch_obj.mesh
        
        

    (spines,
     spines_sdf,
     spines_mesh_idx) = spu.get_spine_meshes_unfiltered_from_mesh(
        mesh,
        delete_temp_file=True,
        return_sdf=True,
        return_mesh_idx = True,
        print_flag=False,
        plot_segmentation=plot_segmentation,
        ensure_mesh_conn_comp=ensure_mesh_conn_comp,
        plot = False,
        **kwargs
    )
    

    if plot_spines_before_filter:
        print(f"plot_spines_before_filter: {len(spines)}")
        nviz.plot_objects(mesh,
                         meshes = spines,
                         meshes_colors="red")
        



    spine_submesh_split_filtered = spines
    
    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered
    
    if filter_out_border_spines:
        if verbose:
            print("Using the filter_out_border_spines option")
        spine_idx = spu.filter_out_border_spines(
            mesh,
            spine_submesh_split_filtered,
            border_percentage_threshold=border_percentage_threshold,
            check_spine_border_perc=check_spine_border_perc,
            verbose=verbose,
            return_idx = True,
        )
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

        if plot_spines_after_border_filter:
            print(f"plot_spines_after_bbox_threshold: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(mesh,
                    meshes = spine_submesh_split_filtered,
                    meshes_colors="red"
            )
        if verbose:
            print(f"After filter_out_border_spines: # of spines = {len(filter_out_border_spines)}")

    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered

    if skeleton_endpoint_nullification and (branch_obj is not None or skeleton is not None):
        if skeleton is None:
            skeleton = branch_obj.skeleton
        if verbose:
            print("Using the skeleton_endpoint_nullification option")


        curr_branch_end_coords = sk.find_skeleton_endpoint_coordinates(skeleton)
        spine_idx = tu.filter_meshes_by_containing_coordinates(
            spine_submesh_split_filtered,
            curr_branch_end_coords,
            distance_threshold=skeleton_endpoint_nullification_distance,
            return_indices=True
        )
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

        if plot_spines_after_skeleton_endpt_nullification:
            print(f"plot_spines_after_skeleton_endpt_nullification: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
            
        if verbose:
            print(f"After skeleton_endpoint_nullification: # of spines = {len(spine_submesh_split_filtered)}")


    if len(spine_submesh_split_filtered) == 0:
        return spine_submesh_split_filtered
    
    if soma_vertex_nullification and (soma_kdtree is not None or soma_kdtree is not None):
        if soma_verts is not None:
            soma_kdtree = KDTree(soma_verts)
        if soma_kdtree is None:
            raise Exception("Requested soma vertex but no soma information given")

        if verbose:
            print("Using the soma_vertex_nullification option")

        spine_idx = spu.filter_out_soma_touching_spines(spine_submesh_split_filtered,
                                                    soma_kdtree=soma_kdtree,
                                                       return_idx = True)
        
        spine_submesh_split_filtered = spine_submesh_split_filtered[spine_idx]
        spines_sdf=spines_sdf[spine_idx]
        spines_mesh_idx = spines_mesh_idx[spine_idx]

        if plot_spines_after_soma_nullification:
            print(f"plot_spines_after_soma_nullification: {len(spine_submesh_split_filtered)}")
            nviz.plot_objects(mesh,
                             meshes = spine_submesh_split_filtered,
                             meshes_colors="red")
            
        if verbose:
            print(f"After soma_vertex_nullification: # of spines = {len(spine_submesh_split_filtered)}")
            
            
    # Creating the spine objects
    spine_objs = [spu.Spine(
        mesh = k,
        mesh_face_idx = k_idx,
        sdf = s
    ) for k,k_idx,s in zip(spine_submesh_split_filtered,spines_mesh_idx,spines_sdf,)]
    
    if plot:
        nviz.plot_objects(
            mesh,
            meshes = [k.mesh for k in spine_objs],
            meshes_colors = "red",
            buffer = 0,
        )
    
    if verbose:
        print(f"Final Number of Spine Objects = {len(spine_objs)}")
        
    return spine_objs


def spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_mesh(
    mesh,
    skeleton = None,
    **kwargs
    ):
    
    return spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj(
        branch_obj = None,
        mesh = mesh,
        skeleton = skeleton,
        **kwargs
        )


def df_from_spine_objs(
    spine_objs,
    attributes_to_skip = (
        "mesh_face_idx",
        "mesh",
        "neck_face_idx",
        "head_face_idx",
        "sdf",
        "skeleton",
    ),
    attributes_to_add = computed_attributes_to_export,
    columns_at_front = computed_attributes_to_export,
    columns_at_back = None,
    attributes = None,
    add_volume_to_area_ratio = False,
    verbose = False,
    verbose_loop = False,
    ):
    """
    Purpose: make a spine attribute
    dataframe from a list of spines

    Pseudocode: 
    1) 
    """
    st = time.time()
    spine_objs = nu.to_list(spine_objs)
    
    dicts = []
    for j,k in enumerate(spine_objs):
        if verbose_loop:
            print(f"spine {j}")
        curr_dict = k.export(
            attributes_to_skip=attributes_to_skip,
            attributes_to_add=attributes_to_add,
            attributes=attributes,
        )
        if add_volume_to_area_ratio:
            curr_dict["spine_volume_to_spine_area"] = spu.spine_volume_to_spine_area(k)
        dicts.append(
            curr_dict
        )
        
        
    
    df = pd.DataFrame.from_records(dicts)
    
    if attributes is None:
        df = pu.order_columns(
            df,
            columns_at_front=columns_at_front,
            columns_at_back=columns_at_back,
        )
    if verbose:
        print(f"Time for generating df {time.time() - st}")
    
    return df


def plot_spine_objs(
    spine_objs,
    branch_obj = None,
    mesh = None,
    plot_mesh_centers = True,
    spine_color = "random",
    mesh_alpha = 1,
    ):
    

    if branch_obj is not None:
        mesh = branch_obj.mesh
        
    scatters = None
    if plot_mesh_centers:
        scatters = [np.vstack([k.mesh_center for k in spine_objs]).reshape(-1,3)]
    nviz.plot_objects(
        mesh,
        meshes = [k.mesh for k in spine_objs],
        meshes_colors = spine_color,
        scatters=scatters,
        mesh_alpha=mesh_alpha,
    )
def filter_spine_objs_from_restrictions(
    spine_objs,
    restrictions,
    spine_df = None,
    verbose=False,
    return_idx = False,
    joiner = "AND",
    plot = False,
    **kwargs
    ):
    """
    Purpose: Want to filter the spines with
    a list of queries
    """
    if spine_df is None:
        spine_df = spu.df_from_spine_objs(
            spine_objs,
            add_volume_to_area_ratio=True,
            **kwargs
        )
        
    
        
    idx = pu.query_table_from_list(
        spine_df,
        restrictions=restrictions,
        verbose_filtering=verbose,
        return_idx = True,
        joiner=joiner,
        **kwargs
    )
    
    spine_objs_new = [spine_objs[k] for k in idx]
    
    if return_idx:
        return idx
    
    if plot:
        plot_spine_objs(spine_objs_new)
    return spine_objs_new

query_spine_objs = filter_spine_objs_from_restrictions


def example_comparing_mesh_segmentation_vs_spine_head_segmentation(
    spine_mesh,
    ):


    cluster_idx = meshu.segment_mesh(
        spine_mesh,
        verbose = False,
        eta = 0.15,
        delta = 1,
    )

    mesh_dict,mesh_face_dict = tu.split_mesh_into_face_groups(
        spine_mesh,
        cluster_idx,
        check_connect_comp=True,
        plot = True
    )

    spu.spine_head_neck(
        spine_mesh,
        plot_segmentation = True,
    )
    
def plot_spine_objs_on_branch(
    spines_obj,
    branch_obj,
    plot_spines_individually = True,
    ):
    
    spines_obj = nu.to_list(spines_obj)
    meshes = [k.mesh for k in spines_obj]
    if plot_spines_individually:
        nviz.plot_objects(
            meshes=meshes,
            meshes_colors = "random"
        )
    nviz.plot_objects(
        branch_obj.mesh,
        meshes = meshes,
        meshes_colors="red",
        scatters=[k.centroid for k in meshes]
    )
    
def spine_volume_to_spine_area(spine_obj):
    if spine_obj.volume is None:
        return 0
    return (spine_obj.volume/spu.volume_divisor)/(spine_obj.area/spu.area_divisor)

def filter_spine_objs_by_size_bare_minimum(
    spine_objs,
    spine_n_face_threshold = None,#6,
    spine_sk_length_threshold = None,#306.6,
    filter_by_volume_threshold = None,#900496.186,
    bbox_oriented_side_max_min = None,#300,
    sdf_mean_min = None,#0,
    spine_volume_to_spine_area_min = None,
    verbose = False,
    ):
    if len(spine_objs) == 0:
        return spine_objs
    
    if spine_n_face_threshold is None:
        spine_n_face_threshold = spine_n_face_threshold_bare_min_global
        
    if spine_sk_length_threshold is None:
        spine_sk_length_threshold = spine_sk_length_threshold_bare_min_global
        
    if filter_by_volume_threshold is None:
        filter_by_volume_threshold = filter_by_volume_threshold_bare_min_global
        
    if bbox_oriented_side_max_min is None:
        bbox_oriented_side_max_min = bbox_oriented_side_max_min_bare_min_global
        
    if sdf_mean_min is None:
        sdf_mean_min = sdf_mean_min_bare_min_global
        
    if spine_volume_to_spine_area_min is None:
        spine_volume_to_spine_area_min = spine_volume_to_spine_area_min_bare_min_global
        
        
    #print(f"filter_by_volume_threshold = {filter_by_volume_threshold}")
    sp_objs_filt = spu.filter_spine_objs_from_restrictions(
        spine_objs,
        restrictions = [
            f"n_faces >= {spine_n_face_threshold}",
            f"(skeletal_length >= {spine_sk_length_threshold}) or (skeletal_length != skeletal_length)",
            f"(volume >= {filter_by_volume_threshold}) or (volume != volume)",
            f"sdf_mean > {sdf_mean_min}",
            f"bbox_oriented_side_max > {bbox_oriented_side_max_min}",
            f"spine_volume_to_spine_area >= {spine_volume_to_spine_area_min}",
        ],
        verbose = verbose,
    )

    return sp_objs_filt
    
    
def example_trying_to_skeletonize_spine(spine_obj):
    from mesh_tools import meshparty_skeletonize as m_sk
    from mesh_tools import skeleton_utils as sk

    mesh = spine_obj.mesh
    root = spine_obj.coordinate
    invalidation_d = 1000
    kwargs = dict()
    sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(
        mesh,
        root=root,
        filter_mesh=False,
        invalidation_d=invalidation_d,
        **kwargs
    )

    (segment_branches_filtered, #skeleton branches
    divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
    segment_widths_median_filtered) = m_sk.skeleton_obj_to_branches(
        sk_meshparty_obj,
        mesh,
        meshparty_n_surface_downsampling = 0,
        meshparty_segment_size = 0,
        combine_close_skeleton_nodes_threshold=0,
        filter_end_nodes=True,
        filter_end_node_length=300,
    )

    if plot:
        nviz.plot_objects(
            mesh,
            sk.stack_skeletons(segment_branches_filtered)
        )
        
def split_head_mesh(
    spine_obj,
    return_face_idx_map = True,
    plot = False,):
    """
    Purpose: want to divide mesh into connected
    components and optionally return the mask
    of mapping fae to component
    """
    head_meshes,head_face_idx = tu.connected_components_from_mesh(
        spine_obj.head_mesh,
        return_face_idx_map = True,
        plot=plot
    )
    
    if return_face_idx_map:
        return head_meshes,head_face_idx
    else:
        return head_meshes
    
def spine_mesh(spine_obj):
    return spine_obj.mesh
    
def mesh_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0
    ):
    
    args = dict()
    if compartment == "head":
        mesh_func = spu.head_mesh_splits_from_index
        args = dict(index=index)
    elif compartment == "neck":
        mesh_func = spu.neck_mesh
    elif compartment == 'spine':
        mesh_func = spu.spine_mesh
    elif compartment == 'no_head':
        mesh_func = spu.neck_mesh
    else:
        raise Exception("")
    
        
    return mesh_func(spine_obj,**args)

def mesh_attribute_from_compartment(
    spine_obj,
    attribute_func,
    compartment = "head",
    index = 0,
    shaft_default_value = None,
    **kwargs
    ):
    
    if type(attribute_func) == str:
        attribute_func = eval(f"tu.{attribute_func}")
        
    if type(compartment) != str:
        compartment,index = compartment_index_from_id(compartment)
        
    if compartment == "shaft":
        return shaft_default_value
    
    curr_mesh = spu.mesh_from_compartment(spine_obj,compartment=compartment,index=index)
    if curr_mesh is None or len(curr_mesh.faces) == 0:
        return None
    return attribute_func(
        curr_mesh,
        **kwargs
    )
    
def width_ray_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    percentile = 50,
    default_value_if_empty = 0,
    ):
    
    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.width_ray_trace_perc,
        compartment = compartment,
        index = index,
        percentile=percentile,
        default_value_if_empty=default_value_if_empty,
        )

def width_ray_80_perc_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    default_value_if_empty = 0,
    ):
    
    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.width_ray_trace_perc,
        compartment = compartment,
        index = index,
        percentile=80,
        default_value_if_empty=default_value_if_empty,
        )

def area_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):
    

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.area,
        compartment = compartment,
        index = index,
        )

def n_faces(
    spine_obj,
    compartment = "head",
    index = 0,
    ):
    
    def my_func(mesh):
        return len(mesh.faces)

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=my_func,
        compartment = compartment,
        index = index,
        )

def volume_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.mesh_volume,
        compartment = compartment,
        index = index,
        )

def skeletal_length_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.skeletal_length_from_mesh,
        compartment = compartment,
        index = index,
    )

bbox_oriented = False
def bbox_min_x_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_x,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_min_y_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_y,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_min_z_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_min_z,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_x_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_x,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_y_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_y,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def bbox_max_z_nm_from_compartment(
    spine_obj,
    compartment = "head",
    index = 0,
    ):

    return mesh_attribute_from_compartment(
        spine_obj,
        attribute_func=tu.bbox_max_z,
        compartment = compartment,
        index = index,
        oriented=bbox_oriented,
    )

def spine_compartment_mesh_functions(
    compartments = ("spine","head",'neck',),
    stats_functions = (
        "width_ray_from_compartment",
        "width_ray_80_perc_from_compartment",
        "area_from_compartment",
        "volume_from_compartment",
        "skeletal_length_from_compartment",
        "n_faces",
        "bbox_min_x_nm_from_compartment",
        "bbox_min_y_nm_from_compartment",
        "bbox_min_z_nm_from_compartment",
        
        "bbox_max_x_nm_from_compartment",
        "bbox_max_y_nm_from_compartment",
        "bbox_max_z_nm_from_compartment",
    ),
    verbose = True
    ):

    """
    Purpose: To generate the size 
    functions for all compartments: spine,head,neck,

    Pseudocode: 
    1) Iterate through all compartments
    2) print out the formatted function
    """

    function_names = []
    function_declaration_str = ""
    for comp in compartments:
        function_declaration_str += (f"#--- Mesh Attribute Functions for {comp} -----\n")
        for st in stats_functions: 
            func_name = f"{comp}_{st.replace('_from_compartment','')}"
            function_names.append(func_name)
            function_declaration_str += (
            f"@property\ndef {func_name}(self,**kwargs):\n"
            f"    return {st}(self,compartment = '{comp}',**kwargs)\n\n"
            )

    if verbose:
        print(f"{function_declaration_str}")
    return function_names

def spine_compartment_mesh_functions_dict(spine_obj):
    """
    Purpose: To compute the statistics for a spine obj
    
    spu.spine_compartment_mesh_functions_dict(spine_obj)
    Output: 
    {'spine_width': 262.76843376430315,
     'spine_width_80_perc': 433.6109345474601,
     'spine_area': 4458314.872099194,
     'spine_volume': 1442177510.8208666,
     'spine_skeletal_length': 4518.759601841594,
     'head_width': 385.4437561195757,
     'head_width_80_perc': 452.5942823041014,
     'head_area': 1848904.3472918314,
     'head_volume': 198657361.45833808,
     'head_skeletal_length': 1580.07824623045,
     'neck_width': 110.51452991062567,
     'neck_width_80_perc': 145.0012250999968,
     'neck_area': 1378146.4731119499,
     'neck_volume': 258763454.06805038,
     'neck_skeletal_length': 2600.629272237006
    }
    
    """
    curr_funcs = spine_compartment_mesh_functions(verbose = False,)
    globs = globals()
    locs = locals()
    return {k:eval(f"spine_obj.{k}",globs,locs) for k in curr_funcs}
    
def plot_spines_objs_with_head_neck_and_coordinates(
    spine_objs,
    branch_obj=None,
    mesh = None,
    head_color = "red",
    neck_color = "aqua",
    no_head_color = "black",
    base_coordinate_color = "pink",
    center_coordinate_color = "orange",
    mesh_alpha = 0.8,
    verbose = False
    ):
    """
    Purpose: Want to plot from a list of spines
    all of the head,necks, centroids and 
    coordinates of spines

    Pseudocode: 
    For each spine:
    a) get the head/neck and put into mesh lists
    b) Get the coordinates and mesh_center and put into scatter
    3) Plot all  with the branch mesh
    """


    if branch_obj is None:
        mesh = mesh
    else:
        mesh = branch_obj.mesh
        
    head_meshes = []
    neck_meshes = []
    no_head_meshes = []
    base_coords = []
    center_coords = []
    for s in spine_objs:
        if s.n_heads == 0:
            no_head_meshes.append(s.no_head_mesh)
        else:
            head_meshes.append(s.head_mesh)
            neck_meshes.append(s.neck_mesh)
        center_coords.append(s.mesh_center)
        if s.coordinate is None:
            continue
        base_coords.append(s.coordinate)
        
    #print(f"head_meshes = {head_meshes}")

    base_coords = np.array(base_coords).reshape(-1,3)
    center_coords = np.array(center_coords).reshape(-1,3)
    
    # concatenating the meshes
    head_meshes = [tu.combine_meshes(head_meshes)]
    neck_meshes = [tu.combine_meshes(neck_meshes)]
    no_head_meshes = [tu.combine_meshes(no_head_meshes)]
    
    if verbose:
        print(f"head_meshes = {head_meshes}")
        print(f"neck_meshes = {neck_meshes}")
        print(f"no_head_meshes = {no_head_meshes}")

    colors = (
        [head_color]*len(head_meshes) +
        [neck_color]*len(neck_meshes) + 
        [no_head_color]*len(no_head_meshes)
    )
    nviz.plot_objects(
        mesh,
        meshes = head_meshes + neck_meshes + no_head_meshes,
        meshes_colors = colors,
        scatters=[base_coords,center_coords],
        scatters_colors=[base_coordinate_color,center_coordinate_color],
        mesh_alpha=mesh_alpha,
        buffer = 0,
    )

def spine_objs_bare_minimum_filt_with_attr_from_branch_obj(
    branch_obj=None,
    soma_verts_on_limb=None,
    soma_kdtree_on_limb=None,
    plot_unfiltered_spines = False,
    plot_filtered_spines = False,
    verbose = False,
    
    #for calculatin distance of spine
    soma_center = None,
    upstream_skeletal_length = None,
    
    #for branch features
    branch_features = None,
    mesh = None,
    skeleton = None,
    **kwargs
    ):

    if mesh is None:
        mesh = branch_obj.mesh
    if skeleton is None and branch_obj is not None:
        skeleton = branch_obj.skeleton
    
    if soma_kdtree_on_limb is None and soma_verts_on_limb is not None:
        soma_kdtree_on_limb = KDTree(soma_verts_on_limb)
    sp_objs = spu.spine_objs_with_border_sk_endpoint_and_soma_filter_from_scratch_on_branch_obj(
        branch_obj=branch_obj,
        soma_kdtree = soma_kdtree_on_limb,
        verbose = verbose,
        plot = plot_unfiltered_spines,
        mesh=mesh,
        skeleton = skeleton,
    )
    
    #raise Exception("")

    sp_objs = spu.calculate_spine_attributes_for_list(
        sp_objs,
        calculate_coordinates=False,
    )

    if verbose:
        print(f"Before filtering len(sp_objs) = {len(sp_objs)}")

    #filters the spine objects
    sp_objs_filt = spu.filter_spine_objs_by_size_bare_minimum(sp_objs)
    if verbose:
        print(f"AFTER filtering len(sp_objs_filt) = {len(sp_objs_filt)}")


    #calculates the center and closest face idx
    sp_objs_filt = spu.calculate_spine_attributes_for_list(
        sp_objs_filt,
        branch_obj = branch_obj,
        calculate_coordinates=True,
        calculate_head_neck = True,
        verbose_time=False,
        soma_center=soma_center,
        upstream_skeletal_length=upstream_skeletal_length,
        mesh = mesh,
        **kwargs
    )

    if plot_filtered_spines:
        spu.plot_spines_objs_with_head_neck_and_coordinates(
            sp_objs_filt,
            branch_obj = branch_obj,
            mesh = mesh
        )

        
    return sp_objs_filt

def spine_objs_near_endpoints(
    spine_objs,
    min_dist = 4_000,
    plot = False):
    
    return spu.query_spine_objs(
        output_spine_objs,
        f"(endpoint_dist_0 < {min_dist}) or (endpoint_dist_1 < {min_dist})",
        plot=True,
    )

def id_from_compartment_index(
    compartment,
    index = 0):
    if compartment == "shaft":
        return -3
    elif compartment == "no_head":
        return -2
    elif compartment == "neck":
        return -1
    elif compartment == "head":
        return 0 + index
    else:
        raise Exception("")
        
def compartment_index_from_id(id):
    if id == -3:
        return "shaft",0
    elif id == -2:
        return "no_head",0
    elif id == -1:
        return "neck",0
    elif id >= 0:
        return "head",id
    else:
        raise Exception("")
        
        
def compartment_idx_for_mesh_face_idx_of_spine(spine_obj):
    """
    Purpose: Create a face map for that spines mesh_face_idx
    from the head,neck, and no_label 
    
    Ex: 
    spine_obj = output_spine_objs[5]
    spu.plot_head_neck(spine_obj)
    tu.split_mesh_into_face_groups(
        spine_obj.mesh,
        spu.compartment_idx_for_mesh_face_idx_of_spine(spine_obj),
        plot=True,
    )
    """
    empty_array = np.ones(len(spine_obj.mesh.faces))
    if spine_obj.n_heads == 0:
        return empty_array * id_from_compartment_index("no_head")
    else:
        curr_idx = empty_array * id_from_compartment_index("neck")
        curr_idx[spine_obj.head_face_idx] = spine_obj.head_mesh_splits_face_idx
        return curr_idx
    
def face_idx_map_from_spine_objs(
    spine_objs,
    branch_obj = None,
    mesh = None, 
    no_spine_index = -1,
    plot = False,
    ):
    """
    Purpose: from a branch
    mesh and spine objs on that branch mesh
    create an array (N,2) that maps every face to the shaft
    or a spine index and the compartment
    """
    if mesh is None:
        mesh = branch_obj.mesh


    face_map = np.ones((len(branch_obj.mesh.faces),2))*no_spine_index
    face_map[:,1] = spu.id_from_compartment_index("shaft")
    for j,s in enumerate(spine_objs):
        face_map[s.mesh_face_idx,0] = j
        face_map[s.mesh_face_idx,1] = spu.compartment_idx_for_mesh_face_idx_of_spine(s)

    if plot:
        tu.split_mesh_into_face_groups(
            branch_obj.mesh,
            face_map[:,1],
            plot = True,
        )

    return face_map

def plot_spine_objs_and_syn_from_syn_df(
    spine_objs,
    syn_df,
    spine_idxs_to_plot = None
    ):
    
    if spine_idxs_to_plot is None:
        spine_idxs = np.arange(len(spine_objs))
    else:
        spine_idxs = spine_idxs_to_plot
    spine_objs_restr = np.array(output_spine_objs)[spine_idxs]
    syn_df_restr = syn_df.query(f"spine_id in {list(spine_idxs)}")
    display(syn_df_restr[["volume","spine_id","spine_compartment"]])
    nviz.plot_objects(
        meshes = [k.mesh for k in spine_objs_restr],
        scatters=[syu.synapse_coordinates_from_synapse_df(syn_df_restr)]
    )
    

def synapse_df_with_spine_match(
    branch_obj,
    spine_objs,
    plot_face_idx_map = False,
    attributes_to_append = (
    "volume",
    "width_ray_80_perc",
    "area"
    ),

    attribute_rename_dict = dict(
        volume = "spine_volume"
    ),
    spine_id_column = "spine_id",
    spine_compartment_column ="spine_compartment",
    verbose = False,
    #filter_away_non_spine_synapses = True,
    ):
    """
    Purpose: Create a dataframe that maps all
    syanpses on the branch to the spine id and
    size of the spine id

    Pseudocode: 
    1) Creae an array the same size as # of faces
    of branch where the values are (spine_id, comp #)
    2) Use the synapse objects of branch
    """


    face_idx_map = spu.face_idx_map_from_spine_objs(
        spine_objs = spine_objs,
        branch_obj = branch_obj,
        plot = plot_face_idx_map,
    )


    syn_df = syu.synapse_df(branch_obj.synapses)
    if len(syn_df) > 0:
        # mapping synapses to spines
        syn_df[[spine_id_column,spine_compartment_column]] = face_idx_map[
            syn_df["closest_branch_face_idx"].to_numpy(),:].astype('int')

        syn_df = syn_df.query(f"{spine_compartment_column} != {spu.id_from_compartment_index('shaft')}")
        if len(syn_df) > 0:
            # appending features to spines
            if attributes_to_append is not None:
                attributes_to_append = nu.to_list(attributes_to_append)

                for att in attributes_to_append:
                    if verbose:
                        print(f"-- Working on adding {att} to synapse df")
                    syn_df[attribute_rename_dict.get(att,att)] = [getattr(spu,f"{att}_from_compartment")(
                        spine_obj = spine_objs[sp_idx],
                        compartment = comp,
                    ) for sp_idx,comp in syn_df[[spine_id_column,spine_compartment_column]].to_numpy()]


            

    if verbose:
        print(f"# of synases = {len(syn_df)}")
    
    return syn_df

def example_syn_df_spine_correlations(syn_df):
    sns.jointplot(data = syn_df,x = "spine_compartment",y = "volume")
    sns.jointplot(data = syn_df.query(f"spine_compartment >= 0"),x = "volume",y = "spine_volume")
    sns.jointplot(data = syn_df.query(f"spine_compartment >= 0"),x = "volume",y = "width_ray_80_perc")
    


                        
def spine_objs_and_synapse_df_computed_from_branch_idx(
    branch_obj=None,
    limb_obj = None,
    branch_idx = None,
    soma_verts_on_limb = None,
    soma_kdtree_on_limb = None,
    upstream_skeletal_length = None,
    plot_branch_mesh_before_spine_detection = False,
    
    #spine detection phase:
    plot_unfiltered_spines  = False,
    plot_filtered_spines = False,
    
    #branch features
    branch_features = None,
    
    verbose = False,
    verbose_computation = False,
    **kwargs
    ):
    """
    Purpose: from a branch object
    will generate spine objects and the
    synapse df of synaspes onto spines

    """
    
    if not verbose_computation:
        tqu.turn_off_tqdm()
    if limb_obj is not None and branch_idx is not None:
        upstream_skeletal_length = nst.total_upstream_skeletal_length(limb_obj,branch_idx)
        branch_obj = limb_obj[branch_idx]

    if plot_branch_mesh_before_spine_detection:
        nviz.plot_objects(
            branch_obj.mesh
        )

    if soma_kdtree_on_limb is None:
        if soma_verts_on_limb is None:
            soma_verts_on_limb =limb_obj.current_touching_soma_vertices
        soma_kdtree_on_limb = KDTree(soma_verts_on_limb)

    #raise Exception("")
    output_spine_objs = spu.spine_objs_bare_minimum_filt_with_attr_from_branch_obj(
        branch_obj,
        soma_kdtree_on_limb =soma_kdtree_on_limb,
        plot_unfiltered_spines = plot_unfiltered_spines,
        plot_filtered_spines = plot_filtered_spines,
        soma_center = np.mean(soma_verts_on_limb,axis=0),
        upstream_skeletal_length=upstream_skeletal_length,
        verbose = verbose_computation,
        branch_features=branch_features,
    )


    syn_df = spu.synapse_df_with_spine_match(
        branch_obj,
        spine_objs = output_spine_objs,
        verbose = verbose_computation,
    )

    if verbose:
        print(f"# of output spines = {len(output_spine_objs)}")
        print(f"# of spine_synapses = {len(syn_df)}")
        
    if not verbose_computation:
        tqu.turn_on_tqdm()

    return output_spine_objs,syn_df

def spine_id_from_limb_branch_spine_idx(
    limb_idx,
    branch_idx,
    spine_idx = 0):
    """
    Purpose: Defines the method used for creating the spine id
    [limb,2][branches,4][spine,4]
    """
    limb_idx = nru.limb_idx(limb_idx)
    return int(f"{limb_idx}{str(branch_idx).zfill(4)}{str(spine_idx).zfill(4)}")

def spine_id_range_from_limb_branch_idx(
    limb_idx,
    branch_idx,
    verbose = False,
    return_dict = False,
    **kwargs
    ):
    """
    Purpose: to come up with a spine id range 
    given a limb and branch

    Pseudocode: 

    """
    spine_id_min = limb_idx*10_000*10_000 + branch_idx*10_000
    spine_id_max = spine_id_min + 9999
    
    if verbose:
        print(f"spine_id_min= {spine_id_min}, spine_id_max = {spine_id_max}")
        
    if return_dict:
        return dict(spine_id_min=spine_id_min,spine_id_max = spine_id_max)
    return spine_id_min,spine_id_max

def spine_id_add_from_limb_branch_idx(limb_idx,branch_idx):
    return spine_id_from_limb_branch_spine_idx(limb_idx,branch_idx)


def spine_objs_and_synapse_df_computed_from_neuron_obj(
    neuron_obj,
    limb_branch_dict = None,
    limb_branch_dict_exclude = None,

    verbose = False,
    **kwargs
    ):


    # -- cycling through all of the branches to compute the spines
    if limb_branch_dict is None:
        limb_branch_dict = neuron_obj.limb_branch_dict

    global_time = time.time()
    soma_center = neuron_obj["S0"].mesh_center

    limb_branch_spine_info = dict()
    for limb_idx in limb_branch_dict.keys():

        if verbose:
            print(f"----Working on limb {limb_idx}")

        bu.set_branches_endpoints_upstream_downstream_idx_on_limb(neuron_obj[limb_idx])
        curr_limb = neuron_obj[limb_idx]

#         if soma_vertex_nullification:
#             soma_verts = np.concatenate([neuron_obj[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])
#             soma_kdtree = KDTree(soma_verts)
#         else:
#             soma_kdtree = None

        for branch_idx in limb_branch_dict[limb_idx]:

            if limb_branch_dict_exclude is not None:
                if limb_idx in limb_branch_dict_exclude:
                    if branch_idx in limb_branch_dict_exclude[limb_idx]:
                        if verbose:
                            print(f"Skipping because in limb_branch exclude")
                        continue


    #         if verbose:
    #             print(f"Working on limb {limb_idx} branch {branch_idx}")
            st = time.time()
            if verbose:
                    print(f"   ---- branch {branch_idx}")
            spine_objs_computed,syn_df_computed = spu.spine_objs_and_synapse_df_computed_from_branch_idx(
                limb_obj = neuron_obj[limb_idx],
                branch_idx = branch_idx,
                verbose = verbose,
                **kwargs
            )

            additive_id = spu.spine_id_add_from_limb_branch_idx(limb_idx,branch_idx)
            for j,s in enumerate(spine_objs_computed):
                s.spine_id = j + additive_id
                s.limb_idx = limb_idx
                s.branch_idx = branch_idx

            if len(syn_df_computed) > 0:
                syn_df_computed["spine_id"] = syn_df_computed["spine_id"].astype('int') + additive_id

            if limb_idx not in limb_branch_spine_info:
                limb_branch_spine_info[limb_idx] = dict()

            limb_branch_spine_info[limb_idx][branch_idx] = dict(spine_objs = spine_objs_computed,syn_df = syn_df_computed)

            if verbose:
                print(f"       -> time = {time.time() - st}")

    if verbose:
        print(f"Time for all spine computation = {time.time() - global_time}")
        
    return limb_branch_spine_info

spine_compartments = ("spine_head","spine_neck","spine_no_head","shaft")
spine_compartments_no_prefix = [k.replace('spine_','') for k in spine_compartments]
def plot_spine_face_idx_dict(
    mesh,
    spine_head_face_idx=None,
    spine_neck_face_idx=None,
    spine_no_head_face_idx=None,
    shaft_face_idx = None,
    head_color = head_color_default,
    neck_color = neck_color_default,
    no_head_color = no_head_color_default,
    shaft_color = shaft_color_default,
    mesh_alpha=0.5,
    synapse_dict = None,
    verbose = True,
    show_at_end = True,
    mesh_to_plot = None,
    compartment_meshes_dict = None,
    scatters = [],
    scatters_colors = [],
    **kwargs
    ):
    if verbose:
        print(f"")
        
    meshes = []
    meshes_colors = []
    
    if mesh_to_plot is None:
        mesh_to_plot = mesh
    for cat in spine_compartments:
        curr_val = eval(f"{cat}_face_idx")
        if curr_val is not None and len(curr_val) > 0:
            meshes.append(mesh.submesh([curr_val],append=True))
            curr_color = eval(f"{cat.replace('spine_','')}_color")
            if verbose:
                print(f"{cat} ({curr_color})")
            meshes_colors.append(curr_color)
            
    #return meshes
    curr_show_at_end = show_at_end
    if show_at_end:
        curr_show_at_end = synapse_dict is None
        
    if compartment_meshes_dict is not None:
        total_mesh = tu.combine_meshes(meshes)
        comp_meshes = []
        comp_colors = []
        for comp,comp_mesh in compartment_meshes_dict.items():
            if len(comp_mesh.faces) == 0:
                continue
            new_comp_mesh = tu.subtract_mesh(comp_mesh,total_mesh,exact_match = False)
            comp_meshes.append(new_comp_mesh)
            comp_colors.append(apu.colors_from_compartments(comp))
            
        meshes += comp_meshes
        meshes_colors += comp_colors
        
    nviz.plot_objects(
        mesh_to_plot,
        meshes = meshes,
        meshes_colors = meshes_colors,
        mesh_alpha=mesh_alpha,
        show_at_end = curr_show_at_end,
        scatters=scatters,
        scatters_colors = scatters_colors,
        **kwargs
    )
    
    if synapse_dict is not None:
        plot_spine_synapse_coords_dict(
        synapse_dict = synapse_dict,
        append_figure = True,
        show_at_end=show_at_end,
        )
    

def plot_spine_synapse_coords_dict(
    mesh=None,
    synapse_dict = None,
    spine_head_synapse_coords = None,
    spine_neck_synapse_coords=  None,
    spine_no_head_synapse_coords=  None,
    shaft_synapse_coords = None,
    head_color = head_color_default,
    neck_color = neck_color_default,
    no_head_color = no_head_color_default,
    shaft_color = shaft_color_default,
    verbose = False,
    scatter_size = 0.08,
    **kwargs
    ):
    if synapse_dict is None:
        synapse_dict= dict()
        
    default_val = np.array([]).reshape(-1,3)
        
    scatters = []
    scatters_colors = []
    for cat in spine_compartments:
        curr_name = f"{cat}_synapse_coords"
        curr_val = eval(curr_name)
        if curr_val is None:
            curr_val = synapse_dict.get(curr_name,None)
                
        if curr_val is not None and len(curr_val) > 0:
            scatters.append(curr_val)
            curr_color = eval(f"{cat.replace('spine_','')}_color")
            if verbose:
                print(f"{cat} ({curr_color})")
            scatters_colors.append(curr_color)
                
    #return meshes
    nviz.plot_objects(
        mesh,
        scatters = scatters,
        scatters_colors = scatters_colors,
        scatter_size=scatter_size,
        **kwargs
    )
    
def spine_compartments_face_idx_for_neuron_mesh(
    limb_branch_spine_dict,
    limb_branch_face_dict,
    compartments = ("head","neck","no_head"),
    compute_shaft_face_idx = True,
    mesh=None,
    n_faces = None,
    plot = False,
    mesh_alpha=1,
    add_n_faces = True,
    ):
    """
    Purpose: To compile the face_idx of a compartment from
    spine objs knowing the branch_face_idx corresponding
    to the larger mesh

    Example: 
    spine_compartment_masks = spu.spine_compartments_face_idx_for_neuron_mesh(
        limb_branch_face_dict=limb_branch_face_dict,
        limb_branch_spine_dict = limb_branch_spine_info_ret,
        mesh = decimated_mesh,
        plot = True,
        mesh_alpha = 1
    )
    """

    comp_face_idx_dict = {k:[] for k in compartments}
    for limb_idx,branch_info in limb_branch_spine_dict.items():
        for b_idx,b_spine_dict in branch_info.items():
            branch_face_idx = limb_branch_face_dict[limb_idx][b_idx]
            for comp in compartments:
                branch_comp_face_idx = [k.mesh_face_idx[getattr(k,f"{comp}_face_idx").astype('int')]
                                        if len(getattr(k,f"{comp}_face_idx")) > 0 else []
                                        for k in b_spine_dict["spine_objs"]]
                if len(branch_comp_face_idx) > 0:
                    branch_comp_face_idx = np.hstack(branch_comp_face_idx)

                branch_comp_face_idx = np.array(branch_comp_face_idx)
                comp_face_idx_dict[comp].append(branch_face_idx[branch_comp_face_idx.astype('int')])

    comp_face_idx_dict_final = dict()
    for comp in compartments:
        curr_arrays = np.array(comp_face_idx_dict[comp])
        if len(curr_arrays) > 0:
            curr_arrays = np.hstack(curr_arrays)
        comp_face_idx_dict_final[f"spine_{comp}_face_idx"] = curr_arrays

    if compute_shaft_face_idx and mesh is not None or n_faces is not None:
        if n_faces is None:
            n_faces = len(mesh.faces)
        all_idx = np.hstack(list(comp_face_idx_dict_final.values()))
        comp_face_idx_dict_final[f"shaft_face_idx"] = np.delete(np.arange(n_faces),np.array(all_idx).astype('int'))

    if add_n_faces:
        comp_face_idx_dict_final.update({f"{k}_n_faces":len(v) for k,v in comp_face_idx_dict_final.items()})
    
    if plot:
        plot_spine_face_idx_dict(mesh,**comp_face_idx_dict_final,mesh_alpha=mesh_alpha)

    return comp_face_idx_dict_final

def spine_objs_and_synapse_df_total_from_limb_branch_spine_dict(
    limb_branch_spine_dict,
    verbose = False,
    ):
    """
    Purpose: To extract all spine objects and 
    synapse dfs and concatenate from
    a limb_branch_spine_dict
    """

    spine_objs_total = []
    syn_df_total = []
    for limb_idx,branch_data in limb_branch_spine_dict.items():
        for b_idx,spine_data in branch_data.items():
            spine_objs = spine_data["spine_objs"]
            syn_df = spine_data["syn_df"]

            if len(syn_df) > 0:
                syn_df_total.append(syn_df)
            spine_objs_total += spine_objs

    if len(syn_df_total) > 0:
        syn_df_total = pu.concat(syn_df_total,axis = 0).reset_index(drop=True)
    else:
        syn_df_total = pu.empty_df()
        
    if verbose:
        print(f"Total # of spines = {len(spine_objs_total)}")
        print(f"Total # of synapses on spines = {len(syn_df_total)}")

    return spine_objs_total,syn_df_total

def features_to_export_for_db():
    total_features = [
        "spine_id",

        #coordinates
        "base_coordinate_x_nm",
        "base_coordinate_y_nm",
        "base_coordinate_z_nm",
        "mesh_center_x_nm",
        "mesh_center_y_nm",
        "mesh_center_z_nm",
        
        #head features
        "n_heads",
        "shaft_border_area",
    
        
        #distances from things
        "downstream_dist",
        "upstream_dist",
        "soma_distance_euclidean",
        "soma_distance",
        
        #branch features
        "compartment",
        "branch_width_overall",
        "branch_skeletal_length",
        "branch_width_at_base",
        
    ]

    #spine size features (and repeat for head/neck)
    size_features = [
        "area",
        "n_faces",
        "volume",
        "skeletal_length",
        "width_ray",
        "width_ray_80_perc",
        "bbox_oriented_side_max",
        "bbox_oriented_side_middle",
        "bbox_oriented_side_min",
#         "bbox_min_x_nm",
#         "bbox_min_y_nm",
#         "bbox_min_z_nm",
#         "bbox_max_x_nm",
#         "bbox_max_y_nm",
#         "bbox_max_z_nm",
    ]
    
    for prefix in ["spine","head","neck"]:
        total_features+= [f"{prefix}_{k}" for k in size_features]

    return total_features

def spine_df_for_db_from_spine_objs(
    spine_objs,
    verbose = False,
    verbose_loop = False,):

    return spu.df_from_spine_objs(
        spine_objs,
        attributes = features_to_export_for_db(),
        verbose = verbose,
        verbose_loop = verbose_loop,
    )
def example_plot_coordinates_from_spine_df_idx(idx,spine_objs):
    spine_obj = spine_objs[idx]
    nviz.plot_objects(
        meshes = [spine_obj.head_mesh,spine_obj.neck_mesh],
        meshes_colors=["red","green"],
        scatters=[np.array([
            spine_df[[f"{comp}_bbox_min_{ax}_nm" for ax in ['x','y','z']]].iloc[idx,:].to_numpy(),
            spine_df[[f"{comp}_bbox_max_{ax}_nm" for ax in ['x','y','z']]].iloc[idx,:].to_numpy()])
                 for comp in ['head','neck']] + [
            spine_df[[f"{comp}_{ax}_nm" for ax in ['x','y','z']]].iloc[idx].to_numpy()
        for comp in ["base_coordinate","mesh_center"]],
        scatters_colors=["red","green","black","purple"],
    )
    
def examle_plot_spines_from_spine_df_query(
    spine_df,
    spine_objs,
    ):
    spine_df_filt = spine_df.query(f"(neck_skeletal_length > 6000) and (n_heads < 2)")

    spu.plot_spine_objs(
        np.array(spine_objs)[spine_df_filt.index.to_numpy().astype('int')],
        mesh = decimated_mesh,
        spine_color = "red",
    )
    
# ---- function ofr extracting new spine_objs from neuron_obj ---
def limb_branch_dict_to_search_for_spines(
    neuron_obj,
    query = None,
    plot = False,
    verbose = False,
    ):
    
    if query is None:
        query = spu.query_global
    
    functions_list = [
        "median_mesh_center",
        "n_faces_branch"
    ]

    
    limb_branch_dict = ns.query_neuron(neuron_obj,
           functions_list=functions_list,
           query=query,
            plot_limb_branch_dict=plot
    )
    
    if verbose:
        print(f"limb_branch_dict_spine_candidates = \n{limb_branch_dict}")

    return limb_branch_dict


def spine_and_syn_df_computed_from_neuron_obj(
    neuron_obj,
    limb_branch_dict = None,
    restrict_to_proofread_filtered_branches = True,
    decimated_mesh = None,
    proofread_faces = None,
    return_spine_compartmenets_face_idxs = True,
    add_neuron_compartments = True,
    compartment_faces_dict = None,
    verbose = False):
    """
    Purpose: To get the spine df and synase
    df corresponding to these spines from segment id
    and split index,

    dataframes can then be written to datajoint

    Pseudocode: 
    1) Download the neuron object
    2) Get the typical limb branch to search 
    3) Get the limb branch dict for faces after proofreading
    and the mesh face idx for the branches
    4) Generate the new spine objects and corresponding synapse df
    5) Generate the spine statistics dataframe from spine objects
    6) If the spine_face_idx is requested:
    - generate the spine face idxs from the limb_branch_face_dict
    and spine objs
    """

    st = time.time()

    if limb_branch_dict is None:
        limb_branch_dict = spu.limb_branch_dict_to_search_for_spines(neuron_obj)

    # 3) Get the limb branch dict for faces after proofreading
    # and the mesh face idx for the branches
    if restrict_to_proofread_filtered_branches or return_spine_compartmenets_face_idxs:
        limb_branch_face_dict = nru.limb_branch_face_idx_dict_from_neuron_obj_overlap_with_face_idx_on_reference_mesh(
            neuron_obj,
            faces_idx = proofread_faces,
            mesh_reference = decimated_mesh,
            limb_branch_dict = limb_branch_dict,
            verbose = False
        )

        curr_limb_branch_dict = limb_branch_face_dict
        
        if add_neuron_compartments:
            if compartment_faces_dict is None:
                raise Exception("compartment_faces_dict is None and requested compartments")
            from neurd import apical_utils as apu
            limb_branch_compartment_dict = apu.limb_branch_compartment_dict_from_limb_branch_face_and_compartment_faces_dict(
                limb_branch_face_dict,
                compartment_faces_dict=compartment_faces_dict,
                verbose = False,
            )
        
    else:
        curr_limb_branch_dict = limb_branch_dict
        
    #

    #4) Generate the new spine objects and corresponding synapse df
    limb_branch_spine_info_ret = spu.spine_objs_and_synapse_df_computed_from_neuron_obj(
        neuron_obj,
        limb_branch_dict= curr_limb_branch_dict,
        verbose = verbose,

    )
    
    #adding the neuron compartment labels
    if add_neuron_compartments:
        for limb_idx,branch_info in limb_branch_spine_info_ret.items():
            for branch_idx,spines_syn in branch_info.items():
                branch_spines = spines_syn["spine_objs"]
                for sp_obj in branch_spines:
                    sp_obj.compartment = limb_branch_compartment_dict[limb_idx][branch_idx]

        
                            

    spine_objs_total,syn_df_total = spu.spine_objs_and_synapse_df_total_from_limb_branch_spine_dict(
        limb_branch_spine_info_ret,
        verbose = verbose
    )
    
    

    spine_df = spu.spine_df_for_db_from_spine_objs(
        spine_objs_total,
        verbose = verbose,
        verbose_loop = False,
    )


    if return_spine_compartmenets_face_idxs:
        spine_compartment_masks = spu.spine_compartments_face_idx_for_neuron_mesh(
            limb_branch_face_dict=limb_branch_face_dict,
            limb_branch_spine_dict = limb_branch_spine_info_ret,
            mesh = decimated_mesh,
            plot = False,
            mesh_alpha = 1
        )

        return_value = spine_df,syn_df_total,spine_compartment_masks
    else:
        return_value = spine_df,syn_df_total

    if verbose:
        print(f"Total time for spine_obj and syn_df from neuron obj = {time.time() - st}")
    
    return return_value


def plot_spine_coordinates_from_spine_df(
    mesh,
    spine_df,
    coordinate_types = (
        "base_coordinate",
        "mesh_center",
        "head_bbox_max",
        "head_bbox_min",
        "neck_bbox_max",
        "neck_bbox_min",
    )
    ):
    
    """
    Ex: 
    scats = spu.plot_spine_coordinates_from_spine_df(
        mesh = neuron_obj.mesh,
        spine_df=spine_df,
    )
    """

    scatters = [pu.coordinates_from_df(
        spine_df,k,filter_away_nans = True ) for k in 
        coordinate_types
    ]
    
    
    #return scatters
    
    nviz.plot_objects(
        mesh,
        scatters=scatters,
        scatters_colors=[
            base_color_default,
            center_color_default,
            head_color_default,
            head_color_default,
            neck_color_default,
            neck_color_default,
        ]
    )
    
def example_plot_small_volume_spines_from_spine_df(
    neuron_obj,
    spine_df):

    spine_df.query(f"spine_volume < {np.percentile(spine_df.spine_volume.to_numpy(),95)}").spine_volume.hist(bins=100)

    curr_df = spine_df.query(f"spine_volume < 10000000")
    
    scats = spu.plot_spine_coordinates_from_spine_df(
        mesh = neuron_obj.mesh,
        spine_df=curr_df,
    )
    
def synapse_spine_match_df_filtering(
    syn_df):
    """
    Purpose: To map columns and 
    filter away columns of synapse df
    for database write
    """

    if len(syn_df) == 0:
        return syn_df
    
    syn_df_renamed = pu.map_column_with_dict_slow(
        syn_df,
        column = "spine_compartment",
        dict_map = {-3:"shaft",-2:"no_head",-1:'neck'},
        default_value = "head",
        verbose = True,
        in_place=False
    )
    
    syn_df_renamed = syn_df_renamed[["syn_id","spine_id","spine_compartment","spine_volume","area","width_ray_80_perc"]]
    syn_df_renamed = pu.rename_columns(
        syn_df_renamed,
        dict(syn_id = "synapse_id",
             area = "spine_area",
            width_ray_80_perc = "spine_width_ray_80_perc")
    )

    return syn_df_renamed

area_divisor = 1_000_000
scale_dict_default = dict(
    volume = 1/1_000_000_000,
    area = 1/1_000_000
)
def scale_stats_df(
    df,
    scale_dict = scale_dict_default,
    in_place = False,
    ):
    """
    Purpose: Want to scale certain columns
    of dataframe by divisors if have keyword
    """
    if not in_place:
        df = df.copy()
    
    
    for k,scale in scale_dict.items():
        for col in df.columns:
            if k in col:
                df.loc[~df[col].isna(),col] = df.loc[~df[col].isna(),col].astype('int')*scale
                
                
    return df

def filter_and_scale_spine_syn_df(spine_df,syn_df):
    syn_df_filt = spu.synapse_spine_match_df_filtering(syn_df)
    syn_df_scaled = spu.scale_stats_df(syn_df_filt)
    spine_df_scaled = spu.scale_stats_df(spine_df)
    return spine_df_scaled,syn_df_scaled

def spine_counts_from_spine_df(spine_df):
    if len(spine_df) == 0:
        return dict(
            n_spine_one_head = 0,
            n_spine_no_head = 0,
            n_spine_multi_head = 0
        )
    
    return dict(
            n_spine_one_head = len(spine_df.query(f"n_heads == 1")),
            n_spine_no_head = len(spine_df.query(f"n_heads == 0")),
            n_spine_multi_head = len(spine_df.query(f"n_heads > 1")),
    )

def spine_features_to_print():
    features = ["n_heads","shaft_border_area"]
    for comp in ["spine","head","neck"]:
        for k in ["area","n_faces","volume","skeletal_length","width_ray","width_ray_80_perc",
                 "bbox_oriented_side_max","bbox_oriented_side_middle","bbox_oriented_side_min"]:
            features.append(f"{comp}_{k}")

    return features

def seg_split_spine(
    df= None,
    segment_id= None,
    split_index= None,
    spine_id = None,
    return_dicts = True,
    ):
    """
    Purpose: Get segment,split_index and spine_ids
    
    Ex: 
    spu.seg_split_spine(spine_df_trans_umap)

    spu.seg_split_spine(
        segment_id = 864691135730167737,
        split_index = 0,
        spine_id = [0,11],
        df = None,
    )
    """
    if df is not None:
        if "segment_id" in df.columns:
            segment_id = df.segment_id.to_list()
        if "split_index" in df.columns:
            split_index = df.split_index.to_list()
        if "spine_id" in df.columns:
            spine_id = df.spine_id.to_list()

    segment_id = nu.to_list(segment_id)
    split_index = nu.to_list(split_index)
    spine_id=nu.to_list(spine_id)

    if len(segment_id) != len(spine_id):
        if len(segment_id) == 1:
            segment_id = segment_id*len(spine_id)
        else:
            raise Exception("")

    if len(split_index) != len(spine_id):
        if len(split_index) == 1:
            split_index = split_index*len(spine_id)
        else:
            raise Exception("")

    if return_dicts:
        return [dict(segment_id=s,split_index = sp,spine_id = spi)
               for s,sp,spi in zip(segment_id,split_index,spine_id)]
    return segment_id,split_index,spine_id

seg_split_spine_from_df = seg_split_spine

def area_of_border_verts(spine_obj,default_value = 0):
    if spine_obj.coordinate_border_verts is None:
        return default_value
    return tu.area_of_vertex_boundary(
        spine_obj.mesh,
        spine_obj.coordinate_border_verts
    )

# ----- new shaft filtering function --------------
def restrict_meshes_to_shaft_meshes_without_coordinates(
    meshes,
    close_hole_area_top_2_mean_max = None, #110_000,
    mesh_volume_max = None,#0.3e9,
    #n_faces_max = 500,
    n_faces_min = None,
    verbose = False,
    plot = False,
    return_idx = True,
    return_all_shaft_if_none = True,
    ):
    """
    Purpose: To restrict a list of meshes
    to those with a high probabilty of being
    a shaft mesh
    """
    if close_hole_area_top_2_mean_max is None:
        close_hole_area_top_2_mean_max = shaft_close_hole_area_top_2_mean_max_global
    if mesh_volume_max is None:
        mesh_volume_max = shaft_mesh_volume_max_global
    if n_faces_min is None:
        n_faces_min = shaft_mesh_n_faces_min_global
    
    query = [
        f"(close_hole_area_top_2_mean > {close_hole_area_top_2_mean_max}) or (mesh_volume > {mesh_volume_max})",
        #f"(close_hole_area_top_2_mean > {close_hole_area_top_2_mean_max}) or (n_faces > {n_faces_max})",
        f"(n_faces > {n_faces_min})"
    ]

    shaft_meshes_idx= tu.query_meshes_from_stats(
        meshes,
        functions = [
            "close_hole_area_top_2_mean",
            "n_faces",
            "mesh_volume"
        ],
        query = query,
        verbose = verbose,
        plot = plot,
        return_idx = return_idx
    )
    
    if len(shaft_meshes_idx) == 0 and return_all_shaft_if_none:
        shaft_meshes_idx = np.arange(len(meshes))

    return shaft_meshes_idx

def split_mesh_into_spines_shaft(
    current_mesh,
    segment_name="",
    clusters=None,
    smoothness=None,
    cgal_folder = Path("./cgal_temp"),
    delete_temp_file=True,
    shaft_threshold = None,
     return_sdf = True,
    print_flag = False,
    plot_segmentation = False,
    plot_shaft = False,
    plot_shaft_buffer = 0,
    **kwargs
    ):
    if clusters is None:
        clusters = clusters_threshold_global
    
    if smoothness is None:
        smoothness = smoothness_threshold_global
        
    if shaft_threshold is None:
        shaft_threshold = shaft_threshold_global
    
    

    #print(f"plot_segmentation= {plot_segmentation}")
    cgal_data,cgal_sdf_data = tu.mesh_segmentation(current_mesh,
                                                  cgal_folder=cgal_folder,
                                                   clusters=clusters,
                                                   smoothness=smoothness,
                                                   return_sdf=True,
                                                   delete_temp_files=delete_temp_file,
                                                   return_meshes=False,
                                                   return_ordered_by_size=False,
                                                   plot_segmentation = plot_segmentation,
                                                  )
    
    #get a look at how many groups and what distribution:
    from collections import Counter
    if print_flag:
        print(f"Counter of data = {Counter(cgal_data)}")

    #gets the meshes that are split using the cgal labels
    split_meshes,split_meshes_idx = tu.split_mesh_into_face_groups(current_mesh,cgal_data,return_idx=True,
                                   check_connect_comp = False)
    
    
    
    split_meshes,split_meshes_idx
    
    
    if len(split_meshes.keys()) <= 1:
        print("There was only one mesh found from the spine process and mesh split, returning empty array")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
        
    
    
#     #Applying a length threshold to get all other possible shaft meshes
#     for spine_id,spine_mesh in split_meshes.items():
#         if len(spine_mesh.faces) < shaft_threshold:
#             spine_meshes.append(spine_mesh)
#             spine_meshes_idx.append(split_meshes_idx[spine_id])
#         else:
#             shaft_meshes.append(spine_mesh)
#             shaft_meshes_idx.append(split_meshes_idx[spine_id])

    meshes = np.array(list(split_meshes.values()))
    meshes_idx = np.array(list(split_meshes.keys()))
    
    #return meshes
    
    sh_idx = spu.restrict_meshes_to_shaft_meshes_without_coordinates(
        meshes,
        **kwargs
    )
    
    shaft_meshes = [meshes[k] for k in sh_idx]
    shaft_meshes_idx = [split_meshes_idx[meshes_idx[k]] for k in sh_idx]
    
    sp_idx = np.delete(np.arange(len(meshes)),sh_idx).astype('int')
    spine_meshes = [meshes[k] for k in sp_idx]
    spine_meshes_idx = [split_meshes_idx[meshes_idx[k]] for k in sp_idx]
    
 
    if len(shaft_meshes) == 0:
        if print_flag:
            print("No shaft meshes detected")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
 
    if len(spine_meshes) == 0:
        if print_flag:
            print("No spine meshes detected")
            
    if plot_shaft:
        print(f"--- initial shaft meshes before graph fixing")
        nviz.plot_objects(
            current_mesh,
            meshes = shaft_meshes,
            meshes_colors = "red",
            buffer = plot_shaft_buffer,
        )
    if return_sdf:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx,cgal_sdf_data
    else:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx

    
def synapse_attribute_dict_from_synapse_df(
    df,
    attribute = "synapse_coords",
    suffix = None,
    verbose = False,):
    """
    Purpose: To extract the coordinates of 
    all the spine categories from a synapse_spine_df

    Psuedocode: 
    Iterate through all of the spine categories
    1) Restrict the dataframe to just categories
    2) Extract coordinates
    3) Put in dictionary
    """
    if suffix is None:
        suffix = f"_{attribute}"
    coordinate_dict = dict()
    for cat in spu.spine_compartments:
        curr_cat = cat.replace(f"spine_","")
        if len(df) > 0:
            curr_df = df.query(f"spine_compartment=='{curr_cat}'")
        else:
            curr_df = []
            
        if attribute == "synapse_coords":
            if len(curr_df) == 0:
                coords = np.array([]).reshape(-1,3)
            else:
                coords = pu.coordinates_from_df(curr_df,name = "synapse")
        elif attribute == "synapse_id":
            if len(curr_df) == 0:
                coords = np.array([]).reshape(-1)
            else:
                coords = curr_df["synapse_id"].to_numpy()
        else:
            raise Exception("")
        
        coordinate_dict[f"{cat}{suffix}"] = coords
    
    if verbose:
        for k,v in coordinate_dict.items():
            print(f"{k}:{len(v)}")
    
    return coordinate_dict

def synapse_coords_from_synapse_df(
    df,
    suffix = None,
    verbose = False,
    return_dict = True,):
    
    return_d =  synapse_attribute_dict_from_synapse_df(
        df,
        attribute = "synapse_coords",
        suffix = suffix,
        verbose = verbose,
    )
    
    if return_dict:
        return return_d
    else:
        return np.vstack(list(return_d.values()))

def synapse_ids_from_synapse_df(
    df,
    suffix = None,
    verbose = False,
    return_dict = True,):
    
    return_d =  synapse_attribute_dict_from_synapse_df(
        df,
        attribute = "synapse_id",
        suffix = suffix,
        verbose = verbose,
    )
    
    if return_dict:
        return return_d
    else:
        return np.hstack(list(return_d.values())).astype('int')
    
    

def plot_spine_embeddings_kde(
    df,
    embeddings = ["umap_0","umap_1"],
    rotation = 0,
    hue = "e_i_predicted",
    excitatory_color = mu.seaborn_orange,
    inhibitory_color = mu.seaborn_blue,
    thresh = 0.2,#0.5,
    levels=5,
    alpha = 0.5,
    ax = None,
    ):
    
    if ax is None:
        figsize = (20,14)
        fig, ax = plt.subplots(1,1,figsize=figsize)

    df_to_plot = df.copy()
    rotation_embeddings = [f"{k}_rotated" for k in embeddings]
    df_to_plot[rotation_embeddings] = (
        (nu.rotation_matrix(rotation)@(df_to_plot[embeddings].to_numpy()).T).T)

    ax = sns.kdeplot(
    #     data = pu.randomly_sample_classes_from_df(
    #         spine_df_trans_umap,
    #         column=hue,
    #         n_samples=50,
    #         seed = 1000,
    #     ),
        data = df_to_plot,
        x = rotation_embeddings[0],
        y = rotation_embeddings[1],
        cbar=False,
        palette=dict(
            inhibitory = inhibitory_color,
            excitatory = excitatory_color,
        ),
        hue = hue,
        thresh = thresh,#0.5,
        levels=levels,
        legend = True,
        fill = True,
        alpha = alpha,
        ax = ax,
        linestyles=":",
    )
    
    mu.set_axes_font_size(ax,50)
    mu.set_axes_tick_font_size(ax,30)
    #mu.set_legend_size(ax,100)

    return mu.set_axes_outside_seaborn(ax)


def spine_compartment_synapses(df,compartment):
    compartment= nu.to_list(compartment)
    return df.query(f"spine_compartment in {list(compartment)}")

def n_spine_compartment_synapses(df,compartment):
    return len(spine_compartment_synapses(df,compartment))
def shaft_synapses(df):
    return spine_compartment_synapses(df,compartment="shaft")
def n_shaft_synapses(df):
    return len(shaft_synapses(df))

def set_shaft_synapses_from_spine_query(
    df,
    query,
    verbose = False,
    in_place = False,
    ):
    """
    Purpose: Receiving a synapse table
    with spine information associated with it
    and filters of which spines to actually keep,
    will flip the current spine compartment
    label of the synapses

    Pseudocode: 
    1) invert the filter to get a filter for all spines
    that should not be spines
    2) Use the query to set the spine compartment as shaft
    """

    if not in_place:
        df = df.copy()


    query_str = pu.query_str_from_list(query,table_type="pandas")
    query_str = f"not ({query_str})"
    if verbose:
        print(f"shaft query = {query_str}")

    if verbose:
        print(f"Before filter # of shaft synapses = {spu.n_shaft_synapses(df)}")
    pu.set_column_subset_value_by_query(
        df,
        query = query_str,
        column = "spine_compartment",
        value = "shaft"
    )

    if verbose:
        print(f"AFTER filter # of shaft synapses = {spu.n_shaft_synapses(df)}")

    return df


# --------- exporting the stats --------------

def spine_synapse_stats_from_synapse_df(
    df,
    grouping_features = (
        "compartment",
        "spine_compartment",
        ),
    grouping_features_backup = (
        "spine_compartment",
    ),
    features = (
        'spine_area',
        'spine_n_faces',
        'spine_skeletal_length',
        'spine_volume',
        'spine_width_ray',
        'syn_spine_area',
        'syn_spine_volume',
        'syn_spine_width_ray_80_perc',
        'synapse_size',
        ),
    prefix = "syn",
    return_dict = True,
    synapse_types = ('postsyn',),
    
    ):
    """
    Purpose: To generate a dictionary/df
    parsing down the categories of a spine
    df into the averages and counts for different
    - neuron compartments
    - spine compartments

    Pseudocode: 
    - limit to only postsyn

    - For specified features
    1) groupby spine compartment and compartment
    - reduce by average
    - reduce by count
    """
    if synapse_types is not None:
        synapse_types = nu.to_list(synapse_types)
        df = df.query(f"synapse_type in {list(synapse_types)}")

    features = list(features)
    
    all_dfs = []
    for gf in [grouping_features,grouping_features_backup]:
        gf = list(gf)

        df_lite = df[gf + features]
        spine_columns = [k for k in features if "spine" in k]
        df_lite = pu.set_column_subset_value_by_query(
            df_lite,
            query = f"spine_compartment == 'shaft'",
            column = spine_columns,
            value = 0
        )


        df_stats = pu.group_df_for_count_and_average(
            df_lite,
            columns = gf,
            default_value = 0,
            return_one_row_df = True,
        )
        
        all_dfs.append(df_stats)
        
    df_stats = pu.concat(all_dfs,axis=1)

    df_stats.columns = [f"n_{k.replace('_unique_counts','')}"
                        if '_unique_counts' in k else k for k in df_stats.columns]
    if prefix is not None:
        df_stats.columns = [f"{prefix}_{k}" for k in df_stats.columns]

    column_ints = [c for c in df_stats if "_n_" in c and "face" not in c]
    if len(column_ints) > 0:
        df_stats[column_ints] = df_stats[column_ints].astype('int')
        
    if return_dict:
        df_stats = pu.df_to_dicts(df_stats)[0]

    return df_stats

spine_features_no_head = (
        'spine_area',
        'spine_n_faces',
        'spine_skeletal_length',
        'spine_volume',
        'spine_width_ray',
        )

spine_features_head_neck = (
        'head_area',
        'head_n_faces',
        'head_skeletal_length',
        'head_volume',
        'head_width_ray',
        'head_width_ray_80_perc',
        
        'neck_area',
        'neck_n_faces',
        'neck_skeletal_length',
        'neck_volume',
        'neck_width_ray',
        'neck_width_ray_80_perc',
        )

spine_features_n_syn_no_head = (
        'spine_n_no_head_syn',
        'spine_max_no_head_syn_size',
        'spine_max_no_head_sp_vol',
)

spine_features_n_syn_head_neck = (
        'spine_n_head_syn',
        'spine_n_neck_syn',
        'spine_max_head_syn_size',
        'spine_max_neck_syn_size',
       'spine_max_head_sp_vol',
       'spine_max_neck_sp_vol',
)

def spine_stats_from_spine_df(
    df,
    grouping_features = (
        "compartment",
        ),
    features_no_head = spine_features_no_head,
    features_head_neck = spine_features_head_neck,
    features_n_syn_no_head = spine_features_n_syn_no_head,
    features_n_syn_head_neck = spine_features_n_syn_head_neck,
    prefix = "sp",
    return_dict = True,
    ):
    
    """
    Purpose: to export spine statistics
    grouped by compartment
    """
    all_dfs = []
    for n_heads,pre,features,n_syn_features in zip(
            [1,0],
            ["head","no_head"],
            [features_head_neck,features_no_head,],
            [features_n_syn_head_neck,features_n_syn_no_head,],
        ):
        features = list(features) + list(n_syn_features)
        grouping_features = list(grouping_features)

        df_lite = df.query(f"n_heads == {n_heads}").reset_index(drop=True)
        if len(df_lite) == 0:
            continue
        
        df_lite = df_lite[grouping_features + features]
   
        df_stats = pu.group_df_for_count_and_average(
            df_lite,
            columns = grouping_features,
            default_value = 0,
            return_one_row_df = True,
        )

        df_stats.columns = [
            f"n_{k.replace('_unique_counts','')}"
                 if '_unique_counts' in k else k for k in df_stats.columns]

        if pre is not None:
            df_stats.columns = [f"{pre}_{k}" for k in df_stats.columns]
            
        all_dfs.append(df_stats)
        
        # ---- want to calculate the stats without aggregating across columns ----
        df_feat = df_lite[features].mean().to_frame().T
        df_feat[f"{pre}_number"] = len(df_lite)
        all_dfs.append(df_feat)

    
    if len(all_dfs) == 0:
        if return_dict:
            return {}
        else:
            return pu.empty_df()
    
    df_stats = pu.concat(all_dfs,axis = 1)

    if prefix is not None:
        df_stats.columns = [f"{prefix}_{k}" for k in df_stats.columns]

    column_ints = [c for c in df_stats if "_n_" in c and "face" not in c
                  and "syn" not in c]
    if len(column_ints) > 0:
        df_stats[column_ints] = df_stats[column_ints].astype('int')
        
    if return_dict:
        df_stats = pu.df_to_dicts(df_stats)[0]

    return df_stats

def add_spine_densities_to_spine_stats_df(
    df,
    head_types = ["no_head","head"],
    skeletal_length_divisor = 1000,
    compartments = None,
    in_place = False,
    ):
    """
    Purpose: Compute spine densities from
    spine_stats_df
    """
    if not in_place:
        df = df.copy()
        
    if compartments is None:
        compartments = apu.dendrite_compartment_labels()
        
    for comp in compartments:
        df[f"{comp}_spine_density"] = (
            df[[f"sp_{k}_n_{comp}" for k in head_types]].sum(axis=1)
            / (df[f"{comp}_skeletal_length"]/skeletal_length_divisor)
        )

    df["total_spine_density"] = (
        df[list(np.concatenate(
            [[f"sp_{k}_n_{comp}" for k in head_types] for comp in compartments]))].sum(axis=1)
        / (df[f"dendrite_skeletal_length"]/skeletal_length_divisor)
    )

    return df

def add_synapse_densities_to_spine_synapse_stats_df(
    df,
    in_place = True,
    compartments = None,
    spine_compartments = None,
    skeletal_length_divisor = 1000,
    eta = 0.000001,
    return_features = False,
    max_value_to_set_to_zero = 100000,
    ):
    """
    Purpose: Want to compute the synapse densities
    """

    if not in_place:
        df = df.copy()


    forgotten_labels = [
        "syn_n_apical_shaft_neck",
        "syn_n_apical_tuft_neck",]

    features_generated = []

    for l in forgotten_labels:
        df[l] = 0

    if compartments is None:
        compartments = apu.dendrite_compartment_labels()

    if spine_compartments is None:
        spine_compartments = spu.spine_compartments_no_prefix

    for sp in spine_compartments:
        df[f"{sp}_synapse_density"] = df[[f"syn_n_{k}_{sp}" for k in compartments]].sum(axis = 1)/(
            df[f"dendrite_skeletal_length"]/skeletal_length_divisor + eta)
        features_generated.append(f"{sp}_synapse_density")

    df[f"spine_synapse_density"] = df[[f"{sp}_synapse_density" for sp in ["head","neck","no_head"]]].sum(axis = 1)
    features_generated.append(f"spine_synapse_density")
    df[f"synapse_density"] = df[[f"{sp}_synapse_density" for sp in ["head","neck","no_head","shaft"]]].sum(axis = 1)
    features_generated.append(f"synapse_density")

    for comp in compartments:
        df[f"{comp}_synapse_density"] = df[[f"syn_n_{comp}_{k}" for k in spine_compartments]].sum(axis = 1)/(
            df[f"{comp}_skeletal_length"]/skeletal_length_divisor + eta)
        features_generated.append(f"{comp}_synapse_density")
        df[f"{comp}_spine_synapse_density"] = df[[f"syn_n_{comp}_{k}" for k in ["head","neck","no_head"]]].sum(axis = 1)/(
            df[f"{comp}_skeletal_length"]/skeletal_length_divisor + eta)
        features_generated.append(f"{comp}_spine_synapse_density")
        for sp in spine_compartments:
            df[f"{comp}_{sp}_synapse_density"] = df[f"syn_n_{comp}_{sp}"]/(df[f"{comp}_skeletal_length"]/skeletal_length_divisor + eta)
            features_generated.append(f"{comp}_{sp}_synapse_density")

    df["soma_syn_area_density"] = df["syn_n_soma_shaft"]/df["max_soma_area"]
    features_generated.append("soma_syn_area_density")
    df["soma_syn_volume_density"] = df["syn_n_soma_shaft"]/df["max_soma_volume"]
    features_generated.append("soma_syn_volume_density")
    
    df["shaft_to_spine_synapse_density_ratio"] = df["shaft_synapse_density"]/(df["spine_synapse_density"] + eta)
    features_generated.append("shaft_to_spine_synapse_density_ratio")
    
    df["shaft_syn_to_spine_syn_density_ratio"] = df["shaft_synapse_density"]/(df["spine_synapse_density"] + eta)
    
    features_generated.append("shaft_syn_to_spine_syn_density_ratio")

    
    if max_value_to_set_to_zero is not None:
        for f in features_generated:
            df = pu.set_column_subset_value_by_query(
                df,
                query = f"{f} >= {max_value_to_set_to_zero}",
                column = f,
                value = 0,
            )
            
    
            
    if return_features:
        return df,features_generated
    return df


def plot_feature_histograms_with_ct_overlay(
    df,
    features = None,
    max_value = None,):
    if features is None:
        features = [k for k in df.columns if k not in 
            ["segment_id","split_index","min_skeletal_length","cell_type"] and "gnn" not in k]
        
    spine_stats_df = df
    exc_df = spine_stats_df.query(f"gnn_cell_type_coarse == 'excitatory'")
    inh_df = spine_stats_df.query(f"gnn_cell_type_coarse == 'inhibitory'")
    for f in features: 
        print(f"----- {f} -----")
        cell_type = "both"
        if "dendrite" in f:
            cell_type = "inhibitory"
        for comp in apu.dendrite_compartment_labels():
            if "dendrite" == comp:
                continue
            if comp in f:
                cell_type = "excitatory"
                break


        if cell_type == 'excitatory':
            curr_df = exc_df
            hue_order = ctu.allen_cell_type_fine_classifier_labels_exc
        elif cell_type == "inhibitory":
            curr_df = inh_df
            hue_order = ctu.allen_cell_type_fine_classifier_labels_inh
        else:
            curr_df = spine_stats_df
            hue_order = ctu.allen_cell_type_fine_classifier_labels


        curr_df = curr_df.query(f"{f} > 0")
        if max_value is not None:
            curr_df = curr_df.query(f"{f} <= {max_value}")
            
        if len(curr_df) == 0:
            continue
        mu.histograms_overlayed(
            df = curr_df,
            column=f,
            hue = "gnn_cell_type_fine",
            density = True,
            hue_order = hue_order,
            title=f)

        plt.show()
        
def feature_summed_over_compartments(
    df,
    feature,
    final_name=None,
    in_place = True,
    compartments = None,
    head_prefix = ["head","no_head"],
    verbose = False,
    ):
    
    if compartments is None:
        compartments = apu.dendrite_compartment_labels()
        
    head_prefix = nu.to_list(head_prefix)
    compartments = nu.to_list(compartments)
        
    if not in_place:
        df = df.copy()
    
    
    columns = list(np.concatenate([[f"sp_{h}_{comp}_{feature}" for h in head_prefix] for comp in compartments]))
    weights = list(np.concatenate([[f"sp_{h}_n_{comp}" for h in head_prefix] for comp in compartments]))
    if verbose:
        print(f"columns = {columns}")
        print(f"weights = {weights}")
    #return columns
    val = np.sum((df[columns].to_numpy()*df[weights].to_numpy()),axis = 1)/(np.sum(df[weights].to_numpy(),axis=1))
    #val = (df[columns]*df[weights]).mean(axis = 1)
    
    if final_name is not None:
        df[final_name] = val
        return df
    else:
        return val

    

def plot_spine_feature_hist(
    df,
    x,
    y,
    x_multiplier = 1,
    y_multiplier = 1,
    title_fontsize = 30,
    axes_label_fontsize = 30,
    axes_tick_fontsize = 25,
    palette = None,
    hue = None,
    legend = False,
    title=None,
    xlabel = None,
    ylabel = None,
    ax = None,
    figsize = (7,10),
    percentile_upper = 99,
    verbose = False,
    print_correlation = False,
    show_at_end = False,
    plot_type = "histplot",
    kde_thresh = 0.2,
    kde_levels = 4,
    min_x = 0,
    min_y = 0,
    text_box_x = 0.95,
    text_box_y = 0.05,
    text_box_horizontalalignment = "right",
    text_box_verticalalignment = "bottom",
    text_box_alpha = 1,
    
    
    # --- plotting correlation textbox ---
    plot_correlation_box = True,
    correlation_type = "corr_pearson",
    text_box_fontsize = 20,
    
    
    # --- lim ---
    xlim = None,
    ylim = None,
    
    # --- legend arguments ---
    legend_title = None,
    legend_fontsize=None,
    ):
    
    if verbose:
        print(f"----{y} vs {x}----")
    
    edge_df_sp_filt = df.query(f"({x}>={min_x}) and ({y}>={min_y})") 
    edge_df_sp_filt = pu.filter_df_by_column_percentile(
        edge_df_sp_filt,
        columns=[x,y],
        percentile_lower=0,
        percentile_upper=percentile_upper,
    ).copy()
    
    edge_df_sp_filt[x] = edge_df_sp_filt[x].astype('float')*x_multiplier
    edge_df_sp_filt[y] = edge_df_sp_filt[y].astype('float')*y_multiplier
    
    
    corr_dict = stu.correlation_scores_all(
            df = edge_df_sp_filt,
            x = x,
            y = y,
            verbose = print_correlation,
            return_p_value=True,
    )
        

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize = figsize)
    
    if plot_type=="histplot" or "dis" in plot_type:
        ax = sns.histplot(
            data = edge_df_sp_filt,
            x = x,
            y = y,
            hue = hue,
            palette = palette,
            ax = ax,
            #kind = "reg",
            legend = legend,
        )
    elif plot_type == "jointplot":
        ax = sns.jointplot(
            data = edge_df_sp_filt,
            x = x,
            y = y,
            hue = hue,
            palette = palette,
            ax = ax,
            #kind = "reg",
            legend = legend,
        )
    elif plot_type == "scatterplot":
        ax = sns.scatterplot(
            data = edge_df_sp_filt,
            x = x,
            y = y,
            hue = hue,
            palette = palette,
            ax = ax,
            #kind = "reg",
            legend = legend,
        )
        
    elif plot_type == 'kdeplot':
        ax = sns.kdeplot(
            data = edge_df_sp_filt,#.query(f"{column}=='{k}'"),
            x = x,
            y = y,#features_to_plot[1],
            hue = hue,
            common_norm = False,
            thresh=kde_thresh,
            levels = kde_levels,
            palette=palette,
            ax = ax,
        )
        
    else:
        raise Exception("")
        
    ax = mu.ax_main(ax)
    
    if title is not None:
        ax.set_title(title,fontsize = title_fontsize)
        
    
    if xlabel is None:
        xlabel = x.replace("_"," ").title()
    if ylabel is None:
        ylabel = y.replace("_"," ").title()
    ax.set_xlabel(xlabel,fontsize = axes_label_fontsize)
    ax.set_ylabel(ylabel,fontsize = axes_label_fontsize)
    mu.set_axes_tick_font_size(ax,fontsize=axes_tick_fontsize)


    
    if plot_correlation_box:
#         curr_value = corr_dict[correlation_type]
#         corr_str = (f"Corr = {curr_value['correlation']:.3f}"
#             f"\n(P = {curr_value['pvalue']:.2f})")
#         mu.text_box_on_ax(
#             ax,
#             x = text_box_x,
#             y = text_box_y,
#             horizontalalignment = text_box_horizontalalignment,
#             verticalalignment = text_box_verticalalignment,
#             text =corr_str,
#             fontsize = text_box_fontsize,
#             alpha = text_box_alpha
#         )
        
        mu.add_correlation_text_box(
            ax,
            corr_dict = corr_dict,
            text_box_fontsize = text_box_fontsize,
            correlation_type = correlation_type,
            text_box_x = text_box_x,
            text_box_y = text_box_y,
            text_box_horizontalalignment = text_box_horizontalalignment,
            text_box_verticalalignment = text_box_verticalalignment,
            text_box_alpha = text_box_alpha,
        )
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
        
    if legend_title is not None and hue is not None:
        mu.set_legend_title(ax,"Cell Type")
    if legend_fontsize is not None and hue is not None:
        mu.set_legend_fontsizes(ax,legend_fontsize)
    
    if show_at_end:
        plt.show()

    return ax

def plot_connetion_type_head_vs_spine_size_by_conn_type_kde(
    df,
    x = "head_volume",
    y = "synapse_size_um",
    hue = "connection_type",
    ax = None,
    figsize = (7,10),
    title = f"Syn Size vs Spine Head Volume",
    title_fontsize = 30,
    xlabel = "Spine Head Volume ($\mu m^3$)",
    ylabel = f"Synapse Cleft Volume ($\mu m^3$)",
    axes_label_fontsize = 30,
    axes_tick_fontsize = 25,
    palette = None,
    kde_thresh = 0.2,
    kde_levels = 4,
    hue_options = None,
    legend_title = None,
    legend_fontsize = 20,
    ):
    
    edge_df_sp_filt = pu.filter_df_by_column_percentile(
        df,
        columns=[x,y],
        percentile_lower=0,
        percentile_upper=99.5,
    )

    edge_df_sp_filt_head = edge_df_sp_filt.query(f"spine_compartment == 'head'")
    

    edge_df_sp_filt_head[x] = edge_df_sp_filt_head[x].astype('float')
    edge_df_sp_filt_head[y] = edge_df_sp_filt_head[y].astype('float')

    
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize = figsize)
        
    if palette is None:
        from neurd import nature_paper_plotting as npp
        palette = npp.exc_inh_combination_palette
    
    if hue_options is not None:
        edge_df_sp_filt_head = edge_df_sp_filt_head.query(f"{hue} in {nu.to_list(hue_options)}")
    
    ax =  sns.kdeplot(
        data = edge_df_sp_filt_head,#.query(f"{column}=='{k}'"),
        x = x,
        y = y,#features_to_plot[1],
        hue = hue,
        common_norm = False,
        thresh=kde_thresh,
        levels = kde_levels,
        palette=palette,
        ax = ax,
    )
    
    ax.set_title(title,fontsize = title_fontsize)
    ax.set_xlabel(xlabel,fontsize = axes_label_fontsize)
    ax.set_ylabel(ylabel,fontsize = axes_label_fontsize)
    mu.set_axes_tick_font_size(ax,fontsize=axes_tick_fontsize)
    
    if legend_title is None:
        legend_title = hue.replace("_"," ").title()
    mu.set_legend_title(
        ax,legend_title
    )
    
    mu.set_legend_fontsizes(ax,legend_fontsize)
    
    return ax

def filter_spine_df_samples(
    df,
    syn_max = 4,
    sp_comp_max = 3,
    restrictions = None,
    ):
    
    if restrictions is None:
        restrictions= []
    
    restrictions += [
        "n_heads == 1",
        f"spine_n_spine_total_syn <= {syn_max}",
        f"spine_n_head_syn <= {sp_comp_max}",
        f"spine_n_neck_syn <= {sp_comp_max}",
    ]
    if f"gnn_cell_type_coarse" in df.columns:
        restrictions.append(f"gnn_cell_type_coarse == cell_type")
    
    df = pu.query_table_from_list(
        df,
        restrictions
    )

    return df

def number_of_columns(df):
    return [k for k in df.columns if "_n_" in k or k[:2] == 'n_']
def convert_number_of_columns_to_dtype(df,dtype = "int"):
    df[number_of_columns(df)] = df[number_of_columns(df)].astype(dtype)
    return df
def plot_spine_attribute_vs_category_from_spine_df_samples(
    df,
    spine_attributes = ["spine_n_head_syn","spine_n_spine_total_syn"],
    category = "cell_type",
    legend_dict_map = dict(
        spine_n_head_syn =  "Spine Head",
        spine_n_spine_total_syn = "All Spine",
    ),
    title_append = "(MICrONS)",
    title = f"Cell Type vs. Average Number\n of Syn on Spine",
    x_label = f"Average # of Synapses",
    legend_title = "Syn Type",
    source = "MICrONS",
    ylabel = "Postsyn Cell Type",
    title_fontsize = 30,
    axes_label_fontsize = 30,
    axes_tick_fontsize = 25,
    legend_fontsize = 20,
    set_legend_outside = False,
    legend_loc = "best",
    **kwargs
    ):
    """
    Purpose: To take a sampling of the spine
    and to plot the average number of synapses
    on each spine vs the cell type
    """

    total_dfs = []
    for x in spine_attributes:
        df_samp_n_head = df[[category,x]]
        df_samp_n_head["category"] = x
        df_samp_n_head = pu.rename_columns(df_samp_n_head,{x:"value"})
        total_dfs.append(df_samp_n_head)

    all_df = pu.concat(total_dfs,axis = 0).reset_index(drop=True)
    
    if legend_dict_map is not None:
        all_df = pu.map_column_with_dict(all_df,column = "category",dict_map = legend_dict_map,
                                            use_default_value = False)

    ax = sns.barplot(
        data = all_df,
        x = "value",
        y = category,
        hue = "category"
    )

    if title_append is not None:
        title= f"{title} {title_append}"

    ax.set_title(
        title,fontsize = title_fontsize
    )
    #ax.set_xlabel("Spine Head Volume ($\mu m^3$)",fontsize = axes_label_fontsize)
    ax.set_xlabel(f"{x_label}",fontsize = axes_label_fontsize)
    ax.set_ylabel(f"{ylabel}",fontsize = axes_label_fontsize)
    mu.set_axes_tick_font_size(
        ax,fontsize=axes_tick_fontsize,
        y_rotation=0,
        x_rotation=0,
        y_tick_alignment="center"
    )

    if set_legend_outside:
        ax = mu.set_legend_outside_seaborn(ax)
    else:
        ax = mu.set_legend_location_seaborn(ax,legend_loc)
#     if legend_dict_map is not None:
#         print(f"Trying to set labels")
#         ax = mu.set_legend_labels_with_dict(
#             ax,
#             legend_dict_map
#         )
    
    mu.set_legend_size(ax,legend_fontsize)
    mu.set_legend_title(ax,legend_title)
    

    
    return ax
# ----------------- Parameters ------------------------

global_parameters_dict_default_spine_identification = dict(
    query="median_mesh_center > 115 and n_faces_branch>100",#previous used median_mesh_center > 140
    calculate_spine_volume=True,

    clusters_threshold=5,#3,#2,
    smoothness_threshold=0.08,#0.12,#0.08,
    shaft_close_hole_area_top_2_mean_max = 110_000,
    shaft_mesh_volume_max = 0.3e9,
    shaft_mesh_n_faces_min = 10,
    shaft_threshold=300,
    
    # --- the new bare minimum thresholds ----
    spine_n_face_threshold_bare_min = 6,
    spine_sk_length_threshold_bare_min = 306.6,
    filter_by_volume_threshold_bare_min = 900496.186,
    bbox_oriented_side_max_min_bare_min = 300,
    spine_volume_to_spine_area_min_bare_min = 0.008,
    sdf_mean_min_bare_min = 0,

    spine_n_face_threshold=25,
    spine_sk_length_threshold = 1_000,
    filter_by_bounding_box_longest_side_length=True,
    side_length_threshold = 5000,
    filter_out_border_spines=False, #this seemed to cause a lot of misses
    skeleton_endpoint_nullification=True,
    skeleton_endpoint_nullification_distance = 2000,
    soma_vertex_nullification = True,
    border_percentage_threshold=0.3,
    check_spine_border_perc=0.4,

    #-------1/20 Addition --------
    filter_by_volume = True,
    filter_by_volume_threshold = 19_835_293, #calculated from experiments   
    
)

global_parameters_dict_default_head_neck_shaft = dict(
    head_smoothness = 0.09,#0.15,
    head_ray_trace_min = 240,
    head_face_min = 10,
    only_allow_one_connected_component_neck = False,
    
)

global_parameters_dict_default = gu.merge_dicts([
    global_parameters_dict_default_spine_identification,
    global_parameters_dict_default_head_neck_shaft
    
])

attributes_dict_default = dict(
)    

global_parameters_dict_microns = {}
attributes_dict_microns = {}

global_parameters_dict_h01_spine_identification = dict(
    spine_n_face_threshold=15,
    spine_sk_length_threshold = 1_000,
    filter_by_volume_threshold = 50_000_000,#19835293, #calculated from experiments   
    
    clusters_threshold=5,
    smoothness_threshold=0.08,#0.12,#0.08,
    shaft_close_hole_area_top_2_mean_max = 260_000,
    shaft_mesh_volume_max = 0.6e9,
    shaft_mesh_n_faces_min = 10,
    
    # bare minimum filters 
    spine_volume_to_spine_area_min_bare_min = 0.01,
    spine_n_face_threshold_bare_min = 10,
    
)
global_parameters_dict_h01_head_neck_shaft = {}

global_parameters_dict_h01 = gu.merge_dicts([
    global_parameters_dict_h01_spine_identification,
    global_parameters_dict_h01_head_neck_shaft
    
])


attributes_dict_h01 = {}

# data_type = "default"
# algorithms = None

# modules_to_set = [spu]

# def set_global_parameters_and_attributes_by_data_type(dt,
#                                                      algorithms_list = None,
#                                                       modules = None,
#                                                      set_default_first = True,
#                                                       verbose=False):
#     if modules is None:
#         modules = modules_to_set
    
#     modu.set_global_parameters_and_attributes_by_data_type(modules,dt,
#                                                           algorithms=algorithms_list,
#                                                           set_default_first = set_default_first,
#                                                           verbose = verbose)
    
# set_global_parameters_and_attributes_by_data_type(data_type,
#                                                    algorithms)

# def output_global_parameters_and_attributes_from_current_data_type(
#     modules = None,
#     algorithms = None,
#     verbose = True,
#     lowercase = True,
#     output_types = ("global_parameters"),
#     include_default = True,
#     algorithms_only = False,
#     **kwargs):
    
#     if modules is None:
#         modules = modules_to_set
    
#     return modu.output_global_parameters_and_attributes_from_current_data_type(
#         modules,
#         algorithms = algorithms,
#         verbose = verbose,
#         lowercase = lowercase,
#         output_types = output_types,
#         include_default = include_default,
#         algorithms_only = algorithms_only,
#         **kwargs,
#         )


#--- from neurd_packages ---
from . import apical_utils as apu
from . import branch_attr_utils as bau
from . import branch_utils as bu
from . import cell_type_utils as ctu
from . import neuron_searching as ns
from . import neuron_statistics as nst
from . import neuron_utils as nru
from . import neuron_visualizations as nviz
from . import synapse_utils as syu
from . import width_utils as wu

#--- from mesh_tools ---
from mesh_tools import skeleton_utils as sk
from mesh_tools import trimesh_utils as tu

#--- from datasci_tools ---
from datasci_tools import dj_utils as dju
from datasci_tools import general_utils as gu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import mesh_utils as meshu
from datasci_tools import module_utils as modu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_dep as np
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import statistics_utils as stu
from datasci_tools import system_utils as su
from datasci_tools import tqdm_utils as tqu

from . import spine_utils as spu