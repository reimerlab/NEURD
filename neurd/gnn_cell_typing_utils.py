"""
Purpose: 
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from copy import deepcopy
import pandas as pd

try:
    from torch_geometric.data import Data
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch.nn import Linear
    from torch_geometric.nn import global_mean_pool, global_max_pool
    import torch_geometric.nn as nn_geo
    import torch_geometric.nn as nn 
    from torch_geometric.loader import DataLoader
except:
    pass

class_name = "label"

# **-- Utilities for dataset -----**

class GraphData(ABC):
    @abstractmethod
    def __init__(self,G,data=None,label=None):
        if data is None:
            data = dict()
            
        self.identifiers = dict(
            segment_id = G.graph['segment_id'],
            split_index = G.graph.get('split_index',0),
        )
        for k,v in self.identifiers.items():
            setattr(self,k,v)
            
        # setting the labe if there is one
        if label is None:
            label = data.get(class_name,None)
        self.label = label
        
        self.essential_properties = [
            "node_list",
            "edge_index",
            "feature_df",
            "node_weight",
        ]

        self.set_essential_properties(data)
        self.check_essential_properties()

    def attribute_str(self, parenthesis = True):
        str_list = [
            f"Nodes = {len(self.node_list)}",
            f"Edges = {len(self.edge_index)}",
            f"Features = {len(self.feature_df.columns)}"
        ]
        comb_str = ", ".join(str_list)
        if not parenthesis:
            return comb_str
        return "(" + comb_str + ")"
        
    def __repr__(self):
        return f"Graph Data {self.attribute_str()}"
            
    def check_essential_properties(self):
        for p in self.essential_properties:
            if getattr(self,p) is None:
                raise AttributeError(f"{p} is None")

    def set_essential_properties(self,data):
        for p in self.essential_properties:
            setattr(self,p,data.get(p,None))

    @property
    def skeletal_length(self):
        return self.feature_df['skeletal_length'].sum()


def limb_prefixes_from_graph(G):
    return set(list([k.split("_")[0] for k in G.nodes()]))

def limb_data_dict_from_G(
    G,
    soma_attributes,
    node_weight_attribute = "skeletal_length",
    features=None,
    return_dict = True
    ):
    if features is None:
        features = nxf.features_to_output_for_gnn
    
    node_list = np.array(list(G.nodes()))
    
    #will already return a symmetrical adjacency matrix
    edge_index = nu.edge_list_from_adjacency_matrix(
        nx.adjacency_matrix(
        G,
        nodelist = node_list,
        weight = "weight"
    ).toarray())
    
    feature_df = xu.feature_matrix_from_G(
        G,
        features=features,
        return_df=True
    )
        
    feature_df = pu.replace_nan_with_default(
        feature_df,
        default = 0
    )
    
    
    node_weight = feature_df[node_weight_attribute].to_numpy()
    
    feature_df["soma_start_angle_max"] = nxu.soma_start_angle_max(G)
    for k,v in soma_attributes.items():    
        feature_df[k] = v

    return_d = dict(
            node_list = node_list,
            edge_index = edge_index,
            node_weight = node_weight,
            feature_df = feature_df,
        )
    if return_dict:
        return return_d
    else:
        return list(return_d.values())


class LimbGraphData(GraphData):
    def __init__(self,G,soma_attributes,limb_idx = None,label=None,features=None,**kwargs):
        
        if "S0" in G.nodes():
            raise ValueError("This graph has a soma node, supposed to be a limb graph")
        node_prefixes = limb_prefixes_from_graph(G)
        if len(node_prefixes) > 1:
            raise ValueError(f"Graph  had multiple different limb prefixes (should have one): {node_prefixes}")

        # set the essential properties
        limb_dict = limb_data_dict_from_G(G,soma_attributes,features=features,**kwargs)
        super().__init__(G,data=limb_dict,label = label)
        
        #set the limb identifier
        self.identifiers['limb_idx'] = limb_idx
        self.limb_idx = limb_idx

    def __repr__(self):
        identifier = f"Neuron {self.segment_id}_{self.split_index} Limb {self.limb_idx}"
        return  f"{identifier} {self.attribute_str()}"


def extract_soma_attributes(G):
    soma_attributes_names = {
        "max_soma_volume":"mesh_volume",
        "n_syn_soma":"n_synapses"
    }
    soma_dict = G.nodes["S0"]
    return {k:soma_dict[v] for k,v in 
                      soma_attributes_names.items() if v in soma_dict}

def clean_fetched_graph(G,verbose = False):
    G =  nxu.fix_flipped_skeleton(G,verbose=verbose)
    G = nxu.fix_attribute(G,verbose = verbose)
    G = nxu.fix_width_inf_nan(G, verbose = verbose)
    return G

def scale_synapse_volume(
    G,
    voxel_to_nm_scaling = (4,4,40),
    in_place = False,
    ):
    """
    The GNN model was trained where the synapse volumes
    weren't correctly scaled by the voxel_to_nm product,
    so adjusting the current synapse volumes by dividing by the 
    voxel_to_nm product to put in the right scale for the GNN
    """
    if not in_place:
        G = deepcopy(G)

    synapse_attribute = "synapse_data"
    product = np.array(voxel_to_nm_scaling).prod()
    for n in G.nodes():
        if synapse_attribute in G.nodes[n]:
            for s in G.nodes[n][synapse_attribute]:
                s["volume"] = s["volume"]/product
    return G
        
def preprocess_neuron_graph(
    G,
    in_place = False,
    verbose = False,
    distance_threshold_min = 0,
    ):
        
    if not in_place:
        G = G.copy()
        
    G_clean = clean_fetched_graph(
        G
    )
    G_clean = scale_synapse_volume(
        G_clean
    )
    
    G1 = nxu.filter_graph(
        G_clean,
        features_to_output=[],
    )
    
    G2 = nxst.add_summary_statistic_over_dynamic_attributes_to_G(G1)
    G2 = nxst.add_any_missing_node_features(G2)
    
    G3 = nxu.nodes_farther_than_distance_from_soma(
                G2,
                verbose = verbose,
                distance_threshold = distance_threshold_min,
                return_subgraph = True,
                distance_type = "downstream",
                from_attributes = True,
    )

    return G3

def combine_limb_data_objs(
    limb_data_objs
    ):
    """
    Purpose
    -------
    Combining a list of limb datasets into a whole 
    neuron graph dataset
    
    Pseudocode
    ----------
    1) hstack the node_list, node_weight
    2) stack the pandas features_df
    3) Stack the edges indexes but add the previous node total
    """
    node_number = [len(k.node_list) for k in limb_data_objs]
    node_number_cumsum = np.insert(np.cumsum(node_number),0,0)[:-1]

    node_list = np.hstack([k.node_list for k in limb_data_objs])
    node_weight = np.hstack([k.node_weight for k in limb_data_objs])

    feature_df = pd.concat([k.feature_df for k in limb_data_objs]).reset_index(drop=True)
    edge_index_list = [k.edge_index for k in limb_data_objs]
    edge_index = np.vstack([
        k + v for k,v in zip(edge_index_list,node_number_cumsum)
    ])
    label_data = [k.label for k in limb_data_objs]
    label = max(label_data,key=label_data.count)

    return dict(
        node_list=node_list,
        node_weight=node_weight,
        feature_df=feature_df,
        edge_index=edge_index,
        label=label
    )
    
class NeuronGraphData(GraphData):
    
    def __init__(
        self,
        G,
        soma_attributes = None,
        limb_skeletal_length_min = 25_000,
        label = None,
        verbose = False,
        features=None,
        **kwargs):
        """
        Purpose
        -------

        Psuedocode
        ----------
        1) Extract out the soma attributes
        2) Create the limb Data objects
        3) Combine the limb data objects into a graph object
        """
        if "S0" not in G.nodes():
            raise ValueError("This graph must contain a soma node")

        if soma_attributes is None:
            soma_attributes = extract_soma_attributes(G)

        G = preprocess_neuron_graph(G,verbose = verbose)
        self.limb_data_objs = [
            LimbGraphData(
                Gl,
                soma_attributes=soma_attributes,
                limb_idx = j,
                label = label,
                features=features,
                **kwargs
            )
            for j,Gl in enumerate(
                nxu.all_limb_graphs_off_soma(G,verbose = verbose)
            )
        ]

        # filters the limbs for a minimum skeletal length
        self.limb_skeletal_length_min = limb_skeletal_length_min
        if self.limb_skeletal_length_min is not None:
            self.limb_data_objs = [k for k in self.limb_data_objs
                                   if k.skeletal_length > limb_skeletal_length_min]
            
        whole_neuron_data = combine_limb_data_objs(
            self.limb_data_objs
        )

        super().__init__(G,data=whole_neuron_data)
            
    def __repr__(self):
        identifier = f"Neuron {self.segment_id}_{self.split_index}"
        return  f"{identifier} {self.attribute_str()}"

    @property
    def num_limbs(self):
        return len(self.limb_data_objs)
    
    
# ** -- utilities for pytorch dataset -- **
class OutputClassConfig:
    """
    Purpose
    -------
        Store all relevant information for the output class
        setup (ex: integer mapping)
    """
    def __init__(
        self,
        class_map
        ):

        self.class_map = class_map
        self.class_map_inv = {v:k for k,v in self.class_map.items()}
        
    def convert_class_to_int(self,y):
        return self.class_map[y]

    @property
    def classes(self):
        return list(self.class_map.keys())

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def class_map_int(self):
        return self.class_map_inv

    def decode_ints(self,data):
        return [self.class_map_inv[k] for k in data]
    
    def __repr__(self):
        return self.class_map.__repr__()
    
    def __str__(self):
        return self.class_map.__str__()

microns_cell_type_map = {
 '23P': 0,
 '4P': 1,
 '5P-IT': 2,
 '5P-NP': 3,
 '5P-PT': 4,
 '6P-CT': 5,
 '6P-IT': 6,
 'BC': 7,
 'BPC': 8,
 'MC': 9,
 'NGC': 10
}



class NeuronDataset():
    """
    Class that will store a list of graph 
    objects and normalize them if need be
    """
    def __init__(
        self,
        graphs,
        normalization_df = None,
        normalization_df_csv = None,
        class_config = None,
        ):

        self._graphs = graphs
        self.normalization_df = None
        self.normalization_df_csv = None
            
        self.load_normalization_df(
            normalization_df = normalization_df,
            normalization_df_csv = normalization_df_csv
        )
        self.class_config = class_config

        self._dataset = None

    def convert_graph_data_to_pytorch_data(
        self,
        G,
        normalize = True,
        feature_df = None):
        """
        Purpose
        -------
            To convert a GraphData object
        into a pytorch Data object
    
        Pseudocode
        ----------
        """
        if feature_df is None:
            feature_df = G.feature_df
            
        if normalize:
            if self.normalization_df is None:
                raise AttributeError(
                    f"normalization_df is None when" \
                    + " creating dataset (normalization turned on)"
                )
            feature_df = pu.normalize_df_with_df(
                    df = feature_df,
                    df_norm=self.normalization_df,)
            
        x = torch.tensor(
            feature_df.to_numpy(),
            dtype=torch.float
        )
            
        edge_index = torch.tensor(
            G.edge_index.T,
            dtype=torch.long
        )

        node_weight = torch.tensor(
            G.node_weight,
            dtype = torch.float
        )
            
        data_dict = dict(
            x = x,
            edge_index=edge_index,
            node_weight=node_weight,
        )
        
        y = getattr(G,class_name,None)
        if y is not None:
            if "int" not in str(type(y)):
                y = self.class_config.convert_class_to_int(y)
            data_dict["y"] = torch.tensor(y,dtype=torch.long)
    
        for k,v in G.identifiers.items():
            data_dict[k] = v
            
        return Data(**data_dict)
        
    def load_normalization_df(
        self,
        normalization_df = None,
        normalization_df_csv = None,):
        
        if normalization_df is None:
            if self.normalization_df_csv is None:
                self.normalization_df_csv = normalization_df_csv
            normalization_df = pu.csv_to_df(self.normalization_df_csv)
        
        self.normalization_df = normalization_df
            
    @property
    def num_graphs(self):
        return len(self._graphs)

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = [self.convert_graph_data_to_pytorch_data(k)
                             for k in self._graphs]
        return self._dataset

class LimbDataset(NeuronDataset):
    def __init__(
        self,
        graphs,
        min_skeletal_length=60_000,
        *args,**kwargs):
        
        self.min_skeletal_length = min_skeletal_length
        filtered_graphs = [
            k for k in graphs
            if k.skeletal_length >= self.min_skeletal_length]
        
        super().__init__(
            graphs = filtered_graphs,
            *args,
            **kwargs)

    def convert_graph_data_to_pytorch_data(
        self,
        G,
        normalize = True):

        feature_df = G.feature_df

        if feature_df["soma_start_angle_max"].max() > 1:
            feature_df["soma_start_angle_max"] /= 180

        return super().convert_graph_data_to_pytorch_data(
            G,
            normalize=normalize,
            feature_df=feature_df
        )
        
        
# ** --- Model architectures --- **

def isnan_any(tensor):
    return torch.any(torch.isnan(tensor))

def global_mean_weighted_pool(x,batch,weights,debug_nan = False):
    """
    Purpose: To do a weighted mean pooling for 
    a batch
    
    Ex: 
    from pytorch_tools import tensor_utils as tenu
    x = torch.Tensor([70,80,90,100,110,120])
    w = torch.Tensor([10,5,15,10,5,15])
    batch = torch.tensor(np.array([0,0,0,1,1,1]),dtype=torch.int64)
    tenu.global_mean_weighted_pool(x,batch,w)
    """
    weights = (weights.unsqueeze(1))
    if debug_nan:
        if tenu.isnan_any(batch):
            raise Exception(f"Nan batch")
    weight_sum = nn_geo.global_add_pool(weights,batch)
    
    
    if debug_nan:
        if isnan_any(weight_sum):
            raise Exception(f"Nan weight_sum")
    weighted_value_sum = nn_geo.global_add_pool(x*weights,batch)
    if debug_nan:
        if isnan_any(weighted_value_sum):
            raise Exception(f"Nan weighted_value_sum")
    
    weight_result = weighted_value_sum/weight_sum
    weight_result[weight_result != weight_result] = 0
    
    if debug_nan:
        if isnan_any(weight_result):
            raise Exception(f"Nan weight_result")
    return weight_result

class NeuronGCN(torch.nn.Module):
    """
    
    """
    def __init__(
        self,
        
        num_node_features,
        num_classes,
        
        activation_function = "relu",
        
        n_hidden_channels=None,
        n_layers = None,
        add_self_loops = False,
        
        #batch norm specifics
        use_bn = True,
        track_running_stats=True,
        
        global_pool_type="mean_weighted",
        global_pool_weight = "node_weight",
        
        **kwargs
        ):
        """
        
            
        """
        
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        
        self.n_hidden_channels=n_hidden_channels
        self.n_layers = n_layers
        self.add_self_loops = add_self_loops
        
        self.use_bn = use_bn
        self.act_func = getattr(F,activation_function)
        
        # -- for the pooling --
        self.global_pool_type = global_pool_type
        self.global_pool_func = eval(f"global_{global_pool_type}_pool")
        self.global_pool_weight = global_pool_weight
        
        previous_layer = self.num_node_features
        
        # creattng the hidden layers and batch norm layers
        for j in range(self.n_layers):
            suffix = self.suffix_from_layer(j)
            n_output = self.n_hidden_channels
            setattr(self,
                    f"conv{suffix}",
                    GCNConv(
                        previous_layer, 
                        n_output,
                        add_self_loops=self.add_self_loops)
            )
            
            if self.use_bn: 
                setattr(
                    self,
                    f"bn{suffix}",
                    torch.nn.BatchNorm1d(
                        n_output,
                        track_running_stats=track_running_stats
                        )
                )
            previous_layer = n_output
            
        # creates the linear layer for class outputs    
        self.lin = Linear(
            previous_layer, 
            self.num_classes
        )
        
    def suffix_from_layer(self,layer):
        return f"{layer}_pool0"
              
    def encode(self,data,**kwargs):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data,"batch",None)
        
        
        for j in range(self.n_layers):
            suffix = self.suffix_from_layer(j)
            
            #print(f"X before conv{suffix}")
            #print(x)
            x = getattr(self,f"conv{suffix}")(
                x, 
                edge_index,
            )
            #print(f"X after conv{suffix}")
            #print(x)
            
            if self.use_bn:
                x = getattr(self,f"bn{suffix}")(x)
                    
            # not running activation on the last layer
            if (j < self.n_layers-1): 
                x = self.act_func(x)

        return x,batch
    
    def pooling(self,x,batch,weights):  
        # using the batch to do global pooling
        return_x = self.global_pool_func(
            x,
            batch,
            weights=weights
        )
        
        return return_x
        
    def forward(self, data,**kwargs):
        weights = getattr(data,f"{self.global_pool_weight}")
        x,batch = self.encode(data,**kwargs)
        x = self.pooling(x,batch,weights)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return F.softmax(x,dim=1)
        
        
class LimbGCN(NeuronGCN):
    def __init__(
        self,
        *args,
        **kwargs):
        
        super().__init__(
            *args,
            **kwargs
        )
        
        # expanding the size of the linear layer because
        # pooling stage creates twice as large vector
        self.lin = Linear(
            self.n_hidden_channels*2, 
            self.num_classes
        )
        
    def pooling(self,x,batch,weights):
        
        return_x = super().pooling(x,batch,weights)
        return_x = torch.hstack(
            [return_x,
             global_max_pool(x,batch)]
        )
        
        return return_x
    
    
# ** -- Inference and training run -- **
def load_saved_model(model,weight_filepath):
    checkpoint = torch.load(weight_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class GnnInput:
    """
    Purpose: To collect the necessary 
    information for running a training
    or inference run:
        a. Model (optional, saved weights)
        b. Dataset
        c. class configuration
    """
    def __init__(
        self,
        dataset_obj,
        
        # for initializing model
        model = None,
        model_cls = None,
        model_architecture_kwargs = None,
        model_weights_filepath = None,

        # for creating the dataset
        batch_size = 20,

        # for decoding the class
        class_config = None,
        ):

        # -- instantiating the model
        self.model = model

        if self.model is None:
            self.model_architecture_kwargs = model_architecture_kwargs
            self.model = model_cls(**model_architecture_kwargs)
        if model_weights_filepath is not None:
            self.model_weights_filepath = model_weights_filepath
            self.load_saved_model()

        # -- creating the dataloader
        self.dataset_obj = dataset_obj
        self.batch_size = batch_size
        self.data_loader = self.create_data_loader()
            
        # for decoding the moel output
        if class_config is None:
            class_config = dataset.class_config

        if class_config is None:
            raise AttributeError("Class Configuration must be set")
        
        self.class_config = class_config
    
    def create_data_loader(self,dataset = None,shuffle=False):
        if dataset is None:
            dataset = self.dataset_obj.dataset
            
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle = shuffle
        )
    
    def load_saved_model(self):
        self.model = load_saved_model(
            self.model,
            self.model_weights_filepath)

            
class Runner:
    """
    Parent class for running inference or training runs
    """
    def __init__(
        self,
        gnn_input,
        *args,
        **kwargs,
    ):
        self.gnn_input = gnn_input
        self.model = gnn_input.model
        self.dataset = gnn_input.dataset_obj.dataset
        self.data_loader = gnn_input.data_loader
        self.class_config = gnn_input.class_config
        
        
def inference_run(model,dataset_loader):
    model.eval()

    prediction_list = []
    y_list = []
    with torch.no_grad():
        for batch in dataset_loader:
            out = model(batch)  # Forward pass
            labels = batch.y # Get the predicted class for 
            prediction_list.append(out.detach().cpu().numpy())
            y_list.append(labels.detach().cpu().numpy())
            
    y_list = np.hstack(y_list)
    prediction_list = np.vstack(prediction_list)
    
    return prediction_list,y_list
            

class InferenceRunner(Runner):
    """
    Purpose: To carry out an inference run
    using input from 
    """
    def __init__(
        self,
        gnn_input,
        *args,
        **kwargs,
    ):
        
        super().__init__(gnn_input,*args,**kwargs)
        
    def run(
        self,
        data_loader=None,
        verbose = False
        ):
        
        if data_loader is None:
            data_loader = self.data_loader

        class_conf,y = inference_run(
            self.model,
            data_loader
        )
        
        # decode the class predictions
        y_pred = np.argmax(class_conf,axis = 1)
        y_pred_class = self.class_config.decode_ints(y_pred)
        y_pred_class_conf = [class_conf[j,k] for j,k in 
                             enumerate(y_pred)]
        y_class = self.class_config.decode_ints(y)
        
        self.y_pred = y_pred
        self.y_pred_class = y_pred_class
        self.y_pred_class_conf = y_pred_class_conf
        self.y_class = y_class
        
    
    @property
    def prediction_df(self):
        graph_idx = np.arange(len(self.y_pred_class))
        df = pd.DataFrame.from_dict(dict(
            graph_idx = graph_idx,
            y_pred_class = self.y_pred_class,
            y_pred_confidence = self.y_pred_class_conf,
            y_true = self.y_class,
        ))
        return df
            
        
class TrainingRunner(Runner):
    """
    To implement a training run
    """
    pass

    




    
#neuron_morphology_tools import
from neuron_morphology_tools import (
    neuron_nx_stats as nxst,
    neuron_nx_feature_processing as nxf,
    neuron_nx_utils as nxu
)


#datasci_tools imports
from datasci_tools import \
    numpy_utils as nu, \
    networkx_utils as xu, \
    pandas_utils as pu
