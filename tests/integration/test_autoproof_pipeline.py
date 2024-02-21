"""
Purpose: 

"""

import unittest

import neurd
from neurd.vdi_microns import volume_data_interface as vdi
from neurd import soma_extraction_utils as sm
from neurd import neuron
from neurd import neuron_pipeline_utils as npu
from neurd import neuron_utils as nru

from datasci_tools import pandas_utils as pu
from datasci_tools import pipeline
from datasci_tools import pathlib_utils as plu

from mesh_tools import trimesh_utils as tu

from pathlib import Path


class TestAutoproofPipeline(unittest.TestCase):
    def setUp(self):
        self.segment_id = 864691135510518224
        
        self.fixture_path = str((plu.parent_directory(plu.parent_directory(Path(__file__))) / Path(f'fixtures')).absolute())
        self.synapse_filepath = str(Path(f'{self.fixture_path}/{self.segment_id}_synapses.csv').absolute())
        
    def test_1_vdi_set_synapse_filepath_and_pipeline_products(self):
        vdi.set_synapse_filepath(
            self.synapse_filepath
        )
        
        self.__class__.products = pipeline.PipelineProducts()
        
        self.assertTrue(pathlib.Path(self.synapse_filepath).exists())
        
        
    def test_2_vdi_fetch_mesh(self):
        self.__class__.mesh = vdi.fetch_segment_id_mesh(
            self.segment_id,
            meshes_directory = self.fixture_path
        )
        
        self.assertIsNotNone(self.__class__.mesh)
        
    # --- Step 1: Decimation --
    def test_3_decimation(self):
        self.__class__.mesh_decimated = tu.decimate(
            self.__class__.mesh,
            decimation_ratio =0.25,
        )
        
        self.assertIsNotNone(self.__class__.mesh_decimated)
        
    def test_4_soma_extraction(self):
        self.__class__.soma_products = sm.soma_indentification(
            self.__class__.mesh,
            verbose=False,
        )
        
        self.__class__.products.set_stage_attrs(
            stage = "soma_identification",
            attr_dict = self.__class__.soma_products,
        )
        
        print(f"self.__class__.soma_products = {self.__class__.soma_products}")
        self.assertIsNotNone(self.__class__.soma_products)
        
    
    def test_5_decomposition(self):
        self.__class__.neuron_obj = neuron.Neuron(
            mesh = self.__class__.mesh,
            segment_id = self.segment_id, # don't need this explicitely if segment_id is already in products
            pipeline_products = self.__class__.products,
            suppress_preprocessing_print=True,
            suppress_output=True,
        )
        
        _ = self.__class__.neuron_obj.calculate_decomposition_products(
            store_in_obj = True,
        )
        
        self.assertIsNotNone(self.__class__.neuron_obj)
        
        
    def test_6_saving_off_and_reloading_neuron(self):
        vdi.save_neuron_obj(
            self.__class__.neuron_obj,
            verbose = False
        )
        self.__class__.neuron_obj = vdi.load_neuron_obj(
            segment_id = self.segment_id,
            mesh_decimated = self.__class__.mesh
        )
        
        self.assertIsNotNone(self.__class__.neuron_obj)
        
        
    def test_7_multi_soma_split(self):
        multi_soma_split_parameters = dict()
        
        _ = self.__class__.neuron_obj.calculate_multi_soma_split_suggestions(
            plot = False,
            store_in_obj = True,
            **multi_soma_split_parameters
        )
        
        self.__class__.neuron_list = self.__class__.neuron_obj.multi_soma_split_execution(
            verbose = False,
        )
        
        self.__class__.n1 = self.__class__.neuron_list[0]
        
        self.assertEqual(len(self.__class__.neuron_list),2)
        
    def test_8_cell_typing(self):
        self.__class__.neuron_obj_axon = npu.cell_type_ax_dendr_stage(
            self.__class__.n1,
            mesh_decimated = self.__class__.mesh,
            plot_axon = False,
        )
        
        self.assertIsNotNone(self.__class__.neuron_obj_axon)
        
    def test_9_autoproof(self):
        self.__class__.neuron_obj_proof = npu.auto_proof_stage(
            self.__class__.neuron_obj_axon,
            mesh_decimated = self.__class__.mesh,
            calculate_after_proof_stats = False,
        )
        
        self.assertIsNotNone(self.__class__.neuron_obj_proof)

        
    
if __name__ == '__main__':
    unittest.main()
        
    
    
        
    