"""
Unit and regression test for the pyvtna package.
"""

# Import package, test suite, and other packages as needed
import sys
import copy
import pytest
import pyvtna
import numpy as np
from pyvtna import align, metrics, notebook, poisoning, readers, signal, testing

class TestImports:
    def test_pyvtna_imported(self):
        """Sample test, will always pass so long as import statement worked."""
        assert "pyvtna" in sys.modules

    def test_modules_imported(self):
        """Sample test, will always pass so long as import statement worked"""
        assert all('pyvtna.'+ elem in sys.modules for elem in "align metrics notebook poisoning readers signal testing".split())

@pytest.fixture
def rxn_data():
    rxn_data = {
        'rxn1': np.array([[1, 2, 0, 0],
                          [2, 1.6, 0.2, 0.2],
                          [3, 1.2, 0.4, 0.4],
                          [4, 0.8, 0.6, 0.6],
                          [5, 0.4, 0.8, 0.8],
                          [6, 0, 1.0, 1.0]]),
        'rxn2': np.array([[1, 1, 0, 0],
                          [2, 0.6, 0.2, 0.2],
                          [3, 0.2, 0.4, 0.4],
                          [4, 0.0, 0.5, 0.5]])
    }
    return rxn_data

@pytest.fixture
def norm_rxn_data():
    norm_rxn_data = {
        'rxn1': np.array([[1, 1, 0, 0],
                          [2, 0.8, 0.1, 0.1],
                          [3, 0.6, 0.2, 0.2],
                          [4, 0.4, 0.3, 0.3],
                          [5, 0.2, 0.4, 0.4],
                          [6, 0, 0.5, 0.5]]),
        'rxn2': np.array([[1, 1, 0, 0],
                          [2, 0.6, 0.2, 0.2],
                          [3, 0.2, 0.4, 0.4],
                          [4, 0.0, 0.5, 0.5]])
    }
    return norm_rxn_data

@pytest.fixture
def rxn_tc():
    rxn_tc = {
        'rxn1': np.array([2., 2., 2., 2., 2., 2.]),
        'rxn2': np.array([1., 1., 1., 1.])
    }
    return rxn_tc

@pytest.fixture
def rxn_mv():
    rxn_mv = {
        'rxn1': 2.,
        'rxn2': 1.
    }
    return rxn_mv

class TestManualReader:
    def test_manual_input_rxn_names(self, rxn_data):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        assert manual_rxn_data.reaction_names == ['rxn1', 'rxn2']

    def test_manual_input_spec_names(self, rxn_data):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        assert manual_rxn_data.species_names == ['a', 'b', 'c']

    def test_manual_input_data(self, rxn_data):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        rxn_match = all([np.allclose(rxn_data[key], manual_rxn_data.reaction_traces[key]) for key in rxn_data])
        assert rxn_match

    def test_manual_input_get_tc(self, rxn_data, rxn_tc):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        tc = manual_rxn_data.get_tc()
        prop_match = all([np.allclose(rxn_tc[key], manual_rxn_data.species_totals[key]) for key in rxn_tc])
        return_match = all([np.allclose(rxn_tc[key], tc[key]) for key in rxn_tc])
        assert (prop_match and return_match)

    def test_manual_input_get_mv(self, rxn_data, rxn_mv):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        mv = manual_rxn_data.get_mv()
        prop_match = all([np.allclose(rxn_mv[key], manual_rxn_data.species_maxes[key]) for key in rxn_mv])
        return_match = all([np.allclose(rxn_mv[key], mv[key]) for key in rxn_mv])
        assert (prop_match and return_match)

    def test_manual_input_get_norms(self, rxn_data, rxn_tc):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        tc = manual_rxn_data.get_species_norms()
        prop_match = all([np.allclose(rxn_tc[key], manual_rxn_data.species_norms[key]) for key in rxn_tc])
        return_match = all([np.allclose(rxn_tc[key], tc[key]) for key in rxn_tc])
        assert (prop_match and return_match)

    def test_manual_input_get_norms_mv(self, rxn_data, rxn_mv):
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        manual_rxn_data.set_norm_method('MV')
        mv = manual_rxn_data.get_species_norms()
        prop_match = all([np.allclose(rxn_mv[key], manual_rxn_data.species_norms[key]) for key in rxn_mv])
        return_match = all([np.allclose(rxn_mv[key], mv[key]) for key in rxn_mv])
        assert (prop_match and return_match)

    def test_manual_input_normalize_rxn_mv(self, rxn_data, norm_rxn_data):
        og_rxn_data = copy.deepcopy(rxn_data)
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        manual_rxn_data.set_norm_method('MV')
        norm_rxns = manual_rxn_data.normalize_columns()
        prop_match = all([np.allclose(norm_rxn_data[key], manual_rxn_data.reaction_traces[key]) for key in norm_rxn_data])
        return_match = all([np.allclose(norm_rxn_data[key], norm_rxns[key]) for key in norm_rxn_data])
        orig_match = all([np.allclose(og_rxn_data[key], manual_rxn_data.original_reaction_traces[key]) for key in og_rxn_data])
        assert (prop_match and return_match and orig_match)

    def test_manual_input_normalize_rxn_tc(self, rxn_data, norm_rxn_data):
        og_rxn_data = copy.deepcopy(rxn_data)
        manual_rxn_data = readers.ManualInput(data=rxn_data, species_names=['a', 'b', 'c'])
        norm_rxns = manual_rxn_data.normalize_columns()
        prop_match = all([np.allclose(norm_rxn_data[key], manual_rxn_data.reaction_traces[key]) for key in norm_rxn_data])
        return_match = all([np.allclose(norm_rxn_data[key], norm_rxns[key]) for key in norm_rxn_data])
        orig_match = all([np.allclose(og_rxn_data[key], manual_rxn_data.original_reaction_traces[key]) for key in og_rxn_data])
        assert (prop_match and return_match and orig_match)


class TestExcelReader:
    def test_excel_reader_rxn_names(self, rxn_data):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        assert excel_rxn_data.reaction_names == ['rxn1', 'rxn2']

    def test_excel_reader_spec_names(self, rxn_data):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        assert excel_rxn_data.species_names == ['a', 'b', 'c']

    def test_excel_reader_data(self, rxn_data):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        rxn_match = all([np.allclose(rxn_data[key], excel_rxn_data.reaction_traces[key]) for key in rxn_data])
        assert rxn_match

    def test_excel_reader_get_tc(self, rxn_data, rxn_tc):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        tc = excel_rxn_data.get_tc()
        prop_match = all([np.allclose(rxn_tc[key], excel_rxn_data.species_totals[key]) for key in rxn_tc])
        return_match = all([np.allclose(rxn_tc[key], tc[key]) for key in rxn_tc])
        assert (prop_match and return_match)

    def test_excel_reader_get_mv(self, rxn_data, rxn_mv):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        mv = excel_rxn_data.get_mv()
        prop_match = all([np.allclose(rxn_mv[key], excel_rxn_data.species_maxes[key]) for key in rxn_mv])
        return_match = all([np.allclose(rxn_mv[key], mv[key]) for key in rxn_mv])
        assert (prop_match and return_match)

    def test_excel_reader_get_norms(self, rxn_data, rxn_tc):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        tc = excel_rxn_data.get_species_norms()
        prop_match = all([np.allclose(rxn_tc[key], excel_rxn_data.species_norms[key]) for key in rxn_tc])
        return_match = all([np.allclose(rxn_tc[key], tc[key]) for key in rxn_tc])
        assert (prop_match and return_match)

    def test_excel_reader_get_norms_mv(self, rxn_data, rxn_mv):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        excel_rxn_data.set_norm_method('MV')
        mv = excel_rxn_data.get_species_norms()
        prop_match = all([np.allclose(rxn_mv[key], excel_rxn_data.species_norms[key]) for key in rxn_mv])
        return_match = all([np.allclose(rxn_mv[key], mv[key]) for key in rxn_mv])
        assert (prop_match and return_match)

    def test_excel_reader_normalize_rxn_mv(self, rxn_data, norm_rxn_data):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        excel_rxn_data.set_norm_method('MV')
        norm_rxns = excel_rxn_data.normalize_columns()
        prop_match = all([np.allclose(norm_rxn_data[key], excel_rxn_data.reaction_traces[key]) for key in norm_rxn_data])
        return_match = all([np.allclose(norm_rxn_data[key], norm_rxns[key]) for key in norm_rxn_data])
        orig_match = all([np.allclose(rxn_data[key], excel_rxn_data.original_reaction_traces[key]) for key in rxn_data])
        assert (prop_match and return_match and orig_match)

    def test_excel_reader_normalize_rxn_tc(self, rxn_data, norm_rxn_data):
        excel_rxn_data = readers.ExcelReader(filename='pyvtna/data/test_rxn.xlsx')
        norm_rxns = excel_rxn_data.normalize_columns()
        prop_match = all([np.allclose(norm_rxn_data[key], excel_rxn_data.reaction_traces[key]) for key in norm_rxn_data])
        return_match = all([np.allclose(norm_rxn_data[key], norm_rxns[key]) for key in norm_rxn_data])
        print(excel_rxn_data.original_reaction_traces)
        orig_match = all([np.allclose(rxn_data[key], excel_rxn_data.original_reaction_traces[key]) for key in rxn_data])
        assert (prop_match and return_match and orig_match)