"""pre plotting data manipulation"""
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class VTNAReader:
    
    def __init__(self):
        self.reaction_traces = {}
        self.original_reaction_traces = {}
        self.reaction_names = []
        self.reactant_names = []
        self.species_totals = {}
        self.species_maxes = {}
        self.species_norms = {}
        self.norm_method = 'TC'
        self.normalized = False

    @staticmethod
    def get_reaction_names(data):
        return list(data.keys())

    @staticmethod
    def get_reactant_names(data):
        reactants = None
        match = True
        # import excel sheets of reaction 1 and 2
        for rxn in VTNAReader.get_reaction_names():
            if reactants is None:
                reactants = list(data[rxn].keys())
            else:
                new_reactants = list(data[rxn].keys())
                if new_reactants != reactants:
                    match = False
                    if len(new_reactants) != len(reactants):
                        raise ValueError("Each reaction must have same number of monitored species.")
        if not match:
            reactants = [str(i) for i in range(1, len(data[rxn].keys()) + 1)]
        return reactants
        
    def load(self, filename):
        pass
    
    def get_tc(self):
        """
        produce a column summing all counts at each timestep to be used for Total ion Count normalization

        Parameters
        ----------

        Returns
        -------

        """
        if len(self.species_totals):
            return self.species_totals
        else:
            for rxn, data in self.reaction_traces.items():
                self.species_totals[rxn] = (df.iloc[:, 1:].sum(axis=1))
            return self.species_totals
        
    def get_mv(self):
        """
        produce a column of the max species values at each timestep to be used for Max Value normalization

        Parameters
        ----------

        Returns
        -------

        """
        if len(self.species_maxes):
            return self.species_maxes
        else:
            for df in self.reaction_traces:
                self.species_maxes.append(df.iloc[:, 1:].max().max())
            return self.species_maxes
        
    def set_norm_method(self, normalization_method): 
        if normalization_method == "TC" or normalization_method == "Total Count":
            self.norm_method = "Total Count"
        elif normalization_method == "MV" or normalization_method == "Max Value":
            self.norm_method = "Max Value"
        else:
            raise ValueError("Normalization Method must be either 'Total Count' ('TC') or 'Max Value' ('MV').")

    def get_species_norms(self, normalization_method=None):
        """
        returns chosen normalization value

        if neither TC nor MV is selected, the operations of Total1 and Total2
        will not change any values
        Parameters
        ----------
        normalization_method

        Returns
        -------

        """
        if normalization_method is None:
            normalization_method = self.norm_method 
        if normalization_method == "TC" or normalization_method == "Total Count":
            self.species_norms = self.get_tc()
        elif normalization_method == "MV" or normalization_method == "Max Value":
            self.species_norms = self.get_mv()
        else:
            self.species_norms = [1]*len(self.reaction_traces)
        return self.species_norms

    def normalize_columns(self):
        """
        normalize all columns by the sum on that time step (excludes the time column in a sheet)
        Parameters
        ----------

        Returns
        -------

        """
        rnorm = []
        for i, df in enumerate(self.reaction_traces):
            rnorm.append(pd.concat([df.iloc[:, 0], df.iloc[:, 1:].div(self.get_species_norms()[i], axis=0)], axis=1))
        self.original_reaction_traces = self.reaction_traces
        self.reaction_traces = rnorm
        self.normalized = True
        return rnorm
    
    def reset_reaction_traces(self):
        self.reaction_traces = self.original_reaction_traces
        self.species_totals = []
        self.species_maxes = []
        self.species_norms = []
        self.normalized = False

    def shift_times(self, shifts):
        """
        function that shifts the time column of each reaction by a set amount.

        Parameters
        ----------
        shifts

        Returns
        -------

        """
        if isinstance(shifts, float):
            shifts = [shifts for _ in self.reaction_traces]
        for i, df in enumerate(self.reaction_traces):
            df.iloc[:, 0] -= shifts[i]
        return self.reaction_traces

    def get_max_times(self):
        """
        Get the final time at the end of reaction data.

        Parameters
        ----------

        Returns
        -------
        maximum time: float

        """
        maxtime = []
        for df in self.reaction_traces:
            maxtime.append(df.iloc[:, 0].max())
        return maxtime

    # update concs to be read from sheet (column 1) and have user update names of species in the website if they desire
    def multiply_concs(self, concs):
        concs = np.array(concs)
        for i, df in enumerate(self.reaction_traces):
            for j in range(1, len(df.columns)):
                df.iloc[:, j] *= concs[i, j]
        return self.reaction_traces

    def select_data(self, reactions=None, species=None):
        """selects data to plot"""
        if reactions is None:   # return all reactions
            if species is None:     # return all species
                return self.reaction_traces
            return [self.reaction_traces[rxn].iloc[:, [0]+[spec+1 for spec in species]]
                    for rxn in range(len(self.reaction_traces))]
        elif species is None:
            return [self.reaction_traces[rxn] for rxn in reactions]
        return [self.reaction_traces[rxn].iloc[:, [0]+[spec+1 for spec in species]] for rxn in reactions]

class ManualInput(VTNAReader):
    def __init__(self, data_dict=None):
        super().__init__()
        if data_dict is not None:
            self.load(data_dict)

    def load(self, data):
        self.reaction_traces = data
        self.original_reaction_traces = data
        self.reaction_names = self.get_reaction_names(data)
        self.reactant_names = self.get_reactant_names(data)


class ExcelReader(VTNAReader):
    
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.load(filename)
    
    def load(self, filename):
        xl = pd.ExcelFile(filename)
        raw_data = []
        existing_reactants = None
        match = True
        # import excel sheets of reaction 1 and 2
        for i in range(len(xl.sheet_names)):
            raw_data.append(pd.read_excel(filename, i))
            if existing_reactants is None:
                existing_reactants = raw_data[-1].columns.tolist()
            else:
                cols = raw_data[-1].columns.tolist()
                if cols != existing_reactants:
                    match = False
                    if len(cols) != len(existing_reactants):
                        raise ValueError("Sheets contain different numbers of species monitored.")
            # print(f"\ncolumns read from experiment {i+1}: \n{len(raw_data[i].columns)}")
        if not match:
            existing_reactants = [str(i) for i in range(1, len(raw_data[1].columns) + 1)]
        self.reaction_traces = raw_data
        self.original_reaction_traces = [df.copy() for df in raw_data]
        self.reaction_names = xl.sheet_names
        self.reactant_names = existing_reactants[1:]
