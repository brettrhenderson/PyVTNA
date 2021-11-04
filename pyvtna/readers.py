"""pre plotting data manipulation"""
from pandas import ExcelFile, read_excel
import numpy as np
import copy

class VTNAReader:
    
    def __init__(self):
        self.reaction_traces = {}
        self.original_reaction_traces = {}
        self.reaction_names = []
        self.species_names = []
        self.species_totals = {}
        self.species_maxes = {}
        self.species_norms = {}
        self.norm_method = 'TC'
        self.normalized = False
        
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
            for rxn_name, rxn_trace in self.reaction_traces.items():
                self.species_totals[rxn_name] = (rxn_trace[:, 1:].sum(axis=1))
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
            for rxn_name, rxn_trace in self.reaction_traces.items():
                self.species_maxes[rxn_name] = rxn_trace[:, 1:].max()
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
            self.species_norms = {rxn_name: 1 for rxn_name in self.reaction_names}
        return self.species_norms

    def normalize_columns(self):
        """
        normalize all columns by the sum on that time step (excludes the time column in a sheet)
        Parameters
        ----------

        Returns
        -------

        """
        if self.normalized:
            return self.reaction_traces
        else:
            for rxn_name, rxn_trace in self.reaction_traces.items():
                norm = self.get_species_norms()[rxn_name]
                if isinstance(norm, np.ndarray):
                    norm = norm.reshape(-1, 1)
                rxn_trace[:, 1:] /= norm
            self.normalized = True
            return self.reaction_traces
    
    def reset_reaction_traces(self):
        self.reaction_traces = copy.deepcopy(self.original_reaction_traces)
        self.species_totals = {}
        self.species_maxes = {}
        self.species_norms = {}
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
            shifts = {rxn_name: shifts for rxn_name in self.reaction_names}
        if isinstance(shifts, list):
            for i, (rxn_name, rxn_trace) in enumerate(self.reaction_traces.items()):
                rxn_trace[:, 0] -= shifts[i]
        elif isinstance(shifts, dict):
            if list(shifts.keys()) != list(self.reaction_names):
                raise ValueError('Shift keys must match reaction names.')
            for rxn_name, rxn_trace in self.reaction_traces.items():
                rxn_trace[:, 0] -= shifts[rxn_name]
        else:
            raise ValueError('Shift keys must either be list or dict with keys equal to reaction names.')
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
        maxtimes = {}
        for rxn_name, rxn_trace in self.reaction_traces.items():
            maxtimes[rxn_name] = rxn_trace[:, 0].max()
        return maxtimes

    def multiply_concs(self, concs):
        """
        Allows a user to calibrate the concentraton values of their reaction traces separately.

        If the user has only ion count data, for example, they can turn that into concentration
        data by normalizing and then multiplying by the known concentration of the reagent that has
        been normalized to 1 at the start of the reaction. Also allows different multiplications for
        each species in each reaction trace, to account for sensitivity of species detection.

        Parameters
        ----------
        concs : dict
            Keys should match the reaction names in `VTNAReader.reaction_names`. If value is a float,
            all species will be multiplied by that value. Else if value is a sequence, each species
            will be multiplied by the corresponding entry in the sequence

        Returns
        -------
        dict
            Updated reaction traces

        """
        if list(concs.keys()) != self.reaction_names:
            raise ValueError('Concentration keys must have keys equal to reaction names.')
        for rxn_name, rxn_trace in self.reaction_traces.items():
            rxn_trace *= np.array(concs[rxn_name])
        return self.reaction_traces

    def select_data(self, reaction_names=None, species_names=None):
        """selects data to plot"""
        if not all([isinstance(spec, int) for spec in species_names]):
            if all([isinstance(spec, str) for spec in species_names]):
                species_names = [self.species_names.index(spec) for spec in species_names].sort()
            else:
                raise ValueError("Species must be either a list of integer indexes or strings identifying species names")
        if reaction_names is None:   # return all reactions
            if species_names is None:     # return all species
                return self.reaction_traces
            return {rxn_name: rxn_trace[:, [0]+[spec + 1 for spec in species_names]]
                    for rxn_name, rxn_trace in self.reaction_traces.items()}
        elif species_names is None:
            return {rxn_name: self.reaction_traces[rxn_name] for rxn_name in reaction_names}
        return {rxn_name: self.reaction_traces[rxn_name][:, [0]+[spec + 1 for spec in species_names]]
                for rxn_name in reaction_names}


class ManualInput(VTNAReader):
    def __init__(self, data=None, species_names=None):
        super().__init__()
        if data is not None:
            self.reaction_names = [str(name) for name in data.keys()]
            self.species_names = species_names
            self.load(data)

    def load(self, data):
        # check the format of reaction data
        for rxn_name, rxn_trace in data.items():
            if self.species_names is None:
                self.species_names = [str(i) for i in range(rxn_trace.shape[1] - 1)]
            else:
                if rxn_trace.shape[1] - 1 != len(self.species_names):
                    raise ValueError("Reaction traces contain different numbers of species monitored.")
        self.reaction_traces = data
        self.original_reaction_traces = copy.deepcopy(data)


class ExcelReader(VTNAReader):
    
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.load(filename)
    
    def load(self, filename):
        xl = ExcelFile(filename)
        reaction_traces = {}
        species_names = None
        reaction_names = xl.sheet_names
        match = True
        for i, rxn_name in enumerate(reaction_names):
            df = read_excel(filename, i)
            reaction_traces[rxn_name] = df.values
            if species_names is None:
                species_names = df.columns.tolist()
            else:
                cols = df.columns.tolist()
                if cols != species_names:
                    match = False
                    if len(cols) != len(species_names):
                        raise ValueError("Sheets contain different numbers of species monitored.")
        if not match:
            species_names = [str(i + 1) for i in range(reaction_traces[rxn_name].shape[1])]
        self.reaction_traces = reaction_traces
        self.original_reaction_traces = copy.deepcopy(reaction_traces)
        self.reaction_names = [str(name) for name in reaction_names]
        self.species_names = species_names[1:]
