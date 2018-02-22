import numpy as np
from DataBaseClass import DataBase

class PseudoData(DataBase):

    def __init__(self, input_data, output_data, diagnoses, sequence_lengths, transform_diagnoses = False, diag_labels_dim = None, data_order = None):
        self._input_data = input_data
        self._data_dim = self._input_data.shape[-1]
        self._max_length = self._input_data.shape[1]
        self._output_data = output_data
        self._target_dim = self._output_data.shape[-1]
        if transform_diagnoses:
            assert diag_labels_dim is not None
            self._diag_labels_dim = diag_labels_dim
            self._diagnosis_labels = self.transform_diagnosis_labels(diagnoses, diag_labels_dim)
        else:
            self._diag_labels_dim = diagnoses.shape[1]
            self._diagnosis_labels = diagnoses
        self._sequence_lengths = sequence_lengths
        if data_order is None:
            self._data_order = np.arange(input_data.shape[0])
        else:
            self._data_order = data_order
        self._input_mean, self._input_chol_cov = self.compute_mean_chol_cov()
        self.normalise()
        