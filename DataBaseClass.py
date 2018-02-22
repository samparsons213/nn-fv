import numpy as np
import abc
from utils import lazy_property
from DiagnosisModel import DiagnosisPrediction
from LSTMModel import LSTM

class DataBase:
    __metaclass__ = abc.ABCMeta

    _input_data = None
    _output_data = None
    _diagnosis_labels = None
    _sequence_lengths = None
    _data_order = None
    _input_mean = None
    _input_chol_cov = None
    _max_length = None
    _data_dim = None
    _target_dim = None
    _diag_labels_dim = None

    @abc.abstractmethod
    def __init__(self, parameter_list):
        ''' Implement concrete constructor here '''
    
    def compute_mean_chol_cov(self):
        all_input_data = np.concatenate([self._input_data[i, :self._sequence_lengths[i]] for i in range(self._input_data.shape[0])])
        input_mean =  np.mean(all_input_data, axis=0)
        input_cov = np.cov(all_input_data, rowvar=False)
        chol_cov = np.linalg.cholesky(input_cov)
        return input_mean, chol_cov

    def transform_diagnosis_labels(self, diagnosis_labels, num_labels):
        num_files = self._input_data.shape[0]
        one_hot_labels = np.zeros([num_files, num_labels], dtype=bool)
        for row in range(num_files):
            one_hot_labels[row, diagnosis_labels[row]] = True
        return one_hot_labels

    def normalise(self, mean = None, chol_cov = None):
        if mean is None:
            mean = self._input_mean
        if chol_cov is None:
            chol_cov = self._input_chol_cov
        num_files = self._input_data.shape[0]
        centred_data = self._input_data - mean
        for i in range(num_files):
            centred_data[i, self._sequence_lengths[i]:] = np.zeros((self._max_length - self._sequence_lengths[i], self._data_dim))
        scaled_centred_data = np.einsum('ijk,kl->ijl', centred_data, np.linalg.inv(chol_cov).T)
        self._normalised_input_data = scaled_centred_data
    
    def random_shuffle(self):
        shuffle_order = np.random.permutation(self._input_data.shape[0])
        self.shuffle(shuffle_order)

    def shuffle(self, shuffle_order):
        self._normalised_input_data = self._normalised_input_data[shuffle_order]
        self._input_data = self._input_data[shuffle_order]
        self._diagnosis_labels = self._diagnosis_labels[shuffle_order]
        self._output_data = self._output_data[shuffle_order]
        self._sequence_lengths = self._sequence_lengths[shuffle_order]
        self._data_order = self._data_order[shuffle_order]
    
    def feed_dict(self, model, keep_prob = 1.0, normalise = False, idx = None):
        if isinstance(idx, int):
            idx = [idx]
        if normalise:
            inp = np.squeeze(self._normalised_input_data[idx])
        else:
            inp = np.squeeze(self._input_data[idx])
        if isinstance(model, DiagnosisPrediction):
            out = np.squeeze(self._diagnosis_labels[idx])
        else:
            out = np.squeeze(self._output_data[idx])
        if (idx is not None) and (len(idx) == 1):
            inp = np.reshape(inp, (1, ) + inp.shape)
            out = np.reshape(out, (1, ) + out.shape)
        if isinstance(model, LSTM):
            feed_dict = {model.data: inp, model.targets: out, model.sequence_lengths: np.squeeze(self._sequence_lengths[idx]),
                model.input_keep_prob: keep_prob, model.output_keep_prob: keep_prob, model.state_keep_prob: keep_prob}
        else:
            feed_dict = {model.data: inp, model.targets: out, model.keep_prob: keep_prob}
        return feed_dict

    def input_data(self, normalise = False):
        if normalise:
            return self._normalised_input_data
        else:
            return self._input_data

    def targets(self, diag_pred = False):
        if diag_pred:
            return self._diagnosis_labels
        else:
            return self._output_data

    @property
    def data_dim(self):
        return self._data_dim

    @property
    def target_dim(self):
        return self._target_dim

    @property
    def data_order(self):
        return self._data_order

    @property
    def input_mean(self):
        return self._input_mean

    @property
    def input_chol_cov(self):
        return self._input_chol_cov

    @property
    def normalised_input_data(self):
        return self._normalised_input_data

    @property
    def sequence_lengths(self):
        return self._sequence_lengths

    @property
    def output_data(self):
        return self._output_data

    @property
    def diagnosis_labels(self):
        return self._diagnosis_labels

