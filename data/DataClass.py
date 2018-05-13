import csv

import numpy as np

from DataBaseClass import DataBase
from DiagnosisModel import DiagnosisPrediction
from models.LSTMModel import LSTM


class Data(DataBase):
    def __init__(self, filenames, diagnosis_labels, data_dim = 4, target_dim = 2, diag_labels_dim = 2, max_length = 3000):
        self._data_dim = data_dim
        self._target_dim = target_dim
        self._diag_labels_dim = diag_labels_dim
        self._filenames = filenames
        self._data_order = np.array(range(len(filenames)))
        self._max_length = max_length
        self._input_data, self._output_data, self._sequence_lengths = self.read_files(filenames)
        self._max_length = max(self._sequence_lengths)
        self._diagnosis_labels = self.transform_diagnosis_labels(diagnosis_labels, diag_labels_dim)
        self._input_mean, self._input_chol_cov = self.compute_mean_chol_cov()
        self.normalise()
        
    def compute_mean_chol_cov(self):
        all_input_data = np.concatenate([self._input_data[i, :self.sequence_lengths[i]] for i in range(self._input_data.shape[0])])
        input_mean =  np.mean(all_input_data, axis=0)
        input_cov = np.cov(all_input_data, rowvar=False)
        chol_cov = np.linalg.cholesky(input_cov)
        return input_mean, chol_cov

    def normalise_input_data(self, mean = None, chol_cov = None):
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

    def read_files(self, filenames):
        num_files = len(filenames)
        input_data = np.zeros([num_files, self.max_length, self.data_dim])
        output_data = np.zeros([num_files, self.max_length, self.target_dim], dtype=bool)
        sequence_lengths = np.zeros(num_files, dtype=int)
        for file_idx in range(num_files):
            with open(filenames[file_idx]) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    temp_input = np.array([float(item) for item in row])
                    input_data[file_idx, sequence_lengths[file_idx]] = temp_input[:self.data_dim]
                    output_data[file_idx, sequence_lengths[file_idx], int(temp_input[-1])] = True
                    sequence_lengths[file_idx] += 1
        max_sl = max(sequence_lengths)
        input_data = input_data[:, :max_sl]
        output_data = output_data[:, :max_sl]
        return input_data, output_data, sequence_lengths

    def shuffle(self, shuffle_order):
        DataBase.shuffle(self, shuffle_order)
        self._filenames = [self._filenames[i] for i in shuffle_order]
    
    def output_labels_one_file(self, file_num):
        return [self.output_data[file_num][i][1] for i in range(self.sequence_lengths[file_num])]

    def one_file_into_windows(self, file_num, window_size, normalise = False, disjoint = True):
        file_num = file_num % self._input_data.shape[0]
        if disjoint:
            num_windows = int(self._sequence_lengths[file_num] / window_size)
            windowed_data = np.reshape(self.input_data(normalise)[file_num, :num_windows*window_size], [num_windows, window_size, self._data_dim])
            windowed_targets = np.reshape(self._output_data[file_num, :num_windows*window_size], [num_windows, window_size, self._target_dim])
        else:
            num_windows = self._sequence_lengths[file_num] + 1 - window_size
            windowed_data = np.array([self.input_data(normalise)[file_num, i:i+window_size] for i in range(num_windows)])
            windowed_targets = np.array([self._output_data[file_num, i:i+window_size] for i in range(num_windows)])
        return windowed_data, windowed_targets

    def feed_dict(self, model, keep_prob = 1.0, normalise = False, idx = None):
        if idx is None:
            batch_size = len(self._filenames)
            idx = range(batch_size)
        if isinstance(idx, int):
            idx = [idx]
        batch_size = len(idx)
        inp = np.zeros([batch_size, model.batch_length, model.data_dim])
        sl = min(model.batch_length, max(self._sequence_lengths))
        if normalise:
            inp[:, :sl] = self._normalised_input_data[idx, :sl]
        else:
            inp[:, :sl] = self._input_data[idx, :sl]
        out = np.zeros([batch_size, model.batch_length, model.target_dim])

        if isinstance(model, DiagnosisPrediction):
            out[:, :sl] = self._diagnosis_labels[idx, :sl]
        else:
            out[:, :sl] = self._output_data[idx, :sl]

        if isinstance(model, LSTM):
            sl = self._sequence_lengths[idx]
            feed_dict = {model.data: inp, model.targets: out, model.sequence_lengths: sl,
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
    def filenames(self):
        return self._filenames

    @property
    def max_length(self):
        return self._max_length

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

