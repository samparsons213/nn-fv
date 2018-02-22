import numpy as np
from pathlib2 import Path
from DataClass import Data
from PseudoDataClass import PseudoData

data_dim = 4
target_dim = 2
training_files = []
training_diagnoses = []
testing_files = []
testing_diagnoses = []
training_rate = 0.5
n_trials = 12
n_ctrl = 20
n_ad = 24
n_pca = 6
max_length = 600

def train_or_test(training_data, testing_data, data_object):
    if np.random.rand() < training_rate:
        training_data.append(data_object)
    else:
        testing_data.append(data_object)
initial_training_data = []
initial_testing_data = []
diagnosis_labels = [0, 1, 1]
#diagnosis_labels = [0, 1, 2]
diag_labels_dim = max(diagnosis_labels) + 1
diagnosis_labels_tmp = [diagnosis_labels[0]]*n_trials
for person in range(1, n_ctrl+1):
    file_names = []
    file_root = "./d5/dim4/ctrl_{:d}_trial_".format(person)
    for trial in range(1, n_trials+1):
        file_path = Path(file_root+"{:d}.csv".format(trial))
        file_names.append(file_path.as_posix())
    train_or_test(initial_training_data, initial_testing_data, Data(file_names, diagnosis_labels_tmp, data_dim, target_dim, diag_labels_dim, max_length))

diagnosis_labels_tmp = [diagnosis_labels[1]]*n_trials
for person in range(1, n_ad+1):
    file_names = []
    file_root = "./d5/dim4/ad_{:d}_trial_".format(person)
    for trial in range(1, n_trials+1):
        file_path = Path(file_root+"{:d}.csv".format(trial))
        file_names.append(file_path.as_posix())
    train_or_test(initial_training_data, initial_testing_data, Data(file_names, diagnosis_labels_tmp, data_dim, target_dim, diag_labels_dim, max_length))

diagnosis_labels_tmp = [diagnosis_labels[2]]*n_trials
for person in range(1, n_pca+1):
    file_names = []
    file_root = "./d5/dim4/pca_{:d}_trial_".format(person)
    for trial in range(1, n_trials+1):
        file_path = Path(file_root+"{:d}.csv".format(trial))
        file_names.append(file_path.as_posix())
    train_or_test(initial_training_data, initial_testing_data, Data(file_names, diagnosis_labels_tmp, data_dim, target_dim, diag_labels_dim, max_length))

n_training_files = len(initial_training_data)
max_td_length = max([sum(data.sequence_lengths) for data in initial_training_data])
max_length = max_td_length
indata = np.zeros([n_training_files, max_td_length, data_dim])
outdata = np.zeros([n_training_files, max_td_length, target_dim])
diagnosis_labels = np.zeros([n_training_files, initial_training_data[0].diagnosis_labels.shape[1]], dtype=bool)
sl = np.zeros(n_training_files, dtype=int)
ctr = 0
for data in initial_training_data:
    start = 0
    sl_i = data.sequence_lengths
    sl[ctr] = sum(sl_i)
    diagnosis_i = data.diagnosis_labels[0]
    for trial in range(n_trials):
        stop = start + sl_i[trial]
        indata[ctr, start:stop] = np.reshape(data.input_data()[trial, :sl_i[trial]], [1, sl_i[trial], data_dim])
        outdata[ctr, start:stop] = np.reshape(data.output_data[trial, :sl_i[trial]], [1, sl_i[trial], target_dim])
        diagnosis_labels[ctr] = diagnosis_i
        start = stop
    ctr += 1
training_data = PseudoData(indata, outdata, diagnosis_labels, sl)

n_testing_files = len(initial_testing_data)
max_td_length = max([sum(data.sequence_lengths) for data in initial_testing_data])
max_length = max([max_length, max_td_length])
indata = np.zeros([n_testing_files, max_td_length, data_dim])
outdata = np.zeros([n_testing_files, max_td_length, target_dim])
diagnosis_labels = np.zeros([n_testing_files, initial_testing_data[0].diagnosis_labels.shape[1]], dtype=bool)
sl = np.zeros(n_testing_files, dtype=int)
ctr = 0
for data in initial_testing_data:
    start = 0
    sl_i = data.sequence_lengths
    sl[ctr] = sum(sl_i)
    diagnosis_i = data.diagnosis_labels[0]
    for trial in range(n_trials):
        stop = start + sl_i[trial]
        indata[ctr, start:stop] = np.reshape(data.input_data()[trial, :sl_i[trial]], [1, sl_i[trial], data_dim])
        outdata[ctr, start:stop] = np.reshape(data.output_data[trial, :sl_i[trial]], [1, sl_i[trial], target_dim])
        diagnosis_labels[ctr] = diagnosis_i
        start = stop
    ctr += 1
testing_data = PseudoData(indata, outdata, diagnosis_labels, sl)
testing_data.normalise(training_data.input_mean, training_data.input_chol_cov)



