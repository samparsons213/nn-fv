import numpy as np
from pathlib2 import Path
from DataClass import Data

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
def train_or_test(training_files, training_diagnoses, testing_files, testing_diagnoses, training_rate, file_path, diagnosis):
    if np.random.rand() < training_rate:
        training_files.append(file_path.as_posix())
        training_diagnoses.append(diagnosis)
    else:
        testing_files.append(file_path.as_posix())
        testing_diagnoses.append(diagnosis)

diagnosis_labels = [0, 1, 1]
#diagnosis_labels = [0, 1, 2]
diag_labels_dim = max(diagnosis_labels) + 1
for trial in range(1, n_trials+1):
    diagnosis = diagnosis_labels[0]
    for person in range(1, n_ctrl+1):
        file_path = Path("./d5/dim4/ctrl_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            train_or_test(training_files, training_diagnoses, testing_files, testing_diagnoses, training_rate, file_path, diagnosis)
    
    diagnosis = diagnosis_labels[1]
    for person in range(1, n_ad+1):
        file_path = Path("./d5/dim4/ad_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            train_or_test(training_files, training_diagnoses, testing_files, testing_diagnoses, training_rate, file_path, diagnosis)
    
    diagnosis = diagnosis_labels[2]
    for person in range(1, n_pca+1):
        file_path = Path("./d5/dim4/pca_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            train_or_test(training_files, training_diagnoses, testing_files, testing_diagnoses, training_rate, file_path, diagnosis)

#training_files = ["d5ctrl_12_trial_1.csv", "d5ctrl_12_trial_2.csv", "d5ctrl_12_trial_3.csv"]
#testing_files = ["d5ctrl_12_trial_4.csv", "d5ctrl_12_trial_5.csv", "d5ctrl_12_trial_6.csv"]
#training_files = ["small_file.csv"]
#testing_files = ["small_file.csv", "small_file.csv"]

max_length = 600
training_data = Data(training_files, training_diagnoses, data_dim, target_dim, diag_labels_dim, max_length)
testing_data = Data(testing_files, testing_diagnoses, data_dim, target_dim, diag_labels_dim, max_length)
testing_data.normalise(training_data.input_mean, training_data.input_chol_cov)
max_length = max(training_data.input_data().shape[1], testing_data.input_data().shape[1])


