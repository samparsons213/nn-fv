import datetime
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from pathlib2 import Path
from timeit import default_timer as timer
from DataClass import Data
from LSTMModel import LSTM

def get_log_sigma0(data):
    '''
    Returns an easy to compute initial estimate for
    the log of the mle of sigma for a Gaussian kernel
    density estimator defined by data and evaluated
    at each point, with sigma not being recomputed for 
    each data point
    
    data:        [N by D] dataset
    '''
    row_diffs = np.reshape(data, [data.shape[0], 1, data.shape[-1]]) - np.reshape(data, (1,) + data.shape)
    n_diffs = row_diffs.shape[0] * (row_diffs.shape[0]-1)
    return np.log(np.abs(row_diffs).sum((0, 1)) / n_diffs)

def neg_log_kde(data, log_sigma, jac=False):
    '''
    evaluates the negative log-likelihood of unseen
    data, defined by a Gaussian kernel density estimator
    centred at each point in kde_data, with common 
    diagonal convariance matrix diag(exp(log_sigma)). 
    sigma is not recomputed for each data point
    
    data:        [N by D] dataset
    log_sigma:   [D] parameter array
    jac:         boolean for returning grad vector
    '''
    sigma = np.exp(log_sigma)
    scaled_row_diffs = (np.reshape(data, [data.shape[0], 1, data.shape[-1]]) - 
                        np.reshape(data, (1,) + data.shape)) / sigma
    scaled_row_diffs_sq = scaled_row_diffs * scaled_row_diffs
    log_pdf = -(scaled_row_diffs_sq.sum(2)/2.0 + log_sigma.sum())
    np.fill_diagonal(log_pdf, -np.inf)
    max_vals = np.amax(log_pdf, axis=1, keepdims=True)
    shifted_log_pdf = log_pdf - max_vals
    fval = -(max_vals + np.log(np.exp(shifted_log_pdf).sum(1, keepdims=True))).sum()
    if not jac:
        return fval
    
    pdf = np.exp(log_pdf)
    grad_ija = (scaled_row_diffs_sq - 1) * np.reshape(pdf, pdf.shape + (1, ))
    s_grad_ia = grad_ija.sum(1, keepdims=True)
    s_pdf_i = np.reshape(pdf.sum(1), pdf.sum(1).shape + (1, 1))
    grad_ratio_ia = np.squeeze(s_grad_ia / s_pdf_i)
    fjac = -grad_ratio_ia.sum(0)
    return (fval, fjac)

def get_sigma_mle(data):
    '''
    Returns the mle of sigma for some data, using the data
    to define a Gaussian kernel density estimator. sigma is
    not recomputed for every data point
    
    data:        [N by D] dataset
    '''
    log_sigma0 = get_log_sigma0(data)
    minimizer = minimize(lambda ls: neg_log_kde(data, ls), log_sigma0, method='L-BFGS-B', jac=True)
    return np.exp(minimizer.x)

def get_log_sigma02(unseen_data, kde_data):
    '''
    Returns an easy to compute initial estimate for
    the log of the mle of sigma for a Gaussian kernel
    density estimator defined by kde_data and evaluated
    on unseen data
    
    unseen_data: [N by D] dataset
    kde_data:    [M by D] dataset
    '''
    N, D = unseen_data.shape
    M = kde_data.shape[0]
    row_diffs = np.reshape(unseen_data, (N, 1, D)) - np.reshape(kde_data, (1, M, D))
    return np.log(np.abs(row_diffs.mean(axis=(0, 1))))

def neg_log_kde2(unseen_data, kde_data, log_sigma, jac=False):
    '''
    evaluates the negative log-likelihood of unseen
    data, defined by a Gaussian kernel density estimator
    centred at each point in kde_data, with common 
    diagonal convariance matrix diag(exp(log_sigma))
    
    unseen_data: [N by D] dataset
    kde_data:    [M by D] dataset
    log_sigma:   [D] parameter array
    jac:         boolean for returning grad vector
    '''
    sigma = np.exp(log_sigma)
    N, D = unseen_data.shape
    M = kde_data.shape[0]
    scaled_row_diffs = (np.reshape(unseen_data, (N, 1, D)) -
                        np.reshape(kde_data, (1, M, D))) / sigma
    scaled_row_diffs_sq = scaled_row_diffs * scaled_row_diffs
    log_probs = -(log_sigma.sum() + scaled_row_diffs_sq.sum(axis=2)/2.0)
    max_vals = np.amax(log_probs, axis=1, keepdims=True)
    shifted_log_probs = log_probs - max_vals
    fval = -(np.log(np.exp(shifted_log_probs).sum(axis=1, keepdims=True)) + max_vals).sum()
    if not jac:
        return fval
    
    probs = np.exp(log_probs)
    grad_ija = (scaled_row_diffs_sq - 1) * np.reshape(probs, (N, M, 1))
    s_grad_ia = grad_ija.sum(axis=1)
    s_probs_i = np.reshape(probs.sum(axis=1), (N, 1))
    grad_ratio_ia = s_grad_ia / s_probs_i
    fjac = -grad_ratio_ia.sum(axis=0)
    return fval, fjac

def get_sigma_mle2(unseen_data, kde_data, retval=False):
    '''
    Returns the mle of sigma for unseen data, using kde_data
    to define a Gaussian kernel density estimator
    
    unseen_data: [N by D] dataset
    kde_data:    [M by D] dataset
    retval:      boolean for returning function val along with
                 its argmax
    '''
    log_sigma0 = get_log_sigma02(unseen_data, kde_data)
    minimizer = minimize(lambda ls: neg_log_kde2(unseen_data, kde_data, ls, True), log_sigma0,
                        method='L-BFGS-B', jac=True)
    mle = np.exp(minimizer.x)
    if retval:
        return mle, minimizer.fun
    return mle

def get_sigma_mle2_all(data):
    '''
    Returns the mle for sigma for each row in turn as the
    unseen data, with all other rows defining the Gaussian 
    kernel density estimator 
    
    data:    [N by D] dataset
    '''
    rows = range(data.shape[0])
    return np.array([get_sigma_mle2(data[[i]], data[[j for j in rows if j != i]]) for i in rows])

def add_file_to_list(file_list, file_path, diagnosis_list, diagnosis):
    file_list.append(file_path.as_posix())
    diagnosis_list.append(diagnosis)

max_length = 600
data_dim = 4
target_dim = 2
diagnosis_labels = [0, 1, 1] # considers tAD and PCA as one class
#diagnosis_labels = [0, 1, 2] # separating tAD and PCA
diag_labels_dim = max(diagnosis_labels) + 1
n_trials = 12
trials_list = range(1, n_trials+1)
#trials_list = np.random.randint(1, n_trials+1, [1]) # train on a random trial

training_diagnosis_label = 0
training_individual = 9
training_files_strings = ["./d5/dim4/ctrl_{:d}_trial_{:d}.csv".format(training_individual, i) for i in trials_list]
training_files_paths = [Path(file_string).as_posix() for file_string in training_files_strings]
training_diagnoses = [training_diagnosis_label] * len(trials_list)
training_data = Data(training_files_paths, training_diagnoses, data_dim, target_dim, diag_labels_dim, max_length)

validating_diagnosis_label = 1
validating_individual = 2
#validating_files_strings = ["./d5/dim4/ctrl_{:d}_trial_{:d}.csv".format(validating_individual, i) for i in trials_list] # validate on control
validating_files_strings = ["./d5/dim4/ad_{:d}_trial_{:d}.csv".format(validating_individual, i) for i in trials_list] # validate on tAD patient
validating_files_paths = [Path(file_string).as_posix() for file_string in validating_files_strings]
validating_diagnoses = [validating_diagnosis_label] * len(trials_list)
validating_data = Data(validating_files_paths, validating_diagnoses, data_dim, target_dim, diag_labels_dim, max_length)

n_ctrl = 20
n_ad = 24
n_pca = 6
testing_files_paths = []
testing_diagnoses = []
for trial in trials_list:
    diagnosis = diagnosis_labels[0]
    for person in range(1, n_ctrl+1):
        file_path = Path("./d5/dim4/ctrl_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            add_file_to_list(testing_files_paths, file_path, testing_diagnoses, diagnosis)
    
    diagnosis = diagnosis_labels[1]
    for person in range(1, n_ad+1):
        file_path = Path("./d5/dim4/ad_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            add_file_to_list(testing_files_paths, file_path, testing_diagnoses, diagnosis)
    
    diagnosis = diagnosis_labels[2]
    for person in range(1, n_pca+1):
        file_path = Path("./d5/dim4/pca_{:d}_trial_{:d}.csv".format(person, trial))
        if file_path.is_file():
            add_file_to_list(testing_files_paths, file_path, testing_diagnoses, diagnosis)

testing_data = Data(testing_files_paths, testing_diagnoses, data_dim, target_dim, diag_labels_dim, max_length)
testing_data.normalise(training_data.input_mean, training_data.input_chol_cov)
max_length = max(training_data.input_data().shape[1], testing_data.input_data().shape[1])

learning_rate = tf.placeholder(tf.float32, ())
adam_epsilon = 1e-5
adam_beta1 = 0.9
adam_beta2 = 0.999
use_gru = False
num_hidden = [4]
model = LSTM(num_hidden, data_dim, target_dim, max_length, use_gru, diag_labels_dim, adam_epsilon)
keep_prob = 0.5
normalise = True

minimize_entropy = False
if minimize_entropy:
    min_text = 'min'
    entropy_op = model.entropy
    min_entropy_op = model.entropy_minimize
else:
    min_text = 'max'
    #entropy_op = model.negative_entropy # doesn't recompute sigma for each observation
    entropy_op = model.negative_entropy2 # recomputes sigma for each observation
    #entropy_op = model.standardised_neg_entropy # normalises each dimension of lstm output to avoid degenerate rescaling
    min_entropy_op = model.negative_entropy_minimize

saver = tf.train.Saver(max_to_keep=n_trials)
sess = tf.Session()
init_op = tf.global_variables_initializer()
checkpoint_folder = './Results/' + '_'.join(str(datetime.datetime.now()).split(' ')) + '/'
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
initialized_variables_save_path = checkpoint_folder + 'initial.ckpt'

num_epochs = 5000
entropies = np.zeros([n_trials, num_epochs])
validating_entropies = np.zeros([n_trials, num_epochs])
optimal_entropies = np.zeros([n_trials])
optimal_validating_entropies = np.zeros([n_trials])

plot_flag = False
write_data = True
learn_from_random_initialisation = False

tv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn')
grads_op = tf.gradients(entropy_op, tv)
no_dropout_feed_dict = training_data.feed_dict(model, 1.0, normalise, 0)
no_dropout_feed_dict[model.entropy_idx] = 0
no_dropout_feed_dict[model.sigma] = np.ones([num_hidden[-1]])

sess.run(init_op)
no_dropout_feed_dict[model.sigma2] = np.ones([model.batch_length, num_hidden[-1]])
model_vars = sess.run(tv, no_dropout_feed_dict)
n_grad_elements = sum([var_i.size for var_i in model_vars])
n_patients = len(testing_files_paths) / n_trials
all_grads = np.zeros([n_trials, n_patients, n_grad_elements])

lambda0 = 0.001
alpha = 1.0
decay_speed = 10
restore_previous_session = False
previous_folder = './Results/2018-02-02_00:50:04.156362/' # needed if restore_previous_session = True
checkpoint_file = 'nh4_trial{:d}_ne2000_eps1.0e-08_max.ckpt' # needed if restore_previous_session = True
start_time = timer()
for entropy_idx in range(len(trials_list)):
    print "\nTrial {:d}".format(trials_list[entropy_idx])
    previous_checkpoint = previous_folder + checkpoint_file.format(trials_list[entropy_idx])
    save_path = checkpoint_folder + 'nh{:d}_trial{:d}_ne{:d}_eps{:2.1e}_{}.ckpt'.format(num_hidden[-1], trials_list[entropy_idx], 
                                                                                        num_epochs, adam_epsilon, min_text)
    no_dropout_feed_dict = training_data.feed_dict(model, 1.0, normalise, entropy_idx)
    no_dropout_feed_dict[model.entropy_idx] = 0
    validating_feed_dict = validating_data.feed_dict(model, 1.0, normalise, entropy_idx)
    validating_feed_dict[model.entropy_idx] = 0
    minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon)
    minimize_op = minimizer.minimize(entropy_op)
    initialized = False
    while not initialized:
        sess.run(tf.global_variables_initializer())
        if restore_previous_session:
            saver.restore(sess, previous_checkpoint)
        model_output2 = sess.run(model.standardised_vals, no_dropout_feed_dict)
        sigma2 = get_sigma_mle2_all(model_output2)
        no_dropout_feed_dict[model.sigma2] = sigma2
        validating_feed_dict[model.sigma2] = sigma2
        initial_entropy = sess.run(entropy_op, no_dropout_feed_dict)
        initial_validating_entropy = sess.run(entropy_op, validating_feed_dict)
        if not (np.isnan(initial_entropy) or np.isnan(initial_validating_entropy)):
            initialized = True
    optimal_entropies[entropy_idx] = initial_entropy
    optimal_validating_entropies[entropy_idx] = initial_validating_entropy
    restore_path = saver.save(sess, save_path)
    print "initial_save to " + restore_path
    print "\nThe initial minimum (negative) entropy for trial {:d} is {:4.2f}\n".format(trials_list[entropy_idx], 
                                                                                        optimal_entropies[entropy_idx])
    print "\nThe initial minimum (negative) validating entropy for trial {:d} is {:4.2f}\n".format(trials_list[entropy_idx], 
                                                                                                   optimal_validating_entropies[entropy_idx])
    dropout_feed_dict = training_data.feed_dict(model, keep_prob, normalise, entropy_idx)
    dropout_feed_dict[model.entropy_idx] = 0
    ctr1 = 0
    ctr2 = 0
    lr = lambda0
    for epoch in range(num_epochs):
        print "\tTrial {:d} epoch {:d}".format(trials_list[entropy_idx], epoch+1)
        model_output = np.reshape(sess.run(model.val, no_dropout_feed_dict), [-1, num_hidden[-1]])[:, :training_data.sequence_lengths[entropy_idx]]
        #model_output = sess.run(model.standardised_vals, no_dropout_feed_dict) # use if optimising standardised entropy
        sigma2 = get_sigma_mle2_all(model_output)
        dropout_feed_dict[model.sigma2] = sigma2
        if (epoch % decay_speed) == 0:
            lr = lr * alpha
        dropout_feed_dict[learning_rate] = lr
        no_dropout_feed_dict[model.sigma2] = sigma2
        validating_feed_dict[model.sigma2] = sigma2
        sess.run(minimize_op, dropout_feed_dict)
        entropies[entropy_idx, epoch] = sess.run(entropy_op, no_dropout_feed_dict)
        validating_entropies[entropy_idx, epoch] = sess.run(entropy_op, validating_feed_dict)
        print "\t\tent: {:4.2f}".format(entropies[entropy_idx, epoch])
        print "\t\tval ent: {:4.2f}".format(validating_entropies[entropy_idx, epoch])
        if entropies[entropy_idx, epoch] < optimal_entropies[entropy_idx]:
            ctr1 += 1
            print "\t\tnew best entropy for trial {:d} number {:d}".format(trials_list[entropy_idx], ctr1)
            optimal_entropies[entropy_idx] = entropies[entropy_idx, epoch]
        if validating_entropies[entropy_idx, epoch] < optimal_validating_entropies[entropy_idx]:
            ctr2 += 1
            print "\t\tnew best validating entropy for trial {:d} number {:d}".format(trials_list[entropy_idx], ctr2)
            optimal_validating_entropies[entropy_idx] = validating_entropies[entropy_idx, epoch]
            restore_path = saver.save(sess, save_path)
    print "\n\tThe minimum (negative) entropy achieved for trial {:d} was {:4.2f}\n".format(trials_list[entropy_idx], 
                                                                                            optimal_entropies[entropy_idx])
    print "\n\tThe minimum (negative) validating_entropy achieved for trial {:d} was {:4.2f}\n".format(trials_list[entropy_idx], 
                                                                                                       optimal_validating_entropies[entropy_idx])
    saver.restore(sess, save_path)
    model_output = sess.run(model.standardised_vals, no_dropout_feed_dict)
    sigma2 = get_sigma_mle2_all(model_output)
    first_patient_idx = entropy_idx * n_patients
    for patient_idx in range(n_patients):
        fd_idx = first_patient_idx + patient_idx
        feed_dict = testing_data.feed_dict(model, 1.0, normalise, fd_idx)
        feed_dict[model.entropy_idx] = 0
        feed_dict[model.sigma2] = sigma2
        grads = sess.run(grads_op, feed_dict)
        all_grads[entropy_idx, patient_idx] = np.concatenate([np.reshape(grad_i, [-1]) for grad_i in grads])
    if plot_flag:
        plt.plot(entropies[entropy_idx])
        plt.plot(validating_entropies[entropy_idx])
        plt.show()
    if write_data:
        all_grads_filename = Path(checkpoint_folder + "all_grads_nh{:d}_trial{:d}_ne{:d}_eps{:2.1e}_{}.csv".format(num_hidden[-1], 
                                  trials_list[entropy_idx], num_epochs, adam_epsilon, min_text)).as_posix()
        with open(all_grads_filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_grads[entropy_idx])

if write_data:
    entropies_filename = checkpoint_folder + "entropies_nh{:d}_ne{:d}_eps{:2.1e}_{}.csv".format(num_hidden[-1], num_epochs, adam_epsilon, min_text)
    with open(entropies_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(np.reshape(entropies, [-1, num_epochs]))
    validating_entropies_filename = (checkpoint_folder + 
                                    "validating_entropies_nh{:d}_ne{:d}_eps{:2.1e}_{}.csv".format(num_hidden[-1], num_epochs, adam_epsilon, min_text))
    with open(validating_entropies_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(np.reshape(validating_entropies, [-1, num_epochs]))
stop_time = timer()

print "Total time taken for optimisation and writing to disk was {:4.2f} seconds".format(stop_time - start_time)

sess.close()