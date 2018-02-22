import tensorflow as tf
from LSTMModel import LSTM
from Conv1DModel import Conv1D
from DiagnosisModel import DiagnosisPrediction
from PseudoDataClass import PseudoData
import numpy as np
import matplotlib.pyplot as plt
import utils
from modelling_functions import train_model, train_lstm_diag

import load_data_script as ld
#import load_data_by_person_script as ld
use_gru = False

''' Use this code block if using the LSTM model for movement analysis '''
#num_hidden = [25]
#model = LSTM(num_hidden, ld.data_dim, ld.target_dim, ld.max_length, use_gru)

''' Use this code block if using the Conv1D model for diagnosis prediction '''
filter_size = 10
num_filters = [32, 64]
fc1_size = 128
all_sequence_lengths = np.concatenate((ld.training_data.sequence_lengths, ld.testing_data.sequence_lengths))
window_size = max(all_sequence_lengths)
conv_target_dim = ld.diag_labels_dim
model = Conv1D(filter_size, num_filters, fc1_size, window_size, ld.data_dim, conv_target_dim)

#intent_num_hidden = [num_filters[0] * 2, num_filters[0] * 4]
intent_num_hidden = [num_filters[0] * 2]
intent_data_dim = num_filters[0]
intent_model = LSTM(intent_num_hidden, intent_data_dim, ld.target_dim, ld.max_length, use_gru)

keep_prob = 0.5

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

num_training_files = ld.training_data.input_data().shape[0]
batch_size = 15
num_batches = int(num_training_files / batch_size) + int(bool(num_training_files % batch_size))
normalise = True
feed_dict = ld.testing_data.feed_dict(model, 1.0, normalise)
diag_pred = isinstance(model, DiagnosisPrediction)
incorrect = sess.run(model.error,feed_dict)
best_test_error = utils.mean_error(incorrect, diag_pred, ld.testing_data.sequence_lengths)
num_epochs = 500
print_flag = True
train_errors, test_errors = train_model(ld.training_data, ld.testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)
#train_errors, test_errors = train_lstm_diag(ld.training_data, ld.testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)
best_test_error = min(best_test_error, min(test_errors))

plt.plot(train_errors)
plt.plot(test_errors)
plt.show()
print 'Best test error was {:4.2f}%'.format(100*best_test_error)

vals, preds = utils.vals_preds(model, ld.testing_data, sess, normalise)
#vals = ld.testing_data.diagnosis_labels
#max_val = vals.shape[1]
#feed_dict = ld.testing_data.feed_dict(model, 1.0, normalise)
#feed_dict[model.diag_labels] = ld.testing_data.diagnosis_labels
#preds = sess.run(model.diag_prediction, feed_dict)
#preds = np.concatenate([np.reshape(preds == i, [preds.size, 1]).astype(int) for i in range(max_val)], axis = 1)
cm = utils.calc_cm(vals, preds)
utils.print_cm(cm)
print "Matthew's correlation coeffecient: {:4.3f}".format(utils.mcc(vals, preds))
print "Youden's J statistic: {:4.3f}".format(utils.youdens_j(cm))
print "F1 score: {:4.3f}".format(utils.f1(cm))

intent_testing_input_data = sess.run(model.conv_layers[0], feed_dict)
intent_testing_data = PseudoData(intent_testing_input_data, ld.testing_data.output_data, ld.testing_data.diagnosis_labels, 
    ld.testing_data.sequence_lengths, False, ld.testing_data.data_order)
feed_dict = ld.training_data.feed_dict(model, 1.0, normalise)
intent_training_input_data = sess.run(model.conv_layers[0], feed_dict)
intent_training_data = PseudoData(intent_training_input_data, ld.training_data.output_data, ld.training_data.diagnosis_labels, 
    ld.training_data.sequence_lengths, False, ld.training_data.data_order)
intent_testing_data.normalise(intent_training_data.input_mean, intent_training_data.input_chol_cov)
num_epochs = 500
#train_errors, test_errors = train_model(intent_training_data, intent_testing_data, intent_model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)
train_errors, test_errors = train_lstm_diag(intent_training_data, intent_testing_data, intent_model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)

if num_epochs > 0:
    best_test_error = min(best_test_error, min(test_errors))
else:
    best_test_error = 0

plt.plot(train_errors)
plt.plot(test_errors)
plt.show()
print 'Best test error was {:4.2f}%'.format(100*best_test_error)

#vals, preds = utils.vals_preds(intent_model, intent_testing_data, sess, normalise)
vals = intent_testing_data.diagnosis_labels
max_val = vals.shape[1]
feed_dict = intent_testing_data.feed_dict(intent_model, 1.0, normalise)
feed_dict[intent_model.diag_labels] = intent_testing_data.diagnosis_labels
preds = sess.run(intent_model.diag_prediction, feed_dict)
preds = np.concatenate([np.reshape(preds == i, [preds.size, 1]).astype(int) for i in range(max_val)], axis = 1)
cm = utils.calc_cm(vals, preds)
utils.print_cm(cm)
print "Matthew's correlation coeffecient: {:4.3f}".format(utils.mcc(vals, preds))
print "Youden's J statistic: {:4.3f}".format(utils.youdens_j(cm))
print "F1 score: {:4.3f}".format(utils.f1(cm))

sess.close()