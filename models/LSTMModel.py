import tensorflow as tf
import numpy as np
import abc
from utils import lazy_property
from MovementAnalysis import MovementAnalysis

class LSTM(MovementAnalysis):

    def __init__(self, num_hidden = 24, data_dim = None, target_dim = None, max_length = None, use_gru = False, diag_dim = 2, epsilon = 3e-1):
        self._num_hidden = num_hidden
        if isinstance(num_hidden, int):
            self._num_layers = 1
            self._num_hidden = [self._num_hidden]
        else:
            self._num_layers = len(num_hidden)
        self._use_gru = use_gru
        self._epsilon = epsilon
        MovementAnalysis.__init__(self, data_dim, target_dim, max_length)
        self._diag_dim = diag_dim
        self.diag_minimize
        self.entropy_minimize
        self.negative_entropy_minimize
        self.standardised_neg_entropy
        self.input_keep_prob
        self.output_keep_prob
        self.state_keep_prob
    
    @lazy_property
    def input_keep_prob(self):
        return tf.placeholder(tf.float32)

    @lazy_property
    def output_keep_prob(self):
        return tf.placeholder(tf.float32)

    @lazy_property
    def state_keep_prob(self):
        return tf.placeholder(tf.float32)

    @lazy_property
    def multi_rnn_cell(self):
        if self._num_layers == 1:
            cell = self.single_cell(self._num_hidden[0])
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell(self._num_hidden[i]) for i in range(self._num_layers)])
        return cell

    @lazy_property
    def dynamic_rnn(self):
        val, state = tf.nn.dynamic_rnn(self.multi_rnn_cell, self.data, dtype=tf.float32, sequence_length=self.sequence_lengths)
        return val, state

    @lazy_property
    def val(self):
        return self.dynamic_rnn[0]

    @lazy_property
    def state(self):
        return self.dynamic_rnn[1]

    @lazy_property
    def prediction_logits(self):
        weight = tf.Variable(tf.truncated_normal([self._num_hidden[-1], int(self.targets.get_shape()[2])], stddev=0.25))
        bias = tf.Variable(tf.constant(0.1, shape=[self.targets.get_shape()[2]]))
        return tf.einsum('ijk,kl->ijl', self.val, weight) + bias        
    
    @lazy_property
    def diagnosis_logits(self):
        diag_weight = tf.Variable(tf.truncated_normal([self._num_hidden[-1], int(self._diag_dim)], stddev=0.25))
        diag_bias = tf.Variable(tf.constant(0.1, shape=[self._diag_dim]))
        mean_vals = tf.reduce_sum(self.val, 1) / tf.reshape(tf.cast(self.sequence_lengths, tf.float32), [-1, 1])
        return tf.matmul(mean_vals, diag_weight) + diag_bias

    @lazy_property
    def diag_labels(self):
        return tf.placeholder(tf.float32, [None, self._diag_dim])

    @lazy_property
    def diag_minimize(self):
        cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.diag_labels, logits=self.diagnosis_logits))
        diag_optimizer = tf.train.AdamOptimizer(epsilon=self._epsilon)
        return diag_optimizer.minimize(cross_ent)

    @lazy_property
    def diag_prediction(self):
        return tf.argmax(self.diagnosis_logits, 1)

    @lazy_property
    def diag_error(self):
        return tf.not_equal(tf.argmax(self.diag_labels, 1), self.diag_prediction)

    @lazy_property
    def sigma(self):
        return tf.placeholder(tf.float32, [self._num_hidden[-1]])

    @lazy_property
    def sigma2(self):
        return tf.placeholder(tf.float32, [self._batch_length, self._num_hidden[-1]])

    @lazy_property
    def entropy_idx(self):
        return tf.placeholder(tf.int32, ())

    @lazy_property
    def negative_entropy(self):
        # sigma is not recomputed for each data point
        scaled_row_diffs = (tf.reshape(self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]], 
                            [self.sequence_lengths[self.entropy_idx], 1, self._num_hidden[-1]]) - 
                            tf.reshape(self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]], 
                            [1, self.sequence_lengths[self.entropy_idx], self._num_hidden[-1]])) / self.sigma
        scaled_row_diffs_sq = scaled_row_diffs * scaled_row_diffs
        log_probs = -(tf.reduce_sum(tf.log(self.sigma)) + tf.reduce_sum(scaled_row_diffs_sq, axis=2)/2.0)
        log_probs = tf.matrix_set_diag(log_probs, tf.fill([self.sequence_lengths[0]], -np.inf))
        return tf.reduce_mean(tf.reduce_logsumexp(log_probs, axis=1)) # ignored additive constant log(self.sequence_lengths[0])
    
    @lazy_property
    def negative_entropy2(self):
        #sigma is recomputed for each data point (uses sigma2 placeholder)
        scaled_row_diffs = ((tf.reshape(self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]], 
                            [self.sequence_lengths[self.entropy_idx], 1, self._num_hidden[-1]]) - 
                            tf.reshape(self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]], 
                            [1, self.sequence_lengths[self.entropy_idx], self._num_hidden[-1]])) / 
                            self.sigma2[:self.sequence_lengths[self.entropy_idx]])
        scaled_row_diffs_sq = scaled_row_diffs * scaled_row_diffs
        log_probs = -(tf.reduce_sum(tf.log(self.sigma2[:self.sequence_lengths[self.entropy_idx]]), axis=1, keepdims=True) + 
                      tf.reduce_sum(scaled_row_diffs_sq, axis=2)/2.0)
        log_probs = tf.matrix_set_diag(log_probs, tf.fill([self.sequence_lengths[0]], -np.inf))
        return tf.reduce_mean(tf.reduce_logsumexp(log_probs, axis=1)) # ignored additive constant log(self.sequence_lengths[0])
        

    @lazy_property
    def standardised_vals(self):
        diffs = (self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]] - 
                tf.reduce_mean(self.val[self.entropy_idx, :self.sequence_lengths[self.entropy_idx]], axis=1, keepdims=True))
        stds = tf.sqrt(tf.reduce_mean(diffs * diffs, axis=1, keepdims=True))
        return diffs / stds

    @lazy_property
    def standardised_neg_entropy(self):
        #sigma is recopmuted for ecah data poiint (uses sigma2 placeholder)
        scaled_row_diffs = ((tf.reshape(self.standardised_vals, [self.sequence_lengths[self.entropy_idx], 1, self._num_hidden[-1]]) -
                            tf.reshape(self.standardised_vals, [1, self.sequence_lengths[self.entropy_idx], self._num_hidden[-1]])) /
                            self.sigma2[:self.sequence_lengths[self.entropy_idx]])
        scaled_row_diffs_sq = scaled_row_diffs * scaled_row_diffs
        log_probs = -(tf.reduce_sum(tf.log(self.sigma2[:self.sequence_lengths[self.entropy_idx]]), axis=1, keepdims=True) + 
                      tf.reduce_sum(scaled_row_diffs_sq, axis=2)/2.0)
        log_probs = tf.matrix_set_diag(log_probs, tf.fill([self.sequence_lengths[0]], -np.inf))
        return tf.reduce_mean(tf.reduce_logsumexp(log_probs, axis=1)) # ignored additive constant log(self.sequence_lengths[0])

    @lazy_property
    def negative_entropy_minimize(self):
        neg_ent_minimizer = tf.train.AdamOptimizer(epsilon=self._epsilon)
        return neg_ent_minimizer.minimize(self.negative_entropy)

    @lazy_property
    def entropy(self):
        return tf.negative(self.negative_entropy)

    @lazy_property
    def entropy_minimize(self):
        ent_minimizer = tf.train.AdamOptimizer(epsilon=self._epsilon)
        return ent_minimizer.minimize(self.entropy)

    def single_cell(self, num_hidden):
        if self._use_gru:
            cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            return tf.nn.rnn_cell.DropoutWrapper(cell,
                self.input_keep_prob, self.output_keep_prob, self.state_keep_prob, False)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            return tf.nn.rnn_cell.DropoutWrapper(cell,
                self.input_keep_prob, self.output_keep_prob, self.state_keep_prob, False)

