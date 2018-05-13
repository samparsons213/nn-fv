import abc
import tensorflow as tf
from utils import lazy_property

class PredictionModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dim = 2, batch_length = 3000):
        self._batch_length = batch_length
        self._data_dim = data_dim
        self.data
        self.sequence_lengths
        self.prediction
        self.minimize
        self.error

    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @property
    def batch_length(self):
        return self._batch_length

    @property
    def data_dim(self):
        return self._data_dim

    @abc.abstractproperty
    def target_dim(self):
        ''' Implement target_dim property here '''

    @abc.abstractproperty
    @lazy_property
    def prediction(self):
        ''' Implement predictions with particulars of target shape here '''

    @lazy_property
    def minimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.prediction_logits))
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(cross_entropy)

    @abc.abstractproperty
    @lazy_property
    def error(self):
        ''' Implement error with particulars of target shape here '''

    @lazy_property
    def confusion_matrix(self):
        return tf.confusion_matrix(self.cm_vals, self.cm_preds)

    @lazy_property
    def data(self):
        return tf.placeholder(tf.float32, [None, self._batch_length, self._data_dim])

    @abc.abstractproperty
    @lazy_property
    def targets(self):
        ''' Implement shape of targets placeholder here '''

    @lazy_property
    def sequence_lengths(self):
        return tf.placeholder(tf.int32, [None])

    @lazy_property
    def cm_vals(self):
        return tf.placeholder(tf.float32, [None])

    @lazy_property
    def cm_preds(self):
        return tf.placeholder(tf.float32, [None])

    @abc.abstractproperty
    @lazy_property
    def prediction_logits(self, parameter_list):
        ''' Model structure is implemented here '''