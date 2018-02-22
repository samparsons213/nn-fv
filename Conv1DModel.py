import tensorflow as tf
import numpy as np
from math import ceil
from utils import lazy_property
from DiagnosisModel import DiagnosisPrediction

class Conv1D(DiagnosisPrediction):

    def __init__(self, filter_size, num_filters, fc1_size = 1024, window_size = 125, data_dim = None, target_dim = None, 
            conv_stride = 1, pool_stride = 2, pool_size = None):
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._fc1_size = fc1_size
        self._window_size = window_size
        self._conv_stride = conv_stride
        self._pool_stride = pool_stride
        if pool_size is None:
            pool_size = pool_stride
        self._pool_size = pool_size
        self._num_layers = len(num_filters)
        self.keep_prob
        DiagnosisPrediction.__init__(self, data_dim, target_dim, window_size)
    
    @lazy_property
    def keep_prob(self):
        return tf.placeholder(tf.float32)

    @lazy_property
    def prediction_logits(self):
        in_channels = [self._data_dim]
        for i in range(self._num_layers-1):
            in_channels.append(self._num_filters[i])
        W = [self.weight_variable([self._filter_size, in_channels[i], self._num_filters[i]]) for i in range(self._num_layers)]
        b = [self.bias_variable([self._num_filters[i]]) for i in range(self._num_layers)]
        self._cache_conv_layers = [tf.nn.relu(self.convLayer(self.data, W[0]) + b[0])] # prepend with _cache_ to be accessed vias lazy propoerty
        self._cache_pool_layers = [self.maxPool(self._cache_conv_layers[0], self._pool_size)] # prepend with _cache_ to be accessed vias lazy propoerty
        for i in range(1, self._num_layers):
            self._cache_conv_layers.append(tf.nn.relu(self.convLayer(self._cache_pool_layers[i-1], W[i]) + b[i]))
            self._cache_pool_layers.append(self.maxPool(self._cache_conv_layers[i], self._pool_size))
        final_win_size = float(self._window_size)
        for i in range(self._num_layers):
            final_win_size = ceil(final_win_size / self._pool_stride)
        final_win_size = int(final_win_size)
        W_fc1 = self.weight_variable([final_win_size * self._num_filters[-1], self._fc1_size])
        b_fc1 = self.bias_variable([self._fc1_size])
        final_pool_flat = tf.reshape(self._cache_pool_layers[-1], [-1, final_win_size * self._num_filters[-1]])
        fc1 = tf.nn.relu(tf.matmul(final_pool_flat, W_fc1) + b_fc1)
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        W_fc2 = self.weight_variable([self._fc1_size, self._target_dim])
        b_fc2 = self.bias_variable([self._target_dim])
        return tf.matmul(fc1_drop, W_fc2) + b_fc2

    @lazy_property
    def conv_layers(self):
        self.prediction_logits
    
    @lazy_property
    def pool_layers(self):
        self.prediction_logits

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.0)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def convLayer(self, x, W):
        return tf.nn.conv1d(x, W, self._conv_stride, 'SAME', True, "NHWC")

    def maxPool(self, x, pool_size):
        return tf.layers.max_pooling1d(x, pool_size, self._pool_stride, 'SAME')