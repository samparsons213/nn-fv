import tensorflow as tf

from models.ModelBase import PredictionModel
from utils import lazy_property


class DiagnosisPrediction(PredictionModel):

    def __init__(self, data_dim = None, target_dim = 2, batch_length = None):
        self._target_dim = target_dim
        PredictionModel.__init__(self, data_dim, batch_length)
        self.targets

    @property
    def target_dim(self):
        return self._target_dim
    
    @lazy_property
    def targets(self):
        return tf.placeholder(tf.float32, [None, self._target_dim])

    @lazy_property
    def prediction(self):
        return tf.argmax(self.prediction_logits, 1)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.targets, 1), self.prediction)
        return mistakes
