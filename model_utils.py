import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell

import rnn_utils



def add_training_op(loss, learning_rate):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)
        return train_op