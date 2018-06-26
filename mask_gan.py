from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell

import properties as prp
import utils


class Model():

    
    def __init__(self, sequence_length=48):
        self.sequence_length = sequence_length

    def init_ops(self):
       

    def add_placeholders(self):

    def inference(self):

    # rnn through each 30', 1h 
    def get_input_representation(self, params):
        fw_cell = LayerNormBasicLSTMCell(params.encoder_size)
        _, fn_state = tf.nn.static_rnn(
            fw_cell,
            inputs,
            dtype=np.float32
        )
        # (new_c, new_h)
        return fn_state

    def create_discriminator():

    def create_generator(self, inputs):
        with tf.variable_scope("generator", reuse=reuse, initializer=initializer):
            state_dec = self.get_input_representation(inputs)
            # push final state of encoder to decoder
            for t in xrange(self.sequence_length):
                dec_in = inputs[:, t]
                dec_out, state_dec = cell_dec(dec_in, state_dec)
                pm2_5 = tf.dense(dec_out, 
                                name="decoder_output",
                                activation=None)
                
                


    def add_loss_generator(self, output):
    
    def add_loss_discriminator(self, output):

    def add_training_op(self, loss):

    def run_epoch(self):
        