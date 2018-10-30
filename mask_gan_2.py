from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import copy
from apgan import APGan
import properties as pr
import rnn_utils



class MaskGan2(APGan):

    def __init__(self, **kwargs):
        super(MaskGan2, self).__init__(**kwargs)
        self.use_rewards = False
        
    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors):
        params = copy.deepcopy(self.e_params)
        params["fw_cell_size"] = 256
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs_vectors = tf.tile(dec_hidden_vectors, [1, self.decoder_length])
            dec_inputs_vectors = tf.reshape(dec_inputs_vectors, [pr.batch_size, self.rnn_hidden_units, self.decoder_length])
            dec_inputs_vectors = tf.transpose(dec_inputs_vectors, [0, 2, 1])
            # dec_inputs_vectors with shape bs x 24 x 256: concatenation of conditional layer vector & uniform random 128D
            dec_inputs_vectors = tf.concat([dec_inputs_vectors, self.z], axis=2)
            dec_concat_vectors = tf.layers.dense(dec_inputs_vectors, 256, name="generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs, _ = rnn_utils.execute_sequence(dec_concat_vectors, params)
            # change to shape bs*24 x 256 => fast execute cnns
            dec_outputs = tf.reshape(dec_outputs, [pr.batch_size * self.decoder_length, 4, 4, 16])
            generate_outputs = rnn_utils.get_cnn_rep(dec_outputs, 2, tf.nn.relu, 8, self.use_batch_norm, self.dropout, False)
            generate_outputs = tf.tanh(tf.layers.flatten(generate_outputs))
            outputs = tf.reshape(generate_outputs, [pr.batch_size, self.decoder_length, pr.grid_size * pr.grid_size])
        return outputs

    # performing GRU before final decision fake/real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    def validate_output(self, inputs, conditional_vectors, is_fake=False):
        conditional_vectors = tf.reshape(conditional_vectors, [pr.batch_size * self.decoder_length, self.rnn_hidden_units])
        inputs = tf.reshape(inputs, [pr.batch_size * self.decoder_length, pr.grid_size, pr.grid_size, 1])
        inputs_rep = rnn_utils.get_cnn_rep(inputs, 3, tf.nn.leaky_relu, 8, self.use_batch_norm, self.dropout, False)
        inputs_rep = tf.layers.flatten(inputs_rep)
        inputs_rep = tf.concat([inputs_rep, conditional_vectors], axis=1)
        inputs_rep_shape = inputs_rep.get_shape()
        inputs_rep = tf.reshape(inputs_rep, [pr.batch_size, self.decoder_length, int(inputs_rep_shape[-1])])
        # push through a GRU layer
        rnn_outputs, _ = rnn_utils.execute_sequence(inputs_rep, self.e_params)
        # real or fake value
        output = tf.layers.dense(rnn_outputs, 1, name="validation_value")
        output = tf.layers.flatten(output)
        rewards = None
        if is_fake:
            rewards = [None] * self.decoder_length
            pred_value = tf.log_sigmoid(output)
            pred_values = tf.unstack(pred_value, axis=1)
            for i in xrange(self.decoder_length - 1, -1,-1):
                rewards[i] = pred_values[i]
                if i != (self.decoder_length - 1):
                    for j in xrange(i + 1, self.decoder_length):
                        rewards[i] += np.power(self.gamma, (j - i)) * rewards[i]
        return output, rewards   
    
    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            conditional_vectors = tf.tile(conditional_vectors, [1, self.decoder_length])
            real_inputs = self.pred_placeholder
            fake_val, fake_rewards = self.validate_output(fake_outputs, conditional_vectors, self.use_rewards)
            real_val, _ = self.validate_output(real_inputs, conditional_vectors)
        return fake_val, real_val, fake_rewards