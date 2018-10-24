from __future__ import print_function
from __future__ import division

import sys
import copy
import time
import numpy as np
import tensorflow as tf

from mask_gan import MaskGan
import properties as pr
import utils
import rnn_utils


class APGan(MaskGan):

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        # set up multiple cnns layers to generate outputs
        self.use_gen_cnn = True
        self.dropout = 0.0
        self.use_batch_norm = False
        self.strides = [4]
        self.beta1 = 0.5
        self.lamda = 100
        self.gmtype = 3
        self.z_dim = 128
        self.z = tf.placeholder(tf.float32, shape=[pr.batch_size, self.decoder_length, self.z_dim])

    def inference(self, is_train=True):
        fake_outputs, conditional_vectors = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        if is_train:
            fake_preds, fake_rewards, real_preds = self.create_discriminator(fake_outputs, conditional_vectors)
            self.dis_loss = self.add_discriminator_loss(fake_preds, real_preds)
            self.gen_loss = self.get_generator_loss(fake_preds, fake_rewards, fake_outputs)
            self.gen_op = self.train_generator(self.gen_loss)
            self.dis_op = self.train_discriminator(self.dis_loss)
        return fake_outputs
    
    # generate output images
    def create_generator(self, enc, dec, att):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input(enc, dec)
            enc_output = self.exe_encoder(enc, False, 0.0)
            attention = None
            if self.use_attention:
                # batch size x rnn_hidden_size
                inputs = tf.nn.embedding_lookup(self.attention_embedding, att)
                attention = self.get_attention_rep(inputs)
            conditional_vectors = self.add_conditional_layer(dec, enc_output, attention)
            outputs = self.exe_decoder(conditional_vectors)
        return outputs, conditional_vectors
    
    def sample_z(self, x, y, z):
        return np.random.uniform(-1., 1., size=[x, y, z])
    
    # the conditional layer that concat all attention vectors => produces a single vector
    def add_conditional_layer(self, dec, enc_output, attention=None):
        with tf.name_scope("conditional"):
            cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
            cnn_dec_input = tf.layers.flatten(cnn_dec_input)
            cnn_shape = cnn.get_shape()
            dec_data = tf.reshape(cnn_dec_input, [pr.batch_size, self.decoder_length, int(cnn_shape[-1])])
            enc_outputs, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
            # add attentional layer here to measure the importance of each timestep.
            fn_state = self.get_softmax_attention(enc_outputs)
            # dec_input with shape bs x 3hidden_size
            dec_input = tf.concat([enc_output, fn_state], axis=1)
            if not attention is None:
                dec_input = tf.concat([dec_input, attention], axis=1)
            # dec_hidden_vectors with shape bs x 128
            dec_hidden_vectors = tf.layers.dense(dec_input, self.rnn_hidden_units, "conditional_layer", activation=tf.nn.tanh)
            return dec_hidden_vectors
    
    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs_vectors = tf.tile(dec_hidden_vectors, [1, self.decoder_length])
            dec_inputs_vectors = tf.transpose(dec_inputs_vectors, [0, 2, 1])
            # dec_hidden_vectors with shape bs x 24 x 256
            dec_inputs_vectors = tf.concat([dec_inputs_vectors, self.z], axis=2)
            dec_outputs = tf.layers.dense(dec_hidden_vectors, 256, "generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs = tf.unstack(dec_outputs, axis=1)
            gen_outputs = []
            for d in dec_outputs:
                out = rnn_utils.get_cnn_rep(d, 2, tf.nn.relu, 8, self.use_batch_norm, self.dropout, False)
                gen_outputs.append(out)
            outputs = tf.stack(gen_outputs, axis=1)
            outputs = tf.tanh(tf.layers.flatten(outputs))
        return outputs

    def validate_output(self, inputs, conditional_vectors, is_fake=True):
        inputs = tf.reshape(inputs, [pr.batch_size * self.decoder_length, pr.grid_size, pr.grid_size])
        inputs_rep = rnn_utils.get_cnn_rep(inputs, 3, tf.nn.leaky_relu, 8, self.use_batch_norm, self.dropout, False)
        inputs_rep = tf.layers.flatten(inputs_rep)
        output = tf.layers.dense(inputs_rep, 1, name="validation_value")
        # reshape from bs * decoder_length x 1 => bs x  decoder_length
        output = tf.reshape(output, [pr.batch_size, self.decoder_length])
        rewards = []
        if is_fake:
            pred_value = tf.log_sigmoid(output)
            pred_values = tf.unstack(pred_value, axis=1)
            for i in xrange(self.decoder_length - 1, -1,-1):
                rewards[i] = pred_value
                if i != (self.decoder_length - 1):
                    for j in xrange(i + 1, self.decoder_length):
                        rewards[i] += np.power(self.gamma, (j - i)) * rewards[i]
        return output, rewards        

    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            conditional_vectors = tf.tile(conditional_vectors, [1, self.decoder_length])
            real_inputs = self.pred_placeholder
            fake_val, fake_rewards = self.validate_output(fake_outputs, conditional_vectors)
            real_val, _ = self.validate_output(real_inputs, conditional_vectors)

        return fake_val, fake_rewards, real_val

    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        ct_t = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
        dec_t = ct_t + self.decoder_length

        feed = {
            self.encoder_inputs : ct_t,
            self.decoder_inputs: dec_t,
            self.z: self.sample_z(pr.batch_size, self.decoder_length, self.z_dim)
        }
        if self.use_attention:
            feed[self.attention_inputs] = ct_t

        if not train:
            pred = session.run([self.outputs], feed_dict=feed)
        else:
            gen_loss, dis_loss, pred, _, _= session.run([self.gen_loss, self.dis_loss, self.outputs, self.gen_op, self.dis_op], feed_dict=feed)
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss

        return pred, total_gen_loss, total_dis_loss