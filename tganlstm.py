from __future__ import print_function
from __future__ import division

import tensorflow as tf

import copy
from tgan import TGAN
import properties as pr
import rnn_utils


# https://github.com/soumith/ganhacks/issues/14
# https://stats.stackexchange.com/questions/251279/why-discriminator-converges-to-1-2-in-generative-adversarial-networks
# https://towardsdatascience.com/understanding-and-optimizing-gans-going-back-to-first-principles-e5df8835ae18
# https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
# flip labels
# add noise to input & label

class TGANLSTM(TGAN):

    def __init__(self, **kwargs):
        super(TGANLSTM, self).__init__(**kwargs)
        self.z_dim = [pr.batch_size, self.decoder_length, 256]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)
        # alpha is used for generator loss function

    # generate output images
    def create_generator(self, enc, dec, att):
        # shape:  batch_size x encoder_length x 1024
        enc, dec, att = self.lookup_input(enc, dec, att)
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            enc_outputs = self.exe_encoder(enc)
            """
                because transportation is a local problem => weather data don't need to be converted to grid heatmap
                Use simple lstm gru to visualize the fluctuation of weather feature => an attentional vector
            """
            att = tf.reshape(tf.transpose(att, [0, 2, 1, 3]), [pr.batch_size * 25, self.attention_length, 9])
            att_outputs, _ = rnn_utils.execute_sequence(att, self.e_params)
            att_outputs = self.get_softmax_attention(att_outputs)
            att_outputs = tf.reshape(att_outputs, [pr.batch_size, 25, 128])
            att_outputs = tf.layers.dense(att_outputs, 32, activation=tf.nn.tanh, name="attention_weathers")
            att_outputs = tf.layers.flatten(att_outputs)
            conditional_vectors = self.add_conditional_layer(enc_outputs, att_outputs)
            outputs = self.exe_decoder(conditional_vectors)
        return outputs, conditional_vectors
    
    def exe_encoder(self, enc):
        params = copy.deepcopy(self.e_params)
        params["fw_cell_size"] = 256
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            msf_output = self.add_msf_networks(enc)
            hidden_output = tf.reshape(msf_output, shape=(pr.batch_size, self.encoder_length, 256))
            # go to lstm
            lstm_output, _ = rnn_utils.execute_sequence(hidden_output, params)
            lstm_output = self.get_softmax_attention(lstm_output)
        return lstm_output
    
    # generator cnn layers
    def add_msf_networks(self, inputs, activation=tf.nn.relu, is_dis=False):
        inputs = tf.reshape(inputs, shape=(pr.batch_size * self.encoder_length, self.grid_size, self.grid_size, 1))
        # input (64, 32, 32, 1) output (64, 32, 32, 16)
        msf1 = rnn_utils.get_multiscale_conv(inputs, 16, activation=activation, prefix="msf1")
        # input (64, 32, 32, 16) output (64, 16, 16, 16)
        msf1_down = rnn_utils.get_cnn_unit(msf1, 32, (5,5), activation, padding="SAME", name="down_sample_1", strides=(2,2))
        msf2 = rnn_utils.get_multiscale_conv(msf1_down, 8, activation=activation, prefix="msf21")
        # input (64, 16, 16, 32) output (64, 8, 8, 32)
        msf2 = rnn_utils.get_cnn_unit(msf2, 32, (5,5), activation, padding="SAME", name="down_sample_2", strides=(2,2))
        # input (64, 8, 8, 32) output (64, 8, 8, 32)
        msf3 = rnn_utils.get_multiscale_conv(msf2, 8, [3,1], activation, prefix="msf31")
        # input (64, 8, 8, 32) output (64, 4, 4, 32)
        msf3 = rnn_utils.get_cnn_unit(msf3, 32, (5,5), activation, padding="SAME", name="down_sample_3", strides=(2,2))
        msf3 = tf.layers.flatten(msf3)
        # change to bs * length x 256
        msf3 = self.add_hidden_layers(msf3)
        shape = msf3.get_shape()
        msf3 = tf.reshape(msf3, shape=(pr.batch_size, int(int(shape[0]) / pr.batch_size) * int(shape[-1])))
        return msf3  

    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs_vectors = tf.tile(dec_hidden_vectors, [1, self.decoder_length])
            dec_inputs_vectors = tf.reshape(dec_inputs_vectors, [pr.batch_size, 256, self.decoder_length])
            dec_inputs_vectors = tf.transpose(dec_inputs_vectors, [0, 2, 1])
            # dec_inputs_vectors with shape bs x 8 x 512: concatenation of conditional layer vector & uniform random
            dec_inputs_vectors = tf.concat([dec_inputs_vectors, self.z], axis=2)
            dec_outputs = tf.layers.dense(dec_inputs_vectors, 512, name="generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs = tf.reshape(dec_outputs, [pr.batch_size * self.decoder_length, 8, 8, 8])
            # perform 3d-conv generation
            ups16 = rnn_utils.get_cnn_transpose_unit(dec_outputs, 32, (5,5), name="upscale_1")
            ups32 = rnn_utils.get_cnn_transpose_unit(ups16, 1, (5,5), name="up_scale2")
            cnn_outputs = tf.layers.flatten(ups32)
            cnn_outputs = tf.tanh(cnn_outputs)
            cnn_outputs = tf.reshape(cnn_outputs, [pr.batch_size, self.decoder_length, self.grid_size, self.grid_size])
        return cnn_outputs

    # call this function from grandparent
    def create_discriminator(self, fake_outputs, conditional_vectors):
        super(TGAN, self).create_discriminator(self, fake_outputs, conditional_vectors)
    
    def add_discriminator_loss(self, fake_preds, real_preds):
        super(TGAN, self).add_discriminator_loss(self, fake_preds, real_preds)
    