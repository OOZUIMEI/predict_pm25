from __future__ import print_function
from __future__ import division

import tensorflow as tf

from apgan import APGan
import properties as pr
import rnn_utils


# https://github.com/soumith/ganhacks/issues/14
# https://stats.stackexchange.com/questions/251279/why-discriminator-converges-to-1-2-in-generative-adversarial-networks
# https://towardsdatascience.com/understanding-and-optimizing-gans-going-back-to-first-principles-e5df8835ae18
# https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
# flip labels
# add noise to input & label

class TGAN(APGan):

    def __init__(self, encoder_length=8, decoder_length=8, attention_length=24, **kwargs):
        super(TGAN, self).__init__(**kwargs)
        # alpha is used for generator loss function
        self.dropout = 0.0
        self.grid_size = 32
        self.z_dim = [pr.batch_size, 256]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.attention_length = attention_length
        self.strides = [1,2]
        self.use_attention = True

    def set_data(self, datasets, train, valid, attention_vectors=None):
        self.datasets = datasets[:,:]
        self.train = train
        self.valid = valid
        self.attention_vectors = attention_vectors
    
    # preserve memory for tensors
    def add_placeholders(self):
        self.embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="Variable")
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))
        self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.grid_size, self. grid_size))
        # weather attention_inputs
        self.attention_embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="attention_embedding")
        self.attention_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.attention_length))
    
    # mapping input indices to dataset
    def lookup_input(self, enc, dec, att):
        enc = tf.nn.embedding_lookup(self.embedding, enc)
        dec = tf.nn.embedding_lookup(self.embedding, dec)
        att = tf.nn.embedding_lookup(self.attention_embedding, att)
        enc.set_shape((pr.batch_size, self.encoder_length, 1024))
        enc = tf.reshape(enc, (pr.batch_size, self.encoder_length, 32, 32, 1))
        dec.set_shape((pr.batch_size, self.decoder_length, 1024))
        dec = tf.reshape(dec, (pr.batch_size, self.decoder_length, 32, 32))
        self.pred_placeholder = dec
        dec = tf.expand_dims(dec, axis=4)
        att.set_shape((pr.batch_size, self.attention_length, 25, 9))
        return enc, dec, att

    def inference(self, is_train=True):
        fake_outputs, conditional_vectors = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        if is_train:
            fake_vals, real_vals, _ = self.create_discriminator(fake_outputs, conditional_vectors)
            self.dis_loss = self.add_discriminator_loss(fake_vals, real_vals)
            self.gen_loss = self.get_generator_loss(fake_vals, fake_outputs)
            self.gen_op = self.train_generator(self.gen_loss)
            self.dis_op = self.train_discriminator(self.dis_loss)
        return fake_outputs
    
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
    
    # the conditional layer that concat all attention vectors => produces a single vector
    def add_conditional_layer(self, enc_outputs, attention):
        with tf.name_scope("conditional"):
            attention_inputs = tf.concat([enc_outputs, attention], axis=1)
            dec_hidden_vectors = tf.layers.dense(attention_inputs, 256, name="conditional_layer", activation=tf.nn.tanh)
            return dec_hidden_vectors
    
    def exe_encoder(self, enc):
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            msf_output = self.add_msf_networks(enc)
            hidden_output = self.add_hidden_layers(msf_output)
        return hidden_output
    
    # generator cnn layers
    def add_msf_networks(self, inputs, activation=tf.nn.relu, is_dis=False):
        # input (64, encode_length, 32, 32, 1) output (64, encod'e_length, 32, 32, 64)
        msf1 = rnn_utils.get_multiscale_conv3d(inputs, 16, activation=activation, prefix="msf1")
        # input (64, encode_length, 32, 32, 64) output (64, encode_length/2, 16, 16, 32)
        msf1_down = rnn_utils.get_cnn3d_unit(msf1, 32, (5,5,5), activation, padding="SAME", name="down_sample_1", strides=(2,2,2))
        # input (64, 16, 16, 64) output (64, 16, 16, 64)
        msf2 = rnn_utils.get_multiscale_conv3d(msf1_down, 8, activation=activation, prefix="msf21")
        # input (64, el/2, 16, 16, 32) output (64, el/4, 8, 8, 32)
        msf2_down = rnn_utils.get_cnn3d_unit(msf2, 32, (5,5,5), activation, padding="SAME", name="down_sample_2", strides=(2,2,2))
        # input (64, el/4, 8, 8, 32) output (64, el/4, 8, 8, 32)
        msf3 = rnn_utils.get_multiscale_conv3d(msf2_down, 8, [3,1], activation, prefix="msf31")
        # new layer (64, el/4, 8, 8, 32) output (64, el/8, 4, 4, 64)
        # msf3 = rnn_utils.get_cnn3d_unit(msf3, 64, (5,5,5), activation, padding="SAME", name="down_sample_3", strides=(2,2,2))
        msf3 = tf.layers.flatten(msf3)
        return msf3

    def add_hidden_layers(self, input_vectors):
        hidden_output = tf.layers.dense(input_vectors, 512, tf.nn.tanh, name="hidden_1")
        hidden_output = tf.nn.dropout(hidden_output, 0.5)
        hidden_output = tf.layers.dense(hidden_output, 256, tf.nn.tanh, name="hidden_2")
        hidden_output = tf.nn.dropout(hidden_output, 0.5)
        return hidden_output

    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors):
        # 512 / 16 / depth
        depth = int(self.decoder_length / 4)
        ft = int(8 / depth)
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            # bs x 256 + bs x 256
            dec_inputs_vectors = tf.concat([dec_hidden_vectors, self.z], axis=1)
            dec_outputs = tf.layers.dense(dec_inputs_vectors, 512, name="generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs = tf.reshape(dec_outputs, [pr.batch_size, depth, 8, 8, ft])
            # perform 3d-conv generation
            ups16 = rnn_utils.get_cnn_transpose3d_unit(dec_outputs, 32, (5,5,5), name="upscale_1")
            ups32 = rnn_utils.get_cnn_transpose3d_unit(ups16, 1, (5,5,5), name="up_scale2")
            cnn_outputs = tf.layers.flatten(ups32)
            cnn_outputs = tf.tanh(cnn_outputs)
            cnn_outputs = tf.reshape(cnn_outputs, [pr.batch_size, self.decoder_length, self.grid_size, self.grid_size])
        return cnn_outputs

    # just decide whether an image is fake or real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    # in conv3d, consider a cube only instead of each frame
    # conditional_vectors is bs x 256
    # change hidden_output from bs x 2048 => bs x 256
    def validate_output(self, inputs, conditional_vectors):
        inputs = tf.expand_dims(inputs, axis=4)
        hidden_output = self.add_msf_networks(inputs, tf.nn.leaky_relu, True)
        hidden_output = self.add_hidden_layers(hidden_output)
        hidden_output = tf.concat([hidden_output, conditional_vectors], axis=1)
        output = tf.layers.dense(hidden_output, 1, name="validation_value")
        return output
    
    # different to apgan
    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            fake_val = self.validate_output(fake_outputs, conditional_vectors)
            real_val = self.validate_output(self.pred_placeholder, conditional_vectors)
        return fake_val, real_val, None


    def get_generator_loss(self, fake_preds, outputs, fake_rewards=None):
        labels = tf.layers.flatten(self.pred_placeholder)
        outputs = tf.layers.flatten(outputs)
        gen_loss = self.add_generator_loss(fake_preds, outputs, labels, fake_rewards)
        return gen_loss