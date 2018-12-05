from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from apgan import APGan
import properties as pr
import rnn_utils


# https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
# flip labels
# add noise to input & label
# transform inputs: flip_top_down, left_right, random_crop, translation, add gaussian noise

class CAPGan(APGan):

    def __init__(self, **kwargs):
        # alpha is used for generator loss function
        super(CAPGan, self).__init__(**kwargs)
        self.z_dim = [pr.batch_size, 128]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)
        self.encoder_length = 25
        self.decoder_length = 25
        self.attention_length = 72
        self.use_attention = False
        self.alpha = 0.005
        self.augment_configs = ["gaussian_noise", "flip_top_down", "flip_left_right", "random_crop"]
        self.flag = tf.placeholder(tf.float32, shape=[self.batch_size, 1])

    def set_data(self, datasets, train, valid, attention_vectors=None):
        dtl = len(datasets)
        print(datasets.shape)
        self.datasets = datasets[int(dtl/2):,:,:]
        self.train = train
        self.valid = valid
        self.attention_vectors = attention_vectors
    
    # mapping input indices to dataset
    def lookup_input(self, enc, dec):
        enc = tf.nn.embedding_lookup(self.embedding, enc)
        dec_f = tf.nn.embedding_lookup(self.embedding, dec)
        enc.set_shape((self.batch_size, self.encoder_length, 25, self.encode_vector_size))
        dec_f.set_shape((self.batch_size, self.encoder_length, 25, self.encode_vector_size))
        dec = dec_f[:,:,:,self.df_ele:]
        dec.set_shape((self.batch_size, self.encoder_length, 25, self.decode_vector_size))
        self.pred_placeholder = tf.reshape(dec_f[:,:,:,0], [pr.batch_size, self.decoder_length, 25, 1])
        return enc, dec
    
    def add_msf_networks(self, inputs, activation=tf.nn.relu, is_dis=False):
        feature_remap = rnn_utils.get_cnn_unit(inputs, 32, (1,1), activation, padding="SAME", name="feature_remap", strides=(1,1))
        # input (64, 25, 25, 64) output (64, 25, 25, 64)
        msf1 = rnn_utils.get_multiscale_conv(feature_remap, 16, activation=activation, prefix="msf1")
        # input (64, 25, 25, 64) output (64, 11, 11, 64)
        msf1_down = rnn_utils.get_cnn_unit(msf1, 32, (5,5), activation, padding="VALID", name="down_sample_1")
        # input (64, 11, 11, 64) output (64, 11, 11, 64)
        if not is_dis:
            msf2 = rnn_utils.get_multiscale_conv(msf1_down, 8, activation=activation, prefix="msf21")
        else: 
            msf2 = msf1_down
        # msf2 = rnn_utils.get_multiscale_conv(msf2, 32, activation=activation, prefix="msf22")
        # input (64, 11, 11, 64) output (64, 3, 3, 64)
        msf2_down = rnn_utils.get_cnn_unit(msf2, 32, (5,5), activation, padding="VALID",name="down_sample_2")
        # input (64, 3, 3, 64) output (64, 3, 3, 64)
        if not is_dis:
            msf3 = rnn_utils.get_multiscale_conv(msf2_down, 8, [3,1], activation, prefix="msf31")
            # msf3 = rnn_utils.get_multiscale_conv(msf3, 8, [3,1], activation, prefix="msf32")
            msf3 = tf.layers.flatten(msf3)
        else:
            msf3 = tf.layers.flatten(msf2_down)
        # hidden_output = tf.layers.dense(msf3, 256, tf.nn.tanh, name="hidden_1")
        hidden_output = tf.layers.dense(msf3, 128, tf.nn.tanh, name="hidden_2")
        hidden_output = tf.nn.dropout(hidden_output, 0.5)
        return hidden_output

    def exe_encoder(self, enc):
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            hidden_output = self.add_msf_networks(enc)
        return hidden_output
    
    def get_decoder_representation(self, dec):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            hidden_output = self.add_msf_networks(dec)
        return hidden_output

    # the conditional layer that concat all attention vectors => produces a single vector
    def add_conditional_layer(self, dec_rep, enc_outputs, attention=None):
        with tf.name_scope("conditional"):
            dec_input = tf.concat([enc_outputs, dec_rep], axis=1)
            if not attention is None:
                dec_input = tf.concat([dec_input, attention], axis=1)
            # dec_hidden_vectors with shape bs x 128
            dec_hidden_vectors = tf.layers.dense(dec_input, self.rnn_hidden_units, name="conditional_layer", activation=tf.nn.tanh)
            return dec_hidden_vectors
    
    #perform dec  oder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            # dec_inputs_vectors with shape bs x 256: concatenation of conditional layer vector & uniform random 128D
            dec_inputs_vectors = tf.concat([dec_hidden_vectors, self.z], axis=1)
            dec_inputs_vectors = tf.layers.dense(dec_hidden_vectors, 256, activation=tf.nn.tanh, name="input_hidden")
            dec_inputs_vectors = tf.reshape(dec_inputs_vectors, [pr.batch_size, 4, 4, 16])
            ups8 = rnn_utils.get_cnn_transpose_unit(dec_inputs_vectors, 128, (3,3), name="up_scale1", strides=(2,2))
            ups16 = rnn_utils.get_cnn_transpose_unit(ups8, 64, (5,5), name="up_scale2", strides=(2,2))
            ups32 = rnn_utils.get_cnn_transpose_unit(ups16, 32, (7,7), name="up_scale3", strides=(2,2))
            msf1 = rnn_utils.get_multiscale_conv(ups32, 16, prefix="decoder_msf")
            cnn_outputs = rnn_utils.get_cnn_unit(msf1, 1, (8, 8), tf.nn.relu, "VALID", "cnn_gen_output", dropout=0.5, strides=(1,1))
            # cnn_outputs = tf.tanh(cnn_outputs)
            cnn_outputs = tf.layers.flatten(cnn_outputs)
            cnn_outputs = tf.layers.dense(cnn_outputs, 625, name="final_hidden_layer", activation=tf.nn.sigmoid)
            # cnn_outputs = cnn_outputs / 2 + 0.5
            cnn_outputs = tf.reshape(cnn_outputs, [pr.batch_size, self.decoder_length, 25, 1])
        return cnn_outputs

    # generate output images
    def create_generator(self, enc, dec, att):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input(enc, dec)
            # get encoder representation
            enc_outputs = self.exe_encoder(enc)
            # get decoder representation
            dec_outputs = self.get_decoder_representation(dec)
            if self.use_attention:
                inputs = tf.nn.embedding_lookup(self.attention_embedding, att)
                attention = self.get_attention_rep(inputs)
            else:
                attention = None
            conditional_vectors = self.add_conditional_layer(dec_outputs, enc_outputs, attention)
            outputs = self.exe_decoder(conditional_vectors)
        return outputs, conditional_vectors
    
    # add generation loss
    # use log_sigmoid instead of log because fake_vals is w * x + b (not the probability value)
    def add_generator_loss(self, fake_vals, outputs, labels, fake_rewards=None):
        loss = tf.losses.mean_squared_error(labels, outputs)
        if self.alpha:
            sigmoid_loss = self.alpha * tf.log_sigmoid(fake_vals)        
            # normal lossmse + (-log(D(G)))
            loss_values = loss - sigmoid_loss
            #loss_values = sigmoid_loss
            loss = tf.reduce_mean(loss_values)
        else:
            for v in tf.trainable_variables():
                if not 'bias' in v.name.lower():
                    loss += 0.0001 * tf.nn.l2_loss(v)
        return loss
    
    def get_generator_loss(self, fake_preds, outputs, fake_rewards=None):
        gen_loss = self.add_generator_loss(fake_preds, outputs, self.pred_placeholder, fake_rewards)
        return gen_loss

    # regular discriminator loss function
    def add_discriminator_loss(self, fake_preds, real_preds):
        # real_labels = tf.constant(0.9, shape=[self.batch_size, 1])
        # fake_labels = tf.zeros([self.batch_size, 1])
        real_labels = np.absolute(self.flag - np.random.uniform(0.8, 1., [self.batch_size, 1]))
        fake_labels = np.absolute(self.flag - np.random.uniform(0., 0.2, [self.batch_size, 1]))
        dis_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(fake_labels, fake_preds))
        dis_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(real_labels, real_preds))
        dis_loss = dis_loss_real + dis_loss_fake
        tf.summary.scalar("dis_loss", dis_loss)
        return dis_loss

    # just decide whether an image is fake or real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    def validate_output(self, inputs, conditional_vectors):
        hidden_output = self.add_msf_networks(inputs, tf.nn.leaky_relu, True)
        hidden_output = tf.concat([hidden_output, conditional_vectors], axis=1)
        output = tf.layers.dense(hidden_output, 1, name="validation_value")
        return output   

    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            conditional_vectors = tf.tile(conditional_vectors, [1, self.decoder_length])
            real_inputs = self.pred_placeholder
            fake_val = self.validate_output(fake_outputs, conditional_vectors)
            real_val = self.validate_output(real_inputs, conditional_vectors)
        return fake_val, real_val, None

    # normalize inputs to [-1,1]
    def normalize_abs_one(self, inputs):
        inputs = (inputs - 0.5) * 2
        return inputs
       
    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = ct_t + self.decoder_length

        feed = {
            self.encoder_inputs : ct_t,
            self.decoder_inputs: dec_t,
            self.z: self.sample_z(),
            self.flag: np.asarray(np.random.randint(0, 1, [pr.batch_size, 1]), dtype=np.float32)
            #self.flag: np.zeros([pr.batch_size, 1], dtype=np.float32)
        }
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])

        if not train:
            pred = session.run([self.outputs], feed_dict=feed)
        else:
            gen_loss, dis_loss, pred, _, _= session.run([self.gen_loss, self.dis_loss, self.outputs, self.gen_op, self.dis_op], feed_dict=feed)
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss

        return pred, total_gen_loss, total_dis_loss
