"""
both decoder and encoder using same attention layers
"""

import numpy as np
import tensorflow as tf

from mask_gan import MaskGan
import properties as pr
import rnn_utils


# https://github.com/soumith/ganhacks/issues/14
# https://stats.stackexchange.com/questions/251279/why-discriminator-converges-to-1-2-in-generative-adversarial-networks
# https://towardsdatascience.com/understanding-and-optimizing-gans-going-back-to-first-principles-e5df8835ae18
# https://stackoverflow.com/questions/42enc_output690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
# Reload graph
# https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125
# https://ww2.arb.ca.gov/resources/sources-air-pollution
# flip labels
# add noise to input & label

class APGan(MaskGan):

    def __init__(self, **kwargs):
        super(APGan, self).__init__(**kwargs)
        # alpha is used for generator loss function
        # [0.001 nodp > 0.001 dp0.5 > 0.005 nodp > 0.005 dp0.5]
        # ? 0.0009
        # 0.0005 is mode collapse. 0.01 is mode collapse when use_flip = False
        self.use_flip = True
        self.alpha = 0.001
        self.use_gen_cnn = True
        self.dropout = 0.5 # help maintain the discriminator (0.5/0.0)
        self.use_batch_norm = False
        self.strides = [2]
        self.beta1 = 0.5
        self.lamda = 100
        # discriminator unit type 6 & 7 is hard to control / should be 4
        self.gmtype = 7
        self.mtype = 7
        self.z_dim = [self.batch_size, self.decoder_length, 128]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)   
        self.flag = tf.placeholder(tf.float32, shape=[self.batch_size, 1]) 

    def inference(self, is_train=True):
        fake_outputs, conditional_vectors = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        if is_train:
            fake_vals, real_vals, fake_rewards = self.create_discriminator(fake_outputs, conditional_vectors)
            self.dis_loss = self.add_discriminator_loss(fake_vals, real_vals)
            self.gen_loss = self.get_generator_loss(fake_vals, fake_outputs, fake_rewards)
            self.gen_op = self.train_generator(self.gen_loss)
            self.dis_op = self.train_discriminator(self.dis_loss)
        return fake_outputs
    
    # generate output images
    def create_generator(self, enc, dec, att):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input(enc, dec)
            fn_state, enc_outputs = self.exe_encoder(enc, False, 0.0)
            attention = None
            if self.use_attention:
                # print("use attention")
                # batch size x rnn_hidden_size
                inputs = tf.nn.embedding_lookup(self.attention_embedding, att)
                attention = self.get_attention_rep(inputs)
            conditional_vectors = self.add_conditional_layer(dec, enc_outputs, attention)
            outputs = self.exe_decoder(conditional_vectors, fn_state)
        return outputs, conditional_vectors
    
    def sample_z(self):
        # better for pm2.5
        # return np.random.uniform(-1., 1., size=self.z_dim)
        # better for pm10
        return np.random.normal(0., 0.01, size=self.z_dim)
    
    def get_generator_loss(self, fake_preds, outputs, fake_rewards=None):
        labels = tf.reshape(self.pred_placeholder, shape=(self.batch_size, self.decoder_length, self.grid_square))
        gen_loss = self.add_generator_loss(fake_preds, outputs, labels, fake_rewards)
        return gen_loss
    
    # add generation loss
    # use log_sigmoid instead of log because fake_vals is w * x + b (not the probability value)
    def add_generator_loss(self, fake_vals, outputs, labels, fake_rewards=None):
        loss = tf.losses.mean_squared_error(labels, outputs)
        if not fake_rewards is None:
            print("Using reinsforcement learning")
            advatages = tf.abs(fake_rewards)
            loss = tf.reduce_mean(tf.multiply(loss, tf.stop_gradient(advatages)))
        else:
            print("Using combined loss function")
            if self.alpha:
                sigmoid_loss = self.alpha * tf.log_sigmoid(fake_vals)
                # sigmoid_loss = self.alpha * tf.losses.sigmoid_cross_entropy(fake_vals, tf.constant(1., shape=[self.batch_size, self.decoder_length]))
                # normal lossmse + (-log(D(G)))
                loss = loss - sigmoid_loss
                #loss_values = sigmoid_loss
                loss = tf.reduce_mean(loss)
                """
                for v in tf.trainable_variables():
                    name = v.name.lower()
                    if 'generator' in name and not 'bias' in name:
                        loss += 0.0001 * tf.nn.l2_loss(v)
                """
            else:
                loss = tf.reduce_mean(loss)
                for v in tf.trainable_variables():
                    if not 'bias' in v.name.lower():
                        loss += 0.0001 * tf.nn.l2_loss(v)
        return loss
    
    # the conditional layer that concat all attention vectors => produces a single vector
    # def add_conditional_layer(self, dec, enc_outputs, attention=None):
    #     with tf.name_scope("conditional"):
    #         cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
    #         cnn_dec_input = tf.layers.flatten(cnn_dec_input)
    #         cnn_shape = cnn_dec_input.get_shape()
    #         dec_data = tf.reshape(cnn_dec_input, [self.batch_size, self.decoder_length, int(cnn_shape[-1])])
    #         dec_rep, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
    #         dec_rep = self.get_softmax_attention(dec_rep)
    #         # add attentional layer here to measure the importance of each timestep.
    #         enc_outputs = self.get_softmax_attention(enc_outputs)
    #         # dec_input with shape bs x 3hidden_size
    #         dec_input = tf.concat([enc_outputs, dec_rep], axis=1)
    #         if not attention is None:
    #             dec_input = tf.concat([dec_input, attention], axis=1)
    #         # dec_hidden_vectors with shape bs x 128 
    #         dec_hidden_vectors = tf.layers.dense(dec_input, 128, name="conditional_layer", activation=tf.nn.tanh)
    #         if self.dropout:
    #             dec_hidden_vectors = tf.nn.dropout(dec_hidden_vectors, 0.5)
    #         return dec_hidden_vectors
    
    # the conditional layer that concat all attention vectors => produces a single vector
    def add_conditional_layer(self, dec, enc_outputs, attention=None):
        with tf.variable_scope("encoder_softmax", initializer=self.initializer):
            enc_outputs = self.get_softmax_attention(enc_outputs)

        with tf.variable_scope("decoder_softmax", initializer=self.initializer):
            cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
            cnn_dec_input = tf.layers.flatten(cnn_dec_input)
            cnn_shape = cnn_dec_input.get_shape()
            dec_data = tf.reshape(cnn_dec_input, [self.batch_size, self.decoder_length, int(cnn_shape[-1])])
            dec_rep, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
            dec_rep = self.get_softmax_attention(dec_rep)

        with tf.name_scope("conditional"):
            # add attentional layer here to measure the importance of each timestep.
            # dec_input with shape bs x 3hidden_size
            dec_input = tf.concat([enc_outputs, dec_rep], axis=1)
            if not attention is None:
                dec_input = tf.concat([dec_input, attention], axis=1)
            # dec_hidden_vectors with shape bs x 128 
            dec_hidden_vectors = tf.layers.dense(dec_input, 128, name="conditional_layer", activation=tf.nn.tanh)
            if self.dropout:
                dec_hidden_vectors = tf.nn.dropout(dec_hidden_vectors, 0.5)
        return dec_hidden_vectors

    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors, fn_state=None):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs_vectors = tf.tile(dec_hidden_vectors, [1, self.decoder_length])
            dec_inputs_vectors = tf.reshape(dec_inputs_vectors, [self.batch_size, self.rnn_hidden_units, self.decoder_length])
            dec_inputs_vectors = tf.transpose(dec_inputs_vectors, [0, 2, 1])
            # dec_inputs_vectors with shape bs x 24 x 256: concatenation of conditional layer vector & uniform random 128D
            dec_inputs_vectors = tf.concat([dec_inputs_vectors, self.z], axis=2)
            dec_outputs = tf.layers.dense(dec_inputs_vectors, 256, name="generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs = tf.unstack(dec_outputs, axis=1)
            gen_outputs = []
            for d in dec_outputs:
                d_ = tf.reshape(d, [self.batch_size, 2, 2, 64])
                out = rnn_utils.get_cnn_rep(d_, 5, tf.nn.relu, 8, self.use_batch_norm, 0.5, False)
                gen_outputs.append(out)
            outputs = tf.stack(gen_outputs, axis=1)
            outputs = tf.tanh(tf.layers.flatten(outputs))
            outputs = tf.reshape(outputs, [self.batch_size, self.decoder_length, pr.grid_size * pr.grid_size])
        return outputs

    # just decide whether an image is fake or real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    def validate_output(self, inputs, conditional_vectors):
        inputs = tf.reshape(inputs, [self.batch_size * self.decoder_length, pr.grid_size, pr.grid_size, 1])
        inputs_rep = rnn_utils.get_cnn_rep(inputs, self.gmtype, tf.nn.leaky_relu, 8, self.use_batch_norm, self.dropout, False)
        inputs_rep = tf.layers.flatten(inputs_rep)
        inputs_rep = tf.concat([inputs_rep, conditional_vectors], axis=1)
        # add new hidden layer in the middle
        # help maintain the discriminator
        # output = tf.layers.dense(inputs_rep, 128, name="hidden_validation")
        # if self.dropout:
        #     output = tf.nn.dropout(output, 0.5)
        output = tf.layers.dense(inputs_rep, 1, name="validation_value")
        # if self.dropout:
        #     output = tf.nn.dropout(output, 0.5)
        output = tf.reshape(output, [self.batch_size, self.decoder_length])
        return output, None

    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            c_d = conditional_vectors.get_shape()
            conditional_vectors = tf.tile(conditional_vectors, [1, self.decoder_length])
            conditional_vectors = tf.reshape(conditional_vectors, [self.batch_size * self.decoder_length, int(c_d[-1])])
            fake_val, fake_rewards = self.validate_output(fake_outputs, conditional_vectors)
            real_val, _ = self.validate_output(self.pred_placeholder, conditional_vectors)
        return fake_val, real_val, fake_rewards

    # regular discriminator loss function
    def add_discriminator_loss(self, fake_preds, real_preds):
        if self.use_flip:
            real_labels = np.absolute(self.flag - np.random.uniform(0.8, 1., [self.batch_size, self.decoder_length]))
            fake_labels = np.absolute(self.flag - np.random.uniform(0., 0.2, [self.batch_size, self.decoder_length]))
            # real_labels = np.absolute(self.flag - tf.constant(0.9, shape=[self.batch_size, self.decoder_length]))
            # fake_labels = np.absolute(self.flag - tf.constant(0.1, shape=[self.batch_size, self.decoder_length]))
        else:
            # smooth one side (real - side)
            real_labels = tf.constant(0.9, shape=[self.batch_size, self.decoder_length])
            fake_labels = tf.zeros([self.batch_size, self.decoder_length])
        dis_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(fake_labels, fake_preds))
        dis_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(real_labels, real_preds))
        dis_loss = dis_loss_real + dis_loss_fake
        tf.summary.scalar("dis_loss", dis_loss)
        return dis_loss
    
    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])

        feed = {
            self.encoder_inputs: ct_t,
            self.decoder_inputs: dec_t,
            self.z: self.sample_z()
        }
        if self.use_flip:
            feed[self.flag] = np.asarray(np.random.randint(0, 1, [self.batch_size, 1]), dtype=np.float32)
        
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])

        if not train:
            pred = session.run([self.outputs], feed_dict=feed)
        else:
            gen_loss, dis_loss, pred, _, _= session.run([self.gen_loss, self.dis_loss, self.outputs, self.gen_op, self.dis_op], feed_dict=feed)
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss

        return pred, total_gen_loss, total_dis_loss
