from __future__ import print_function
from __future__ import division

import sys
import copy
import time
import numpy as np
import tensorflow as tf

from baseline_cnnlstm import BaselineModel
import properties as pr
import utils
import rnn_utils

"""

# case -2: CNN + 1LSTM + GAN + Policy gradient with Actor - critic (worst)
# case -1: LSTM + Regular training method

CNNs-LSTM models

case 0: CNN-32 + 1LSTM + Regular training method
case 10: 2 CNNs + 1LSTM + Regular training method + dropout + batchnorm + l1
case 11: 2 CNNs + 1LSTM + Regular training method + dropout + batchnorm
case 2: 2 CNNs + 1LSTM + Regular training method

# GAN Models
# case 1: 1CNN + 1LSTM + GAN + MSE                                                     26.609738679949896
# case 3: 4 CNNs + 1LSTM + GAN + MSE                                                     28.384105546904856
# case 4: 2 CNNs + 1LSTM + GAN + MSE + dropout & batchnorm                               25.417205501456465
# case 5: 2 CNNs + 1LSTM + GEN CNNs + GAN + MSE + dropout & batchnorm     

case 6: 2 CNNs + 1LSTM + GEN CNNs + GAN + MSE + dropout + batchnorm + l1 
case 7: 2 CNNs + 1LSTM + GEN CNNs + GAN + MSE


Instance noice, a trick for stabilizing gan
https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
https://github.com/soumith/ganhacks
"""


class MaskGan(BaselineModel):

    def __init__(self, gamma=0.9, dis_learning_rate=0.0002, gen_learning_rate=0.0002, critic_learning_rate=0.001, use_critic=False, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.gen_loss_type = 1
        self.gamma = gamma
        self.dis_learning_rate = dis_learning_rate
        self.gen_learning_rate = gen_learning_rate
        self.critic_learning_rate = critic_learning_rate
        # critic loss
        self.use_critic = use_critic
        # set up multiple cnns layers to generate outputs
        self.use_gen_cnn = True
        # add l1 norm to loss
        self.use_l1 = False
        self.is_clip = True
        self.beta1 = 0.5
        self.lamda = 100

    def init_ops(self):
        self.add_placeholders()
        self.outputs = self.inference()
        self.merged = tf.summary.merge_all()

    def inference(self):
        enc_output, dec, outputs, estimated_values, attention = self.create_generator()
        fake_preds, fake_rewards, real_preds = self.create_discriminator(enc_output, dec, outputs, attention)
        self.dis_loss, dis_loss_real, dis_loss_fake = self.add_discriminator_loss(fake_preds, real_preds)
        # use tanh on generator output or not. because y = tanh(x) if x > 0 then y always belongs to [0, 1]
        if self.use_gen_cnn:
            outputs = tf.tanh(outputs)
        if self.gen_loss_type == 0:
            # use log(G)
            self.gen_loss = self.add_generator_loss(fake_preds, fake_rewards, estimated_values)
        else: 
            # use mse of G & labels
            labels = tf.reshape(self.pred_placeholder, shape=(self.batch_size, self.decoder_length, self.grid_square))
            self.gen_loss = self.add_generator_loss(fake_preds, fake_rewards, estimated_values, outputs, labels)
        if self.use_critic:
            self.critic_loss = self.add_critic_loss(fake_rewards, estimated_values)
            self.critic_op = self.train_critic(self.critic_loss)
        self.gen_op = self.train_generator(self.gen_loss)
        self.dis_op = self.train_discriminator(self.dis_loss)
        
        return outputs
    
    def create_generator(self):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input()
            enc_output = self.exe_encoder(enc)
            # estimated_values [0, inf]
            attention = None
            if self.use_attention:
                # batch size x rnn_hidden_size
                inputs = tf.nn.embedding_lookup(self.attention_embedding, self.attention_inputs)
                attention = self.get_attention_rep(inputs)
            outputs, estimated_values = self.exe_decoder_critic(dec, enc_output, attention)
        return enc_output, dec, outputs, estimated_values, attention

    #perform decoder with critic estimated award
    def exe_decoder_critic(self, dec, enc_output, attention=None):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            # estimated_values [0, inf], outputs: [0, 1]
            params = copy.deepcopy(self.e_params)
            params["fw_cell"] = "gru_block"
            outputs, estimated_values = rnn_utils.execute_decoder_critic(dec, enc_output, self.decoder_length, params, attention, use_critic=self.use_critic, cnn_gen=self.use_gen_cnn)
            # batch_size x decoder_length x grid_size x grid_size
            outputs = tf.stack(outputs, axis=1)
            # batch_size x decoder_length
        return outputs, estimated_values

    # inputs is either from generator or from real context
    # enc_output: last hidden layer of encoder
    # dec: decoder vectors without pm2.5
    # output: fake_preds: fake prediction pm2.5 (logits value): this is D value before performing sigmoid
    def create_discriminator(self, enc_output, dec, outputs, attention=None):
        outputs_ = tf.expand_dims(tf.reshape(outputs, [self.batch_size, self.decoder_length, self.grid_size, self.grid_size]), axis=4)
        params = copy.deepcopy(self.e_params)
        params["de_output_size"] = 1
        params["fw_cell"] = "gru_block"
        dec_real = tf.concat([dec, tf.expand_dims(self.pred_placeholder, axis=4)], axis=4)
        dec_fake = tf.concat([dec, outputs_], axis=4)
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            # get probability of reality (either fake or real)
            fake_preds, fake_rewards = rnn_utils.execute_decoder_dis(dec_fake, enc_output, self.decoder_length, params, self.gamma, attention)
            real_preds, _ = rnn_utils.execute_decoder_dis(dec_real, enc_output, self.decoder_length, params, self.gamma, attention, False)
        return tf.squeeze(tf.stack(fake_preds, axis=1), [2]), fake_rewards, tf.squeeze(tf.stack(real_preds, axis=1), [2])

    # mse training
    def  add_critic_loss(self, rewards, estimated_values):
        loss = tf.losses.mean_squared_error(labels=rewards, predictions=estimated_values)
        tf.summary.scalar("critic_loss", loss)
        return loss

    # add generation loss
    # type 1: regular loss log(D(G))
    # type 2: ||(fake - real)||22
    def add_generator_loss(self, fake_preds, rewards, estimated_values, outputs=None, labels=None):
        r_ = tf.squeeze(tf.stack(rewards, axis=1))
        if self.use_critic:
            e_ = tf.squeeze(tf.stack(estimated_values, axis=1))
            advantages = tf.subtract(r_, e_)
            advantages = tf.clip_by_value(advantages, -5, 5)
        else:
            advantages = tf.abs(r_)
        # fake_labels = tf.constant(1, shape=[self.batch_size, self.decoder_length])
        if labels is not None:
            loss_values = tf.losses.mean_squared_error(labels, outputs)
        else:
            loss_values = tf.log_sigmoid(fake_preds)
        loss = tf.reduce_mean(tf.multiply(loss_values, tf.stop_gradient(advantages)))
        tf.summary.scalar("gen_loss", loss)
        return loss

    # regular discriminator loss function
    def add_discriminator_loss(self, fake_preds, real_preds):
        real_labels = tf.constant(0.9, shape=[self.batch_size, self.decoder_length])
        fake_labels = tf.zeros([self.batch_size, self.decoder_length])
        dis_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(fake_labels, fake_preds))
        dis_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(real_labels, real_preds))
        dis_loss = dis_loss_real + dis_loss_fake
        tf.summary.scalar("dis_loss", dis_loss)
        return dis_loss, dis_loss_real, dis_loss_fake

    def train_critic(self, loss):
        with tf.name_scope("train_critic"):
            critic_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate, self.beta1)
            critic_vars = [
                v for v  in tf.trainable_variables() if ("critic_linear_output" in v.op.name or "decoder_reward" in v.op.name or v.op.name.startswith("discriminator/rnn"))
            ]
            critic_grads = tf.gradients(loss, critic_vars)
            if self.is_clip:
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, 10.)
            critic_train_op = critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))        
        return critic_train_op
    
    def train_discriminator(self, loss):
        with tf.name_scope("train_discriminator"):
            dis_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate, self.beta1)
            dis_vars = [v for v in tf.trainable_variables() if v.op.name.startswith("discriminator")]
            dis_grads = tf.gradients(loss, dis_vars)
            if self.is_clip:
                dis_grads, _ = tf.clip_by_global_norm(dis_grads, 10.)
            dis_train_op = dis_optimizer.apply_gradients(zip(dis_grads, dis_vars))
            return dis_train_op

    # policy gradient
    def train_generator(self, loss):
        with tf.name_scope("train_generator"):
            gen_optimizer = tf.train.AdamOptimizer(self.gen_learning_rate, self.beta1)
            gen_vars = [v for v in tf.trainable_variables() if v.op.name.startswith("generator")]
            if self.use_l1:
                l1_norm = tf.contrib.layers.l1_regularizer(scale=self.lamada, gen_vars)
                loss += l1_norm
            
            if self.gen_loss_type == 0:
                # gradient ascent, maximum reward  => descent with minimizing the loss
                gen_grads = tf.gradients(-loss, gen_vars)
            else:
                # using mse without critic
                gen_grads = tf.gradients(loss, gen_vars)

            if self.is_clip:
                gen_grads, _ = tf.clip_by_global_norm(gen_grads, 10.)
            gen_train_op = gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
            return gen_train_op

    # using stride to reduce the amount of data to loop over time intervals
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, verbose=True, train=False, shuffle=True, stride=4):
        st = time.time()
        if not train:
            train_op = tf.no_op()
        dt_length = len(data)
        # print("data_size: ", dt_length)
        cons_b = self.batch_size * stride
        total_steps = dt_length // cons_b
        total_gen_loss = []
        total_dis_loss = []
        total_critic_loss = []
        preds = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        for step in xrange(total_steps):
            index = range(step * cons_b,
                          (step + 1) * cons_b, stride)
            # just the starting points of encoding batch_size,
            ct_t = ct[index]
            # switch batchsize, => batchsize * encoding_length (x -> x + 24)
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
            dec_t = ct_t + self.decoder_length

            feed = {
                self.encoder_inputs : ct_t,
                self.decoder_inputs: dec_t,
            }
            if self.use_attention:
                feed[self.attention_inputs] = ct_t

            if self.use_critic:
                gen_loss, dis_loss, critic_loss, pred, _, _, _= session.run(
                    [self.gen_loss, self.dis_loss, self.critic_loss, self.outputs, self.gen_op, 
                    self.dis_op, self.critic_op], feed_dict=feed)
                total_critic_loss.append(critic_loss) 
            else:
                gen_loss, dis_loss, pred, _, _= session.run([self.gen_loss, self.dis_loss, self.outputs, self.gen_op, self.dis_op], feed_dict=feed)
            
            total_gen_loss.append(gen_loss) 
            total_dis_loss.append(dis_loss)
            
            if verbose:
                if self.use_critic:
                    sys.stdout.write('\r{} / {} gen_loss = {} | dis_loss = {} | critic_loss = {}'.format(
                        step, total_steps, np.mean(total_gen_loss), np.mean(total_dis_loss), np.mean(total_critic_loss)))
                else:
                    sys.stdout.write('\r{} / {} gen_loss = {} | dis_loss = {}'.format(
                        step, total_steps, np.mean(total_gen_loss), np.mean(total_dis_loss)))

                sys.stdout.flush()

            if not train:
                preds.append(pred)

        if verbose:
            sys.stdout.write("\r")
            dur = time.time() - st
            print("Running time: %.2f" % dur)
        if train_writer is not None:
            total_gen_loss, total_dis_loss = np.mean(total_gen_loss), np.mean(total_dis_loss)
            summary = tf.Summary()
            summary.value.add(tag= "Generator Loss", simple_value=total_gen_loss)
            summary.value.add(tag= "Discriminator Loss", simple_value=total_dis_loss)
            if self.use_critic:
                summary.value.add(tag= "Critic Loss", simple_value=np.mean(total_critic_loss))
            train_writer.add_summary(summary, num_epoch)
        return preds