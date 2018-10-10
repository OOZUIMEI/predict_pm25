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
 
# case 1: CNN-32 + 1LSTM + Regular training method                                     10.924795863322162
# case 2: 2 CNNs + 1LSTM + Regular training method                                     10.814500091143062
# case 3: 2 CNNs + 1LSTM + Regular training method + dropout                           10.94596252670948
# case 4: 2 CNNs + 1LSTM + Regular training method + dropout + batchnorm               10.877427656329758
# case 5: 2 CNNs + 1LSTM + Regular training method + Gen with CNNs + DP + BN           14.06973230155157    

# GAN Models (LSTM + CNNs)
#case 1: 2 CNNs + 1LSTM + MSE                                                           11.115757976757626                                                      
#case 3: 2 CNNs + 1LSTM + MSE + dp & batchnorm                                          11.401404469056558
#case 4: 2 CNNs + 1LSTM + GEN CNNs + MSE + dp                                           11.212858479217973
#case 5: 2 CNNs + 1LSTM + GEN CNNs + MSE + dp & batchnorm                               11.280975725838095

# RNN                                                                                   36.91
# LSTM 1 layers                                                                         36.41                                                                     
# GRU 1 layers                                                                          36.48             
# GRU + LSTM                                                                            36.05
# Neural networks                                                                       36.36

# ADAIN

# Stacked Autoencoder

# GAN + CNNs



Instance noice, a trick for stabilizing gan
https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
https://github.com/soumith/ganhacks
"""


class MaskGan(BaselineModel):

    def __init__(self, gamma=0.9, learning_rate=0.0002, critic_learning_rate=0.001, use_critic=False, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.gen_loss_type = 1
        self.gamma = gamma
        self.learning_rate = learning_rate
        # critic loss
        self.use_critic = use_critic
        # set up multiple cnns layers to generate outputs
        self.use_gen_cnn = False
        # add l1 norm to loss
        self.use_l1 = False
        self.is_clip = True
        self.beta1 = 0.5
        self.lamda = 100
        self.gmtype = 3
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)

    def init_ops(self):
        self.add_placeholders()
        self.outputs = self.inference()
        # self.merged = tf.summary.merge_all()

    def inference(self):
        enc_output, dec, outputs, tanh_inputs, estimated_values, attention = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        fake_preds, fake_rewards, real_preds = self.create_discriminator(enc_output, dec, outputs, attention)
        # use tanh on generator output or not. because y = tanh(x) if x > 0 then y always belongs to [0, 1]
        self.dis_loss = self.add_discriminator_loss(fake_preds, real_preds)
        self.gen_loss = self.get_generator_loss(fake_preds, fake_rewards, estimated_values, tanh_inputs)
        if self.use_critic:
            self.critic_loss = self.add_critic_loss(fake_rewards, estimated_values)
            self.critic_op = self.train_critic(self.critic_loss)
        self.gen_op = self.train_generator(self.gen_loss)
        self.dis_op = self.train_discriminator(self.dis_loss)
        
        return outputs
       
    def create_generator(self, enc, dec, att):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input(enc, dec)
            enc_output = self.exe_encoder(enc, False, 0.0)
            # estimated_values [0, inf]
            attention = None
            if self.use_attention:
                # batch size x rnn_hidden_size
                inputs = tf.nn.embedding_lookup(self.attention_embedding, att)
                attention = self.get_attention_rep(inputs)
            outputs, estimated_values = self.exe_decoder_critic(dec, enc_output, attention)
            tanh_outputs = None
            if self.use_gen_cnn:
                tanh_outputs = tf.tanh(outputs)
            else:
                tanh_outputs = outputs
        return enc_output, dec, outputs, tanh_outputs, estimated_values, attention

    #perform decoder with critic estimated award
    def exe_decoder_critic(self, dec, enc_output, attention=None):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            # estimated_values [0, inf], outputs: [0, 1]
            params = copy.deepcopy(self.e_params)
            params["fw_cell"] = "gru_block"
            outputs, estimated_values = rnn_utils.execute_decoder_critic(dec, enc_output, self.decoder_length, params, attention, use_critic=self.use_critic, cnn_gen=self.use_gen_cnn, mtype=self.gmtype)
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
            fake_preds, fake_rewards = rnn_utils.execute_decoder_dis(dec_fake, enc_output, self.decoder_length, params, self.gamma, attention, mtype=self.gmtype)
            real_preds, _ = rnn_utils.execute_decoder_dis(dec_real, enc_output, self.decoder_length, params, self.gamma, attention, False, mtype=self.gmtype)
        return tf.squeeze(tf.stack(fake_preds, axis=1), [2]), fake_rewards, tf.squeeze(tf.stack(real_preds, axis=1), [2])

    # mse training
    def  add_critic_loss(self, rewards, estimated_values):
        loss = tf.losses.mean_squared_error(labels=rewards, predictions=estimated_values)
        tf.summary.scalar("critic_loss", loss)
        return loss

    def get_generator_loss(self, fake_preds, fake_rewards, estimated_values, outputs):
        if self.use_gen_cnn:
            outputs = tf.tanh(outputs)
        if self.gen_loss_type == 0:
            # use log(G)
            gen_loss = self.add_generator_loss(fake_preds, fake_rewards, estimated_values)
        else: 
            # use mse of G & labels
            labels = tf.reshape(self.pred_placeholder, shape=(self.batch_size, self.decoder_length, self.grid_square))
            gen_loss = self.add_generator_loss(fake_preds, fake_rewards, estimated_values, outputs, labels)
        return gen_loss

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
        return dis_loss

    def train_critic(self, loss):
        with tf.name_scope("train_critic"):
            critic_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
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
            dis_grads, dis_vars = self.get_optimization(loss, "discriminator")
            dis_grads, _ = tf.clip_by_global_norm(dis_grads, 10.)
            dis_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            dis_train_op = dis_optimizer.apply_gradients(zip(dis_grads, dis_vars))
            return dis_train_op
    
    # policy gradient
    def train_generator(self, loss):
        with tf.name_scope("train_generator"):
            gen_vars = [v for v in tf.trainable_variables() if v.op.name.startswith("generator")]
            gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1) 
            if self.gen_loss_type == 0:
                gen_grads = tf.gradients(-loss, gen_vars)
            else:
                gen_grads = tf.gradients(loss, gen_vars)
            gen_grads, _ = tf.clip_by_global_norm(gen_grads, 10.)
            gen_train_op = gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
            return gen_train_op

    def get_optimization(self, loss, name_scope):
        vars_ = [v for v in tf.trainable_variables() if v.op.name.startswith(name_scope)]
        #grads = self.optimizer.compute_gradients(loss, vars_)
        grads = tf.gradients(loss, vars_)
        return grads, vars_

    # using stride to reduce the amount of data to loop over time intervals
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, verbose=True, train=False, shuffle=True, stride=4):
        st = time.time()
        if not train:
            self.gen_op = tf.no_op()
            self.dis_op = tf.no_op()
        dt_length = len(data)
        # print("data_size: ", dt_length)
        total_gen_loss = []
        total_dis_loss = []
        total_critic_loss = []
        preds = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
      
        strides = [4, 8, 12]
        if train:
            np.random.shuffle(strides)
            stride = strides[0]
      
        cons_b = self.batch_size * stride

        total_steps = dt_length // cons_b
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

    # reference from cifa_multiple_gpu code
    def average_gradients(self, tower_grads):
        average_grads = []
        vars_ = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            vars_.append(v)
            average_grads.append(grad)
        average_grads, _ = tf.clip_by_global_norm(average_grads, 10.)
        average_grads = zip(average_grads, vars_)
        return average_grads

    def prefetch_queue(self, session, enqueue, data, total_steps, global_step, preload_nums=8):
        end_step = global_step + preload_nums
        if end_step < total_steps:
            for step in xrange(global_step, end_step):
                index = range(step * cons_b, (step + 1) * cons_b, stride)
                ct_t = ct[index]
                ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
                dec_t = ct_t + self.decoder_length
                q_dc = {
                    self.encoder_inputs: ct_t,
                    self.decoder_inputs: dec_t,
                    self.attention_inputs: ct_t
                }
                session.run(enqueue, feed_dict=q_dc)

    def run_multiple_gpu(self, sess, url_data, url_attention, url_weight, train_writer=None, offset=0, train=False, shuffle=True, stride=4, gpu_nums=2, max_steps=1200):
        if not train:
            train_op = tf.no_op()
        max_load = gpu_nums * 2
        queue = tf.FIFOQueue(max_load, dtypes=[tf.int32, tf.int32, tf.int32], name="data_lookup")
        enqueue = queue.enqueue([self.encoder_inputs, self.decoder_inputs, self.attention_inputs])
        # prepare data
        devices = pr.gpu_devices.split(",")
        with tf.variable_scope(tf.get_variable_scope()):
            dis_vars, gen_vars = None,None
            tower_dis_grads = []
            tower_gen_grads = []
            for x in xrange(gpu_nums):
                with tf.device("gpu:%s" % devices[x]):
                    with tf.name_scope("tenant_%s" % x):
                        enc, dec, att = queue.dequeue()
                        enc_output, dec, outputs, tanh_inputs, estimated_values, attention = self.create_generator(enc, dec, att)
                        fake_preds, fake_rewards, real_preds = self.create_discriminator(enc_output, dec, outputs, attention)
                        dis_loss = self.add_discriminator_loss(fake_preds, real_preds)
                        gen_loss = self.get_generator_loss(fake_preds, fake_rewards, estimated_values, tanh_inputs)
                        dis_grads, _ = self.get_optimization(dis_loss, "discriminator")
                        gen_grads, _ = self.get_optimization(gen_loss, "generator")
                        tower_gen_grads.append(gen_grads)
                        tower_dis_grads.append(dis_grads)
                        tf.get_variable_scope().reuse_variables()
        gen_grads = self.average_gradients(tower_gen_grads)   
        dis_grads = self.average_gradients(tower_dis_grads)
        gen_train_op = self.optimizer.apply_gradients(gen_grads)        
        dis_train_op = self.optimizer.apply_gradients(dis_grads)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # load data
        print("Loading dataset")
        datasets = utils.load_file(url_data)
        if url_attention:
            att_data = utils.load_file(url_attention)
        lt = len(datasets)
        data, _ = utils.process_data_grid(lt, self.batch_size, self.encoder_length, self.decoder_length, True)
        self.set_data(datasets, data, None, att_data)
        self.assign_datasets(sess)
        dt_length = len(data)
        cons_b = self.batch_size * stride
        total_steps = dt_length // cons_b
        preds = []
        summary = tf.Summary()
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        self.prefetch_queue(sess, enqueue, data, total_steps, 0, max_load)
        # for i in xrange(max_steps):
        #     for b in xrange(total_steps):
        #         if b and b % max_load == 0:
        #             self.prefetch_queue(session, enqueue, data, total_steps, b, max_load)
        #         if train:
        #             d_loss, g_loss, _, _ = sess.run([gen_loss, dis_loss, gen_train_op, dis_train_op])
        #             if train_writer is not None:
        #                 summary.value.add(tag= "Generator Loss", simple_value=d_loss)
        #                 summary.value.add(tag= "Discriminator Loss", simple_value=g_loss)
        #                 train_writer.add_summary(summary, offset + i)
        #             if epoch % 10 == 0 or (epoch + 1) == max_steps:
        #                 utils.update_progress((i + 1.0) / p.total_iteration)
        #                 saver.save(session, 'weights/%s.weights' % url_weight)
        #         else:
        #             pred = sess.run([outputs])
        #             preds.append(pred)
        return preds
