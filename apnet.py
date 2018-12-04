from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from apgan import APGan
import properties as pr

# add noise to input & label
# alpha is zero => use only content loss in APNet

class APNet(APGan):

    def __init__(self, **kwargs):
        super(APNet, self).__init__(**kwargs)
        self.alpha = 0

    def inference(self, is_train=True):
        fake_outputs, _ = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        if is_train:
            self.gen_loss = self.get_generator_loss(None, fake_outputs, None)
            self.gen_op = self.train_generator(self.gen_loss)
        return fake_outputs
    
    # add generation loss
    # use log_sigmoid instead of log because fake_vals is w * x + b (not the probability value)
    def add_generator_loss(self, fake_vals, outputs, labels, fake_rewards=None):
        mse_loss = tf.losses.mean_squared_error(labels, outputs)
        if not fake_rewards is None:
            print("Using reinsforcement learning")
            advatages = tf.abs(fake_rewards)
            loss = tf.reduce_mean(tf.multiply(mse_loss, tf.stop_gradient(advatages)))
        else:
            print("Using combined loss function")
            if self.alpha:
                sigmoid_loss = self.alpha * tf.log_sigmoid(fake_vals)
                # sigmoid_loss = self.alpha * tf.losses.sigmoid_cross_entropy(fake_vals, tf.constant(1., shape=[self.batch_size, self.decoder_length]))
                # normal lossmse + (-log(D(G)))
                loss_values = mse_loss - sigmoid_loss
                #loss_values = sigmoid_loss
                loss = tf.reduce_mean(loss_values)
            else:
                loss = mse_loss
        return loss
    
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
        }
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])

        if not train:
            pred = session.run([self.outputs], feed_dict=feed)
        else:
            gen_loss, pred, _= session.run([self.gen_loss, self.outputs, self.gen_op], feed_dict=feed)
            total_gen_loss += gen_loss

        return pred, total_gen_loss, 0
