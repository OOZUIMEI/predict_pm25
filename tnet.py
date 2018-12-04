from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tgan import TGAN
import properties as pr
import rnn_utils

#APNet for transportation data

class TNet(TGAN):

    def __init__(self, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.alpha = 0
        # alpha is used for generator loss function

    def inference(self, is_train=True):
        fake_outputs, _ = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        self.gen_loss = self.get_generator_loss(None, fake_outputs)
        if is_train:
            self.gen_op = self.train_generator(self.gen_loss)
        return fake_outputs
    
           
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
            pred, gen_loss = session.run([self.outputs, self.gen_loss], feed_dict=feed)
            total_gen_loss += gen_loss
        else:
            gen_loss, pred, _, = session.run([self.gen_loss, self.outputs, self.gen_op], feed_dict=feed)
            total_gen_loss += gen_loss

        return pred, total_gen_loss, 0
