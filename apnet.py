import numpy as np
import tensorflow as tf
from copy import deepcopy
from apgan import APGan
import properties as pr
import rnn_utils
# add noise to input & label
# alpha is zero => use only content loss in APNet

class APNet(APGan):

    def __init__(self, **kwargs):
        super(APNet, self).__init__(**kwargs)
        self.dropout = 0.5
        self.alpha = 0
        self.mtype = 3
        self.use_attention = False

    def inference(self, is_train=True):
        fake_outputs, _ = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        self.gen_loss = self.get_generator_loss(None, fake_outputs, None)
        if is_train:
            self.gen_op = self.train_generator(self.gen_loss)
        return fake_outputs
    
    #perform decoder to produce outputs of the generator
    # dec_hidden_vectors: bs x rnn_hidden_units
    def exe_decoder(self, dec_hidden_vectors, fn_state):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = deepcopy(self.e_params)
            if "gru" in self.e_params["fw_cell"]:
                params["fw_cell"] = "gru_block"
            else:
                params["fw_cell"] = "lstm_block"
            outputs = rnn_utils.execute_decoder_cnn(None, fn_state, self.decoder_length, params, dec_hidden_vectors, self.use_cnn, self.use_gen_cnn, self.mtype, self.use_batch_norm, self.dropout)
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.reshape(outputs, [self.batch_size, self.decoder_length, pr.grid_size * pr.grid_size])
        return outputs

    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])

        feed = {
            self.encoder_inputs : ct_t,
            self.decoder_inputs: dec_t
        }
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])

        if not train:
            pred, gen_loss = session.run([self.outputs, self.gen_loss], feed_dict=feed)
            total_gen_loss += gen_loss
        else:
            gen_loss, pred, _= session.run([self.gen_loss, self.outputs, self.gen_op], feed_dict=feed)
            total_gen_loss += gen_loss

        return pred, total_gen_loss, 0
