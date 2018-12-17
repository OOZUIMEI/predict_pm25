import tensorflow as tf
from copy import deepcopy
from apgan import APGan
import properties as pr
import rnn_utils
# add noise to input & label
# alpha is zero => use only content loss in APNet

class APGAN_LSTM(APGan):

    def __init__(self, **kwargs):
        super(APGAN_LSTM, self).__init__(**kwargs)
        self.z_dim = [pr.batch_size, 128]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)   
        self.dropout = 0.5
        self.alpha = 0.0002 
        #self.alpha = 0

    #perform decoder to produce outputs of the generator
    # dec_hidden_vectors: bs x rnn_hidden_units
    def exe_decoder(self, dec_hidden_vectors, fn_state):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = deepcopy(self.e_params)
            if "gru" in self.e_params["fw_cell"]:
                params["fw_cell"] = "gru_block"
            else:
                params["fw_cell"] = "lstm_block"
            # concatenate with z noise
            dec_hidden_vectors = tf.concat([dec_hidden_vectors, self.z], axis=1)
            dec_hidden_vectors = tf.layers.dense(dec_hidden_vectors, self.rnn_hidden_unis, name="generation_hidden_seed", activation=tf.nn.tanh)
            outputs = rnn_utils.execute_decoder_cnn(None, fn_state, self.decoder_length, params, dec_hidden_vectors, self.use_cnn, self.use_gen_cnn, self.mtype, self.use_batch_norm, self.dropout)
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.reshape(outputs, [pr.batch_size, self.decoder_length, pr.grid_size * pr.grid_size])

    """ 
    # performing GRU before final decision fake/real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    def validate_output(self, inputs, conditional_vectors):
        conditional_vectors = tf.reshape(conditional_vectors, [pr.batch_size * self.decoder_length, self.rnn_hidden_units])
        inputs = tf.reshape(inputs, [pr.batch_size * self.decoder_length, pr.grid_size, pr.grid_size, 1])
        inputs_rep = rnn_utils.get_cnn_rep(inputs, self.gmtype, tf.nn.leaky_relu, 8, self.use_batch_norm, self.dropout, False)
        inputs_rep = tf.layers.flatten(inputs_rep)
        inputs_rep = tf.concat([inputs_rep, conditional_vectors], axis=1)
        inputs_rep_shape = inputs_rep.get_shape()
        inputs_rep = tf.reshape(inputs_rep, [pr.batch_size, self.decoder_length, int(inputs_rep_shape[-1])])
        # push through a GRU layer
        rnn_outputs, _ = rnn_utils.execute_sequence(inputs_rep, self.e_params)
        # real or fake value
        output = tf.layers.dense(rnn_outputs, 1, name="validation_value")
        output = tf.layers.flatten(output)
        # rewards = None
        # if is_fake:
        #     rewards = [None] * self.decoder_length
        #     pred_value = tf.log_sigmoid(output)
        #     pred_values = tf.unstack(pred_value, axis=1)
        #     for i in xrange(self.decoder_length - 1, -1,-1):
        #         rewards[i] = pred_values[i]
        #         if i != (self.decoder_length - 1):
        #             for j in xrange(i + 1, self.decoder_length):
        #                 rewards[i] += np.power(self.gamma, (j - i)) * rewards[i]
        return output, None   
    """
