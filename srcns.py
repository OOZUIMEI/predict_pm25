import tensorflow as tf

from tnetlstm import TNetLSTM
import copy
import properties as pr
import rnn_utils

#APNet for transportation data

class SRCN(TNetLSTM):

    def __init__(self, **kwargs):
        super(SRCN, self).__init__(**kwargs)
        self.use_attention = False
        # alpha is used for generator loss function

    # generator cnn layers
    def add_msf_networks(self, inputs, activation=tf.nn.relu, is_dis=False):
        inputs = tf.reshape(inputs, shape=(pr.batch_size * self.encoder_length, self.grid_size, self.grid_size, 1))
        # input (64, 32, 32, 1) output (64, 16, 16, 16)
        msf1 = rnn_utils.get_cnn_unit(inputs, 16, (3,3), activation, padding="SAME", name="sample_1", strides=(1,1))
        # input (64, 32, 32, 16) output (64, 16, 16, 16)
        msf1_down = rnn_utils.get_maxpool(msf1, (2,2), padding="SAME", name="down_sample_1", strides=(2,2))
        # input (64, 16, 16, 16) output (64, 16, 16, 32)
        msf2 = rnn_utils.get_cnn_unit(msf1_down, 32, (3,3), activation, padding="SAME", name="sample_2", strides=(1,1))
        # input (64, 16, 16, 32) output (64, 8, 8, 32)
        msf2_down = rnn_utils.get_maxpool(msf2, (2,2), padding="SAME", name="down_sample_2", strides=(2,2))
        # input (64, 8, 8, 32) output (64, 8, 8, 128)
        msf3 = rnn_utils.get_cnn_unit(msf2_down, 64, (3,3), activation, padding="SAME", name="sample_3", strides=(1,1))
        msf32 = rnn_utils.get_cnn_unit(msf3, 64, (3,3), activation, padding="SAME", name="sample_4", strides=(1,1))
        msf3_down = rnn_utils.get_maxpool(msf32, (2,2), padding="SAME", name="down_sample_4", strides=(2,2))
        msf3_ = tf.layers.flatten(msf3_down)
        return msf3_

    def exe_encoder(self, enc):
        params = copy.deepcopy(self.e_params)
        params["fw_cell_size"] = 256
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            msf_output = self.add_msf_networks(enc)
            msf_output = self.add_hidden_layers(msf_output)
            hidden_output = tf.reshape(msf_output, shape=(pr.batch_size, self.encoder_length, 256))
            # go to lstm
            _, final_state = rnn_utils.execute_sequence(hidden_output, params)
        return final_state

    # perform decoder to produce outputs of the generator
    # modify th e
    def exe_decoder(self, dec_hidden_vectors):
        params = copy.deepcopy(self.e_params)
        if "gru" in self.e_params["fw_cell"]:
            params["fw_cell"] = "gru_block"
        elif "rnn" in self.e_params["fw_cell"]:
            params["fw_cell"] = "rnn"
        else:
            params["fw_cell"] = "lstm_block"
        params["de_output_size"] = 1024
        params["fw_cell_size"] = 256
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs = tf.zeros([pr.batch_size, self.decoder_length, 256])
            outputs = rnn_utils.execute_decoder(dec_inputs, dec_hidden_vectors, self.decoder_length, params)
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.reshape(outputs, [pr.batch_size, self.decoder_length, self.grid_size, self.grid_size])
        return outputs