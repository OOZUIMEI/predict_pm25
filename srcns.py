import tensorflow as tf

from tnetlstm import TNetLSTM
import rnn_utils

#APNet for transportation data

class SVCNs(TNetLSTM):

    def __init__(self, **kwargs):
        super(SVCNs, self).__init__(**kwargs)
        # alpha is used for generator loss function

    # generator cnn layers
    def add_msf_networks(self, inputs, activation=tf.nn.relu, is_dis=False):
        # input (64, 32, 32, 1) output (64, 16, 16, 16)
        msf1 = rnn_utils.get_cnn_unit(inputs, 16, (3,3), activation, padding="SAME", name="sample_1", strides=(1,1))
        # input (64, 32, 32, 16) output (64, 16, 16, 16)
        msf1_down = rnn_utils.get_maxpool(msf1, (2,2), padding="SAME", name="down_sample_1", strides=(2,2))
        # input (64, 16, 16, 16) output (64, 16, 16, 32)
        msf2 = rnn_utils.get_cnn_unit(msf1_down, 32, (3,3), activation, padding="SAME", name="sample_1", strides=(1,1))
        # input (64, 16, 16, 32) output (64, 8, 8, 32)
        msf2_down = rnn_utils.get_maxpool(msf2, (2,2), padding="SAME", name="down_sample_2", strides=(2,2))
        # input (64, 8, 8, 32) output (64, 8, 8, 128)
        msf3 = rnn_utils.get_cnn_unit(msf2_down, 64, (3,3), activation, padding="SAME", name="sample_1", strides=(1,1))
        msf32 = rnn_utils.get_cnn_unit(msf3, 64, (3,3), activation, padding="SAME", name="sample_1", strides=(1,1))
        msf3_down = rnn_utils.get_maxpool(msf32, (2,2), padding="SAME", name="down_sample_2", strides=(2,2))
        msf3_ = tf.layers.flatten(msf3_down)
        return msf3_
