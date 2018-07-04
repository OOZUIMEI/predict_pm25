from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell, MultiRNNCell

import properties as prp
import utils


def get_cell(cell_type, size):
    if cell_type == "layer_norm_basic":
        cell = LayerNormBasicLSTMCell(size)
    elif cell_type == "gru":
        cell = GRUCell(size)
    else:
        cell = BasicLSTMCell(size)
    return cell


# rnn through each 30', 1h 
def execute_sequence(inputs, params):
    # 1 is bidireciton
    # note: state_size of MultiRNNCell must be equal to size input_size
    fw_cell = get_cell(params["fw_cell"], params["fw_cell_size"])
    if "rnn_layer" in params and params["rnn_layer"] > 1:
        fw_cell = MultiRNNCell([fw_cell] * params["rnn_layer"])
    if params["type"] == 1:
        bw_cell = get_cell(params["bw_cell"], params["bw_cell_size"])
        
        if "rnn_layer" in params and params["rnn_layer"] > 1:
            bw_cell = MultiRNNCell([bw_cell] * params["rnn_layer"])
        
        outputs, fn_state  = tf.nn.static_bidirectional_rnn(
            fw_cell,
            bw_cell,
            inputs, 
            dtype=np.float32
        )
    # default is one direction static rnn
    else:        
        outputs, fn_state = tf.nn.static_rnn(
            fw_cell,
            inputs,
            dtype=np.float32
        )
    # new_h, (new_c, new_h)
    return outputs, fn_state


def execute_decoder(inputs, init_state, sequence_length, params):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]))
    dec_out = None
    outputs = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    for t in xrange(sequence_length):
        dec_in = tf.concat([inputs[:, t], pm2_5], axis=1)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        pm2_5 = tf.layers.dense(dec_out, 
                        params["de_output_size"],
                        name="decoder_output",
                        activation=tf.nn.sigmoid)
        outputs.append(pm2_5)
    return outputs



def execute_decoder_mask(inputs, init_state, sequence_length, is_training=True, masks=None):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = None
    dec_out = None
    for t in xrange(sequence_length):
        dec_in = inputs[:, t]
        if is_training and masks:
            dec_in = tf.where(masks[:, t], dec_in, pm2_5)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        pm2_5 = tf.dense(dec_out, 
                        name="decoder_output",
                        activation=None)
    return dec_out, dec_state