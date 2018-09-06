from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell, MultiRNNCell, LSTMBlockFusedCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
import properties as prp
import utils


def get_cell(cell_type, size):
    if cell_type == "layer_norm_basic":
        cell = LayerNormBasicLSTMCell(size)
    elif cell_type == "lstm_block_fused":
        cell = tf.contrib.rnn.LSTMBlockFusedCell(size)
    elif cell_type == "cudnn_lstm":
        cell = CudnnLSTM(1, size)
    elif cell_type == "cudnn_gru":
        cell = CudnnGru(1, size)
    else:
        cell = BasicLSTMCell(size)
    return cell


# rnn through each 30', 1h 
def execute_sequence(inputs, params):
    if prp.device and "gpu" not in prp.device:
        params["fw_cell"] = "basic"
    # 1 is bidireciton
    # note: state_size of MultiRNNCell must be equal to size input_size
    fw_cell = get_cell(params["fw_cell"], params["fw_cell_size"])
    if "cudnn" in params["fw_cell"] or params["fw_cell"] == "lstm_block_fused":
        shape = inputs.get_shape()
        inputs = tf.transpose(inputs, [1, 0, 2])
        if "cudnn" in params["fw_cell"]:
            outputs, fn_state = fw_cell(inputs)
        else:
            outputs, fn_state = fw_cell(inputs, dtype=tf.float32)
        h = tf.squeeze(fn_state[1])
        c = tf.squeeze(fn_state[0])
        if shape[0] == 1:
            fn_state = (tf.reshape(h, [1, h.get_shape()[0]]), tf.reshape(c, c.get_shape()[0]))
        else:
            fn_state = (h, c)
        outputs = tf.transpose(outputs, [1, 0, 2])
    else:
        if "rnn_layer" in params and params["rnn_layer"] > 1:
            fw_cell = MultiRNNCell([fw_cell] * params["rnn_layer"])
        exe_inputs = tf.unstack(inputs, axis=1)
        outputs, fn_state = tf.nn.static_rnn(
            fw_cell,
            exe_inputs,
            dtype=np.float32
        )
    return outputs, fn_state


# not perform cnn on pm2_5 output
def execute_decoder(inputs, init_state, sequence_length, params, attention=None, dropout=None):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]))
    dec_out = None
    outputs = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    for t in xrange(sequence_length):
        dec_in = tf.concat([inputs[:, t], pm2_5], axis=1)
        # need to do cnn here
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        # pm2_5 with shape batchsize x (grid_size * grid_size)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        pm2_5 = tf.layers.dense(dec_out, 
                        params["de_output_size"],
                        name="decoder_output",
                        activation=tf.nn.sigmoid)
        if dropout:
            pm2_5 = tf.nn.dropout(pm2_5, dropout)
        outputs.append(pm2_5)
    return outputs


# perform cnn on pm2_5 output
def execute_decoder_cnn(inputs, init_state, sequence_length, params, attention=None, dropout=None, mask=None):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]), dtype=np.float32)
    dec_out = None
    outputs = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        pm2_5_t = tf.expand_dims(tf.reshape(pm2_5, [params["batch_size"], params["grid_size"], params["grid_size"]]), 3)
        dec_in = tf.concat([input_t, pm2_5_t], axis=3)
        # need to do cnn here
        dec_in = tf.transpose(dec_in, [0,3,1,2])
        dec_in = tf.expand_dims(dec_in, 4)
        dec_in = tf.layers.conv3d(
            inputs=dec_in,
            strides=params["decoder_strides"],
            filters=params["decoder_filter"],
            kernel_size=params["decoder_kernel"],
            padding="valid"
        )
        # bs x 12 x 12
        dec_in = tf.reshape(tf.reshape(tf.squeeze(dec_in), [-1]), [params["batch_size"], 144])
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        pm2_5 = tf.layers.dense(dec_out, 
                        params["de_output_size"],
                        name="decoder_output",
                        activation=tf.nn.sigmoid)
        if dropout:
            pm2_5 = tf.nn.dropout(pm2_5, dropout)
        outputs.append(pm2_5)
    return outputs


# perform cnn on pm2_5 output
# estimated value of critic: 0 - inf
# outputs: pm2.5 images
def execute_decoder_critic(inputs, init_state, sequence_length, params, attention=None, dropout=None, mask=None):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]), dtype=np.float32)
    dec_out = None
    outputs = []
    estimated_values = []
    cell_dec = get_cell("basic", params["fw_cell_size"])
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        pm2_5_t = tf.expand_dims(tf.reshape(pm2_5, [params["batch_size"], params["grid_size"], params["grid_size"]]), 3)
        dec_in = tf.concat([input_t, pm2_5_t], axis=3)
        # need to do cnn here
        dec_in = tf.transpose(dec_in, [0,3,1,2])
        dec_in = tf.expand_dims(dec_in, 4)
        dec_in = tf.layers.conv3d(
            inputs=dec_in,
            strides=params["decoder_strides"],
            filters=params["decoder_filter"],
            kernel_size=params["decoder_kernel"],
            padding="valid"
        )
        # bs x 12 x 12
        dec_in = tf.reshape(tf.reshape(tf.squeeze(dec_in), [-1]), [params["batch_size"], 144])
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        # belong to generator
        pm2_5 = tf.layers.dense(dec_out, params["de_output_size"], name="decoder_output", activation=tf.nn.sigmoid)
        if dropout:
            pm2_5 = tf.nn.dropout(pm2_5, dropout)
        # belong to critic
        e_value = tf.layers.dense(dec_out, 1, name="critic_linear_output", activation=None)
        outputs.append(pm2_5)
        estimated_values.append(e_value)
    return outputs, estimated_values

# output: predictions - probability [0, 1], rewards [0, 1]
def execute_decoder_dis(inputs, init_state, sequence_length, params, gamma, attention=None, is_fake=True, dropout=None, mask=None):
    # push final state of encoder to decoder
    dec_state = init_state
    dec_out = None
    predictions = []
    cell_dec = get_cell("basic", params["fw_cell_size"])
    rewards = []
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        # need to do cnn here
        dec_in = tf.transpose(input_t, [0,3,1,2])
        dec_in = tf.expand_dims(dec_in, 4)
        dec_in = tf.layers.conv3d(
            inputs=dec_in,
            strides=params["decoder_strides"],
            filters=params["decoder_filter"],
            kernel_size=params["decoder_kernel"],
            padding="valid"
        )
        # bs x 12 x 12
        dec_in = tf.reshape(tf.reshape(tf.squeeze(dec_in), [-1]), [params["batch_size"], 144])
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        # belong to disciminator
        dec_out = tf.layers.dense(dec_out, 1, name="decoder_linear_value", activation=None)
        if dropout:
            dec_out = tf.nn.dropout(dec_out, dropout)
        # belong to critic
        pred_value = tf.log_sigmoid(dec_out, name="decoder_reward")
        if is_fake:
            if rewards:
                lt = len(rewards)
                for i in xrange(lt):
                    rewards[i] += np.power(gamma, (lt - i)) * pred_value
            # r * gammar ^ 0
            rewards.append(pred_value)
        predictions.append(dec_out)
    return predictions, rewards


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