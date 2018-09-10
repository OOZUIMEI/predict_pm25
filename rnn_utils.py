from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell, MultiRNNCell, LSTMBlockFusedCell, LSTMBlockCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
import properties as prp
import utils


def get_cell(cell_type, size, layers=1):
    if cell_type == "layer_norm_basic":
        cell = LayerNormBasicLSTMCell(size)
    elif cell_type == "lstm_block_fused":
        cell = tf.contrib.rnn.LSTMBlockFusedCell(size)
    elif cell_type == "cudnn_lstm":
        cell = CudnnLSTM(layers, size)
    elif cell_type == "cudnn_gru":
        cell = CudnnGru(layers, size)
    elif cell_type == "lstm_block":
        cell = LSTMBlockCell(size)
    else:
        cell = BasicLSTMCell(size)
    return cell


# rnn through each 30', 1h 
def execute_sequence(inputs, params):
    if prp.device and "gpu" not in prp.device:
        if "cudnn" in params["fw_cell"]:
            params["fw_cell"] = "lstm_block_fused"
        else:
            params["fw_cell"] = "basic"
    # 1 is bidireciton
    # note: state_size of MultiRNNCell must be equal to size input_size
    fw_cell = get_cell(params["fw_cell"], params["fw_cell_size"])
    if "cudnn" in params["fw_cell"] or params["fw_cell"] == "lstm_block_fused":
        inputs = tf.transpose(inputs, [1, 0, 2])
        if "cudnn" in params["fw_cell"]:
            outputs, fn_state = fw_cell(inputs)
            c = tf.squeeze(fn_state[1], [0])
            h = tf.squeeze(fn_state[0], [0])
            fn_state = (c, h)
        else:
            outputs, fn_state = fw_cell(inputs, dtype=tf.float32)
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
        dec_in = get_cnn_rep(dec_in)
        dec_in = tf.flatten(dec_in))
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
# this is for generator
def execute_decoder_critic(inputs, init_state, sequence_length, params, attention=None, mask=None, use_critic=True):
    # push final state of encoder to decoder
    dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]), dtype=np.float32)
    dec_out = None
    outputs = []
    estimated_values = []
    cell_dec = get_cell("lstm_block", params["fw_cell_size"])
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        pm2_5_t = tf.expand_dims(tf.reshape(pm2_5, [params["batch_size"], params["grid_size"], params["grid_size"]]), 3)
        dec_in = tf.concat([input_t, pm2_5_t], axis=3)
        # need to do cnn here
        dec_in = get_cnn_rep(dec_in, type=2)
        dec_in = tf.flatten(dec_in))
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        # belong to generator
        pm2_5 = tf.layers.dense(dec_out, params["de_output_size"], name="decoder_output", activation=tf.nn.sigmoid)
        # belong to critic
        if use_critic:
            e_value = tf.layers.dense(dec_out, 1, name="critic_linear_output", activation=None)
            estimated_values.append(e_value)
        outputs.append(pm2_5)
    return outputs, estimated_values

# output: predictions - probability [0, 1], rewards [0, 1]
# this is for discriminator
def execute_decoder_dis(inputs, init_state, sequence_length, params, gamma, attention=None, is_fake=True):
    # push final state of encoder to decoder
    dec_state = init_state
    dec_out = None
    predictions = []
    cell_dec = get_cell("lstm_block", params["fw_cell_size"])
    rewards = []
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        # need to do cnn here
        dec_in = get_cnn_rep(input_t, 3, tf.nn.leaky_relu)
        dec_in = tf.flatten(dec_in)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        # belong to disciminator
        dec_out = tf.layers.dense(dec_out, 1, name="decoder_linear_value", activation=None)
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


def get_cnn_rep(cnn_inputs, type=1, activation=tf.nn.relu):
    inp_shape = cnn_inputs.get_shape()
    inp_length = len(inp_shape) 
    upscale_k = (5, 5)
    strides = (2,2)
    if inp_length == 5:
        length = inp_shape[0] * inp_shape[1]
    else: 
        length = inp_shape[0]
    if type == 0:
        if inp_length == 5:
            cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
        cnn_inputs = tf.expand_dims(cnn_inputs, 4)
        cnn_outputs = tf.layers.conv3d(
            inputs=cnn_inputs,
            strides=(2,2,1),
            filters=1,
            kernel_size=(inp_shape[4],3,3)
        )
        #output should have shape: bs * length, 12, 12
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif type == 1:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution
        """
        if inp_length == 5:
            cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
        # 25 x 25 x H => 8x8x32 => 16x16x16 => 16x16x1
        conv1 = get_cnn_unit(cnn_inputs, 32, (11,11), None, "VALID", "conv1")
        conv2 = get_cnn_transpose_unit(conv1, 16, upscale_k, None, "SAME", "transpose_conv1")
        conv3 = get_cnn_transpose_unit(conv2, 8, upscale_k, None, "SAME", "transpose_conv2")
        cnn_outputs = get_cnn_unit(conv3, 1, upscale_k, None, "SAME", "")
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif type == 2:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for Generator output
        """
        if inp_length == 5:
            cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
        # normalize input to [-1, 1] in generator
        cnn_inputs = tf.tanh(cnn_inputs)
        # input should be 4 * 4 * 32 => 8 x 8 x 32 => 16 x 16 x 16 => 32 x 32 x 8 => 25x25x1
        conv1 = get_cnn_transpose_unit(cnn_inputs, 32, upscale_k, activation, "SAME", "transpose_conv1", True, 0.5)
        conv2 = get_cnn_transpose_unit(conv1, 16, upscale_k, activation, "SAME", "transpose_conv2", True, 0.5)
        conv3 = get_cnn_transpose_unit(conv2, 8, upscale_k, activation, "SAME", "transpose_conv3", True, 0.5)
        cnn_outputs = get_cnn_unit(conv3, 1, (8, 8), activation, "SAME", "cnn_gen_output", True, 0.5)
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
     elif type == 3:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for discriminator
            use dropout and batch_normalization
        """
        if inp_length == 5:
            cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
        # normalize input to [-1, 1] in generator
        cnn_inputs = tf.tanh(cnn_inputs)
        # 25 x 25 x H => 8x8x32 => 4x4x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (11,11), activation, "VALID", "rep_conv1", True, 0.5)
        cnn_outputs = get_cnn_unit(conv1, 32, upscale_k, activation, "SAME", "rep_conv2", True, 0.5)
    else:
        """
            Use for representation steps of both encoder and decoder
            provide the cnn representation of input images from 25 x 25 => 4 * 4 * 64 dimension
        """
        if inp_length == 5:
            cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
        # 25 x 25 x H => 8x8x32 => 4x4x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (11,11), activation, "VALID", "rep_conv1")
        cnn_outputs = get_cnn_unit(conv1, 32, upscale_k, activation, "SAME", "rep_conv2")
    return cnn_outputs


def get_cnn_unit(cnn_inputs, filter, kernel, activation=tf.nn.relu, padding="VALID", name="", use_batch_norm=False, dropout=0.0, strides=(2,2)):
    cnn_outputs = tf.layers.conv2d(
        inputs=cnn_inputs,
        strides=strides,
        filters=1,
        kernel_size=kernel,
        padding=padding,
        activation=activation
    )
    if dropout != 0.0:
        cnn_outputs = tf.layers.dropout(cnn_outputs)
    if use_batch_norm:
        cnn_outputs = tf.layers.batch_normalization(cnn_outputs, name=name)
    return cnn_outputs


def get_cnn_transpose_unit(cnn_inputs, filter, kernel, activation=tf.nn.relu, padding="SAME", name="", use_batch_norm=False, dropout=0.0, strides=(2,2)):
    cnn_outputs = tf.layers.conv2d_transpose(
        inputs=cnn_inputs,
        strides=strides,
        filters=1,
        kernel_size=kernel,
        padding=padding,
        activation=activation,
        name=name
    )
    if dropout != 0.0:
        cnn_outputs = tf.layers.dropout(cnn_outputs)
    if use_batch_norm:
        cnn_outputs = tf.layers.batch_normalization(cnn_outputs, name=name + "_bn")
    return cnn_outputs