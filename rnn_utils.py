from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell, MultiRNNCell, LSTMBlockFusedCell, LSTMBlockCell, GRUBlockCell, BasicRNNCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU, CudnnRNNTanh
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
        cell = CudnnGRU(layers, size)
    elif cell_type == "lstm_block":
        cell = LSTMBlockCell(size)
    elif cell_type == "gru_block":
        cell = GRUBlockCell(size)
    elif cell_type == "rnn":
        cell = BasicRNNCell(size)
    elif cell_type == "cudnn_rnn":
        cell = CudnnRNNTanh(layers, size)
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
    if  "fw_layers" not in params:
        params["fw_layers"] = 1
    # note: state_size of MultiRNNCell must be equal to size input_size
    fw_cell = get_cell(params["fw_cell"], params["fw_cell_size"], params["fw_layers"])
    if "cudnn" in params["fw_cell"] or params["fw_cell"] == "lstm_block_fused":
        inputs = tf.transpose(inputs, [1, 0, 2])
        if "cudnn" in params["fw_cell"]:
            outputs, fn_state = fw_cell(inputs)
            if params["fw_cell"] == "cudnn_lstm":
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
    if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
        dec_state = tf.squeeze(init_state[0], [0])
    else:
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
                        name="decoder_output_relu",
                        activation=tf.nn.relu)
        pm2_5 = tf.layers.dense(dec_out, 
                        params["de_output_size"],
                        name="decoder_output_tanh",
                        activation=tf.nn.tanh)
        if dropout:
            pm2_5 = tf.nn.dropout(pm2_5, dropout)
        outputs.append(pm2_5)
    return outputs


# perform cnn on pm2_5 output
def execute_decoder_cnn(inputs, init_state, sequence_length, params, attention=None, cnn_rep=True, cnn_gen=False, mtype=4, use_batch_norm=True, dropout=0.5):
    # push final state of encoder to decoder
    if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
        dec_state = tf.squeeze(init_state[0], [0])
    else:
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
        if cnn_rep:
            dec_in = get_cnn_rep(dec_in, mtype=mtype)
        dec_in = tf.layers.flatten(dec_in)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        
        if cnn_gen:
            pm2_5_input = tf.layers.dense(dec_out, 256, name="decoder_output_cnn")
            pm2_5_input = tf.reshape(pm2_5_input, [params["batch_size"], 4, 4, 16])
            pm2_5_cnn = get_cnn_rep(pm2_5_input, 2, max_filters=16, use_batch_norm=use_batch_norm, dropout=dropout)
            pm2_5_cnn = tf.tanh(pm2_5_cnn)
            pm2_5 = tf.layers.flatten(pm2_5_cnn)
        else:
            pm2_5 = tf.layers.dense(dec_out, 
                        params["de_output_size"],
                        name="decoder_output",
                        activation=tf.nn.sigmoid)
        outputs.append(pm2_5)
    return outputs


# perform cnn on pm2_5 output
# estimated value of critic: 0 - inf
# outputs: pm2.5 images
# this is for generator
def execute_decoder_critic(inputs, init_state, sequence_length, params, attention=None, use_critic=True, cnn_gen=True, mtype=3, use_batch_norm=True, dropout=0.5):
    # push final state of encoder to decoder
    if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
        dec_state = tf.squeeze(init_state[0], [0])
    else:
        dec_state = init_state
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]), dtype=np.float32)
    dec_out = None
    outputs = []
    estimated_values = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        pm2_5_t = tf.expand_dims(tf.reshape(pm2_5, [params["batch_size"], params["grid_size"], params["grid_size"]]), 3)
        dec_in = tf.concat([input_t, pm2_5_t], axis=3)
        # need to do cnn here
        dec_in = get_cnn_rep(dec_in, mtype=mtype)
        dec_in = tf.layers.flatten(dec_in)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        # belong to generator
        if cnn_gen:
            pm2_5_input = tf.layers.dense(dec_out, 256, name="decoder_output_cnn")
            pm2_5_input = tf.reshape(pm2_5_input, [params["batch_size"], 4, 4, 16])
            pm2_5_cnn = get_cnn_rep(pm2_5_input, 2, max_filters=16, use_batch_norm=use_batch_norm, dropout=dropout)
            pm2_5 = tf.layers.flatten(pm2_5_cnn)
        else:
            pm2_5 = tf.layers.dense(dec_out, params["de_output_size"], name="decoder_output", activation=tf.nn.sigmoid)
        # belong to critic
        if use_critic:
            e_value = tf.layers.dense(dec_out, 1, name="critic_linear_output", activation=None)
            estimated_values.append(e_value)
        outputs.append(pm2_5)
    return outputs, estimated_values

# output: predictions - probability [0, 1], rewards [0, 1]
# this is for discriminator
def execute_decoder_dis(inputs, init_state, sequence_length, params, gamma, attention=None, is_fake=True, mtype=3, use_batch_norm=True, dropout=0.5):
    # push final state of encoder to decoder
    if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
        dec_state = tf.squeeze(init_state[0], [0])
    else:
        dec_state = init_state
    dec_out = None
    predictions = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    rewards = []
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        input_t = inputs[:, t]
        # need to do cnn here
        dec_in = get_cnn_rep(input_t, mtype, tf.nn.leaky_relu, use_batch_norm=use_batch_norm, dropout=dropout)
        dec_in = tf.layers.flatten(dec_in)
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


def get_cnn_rep(cnn_inputs, mtype=4, activation=tf.nn.relu, max_filters=8, use_batch_norm=True, dropout=0.5):
    inp_shape = cnn_inputs.get_shape()
    inp_length = len(inp_shape) 
    upscale_k = (5, 5)
    if inp_length == 5:
        length = inp_shape[0] * inp_shape[1]
    else: 
        length = inp_shape[0]
    
    if inp_length == 5:
        cnn_inputs = tf.reshape(cnn_inputs, [length, inp_shape[2], inp_shape[2], inp_shape[-1]])
    if mtype == 0:
        # cnn_inputs = tf.transpose(cnn_inputs, [0,2,3,1])
        # cnn_inputs = tf.expand_dims(cnn_inputs,4)
        # cnn_outputs = tf.layers.conv3d(
        #         inputs=cnn_inputs,                
        #         strides=(1,2,2),
        #         filters=1,
        #         kernel_size=(inp_shape[-1],3,3)
        # )
        cnn_outputs = get_cnn_unit(cnn_inputs, 32, upscale_k, name="basic_conv")
    elif mtype == 1:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution (for original GAN with mse)
        """
        # 25 x 25 x H => 8x8x32 => 16x16x16 => 32x32x8 => 16x16x1
        conv1 = get_cnn_unit(cnn_inputs, 32, (11,11), None, "VALID", "conv1")
        conv2 = get_cnn_transpose_unit(conv1, 16, upscale_k, None, "SAME", "transpose_conv1", use_batch_norm, dropout)
        conv3 = get_cnn_transpose_unit(conv2, 8, upscale_k, None, "SAME", "transpose_conv2", use_batch_norm, dropout)
        cnn_outputs = get_cnn_unit(conv3, 1, upscale_k, None, "SAME", "")
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif mtype == 2:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for Generator output
        """
        # normalize input to [-1, 1] in generator
        cnn_inputs = tf.tanh(cnn_inputs)
        # input should be 4 * 4 * 8 => 8 x 8 x 8 => 16 x 16 x 4 => 32 x 32 x 2 => 25x25x1
        conv1 = get_cnn_transpose_unit(cnn_inputs, max_filters, upscale_k, activation, "SAME", "transpose_conv1", use_batch_norm, dropout)
        conv2 = get_cnn_transpose_unit(conv1, max_filters / 2, upscale_k, activation, "SAME", "transpose_conv2", use_batch_norm, dropout)
        conv3 = get_cnn_transpose_unit(conv2, max_filters / 4, upscale_k, activation, "SAME", "transpose_conv3", use_batch_norm, dropout)
        cnn_outputs = get_cnn_unit(conv3, 1, (8, 8), activation, "VALID", "cnn_gen_output", use_batch_norm, dropout, strides=(1,1))
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    else:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for discriminator
            use dropout and batch_normalization
        """
        # normalize input to [-1, 1] in generator
        cnn_inputs = tf.tanh(cnn_inputs)
        # 25 x 25 x H => 8x8x8 => 4x4x8
        conv1 = get_cnn_unit(cnn_inputs, max_filters, (11,11), activation, "VALID", "rep_conv1", use_batch_norm, dropout)
        cnn_outputs = get_cnn_unit(conv1, max_filters, upscale_k, activation, "SAME", "rep_conv2", use_batch_norm, dropout)
    # else:
    #     """
    #         Use for representation steps of both encoder and decoder
    #         provide the cnn representation of input images from 25 x 25 => 4 * 4 * 64 dimension
    #     """
    #     cnn_inputs = tf.tanh(cnn_inputs)
    #     # 25 x 25 x H => 8x8x8 => 4x4x8
    #     conv1 = get_cnn_unit(cnn_inputs, max_filters, (11,11), activation, "VALID", "rep_conv1")
    #     cnn_outputs = get_cnn_unit(conv1, max_filters, upscale_k, activation, "SAME", "rep_conv2")
    return cnn_outputs


def get_cnn_unit(cnn_inputs, filters, kernel, activation=tf.nn.relu, padding="VALID", name="", use_batch_norm=False, dropout=0.0, strides=(2,2)):
    cnn_outputs = tf.layers.conv2d(
        inputs=cnn_inputs,
        strides=strides,
        filters=int(filters),
        kernel_size=kernel,
        padding=padding,
        activation=activation,
        name=name
    )
    if dropout != 0.0:
        cnn_outputs = tf.layers.dropout(cnn_outputs)
    if use_batch_norm:
        cnn_outputs = tf.layers.batch_normalization(cnn_outputs, name=name + "_bn", fused=True)
    return cnn_outputs


def get_cnn_transpose_unit(cnn_inputs, filters, kernel, activation=tf.nn.relu, padding="SAME", name="", use_batch_norm=False, dropout=0.0, strides=(2,2)):
    cnn_outputs = tf.layers.conv2d_transpose(
        inputs=cnn_inputs,
        strides=strides,
        filters=int(filters),
        kernel_size=kernel,
        padding=padding,
        activation=activation,
        name=name
    )
    if dropout != 0.0:
        cnn_outputs = tf.layers.dropout(cnn_outputs)
    if use_batch_norm:
        cnn_outputs = tf.layers.batch_normalization(cnn_outputs, name=name + "_bn", fused=True)
    return cnn_outputs
