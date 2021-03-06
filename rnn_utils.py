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


"""
https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
https://towardsdatascience.com/transpose-convolution-77818e55a123
https://distill.pub/2016/deconv-checkerboard/
"""
def get_softmax_attention(inputs):
    logits = tf.layers.dense(inputs, units=1, activation=None, name="attention_logits")
    attention_logits = tf.squeeze(logits)
    attention = tf.nn.softmax(attention_logits)
    outputs = tf.transpose(inputs, [2, 0, 1])
    outputs = tf.multiply(outputs, attention)
    outputs = tf.transpose(outputs, [1, 2, 0])
    outputs = tf.reduce_sum(outputs, axis=1)
    return outputs


# 'unidirectional' or 'bidirectional'
def get_cell(cell_type, size, layers=1, direction='unidirectional'):
    if cell_type == "layer_norm_basic":
        cell = LayerNormBasicLSTMCell(size)
    elif cell_type == "lstm_block_fused":
        cell = tf.contrib.rnn.LSTMBlockFusedCell(size)
    elif cell_type == "cudnn_lstm":
        cell = CudnnLSTM(layers, size, direction=direction)
    elif cell_type == "cudnn_gru":
        cell = CudnnGRU(layers, size, direction=direction)
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


# elmo softmax
def execute_sequence(inputs, params):
    outputs, fn_state = None, None
    if params["rnn_layer"] > 1:
        outputs = inputs
        all_outputs = []
        for _ in xrange(params["rnn_layer"]):
            outputs, fn_state = execute_sequence_engine(inputs, params)
            all_outputs.append(outputs)
        all_outputs = tf.concat(all_outputs, axis=2)
        shape = all_outputs.get_shape()
        dim = int(int(shape[2]) / params["rnn_layer"])
        all_outputs = tf.reshape(all_outputs, (shape[0] * shape[1], params["rnn_layer"], dim), name="multilayer_reshape_softmax")
        all_outputs = get_softmax_attention(all_outputs)
        outputs = tf.reshape(all_outputs, (shape[0], shape[1], dim), name="multilayer_reshape")
    else:
        outputs, fn_state = execute_sequence_engine(inputs, params)
        # residual network, like elmo, connect bidirectional vectors & inputs
    if "elmo" in params and  params["elmo"]:
        outputs = tf.concat([outputs, inputs], axis=2, name="elmo_connection")
    return outputs, fn_state


# rnn through each 30', 1h 
def execute_sequence_engine(inputs, params):
    if prp.device and "gpu" not in prp.device:
        if "cudnn" in params["fw_cell"]:
            params["fw_cell"] = "lstm_block_fused"
        else:
            params["fw_cell"] = "basic"
    # 1 is bidireciton
    if  "rnn_layer" not in params:
        params["rnn_layer"] = 1
    # note: state_size of MultiRNNCell must be equal to size input_size
    fw_cell = get_cell(params["fw_cell"], params["fw_cell_size"], params["rnn_layer"], params["direction"])
    if "cudnn" in params["fw_cell"] or params["fw_cell"] == "lstm_block_fused":
        inputs = tf.transpose(inputs, [1, 0, 2])
        if "cudnn" in params["fw_cell"]:
            outputs, fn_state = fw_cell(inputs)
            if params["fw_cell"] == "cudnn_lstm":
                c_shape = fn_state[0].get_shape()
                if c_shape[0] == 1:
                    c = tf.squeeze(fn_state[1], [0])
                    h = tf.squeeze(fn_state[0], [0])
                else:
                    c = tf.squeeze(tf.gather(fn_state[1], [-1]), [0])
                    h = tf.squeeze(tf.gather(fn_state[0], [-1]), [0])
                fn_state = (c, h)
            # if params["direction"] == "bidirectional":
            #     shape = outputs.get_shape()
            #     outputs = tf.reshape(outputs, [shape[0]* shape[1], 2, params["fw_cell_size"]])
            #     with tf.variable_scope("bidirectional_softmax", initializer= tf.contrib.layers.xavier_initializer()):
            #         # get softmax of bidirectional vectors
            #         outputs = get_softmax_attention(outputs)
            #         # reshape to original shape
            #         outputs = tf.reshape(outputs, [shape[0], shape[1], params["fw_cell_size"]])
            
        else:
            outputs, fn_state = fw_cell(inputs, dtype=tf.float32)
        # => bs x length x hsize
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
    #if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
    #    print(init_state)
    #    dec_state = tf.squeeze(init_state[0], [0])
    #else:
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
                        name="decoder_output_sigmoid",
                        activation=tf.nn.sigmoid)
        if dropout:
            pm2_5 = tf.nn.dropout(pm2_5, dropout)
        outputs.append(pm2_5)
    return outputs


# perform cnn on pm2_5 output
def execute_decoder_cnn(inputs, init_state, sequence_length, params, attention=None, cnn_rep=True, cnn_gen=False, mtype=4, use_batch_norm=True, dropout=0.5, dctype=5, offset=0):
    # push final state of encoder to decoder
    if not init_state is None:
        if params["fw_cell"] == "gru_block" or params["fw_cell"] == "rnn":
            dec_state = tf.squeeze(init_state[0], [0])
        else:
            dec_state = init_state
    else:
        # generate outputs without the need of encoder
        zeros = np.zeros((params["batch_size"], params["fw_cell_size"]), dtype=np.float32)
        dec_state = (zeros, zeros)
    pm2_5 = np.zeros((params["batch_size"], params["de_output_size"]), dtype=np.float32)
    dec_out = None
    outputs = []
    cell_dec = get_cell(params["fw_cell"], params["fw_cell_size"])
    for t in xrange(sequence_length):
        # shape of input_t bs x grid_size x grid_size x hidden_size
        pm2_5_t = tf.expand_dims(tf.reshape(pm2_5, [params["batch_size"], params["grid_size"], params["grid_size"]]), 3)
        if inputs is not None:
            input_t = inputs[:, t]
            dec_in = tf.concat([input_t, pm2_5_t], axis=3)  
        else:
            dec_in = pm2_5_t
        # need to do cnn here
        if cnn_rep:
            dec_in = get_cnn_rep(dec_in, mtype=mtype, use_batch_norm=use_batch_norm, dropout=dropout)
        dec_in = tf.layers.flatten(dec_in)
        dec_out, dec_state = cell_dec(dec_in, dec_state)
        if attention is not None: 
            dec_out = tf.concat([dec_out, attention], axis=1)
        if not offset or t >= offset:
            if cnn_gen:
                pm2_5_input = tf.layers.dense(dec_out, 256, name="decoder_output_cnn")
                if mtype == 3 or mtype == 6 or mtype == 8 or mtype == 9:
                    pm2_5_input = tf.reshape(pm2_5_input, [params["batch_size"], 2, 2, 64])
                else:
                    pm2_5_input = tf.reshape(pm2_5_input, [params["batch_size"], 4, 4, 16])
                pm2_5_cnn = get_cnn_rep(pm2_5_input, dctype, max_filters=64, use_batch_norm=use_batch_norm, dropout=dropout)
                pm2_5 = tf.layers.flatten(pm2_5_cnn)
            else:
                # pm2_5 = tf.layers.dense(dec_out, 
                #             params["de_output_size"],
                #             name="decoder_output",
                #             activation=tf.nn.sigmoid)
                pm2_5 = tf.layers.dense(dec_out, 
                            params["de_output_size"],
                            name="decoder_output",
                            activation=tf.nn.relu)
            outputs.append(pm2_5)
    return outputs


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


def get_cnn_rep(cnn_inputs, mtype=4, activation=tf.nn.relu, max_filters=8, use_batch_norm=True, dropout=0.5, normalize=True):
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
        conv1 = get_cnn_transpose_unit(cnn_inputs, max_filters, upscale_k, activation, "SAME", "transpose_conv1", use_batch_norm, dropout)
        conv2 = get_cnn_transpose_unit(conv1, max_filters / 2, upscale_k, activation, "VALID", "transpose_conv2", use_batch_norm, dropout)
        cnn_outputs = get_cnn_transpose_unit(conv2, max_filters / 4, upscale_k, activation, "VALID", "transpose_conv3", use_batch_norm, dropout)
        cnn_outputs = get_cnn_unit(conv3, 1, (8, 8), activation, "VALID", "cnn_gen_output", use_batch_norm, dropout, strides=(1,1))
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif mtype == 3:
        """
            use for generator only
        """
        # 25 x 25 x H => 11x11x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (5,5), activation, "VALID", "rep_conv1", use_batch_norm, dropout)
        # 11x11x32 => 11 x 11 x 32
        msf1 = get_multiscale_conv(conv1, 8, activation=activation, prefix="msf1")
        # 11x11x32 => 4x4x32
        conv2 = get_cnn_unit(msf1, 32, (5,5), activation, "VALID", "rep_conv2", use_batch_norm, dropout)
        # 4x4x32 => 4x4x64
        msf2 = get_multiscale_conv(conv2, 16, activation=activation, prefix="msf2")
        # 4x4x32 => 2x2x16
        cnn_outputs = get_cnn_unit(msf2, 16, (3,3), activation, "SAME", "rep_conv3", use_batch_norm, dropout)
        # msf3 = get_multiscale_conv(conv3, 8, [1,3], activation=activation, prefix="msf3_")
        # cnn_outputs = get_cnn_unit(msf3, 16, (3,3), activation, "SAME", "rep_output", use_batch_norm, dropout)
    elif mtype == 4:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for discriminator
            use dropout and batch_normalization
        """
        # 25 x 25 x H => 8x8x8 => 4x4x8
        conv1 = get_cnn_unit(cnn_inputs, max_filters, (11,11), activation, "VALID", "rep_conv1", use_batch_norm, dropout)
        cnn_outputs = get_cnn_unit(conv1, max_filters, upscale_k, activation, "SAME", "rep_conv2", use_batch_norm, dropout)
    elif mtype == 5:
        """
            use structure of DCGAN with the mixture of both tranposed convolution and convolution for Generator output
        """
        # input should be 2 * 2 * 64  => 4 * 4 * 32 => 11 * 11 * 16 => 25 x 25 x 1
        conv1 = get_cnn_transpose_unit(cnn_inputs, max_filters, upscale_k, activation, "SAME", "transpose_conv1", use_batch_norm, dropout)
        conv2 = get_cnn_transpose_unit(conv1, max_filters / 2, upscale_k, activation, "VALID", "transpose_conv2", use_batch_norm, dropout)
        cnn_outputs = get_cnn_transpose_unit(conv2, 1, upscale_k, activation, "VALID", "transpose_conv3", use_batch_norm, dropout)
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif mtype == 6:
        """
            remove all msf layers (same as == 3)
        """
        # 25 x 25 x H => 11x11x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (5,5), activation, "VALID", "rep_conv1", use_batch_norm, dropout)
        # 11x11x32 => 4x4x32
        conv2 = get_cnn_unit(conv1, 32, (5,5), activation, "VALID", "rep_conv2", use_batch_norm, dropout)
        # 4x4x32 => 2x2x16
        cnn_outputs = get_cnn_unit(conv2, 16, (3,3), activation, "SAME", "rep_conv3", use_batch_norm, dropout)
    elif mtype == 7:
        """
            same as 5 but for us data (output 32 x 32) 
            input should be 2 * 2 * 64  => 4 * 4 * 32 => 8 * 8 * 16 => 16 x 16 x 8 => 32 x 32 x 1
        """
        conv1 = get_cnn_transpose_unit(cnn_inputs, max_filters, upscale_k, activation, "SAME", "transpose_conv1", use_batch_norm, dropout)
        conv2 = get_cnn_transpose_unit(conv1, max_filters / 2, upscale_k, activation, "SAME", "transpose_conv2", use_batch_norm, dropout)
        conv3 = get_cnn_transpose_unit(conv2, max_filters / 4, upscale_k, activation, "SAME", "transpose_conv3", use_batch_norm, dropout)
        cnn_outputs = get_cnn_transpose_unit(conv3, 1, upscale_k, activation, "SAME", "transpose_conv4", use_batch_norm, dropout)
        cnn_outputs = tf.squeeze(cnn_outputs, [-1])
    elif mtype == 8:
        # 32 x 32 x H => 16x16x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (5,5), activation, "SAME", "rep_conv1", use_batch_norm, dropout)
        # 16x16x32 => 16 x 16 x 32
        msf1 = get_multiscale_conv(conv1, 8, activation=activation, prefix="msf1")
        # 16x16x32 => 8x8x32
        conv2 = get_cnn_unit(msf1, 32, (5,5), activation, "SAME", "rep_conv2", use_batch_norm, dropout)
        # 8x8x32 => 8x8x64
        msf2 = get_multiscale_conv(conv2, 16, activation=activation, prefix="msf2")
        # 8x8x64 => 4x4x16
        cnn_outputs = get_cnn_unit(msf2, 16, (3,3), activation, "SAME", "rep_conv3", use_batch_norm, dropout)
    elif mtype == 9:
        # same as 8 but remove msf
        # 32 x 32 x H => 16x16x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (5,5), activation, "SAME", "rep_conv1", use_batch_norm, dropout)
        # 16x16x32 => 8x8x32
        conv2 = get_cnn_unit(conv1, 32, (5,5), activation, "SAME", "rep_conv2", use_batch_norm, dropout)
        # 8x8x64 => 4x4x16
        cnn_outputs = get_cnn_unit(conv2, 16, (3,3), activation, "SAME", "rep_conv3", use_batch_norm, dropout)
    else:
        # 25 x 25 x H => 11x11x32
        conv1 = get_cnn_unit(cnn_inputs, 32, (5,5), activation, "VALID", "rep_conv1", use_batch_norm, dropout)
        # 11x11x32 => 4x4x8 (32)
        cnn_outputs = get_cnn_unit(conv1, 8, (5,5), activation, "VALID", "rep_conv2", use_batch_norm, dropout)
    return cnn_outputs


# get multiscale convolution output
def get_multiscale_conv(inputs, filters, kernel=[7,5,3,1], activation=tf.nn.relu, is_trans=False, prefix="msf", strides=(1,1), use_softmax=False):
    convs = []
    conv_shape = None
    for k in kernel:
        if not is_trans:
            conv1 = get_cnn_unit(inputs, filters, (k, k), activation, "SAME", "%s_%ix%i" % (prefix, k,k), strides=strides)
        else:
            conv1 = get_cnn_transpose_unit(inputs, filters, (k, k), activation, "SAME", "%s_%ix%i" % (prefix, k,k), strides=strides)
        if not conv_shape:
            conv_shape = conv1.get_shape()
        if use_softmax:
            conv1 = tf.layers.flatten(conv1)
        convs.append(conv1)
    if use_softmax:
        # get shape: bs * l x w x h x k
        temp_shape = convs[0].get_shape()
        convs = tf.reshape(tf.concat(convs, axis=1), (temp_shape[0], len(convs), temp_shape[1]))
        # conv = tf.transpose(convs, [0, 2, 1])
        with tf.variable_scope("attention_%s" %prefix, initializer=tf.contrib.layers.xavier_initializer()):
            conv_ = get_softmax_attention(convs)
            conv_ = tf.reshape(conv_, conv_shape)
    else:
        conv_ = tf.concat(convs, axis=-1)
    return conv_


# get multiscale convolution output
def get_multiscale_conv3d(inputs, filters, kernel=[7,5,3,1], activation=tf.nn.relu, is_trans=False, prefix="msf",strides=(1,1,1)):
    convs = []
    for k in kernel:
        if not is_trans:
            conv1 = get_cnn3d_unit(inputs, filters, (k, k, k), activation, "SAME", "%s_%ix%i" % (prefix, k,k), strides=strides)
        else:
            conv1 = get_cnn_transpose3d_unit(inputs, filters, (k, k, k), activation, "SAME", "%s_%ix%i" % (prefix, k,k), strides=strides)
        convs.append(conv1)
    conv_ = tf.concat(convs, axis=-1)
    return conv_


# get convolution output
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


# get convolution output of pooling 2d
def get_maxpool(cnn_inputs, pool_size, padding="VALID", name="", strides=(2,2)):
    cnn_outputs = tf.layers.max_pooling2d(
        inputs=cnn_inputs,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        name=name
    )
    return cnn_outputs


# get convolution output
def get_cnn3d_unit(cnn_inputs, filters, kernel, activation=tf.nn.relu, padding="VALID", name="", use_batch_norm=False, dropout=0.0, strides=(2,2,2)):
    cnn_outputs = tf.layers.conv3d(
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


def get_cnn_transpose3d_unit(cnn_inputs, filters, kernel, activation=tf.nn.relu, padding="SAME", name="", use_batch_norm=False, dropout=0.0, strides=(2,2,2)):
    cnn_outputs = tf.layers.conv3d_transpose(
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
