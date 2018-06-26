from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LayerNormBasicLSTMCell

import properties as prp
import utils
import model_utils
import rnn_utils
import heatmap


class BaselineModel():

    
    def __init__(self, encoder_length=24, decoder_length=24, grid_size=60, rnn_hidden_units=128, 
                encode_vector_size=12, decode_vector_size=6, learning_rate=0.01, batch_size=64, loss="mae", df_ele=6):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.sequence_length = encoder_length + decoder_length
        self.is_training = True
        self.grid_size = grid_size
        self.rnn_hidden_units=128
        self.encode_vector_size = encode_vector_size
        self.decode_vector_size = decode_vector_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grid_square = grid_size * grid_size
        self.loss = loss
        self.dropout = 0.9
        self.df_ele = df_ele

    
    def set_training(self, training):
        self.is_training = training

    def set_data(self, datasets, train, valid):
        self.datasets = datasets
        self.train = train
        self.valid = valid

    def init_ops(self):
       self.add_placeholders()
       self.output = self.inference()
       self.loss_op = self.add_loss(self.output)
       self.train_op = model_utils.add_training_op(self.loss_op, self.learning_rate)
       self.merged = tf.summary.merge_all()
    
    # preserve memory for tensors
    def add_placeholders(self):
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))
        # self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        embedding = tf.Variable(self.datasets, name="Embedding")
        enc = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        dec_f = tf.nn.embedding_lookup(embedding, self.decoder_inputs)
        dec = dec_f[:,:,:,:,self.df_ele:]
        self.pred_placeholder = dec_f[:,:,:,:,0]
        initializer=tf.contrib.layers.xavier_initializer()
        ecs = self.grid_square * self.encode_vector_size
        dcs = self.grid_square * self.decode_vector_size
        grd_cnn = 13 * 13
        e_params = {
            "fw_cell_size" : self.rnn_hidden_units,
            "fw_cell": "basic",
            "de_output_size": self.grid_square,
            "batch_size" : self.batch_size,
            "type": 0
        }
        with tf.variable_scope("encoder", initializer=initializer):
            # add one cnn layer here
            cnn = self.get_cnn_rep(enc, self.encoder_length, self.encode_vector_size)
            enc_data = tf.unstack(tf.reshape(cnn, [self.batch_size, self.encoder_length, grd_cnn]), axis=1)
            # then push through lstm
            _, enc_output = rnn_utils.execute_sequence(enc_data, e_params)
        
        with tf.variable_scope("decoder", initializer=initializer, reuse=tf.AUTO_REUSE):
            # add one cnn layer before decoding using lstm
            cnn = self.get_cnn_rep(dec, self.decoder_length, self.decode_vector_size)
            dec_data = tf.reshape(cnn, [self.batch_size, self.decoder_length, grd_cnn])
            #finally push -> decoder
            outputs = rnn_utils.execute_decoder(dec_data, enc_output, self.decoder_length, e_params)
            outputs = tf.stack(outputs, axis=1)
        return outputs


    def get_cnn_rep(self, dec, length, vector_size):
        cnn_inputs = tf.transpose(dec, [0,1,4,2,3])
        cnn_inputs = tf.reshape(cnn_inputs, [-1])
        # cnn_inputs = tf.reshape(self.decoder_inputs, [-1])
        
        cnn_inputs = tf.reshape(cnn_inputs, [self.batch_size * length, vector_size, self.grid_size, self.grid_size])
        cnn_inputs = tf.expand_dims(cnn_inputs, 4)
        cnn = tf.layers.conv3d(
            inputs=cnn_inputs,
            strides=(1,2,2),
            filters=1,
            kernel_size=(vector_size,6,6),
            padding="valid"
        )
        #output should have shape: bs * length, 28, 28, 1
        cnn = tf.reshape(tf.squeeze(cnn), [-1])
        return cnn
    
    # add loss function
    # output will be loss scalar    
    def add_loss(self, output):
        matrix_size = [self.batch_size, self.decoder_length * self.grid_square]
        y = tf.reshape(tf.reshape(output, [-1]), matrix_size)
        pred = tf.reshape(self.pred_placeholder, matrix_size)
        if self.loss is 'mse':
            loss_op = tf.losses.mean_squared_error
        else: 
            loss_op = tf.losses.absolute_difference
        
        loss = loss_op(labels=pred, predictions=y)
        loss = tf.reduce_mean(loss)
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
             if not 'bias' in v.name.lower():
                 loss += 0.0001 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False, shuffle=True):
        dp = self.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        dt_length = len(data)
        # print("data_size: ", dt_length)
        total_steps = dt_length // self.batch_size
        total_loss = []
        accuracy = 0
        r = np.random.permutation(dt_length)
        ct = np.asarray(data, dtype=np.float32)
        ct = ct[r]
        preds = []
        for step in range(total_steps):
            index = range(step * self.batch_size,
                          (step + 1) * self.batch_size)
            # just the starting points of encoding batch_size,
            ct_t = ct[index]
            # switch batchsize, => batchsize * encoding_length
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
            dec_t = ct_t + self.decoder_length
            # convert 1-d data to map
            # pred_t = self.convert_preds_to_grid(pred_t)
            # ct_t = self.convert_context_to_grid(ct_t)
            # dec_t = self.convert_context_to_grid(dec_t)

            feed = {
                self.encoder_inputs: ct_t,
                # self.pred_placeholder: pred_t,
                self.decoder_inputs: dec_t
            }
            
            loss, pred, summary, _ = session.run(
                [self.loss_op, self.output, self.merged, train_op], feed_dict=feed)
            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            
            pred = [int(round(x)) if x > 0 else 0 for x in pred]
            preds += pred
            acc = utils.calculate_accuracy(pred, pred_labels, self.range, self.is_classify)
            accuracy += acc
            total_loss.append(loss)

        avg_acc = 0.
        if total_steps:
            # print(accuracy)
            avg_acc = accuracy * 1.0 / len(preds)
        return np.mean(total_loss), avg_acc, preds, pr.tolist()
        