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
                encode_vector_size=12, decode_vector_size=6, learning_rate=0.01, batch_size=64, loss="mae"):
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
        self.map = heatmap.build_map()

    
    def set_training(self, training):
        self.is_training = training

    def set_data(self, train, valid):
        self.train = train
        self.valid = valid

    def init_ops(self):
       self.add_placeholders()
       self.output = self.inference()
       self.loss_op = self.add_loss(self.output)
       self.train_op = model_utils.add_training_op(self.loss_op, self.learning_rate)
    
    # preserve memory for tensors
    def add_placeholders(self):
        self.encoder_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.encoder_length, self.encode_vector_size, self.grid_size, self.grid_size))
        self.decoder_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.decode_vector_size, self.grid_size, self.grid_size))
        self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.grid_size, self.grid_size))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        initializer=tf.contrib.layers.xavier_initializer()
        ecs = self.grid_square * self.encode_vector_size
        dcs = self.grid_square * self.decode_vector_size
        grd_cnn = 14 * 14
        e_params = {
            "fw_cell_size" : self.rnn_hidden_units,
            "fw_cell": "basic",
            "de_output_size": self.grid_square,
            "batch_size" : self.batch_size,
            "type": 0
        }

        with tf.variable_scope("encoder", initializer=initializer):
            # add one cnn layer here
            cnn_inputs = tf.reshape(self.encoder_inputs, [-1])
            cnn_inputs = tf.reshape(cnn_inputs, [self.batch_size * self.encoder_length, self.encode_vector_size, self.grid_size, self.grid_size])
            cnn_inputs = tf.expand_dims(cnn_inputs, 4)
            cnn = tf.layers.conv3d(
                inputs=cnn_inputs,
                strides=(1,2,2),
                filters=1,
                kernel_size=(self.encode_vector_size,6,6),
                padding="valid"
            )
            pooling = tf.layers.max_pooling3d(cnn, (1,2,2), (1,2,2))
            #output should have shape: bs * length, 28, 28, 1
            cnn = tf.reshape(tf.squeeze(pooling), [-1])
            enc_data = tf.unstack(tf.reshape(cnn, [self.batch_size, self.encoder_length, grd_cnn]), axis=1)
            # then push through lstm
            _, enc_output = rnn_utils.execute_sequence(enc_data, e_params)
      
        with tf.variable_scope("decoder", initializer=initializer, reuse=tf.AUTO_REUSE):
            # add one cnn layer before decoding using lstm
            cnn_inputs = tf.reshape(self.decoder_inputs, [-1])
            cnn_inputs = tf.reshape(cnn_inputs, [self.batch_size * self.decoder_length, self.decode_vector_size, self.grid_size, self.grid_size])
            cnn_inputs = tf.expand_dims(cnn_inputs, 4)
            cnn = tf.layers.conv3d(
                inputs=cnn_inputs,
                strides=(1,2,2),
                filters=1,
                kernel_size=(self.decode_vector_size,6,6),
                padding="valid"
            )
            pooling = tf.layers.max_pooling3d(cnn, (1,2,2), (1,2,2))
            #output should have shape: bs * length, 28, 28, 1
            cnn = tf.reshape(tf.squeeze(pooling), [-1])
            dec_data = tf.reshape(cnn, [self.batch_size, self.decoder_length, grd_cnn])
            #finally push -> decoder
            outputs = rnn_utils.execute_decoder(dec_data, enc_output, self.decoder_length, e_params)
            outputs = tf.stack(outputs, axis=1)
        return outputs


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

    # convert 25 district record to map for 3-d dimensions
    def convert_preds_to_grid(self, preds):
        res = []
        for b in preds:
            res_t = []
            for t in b:
                p = heatmap.fill_map(t, self.map, False) 
                res_t.append(p)
            res.append(res_t)
        return np.asarray(res, dtype=np.float)

    # 4-d dimensions
    def convert_context_to_grid(self, context):
        res = []
        for b in context:
            res_t = []
            for t in b:
                # transpose 25 * 12 => 12 * 25 (12 is no of elements)
                res_e = []
                t_ = t.transpose()
                for e in t_:
                    # p with shape 60 x 60
                    p = heatmap.fill_map(e, self.map, False)
                    res_e.append(p)
                res_t.append(res_e)
            res.append(res_t)
        # batch * length * elements * 60 * 60
        return np.asarray(res, dtype=np.float)

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False, shuffle=True):
        dp = self.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        dt_length = len(data[0])
        # print("data_size: ", dt_length)
        total_steps = dt_length // self.batch_size
        total_loss = []
        accuracy = 0
        r = np.random.permutation(dt_length)
        ct, pr, dec = data
        ct, pr, dec = np.asarray(ct, dtype=np.float32), np.asarray(pr, dtype=np.int32), np.asarray(dec, dtype=np.float32)
        if shuffle:
            # shuffle data
            ct, pr, dec = ct[r], pr[r], dec[r]
        preds = []
        for step in range(total_steps):
            index = range(step * self.batch_size,
                          (step + 1) * self.batch_size)
            pred_t = pr[index]
            ct_t = ct[index]
            dec_t = dec[index]

            # convert 1-d data to map
            pred_t = self.convert_preds_to_grid(pred_t)
            ct_t = self.convert_context_to_grid(ct_t)
            dec_t = self.convert_context_to_grid(dec_t)

            feed = {
                self.encoder_inputs: ct_t,
                self.pred_placeholder: pred_t,
                self.decoder_inputs: dec_t
            }
            
            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)
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
        