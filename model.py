from __future__ import print_function
from __future__ import division

import sys
import time
    
import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell
from attention_cell import AttentionCell

import properties as pr
import utils


class Model():

    
    def __init__(self, max_sent_len=24, max_input_len=30, embed_size=12, hidden_size=128, hidden2_size=64, relu_size=64, learning_rate = 0.001, batch_size=54, 
                lr_decayable=True, using_bidirection=False, fw_cell='basic', bw_cell='gru', is_classify=True, target=5, loss='softmax', 
                acc_range=10, use_tanh_prediction=True, input_rnn=True, sight=1, dvs=4, use_decoder=True, is_weighted=0):
        self.max_sent_len = max_sent_len
        self.max_input_len = max_input_len
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decayable = lr_decayable
        self.using_bidirection = using_bidirection
        self.fw_cell = fw_cell
        self.bw_cell = bw_cell
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.relu_size = relu_size
        self.l2 = 0.0001
        self.target = target
        self.dropout = 0.9
        self.lr_depr = 100
        self.decay_rate = 0.95
        self.range = acc_range
        # self.input_len = tf.fill([self.batch_size, 1], self.max_input_len)
        self.is_classify = is_classify
        self.use_tanh_prediction = use_tanh_prediction
        self.loss = loss
        self.input_rnn = input_rnn
        self.decode_length = sight
        self.decode_vector_size = dvs
        self.use_decoder = use_decoder
        self.is_weighted = is_weighted

    def set_data(self, train, valid):
        self.train = train
        self.valid = valid
        
    def init_ops(self):
        # init memory
        self.add_placeholders()
        # init model
        self.output = self.inference()
        # init prediction step
        self.pred = self.get_predictions(self.output)
        # init cost function
        self.calculate_loss = self.add_loss_op(self.output)
        # init gradient
        self.train_step = self.add_training_op(self.calculate_loss)

        self.merged = tf.summary.merge_all()

    def add_placeholders(self):
        """add data placeholder to graph """
        
        if self.max_input_len > 1:
            self.input_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.max_sent_len, self.max_input_len, self.embed_size))  
            self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size * self.max_sent_len,))
        else:
            self.input_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.max_sent_len, self.embed_size))    

        self.decode_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decode_length, self.decode_vector_size))  
        self.pred_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,))
        if self.is_weighted:
            self.weight_labels = tf.placeholder(tf.float32, shape(self.batch_size,))
        else:
            self.weight_labels = None
        # else:
        #     self.pred_placeholder = tf.placeholder(
        #         tf.int32, shape=(self.batch_size,1))
        # place holder for start vs end position
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.iteration = tf.placeholder(tf.int32)

    def inference(self):
        """Performs inference on the DMN model"""
        with tf.variable_scope("word_shift", initializer=tf.contrib.layers.xavier_initializer()):
            if self.max_input_len > 1:
                word_reps = tf.reduce_mean(self.input_placeholder, axis=2)
            else:
                word_reps = self.input_placeholder

        with tf.variable_scope("sentence", initializer=tf.contrib.layers.xavier_initializer()):
            # decode with attention 
            sent_reps, fn_state = self.get_input_representation(word_reps, self.fw_cell)
            sent_reps = tf.reduce_mean(sent_reps, axis=0)
        
        if self.use_decoder:
            # add attention layer
            with tf.variable_scope("attention", initializer=tf.contrib.layers.xavier_initializer()):
                attention = tf.layers.dense(sent_reps, 
                                            self.hidden_size, 
                                            activation=tf.nn.tanh,
                                            name="attention")
            
            with tf.variable_scope("decoder", initializer=tf.contrib.layers.xavier_initializer()):
                attention = tf.reshape(tf.tile(attention, [1, self.decode_length]), [self.batch_size, self.decode_length, self.hidden_size])
                decode_input = tf.concat([attention, self.decode_placeholder], axis=2)
                _, fn_state = self.get_input_representation(decode_input, "basic", self.decode_length)
                # _, fn_state = self.get_input_representation(self.decode_placeholder, "att", self.decode_length, attention=attention)
                input_hidden = fn_state[1]
        else:
            input_hidden = sent_reps
        with tf.variable_scope("hidden", initializer=tf.contrib.layers.xavier_initializer()):
            if self.is_classify:
                output = tf.layers.dense(input_hidden,
                                    self.hidden_size,
                                    activation=tf.nn.tanh,
                                    name="h2")
                
                output = tf.layers.dense(output,
                                        self.relu_size,
                                        activation=tf.nn.relu,
                                        name="relu")

                output = tf.nn.dropout(output, self.dropout_placeholder)
            else:
                if self.use_tanh_prediction:
                    output = tf.layers.dense(input_hidden,
                                        self.hidden2_size,
                                        activation=tf.nn.tanh,
                                        name="h2")
                else:
                    # simple model with only one hidden layer to predict
                    output = sent_reps
            output = tf.layers.dense(output,
                                    self.target,
                                    name="fn")
        return output

    # rnn through each 30', 1h 
    def get_input_representation(self, inputs, fw_cell="basic", length=None, attention=None):
        if fw_cell == 'basic':
            fw_cell = BasicLSTMCell(self.embed_size)
        elif fw_cell == 'att':
            fw_cell = AttentionCell(self.embed_size, max_val=pr.pm25_max, min_val=0.0, attention=attention)
        else:
            fw_cell = GRUCell(self.embed_size)
        if not self.using_bidirection:
            if not length:
                length = self.max_sent_len
            inputs = tf.unstack(inputs, length, 1)
            # outputs is array of max_time of [batch_size, cell_bw.output_size]
            outputs, fn_state = tf.nn.static_rnn(
                fw_cell,
                inputs,
                dtype=np.float32
            )
        else:
            if self.bw_cell == 'basic':
                back_cell = BasicLSTMCell(self.embed_size)
            else:
                back_cell = GRU
                Cell(self.embed_size)
            outputs, fn_state  = tf.nn.static_bidirectional_rnn(
                fw_cell,
                back_cell,
                inputs, 
                dtype=np.float32
            )
            outputs = tf.concat(outputs, 2)
        return outputs, fn_state

    def add_loss_op(self, output):
        """Calculate loss"""
        if self.loss == 'softmax':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.pred_placeholder, weights=self.weight_labels)
        elif self.loss == 'mse':
            pred = tf.reshape(self.pred_placeholder, [self.batch_size, 1])
            loss = tf.losses.mean_squared_error(predictions=output, labels=pred, weights=self.weight_labels)
        elif self.loss == 'mae': 
            pred = tf.reshape(self.pred_placeholder, [self.batch_size, 1])
            loss = tf.losses.absolute_difference(labels=pred, predictions=output, weights=self.weight_labels)
        
        loss = tf.reduce_mean(loss)
        # add l2 regularization for all variables except biases
        # for v in tf.trainable_variables():
        #     if not 'bias' in v.name.lower():
        #         loss += self.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss

    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        if self.lr_decayable:
            lr = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.iteration, decay_steps=self.lr_depr, decay_rate=self.decay_rate)
        else:
            lr = self.learning_rate
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)
        return train_op
   
    def get_predictions(self, output):
        if self.is_classify:
            pred = tf.nn.softmax(output)
            pred = tf.argmax(pred, 1)
        else:
            pred = output
        return pred

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
        ct, ct_l, pr, dec = data
        ct, pr, dec = np.asarray(ct, dtype=np.float32), np.asarray(pr, dtype=np.int32), np.asarray(dec, dtype=np.float32)
        if shuffle:
            # shuffle data
            ct, pr, dec = ct[r], pr[r], dec[r]
            if ct_l:
                ct_l = ct_l[r]
        elif ct_l:
            ct_l = np.asarray(ct_l, dtype=np.int32)
        preds = []
        pred_classes = None
        for step in range(total_steps):
            index = range(step * self.batch_size,
                          (step + 1) * self.batch_size)
            pred_labels = pr[index]
            feed = {self.input_placeholder: ct[index],
                    self.pred_placeholder: pred_labels,
                    self.dropout_placeholder: dp,
                    self.decode_placeholder: dec[index],
                    self.iteration: num_epoch}
            if self.is_weighted:
                pred_classes = np.array([utils.get_pm25_class(pm) for pm in pred_labels])
                pred_classes = pred_classes[pred_labels]
                feed[self.weight_labels] = pred_classes
            # if self.input_rnn:
            #     l = np.reshape(ct_l[index], [self.batch_size * self.max_sent_len])
            #     feed[self.input_len_placeholder] = l
            
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

        #     if verbose and step % verbose == 0:
        #         sys.stdout.write('\r{} / {} : loss = {}'.format(
        #             step, total_steps, np.mean(total_loss)))
        #         sys.stdout.flush()

        # if verbose:
        #     sys.stdout.write('\r')
        avg_acc = 0.
        if total_steps:
            # print(accuracy)
            avg_acc = accuracy * 1.0 / len(preds)
        return np.mean(total_loss), avg_acc, preds, pr.tolist()
