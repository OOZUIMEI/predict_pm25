from __future__ import print_function
from __future__ import division

import sys
import time
    
import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell


class Model():

    
    def __init__(self, max_input_len=30, embed_size=12, hidden_size=128, relu_size=64, learning_rate = 0.001, batch_size=54, 
                lr_decayable=True, using_bidirection=False, fw_cell='basic', bw_cell='gru', is_classify=True, target=5, loss='softmax', 
                acc_range=10, use_tanh_prediction=True):
        self.max_input_len = max_input_len
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decayable = lr_decayable
        self.using_bidirection = using_bidirection
        self.fw_cell = fw_cell
        self.bw_cell = bw_cell
        self.hidden_size = hidden_size
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
        
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.max_input_len, self.embed_size))  
        # if self.is_classify:
        self.pred_placeholder = tf.placeholder(
            tf.int32, shape=(self.batch_size,))
        # else:
        #     self.pred_placeholder = tf.placeholder(
        #         tf.int32, shape=(self.batch_size,1))
        # place holder for start vs end position
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.iteration = tf.placeholder(tf.int32)

    def inference(self):
        """Performs inference on the DMN model"""
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation rnn')
            word_reps = self.get_input_representation()
            word_reps = tf.reduce_mean(word_reps, axis=0)
        
        with tf.variable_scope("hidden", initializer=tf.contrib.layers.xavier_initializer()):
            if self.is_classify:
                output = tf.layers.dense(word_reps,
                                    self.hidden_size,
                                    activation=tf.nn.tanh,
                                    name="h2")
                
                output = tf.layers.dense(word_reps,
                                        self.relu_size,
                                        activation=tf.nn.relu,
                                        name="relu")

                output = tf.nn.dropout(output, self.dropout_placeholder)
            else:
                if self.use_tanh_prediction:
                    output = tf.layers.dense(word_reps,
                                        self.hidden_size,
                                        activation=tf.nn.tanh,
                                        name="h2")
                else:
                    # simple model with only one hidden layer to predict
                    output = word_reps
            output = tf.layers.dense(output,
                                    self.target,
                                    name="fn")
        return output

    def get_input_representation(self):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        inputs = tf.unstack(self.input_placeholder, self.max_input_len, 1)
        # use encoding to get sentence representation plus position encoding
        # (like fb represent)
        if self.fw_cell == 'basic':
            fw_cell = BasicLSTMCell(self.embed_size)
        else:
            fw_cell = GRUCell(self.embed_size)
        if not self.using_bidirection:
            # outputs with [batch_size, max_time, cell_bw.output_size]
            outputs, _ = tf.nn.static_rnn(
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
            outputs, _  = tf.nn.static_bidirectional_rnn(
                fw_cell,
                back_cell,
                inputs, 
                dtype=np.float32
            )
            outputs = tf.concat(outputs, 2)
        return outputs

    def add_loss_op(self, output):
        """Calculate loss"""
        if self.loss == 'softmax':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.pred_placeholder)
        elif self.loss == 'mse':
            pred = tf.reshape(self.pred_placeholder, [self.batch_size, 1])
            loss = tf.losses.mean_squared_error(predictions=output, labels=pred)
        elif self.loss == 'mae': 
            pred = tf.reshape(self.pred_placeholder, [self.batch_size, 1])
            loss = tf.losses.absolute_difference(labels=pred, predictions=output)
        
        loss = tf.reduce_mean(loss)
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.l2 * tf.nn.l2_loss(v)

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

    def calculate_accuracy(self, pred, pred_labels):
        accuracy = 0
        if self.is_classify:
            accuracy = np.sum(pred == pred_labels)
        else:
            accuracy = np.sum([1 for x, y in zip(pred, pred_labels) if abs(x - y) <= self.range])

        return accuracy

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        dp = self.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        dt_length = len(data[0])
        total_steps = dt_length // self.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        r = np.random.permutation(dt_length)
        ct, pr = data
        ct, pr = np.asarray(ct, dtype=np.float32), np.asarray(pr, dtype=np.float32)
        ct, pr = ct[r], pr[r]
        preds = []
        for step in range(total_steps):
            index = range(step * self.batch_size,
                          (step + 1) * self.batch_size)
            feed = {self.input_placeholder: ct[index],
                    self.pred_placeholder: pr[index],
                    self.dropout_placeholder: dp,
                    self.iteration: num_epoch}
            
            pred_labels = pr[step * self.batch_size:(step + 1) * self.batch_size]
            
            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)
            if train_writer is not None:
                train_writer.add_summary(
                    summary, num_epoch * total_steps + step)
            
            preds += [x for x in pred]
            accuracy += self.calculate_accuracy(pred, pred_labels)

            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        avg_acc = 0.
        if total_steps:
            avg_acc = accuracy * 1.0 / dt_length
        return np.sum(total_loss), avg_acc, preds, pr.tolist()
