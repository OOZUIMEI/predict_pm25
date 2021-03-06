from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
import properties as pr
import model_utils
import utils


class NeuralNetwork(object):

    def __init__(self, encoder_length=24, encoder_vector_size=15, decoder_length=8, decoder_vector_size=9, attention_length=24, attention_vector_size=17, 
                learning_rate=0.01, dtype="grid", forecast_factor=0, **kwargs):
        # super(NeuralNetwork, self).__init__(**kwargs)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.attention_length = attention_length
        self.encoder_vector_size = encoder_vector_size
        self.decoder_vector_size = decoder_vector_size
        self.attention_vector_size = attention_vector_size
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.forecast_factor = forecast_factor
        self.initializer = tf.contrib.layers.xavier_initializer()
    
    def add_placeholders(self):
        self.embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, name="Data_Embedding", trainable=False)
        self.attention_embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="attention_embedding")
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(pr.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(pr.batch_size, self.decoder_length))
        self.attention_inputs = tf.placeholder(tf.int32, shape=(pr.batch_size, self.attention_length))
        self.dropout_placeholder = tf.Variable(0.5, False, name="dropout", dtype=tf.float32)

    def set_data(self, datasets, train, valid, attention, session):
        self.datasets = datasets
        self.train = train
        self.valid = valid
        assign_ops = tf.assign(self.embedding, self.datasets, False)
        session.run(assign_ops)
    
    def init_model(self):
        self.add_placeholders()
        self.output = self.inference()
        self.loss = self.add_loss(self.output)
        self.train_op = model_utils.add_training_op(self.loss, self.learning_rate)

    def lookup_input(self):
        print("predict %i" % self.forecast_factor)
        enc = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        enc.set_shape((pr.batch_size, self.encoder_length, 25, self.encoder_vector_size))
        dec_f = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
        dec_f.set_shape((pr.batch_size, self.decoder_length, 25, self.encoder_vector_size))
        # predict only one timestep
        self.pred_placeholder = dec_f[:,self.decoder_length - 1,:,self.forecast_factor]
        return enc
    
    def inference(self):
        enc = self.lookup_input()
        enc = tf.reshape(tf.transpose(enc, [0, 2, 1, 3]), shape=[pr.batch_size,25,self.encoder_length*self.encoder_vector_size])
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            enc_out = self.add_neural_nets(enc)
            enc_out = tf.layers.dropout(enc_out, self.dropout_placeholder)
        
        with tf.variable_scope("prediction", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            pred = tf.layers.dense(enc_out, units=1, activation=tf.nn.sigmoid, name="final_hidden_sigmoid")
            pred = tf.layers.dropout(pred, self.dropout_placeholder)
            pred = tf.squeeze(pred, axis=2)
        return pred
    
    def add_single_net(self, inputs, units=100, activation=tf.nn.relu, name="hidden_relu_basic"):
        out_hid1 = tf.layers.dense(inputs, units=units, activation=activation, name=name)
        return out_hid1
    
    def add_neural_nets(self, inputs, activation=tf.nn.tanh):
        out_hid1 = tf.layers.dense(inputs, units=256, activation=activation, name="hidden_256")
        out_hid2 = tf.layers.dense(out_hid1, units=128, activation=activation, name="hidden_128")
        out_hid3 = tf.layers.dense(out_hid2, units=64, activation=activation, name="hidden_64")
        out_hid4 = tf.layers.dense(out_hid3, units=32, activation=activation, name="hidden_32")
        return out_hid4

    def add_loss(self, pred, labels=None):
        if labels is None:
            labels = tf.layers.flatten(self.pred_placeholder)
        preds = tf.layers.flatten(pred)
        losses = tf.losses.mean_squared_error(labels=labels, predictions=preds)
        losses = tf.reduce_mean(losses)
        for x in tf.trainable_variables():
            if "bias" not in x.name.lower():
                losses += 0.0001 * tf.nn.l2_loss(x)
        return losses
    
     # operation of each epoch
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=True, train=False, shuffle=True, stride=4):
        if train_op is None:
            train_op = tf.no_op()
        dt_length = len(data)
        # print("data_size: ", dt_length)
        cons_b = pr.batch_size * stride
        total_steps = dt_length // cons_b
        total_loss = 0.0
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        preds = []
        for step in range(total_steps):
            index = range(step * cons_b, (step + 1) * cons_b, stride)
            # just the starting points of encoding batch_size,
            ct_ = ct[index]
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_])
            dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in ct_])
            # print(np.shape(ct_t))
            # print(np.shape(dec_t))
            feed = {
                self.encoder_inputs : ct_t,
                self.decoder_inputs: dec_t
            }
            l, pred, _= session.run([self.loss, self.output, train_op], feed_dict=feed)
            
            total_loss += l
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(step, total_steps, total_loss / (step + 1)))
                sys.stdout.flush()

            preds.append(pred)

        if verbose:
            sys.stdout.write("\r")
        
        total_loss = total_loss / total_steps
        if train_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag= "Total Loss", simple_value=total_loss)
            train_writer.add_summary(summary, num_epoch)
        return total_loss, preds
    
