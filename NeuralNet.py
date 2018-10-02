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

    def __init__(self, encoder_length=24, encoder_vector_size=15, decoder_length=24, decoder_vector_size=9, attention_length=24, attention_vector_size=17, learning_rate=0.01, dtype="grid"):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.attention_length = attention_length
        self.encoder_vector_size = encoder_vector_size
        self.decoder_vector_size = decoder_vector_size
        self.attention_vector_size = attention_vector_size
        self.learning_rate = learning_rate
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.dtype = dtype
    
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
        self.attention_vectors = attention
        assign_ops = tf.assign(self.embedding, self.datasets, False)
        session.run(assign_ops)
        att_ops = tf.assign(self.attention_embedding, self.attention_vectors, False)
        session.run(att_ops)
    
    def init_model(self):
        self.add_placeholders()
        self.output = self.inference()
        self.loss = self.add_loss(self.output)
        self.train_op = model_utils.add_training_op(self.loss, self.learning_rate)

    def lookup_input(self):
        enc = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        dec_f = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
        if self.dtype == "grid":
            enc.set_shape((pr.batch_size, self.encoder_length, pr.map_size, pr.map_size, self.encoder_vector_size))
            dec_f.set_shape((pr.batch_size, self.encoder_length, pr.map_size, pr.map_size, self.encoder_vector_size))
            # embedding = tf.Variable(self.datasets, name="embedding")
            dec = dec_f[:,:,:,:,6:]
            dec.set_shape((pr.batch_size, self.encoder_length, pr.map_size, pr.map_size, self.decoder_vector_size))
            self.pred_placeholder = dec_f[:,:,:,:,0]
        else:
            enc.set_shape((pr.batch_size, self.encoder_length, 25, self.encoder_vector_size))
            dec_f.set_shape((pr.batch_size, self.encoder_length, 25, self.encoder_vector_size))
            dec = dec_f[:,:,:,6:]
            dec.set_shape((pr.batch_size, self.encoder_length, 25, self.decoder_vector_size))
            self.pred_placeholder = dec_f[:,:,:,0]
        
        att = tf.nn.embedding_lookup(self.attention_embedding, self.decoder_inputs)
        att.set_shape((pr.batch_size, self.attention_length, self.attention_vector_size))
        return enc, dec, att
    
    def inference(self):
        enc, dec, att = self.lookup_input()
        att = tf.reshape(att, shape=(pr.batch_size, self.attention_length * self.attention_vector_size))

        with tf.variable_scope("attention", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            att_out = self.add_neural_nets(att)
        
        if self.dtype != "grid":
            enc = tf.reshape(tf.transpose(enc, [0, 2, 1, 3]), shape=(pr.batch_size, 25, self.encoder_length * self.encoder_vector_size))
            with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                enc_out = self.add_neural_nets(enc)
                enc_shape = enc_out.get_shape()
                enc_out = tf.reshape(tf.tile(enc_out, [1, self.decoder_length, 1]), shape=(enc_shape[0], self.decoder_length, enc_shape[1], enc_shape[-1]))
            
            att_shape = att_out.get_shape()
            att_out = tf.reshape(tf.tile(att_out, [1, self.decoder_length * 25]), shape=(att_shape[0], self.decoder_length, 25, att_shape[-1]))

            with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                dec_out = self.add_neural_nets(dec)
                # dec_out has shape batch_size x 24 x 25 x 96
            dec_out = tf.concat([dec_out, enc_out, att_out], axis=3)
            with tf.variable_scope("prediction", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                pred_hidden = self.add_neural_nets(dec_out)
                # pred_relu = tf.layers.dense(pred_hidden, units=1, activation=tf.nn.relu, name="relu_hidden")
                # # output has shape batch_size x 24 x 25
                # pred = tf.layers.dense(pred_relu, units=1, activation=tf.nn.tanh, name="final_hidden")
                pred = tf.layers.dense(pred_hidden, units=1, activation=tf.nn.sigmoid, name="final_hidden_sigmoid")
                pred = tf.layers.dropout(pred, self.dropout_placeholder)
                pred = tf.squeeze(pred, axis=3)
        else:
            enc = tf.layers.flatten(enc)
            with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                enc_out = self.add_neural_nets(enc)
                enc_shape = enc_out.get_shape()
                enc_out = tf.reshape(tf.tile(enc_out, [1, self.decoder_length]), shape=(pr.batch_size, self.decoder_length, enc_shape[-1]))
            att_shape = att_out.get_shape()
            att_out = tf.reshape(tf.tile(att_out, [1, self.decoder_length]), shape=(pr.batch_size, self.decoder_length, enc_shape[-1]))
            with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                dec_in = tf.reshape(dec, shape=(pr.batch_size, self.decoder_length, 625 * self.decoder_vector_size))
                dec_out = self.add_neural_nets(dec_in)
                # dec_out has shape batch_size x 24 x 25 x 96
            dec_out = tf.concat([dec_out, enc_out, att_out], axis=2)

            with tf.variable_scope("prediction", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                pred = tf.layers.dense(dec_out, units=625, activation=tf.nn.sigmoid, name="final_hidden_sigmoid")
                pred = tf.layers.dropout(pred, self.dropout_placeholder)
        return pred
        
    def add_neural_nets(self, inputs):
        out_hid1 = tf.layers.dense(inputs, units=256, activation=tf.nn.tanh, name="hidden_256")
        out_hid2 = tf.layers.dense(out_hid1, units=128, activation=tf.nn.tanh, name="hidden_128")
        out_hid3 = tf.layers.dense(out_hid2, units=64, activation=tf.nn.tanh, name="hidden_64")
        out_hid4 = tf.layers.dense(out_hid3, units=32, activation=tf.nn.tanh, name="hidden_32")
        out_hid4 = tf.layers.dropout(out_hid4, self.dropout_placeholder)
        return out_hid4

    def add_loss(self, pred):
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
            ct_t = ct[index]
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
            dec_t = ct_t + self.decoder_length

            feed = {
                self.encoder_inputs : ct_t,
                self.decoder_inputs: dec_t,
                self.attention_inputs: ct_t
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
    
