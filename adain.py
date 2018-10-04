import tensorflow as tf 
import numpy as np
from NeuralNet import NeuralNetwork
import properties as pr
import rnn_utils


# reference: a neural attention model for urban air quality inference: learning the weights of monitoring stations

class  Adain(NeuralNetwork):
    
    def __init__(self, rnn_hidden_units=300, **kwargs):
        super(Adain, self).__init__(**kwargs)
        self.rnn_hidden_units = rnn_hidden_units
        self.params = {
            "fw_cell": "cudnn_lstm",
            "fw_cell_size": self.rnn_hidden_units,
            "fw_layers": 2
        }
        self.dtype = "dis"

    def inference(self):
        
        enc, _, china_att = self.lookup_input()
        self.lstm_cell = "cudnn_lstm"
        # first represent china as a vectors
        with tf.variable_scope("china_attention", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            china_att_rep = self.get_attention_rep(china_att, self.attention_length, self.params["fw_cell"], self.params["fw_cell_size"])
            att_shape = china_att_rep.get_shape()
            china_att_out = tf.reshape(tf.tile(china_att_rep, [1, self.encoder_length * 25]), shape=(att_shape[0], self.encoder_length, 25, att_shape[-1]))

        enc_inputs = tf.concat([enc, china_att_out], axis=3)
        enc_inputs = tf.transpose(enc_inputs, [0, 2, 1, 3])
        
        with tf.variable_scope("encoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            # feed data stations to a single net then concat with lstm layers
            # feed outputs to double net
            enc = tf.reshape(enc, shape=(pr.batch_size * 25, self.encoder_length, self.encoder_vector_size))
            _, enc_lstm = rnn_utils.execute_sequence(enc, self.params)
            # b x 25 x 24 
            enc_nn = self.add_single_net(tf.layers.flatten(enc))
            enc_combined = tf.concat([enc_lstm[-1], enc_nn], axis=1)
            enc_combined_s = enc_combined.get_shape()
            enc_combined = tf.reshape(enc_combined, shape=(pr.batch_size, 25, enc_combined_s[-1]))
            enc_outputs = self.add_upper_net(enc_combined)

            # concat station rep vectors with others' attention vectors
            outputs = []
            for x in xrange(25):                
                indices = range(25)
                del(indices[x])
                others = tf.gather(enc_outputs, indices, axis=1)
                current = tf.squeeze(tf.gather(enc_outputs, [x], axis=1))
                current_ = tf.reshape(tf.tile(current, [1, 24]), shape=(pr.batch_size, 24, 200))
                others = tf.concat([others, current_], axis=1)
                with tf.name_scope("attention_score_%i" % x):
                    attention_score = tf.layers.dense(others, units=1, name="softmax_score")
                    attention_score = tf.nn.softmax(tf.squeeze(attention_score, axis=-1), name="softmax_prob")
                others = tf.transpose(others, [2, 0, 1])
                attention_vectors = tf.multiply(others, attention_score)
                attention_vectors = tf.transpose(attention_vectors, [1, 2, 0])
                attention_vectors = tf.reduce_sum(attention_vectors, axis=1)
                current = tf.concat([current, attention_vectors], axis=1)
                with tf.name_scope("prediction_%i" % x):
                    pred = tf.layers.dense(current, units=self.decoder_length, activation=tf.nn.sigmoid, name="predictions")
                    pred = tf.layers.dropout(pred, self.dropout_placeholder)
                    tf.get_variable_scope().reuse_variables()
                outputs.append(pred)

            outputs = tf.stack(outputs, axis=1)
            outputs = tf.transpose(outputs, [0, 2, 1])

        return outputs

    def add_single_net(self, inputs, units=100, activation=tf.nn.relu, name="hidden_relu_basic"):
        out_hid1 = tf.layers.dense(inputs, units=units, activation=activation, name=name)
        return out_hid1

    def add_upper_net(self, inputs):
        out_hid1 = tf.layers.dense(inputs, units=200, activation=tf.nn.relu, name="hidden_relu_upper_1")
        out_hid2 = tf.layers.dense(out_hid1, units=200, activation=tf.nn.relu, name="hidden_relu_upper_2")
        out_hid2 = tf.layers.dropout(out_hid2, self.dropout_placeholder)
        return out_hid2


    # china representation
    def get_attention_rep(self, inputs, attention_length, lstm_cell, hidden_size):
        with tf.variable_scope("attention_rep", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = {
                "fw_cell": lstm_cell,
                "fw_cell_size": hidden_size
            }
            outputs, _ = rnn_utils.execute_sequence(inputs, params)
            # outputs = tf.stack(outputs, axis=1)
            attention_logits = tf.squeeze(tf.layers.dense(outputs, units=1, activation=None, name="attention_logits"))
            attention = tf.nn.softmax(attention_logits)
            outputs = tf.transpose(outputs, [2, 0, 1])
            outputs = tf.multiply(outputs, attention)
            outputs = tf.transpose(outputs, [1, 2, 0])
            outputs = tf.reduce_sum(outputs, axis=1)
        return outputs          