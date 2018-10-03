import tensorflow as tf 
import numpy as np
from NeuralNet import NeuralNetwork
import rnn_utils


class  Adain(NeuralNetwork):
    
    def __init__(self, rnn_hidden_units=300, attention_length=24):
        self.rnn_hidden_units = rnn_hidden_units
        self.attention_length = attention_length
        self.params = {
            "fw_cell": "cudnn_lstm",
            "fw_cell_size": self.rnn_hidden_units,
            "fw_layers": 2
        }

    def inference(self):
        
        enc, _, china_att = self.lookup_input()
        self.lstm_cell = "cudnn_lstm"
        # b x 24 x 25 x H
        with tf.variable_scope("china_attention", initializer=self.initializer(), reuse=tf.AUTO_REUSE):
            china_att_rep = self.get_attention_rep(china_att, self.attention_length, self.params["fw_cell"], self.params["fw_cell_cize"])
            att_shape = att_out.get_shape()
            china_att_out = tf.reshape(tf.tile(china_att_rep, [1, self.encoder_length * 25]), shape=(att_shape[0], self.encoder_length, 25, att_shape[-1]))

        enc_inputs = tf.concat([enc_inputs, china_att_out], axis=3)
        enc_inputs = tf.transpose(enc_inputs, [0, 2, 1, 3])
        
        with tf.variable_scope("encoder", initializer=self.initializer(), reuse=tf.AUTO_REUSE)
            enc = tf.reshape(enc, shape=(pr.batch_size * 25, self.encoder_length, self.encoder_vector_size))
            _, enc_lstm = rnn_utils.execute_sequence(enc, self.params)
            # b x 25 x 24 
            enc_nn = self.add_single_net(tf.layers.flatten(enc))
            enc_combined = tf.concat([enc_lstm, enc_nn], axis=1)
            enc_combined_s = enc_combined.get_shape()
            enc_combined = tf.reshape(enc_combined, shape=(pr.batch_size, 25, enc_combined_s[-1]))
            enc_ouputs = self.add_upper_net(enc_combined)

            outputs = []
            for x in xrange(25):                
                indices = range(25)
                del(indices[x])
                others = tf.gather(enc_outputs, indices)
                current = tf.gather(enc_outputs, [x])
                current_ = tf.tile (current, [1, 24])       
                others = tf.concat([others, current_], axis=1)
                attention_score = tf.nn.softmax(others)
                attention_vectors = tf.reduce_sum(others * attention_score)
                current = tf.concat([current, attention_vectors], axis=1)           
                outputs.append(current)
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.layers.dense(outputs, activation=tf.nn.sigmoid, "predictions")

        return outputs

    def add_single_net(self, inputs):
        out_hid1 = tf.layers.dense(inputs, units=100, activation=tf.nn.relu, name="hidden_relu_basic")
        return out_hid1

    def add_upper_net(self, inputs):
        out_hid1 = tf.layers.dense(inputs, units=200, activation=tf.nn.relu, name="hidden_relu_upper_1")
        out_hid2 = tf.layers.dense(out_hid1, units=200, activation=tf.nn.relu, name="hidden_relu_upper_2")
        return out_hid2


    # china representation
    def get_attention_rep(self, inputs, attention_length, lstm_cell, hidden_size):
        with tf.variable_scope("attention_rep", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = {
                "fw_cell": lstm_cell,
                "fw_cell_size": hidden_size
            }
            inputs.set_shape((pr.batch_size, attention_length, hidden_size))
            # inputs = tf.unstack(inputs, self.attention_length, 1)
            outputs, _ = rnn_utils.execute_sequence(inputs, params)
            # outputs = tf.stack(outputs, axis=1)
            attention_logits = tf.squeeze(tf.layers.dense(outputs, units=1, activation=None, name="attention_logits"))
            attention = tf.nn.softmax(attention_logits)
            outputs = tf.transpose(outputs, [2, 0, 1])
            outputs = tf.multiply(outputs, attention)
            outputs = tf.transpose(outputs, [1, 2, 0])
            outputs = tf.reduce_sum(outputs, axis=1)
        return outputs          