from __future__ import print_function
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
import properties as prp
import utils
import model_utils
import rnn_utils
import heatmap


class BaselineModel(object):

    
    def __init__(self, encoder_length=24, decoder_length=24, grid_size=25, rnn_hidden_units=128, 
                encode_vector_size=12, decode_vector_size=6, learning_rate=0.01, batch_size=64, loss="mae", 
                df_ele=6, rnn_layers=1, dtype="grid", attention_length=24, atttention_hidden_size=17,
                use_attention=True, use_cnn=False, no_cnn_decoder=False):
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
        self.dtype = dtype        
        self.map = heatmap.build_map()
        self.use_cnn = use_cnn
        self.districts = 25
        self.rnn_layers = rnn_layers
        self.no_cnn_decoder = no_cnn_decoder
        self.atttention_hidden_size = atttention_hidden_size
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.e_params = {
            "fw_cell_size" : self.rnn_hidden_units,
            "fw_cell": "cudnn_lstm",
            "batch_size" : self.batch_size,
            "type": 0,
            "rnn_layer": self.rnn_layers,
            "decoder_strides": (1,2,2),
            "decoder_kernel": (self.decode_vector_size + 1,3,3),
            "grid_size": grid_size,
            "decoder_filter": 1
        }
        self.dropout = 0.9
        self.use_attention = use_attention
        self.attention_length = attention_length
        if self.dtype == "grid":
            if self.use_cnn:
                self.grd_cnn = 12 * 12
            else:
                self.grd_cnn = self.grid_square * self.encode_vector_size
            # size of output prediction matrix (24 * 24)
            self.e_params["de_output_size"] = self.grid_square
            if self.rnn_layers > 1:
                e_params["fw_cell_size"] = self.grd_cnn
        else:
            self.e_params["de_output_size"] = self.districts
            if self.rnn_layers > 1:
                self.e_params["fw_cell_size"] = self.districts
    
    def set_training(self, training):
        self.is_training = training

    def set_data(self, datasets, train, valid, attention_vectors=None):
        self.datasets = datasets
        self.train = train
        self.valid = valid
        self.attention_vectors = attention_vectors

    def init_ops(self):
       self.add_placeholders()
       self.output = self.inference()
       self.loss_op = self.add_loss(self.output)
       self.train_op = model_utils.add_training_op(self.loss_op, self.learning_rate)
       self.merged = tf.summary.merge_all()
    
    # preserve memory for tensors
    def add_placeholders(self):
        self.embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="Variable")
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))
        if self.dtype == "grid":
            self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.grid_size, self. grid_size))
        # china attention_inputs
        self.attention_embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="attention_embedding")
        self.attention_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.attention_length))
        self.dropout_placeholder = tf.Variable(self.dropout, False, name="dropout", dtype=tf.float32)

    def inference(self):
        # embedding = tf.Variable(self.datasets, name="Embedding")
        # check if dtype is grid then just look up index from the datasets 
        enc, dec = self.lookup_input()
        enc_output = self.exe_encoder(enc)
        attention = None
        if self.use_attention:
            # batch size x rnn_hidden_size
            inputs = tf.nn.embedding_lookup(self.attention_embedding, self.attention_inputs)
            attention = self.get_attention_rep(inputs)
        outputs = self.exe_decoder(dec, enc_output, attention)
        return outputs

    # perform encoder
    def exe_encoder(self, enc):
        with tf.variable_scope("encoder", initializer=self.initializer):
            if self.dtype == "grid":
                if self.use_cnn:
                    # add one cnn layer here
                    cnn = self.get_cnn_rep(enc, self.encoder_length, self.encode_vector_size)
                else:
                    cnn = enc
                enc_data = tf.reshape(cnn, [self.batch_size, self.encoder_length, self.grd_cnn])
                # enc_data = tf.unstack(enc_data, axis=1)
            else:
                enc = tf.reshape(tf.reshape(enc, [-1]), [self.batch_size, self.encoder_length, self.districts * self.encode_vector_size])
                # enc_data = tf.unstack(enc, axis=1)
            # then push through lstm
            _, enc_output = rnn_utils.execute_sequence(enc_data, self.e_params)
            if self.rnn_layers > 1:
                enc_output = enc_output[-1]
        return enc_output

    # mapping input indices to dataset
    def lookup_input(self):
        enc = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        enc.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        dec_f = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
        dec_f.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        if self.dtype == "grid":
            # embedding = tf.Variable(self.datasets, name="embedding")
            dec = dec_f[:,:,:,:,self.df_ele:]
            dec.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.decode_vector_size))
            self.pred_placeholder = dec_f[:,:,:,:,0]
        else:
            dec = dec_f[:,:,:,self.df_ele:]
            dec.set_shape((self.batch_size, self.encoder_length, 25, self.decode_vector_size))
            self.pred_placeholder = dec_f[:,:,:,0]
        return enc, dec

    #perform decoder
    def exe_decoder(self, dec, enc_output, attention=None):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            if self.dtype == "grid":
                if self.no_cnn_decoder:
                    if self.use_cnn:
                        # add one cnn layer before decoding using lstm
                        cnn = self.get_cnn_rep(dec, self.decoder_length, self.decode_vector_size)
                    else:
                        cnn = dec
                        grd_cnn = self.grid_square * self.decode_vector_size
                    dec_data = tf.reshape(cnn, [self.batch_size, self.decoder_length, grd_cnn])
                else:
                    outputs = rnn_utils.execute_decoder_cnn(dec, enc_output, self.decoder_length, self.e_params, attention)
            else:
                dec_data = tf.reshape(tf.reshape(dec, [-1]), [self.batch_size, self.decoder_length, self.districts * self.decode_vector_size])
            #finally push -> decoder
            if self.no_cnn_decoder:
                outputs = rnn_utils.execute_decoder(dec_data, enc_output, self.decoder_length, self.e_params, attention, self.dropout_placeholder)
            else:
                outputs = tf.stack(outputs, axis=1)
        return outputs

    # get cnn representation of input images 
    def get_cnn_rep(self, dec, length, vector_size):
        cnn_inputs = tf.transpose(dec, [0,1,4,2,3])
        #flatten
        cnn_inputs = tf.reshape(cnn_inputs, [-1])
        # cnn_inputs = tf.reshape(self.decoder_inputs, [-1])
        
        cnn_inputs = tf.reshape(cnn_inputs, [self.batch_size * length, vector_size, self.grid_size, self.grid_size])
        cnn_inputs = tf.expand_dims(cnn_inputs, 4)
        cnn = tf.layers.conv3d(
            inputs=cnn_inputs,
            strides=(1,2,2),
            filters=1,
            kernel_size=(vector_size,3,3),
            padding="valid"
        )
        #output should have shape: bs * length, 12, 12, 1
        cnn = tf.reshape(tf.squeeze(cnn), [-1])
        return cnn
    
    # china representation
    def get_attention_rep(self, inputs):
        with tf.variable_scope("attention_rep", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = {
                "fw_cell": "cudnn_lstm",
                "fw_cell_size": self.rnn_hidden_units
            }
            inputs.set_shape((self.batch_size, self.attention_length, self.atttention_hidden_size))
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
    
    # add loss function
    # output will be loss scalar    
    def add_loss(self, output):
        if self.dtype == "grid":
            matrix_size = [self.batch_size, self.decoder_length * self.grid_square]
            y = tf.reshape(tf.reshape(output, [-1]), matrix_size)
            pred = tf.reshape(self.pred_placeholder, matrix_size)
        else:
            y = output
            pred = self.pred_placeholder
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

    def map_to_grid(self, ct):
        res = []
        for b in ct:
            res_t = []
            for t in b:
                g = heatmap.fill_map(t, self.map)
                res_t.append(g)
            res.append(res_t)
        return np.asarray(res, dtype=np.float32)

    # assign datasets to embedding 
    def assign_datasets(self, session):
        assign_ops = tf.assign(self.embedding, self.datasets, False)
        session.run(assign_ops)
        if self.use_attention:
            att_ops = tf.assign(self.attention_embedding, self.attention_vectors, False)
            session.run(att_ops)
    
    # operation of each epoch
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=True, train=False, shuffle=True):
        dp = self.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        dt_length = len(data)
        # print("data_size: ", dt_length)
        total_steps = dt_length // self.batch_size
        total_loss = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        preds = []
        for step in range(total_steps):
            index = range(step * self.batch_size,
                          (step + 1) * self.batch_size)
            # just the starting points of encoding batch_size,
            ct_t = ct[index]
            # switch batchsize, => batchsize * encoding_length
            # lookup from x => x + encoder_length
            # ct_t = batch_size x encoder_length
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
            dec_t = ct_t + self.decoder_length

            feed = {
                self.encoder_inputs : ct_t,
                self.decoder_inputs: dec_t,
            }
            if self.use_attention:
                feed[self.attention_inputs] = ct_t

            loss, pred, _= session.run(
                [self.loss_op, self.output, train_op], feed_dict=feed)
            
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

            preds.append(pred)

        if verbose:
            sys.stdout.write("\r")
        
        total_loss = np.mean(total_loss)
        if train_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag= "Total Loss", simple_value=total_loss)
            train_writer.add_summary(summary, num_epoch)
        return total_loss, preds
        