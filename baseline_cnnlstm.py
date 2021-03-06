from __future__ import print_function
from __future__ import division

import sys
import time
import copy
import numpy as np
import tensorflow as tf
import properties as pr
import utils
import model_utils
import rnn_utils
import heatmap


class BaselineModel(object):

    
    def __init__(self, encoder_length=24, decoder_length=24, grid_size=25, rnn_hidden_units=128, 
                encode_vector_size=12, decode_vector_size=6, learning_rate=0.01, batch_size=64, loss="mse", 
                df_ele=6, rnn_layers=1, dtype="grid", attention_length=24, atttention_hidden_size=17,
                use_attention=True, use_cnn=False, use_gen_cnn=True, forecast_factor=1, mtype=6, num_class=0, districts=25, **kwargs):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.sequence_length = encoder_length + decoder_length
        self.is_training = True
        self.grid_size = grid_size
        self.rnn_hidden_units=rnn_hidden_units
        self.encode_vector_size = encode_vector_size
        self.decode_vector_size = decode_vector_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grid_square = grid_size * grid_size
        self.loss = loss
        self.df_ele = encode_vector_size - decode_vector_size
        self.dtype = dtype        
        self.map = heatmap.build_map()
        self.use_cnn = use_cnn
        self.districts = districts
        self.rnn_layers = rnn_layers
        # attention config
        self.use_attention = use_attention
        self.attention_length = attention_length
        self.atttention_hidden_size = atttention_hidden_size

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.dropout = 0.9
        self.num_class = num_class
        self.e_params = {
            "fw_cell_size" : self.rnn_hidden_units,
            "fw_cell": "cudnn_lstm",
            "batch_size" : self.batch_size,
            "type": 0,
            "rnn_layer": self.rnn_layers,
            "grid_size": grid_size,
            "dropout": self.dropout,
            "direction": "unidirectional",
            "de_output_size" : self.grid_square,
            "num_class": num_class,
            "districts": self.districts
        }
        # if self.dtype == "grid":
        #     if self.use_cnn:
        #         self.grd_cnn = 12 * 12
        #     else:
        #         self.grd_cnn = self.grid_square * self.encode_vector_size
        #     # size of output prediction matrix (24 * 24)
        #     self.e_params["de_output_size"] = self.grid_square
        #     if self.rnn_layers > 1:
        #         self.e_params["fw_cell_size"] = self.grd_cnn
        # else:
        #     self.e_params["de_output_size"] = 1
        #     if self.rnn_layers > 1:
        #         self.e_params["fw_cell_size"] = self.districts
        # true mean use generate output by transposed cnn whereas. cnn-lstm doesn't use trans cnn generation 
        self.use_gen_cnn = use_gen_cnn
        # 6 is regular cnn-lstm, 3 is msf apnet
        self.mtype = mtype
        self.use_batch_norm = False
        # 0 is predict pm2.5 while 1 is predict pm10
        self.forecast_factor = forecast_factor
        self.all_pred = True
    
    def set_training(self, training):
        self.is_training = training

    def set_data(self, datasets, train, valid, attention_vectors=None, labels=None):
        self.datasets = datasets
        self.train = train
        self.valid = valid
        self.labels = labels
        self.attention_vectors = attention_vectors

    def init_ops(self, is_train=True):
       self.add_placeholders()
       self.output = self.inference()
       self.loss_op = self.add_loss(self.output)
       if is_train:
           self.train_op = model_utils.add_training_op(self.loss_op, self.learning_rate)
       self.merged = tf.summary.merge_all()
    
    # preserve memory for tensors
    def add_placeholders(self):
        self.embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="Variable")
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))
        if self.dtype == "grid":
            if self.all_pred:
                self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.grid_size, self. grid_size))
            else:
                self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.grid_size, self. grid_size))
        # china attention_inputs
        self.attention_embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="attention_embedding")
        self.label_embedding = tf.Variable([], validate_shape=False, dtype=tf.int32, trainable=False, name="label_class_embedding")
        self.attention_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.attention_length))
        self.dropout_placeholder = tf.Variable(self.dropout, False, name="dropout", dtype=tf.float32)


    def inference(self):
        # embedding = tf.Variable(self.datasets, name="Embedding")
        # check if dtype is grid then just look up index from the datasets 
        enc, dec = self.lookup_input(self.encoder_inputs, self.decoder_inputs)
        # return final_state and enc_output (enc_ou)
        final_state, _  = self.exe_encoder(enc)
        attention = None
        if self.use_attention:
            # batch size x rnn_hidden_size
            inputs = tf.nn.embedding_lookup(self.attention_embedding, self.attention_inputs)
            attention = self.get_attention_rep(inputs)
        outputs = self.exe_decoder(dec, final_state, attention)
        return outputs

    # mapping input indices to dataset
    def lookup_input(self, enc, dec):
        enc = tf.nn.embedding_lookup(self.embedding, enc)
        dec_f = tf.nn.embedding_lookup(self.embedding, dec)
        print("predict %i" % self.forecast_factor)
        if self.dtype == "grid":
            enc.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
            dec_f.set_shape((self.batch_size, self.decoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
            # embedding = tf.Variable(self.daitasets, name="embedding")
            dec = dec_f[:,:,:,:,self.df_ele:]
            dec.set_shape((self.batch_size, self.decoder_length, self.grid_size, self.grid_size, self.decode_vector_size))
            # 0 is pm2.5 1 is pm10
            if self.all_pred:
                self.pred_placeholder = dec_f[:,:,:,:, self.forecast_factor]
            else:
                self.pred_placeholder = dec_f[:,-1,:,:, self.forecast_factor]
        else:
            enc.set_shape((self.batch_size, self.encoder_length, 25, self.encode_vector_size))
            dec_f.set_shape((self.batch_size, self.decoder_length, 25, self.encode_vector_size))
            dec = dec_f[:,:,:,self.df_ele:]
            dec.set_shape((self.batch_size, self.decoder_length, 25, self.decode_vector_size))
            if self.all_pred:
                self.pred_placeholder = dec_f[:,:,:, self.forecast_factor]
                self.pred_placeholder = tf.reduce_mean(self.pred_placeholder, axis=2)
            else:
                self.pred_placeholder = dec_f[:,-1,:, self.forecast_factor]
                self.pred_placeholder = tf.reduce_mean(self.pred_placeholder, axis=1)
        self.pred_class_placeholder = None
        if self.num_class:
            self.pred_class_placeholder = tf.nn.embedding_lookup(self.label_embedding, self.decoder_inputs)
            self.pred_class_placeholder.set_shape((self.batch_size, self.decoder_length, self.districts))
        return enc, dec

    # perform encoder
    def exe_encoder(self, enc, use_batch_norm=None, dropout=None):
        if not dropout:
            dropout = self.dropout
        if not use_batch_norm:
            self.use_batch_norm = use_batch_norm
        
        with tf.variable_scope("encoder", initializer=self.initializer):
            if self.dtype == "grid":
                if self.use_cnn:
                    # add one cnn layer here
                    print(enc.get_shape())
                    cnn = rnn_utils.get_cnn_rep(enc, mtype=self.mtype, use_batch_norm=use_batch_norm, dropout=dropout)
                else:
                    cnn = enc
                cnn = tf.layers.flatten(cnn)
                cnn_shape = cnn.get_shape()
                # last_dim = cnn_shape[-1] * cnn_shape[-2]
                # last_dim = int(cnn_shape[-1]) / self.encoder_length
                last_dim = int(cnn_shape[-1])
                enc_data = tf.reshape(cnn, [self.batch_size, self.encoder_length, last_dim])
                # enc_data = tf.unstack(enc_data, axis=1)
            else:
                enc_data = tf.reduce_mean(enc, axis=1)
            # then push through lstm
            enc_output, fn_state = rnn_utils.execute_sequence(enc_data, self.e_params)
            if self.rnn_layers > 1:
                fn_state = fn_state[-1]
        return fn_state, enc_output

    #perform decoder
    def exe_decoder(self, dec, enc_output, attention=None):
        params = copy.deepcopy(self.e_params)
        if "gru" in self.e_params["fw_cell"]:
            params["fw_cell"] = "gru_block"
        elif "rnn" in self.e_params["fw_cell"]:
            params["fw_cell"] = "rnn"
        else:
            params["fw_cell"] = "lstm_block"
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            if self.dtype == "grid":                
                outputs, _ = rnn_utils.execute_decoder_cnn(dec, enc_output, self.decoder_length, params, attention, self.use_cnn, self.use_gen_cnn, self.mtype, self.use_batch_norm, self.dropout)
                outputs = tf.stack(outputs, axis=1)
            else: 
                if "gru" in self.e_params["fw_cell"] or self.e_params["fw_cell"] == "rnn":
                    enc_output = tf.squeeze(enc_output[0], 0)
                dec_data = tf.transpose(dec,[0, 2, 1, 3])
                dec_data = tf.reduce_mean(dec_data, axis=1)
                outputs = rnn_utils.execute_decoder(dec_data, enc_output, self.decoder_length, params, attention, self.dropout_placeholder)
                outputs = tf.reshape(tf.stack(outputs, axis=1), shape=(pr.batch_size, self.decoder_length))
        return outputs
    
    # china representation
    def get_attention_rep(self, inputs):
        with tf.variable_scope("attention_rep", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = {
                "fw_cell": self.e_params["fw_cell"],
                "fw_cell_size": self.rnn_hidden_units,
                "direction": self.e_params["direction"],
                "rnn_layer": self.e_params["rnn_layer"]
            }
            inputs.set_shape((self.batch_size, self.attention_length, self.atttention_hidden_size))
            # inputs = tf.unstack(inputs, self.attention_length, 1)
            outputs, _ = rnn_utils.execute_sequence(inputs, params)
            # outputs = tf.stack(outputs, axis=1)
            outputs = self.get_softmax_attention(outputs)
        return outputs
    
    # return attentional vector of a set of vectors
    # compute softmax -> multiple softmax score with original vectors -> reduce_sum
    def get_softmax_attention(self, inputs):
        return rnn_utils.get_softmax_attention(inputs)
    
    # add loss function
    # output will be loss scalar    
    def add_loss(self, output):
        if self.dtype == "grid":
            y = tf.layers.flatten(output)
            pred = tf.layers.flatten(self.pred_placeholder)
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
        if self.num_class:
            lbl_ops = tf.assign(self.label_embedding, self.labels, False)
            session.run(lbl_ops)
    
    # operation of each epoch
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, verbose=True, train=False, shuffle=True, stride=4):
        dt_length = len(data)
        # print("data_size: ", dt_length)
        cons_b = self.batch_size * stride
        total_steps = dt_length // cons_b
        total_loss = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        preds = []
        for step in range(total_steps):
            index = range(step * cons_b,
                          (step + 1) * cons_b, stride)
            # just the starting points of encoding batch_size,
            idx = ct[index]
            # switch batchsize, => batchsize * encoding_length
            # lookup from x => x + encoder_length
            # ct_t = batch_size x encoder_length
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
            dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])

            feed = {
                self.encoder_inputs : ct_t,
                self.decoder_inputs: dec_t,
            }
            if self.use_attention:
                feed[self.attention_inputs] = ct_t

            if train:
                loss, pred, _= session.run([self.loss_op, self.output, self.train_op], feed_dict=feed)
            else:
                loss, pred = session.run([self.loss_op, self.output], feed_dict=feed)

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
        
