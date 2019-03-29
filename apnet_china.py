import time
import numpy as np
import tensorflow as tf
from copy import deepcopy
import properties as pr
import rnn_utils

"""
districts ~ stations
"""
class APNetChina():

    def __init__(self, encoder_length=24, decoder_length=24, grid_size=32, rnn_hidden_units=128, 
                encode_vector_size=12, decode_vector_size=6, learning_rate=0.00001, batch_size=64, 
                attention_length=24, atttention_hidden_size=17, use_attention=True, num_class=6, 
                districts=36, encoder_type=9, **kwargs):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.grid_size = grid_size
        self.rnn_hidden_units = rnn_hidden_units
        self.encode_vector_size = encode_vector_size
        self.decode_vector_size = decode_vector_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.attention_length = attention_length
        self.atttention_hidden_size = atttention_hidden_size
        self.use_attention = use_attention
        self.use_gen_cnn = False
        self.forecast_factor = 0
        self.num_class = num_class
        self.districts = districts
        self.dropout = 0.5
        self.mtype = encoder_type
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.e_params = {
            "fw_cell_size" : self.rnn_hidden_units,
            "fw_cell": "cudnn_lstm",
            "batch_size" : self.batch_size,
            "type": 0,
            "rnn_layer": 1,
            "grid_size": grid_size,
            "dropout": self.dropout,
            "direction": "unidirectional",
            "de_output_size" : grid_size * grid_size,
            "num_class": num_class,
            "districts": self.districts
        }
        self.trainable_scope = ""
    
    def init_ops(self, is_train=True):
        self.add_placeholders()
        self.outputs = self.inference(is_train)
    
    def set_data(self, datasets, train, valid, attention_vectors=None, labels=None):
        self.datasets = datasets
        self.train = train
        self.valid = valid
        self.labels = labels
        self.attention_vectors = attention_vectors
    
    def assign_datasets(self, session):
        assign_ops = tf.assign(self.embedding, self.datasets, False)
        session.run(assign_ops)
        if self.use_attention:
            att_ops = tf.assign(self.attention_embedding, self.attention_vectors, False)
            session.run(att_ops)
        if self.num_class:
            lbl_ops = tf.assign(self.label_embedding, self.labels, False)
            session.run(lbl_ops)
    
     # preserve memory for tensors
    def add_placeholders(self):
        self.embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="Variable_embedding")
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_length))
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))
        self.pred_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.decoder_length, self.grid_size, self. grid_size))
        self.pred_class_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length, self.districts))
        # weather attention_inputs
        self.attention_embedding = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="attention_embedding")
        self.label_embedding = tf.Variable([], validate_shape=False, dtype=tf.int32, trainable=False, name="label_embedding")

    def inference(self, is_train=True):
        outputs, classes = self.create_generator(self.encoder_inputs, self.decoder_inputs)
        self.gen_loss = None
        if is_train:
            self.gen_loss = self.get_generator_loss(outputs, classes)
            self.gen_op = self.train_generator(self.gen_loss, self.trainable_scope)
        self.outputs = tf.reshape(outputs, [self.batch_size, self.decoder_length, self.grid_size * self.grid_size])
        return outputs

    # generate output images
    def create_generator(self, enc, dec):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, enc_w, dec_w = self.lookup_input(enc, dec)
            en_hidden_vector, fn_state, enc_w_h  = self.exe_encoder(enc, enc_w)
            # use future weather forecast for decoder
            if self.use_attention:
                with tf.variable_scope("future_forecast", self.initializer, reuse=tf.AUTO_REUSE):
                    forecast_outputs, _ = rnn_utils.execute_sequence(dec_w, self.e_params)
                    forecast_attention = rnn_utils.get_softmax_attention(forecast_outputs)
                    # concate enc_w_rep, dec_w_rep
                    weather_hidden_vector = tf.concat([en_hidden_vector, enc_w_h, forecast_attention], axis=1)
                outputs, classes = self.exe_decoder(forecast_outputs, fn_state, weather_hidden_vector, enc)
            else:
                outputs, classes = self.exe_decoder(None, fn_state, en_hidden_vector, enc)
        
        return outputs, classes
    
    # mapping input indices to dataset
    def lookup_input(self, enc, dec):
        # encoder pollutants inputs
        enc_vectors = tf.nn.embedding_lookup(self.embedding, enc)
        enc_vectors.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        #encoder weather
        enc_w = tf.nn.embedding_lookup(self.attention_embedding, enc)
        enc_w.set_shape((self.batch_size, self.encoder_length, self.atttention_hidden_size))
        #decoder weather
        dec_w = tf.nn.embedding_lookup(self.attention_embedding, dec)
        dec_w.set_shape((self.batch_size, self.decoder_length, self.atttention_hidden_size))
        # labels for reg and class
        dec_vectors = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
        dec_vectors.set_shape((self.batch_size, self.decoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        self.pred_placeholder = dec_vectors[:,:,:,:,self.forecast_factor]
        self.pred_class_placeholder = tf.nn.embedding_lookup(self.label_embedding, self.decoder_inputs)
        self.pred_class_placeholder.set_shape((self.batch_size, self.decoder_length, self.districts))
        return enc_vectors, enc_w, dec_w


    """
    enc: encoder inputs (pollutants grid)
    enc_weather: encoder weather inputs representation (pushed through previous LSTMs)
    """
    def exe_encoder(self, enc, enc_weather):

        with tf.variable_scope("encoder", initializer=self.initializer):
            # add cnns layers here
            # now, default using batch norm in generator encoder
            cnn = rnn_utils.get_cnn_rep(enc, mtype=self.mtype, use_batch_norm=True, dropout=self.dropout)
            cnn = tf.layers.flatten(cnn)
            cnn_shape = cnn.get_shape()
            last_dim = int(cnn_shape[-1])
            enc_data = tf.reshape(cnn, [self.batch_size, self.encoder_length, last_dim])
            # if use weather attention => concate pollutants output with weather outputs
            if self.use_attention:
                with tf.variable_scope("weather_encoder", initializer=self.initializer):
                    enc_weather_output, _ = rnn_utils.execute_sequence(enc_weather, self.e_params)
                    # get hidden attention of encoder weathers
                    enc_weather_hidden = rnn_utils.get_softmax_attention(enc_weather_output)
                    enc_data = tf.concat([enc_data, enc_weather_output], axis=2)
            else:
                enc_weather_hidden = None
            # then push though lstm
            enc_outputs, fn_state = rnn_utils.execute_sequence(enc_data, self.e_params)
            enc_outputs = rnn_utils.get_softmax_attention(enc_outputs)
            
        return enc_outputs, fn_state, enc_weather_hidden

    #perform decoder to produce outputs of the generator
    """
    forecast_outputs: future weather outputs representation
    fn_state: last state of encoder 
    weather_hidden_vector: concatenation of enc & dec weather attention
    enc_inputs: encoder pollutants inputs
    """
    def exe_decoder(self, forecast_outputs, fn_state, weather_hidden_vector, enc_inputs):
        last_enc_timestep = enc_inputs[:,:,:,0]
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = deepcopy(self.e_params)
            if "gru" in self.e_params["fw_cell"]:
                params["fw_cell"] = "gru_block"
            else:
                params["fw_cell"] = "lstm_block"
            outputs, classes = rnn_utils.execute_decoder_cnn(forecast_outputs, fn_state, self.decoder_length, params, weather_hidden_vector, True, \
                                                            self.use_gen_cnn, self.mtype, False, self.dropout, dctype=5, classify=bool(self.num_class), \
                                                            init_input=last_enc_timestep, grid_input=False)
            outputs = tf.stack(outputs, axis=1)
            if classes:
                classes = tf.stack(classes, axis=1)
        return outputs, classes

    # add generation loss
    def get_generator_loss(self, outputs, classes):
        loss = tf.losses.mean_squared_error(tf.layers.flatten(self.pred_placeholder * 300.0), tf.layers.flatten(outputs))
        if self.num_class:
            class_labels = tf.one_hot(self.pred_class_placeholder, self.num_class)
            classes = tf.reshape(classes, (self.batch_size, self.decoder_length, self.districts, self.num_class))
            class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_labels, logits=classes)
            loss += class_loss

        loss = tf.reduce_mean(loss)
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += 0.0001 * tf.nn.l2_loss(v)
        
        return loss
    
    def train_generator(self, loss, sw=""):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        if sw:
            var = [v for v in tf.trainable_variables() if sw in v.op.name]
            grads = tf.gradients(loss, var)
            train_op = opt.apply_gradients(zip(grads, var))
        else:
            gvs = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(gvs)
        return train_op
    
    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])

        feed = {
            self.encoder_inputs: ct_t,
            self.decoder_inputs: dec_t,
        }
        
        if not train:
            if not self.gen_loss is None:
                gen_loss, pred, label = session.run([self.gen_loss, self.outputs, self.pred_placeholder], feed_dict=feed)
                loss += gen_loss
            else:
                pred, label = session.run([self.outputs, self.pred_placeholder], feed_dict=feed)
        else:
            gen_loss, pred, _= session.run([self.gen_loss, self.outputs, self.gen_op], feed_dict=feed)
            loss += gen_loss
            label = None

        return pred, loss, label

    # using stride to reduce the amount of data to loop over time intervals
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train=False, shuffle=True, stride=2):
        dt_length = len(data)
        total_gen_loss = 0.0
        preds = []
        labels = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]
        
        if self.batch_size >= stride:
            cons_b = self.batch_size * stride
        else:
            cons_b = self.batch_size
            stride = 1
        
        total_steps = dt_length // cons_b
        for step in xrange(total_steps):
            index = range(step * cons_b, (step + 1) * cons_b, stride)
            pred, total_gen_loss, label = self.iterate(session, ct, index, train, total_gen_loss)
            if not train:
                preds.append(pred)  
                labels.append(label)    
        
        total_gen_loss = total_gen_loss / total_steps
        if train_writer:
            summary = tf.Summary()
            summary.value.add(tag= "Loss", simple_value=total_gen_loss)
            train_writer.add_summary(summary, num_epoch)
        return total_gen_loss, (preds, labels)
