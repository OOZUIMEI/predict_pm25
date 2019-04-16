import numpy as np
import tensorflow as tf
from copy import deepcopy
from apgan import APGan
import properties as pr
import rnn_utils

# add noise to input & label
# alpha is zero => use only content loss in APNet
# embedding: 
# https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/
# https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

class APNet(APGan):

    def __init__(self, offset=0, **kwargs):
        super(APNet, self).__init__(**kwargs)
        self.beta1 = 0.5
        self.alpha = 0
        self.attention_length = 96
        self.e_params["direction"] = "unidirectional"
        self.e_params["rnn_layer"] = 1
        self.e_params["dropout"] = self.dropout
        self.e_params["elmo"] = False
        # use up-scale transposed cnn or not 
        # self.use_gen_cnn = False
        # flag checks whether uses weather or not
        self.use_weather = True 
        # flag checks weather uses encoder or not. if not it's merely decode with weather or with china
        self.use_encoder = True
        self.enc_att_dis = None
        self.offset = offset
        print("only pred exe from %i" % self.offset)
    
    def inference(self, is_train=True):
        fake_outputs, _, classes = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        self.gen_loss = self.get_generator_loss(None, fake_outputs, None, classes)
        if is_train:
            self.gen_op = self.train_generator(self.gen_loss)
        return fake_outputs
    
    # mapping input indices to dataset
    def lookup_input(self, enc, dec):
        self.pred_class_placeholder = None
        if self.use_encoder:
            enc = tf.nn.embedding_lookup(self.embedding, enc)
            enc.set_shape((self.batch_size, self.encoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        else:
            enc = None
        dec_f = tf.nn.embedding_lookup(self.embedding, dec)
        dec_f.set_shape((self.batch_size, self.decoder_length, self.grid_size, self.grid_size, self.encode_vector_size))
        self.pred_placeholder = dec_f[:,self.offset:,:,:,self.forecast_factor]
        if self.use_weather:
            dec = dec_f[:,:,:,:,self.df_ele:]
            dec.set_shape((self.batch_size, self.decoder_length, self.grid_size, self.grid_size, self.decode_vector_size))
            # 0 is pm2.5 1 is pm10
        else:
            if self.use_encoder:
                enc = enc[:,:,:,:,:self.df_ele]
            dec = None
        return enc, dec
    

    # this is different to prevous apnet model. 
    # decode and encoder do not shared softmax weights => when predicting long (>24) => decrease loss then increase
    def add_conditional_layer(self, dec, enc_outputs, attention=None):
        if self.use_encoder:
            with tf.variable_scope("encoder_attention", initializer=self.initializer):
                enc_outputs = self.get_softmax_attention(enc_outputs)
                # enc_outputs = tf.reduce_mean(enc_outputs, axis=1)
                # self.enc_att_dis = enc_att_dis

        # add attentional layer here to measure the importance of each timestep. (past hidden, future forecast, china)
        with tf.variable_scope("conditional", initializer=self.initializer):
            # use weather to validate impact factor to seoul weather
            if self.use_weather:
                cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
                cnn_dec_input = tf.layers.flatten(cnn_dec_input)
                cnn_shape = cnn_dec_input.get_shape()
                dec_data = tf.reshape(cnn_dec_input, [self.batch_size, self.decoder_length, int(cnn_shape[-1])])
                dec_rep, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
                # this is forecast weather vector
                # hidden_input = tf.reduce_mean(dec_rep, axis=1)
                hidden_input = self.get_softmax_attention(dec_rep)
                if self.use_encoder:
                    #concat encoder output and weather forecast
                    hidden_input = tf.concat([enc_outputs, hidden_input], axis=1)
            else:
                hidden_input = enc_outputs

            if not attention is None and not hidden_input is None:
                # concate with china factor
                hidden_input = tf.concat([hidden_input, attention], axis=1)
            elif not attention is None:
                hidden_input = attention

            # get output shape to check the number of concatenation vectors:
            # if hidden_input is none then die
            hidden_input_shape = hidden_input.get_shape()
            if hidden_input_shape[-1] != 128:
                # dec_hidden_vectors with shape bs x 128 
                dec_hidden_vectors = tf.layers.dense(hidden_input, 128, name="conditional_layer", activation=tf.nn.tanh)
                if self.dropout:
                    dec_hidden_vectors = tf.nn.dropout(dec_hidden_vectors, 0.5)
            else:
                # what if all vectors is none:
                dec_hidden_vectors = hidden_input
                
            return dec_hidden_vectors

    #perform decoder to produce outputs of the generator
    # dec_hidden_vectors: bs x rnn_hidden_units
    def exe_decoder(self, dec_hidden_vectors, fn_state):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            params = deepcopy(self.e_params)
            if "gru" in self.e_params["fw_cell"]:
                params["fw_cell"] = "gru_block"
            else:
                params["fw_cell"] = "lstm_block"
            dctype = 5
            if self.grid_size == 32:
                dctype = 7
            outputs = rnn_utils.execute_decoder_cnn(None, fn_state, self.decoder_length, params, dec_hidden_vectors, \
                                                    self.use_cnn, self.use_gen_cnn, self.mtype, self.use_batch_norm, \
                                                    self.dropout, dctype=dctype, offset=self.offset)
            classes = None
            outputs = tf.stack(outputs, axis=1)
            outputs = tf.reshape(outputs, [self.batch_size, self.decoder_length - self.offset, self.grid_size * self.grid_size])
        return outputs, classes

    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])
        feed = {
            self.encoder_inputs : ct_t,
            self.decoder_inputs: dec_t
        }
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])
        
        if not train:
            pred, gen_loss = session.run([self.outputs, self.gen_loss], feed_dict=feed)
            total_gen_loss += gen_loss
            # print("preds", pred[0,0,:10])
        else:
            gen_loss, pred, _ = session.run([self.gen_loss, self.outputs, self.gen_op], feed_dict=feed)
            total_gen_loss += gen_loss

        return pred, total_gen_loss, 0

    
