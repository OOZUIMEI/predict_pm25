"""
both decoder and encoder using same attention layers
"""
import time
import numpy as np
import tensorflow as tf

# from mask_gan import MaskGan
from baseline_cnnlstm import BaselineModel
import properties as pr
import rnn_utils


# https://github.com/soumith/ganhacks/issues/14
# https://stats.stackexchange.com/questions/251279/why-discriminator-converges-to-1-2-in-generative-adversarial-networks
# https://towardsdatascience.com/understanding-and-optimizing-gans-going-back-to-first-principles-e5df8835ae18
# https://stackoverflow.com/questions/42enc_output690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
# https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
# Reload graph
# https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125
# https://ww2.arb.ca.gov/resources/sources-air-pollution
# flip labels
# add noise to input & label

class APGan(BaselineModel):

    def __init__(self, gamma=0.9, learning_rate=0.0002, use_cnn=True, **kwargs):
        super(APGan, self).__init__(**kwargs)
        # alpha is used for generator loss function
        # [0.001 nodp > 0.001 dp0.5 > 0.005 nodp > 0.005 dp0.5]
        # ? 0.0009
        # 0.0005 is mode collapse. 0.01 is mode collapse when use_flip = False
        self.use_flip = True
        self.alpha = 0.001
        self.use_gen_cnn = True
        self.dropout = 0.5 # help maintain the discriminator (0.5/0.0)
        self.use_batch_norm = False
        self.strides = [2]
        self.beta1 = 0.5
        self.lamda = 100
        # discriminator unit type 6 & 7 is hard to control / should be 4
        self.gmtype = 10
        self.all_pred = True
        self.learning_rate = learning_rate
        self.use_cnn = use_cnn
        self.z_dim = [self.batch_size, self.decoder_length, 128]
        self.z = tf.placeholder(tf.float32, shape=self.z_dim)   
        self.flag = tf.placeholder(tf.float32, shape=[self.batch_size, 1]) 
        # set up multiple cnns layers to generate outputs
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)

    def init_ops(self, is_train=True):
        self.add_placeholders()
        self.outputs = self.inference(is_train)
        # self.merged = tf.summary.merge_all()

    def inference(self, is_train=True):
        fake_outputs, conditional_vectors, _ = self.create_generator(self.encoder_inputs, self.decoder_inputs, self.attention_inputs)
        if is_train:
            fake_vals, real_vals, fake_rewards = self.create_discriminator(fake_outputs, conditional_vectors)
            self.dis_loss = self.add_discriminator_loss(fake_vals, real_vals)
            self.gen_loss = self.get_generator_loss(fake_vals, fake_outputs, fake_rewards)
            self.gen_op = self.train_generator(self.gen_loss)
            self.dis_op = self.train_discriminator(self.dis_loss)
        return fake_outputs
    
    # generate output images
    def create_generator(self, enc, dec, att):
        with tf.variable_scope("generator", self.initializer, reuse=tf.AUTO_REUSE):
            # shape:  batch_size x decoder_length x grid_size x grid_size
            enc, dec = self.lookup_input(enc, dec)
            if not enc is None:
                fn_state, enc_outputs = self.exe_encoder(enc, False, 0.0)
            else:
                # do not use internal historical data
                fn_state, enc_outputs = None, None
            # maybe use only china factor or weather forecast 
            attention = None
            if self.use_attention:
                # print("use attention")
                # batch size x rnn_hidden_size
                inputs = tf.nn.embedding_lookup(self.attention_embedding, att)
                attention = self.get_attention_rep(inputs)
            conditional_vectors = self.add_conditional_layer(dec, enc_outputs, attention)
            outputs, classes = self.exe_decoder(conditional_vectors, fn_state)
        return outputs, conditional_vectors, classes
    
    def sample_z(self):
        # better for pm2.5
        # return np.random.uniform(-1., 1., size=self.z_dim)
        # better for pm10
        return np.random.normal(0., 0.01, size=self.z_dim)
    
    def get_generator_loss(self, fake_preds, outputs, fake_rewards=None, classes=None):
        if self.all_pred:
            labels = tf.reshape(self.pred_placeholder, shape=(self.batch_size, self.decoder_length, self.grid_square))
        else:
            labels = tf.reshape(self.pred_placeholder, shape=(self.batch_size, self.grid_square))
        gen_loss = self.add_generator_loss(fake_preds, outputs, labels, fake_rewards, classes, self.pred_class_placeholder)
        return gen_loss
    
    # add generation loss
    # use log_sigmoid instead of log because fake_vals is w * x + b (not the probability value)
    def add_generator_loss(self, fake_vals, outputs, labels, fake_rewards=None, classes=None, class_labels=None):
        loss = tf.losses.mean_squared_error(labels, outputs)
        if not classes is None and not class_labels is None:
            class_labels = tf.cast(class_labels, tf.int32)
            class_labels = tf.one_hot(class_labels, self.num_class)
            print("class", class_labels.get_shape())
            classes = tf.reshape(classes, (self.batch_size, self.decoder_length, self.districts, self.num_class))
            class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_labels, logits=classes)
            loss += class_loss
        
        if not fake_rewards is None:
            print("Using reinsforcement learning")
            advatages = tf.abs(fake_rewards)
            loss = tf.reduce_mean(tf.multiply(loss, tf.stop_gradient(advatages)))
        else:
            print("Using combined loss function")
            if self.alpha:
                sigmoid_loss = self.alpha * tf.log_sigmoid(fake_vals)
                # sigmoid_loss = self.alpha * tf.losses.sigmoid_cross_entropy(fake_vals, tf.constant(1., shape=[self.batch_size, self.decoder_length]))
                # normal lossmse + (-log(D(G)))
                loss = loss - sigmoid_loss
                #loss_values = sigmoid_loss
                loss = tf.reduce_mean(loss)
            else:
                loss = tf.reduce_mean(loss)
                for v in tf.trainable_variables():
                    if not 'bias' in v.name.lower():
                        loss += 0.0001 * tf.nn.l2_loss(v)
        return loss
    
    # the conditional layer that concat all attention vectors => produces a single vector
    #"""
    def add_conditional_layer(self, dec, enc_outputs, attention=None):
        with tf.name_scope("conditional"):
            cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
            cnn_dec_input = tf.layers.flatten(cnn_dec_input)
            cnn_shape = cnn_dec_input.get_shape()
            dec_data = tf.reshape(cnn_dec_input, [self.batch_size, self.decoder_length, int(cnn_shape[-1])])
            dec_rep, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
            dec_rep = self.get_softmax_attention(dec_rep)
            # add attentional layer here to measure the importance of each timestep.
            enc_outputs = self.get_softmax_attention(enc_outputs)
            # dec_input with shape bs x 3hidden_size
            dec_input = tf.concat([enc_outputs, dec_rep], axis=1)
            if not attention is None:
                dec_input = tf.concat([dec_input, attention], axis=1)
            # dec_hidden_vectors with shape bs x 128 
            dec_hidden_vectors = tf.layers.dense(dec_input, 128, name="conditional_layer", activation=tf.nn.tanh)
            if self.dropout:
                dec_hidden_vectors = tf.nn.dropout(dec_hidden_vectors, 0.5)
            return dec_hidden_vectors
    
    """

    # the conditional layer that concat all attention vectors => produces a single vector
    def add_conditional_layer(self, dec, enc_outputs, attention=None):
        with tf.variable_scope("encoder_softmax", initializer=self.initializer):
            enc_outputs = self.get_softmax_attention(enc_outputs)

        with tf.variable_scope("decoder_softmax", initializer=self.initializer):
            cnn_dec_input = rnn_utils.get_cnn_rep(dec, mtype=self.mtype, use_batch_norm=self.use_batch_norm, dropout=self.dropout)
            cnn_dec_input = tf.layers.flatten(cnn_dec_input)
            cnn_shape = cnn_dec_input.get_shape()
            dec_data = tf.reshape(cnn_dec_input, [self.batch_size, self.decoder_length, int(cnn_shape[-1])])
            dec_rep, _ = rnn_utils.execute_sequence(dec_data, self.e_params)
            dec_rep = self.get_softmax_attention(dec_rep)

        with tf.name_scope("conditional"):
            # add attentional layer here to measure the importance of each timestep.
            # dec_input with shape bs x 3hidden_size
            dec_input = tf.concat([enc_outputs, dec_rep], axis=1)
            if not attention is None:
                dec_input = tf.concat([dec_input, attention], axis=1)
            # dec_hidden_vectors with shape bs x 128 
            dec_hidden_vectors = tf.layers.dense(dec_input, 128, name="conditional_layer", activation=tf.nn.tanh)
            if self.dropout:
                dec_hidden_vectors = tf.nn.dropout(dec_hidden_vectors, 0.5)
        return dec_hidden_vectors
    """

    #perform decoder to produce outputs of the generator
    def exe_decoder(self, dec_hidden_vectors, fn_state=None):
        with tf.variable_scope("decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            dec_inputs_vectors = tf.tile(dec_hidden_vectors, [1, self.decoder_length])
            dec_inputs_vectors = tf.reshape(dec_inputs_vectors, [self.batch_size, self.rnn_hidden_units, self.decoder_length])
            dec_inputs_vectors = tf.transpose(dec_inputs_vectors, [0, 2, 1])
            # dec_inputs_vectors with shape bs x 24 x 256: concatenation of conditional layer vector & uniform random 128D
            dec_inputs_vectors = tf.concat([dec_inputs_vectors, self.z], axis=2)
            dec_outputs = tf.layers.dense(dec_inputs_vectors, 256, name="generation_hidden_seed", activation=tf.nn.tanh)
            dec_outputs = tf.unstack(dec_outputs, axis=1)
            gen_outputs = []
            dtype = 5
            if self.grid_size == 32:
                dtype = 7
            for d in dec_outputs:
                d_ = tf.reshape(d, [self.batch_size, 2, 2, 64])
                out = rnn_utils.get_cnn_rep(d_, dtype, tf.nn.relu, 8, self.use_batch_norm, self.dropout, False)
                gen_outputs.append(out)
            outputs = tf.stack(gen_outputs, axis=1)
            outputs = tf.tanh(tf.layers.flatten(outputs))
            outputs = tf.reshape(outputs, [self.batch_size, self.decoder_length, self.grid_size * self.grid_size])
        return outputs, None

    # just decide whether an image is fake or real
    # calculate the outpute validation of discriminator
    # output is the value of a dense layer w * x + b
    def validate_output(self, inputs, conditional_vectors):
        inputs = tf.reshape(inputs, [self.batch_size * self.decoder_length, self.grid_size, self.grid_size, 1])
        inputs_rep = rnn_utils.get_cnn_rep(inputs, self.gmtype, tf.nn.leaky_relu, 8, self.use_batch_norm, self.dropout, False)
        inputs_rep = tf.layers.flatten(inputs_rep)
        inputs_rep = tf.concat([inputs_rep, conditional_vectors], axis=1)
        # add new hidden layer in the middle
        # help maintain the discriminator
        # output = tf.layers.dense(inputs_rep, 128, name="hidden_validation")
        # if self.dropout:
        #     output = tf.nn.dropout(output, 0.5)
        output = tf.layers.dense(inputs_rep, 1, name="validation_value")
        # if self.dropout:
        #     output = tf.nn.dropout(output, 0.5)
        output = tf.reshape(output, [self.batch_size, self.decoder_length])
        return output, None

    def create_discriminator(self, fake_outputs, conditional_vectors):
        with tf.variable_scope("discriminator", self.initializer, reuse=tf.AUTO_REUSE):
            c_d = conditional_vectors.get_shape()
            conditional_vectors = tf.tile(conditional_vectors, [1, self.decoder_length])
            conditional_vectors = tf.reshape(conditional_vectors, [self.batch_size * self.decoder_length, int(c_d[-1])])
            fake_val, fake_rewards = self.validate_output(fake_outputs, conditional_vectors)
            real_val, _ = self.validate_output(self.pred_placeholder, conditional_vectors)
        return fake_val, real_val, fake_rewards

    # regular discriminator loss function
    def add_discriminator_loss(self, fake_preds, real_preds):
        if self.use_flip:
            real_labels = np.absolute(self.flag - np.random.uniform(0.8, 1., [self.batch_size, self.decoder_length]))
            fake_labels = np.absolute(self.flag - np.random.uniform(0., 0.2, [self.batch_size, self.decoder_length]))
            # real_labels = np.absolute(self.flag - tf.constant(0.9, shape=[self.batch_size, self.decoder_length]))
            # fake_labels = np.absolute(self.flag - tf.constant(0.1, shape=[self.batch_size, self.decoder_length]))
        else:
            # smooth one side (real - side)
            real_labels = tf.constant(0.9, shape=[self.batch_size, self.decoder_length])
            fake_labels = tf.zeros([self.batch_size, self.decoder_length])
        dis_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(fake_labels, fake_preds))
        dis_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(real_labels, real_preds))
        dis_loss = dis_loss_real + dis_loss_fake
        tf.summary.scalar("dis_loss", dis_loss)
        return dis_loss
    
    # operate in each interation of an epoch
    def iterate(self, session, ct, index, train, total_gen_loss, total_dis_loss):
        # just the starting points of encoding batch_size,
        idx = ct[index]
        # switch batchsize, => batchsize * encoding_length (x -> x + 24)
        ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in idx])
        dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in idx])

        feed = {
            self.encoder_inputs: ct_t,
            self.decoder_inputs: dec_t,
            self.z: self.sample_z()
        }
        if self.use_flip:
            feed[self.flag] = np.asarray(np.random.randint(0, 1, [self.batch_size, 1]), dtype=np.float32)
        
        if self.use_attention:
            feed[self.attention_inputs] = np.asarray([range(int(x), int(x) + self.attention_length) for x in idx])

        if not train:
            pred = session.run([self.outputs], feed_dict=feed)
        else:
            gen_loss, dis_loss, pred, _, _= session.run([self.gen_loss, self.dis_loss, self.outputs, self.gen_op, self.dis_op], feed_dict=feed)
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss

        return pred, total_gen_loss, total_dis_loss

    def train_discriminator(self, loss):
        with tf.name_scope("train_discriminator"):
            dis_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            dis_vars = [v for v in tf.trainable_variables() if v.op.name.startswith("discriminator")]
            dis_grads = tf.gradients(loss, dis_vars)
            dis_grads, _ = tf.clip_by_global_norm(dis_grads, 10.)
            dis_train_op = dis_optimizer.apply_gradients(zip(dis_grads, dis_vars))
            return dis_train_op
    
    # policy gradient
    def train_generator(self, loss):
        with tf.name_scope("train_generator"):
            gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            gen_vars = [v for v in tf.trainable_variables() if v.op.name.startswith("generator")]
            # if self.gen_loss_type == 0:
            ## gradient ascent, maximum reward  => ascent the generator loss
            # gen_grads = tf.gradients(-loss, gen_vars)
            # else:
            # using mse without critic
            gen_grads = tf.gradients(loss, gen_vars)
            gen_grads, _ = tf.clip_by_global_norm(gen_grads, 10.)
            gen_train_op = gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
            return gen_train_op
    
    # using stride to reduce the amount of data to loop over time intervals
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, verbose=False, train=False, shuffle=True, stride=4):
        # print("Using stride: %i" % stride)
        st = time.time()
        dt_length = len(data)
        # print("data_size: ", dt_length)
        total_gen_loss = 0.0
        total_dis_loss = 0.0
        preds = []
        ct = np.asarray(data, dtype=np.float32)
        if shuffle:
            r = np.random.permutation(dt_length)
            ct = ct[r]

        # if train:
        #     np.random.shuffle(self.strides)        
        
        if len(self.strides) > 1:
            stride = self.strides[0]
        
        if self.batch_size >= stride:
            cons_b = self.batch_size * stride
        else:
            cons_b = self.batch_size
            stride = 1
        total_steps = dt_length // cons_b
        for step in xrange(total_steps):
            index = range(step * cons_b, (step + 1) * cons_b, stride)
            pred, total_gen_loss, total_dis_loss = self.iterate(session, ct, index, train, total_gen_loss, total_dis_loss)
            if not train:
                preds.append(pred)      
        
        total_gen_loss, total_dis_loss = total_gen_loss / total_steps, total_dis_loss / total_steps
        if train_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag= "Generator Loss", simple_value=total_gen_loss)
            summary.value.add(tag= "Discriminator Loss", simple_value=total_dis_loss)
            train_writer.add_summary(summary, num_epoch)
        dur = time.time() - st
        #print("%.4f" % dur)
        return total_gen_loss, preds
