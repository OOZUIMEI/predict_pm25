from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import properties as pr
from NeuralNet import NeuralNetwork

# reference: deep learning architecture for air quality predictions
# hidden layer = 300
# the number of total layers in stack = 3
# time intervals in the paper 
class StackAutoEncoder(NeuralNetwork):
    
    def __init__(self, pre_train=False, learning_rate=0.01, **kwargs): 
        super(StackAutoEncoder, self).__init__(**kwargs)
        self.pre_train_iter = 10
        self.time_intervals = 8
        self.pre_train = pre_train
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def inference(self):
        with tf.name_scope("embedding"):
            enc = self.process_inputs()

        with tf.name_scope("train_stack"):
            output_ae_0, train_ae_0 = self.add_encoder(enc)
            output_ae_1, train_ae_1 = self.add_encoder(output_ae_0)
            output_ae_2, train_ae_2 = self.add_encoder(output_ae_1)

        self.train_ae_0 = train_ae_0
        self.train_ae_1 = train_ae_1
        self.train_ae_2 = train_ae_2

        with tf.variable_scope("prediction_layer", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            outputs = self.add_single_net(output_ae_2, 25, tf.nn.sigmoid, "prediction_sigmoid")
            outputs = tf.reshape(outputs, shape=(pr.batch_size, 25))
            
        return outputs
    
    # lookup input's vectors from datasets
    def process_inputs(self):
        enc = self.lookup_input()
        # print(enc.get_shape())
        enc = tf.gather(enc, range(self.encoder_length - self.time_intervals, self.encoder_length), axis=1)
        # enc: b x 8 x 25 x H
        enc = tf.transpose(enc, [0, 2, 1, 3])
        enc = tf.layers.flatten(enc, name="encoder_flatten") # B X D
        return enc

    # build stack autoencoder netoworkds
    def add_encoder(self, inputs, layer=0):
        scope_name = "ae_layer_%i" % layer
        with tf.variable_scope(scope_name, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            shape = inputs.get_shape()
            ae_vectors = self.add_single_net(inputs, 300, tf.nn.sigmoid, "ae_en_sigmoid")
            # retrieve hidden size of inputs
            output_ae = self.add_single_net(ae_vectors, shape[-1], tf.nn.sigmoid, "ae_de_sigmoid")
            if self.pre_train:
                ae_loss = tf.losses.mean_squared_error(labels=inputs, predictions=output_ae)
                train_weights = []
                for x in tf.trainable_variables():
                    if x.op.name.startswith(scope_name):
                        train_weights.append(x)
                        if "bias" not in x.name.lower():
                            ae_loss += tf.nn.l2_loss(x)
                grads = self.optimizer.compute_gradients(ae_loss, train_weights)
                train_ae = self.optimizer.apply_gradients(grads)
            else:
                train_ae = None
        return output_ae, train_ae
    
    # operation of each epoch
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=True, train=False, shuffle=True, stride=4):
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
        if train_op is None:
            train_op = tf.no_op()
        elif self.pre_train:
            # pretrain if needed
            # only pretrain with the second half of data
            for pr_i in xrange(self.pre_train_iter):
                for step in xrange(total_steps/2, total_steps):
                    index = range(step * cons_b, (step + 1) * cons_b, stride)
                    # just the starting points of encoding batch_size,
                    ct_ = ct[index]
                    ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_])
                    dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in ct_])

                    feed = {
                        self.encoder_inputs : ct_t,
                        self.decoder_inputs: dec_t
                    }
                    session.run([self.train_ae_0, self.train_ae_1, self.train_ae_2], feed_dict=feed)

        # training with all layers of the model
        for step in xrange(total_steps):
            index = range(step * cons_b, (step + 1) * cons_b, stride)
            # just the starting points of encoding batch_size,
            ct_ = ct[index]
            ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_])
            dec_t = np.asarray([range(int(x) + self.encoder_length, int(x) + self.encoder_length + self.decoder_length) for x in ct_])

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

