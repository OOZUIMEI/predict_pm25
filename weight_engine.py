import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell 
from tensorflow.contrib.cudnn_rnn import CudnnLSTM


with tf.device("gpu"):
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph("./weights/seoul_china_pre.weights.meta", clear_devices=True)
    with tf.Session() as sess:
        imported_meta.restore(sess, "./weights/seoul_china_pre.weights")
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        print(var)