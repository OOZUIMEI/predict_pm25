"""
index of N/A in one_hot feature 0,41,73,112,129,146
39 in case of only seoul & not integrate china
"""

import utils
import tensorflow as tf
import numpy as np
import os
import sys
import time
from datetime import datetime
import argparse
import properties as p
import heatmap

from baseline_cnnlstm import BaselineModel


def convert_element_to_grid(self, context):
    res = []
    for b in context:
        res_t = []
        for t in b:
            p = heatmap.fill_map(t, self.map, False) 
            res_t.append(p)
        res.append(res_t)
    return np.asarray(res, dtype=np.float)


# pre-process data for training 
# convert 1-d data to grid-data
def process_data(dtlength, batch_size, encoder_length, decoder_length):
    # ma = heatmap.build_map()
    maximum = (dtlength - encoder_length - decoder_length) // batch_size * batch_size
    end = maximum + encoder_length + decoder_length
    # random from 0 -> maximum_index to separate valid & train set
    train_length = int((maximum * 0.8) // batch_size * batch_size)
    r = np.random.permutation(maximum)
    indices = np.asarray(range(maximum), dtype=np.int32)[r]
    train = indices[:train_length]
    valid = indices[train_length:]
    return train, valid, end



def main(url_feature="", batch_size=126, encoder_length=24, embed_size=None, loss=None, decoder_length=24, decoder_size=4, grid_size=25, weight_prefix="sp", dtype=""):
    model = BaselineModel(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, dtype="grid", grid_size=grid_size)
    print("==> Loading dataset")
    utils.assert_url(url_feature)
    dataset = utils.load_file(url_feature)
    if not dataset:
        raise ValueError("Not supported data")
    dataset = np.asarray(dataset, dtype=np.float32)
    train, valid, end = process_data(len(dataset), batch_size, encoder_length, decoder_length)
    model.set_data(dataset[:end], train, valid)
    
    print('==> initializing models')
    with tf.device('/%s' % p.device):
        model.init_ops()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    gpu_options = None
    if p.device == "gpu":
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p.gpu_fraction, allow_growth=True)
       
    tconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=tconfig) as session:
        
        sum_dir = 'summaries/' + time.strftime("%Y-%m-%d %H %M")
        if not utils.check_file(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        best_val_loss = float('inf')
        best_overall_val_loss = float('inf')

        print('==> starting training')
        train_losses = []
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss = model.run_epoch(
                session, train, epoch, train_writer,
                train_op=model.train_op, train=True)
            train_losses.append(train_loss)
            print('Training loss: {}'.format(train_loss))
       
            valid_loss = model.run_epoch(session, valid)
            print('Validation loss: {}'.format(valid_loss))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%s.weights' % weight_prefix)

            if (epoch - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        tm = utils.clear_datetime(datetime.strftime(utils.get_datetime_now(), "%Y-%m-%d %H:%M:%S"))
        l_fl = "train_loss_%s_%s" % (weight_prefix, tm)
        utils.save_file(tm, train_losses)
            


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python train.py -pr "vectors/labels" -f "vectors/full_data" -fl "vectors/full_data_len" -p "train_basic_64b_tanh_12h_" -fw "basic" -dc 1 -l mae -r 10 -usp 1 -e 13 -bs 126 -sl 24 -ir 0 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature", help="prefix to save weighted files")
    parser.add_argument("-p", "--weight_prefix", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", default='mae')
    parser.add_argument("-e", "--embed_size", type=int, default=12)
    parser.add_argument("-el", "--encoder_length", type=int, default=24)
    parser.add_argument("-dl", "--decoder_length", type=int, default=24)
    parser.add_argument("-ds", "--decoder_size", type=int, default=6)
    parser.add_argument("-g", "--grid_size", type=int, default=25)
    parser.add_argument("-dt", "--dtype", default='grid')

    args = parser.parse_args()

    main(args.feature, args.batch_size, args.encoder_length, args.embed_size, args.loss, args.decoder_length, args.decoder_size, dtype=args.dtype)
    