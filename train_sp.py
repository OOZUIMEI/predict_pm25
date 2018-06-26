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
import argparse
import properties as p

from baseline_cnnlstm import BaselineModel


def process_data(dataset, batch_size, encoder_length, decoder_length, fr_ele=6):
    len_dataset = len(dataset)
    dlength = len_dataset - 1
    # total batch
    dlength_b = len_dataset // batch_size
    maximum = dlength_b * batch_size
    # dataset = dataset[:-1]
    new_data = []
    new_pred = []
    decode_vec = []
    dat = np.asarray(dataset)
    # convert district data to map
    
    for x in xrange(maximum):
        # encoding starting
        e = x + encoder_length
        # decoding starting index
        d_e = e + decoder_length
        if e <= dlength and d_e <= len_dataset:
            e_arr = dat[x : e]
            # extract predict value & decoding context vector
            d_arr = dat[e:d_e,:,fr_ele:]
            # append vector to matrix
            pred = dat[e:d_e,:,0]
            new_data.append(e_arr)
            new_pred.append(pred)
            decode_vec.append(d_arr)
        else:
            break
    new_data, new_pred, decode_vec = np.asarray(new_data, dtype=np.float32), np.asarray(new_pred, dtype=np.int32), np.asarray(decode_vec, dtype=np.float32)
    train_len = int(dlength_b * 0.8) * batch_size
    # permutation to balance distribution
    r = np.random.permutation(dlength_b)
    new_data, new_pred, decode_vec = new_data[r], new_pred[r], decode_vec[r]
    # training set = 80% of dataset
    train_data = new_data[:train_len]
    train_pred = new_pred[:train_len]
    train_dec = decode_vec[:train_len]
    # generate validate set 20% of dataset
    valid_data = new_data[train_len:]
    valid_pred = new_pred[train_len:]
    valid_dec = decode_vec[train_len:]        

    train = (train_data, train_pred, train_dec)
    dev = (valid_data, valid_pred, valid_dec)
    return train, dev



def main(url_feature="", batch_size=126, encoder_length=24, embed_size=None, loss=None, decoder_length=24, decoder_size=4):
    model = BaselineModel(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size)
    with tf.device('/%s' % p.device):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    print("Loading dataset")
    utils.assert_url(url_feature)
    dataset = utils.load_file(url_feature)
    train, dev = process_data(dataset, batch_size, encoder_length, decoder_length)
    model.set_data(train, dev)
    
    gpu_options = None
    if p.device == "gpu":
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p.gpu_fraction)
       
    tconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=tconfig) as session:

        sum_dir = 'summaries/' + time.strftime("%Y-%m-%d %H %M")
        if not utils.check_file(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_overall_val_loss = float('inf')

        print('==> starting training')
        train_losses, train_accuracies = [], []
        val_losses, val_acces, best_preds, best_lb = [], [], [], []
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy, _, _ = model.run_epoch(
                session, model.train, epoch, train_writer,
                train_op=model.train, train=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            print('Training loss: {}'.format(train_loss))
       
            valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, model.valid)
            val_losses.append(valid_loss)
            val_acces.append(valid_accuracy)
            print('Validation loss: {}'.format(valid_loss))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%sdaegu.weights' % prefix)
                    best_preds = preds
                    best_lb = lb
                    if best_val_accuracy < valid_accuracy:
                        best_val_accuracy = valid_accuracy

            if (epoch - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python train.py -pr "vectors/labels" -f "vectors/full_data" -fl "vectors/full_data_len" -p "train_basic_64b_tanh_12h_" -fw "basic" -dc 1 -l mae -r 10 -usp 1 -e 13 -bs 126 -sl 24 -ir 0 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature", help="prefix to save weighted files")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", default='mae')
    parser.add_argument("-e", "--embed_size", type=int, default=12)
    parser.add_argument("-el", "--encoder_length", type=int, default=24)
    parser.add_argument("-dl", "--decoder_length", type=int, default=24)
    parser.add_argument("-ds", "--decoder_size", type=int, default=6)

    args = parser.parse_args()

    main(args.feature, args.batch_size, args.encoder_length, args.embed_size, args.loss, args.decoder_length, args.decoder_size)
    