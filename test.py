import utils
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import properties as p

from model import Model


def save_predictions(best_preds, best_lb, prefix=""):
    tmp = ""
    for x, y in zip(best_preds, best_lb):
        tmp += "%i|%i\n" % (x, y)
    utils.save_file("%stest_preds.txt" % prefix, tmp, False)


def process_data(dataset, data_len, pred, batch_size, max_sent):
    dlength = len(dataset) - 1
    # total batch
    dlength_b = dlength // batch_size
    maximum = dlength_b * batch_size
    dataset = dataset[:-1]
    data_len = data_len[:-1]
    new_data, new_data_len, new_pred = [], [], []
    for x in xrange(maximum):
        e = x + max_sent
        if e <= dlength:
            arr = dataset[x : e]
            arr_l = data_len[x : e]
            new_data.append(arr)
            new_data_len.append(arr_l)
            new_pred.append(pred[e])
        else:
            break
    r = np.random.permutation(len(new_data_len))
    data = (new_data, new_data_len, new_pred)
    return data



def main(prefix="", url_feature="", url_pred="", url_len="", url_weight="", batch_size=126, max_input_len=30, max_sent_length=24, 
        embed_size=13, acc_range=10):
    utils.assert_url(url_feature)
    if url_pred:
        utils.assert_url(url_pred)
        utils.assert_url(url_len)
        dataset = utils.load_file(url_feature)
        pred = utils.load_file(url_pred)
        data_len = utils.load_file(url_len)
        test = process_data(dataset, data_len, pred, batch_size, max_sent_length)
    else:
        test = utils.load_file(url_feature)
    # init model
    model = Model(max_input_len=max_input_len, max_sent_len=max_sent_length, embed_size=embed_size, 
                 using_bidirection=False, fw_cell="basic", bw_cell="basic", batch_size=batch_size,
                 is_classify=False, use_tanh_prediction=True, target=1, loss="mae", acc_range=acc_range, input_rnn=False)
    
    
    # model.init_data_node()
    tf.reset_default_graph()
    with tf.device('/%s' % p.device):
        model.init_ops()
        saver = tf.train.Saver()
    
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    
    with tf.Session(config=tconfig) as session:
        init = tf.global_variables_initializer()
        session.run(init)
        # saver = tf.train.import_meta_graph(url_weight + ".meta")
        saver.restore(session, url_weight)
        print('==> running model')
        valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, test)
        print('Validation loss: {}'.format(valid_loss))
        print('Validation accuracy: {}'.format(valid_accuracy))
        tmp = 'Test validation accuracy: %.4f' % valid_accuracy
        utils.save_file("%stest_accuracy.txt" % prefix, tmp, False)
        save_predictions(preds, lb, prefix)


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python test.py -f vectors/full_norm_test -fl vectors/full_norm_test_len -pr vectors/labels_test.pkl -wurl "weights/test10_48daegu.weights" -p "test" -sl 48
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_path", help="prefix to save weighted files")
    parser.add_argument("-pr", "--pred_path", help="prefix to save weighted files")
    parser.add_argument("-fl", "--feature_len_path", help="prefix to save weighted files")
    parser.add_argument("-wurl", "--w_url", help="prefix to save weighted files")
    parser.add_argument("-p", "--prefix", help="prefix to save weighted files")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-r", "--acc_range", type=int, default=10)
    parser.add_argument("-e", "--embed_size", type=int, default=11)
    parser.add_argument("-il", "--input_size", type=int, default=30)
    parser.add_argument("-sl", "--sent_size", type=int, default=24)

    args = parser.parse_args()

    main(args.prefix, args.feature_path, args.pred_path, args.feature_len_path, args.w_url, args.batch_size, 
        args.input_size, args.sent_size, args.embed_size, args.acc_range)
    