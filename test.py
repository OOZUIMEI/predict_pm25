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
    utils.save_file("test_acc/%s_test_preds.txt" % prefix, tmp, False)


def process_data(dataset, data_len, pred, batch_size, max_sent, sight=1):
    dlength = len(dataset) - 1
    # total batch
    dlength_b = dlength // batch_size
    maximum = dlength_b * batch_size
    dataset = dataset[:-1]
    data_len = data_len[:-1]
    new_data, new_data_len, new_pred = [], [], []
    decode_vec = []
    for x in xrange(maximum):
        e = x + max_sent
        p_i = e + sight - 1
        d_e = p_i + 1
        if e <= dlength and d_e <= dlength:
            arr = dataset[x : e]
            arr_l = data_len[x : e]
            arr_d =[x[0][8:12] for x in dataset[e : d_e]] 
            new_data.append(arr)
            new_data_len.append(arr_l)
            new_pred.append(pred[p_i])
            decode_vec.append(arr_d)
        else:
            break
    data = (new_data, new_data_len, new_pred, decode_vec)
    return data



def main(prefix="", url_feature="", url_pred="", url_len="", url_weight="", batch_size=126, max_input_len=30, max_sent_length=24, 
        embed_size=13, acc_range=10, sight=1, is_classify=0, decoder=1, decoder_size=4):
    # init model
    model = Model(max_input_len=max_input_len, max_sent_len=max_sent_length, embed_size=embed_size, 
                 using_bidirection=False, fw_cell="basic", bw_cell="basic", batch_size=batch_size,
                 is_classify=is_classify, use_tanh_prediction=True, target=5 if is_classify else 1, 
                 loss="softmax" if is_classify else "mae", acc_range=acc_range, input_rnn=False, sight=sight, 
                 use_decoder=decoder, dvs=decoder_size)
    
    # model.init_data_node()
    tf.reset_default_graph()
    with tf.device('/%s' % p.device):
        model.init_ops()
        saver = tf.train.Saver()
    
    utils.assert_url(url_feature)
    if url_pred:
        utils.assert_url(url_pred)
        utils.assert_url(url_len)
        dataset = utils.load_file(url_feature)
        pred = utils.load_file(url_pred, False)
        if is_classify:
            pred = [utils.get_pm25_class(round(float(x.replace("\n", "")))) for x in pred]
        else:
            pred = [round(float(x.replace("\n", ""))) for x in pred]
        data_len = utils.load_file(url_len)
        test = process_data(dataset, data_len, pred, batch_size, max_sent_length, sight)
    else:
        test = utils.load_file(url_feature)
        
    tconfig = tf.ConfigProto(allow_soft_placement=True)
    
    with tf.Session(config=tconfig) as session:
        init = tf.global_variables_initializer()
        session.run(init)
        # saver = tf.train.import_meta_graph(url_weight + ".meta")
        saver.restore(session, url_weight)
        print('==> running model')
        valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, test,  shuffle=False)
        print('Validation loss: {}'.format(valid_loss))
        print('Validation accuracy: {}'.format(valid_accuracy))
        tmp = 'Test validation accuracy: %.4f' % valid_accuracy
        # utils.save_file("test_acc/%s_test_acc.txt" % prefix, tmp, False)
        utils.save_predictions(preds, lb,  p.test_preds % prefix)


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
    parser.add_argument("-s", "--sight", type=int, default=1)
    parser.add_argument("-c", "--classify", type=int, default=0)
    parser.add_argument("-d", "--decoder", type=int, default=1)
    parser.add_argument("-ds", "--decoder_size", type=int, default=4)

    args = parser.parse_args()

    main(args.prefix, args.feature_path, args.pred_path, args.feature_len_path, args.w_url, args.batch_size, 
        args.input_size, args.sent_size, args.embed_size, args.acc_range, args.sight, args.classify, args.decoder, args.decoder_size)
    