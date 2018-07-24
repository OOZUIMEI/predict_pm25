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

from model import Model


def main(prefix="", url_feature="", url_pred="", url_len="",  url_feature1="", url_pred1="", url_len1="", 
        batch_size=126, max_input_len=30, max_sent_length=24, lr_decayable=False, using_bidirection=False, 
        forward_cell='', backward_cell='', embed_size=None, is_classify=True, loss=None, acc_range=None, 
        usp=None, input_rnn=None, reload_data=True, pred_sight=1, decoder=1, decoder_size=4, is_weighted=0, 
        context_info=1, rnn_layer=1):
    target = 5 if is_classify else 1
    model = Model(max_input_len=max_input_len, max_sent_len=max_sent_length, embed_size=embed_size, learning_rate = 0.001, lr_decayable=lr_decayable, 
                 using_bidirection=using_bidirection, fw_cell=forward_cell, bw_cell=backward_cell, batch_size=batch_size,
                 target=target, is_classify=is_classify, loss=loss, acc_range=acc_range, use_tanh_prediction=usp, input_rnn=input_rnn, 
                 sight=pred_sight, dvs=decoder_size, use_decoder=decoder, is_weighted=is_weighted, rnn_layer=rnn_layer)
    # model.init_data_node()
    with tf.device('/%s' % p.device):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    if reload_data:
        print("Loading dataset")
        utils.assert_url(url_feature)
        utils.assert_url(url_pred)
        if max_input_len > 1:
            utils.assert_url(url_len)
            data_len = utils.load_file(url_len)
        else:
            data_len = None
        # if url_feature1:
        #     utils.assert_url(url_feature1)
        #     utils.assert_url(url_pred1)
        #     if max_input_len > 1:
        #         utils.assert_url(url_len1)
        dataset = utils.load_file(url_feature)
        pred = utils.load_file(url_pred, False)
        if is_classify:
            pred = [utils.get_pm25_class(round(float(x.replace("\n", "")))) for x in pred]
        else:
            pred = [round(float(x.replace("\n", ""))) for x in pred]
        train, dev = utils.process_data(dataset, data_len, pred, batch_size, max_input_len, max_sent_length, False, pred_sight, context_info)
        # utils.save_file(p.train_url % ("_" + prefix + "_" + str(max_sent_length)), train)
        # utils.save_file(p.dev_url % ("_" + prefix + "_" +str(max_sent_length)), dev)
    else:
        utils.assert_url(p.train_url)
        utils.assert_url(p.dev_url)
        train = utils.load_file(p.train_url)
        dev = utils.load_file(p.dev_url)
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

        if url_pred1:
            print('==> restoring weights')
            saver.restore(session, '%s' % url_pred1)
        
        print('==> starting training')
        train_losses, train_accuracies = [], []
        val_losses, val_acces, best_preds, best_lb = [], [], [], []
        for epoch in xrange(p.total_iteration):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy, _, _ = model.run_epoch(
                session, model.train, epoch, train_writer,
                train_op=model.train_step, train=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            print('Training loss: {}'.format(train_loss))
            # print('Training accuracy: {}'.format(train_accuracy))
       
            valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, model.valid)
            val_losses.append(valid_loss)
            val_acces.append(valid_accuracy)
            print('Validation loss: {}'.format(valid_loss))
            # print('Validation accuracy: {}'.format(valid_accuracy))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%sdaegu.weights' % prefix)
                    best_preds = preds
                    best_lb = lb
                    # utils.save_predictions(best_preds, best_lb, p.train_preds % prefix)
                    if best_val_accuracy < valid_accuracy:
                        best_val_accuracy = valid_accuracy

            if (epoch - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        # tmp = 'Best validation accuracy: %.4f' % best_val_accuracy
        # print(tmp)
        # utils.save_file("accuracies/%saccuracy.txt" % prefix, tmp, False)
        utils.save_file("logs/%slosses.pkl" % prefix, {"train_loss": train_losses, "train_acc": train_accuracies, "valid_loss": val_losses, "valid_acc" : val_acces})
        # utils.save_predictions(best_preds, best_lb, p.train_preds % prefix) 
        # if url_feature1:
        #     dataset_t = utils.load_file(url_feature1)
        #     data_len_t = utils.load_file(url_len1)
        #     pred_t = utils.load_file(url_pred1)
        #     _, test = process_data(dataset_t, data_len_t, pred_t, batch_size, max_sent_length)
        #     valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, test)
        #     print("Test", valid_loss, valid_accuracy)
        #     utils.save_predictions(preds, lb, p.train_preds % (prefix + "_test_"))


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python train.py -pr "vectors/labels" -f "vectors/full_data" -fl "vectors/full_data_len" -p "train_basic_64b_tanh_12h_" -fw "basic" -dc 1 -l mae -r 10 -usp 1 -e 13 -bs 126 -sl 24 -ir 0 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_path", help="prefix to save weighted files")
    parser.add_argument("-pr", "--pred_path", help="prefix to save weighted files")
    parser.add_argument("-fl", "--feature_len_path", help="prefix to save weighted files")
    parser.add_argument("-f1", "--feature_path1", help="prefix to save weighted files")
    parser.add_argument("-pr1", "--pred_path1", help="prefix to save weighted files")
    parser.add_argument("-fl1", "--feature_len_path1", help="prefix to save weighted files")
    parser.add_argument("-p", "--prefix", help="prefix to save weighted files")
    parser.add_argument("-fw", "--forward_cell", default='basic')
    parser.add_argument("-bw", "--backward_cell", default='basic')
    parser.add_argument("-bd", "--bidirection", type=int)
    parser.add_argument("-dc", "--lr_decayable", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-c", "--classify", type=int, default=0)
    parser.add_argument("-l", "--loss", default='mae')
    parser.add_argument("-r", "--acc_range", type=int, default=10)
    parser.add_argument("-usp", "--use_tanh_pred", type=int, default=1)
    parser.add_argument("-e", "--embed_size", type=int, default=12)
    parser.add_argument("-il", "--input_size", type=int, default=1)
    parser.add_argument("-sl", "--sent_size", type=int, default=24)
    parser.add_argument("-ir", "--input_rnn", type=int, default=0)
    parser.add_argument("-rl", "--reload_data", type=int, default=1)
    parser.add_argument("-s", "--pred_sight", type=int, default=1)
    parser.add_argument("-d", "--decoder", type=int, default=1)
    parser.add_argument("-ds", "--decoder_size", type=int, default=4)
    parser.add_argument("-w", "--weighted", help="add weighted ratio to loss value", type=int, default=0)
    parser.add_argument("-ci", "--context_info", type=int, default=1)
    parser.add_argument("-rn", "--rnn_layer", type=int, default=1)

    args = parser.parse_args()

    main(args.prefix, args.feature_path, args.pred_path, args.feature_len_path, args.feature_path1, args.pred_path1, args.feature_len_path1,
        args.batch_size, args.input_size, args.sent_size, args.lr_decayable, 
        args.bidirection, args.forward_cell, args.backward_cell, args.embed_size, args.classify, 
        args.loss, args.acc_range, args.use_tanh_pred, args.input_rnn, args.reload_data, args.pred_sight, args.decoder, args.decoder_size, 
        args.weighted, args.context_info, args.rnn_layer)
    
