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
    utils.save_file("%spreds.txt" % prefix, tmp, False)


def main(prefix="", url_feature="", url_pred="", batch_size=81, lr_decayable=False, using_bidirection=False, 
        forward_cell='', backward_cell='', target=5, is_classify=True, loss=None, acc_range=None, usp=None):
    if utils.check_file(url_feature):
        print("Loading dataset")
        dataset = utils.load_file(url_feature)
    else:
        raise ValueError("%s is not existed" % url_feature)
    if utils.check_file(url_pred):
        pred = utils.load_file(url_pred)
    else:
        raise ValueError("%s is not existed" % url_pred)
        # config.strong_supervision = True
    model = Model(max_input_len=30, embed_size=12, learning_rate = 0.001, lr_decayable=lr_decayable, 
                 using_bidirection=using_bidirection, fw_cell=forward_cell, bw_cell=backward_cell, 
                 target=target, is_classify=is_classify, loss=loss, acc_range=acc_range, use_tanh_prediction=usp)
    
    dlength = len(dataset)
    dlength_b = dlength // batch_size
    dlength = dlength_b * batch_size
    dataset = dataset[:dlength]
    pred = pred[:dlength]
    train_len = int(dlength_b * 0.8) * batch_size
    train_data = dataset[:train_len]
    train_pred = pred[:train_len]
    valid_data = dataset[train_len:]
    valid_pred = pred[train_len:]
    train = (train_data, train_pred)
    dev = (valid_data, valid_pred)
    model.set_data(train, dev)
    # model.init_data_node()
    with tf.device('/cpu'):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = tf.ConfigProto(allow_soft_placement=True)

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

        # if restore:
        #     print('==> restoring weights')
        #     saver.restore(session, 'weights/%sdaegu.weights' % prefix)
        
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
            print('Training accuracy: {}'.format(train_accuracy))
       
            valid_loss, valid_accuracy, preds, lb = model.run_epoch(session, model.valid)
            val_losses.append(valid_loss)
            val_acces.append(valid_accuracy)
            print('Validation loss: {}'.format(valid_loss))
            print('Validation accuracy: {}'.format(valid_accuracy))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%sdaegu.weights' % prefix)
                    best_preds = preds
                    best_lb = lb
                    save_predictions(best_preds, best_lb, prefix)
                    if best_val_accuracy < valid_accuracy:
                        best_val_accuracy = valid_accuracy

            if (epoch - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        tmp = 'Best validation accuracy: %.4f' % best_val_accuracy
        print(tmp)
        utils.save_file("%saccuracy.txt" % prefix, tmp, False)
        utils.save_file("logs/%slosses.pkl" % prefix, {"train_loss": train_losses, "train_acc": train_accuracies, "valid_loss": val_losses, "valid_acc" : val_acces})
        save_predictions(best_preds, best_lb, prefix) 


if __name__ == "__main__":
    # python train_sentiment.py -dc 1 -b weights/8x8_code_book.pkl -w weights/8x8_code_words_training.txt -bs 8 -ws 8 -p '8x8_'
    # python train_sentiment.py -dc 1 -p 'aaa' -bd 1 -fw 'basic' -bw 'basic'
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_path", help="prefix to save weighted files")
    parser.add_argument("-pr", "--pred_path", help="prefix to save weighted files")
    parser.add_argument("-p", "--prefix", help="prefix to save weighted files")
    parser.add_argument("-fw", "--forward_cell", default='basic')
    parser.add_argument("-bw", "--backward_cell", default='basic')
    parser.add_argument("-bd", "--bidirection", type=int)
    parser.add_argument("-dc", "--lr_decayable", type=int)
    parser.add_argument("-bs", "--batch_size", type=int, default=54)
    parser.add_argument("-t", "--target", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=1)
    parser.add_argument("-l", "--loss", default='softmax')
    parser.add_argument("-r", "--acc_range", type=int, default=10)
    parser.add_argument("-usp", "--use_tanh_pred", type=int, default=1)

    args = parser.parse_args()

    main(args.prefix, args.feature_path, args.pred_path, args.batch_size, args.lr_decayable, 
        args.bidirection, args.forward_cell, args.backward_cell, args.target, args.classify, 
        args.loss, args.acc_range, args.use_tanh_pred)
    