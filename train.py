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


def process_data(dataset, data_len, pred, batch_size, max_sent):
    dlength = len(dataset) - 1
    # total batch
    dlength_b = dlength // batch_size
    dlength = dlength_b * batch_size
    dataset = dataset[:dlength]
    data_len = data_len[:dlength]
    new_data = []
    new_data_len = []
    new_pred = []
    for x in xrange(dlength):
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
    new_data, new_data_len, new_pred = np.asarray(new_data, dtype=np.float32), np.asarray(new_data_len, dtype=np.int32), np.asarray(new_pred, dtype=np.int32)
    new_data, new_data_len, new_pred = new_data[r], new_data_len[r], new_pred[r]
    
    train_len = int(dlength_b * 0.8) * batch_size
    train_data = new_data[:train_len]
    train_data_len = new_data_len[:train_len]
    train_pred = new_pred[:train_len]
    valid_data = new_data[train_len:]
    valid_data_len = new_data_len[train_len:]
    valid_pred = new_pred[train_len:]
    train = (train_data, train_data_len, train_pred)
    dev = (valid_data, valid_data_len, valid_pred)
    return train, dev



def main(prefix="", url_feature="", url_pred="", url_len="", batch_size=126, max_input_len=30, max_sent_length=24, lr_decayable=False, using_bidirection=False, 
        forward_cell='', backward_cell='', embed_size=None, target=5, is_classify=True, loss=None, acc_range=None, usp=None):
    if utils.check_file(url_feature):
        print("Loading dataset")
        dataset = utils.load_file(url_feature)
    else:
        raise ValueError("%s is not existed" % url_feature)
    if utils.check_file(url_pred):
        pred = utils.load_file(url_pred)
    else:
        raise ValueError("%s is not existed" % url_pred)
    if utils.check_file(url_len):
        data_len = utils.load_file(url_len)
    else:
        raise ValueError("%s is not existed" % url_len)
        # config.strong_supervision = True
    model = Model(max_input_len=max_input_len, max_sent_len=max_sent_length, embed_size=embed_size, learning_rate = 0.001, lr_decayable=lr_decayable, 
                 using_bidirection=using_bidirection, fw_cell=forward_cell, bw_cell=backward_cell, batch_size=batch_size,
                 target=target, is_classify=is_classify, loss=loss, acc_range=acc_range, use_tanh_prediction=usp)
    
    train, dev = process_data(dataset, data_len, pred, batch_size, max_sent_length)
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
    # 10110 -> 10080 -> 126 batch 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_path", help="prefix to save weighted files")
    parser.add_argument("-pr", "--pred_path", help="prefix to save weighted files")
    parser.add_argument("-fl", "--feature_len_path", help="prefix to save weighted files")
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
    parser.add_argument("-e", "--embed_size", type=int, default=12)
    parser.add_argument("-il", "--input_size", type=int, default=30)
    parser.add_argument("-sl", "--sent_size", type=int, default=12)

    args = parser.parse_args()

    main(args.prefix, args.feature_path, args.pred_path, args.feature_len_path, args.batch_size, args.input_size, args.sent_size, args.lr_decayable, 
        args.bidirection, args.forward_cell, args.backward_cell, args.embed_size, args.target, args.classify, 
        args.loss, args.acc_range, args.use_tanh_pred)
    