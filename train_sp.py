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
import re
from datetime import datetime, timedelta
import argparse
import properties as p
import heatmap
import craw_seoul_aqi as aqi
import craw_aws as aws

import process_sp_vector as psv
from baseline_cnnlstm import BaselineModel
from mask_gan import MaskGan
# import matplotlib
# import matplotlib.pyplot as plt
from  spark_engine import SparkEngine
import district_neighbors as dd


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
def process_data(dtlength, batch_size, encoder_length, decoder_length=None, is_test=False):
    # ma = heatmap.build_map()
    # maximum = (dtlength - encoder_length - decoder_length) // batch_size * batch_size
    maximum = dtlength - encoder_length - decoder_length
    # end = maximum + encoder_length + decoder_length
    # random from 0 -> maximum_index to separate valid & train set
    indices = np.asarray(range(maximum), dtype=np.int32)
    if not is_test:
        train_length = int((maximum * 0.8) // batch_size * batch_size)
        r = np.random.permutation(maximum)
        indices = indices[r]
        train = indices[:train_length]
        valid = indices[train_length:]
    else:
        train, valid = indices, None
    return train, valid


def execute(path, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer=None, offset=0):
    print("==> Loading dataset")
    dataset = utils.load_file(path)
    last_epoch = 0
    if dataset:
        dataset = np.asarray(dataset, dtype=np.float32)
        lt = len(dataset)
        train, valid = process_data(lt, batch_size, encoder_length, decoder_length, is_test)
        if attention_url:
            attention_data = utils.load_file(attention_url)
        else:
            attention_data = None
        
        model.set_data(dataset, train, valid, attention_data)
        model.assign_datasets(session)
        if not is_test:
            best_val_epoch = 0
            best_val_loss = float('inf')
            # best_overall_val_loss = float('inf')

            print('==> starting training')
            train_losses = []
            train_f, valid_f = train_writer
            for epoch in xrange(p.total_iteration):
                print('Epoch {}'.format(epoch))
                start = time.time()

                train_loss, _ = model.run_epoch(session, train, offset + epoch, train_f,train_op=model.train_op, train=True)
                train_losses.append(train_loss)
                print('Training loss: {}'.format(train_loss))

                valid_loss, _ = model.run_epoch(session, valid, offset + epoch, train_writer=valid_f)
                print('Validation loss: {}'.format(valid_loss))

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    # if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    # best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%s.weights' % url_weight)

                if (epoch - best_val_epoch) > p.early_stopping:
                    last_epoch += epoch + 1
                    break
                print('Total time: {}'.format(time.time() - start))
            tm = utils.clear_datetime(datetime.strftime(utils.get_datetime_now(), "%Y-%m-%d %H:%M:%S"))
            l_fl = "train_loss/train_loss_%s_%s" % (url_weight, tm)
            utils.save_file(l_fl, train_losses)
        else:
            saver.restore(session, url_weight)
            print('==> running model')
            loss, preds = model.run_epoch(session, model.train, shuffle=False)
            l_str = 'Test mae loss: %.4f' % loss
            print(l_str)
            pt = re.compile("weights/([A-Za-z0-9_.]*).weights")
            name = pt.match(url_weight)
            name_s = name.group(1)
            utils.save_file("test_sp/%s_loss.txt" % name_s, l_str, use_pickle=False)
            utils.save_file("test_sp/%s" % name_s, preds)
    return last_epoch


def get_gpu_options():
    gpu_options = None
    if "gpu" in p.device:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p.gpu_fraction)
        os.environ["CUDA_VISIBLE_DEVICES"]=p.gpu_devices
    configs = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return configs


def main(url_feature="", attention_url="", url_weight="sp", batch_size=128, encoder_length=24, embed_size=None, loss=None, decoder_length=24, decoder_size=4, grid_size=25, rnn_layers=1,
        dtype="grid", is_folder=False, is_test=False, use_cnn=True, restore=False):
    model = BaselineModel(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, rnn_layers=rnn_layers,
                        dtype=dtype, grid_size=grid_size, use_cnn=use_cnn)
    print('==> initializing models')
    with tf.device('/%s' % p.device):
        model.init_ops()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    utils.assert_url(url_feature)

    tconfig = get_gpu_options()
    sum_dir = 'summaries'
    if not utils.check_file(sum_dir):
        os.makedirs(sum_dir)

    train_writer = None
    valid_writer = None
    with tf.Session(config=tconfig) as session:
        if not restore:
            session.run(init)
        else:
            print("==> Reload pre-trained weights")
            saver.restore(session, url_weight)
            url_weight = url_weight.split("/")[-1]
            url_weight = url_weight.rstrip(".weights")
        
        if not is_test:
            suf = time.strftime("%Y.%m.%d_%H.%M")
            train_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_train", session.graph, filename_suffix=suf)
            valid_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_valid", session.graph, filename_suffix=suf)

        folders = None
        
        if is_folder:
            folders = os.listdir(url_feature)
            if attention_url:
                a_folders = os.listdir(attention_url)
                folders = zip(folders, a_folders)
            last_epoch = 0
            for i, files in enumerate(folders):
                if attention_url:
                    x, y = files
                    att_url = os.path.join(attention_url, y)
                    print("==> Training set (%i, %s, %s)" % (i + 1, x, y))
                else: 
                    x = files
                    print("==> Training set (%i, %s)" % (i + 1, x))
                last_epoch = execute(os.path.join(url_feature, x), att_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, (train_writer, valid_writer), last_epoch)
        else:
            execute(url_feature, att_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, (train_writer, valid_writer))


def execute_gan(path, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer=None, offset=0):
    print("==> Loading dataset")
    dataset = utils.load_file(path)
    if dataset:
        dataset = np.asarray(dataset, dtype=np.float32)
        lt = len(dataset)
        train, _ = process_data(lt, batch_size, encoder_length, decoder_length, True)
        # in gan, we don't need to validate
        # load attention data
        if attention_url:
            attention_data = utils.load_file(attention_url)
        else:
            attention_data = None
        
        model.set_data(dataset, train, None, attention_data)
        model.assign_datasets(session)

        if not is_test:
            print('==> starting training')
            train_f = train_writer
            for epoch in xrange(p.total_iteration):
                _ = model.run_epoch(session, train, offset + epoch, train_f, train=True, verbose=False)
                if epoch % 10 == 0:
                    utils.update_progress(epoch * 1.0 / p.total_iteration)
                    saver.save(session, 'weights/%s.weights' % url_weight)
            saver.save(session, 'weights/%s.weights' % url_weight)
        else:
            saver.restore(session, url_weight)
            print('==> running model')
            preds = model.run_epoch(session, train, train=False, verbose=False, shuffle=False)
            # l_str = 'Test mae loss: %.4f' % loss
            pt = re.compile("weights/([A-Za-z0-9_.]*).weights")
            name = pt.match(url_weight)
            name_s = name.group(1)
            utils.save_file("test_sp/%s" % name_s, preds)


def train_gan(url_feature="", attention_url="", url_weight="sp", batch_size=128, encoder_length=24, embed_size=None, loss=None, decoder_length=24, decoder_size=4, grid_size=25, rnn_layers=1, 
            dtype="grid", is_folder=False, is_test=False, restore=False):
    model = MaskGan(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, rnn_layers=rnn_layers, grid_size=grid_size, use_cnn=1)
    print('==> initializing models')
    with tf.device('/%s' % p.device):
        model.init_ops()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    utils.assert_url(url_feature)

    tconfig = get_gpu_options()
    sum_dir = 'summaries'
    if not utils.check_file(sum_dir):
        os.makedirs(sum_dir)
    
    train_writer = None
    
    with tf.Session(config=tconfig) as session:       
        if not restore:
            session.run(init)
        else:
            print("==> Reload pre-trained weights")
            saver.restore(session, url_weight)
            csn = int(time.time())

        if not is_test:
            url_weight = url_weight.split("/")[-1]
            url_weight = url_weight.rstrip(".weights")
            suf = time.strftime("%Y.%m.%d_%H.%M")
            train_writer = tf.summary.FileWriter("%s/%s_%i" % (sum_dir, url_weight, csn), session.graph, filename_suffix=suf)

        folders = None
        if is_folder:
            folders = os.listdir(url_feature)
            if attention_url:
                a_folders = os.listdir(attention_url)
                folders = zip(folders, a_folders)
            for i, files in enumerate(folders):
                if attention_url:
                    x, y = files
                    att_url = os.path.join(attention_url, y)
                    print("==> Training set (%i, %s, %s)" % (i + 1, x, y))
                else: 
                    x = files
                    print("==> Training set (%i, %s)" % (i + 1, x))
                execute_gan(os.path.join(url_feature, x), att_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer, i * p.total_iteration)
        else:
            execute_gan(url_feature, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer)


def get_prediction_real_time(sparkEngine, url_weight="", dim=12):
    # continuously crawl aws and aqi & weather
    encoder_length = 24
    decoder_length = 24
    end = utils.get_datetime_now()
    # end = datetime.strptime("2018-06-19 11:01:00", p.fm)
    # e_ = end.strftime(p.fm)
    start = end - timedelta(days=1)
    start = start.replace(minute=0, second=0, microsecond=0)
    # s_ = start.strftime(p.fm)
    # 2. process normalize data
    vectors, w_pred, timestamp = sparkEngine.process_vectors(start, end, dim)
    v_l = len(vectors)
    if v_l:
        sp_vectors = psv.convert_data_to_grid_exe(vectors)
        if v_l < encoder_length:
            sp_vectors = np.pad(sp_vectors, ((encoder_length - v_l,0), (0,0), (0,0), (0, 0)), 'constant', constant_values=0)
        
        # repeat for 25 districts
        if w_pred:
            w_pred = np.repeat(np.expand_dims(w_pred, 1), 25, 1)
            de_vectors = psv.convert_data_to_grid_exe(w_pred)
            # pad to fill top elements of decoder vectors
            de_vectors = np.pad(de_vectors, ((0, 0), (0, 0), (0, 0), (6, 0)), 'constant', constant_values=0)
        else:
            # know nothing about future weather forecast
            de_vectors = np.zeros((decoder_length, 25, 25, dim))
        sp_vectors = np.concatenate((sp_vectors, de_vectors), axis=0)
        # 4. Feed to model
        model = BaselineModel(encoder_length=encoder_length, encode_vector_size=12, batch_size=1, decoder_length=decoder_length, rnn_layers=1,
                        dtype='grid', grid_size=25, use_cnn=True)
        model.set_data(sp_vectors, [0], None)
        
        with tf.device('/%s' % p.device):
            model.init_ops()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
        
        tconfig = get_gpu_options()
        with tf.Session(config=tconfig) as session:
            print('==> initializing models')
            session.run(init)
            print('==> running model')
            saver.restore(session, 'weights/%s' % p.prediction_weight)
            print('==> running model')
            _, preds = model.run_epoch(session, model.train, shuffle=False, verbose=False)
            preds = np.reshape(np.squeeze(preds), (decoder_length, 25, 25))
            # _, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(preds[0], cmap="gray")
            return preds, timestamp
    return [], []
    

def  get_districts_preds(preds):
    res = []
    for d_t in preds:
        r_t = []
        for x, y in p.dis_points:
            # x = col, y = row
            r_t.append(d_t[y][x] * 500)
        res.append(r_t)
    return res


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python train.py -pr "vectors/labels" -f "vectors/full_data" -fl "vectors/full_data_len" -p "train_basic_64b_tanh_12h_" -fw "basic" -dc 1 -l mae -r 10 -usp 1 -e 13 -bs 126 -sl 24 -ir 0 
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--feature", help="")
    parser.add_argument("-f", "--folder", default=0, type=int,  help="prefix to save weighted files")
    parser.add_argument("-w", "--url_weight", type=str, default="")
    parser.add_argument("-au", "--attention_url", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", default='mae')
    parser.add_argument("-e", "--embed_size", type=int, default=12)
    parser.add_argument("-el", "--encoder_length", type=int, default=24)
    parser.add_argument("-dl", "--decoder_length", type=int, default=24)
    parser.add_argument("-ds", "--decoder_size", type=int, default=6)
    parser.add_argument("-g", "--grid_size", type=int, default=25)
    parser.add_argument("-dt", "--dtype", default='grid')
    parser.add_argument("-t", "--is_test", default=0, help="is testing", type=int)
    parser.add_argument("-cnn", "--use_cnn", default=1, help="using cnn or not", type=int)
    parser.add_argument("-r", "--rnn_layers", default=1, help="number of rnn layers", type=int)
    parser.add_argument("-a", "--adversarial", default=1, help="Using adversarial networks", type=int)
    parser.add_argument("-m", "--model", default="GAN")
    parser.add_argument("-rs", "--restore", default=0, help="Restore pre-trained models", type=int)

    args = parser.parse_args()
    # if not os.path.exists("missing.pkl"):
    # sparkEngine = SparkEngine()
    # preds, timestamp = get_prediction_real_time(sparkEngine)
        # utils.save_file("missing.pkl", preds)
    # else:
    #     preds = utils.load_file("missing.pkl")
    # preds = np.reshape(np.squeeze(preds), (24, 25, 25))
    # prediction = get_districts_preds(preds)
    # print(prediction[0])
    if args.model == "GAN":
        train_gan(args.feature, args.attention_url, args.url_weight, args.batch_size, args.encoder_length, args.embed_size, args.loss, args.decoder_length, args.decoder_size, 
            args.grid_size, args.rnn_layers, dtype=args.dtype, is_folder=bool(args.folder), is_test=bool(args.is_test), restore=bool(args.restore))
    else:
        main(args.feature, args.attention_url, args.url_weight, args.batch_size, args.encoder_length, args.embed_size, args.loss, args.decoder_length, args.decoder_size, 
        args.grid_size, args.rnn_layers, dtype=args.dtype, is_folder=bool(args.folder), is_test=bool(args.is_test), use_cnn=bool(args.use_cnn),  restore=bool(args.restore))
