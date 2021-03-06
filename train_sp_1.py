"""
index of N/A in one_hot feature 0,41,73,112,129,146
39 in case of only seoul & not integrate china
"""

import utils
import tensorflow as tf
from numba import cuda
import numpy as np
import math
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
from NeuralNet import NeuralNetwork
from adain import Adain
from stack_autoencoder import StackAutoEncoder
from mask_gan import MaskGan
from apgan import APGan
from apgan_lstm import APGAN_LSTM
from mask_gan_2 import MaskGan2
from capgan import CAPGan
from tgan import TGAN
from tganlstm import TGANLSTM
from tnetlstm import TNetLSTM
from tnet import TNet
from apnet import APNet
from srcns import SRCN

import matplotlib
import matplotlib.pyplot as plt
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


def get_gpu_options(use_fraction=True):
    gpu_options = None
    device_count = None
    if "gpu" in p.device:
        if use_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p.gpu_fraction)
            os.environ["CUDA_VISIBLE_DEVICES"]=p.gpu_devices
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
            os.environ["CUDA_VISIBLE_DEVICES"]="2"
    else:
        device_count={"GPU":0}
    configs = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, device_count=device_count)
    return configs


def execute(path, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer=None, offset=0, validation_url="", attention_valid_url="", best_val_loss=None):
    print("==> Loading dataset")
    dataset = utils.load_file(path)
    global_t = offset
    if dataset:
        dataset = np.asarray(dataset, dtype=np.float32)
        lt = len(dataset)
        if validation_url:
            valid_set = utils.load_file(validation_url)
            valid_set = np.asarray(valid_set, dtype=np.float32)
            valid_length = len(valid_set)
            print("using separated valid data %i" % valid_length)
            train, _ = utils.process_data_grid(lt, batch_size, encoder_length, decoder_length, True)
            valid, _ = utils.process_data_grid(valid_length, batch_size, encoder_length, decoder_length, True)
            valid = valid + lt
            if attention_url:
                valid_att_data = utils.load_file(attention_valid_url)
                print("using separated valid att %i" % len(valid_att_data))
            dataset = np.concatenate((dataset, valid_set), axis=0)
            del(valid_set) 
        else:
            train, valid = utils.process_data_grid(lt, batch_size, encoder_length, decoder_length, is_test)
        if attention_url:
            attention_data = utils.load_file(attention_url)
            if attention_valid_url:
                attention_data += valid_att_data
                del(valid_att_data)
        else:
            attention_data = None
        model.set_data(dataset, train, valid, attention_data)
        model.assign_datasets(session)
        if not is_test:
            best_val_epoch = 0
            if not best_val_loss:
                best_val_loss = float('inf')
            # best_overall_val_loss = float('inf')
            print('==> starting training')
            train_f, valid_f = train_writer
            for epoch in xrange(p.total_iteration):
                print('Epoch {}'.format(epoch))
                start = time.time()
                global_t = offset + epoch

                train_loss, _ = model.run_epoch(session, train, global_t, train_f, train=True, stride=2)
                print('Training loss: {}'.format(train_loss))

                valid_loss, _ = model.run_epoch(session, valid, global_t, train_writer=valid_f, stride=2)
                print('Validation loss: {}'.format(valid_loss))

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    # if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    # best_overall_val_loss = best_val_loss
                    saver.save(session, 'weights/%s.weights' % url_weight)

                if (epoch - best_val_epoch) > p.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))
        else:
            # saver.restore(session, url_weight)
            print('==> running model')
            loss, preds = model.run_epoch(session, model.train, shuffle=False, stride=2)
            l_str = 'Test mae loss: %.4f' % loss
            print(l_str)
            pt = re.compile("weights/([A-Za-z0-9_.]*).weights")
            name = pt.match(url_weight)
            if name:
                name_s = name.group(1)
            else:
                name_s = url_weight
            utils.save_file("test_sp/%s" % name_s, preds)
    return global_t, best_val_loss


def train_baseline(url_feature="", attention_url="", url_weight="sp", batch_size=128, encoder_length=24, embed_size=None, loss=None, decoder_length=24, decoder_size=4, grid_size=25, rnn_layers=1, dtype="grid", is_folder=False, is_test=False, use_cnn=True, restore=False, model_name="", validation_url="", attention_valid_url="", best_val_loss=None, forecast_factor=0):
    if model_name == "APNET":
        model = APNet(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decoder_length=decoder_length, decode_vector_size=decoder_size, grid_size=grid_size,  use_attention=bool(attention_url), forecast_factor=forecast_factor)
    elif model_name == "TNET": 
        model = TNet(encoder_length=8, decoder_length=8, grid_size=32)
    elif model_name == "TNETLSTM": 
        model = TNetLSTM(encoder_length=8, decoder_length=8, grid_size=32)
    elif model_name == "SRCN":
        model = SRCN(encoder_length=8, decoder_length=8, grid_size=32)
    else:
        model = BaselineModel(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decoder_length=decoder_length, decode_vector_size=decoder_size, rnn_layers=rnn_layers, dtype=dtype, grid_size=grid_size, use_cnn=use_cnn, loss=loss, use_attention=bool(attention_url), forecast_factor=forecast_factor)
    print('==> initializing models')
    with tf.device('/%s' % p.device):
        model.init_ops(is_train=(not is_test))
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
            # suf = time.strftime("%Y.%m.%d_%H.%M")
            csn = int(time.time())
            train_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_train_" + str(csn), session.graph)
            valid_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_valid_" + str(csn), session.graph)
            if restore:
                url_weight = url_weight + "_" + str(csn)
       
        folders = None
        best_val_loss = float('inf')
        if is_folder:
            folders = sorted(os.listdir(url_feature))
            if attention_url:
                a_folders = sorted(os.listdir(attention_url))
                folders = zip(folders, a_folders)
            last_epoch = 0
            for i, files in enumerate(folders):
                if attention_url:
                    x, y = files
                    att_url = os.path.join(attention_url, y)
                    print("==> Training set (%i, %s, %s)" % (i + 1, x, y))
                else: 
                    x = files
                    att_url = ""
                    print("==> Training set (%i, %s)" % (i + 1, x))
                last_epoch, best_val_loss = execute(os.path.join(url_feature, x), att_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, 
                                    is_test, (train_writer, valid_writer), last_epoch, validation_url=validation_url, attention_valid_url=attention_valid_url, best_val_loss=best_val_loss)
                if not is_test:
                    print("best val loss:" + str(best_val_loss))
        else:
            _, best_val_loss = execute(url_feature, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, (train_writer, valid_writer), 
                        validation_url=validation_url, attention_valid_url=attention_valid_url, best_val_loss=best_val_loss)
            if not is_test:
                print("best val loss:" + str(best_val_loss))


def save_gan_preds(url_weight, preds):
    shape = np.shape(preds)
    pt = re.compile("weights/([A-Za-z0-9_.]*).weights")
    name = pt.match(url_weight)
    if name:
        name_s = name.group(1)
    else: 
        name_s = url_weight
    pr_s = shape[0] * p.batch_size
    if shape[-1] == 1 and len(shape) == 4:
        preds = np.reshape(preds, (pr_s, shape[-3], shape[-2]))
    elif shape[1] == 1:
        shape = list(shape)
        shape = [pr_s] + shape[3:]
        preds = np.reshape(preds, tuple(shape))
    else:
        preds = np.reshape(preds, (pr_s, shape[-2], shape[-1]))
    utils.save_file("test_sp/%s" % name_s, preds)


def get_gan_model(model_name, encoder_length, embed_size, batch_size, decoder_size, decoder_length, grid_size, is_test, forecast_factor, use_attention=True):
    if model_name == "APGAN":
        model = APGan(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, decoder_length=decoder_length, grid_size=grid_size, forecast_factor=forecast_factor, use_attention=use_attention)
    elif model_name == "MASKGAN":
        model = MaskGan(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, grid_size=grid_size, use_cnn=1, forecast_factor=forecast_factor, use_attention=use_attention)
    elif model_name == "APGAN_LSTM":
        model = APGAN_LSTM(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, decoder_length=decoder_length, grid_size=grid_size, forecast_factor=forecast_factor, use_attention=use_attention)
    elif model_name == "CAPGAN":
        model = CAPGan(encoder_length=encoder_length, encode_vector_size=embed_size, batch_size=batch_size, decode_vector_size=decoder_size, grid_size=grid_size, forecast_factor=forecast_factor, use_attention=use_attention)
    elif model_name == "TGAN":
        model = TGAN(encoder_length=8, decoder_length=8, grid_size=32)
    else:
        model = TGANLSTM(encoder_length=8, decoder_length=8, grid_size=32)
    with tf.device('/%s' % p.device):
        model.init_ops(not is_test)
    return model


def execute_gan(path, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer=None, offset=0):
    #if restore and not is_test:
    #    tf.reset_default_graph()
    #    print(tf.get_default_graph())
        #with tf.device('/%s' % p.device):
        #    model.init_ops(not is_test)
        #    #model.add_placeholders()
        #trainable_vars = tf.trainable_variables()
        #saver = tf.train.Saver(trainable_vars)
    print("==> Loading dataset")
    dataset = utils.load_file(path)
    if dataset:
        dataset = np.asarray(dataset, dtype=np.float32)
        lt = len(dataset)
        train, _ = utils.process_data_grid(lt, batch_size, encoder_length, decoder_length, True)
   
    attention_data = None
    if attention_url:
        attention_data = utils.load_file(attention_url)
   
    model.set_data(dataset, train, None, attention_data)
        #with tf.Session(config=gpu_configs) as session:
            #init = tf.global_variables_initializer()
            #session.run(init)
    model.assign_datasets(session)
    if not is_test:   
        print("start training")
        for epoch in xrange(100):
             _ = model.run_epoch(session, train, offset + epoch, train_writer, train=True, verbose=False, stride=2)
        saver.save(session, 'weights/%s.weights' % url_weight)
    else:
         # saver.restore(session, url_weight)
        print('==> running model')
        _, preds = model.run_epoch(session, train, train=False, verbose=False, shuffle=False, stride=2)
        save_gan_preds(url_weight, preds)


def train_gan(url_feature="", attention_url="", url_weight="sp", batch_size=128, encoder_length=24, embed_size=None, 
    decoder_length=24, decoder_size=4, grid_size=25, is_folder=False, is_test=False, restore=False, model_name="APGAN", forecast_factor=0):
    gpu_configs = get_gpu_options()
    folders = None
    train_writer = None
    if not is_test:
        csn = int(time.time())
        url_weight = url_weight.split("/")[-1]
        url_weight = url_weight.rstrip(".weights")
        train_writer =  tf.summary.FileWriter("summaries/%s_%i" % (url_weight, csn))
    if is_folder:
        folders = sorted(os.listdir(url_feature))
        if attention_url:
            a_folders = sorted(os.listdir(attention_url))
            folders = zip(folders, a_folders)
        fl = len(folders)
        # train each year in 100 epochs then repeat until it match total_iteration
        repeat_steps = int(p.total_iteration / 100)
        for t in xrange(repeat_steps):
            tf.reset_default_graph()
            model = get_gan_model(model_name, encoder_length, embed_size, batch_size, decoder_size, decoder_length, grid_size, False, forecast_factor=forecast_factor, use_attention=bool(attention_url))
            saver = tf.train.Saver(max_to_keep=2)
            with tf.Session(config=gpu_configs) as session:
                train_writer.add_graph(session.graph, global_step=(fl * t * 100))
                if restore:
                    print("restore previous weights")
                    saver.restore(session, "weights/%s.weights" % url_weight)
                else:
                    init = tf.global_variables_initializer()
                    session.run(init)
                # loop over data folders
                for i, files in enumerate(folders):
                    offset = (fl * t + i) * 100
                    if attention_url:
                        x, y = files
                        att_url = os.path.join(attention_url, y)
                        print("==> Training set (%i, %s, %s)" % (i + 1, x, y))
                    else: 
                        x = files
                        att_url = None
                        print("==> Training set (%i, %s)" % (i + 1, x))
                    # train each year 2013 - 2016 in 100 epochs intervally
                    execute_gan(os.path.join(url_feature, x), att_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer, offset)
                session.close()
            restore = True
    else:
        model = get_gan_model(model_name, encoder_length, embed_size, batch_size, decoder_size, decoder_length, grid_size, True, forecast_factor=forecast_factor, use_attention=bool(attention_url))
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session(config=gpu_configs) as session:
            if restore:
                print("reload pretrained model")
                saver.restore(session, "weights/%s.weights" % url_weight)
            else:
                init = tf.global_variables_initializer()
                session.run(init)
            if train_writer is not None:
                train_writer.add_graph(session.graph)
            execute_gan(url_feature, attention_url, url_weight, model, session, saver, batch_size, encoder_length, decoder_length, is_test, train_writer)


# def  get_districts_preds(preds):
#     res = []
#     for d_t in preds:
#         r_t = []
#         for x, y in p.dis_points:
#             # x = col, y = row
#             r_t.append(d_t[y][x] * 500)
#         res.append(r_t)
#     return res


"""
average data of area => 25 points
output should be 24 x 25
"""
def aggregate_predictions(preds):
    outputs = []
    # loop over timesteps
    for t in preds:
        # 25 x 25
        out_ = []
        for d in p.dis_points:
            val = 0.0
            for x, y in d:
                val += t[y][x]
            if val != 0.0:
                val = val / len(d) * 300
            out_.append(val)
        outputs.append(out_)
    return outputs


"""
activate spark engine & real time prediction service
"""        
def get_prediction_real_time(sparkEngine, model=None, url_weight="", dim=15, prediction_weight="", encoder_length=24, decoder_length=24, attention_length=24, is_close_cuda=True):
    # continuously crawl aws and aqi & weather
    end = utils.get_datetime_now()
    end = end - timedelta(hours=1)
    # end = datetime.strptime("2018-06-19 11:01:00", p.fm)
    # e_ = end.strftime(p.fm)
    start = end - timedelta(hours=23)
    start = start.replace(minute=0, second=0, microsecond=0)
    # s_ = start.strftime(p.fm)
    # 2. process normalize data
    vectors, w_pred, china_vectors, timestamp = sparkEngine.process_vectors(start, end, dim)
    v_l = len(vectors)
    if v_l:
        sp_vectors = psv.convert_data_to_grid_exe(vectors)
        if v_l < encoder_length:
            sp_vectors = np.pad(sp_vectors, ((encoder_length - v_l,0), (0,0), (0,0), (0, 0)), 'constant', constant_values=0)
        # repeat for 25 districts
        if w_pred:
            w_pred = np.repeat(np.expand_dims(w_pred, 1), p.grid_size, 1)
            de_vectors = psv.convert_data_to_grid_exe(w_pred)
            # pad to fill top elements of decoder vectors
            de_vectors = np.pad(de_vectors, ((0, 0), (0, 0), (0, 0), (6, 0)), 'constant', constant_values=0)
        else:
            # know nothing about future weather forecast
            de_vectors = np.zeros((decoder_length, p.grid_size, p.grid_size, dim))
        sp_vectors = np.concatenate((sp_vectors, de_vectors), axis=0)

        c_l = len(china_vectors)
        if c_l < attention_length:
            # print(attention_length - c_l)
            china_vectors = np.pad(china_vectors, ((attention_length - c_l, 0), (0, 0)), 'constant', constant_values=0)

        # 4. Feed to model
        if model is None:
            # model = BaselineModel(encoder_length=encoder_length, encode_vector_size=12, batch_size=1, decoder_length=decoder_length, rnn_layers=1,
            #                 dtype='grid', grid_size=25, use_cnn=True)
            # model.set_data(sp_vectors, [0], None)
            # model = MaskGan(encoder_length=encoder_length, encode_vector_size=15, batch_size=1, decode_vector_size=9, grid_size=25, use_cnn=True)
            model = APGan(encoder_length=24, decoder_length=24, encode_vector_size=15, batch_size=1, decode_vector_size=9, grid_size=25, forecast_factor=0)
            # model = APNet(encoder_length=24, decoder_length=24, encode_vector_size=15, batch_size=1, decode_vector_size=9, grid_size=25, forecast_factor=0)
        model.set_data(sp_vectors, [0], None, china_vectors)
        with tf.device('/%s' % p.device):
            model.init_ops(is_train=False)
            saver = tf.train.Saver()
        tconfig = get_gpu_options(False)        
        with tf.Session(config=tconfig) as session:
            model.assign_datasets(session)    
            preds_pm25 = realtime_execute(model, session, saver, decoder_length, p.prediction_weight_pm25)
            model.forecast_factor = 1
            preds_pm10 = realtime_execute(model, session, saver, decoder_length, p.prediction_weight_pm10)
            china_vectors = np.array(china_vectors)
            # print("china", china_vectors.shape)
            # tf.reset_default_graph()
            # session.close()
            if is_close_cuda:
                cuda.select_device(0)
                cuda.close()
        return (preds_pm25, preds_pm10), timestamp, np.transpose(china_vectors[:,:2] * 500)
    else:
        return ([],[]), [], []
    

def realtime_execute(model, session, saver, decoder_length, prediction_weight):
        print('==> restore model')
        saver.restore(session, 'weights/%s' % prediction_weight)
        print('==> running model')
        _, preds = model.run_epoch(session, model.train, train=False, verbose=False, shuffle=False)
        preds = np.reshape(preds, (decoder_length, p.grid_size, p.grid_size))
        # utils.save_file("test_acc/current_preds", preds) 
        preds = aggregate_predictions(preds)
        return preds


# call neural networks, stack autoencoder, or adain 
def run_neural_nets(url_feature="", attention_url="", url_weight="sp", encoder_length=24, encoder_size=15, decoder_length=8, decoder_size=9, is_test=False, restore=False, model="NN", pre_train=False):
    if model == "NN":
        model = NeuralNetwork(encoder_length=encoder_length, encoder_vector_size=encoder_size, decoder_length=decoder_length, decoder_vector_size=decoder_size)
    elif model == "SAE":
        model = StackAutoEncoder(encoder_length=encoder_length, encoder_vector_size=encoder_size, decoder_length=decoder_length, pre_train=pre_train)
    else:
        model = Adain(encoder_length=encoder_length, encoder_vector_size=encoder_size, decoder_length=decoder_length)
    print('==> initializing models')
    with tf.device('/%s' % p.device):
        model.init_model()
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
            url_weight = url_weight.split("/")[-1]
            url_weight = url_weight.rstrip(".weights")
        
        if not is_test:
            suf = time.strftime("%Y.%m.%d_%H.%M")
            train_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_train", session.graph, filename_suffix=suf)
            valid_writer = tf.summary.FileWriter(sum_dir + "/" + url_weight + "_valid", session.graph, filename_suffix=suf)

        print("==> Loading dataset")
        dataset = utils.load_file(url_feature)
        if dataset:
            dataset = np.asarray(dataset, dtype=np.float32)
            lt = len(dataset)
            st = int(lt/2)
            lt = lt - st
            dataset = dataset[st:,:,:]
            train, valid = utils.process_data_grid(lt, p.batch_size, encoder_length, decoder_length, is_test)
            if attention_url:
                attention_data = utils.load_file(attention_url)
            else:
                attention_data = None
            model.set_data(dataset, train, valid, attention_data, session)
            if not is_test:
                best_val_epoch = 0
                best_val_loss = float('inf')
                # best_overall_val_loss = float('inf')
                print('==> starting training')
                for epoch in xrange(p.total_iteration):
                    print('Epoch {}'.format(epoch))
                    start = time.time()
                    train_loss, _ = model.run_epoch(session, train, epoch, train_writer, train_op=model.train_op, train=True)
                    print('Training loss: {}'.format(train_loss))

                    valid_loss, _ = model.run_epoch(session, valid, epoch, valid_writer)
                    print('Validation loss: {}'.format(valid_loss))

                    if valid_loss < best_val_loss:
                        best_val_loss = valid_loss
                        best_val_epoch = epoch
                        print('Saving weights')
                        saver.save(session, 'weights/%s.weights' % url_weight)

                    if (epoch - best_val_epoch) > p.early_stopping:
                        break
                    print('Total time: {}'.format(time.time() - start))
            else:
                # saver.restore(session, url_weight)
                print('==> running model')
                _, preds = model.run_epoch(session, model.train, shuffle=False)
                pt = re.compile("weights/([A-Za-z0-9_.]*).weights")
                name = pt.match(url_weight)
                if name:
                    name_s = name.group(1)
                else:
                    name_s = url_weight
                utils.save_file("test_sp/%s" % name_s, preds)


if __name__ == "__main__":
    # 10110 -> 10080 -> 126 batch 
    # python train.py -pr "vectors/labels" -f "vectors/full_data" -fl "vectors/full_data_len" -p "train_basic_64b_tanh_12h_" -fw "basic" -dc 1 -l mae -r 10 -usp 1 -e 13 -bs 126 -sl 24 -ir 0 
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--feature", help="a path to datasets (either to a file or a folder)")
    parser.add_argument("-v", "--validation_url", help="a path to validation set")
    parser.add_argument("-f", "--folder", default=0, type=int,  help="either train a folder or just train a file")
    parser.add_argument("-w", "--url_weight", type=str, default="")
    parser.add_argument("-au", "--attention_url", type=str, default="")
    parser.add_argument("-av", "--valid_attention_url", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", default='mse')
    parser.add_argument("-e", "--embed_size", type=int, default=15)
    parser.add_argument("-el", "--encoder_length", type=int, default=24)
    parser.add_argument("-dl", "--decoder_length", type=int, default=24)
    parser.add_argument("-ds", "--decoder_size", type=int, default=9)
    parser.add_argument("-g", "--grid_size", type=int, default=25, help="size of grid")
    parser.add_argument("-dt", "--dtype", default='grid', help="dtype is either 'grid' or 'dis' that mean use grid data of just station data")
    parser.add_argument("-t", "--is_test", default=0, help="is testing", type=int)
    parser.add_argument("-cnn", "--use_cnn", default=1, help="using cnn or not in mining input's vectors", type=int)
    parser.add_argument("-r", "--rnn_layers", default=1, help="number of rnn layers", type=int)
    parser.add_argument("-m", "--model", default="GAN")
    parser.add_argument("-rs", "--restore", default=0, help="Restore pre-trained model", type=int)
    parser.add_argument("-p", "--pretrain", default=0, help="Pretrain model: only use of SAE networks", type=int)
    parser.add_argument("-bv", "--best_val_loss", type=float, help="best validation loss from previous training")
    parser.add_argument("-fc", "--forecast_factor", type=int, default=0, help="0 is pm2.5 1 is pm10")
    args = parser.parse_args()
    """
    # sparkEngine = SparkEngine()
    # preds, timestamp, china = get_prediction_real_time(sparkEngine)
    """
    #  0.00183376428791 0.00183376425411552
    if "GAN" in args.model:
        train_gan(args.feature, args.attention_url, args.url_weight, args.batch_size, args.encoder_length, args.embed_size, args.decoder_length, args.decoder_size, 
            args.grid_size, is_folder=bool(args.folder), is_test=bool(args.is_test), restore=bool(args.restore), model_name=args.model, forecast_factor=args.forecast_factor)
    elif args.model in ["CNN_LSTM", "TNET", "TNETLSTM", "APNET", "SRCN"] :
        train_baseline(args.feature, args.attention_url, args.url_weight, args.batch_size, args.encoder_length, args.embed_size, args.loss, args.decoder_length, args.decoder_size, 
        args.grid_size, args.rnn_layers, dtype=args.dtype, is_folder=bool(args.folder), is_test=bool(args.is_test), use_cnn=bool(args.use_cnn),  restore=bool(args.restore), 
        model_name=args.model, validation_url=args.validation_url, attention_valid_url=args.valid_attention_url, best_val_loss=args.best_val_loss, forecast_factor=args.forecast_factor)
    elif args.model == "ADAIN":
        run_neural_nets(args.feature, args.attention_url, args.url_weight, args.encoder_length, args.embed_size, args.decoder_length, args.decoder_size, bool(args.is_test), bool(args.restore), args.model)
    elif args.model == "SAE":
        run_neural_nets(args.feature, args.attention_url, args.url_weight, args.encoder_length, args.embed_size, args.decoder_length, args.decoder_size, bool(args.is_test), bool(args.restore), args.model, bool(args.pretrain))
    elif args.model == "NN":
        run_neural_nets(args.feature, args.attention_url, args.url_weight, args.encoder_length, args.embed_size, args.decoder_length, args.decoder_size, bool(args.is_test), bool(args.restore))
    elif args.model == "TGAN" or args.model == "TGANLSTM":
        train_gan(args.feature, "", args.url_weight, args.batch_size, args.encoder_length, 1, args.decoder_length, 1, 32, False, is_test=bool(args.is_test), restore=bool(args.restore), model_name=args.model)