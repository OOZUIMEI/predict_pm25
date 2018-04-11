import pickle 
import os.path as path
import sys
import os
import codecs
import numpy as np
import math
from datetime import datetime
import time


def save_predictions(best_preds, best_lb, path):
    tmp = "preds,labels\n"
    for x, y in zip(best_preds, best_lb):
        tmp += "%i,%i\n" % (x, y)
    save_file(path, tmp, False)


def validate_path(name):
    paths = name.split("/")
    paths = paths[:-1]
    tmp = ""
    if paths:
        for e, folder in enumerate(paths):
            if folder:
                tmp += folder
                if not path.exists(tmp):
                    os.makedirs(tmp)
                tmp += "/"


def save_file(name, obj, use_pickle=True):
    validate_path(name)
    with open(name, 'wb') as f:
        if use_pickle:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        else: 
            f.write(obj)


def save_file_utf8(name, obj):
    with codecs.open(name, "w", "utf-8") as file:
        file.write(u'%s' % obj)


def array_to_str(obj, delimiter="\n"):
    tmp = ""
    l = len(obj) - 1
    for i, x in enumerate(obj):
        tmp += str(x)
        if i < l:
            tmp += delimiter
    return tmp


def load_file(pathfile, use_pickle=True):
    if path.exists(pathfile):
        if use_pickle:
            with open(pathfile, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(pathfile, 'rb') as file:
                data = file.readlines()
        return data 


def load_file_utf8(pathfile):
    if path.exists(pathfile):
        with codecs.open(pathfile, "r", "utf-8") as file:
            data = file.readlines()
        return data 


def check_file(pathfile):
    return path.exists(pathfile)


def assert_url(url):
    if not check_file(url):
        raise ValueError("%s is not existed" % url)


def intersect(c1, c2):
    return list(set(c1).intersection(c2))


def sub(c1, c2):
    return list(set(c1)- set(c2))


def update_progress(progress, sleep=0.01, barLength=20):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    time.sleep(sleep)


def calculate_accuracy(pred, pred_labels, rng, is_classify):
    accuracy = 0
    if is_classify:
        accuracy = np.sum([1 for x, y in zip(pred, pred_labels) if x == y])
    else:
        accuracy = np.sum([1 for x, y in zip(pred, pred_labels) if abs(x - y) <= rng])
        # print([(x, y) for x, y in zip(pred, pred_labels) if abs(round(x) - round(y)) <= rng])
    return accuracy


def Linear(AQIhigh, AQIlow, Conchigh, Conclow, Concentration):
    Conc = float(Concentration)
    a = ((Conc - Conclow) / (Conchigh - Conclow)) * (AQIhigh - AQIlow) + AQIlow
    linear = round(a)
    return linear


# convert pm10 micro value to aqi value
def AQIPM10(Concentration):
    Conc = float(Concentration)
    c = math.floor(Conc)
    if (c >= 0 and c < 55):
        AQI = Linear(50, 0, 54, 0, c)
    elif(c >= 55 and c < 155):
        AQI = Linear(100, 51, 154, 55, c)
    elif(c >= 155 and c < 255):
        AQI = Linear(150, 101, 254, 155, c)
    elif(c >= 255 and c < 355):
        AQI = Linear(200, 151, 354, 255, c)
    elif(c >= 355 and c < 425):
        AQI = Linear(300, 201, 424, 355, c)
    elif(c >= 425 and c < 505):
        AQI = Linear(400, 301, 504, 425, c)
    elif(c >= 505 and c < 605):
        AQI = Linear(500, 401, 604, 505, c)
    else:
        AQI = 0
    return AQI


# convert pm25 micro value to aqi value
def AQIPM25(Concentration):
    Conc = float(Concentration)
    c = (math.floor(10 * Conc)) / 10;
    if (c >= 0 and c < 12.1):
        AQI = Linear(50, 0, 12, 0, c)
    elif (c >= 12.1 and c < 35.5):
        AQI = Linear(100, 51, 35.4, 12.1, c)
    elif (c >= 35.5 and c < 55.5):
        AQI = Linear(150, 101, 55.4, 35.5, c)
    elif (c >= 55.5 and c < 150.5):
        AQI = Linear(200, 151, 150.4, 55.5, c)
    elif (c >= 150.5 and c < 250.5):
        AQI = Linear(300, 201, 250.4, 150.5, c)
    elif (c >= 250.5 and c < 350.5):
        AQI = Linear(400, 301, 350.4, 250.5, c)
    elif (c >= 350.5 and c < 500.5):
        AQI = Linear(500, 401, 500.4, 350.5, c)
    else:
        AQI = 0
    return AQI


def get_pm25_class(index):
    cl = 0
    if index <= 50:
        cl = 0
    elif index > 50 and index <= 100:
        cl = 1
    elif index > 100 and index <= 150:
        cl = 2
    elif index > 150 and index <= 200:
        cl = 3
    elif index > 200 and index <= 300:
        cl = 4
    else:
        cl = 5
    return cl


# pre-process data to batch
def process_data(dataset, data_len, pred, batch_size, max_input_len, max_sent, 
                is_test=False, pred_sight=1, is_classify=False, context_meaning=1, fr_ele = 7):
    len_dataset = len(dataset)
    dlength = len_dataset - 1
    # total batch
    dlength_b = len_dataset // batch_size
    maximum = dlength_b * batch_size
    # dataset = dataset[:-1]
    # if data_len:
    #     data_len = data_len[:-1]
    new_data = []
    new_data_len = []
    new_pred = []
    decode_vec = []
    if context_meaning:
        zeros = np.zeros(fr_ele).tolist()
    for x in xrange(maximum):
        e = x + max_sent
        p_i = e + pred_sight - 1
        d_e = p_i + 1
        if e <= dlength and d_e <= len_dataset:
            arr = dataset[x : e]
            # if input len is  like a sentence            
            if data_len:
                arr_l = data_len[x : e]
                new_data_len.append(arr_l)
                arr_d =[y[0][fr_ele:] for y in dataset[e : d_e]]
            else:
                arr_d =[y[fr_ele:] for y in dataset[e : d_e]]
                if context_meaning:
                    arr = [zeros + en_c[fr_ele:] for en_c in arr]
            # append vector to matrix
            new_data.append(arr)
            new_pred.append(pred[p_i])
            decode_vec.append(arr_d)
        else:
            break
    # print(np.shape(decode_vec))
    if not is_test:
        new_data, new_pred, decode_vec = np.asarray(new_data, dtype=np.float32), np.asarray(new_pred, dtype=np.int32), np.asarray(decode_vec, dtype=np.float32)
        train_len = int(dlength_b * 0.8) * batch_size
        # training set = 80% of dataset
        train_data = new_data[:train_len]
        train_pred = new_pred[:train_len]
        train_dec = decode_vec[:train_len]
        # generate validate set 20% of dataset
        valid_data = new_data[train_len:]
        valid_pred = new_pred[train_len:]
        valid_dec = decode_vec[train_len:]        

        if data_len:
            new_data_len = np.asarray(new_data_len, dtype=np.int32)
            new_data_len = new_data_len[r]
            train_data_len = new_data_len[:train_len]
            valid_data_len = new_data_len[train_len:]
        else: 
            train_data_len, valid_data_len = None, None
        train = (train_data, train_data_len, train_pred, train_dec)
        dev = (valid_data, valid_data_len, valid_pred, valid_dec)
    else:
        train = None
        dev = (new_data, new_data_len, new_pred, decode_vec)
    return train, dev


def now_timestamp():
    return time.mktime(time.localtime())


def get_datetime_now():
    return datetime.fromtimestamp(now_timestamp())


# format hour, day < 10 to 10 format
def format10(no):
    if no < 10:
        return "0" + str(no)
    else:
        return str(no)


# change datetime str to filename format
def clear_datetime(dt):
    tmp = ""
    for x in dt:
        if x == "-" or x == ":":
            tmp += "."
        elif x == " ":
            tmp += "_"
        else:
            tmp += x
    return tmp