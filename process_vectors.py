import argparse
import numpy as np
import pickle
import utils
import time
import re
from datetime import datetime


file_path = "/home/alex/Documents/datasets/featured_vectors"
m = 30
fixed = datetime.strptime("2017-06-01 00:00:00", '%Y-%m-%d %H:%M:%S')

# check if date > fixed date
def get_date_offset(tm):
    d1 = datetime.fromtimestamp(tm)
    delta = d1 - fixed
    d = delta.days
    s = delta.seconds
    if s:
        s = s // 3600
    return d, s


def get_time(tm):
    x = float(tm)/1000000
    t = time.localtime(x)
    return time.strftime('%Y-%m-%d %H:%M:%S', t)

def get_vector(v):
    v = v.rstrip("\n").replace("[[", "").replace("]]", "")
    v = v.split(",")
    return v


def compact_vectors(url):
    data = utils.load_file(url, False)
    length = len(data)
    plength = (length // m) * m
    b_data = data[:plength]
    # r_data = data[plength:]
    arr = []
    tmp = []
    i = 0
    for v in b_data:
        i += 1
        v = get_vector(v)
        tmp.append([float(x) for x in v])
        if i == m:
            arr.append(tmp)
            i = 0
            tmp = []
    return arr


def clear_vector(url):
    arr = compact_vectors(url)
    with open("feature_vectors_rnn.pkl", 'wb') as file:
        pickle.dump(arr, file, pickle.HIGHEST_PROTOCOL)


def get_vectors(url):
    data = utils.load_file(url, False)
    arr = [get_vector(v) for v in data]
    return arr


# concate vector with pressure and weather data
def concate_vector(url1, url2, url3, url4, sent_length):
    vectors = []
    data1 = utils.load_file(url1, False)
    data2 = get_vectors(url2)
    pres = get_vectors(url3)
    pr_len = len(pres)
    weathers = get_vectors(url4)
    w_len = len(weathers)

    for i, v in enumerate(data1):
        v_ = v.replace("\n", "").split(",")
        t = get_time(v_[0])
        arr = [float(x) for x in v_[1:]] + [float(x) for x in data2[i]]
        # plus pressure
        tm = float(v_[0]) / 1000000
        d, h = get_date_offset(tm)
        if d >= 0 and d  < pr_len:
            arr += pres[d]
        else: 
            arr += [0.0]
        
        # plus weather conditions
        if d >= 0 and h >= 0:
            w_i = d * 24 + h
            if w_i < w_len:
                arr += weathers[w_i]
            else:
                arr += [0.0, 0.0, 0.0]
        else:
            arr += [0.0, 0.0, 0.0]
        
        # vectors.append(arr)
    utils.save_file("full_vectors.pkl", vectors)


# generate onehot vector from string indices and length of vector
def to_one_hot(st, length, na):
    oh_c = st.split(",")
    oh_v = np.zeros(length)
    for o_v in oh_c:
        idx = int(o_v)
        if not na or (idx not in na):
            oh_v[idx] = 1.0
    return oh_v


# parser completed vector operation
def parse_full_vector(url, out_url, m, d, not_available="", one_hot_included=False):
    if not_available:
        na = [int(x) for x in not_available.split(",")]
    else:
        na = None
    data = utils.load_file(url, False)
    res = []
    res_l = []
    df = np.zeros(d, dtype=np.float32).tolist()
    pattern = re.compile('(\d{4}-\d\d-\d\d\s\d\d:\d\d:\d\d),\((\d+),\[((\d+,*)+)\],\[(1.0,?)+\]\),\[(([\dE\-\.],?)+)\],([01])')
    pattern_no_oh = re.compile('(\d{4}-\d\d-\d\d\s\d\d:\d\d:\d\d.0),([\(\[\d.\],]*)')
    str_ = "%s,[" % d
    for r in data:
        # each row record
        d_ = r.rstrip("\n")
        if m > 1:
            arr_ = d_.split(",WrappedArray")
            cm_features = arr_[-2]
            cm_features_ = cm_features.lstrip("(").rstrip(")").split("],")
            oh_features = arr_[-1]
            oh_features_ = oh_features.lstrip("(").rstrip(")").split("]),")
            vecs = np.zeros((m, d))
            for i, row_v in enumerate(zip(cm_features_, oh_features_)):
                e, o = row_v               
                e_ = e.lstrip("[")
                o_ = o.lstrip("(")
                ae = e_.split(",")
                ae = [float(a_i.replace("[", "").replace("]", "")) for a_i in ae]
                ao = o_.split(",[")
                # print(len(ae))
                # onehot length = 
                o_l = ao[0]
                o_d = ao[1].rstrip("]")
                # print(ao)
                tmp = np.zeros(int(o_l.replace("(", "")))
                tmp = to_one_hot(o_d, int(o_l.replace("(", "")), na)
                tmp = ae + tmp.tolist()     
                vecs[i] = tmp
            res_l.append(len(cm_features_))
        elif m:
            if one_hot_included:
                mgrp = pattern.search(d_)
                # init onehot vector features
                oh_l = int(mgrp.group(2))
                oh_v = to_one_hot(mgrp.group(3), oh_l, na)
                # is holiday or not
                holi = float(mgrp.group(8))
                np.append(oh_v, holi)
                # parse common features
                fe = mgrp.group(6)
                fe_v = [float(f_v) for f_v in fe.split(",")]
                vecs = fe_v + oh_v.tolist()
            else:
                mgrp = pattern_no_oh.search(d_)
                f = mgrp.group(2).lstrip("(").rstrip("]")
                if str_ in f:
                    arr_ = f.split("],[")
                    first_ = arr_[0].replace(str_, "").split(",")
                    sec = arr_[-1].split(",")
                    vecs = [0.0]  * d
                    for i, v_i in enumerate(first_):
                        id_v = int(v_i)
                        vecs[id_v] = float(sec[i])
                else:
                    f = f.lstrip("[")
                    vecs = [float(x) for x in f.split(",")]
        else:
            vecs = np.zeros(d)
            vecs = vecs.tolist()
        res.append(vecs)
    if res_l:
        utils.save_file("%s_len" % out_url, res_l)
    utils.save_file("%s" % out_url, res)


def merge_vectors(url, url1, url2):
    data1 = utils.load_file(url)
    data2 = utils.load_file(url1)
    arr = []
    for x, y in zip(data1, data2):
        shi = [x1 + y1 for x1, y1 in zip(x, y)]
        arr.append(shi)            
        print(np.shape(shi))
    print(np.shape(arr))
            

def get_labels(url, url1, url2=None):
    print(url)
    data = utils.load_file(url, False)
    res = []
    for d in data:
        d_ = d.rstrip("\n")
        arr = d_.split(",WrappedArray")
        e = arr[1]
        e_ = e.lstrip("(").rstrip(")").lstrip("[").split(',')
        res.append(int(round(float(e_[0]) * 500)))
    # if valid url2 then save text file
    if url2:
        tmp = utils.array_to_str(res)
        utils.save_file(url2, tmp, False)
    utils.save_file(url1, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="use prefix folder to shorten url")
    parser.add_argument("-u", "--url", help="input file url")
    parser.add_argument("-u1", "--url1", help="output file url")
    parser.add_argument("-u2", "--url2")
    parser.add_argument("-u3", "--url3")
    parser.add_argument("-t", "--task", type=int, default=2)
    parser.add_argument("-s", "--sent_length", type=int, default=30)
    parser.add_argument("-d", "--doc_length", type=int, default=12)
    parser.add_argument("-m", "--max", type=int, default=1, help="max length of collect list")
    parser.add_argument("-dim", "--dim", type=int, default=13)
    parser.add_argument("-o", "--not_available", type=str, default="41,73,112,129,146")
    args = parser.parse_args()

    if args.prefix:
        args.url = args.prefix + "/" + args.url
        args.url1 = args.prefix + "/" + args.url1

    if args.task == 1:
        u2 = args.prefix + "/" + args.url2 
        u3 = args.prefix + "/" + args.url3
        # -u normalization.csv -u1 short_feature_vectors -u2 pressures -u3 weathers
        concate_vector(u, u1, u2, u3, args.sent_length, args.doc_length)
    elif args.task == 2:
        # parse completed vectors
        parse_full_vector(args.url, args.url1, args.max, args.dim, args.not_available)
    elif args.task == 3:
        args.url2 = args.prefix + "/" + args.url2
        merge_vectors(args.url, args.url1, args.url2)
    elif args.task == 4:
        args.url2 = args.prefix + "/" + args.url2
        get_labels(args.url, args.url1, args.url2)
    else:
        clear_vector(args.url)
        



