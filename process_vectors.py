import argparse
import numpy as np
import pickle
import utils
import time
from datetime import datetime


file_path = "/home/alex/Documents/datasets/featured_vectors"
m = 30
fixed = datetime.strptime("2017-06-01 00:00:00", '%Y-%m-%d %H:%M:%S')


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


def parse_full_vector(url, out_url, m, d):
    print(url)
    data = utils.load_file(url, False)
    res = []
    res_l = []
    df = np.zeros(d, dtype=np.float32).tolist()
    for d in data:
        # each row record
        d_ = d.rstrip("\n")
        arr = d_.split(",WrappedArray")
        arr = arr[1:]
        vecs = None
        for i, e in enumerate(arr):
            if not i:
                vecs = [[] for l_i in xrange(m)]
            e_ = e.lstrip("(").rstrip(")")
            a = None
            if "[" in e_:
                a = e_.split("], ")
                for a_i, a_ in enumerate(a):
                    a_v = a_.replace("[", "").replace("]", "").replace(")", "")
                    a_v = a_v.split(",")
                    for a_j in a_v:
                        a_j = a_j.strip()
                        if a_j == "null":
                            vecs[a_i].append(float(0))
                        else:
                            vecs[a_i].append(float(a_j))
            else:
                fl = False
                a = e_.split(",")
                for a_i, a_ in enumerate(a):
                    a_ = a_.strip().replace("[", "").replace("]", "").replace(")", "")
                    if a_ == "null":
                        vecs[a_i].append(float(0))
                    else:
                        vecs[a_i].append(float(a_))
            # check first aggregated vector to get length of shift
            if not i:
                l = len(a)
                res_l.append(l)
                re = m - l
                if re:
                    for x in xrange(re):
                        vecs[x + l] = df
        # s = np.shape(vecs)
        # if len(s) != 2 or s[1] == 1:
        #     break
        res.append(vecs)
    print(np.shape(res))
    print("len", len(res_l))
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
    data = utils.load_file(url, False)
    res = []
    for d in data:
        d_ = d.rstrip("\n")
        arr = d_.split(",WrappedArray")
        e = arr[1]
        e_ = e.lstrip("(").rstrip(")").split(',')
        res.append(int(round(float(e_[0]) * 500)))
    # if valid url2 then save text file
    if url2:
        tmp = utils.array_to_str(res)
        utils.save_file(url2, tmp, False)
    utils.save_file(url1, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix")
    parser.add_argument("-u", "--url")
    parser.add_argument("-u1", "--url1")
    parser.add_argument("-u2", "--url2")
    parser.add_argument("-u3", "--url3")
    parser.add_argument("-t", "--task", type=int, default=0)
    parser.add_argument("-s", "--sent_length", type=int, default=30)
    parser.add_argument("-d", "--doc_length", type=int, default=12)
    parser.add_argument("-m", "--max", type=int, default=30)
    parser.add_argument("-dim", "--dim", type=int, default=13)
    args = parser.parse_args()
    if args.task == 1:
        if args.prefix:
            u = args.prefix + "/" + args.url 
            u1 = args.prefix + "/" + args.url1 
            u2 = args.prefix + "/" + args.url2 
            u3 = args.prefix + "/" + args.url3
        else:
            u = args.url 
            u1 = args.url1 
            u2 = args.url2 
            u3 = args.url3
        # -u normalization.csv -u1 short_feature_vectors -u2 pressures -u3 weathers
        concate_vector(u, u1, u2, u3, args.sent_length, args.doc_length)
    elif args.task == 2:
        if args.prefix:
            args.url = args.prefix + "/" + args.url
            args.url1 = args.prefix + "/" + args.url1
        parse_full_vector(args.url, args.url1, args.max, args.dim)
    elif args.task == 3:
        if args.prefix:
            args.url = args.prefix + "/" + args.url
            args.url1 = args.prefix + "/" + args.url1
            args.url2 = args.prefix + "/" + args.url2
        merge_vectors(args.url, args.url1, args.url2)
    elif args.task == 4:
        if args.prefix:
            args.url = args.prefix + "/" + args.url
            args.url1 = args.prefix + "/" + args.url1
            args.url2 = args.prefix + "/" + args.url2
        get_labels(args.url, args.url1, args.url2)
    else:
        clear_vector(args.url)
        



