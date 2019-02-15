import os
import argparse
import numpy as np
from scipy.interpolate import interp1d, griddata
from copy import deepcopy
import pickle
import utils
import time
import re
import math
from datetime import datetime
import heatmap
import properties as pr


file_path = "/home/alex/Documents/datasets/spatio_temporal_ck/"


# parser completed vector operation
def parse_vector(url, out_url, dim):
    data = utils.load_file(file_path + url, False)
    pattern = re.compile('WrappedArray\(((\d+,*\ *)*)\),')
    # el_pat = re.compile('((\(\d+,\[(\d+,?)+\],)?(\[(\d\.[\dE-]*,*)+\]*[,\ \)]*))')
    sub_pa = re.compile('(\d+),(\[(\d,?)*\]),(\[(\d\.[\dE-]*,*)+\]*[,\ \)]*)')
    res = []
    str_ = "%s,[" % dim
    count = 0
    for r in data:
        # each row record
        dis_vectors = [np.zeros(dim, dtype=np.float).tolist()] * 25
        d_ = r.rstrip("\n")
        # print(d_)
        mgrp = pattern.search(d_)
        if(mgrp):
            dis_p = mgrp.group(0)
            dis = mgrp.group(1)
            dis_codes = dis.split(", ")
            features = d_.split(dis_p)[-1]
            features = features.lstrip("WrappedArray(").rstrip(")]")
            features_v = re.split(", \(|, \[", features)
            for d in dis_codes:
                idx = int(d) - 1
                f = features_v[idx]
                if f[-1] is ")":
                    # string format is (12,[1,3,4,5,6,10],[0.1120000034570694,xxx,...])
                    mf = sub_pa.match(f)
                    elz = [0.0]*dim
                    ex = mf.group(2)[1:-1].split(",")
                    ex_v = mf.group(4)[1:-2].split(",")
                    for y, y_v in enumerate(ex_v):
                        idy = int(ex[y]) - 1
                        elz[idy] = float(y_v)
                    dis_vectors[idx] = elz
                else:
                    if str_ in f:
                        # contain zeros inside
                        arr_ = f.split("],[")
                        first_ = arr_[0].replace(str_, "").split(",")
                        sec = arr_[-1].split(",")
                        v_ = [0.0]  * dim
                        for i, v_i in enumerate(first_):
                            id_v = int(v_i)
                            v_[id_v] = float(sec[i])
                        dis_vectors[idx] = v_
                    else:
                        f_ = f.replace("[", "").replace("]", "")
                        dis_vectors[idx] = [float(x) for x in f_.split(",")]
        else:
            mf = sub_pa.search(d_)
            f = mf.group(0).rstrip("]))")
            arr_ = f.split("],[")
            first_ = arr_[0].replace(str_, "").split(",")
            sec = arr_[-1].split(",")
            v_ = [0.0]  * dim
            for i, v_i in enumerate(first_):
                id_v = int(v_i)
                v_[id_v] = float(sec[i])
            for i in xrange(25):
                dis_vectors[i] = v_
        res.append(dis_vectors)   
    print(np.shape(res))
    utils.save_file("%s%s" % (file_path, out_url), res)
    return res


def convert_data_to_grid_exe(data):
    grid = heatmap.build_map()
    lt = len(data)
    res = []
    for i, t in enumerate(data):
        g = heatmap.fill_map(t, grid)
        res.append(g)
    return res


def convert_data_to_grid(url, out_url, url_att="", out_url_att="", part=1):
    grid = heatmap.build_map(pr.map_size)
    data = utils.load_file(url)
    lt = len(data)
    attention_data = None
    att_part = None
    print(url_att)
    if url_att:
        attention_data = utils.load_file(url_att)
        alt = len(attention_data)
        if lt != alt:
            raise ValueError("Attention & Main Data need same length while %s and %s" % (lt, alt))
        data = zip(data, attention_data)
        att_part = []
    res = []
    if part != 1:
        bound = int(math.ceil(float(lt) / part))
    else:
        bound = lt
    for i, row in enumerate(data):
        if url_att:
            t, a = row
        else:
            t = row
        if i and (i % bound) == 0:
            p_i = i / bound
            out_url_name = out_url + "_" + str(p_i)
            utils.save_file(out_url_name, res)        
            if url_att:
                att_out_url_name = out_url_att + "_" + str(p_i)
                utils.save_file(att_out_url_name, att_part)
            res = []
            att_part = []
        g = heatmap.fill_map(t, grid)
        res.append(g)
        if url_att:
            att_part.append(a)
        utils.update_progress(float(i)/lt)
    if part == 1:
        out_url_name = out_url
    else:
        out_url_name = out_url + "_" + str(part)
    utils.save_file(out_url_name, res)
    if url_att:
        att_out_url_name = out_url_att + "_" + str(part)
        utils.save_file(att_out_url_name, att_part)


# hour: [0,8759], 
# day: [0,364]
def convert_transport_data(url):
    print("Converting: %s" % url)
    name = url.split("/")[-1]
    name = name.split(".")[0]
    data = utils.load_file(url, False)
    year_length = (int(data[-1].rstrip("\n").split(",")[-1]) + 1) * 24
    days = [[0] * 1024] * year_length
    old_h = -1
    old_d = -1
    one_hour = []
    for x in data[1:]:
        rows = x.rstrip("\n").split(",")
        d = int(rows[-1])
        h = int(rows[-2]) % 24
        idx = int(rows[1])
        if old_h != h:
            if old_h != -1:
                days[old_d*24 + old_h] = one_hour
            one_hour = [0]  * 1024
        if idx < 1024:
            one_hour[idx] = float(rows[2])
        old_h = h
        old_d = d
    days[old_d*24 + old_h] = one_hour
    utils.save_file("vectors/transportation/%s.pkl" % name, days)


def interpolate_grid(data, idx, x, y, X, Y):
    py = np.array([y[i / 32] for i in idx])
    px = np.array([x[i % 32]  for i in idx])
    inter_data = griddata((px, py), data, (X, Y), method="nearest")
    return inter_data


# 2014-2017: data have full length
def convert_us_data(url):
    print("Converting: %s" % url)
    # start = datetime.strptime(start_date, "%Y%m%d") 
    name = url.split("/")[-1]
    name = name.split(".")[0]
    data = utils.load_file(url, False)
    days = []
    old_h = -1
    # old_d = -1
    one_hour = []
    stations = []
    # prepare x, y  32 x 32 for interpolation
    xi = np.linspace(-1,1,32)
    yi =  np.linspace(-1,1,32)
    X, Y = np.meshgrid(xi,yi)
    for x in data[1:]:
        rows = x.rstrip("\n").split(",")
        tm = rows[0]
        # d = datetime.strptime(tm[0:8], "%Y%m%d") 
        # dt = (d - start).days
        h = int(tm[-2:]) % 24
        idx = int(rows[1])
        if old_h != h:
            if old_h != -1:
                inter_data = interpolate_grid(np.array(one_hour), stations, xi, yi, X, Y)
                days.append(inter_data)
            # one_hour = [0]  * 1024
            one_hour = []
            stations = []
        if idx < 1024:
            one_hour.append(float(rows[-1]))
            stations.append(idx)
        old_h = h
        # old_d = dt
    inter_data = interpolate_grid(np.array(one_hour), stations, xi, yi, X, Y)
    days.append(inter_data)
    print("output_shape", np.shape(days))
    utils.save_file("/media/data/us_data/grid/%s.pkl" % name, np.array(days))


# def interpolate_missing(url="vectors/sp_china_combined/sp_seoul_train_bin"):
#     print("Converting: %s" % url)
#     data = utils.load_file(url)
#     data = np.array(data)
#     L = len(data)
#     idx = np.arange(L)
#     data = np.transpose(data, (2, 1, 0))
#     for t_i,t  in enumerate(data):
#         for y_i, y in enumerate(t):
#             x = np.array(deepcopy(idx))
#             zeros = np.where(y == 0.)
#             if len(zeros):
#                 xold = np.delete(x, zeros)
#                 yold = np.delete(y, zeros)
#                 if len(xold) >= 2 and len(yold) >= 2:
#                     f = interp1d(xold, yold, fill_value="extrapolate")
#                     y_new = f(x)
#                     data[t_i, y_i,:] = y_new
#     data = np.transpose(data, (0, 2, 1))
#     for t_i,t  in enumerate(data):
#         for y_i, y in enumerate(t):
#             x = np.arange(len(y))
#             zeros = np.where(y == 0.)
#             if len(zeros):
#                 xold = np.delete(x, zeros)
#                 yold = np.delete(y, zeros)
#                 if len(xold) >= 2 and len(yold) >= 2:
#                     f = interp1d(xold, yold, fill_value="extrapolate")
#                     y_new = f(x)
#                     data[t_i, y_i,:] = y_new
#                 else:
#                     data[t_i, y_i,:] = (data[t_i, y_i-1,:] + data[t_i, y_i+1,:])/2
#     data = np.transpose(data, (1, 2, 0))
#     data = data.tolist()
#     utils.save_file("vectors/sp_china_combined/sp_seoul_train_bin_ip", data)


def interpolate_missing(url="vectors/sp_china_combined/sp_seoul_train_bin"):
    print("Converting: %s" % url)
    data = utils.load_file(url)
    data = np.array(data)
    data = np.transpose(data, (2, 1, 0))
    for t_i,t  in enumerate(data):
        for y_i, y in enumerate(t):
            zeros = np.where(y == 0.)
            if len(zeros):
                yold = np.delete(y, zeros)
                m = np.mean(yold)
                for i in zeros:
                    data[t_i,y_i,i] = m
    # data = np.transpose(data, (2, 1, 0))
    # zeros = np.where(data == 0)
    # print(zeros)
    # data = data.tolist()
    # utils.save_file("vectors/sp_china_combined/sp_seoul_train_bin_ip", data)


# change missing values to mean by default
def interpolate_missing_china(url="vectors/sp_china_combined/sp_china_test_bin"):
    print("Converting: %s" % url)
    data = utils.load_file(url)
    data = np.array(data)
    data = data.transpose()
    for t_i,t in enumerate(data):
        # x = np.array(deepcopy(idx))
        zeros = np.where(t == 0.)
        if len(zeros):
            yold = np.delete(t, zeros)
            m = np.mean(yold)
            for i in zeros:
                data[t_i,i] = m
    # data = data.transpose()
    # data = data.tolist()
    # utils.save_file("vectors/sp_china_combined/sp_china_test_bin_ip", data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="prefix to fix")
    parser.add_argument("-u", "--url")
    parser.add_argument("-u1", "--url1")
    parser.add_argument("-au", "--aurl")
    parser.add_argument("-au1", "--aurl1")
    parser.add_argument("-dim", "--dim", type=int, default=15)
    # parser.add_argument("-dim", "--dim", type=int, default=12)
    parser.add_argument("-s", "--part", type=int, default=1)
    parser.add_argument("-t", "--task", type=int, default=0)
    args = parser.parse_args()

    if args.task == 0:
        parse_vector(args.url, args.url1, args.dim)
    elif args.task == 1:
        convert_data_to_grid(args.url, args.url1, args.aurl, args.aurl1, args.part)
    elif args.task == 2:
        convert_transport_data(args.url)
    elif args.task == 3:
        # convert us data
        # path = "/media/data/us_data/output_map"
        # path2 = "/media/data/us_data/files_grid"
        # files = os.listdir(path2)
        # merge = []
        # for x in sorted(files):
        #     print(x)
        #     data = utils.load_file(path2 + "/" + x)
        #     merge.append(data)
        # # utils.save_file(path + "/merge.pkl", merge)
        #     # convert_us_data(path + "/" + x)
        # # data = utils.load_file(path2 + "/merge2.pkl")
        # data = np.array(merge)
        # test = data[:,-8760:,:,:]
        # train = data[:,:-8760,:,:]
        # utils.save_file(path2 + "/train.pkl", np.transpose(train, [1, 0, 2, 3]).tolist())
        # utils.save_file(path2 + "/test.pkl", np.transpose(test, [1, 0, 2, 3]).tolist())

        # data = utils.load_file("/media/data/us_data/additional_vectors.csv", False)
        # totals = []
        # for d in data[1:]:
        #     rows = d.rstrip("\n").split(",")
        #     month = float(rows[1])
        #     day = float(rows[2])
        #     hour = float(rows[3])
        #     v = [month, day, hour]
        #     totals.append(v)
        # train = totals[:-8760]
        # test = totals[-8760:]
        # print(np.shape(train), np.shape(test))
        # utils.save_file("/media/data/us_data/train_att.pkl", train)
        # utils.save_file("/media/data/us_data/test_att.pkl", test)
        data = utils.load_file("/media/data/us_data/train/train_.pkl")
        # data = np.transpose(data, [0, 2, 3, 1])
        data = np.array(data)
        # data1 = data[:8760,:,:,:]
        # data2 = data[8760:17520,:,:,:]
        data3 = data[17520:,:,:,:]
        # utils.save_file("/media/data/us_data/train/train_2014.pkl", data1.tolist())
        # utils.save_file("/media/data/us_data/train/train_2015.pkl", data2.tolist())
        utils.save_file("/media/data/us_data/train/train_2016.pkl", data3.tolist())

        # data = utils.load_file("/media/data/us_data/train/train_att.pkl")
        # data = np.array(data)
        # data1 = data[:8760,:]
        # data2 = data[8760:17520,:]
        # data3 = data[17520:,:]
        # utils.save_file("/media/data/us_data/train/train_att_2014.pkl", data1.tolist())
        # utils.save_file("/media/data/us_data/train/train_att_2015.pkl", data2.tolist())
        # utils.save_file("/media/data/us_data/train/train_att_2016.pkl", data3.tolist())
    else:
        interpolate_missing()


