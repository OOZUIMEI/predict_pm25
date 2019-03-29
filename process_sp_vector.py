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
def interpolate_grid_data(url, output_url="/media/data/us_data/grid", hour_start=0, header=0, consts=[]):
    print("Converting: %s" % url)
    # start = datetime.strptime(start_date, "%Y%m%d") 
    name = url.split("/")[-1]
    name = name.split(".")[0]
    data = utils.load_file(url, False)
    dims = len(data[0].split(",")) - 2
    days = []
    old_h = -1
    # old_d = -1
    one_hour = np.reshape([], (dims,0)).tolist()
    stations = np.reshape([], (dims,0)).tolist()
    # prepare x, y  32 x 32 for interpolation
    xi = np.linspace(-1,1,32)
    yi =  np.linspace(-1,1,32)
    X, Y = np.meshgrid(xi,yi)
    month = 0
    for x_i, x in enumerate(data[header:]):
        rows = x.rstrip("\n").split(",")
        tm = rows[0]
        # d = datetime.strptime(tm[0:8], "%Y%m%d") 
        # dt = (d - start).days
        if "-" in tm:
            h = int(tm[11:13])
        else:
            h = int(tm[-2:])
        if hour_start == 0:
            h = h % 24
        else:
            h = (h - 1) % 24
        if old_h != h:
            if old_h != -1:
                one_hour_grid = []
                for i in xrange(dims):
                    if one_hour[i]:
                        inter_data = interpolate_grid(np.array(one_hour[i]), stations[i], xi, yi, X, Y)
                    else:
                        inter_data = [consts[month][i]] * 1024
                        inter_data = np.reshape(inter_data, (32,32))
                        inter_data = inter_data.tolist()
                    one_hour_grid.append(inter_data)
                days.append(one_hour_grid)
            one_hour = np.reshape([], (dims,0)).tolist()
            stations = np.reshape([], (dims,0)).tolist()
        if rows[1]:
            idx = int(rows[1])
            if idx < 1024:
                if "-" in tm:
                    month = int(tm[5:7]) - 1
                else:
                    month = int(tm[4:6]) - 1
                for i in xrange(dims):
                    if rows[i+2]:
                        v = float(rows[i + 2])
                        if v != -1.0:
                            one_hour[i].append(v)
                            stations[i].append(idx)
        old_h = h
        utils.update_progress((x_i+1.0) / len(data))
    one_hour_grid = []
    for i in xrange(dims):
        if one_hour[i]:
            inter_data = interpolate_grid(np.array(one_hour[i]), stations[i], xi, yi, X, Y)
        else:
            inter_data = [consts[month][i]] * 1024
            inter_data = np.reshape(inter_data, (32,32))
            inter_data = inter_data.tolist()
        one_hour_grid.append(inter_data)
    days.append(one_hour_grid)
    days = np.transpose(days, [0,2,3,1])
    print("output_shape", np.shape(days))
    utils.save_file("%s/%s" % (output_url, name), days)



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


"""
# start ~ vector elements are started from
# header ~ =0 ~ does not have header, start looping from row ~ header value
"""
def convert_csv_to_vectors(url, output_url, start=0, header=0, one_hots=None, one_hots_dims=0):
    data = utils.load_file(url, False)
    dims = len(data[0].split(",")) - start
    if one_hots:
        dims = dims - len(one_hots)
    outputs = []
    for d in data[header:100]:
        row = d.rstrip("\n").split(",")
        if one_hots:
            v = [0.0] * dims
            j = 0
            for i_t, i_x in enumerate(row):
                if i_t >= start and not i_t in one_hots:
                    v[j] = float(i_x)
                    j += 1
            if len(one_hots) == 1:
                o_h = to_one_hot(int(row[one_hots[0]]), one_hots_dims)
                v += o_h
            else:
                for idx in one_hots:
                    o_h = to_one_hot(int(row[idx]), one_hots_dims)
                    v += o_h
            outputs.append(v)
        else:
            row = [float(i) for i in row if i >= start]
    name = url.split("/")[-1]
    name = name.split(".")[0]
    print("output:", np.shape(outputs))
    utils.save_file("%s/%s" % (output_url, name), outputs)


def to_one_hot(idx, length):
    oh_v = [0.0] * length
    oh_v[idx] = 1.0
    return oh_v


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
        path = "/media/data/us_data/output_map"
        path2 = "/media/data/us_data/files_grid"
        # files = os.listdir(path2)
        # merge = []
        # for x in sorted(files):
        #     print(x)
        #     data = utils.load_file(path2 + "/" + x)
        #     merge.append(data)
        # utils.save_file(path + "/merge.pkl", merge)
            # convert_us_data(path + "/" + x)
        # data = utils.load_file(path2 + "/merge2.pkl")
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
        # data = utils.load_file("/media/data/us_data/train/train_.pkl")
        # data = np.transpose(data, [0, 2, 3, 1])
        # data = np.array(data)
        # data1 = data[:8760,:,:,:]
        # data2 = data[8760:17520,:,:,:]
        # data3 = data[17520:,:,:,:]
        # utils.save_file("/media/data/us_data/train/train_2014.pkl", data1.tolist())
        # utils.save_file("/media/data/us_data/train/train_2015.pkl", data2.tolist())
        # utils.save_file("/media/data/us_data/train/train_2016.pkl", data3.tolist())

        # data = utils.load_file("/media/data/us_data/train/train_att.pkl")
        # data = np.array(data)
        # data1 = data[:8760,:]
        # data2 = data[8760:17520,:]
        # data3 = data[17520:,:]
        # utils.save_file("/media/data/us_data/train/train_att_2014.pkl", data1.tolist())
        # utils.save_file("/media/data/us_data/train/train_att_2015.pkl", data2.tolist())
        # utils.save_file("/media/data/us_data/train/train_att_2016.pkl", data3.tolist())
    elif args.task == 4:
        # dim ~ hour start [0:23] [1:24], part ~ header existed?
        # [[0.10862145660706499,0.1207263561788825,0.1349276253243588,0.10789215467642199,0.04640563619311315,0.07562307555079782],
        # [0.10631524138592956,0.13076554853346237,0.10330739353377906,0.08249876360039543,0.07745439048562919,0.06090791780143575], 
        # [0.09026401306071678,0.15504747835401195,0.10691856104415078,0.06169640379049142,0.10149873105533212,0.04558523718864022],
        # [0.07041190411083767,0.12555243292682924,0.08880862068481207,0.03675579941658605,0.15480935004642568,0.020833391980738993],
        # [0.06274248236604381,0.12554422884465688,0.09506018790077411,0.041311775858436456,0.1776011378961564,0.030970654221613245],
        # [0.055922751829905894,0.08854474860944463,0.08449050215611768,0.03931533715293025,0.1862889920542799,0.01732330221651997],
        # [0.08778106189920044,0.10213031079036185,0.07907813166011342,0.0460190924308913,0.19881304783599044,0.015188673270909636],
        # [0.06295408578402882,0.10010930352094585,0.0879987122572556,0.04328257059396355,0.17978478184719623,0.013598793633498115],
        # [0.06609576733094873,0.09889653398321137,0.09770832887975789,0.04857684761281943,0.11220235709556392,0.01607636125654456],
        # [0.1234402850905355,0.16154950248756209,0.1362586530618534,0.06580628629996563,0.057115449265459424,0.020934336078404697],
        # [0.10419031141868512,0.14273071916177302,0.1318221472693501,0.08436356109464767,0.04640406614022088,0.03610405370063641],
        # [0.07631691222245278,0.10902754233279727,0.11184361337361813,0.09113806287170875,0.059421380694189396,0.06287732177755165]]
        beijing_avg = [[0.34613868294811845,0.19979132676397415,0.13492762532435773,0.10789215467642248,0.04640563619311558,0.07562307555079657],
                        [0.345999296157668,0.2113042097140975,0.10330739353377878,0.08249876360040012,0.07745439048562733,0.06090791780143249],
                        [0.2985700588250635,0.25368477607405265,0.1069185610441515,0.061696403790497165,0.10149873105532839,0.04558523718863989],
                        [0.23425925925925664,0.20492016637053742,0.08880862068481195,0.03675579941658443,0.15480935004642468,0.02083339198073907],
                        [0.2090963714611835,0.20772722050925624,0.09506018790077378,0.04131177585843119,0.17760113789615192,0.030970654221613946],
                        [0.18638709771116696,0.14646035915722486,0.08449050215611764,0.03931533715292315,0.18628899205427843,0.017323302216521834],
                        [0.2925529302548554,0.16908991852709024,0.07907813166011297,0.04601909243088076,0.19881304783598913,0.015188673270909166],
                        [0.20983558706911157,0.1656729062312251,0.08799871225725524,0.04328257059395802,0.17978478184719374,0.013598793633497757],
                        [0.22031922443649898,0.16373598341591314,0.09770832887975878,0.04857684761281418,0.11220235709555247,0.01607636125654496],
                        [0.396888189717916,0.26736161121230884,0.1362586530618541,0.06580628629996631,0.057115449265462345,0.020934336078404742],
                        [0.32582088502546136,0.23497741503521985,0.13182214726935193,0.08436356109464893,0.0464040661402242,0.03610405370063617],
                        [0.24133762191327165,0.18011511250635098,0.11184361337361808,0.09113806287170777,0.05942138069419045,0.06287732177755145]]
        interpolate_grid_data(args.url, args.url1, 1, 0, beijing_avg)
    elif args.task == 5:
        # convert csv to tensor vectors (weather & other additional factors)
        # seoul
        # convert_csv_to_vectors(args.url, args.url1, 1, 1, [12], 10)
        # china
        convert_csv_to_vectors(args.url, args.url1, 1, 0, [12], 10)
    else:
        interpolate_missing()


