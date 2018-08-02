import argparse
import numpy as np
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
    for r in data:
        # each row record
        dis_vectors = [np.zeros(dim, dtype=np.float).tolist()] * 25
        d_ = r.rstrip("\n")
        # print(d_)
        mgrp = pattern.search(d_)
        if(mgrp):
            # init onehot vector features
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
            res.append(dis_vectors)   
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


def convert_data_to_grid(url, out_url, part=1):
    grid = heatmap.build_map(pr.map_size)
    data = utils.load_file(url)
    lt = len(data)
    res = []
    if part != 1:
        bound = int(math.ceil(float(lt) / part))
    else:
        bound = lt
    for i, t in enumerate(data):
        if i and (i % bound) == 0:
            out_url_name = out_url + "_" + str(i / bound)
            utils.save_file(out_url_name, res)        
            res = []
        g = heatmap.fill_map(t, grid)
        res.append(g)
        utils.update_progress(float(i)/lt)

    if part == 1:
        out_url_name = out_url
    else:
        out_url_name = out_url + "_" + str(part)
    utils.save_file(out_url_name, res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="prefix to fix")
    parser.add_argument("-u", "--url")
    parser.add_argument("-u1", "--url1")
    parser.add_argument("-dim", "--dim", type=int, default=15)
    # parser.add_argument("-dim", "--dim", type=int, default=12)
    parser.add_argument("-s", "--part", type=int, default=12)
    parser.add_argument("-t", "--task", type=int, default=0)
    args = parser.parse_args()

    if args.task == 0:
        parse_vector(args.url, args.url1, args.dim)
    else:
        convert_data_to_grid(args.url, args.url1, args.part)
        



