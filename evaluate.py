from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import argparse
import itertools
import numpy as np
import heatmap
from math import sqrt
import properties as pr
import utils
from crawling_base import Crawling  
import aqi
from utils import save_file


# old eval
def evaluate(pred, labs, rg, is_classify=False, verbose=True):
    l = len(pred)
    if is_classify:
        acc = utils.calculate_accuracy(pred, labs, rg, True)
        acc = float(acc) / l * 100
        print("classified accuracy:%.6f" % acc)
    else:
        pred = [utils.boost_pm25(x) for x in pred]
        pred_ = [utils.get_pm25_class(x) for x in pred]
        labs_ = [utils.get_pm25_class(x) for x in labs]
        # precision_score(labs_, pred_, average='weighted')
        cacc = utils.calculate_accuracy(pred_, labs_, 0, True)
        cacc = float(cacc) / l * 100
        # print(test)
        acc = utils.calculate_accuracy(pred, labs, rg, False)
        acc = float(acc) / l * 100
        mae = mean_absolute_error(labs, pred)
        rmse = sqrt(mean_squared_error(labs, pred))
        r2 = r2_score(labs, pred)
        if verbose:
            print("classified accuracy:%.6f" % cacc)
            print("accuracy:%.6f" % acc)
            print("mae:%.6f" % mae)
            print("rmse:%.6f" % rmse)
            print("r2_score:%.6f" % r2)


# evaluate grid training
def evaluate_sp(url, url2, decoder_length=24, is_grid=True, grid_eval=True):
    cr = Crawling() 
    map_ = heatmap.build_map()
    data = utils.load_file(url)
    if type(data) is list:
        data = np.asarray(data)
    if len(data.shape) == 4:
        lt = data.shape[0] * data.shape[1]
    else:
        lt = data.shape[0]
    if is_grid:
        data = np.reshape(data, (lt, data.shape[-2], 25, 25))
    else:
        data = np.reshape(data, (lt, data.shape[-2], 25))
    labels = utils.load_file(url2)
    labels = np.asarray(labels)
    loss_mae = 0.0
    loss_rmse = 0.0
    r2_total = 0.0
    for i, d in enumerate(data):
        d = d[:decoder_length,:,:]
        pred_t  = []
        if is_grid:
            for d_ in d:
                d_t = heatmap.clear_interpolate_bound(np.asarray(d_), map_)
                pred_t.append(d_t)
        else:
            if grid_eval:
                for d_ in d:
                    d_t = heatmap.fill_map(d_, map_)
                    pred_t.append(d_t)
            else:
                pred_t = d
        lb_i = i * pr.strides + 24
        lbt = labels[lb_i:(lb_i+decoder_length),:,0]
        if grid_eval:
            lbg = []
            for x in lbt:
                x_l = heatmap.fill_map(x, map_)
                lbg.append(x_l)
            lbg = np.asarray(lbg)
            lbg = lbg.flatten()
        else:
            lbg = lbt.flatten()
        pred_t = np.asarray(pred_t)
        pred_t = pred_t.flatten()
        mae, mse, r2 = get_evaluation(pred_t, lbg)
        loss_mae += mae
        loss_rmse += mse
        r2_total += r2
        utils.update_progress((i + 1.0) / lt)
    loss_mae = loss_mae / lt * 300
    loss_rmse = sqrt(loss_rmse / lt) * 300
    r2_total = r2_total / lt
    print("MAE: %.6f %.6f" % (loss_mae, cr.ConcPM25(loss_mae)))
    print("RMSE: %.6f %.6f" % (loss_rmse, cr.ConcPM25(loss_rmse)))
    print("R2 Score: %.6f" % r2_total)


def convert_coordinate_to_idx():
    dis = []
    for d in pr.dis_points:
        dis_idx = []
        for x, y in d:
            idx = y * pr.grid_size + x
            dis_idx.append(idx)
        a = sorted(dis_idx)
        dis.append(a)
    utils.save_file("district_idx.pkl", dis)
    return dis


# timestep data is numpy
# districts are pre-converted data
def aggregate_predictions(districts, timstep_data):
    outputs = []
    # loop over timesteps
    for d in districts:
        # mapping pred data with district idx
        # make sure timestep_data is numpy format
        dis_pred = timstep_data[d]
        val = np.mean(dis_pred)
        outputs.append(val)
    return outputs


# evaluate grid training
def evaluate_by_districts(url, url2, stride=2, encoder_length=24, decoder_length=24, forecast_factor=0, is_classify=False, confusion_title="", norm=True, is_grid=True, offset=48):
    if not utils.validate_path("district_idx.pkl"):
        districts = convert_coordinate_to_idx()
    else:
        districts = utils.load_file("district_idx.pkl")
    data = utils.load_file(url)
    print(np.shape(data))
    if type(data) is list:
        data = np.asarray(data)
    if len(data.shape) == 4:
        lt = data.shape[0] * data.shape[1]
        # if not is_grid:
        # data = np.reshape(data, (lt, data.shape[-1]))
        # else:
        data = np.reshape(data, (lt, data.shape[-2], data.shape[-1]))
    else:
        lt = data.shape[0]
        data = np.reshape(data, (lt, data.shape[-2], data.shape[-1]))
    days = int(math.ceil(data.shape[1] / 24.0))
    if days > 2:
        st_h = (days - 1) * 24
    else: 
        st_h = 0
    labels = utils.load_file(url2)
    labels = np.asarray(labels)
    if not is_classify:
        loss_mae = [0.0] * decoder_length
        # loss_rmse = [0.0] * decoder_length
    elif not confusion_title:
        acc = 0.
    else:
        acc = None
    cr = Crawling() 
    for i, d in enumerate(data):
        if not is_grid:
            d = d[st_h:decoder_length]
        else:
            d = d[st_h:decoder_length,:]
        lb_i = i * stride + encoder_length
        lbt = labels[lb_i+offset:(lb_i+offset+decoder_length),:,forecast_factor]
        if not confusion_title:
            a = 0.
        else:
            a = None
        for t_i, (t, l_t) in enumerate(zip(d, lbt)):
            t_i += st_h
            if is_grid:
                pred_t = aggregate_predictions(districts, t)
                pred_t = np.array(pred_t)
            else:
                pred_t = t
            pred_t = pred_t.flatten()
            if not is_classify:
                if not forecast_factor:
                    # mae, mse, _ = get_evaluation(pred_t, l_t)
                    #mae = mean_absolute_error(pred_t * 300, l_t * 300)
                    mae = mean_absolute_error([cr.ConcPM25(x*300) for x in pred_t], [cr.ConcPM25(x * 300) for x in l_t])
                    loss_mae[t_i] += mae
                    # loss_rmse[t_i] += mse
                else:
                    mae = mean_absolute_error([cr.ConcPM10(x*300) for x in pred_t], [cr.ConcPM10(x * 300) for x in l_t])
                    loss_mae[t_i] += mae
            elif not confusion_title:
                a += classify_data(pred_t, l_t, forecast_factor, tp="G")
            elif a is None:
                a = classify_data(pred_t, l_t, forecast_factor, True, tp="G")
            else:
                a += classify_data(pred_t, l_t, forecast_factor, True, tp="G")
        if is_classify:
            a = a / decoder_length
            if not confusion_title:
                acc += a
            elif acc is None:
                acc = a
            else:
                acc += a
        utils.update_progress((i + 1.0) / lt)
    if not is_classify:
        # print mae loss score
        # caculate loss for each timestep
        loss_mae = np.array(loss_mae) / lt
        # loss_rmse = [sqrt(x / lt)  * 300 for x in loss_rmse]
        # calculate accumulated loss
        no_h = len(loss_mae)
        if no_h > 24:
            print("hourly errors", loss_mae[:24])
            days = math.ceil(no_h * 1.0 / 24)
            for x in xrange(1, int(days)):
                ed = (x+1) * 24
                if ed > no_h:
                    ed = no_h
                print("day %i" % (x + 1), np.mean(loss_mae[x*24:ed]))
        else:
            print(loss_mae)
        # print(loss_mae)
        #print_accumulate_error(loss_mae, loss_rmse, decoder_length, forecast_factor)
    elif not confusion_title:
        # print classification score
        acc = acc / lt * 100
        print("accuracy %.4f" % acc)
    else:
        name = url.split("/")[-1]
        # print confusion matrix
        utils.save_file("results/confusion/confusion_%s" % name, acc)
        draw_confusion_matrix(acc, confusion_title, norm)


# evaluate grid training
def evaluate_us(url, url2, stations, stride=2, encoder_length=48, decoder_length=48, forecast_factor=0, eval_stat=False):
    data = utils.load_file(url)
    if type(data) is list:
        data = np.asarray(data)
    if len(data.shape) == 4:
        lt = data.shape[0] * data.shape[1]
        data = np.reshape(data, (lt, data.shape[-2], data.shape[-1]))
    else:
        lt = data.shape[0]
        data = np.reshape(data, (lt, data.shape[-2], data.shape[-1]))
    
    labels = utils.load_file(url2)
    print(data.shape)
    if not eval_stat:
        loss_mae = [0.0] * decoder_length
    else:
        loss_mae = np.array([[0.0] * decoder_length] * len(stations))
    for i, d_ in enumerate(data):
        d = d_[:decoder_length,:]
        lb_i = i * stride + encoder_length
        lbt = labels[lb_i:(lb_i+decoder_length),:]
        for t_i, (t, l_t) in enumerate(zip(d, lbt)):
            l_t = l_t.flatten()
            pred_t = []
            label_t = []
            t_ = np.reshape(t, (32, 32))
            for r in stations:
                er = r / 32
                ex = r % 32
                sty = er - 1
                if sty < 0:
                    sty = 0
                stx = ex - 1
                if stx < 0:
                    stx = 0
                pred_point = t_[sty:er+1,stx:ex+2]
                pred_t.append(np.mean(pred_point))
                label_t.append(l_t[r])
            if not eval_stat:
                mae = mean_absolute_error(pred_t, label_t)
                loss_mae[t_i] += mae
            else:
                # loss of all stations at timestep t_i
                # print("pre",[prt*300 for prt in pred_t])
                mae = [abs(p_s - l_s) for p_s, l_s in zip(pred_t,label_t)]
                for m_i, m in enumerate(mae):
                    loss_mae[m_i, t_i] += m
        utils.update_progress((i + 1.0) / lt)
    # print mae loss score
    # caculate loss for each timestep
    barrier = 300
    if forecast_factor:
        barrier = 500
    loss_mae = np.array(loss_mae) / lt  * barrier
    # print loss
    for i, x in enumerate(loss_mae):
        print(x)


# evaluate grid china training
def evaluate_china(url, url2, stations, encoder_length=12, decoder_length=24, eval_stat=False):
    cr = Crawling()
    # missing = [136,156,824,1028,1053,1084,1085,1457,1460,1461,1462,1464,1465,1476,1477, \
    #         1478,1479,1481,1482,1484,1485,1486,1487,1510,1841,3814,4246,4268,4301,4311, \
    #         4313,4317,4329,4331,4333,4349,4360,4504,4524,4529,4532,4535,4580,4860,5270, \
    #         5457,5489,5509,5962,6007,6011,6039,6054,6125,6172,6189,6192,6201,6230,6234, \
    #         6246,6254,6255,6256,6257,6258,6295,6300,6319,6336,6356,6362,6371,6372,6394, \
    #         6491,6492,6493,6494,6497,6517,6519,6523,6539,6559,6564,6568,6569,6570,6612, \
    #         6613,6614,6615,6616,6617,6618,6619,6620,6621,6622,6623,6624,6625,6626,6627, \
    #         6628,6629,6630,6631,6632,6633,6634,6635,6636,6637,6638,6687,6704,6754,6832, \
    #         6849,6858,6873,6979,7255,8099,8100,8101,8225,8226,8227,8228,8229,8230,8231, \
    #         8232,8233,8234,8235,8236,8237,8238,8239,8240,8241,8273,8274,8275,8276,8277, \
    #         8278,8279,8280,8281,8282,8283,8284,8285,8286,8287,8288,8289,8290,8291,8344, \
    #         8421,8422,8423,8424,8425,8426,8427,8428,8429,8430,8431,8432,8495,8496,8497, \
    #         8498,8499,8500,8501,8502,8503,8504,8631,8759]
    data = utils.load_file(url)
    labels = utils.load_file(url2)
    if type(data) is list:
        data = np.asarray(data)
    if len(data.shape) == 4:
        lt = data.shape[0] * data.shape[1]
        data = np.reshape(data, (lt, data.shape[-2], data.shape[-1]))
        labels =  np.reshape(labels, (lt, data.shape[-2], data.shape[-1])) 
    else:
        lt = len(data)
        data = np.array(data)
        labels = np.array(labels)
    if not eval_stat:
        loss_mae = [0.0] * decoder_length
    else:
        loss_mae = np.array([[0.0] * decoder_length] * len(stations))
    classes = [0.0] * decoder_length
    # loss_lt = 0
    loss_total = [0.0] * decoder_length
    # ckt = 0
    valid = []
    for i, (d_, l_) in enumerate(zip(data, labels)):
        # st = 2 * i
        # ed = st + decoder_length
        # flag = False
        # for ck in range(st, ed):
        #     if ck in missing:
        #         flag = True
        #         break
        # if flag:
        #     continue                
        d = d_[:decoder_length,:]
        lbt = l_[:decoder_length,:]
        for t_i, (t, l_t) in enumerate(zip(d, lbt)):
            pred_t = []
            label_t = []
            for r in stations:
                #pred_t.append(t[r])
                #label_t.append(l_t[r])
                # if not math.isnan(t[r]):
                pr = t[r]
                if pr < 0:
                    pr = 0
                print(pr)
                pred_t.append(cr.aqi_pm25_china(pr))
                label_t.append(cr.aqi_pm25_china(l_t[r] * 300))
                # label_t.append(l_t[r])
                # print(t[r], l_t[r])
            # if pred_t:
            if not eval_stat:
                mae = mean_absolute_error(pred_t, label_t)
                classes[t_i] += accuracy_score([cr.aqi_pm25_china_class(x) for x in pred_t], [cr.aqi_pm25_china_class(x) for x in label_t])
                if mae > 80:
                    valid.append("%i,%i" % (i, t_i))
                if mae < 80:
                    loss_total[t_i] += 1
                    loss_mae[t_i] += mae
            else:
                mae = [abs(p_s - l_s) for p_s, l_s in zip(pred_t,label_t)]
                for m_i, m in enumerate(mae):
                    loss_mae[m_i, t_i] += m
        # loss_lt += 1
        # utils.update_progress((i + 1.0) / lt)
    va = "\n".join(valid)
    save_file("outline_china", va, False)
    loss_mae = [x / y if y > 0 else 0 for x, y in zip(loss_mae, loss_total)]
    for i, x in enumerate(loss_mae):
        # if not eval_stat:
        print(x, loss_total[i])
        # else:
            # print([y/loss_lt for y in x])
    # print("accumulated:", np.mean(loss_mae[1:6]))
    print(np.array(classes) / lt)


# evaluate grid training
def evaluate_single_pred(url, url2, decoder_length=8):
    cr = Crawling() 
    data = utils.load_file(url)
    if type(data) is list:
        data = np.asarray(data)
    lt = data.shape[0] * data.shape[1]
    data = np.reshape(data, (lt, 25))
    dtl = len(data)
    labels = utils.load_file(url2)
    labels = np.asarray(labels)
    loss_mae = 0.0
    loss_rmse = 0.0
    r2_total = 0.0
    for i, d in enumerate(data):
        pred_t = np.asarray(d).flatten()
        lb_i = i * pr.strides + 24
        lbt = labels[lb_i:(lb_i+decoder_length),:,0]
        lbg = lbt[decoder_length - 1,:].flatten()
        mae, mse, r2 = get_evaluation(pred_t, lbg)
        loss_mae += mae
        loss_rmse += mse
        r2_total += r2
        utils.update_progress((i + 1.0) / dtl)
    loss_mae = loss_mae / lt * 300
    loss_rmse = sqrt(loss_rmse / lt) * 300
    r2_total = r2_total / lt
    print("MAE: %.6f %.6f" % (loss_mae, cr.ConcPM25(loss_mae)))
    print("RMSE: %.6f %.6f" % (loss_rmse, cr.ConcPM25(loss_rmse)))
    print("R2 score: %.6f" % r2_total)


# predict multiple dimension
# pm2.5, pm10
def evaluate_multi(url, url2, time_lags=24):
    cr = Crawling() 
    preds = utils.load_file(url)
    preds = np.array(preds)
    lt = len(preds)
    labels = utils.load_file(url2)
    labels = np.array(labels)

    loss_mae0, loss_mae1 = 0.0, 0.0
    loss_rmse0, loss_rmse1 = 0.0, 0.0
    r2_0, r2_1 = 0.0, 0.0
    for i, d in enumerate(preds):
        lb_i = i * pr.strides + time_lags + 1
        mae0, mse0, r2 = get_evaluation(d[:time_lags,:], labels[lb_i:(lb_i+time_lags),:,0])
        # mae1, mse1 = get_evaluation(d[:time_lags,:,1], labels[lb_i:(lb_i+time_lags),:,1])
        loss_rmse0 += mse0  
        # loss_rmse1 += mse1
        loss_mae0 += mae0
        # loss_mae1 += mae1
        r2_0 += r2
    loss_mae0 = loss_mae0 / lt * 300
    loss_mae1 = loss_mae1 / lt * 300
    loss_rmse0 = sqrt(loss_rmse0 / lt) * 300
    loss_rmse1 = sqrt(loss_rmse1 / lt) * 300
    r2_0 = r2_0 / lt
    print("MAE: %.6f %.6f" % (loss_mae0, cr.ConcPM25(loss_mae0)))
    print("RMSE: %.6f %.6f" % (loss_rmse0, cr.ConcPM25(loss_rmse0)))
    print("R2 Score: %.6f" % r2_0)
    
    # print("MAE PM10: %.6f" % loss_mae1)
    # print("RMSE PM10: %.6f" % loss_rmse1)
    # labels0 = labels[:,:,0].flatten()
    # labels1 = labels[:,:,1].flatten()
    # std0 = np.std(labels0)
    # std1 = np.std(labels1)
    # m0 = np.mean(labels0)
    # m1 = np.mean(labels1)
    # print(m0, std0)
    # print(m1, std1)


# predict multiple dimension
# pm2.5, pm10
def evaluate_transportation(url, url2, pred_length=8):
    preds = utils.load_file(url)
    preds = np.array(preds)
    lt = len(preds)
    labels = utils.load_file(url2)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 32, 32)
    shape = np.shape(preds)
    if preds.shape[-1] < pred_length:
        print("data shape is ", preds.shape)
        pred_length = preds[-1]
    loss_mae0 = [0.0] * pred_length
    loss_rmse0 = [0.0] * pred_length
    r2_total = 0.0
    for i, d in enumerate(preds):
        # 8 is encoder_length
        lb_i = i + 8 
        # labels[lb_i:(pred_length+lb_i),:,:]
        for x in xrange(pred_length):
            mae0, mse0, _ = get_evaluation(d[x,:,:], labels[lb_i+x,:,:])
            # mae0, mse0, r2 = get_evaluation(d[0,:,:], labels[lb_i,:,:])
            loss_rmse0[x] += mse0 
            loss_mae0[x] += mae0
            # r2_total += r2
    loss_mae0 = [(x / lt * 131) for x in loss_mae0]
    loss_rmse0 = [(sqrt(x / lt) * 131) for x in loss_rmse0]
    # r2_total = r2_total / lt
    # print("MAE: %.6f" % loss_mae0)
    # print("RMSE: %.6f" % loss_rmse0)
    # print("R2 Score: %.6f" % r2_total)
    print_accumulate_error(loss_mae0, loss_rmse0, pred_length, 0)


# evaluate grid training
def evaluate_lstm(url, url2, decoder_length=24, forecast_factor=0, is_classify=False):
    data = utils.load_file(url)
    if type(data) is list:
        data = np.asarray(data)
    lt = data.shape[0] * data.shape[1]
    data = np.reshape(data, (lt, data.shape[-1]))
    if decoder_length > data.shape[-1]:
        decoder_length = data.shape[-1]
    dtl = len(data)
    labels = utils.load_file(url2)
    labels = np.asarray(labels)
    if not is_classify:
        loss_mae = [0.0] * decoder_length
        loss_rmse = [0.0] * decoder_length
    else:
        acc = 0.
    #: r2_total = 0.0
    cr = Crawling() 
    for i, d in enumerate(data):
        if decoder_length < data.shape[-1]:
            pred_t = d[:decoder_length]
        else:
            pred_t = d
        lb_i = i * pr.strides + 24
        lbt = np.mean(labels[lb_i:(lb_i+decoder_length),:,forecast_factor], axis=1)
        a = 0.
        for t_i, (p, l) in enumerate(zip(pred_t, lbt)):
            if not is_classify:
                # mae, mse, _ = get_evaluation(p, l)
                mae = abs(cr.ConcPM10(p*300) - cr.ConcPM10(l * 300))
                loss_mae[t_i] += mae
                # loss_rmse[t_i] += mse
            else:
                a += classify_data(pred_t, lbt, forecast_factor)
        if is_classify:
            a = a / decoder_length
            acc += a
        # r2_total += r2
        utils.update_progress((i + 1.0) / dtl)
    if not is_classify:
        loss_mae = np.array(loss_mae) / lt
        # loss_rmse = [sqrt(x / lt)  * 300 for x in loss_rmse]
        # print("R2 score: %.6f" % r2_total)
        print_accumulate_error(loss_mae, loss_rmse, decoder_length, forecast_factor=forecast_factor)
    else: 
        acc = acc / lt * 100
        print("accuracy %.4f" % acc)


def get_evaluation(pr, lb):
    pr = pr.flatten()
    lb = lb.flatten()
    mse = mean_squared_error(pr, lb)
    mae = mean_absolute_error(pr, lb)
    # r2 = r2_score(lb, pr)
    return mae, mse, 0.0


def print_accumulate_error(loss_mae, loss_rmse, decoder_length, forecast_factor=0):
    cr = Crawling() 
    for x in xrange(decoder_length):
        print("%ih" % (x + 1))
        if not forecast_factor:
            print("S MAE: %.6f %.6f" % (loss_mae[x], cr.ConcPM25(loss_mae[x])))
            print("S RMSE: %.6f %.6f" % (loss_rmse[x], cr.ConcPM25(loss_rmse[x])))
        else: 
            print("S PM10 MAE: %.6f %.6f" % (loss_mae[x], cr.ConcPM10(loss_mae[x])))
            print("S PM10 RMSE: %.6f %.6f" % (loss_rmse[x], cr.ConcPM10(loss_rmse[x])))
        if x > 0:
            loss_mae[x] += loss_mae[x-1]
            t_mae = loss_mae[x] / (x + 1)
            loss_rmse[x] += loss_rmse[x-1]
            t_rmse = loss_rmse[x] / (x + 1)
            if not forecast_factor:
                print("T MAE: %.6f %.6f" % (t_mae, cr.ConcPM25(t_mae)))
                print("T RMSE: %.6f %.6f" % (t_rmse, cr.ConcPM25(t_rmse)))
            else:
                print("T PM10 MAE: %.6f %.6f" % (t_mae, cr.ConcPM10(t_mae)))
                print("T PM10 RMSE: %.6f %.6f" % (t_rmse, cr.ConcPM10(t_rmse)))
            

def print_us_error(loss_mae, loss_rmse, decoder_length, forecast_factor=0):
    for x in xrange(decoder_length):
        print("%ih" % (x + 1))
        if not forecast_factor:
            print("S MAE: %.6f" % (loss_mae[x]))
        else: 
            print("S PM10 MAE: %.6f" % (loss_mae[x]))
        if x > 0:
            loss_mae[x] += loss_mae[x-1]
            t_mae = loss_mae[x] / (x + 1)
            if not forecast_factor:
                print("T MAE: %.6f" % (t_mae))
            else:
                print("T PM10 MAE: %.6f" % (t_mae))


def classify_data(pr, lb, factor, is_conf=False, tp="S"):
    # use seoul metric
    if tp == "S":
        pr = [get_class(x * 300, factor) for x in pr]
        lb = [get_class(x * 300, factor) for x in lb]
        if is_conf:
            acc = confusion_matrix(lb, pr, labels=[0,1,2,3])
        else:
            acc = accuracy_score(lb, pr)
    else: 
        # use global metric
        pr_ = [get_aqi_class(x * 300) for x in pr]
        lb_ = [get_aqi_class(x * 300) for x in lb]
        if is_conf:
            acc = confusion_matrix(lb_, pr_, labels=[0,1,2,3,4,5])
        else:
            acc = accuracy_score(lb_, pr_)
    return acc


def get_class(x, factor):    
    if not factor: # classify pm25
        x = round(aqi.ConcPM25(x))
        if x <= 15:
            return 0
        elif x <= 35:
            return 1
        elif x <= 75:
            return 2
        else:
            return 3
    else: # classify pm10
        x = round(aqi.ConcPM10(x))
        if x <= 30:
            return 0
        elif x <= 80:
            return 1
        elif x <= 150:
            return 2
        else:
            return 3


def get_aqi_class(x):    
    if x <= 50:
        return 0
    elif x <= 100:
        return 1
    elif x <= 150:
        return 2
    elif x <= 200:
        return 3
    elif x <= 300:
        return 4
    return 5
    
    

def draw_confusion_matrix(conf, title="Confusion Matrix", norm=True):
    plt.figure()
    tick_marks = [0,1,2,3]
    classes = ["Good", "Average", "Bad", "Very Bad"]
    if norm:
        conf = conf.astype("float") / conf.sum(axis=1, keepdims=True)
    plt.imshow(conf, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fm = ".3f" if norm else "d"
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i,j], fm), horizontalalignment="center", color="white" if conf[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("Label")
    plt.xlabel("Prediction")
    name = title.lower().replace(" ", "_").replace(".", "")
    plt.savefig("results/figures/%s.png" % name, format="png", bbox_inches="tight")
    # plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="predictions file path")
    parser.add_argument("-l", "--url2", help="labels file path")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)
    parser.add_argument("-cf", "--confusion", default="")
    parser.add_argument("-t", "--task", type=int, default=0)
    parser.add_argument("-g", "--grid", type=int, default=1)
    parser.add_argument("-gev", "--grid_eval", type=int, default=1)
    parser.add_argument("-tl", "--time_lags", type=int, default=24)
    parser.add_argument("-f", "--forecast_factor", type=int, default=0)
    parser.add_argument("-o", "--offset", type=int, default=0)
    """
    SVR         & \textbf{0.657} & 0.618 & 0.582 & 0.551 & 0.525 & 0.501 & 0.480 & 0.462 & 0.446 & 0.432 & 0.419 \\
    SAE         & 0.462 & 0.421 & 0.384 & 0.379 & 0.349 & 0.315 & 0.310 & 0.289 & 0.278 & 0.281 & 0.262 \\
    ADAIN       & 0.442 & 0.418 & 0.398 & 0.369 & 0.340 & 0.320 & 0.311 & 0.294 & 0.279 & 0.266 & 0.256 \\
    FC-LSTM     & 0.574 & 0.512 & 0.437 & 0.392 & 0.376 & 0.359 & 0.329 & 0.321 & 0.302 & 0.291 & 0.288 \\
    FC-GRU      & 0.563 & 0.496 & 0.422 & 0.381 & 0.363 & 0.347 & 0.324 & 0.316 & 0.294 & 0.286 & 0.284 \\
    ConvLSTM    & 0.658 & 0.651 & 0.651 & 0.651 & 0.647 & 0.642 & 0.638 & 0.635 & 0.633 & 0.632 & 0.631 \\
    \hline
    APNet       & \underline{0.716} & \underline{0.700} & \underline{0.697} & \underline{0.695} & \underline{0.686} & \underline{0.668} & \underline{0.662} & \underline{0.650} & \underline{0.657} & \underline{0.647} & \underline{0.638} \\
    APGAN       & \textbf{0.752} & \textbf{0.743} & \textbf{0.737} & \textbf{0.729} & \textbf{0.720} & \textbf{0.678} & \textbf{0.689} & \textbf{0.680} & \textbf{0.682} & \textbf{0.682} & \textbf{0.683} \\
    """
    args = parser.parse_args()
    if args.task == 0:
        # evaluate_sp(args.url, args.url2, args.time_lags, bool(args.grid), bool(args.grid_eval))
        for x in range(28, 52, 4):
            print("%ih" % x)
            evaluate_sp(args.url, args.url2, x, bool(args.grid), bool(args.grid_eval))
    elif args.task == 1:
        # SAE MAE 8h: MAE: 39.32 RMSE: 44.17
        # Neural nets 8h: MAE: 38.66 RMSE: 45.67
        evaluate_single_pred(args.url, args.url2, args.time_lags)
    elif args.task == 2:
        evaluate_transportation(args.url, args.url2, args.time_lags)
    elif args.task == 3:
        evaluate_lstm(args.url, args.url2, args.time_lags, args.forecast_factor, args.classify)
    elif args.task == 4:
        evaluate_by_districts(args.url, args.url2, pr.strides, decoder_length=args.time_lags, forecast_factor=args.forecast_factor, is_classify=args.classify, confusion_title=args.confusion, is_grid=bool(args.grid), offset=args.offset)
    elif args.task == 5:
        print(args.forecast_factor, args.time_lags)
        # us_pm10 = [344,404,414,439,440,497,500,502,505,523,558,582,583,601,603,612,613,632,640,641,690,762,770,800,801,818,864,908,940,960,992,1013] 
        us_pm10 = [414,603,612,641,864,960,992] 
        # us_pm25 = [31,60,184,250,402,444,470,523,525,558,582,583,586,589,632,641,690,770,800,801,818,864,960,1013]
        us_pm25 = [31,60,184,250,641,690,770,818,864,960,1013]
        if args.forecast_factor:
            evaluate_us(args.url, args.url2, us_pm10, pr.strides, encoder_length=args.time_lags, decoder_length=args.time_lags, forecast_factor=args.forecast_factor, eval_stat=False)
        else:
            evaluate_us(args.url, args.url2, us_pm25, pr.strides, encoder_length=args.time_lags, decoder_length=args.time_lags, forecast_factor=args.forecast_factor, eval_stat=False)
    elif args.task == 6:
        china = [9,32,204,214,228,270,293,328,331,362,363,364,371,389,394,395,396,397,419,426,429,430,456,459,486,581,608,627,639,711,774,818,832,855,928,1018]
        evaluate_china(args.url, args.url2, china, decoder_length=args.time_lags)
    else:
        # train_data
        # pm25: 0.24776679025820308, 0.11997866025609479
        # pm10: 0.13238910415838293, 0.07497394009926399
        # test data 
        # (0.25498451500807523, 0.12531802770836317)
        # (0.12908470683363008, 0.06754419659245953)
        # evaluate_multi(args.url, args.url2, args.time_lag)
        pm10_title = ["Confusion Matrix of ConvLSTM on PM10","Confusion Matrix of APNET Simple on PM10","Confusion Matrix of APNET on PM10","Confusion Matrix of APGAN on PM10", "Confusion Matrix of APGAN Simple on PM10"]
        pm25_title = ["Confusion Matrix of ConvLSTM on PM2.5","Confusion Matrix of APNET Simple on PM2.5","Confusion Matrix of APNET on PM2.5","Confusion Matrix of APGAN on PM2.5", "Confusion Matrix of APGAN Simple on PM2.5"]
        pm10 = ["cnn_simple6_pm10","apnet_noatt_pm10","apnet_lstm_pm10_1544774761","apgan77gau_apnet", "apgannoattgau1_apnet_noatt"]
        pm25 = ["cnn_simple_6_48","apnet_noatt_3","apnet_lstm_gen_dp_1544702820_24","apgan77_apnet", "apgannoatt_apnetnoatt"]
        for x, t in zip(pm10, pm10_title):
            print(x, t)
            conf = utils.load_file("results/confusion/confusion_%s" % x)
            draw_confusion_matrix(conf, t, True)
            # evaluate_by_districts("test_sp/apnet_apgan/pm10/" + x, "vectors/sp_china_combined/seoul_test_labels", 2, decoder_length=24, forecast_factor=1, is_classify=True, confusion_title=t, norm=False)
        
        for x, t in zip(pm25, pm25_title):
            print(x, t)
            conf = utils.load_file("results/confusion/confusion_%s" % x)
            draw_confusion_matrix(conf, t, True)
            # evaluate_by_districts("test_sp/apnet_apgan/pm25/" + x, "vectors/sp_china_combined/seoul_test_labels", 2, decoder_length=24, forecast_factor=0, is_classify=True, confusion_title=t, norm=False)
