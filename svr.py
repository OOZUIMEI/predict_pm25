import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import utils
import argparse
import evaluate as ev
import aqi


def process_data(data, time_lags=8, history = 16, future = 12, forecast_factor=0):
    data = np.asarray(data, dtype=np.float32)
    dt_len = len(data)
    # 0 is pm25 while 1 is pm10
    pm2_5_ = [float(x) for x in data[:,forecast_factor]]
    training_data = []
    labels = []
    total = history + future
    for x in xrange(0, dt_len, time_lags):
        if x + total <= dt_len:
            dt = np.array(data[x:x+history,:])
            dt_future = np.array(data[x+history:x+total,6:])
            l = pm2_5_[x + total - 1]
            training_data.append(np.concatenate((dt.flatten(), dt_future.flatten())))
            labels.append(l)
    return training_data, labels


def train(url="vectors/sp_china_combined/sp_seoul_train_bin", future=12, forecast_factor=0):
    data_train = utils.load_file(url)
    dt_len = len(data_train) / 2
    data_train = np.array(data_train[dt_len:])
    svrs = []
    for x in xrange(25):
        data = data_train[:,x,:]
        data, labels = process_data(data, future=future, forecast_factor=forecast_factor)
        svr = SVR()
        svr.fit(data, labels)
        svrs.append(svr)
    return svrs

def test(svrs, url="vectors/sp_china_combined/sp_seoul_test_bin", future=12, forecast_factor=0):
    data = utils.load_file(url)
    data = np.array(data)
    preds = []
    labels = []
    for x in xrange(25):
        dt = data[:,x,:]
        data_train, lb = process_data(dt, time_lags=4, future=future, forecast_factor=forecast_factor)
        pred = svrs[x].predict(data_train)
        preds.append(pred)
        labels.append(lb)
    return preds, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--factor", type=int, default=0)
    parser.add_argument("-tl", "--time_lags", type=int, default=12)
    args = parser.parse_args()
    
    svrs = train(future=args.time_lags, forecast_factor=args.factor)
    preds, labels = test(svrs, future=args.time_lags, forecast_factor=args.factor)
    utils.save_file("svr_%i_%i" % (args.factor, args.time_lags), preds)
    lt = len(preds)
    # print(np.shape(preds))
    # print(preds)
    mae = 0.0
    rmse = 0.0
    r2_total = 0.0
    acc = 0.
    for pr, lb in zip(preds, labels):
        mae += mean_absolute_error(lb, pr)
        rmse += sqrt(mean_squared_error(lb, pr))
        r2_total += r2_score(lb, pr)
        acc += ev.classify_data(lb, pr, args.factor)

    rmse = rmse * 12
    mae = mae * 12
    # r2_total = r2_total / 25
    acc = acc / lt

    if args.factor:
        rmse_ = aqi.ConcPM10(rmse)
        mae_ = aqi.ConcPM10(mae)
    else:
        rmse_ = aqi.ConcPM25(rmse)
        mae_ = aqi.ConcPM25(mae)
            
    print("mae: %.4f - concentration: %.4f" % (mae, mae_))
    print("rmse: %.4f - concentration: %.4f" % (rmse, rmse_))
    # print("r2 score: %.4f" % r2_total)
    print("accuracy: %.4f" % acc)
