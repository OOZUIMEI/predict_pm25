import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import utils



def process_data(data, time_lags=8):
    data = np.asarray(data, dtype=np.float32)
    dt_len = len(data)
    # 0 is pm25 while 1 is pm10
    pm2_5_ = [float(x) for x in data[:,1]]
    training_data = []
    labels = []
    history = 8
    total = 24
    for x in xrange(0, dt_len, time_lags):
        if x + total <= dt_len:
            dt = np.array(data[x:x+history,:])
            dt_future = np.array(data[x+history:x+total,6:])
            l = pm2_5_[x + total - 1]
            training_data.append(np.concatenate((dt.flatten(), dt_future.flatten())))
            labels.append(l)
    return training_data, labels


def train(url="vectors/sp_seoul_train_bin"):
    data_train = utils.load_file(url)
    dt_len = len(data_train) / 2
    data_train = np.array(data_train[dt_len:])
    svrs = []
    for x in xrange(25):
        data = data_train[:,x,:]
        data, labels = process_data(data)
        svr = SVR()
        svr.fit(data, labels)
        svrs.append(svr)
    return svrs

def test(svrs, url="vectors/sp_seoul_test_bin"):
    data = utils.load_file(url)
    data = np.array(data)
    preds = []
    labels = []
    for x in xrange(25):
        dt = data[:,x,:]
        data_train, lb = process_data(dt, time_lags=4)
        pred = svrs[x].predict(data_train)
        preds.append(pred)
        labels.append(lb)
    return preds, labels


svrs = train()
preds, labels = test(svrs)
# print(preds)
mae = 0.0
rmse = 0.0
for pr, lb in zip(preds, labels):
    mae += mean_absolute_error(lb, pr)
    rmse += sqrt(mean_squared_error(lb, pr))

rmse = rmse * 12
mae = mae * 12
"""
# 8 mae: 19.73 # rmse: 25.83
# 12 mae: 21.29 rmse: 27.72
# 16 mae: 22.38 rmse: 29.04
# 20 mae: 23.21 rmse: 29.92
# 24 mae: 23.81 rmse: 30.59


# pm10 prediction
# 8h mae: 13.72 rmse: 17.42
# 16h mae: 14.97 rmse: 18.83
# 24h mae: 15.74 rmse: 19.63

"""
print("mae: %.2f" % mae)
print("rmse: %.2f" % rmse)