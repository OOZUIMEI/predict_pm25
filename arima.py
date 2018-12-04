from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import utils

#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# url_train = "vectors/sp_seoul_train_bin"
url_test = "vectors/sp_china_combined/sp_seoul_test_bin"

# data_train = utils.load_file(url_train)
# data_train = np.array(data_train)

# pm2_5_train = np.mean(data_train[-1000:,:,0], axis=1)

data_test = utils.load_file(url_test)
data_test = np.array(data_test)
pm2_5_test = data_test[:,:,1]


pm2_5_test = np.mean(pm2_5_test, axis=1)
pm2_5_train = pm2_5_test[:500]
pm2_5_test = pm2_5_test[500:]
print(len(pm2_5_test))
# series = np.concatenate((pm2_5_train, pm2_5_test), axis=0)
# lt = len(series)
# train_data = pm2_5_train[:]
total_results = 500
prediction_steps = 25
running_steps = total_results / prediction_steps - 1

model = ARIMA(pm2_5_train, order=(3, 1, 1))
model_fit = model.fit(disp=0)
outputs,_,_ = model_fit.forecast(steps=prediction_steps)

preds = outputs[:]

for x in xrange(running_steps):
    pm2_5_train = pm2_5_train[prediction_steps:]
    pm2_5_train = np.concatenate((pm2_5_train, outputs), axis=0)
    model = ARIMA(pm2_5_train, order=(3, 1, 1))
    model_fit = model.fit(disp=0)
    outputs,_,_ = model_fit.forecast(steps=prediction_steps)
    
    preds = np.concatenate((preds, outputs), axis=0)

preds = [x if x >= 0 else 0.0 for x in preds]
labels = pm2_5_test[0:total_results]
mae = mean_absolute_error(labels, preds) 
mse = mean_squared_error(labels, preds)
rmse = sqrt(mse)
r2_total = r2_score(labels, preds)

rmse = rmse * 300
mae = mae * 300

# mae: 28.43 rmse: 35.41
# mae: 15.81 rmse: 19.84 r2 score: -0.20
print("mae: %.2f" % mae)
print("rmse: %.2f" % rmse)
print("r2 score: %.2f" % r2_total)

# residuals = DataFrame(outputs)
# residuals.plot()
# pyplot.show()