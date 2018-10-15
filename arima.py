from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import utils

#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
url_train = "vectors/sp_seoul_train_bin"
url_test = "vectors/sp_seoul_test_bin"

data_train = utils.load_file(url_train)
data_train = np.array(data_train)

pm2_5_train = np.mean(data_train[-6000:,:,0], axis=1)

data_test = utils.load_file(url_test)
data_test = np.array(data_test)
pm2_5_test = data_test[:,:,0]


pm2_5_test = np.mean(pm2_5_test, axis=1)
series = np.concatenate((pm2_5_train, pm2_5_test), axis=0)
lt = len(series)
train_data = series[0:8000]
mae = 0.0
rmse = 0.0
model = ARIMA(train_data, order=(8, 1, 1))
model_fit = model.fit(disp=0)
outputs,_,_ = model_fit.forecast(steps=10000)
outputs = [x if x >= 0 else 0.0 for x in outputs]
labels = pm2_5_test[2000:12000]
mae += mean_absolute_error(labels, outputs) 
mse = mean_squared_error(labels, outputs)
rmse += sqrt(mse)

rmse = rmse * 300
mae = mae * 300
print("mae: %.2f" % mae)
print("rmse: %.2f" % rmse)

# # residuals = DataFrame(outputs)
# # residuals.plot()
# # pyplot.show()