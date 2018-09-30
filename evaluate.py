import utils
import argparse
import numpy as np
import heatmap
from math import sqrt
import  utils
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(pred, labs, rg, is_classify=False, verbose=True):
    l = len(pred)
    if is_classify:
        acc = utils.calculate_accuracy(pred, labs, rg, True)
        acc = float(acc) / l * 100
        print("classified accuracy:%.2f" % acc)
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
            print("classified accuracy:%.2f" % cacc)
            print("accuracy:%.2f" % acc)
            print("mae:%.2f" % mae)
            print("rmse:%.2f" % rmse)
            print("r2_score:%.2f" % r2)


def evaluate_sp():
    map_ = heatmap.build_map()
    data = utils.load_file("test_sp/gan_case4")
    if type(data) is list:
        data = np.asarray(data)
    if len(data.shape) == 4:
        lt = data.shape[0] * data.shape[1]
    else:
        lt = data.shape[0]
    data = np.reshape(data, (lt, data.shape[-2], 25, 25))
    labels = utils.load_file("vectors/sp_seoul_test_bin")
    labels = np.asarray(labels)
    dtl = len(data)
    loss = 0.0
    for i, d in enumerate(data):
        lb_i = i * 4 + 24
        lbt = labels[lb_i:(lb_i+24),:,0]
        lbg = []
        for x in lbt:
            x_l = heatmap.fill_map(x, map_)
            lbg.append(x_l)
        lbg = np.asarray(lbg)
        lbg = lbg.flatten()
        pred_t  = []
        for d_ in d:
            d_t = heatmap.clear_interpolate_bound(np.asarray(d_), map_)
            pred_t.append(d_t)
        pred_t = np.asarray(pred_t)
        pred_t = pred_t.flatten()
        loss += mean_squared_error(lbg, pred_t)
        # utils.update_progress(i * 1.0 / dtl)
    loss = loss / dtl * 300
    print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="predictions file path")
    parser.add_argument("-l", "--url2", help="labels file path")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)

    args = parser.parse_args()
    preds = utils.load_file(args.url)
    labels = utils.load_file(args.url2)
    loss = 0.0
    for i, d in enumerate(preds):
        lb_i = i * 4 + 24
        lbt = labels[lb_i:(lb_i+24),:,0]
        lbg = np.array(lbt)
        lbg = lbg.flatten()
        pred_t = d.flatten()
        loss += mean_squared_error(lbg, pred_t)
    print(loss)