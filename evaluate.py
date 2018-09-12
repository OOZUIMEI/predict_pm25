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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)

    args = parser.parse_args()

    # a = utils.load_file(args.url, False)
    # labs = []
    # preds = []
    # for i, x in enumerate(a):
    #     if i:
    #         x_ = x.rstrip("\n")
    #         x_ = x_.split(",")
    #         if x_:
    #             labs.append(int(x_[-1]))
    #             preds.append(int(x_[0]))
    # evaluate(preds, labs, args.range, args.classify)
    map_ = heatmap.build_map()
    data = utils.load_file("test_sp/gan_cuda_m")
    data = np.reshape(data, (data.shape[0], data.shape[1], 25, 25))
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
        loss += mean_absolute_error(lbg, pred_t)
        # utils.update_progress(i * 1.0 / dtl)
    loss = loss / dtl * 300
    print(loss)
