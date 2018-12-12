import utils
import argparse
import numpy as np
import heatmap
from math import sqrt
import  utils
from crawling_base import Crawling  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        lb_i = i * 4 + 24
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
        lb_i = i * 4 + 24
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
        lb_i = i * 4 + time_lags + 1
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
def evaluate_transportation(url, url2):
    preds = utils.load_file(url)
    preds = np.array(preds)
    lt = len(preds)
    labels = utils.load_file(url2)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 32, 32)
    shape = np.shape(preds)
    # pred_length = shape[1]
    pred_length = 1
    loss_mae0 = 0.0
    loss_rmse0 = 0.0
    r2_total = 0.0
    for i, d in enumerate(preds):
        lb_i = i + 8
        mae0, mse0, r2 = get_evaluation(d[:pred_length,:,:], labels[lb_i:(pred_length+lb_i),:,:])
        # mae0, mse0, r2 = get_evaluation(d[0,:,:], labels[lb_i,:,:])
        loss_rmse0 += mse0 
        loss_mae0 += mae0
        r2_total += r2
    loss_mae0 = loss_mae0 / lt * 131
    loss_rmse0 = sqrt(loss_rmse0 / lt) * 131
    r2_total = r2_total / lt
    print("MAE: %.6f" % loss_mae0)
    print("RMSE: %.6f" % loss_rmse0)
    print("R2 Score: %.6f" % r2_total)


# evaluate grid training
def evaluate_lstm(url, url2, decoder_length=24):
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
    loss_mae = 0.0
    loss_rmse = 0.0
    r2_total = 0.0
    for i, d in enumerate(data):
        if decoder_length < data.shape[-1]:
            pred_t = d[:decoder_length]
        pred_t = pred_t.flatten()
        lb_i = i * 4 + 24
        lbt = np.mean(labels[lb_i:(lb_i+decoder_length),:,0], axis=1)
        mae, mse, r2 = get_evaluation(pred_t, lbt)
        loss_mae += mae
        loss_rmse += mse
        r2_total += r2
        utils.update_progress((i + 1.0) / dtl)
    loss_mae = loss_mae / lt * 300
    loss_rmse = sqrt(loss_rmse / lt) * 300
    r2_total = r2_total / lt
    print("MAE: %.6f" % loss_mae)
    print("RMSE: %.6f" % loss_rmse)
    print("R2 score: %.6f" % r2_total)


def get_evaluation(pr, lb):
    pr = pr.flatten()
    lb = lb.flatten()
    mse = mean_squared_error(pr, lb)
    mae = mean_absolute_error(pr, lb)
    r2 = r2_score(lb, pr)
    return mae, mse, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="predictions file path")
    parser.add_argument("-l", "--url2", help="labels file path")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)
    parser.add_argument("-t", "--task", type=int, default=0)
    parser.add_argument("-g", "--grid", type=int, default=1)
    parser.add_argument("-gev", "--grid_eval", type=int, default=1)
    parser.add_argument("-tl", "--time_lags", type=int, default=24)

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
        evaluate_transportation(args.url, args.url2)
    elif args.task == 3:
        evaluate_lstm(args.url, args.url2, args.time_lags)
    else:
        # train_data
        # pm25: 0.24776679025820308, 0.11997866025609479
        # pm10: 0.13238910415838293, 0.07497394009926399
        # test data 
        # (0.25498451500807523, 0.12531802770836317)
        # (0.12908470683363008, 0.06754419659245953)
        evaluate_multi(args.url, args.url2, args.time_lag)