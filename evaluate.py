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


# evaluate grid training
def evaluate_sp(url, url2, is_grid=True, grid_eval=True):
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
    dtl = len(data)
    loss_mae = 0.0
    loss_rmse = 0.0
    for i, d in enumerate(data):
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
        lbt = labels[lb_i:(lb_i+24),:,1]
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
        mae, mse = get_evaluation(pred_t, lbg)
        loss_mae += mae
        loss_rmse += mse
        utils.update_progress((i + 1.0) / dtl)
    loss_mae = loss_mae / lt * 300
    loss_rmse = sqrt(loss_rmse / lt) * 300
    print("MAE: %.2f" % loss_mae)
    print("RMSE: %.2f" % loss_rmse)


# predict multiple dimension
# pm2.5, pm10
def evaluate_multi(url, url2, time_lags=24):
    preds = utils.load_file(url)
    preds = np.array(preds)
    st = 0
    preds = preds[st:,:,:,:]
    lt = len(preds)
    labels = utils.load_file(url2)
    labels = np.array(labels)

    loss_mae0, loss_mae1 = 0.0, 0.0
    loss_rmse0, loss_rmse1 = 0.0, 0.0
    m = 0
    for i, d in enumerate(preds):
        lb_i = (st + i) * 4 + 25
        
        mae0, mse0 = get_evaluation(d[:time_lags,:,0], labels[lb_i:(lb_i+time_lags),:,0])
        mae1, mse1 = get_evaluation(d[:time_lags,:,1], labels[lb_i:(lb_i+time_lags),:,1])
        loss_rmse0 += mse0 
        loss_rmse1 += mse1
        loss_mae0 += mae0
        loss_mae1 += mae1
        # print(d[:time_lags,:,0], labels[lb_i-25:(lb_i-25+time_lags),:,0])
        if (mae0 * 300) < 30:
            m += 1
            # print("11",i)
        # break
    loss_mae0 = loss_mae0 / lt * 300
    loss_mae1 = loss_mae1 / lt * 300
    loss_rmse0 = sqrt(loss_rmse0 / lt) * 300
    loss_rmse1 = sqrt(loss_rmse1 / lt) * 300
    print("MAE PM2.5: %.2f" % loss_mae0)
    print("RMSE PM2.5: %.2f" % loss_rmse0)
    print("MAE PM10: %.2f" % loss_mae1)
    print("RMSE PM10: %.2f" % loss_rmse1)
    labels0 = labels[:,:,0].flatten()
    labels1 = labels[:,:,1].flatten()
    std0 = np.std(labels0)
    std1 = np.std(labels1)
    m0 = np.mean(labels0)
    m1 = np.mean(labels1)
    print(m0, std0)
    print(m1, std1)

def get_evaluation(pr, lb):
    pr = pr.flatten()
    lb = lb.flatten()
    mse = mean_squared_error(pr, lb)
    mae = mean_absolute_error(pr, lb)
    return mae, mse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="predictions file path")
    parser.add_argument("-l", "--url2", help="labels file path")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)
    parser.add_argument("-t", "--task", type=int, default=0)
    parser.add_argument("-g", "--grid", type=int, default=0)
    parser.add_argument("-gev", "--grid_eval", type=int, default=0)
    parser.add_argument("-tl", "--time_lags", type=int, default=24)

    args = parser.parse_args()
    if args.task == 0:
        # ADAIN: MAE: 75.96 RMSE: 79.24
        # SAE MAE: 78.66 RMSE: 89.31
        # Neural nets: MAE: 61.27 RMSE: 66.95
        evaluate_sp(args.url, args.url2, bool(args.grid), bool(args.grid_eval))
    else:
        # train_data
        # pm25: 0.24776679025820308, 0.11997866025609479
        # pm10: 0.13238910415838293, 0.07497394009926399
        # test data 
        # (0.25498451500807523, 0.12531802770836317)
        # (0.12908470683363008, 0.06754419659245953)
        evaluate_multi(args.url, args.url2)
