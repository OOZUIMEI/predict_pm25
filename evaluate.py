import utils
import argparse
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(pred, labs, rg, is_classify=False):
    l = len(pred)
    if is_classify:
        acc = utils.calculate_accuracy(pred, labs, rg, True)
        acc = float(acc) / l * 100
        print("classified accuracy:%.2f" % acc)
    else:
        pred_ = [utils.get_pm25_class(x) for x in pred]
        labs_ = [utils.get_pm25_class(x) for x in labs]
        # precision_score(labs_, pred_, average='weighted')
        cacc = utils.calculate_accuracy(pred_, labs_, 0, True)
        cacc = float(cacc) / l * 100
        print("classified accuracy:%.2f" % cacc)
        # print(test)
        acc = utils.calculate_accuracy(pred, labs, rg, False)
        acc = float(acc) / l * 100
        print("accuracy:%.2f" % acc)
        mae = mean_absolute_error(labs, pred)
        rmse = sqrt(mean_squared_error(labs, pred))
        r2 = r2_score(labs, pred)
        print("mae:%.2f" % mae)
        print("rmse:%.2f" % rmse)
        print("r2_score:%.2f" % r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-r", "--range", type=int, default=5)
    parser.add_argument("-c", "--classify", type=int, default=0)

    args = parser.parse_args()

    a = utils.load_file(args.url, False)
    labs = []
    preds = []
    for i, x in enumerate(a):
        if i:
            x_ = x.rstrip("\n")
            x_ = x_.split(",")
            if x_:
                labs.append(int(x_[-1]))
                preds.append(int(x_[0]))
    evaluate(preds, labs, args.range, args.classify)
