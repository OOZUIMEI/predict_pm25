import utils
import argparse
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-r", "--range", type=int, default=5)

    args = parser.parse_args()

    a = utils.load_file(args.url, False)
    labs = []
    pred = []
    for i, x in enumerate(a):
        if i:
            x_ = x.rstrip("\n")
            x_ = x_.split(",")
            if x_:
                labs.append(int(x_[-1]))
                pred.append(int(x_[0]))
    l = len(pred)
    acc = utils.calculate_accuracy(pred, labs, args.range, False)
    acc = float(acc) / l * 100
    pred_ = [utils.get_pm25_class(x) for x in pred]
    labs_ = [utils.get_pm25_class(x) for x in labs]
    cacc = utils.calculate_accuracy(pred_, labs_, 0, True)
    cacc = float(cacc) / l * 100
    mae = mean_absolute_error(labs, pred)
    rmse = sqrt(mean_squared_error(labs, pred))
    print("classified accuracy:%.2f" % cacc)
    print("accuracy:%.2f" % acc)
    print("mae:%.2f" % mae)
    print("rmse:%.2f" % rmse)