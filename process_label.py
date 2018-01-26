import utils
from datetime import datetime


prefix = "/home/alex/Documents/datasets/airkorea"
file = prefix + "/aqi_01_2018"
start = datetime.strptime("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
tm = (start - datetime(1970,1,1)).total_seconds()
data = utils.load_file(file, False)
arr = []
for i, x in enumerate(data):
    t = tm + 3600 * i
    s = datetime.utcfromtimestamp(t)
    s_ = s.strftime("%Y-%m-%d %H:%M:%S")
    arr.append(s_ + "," + x.rstrip("\n"))
tmp = utils.array_to_str(arr)
utils.save_file("pm25_01.csv", tmp, False)
    