import time


file_path = "/home/alex/Documents/datasets/timestamp_60.csv"
file_path2 = "/home/alex/Documents/datasets/tsp_60.txt"


with open(file_path, 'rb') as file:
    data = file.readlines()
    tmp = ""
    for x in data:
        x = x.replace("\n", '')
        x = float(x)/1000000
        t = time.localtime(x)
        tmp += time.strftime('%Y-%m-%d %H:%M:%S', t)
        tmp += '\n'


with open(file_path2, 'wb') as file:
    file.write(tmp)
