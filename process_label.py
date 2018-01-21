import pickle 


file_path = "/home/alex/Documents/datasets/pm25_f.csv"
file_path2 = "labels.pkl"
file_path3 = "label_values.pkl"

m = 30
pm25range = [(0, 50),(51,100),(101,150),(151,200),(201,300),(301,400),(401,500)]


def get_label2(pm25):
    i, j = 0, 0
    for x, y in pm25range:
        if pm25 >= x and pm25 <= y:
            i = j
            break
        j += 1
    return i


with open(file_path, 'rb') as file:
    data = file.readlines()
    i = 0
    total = 0
    avg = 0.0
    arr = []
    arr_v = []
    for x in data:
        i += 1
        dt = x.replace("\n", "").split(",")
        if dt:
            if i == m:
                i = 0
                avg = float(total) / m
                lb = get_label2(avg)
                arr_v.append(int(round(avg)))
                arr.append(lb)
                total = 0
            total += float(dt[-1])
    if i:
        avg = float(total) / i
        lb = get_label2(avg)
        arr_v.append(int(round(avg) - 1))
        arr.append(lb)
    del arr[0]
    del arr_v[0]
    print(arr)
    # print("arrv", arr_v)

with open(file_path2, 'wb')  as file:
    pickle.dump(arr, file, pickle.HIGHEST_PROTOCOL)

with open(file_path3, 'wb')  as file:
    pickle.dump(arr_v, file, pickle.HIGHEST_PROTOCOL)