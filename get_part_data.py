import os
import numpy as np
import utils

# data1 = utils.load_file("vectors/sp_china_combined/seoul_16")
# data1 = np.array(data1)
# print(data1.shape)

# data1 = utils.load_file("vectors/sp_china_combined/seoul_1")
# data2 = utils.load_file("vectors/sp_china_combined/china_1")
# data1 = np.array(data1)
# data2 = np.array(data2)

# #data1 = data1[-43848:-26304,:,:]
# #data2 = data2[-43848:-26304,:]

# data11 = data1[:8760,:,:,:]
# data21 = data2[:8760,:]

# utils.save_file("vectors/sp_china_combined/seoul_14", data11.tolist())
# utils.save_file("vectors/sp_china_combined/china_14", data21.tolist())

# data11 = data1[8760:17520,:,:,:]
# data21 = data2[8760:17520,:]

# utils.save_file("vectors/sp_china_combined/seoul_15", data11.tolist())
# utils.save_file("vectors/sp_china_combined/china_15", data21.tolist())

# data11 = data1[17520:,:,:,:]
# data21 = data2[17520:,:]

# utils.save_file("vectors/sp_china_combined/seoul_16", data11.tolist())
# utils.save_file("vectors/sp_china_combined/china_16", data21.tolist())

"""# process shandong data

data = utils.load_file("/media/data/datasets/shandong/shandong_norm.csv", False)
res = []
for x in data:
    x = x.rstrip("\n")
    x = x.split(",")
    x_ = [float(d) for d in x]
    res.append(x_)

res = np.array(res)
print(res.shape)
r2014 = res[:8760,:] 
r2015 = res[8760:17520,:]
r2016 = res[17520:26304,:]
r2017 = res[26304:35064,:]
r2018 = res[35064:,:]

c2014 = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/china/china_14")
c2015 = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/china/china_15")
c2016 = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/china/china_16")
c2017 = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/china_test")
c2018 = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/china_valid")

c2014 = np.concatenate([c2014, r2014], axis=1)
c2015 = np.concatenate([c2015, r2015], axis=1)
c2016 = np.concatenate([c2016, r2016], axis=1)
c2017 = np.concatenate([c2017, r2017], axis=1)
c2018 = np.concatenate([c2018, r2018], axis=1)

print(c2014.shape, c2015.shape, c2016.shape, c2017.shape, c2018.shape)
utils.save_file("vectors/sp_china_combined/china_/china_2014", c2014.tolist())
utils.save_file("vectors/sp_china_combined/china_/china_2015", c2015.tolist())
utils.save_file("vectors/sp_china_combined/china_/china_2016", c2016.tolist())
utils.save_file("vectors/sp_china_combined/china_/china_2017", c2017.tolist())
utils.save_file("vectors/sp_china_combined/china_/china_2018", c2018.tolist())

"""
"""
## process seoul data (weather 2014010101 -2018010101)
weather = utils.load_file("/media/data/datasets/seoul_stations/weather")
weather = np.array(weather)

r2015 = weather[8760:17520,:]
r2016 = weather[17520:26304,:]
r2017 = weather[26304:,:]

print(r2015.shape, r2016.shape, r2017.shape)
utils.save_file("/media/data/datasets/seoul_stations/weather_2015", r2015.tolist())
utils.save_file("/media/data/datasets/seoul_stations/weather_2016", r2015.tolist())
utils.save_file("/media/data/datasets/seoul_stations/weather_2017", r2015.tolist())

"""
"""
# transpose grid data
# path = "/media/data/datasets/seoul_stations/grid"
# for f in os.listdir(path):
#     dat = utils.load_file("%s/%s" % (path, f))
#     dat = np.transpose(dat, [0, 2, 3, 1])
#     utils.save_file("/media/data/datasets/seoul_stations/%s" % f, dat.tolist())

dat = utils.load_file("/media/data/datasets/china_town/beijing")
dat = np.transpose(dat, [0, 2, 3, 1])
utils.save_file("/media/data/datasets/china_town/beijing_", dat.tolist())
"""

# def process_data_china(dtlength, batch_size, encoder_length, decoder_length):
#     indices = np.asarray(range(dtlength), dtype=np.int32)
#     train_, valid_ = [], []
#     for x in xrange(12):
#         st = x * 720
#         train = indices[st:st+480 - encoder_length - decoder_length].tolist()
#         valid = indices[st+480:st+720 - encoder_length -  decoder_length].tolist()
#         print("train",train)
#         print("valid", valid)
#         train_ += train
#         valid_ += valid
#     print(np.shape(train_), np.shape(valid_))
#     return train_, valid_

# process_data_china(8670, 32, 48, 48)

# dat = utils.load_file("/media/data/datasets/china_town/beijing_snorm")
# dat = utils.load_file("/media/data/datasets/china_town/beijing_weather")
# dat = utils.load_file("/media/data/datasets/china_town/beijing_labels")
# dat = np.array(dat)
# months = [31,30,31,31,30,31,30,31,31,28,31,30]
# for i, m in enumerate(months):
#     ed = m * 24
#     if i == 0:
#         st = 0
#     else:
#         st = np.sum(months[:i]) * 24
#         ed += st
#     # print(st, ed)
#     # d = dat[st:ed,:,:,:]
#     d = dat[st:ed,:]
#     mname = (i + 5) % 12
#     if mname == 0:
#         mname = 12
#     print("saving:", mname)
#     utils.save_file("/media/data/datasets/china_town/beijing_%i" % mname, d)


# dat = utils.load_file("/media/data/datasets/china_town/test/weather/beijing_3")
# dat = np.array(dat)
# print(dat.shape)
# print(dat[10,:])

dat = utils.load_file("/media/data/pm25_working/vectors/sp_china_combined/dat2017")
# print(dat)