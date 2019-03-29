import os
import numpy as np
from utils import load_file, save_file, array_to_str
from crawling_base import Crawling
from evaluate import get_class, get_aqi_class


def convert_pm(pm, us_pm, r):
    crawling = Crawling()
    pm_ = []
    # extract stations class
    for d in pm:
        pr = []
        for s in us_pm:
            v = d[s] * r
            #pr.append(get_aqi_class(crawling.AQIPM25(v)))
            pr.append(crawling.aqi_pm25_china_class(v))
        pm_.append(pr)

    # extract class for all pixels
    # for d in pm:
    #     pr = []
    #     for idx in d:
    #         pr.append(get_aqi_class(crawling.AQIPM25(idx)))
    #     pm_.append(pr)
    return pm_



def convert_file(f, stations=[], factor=0, norm=0):
    data = load_file(f)
    print("input", np.shape(data))
    data = np.array(data)
    pm = np.reshape(data[:,:,:,factor], [len(data), 1024])
    pm_ = convert_pm(pm, stations, norm)
    return pm_

beijing = [9,32,204,214,228,270,293,328,331,362,363,364,371,389,394,395,396,397,419,426,429,430,456,459,486,581,608,627,639,711,774,818,832,855,928,1018]
seoul = [496,463,530,529,529,434,440,437,405,601,564,565,659,658,852,684,557,461,460,421,548,265,362,330,207,240,204,341,273,339,212,346,444,412,74,818,358,855,727]
us_pm10 = [344,404,414,439,440,497,500,502,505,523,558,582,583,601,603,612,613,632,640,641,690,762,770,800,801,818,864,908,940,960,992,1013] 
us_pm25 = [31,60,184,250,402,444,470,523,525,558,582,583,586,589,632,641,690,770,800,801,818,864,960,1013]

"""
convert us data to classification
# convert_file("/media/data/us_data/train/test.pkl", "test")
# convert_file("/media/data/us_data/train/us/train_2014.pkl", "2014")
# convert_file("/media/data/us_data/train/us/train_2015.pkl", "2015")
# convert_file("/media/data/us_data/train/us/train_2016.pkl", "2016")
save_file("/media/data/us_data/train/classlb_pm25%s" % suffix, pm25_)
save_file("/media/data/us_data/train/classlb_pm10%s" % suffix, pm10_)
"""
"""
# get labels from aqi grid for seoul
for f in os.listdir("/media/data/datasets/seoul_stations/grid"):
    pm = convert_file("/media/data/datasets/seoul_stations/grid/%s" % f, seoul, 0, 175.0)
    save_file("/media/data/datasets/seoul_stations/%s" % f, pm)
pm = convert_file("/media/data/datasets/china_town/beijing", beijing, 0, 1000.0)
save_file("/media/data/datasets/china_town/beijing_labels", pm)
"""
# pm = convert_file("/media/data/datasets/china_town/beijing_fulldate", beijing, 0, 1000.0)
data = load_file("/media/data/datasets/china_town/beijing_snorm")
print("input", np.shape(data))
data = np.array(data)
pm = np.reshape(data[:,:,:,0], [len(data), 1024])
pm_ = convert_pm(pm, beijing, 300.0)
save_file("/media/data/datasets/china_town/beijing_labels_cn", pm_)
