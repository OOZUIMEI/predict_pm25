from pyspark.sql.column import Column, _to_java_column, _to_seq
import properties as p

def fix_float(x):
    res = 0.0
    try:
        if x and x != "-" and x != "&nbsp" and x != "":
            res = float(x)
    except Exception as e:
        print(x)
    return res


def fix_int(x):
    res = 0
    if x and x != "&nbsp" and x != "-" and x != "":
        res = int(x)
    return res


def get_district_code(x):
    return p.district_codes.index(x) + 1


def normalize_pm(x):
    res = 0.0
    if x > 300:
        res = 1.0 
    else:
        res = float(x) / 300
    return res


def normalize_agl(x):
    if x:
        return x / 360
    return 0


def get_holiday(x, date):
    if x == "Sat" or x == "Sun" or date in p.holidays: 
        return 1
    return 0


def  get_holiday_china(x, date):
    if x == "Sat" or x == "Sun" or date in p.china_holidays: 
        return 1
    return 0


# def humidity(h): 
#     if h:
#         return h / 100 
#     return 0.0


def wdm(col):
    #https://stackoverflow.com/questions/33233737/spark-how-to-map-python-with-scala-or-java-user-defined-functions
    #https://stackoverflow.com/questions/36023860/how-to-use-a-scala-class-inside-pyspark
    wdm = sc._jvm.com.prediction.udf.WindDirectionMean.apply
    return Column(wdm(_to_seq(sc, [col], _to_java_column)))


def convert_dir_to_agl(d):
    agl = 0.0
    agl_arr  = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    dir_arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    if d:
        idx = dir_arr.index(d)
        if idx > -1 and idx < len(agl_arr):
            agl = agl_arr[idx]
    return agl / 360


def humidity(d):
    d_ = 0.0
    if isinstance(d, str):
        try:
            d_ = float(d) / 100
        except Exception as e:
            print(e, d)
    elif d:
        d_ = d * 1.0 / 100
    return d_
