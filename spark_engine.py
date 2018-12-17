from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from datetime import timedelta
import numpy as np
import properties as p
import utils
import udf as udf_utils

class SparkEngine():

    def __init__(self):
        self.features = ["o3_val","no2_val","co_val","so2_val", "temp", "precip", "humidity", "wind_sp", "wind_gust"]
        self.w_features = ["temp", "precip", "humidity", "wind_sp", "wind_gust"]
        self.spark = self.init_spark()
        self.assembler = VectorAssembler(inputCols=["pm2_5_norm", "pm10_norm", "features", "wind_agl"], outputCol="all_features")
        # init udf function
        self.udf_float = udf(udf_utils.fix_float)
        self.udf_int = udf(udf_utils.fix_int)
        self.udf_district_code = udf(udf_utils.get_district_code)
        self.udf_pm = udf(udf_utils.normalize_pm)
        self.udf_agl = udf(udf_utils.normalize_agl)
        self.udf_dir = udf(udf_utils.convert_dir_to_agl)
        self.udf_humidity = udf(udf_utils.humidity)
        self.udf_holiday = udf(udf_utils.get_holiday)
        self.udf_china_holiday = udf(udf_utils.get_holiday_china)
        self.sp_schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("district_code", IntegerType(), True),
            StructField("pm10_val", DoubleType(), True),
            StructField("pm2_5_val", DoubleType(), True),
            StructField("o3_val", DoubleType(), True),
            StructField("no2_val", DoubleType(), True),
            StructField("co_val", DoubleType(), True),
            StructField("so2_val", DoubleType(), True),
            StructField("pm10_aqi", DoubleType(), True),
            StructField("pm2_5_aqi", DoubleType(), True)
        ])
        self.sw_schema = StructType([
                StructField("timestamp", StringType(), True),
                StructField("district_code", IntegerType(), True),
                StructField("wind_agl", StringType(), True),
                StructField("wind_dir", StringType(), True),
                StructField("wind_sp", StringType(), True),
                StructField("wind_gust", StringType(), True),
                StructField("temp", StringType(), True),
                StructField("precip", StringType(), True),
                StructField("is_rain", StringType(), True),
                StructField("humidity", StringType(), True)
            ])
        self.sw_pred = StructType([
            StructField("timestamp", StringType(), True),
            StructField("forecast", StringType(), True),
            StructField("temp", IntegerType(), True),
            StructField("feel_like", IntegerType(), True),
            StructField("wind_sp", StringType(), True),
            StructField("wind_dir", StringType(), True),
            StructField("wind_gust", StringType(), True),
            StructField("cloud", IntegerType(), True),
            StructField("humidity", IntegerType(), True),
            StructField("precip", DoubleType(), True),
            StructField("pressure", IntegerType(), True),
            StructField("city", StringType(), True)
        ])
        self.aqi_schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("city", StringType(), True),
            StructField("pm2_5", StringType(), True)
        ])
        # the order of vectors
        self.features = ["pm2_5_norm","pm10_norm","o3_val","no2_val","co_val","so2_val","temp","precip","wind_sp","wind_gust","wind_agl","humidity","is_holiday","hour","month"]
    

    def init_spark(self):
        conf = (SparkConf().setAppName("prediction_airpollution"))
        conf.set("spark.driver.memory", "2g")
        conf.set("spark.executor.memory", "1g")
        conf.set("spark.ui.port", "31040")
        conf.set("spark.sql.shuffle.partitions", "200")
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        return spark

    def normalize_vector(self, df_data, cols, orb = "timestamp"):
        vectorAssembler = VectorAssembler(inputCols=cols, outputCol="features_norm")
        mxm = MinMaxScaler(inputCol="features_norm", outputCol="features")
        output = vectorAssembler.transform(df_data)
        mxm_scaler = mxm.fit(output)
        trans = mxm_scaler.transform(output)
        return trans

    def process_vectors(self, start=None, end=None, dim=12):
        # engine do something
        p1 = "data/seoul_aqi.csv"
        p2 = "data/seoul_aws.csv"
        p3 = "data/weather_forecasts.csv"
        p4 = "data/aqicn.csv"
        
        air = self.spark.read.format("csv").option("header", "false").schema(self.sp_schema).load(p1) \
                    .select("timestamp","district_code","pm10_val","pm2_5_val","o3_val","no2_val","co_val","so2_val","pm10_aqi","pm2_5_aqi",
                    self.udf_pm(col("pm2_5_aqi")).cast("double").alias("pm2_5_norm"), self.udf_pm(col("pm10_aqi")).cast("double").alias("pm10_norm"))
        aws = self.spark.read.format("csv").option("header", "false").schema(self.sw_schema).load(p2) \
                    .select(col("timestamp"),"district_code", self.udf_int(col("wind_agl")).alias("wind_agl"), "wind_dir", self.udf_float(col("wind_sp")).alias("wind_sp"),
                        self.udf_float(col("wind_gust")).alias("wind_gust"), self.udf_float(col("temp")).alias("temp"),self.udf_float(col("precip")).alias("precip"),
                        self.udf_float(col("humidity")).alias("humidity"))
        
        aws_mean = aws.withColumn("timestamp", ((unix_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss") / 3600).cast("long") * 3600).cast("timestamp")) \
                    .filter(col("timestamp").isNotNull()) \
                    .select("timestamp", "district_code", "wind_agl", self.udf_district_code(col("district_code")).alias("code"), "wind_sp", "wind_gust", "temp", "precip", "humidity") \
                    .groupBy("code", "timestamp") \
                    .agg(avg("wind_agl").alias("wind_agl"), avg("wind_sp").alias("wind_sp"), avg("wind_gust").alias("wind_gust"), \
                        avg("temp").alias("temp"), avg("precip").alias("precip"), avg("humidity").alias("humidity"))
        
        merge = air.filter(col("district_code") != 0) \
                    .join(aws_mean, [air.timestamp == aws_mean.timestamp, air.district_code == aws_mean.code], "left_outer") \
                    .withColumn("is_holiday", self.udf_holiday(date_format(air.timestamp, "E"), air.timestamp)) \
                    .withColumn("hour", hour(air.timestamp).cast("double") / 23) \
                    .withColumn("month", (month(air.timestamp).cast("double") - 1) / 12) \
                    .select(air.timestamp, air.district_code, "o3_val", "no2_val", "co_val", "so2_val", "is_holiday", "hour", "month",\
                            "pm2_5_norm", "pm10_norm", "temp", "precip", self.udf_humidity(col("humidity")).alias("humidity"), 
                            self.udf_agl(col("wind_agl")).cast('double').alias("wind_agl"), "wind_sp", "wind_gust") \
                    .na.fill(0.0, ["o3_val","no2_val","co_val","so2_val", "temp", "precip", "humidity", "wind_sp", "wind_gust", "pm2_5_norm", "pm10_norm", "wind_agl"])
       
        w_pred = self.spark.read.format("csv").option("header", "false").schema(self.sw_pred).load(p3) \
                     .groupBy("timestamp", "city").agg(last("temp").alias("temp"),last("precip").alias("precip"), \
                            last("humidity").alias("humidity"),last("wind_sp").alias("wind_sp"), \
                            last("wind_gust").alias("wind_gust"),last("wind_dir").alias("wind_dir")) \
                     .select("timestamp", "temp", "precip", "humidity", "wind_sp", "wind_gust", self.udf_dir("wind_dir").cast("double").alias("wind_agl"), "city")
        aqicn = self.spark.read.format("csv").option("header", "false").schema(self.aqi_schema).load(p4) \
                    .groupBy("timestamp", "city").agg(last("pm2_5").alias("pm2_5")) \
                    .select("timestamp", "city", "pm2_5")
        
        if start and end:
            st_ = start.strftime(p.fm)
            ed_ = end.strftime(p.fm)
            merge = merge.filter((col("timestamp") >= st_) & (col("timestamp") <= ed_))
            end = end + timedelta(days=1)
            ed2_ = end.strftime(p.fm)
        
        else:
            st_, ed_, ed2_ = None, None, None
        final = merge.groupBy("timestamp") \
                     .agg(collect_list("district_code").alias("district_code"), \
                        collect_list("o3_val").alias("o3_val"), \
                        collect_list("no2_val").alias("no2_val"), \
                        collect_list("co_val").alias("co_val"), \
                        collect_list("so2_val").alias("so2_val"), \
                        collect_list("is_holiday").alias("is_holiday"), \
                        collect_list("hour").alias("hour"), \
                        collect_list("month").alias("month"), \
                        collect_list("pm2_5_norm").alias("pm2_5_norm"), \
                        collect_list("pm10_norm").alias("pm10_norm"), \
                        collect_list("temp").alias("temp"), \
                        collect_list("precip").alias("precip"), \
                        collect_list("humidity").alias("humidity"), \
                        collect_list("wind_agl").alias("wind_agl"), \
                        collect_list("wind_sp").alias("wind_sp"), \
                        collect_list("wind_gust").alias("wind_gust")) \
                     .orderBy("timestamp").limit(24)
        final = final.collect()
        res = []
        mx_val = np.array(p.max_values)
        mn_val = np.array(p.min_values)
        del_val = mx_val - mn_val
        
        for d in final:
            dis_vectors = [np.zeros(dim, dtype=np.float).tolist()] * 25
            dis = d["district_code"]
            values = np.array([d[x] for x in self.features[2:10]])
            values = np.transpose(values)
            # min-max standardize for 2->9th element
            values = (values - mn_val) / del_val
            values_norm_1 = np.transpose(np.array([d[x] for x in self.features[:2]]))
            values_norm_2 = np.transpose(np.array([d[x] for x in self.features[10:]]))
            # concatenate to a single vector
            values = np.concatenate((values_norm_1, values, values_norm_2), axis=1)
            for i, x in enumerate(dis):
                idx = int(x) - 1
                dis_vectors[idx] = [float(y) if float(y) > 0 and float(y) < 1 else (1 if float(y) >= 1 else 0) for y in values[i]]
            res.append(dis_vectors)     
        
        # future forecast of seoul
        seoul_w_pred = w_pred.withColumn("is_holiday", self.udf_holiday(date_format(col("timestamp"), "E"), col("timestamp"))) \
                             .withColumn("hour", hour(col("timestamp")).cast("double") / 23) \
                             .withColumn("month", (month(w_pred.timestamp).cast("double") - 1) / 12) \
                             .filter(col("city") == "seoul")
        if not ed_ is None and not ed2_ is None:
            seoul_w_pred = seoul_w_pred.filter((col("timestamp") >= ed_) & (col("timestamp") <= ed2_))

        seoul_w_pred = seoul_w_pred.orderBy("timestamp").limit(24).collect()
        w_ = []
        timestamp = []
        w_max = np.array(p.max_values[4:])
        w_min = np.array(p.min_values[4:])
        w_delta = w_max - w_min
        for x in seoul_w_pred:
            # normalize future forecast according to 0 -- 1 corresponding to min -- max of training set
            dt_v = np.array([float(x['temp']),float(x["precip"]),self.process_wind_no(x["wind_sp"], "m/s"),self.process_wind_no(x["wind_gust"])])
            a = self.min_max_scaler(dt_v, w_min, w_delta)
            a = self.set_boundary(a)
            a = a + [float(x["wind_agl"]),float(x["humidity"])/100.0, float(x["is_holiday"]),float(x["hour"]),float(x["month"])]
            timestamp.append(x['timestamp'])
            w_.append(a)
        # process vectors for china factors from weather forecasts & aqi cn data
        # "b_pm2_5", "s_pm2_5", "b_wdir", "b_humidity", "s_wdir", "s_humidity", "month", "hour", "is_holiday"
        # "s_temp","s_wsp","s_gust","s_precip","b_temp","b_wsp","b_gust","b_precip"
        if not st_ is None and not ed_ is None:
            w_pred = w_pred.filter((col("timestamp") >= st_) & (col("timestamp") <= ed_))
            aqicn = aqicn.filter((col("timestamp") >= st_) & (col("timestamp") <= ed_))
        
        beijing_w_pred = w_pred.filter(col("city") == "beijing")
        beijing_w_pred = beijing_w_pred.withColumn("is_holiday", self.udf_china_holiday(date_format(col("timestamp"), "E"), col("timestamp"))) \
                                       .withColumn("hour", hour(col("timestamp")).cast("double") / 23) \
                                       .withColumn("month", (month(w_pred.timestamp).cast("double") - 1) / 12) \
                                       .orderBy("timestamp").limit(24).collect()
        shenyang_w_pred = w_pred.filter(col("city") == "shenyang").orderBy("timestamp").limit(24).collect()
        aqicn_be = aqicn.filter(col("city") == "beijing").orderBy("timestamp").limit(24).collect()
        aqicn_sh = aqicn.filter(col("city") == "shenyang").orderBy("timestamp").limit(24).collect()

        china_vectors = []
        china_max = np.array(p.max_cn_values)
        china_min = np.array(p.min_cn_values)
        china_delta = china_max - china_min
        for ab, ash, wb, wsh in zip(aqicn_be, aqicn_sh, beijing_w_pred, shenyang_w_pred):
            ab_ = float(ab['pm2_5']) / 500
            ash_ = float(ash['pm2_5']) / 500
            wb_1, wb_2 = self.get_china_weather_factors(wb)
            wsh_1, wsh_2 = self.get_china_weather_factors(wsh)
            wcn_2 = self.min_max_scaler(np.array(wsh_2 + wb_2), china_min, china_delta)
            china_vector = [ab_] + [ash_] + wb_1 + wsh_1 + [float(wb["month"]), float(wb["hour"]), float(wb["is_holiday"])] + self.set_boundary(wcn_2)
            china_vectors.append(china_vector)
        print(np.shape(china_vectors))
        return res, w_, china_vectors, timestamp

    def get_china_weather_factors(self, rows):
        wsp = self.process_wind_no(rows["wind_sp"])
        wg = self.process_wind_no(rows["wind_gust"])
        return [float(rows["wind_agl"]),float(rows["humidity"])/100.0], [rows['temp'], wsp, wg, float(rows["precip"])]
    
    def min_max_scaler(self, inputs, min_v, delta):
        return (inputs - min_v) / delta
    
    def set_boundary(self, inputs):
        return [1.0 if y > 1.0 else y if y > 0.0 else 0.0 for y in inputs]

    def process_wind_no(self, w, convert=""):
        val = 0.0
        wsp = w.split(" ")
        if wsp:
            val = float(wsp[0])
        else:
            val = 0.0
        if "km" in w and val != 0.0:
            if convert == "m/s":
                val = val * 0.277778
            else:
                # convert to mph  
                val = val * 0.621371  
        elif convert == "m/s" and val != 0.0:
            # default mph
            val = val * 0.44704
        return val
