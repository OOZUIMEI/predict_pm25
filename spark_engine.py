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
        self.spark = self.init_spark()
        self.assembler = VectorAssembler(inputCols=["pm2_5_norm", "pm10_norm", "features", "wind_agl"], outputCol="all_features")
        # init udf function
        self.udf_float = udf(udf_utils.fix_float)
        self.udf_int = udf(udf_utils.fix_int)
        self.udf_district_code = udf(udf_utils.get_district_code)
        self.udf_pm = udf(udf_utils.normalize_pm)
        self.udf_agl = udf(udf_utils.normalize_agl)
        self.udf_dir = udf(udf_utils.convert_dir_to_agl)
        self.udf_hum = udf(udf_utils.humidity)
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
            StructField("wind_sp", IntegerType(), True),
            StructField("wind_dir", StringType(), True),
            StructField("wind_gust", IntegerType(), True),
            StructField("cloud", IntegerType(), True),
            StructField("humidity", IntegerType(), True),
            StructField("precip", DoubleType(), True),
            StructField("pressure", IntegerType(), True)
        ])

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

    def process_vectors(self, start=None, end=None):
        # engine do something
        p1 = "data/seoul_aqi.csv"
        p2 = "data/seoul_aws.csv"
        p3 = "data/seoul_weather.csv"
        dim = 12
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
                    .select(air.timestamp, air.district_code, "o3_val", "no2_val", "co_val", "so2_val", \
                            "pm2_5_norm", "pm10_norm", "temp", "precip", "humidity", self.udf_agl(col("wind_agl")).cast('double').alias("wind_agl"), "wind_sp", "wind_gust") \
                    .na.fill(0.0, ["pm2_5_norm", "pm10_norm", "o3_val","no2_val","co_val","so2_val", "temp", "precip", "humidity", "wind_agl", "wind_sp", "wind_gust"])

        w_pred = self.spark.read.format("csv").option("header", "false").schema(self.sw_pred).load(p3) \
                     .groupBy("timestamp").agg(last("temp").alias("temp"),last("precip").alias("precip"), \
                            last("humidity").alias("humidity"),last("wind_sp").alias("wind_sp"), \
                            last("wind_gust").alias("wind_gust"),last("wind_dir").alias("wind_dir")) \
                     .select("timestamp", "temp", "precip", "humidity", "wind_sp", "wind_gust", self.udf_dir("wind_dir").cast("double").alias("wind_agl"))
                     
        vectors = self.normalize_vector(merge, self.features)
        final = self.assembler.transform(vectors)
        if start and end:
            st_ = start.strftime(p.fm)
            ed_ = end.strftime(p.fm)
            final = final.filter((col("timestamp") >= st_) & (col("timestamp") <= ed_))
            end = end + timedelta(days=1)
            ed2_ = end.strftime(p.fm)
            w_pred = w_pred.filter((col("timestamp") >= ed_) & (col("timestamp") <= ed2_))
        # future forecast
        w_pred = w_pred.orderBy("timestamp").collect()
        w_ = []
        timestamp = []
        for x in w_pred:
            a = [x['temp'],x["precip"],x["humidity"],x["wind_sp"],x["wind_gust"], x["wind_agl"]]
            timestamp.append(x['timestamp'])
            w_.append(a)
        

        final = final.select("timestamp", "district_code", "all_features") \
                            .groupBy("timestamp") \
                            .agg(collect_list("district_code").alias("district_code"), \
                                collect_list("all_features").alias("all_features")) \
                            .orderBy("timestamp").collect()
        res = []
        for d in final:
            dis_vectors = [np.zeros(dim, dtype=np.float).tolist()] * 25
            dis = d["district_code"]
            features = d["all_features"]
            for i, x in enumerate(dis):
                idx = int(x) - 1
                dis_vectors[idx] = features[i]
            res.append(dis_vectors)       
        
        return res, w_, timestamp