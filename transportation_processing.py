from pyspark.sql import *
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from operator import add
from functools import reduce
from pyspark import SparkConf


conf = (SparkConf().setAppName("transportation"))
conf.set("spark.driver.memory", "2g")
conf.set("spark.executor.memory", "1g")
conf.set("spark.ui.port", "31040")
conf.set("spark.sql.shuffle.partitions", "200")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

cols = ["_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28","_c29","_c30","_c31","_c32","_c33","_c34"]
data = spark.read.format("csv").load("/media/home/alex/Documents/datasets/car_speed/2018/2018_06.csv")
data1 = data.withColumn('total', reduce(add, [F.col(x) for x in cols])) \
        .select(sum("total").alias("total")).collect()
data1 = data1[0]["total"]
total_c = data.count()
data1 = data1 / total_c / 24
# 27.646148544