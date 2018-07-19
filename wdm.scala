package com.prediction.udf

import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction

val half = math.Pi / 180

class WindDirectionMean extends org.apache.spark.sql.expressions.UserDefinedAggregateFunction {
    // input fields for your aggregate function
    def inputSchema: org.apache.spark.sql.types.StructType = org.apache.spark.sql.types.StructType(org.apache.spark.sql.types.StructField("value", org.apache.spark.sql.types.DoubleType) :: Nil)
    // internal fields for computing aggregate
    def bufferSchema: org.apache.spark.sql.types.StructType =
    
    org.apache.spark.sql.types.StructType(
        org.apache.spark.sql.types.StructField("count", org.apache.spark.sql.types.LongType) ::
        org.apache.spark.sql.types.StructField("sin", org.apache.spark.sql.types.DoubleType) :: 
        org.apache.spark.sql.types.StructField("cos", org.apache.spark.sql.types.DoubleType) :: Nil
    )
    
    def dataType: org.apache.spark.sql.types.DataType = org.apache.spark.sql.types.DoubleType 
    def deterministic: Boolean = true
    def initialize(buffer: org.apache.spark.sql.expressions.MutableAggregationBuffer): Unit = {
        buffer(0) = 0L
        buffer(1) = 0.0
        buffer(2) = 0.0
    }
    def update(buffer: org.apache.spark.sql.expressions.MutableAggregationBuffer, input: org.apache.spark.sql.Row): Unit = {
        buffer(0) = buffer.getAs[Long](0) + 1
        var row = input.getAs[Double](0) * half
        buffer(1) = buffer.getAs[Double](1) +  math.sin(row)
        buffer(2) = buffer.getAs[Double](2) + math.cos(row)
    }
    
    def merge(buffer1: org.apache.spark.sql.expressions.MutableAggregationBuffer, buffer2: org.apache.spark.sql.Row): Unit = {
        buffer1(0) = buffer1.getAs[Long](0) + buffer2.getAs[Long](0)
        buffer1(1) = buffer1.getAs[Double](1) + buffer2.getAs[Double](1)
        buffer1(2) = buffer1.getAs[Double](2) + buffer2.getAs[Double](2)
    }
      
    def evaluate(buffer: org.apache.spark.sql.Row): Any = {
        (math.atan(buffer.getDouble(1) / buffer.getDouble(2)) / half + 360) % 360
    }
}
sqlContext.udf.register("wdm", new WindDirectionMean)