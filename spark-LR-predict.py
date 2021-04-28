# PySpark 2.4.6
# 
# 
import os, sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import numpy as np

from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f

spark = SparkSession.builder.appName('fraud-detection').master("local[*]").getOrCreate()

LR_MODEL_SAVEPATH = 'saves/LRBalancedModel'
PIPELINE_SAVEPATH = 'saves/pipelineModelBalanced'

def predict(testSamples, pipelineModel):
    testDf = spark.createDataFrame([Row(**i) for i in testSamples])
    for col in testDf.columns:
        testDf = testDf.withColumn(col, testDf[col].cast(FloatType()))
    all_input_cols = testDf.columns[1:-1]
    testDf = pipelineModel.transform(testDf)
    selectedCols = ['features'] + all_input_cols 

    testDf = testDf.select(selectedCols)    
    lrModel = LogisticRegressionModel.load(LR_MODEL_SAVEPATH)    
    outputDf = lrModel.transform(testDf)
    predictions = outputDf.select(f.collect_list('prediction')).first()[0]
    return predictions


if __name__=='__main__':
    pipelineModel = PipelineModel.load(PIPELINE_SAVEPATH)
    testSamples = [
        {
        "_c0": 650,
        "Time": 99,
        "V1": "-0.8839956497728281",
        "V2": "-0.150764822957996",
        "V3": "2.2917907214775495",
        "V4": "-0.26345226832778196",
        "V5": "-0.8145352842846351",
        "V6": "0.955840627763703",
        "V7": "0.0976306732312271",
        "V8": "0.474046969090009",
        "V9": "0.139512299928856",
        "V10": "-0.7298612019237679",
        "V11": "0.711062608544338",
        "V12": "0.095006434720644",
        "V13": "-1.09750534430121",
        "V14": "-0.0597015195949617",
        "V15": "0.23455722529490802",
        "V16": "-0.142193908419341",
        "V17": "0.193357555365588",
        "V18": "0.21785331399354502",
        "V19": "1.1555711211730901",
        "V20": "0.35875101019677796",
        "V21": "0.0709014399364962",
        "V22": "0.0518320695040774",
        "V23": "0.110297657345214",
        "V24": "-0.260628692852532",
        "V25": "-0.0975487192089246",
        "V26": "1.15543923721475",
        "V27": "-0.0211993299630798",
        "V28": "0.0625654360473211",
        "Amount": 142.71
        # "Class": 0
        }
    ]
    predictions = predict(testSamples, pipelineModel)
    print(predictions)