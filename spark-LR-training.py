# PySpark 2.4.6
# 
# 

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

DATASET_DIR = '../kaggle-dataset'

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('fraud-detection').master("local[*]").getOrCreate()

from pyspark.sql.types import FloatType

spark_df = spark.read.csv(os.path.join(DATASET_DIR, 'train.csv'), header = True, inferSchema = True)

for col in spark_df.columns:
    spark_df = spark_df.withColumn(col, spark_df[col].cast(FloatType()))

pos_spark_df = spark_df.filter(spark_df.Class == 1)
neg_spark_df = spark_df.filter(spark_df.Class == 0)
neg_spark_df_2 = neg_spark_df.limit(pos_spark_df.count())

spark_df_balanced = pos_spark_df.union(neg_spark_df_2)
spark_df_unbalanced = pos_spark_df.union(neg_spark_df)
# print(spark_df_balanced.columns)
# print(spark_df_unbalanced.count())

all_columns = spark_df_balanced.columns
# print(all_columns)


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

stages = []
all_input_cols = all_columns[1:-1]
print(all_input_cols)
assembler = VectorAssembler(inputCols=all_input_cols, outputCol="features")
stages += [assembler]


pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(spark_df_balanced)
pipelineModel.write().overwrite().save('saves/pipelineModelBalanced')
spark_df_balanced_2 = pipelineModel.transform(spark_df_balanced)
selectedCols = ['Class', 'features'] + all_input_cols
spark_df_balanced_2 = spark_df_balanced_2.select(selectedCols)
# spark_df_balanced_2.printSchema()

from pyspark.ml import PipelineModel

pipelineModelLoaded = PipelineModel.load("saves/pipelineModelBalanced")
spark_df_balanced_2 = pipelineModelLoaded.transform(spark_df_balanced)
selectedCols = ['Class', 'features'] + all_input_cols
spark_df_balanced_2 = spark_df_balanced_2.select(selectedCols)

train, test = spark_df_balanced_2.randomSplit([0.9, 0.1], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
# print(test.show(5))

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
save_path = 'saves/LRBalancedModel'

lr = LogisticRegression(featuresCol = 'features', labelCol = 'Class', maxIter=10)
lrModel = lr.fit(train)
lrModel.write().overwrite().save(save_path)



# import matplotlib.pyplot as plt
# import numpy as np

# beta = np.sort(lrModel.coefficients)

# plt.plot(beta)
# plt.ylabel('Beta Coefficients')
# plt.show()

# trainingSummary = lrModel.summary

# roc = trainingSummary.roc.toPandas()
# plt.plot(roc['FPR'],roc['TPR'])
# plt.ylabel('False Positive Rate')
# plt.xlabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

# print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))



from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
import pyspark.sql.functions as f

# read pickled model
persistedModel = LogisticRegressionModel.load(save_path)

# predict
predictionsDf = persistedModel.transform(test)

predictions = predictionsDf.select(f.collect_list('prediction')).first()[0]
print(predictions)