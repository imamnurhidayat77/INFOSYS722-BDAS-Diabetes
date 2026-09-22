from spark_session import get_spark

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

spark=get_spark()

df=spark.read.parquet(
    "data/model_ready.parquet"
)

df=df.withColumnRenamed(
    "Diabetes_binary",
    "label"
)

train,test=df.randomSplit(
    [0.8,0.2],
    seed=42
)

# Logistic Regression

lr=LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

lr_model=lr.fit(train)

lr_prediction=lr_model.transform(test)

lr_prediction.write.mode("overwrite") \
    .parquet("output/lr_prediction")


# Random Forest

rf=RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100
)

rf_model=rf.fit(train)

rf_prediction=rf_model.transform(test)

rf_prediction.write.mode("overwrite") \
    .parquet("output/rf_prediction")

spark.stop()