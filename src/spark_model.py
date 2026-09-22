from spark_session import get_spark

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)


spark = get_spark()


print("==============================")
print("STEP 5-7 MODEL TRAINING")
print("==============================")


df = spark.read.parquet(
    "data/model_ready.parquet"
)


df = df.withColumnRenamed(
    "Diabetes_binary",
    "label"
)


train, test = df.randomSplit(
    [0.8,0.2],
    seed=42
)


print("Training rows:")
print(train.count())


print("Testing rows:")
print(test.count())


# --------------------------
# Logistic Regression
# --------------------------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)


lr_model = lr.fit(train)


lr_prediction = lr_model.transform(test)


print("\nLogistic Regression Result")

lr_prediction.select(
    "label",
    "prediction",
    "probability"
).show(10)


lr_prediction.write \
    .mode("overwrite") \
    .parquet(
        "output/lr_prediction"
    )


# --------------------------
# Random Forest
# --------------------------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    seed=42
)


rf_model = rf.fit(train)


rf_prediction = rf_model.transform(test)


print("\nRandom Forest Result")

rf_prediction.select(
    "label",
    "prediction",
    "probability"
).show(10)


rf_prediction.write \
    .mode("overwrite") \
    .parquet(
        "output/rf_prediction"
    )


print("\nModel output saved")


spark.stop()