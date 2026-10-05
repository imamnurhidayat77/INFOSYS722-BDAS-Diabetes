from spark_session import get_spark

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)

from pyspark.ml.clustering import KMeans


spark = get_spark()


print("==============================")
print("STEP 7 - DATA MINING")
print("==============================")


# Load transformed data

df = spark.read.parquet(
    "data/model_ready.parquet"
)


train = df.filter(
    df.Partition == "Training"
)


test = df.filter(
    df.Partition == "Testing"
)



print("Training:")
print(train.count())

print("Testing:")
print(test.count())



# =====================================================
# LOGISTIC REGRESSION
# =====================================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="Diabetes_binary",
    maxIter=100
)


lr_model = lr.fit(train)


lr_result = lr_model.transform(test)



# =====================================================
# DECISION TREE
# =====================================================

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    maxDepth=5,
    seed=42
)


dt_model = dt.fit(train)


dt_result = dt_model.transform(test)



# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    numTrees=100,
    maxDepth=8,
    seed=42
)


rf_model = rf.fit(train)


rf_result = rf_model.transform(test)



# =====================================================
# GRADIENT BOOSTED TREE
# =====================================================

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    maxIter=50,
    maxDepth=5,
    seed=42
)


gbt_model = gbt.fit(train)


gbt_result = gbt_model.transform(test)



# =====================================================
# SAVE CLASSIFICATION RESULTS
# =====================================================


lr_result.write.mode("overwrite") \
    .parquet(
        "data/result_logistic.parquet"
    )


dt_result.write.mode("overwrite") \
    .parquet(
        "data/result_tree.parquet"
    )


rf_result.write.mode("overwrite") \
    .parquet(
        "data/result_rf.parquet"
    )


gbt_result.write.mode("overwrite") \
    .parquet(
        "data/result_gbt.parquet"
    )



# =====================================================
# KMEANS CLUSTERING
# =====================================================


kmeans = KMeans(
    k=5,
    seed=42,
    featuresCol="features"
)


kmeans_model = kmeans.fit(train)


cluster_result = kmeans_model.transform(train)


cluster_result.write.mode("overwrite") \
    .parquet(
        "data/result_cluster.parquet"
    )


print("Models completed")


spark.stop()