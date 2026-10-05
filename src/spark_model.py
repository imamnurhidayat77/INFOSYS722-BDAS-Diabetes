from spark_session import get_spark

from pyspark.sql.functions import (
    col,
    when
)

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


# =====================================================
# LOAD MODEL READY DATA
# =====================================================

df = spark.read.parquet(
    "data/model_ready.parquet"
)


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

train = df.filter(
    col("Partition") == "Training"
)


test = df.filter(
    col("Partition") == "Testing"
)


print("\nTraining rows:")
print(train.count())


print("\nTesting rows:")
print(test.count())



# =====================================================
# CREATE CLASS WEIGHT
# =====================================================
#
# Handle imbalance:
# minority class receives higher weight
#

class_count = (
    train
    .groupBy("Diabetes_binary")
    .count()
    .collect()
)


counts = {}

for row in class_count:
    counts[row["Diabetes_binary"]] = row["count"]


majority = max(counts.values())
minority = min(counts.values())


minority_weight = majority / minority


print("\nClass counts:")
print(counts)


print("\nMinority class weight:")
print(minority_weight)



train_weighted = train.withColumn(
    "weight",
    when(
        col("Diabetes_binary") == 1,
        minority_weight
    )
    .otherwise(1.0)
)



# =====================================================
# 1. LOGISTIC REGRESSION
# =====================================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="Diabetes_binary",
    weightCol="weight",
    maxIter=100
)


lr_model = lr.fit(
    train_weighted
)


lr_result = lr_model.transform(
    test
)


print("\nLogistic completed")



# =====================================================
# 2. DECISION TREE
# =====================================================

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    weightCol="weight",
    maxDepth=5,
    seed=42
)


dt_model = dt.fit(
    train_weighted
)


dt_result = dt_model.transform(
    test
)


print("Decision Tree completed")



# =====================================================
# 3. RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    weightCol="weight",
    numTrees=100,
    maxDepth=8,
    seed=42
)


rf_model = rf.fit(
    train_weighted
)


rf_result = rf_model.transform(
    test
)


print("Random Forest completed")



# =====================================================
# 4. GRADIENT BOOSTED TREE
# =====================================================

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="Diabetes_binary",
    maxIter=50,
    maxDepth=5,
    seed=42
)


gbt_model = gbt.fit(
    train_weighted
)


gbt_result = gbt_model.transform(
    test
)


print("GBT completed")



# =====================================================
# SAVE CLASSIFICATION OUTPUT
# =====================================================

lr_result.write \
    .mode("overwrite") \
    .parquet(
        "data/result_logistic.parquet"
    )


dt_result.write \
    .mode("overwrite") \
    .parquet(
        "data/result_tree.parquet"
    )


rf_result.write \
    .mode("overwrite") \
    .parquet(
        "data/result_rf.parquet"
    )


gbt_result.write \
    .mode("overwrite") \
    .parquet(
        "data/result_gbt.parquet"
    )



# =====================================================
# 5. KMEANS CLUSTERING
# =====================================================
#
# Unsupervised descriptive analysis
#

kmeans = KMeans(
    k=5,
    seed=42,
    featuresCol="features"
)


kmeans_model = kmeans.fit(
    train
)


cluster_result = kmeans_model.transform(
    train
)


cluster_result.write \
    .mode("overwrite") \
    .parquet(
        "data/result_cluster.parquet"
    )


print("KMeans completed")



# =====================================================
# FEATURE IMPORTANCE OUTPUT
# =====================================================


print("\nRandom Forest Feature Importance")

print(
    rf_model.featureImportances
)


print("\nDecision Tree Feature Importance")

print(
    dt_model.featureImportances
)



print("\nALL MODELS COMPLETED")


spark.stop()