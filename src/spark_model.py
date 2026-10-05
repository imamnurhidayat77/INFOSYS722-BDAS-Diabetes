from spark_session import get_spark

import math

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    when,
    lit,
    desc,
    avg,
    count as spark_count,
    sum as spark_sum,
    row_number
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator
)

from pyspark.ml.functions import vector_to_array


spark = get_spark()

print("=" * 60)
print("STEP 5 / 6 / 7 - METHODS, ALGORITHMS, DATA MINING")
print("=" * 60)


# =====================================================
# 1. LOAD STEP 4 OUTPUT
# =====================================================

TRAIN_PATH = "data/train_step4_transformed.parquet"
TEST_PATH = "data/test_step4_transformed.parquet"

train_df = spark.read.parquet(TRAIN_PATH)
test_df = spark.read.parquet(TEST_PATH)

print("\n1. Loaded Step 4 Outputs")
print("Training rows :", train_df.count())
print("Testing rows  :", test_df.count())


# =====================================================
# 2. ADD CLASS WEIGHTS
# =====================================================
#
# Helps address class imbalance.
#

label_col = "Diabetes_binary"

label_counts = (
    train_df.groupBy(label_col)
    .count()
    .collect()
)

count_dict = {row[label_col]: row["count"] for row in label_counts}

neg_count = count_dict.get(0, 1)
pos_count = count_dict.get(1, 1)
total_count = neg_count + pos_count

neg_weight = total_count / (2.0 * neg_count)
pos_weight = total_count / (2.0 * pos_count)

train_df = train_df.withColumn(
    "classWeightCol",
    when(col(label_col) == 1, lit(pos_weight)).otherwise(lit(neg_weight))
)

test_df = test_df.withColumn(
    "classWeightCol",
    when(col(label_col) == 1, lit(pos_weight)).otherwise(lit(neg_weight))
)

print("\n2. Class Weights")
print("Negative class count :", neg_count)
print("Positive class count :", pos_count)
print("Negative class weight:", round(neg_weight, 6))
print("Positive class weight:", round(pos_weight, 6))


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def compute_confusion_metrics(df_pred, label_col="Diabetes_binary", pred_col="prediction"):
    """
    Compute confusion-based metrics at the model's prediction threshold.
    """

    tp = df_pred.filter((col(label_col) == 1) & (col(pred_col) == 1)).count()
    tn = df_pred.filter((col(label_col) == 0) & (col(pred_col) == 0)).count()
    fp = df_pred.filter((col(label_col) == 0) & (col(pred_col) == 1)).count()
    fn = df_pred.filter((col(label_col) == 1) & (col(pred_col) == 0)).count()

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def compute_rank_metrics(df_pred, label_col="Diabetes_binary", score_col="score"):
    """
    Compute:
    - top 20% capture
    - top 20% lift
    - places per true case at top 20%
    - contact depth required to reach 60% capture
    """

    total = df_pred.count()
    positives = df_pred.filter(col(label_col) == 1).count()

    top_n = math.ceil(total * 0.20)

    rank_window = Window.orderBy(desc(score_col))

    ranked = (
        df_pred
        .select(label_col, score_col)
        .withColumn("rank", row_number().over(rank_window))
    )

    top20_df = ranked.filter(col("rank") <= top_n)

    captured_top20 = top20_df.agg(
        spark_sum(col(label_col)).alias("captured")
    ).first()["captured"]

    captured_top20 = captured_top20 if captured_top20 is not None else 0

    capture_rate_top20 = captured_top20 / positives if positives else 0.0
    lift_top20 = capture_rate_top20 / 0.20 if positives else 0.0
    places_per_true_top20 = top_n / captured_top20 if captured_top20 else None

    # Operating point to reach 60% capture
    target_capture = positives * 0.60

    cumulative_window = Window.orderBy(desc(score_col)).rowsBetween(Window.unboundedPreceding, 0)

    ranked_cum = ranked.withColumn(
        "cum_positives",
        spark_sum(col(label_col)).over(cumulative_window)
    )

    reach60_row = (
        ranked_cum
        .filter(col("cum_positives") >= lit(target_capture))
        .orderBy("rank")
        .first()
    )

    if reach60_row:
        contacts_to_60 = reach60_row["rank"]
        depth_to_60 = contacts_to_60 / total
    else:
        contacts_to_60 = None
        depth_to_60 = None

    return {
        "Top20_Contacts": top_n,
        "Top20_Captured": captured_top20,
        "Top20_CaptureRate": capture_rate_top20,
        "Top20_Lift": lift_top20,
        "Top20_PlacesPerTrueCase": places_per_true_top20,
        "Contacts_to_60pct_Capture": contacts_to_60,
        "Depth_to_60pct_Capture": depth_to_60
    }


def evaluate_classifier(model_name, predictions_df, label_col="Diabetes_binary"):
    """
    Evaluate a classifier:
    - AUC
    - confusion metrics
    - ranking metrics
    """

    # Add scalar score from probability vector
    scored = predictions_df.withColumn(
        "score",
        vector_to_array(col("probability"))[1]
    )

    # AUC
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(scored)

    # Confusion-based metrics
    conf = compute_confusion_metrics(scored, label_col=label_col, pred_col="prediction")

    # Rank-based metrics
    rank_metrics = compute_rank_metrics(scored, label_col=label_col, score_col="score")

    results = {
        "Model": model_name,
        "AUC": auc,
        **conf,
        **rank_metrics
    }

    return scored, results


def evaluate_risktest5_baseline(df_test, label_col="Diabetes_binary"):
    """
    Evaluate RiskTest5 baseline:
    - confusion using RiskTest5_flag
    - places per true case using RiskTest5_flag
    """

    baseline_pred = df_test.withColumn("prediction", col("RiskTest5_flag").cast("double"))

    conf = compute_confusion_metrics(
        baseline_pred,
        label_col=label_col,
        pred_col="prediction"
    )

    flagged_count = baseline_pred.filter(col("RiskTest5_flag") == 1).count()
    true_positives = baseline_pred.filter(
        (col("RiskTest5_flag") == 1) & (col(label_col) == 1)
    ).count()

    places_per_true = flagged_count / true_positives if true_positives else None

    # Ranking by RiskTest5 raw score for gains/lift style comparison
    ranked_baseline = baseline_pred.withColumn("score", col("RiskTest5").cast("double"))

    rank_metrics = compute_rank_metrics(
        ranked_baseline,
        label_col=label_col,
        score_col="score"
    )

    results = {
        "Model": "RiskTest5_Baseline",
        "AUC": None,
        **conf,
        **rank_metrics,
        "Baseline_FlaggedCount": flagged_count,
        "Baseline_PlacesPerTrueCase": places_per_true
    }

    return ranked_baseline, results


# =====================================================
# 4. DEFINE MODELS
# =====================================================

models = {
    "LogisticRegression": LogisticRegression(
        featuresCol="features",
        labelCol="Diabetes_binary",
        weightCol="classWeightCol",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0
    ),

    "DecisionTree": DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Diabetes_binary",
        weightCol="classWeightCol",
        maxDepth=4,
        minInstancesPerNode=400
    ),

    "RandomForest": RandomForestClassifier(
        featuresCol="features",
        labelCol="Diabetes_binary",
        weightCol="classWeightCol",
        numTrees=200,
        maxDepth=8,
        minInstancesPerNode=25,
        featureSubsetStrategy="sqrt",
        seed=42
    ),

    "GBT": GBTClassifier(
        featuresCol="features",
        labelCol="Diabetes_binary",
        weightCol="classWeightCol",
        maxIter=50,
        maxDepth=4,
        stepSize=0.1,
        seed=42
    )
}


# =====================================================
# 5. TRAIN AND EVALUATE CLASSIFIERS
# =====================================================

all_results = []

print("\n3. Baseline Evaluation")

baseline_df, baseline_results = evaluate_risktest5_baseline(test_df)
all_results.append(baseline_results)

print("Baseline model:", baseline_results["Model"])
print("Baseline recall:", round(baseline_results["Recall"], 4))
print("Baseline precision:", round(baseline_results["Precision"], 4))
print("Baseline top-20 capture:", round(baseline_results["Top20_CaptureRate"], 4))
print("Baseline top-20 lift:", round(baseline_results["Top20_Lift"], 4))
print("Baseline places per true case:", baseline_results["Baseline_PlacesPerTrueCase"])


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 60)

    fitted_model = model.fit(train_df)
    predictions = fitted_model.transform(test_df)

    scored_df, results = evaluate_classifier(model_name, predictions)
    all_results.append(results)

    # Save predictions
    pred_path = f"data/predictions_{model_name}.parquet"
    scored_df.write.mode("overwrite").parquet(pred_path)

    print(f"\nResults for {model_name}")
    print("AUC                  :", round(results["AUC"], 4))
    print("Accuracy             :", round(results["Accuracy"], 4))
    print("Precision            :", round(results["Precision"], 4))
    print("Recall               :", round(results["Recall"], 4))
    print("F1                   :", round(results["F1"], 4))
    print("Top20 Capture        :", round(results["Top20_CaptureRate"], 4))
    print("Top20 Lift           :", round(results["Top20_Lift"], 4))
    print("Places / True Case   :", results["Top20_PlacesPerTrueCase"])
    print("Contacts to 60% cap  :", results["Contacts_to_60pct_Capture"])
    print("Depth to 60% cap     :", results["Depth_to_60pct_Capture"])
    print("Saved:", pred_path)


# =====================================================
# 6. SAVE MODEL METRICS
# =====================================================

metrics_df = spark.createDataFrame(all_results)

metrics_output_path = "data/model_metrics_step7.csv"

(
    metrics_df.coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(metrics_output_path)
)

print("\n4. Model Metrics Saved")
print(metrics_output_path)


# =====================================================
# 7. KMEANS SCREENING (k = 4, 5, 6)
# =====================================================

print("\n5. KMeans Screening")

cluster_results = []

cluster_evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

# KMeans is run on the natural training partition
for k in [4, 5, 6]:

    km = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        seed=42
    )

    km_model = km.fit(train_df)
    clustered = km_model.transform(train_df)

    silhouette = cluster_evaluator.evaluate(clustered)

    cluster_results.append({
        "k": k,
        "silhouette": silhouette
    })

    print(f"k = {k}, silhouette = {round(silhouette, 6)}")


# Pick best k
best_cluster = max(cluster_results, key=lambda x: x["silhouette"])
best_k = best_cluster["k"]

print("\nBest k selected:", best_k)
print("Best silhouette:", round(best_cluster["silhouette"], 6))


# =====================================================
# 8. FINAL KMEANS FIT AND CLUSTER PROFILE
# =====================================================

print("\n6. Final KMeans Fit and Cluster Profiling")

final_km = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=best_k,
    seed=42
)

final_km_model = final_km.fit(train_df)
clustered_train = final_km_model.transform(train_df)

cluster_profile = (
    clustered_train
    .groupBy("cluster")
    .agg(
        spark_count("*").alias("n"),
        avg("Age").alias("Age_mean"),
        avg("GenHlth").alias("GenHlth_mean"),
        avg("Income").alias("Income_mean"),
        avg("BMI").alias("BMI_mean"),
        avg("RiskTest5").alias("RiskTest5_mean"),
        avg("ComorbidityCount").alias("ComorbidityCount_mean"),
        avg("Diabetes_binary").alias("Positive_Prevalence")
    )
    .orderBy("cluster")
)

cluster_profile.show(truncate=False)

cluster_profile_path = "data/kmeans_cluster_profile.csv"

(
    cluster_profile.coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(cluster_profile_path)
)

print("Saved:", cluster_profile_path)


# Save k screening results
cluster_screen_df = spark.createDataFrame(cluster_results)

cluster_screen_path = "data/kmeans_screening_results.csv"

(
    cluster_screen_df.coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(cluster_screen_path)
)

print("Saved:", cluster_screen_path)


# =====================================================
# 9. SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nClassification outputs saved:")
for model_name in models.keys():
    print(f"- data/predictions_{model_name}.parquet")

print("- data/model_metrics_step7.csv")

print("\nClustering outputs saved:")
print("- data/kmeans_screening_results.csv")
print("- data/kmeans_cluster_profile.csv")

print("\nDone.")


spark.stop()