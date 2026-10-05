from spark_session import get_spark

import math

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    desc,
    when,
    lit,
    row_number,
    sum as spark_sum,
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# ============================================================
# SPARK SESSION
# ============================================================

spark = get_spark()

print("=" * 70)
print("STEP 8 - INTERPRETATION AND EVALUATION")
print("=" * 70)


# ============================================================
# 1. LATEST STEP 7 ARTEFACTS
# ============================================================

models = {
    "Logistic Regression":
        "data/predictions_LogisticRegression.parquet",

    "Decision Tree":
        "data/predictions_DecisionTree.parquet",

    "Random Forest":
        "data/predictions_RandomForest.parquet",

    "GBT":
        "data/predictions_GBT.parquet",
}


# ============================================================
# 2. AUC EVALUATOR
# ============================================================

auc_eval = BinaryClassificationEvaluator(
    labelCol="Diabetes_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)


# ============================================================
# 3. CLASSIFIER EVALUATION
# ============================================================

def evaluate_model(name, df):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # Basic audit
    # --------------------------------------------------------

    total_records = df.count()

    print("\nTest rows:", total_records)

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = auc_eval.evaluate(df)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nConfusion Matrix")

    (
        df
        .groupBy(
            "Diabetes_binary",
            "prediction"
        )
        .count()
        .orderBy(
            "Diabetes_binary",
            "prediction"
        )
        .show()
    )

    tp = (
        df
        .filter(
            (col("Diabetes_binary") == 1)
            &
            (col("prediction") == 1)
        )
        .count()
    )

    tn = (
        df
        .filter(
            (col("Diabetes_binary") == 0)
            &
            (col("prediction") == 0)
        )
        .count()
    )

    fp = (
        df
        .filter(
            (col("Diabetes_binary") == 0)
            &
            (col("prediction") == 1)
        )
        .count()
    )

    fn = (
        df
        .filter(
            (col("Diabetes_binary") == 1)
            &
            (col("prediction") == 0)
        )
        .count()
    )

    check_total = tp + tn + fp + fn

    if check_total != total_records:
        raise ValueError(
            f"{name}: confusion matrix does not reconcile "
            f"({check_total} != {total_records})"
        )

    # --------------------------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------------------------

    accuracy = (
        (tp + tn) / total_records
        if total_records > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\nClassification Metrics")
    print("AUC       :", round(auc, 6))
    print("Accuracy  :", round(accuracy, 6))
    print("Precision :", round(precision, 6))
    print("Recall    :", round(recall, 6))
    print("F1        :", round(f1, 6))

    # --------------------------------------------------------
    # POSITIVE-CLASS PROBABILITY
    # --------------------------------------------------------

    scored = (
        df
        .withColumn(
            "positive_probability",
            vector_to_array(
                col("probability")
            )[1]
        )
    )

    total_positive = (
        scored
        .filter(
            col("Diabetes_binary") == 1
        )
        .count()
    )

    # --------------------------------------------------------
    # GLOBAL RANKING
    # --------------------------------------------------------
    #
    # Global ranking is required because the business question
    # is: who should be contacted first across the full held-out
    # population?
    #
    # A global ordering may generate a Spark Window warning.
    # This is expected for this evaluation operation.
    # --------------------------------------------------------

    ranking_window = (
        Window
        .orderBy(
            desc("positive_probability")
        )
    )

    ranked = (
        scored
        .withColumn(
            "rank",
            row_number().over(
                ranking_window
            )
        )
    )

    # --------------------------------------------------------
    # TOP 20% CAPTURE
    # --------------------------------------------------------

    top20_limit = math.ceil(
        total_records * 0.20
    )

    top20 = (
        ranked
        .filter(
            col("rank") <= top20_limit
        )
    )

    top20_positive = (
        top20
        .filter(
            col("Diabetes_binary") == 1
        )
        .count()
    )

    top20_gain = (
        top20_positive / total_positive
        if total_positive > 0
        else 0.0
    )

    top20_lift = (
        top20_gain / 0.20
        if total_positive > 0
        else 0.0
    )

    places_per_case = (
        top20_limit / top20_positive
        if top20_positive > 0
        else None
    )

    # --------------------------------------------------------
    # CONTACT DEPTH REQUIRED FOR 60% CAPTURE
    # --------------------------------------------------------

    cumulative_window = (
        Window
        .orderBy(
            desc("positive_probability")
        )
        .rowsBetween(
            Window.unboundedPreceding,
            0
        )
    )

    ranked = (
        ranked
        .withColumn(
            "cumulative_positive",
            spark_sum(
                col("Diabetes_binary")
            ).over(
                cumulative_window
            )
        )
    )

    target_positive = (
        total_positive * 0.60
    )

    reach60 = (
        ranked
        .filter(
            col("cumulative_positive")
            >= target_positive
        )
        .orderBy("rank")
        .first()
    )

    if reach60:

        contacts60 = int(
            reach60["rank"]
        )

        depth60 = (
            contacts60 /
            total_records
        )

    else:

        contacts60 = None
        depth60 = None

    print("\nRanking Metrics")
    print(
        "Top20 contacts        :",
        top20_limit
    )
    print(
        "Top20 captured cases  :",
        top20_positive
    )
    print(
        "Top20 capture         :",
        round(top20_gain, 6)
    )
    print(
        "Top20 lift            :",
        round(top20_lift, 6)
    )
    print(
        "Places per true case  :",
        round(places_per_case, 6)
        if places_per_case is not None
        else None
    )
    print(
        "Contacts to 60%       :",
        contacts60
    )
    print(
        "Depth to 60%          :",
        round(depth60, 6)
        if depth60 is not None
        else None
    )

    return {
        "Model": name,
        "Test_Rows": int(total_records),

        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),

        "AUC": float(auc),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),

        "Top20_Contacts":
            int(top20_limit),

        "Top20_Captured":
            int(top20_positive),

        "Top20_Capture":
            float(top20_gain),

        "Top20_Lift":
            float(top20_lift),

        "Places_Per_True_Case":
            float(places_per_case)
            if places_per_case is not None
            else None,

        "Contacts_to_60pct":
            int(contacts60)
            if contacts60 is not None
            else None,

        "Depth_to_60pct":
            float(depth60)
            if depth60 is not None
            else None,
    }


# ============================================================
# 4. RUN FOUR SUPERVISED MODELS
# ============================================================

results = []

for name, path in models.items():

    print("\nLoading:", path)

    df = spark.read.parquet(path)

    result = evaluate_model(
        name,
        df
    )

    results.append(result)


# ============================================================
# 5. FINAL MODEL COMPARISON
# ============================================================

comparison = spark.createDataFrame(
    results
)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

(
    comparison
    .orderBy(
        col("AUC").desc()
    )
    .show(
        truncate=False
    )
)

comparison_output = (
    "data/final_model_comparison"
)

(
    comparison
    .coalesce(1)
    .write
    .mode("overwrite")
    .option(
        "header",
        True
    )
    .csv(
        comparison_output
    )
)

print(
    "\nSaved:",
    comparison_output
)


# ============================================================
# 6. LOAD LATEST GBT + LOGISTIC OUTPUTS
# ============================================================

gbt_df = spark.read.parquet(
    "data/predictions_GBT.parquet"
)

logistic_df = spark.read.parquet(
    "data/predictions_LogisticRegression.parquet"
)


# ============================================================
# 7. RISKTEST5 BASELINE
# ============================================================

print("\n")
print("=" * 70)
print("RISKTEST5 BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# Binary RiskTest5_flag performance
# ------------------------------------------------------------

baseline_tp = (
    gbt_df
    .filter(
        (col("Diabetes_binary") == 1)
        &
        (col("RiskTest5_flag") == 1)
    )
    .count()
)

baseline_fp = (
    gbt_df
    .filter(
        (col("Diabetes_binary") == 0)
        &
        (col("RiskTest5_flag") == 1)
    )
    .count()
)

baseline_fn = (
    gbt_df
    .filter(
        (col("Diabetes_binary") == 1)
        &
        (col("RiskTest5_flag") == 0)
    )
    .count()
)

baseline_tn = (
    gbt_df
    .filter(
        (col("Diabetes_binary") == 0)
        &
        (col("RiskTest5_flag") == 0)
    )
    .count()
)

baseline_precision = (
    baseline_tp /
    (baseline_tp + baseline_fp)
    if (baseline_tp + baseline_fp) > 0
    else 0.0
)

baseline_recall = (
    baseline_tp /
    (baseline_tp + baseline_fn)
    if (baseline_tp + baseline_fn) > 0
    else 0.0
)

baseline_f1 = (
    2
    * baseline_precision
    * baseline_recall
    /
    (
        baseline_precision
        + baseline_recall
    )
    if (
        baseline_precision
        + baseline_recall
    ) > 0
    else 0.0
)

flagged_count = (
    baseline_tp
    + baseline_fp
)

baseline_places_flag = (
    flagged_count / baseline_tp
    if baseline_tp > 0
    else None
)

print("\nRiskTest5_flag confusion counts")
print("TP:", baseline_tp)
print("TN:", baseline_tn)
print("FP:", baseline_fp)
print("FN:", baseline_fn)

print("\nRiskTest5_flag metrics")
print(
    "Precision:",
    round(
        baseline_precision,
        6
    )
)
print(
    "Recall:",
    round(
        baseline_recall,
        6
    )
)
print(
    "F1:",
    round(
        baseline_f1,
        6
    )
)
print(
    "Flagged respondents:",
    flagged_count
)
print(
    "Places per true case:",
    round(
        baseline_places_flag,
        6
    )
    if baseline_places_flag
    is not None
    else None
)


# ------------------------------------------------------------
# RiskTest5 ranking at the same top-20% capacity
# ------------------------------------------------------------

baseline_total = gbt_df.count()

baseline_positive_total = (
    gbt_df
    .filter(
        col("Diabetes_binary") == 1
    )
    .count()
)

baseline_window = (
    Window
    .orderBy(
        desc("RiskTest5")
    )
)

baseline_ranked = (
    gbt_df
    .withColumn(
        "baseline_rank",
        row_number().over(
            baseline_window
        )
    )
)

baseline_top20_limit = math.ceil(
    baseline_total * 0.20
)

baseline_top20 = (
    baseline_ranked
    .filter(
        col("baseline_rank")
        <= baseline_top20_limit
    )
)

baseline_positive_top20 = (
    baseline_top20
    .filter(
        col("Diabetes_binary") == 1
    )
    .count()
)

baseline_capture = (
    baseline_positive_top20 /
    baseline_positive_total
    if baseline_positive_total > 0
    else 0.0
)

baseline_lift = (
    baseline_capture / 0.20
    if baseline_positive_total > 0
    else 0.0
)

baseline_places_top20 = (
    baseline_top20_limit /
    baseline_positive_top20
    if baseline_positive_top20 > 0
    else None
)

print("\nRiskTest5 ranking metrics")
print(
    "Top20 contacts:",
    baseline_top20_limit
)
print(
    "Top20 captured:",
    baseline_positive_top20
)
print(
    "Top20 capture:",
    round(
        baseline_capture,
        6
    )
)
print(
    "Top20 lift:",
    round(
        baseline_lift,
        6
    )
)
print(
    "Top20 places per true case:",
    round(
        baseline_places_top20,
        6
    )
    if baseline_places_top20
    is not None
    else None
)


# ------------------------------------------------------------
# Save baseline summary
# ------------------------------------------------------------

baseline_summary = spark.createDataFrame(
    [{
        "Baseline":
            "RiskTest5",

        "TP":
            int(baseline_tp),

        "TN":
            int(baseline_tn),

        "FP":
            int(baseline_fp),

        "FN":
            int(baseline_fn),

        "Precision":
            float(
                baseline_precision
            ),

        "Recall":
            float(
                baseline_recall
            ),

        "F1":
            float(
                baseline_f1
            ),

        "Flagged_Count":
            int(
                flagged_count
            ),

        "Flagged_Places_Per_True_Case":
            float(
                baseline_places_flag
            )
            if baseline_places_flag
            is not None
            else None,

        "Top20_Capture":
            float(
                baseline_capture
            ),

        "Top20_Lift":
            float(
                baseline_lift
            ),

        "Top20_Places_Per_True_Case":
            float(
                baseline_places_top20
            )
            if baseline_places_top20
            is not None
            else None,
    }]
)

(
    baseline_summary
    .coalesce(1)
    .write
    .mode("overwrite")
    .option(
        "header",
        True
    )
    .csv(
        "data/risktest5_baseline_step8"
    )
)


# ============================================================
# 8. BO4 EQUITY ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("BO4 EQUITY ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# GBT income-band recall
# ------------------------------------------------------------

gbt_equity = (
    gbt_df
    .groupBy(
        "Income"
    )
    .agg(

        spark_sum(
            when(
                col("Diabetes_binary") == 1,
                1
            ).otherwise(0)
        ).alias(
            "Actual_Positive"
        ),

        spark_sum(
            when(
                (
                    col("Diabetes_binary") == 1
                )
                &
                (
                    col("prediction") == 1
                ),
                1
            ).otherwise(0)
        ).alias(
            "GBT_TP"
        ),

        spark_sum(
            when(
                (
                    col("Diabetes_binary") == 1
                )
                &
                (
                    col("RiskTest5_flag") == 1
                ),
                1
            ).otherwise(0)
        ).alias(
            "RiskTest_TP"
        ),
    )
)


# ------------------------------------------------------------
# Logistic income-band recall
# ------------------------------------------------------------

logistic_equity = (
    logistic_df
    .groupBy(
        "Income"
    )
    .agg(

        spark_sum(
            when(
                (
                    col("Diabetes_binary") == 1
                )
                &
                (
                    col("prediction") == 1
                ),
                1
            ).otherwise(0)
        ).alias(
            "Logistic_TP"
        )
    )
)


# ------------------------------------------------------------
# Join and calculate recall
# ------------------------------------------------------------

equity = (
    gbt_equity
    .join(
        logistic_equity,
        on="Income",
        how="left"
    )
    .withColumn(
        "RiskTest_Recall",
        when(
            col("Actual_Positive") > 0,
            col("RiskTest_TP")
            /
            col("Actual_Positive")
        )
    )
    .withColumn(
        "Logistic_Recall",
        when(
            col("Actual_Positive") > 0,
            col("Logistic_TP")
            /
            col("Actual_Positive")
        )
    )
    .withColumn(
        "GBT_Recall",
        when(
            col("Actual_Positive") > 0,
            col("GBT_TP")
            /
            col("Actual_Positive")
        )
    )
    .withColumn(
        "GBT_vs_RiskTest",
        col("GBT_Recall")
        -
        col("RiskTest_Recall")
    )
    .withColumn(
        "Logistic_vs_RiskTest",
        col("Logistic_Recall")
        -
        col("RiskTest_Recall")
    )
)


print(
    "\nRecall by Income band"
)

(
    equity
    .orderBy(
        "Income"
    )
    .show(
        truncate=False
    )
)


(
    equity
    .coalesce(1)
    .write
    .mode("overwrite")
    .option(
        "header",
        True
    )
    .csv(
        "data/bo4_equity_comparison"
    )
)

print(
    "\nSaved:",
    "data/bo4_equity_comparison"
)


# ============================================================
# 9. KMEANS EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("KMEANS EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Read latest Step 7 KMeans screening output
# ------------------------------------------------------------

kmeans_screen = (
    spark.read
    .option(
        "header",
        True
    )
    .option(
        "inferSchema",
        True
    )
    .csv(
        "data/kmeans_screening_results.csv"
    )
)

print(
    "\nKMeans screening results"
)

(
    kmeans_screen
    .orderBy(
        "k"
    )
    .show(
        truncate=False
    )
)


best_k_row = (
    kmeans_screen
    .orderBy(
        col(
            "silhouette"
        ).desc()
    )
    .first()
)

if best_k_row is not None:

    print(
        "Best k:",
        best_k_row["k"]
    )

    print(
        "Best silhouette:",
        best_k_row[
            "silhouette"
        ]
    )


# ------------------------------------------------------------
# Read latest cluster profile
# ------------------------------------------------------------

kmeans_profile = (
    spark.read
    .option(
        "header",
        True
    )
    .option(
        "inferSchema",
        True
    )
    .csv(
        "data/kmeans_cluster_profile.csv"
    )
)

print(
    "\nFinal KMeans cluster profile"
)

(
    kmeans_profile
    .orderBy(
        "cluster"
    )
    .show(
        truncate=False
    )
)


# ============================================================
# 10. FINAL TRACEABILITY CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FINAL STEP 8 TRACEABILITY")
print("=" * 70)

expected_test_rows = 127533

for name, path in models.items():

    n = (
        spark
        .read
        .parquet(path)
        .count()
    )

    status = (
        "OK"
        if n == expected_test_rows
        else "CHECK"
    )

    print(
        f"{name}: "
        f"{n:,} rows "
        f"[{status}]"
    )


print("\nStep 8 outputs:")
print(
    "- data/final_model_comparison"
)
print(
    "- data/risktest5_baseline_step8"
)
print(
    "- data/bo4_equity_comparison"
)
print(
    "- data/kmeans_screening_results.csv"
)
print(
    "- data/kmeans_cluster_profile.csv"
)


print("\n" + "=" * 70)
print("STEP 8 COMPLETED")
print("=" * 70)


spark.stop()