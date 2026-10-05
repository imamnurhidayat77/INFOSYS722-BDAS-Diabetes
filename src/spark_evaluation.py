from spark_session import get_spark


from pyspark.sql.functions import (
    col,
    desc,
    when,
    lit,
    row_number,
    sum as spark_sum,
    count
)


from pyspark.sql.window import Window


from pyspark.sql.types import DoubleType


from pyspark.sql.functions import udf


from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator
)


import math



spark = get_spark()



print("=" * 70)
print("STEP 8 - INTERPRETATION AND EVALUATION")
print("=" * 70)



# =====================================================
# PROBABILITY EXTRACTION
# =====================================================


def extract_probability(v):

    if v is None:
        return 0.0

    return float(v[1])



probability_udf = udf(
    extract_probability,
    DoubleType()
)



# =====================================================
# MODEL FILES
# =====================================================


models = {

    "Logistic Regression":
        "data/result_logistic.parquet",

    "Decision Tree":
        "data/result_tree.parquet",

    "Random Forest":
        "data/result_rf.parquet",

    "GBT":
        "data/result_gbt.parquet"

}



# =====================================================
# EVALUATORS
# =====================================================


auc_eval = BinaryClassificationEvaluator(
    labelCol="Diabetes_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)



# =====================================================
# STORAGE
# =====================================================


model_results = []



# =====================================================
# FUNCTION
# =====================================================


def evaluate_model(
    name,
    df
):


    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)



    # -------------------------------------
    # AUC
    # -------------------------------------

    auc = auc_eval.evaluate(df)



    # -------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------

    print("\nConfusion Matrix")


    df.groupBy(
        "Diabetes_binary",
        "prediction"
    ).count().show()



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



    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn
    )


    precision = (
        tp /
        (tp + fp)
        if tp + fp > 0
        else 0
    )


    recall = (
        tp /
        (tp + fn)
        if tp + fn > 0
        else 0
    )


    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )



    print("\nMetrics")

    print("AUC       :", auc)
    print("Accuracy  :", accuracy)
    print("Precision :", precision)
    print("Recall    :", recall)
    print("F1        :", f1)



    # -------------------------------------
    # RANKING METRICS
    # -------------------------------------


    scored = (
        df
        .withColumn(
            "positive_probability",
            probability_udf(
                col("probability")
            )
        )
    )


    total_records = scored.count()


    total_positive = (
        scored
        .filter(
            col("Diabetes_binary") == 1
        )
        .count()
    )



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
            row_number()
            .over(
                ranking_window
            )
        )
    )



    top20_limit = int(
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
        top20_positive /
        total_positive
    )


    top20_lift = (
        top20_gain /
        0.20
    )


    places_per_true_case = (
        top20_limit /
        top20_positive
    )



    print("\nRanking Metrics")

    print(
        "Top20 Capture:",
        top20_gain
    )

    print(
        "Top20 Lift:",
        top20_lift
    )

    print(
        "Places per true case:",
        places_per_true_case
    )



    # -------------------------------------
    # OPERATING POINT 60%
    # -------------------------------------


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
            )
            .over(
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
            >=
            target_positive
        )
        .orderBy("rank")
        .first()
    )



    if reach60:

        contacts_60 = reach60["rank"]

        depth_60 = (
            contacts_60 /
            total_records
        )

    else:

        contacts_60 = None
        depth_60 = None



    print(
        "Contacts to 60% capture:",
        contacts_60
    )


    print(
        "Depth to 60% capture:",
        depth_60
    )



    return {

        "Model": name,

        "AUC": float(auc),

        "Accuracy": float(accuracy),

        "Precision": float(precision),

        "Recall": float(recall),

        "F1": float(f1),

        "Top20_Gain": float(top20_gain),

        "Top20_Lift": float(top20_lift),

        "Places_Per_True_Case": float(
            places_per_true_case
        ),

        "Contacts_to_60pct": contacts_60,

        "Depth_to_60pct": depth_60

    }



# =====================================================
# RUN CLASSIFIERS
# =====================================================


for name, path in models.items():


    df = spark.read.parquet(path)


    result = evaluate_model(
        name,
        df
    )


    model_results.append(
        result
    )



# =====================================================
# MODEL COMPARISON TABLE
# =====================================================


comparison_df = spark.createDataFrame(
    model_results
)



print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)


comparison_df.show(
    truncate=False
)



(
    comparison_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option(
        "header",
        True
    )
    .csv(
        "data/final_model_comparison"
    )
)



# =====================================================
# BASELINE RISKTEST5
# =====================================================


print("\n")
print("=" * 60)
print("RISKTEST5 BASELINE")
print("=" * 60)



baseline = spark.read.parquet(
    "data/result_gbt.parquet"
)



baseline_count = (
    baseline
    .filter(
        col("RiskTest5_flag") == 1
    )
    .count()
)



baseline_positive = (
    baseline
    .filter(
        (col("RiskTest5_flag") == 1)
        &
        (col("Diabetes_binary") == 1)
    )
    .count()
)



print(
    "Baseline flagged:",
    baseline_count
)


print(
    "Baseline true positives:",
    baseline_positive
)


print(
    "Baseline places per true case:",

    baseline_count /
    baseline_positive
)



# =====================================================
# EQUITY ANALYSIS
# =====================================================


print("\n")
print("=" * 60)
print("BO4 EQUITY ANALYSIS")
print("=" * 60)



gbt = spark.read.parquet(
    "data/result_gbt.parquet"
)



equity = (

    gbt
    .groupBy(
        "Income"
    )
    .agg(

        spark_sum(
            when(
                (
                    col("Diabetes_binary")==1
                )
                &
                (
                    col("prediction")==1
                ),

                1

            )
            .otherwise(0)

        )
        .alias(
            "True_Positive"
        ),


        spark_sum(
            when(
                col("Diabetes_binary")==1,

                1

            )
            .otherwise(0)

        )
        .alias(
            "Actual_Positive"
        )

    )

)



equity = equity.withColumn(

    "Recall",

    col("True_Positive")
    /
    col("Actual_Positive")

)



equity.orderBy(
    "Income"
).show()



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
        "data/income_equity_analysis"
    )
)



# =====================================================
# KMEANS
# =====================================================


print("\n")
print("=" * 60)
print("KMEANS EVALUATION")
print("=" * 60)



cluster_df = spark.read.parquet(
    "data/result_cluster.parquet"
)



cluster_df.groupBy(
    "prediction"
).count().show()



silhouette = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette"
).evaluate(
    cluster_df
)



print(
    "Silhouette:",
    silhouette
)



(
    cluster_df
    .groupBy(
        "prediction"
    )
    .agg(

        count("*")
        .alias(
            "cluster_size"
        )

    )
    .show()
)



print("\nSTEP 8 COMPLETED")



spark.stop()