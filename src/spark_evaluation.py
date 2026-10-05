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



spark = get_spark()



print("=" * 70)
print("STEP 8 - INTERPRETATION AND EVALUATION")
print("=" * 70)



# =====================================================
# Probability extractor
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



auc_eval = BinaryClassificationEvaluator(
    labelCol="Diabetes_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)



results = []



# =====================================================
# CLASSIFIER EVALUATION FUNCTION
# =====================================================


def evaluate_model(name, df):


    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)



    auc = auc_eval.evaluate(df)



    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------

    print("\nConfusion Matrix")

    (
        df
        .groupBy(
            "Diabetes_binary",
            "prediction"
        )
        .count()
        .show()
    )



    tp = (
        df
        .filter(
            (col("Diabetes_binary")==1)
            &
            (col("prediction")==1)
        )
        .count()
    )


    tn = (
        df
        .filter(
            (col("Diabetes_binary")==0)
            &
            (col("prediction")==0)
        )
        .count()
    )


    fp = (
        df
        .filter(
            (col("Diabetes_binary")==0)
            &
            (col("prediction")==1)
        )
        .count()
    )


    fn = (
        df
        .filter(
            (col("Diabetes_binary")==1)
            &
            (col("prediction")==0)
        )
        .count()
    )



    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn
    )



    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )


    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )


    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )



    print("\nMetrics")

    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)



    # =================================================
    # RANKING METRICS
    # =================================================


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
            col("Diabetes_binary")==1
        )
        .count()
    )



    ranking_window = (

        Window
        .partitionBy(
            lit(1)
        )
        .orderBy(
            desc(
                "positive_probability"
            )
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
            col("Diabetes_binary")==1
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


    places_per_case = (
        top20_limit /
        top20_positive
    )



    # 60% operating point

    cumulative_window = (

        Window
        .partitionBy(
            lit(1)
        )
        .orderBy(
            desc(
                "positive_probability"
            )
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


    target = total_positive * 0.60


    reach60 = (

        ranked
        .filter(
            col("cumulative_positive")
            >= target
        )
        .orderBy(
            "rank"
        )
        .first()

    )


    if reach60:

        contacts60 = reach60["rank"]

        depth60 = (
            contacts60 /
            total_records
        )

    else:

        contacts60 = None
        depth60 = None



    print("\nRanking Metrics")

    print(
        "Top20 Gain:",
        top20_gain
    )

    print(
        "Top20 Lift:",
        top20_lift
    )

    print(
        "Places per true case:",
        places_per_case
    )

    print(
        "Contacts to 60%:",
        contacts60
    )

    print(
        "Depth to 60%:",
        depth60
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
            places_per_case
        ),

        "Contacts_to_60pct": contacts60,

        "Depth_to_60pct": depth60

    }



# =====================================================
# RUN MODELS
# =====================================================


for name, path in models.items():

    df = spark.read.parquet(path)

    result = evaluate_model(
        name,
        df
    )

    results.append(result)



# =====================================================
# MODEL COMPARISON
# =====================================================


comparison = spark.createDataFrame(
    results
)


print("\nFINAL MODEL COMPARISON")


comparison.show(
    truncate=False
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
        "data/final_model_comparison"
    )
)



# =====================================================
# RISKTEST5 BASELINE RANKING
# =====================================================


print("\n")
print("=" * 60)
print("RISKTEST5 BASELINE")
print("=" * 60)



gbt_df = spark.read.parquet(
    "data/result_gbt.parquet"
)



baseline_window = (

    Window
    .partitionBy(
        lit(1)
    )
    .orderBy(
        desc("RiskTest5")
    )

)



baseline_ranked = (

    gbt_df
    .withColumn(
        "baseline_rank",
        row_number()
        .over(
            baseline_window
        )
    )

)



total_records = baseline_ranked.count()


total_positive = (
    baseline_ranked
    .filter(
        col("Diabetes_binary")==1
    )
    .count()
)



top20_limit = int(
    total_records * 0.20
)



baseline_top20 = (
    baseline_ranked
    .filter(
        col("baseline_rank")
        <= top20_limit
    )
)



baseline_positive = (
    baseline_top20
    .filter(
        col("Diabetes_binary")==1
    )
    .count()
)



baseline_capture = (
    baseline_positive /
    total_positive
)



baseline_places = (
    top20_limit /
    baseline_positive
)



print(
    "Baseline Top20 Capture:",
    baseline_capture
)


print(
    "Baseline Places per True Case:",
    baseline_places
)



# =====================================================
# BO4 EQUITY COMPARISON
# =====================================================


print("\n")
print("=" * 60)
print("BO4 EQUITY ANALYSIS")
print("=" * 60)



equity = (

    gbt_df

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
            "GBT_TP"
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
        ),


        spark_sum(

            when(

                (
                    col("Diabetes_binary")==1
                )
                &
                (
                    col("RiskTest5_flag")==1
                ),

                1

            )
            .otherwise(0)

        )
        .alias(
            "RiskTest_TP"
        )

    )

)



equity = (

    equity

    .withColumn(
        "GBT_Recall",

        col("GBT_TP")
        /
        col("Actual_Positive")
    )

    .withColumn(

        "RiskTest_Recall",

        col("RiskTest_TP")
        /
        col("Actual_Positive")

    )

    .withColumn(

        "Difference",

        col("GBT_Recall")
        -
        col("RiskTest_Recall")

    )

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
        "data/bo4_equity_comparison"
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



cluster_df.groupBy(
    "prediction"
).count().show()



print("\nSTEP 8 COMPLETED")



spark.stop()