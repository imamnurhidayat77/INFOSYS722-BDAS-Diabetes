from spark_session import get_spark


from pyspark.sql.functions import (
    col,
    desc,
    udf
)


from pyspark.sql.types import DoubleType


from pyspark.ml.linalg import Vector


from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    ClusteringEvaluator
)



spark = get_spark()



print("==============================")
print("STEP 8 - INTERPRETATION")
print("==============================")



# =====================================================
# EXTRACT POSITIVE PROBABILITY
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
# CLASSIFICATION EVALUATION
# =====================================================


for name, path in models.items():


    print("\n==============================")
    print(name)
    print("==============================")


    df = spark.read.parquet(path)



    auc = auc_eval.evaluate(df)



    accuracy = MulticlassClassificationEvaluator(
        labelCol="Diabetes_binary",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(df)



    precision = MulticlassClassificationEvaluator(
        labelCol="Diabetes_binary",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(df)



    recall = MulticlassClassificationEvaluator(
        labelCol="Diabetes_binary",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(df)



    f1 = MulticlassClassificationEvaluator(
        labelCol="Diabetes_binary",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(df)



    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)



    print("\nConfusion Matrix")

    df.groupBy(
        "Diabetes_binary",
        "prediction"
    ).count().show()



    # =================================================
    # TOP 20 GAIN / LIFT
    # =================================================


    ranked = (
        df
        .withColumn(
            "positive_probability",
            probability_udf(
                col("probability")
            )
        )
        .orderBy(
            desc("positive_probability")
        )
    )



    total_records = ranked.count()


    top20_records = int(
        total_records * 0.2
    )


    top20 = ranked.limit(
        top20_records
    )



    total_positive = (
        ranked
        .filter(
            col("Diabetes_binary") == 1
        )
        .count()
    )



    top20_positive = (
        top20
        .filter(
            col("Diabetes_binary") == 1
        )
        .count()
    )



    gain = (
        top20_positive /
        total_positive
    )


    lift = (
        gain /
        0.2
    )



    print(
        "Top 20% Gain:",
        gain
    )


    print(
        "Top 20% Lift:",
        lift
    )



    results.append(
        (
            name,
            float(auc),
            float(accuracy),
            float(precision),
            float(recall),
            float(f1),
            float(gain),
            float(lift)
        )
    )



# =====================================================
# SAVE MODEL COMPARISON
# =====================================================


metric_df = spark.createDataFrame(
    results,
    [
        "Model",
        "AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Top20_Gain",
        "Top20_Lift"
    ]
)



print("\nMODEL COMPARISON")

metric_df.show()



metric_df.write \
    .mode("overwrite") \
    .option(
        "header",
        True
    ) \
    .csv(
        "data/model_metrics"
    )



# =====================================================
# KMEANS
# =====================================================


print("\n==============================")
print("KMEANS EVALUATION")
print("==============================")


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
    "Silhouette Score:",
    silhouette
)



print("\nEVALUATION COMPLETED")



spark.stop()