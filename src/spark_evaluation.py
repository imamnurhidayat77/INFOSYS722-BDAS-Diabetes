from spark_session import get_spark


from pyspark.sql.functions import (
    col,
    row_number,
    count,
    desc,
    sum as spark_sum,
    lit
)


from pyspark.sql.window import Window


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



multi_eval = MulticlassClassificationEvaluator(
    labelCol="Diabetes_binary"
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



    accuracy = multi_eval.evaluate(
        df,
        {
            "metricName":"accuracy"
        }
    )


    precision = multi_eval.evaluate(
        df,
        {
            "metricName":"weightedPrecision"
        }
    )


    recall = multi_eval.evaluate(
        df,
        {
            "metricName":"weightedRecall"
        }
    )


    f1 = multi_eval.evaluate(
        df,
        {
            "metricName":"f1"
        }
    )



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
    # TOP 20% GAIN / LIFT
    # =================================================


    ranked = (
        df
        .withColumn(
            "probability_positive",
            col("probability")[1]
        )
        .orderBy(
            desc("probability_positive")
        )
    )


    total = ranked.count()


    top20_count = int(total * 0.2)



    top20 = ranked.limit(
        top20_count
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
# SAVE CLASSIFICATION METRICS
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
# KMEANS EVALUATION
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



cluster_eval = ClusteringEvaluator(
    featuresCol="features",
    metricName="silhouette"
)



silhouette = cluster_eval.evaluate(
    cluster_df
)



print(
    "Silhouette Score:",
    silhouette
)



print("\nEVALUATION COMPLETED")



spark.stop()