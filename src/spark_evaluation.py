from spark_session import get_spark

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


spark = get_spark()


print("==============================")
print("STEP 8 - INTERPRETATION")
print("==============================")


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



binary_eval = BinaryClassificationEvaluator(
    labelCol="Diabetes_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)


multi_eval = MulticlassClassificationEvaluator(
    labelCol="Diabetes_binary"
)



for name,path in models.items():

    print("\n================")
    print(name)
    print("================")


    df = spark.read.parquet(path)


    auc = binary_eval.evaluate(df)


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


    print("Confusion Matrix")

    df.groupBy(
        "Diabetes_binary",
        "prediction"
    ).count().show()



# =====================================================
# CLUSTER EVALUATION
# =====================================================

cluster = spark.read.parquet(
    "data/result_cluster.parquet"
)


print("================")
print("CLUSTER RESULT")
print("================")


cluster.groupBy(
    "prediction"
).count().show()



spark.stop()