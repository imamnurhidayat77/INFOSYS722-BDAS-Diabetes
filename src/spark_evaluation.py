from spark_session import get_spark

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

import pandas as pd


spark = get_spark()


print("==============================")
print("STEP 8 - MODEL EVALUATION")
print("==============================")


prediction = spark.read.parquet(
    "output/rf_prediction"
)


print("\nConfusion Matrix")

prediction.groupBy(
    "label",
    "prediction"
).count().show()


# AUC

auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)


auc = auc_eval.evaluate(
    prediction
)


# Accuracy

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)


accuracy = acc_eval.evaluate(
    prediction
)


print("AUC:")
print(auc)


print("Accuracy:")
print(accuracy)



# Save metrics

metrics = pd.DataFrame(
    {
        "model": [
            "Random Forest"
        ],
        "AUC": [
            auc
        ],
        "Accuracy": [
            accuracy
        ]
    }
)


metrics.to_csv(
    "output/bdas_metrics.csv",
    index=False
)


print("Saved: bdas_metrics.csv")


spark.stop()