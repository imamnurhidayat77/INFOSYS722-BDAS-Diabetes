from spark_session import get_spark

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

spark=get_spark()

prediction=spark.read.parquet(
    "../output/rf_prediction"
)

auc_eval=BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

auc=auc_eval.evaluate(prediction)

accuracy_eval=MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy=accuracy_eval.evaluate(prediction)

print("AUC:",auc)

print("Accuracy:",accuracy)


spark.stop()