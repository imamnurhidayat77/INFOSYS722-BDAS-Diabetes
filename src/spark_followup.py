from spark_session import get_spark
from pyspark.sql.functions import col, udf, desc
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = get_spark()


def prob_udf():
    return udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())


# =====================================================
# A. BO4: recall per income band, model vs baseline
# =====================================================

for name, path in [("LR", "data/result_logistic.parquet"), ("GBT", "data/result_gbt.parquet")]:
    df = spark.read.parquet(path).withColumn("prob", prob_udf()(col("probability")))
    base = spark.read.parquet("data/model_ready.parquet").filter(col("Partition") == "Testing")
    print(f"===== BO4 {name} =====")
    for band, cond in [("all", None), ("inc1", col("Income") == 1), ("inc2", col("Income") == 2)]:
        d = df if cond is None else df.filter(cond)
        tp = d.filter((col("Diabetes_binary") == 1) & (col("prediction") == 1.0)).count()
        pos = d.filter(col("Diabetes_binary") == 1).count()
        print(band, "model recall:", tp / pos, f"({tp}/{pos})")
    for band, cond in [("all", None), ("inc1", col("Income") == 1), ("inc2", col("Income") == 2)]:
        b = base if cond is None else base.filter(cond)
        tp = b.filter((col("Diabetes_binary") == 1) & (col("RiskTest5_flag") == 1)).count()
        pos = b.filter(col("Diabetes_binary") == 1).count()
        print(band, "baseline recall:", tp / pos, f"({tp}/{pos})")


# =====================================================
# B. Gains curve + exact 60% depth (GBT + LR)
# =====================================================

for name, path in [("GBT", "data/result_gbt.parquet"), ("LR", "data/result_logistic.parquet")]:
    df = spark.read.parquet(path).withColumn("prob", prob_udf()(col("probability"))).orderBy(desc("prob"))
    total_pos = df.filter(col("Diabetes_binary") == 1).count()
    rows = df.select("Diabetes_binary", "prob").collect()
    n = len(rows)
    print(f"===== GAINS {name} (n={n}, pos={total_pos}) =====")
    for depth in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = int(n * depth)
        cap = sum(1 for r in rows[:k] if r["Diabetes_binary"] == 1) / total_pos
        print(f"depth={depth:.2f} capture={cap:.4f}")


# =====================================================
# C. Threshold sweep (GBT + LR)
# =====================================================

for name, path in [("GBT", "data/result_gbt.parquet"), ("LR", "data/result_logistic.parquet")]:
    df = spark.read.parquet(path).withColumn("prob", prob_udf()(col("probability")))
    print(f"===== THRESH {name} =====")
    for t in [0.3, 0.4, 0.5, 0.6]:
        tp = df.filter((col("Diabetes_binary") == 1) & (col("prob") >= t)).count()
        fp = df.filter((col("Diabetes_binary") == 0) & (col("prob") >= t)).count()
        fn = df.filter((col("Diabetes_binary") == 1) & (col("prob") < t)).count()
        print(f"t={t} recall={tp/(tp+fn):.4f} precision={tp/(tp+fp):.4f} contacted={tp+fp}")


# =====================================================
# D. RF numTrees sensitivity (weighted, like original)
# =====================================================

train = spark.read.parquet("data/model_ready.parquet").filter(col("Partition") == "Training")
test = spark.read.parquet("data/model_ready.parquet").filter(col("Partition") == "Testing")
cc = {r["Diabetes_binary"]: r["count"] for r in train.groupBy("Diabetes_binary").count().collect()}
w = max(cc.values()) / min(cc.values())
from pyspark.sql.functions import when
tr = train.withColumn("weight", when(col("Diabetes_binary") == 1, w).otherwise(1.0))
ev = BinaryClassificationEvaluator(labelCol="Diabetes_binary", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
for nt in [50, 200]:
    m = RandomForestClassifier(featuresCol="features", labelCol="Diabetes_binary",
        weightCol="weight", numTrees=nt, maxDepth=8, seed=42).fit(tr)
    print(f"RF trees={nt} AUC={ev.evaluate(m.transform(test)):.6f}")
spark.stop()
