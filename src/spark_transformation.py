from spark_session import get_spark

from pyspark.ml.feature import VectorAssembler


spark = get_spark()


df = spark.read.parquet(
    "data/clean_diabetes.parquet"
)


print("==============================")
print("STEP 4 - DATA TRANSFORMATION")
print("==============================")


target = "Diabetes_binary"


features = [
    c for c in df.columns
    if c != target and c != "ID"
]


print("Number of features:")
print(len(features))


assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)


df_model = assembler.transform(df)


df_model = df_model.select(
    "features",
    target
)


df_model.show(5)


df_model.write \
    .mode("overwrite") \
    .parquet(
        "data/model_ready.parquet"
    )


print("Saved: model_ready.parquet")


spark.stop()