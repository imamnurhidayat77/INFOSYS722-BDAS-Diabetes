from spark_session import get_spark

from pyspark.ml.feature import VectorAssembler

spark=get_spark()

df=spark.read.parquet(
    "data/clean_diabetes.parquet"
)

target="Diabetes_binary"

features=[
    c for c in df.columns
    if c != target
]

assembler=VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_vector=assembler.transform(df)

df_final=df_vector.select(
    "features",
    target
)

df_final.write.mode("overwrite") \
    .parquet("data/model_ready.parquet")

df_final.show(5)

spark.stop()