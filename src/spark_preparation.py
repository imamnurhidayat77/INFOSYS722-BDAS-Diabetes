from spark_session import get_spark
from pyspark.sql.functions import col

DATA_PATH="data/brfss2023_diabetes_analysis.csv"

spark=get_spark()

df=(
    spark.read
    .option("header",True)
    .option("inferSchema",True)
    .csv(DATA_PATH)
)

print("Before cleaning:")
print(df.count())

# remove missing target
df_clean=df.dropna(
    subset=["Diabetes_binary"]
)

# fill missing numeric values
df_clean=df_clean.fillna(0)

print("After cleaning:")
print(df_clean.count())

df_clean.write.mode("overwrite") \
    .parquet("data/clean_diabetes.parquet")

spark.stop()