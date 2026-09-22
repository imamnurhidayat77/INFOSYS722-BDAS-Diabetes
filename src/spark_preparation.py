from spark_session import get_spark


DATA_PATH = "data/brfss2023_diabetes_analysis.csv"


spark = get_spark()


df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)


print("==============================")
print("STEP 3 - DATA PREPARATION")
print("==============================")


print("Before cleaning:")
print(df.count())


df_clean = df.dropna(
    subset=["Diabetes_binary"]
)


df_clean = df_clean.fillna(0)


print("After cleaning:")
print(df_clean.count())


df_clean.write \
    .mode("overwrite") \
    .parquet(
        "data/clean_diabetes.parquet"
    )


print("Saved: clean_diabetes.parquet")


spark.stop()