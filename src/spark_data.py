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
print("STEP 2 - DATA UNDERSTANDING")
print("==============================")


print("\nNumber of records:")
print(df.count())


print("\nSchema:")
df.printSchema()


print("\nSummary statistics:")
df.describe().show()


spark.stop()