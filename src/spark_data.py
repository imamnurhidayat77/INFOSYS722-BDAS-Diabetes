from spark_session import get_spark

DATA_PATH = "../data/brfss2023_diabetes_analysis.csv"

spark = get_spark()

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

print("Number of records:")
print(df.count())

print("\nSchema:")
df.printSchema()

print("\nSummary:")
df.describe().show()

spark.stop()