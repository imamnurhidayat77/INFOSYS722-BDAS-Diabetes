from spark_session import get_spark

from pyspark.sql.functions import (
    col,
    when,
    lit,
    rand
)


DATA_PATH = "data/brfss2023_diabetes_analysis.csv"


spark = get_spark()


print("==============================")
print("STEP 3 - DATA PREPARATION")
print("==============================")


# =====================================================
# 1. LOAD DATA
# =====================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)


print("\nOriginal dataset:")
print("Rows :", df.count())
print("Cols :", len(df.columns))


df.printSchema()



# =====================================================
# 2. REMOVE ONLY INVALID TARGET
# =====================================================
# Diabetes_binary adalah target.
# Record tanpa target tidak berguna untuk supervised learning.

df_clean = df.filter(
    col("Diabetes_binary").isNotNull()
)


print("\nAfter target cleaning:")
print("Rows :", df_clean.count())



# =====================================================
# 3. CREATE BMI BAND
# =====================================================

df_clean = df_clean.withColumn(
    "BMI_band",
    when(col("BMI") < 18.5, 1)
    .when(col("BMI") < 25, 2)
    .when(col("BMI") < 30, 3)
    .otherwise(4)
)



# =====================================================
# 4. CREATE HEALTH DAY BANDS
# =====================================================

df_clean = df_clean.withColumn(
    "MentHlth_band",
    when(col("MentHlth") == 0, 0)
    .when(col("MentHlth") <= 14, 1)
    .otherwise(2)
)


df_clean = df_clean.withColumn(
    "PhysHlth_band",
    when(col("PhysHlth") == 0, 0)
    .when(col("PhysHlth") <= 14, 1)
    .otherwise(2)
)



# =====================================================
# 5. CREATE RiskTest5
# =====================================================
#
# Approximation of CDC risk test:
# Age
# Sex
# BMI
# HighBP
# PhysicalActivity
#

df_clean = df_clean.withColumn(
    "RiskTest5",

    (
        when(col("Age") >= 9, 1)
        .otherwise(0)

        +

        when(col("Sex") == 0, 1)
        .otherwise(0)

        +

        when(col("BMI") >= 30, 1)
        .otherwise(0)

        +

        when(col("HighBP") == 1, 1)
        .otherwise(0)

        +

        when(col("PhysActivity") == 0, 1)
        .otherwise(0)
    )
)



df_clean = df_clean.withColumn(
    "RiskTest5_flag",
    when(col("RiskTest5") >= 5, 1)
    .otherwise(0)
)



# =====================================================
# 6. CREATE COMORBIDITY COUNT
# =====================================================

df_clean = df_clean.withColumn(
    "ComorbidityCount",

    (
        when(col("Stroke") == 1, 1)
        .otherwise(0)

        +

        when(col("HeartDiseaseorAttack") == 1, 1)
        .otherwise(0)

        +

        when(col("HighBP") == 1, 1)
        .otherwise(0)

        +

        when(col("HighChol") == 1, 1)
        .otherwise(0)
    )
)



# =====================================================
# 7. CREATE TRAINING / TESTING PARTITION
# =====================================================
#
# Keep partition before modelling.
# No randomSplit during ML stage.

df_clean = df_clean.withColumn(
    "Partition",
    when(
        rand(seed=42) <= 0.7,
        "Training"
    )
    .otherwise("Testing")
)



# =====================================================
# 8. DATA AUDIT
# =====================================================

print("\nFinal prepared dataset:")
print("Rows :", df_clean.count())
print("Cols :", len(df_clean.columns))


print("\nPartition:")
df_clean.groupBy(
    "Partition"
).count().show()


print("\nTarget distribution:")
df_clean.groupBy(
    "Diabetes_binary"
).count().show()



# =====================================================
# 9. SAVE OUTPUT
# =====================================================

(
    df_clean
    .write
    .mode("overwrite")
    .parquet(
        "data/clean_diabetes.parquet"
    )
)


print(
    "\nSaved: data/clean_diabetes.parquet"
)


spark.stop()