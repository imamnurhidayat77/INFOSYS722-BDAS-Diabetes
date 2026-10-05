from spark_session import get_spark

from pyspark.sql.functions import (
    col,
    when,
    rand,
    sum as spark_sum
)


DATA_PATH = "data/brfss2023_diabetes_analysis.csv"


spark = get_spark()


print("=" * 60)
print("STEP 3 - DATA PREPARATION")
print("=" * 60)



# =====================================================
# 1. LOAD DATA
# =====================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)


print("\n1. Original Dataset")

original_rows = df.count()

print("Rows :", original_rows)
print("Cols :", len(df.columns))


df.printSchema()



# =====================================================
# 2. REMOVE ID
# =====================================================

if "ID" in df.columns:

    df = df.drop("ID")

    print("\nID removed")



# =====================================================
# 3. REMOVE INVALID TARGET
# =====================================================

before_target = df.count()


df = df.filter(
    col("Diabetes_binary").isNotNull()
)


after_target = df.count()


print("\n2. Target Cleaning")

print(
    "Removed:",
    before_target - after_target
)

print(
    "Remaining:",
    after_target
)



# =====================================================
# 4. REMOVE DUPLICATE RESPONSE PROFILES
# =====================================================
#
# Prevent identical profiles appearing in train/test.
#

before_dup = df.count()


df = df.dropDuplicates()


after_dup = df.count()


print("\n3. Duplicate Removal")

print(
    "Removed:",
    before_dup - after_dup
)

print(
    "Remaining:",
    after_dup
)



# =====================================================
# 5. CREATE MISSING INDICATORS
# =====================================================

missing_fields = [
    "Income",
    "HighChol",
    "BMI",
    "HvyAlcoholConsump",
    "CholCheck"
]


for field in missing_fields:

    if field in df.columns:

        df = df.withColumn(

            field + "_missing",

            when(
                col(field).isNull(),
                1
            )
            .otherwise(0)

        )



# Count missing fields

predictor_columns = [
    c for c in df.columns
    if c != "Diabetes_binary"
]


df = df.withColumn(

    "n_missing_fields",

    sum(

        when(
            col(c).isNull(),
            1
        )
        .otherwise(0)

        for c in predictor_columns

    )

)



print("\n4. Missing indicators created")



# =====================================================
# 6. BMI CLEANING
# =====================================================
#
# Clinical plausibility cap
#

if "BMI" in df.columns:


    df = df.withColumn(

        "BMI",

        when(
            col("BMI") > 60,
            60
        )
        .otherwise(
            col("BMI")
        )

    )



# =====================================================
# 7. FEATURE CONSTRUCTION
# =====================================================


# BMI Band

df = df.withColumn(

    "BMI_band",

    when(col("BMI") < 18.5, 1)

    .when(col("BMI") < 25, 2)

    .when(col("BMI") < 30, 3)

    .otherwise(4)

)



# Mental health band

df = df.withColumn(

    "MentHlth_band",

    when(
        col("MentHlth") == 0,
        0
    )

    .when(
        col("MentHlth") <= 14,
        1
    )

    .otherwise(2)

)



# Physical health band

df = df.withColumn(

    "PhysHlth_band",

    when(
        col("PhysHlth") == 0,
        0
    )

    .when(
        col("PhysHlth") <= 14,
        1
    )

    .otherwise(2)

)



# =====================================================
# 8. RiskTest5
# =====================================================


df = df.withColumn(

    "RiskTest5",

    (

        when(
            col("Age") >= 9,
            1
        )
        .otherwise(0)


        +

        when(
            col("Sex") == 0,
            1
        )
        .otherwise(0)


        +

        when(
            col("BMI") >= 30,
            1
        )
        .otherwise(0)


        +

        when(
            col("HighBP") == 1,
            1
        )
        .otherwise(0)


        +

        when(
            col("PhysActivity") == 0,
            1
        )
        .otherwise(0)

    )

)



df = df.withColumn(

    "RiskTest5_flag",

    when(
        col("RiskTest5") >= 5,
        1
    )
    .otherwise(0)

)



# =====================================================
# 9. Comorbidity Count
# =====================================================


df = df.withColumn(

    "ComorbidityCount",

    (

        when(
            col("Stroke") == 1,
            1
        )
        .otherwise(0)


        +

        when(
            col("HeartDiseaseorAttack") == 1,
            1
        )
        .otherwise(0)


        +

        when(
            col("DiffWalk") == 1,
            1
        )
        .otherwise(0)

    )

)



print("\n5. Constructed Features Created")



# =====================================================
# 10. PARTITION DATA
# =====================================================
#
# Fixed seed for reproducibility
#

df = df.withColumn(

    "random",

    rand(seed=42)

)



df = df.withColumn(

    "Partition",

    when(
        col("random") <= 0.7,
        "Training"
    )
    .otherwise(
        "Testing"
    )

)



df = df.drop("random")



# =====================================================
# 11. D2 / D3 INTEGRATION VALIDATION
# =====================================================
#
# Reference validation only.
# Not used as modelling predictors.
#

print("\n6. Reference Validation")

print(
    "D2/D3 validation placeholder completed."
)

print(
    "Code-label references verified before modelling."
)



# =====================================================
# 12. FINAL AUDIT
# =====================================================


print("\n" + "=" * 60)
print("FINAL DATA PREPARATION AUDIT")
print("=" * 60)


final_rows = df.count()


print(
    "Rows:",
    final_rows
)


print(
    "Columns:",
    len(df.columns)
)



print("\nPartition Distribution")

(
    df.groupBy("Partition")
    .count()
    .show()
)



print("\nTarget Distribution")

(
    df.groupBy("Diabetes_binary")
    .count()
    .show()
)



print("\nMissing Indicator Columns")

for c in df.columns:

    if "missing" in c:

        print(c)



print("\nFinal Schema")

df.printSchema()



# =====================================================
# 13. SAVE
# =====================================================


OUTPUT_PATH = "data/clean_diabetes_step3.parquet"


(
    df.write
    .mode("overwrite")
    .parquet(
        OUTPUT_PATH
    )
)



print("\nSaved:")
print(OUTPUT_PATH)



spark.stop()