from spark_session import get_spark

from pyspark.sql.functions import (
    col,
    when,
    lit,
    rand,
    count,
    sum as spark_sum
)

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


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


original_rows = df.count()


print("\nOriginal dataset")
print("Rows :", original_rows)
print("Cols :", len(df.columns))


df.printSchema()



# =====================================================
# 2. REMOVE ID COLUMN
# =====================================================
#
# ID provides no predictive information and can create
# memorisation behaviour.
#

if "ID" in df.columns:

    df = df.drop("ID")

    print("\nRemoved ID column")



# =====================================================
# 3. REMOVE INVALID TARGET
# =====================================================

df = df.filter(
    col("Diabetes_binary").isNotNull()
)


after_target_rows = df.count()


print("\nAfter target cleaning")
print(
    "Removed:",
    original_rows - after_target_rows
)

print(
    "Rows:",
    after_target_rows
)



# =====================================================
# 4. REMOVE DUPLICATE RESPONSE PROFILES
# =====================================================
#
# Prevent identical feature-target profiles appearing
# in both train and test.
#

before_duplicates = df.count()


duplicate_columns = df.columns


df = df.dropDuplicates(
    duplicate_columns
)


after_duplicates = df.count()


print("\nDuplicate removal")
print(
    "Removed:",
    before_duplicates - after_duplicates
)

print(
    "Remaining:",
    after_duplicates
)



# =====================================================
# 5. CREATE MISSING VALUE INDICATORS
# =====================================================
#
# Preserve information carried by missing responses.
#

missing_columns = [
    "Income",
    "HighChol",
    "BMI",
    "HvyAlcoholConsump",
    "CholCheck"
]


for c in missing_columns:

    if c in df.columns:

        df = df.withColumn(
            f"{c}_missing",

            when(
                col(c).isNull(),
                1
            )
            .otherwise(0)
        )



# Total missing count

all_predictors = [
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

        for c in all_predictors
    )
)



print("\nMissing indicators created")



# =====================================================
# 6. BMI PLAUSIBILITY CLEANING
# =====================================================
#
# Clinical cap rather than statistical removal.
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
# 7. CONSTRUCT BMI BAND
# =====================================================


df = df.withColumn(

    "BMI_band",

    when(col("BMI") < 18.5, 1)

    .when(col("BMI") < 25, 2)

    .when(col("BMI") < 30, 3)

    .otherwise(4)

)



# =====================================================
# 8. CONSTRUCT HEALTH DAY BANDS
# =====================================================


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
# 9. CONSTRUCT RiskTest5
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
# 10. CONSTRUCT COMORBIDITY COUNT
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



# =====================================================
# 11. CREATE TRAIN / TEST PARTITION
# =====================================================
#
# Fixed seed for reproducibility.
# Stratification handled later in modelling if required.
#

df = df.withColumn(
    "rand_value",
    rand(seed=42)
)


df = df.withColumn(

    "Partition",

    when(
        col("rand_value") <= 0.7,
        "Training"
    )

    .otherwise(
        "Testing"
    )

)



df = df.drop(
    "rand_value"
)



# =====================================================
# 12. FINAL AUDIT
# =====================================================


print("\n==============================")
print("FINAL PREPARED DATASET")
print("==============================")


print(
    "Rows:",
    df.count()
)

print(
    "Columns:",
    len(df.columns)
)



print("\nPartition distribution")

(
    df.groupBy("Partition")
    .count()
    .show()
)



print("\nTarget distribution")

(
    df.groupBy("Diabetes_binary")
    .count()
    .show()
)



print("\nMissing indicator summary")

(
    df.select(
        [
            c for c in df.columns
            if "missing" in c
        ]
    )
    .describe()
    .show()
)



print("\nFinal schema")

df.printSchema()



# =====================================================
# 13. SAVE OUTPUT
# =====================================================


(
    df.write
    .mode("overwrite")
    .parquet(
        "data/clean_diabetes_step3.parquet"
    )
)


print(
    "\nSaved:"
    " data/clean_diabetes_step3.parquet"
)


spark.stop()