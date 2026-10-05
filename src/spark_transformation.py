from spark_session import get_spark

from pyspark.sql.functions import col

from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler
)


spark = get_spark()


print("==============================")
print("STEP 4 - DATA TRANSFORMATION")
print("==============================")


# =====================================================
# LOAD PREPARED DATA
# =====================================================

df = spark.read.parquet(
    "data/clean_diabetes.parquet"
)


target = "Diabetes_binary"



# =====================================================
# REMOVE NON-FEATURE COLUMNS
# =====================================================

exclude_columns = [
    "ID",
    "Diabetes_binary",
    "Partition"
]


features = [
    c for c in df.columns
    if c not in exclude_columns
]


print("\nSelected features:")

for f in features:
    print(f)


print("\nNumber of features:")
print(len(features))



# =====================================================
# CHECK NULL VALUES
# =====================================================

print("\nNull value check:")

for c in features:

    null_count = (
        df
        .filter(
            col(c).isNull()
        )
        .count()
    )

    if null_count > 0:
        print(
            c,
            ":",
            null_count
        )



# =====================================================
# HANDLE REMAINING NULL VALUES
# =====================================================
#
# Missing values are handled after feature selection.
# This prevents leakage into unnecessary columns.
#

df = df.fillna(
    0,
    subset=features
)



# =====================================================
# VECTOR ASSEMBLY
# =====================================================

assembler = VectorAssembler(
    inputCols=features,
    outputCol="raw_features",
    handleInvalid="error"
)


df_vector = assembler.transform(
    df
)



print("\nVectorAssembler completed")



# =====================================================
# STANDARD SCALING
# =====================================================
#
# Fit scaler only using Training data.
# Testing data remains unseen.
#

training_data = (
    df_vector
    .filter(
        col("Partition") == "Training"
    )
)


scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)


scaler_model = scaler.fit(
    training_data
)



df_scaled = scaler_model.transform(
    df_vector
)



# =====================================================
# FINAL MODEL DATASET
# =====================================================

df_model = df_scaled.select(
    "features",
    "Diabetes_binary",
    "Partition",
    "Income",
    "RiskTest5",
    "RiskTest5_flag",
    "ComorbidityCount"
)



print("\nModel ready:")
df_model.show(5)



# =====================================================
# SAVE OUTPUT
# =====================================================

(
    df_model
    .write
    .mode("overwrite")
    .parquet(
        "data/model_ready.parquet"
    )
)


print(
    "\nSaved: data/model_ready.parquet"
)


spark.stop()