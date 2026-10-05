from spark_session import get_spark

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
# REMOVE DATA LEAKAGE COLUMNS
# =====================================================

exclude_columns = [
    "ID",
    "Diabetes_binary",
    "Partition"
]



# =====================================================
# DEFINE FEATURES EXPLICITLY
# =====================================================

features = [
    c for c in df.columns
    if c not in exclude_columns
]


print("Selected features:")
for f in features:
    print(f)


print("\nNumber of features:")
print(len(features))



# =====================================================
# VECTOR ASSEMBLY
# =====================================================

assembler = VectorAssembler(
    inputCols=features,
    outputCol="raw_features"
)


df_vector = assembler.transform(df)



# =====================================================
# STANDARD SCALING
# =====================================================

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)


scaler_model = scaler.fit(
    df_vector.filter(
        df_vector.Partition == "Training"
    )
)


df_model = scaler_model.transform(
    df_vector
)



# =====================================================
# KEEP IMPORTANT COLUMNS
# =====================================================

df_model = df_model.select(
    "features",
    target,
    "Partition",
    "Income",
    "RiskTest5",
    "ComorbidityCount"
)



print("\nModel ready:")
df_model.show(5)



# =====================================================
# SAVE
# =====================================================

(
    df_model
    .write
    .mode("overwrite")
    .parquet(
        "data/model_ready.parquet"
    )
)


print("\nSaved: model_ready.parquet")


spark.stop()