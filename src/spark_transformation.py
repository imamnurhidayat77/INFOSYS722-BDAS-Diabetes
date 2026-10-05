from spark_session import get_spark

from pyspark.sql.functions import col
from pyspark.ml.feature import (
    Imputer,
    VectorAssembler,
    ChiSqSelector,
    StandardScaler
)


spark = get_spark()

print("=" * 60)
print("STEP 4 - DATA TRANSFORMATION")
print("=" * 60)


# =====================================================
# 1. LOAD STEP 3 OUTPUT
# =====================================================

INPUT_PATH = "data/clean_diabetes_step3.parquet"

df = spark.read.parquet(INPUT_PATH)

print("\n1. Loaded Step 3 Output")
print("Rows :", df.count())
print("Cols :", len(df.columns))


# =====================================================
# 2. SPLIT TRAIN / TEST USING STORED PARTITION
# =====================================================

train_df = df.filter(col("Partition") == "Training")
test_df = df.filter(col("Partition") == "Testing")

print("\n2. Partition Check")
print("Training rows :", train_df.count())
print("Testing rows  :", test_df.count())


# =====================================================
# 3. DEFINE INPUT FEATURES
# =====================================================
#
# Exclude target and partition only.
# Step 4.1 reduction will be done formally using ChiSqSelector.
#

exclude_cols = ["Diabetes_binary", "Partition"]

input_features = [
    c for c in df.columns
    if c not in exclude_cols
]

print("\n3. Input Features Before Reduction")
print("Feature count :", len(input_features))
for f in input_features:
    print(f)


# =====================================================
# 4. TRAINING-ONLY IMPUTATION
# =====================================================
#
# Use median imputation learned from training data only.
# Apply same fitted parameters to test data.
#

imputer = Imputer(
    inputCols=input_features,
    outputCols=input_features,
    strategy="median"
)

imputer_model = imputer.fit(train_df)

train_imputed = imputer_model.transform(train_df)
test_imputed = imputer_model.transform(test_df)

print("\n4. Imputation Completed")
print("Training-only imputation model fitted and applied.")


# =====================================================
# 5. ASSEMBLE RAW FEATURE VECTOR
# =====================================================

raw_assembler = VectorAssembler(
    inputCols=input_features,
    outputCol="raw_features"
)

train_vector = raw_assembler.transform(train_imputed)
test_vector = raw_assembler.transform(test_imputed)

print("\n5. Raw Feature Vector Created")
print("Vector column: raw_features")


# =====================================================
# 6. FEATURE REDUCTION (4.1)
# =====================================================
#
# Use ChiSqSelector on training data only.
# Select top features associated with target.
#

TOP_FEATURES = 24

selector = ChiSqSelector(
    numTopFeatures=TOP_FEATURES,
    featuresCol="raw_features",
    outputCol="selected_raw_features",
    labelCol="Diabetes_binary"
)

selector_model = selector.fit(train_vector)

train_selected = selector_model.transform(train_vector)
test_selected = selector_model.transform(test_vector)

selected_indices = selector_model.selectedFeatures
selected_feature_names = [input_features[i] for i in selected_indices]

print("\n6. Feature Reduction Completed")
print("Original feature count :", len(input_features))
print("Selected feature count :", len(selected_feature_names))

print("\nSelected Feature Indices:")
print(selected_indices)

print("\nSelected Feature Names:")
for f in selected_feature_names:
    print(f)


# =====================================================
# 7. PROJECTION / SCALING (4.2)
# =====================================================
#
# StandardScaler fitted on training selected features only.
#

scaler = StandardScaler(
    inputCol="selected_raw_features",
    outputCol="features",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(train_selected)

train_final = scaler_model.transform(train_selected)
test_final = scaler_model.transform(test_selected)

print("\n7. Projection / Scaling Completed")
print("Output vector column: features")


# =====================================================
# 8. FINAL AUDIT
# =====================================================

print("\n" + "=" * 60)
print("FINAL STEP 4 AUDIT")
print("=" * 60)

print("\nTraining rows :", train_final.count())
print("Testing rows  :", test_final.count())

print("\nColumns in transformed training set:")
for c in train_final.columns:
    print(c)

print("\nSelected feature count:", len(selected_feature_names))
print("Selected features:")
for f in selected_feature_names:
    print("-", f)


# =====================================================
# 9. SAVE OUTPUT
# =====================================================

TRAIN_OUTPUT = "data/train_step4_transformed.parquet"
TEST_OUTPUT = "data/test_step4_transformed.parquet"

(
    train_final.write
    .mode("overwrite")
    .parquet(TRAIN_OUTPUT)
)

(
    test_final.write
    .mode("overwrite")
    .parquet(TEST_OUTPUT)
)

print("\nSaved outputs:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)


# =====================================================
# 10. OPTIONAL: SAVE SELECTED FEATURE LIST AS TXT
# =====================================================

feature_output_path = "data/selected_features_step4.txt"

with open(feature_output_path, "w") as f:
    f.write("STEP 4 SELECTED FEATURES\n")
    f.write("========================\n")
    f.write(f"Original feature count: {len(input_features)}\n")
    f.write(f"Selected feature count: {len(selected_feature_names)}\n\n")
    for i, feat in enumerate(selected_feature_names, start=1):
        f.write(f"{i}. {feat}\n")

print(feature_output_path)


spark.stop()