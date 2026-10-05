from functools import reduce

from spark_session import get_spark

from pyspark.sql.functions import (
    col,
    when,
    rand,
    lit,
)


# =====================================================
# PATHS
# =====================================================

DATA_PATH = "data/brfss2023_diabetes_analysis.csv"
D2_PATH = "data/ref_code_labels.csv"
D3_PATH = "data/ref_bmi_thresholds.csv"
OUTPUT_PATH = "data/clean_diabetes_step3.parquet"


spark = get_spark()

print("=" * 70)
print("STEP 3 - DATA PREPARATION")
print("=" * 70)


# =====================================================
# 1. LOAD RESPONDENT DATA
# =====================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

print("\n1. Original Dataset")
print("Rows :", df.count())
print("Cols :", len(df.columns))
df.printSchema()


# =====================================================
# 2. REMOVE IDENTIFIER
# =====================================================

if "ID" in df.columns:
    df = df.drop("ID")
    print("\nID removed")


# =====================================================
# 3. REMOVE RECORDS WITH NO TARGET
# =====================================================

before_target = df.count()

df = df.filter(
    col("Diabetes_binary").isNotNull()
)

after_target = df.count()

print("\n2. Target Cleaning")
print("Removed   :", before_target - after_target)
print("Remaining :", after_target)


# =====================================================
# 4. REMOVE DUPLICATE RESPONSE PROFILES
# =====================================================
#
# ID has already been removed, so duplicate detection is based
# on the analytical response profile rather than the row key.
# Removing duplicates before partitioning prevents identical
# feature-target profiles appearing in both train and test.
# =====================================================

before_dup = df.count()

df = df.dropDuplicates()

after_dup = df.count()

print("\n3. Duplicate Removal")
print("Removed   :", before_dup - after_dup)
print("Remaining :", after_dup)


# Keep the original predictor names before adding indicators / constructs.
base_predictors = [
    c for c in df.columns
    if c != "Diabetes_binary"
]


# =====================================================
# 5. CREATE MISSINGNESS INDICATORS
# =====================================================

missing_fields = [
    "Income",
    "HighChol",
    "BMI",
    "HvyAlcoholConsump",
    "CholCheck",
]

for field in missing_fields:
    if field in df.columns:
        df = df.withColumn(
            f"{field}_missing",
            when(col(field).isNull(), lit(1)).otherwise(lit(0)),
        )

missing_terms = [
    when(col(c).isNull(), lit(1)).otherwise(lit(0))
    for c in base_predictors
]

if missing_terms:
    df = df.withColumn(
        "n_missing_fields",
        reduce(lambda a, b: a + b, missing_terms),
    )
else:
    df = df.withColumn("n_missing_fields", lit(0))

print("\n4. Missing Indicators Created")
print("Indicators:")
for field in missing_fields:
    if f"{field}_missing" in df.columns:
        print("-", f"{field}_missing")
print("- n_missing_fields")


# =====================================================
# 6. BMI PLAUSIBILITY CAP
# =====================================================
#
# Retain high-risk respondents, but limit implausible telephone-
# survey extremes. Missing BMI stays missing for training-only
# imputation in Step 4.
# =====================================================

if "BMI" in df.columns:
    bmi_over_60 = df.filter(col("BMI") > 60).count()

    df = df.withColumn(
        "BMI",
        when(col("BMI") > 60, lit(60.0)).otherwise(col("BMI")),
    )

    print("\n5. BMI Plausibility Treatment")
    print("BMI values capped at 60:", bmi_over_60)


# =====================================================
# 7. LOAD D2 / D3 REFERENCE DATA
# =====================================================
#
# D2 = code-label definitions.
# D3 = BMI band definitions.
# These are reference sources, not respondent-level predictors.
# =====================================================

d2 = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(D2_PATH)
)

d3 = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(D3_PATH)
)

print("\n6. Reference Data Loaded")
print("D2 rows:", d2.count())
print("D3 rows:", d3.count())


# =====================================================
# 8. D2 CODE-LABEL INTEGRATION / VALIDATION
# =====================================================
#
# Temporary many-to-one reference joins verify that every observed
# non-null code has a definition in D2. Labels are not retained as
# model predictors because they duplicate the numeric codes.
# =====================================================

print("\n7. D2 Code-Label Validation")

reference_variables = [
    "GenHlth",
    "Age",
    "Education",
    "Income",
    "Sex",
    "Diabetes_binary",
]

for variable in reference_variables:
    if variable not in df.columns:
        raise ValueError(f"Expected field missing from D1a: {variable}")

    observed = (
        df
        .filter(col(variable).isNotNull())
        .select(col(variable).cast("double").alias("observed_code"))
        .distinct()
    )

    reference = (
        d2
        .filter(col("variable") == variable)
        .select(
            col("code").cast("double").alias("reference_code"),
            col("label").alias("reference_label"),
        )
    )

    check = observed.join(
        reference,
        observed["observed_code"] == reference["reference_code"],
        "left",
    )

    invalid_count = check.filter(
        col("reference_label").isNull()
    ).count()

    observed_codes = [
        r["observed_code"]
        for r in observed.orderBy("observed_code").collect()
    ]

    print(
        f"{variable}: observed={observed_codes}; "
        f"unmapped={invalid_count}"
    )

    if invalid_count != 0:
        print("Unmapped codes:")
        (
            check
            .filter(col("reference_label").isNull())
            .show(truncate=False)
        )
        raise ValueError(
            f"D2 validation failed for {variable}: "
            f"{invalid_count} observed code(s) have no reference label."
        )

print("D2 code-label validation: PASSED")


# =====================================================
# 9. D3 BMI REFERENCE INTEGRATION
# =====================================================
#
# The WHO international rows in D3 directly define BMI_band.
# This makes D3 an implemented reference source rather than a
# decorative file. Null BMI remains null until Step 4 imputation.
# =====================================================

who_bmi_ref = (
    d3
    .filter(col("standard") == "WHO_International")
    .select(
        col("band_code").cast("int").alias("band_code"),
        col("band_label"),
        col("bmi_lower_inclusive").cast("double").alias("bmi_lower_inclusive"),
        col("bmi_upper_exclusive").cast("double").alias("bmi_upper_exclusive"),
    )
    .orderBy("band_code")
)

who_rules = who_bmi_ref.collect()

if len(who_rules) != 4:
    raise ValueError(
        "D3 validation failed: expected four WHO_International BMI bands."
    )

print("\n8. D3 WHO BMI Reference")
who_bmi_ref.show(truncate=False)

bmi_band_expr = when(
    col("BMI").isNull(),
    lit(None).cast("int"),
)

for rule in who_rules:
    lower = float(rule["bmi_lower_inclusive"])
    upper = float(rule["bmi_upper_exclusive"])
    code = int(rule["band_code"])

    bmi_band_expr = bmi_band_expr.when(
        (col("BMI") >= lit(lower)) & (col("BMI") < lit(upper)),
        lit(code),
    )

bmi_band_expr = bmi_band_expr.otherwise(
    lit(None).cast("int")
)

df = df.withColumn("BMI_band", bmi_band_expr)


# Temporary range join validates that the constructed BMI band matches D3.
bmi_observed = (
    df
    .filter(col("BMI").isNotNull())
    .select("BMI", "BMI_band")
    .alias("obs")
)

bmi_ref_alias = who_bmi_ref.alias("ref")

bmi_check = bmi_observed.join(
    bmi_ref_alias,
    (
        (col("obs.BMI") >= col("ref.bmi_lower_inclusive"))
        &
        (col("obs.BMI") < col("ref.bmi_upper_exclusive"))
    ),
    "left",
)

bmi_unmapped = bmi_check.filter(
    col("ref.band_code").isNull()
).count()

bmi_mismatch = bmi_check.filter(
    col("ref.band_code").isNotNull()
    &
    (col("obs.BMI_band") != col("ref.band_code"))
).count()

print("D3 non-null BMI rows checked:", bmi_observed.count())
print("D3 unmapped BMI rows        :", bmi_unmapped)
print("D3 BMI band mismatches      :", bmi_mismatch)

if bmi_unmapped != 0 or bmi_mismatch != 0:
    raise ValueError(
        "D3 BMI validation failed: check BMI thresholds / constructed band."
    )

print("D3 BMI reference integration: PASSED")


# =====================================================
# 10. HEALTH-DAY BAND CONSTRUCTION
# =====================================================
#
# Preserve missing values here so Step 4 can impute from the
# training partition only. Do not silently turn missing responses
# into the highest-risk category.
# =====================================================

df = df.withColumn(
    "MentHlth_band",
    when(col("MentHlth").isNull(), lit(None).cast("int"))
    .when(col("MentHlth") == 0, lit(0))
    .when(col("MentHlth") <= 14, lit(1))
    .otherwise(lit(2)),
)

df = df.withColumn(
    "PhysHlth_band",
    when(col("PhysHlth").isNull(), lit(None).cast("int"))
    .when(col("PhysHlth") == 0, lit(0))
    .when(col("PhysHlth") <= 14, lit(1))
    .otherwise(lit(2)),
)


# =====================================================
# 11. RISKTEST5 CONSTRUCTION
# =====================================================
#
# Five available CDC risk-test components:
#   age, sex, BMI, high blood pressure, physical inactivity.
# Family history and gestational-diabetes history are unavailable.
#
# The scoring mirrors the project's Iteration 3 approximation:
#   Age: <40=0, 40-49=1, 50-59=2, >=60=3
#   BMI: <25=0, 25-29.9=1, 30-39.9=2, >=40=3
#   Male sex=1, HighBP=1, Physically inactive=1
# Maximum available score = 9; referral stand-in = score >= 5.
#
# Important correction: D2 defines Sex=0 as Female and Sex=1 as Male.
# =====================================================

age_points = (
    when(col("Age") < 5, lit(0))
    .when(col("Age") < 7, lit(1))
    .when(col("Age") < 9, lit(2))
    .otherwise(lit(3))
)

bmi_points = (
    when(col("BMI") < 25, lit(0))
    .when(col("BMI") < 30, lit(1))
    .when(col("BMI") < 40, lit(2))
    .otherwise(lit(3))
)

risk_component_missing = (
    col("Age").isNull()
    | col("Sex").isNull()
    | col("BMI").isNull()
    | col("HighBP").isNull()
    | col("PhysActivity").isNull()
)

risk_score = (
    age_points
    + bmi_points
    + when(col("Sex") == 1, lit(1)).otherwise(lit(0))
    + when(col("HighBP") == 1, lit(1)).otherwise(lit(0))
    + when(col("PhysActivity") == 0, lit(1)).otherwise(lit(0))
)

df = df.withColumn(
    "RiskTest5",
    when(
        risk_component_missing,
        lit(None).cast("int"),
    ).otherwise(risk_score.cast("int")),
)

df = df.withColumn(
    "RiskTest5_flag",
    when(
        col("RiskTest5").isNull(),
        lit(None).cast("int"),
    )
    .when(col("RiskTest5") >= 5, lit(1))
    .otherwise(lit(0)),
)


# =====================================================
# 12. COMORBIDITY COUNT
# =====================================================
#
# Same compact descriptive burden used in Iteration 3:
# Stroke + HeartDiseaseorAttack + DiffWalk.
# Missing constituent values remain missing until Step 4 imputation.
# =====================================================

comorbidity_missing = (
    col("Stroke").isNull()
    | col("HeartDiseaseorAttack").isNull()
    | col("DiffWalk").isNull()
)

comorbidity_score = (
    when(col("Stroke") == 1, lit(1)).otherwise(lit(0))
    + when(col("HeartDiseaseorAttack") == 1, lit(1)).otherwise(lit(0))
    + when(col("DiffWalk") == 1, lit(1)).otherwise(lit(0))
)

df = df.withColumn(
    "ComorbidityCount",
    when(
        comorbidity_missing,
        lit(None).cast("int"),
    ).otherwise(comorbidity_score.cast("int")),
)

print("\n9. Constructed Features Created")
print("- BMI_band (from D3 WHO reference)")
print("- MentHlth_band")
print("- PhysHlth_band")
print("- RiskTest5 (0-9)")
print("- RiskTest5_flag (>=5)")
print("- ComorbidityCount")


# =====================================================
# 13. CONSTRUCTION AUDIT
# =====================================================

print("\n10. Constructed Feature Audit")

print("RiskTest5 range / distribution:")
(
    df.groupBy("RiskTest5")
    .count()
    .orderBy("RiskTest5")
    .show(20, truncate=False)
)

print("BMI band distribution:")
(
    df.groupBy("BMI_band")
    .count()
    .orderBy("BMI_band")
    .show(truncate=False)
)

print("Comorbidity distribution:")
(
    df.groupBy("ComorbidityCount")
    .count()
    .orderBy("ComorbidityCount")
    .show(truncate=False)
)


# =====================================================
# 14. REPRODUCIBLE 70:30 PARTITION
# =====================================================

# rand(seed=42) makes the assignment reproducible for this fixed
# Spark input / execution pipeline. The stored Partition column is
# reused by every later step; models do not resplit the data.

df = df.withColumn(
    "_partition_random",
    rand(seed=42),
)

df = df.withColumn(
    "Partition",
    when(
        col("_partition_random") <= 0.7,
        lit("Training"),
    ).otherwise(lit("Testing")),
)

df = df.drop("_partition_random")


# =====================================================
# 15. FINAL DATA PREPARATION AUDIT
# =====================================================

print("\n" + "=" * 70)
print("FINAL DATA PREPARATION AUDIT")
print("=" * 70)

final_rows = df.count()

print("Rows    :", final_rows)
print("Columns :", len(df.columns))

print("\nPartition Distribution")
(
    df.groupBy("Partition")
    .count()
    .orderBy("Partition")
    .show()
)

print("\nTarget Distribution")
(
    df.groupBy("Diabetes_binary")
    .count()
    .orderBy("Diabetes_binary")
    .show()
)

print("\nMissing Indicator Columns")
for c in df.columns:
    if "missing" in c:
        print(c)

print("\nFinal Schema")
df.printSchema()


# =====================================================
# 16. SAVE STEP 3 OUTPUT
# =====================================================

(
    df.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nD2 integration status: PASSED")
print("D3 integration status: PASSED")
print("STEP 3 COMPLETED")

spark.stop()
