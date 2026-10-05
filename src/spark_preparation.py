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
