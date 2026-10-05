from pyspark.sql import SparkSession


def get_spark():

    spark = (
        SparkSession.builder
        .appName(
            "INFOSYS722-BDAS-Diabetes"
        )
        .getOrCreate()
    )

    return spark