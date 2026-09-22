from pyspark.sql import SparkSession


def get_spark():

    spark = (
        SparkSession.builder
        .appName("INFOSYS722-BDAS-Diabetes")
        .master("local[*]")
        .getOrCreate()
    )

    return spark


if __name__ == "__main__":

    spark = get_spark()

    print("Spark Version:", spark.version)

    spark.stop()