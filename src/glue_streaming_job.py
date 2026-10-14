"""Spark Structured Streaming job shaped for AWS Glue streaming.

The same code runs in two places:
1. Locally with `spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1`
   reading from Kafka, which is great for development and tests.
2. As a Glue 4.0 streaming job reading from Kinesis, which is how it runs in
   production. On Glue the `source` argument switches the reader, and the
   `getResolvedOptions` args supply the stream name.

The job lands raw events in bronze (append-only Parquet), dedupes and enriches
into silver, and computes per-state risk aggregates into gold. Malformed rows
go to a quarantine path instead of killing the stream.
"""
from __future__ import annotations

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from .config import Settings

# The event schema is declared explicitly - never infer it in production,
# since a single malformed batch can silently change the types.
EVENT_SCHEMA = StructType(
    [
        StructField("event_type", StringType(), True),
        StructField("application_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("loan_amount", IntegerType(), True),
        StructField("annual_income", IntegerType(), True),
        StructField("credit_score", IntegerType(), True),
        StructField("loan_to_value", DoubleType(), True),
        StructField("debt_to_income", DoubleType(), True),
        StructField("interest_rate", DoubleType(), True),
        StructField("term_months", IntegerType(), True),
        StructField("property_type", StringType(), True),
        StructField("purpose", StringType(), True),
        StructField("state", StringType(), True),
        StructField("first_time_buyer", StringType(), True),
        StructField("amount_due", DoubleType(), True),
        StructField("amount_paid", DoubleType(), True),
        StructField("payment_status", StringType(), True),
        StructField("days_late", IntegerType(), True),
    ]
)


def build_spark(cfg: Settings, source: str) -> SparkSession:
    """Build the session. On Glue the job parameters configure the rest."""
    builder = SparkSession.builder.appName(cfg.app_name)
    if source == "kafka":  # local dev path
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )
    return builder.getOrCreate()


def read_stream(spark: SparkSession, cfg: Settings, source: str):
    """Read raw events from Kafka (local) or Kinesis (Glue)."""
    if source == "kinesis":
        return (
            spark.readStream.format("kinesis")
            .option("streamName", cfg.kinesis_stream)
            .option("endpointUrl", f"https://kinesis.{cfg.aws_region}.amazonaws.com")
            .option("startingPosition", "TRIM_HORIZON")
            .load()
        )
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg.bootstrap_servers)
        .option("subscribe", cfg.topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", cfg.max_rate_per_partition)
        .load()
    )


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "kafka"
    cfg = Settings()
    spark = build_spark(cfg, source)
    spark.sparkContext.setLogLevel("WARN")
    print(f"Starting {cfg.app_name} (source={source})")

    raw = read_stream(spark, cfg, source)
    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS payload")
        .select(F.from_json("payload", EVENT_SCHEMA).alias("e"), F.col("payload"))
        .select("e.*", "payload")
    )
    # Rows that failed to parse have a null key field - quarantine them
    good = parsed.filter(F.col("application_id").isNotNull())
    bad = parsed.filter(F.col("application_id").isNull())

    enriched = (
        good.withColumn("event_time", F.to_timestamp("timestamp"))
        .withColumn("payment_ratio", F.col("amount_paid") / F.col("amount_due"))
        .withWatermark("event_time", f"{cfg.watermark_minutes} minutes")
        .dropDuplicates(["application_id", "event_type", "timestamp"])
    )

    # Bronze: raw append-only landing, partitioned by event type
    bronze = (
        enriched.writeStream.format("parquet")
        .option("path", f"{cfg.raw_bucket}/events")
        .option("checkpointLocation", f"{cfg.checkpoint_dir}/bronze")
        .partitionBy("event_type")
        .outputMode("append")
        .trigger(processingTime="2 minutes")
        .start()
    )
    # Quarantine: malformed payloads for later inspection
    quarantine = (
        bad.writeStream.format("parquet")
        .option("path", f"{cfg.raw_bucket}/quarantine")
        .option("checkpointLocation", f"{cfg.checkpoint_dir}/quarantine")
        .outputMode("append")
        .start()
    )
    # Gold: per-state rolling risk aggregates over 1-hour windows
    gold = (
        enriched.filter(F.col("event_type") == "payment")
        .groupBy(F.window("event_time", "1 hour"), "state")
        .agg(
            F.count("*").alias("payment_count"),
            F.avg("payment_ratio").alias("avg_payment_ratio"),
            F.sum(F.when(F.col("payment_status") == "missed", 1).otherwise(0)).alias("missed_count"),
        )
        .writeStream.format("parquet")
        .option("path", f"{cfg.output_bucket}/state_risk")
        .option("checkpointLocation", f"{cfg.checkpoint_dir}/gold")
        .outputMode("append")
        .trigger(processingTime="5 minutes")
        .start()
    )
    print("Streams started: bronze, quarantine, gold. Awaiting termination...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
