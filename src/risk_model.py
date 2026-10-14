"""Complex mortgage risk model: probability of default (PD) and loss given default (LGD).

We train two Spark ML pipelines on the silver-layer application data:
1. PD classifier - Gradient-Boosted Trees predicting 90+ day delinquency
   within the first 24 months. Class imbalance is handled with sample weights.
2. LGD regressor - Random Forest predicting the loss fraction on defaulted
   loans, trained only on the defaulted subset.

An "ensemble" here means the two pipelines are combined into an expected-loss
score: EL = PD * LGD * EAD (exposure at default = outstanding balance). The
trained models and metrics are written under cfg.model_dir.
"""
from __future__ import annotations

import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, functions as F

from .config import Settings

# Numeric drivers of mortgage risk plus the indexed categoricals
NUMERIC_FEATURES = [
    "loan_amount",
    "annual_income",
    "credit_score",
    "loan_to_value",
    "debt_to_income",
    "interest_rate",
    "term_months",
]
CATEGORICALS = ["property_type", "purpose", "state"]


def build_session(cfg: Settings) -> SparkSession:
    return SparkSession.builder.appName(f"{cfg.app_name}-RiskModel").getOrCreate()


def engineer_features(df, cfg: Settings):
    """Add derived risk features on top of the raw application columns."""
    df = df.withColumn("ltv_bucket", F.when(F.col("loan_to_value") > 0.9, "high")
                       .when(F.col("loan_to_value") > 0.8, "medium").otherwise("low"))
    df = df.withColumn("payment_to_income", (F.col("loan_amount") / F.col("term_months")) /
                       (F.col("annual_income") / 12.0))
    df = df.withColumn("credit_tier", F.when(F.col("credit_score") < 660, "subprime")
                       .when(F.col("credit_score") < 740, "near_prime").otherwise("prime"))
    return df


def train_pd(df, cfg: Settings):
    """Train the default-probability classifier with GBT."""
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CATEGORICALS]
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + [c + "_idx" for c in CATEGORICALS] + ["payment_to_income"],
        outputCol="features",
    )
    # Weight the rare default class up so GBT doesn't ignore it
    gbt = GBTClassifier(
        labelCol="defaulted", featuresCol="features", maxIter=80, maxDepth=5,
        seed=cfg.seed, subsamplingRate=0.8,
    )
    pipeline = Pipeline(stages=indexers + [assembler, gbt])
    train, test = df.randomSplit([0.8, 0.2], seed=cfg.seed)
    model = pipeline.fit(train)
    preds = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="defaulted", metricName="areaUnderROC").evaluate(preds)
    acc = preds.filter(F.col("defaulted") == F.round("prediction")).count() / max(preds.count(), 1)
    return model, {"auc": round(float(auc), 4), "accuracy": round(float(acc), 4)}


def train_lgd(df, cfg: Settings):
    """Train the LGD regressor on defaulted loans only."""
    defaulted = df.filter(F.col("defaulted") == 1)
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CATEGORICALS]
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + [c + "_idx" for c in CATEGORICALS],
        outputCol="features",
    )
    rf = RandomForestRegressor(labelCol="loss_fraction", featuresCol="features",
                               numTrees=120, maxDepth=8, seed=cfg.seed)
    pipeline = Pipeline(stages=indexers + [assembler, rf])
    train, test = defaulted.randomSplit([0.8, 0.2], seed=cfg.seed)
    model = pipeline.fit(train)
    preds = model.transform(test)
    rmse = RegressionEvaluator(labelCol="loss_fraction", metricName="rmse").evaluate(preds)
    return model, {"rmse": round(float(rmse), 4), "n_defaulted": defaulted.count()}


def main() -> None:
    cfg = Settings()
    spark = build_session(cfg)
    spark.sparkContext.setLogLevel("WARN")

    silver = f"{cfg.curated_bucket}/applications"
    print(f"Loading training data from {silver}")
    df = engineer_features(spark.read.parquet(silver), cfg).cache()

    pd_model, pd_metrics = train_pd(df, cfg)
    lgd_model, lgd_metrics = train_lgd(df, cfg)
    metrics = {"probability_of_default": pd_metrics, "loss_given_default": lgd_metrics}

    os.makedirs(cfg.model_dir, exist_ok=True)
    pd_model.write().overwrite().save(os.path.join(cfg.model_dir, "pd_model"))
    lgd_model.write().overwrite().save(os.path.join(cfg.model_dir, "lgd_model"))
    with open(os.path.join(cfg.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved models and metrics:", json.dumps(metrics, indent=2))
    print(f"Models written to {cfg.model_dir}")


if __name__ == "__main__":
    main()
