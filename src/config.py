"""Shared configuration for the mortgage-risk-ml project.

Every setting can be overridden with an environment variable, which is how
secrets and environment-specific values stay out of the code. On AWS Glue the
job parameters are injected by the Glue console; locally we fall back to the
defaults below (dev Kafka on localhost, data under ./data).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    # Kafka / streaming
    bootstrap_servers: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092"))
    topic: str = field(default_factory=lambda: _env("MORTGAGE_TOPIC", "mortgage-events"))
    kinesis_stream: str = field(default_factory=lambda: _env("KINESIS_STREAM", "mortgage-events-dev"))
    consumer_group: str = field(default_factory=lambda: _env("CONSUMER_GROUP", "mortgage-risk-dev"))

    # Storage (placeholders - set real values via env vars on AWS)
    raw_bucket: str = field(default_factory=lambda: _env("RAW_BUCKET", "s3://your-bucket/bronze"))
    curated_bucket: str = field(default_factory=lambda: _env("CURATED_BUCKET", "s3://your-bucket/silver"))
    output_bucket: str = field(default_factory=lambda: _env("OUTPUT_BUCKET", "s3://your-bucket/gold"))
    checkpoint_dir: str = field(default_factory=lambda: _env("CHECKPOINT_DIR", "./checkpoints"))

    # Glue job metadata (informational; real ARNs come from the AWS console)
    glue_job_name: str = field(default_factory=lambda: _env("GLUE_JOB_NAME", "mortgage-risk-streaming"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    glue_iam_role: str = field(default_factory=lambda: _env("GLUE_IAM_ROLE", "arn:aws:iam::ACCOUNT_ID:role/glue-service-role"))

    # Spark tuning
    app_name: str = "MortgageRiskStreaming"
    max_rate_per_partition: str = field(default_factory=lambda: _env("MAX_RATE", "500"))
    watermark_minutes: str = field(default_factory=lambda: _env("WATERMARK_MIN", "30"))

    # Model paths and risk thresholds
    model_dir: str = field(default_factory=lambda: _env("MODEL_DIR", "./models"))
    high_risk_threshold: float = 0.65
    seed: int = 42

    def summary(self) -> str:
        return (
            f"Settings(topic={self.topic}, brokers={self.bootstrap_servers}, "
            f"region={self.aws_region}, model_dir={self.model_dir})"
        )

    def bronze_path(self, table: str = "events") -> str:
        """S3 or local path for a bronze landing table."""
        return f"{self.raw_bucket}/{table}"

    def gold_path(self, table: str = "state_risk") -> str:
        """S3 or local path for a gold aggregate table."""
        return f"{self.output_bucket}/{table}"

    def checkpoint_path(self, stream: str) -> str:
        """Checkpoint directory for one streaming query."""
        return f"{self.checkpoint_dir}/{stream}"


def ensure_dirs(cfg: Settings) -> None:
    """Create local directories used in dev mode. No-op on S3 paths."""
    import pathlib

    for path in (cfg.checkpoint_dir, cfg.model_dir):
        if not path.startswith("s3://"):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def load() -> Settings:
    """Convenience entry point used by scripts and tests."""
    cfg = Settings()
    ensure_dirs(cfg)
    return cfg


if __name__ == "__main__":
    print(load().summary())
