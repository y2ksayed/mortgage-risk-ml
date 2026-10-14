"""Simulates mortgage application and payment events for the streaming pipeline.

The generator produces two event types:
- application: a new mortgage application with borrower and loan attributes
- payment: a monthly payment (on time, late, or missed)

Everything is synthetic, so the pipeline can be exercised end to end without
touching any real customer data. Run it with `--rate` events per second and
`--duration` in seconds; both are optional.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer

from .config import Settings

STATES = ["CA", "TX", "FL", "NY", "IL", "WA", "GA", "NC"]
PROPERTY_TYPES = ["single_family", "condo", "townhouse", "multi_family"]
PURPOSES = ["purchase", "refinance", "cash_out_refi"]


def random_application(app_id: int) -> dict:
    """Build one synthetic mortgage application event."""
    loan_amount = random.randint(120_000, 900_000)
    annual_income = random.randint(45_000, 400_000)
    credit_score = random.randint(580, 840)
    ltv = round(random.uniform(0.55, 0.98), 3)
    dti = round(random.uniform(0.15, 0.55), 3)
    return {
        "event_type": "application",
        "application_id": f"APP-{app_id:06d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_to_value": ltv,
        "debt_to_income": dti,
        "interest_rate": round(random.uniform(0.055, 0.095), 4),
        "term_months": random.choice([180, 240, 360]),
        "property_type": random.choice(PROPERTY_TYPES),
        "purpose": random.choice(PURPOSES),
        "state": random.choice(STATES),
        "first_time_buyer": random.random() < 0.4,
    }


def random_payment(app_id: int) -> dict:
    """Build one synthetic monthly payment event for an existing application."""
    roll = random.random()
    status = "on_time" if roll < 0.85 else ("late" if roll < 0.95 else "missed")
    return {
        "event_type": "payment",
        "application_id": f"APP-{app_id:06d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount_due": round(random.uniform(900, 4200), 2),
        "amount_paid": 0.0 if status == "missed" else round(random.uniform(900, 4200), 2),
        "payment_status": status,
        "days_late": 0 if status == "on_time" else random.randint(1, 60),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate mortgage events into Kafka")
    parser.add_argument("--rate", type=float, default=2.0, help="events per second")
    parser.add_argument("--duration", type=int, default=60, help="run duration in seconds")
    parser.add_argument("--topic", default=None, help="override Kafka topic")
    args = parser.parse_args()

    cfg = Settings()
    topic = args.topic or cfg.topic
    producer = KafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"Producing to topic '{topic}' at ~{args.rate} events/sec")

    deadline = time.time() + args.duration
    app_counter, sent = 0, 0
    while time.time() < deadline:
        app_counter += 1
        # Roughly one application per ten payments, like a real book
        event = random_application(app_counter) if sent % 10 == 0 else random_payment(
            random.randint(1, max(app_counter, 1))
        )
        producer.send(topic, value=event)
        sent += 1
        if sent % 50 == 0:
            print(f"sent {sent} events ({event['event_type']})")
        time.sleep(1.0 / args.rate)
    producer.flush()
    print(f"Done. Sent {sent} events.")


if __name__ == "__main__":
    main()
