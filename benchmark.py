#!/usr/bin/env python3
"""
benchmark.py — Synthetic data benchmark for the Claude Code memory system.

Three realistic projects:
  1. "fintech-api"    — A Python/FastAPI fintech backend
  2. "mobile-app"     — A React Native mobile app
  3. "data-pipeline"  — A Spark/Airflow data engineering pipeline

Tests at 100, 500, 1000, 10000 items across all three repos.

Measures:
  - Query latency (session_recall, prompt_recall, search_facts)
  - Correctness (scope isolation, auto-promotion, dedup, decay)
  - Token budget compliance
  - Storage size (DuckDB file)

Requires: Ollama running with nomic-embed-text pulled.
Usage: python3 benchmark.py
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg
from memory import db, embeddings, recall
from memory.decay import compute_decay_score
from memory.retrieval import parallel_retrieve, reciprocal_rank_fusion, ScoredItem
from memory.consolidation import run_consolidation, run_semantic_forgetting
from memory.config import GLOBAL_SCOPE, SESSION_TOKEN_BUDGET, PROMPT_TOKEN_BUDGET, CHARS_PER_TOKEN

# ── Check Ollama ──────────────────────────────────────────────────────────

if not embeddings.is_ollama_available():
    print("ERROR: Ollama is not running or nomic-embed-text is not pulled.")
    print("  ollama pull nomic-embed-text")
    sys.exit(1)

# ── Three realistic project scopes ───────────────────────────────────────

SCOPES = {
    "fintech": "/Users/dev/projects/fintech-api",
    "mobile":  "/Users/dev/projects/mobile-app",
    "pipeline": "/Users/dev/projects/data-pipeline",
}

# ── Synthetic data generators ─────────────────────────────────────────────

FINTECH_FACTS = [
    # Core stack
    ("FastAPI serves the REST endpoints", "technical", "long", "high"),
    ("PostgreSQL is the primary database", "technical", "long", "high"),
    ("Stripe handles payment processing", "technical", "long", "high"),
    ("Redis caches session tokens", "technical", "long", "high"),
    ("Alembic manages database migrations", "technical", "long", "high"),
    ("SQLAlchemy ORM is used for all database queries", "technical", "long", "high"),
    ("We use Pydantic v2 for request/response validation", "technical", "long", "high"),
    ("The CI pipeline runs on GitHub Actions", "technical", "long", "high"),
    # Auth & security
    ("JWT tokens expire after 24 hours", "technical", "medium", "high"),
    ("Rate limiting is set to 100 req/min per user", "technical", "medium", "high"),
    ("The KYC flow uses Plaid for identity verification", "technical", "long", "high"),
    ("Two-factor auth is mandatory for transfers over $10,000", "decision", "long", "high"),
    ("OAuth2 scopes restrict third-party API access to read-only by default", "technical", "long", "high"),
    ("Session cookies use SameSite=Strict and HttpOnly flags", "technical", "medium", "high"),
    ("IP geolocation blocks login attempts from OFAC-sanctioned countries", "decision", "long", "high"),
    ("Password hashing uses argon2id with 64MB memory cost", "technical", "long", "high"),
    # Payments & ledger
    ("Account balances are stored as integer cents to avoid float rounding", "technical", "long", "high"),
    ("Celery workers process async transaction reconciliation", "technical", "medium", "high"),
    ("The fraud detection model runs in a sidecar container", "technical", "medium", "medium"),
    ("ACH transfers settle in 2-3 business days via Plaid", "technical", "medium", "high"),
    ("Wire transfers require manual compliance officer approval over $50k", "decision", "long", "high"),
    ("The double-entry ledger ensures every debit has a matching credit", "technical", "long", "high"),
    ("Idempotency keys prevent duplicate payment processing on retries", "technical", "long", "high"),
    ("The dispute resolution workflow has 5 states from open to resolved", "contextual", "medium", "medium"),
    ("Chargeback notifications arrive via Stripe webhook within 24 hours", "technical", "medium", "high"),
    ("Monthly interest calculations run as a scheduled Celery beat task", "technical", "medium", "high"),
    # API design
    ("The API response time p99 target is 200ms", "numerical", "medium", "high"),
    ("Webhook retries use exponential backoff with 5 max attempts", "technical", "medium", "high"),
    ("The compliance team requires audit logs for all transactions", "decision", "long", "high"),
    ("API versioning uses URL path prefix /v1/ and /v2/", "technical", "long", "high"),
    ("GraphQL is available for the admin dashboard but not public API", "decision", "medium", "high"),
    ("Pagination uses cursor-based tokens instead of offset/limit", "technical", "long", "high"),
    ("Request tracing uses OpenTelemetry with W3C trace context headers", "technical", "medium", "high"),
    ("Error responses follow RFC 7807 Problem Details format", "technical", "long", "high"),
    # Infrastructure
    ("The staging environment mirrors prod with synthetic data", "contextual", "medium", "medium"),
    ("Docker Compose is used for local development", "technical", "medium", "medium"),
    ("Production runs on AWS ECS Fargate with auto-scaling", "technical", "long", "high"),
    ("Database backups run every 6 hours with 30-day retention", "technical", "medium", "high"),
    ("CloudFront CDN serves static assets and the marketing site", "technical", "medium", "medium"),
    ("RDS Multi-AZ provides automatic failover for PostgreSQL", "technical", "long", "high"),
    ("The VPC has separate public and private subnets across 3 AZs", "technical", "long", "high"),
    ("NAT Gateway enables outbound internet from private subnets", "technical", "medium", "medium"),
    # Monitoring & observability
    ("Prometheus scrapes application metrics every 15 seconds", "technical", "medium", "high"),
    ("Grafana dashboards show real-time transaction volume and error rates", "technical", "medium", "medium"),
    ("PagerDuty alerts the on-call engineer for p99 latency over 500ms", "technical", "medium", "high"),
    ("Structured JSON logging outputs to CloudWatch Log Groups", "technical", "long", "high"),
    ("The health check endpoint returns database and Redis connectivity status", "technical", "medium", "high"),
    # Compliance & business
    ("SOC 2 Type II audit is completed annually in Q4", "contextual", "long", "high"),
    ("PCI DSS compliance requires quarterly vulnerability scans", "decision", "long", "high"),
    ("The money transmitter license covers 48 US states", "contextual", "long", "high"),
    ("Customer support tickets average 4-hour first response time", "numerical", "medium", "medium"),
    ("The referral program gives both parties a $25 bonus", "contextual", "short", "medium"),
    ("Monthly active users exceeded 150,000 in the last quarter", "numerical", "medium", "high"),
    ("The waitlist conversion rate is approximately 34%", "numerical", "short", "medium"),
    ("Revenue per user averages $8.50 per month from interchange fees", "numerical", "medium", "high"),
    # Testing
    ("pytest runs 2,400 tests with 89% line coverage", "numerical", "medium", "high"),
    ("Integration tests use testcontainers for PostgreSQL and Redis", "technical", "long", "high"),
    ("Load testing with Locust simulates 10,000 concurrent users", "technical", "medium", "high"),
    ("Contract tests verify Stripe webhook payload schemas", "technical", "medium", "high"),
    ("Mutation testing with mutmut covers the payment processing module", "technical", "medium", "medium"),
]

MOBILE_FACTS = [
    # Core stack
    ("React Native 0.74 is the mobile framework", "technical", "long", "high"),
    ("Expo is used for build and deployment", "technical", "long", "high"),
    ("TypeScript is enforced across the codebase", "technical", "long", "high"),
    ("Zustand manages client-side state", "technical", "long", "high"),
    ("React Navigation handles screen routing", "technical", "long", "high"),
    ("The app targets iOS 15+ and Android 12+", "technical", "long", "high"),
    ("Hermes JS engine is enabled for Android performance", "technical", "long", "high"),
    # Notifications & messaging
    ("Push notifications use Firebase Cloud Messaging", "technical", "long", "high"),
    ("In-app notification center stores the last 100 notifications locally", "technical", "medium", "high"),
    ("Silent push notifications trigger background data sync", "technical", "medium", "high"),
    ("Notification channels on Android separate transaction alerts from marketing", "technical", "medium", "medium"),
    ("Rich push notifications display transaction amount and merchant logo", "technical", "medium", "high"),
    # Auth & biometrics
    ("Biometric auth is available on supported devices", "technical", "medium", "high"),
    ("Face ID and fingerprint are preferred over PIN entry", "decision", "long", "high"),
    ("The auth token refreshes silently 5 minutes before expiry", "technical", "medium", "high"),
    ("Failed biometric attempts fall back to 6-digit PIN after 3 tries", "technical", "medium", "high"),
    ("Device binding prevents account access from unregistered phones", "technical", "long", "high"),
    # UI/UX
    ("The design system uses a custom theme with 8px grid", "technical", "medium", "medium"),
    ("Accessibility labels are required on all interactive elements", "decision", "long", "high"),
    ("The onboarding flow has 4 screens with skip option", "contextual", "medium", "medium"),
    ("The splash screen loads in under 1.5 seconds", "numerical", "short", "medium"),
    ("Dark mode follows the system preference by default", "technical", "medium", "high"),
    ("Bottom sheet modals replace full-screen overlays for quick actions", "technical", "medium", "medium"),
    ("Skeleton loading screens show during API data fetches", "technical", "medium", "medium"),
    ("The transaction list uses FlashList for 60fps scrolling on large datasets", "technical", "long", "high"),
    ("Haptic feedback confirms successful transfers with a medium impact", "technical", "short", "medium"),
    ("Pull-to-refresh updates account balance and recent transactions", "technical", "medium", "medium"),
    ("Lottie animations play during transfer processing states", "technical", "short", "medium"),
    # Offline & storage
    ("Offline mode caches the last 50 transactions locally", "technical", "medium", "high"),
    ("MMKV replaces AsyncStorage for faster key-value persistence", "technical", "long", "high"),
    ("Pending offline transactions queue and sync when connectivity returns", "technical", "medium", "high"),
    ("SQLite stores the local transaction cache with 30-day TTL", "technical", "medium", "high"),
    # Networking
    ("Deep links follow the scheme myfinapp://", "technical", "medium", "high"),
    ("CodePush enables OTA updates for JS bundle changes", "technical", "medium", "high"),
    ("Axios interceptors attach auth headers and handle 401 token refresh", "technical", "medium", "high"),
    ("Certificate pinning validates the API server TLS certificate", "technical", "long", "high"),
    ("Request retry logic uses jitter to prevent thundering herd on reconnect", "technical", "medium", "high"),
    # Testing & CI
    ("Detox is used for E2E testing", "technical", "long", "high"),
    ("Sentry captures crash reports and performance traces", "technical", "long", "high"),
    ("The app bundle size target is under 25MB", "numerical", "medium", "high"),
    ("App Store review requires privacy nutrition labels", "contextual", "medium", "medium"),
    ("Jest runs 1,800 unit tests covering all Zustand stores and utilities", "numerical", "medium", "high"),
    ("Maestro scripts automate the critical user flows for regression testing", "technical", "medium", "high"),
    ("Flipper is used for debugging network requests and React component trees", "technical", "short", "medium"),
    ("The EAS Build pipeline produces separate dev, staging, and prod binaries", "technical", "long", "high"),
    ("Crashlytics tracks crash-free rate which is currently at 99.4%", "numerical", "medium", "high"),
    # Platform-specific
    ("iOS uses Keychain for secure credential storage", "technical", "long", "high"),
    ("Android uses EncryptedSharedPreferences for sensitive data", "technical", "long", "high"),
    ("Universal Links on iOS handle app-to-web deep link handoff", "technical", "medium", "high"),
    ("Android App Links verified via assetlinks.json on the API domain", "technical", "medium", "high"),
    ("The iOS widget shows account balance using WidgetKit", "technical", "medium", "medium"),
    ("Android Adaptive Icons follow the 108dp safe zone specification", "technical", "short", "low"),
    # Business metrics
    ("Daily active users average 42,000 with peak on Fridays", "numerical", "medium", "high"),
    ("App Store rating is 4.7 with 12,000 reviews", "numerical", "medium", "medium"),
    ("The median session duration is 2 minutes 15 seconds", "numerical", "short", "medium"),
    ("30-day retention rate is 68% for users who complete onboarding", "numerical", "medium", "high"),
    ("The check deposit feature processes 3,200 checks per day via OCR", "numerical", "medium", "high"),
]

PIPELINE_FACTS = [
    # Core stack
    ("Apache Spark processes the raw transaction data", "technical", "long", "high"),
    ("Airflow orchestrates the daily ETL pipeline", "technical", "long", "high"),
    ("Data lands in S3 as Parquet files partitioned by date", "technical", "long", "high"),
    ("Snowflake is the analytics data warehouse", "technical", "long", "high"),
    ("dbt transforms raw tables into analytics models", "technical", "long", "high"),
    ("Great Expectations validates data quality on ingestion", "technical", "long", "high"),
    ("Delta Lake provides ACID transactions on the data lake", "technical", "long", "high"),
    ("Kafka streams real-time events to the fraud detection system", "technical", "long", "high"),
    ("AWS Glue Catalog serves as the metastore", "technical", "long", "high"),
    ("Terraform provisions all pipeline infrastructure", "technical", "long", "high"),
    # Scheduling & orchestration
    ("The pipeline runs daily at 02:00 UTC", "technical", "medium", "high"),
    ("Spark cluster autoscales between 2 and 20 nodes", "technical", "medium", "high"),
    ("The SLA for daily pipeline completion is 06:00 UTC", "numerical", "medium", "high"),
    ("Airflow DAGs use the KubernetesPodOperator for Spark job submission", "technical", "long", "high"),
    ("Task retries are capped at 3 with 10-minute delays between attempts", "technical", "medium", "high"),
    ("Airflow sensors poll S3 for new bank feed files before triggering ETL", "technical", "medium", "high"),
    ("The backfill mechanism replays failed partitions without reprocessing all data", "technical", "long", "high"),
    ("DAG dependencies enforce that staging tables load before production views", "technical", "medium", "high"),
    # Data quality & governance
    ("PII columns are encrypted with AES-256 at rest", "technical", "long", "high"),
    ("The data retention policy is 7 years for financial data", "decision", "long", "high"),
    ("Schema evolution uses Avro for backward compatibility", "technical", "long", "high"),
    ("The reconciliation job compares bank feeds with internal ledger", "technical", "medium", "high"),
    ("Data lineage is tracked in OpenLineage and visualized in Marquez", "technical", "long", "high"),
    ("Column-level access control in Snowflake restricts PII to authorized roles", "technical", "long", "high"),
    ("Anomaly detection flags transactions deviating more than 3 sigma from daily mean", "technical", "medium", "high"),
    ("Row-count assertions catch silent data drops between pipeline stages", "technical", "medium", "high"),
    ("Freshness checks alert if source tables haven't updated in 6 hours", "technical", "medium", "high"),
    ("The data catalog tags each table with owner, SLA tier, and PII classification", "technical", "long", "high"),
    # Storage & formats
    ("Parquet files use Snappy compression averaging 8:1 ratio on transaction data", "technical", "medium", "high"),
    ("The raw zone retains unmodified source extracts for 90 days", "decision", "long", "high"),
    ("Iceberg tables enable time-travel queries on the curated zone", "technical", "long", "high"),
    ("Partitioning by transaction_date and currency reduces scan volume by 95%", "technical", "long", "high"),
    ("Z-ordering on customer_id optimizes point lookups in Delta tables", "technical", "medium", "high"),
    # Analytics & BI
    ("Tableau dashboards refresh from Snowflake views hourly", "technical", "medium", "medium"),
    ("The executive revenue dashboard aggregates data from 14 source tables", "contextual", "medium", "medium"),
    ("Looker semantic layer defines canonical metric definitions for ARR and churn", "technical", "long", "high"),
    ("Ad-hoc query access uses Snowflake's reader accounts with 30-day expiry", "technical", "medium", "medium"),
    ("dbt tests enforce unique and not-null constraints on all surrogate keys", "technical", "long", "high"),
    # Cost & performance
    ("Cost monitoring alerts trigger at $500/day Spark spend", "numerical", "medium", "high"),
    ("The test suite uses synthetic PII-free datasets", "contextual", "medium", "medium"),
    ("Spark adaptive query execution reduced average job time by 40%", "numerical", "medium", "high"),
    ("Snowflake warehouse auto-suspends after 5 minutes of idle time", "technical", "medium", "high"),
    ("Monthly pipeline compute costs average $12,000 across all environments", "numerical", "medium", "high"),
    ("Spot instances for Spark workers reduce EC2 costs by 65%", "technical", "medium", "high"),
    # Streaming
    ("Kafka Connect ingests CDC events from PostgreSQL using Debezium", "technical", "long", "high"),
    ("The Kafka cluster has 12 partitions per topic for throughput", "technical", "medium", "high"),
    ("Schema Registry enforces Avro compatibility for all Kafka topics", "technical", "long", "high"),
    ("Consumer lag monitoring alerts when processing falls behind by 5 minutes", "technical", "medium", "high"),
    ("The dead-letter queue captures malformed events for manual review", "technical", "medium", "high"),
    # ML features
    ("Feature Store serves precomputed features to the fraud model via Redis", "technical", "long", "high"),
    ("Training data snapshots are versioned in DVC with S3 backend", "technical", "medium", "high"),
    ("The feature engineering pipeline produces 147 features per transaction", "numerical", "medium", "high"),
    ("Model serving uses SageMaker endpoints with A/B traffic splitting", "technical", "long", "high"),
    ("Feature drift monitoring compares daily distributions using PSI metric", "technical", "medium", "high"),
]

# Facts that naturally appear across all three projects (candidates for auto-promotion)
CROSS_PROJECT_FACTS = [
    ("The team uses GitHub for version control", "technical", "long", "high"),
    ("Code reviews require at least one approval", "decision", "long", "high"),
    ("Python 3.12 is the standard runtime version", "technical", "long", "high"),
    ("Datadog monitors all production services", "technical", "long", "high"),
    ("The user prefers explicit error messages over silent failures", "personal", "long", "high"),
    ("All secrets are stored in AWS Secrets Manager", "technical", "long", "high"),
    ("The team follows trunk-based development", "decision", "long", "high"),
    ("Pre-commit hooks enforce linting and formatting", "technical", "medium", "high"),
]

FINTECH_ENTITIES = ["FastAPI", "PostgreSQL", "Stripe", "Redis", "Celery", "Alembic", "Plaid", "SQLAlchemy", "Pydantic", "GitHub Actions"]
MOBILE_ENTITIES = ["React Native", "Expo", "TypeScript", "Zustand", "Firebase", "Detox", "Sentry", "Hermes", "CodePush", "React Navigation"]
PIPELINE_ENTITIES = ["Apache Spark", "Airflow", "Snowflake", "dbt", "Delta Lake", "Kafka", "AWS Glue", "Terraform", "Great Expectations", "Tableau"]
CROSS_ENTITIES = ["GitHub", "Python", "Datadog", "AWS", "Docker"]

FINTECH_DECISIONS = [
    ("Use PostgreSQL over MySQL for ACID compliance", "long"),
    ("Implement rate limiting at the API gateway level", "medium"),
    ("Store all monetary values as integer cents", "long"),
]
MOBILE_DECISIONS = [
    ("Use Zustand over Redux for simpler state management", "long"),
    ("Target iOS 15+ to use latest SwiftUI bridges", "long"),
    ("Implement biometric auth before PIN fallback", "medium"),
]
PIPELINE_DECISIONS = [
    ("Use Parquet over CSV for columnar storage efficiency", "long"),
    ("Run the ETL at 02:00 UTC to avoid business hours", "medium"),
    ("Encrypt PII at the column level, not table level", "long"),
]

FINTECH_RELATIONSHIPS = [
    ("FastAPI", "PostgreSQL", "uses", "FastAPI connects to PostgreSQL via SQLAlchemy"),
    ("FastAPI", "Redis", "uses", "FastAPI caches sessions in Redis"),
    ("Stripe", "FastAPI", "enables", "Stripe SDK is called from FastAPI payment endpoints"),
    ("Celery", "Redis", "uses", "Celery uses Redis as message broker"),
    ("Alembic", "PostgreSQL", "manages", "Alembic runs migrations against PostgreSQL"),
]
MOBILE_RELATIONSHIPS = [
    ("React Native", "Expo", "uses", "React Native apps are built and deployed via Expo"),
    ("Zustand", "React Native", "part_of", "Zustand manages state inside React Native components"),
    ("Firebase", "React Native", "enables", "Firebase provides push notifications to the app"),
    ("Sentry", "React Native", "enables", "Sentry captures crashes from the React Native app"),
    ("Detox", "React Native", "enables", "Detox runs E2E tests against the React Native app"),
]
PIPELINE_RELATIONSHIPS = [
    ("Airflow", "Apache Spark", "enables", "Airflow triggers Spark jobs on schedule"),
    ("Apache Spark", "Delta Lake", "uses", "Spark reads and writes Delta Lake tables"),
    ("dbt", "Snowflake", "uses", "dbt transforms data inside Snowflake"),
    ("Kafka", "Apache Spark", "enables", "Kafka streams events consumed by Spark"),
    ("Terraform", "AWS Glue", "manages", "Terraform provisions the Glue Catalog"),
]


# Templates for generating semantically distinct facts beyond the base pools.
# Each template produces a unique sentence when formatted with varying parameters.
FINTECH_TEMPLATES = [
    ("The {endpoint} endpoint handles {volume} requests per day", "numerical", "medium", "high"),
    ("The {service} microservice communicates via {protocol} with {target}", "technical", "medium", "high"),
    ("Database table {table} has {count} million rows and is partitioned by {col}", "technical", "medium", "high"),
    ("The {feature} feature was shipped in sprint {sprint} and affects {module}", "contextual", "short", "medium"),
    ("Error code {code} means {meaning} and triggers {action}", "technical", "medium", "high"),
    ("The {env} environment uses {instance} instances with {mem}GB RAM each", "technical", "medium", "medium"),
    ("API client {client} has a rate limit of {limit} requests per {window}", "technical", "medium", "high"),
    ("The {report} report aggregates data from {sources} upstream tables", "contextual", "medium", "medium"),
    ("Celery task {task} has a {timeout}-second timeout and {retries} max retries", "technical", "medium", "high"),
    ("The {metric} metric threshold for alerting is {threshold}", "numerical", "medium", "high"),
]
MOBILE_TEMPLATES = [
    ("Screen {screen} renders in {time}ms on {device} devices", "numerical", "short", "medium"),
    ("The {component} component uses {hook} hook for {purpose}", "technical", "medium", "medium"),
    ("Navigation flow from {from_screen} to {to_screen} passes {params} params", "technical", "medium", "medium"),
    ("The {gesture} gesture handler on {screen} triggers {action}", "technical", "short", "medium"),
    ("Font {font} at size {size}px is used for {element} throughout the app", "technical", "medium", "low"),
    ("The {animation} animation plays at {fps}fps and lasts {duration}ms", "technical", "short", "low"),
    ("Image caching uses {strategy} strategy with {maxSize}MB disk limit", "technical", "medium", "high"),
    ("The {feature} feature flag is controlled by {service} and defaults to {default}", "technical", "medium", "medium"),
    ("Localization supports {count} languages with {fallback} as fallback", "technical", "medium", "high"),
    ("The {sensor} sensor data is sampled every {interval}ms for {purpose}", "technical", "short", "medium"),
]
PIPELINE_TEMPLATES = [
    ("The {table} table in {zone} zone contains {rows} million rows", "numerical", "medium", "high"),
    ("Spark job {job} processes {volume}GB of data in approximately {time} minutes", "numerical", "medium", "high"),
    ("The {source} source system delivers {format} files via {method} at {schedule}", "technical", "medium", "high"),
    ("Column {column} in {table} has {null_pct}% null rate which is {status}", "numerical", "short", "medium"),
    ("The {model} dbt model materializes as {strategy} and depends on {upstream}", "technical", "medium", "high"),
    ("Partition {partition} of {topic} Kafka topic has {lag} messages of consumer lag", "numerical", "short", "medium"),
    ("The {dashboard} Tableau dashboard is viewed by {viewers} users per week", "numerical", "short", "medium"),
    ("Data quality rule {rule} checks {assertion} on {table} with {severity} severity", "technical", "medium", "high"),
    ("The {pipeline} sub-pipeline takes {duration} minutes and costs ${cost} per run", "numerical", "medium", "high"),
    ("Snowflake query {query_id} scans {bytes}GB and runs in {seconds} seconds", "numerical", "short", "low"),
]

# Value pools for template interpolation
_FINTECH_VALS = {
    "endpoint": ["/accounts", "/transfers", "/payments", "/auth/login", "/webhooks", "/cards", "/statements", "/kyc/verify", "/disputes", "/rewards", "/notifications", "/settings", "/beneficiaries", "/fx/rates", "/compliance/check"],
    "volume": ["12,000", "45,000", "3,200", "890", "156,000", "23,500", "67,000", "9,400", "1,100", "78,000"],
    "service": ["auth", "ledger", "notifications", "compliance", "reporting", "fraud", "billing", "kyc", "fx", "rewards"],
    "protocol": ["gRPC", "REST", "AMQP", "GraphQL", "WebSocket"],
    "target": ["the gateway", "the ledger", "the notification service", "Stripe", "the fraud engine"],
    "table": ["transactions", "accounts", "users", "audit_log", "balances", "transfers", "cards", "beneficiaries", "fx_orders", "rewards_ledger"],
    "count": ["2.4", "18", "0.8", "45", "120", "6.7", "31", "0.3", "89", "14"],
    "col": ["created_date", "account_id", "currency", "status", "region"],
    "feature": ["instant transfers", "virtual cards", "bill pay", "round-up savings", "credit score", "budget tracking", "joint accounts", "recurring payments", "check deposit", "crypto buying"],
    "sprint": list(range(1, 50)),
    "module": ["payments", "accounts", "onboarding", "settings", "compliance", "notifications", "analytics", "support"],
    "code": ["ERR_INSUFFICIENT_FUNDS", "ERR_RATE_LIMITED", "ERR_KYC_PENDING", "ERR_DUPLICATE_TXN", "ERR_ACCOUNT_FROZEN", "ERR_INVALID_ROUTING", "ERR_STRIPE_DECLINED", "ERR_SESSION_EXPIRED", "ERR_SANCTIONS_HIT", "ERR_CARD_EXPIRED"],
    "meaning": ["insufficient balance", "too many requests", "identity verification incomplete", "duplicate transaction detected", "account is frozen", "invalid routing number", "card declined by issuer", "session has expired", "sanctions screening failed", "card expiration date passed"],
    "action": ["user notification", "automatic retry", "support ticket creation", "account review", "transaction reversal"],
    "env": ["staging", "production", "development", "performance-test"],
    "instance": ["t3.medium", "m5.large", "c5.xlarge", "r5.2xlarge"],
    "mem": ["4", "8", "16", "32"],
    "client": ["mobile-ios", "mobile-android", "web-dashboard", "partner-api", "internal-ops"],
    "limit": ["60", "120", "500", "1000", "30"],
    "window": ["minute", "second", "hour"],
    "report": ["daily settlement", "monthly revenue", "compliance audit", "fraud summary", "customer growth"],
    "sources": ["3", "7", "12", "5", "9"],
    "task": ["reconcile_ach", "send_statement", "compute_interest", "check_sanctions", "generate_1099"],
    "timeout": ["30", "60", "120", "300", "600"],
    "retries": ["3", "5", "1", "2"],
    "metric": ["error_rate", "p99_latency", "transaction_throughput", "memory_usage", "queue_depth"],
    "threshold": ["1%", "500ms", "10,000 tps", "85%", "1,000 messages"],
}
_MOBILE_VALS = {
    "screen": ["HomeScreen", "TransferScreen", "SettingsScreen", "ProfileScreen", "TransactionDetailScreen", "LoginScreen", "OnboardingScreen", "CardScreen", "BudgetScreen", "SupportScreen", "NotificationsScreen", "StatementsScreen", "ContactsScreen", "QRScanScreen", "DepositCheckScreen"],
    "time": ["120", "85", "200", "45", "310", "65", "150", "95", "180", "55"],
    "device": ["low-end Android", "iPhone 14", "Pixel 7", "Galaxy S23", "iPad", "iPhone SE"],
    "component": ["BalanceCard", "TransactionRow", "AmountInput", "AccountPicker", "CurrencySelector", "DateRangePicker", "PinPad", "BiometricPrompt", "NotificationBell", "SpendingChart"],
    "hook": ["useQuery", "useMemo", "useCallback", "useEffect", "useReducer", "useRef", "useContext"],
    "purpose": ["data fetching", "memoization", "event handling", "side effects", "complex state", "DOM reference", "theme access"],
    "from_screen": ["Home", "Login", "Transfer", "Settings", "Notifications"],
    "to_screen": ["TransactionDetail", "Receipt", "Confirmation", "EditProfile", "Support"],
    "params": ["accountId, amount", "transactionId", "userId, token", "cardId, status", "notificationId"],
    "gesture": ["swipe-to-dismiss", "long-press", "pinch-to-zoom", "double-tap", "pan"],
    "action": ["delete transaction", "show context menu", "zoom chart", "quick-pay", "drag to reorder"],
    "font": ["Inter", "SF Pro", "Roboto", "Montserrat"],
    "size": ["12", "14", "16", "18", "24", "32"],
    "element": ["body text", "headings", "captions", "button labels", "input fields"],
    "animation": ["transfer-success", "card-flip", "balance-update", "confetti", "shimmer-load"],
    "fps": ["30", "60"],
    "duration": ["300", "500", "800", "1200", "200"],
    "strategy": ["LRU", "FIFO", "size-based eviction"],
    "maxSize": ["50", "100", "200"],
    "feature": ["instant-transfer", "virtual-card", "dark-mode", "biometric-login", "spending-insights", "check-deposit", "crypto-buy", "round-up", "joint-account", "bill-pay"],
    "service": ["LaunchDarkly", "Firebase Remote Config", "Statsig"],
    "default": ["enabled", "disabled"],
    "count": ["12", "24", "6", "3"],
    "fallback": ["en-US", "en"],
    "sensor": ["accelerometer", "gyroscope", "GPS"],
    "interval": ["100", "250", "500", "1000"],
}
_PIPELINE_VALS = {
    "table": ["raw_transactions", "dim_customers", "fact_daily_balances", "stg_bank_feeds", "agg_monthly_revenue", "dim_merchants", "fact_transfers", "raw_card_events", "dim_currencies", "fact_fraud_scores", "stg_plaid_accounts", "agg_cohort_retention", "dim_products", "fact_interest_accruals", "raw_webhook_events"],
    "zone": ["raw", "staging", "curated", "consumption", "archive"],
    "rows": ["0.5", "3.2", "18", "120", "45", "0.1", "7.8", "250", "1.4", "62"],
    "job": ["txn_enrichment", "daily_aggregation", "fraud_scoring", "customer_360", "settlement_reconciliation", "interest_calculation", "regulatory_reporting", "churn_prediction", "feature_engineering", "data_quality_scan"],
    "volume": ["12", "45", "3", "120", "8", "25", "67", "0.5", "200", "15"],
    "time": ["8", "22", "45", "3", "12", "90", "5", "35", "60", "18"],
    "source": ["Plaid", "Stripe", "Salesforce", "Intercom", "internal PostgreSQL", "partner SFTP", "Marqeta", "Galileo", "Unit", "Synapse"],
    "format": ["JSON", "CSV", "Avro", "Parquet", "XML"],
    "method": ["S3 drop", "Kafka stream", "REST API pull", "SFTP", "CDC via Debezium"],
    "schedule": ["hourly", "every 15 minutes", "daily at 01:00 UTC", "real-time", "weekly"],
    "column": ["customer_id", "amount", "merchant_name", "category", "ip_address", "device_fingerprint", "routing_number", "ssn_encrypted", "email", "phone_number"],
    "null_pct": ["0.1", "2.3", "15", "0.0", "8.7", "42", "0.5", "1.2", "0.0", "5.6"],
    "status": ["acceptable", "above threshold", "critical", "expected", "under investigation"],
    "model": ["fct_daily_revenue", "dim_customer_profile", "int_transaction_enriched", "stg_bank_feed_parsed", "agg_weekly_cohort"],
    "strategy": ["incremental", "table", "ephemeral", "view"],
    "upstream": ["stg_transactions, dim_customers", "raw_bank_feeds", "int_enriched, dim_merchants", "stg_plaid_accounts", "fct_daily_revenue"],
    "partition": ["0", "3", "7", "11", "5"],
    "topic": ["raw-transactions", "fraud-alerts", "balance-updates", "account-events", "cdc-postgres"],
    "lag": ["0", "1,200", "45,000", "120", "8,900"],
    "dashboard": ["Executive Revenue", "Fraud Ops", "Customer Growth", "Pipeline Health", "Data Quality"],
    "viewers": ["12", "45", "120", "8", "30"],
    "rule": ["row_count_check", "null_ratio_assert", "freshness_check", "schema_match", "referential_integrity", "value_range_check", "uniqueness_assert", "format_validation"],
    "assertion": ["row count > 0", "null ratio < 5%", "max_timestamp within 6h", "columns match schema v3", "all FKs resolve", "amount between 0 and 1M", "no duplicate IDs", "dates in ISO 8601"],
    "severity": ["critical", "warning", "info"],
    "pipeline": ["bank-feed-ingestion", "fraud-scoring", "daily-settlement", "customer-360", "regulatory-extract"],
    "duration": ["4", "12", "25", "8", "45"],
    "cost": ["2.50", "8.00", "15.00", "3.20", "22.00"],
    "query_id": ["QRY-001", "QRY-042", "QRY-189", "QRY-007", "QRY-256"],
    "bytes": ["0.5", "12", "45", "120", "3"],
    "seconds": ["2", "15", "45", "120", "8"],
}


def _fill_template(template: str, vals: dict, rng: random.Random) -> str:
    """Fill a template string with random values from the value pools."""
    import re as _re
    def replacer(m):
        key = m.group(1)
        pool = vals.get(key, [str(rng.randint(1, 100))])
        return str(rng.choice(pool))
    return _re.sub(r'\{(\w+)\}', replacer, template)


def generate_facts(n: int, scope_name: str) -> list[tuple[str, str, str, str]]:
    """Generate n semantically distinct facts for a given project scope."""
    if scope_name == "fintech":
        base, templates, vals = FINTECH_FACTS, FINTECH_TEMPLATES, _FINTECH_VALS
    elif scope_name == "mobile":
        base, templates, vals = MOBILE_FACTS, MOBILE_TEMPLATES, _MOBILE_VALS
    else:
        base, templates, vals = PIPELINE_FACTS, PIPELINE_TEMPLATES, _PIPELINE_VALS

    rng = random.Random(42 + hash(scope_name))
    facts = []
    seen_texts = set()

    # First: all base facts
    for fact in base:
        if len(facts) >= n:
            break
        facts.append(fact)
        seen_texts.add(fact[0])

    # Then: generate from templates with random fills
    attempts = 0
    while len(facts) < n and attempts < n * 10:
        tmpl_tuple = templates[attempts % len(templates)]
        text = _fill_template(tmpl_tuple[0], vals, rng)
        attempts += 1
        if text not in seen_texts:
            seen_texts.add(text)
            facts.append((text, tmpl_tuple[1], tmpl_tuple[2], tmpl_tuple[3]))

    return facts[:n]


# ── Benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(target_count: int) -> dict:
    """
    Run the full benchmark for a given total item count.
    Items are distributed: 40% fintech, 30% mobile, 30% pipeline.
    Plus cross-project facts injected into all three scopes.
    """
    db_path = Path(tempfile.mktemp(suffix=f"_bench_{target_count}.duckdb"))
    _cfg.DB_PATH = db_path
    conn = db.get_connection(db_path=str(db_path))

    results = {
        "target_count": target_count,
        "actual_count": 0,
        "embed_time_s": 0,
        "insert_time_s": 0,
    }

    # Distribution per scope
    n_fintech = int(target_count * 0.4)
    n_mobile = int(target_count * 0.3)
    n_pipeline = target_count - n_fintech - n_mobile

    scope_data = {
        "fintech": (SCOPES["fintech"], generate_facts(n_fintech, "fintech"),
                    FINTECH_ENTITIES, FINTECH_DECISIONS, FINTECH_RELATIONSHIPS),
        "mobile": (SCOPES["mobile"], generate_facts(n_mobile, "mobile"),
                   MOBILE_ENTITIES, MOBILE_DECISIONS, MOBILE_RELATIONSHIPS),
        "pipeline": (SCOPES["pipeline"], generate_facts(n_pipeline, "pipeline"),
                     PIPELINE_ENTITIES, PIPELINE_DECISIONS, PIPELINE_RELATIONSHIPS),
    }

    session_id = f"bench-{target_count}"
    db.upsert_session(conn, session_id, "benchmark", "/tmp", "/tmp/bench.jsonl", 0, "Benchmark run")

    total_items = 0
    t_embed_total = 0
    t_insert_total = 0

    # ── Insert scoped data ────────────────────────────────────────────────
    for scope_name, (scope_path, facts, entities, decisions, rels) in scope_data.items():
        print(f"  [{scope_name}] Inserting {len(facts)} facts + {len(entities)} entities + {len(decisions)} decisions + {len(rels)} relationships...")

        # Entities
        for name in entities:
            t0 = time.time()
            emb = embeddings.embed(name)
            t_embed_total += time.time() - t0
            t0 = time.time()
            db.upsert_entity(conn, name, embedding=emb, scope=scope_path)
            t_insert_total += time.time() - t0
            total_items += 1

        # Facts
        for text, cat, tc, conf in facts:
            t0 = time.time()
            emb = embeddings.embed(text)
            t_embed_total += time.time() - t0
            t0 = time.time()
            db.upsert_fact(conn, text, cat, tc, conf, emb, session_id, compute_decay_score, scope=scope_path)
            t_insert_total += time.time() - t0
            total_items += 1

        # Decisions
        for text, tc in decisions:
            t0 = time.time()
            emb = embeddings.embed(text)
            t_embed_total += time.time() - t0
            t0 = time.time()
            db.upsert_decision(conn, text, tc, emb, session_id, compute_decay_score, scope=scope_path)
            t_insert_total += time.time() - t0
            total_items += 1

        # Relationships
        for from_e, to_e, rel_type, desc in rels:
            t0 = time.time()
            db.upsert_relationship(conn, from_e, to_e, rel_type, desc, session_id, scope=scope_path)
            t_insert_total += time.time() - t0
            total_items += 1

    # ── Insert cross-project facts (into all 3 scopes for auto-promotion) ─
    print(f"  [cross] Inserting {len(CROSS_PROJECT_FACTS)} cross-project facts into all 3 scopes...")
    for text, cat, tc, conf in CROSS_PROJECT_FACTS:
        emb = embeddings.embed(text)
        for scope_path in SCOPES.values():
            db.upsert_fact(conn, text, cat, tc, conf, emb, session_id, compute_decay_score, scope=scope_path)
        total_items += 1  # count once (deduped)

    for name in CROSS_ENTITIES:
        emb = embeddings.embed(name)
        for scope_path in SCOPES.values():
            db.upsert_entity(conn, name, embedding=emb, scope=scope_path)
        total_items += 1

    results["actual_count"] = total_items
    results["embed_time_s"] = round(t_embed_total, 3)
    results["insert_time_s"] = round(t_insert_total, 3)

    # ── Build FTS indexes for BM25 retrieval ──────────────────────────────
    db.rebuild_fts_indexes(conn)

    # ── Measure storage size ──────────────────────────────────────────────
    conn.close()
    results["db_size_kb"] = round(db_path.stat().st_size / 1024, 1)

    # ── Query latency benchmarks ──────────────────────────────────────────
    conn = db.get_connection(db_path=str(db_path), read_only=True)

    # Session recall — scoped to fintech
    times = []
    for _ in range(5):
        t0 = time.time()
        ctx = recall.session_recall(conn, scope=SCOPES["fintech"])
        times.append(time.time() - t0)
    results["session_recall_ms"] = round(min(times) * 1000, 2)
    results["session_recall_avg_ms"] = round(sum(times) / len(times) * 1000, 2)

    # Format and check token budget
    formatted = recall.format_session_context(ctx)
    session_tokens = len(formatted) // CHARS_PER_TOKEN
    results["session_context_tokens"] = session_tokens
    results["session_budget_ok"] = session_tokens <= SESSION_TOKEN_BUDGET

    # Prompt recall — semantic search scoped to fintech
    query_text = "How does the payment processing work with Stripe?"
    query_emb = embeddings.embed(query_text)
    times = []
    for _ in range(5):
        t0 = time.time()
        pctx = recall.prompt_recall(conn, query_emb, query_text, scope=SCOPES["fintech"], db_path=str(db_path))
        times.append(time.time() - t0)
    results["prompt_recall_ms"] = round(min(times) * 1000, 2)
    results["prompt_recall_avg_ms"] = round(sum(times) / len(times) * 1000, 2)

    # Format and check prompt token budget
    prompt_formatted = recall.format_prompt_context(pctx)
    prompt_tokens = len(prompt_formatted) // CHARS_PER_TOKEN
    results["prompt_context_tokens"] = prompt_tokens
    results["prompt_budget_ok"] = prompt_tokens <= PROMPT_TOKEN_BUDGET

    # Raw search_facts latency
    times = []
    for _ in range(5):
        t0 = time.time()
        db.search_facts(conn, query_emb, limit=8, scope=SCOPES["fintech"])
        times.append(time.time() - t0)
    results["search_facts_ms"] = round(min(times) * 1000, 2)

    # ── 4-way parallel retrieval benchmark ─────────────────────────────
    conn.close()
    times = []
    for _ in range(5):
        t0 = time.time()
        ret = parallel_retrieve(
            db_path=str(db_path),
            query_text=query_text,
            query_embedding=query_emb,
            scope=SCOPES["fintech"],
            limit=10,
            timeout_ms=2000,
        )
        times.append(time.time() - t0)
    results["parallel_retrieve_ms"] = round(min(times) * 1000, 2)
    results["parallel_retrieve_avg_ms"] = round(sum(times) / len(times) * 1000, 2)
    results["parallel_retrieve_items"] = len(ret.items)
    results["parallel_retrieve_strategies"] = ret.strategy_counts
    results["parallel_retrieve_exceeded"] = ret.exceeded_budget

    # BM25-only latency
    times = []
    for _ in range(5):
        t0 = time.time()
        parallel_retrieve(
            db_path=str(db_path), query_text="Stripe payment processing",
            query_embedding=None, scope=SCOPES["fintech"], limit=10,
            timeout_ms=2000, strategies=["bm25"],
        )
        times.append(time.time() - t0)
    results["bm25_only_ms"] = round(min(times) * 1000, 2)

    # ── Consolidation benchmark ────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        conn_rw2 = db.get_connection(db_path=str(db_path))
        uncons = db.get_unconsolidated_facts(conn_rw2, limit=100, scope=SCOPES["fintech"])
        results["unconsolidated_facts"] = len(uncons)

        t0 = time.time()
        consol_stats = run_consolidation(conn_rw2, api_key, SCOPES["fintech"], quiet=True)
        results["consolidation_time_s"] = round(time.time() - t0, 2)
        results["consolidation_stats"] = consol_stats

        # Semantic forgetting
        t0 = time.time()
        forget_stats = run_semantic_forgetting(conn_rw2, SCOPES["fintech"])
        results["semantic_forgetting_time_s"] = round(time.time() - t0, 3)
        results["semantic_forgetting_stats"] = forget_stats

        # Check observations were created
        try:
            obs_stats = db.get_stats(conn_rw2).get("observations", {})
            results["observations_created"] = obs_stats.get("total", 0)
        except Exception:
            results["observations_created"] = 0

        conn_rw2.close()
    else:
        results["unconsolidated_facts"] = "SKIPPED (no ANTHROPIC_API_KEY)"
        results["consolidation_stats"] = "SKIPPED"
        results["observations_created"] = "SKIPPED"

    # ── Observations in recall ─────────────────────────────────────────
    conn = db.get_connection(db_path=str(db_path), read_only=True)
    ctx_with_obs = recall.session_recall(conn, scope=SCOPES["fintech"])
    results["observations_in_recall"] = len(ctx_with_obs.get("observations", []))

    # ── Correctness checks ────────────────────────────────────────────────
    correctness = {}

    # 1. Scope isolation: fintech query should NOT return mobile/pipeline-only facts
    fintech_facts = db.get_facts_by_temporal(conn, "long", 100, scope=SCOPES["fintech"])
    fintech_texts = [f["text"] for f in fintech_facts]
    mobile_only = [t for t in fintech_texts if "React Native" in t or "Expo" in t or "Zustand" in t]
    pipeline_only = [t for t in fintech_texts if "Airflow" in t or "Spark" in t or "Snowflake" in t]
    correctness["scope_isolation"] = len(mobile_only) == 0 and len(pipeline_only) == 0

    # 2. Global visibility: cross-project facts should appear in all scopes
    for scope_name, scope_path in SCOPES.items():
        scoped_facts = db.get_facts_by_temporal(conn, "long", 200, scope=scope_path)
        scoped_texts = [f["text"] for f in scoped_facts]
        # At least some cross-project facts should be visible (they were auto-promoted)
        cross_visible = any("GitHub" in t for t in scoped_texts)
        correctness[f"global_visible_{scope_name}"] = cross_visible

    # 3. Auto-promotion: cross-project facts seen in 3 scopes should be __global__
    cross_fact_text = CROSS_PROJECT_FACTS[0][0]
    cross_emb = embeddings.embed(cross_fact_text)
    cross_hits = db.search_facts(conn, cross_emb, limit=1, threshold=0.85)
    if cross_hits:
        cross_scope = cross_hits[0].get("scope", "?")
        correctness["auto_promotion"] = cross_scope == GLOBAL_SCOPE
    else:
        correctness["auto_promotion"] = False

    # 4. Dedup: inserting the same fact twice should not create duplicates
    pre_count = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active=TRUE").fetchone()[0]
    conn.close()
    conn_rw = db.get_connection(db_path=str(db_path))
    dup_emb = embeddings.embed("FastAPI serves the REST endpoints")
    db.upsert_fact(conn_rw, "FastAPI serves the REST endpoints",
                   "technical", "long", "high", dup_emb, session_id, compute_decay_score,
                   scope=SCOPES["fintech"])
    post_count = conn_rw.execute("SELECT COUNT(*) FROM facts WHERE is_active=TRUE").fetchone()[0]
    correctness["dedup_works"] = post_count == pre_count
    conn_rw.close()
    conn = db.get_connection(db_path=str(db_path), read_only=True)

    # 5. Entity scope isolation
    fintech_entities = db.get_top_entities(conn, 100, scope=SCOPES["fintech"])
    mobile_leak = any(e in fintech_entities for e in ["Detox", "Hermes", "CodePush"])
    correctness["entity_scope_isolation"] = not mobile_leak

    # 6. Relationship scope isolation
    fintech_rels = db.get_relationships_for_entities(conn, ["Airflow"], scope=SCOPES["fintech"])
    correctness["rel_scope_isolation"] = len(fintech_rels) == 0

    results["correctness"] = correctness

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = db.get_stats(conn)
    results["db_stats"] = {
        "facts_active": stats["facts"]["total"],
        "facts_long": stats["facts"]["long"],
        "facts_medium": stats["facts"]["medium"],
        "facts_short": stats["facts"]["short"],
        "entities": stats["entities"]["total"],
        "relationships": stats["relationships"]["total"],
        "decisions": stats["decisions"]["total"],
    }

    conn.close()

    # Cleanup
    try:
        db_path.unlink()
    except Exception:
        pass

    return results


def print_results(results: dict) -> None:
    n = results["target_count"]
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {n:,} target items ({results['actual_count']:,} actual)")
    print(f"{'=' * 70}")

    print(f"\n  Storage:")
    print(f"    DB size:        {results['db_size_kb']:,.1f} KB")
    print(f"    Embed time:     {results['embed_time_s']:.1f}s")
    print(f"    Insert time:    {results['insert_time_s']:.3f}s")

    print(f"\n  Query Latency:")
    print(f"    session_recall:     {results['session_recall_ms']:.2f}ms (best), {results['session_recall_avg_ms']:.2f}ms (avg)")
    print(f"    prompt_recall:      {results['prompt_recall_ms']:.2f}ms (best), {results['prompt_recall_avg_ms']:.2f}ms (avg)")
    print(f"    search_facts:       {results['search_facts_ms']:.2f}ms (best)")
    print(f"    parallel_retrieve:  {results['parallel_retrieve_ms']:.2f}ms (best), {results['parallel_retrieve_avg_ms']:.2f}ms (avg)")
    print(f"    bm25_only:          {results['bm25_only_ms']:.2f}ms (best)")
    print(f"    strategies:         {results['parallel_retrieve_strategies']}")
    print(f"    items returned:     {results['parallel_retrieve_items']}")
    exceeded = "YES" if results['parallel_retrieve_exceeded'] else "no"
    print(f"    budget exceeded:    {exceeded}")

    print(f"\n  Token Budget:")
    session_ok = "PASS" if results["session_budget_ok"] else "FAIL"
    prompt_ok = "PASS" if results["prompt_budget_ok"] else "FAIL"
    print(f"    Session context: {results['session_context_tokens']} tokens (budget: {SESSION_TOKEN_BUDGET}) [{session_ok}]")
    print(f"    Prompt context:  {results['prompt_context_tokens']} tokens (budget: {PROMPT_TOKEN_BUDGET}) [{prompt_ok}]")

    print(f"\n  Correctness:")
    for check, passed in results["correctness"].items():
        status = "PASS" if passed else "FAIL"
        print(f"    {check:30s} [{status}]")

    print(f"\n  Consolidation:")
    print(f"    unconsolidated:     {results.get('unconsolidated_facts', 'N/A')}")
    print(f"    consolidation:      {results.get('consolidation_stats', 'N/A')}")
    if isinstance(results.get('consolidation_time_s'), (int, float)):
        print(f"    consolidation time: {results['consolidation_time_s']:.2f}s")
    print(f"    observations:       {results.get('observations_created', 'N/A')}")
    print(f"    in recall:          {results.get('observations_in_recall', 0)}")
    if isinstance(results.get('semantic_forgetting_stats'), dict):
        sf = results['semantic_forgetting_stats']
        print(f"    semantic forget:    {sf.get('superseded', 0)} superseded / {sf.get('pairs_checked', 0)} pairs checked")

    print(f"\n  DB Stats:")
    for k, v in results["db_stats"].items():
        print(f"    {k:20s} {v}")

    print(f"{'=' * 70}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Claude Code Memory System — Benchmark")
    print(f"Ollama model: {_cfg.OLLAMA_MODEL}")
    print(f"Embedding dim: {_cfg.EMBEDDING_DIM}")
    print()

    all_results = []
    for n in [100, 500, 1000, 10000]:
        print(f"\n--- Running benchmark: {n:,} items ---")
        results = run_benchmark(n)
        print_results(results)
        all_results.append(results)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Items':>8s}  {'DB Size':>10s}  {'Embed':>8s}  {'SessRecall':>11s}  {'PromptRecall':>13s}  {'Search':>8s}  {'SessTok':>8s}  {'Correct':>8s}")
    print(f"  {'':->8s}  {'':->10s}  {'':->8s}  {'':->11s}  {'':->13s}  {'':->8s}  {'':->8s}  {'':->8s}")
    for r in all_results:
        n_correct = sum(1 for v in r["correctness"].values() if v)
        n_total = len(r["correctness"])
        print(f"  {r['target_count']:>8,d}"
              f"  {r['db_size_kb']:>8.1f}KB"
              f"  {r['embed_time_s']:>6.1f}s"
              f"  {r['session_recall_avg_ms']:>9.2f}ms"
              f"  {r['prompt_recall_avg_ms']:>11.2f}ms"
              f"  {r['search_facts_ms']:>6.2f}ms"
              f"  {r['session_context_tokens']:>8d}"
              f"  {n_correct}/{n_total}")
    print(f"{'=' * 90}")
