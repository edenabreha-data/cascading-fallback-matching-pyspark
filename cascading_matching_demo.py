# Databricks notebook source
# MAGIC %md
# MAGIC # Cascading Fallback Matching (PySpark)
# MAGIC Given a request (a set of service codes plus an optional provider and specialty), find historical
# MAGIC visits that match. Start with the most specific filter and fall back to broader ones when there
# MAGIC are not enough matches:
# MAGIC
# MAGIC 1. **Provider level**: matches for that provider, if there are at least `min_matches`
# MAGIC 2. **Specialty level**: matches for that specialty, if there are at least `min_matches`
# MAGIC 3. **Facility level**: all matches, no provider or specialty filter
# MAGIC
# MAGIC All data in this notebook is **synthetic**. Runs on Databricks Free Edition or locally with `pip install pyspark`.

# COMMAND ----------

import random
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("cascading-matching").getOrCreate()
try:
    spark.conf.set("spark.sql.shuffle.partitions", "8")  # small data, so avoid 200 tiny tasks
except Exception:
    pass  # some managed/serverless environments do not allow changing this setting

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic data
# MAGIC Three tables: `visits`, `visit_services` (one row per code on a visit), and `provider_specialty`
# MAGIC (a provider can have more than one specialty, which creates duplicate rows after a join and must be handled).

# COMMAND ----------

def build_synthetic_data(n_visits: int = 5000, seed: int = 42):
    rnd = random.Random(seed)
    n_providers, n_specialties, n_codes = 40, 6, 30

    # Each provider has one primary specialty; ~15% also have a second one
    provider_specialty = []
    for p in range(1, n_providers + 1):
        s = rnd.randint(1, n_specialties)
        provider_specialty.append((p, s))
        if rnd.random() < 0.15:
            provider_specialty.append((p, (s % n_specialties) + 1))

    start = date(2024, 1, 1)
    visits, visit_services = [], []
    for v in range(1, n_visits + 1):
        # Skewed so a few providers are busy and many are sparse
        provider = min(int(rnd.expovariate(0.12)) + 1, n_providers)
        visits.append((
            v,
            provider,
            rnd.randint(1, 3),                              # facility_id
            rnd.choice(["outpatient", "inpatient"]),
            round(rnd.uniform(200, 20000), 2),              # total_charge
            start + timedelta(days=rnd.randint(0, 700)),
        ))
        for c in rnd.sample(range(1, n_codes + 1), rnd.randint(1, 4)):
            visit_services.append((v, f"S{c:03d}"))

    visits_df = spark.createDataFrame(
        visits, "visit_id int, provider_id int, facility_id int, visit_type string, total_charge double, service_date date")
    services_df = spark.createDataFrame(visit_services, "visit_id int, service_code string")
    prov_spec_df = spark.createDataFrame(provider_specialty, "provider_id int, specialty_id int")
    return visits_df, services_df, prov_spec_df

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. The matching logic

# COMMAND ----------

@dataclass
class MatchRequest:
    service_codes: list
    facility_id: Optional[int] = None
    provider_id: Optional[int] = None
    specialty_id: Optional[int] = None
    visit_type: Optional[str] = None


LEVEL_NAMES = {1: "provider", 2: "specialty", 3: "facility"}


def find_candidate_visits(req: MatchRequest, visits, services, prov_spec) -> DataFrame:
    """Visits that contain ALL requested codes (and match facility / visit type if given)."""
    codes = list(set(req.service_codes))
    matched_ids = (
        services.filter(F.col("service_code").isin(codes))
        .groupBy("visit_id")
        .agg(F.countDistinct("service_code").alias("n"))
        .filter(F.col("n") == len(codes))
        .select("visit_id")
    )
    df = visits.join(matched_ids, "visit_id")
    if req.facility_id is not None:
        df = df.filter(F.col("facility_id") == req.facility_id)
    if req.visit_type is not None:
        df = df.filter(F.col("visit_type") == req.visit_type)

    # A provider with two specialties produces two rows per visit after this join.
    # Counts use distinct visit_id and the final result collapses back to one row per visit.
    return df.join(prov_spec, "provider_id", "left")


def cascade_match(req: MatchRequest, visits, services, prov_spec, min_matches: int = 5):
    """Return (matches_df, level, counts) using provider -> specialty -> facility fallback."""
    # No .cache() here so it also works on serverless compute (Databricks Free Edition).
    candidates = find_candidate_visits(req, visits, services, prov_spec)

    provider_count = (
        candidates.filter(F.col("provider_id") == req.provider_id).select("visit_id").distinct().count()
        if req.provider_id is not None else 0
    )
    specialty_count = (
        candidates.filter(F.col("specialty_id") == req.specialty_id).select("visit_id").distinct().count()
        if req.specialty_id is not None else 0
    )
    total_count = candidates.select("visit_id").distinct().count()

    if provider_count >= min_matches:
        level, df = 1, candidates.filter(F.col("provider_id") == req.provider_id)
    elif specialty_count >= min_matches:
        level, df = 2, candidates.filter(F.col("specialty_id") == req.specialty_id)
    else:
        level, df = 3, candidates

    result = (
        df.groupBy("visit_id", "provider_id", "facility_id", "visit_type", "total_charge", "service_date")
        .agg(F.max("specialty_id").alias("specialty_id"))
        .withColumn("cascade_level", F.lit(level))
    )
    counts = {"provider": provider_count, "specialty": specialty_count, "all": total_count}
    return result, level, counts

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Run logging
# MAGIC Each run records the inputs, counts at each level, and which level was used, to make results explainable and debuggable.

# COMMAND ----------

RUN_LOG = []

def log_run(req: MatchRequest, level: int, counts: dict, min_matches: int, rows_returned: int):
    RUN_LOG.append({
        "service_codes": ",".join(sorted(req.service_codes)),
        "facility_id": req.facility_id,
        "provider_id": req.provider_id,
        "specialty_id": req.specialty_id,
        "min_matches": min_matches,
        "provider_matches": counts["provider"],
        "specialty_matches": counts["specialty"],
        "total_matches": counts["all"],
        "level_used": LEVEL_NAMES[level],
        "rows_returned": rows_returned,
    })

def run(req: MatchRequest, visits, services, prov_spec, min_matches: int = 5):
    result, level, counts = cascade_match(req, visits, services, prov_spec, min_matches)
    n = result.count()
    log_run(req, level, counts, min_matches, n)
    return result, level, counts

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Demo on the synthetic data

# COMMAND ----------

visits, services, prov_spec = build_synthetic_data()

requests = [
    MatchRequest(["S001"], provider_id=1, specialty_id=2),                        # busy provider
    MatchRequest(["S001", "S002"], provider_id=38, specialty_id=3),               # sparse provider
    MatchRequest(["S005", "S006", "S007"], provider_id=39, specialty_id=4),       # very specific, few matches
]
for r in requests:
    res, level, counts = run(r, visits, services, prov_spec, min_matches=5)
    print(f"{r.service_codes}: level={LEVEL_NAMES[level]}, counts={counts}, rows={res.count()}")

LOG_SCHEMA = ("service_codes string, facility_id int, provider_id int, specialty_id int, min_matches int, "
              "provider_matches long, specialty_matches long, total_matches long, level_used string, rows_returned long")
spark.createDataFrame(RUN_LOG, LOG_SCHEMA).show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Tests
# MAGIC Small, hand-built data so each fallback level is exercised deterministically.

# COMMAND ----------

def _tiny():
    v = spark.createDataFrame(
        [(1, 10, 1, "outpatient", 100.0, date(2025, 1, 1)),
         (2, 10, 1, "outpatient", 110.0, date(2025, 1, 2)),
         (3, 10, 1, "outpatient", 120.0, date(2025, 1, 3)),
         (4, 20, 1, "outpatient", 130.0, date(2025, 1, 4)),
         (5, 20, 1, "outpatient", 140.0, date(2025, 1, 5)),
         (6, 30, 1, "outpatient", 150.0, date(2025, 1, 6))],
        "visit_id int, provider_id int, facility_id int, visit_type string, total_charge double, service_date date")
    s = spark.createDataFrame([(i, "A") for i in range(1, 7)], "visit_id int, service_code string")
    # provider 10 and 20 are both specialty 1; provider 20 ALSO has specialty 2 (duplicate-row case)
    ps = spark.createDataFrame([(10, 1), (20, 1), (20, 2), (30, 3)], "provider_id int, specialty_id int")
    return v, s, ps

def test_provider_level():
    v, s, ps = _tiny()
    res, level, _ = cascade_match(MatchRequest(["A"], provider_id=10, specialty_id=1), v, s, ps, min_matches=3)
    assert level == 1 and res.count() == 3

def test_specialty_fallback():
    v, s, ps = _tiny()
    # provider 30 has 1 match (<3), specialty 1 has 5 visits (>=3)
    res, level, _ = cascade_match(MatchRequest(["A"], provider_id=30, specialty_id=1), v, s, ps, min_matches=3)
    assert level == 2 and res.count() == 5

def test_facility_fallback():
    v, s, ps = _tiny()
    res, level, _ = cascade_match(MatchRequest(["A"], provider_id=30, specialty_id=3), v, s, ps, min_matches=3)
    assert level == 3 and res.count() == 6

def test_no_duplicates_for_multi_specialty_provider():
    v, s, ps = _tiny()
    res, _, _ = cascade_match(MatchRequest(["A"]), v, s, ps, min_matches=3)
    assert res.count() == res.select("visit_id").distinct().count() == 6

def test_requires_all_codes():
    v, s, ps = _tiny()
    res, _, _ = cascade_match(MatchRequest(["A", "B"]), v, s, ps, min_matches=1)
    assert res.count() == 0

for t in [test_provider_level, test_specialty_fallback, test_facility_fallback,
          test_no_duplicates_for_multi_specialty_provider, test_requires_all_codes]:
    t()
    print("PASS", t.__name__)
