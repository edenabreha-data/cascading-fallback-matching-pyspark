# Cascading Fallback Matching (PySpark)

A PySpark pattern for finding historical records that match a request, starting with the most specific
filter and automatically falling back to broader ones when there isn't enough data.

> All data is **synthetic**, generated in the notebook. No real or proprietary data is used.

## The problem

When you estimate or predict something from history, the most relevant records (for example, the same
provider) are often too sparse to trust. This pipeline widens the filter step by step until the sample is big enough:

| Level | Filter | Used when |
|-------|--------|-----------|
| 1 | Provider | provider has at least `min_matches` matching visits |
| 2 | Specialty | otherwise, specialty has at least `min_matches` |
| 3 | Facility (all) | otherwise, use every matching visit |

A visit matches only if it contains **all** requested service codes.

## What it demonstrates

- Multi-level fallback logic with a configurable threshold
- "Contains all of these codes" matching using `groupBy` + `countDistinct`
- Handling duplicate rows from one-to-many joins (providers with multiple specialties)
- Run logging: every call records its inputs, counts per level, and the level used
- Unit tests covering each fallback level plus the edge cases (duplicates, missing codes)

## Run it

**Databricks Free Edition:** import `cascading_matching_demo.py` as a notebook and run all cells.

**Locally:**
```bash
pip install pyspark
python cascading_matching_demo.py
```

Expected output: three demo requests that land on provider, specialty, and facility level respectively,
a run-log table, and five `PASS` lines.

## Example run log

| service_codes | provider_matches | specialty_matches | total_matches | level_used | rows_returned |
|---|---|---|---|---|---|
| S001 | 46 | 51 | 411 | provider | 46 |
| S001,S002 | 0 | 5 | 27 | specialty | 5 |
| S005,S006,S007 | 0 | 1 | 3 | facility | 3 |

## Possible extensions

- Persist the run log to a Delta table
- Add a code-crosswalk step so sparse new codes can borrow history from older equivalents
- Add date-window filters and recency weighting
- Benchmark on a larger synthetic dataset and compare caching and partitioning strategies
