# Aggregate Pushdown in Mature Query Engines

Background research for the Lance `feat-aggregate-pushdown` work. The motivating use case is `COUNT(DISTINCT col)` directly from a bitmap index, but this report maps the broader design space.

## 1. Executive Summary

- **Three distinct families of "aggregate pushdown"** are conflated in vendor docs. Keep them separate when designing Lance's APIs: (a) *metadata-only execution* — answer the aggregate from per-fragment statistics with zero data IO (Snowflake, Iceberg PR #6622, SQL Server segment metadata for `MIN/MAX/COUNT`); (b) *scan-local aggregation* — run the aggregate inside the scan operator over compressed/encoded data, eliminating a separate Aggregate node (SQL Server "Aggregate Pushdown" since 2016, ClickHouse `optimize_aggregation_in_order`); (c) *materialized/pre-aggregated structures* — separate physical artifact that answers many GROUP BYs (ClickHouse projections, Pinot star-tree, SQL Server indexed views, AggregatingMergeTree).
- **`MIN/MAX` is the universally-supported case.** Every engine has a `MinMaxAggPath`-equivalent that either reads endpoints from a sorted index (Postgres) or reads per-segment min/max statistics (everyone else). Lance has min/max page statistics already — turning these into an `Aggregate` rewrite is the lowest-hanging fruit and matches `preprocess_minmax_aggregates` in Postgres almost exactly.
- **`COUNT(*)` from metadata is universally supported but with caveats.** Without a predicate, every engine answers from row counts in fragment/manifest metadata. *With* a predicate, only fragments whose stats *prove* full inclusion or full exclusion can be skipped — partial fragments must still be scanned. DataFusion's "Fully Matching / Partially Matching / Not Matching" trichotomy (the limit-pruning blog post, March 2026) is the cleanest articulation.
- **`COUNT(DISTINCT)` from a bitmap index is unusual but legitimate.** Druid's `cardinality` aggregator returns approximate distinct counts directly from per-value bitmaps. *Exact* `COUNT(DISTINCT)` from a bitmap is also trivial — it is the dictionary size after applying the predicate's row mask. Lance's bitmap index already has per-value posting lists, so exact distinct count is the natural fit, not HLL.
- **Partial vs. full pushdown matters at the planner level.** Spark's `SupportsPushDownAggregates.supportCompletePushDown()` is the canonical API: per-fragment partial aggregates with a final reduction step in the engine. This is also how Postgres's partition-wise aggregate and postgres_fdw work (`combinefunc`/`serialfunc`/`deserialfunc`). Lance will likely need the same split because indexes are per-fragment.
- **NULL semantics differ between aggregates and become an issue.** `COUNT(*)` counts rows; `COUNT(col)` skips nulls; `MIN/MAX` skip nulls. Iceberg's PR #6622 distinguishes "stat is null because column is all-null" (legal answer) from "stat is missing" (abort pushdown) via a `hasValue` flag. Lance needs the same distinction.
- **Predicate compatibility is the gating constraint.** A pushed aggregate is only legal if the predicate is *also* fully evaluable from the same metadata — otherwise the count/min/max applies to an over-set of rows. This is the source of most correctness bugs in this area (cf. the Iceberg "Fix aggregate pushdown" thread).
- **GROUP BY pushdown is the hard mode.** SQL Server's "grouped aggregate pushdown" only fires when the grouping key bit-packs into ≤10 bits *and* a runtime "benefit measure" stays above a threshold. Pinot's star-tree solves it with a precomputed index. ClickHouse's projections do too. There is no cheap implementation — Lance should defer until non-grouped pushdown is solid.
- **MVCC/visibility is an issue only for transactional engines.** Postgres's index-only-scan has to consult the visibility map; SQL Server's pushdown only applies to "compressed rowgroups" not the delta store. Lance's append-only/versioned model sidesteps this — but the analogue is *deletion vectors / row-level deletes*. Iceberg PR #6622 explicitly disables aggregate pushdown when row-level deletes exist. Lance must do the same when deletion vectors apply.
- **The optimizer integration is consistently a dedicated planner pass, not a generic rule.** Postgres's `preprocess_minmax_aggregates` runs in `grouping_planner` just before `query_planner`. DataFusion's `AggregateStatistics` is a `PhysicalOptimizerRule`. Spark uses a V2 datasource interface (`SupportsPushDownAggregates`). The pattern is consistent: detect the shape, build an alternative path, let the cost model choose.

---

## 2. Taxonomy of Techniques

```
                    ┌─────────────────────────────────────────────┐
                    │            Aggregate Pushdown               │
                    └─────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│  Metadata-    │              │   Scan-local  │              │ Materialized/ │
│  only         │              │  aggregation  │              │ Pre-aggregated│
│  (no IO)      │              │  (closer to   │              │   artifacts   │
│               │              │   data)       │              │               │
└───────────────┘              └───────────────┘              └───────────────┘
   │                                  │                                │
   │ MIN/MAX from index endpoints     │ Aggregate inside scan         │ Indexed views (MSSQL)
   │   (Postgres MinMaxAggPath)       │   over compressed data        │ Projections (ClickHouse)
   │                                  │   (MSSQL agg pushdown)        │ AggregatingMergeTree
   │ MIN/MAX/COUNT from zone-maps     │                               │ Star-tree (Pinot)
   │   (Iceberg, Snowflake, MSSQL     │ SIMD-vectorized agg over      │ Materialized views (PG, Snowflake)
   │    segments, DuckDB zonemap)     │   bit-packed encoded data     │
   │                                  │                               │ Roll-up tables (Druid)
   │ COUNT(*) from row counts         │ Grouped agg pushdown          │
   │                                  │   (MSSQL, 2019+)              │
   │ COUNT DISTINCT from bitmap       │                               │
   │   dictionary (Druid)             │                               │
   │                                  │                               │
   │ HLL distinct from sketches       │                               │
   │   (Druid hyperUnique, BQ, Snow)  │                               │
   └──────────────────────────────────┴───────────────────────────────┘

                       Orthogonal axis: partial vs. complete
       ┌─────────────────────────────────────────────────────────────────┐
       │ Complete: source returns final answer (single fragment, or      │
       │           commutative aggregate over independent fragments      │
       │           where source does the reduction itself).              │
       │ Partial:  source returns per-fragment partial aggregate state;  │
       │           engine reduces with combinefunc.                      │
       │   - Spark: SupportsPushDownAggregates.supportCompletePushDown() │
       │   - Postgres: partial aggregates (combine/serial/deserialfunc)  │
       │   - postgres_fdw: per-foreign-server partial aggregation        │
       └─────────────────────────────────────────────────────────────────┘
```

---

## 3. Per-Engine Sections

### 3.1 PostgreSQL

**MIN/MAX via `MinMaxAggPath` — `src/backend/optimizer/plan/planagg.c`.** The function `preprocess_minmax_aggregates(PlannerInfo *root)` is called by `grouping_planner` just before `query_planner`. It checks: only aggregates in target list, single base relation, no `GROUP BY`/window/CTE, single-argument MIN/MAX (recognized via its sort operator from `pg_aggregate` — `fetch_agg_sort_op`), no `DISTINCT`/`ORDER BY`/`FILTER` on the aggregate, no mutable functions, no row-type args. For each matching aggregate, `build_minmax_path` builds an effective `SELECT col FROM t WHERE col IS NOT NULL ORDER BY col [DESC] LIMIT 1` subquery and registers a `MinMaxAggPath` against the `UPPERREL_GROUP_AGG` upper rel. Cost model decides between this and a scan-based `AggPath`. ([planagg.c](https://doxygen.postgresql.org/planagg_8c_source.html), [Cybertec write-up](https://www.cybertec-postgresql.com/en/speeding-up-min-and-max/))

**`COUNT(*)` and index-only scans.** Postgres has no `count(*)`-from-index optimization analogous to MIN/MAX. The closest is *index-only scan*: a btree scan that skips heap access when the visibility map's all-visible bit is set for the heap page. `EXPLAIN` reports `Heap Fetches: N` for cases where the VM bit was not set. Index-only scans require: index type supports it (btree always; GiST/SP-GiST for some opclasses; GIN never); query references only indexed columns; relevant heap pages are all-visible (requires `VACUUM`). With predicates, btree can do `LooseIndexScan`-like skips, but a full `count(*)` still walks every index entry. ([Index-Only Scans docs](https://www.postgresql.org/docs/current/indexes-index-only-scans.html))

**Partition-wise aggregate and FDW pushdown.** Postgres 10 added remote aggregation in `postgres_fdw`; subsequent commits added partition-wise aggregate, which decomposes the top-level Agg into per-partition Aggs that can each be pushed to a foreign server. The plan ends with a final `Aggregate` whose `combinefunc`/`serialfunc`/`deserialfunc` (declared in `CREATE AGGREGATE`) merge the partials. Enabled by `enable_partitionwise_aggregate` GUC. Restrictions: no `DISTINCT`/`ORDER BY` in aggregate, no `HAVING`, not `array_agg`. ([EDB Aggregate Push-down post](https://www.enterprisedb.com/blog/postgresql-aggregate-push-down-postgresfdw), [commit message](https://www.postgresql.org/message-id/E1f30tV-0003rh-27@gemulon.postgresql.org))

### 3.2 DuckDB

DuckDB auto-builds **zonemaps** (per-row-group min/max) for all general-purpose types and uses them for both predicate pushdown and "computing aggregations" (Indexing docs). Row groups are ~122,880 rows. The optimizer pipeline (Filter Pushdown, Join Order, TopN, Expression Rewriter, Filter Pull-up, IN Rewriter, Statistics Propagation, Reorder Filters, Join Filter Pushdown) does not document a dedicated metadata-only-aggregate rule, but Statistics Propagation does fold known constants (e.g., `MIN/MAX` of a column with known range) at plan time. The **ART index** is documented as not affecting aggregation/join/sort performance — it is only for point lookups and PK enforcement. ([Indexing](https://duckdb.org/docs/current/guides/performance/indexing), [Optimizers blog](https://duckdb.org/2024/11/14/optimizers))

### 3.3 SQL Server (Columnstore)

**Segment elimination** drops rowgroups whose per-segment min/max prove a predicate cannot match. Numeric/date types since 2014; string/binary/guid since 2022. Each rowgroup also stores row count for instant `COUNT(*)`. ([SQLpassion segment elimination](https://www.sqlpassion.at/archive/2017/01/30/columnstore-segment-elimination/))

**Aggregate Pushdown (2016+).** The Aggregate operator is fused into the Columnstore Scan; aggregation runs on compressed/bit-packed data with SIMD. Supports `MIN`, `MAX`, `SUM`, `COUNT`, `COUNT(*)` when input+output fit in 64 bits (int family, money, decimal/numeric with precision ≤18, date/time types). **Not supported**: `DISTINCT`, string columns, virtual columns, delta store rows (which still flow up to the Aggregate node). EXPLAIN exposes `ActualLocallyAggregatedRows`. ([Microsoft post](https://learn.microsoft.com/en-us/archive/blogs/sql_server_team/columnstore-index-performance-sql-server-2016-aggregate-pushdown))

**Grouped Aggregate Pushdown (2019+).** Extends to `GROUP BY`. Each output batch (~900 rows) makes a *runtime* choice between fast (pushdown) and slow paths based on a "benefit measure" starting at 100 and decremented when batches contain few rows per key (22% penalty for <8/key, 11% for 8–16/key). Disables entirely when bit-packed grouping key exceeds 10 bits. Pure RLE keys always fast-path. ([Paul White, SQLPerformance](https://sqlperformance.com/2019/04/sql-plan/grouped-aggregate-pushdown))

**Indexed Views.** Materialized `SELECT ... GROUP BY` results with synchronous maintenance. Optimizer can use them transparently if `EXPAND VIEWS` is off — purely planner-side pattern match against `SELECT` shape.

### 3.4 ClickHouse

**Granule-level min/max + skip indexes.** Default granule is 8192 rows; the primary key (sparse) gives row-range pruning, and explicit `minmax`/`set`/`bloom_filter` skip indexes augment it. The `optimize_use_implicit_projections` and `optimize_use_projections` flags drive the optimizer to consider projections.

**Projections** (transparent materialized aggregates). When a projection defines `GROUP BY`, the underlying engine becomes `AggregatingMergeTree` and aggregate columns become `AggregateFunction(...)` states. The optimizer "automatically samples the primary keys and chooses a table that can generate the same correct result, but requires the least amount of data to be read." Since 25.5, projections can store only sorting keys + `_part_offset` to act as a pure index. ([Projections docs](https://clickhouse.com/docs/data-modeling/projections))

**AggregatingMergeTree.** Stores partial states for aggregations; `min`/`max` need no extra merge cost ("require no extra steps to calculate the final result from the intermediate steps"). The `SimpleAggregateFunction` combinator is an optimized form for aggregates whose state is just the result (`min`, `max`, `sum`, `any`, `anyLast`). ([Altinity KB](https://kb.altinity.com/altinity-kb-queries-and-syntax/simplestateif-or-ifstate-for-simple-aggregate-functions/))

### 3.5 Apache Druid

**Bitmap indexes per dictionary entry.** For each distinct value in a (string) column, Druid stores one Roaring-compressed bitmap of matching rows. Combined with a dictionary mapping string→int. ([Segments doc](https://druid.apache.org/docs/latest/design/segments/))

**`cardinality` and `hyperUnique` aggregators.** `COUNT(DISTINCT)` in SQL is translated to `cardinality`, which returns an *approximate* count via HyperLogLog over the dimension values; `hyperUnique` is the recommended path when you only need the count, not the values — it's stored as an HLL sketch in the segment, so the count is computed by merging sketches across segments, no per-row work. Druid recommends DataSketches (theta/HLL) for new use cases. ([HLL old docs](https://druid.apache.org/docs/latest/querying/hll-old.html), [CALCITE-1670](https://issues.apache.org/jira/browse/CALCITE-1670))

For *exact* distinct count, Druid does not push down — it runs a groupBy and counts. The bitmap-per-value structure means exact distinct count *could* be answered as "number of bitmaps in the dictionary whose intersection with the predicate mask is non-empty" — this is exactly the Lance opportunity.

### 3.6 Apache Pinot — Star-Tree Index

Pre-aggregated multi-dimensional tree. Each level splits on a dimension; each internal node has a "star" child holding the aggregate with that dimension dropped. The planner pattern-matches a query's `GROUP BY` dimensions and aggregate functions against an available star-tree's schema. Aggregations are *materialized* at build time. Reported gains: "99.76% reduction in latency vs. no Star-Tree Index (6.3 seconds to 15 ms)" and "99.99999% reduction in amount of data scanned." Supports COUNT/SUM/MIN/MAX/etc.; approximate distinct via DataSketches theta/HLL stored as the aggregate value at the node. ([Pinot docs](https://docs.pinot.apache.org/basics/indexing/star-tree-index), [Part 3 blog](https://startree.ai/resources/star-tree-index-in-apache-pinot-part-3-understanding-the-impact-in-real-customer/))

### 3.7 Snowflake

**Micro-partition metadata** stored per partition: column value ranges, distinct counts, and "additional properties." Metadata is in the cloud-services layer, queried before any data IO. `count(*)`, `MIN(col)`, `MAX(col)` on a partition-aligned column with no predicate (or with a predicate that aligns with metadata) can return from metadata alone, hence the well-known "instant `COUNT(*)`" on Snowflake. ([Micro-partitions docs](https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions))

**Snowflake Optima (2024-2025).** Dynamically generates *additional* lightweight per-micro-partition metadata for high-frequency "hot" expressions seen in workloads — extending min/max-style pruning to expressions like `LOWER(col) = ...`. ([Optima blog](https://www.snowflake.com/en/engineering-blog/snowflake-optima-metadata-query-pruning/))

### 3.8 Parquet / Iceberg

**Parquet** stores per-row-group and per-page min/max, null count, distinct count (optional, often unset by writers). These drive predicate pushdown but are also enough material for aggregate pushdown.

**Iceberg PR #6622** (`huaxingao`, merged) implemented `MIN/MAX/COUNT` pushdown through Spark's `SupportsPushDownAggregates`. Key classes: `AggregateEvaluator`, `BoundAggregate` (with `hasValue` to distinguish "all-null column" from "stats missing"), `MaxAggregate`, `MinAggregate`, `CountNonNull`. `SparkScanBuilder` orchestrates. Restrictions explicitly enumerated:
- **No GROUP BY** ("Group by aggregation push down is not supported")
- **No row-level deletes** ("Skipped aggregate pushdown: detected row level deletes")
- **No complex types lacking stats**
- **No truncated string metrics** (default mode truncates strings; can't reason about MIN/MAX)

Toggle: `spark.sql.iceberg.aggregate-push-down-enabled`. ([PR #6622](https://github.com/apache/iceberg/pull/6622))

### 3.9 Spark V2 — `SupportsPushDownAggregates`

The data-source-side contract used by Iceberg, JDBC, file sources. `pushAggregation(Aggregation): boolean` to attempt pushdown; `supportCompletePushDown(Aggregation): boolean` to declare whether the source returns final or partial. If partial, Spark inserts a final Aggregate above the V2 scan with the combine semantics. Filter pushdown happens *first*, then aggregate pushdown — so the data source sees already-filtered fragments. ([Spark JavaDoc](https://spark.apache.org/docs/3.4.3/api/java/org/apache/spark/sql/connector/read/SupportsPushDownAggregates.html))

### 3.10 DataFusion

Has an `AggregateStatistics` physical optimizer rule that converts `MIN/MAX/COUNT(*)` over a scan with exact statistics into a constant `ProjectionExec` — pure metadata-only execution. Issue [#19938](https://github.com/apache/datafusion/issues/19938) proposes extending min/max statistics to drive group-by *layout* (use a `Vec` indexed by `value - min` when the range is small). The "Limit Pruning" blog (March 2026) describes a clean three-tier model: *Not Matching* / *Partially Matching* / *Fully Matching* row groups, where Fully-Matching groups can satisfy `LIMIT` without row-level filtering — directly applicable to aggregate pushdown: Fully-Matching row groups can contribute exact counts from their row-count statistic. ([Query Optimizer docs](https://datafusion.apache.org/library-user-guide/query-optimizer.html), [Limit Pruning blog](https://datafusion.apache.org/blog/2026/03/20/limit-pruning/))

---

## 4. Index-Type → Aggregate-Type Matrix

| Index / Metadata           | `COUNT(*)`         | `MIN/MAX`         | `SUM`            | `COUNT(col)` (non-null) | `COUNT(DISTINCT col)`         | `GROUP BY` cardinality       |
|---                         |---                 |---                |---               |---                      |---                            |---                           |
| Row count per fragment     | Yes (no pred)      | No                | No               | Need null count         | No                            | No                           |
| Zone map (min/max)         | No*                | **Yes**           | No               | No                      | No                            | No                           |
| Null count per fragment    | Yes (with above)   | No                | No               | **Yes** (no pred)       | No                            | No                           |
| Distinct count per frag.   | No                 | No                | No               | No                      | Approx (upper bound)†         | No                           |
| Btree (ordered)            | Walk index         | **Yes** O(log n)  | Walk index       | Walk index              | Loose-index scan              | Stream-grouped scan          |
| Bitmap (one-per-value)     | Sum of all bitmaps | **Yes** (first/last value with non-empty bitmap) | No | Bitmap union cardinality | **Yes** (count of values with non-empty bitmap intersected with predicate mask) | **Yes** (cardinality of each bitmap, partition by value) |
| HLL/Theta sketch           | No                 | No                | No               | No                      | **Yes** (approximate)         | Per-group sketch merge        |
| Materialized view / projection / star-tree | Yes | Yes | Yes | Yes | Yes (if pre-aggregated) | **Yes**                      |

*`COUNT(*)` from a zone map alone needs row count too — but every engine stores both per fragment, so in practice this is a single lookup.
†Per-fragment distinct counts cannot be summed (overlap); they bound the answer above.

The bitmap row is the strongest case for Lance. Bitmap-cardinality identities:
```
COUNT(col)         = popcount(  OR_v posting[v]          )    over predicate-masked rows
COUNT(DISTINCT col)= |{ v : posting[v] AND mask != ∅ }|
COUNT(*) WHERE col=v = popcount( posting[v] AND mask )
GROUP BY col, COUNT(*) = for v in dict: emit (v, popcount(posting[v] AND mask))
```

---

## 5. Planner Integration Patterns

Three recurring shapes, in order of complexity:

**(a) Pre-planner rewrite (Postgres pattern).** A dedicated function — `preprocess_minmax_aggregates` — runs *before* the main path enumeration. It builds an alternate path (`MinMaxAggPath`) parallel to the normal Aggregate-over-Scan path. The cost model picks the winner. Pros: keeps the special case out of the general optimizer. Cons: each new shape is a new bespoke function.

**(b) Physical-optimizer rule (DataFusion `AggregateStatistics`).** A late physical-plan rewrite that inspects the plan tree for `AggregateExec { mode: Final, expr: [Min|Max|Count], input: ScanExec }` and, if the scan can produce exact statistics for those columns, replaces the whole subtree with a `ProjectionExec` of constants. Pros: composes with existing rules. Cons: must reason about partial-vs-final aggregate modes; needs exact (not estimated) statistics.

**(c) Data-source interface (Spark V2 `SupportsPushDownAggregates`).** The optimizer hands the data source an `Aggregation` description; the source returns whether (and how completely) it can satisfy it. If partial, optimizer inserts a final-stage Aggregate above. Pros: clean separation; the source owns correctness. Cons: API surface is large; partial-aggregate plumbing must be wired.

**Recommendation for Lance.** Mirror Spark V2's contract at the `Scan` level, but execute the dispatch in DataFusion's physical optimizer (since Lance plans through DataFusion already). The `Scan` would expose `try_pushdown_aggregate(agg, filter) -> Option<PartialAggregateResult>`. The optimizer rule walks `AggregateExec(final) → AggregateExec(partial) → Scan` patterns and asks the scan whether it can satisfy. Index access lives inside the scan (or its `MetricsProvider`), not in the optimizer.

---

## 6. Correctness Gotchas

1. **Predicate-must-be-fully-evaluable-by-index.** If the index can evaluate `col = 5` but not `f(col) = 5`, the predicate must be either rejected by the index entirely or split. A pushed aggregate over a partially-filtered set is silently wrong. Iceberg's PR thread had multiple iterations on this.

2. **NULL handling per aggregate.** `COUNT(*)` counts rows including nulls; `COUNT(col)` and `MIN/MAX` skip nulls. Need both row count and null count per fragment. Iceberg's `BoundAggregate.hasValue` distinguishes "stat exists and column is all-null (legal answer for MIN/MAX = NULL)" from "stat missing → abort."

3. **Row-level deletes / deletion vectors / MVCC.** Stale statistics. Postgres: visibility map. SQL Server: delta rowgroups bypass pushdown. Iceberg: aggregate pushdown disabled if row-level deletes exist on touched files. **Lance equivalent: deletion vectors.** Pushdown must either consult deletion vector population (row count − deleted count) or abort.

4. **Empty input vs zero.** `COUNT` on zero rows is `0`; `MIN/MAX/SUM` on zero rows is `NULL`. The fast path must return the right type, not silently coerce.

5. **`COUNT(DISTINCT)` overlap across fragments.** Per-fragment distinct counts cannot be summed. Two paths: (a) merge an exact structure (sorted dictionary or bitmap union) across fragments; (b) merge HLL/theta sketches for approximate answer. Lance bitmap indexes naturally support (a) via posting-list union.

6. **Truncated/lossy statistics.** Parquet writers commonly truncate string min/max. Iceberg refuses pushdown in this case. Lance should mark such stats as inexact and refuse.

7. **`MIN/MAX` sort operator vs. aggregate sort order.** Postgres's `fetch_agg_sort_op` looks up the agg's sort operator from `pg_aggregate`. A user-defined min-like aggregate is not eligible unless registered correctly. Lance's analogue: only well-known `MIN`/`MAX` over orderable types qualify; do not try to be clever with user-defined aggregates.

8. **GROUP BY combined with aggregate pushdown is partial by definition.** Each fragment emits `(group_key, partial_agg)`, and the engine reduces across fragments. The fragment-side dedup is *not* a complete `GROUP BY` — duplicates across fragments are normal and required for correctness. SQL Server's docs: "the data source can still output data with duplicated keys, which is OK as Spark will do GROUP BY key again."

9. **Aggregate-over-filter ordering.** Spark V2 explicitly pushes filters *before* aggregates. Lance's scan API should follow: aggregate pushdown receives the post-filter view.

10. **Approximate vs exact must be explicit in the API.** Calcite Druid translation of `COUNT(DISTINCT)` to `cardinality` was filed as a bug (CALCITE-1670) because users didn't expect approximate semantics. Lance should never silently approximate.

---

## 7. Open Questions / Things I Couldn't Pin Down Authoritatively

- **DuckDB's exact metadata-only path.** Multiple sources say zonemaps drive "computing aggregations" but I could not find a named optimizer rule (e.g., a `count_star_metadata` rule) in either the optimizer blog or the indexing docs. Need to read `src/optimizer/` in the DuckDB tree directly — start at [`optimizer.cpp`](https://github.com/duckdb/duckdb/blob/main/src/optimizer/optimizer.cpp) and look for statistics-propagation paths that fold to constants.
- **ClickHouse projection selection cost model.** Docs say "the optimizer automatically samples the primary keys" but I did not find a description of the tie-breaking when multiple projections could serve. Likely in `Processors/QueryPlan/Optimizations/optimizeUseAggregateProjection.cpp` in source.
- **Snowflake metadata-only execution rules.** Marketing-level confirmation that COUNT/MIN/MAX from metadata works, but no published planner doc. The Optima blog is the closest thing and is high-level.
- **Pinot star-tree planner matching.** Docs describe the structure but not the matcher. The pattern from the description is "exact match on dimension subset + supported aggregate"; needs source-code confirmation (see `pinot-segment-spi`).
- **Druid exact COUNT(DISTINCT) status.** There is a community "Exact Cardinality Count" extension PR but it is not in core. Mainline path is HLL-approximate. Worth a follow-up: does Druid's bitmap structure make exact distinct count "free enough" that someone proposed a core impl? (The PR exists; review comments would tell us why it didn't merge.)
- **Postgres `count(*)` from index.** I expected a planner rewrite analogous to MinMaxAggPath. I couldn't find one — it appears `count(*)` always goes through an actual scan (possibly index-only), never a metadata read. Worth confirming on `pgsql-hackers`; multiple threads have proposed it and been declined for MVCC reasons.
- **Iceberg manifest-only `MIN/MAX` correctness with column nullability.** PR #6622 introduces `hasValue` but I didn't trace whether mixed-null + non-null fragments are merged correctly when *some* fragments have stats and *others* don't. Worth reading the test cases before mirroring the design.

---

### Sources

- PostgreSQL: [planagg.c source](https://doxygen.postgresql.org/planagg_8c_source.html) · [Cybertec MIN/MAX speedup](https://www.cybertec-postgresql.com/en/speeding-up-min-and-max/) · [Index-Only Scans](https://www.postgresql.org/docs/current/indexes-index-only-scans.html) · [Wiki: Index-only scans](https://wiki.postgresql.org/wiki/Index-only_scans) · [EDB Aggregate Push-down](https://www.enterprisedb.com/blog/postgresql-aggregate-push-down-postgresfdw) · [Partition-wise aggregation commit](https://www.postgresql.org/message-id/E1f30tV-0003rh-27@gemulon.postgresql.org)
- DuckDB: [Indexing](https://duckdb.org/docs/current/guides/performance/indexing) · [Indexes](https://duckdb.org/docs/current/sql/indexes) · [Optimizers blog](https://duckdb.org/2024/11/14/optimizers) · [Row Groups (DeepWiki)](https://deepwiki.com/duckdb/duckdb/7.2-column-storage)
- SQL Server: [Aggregate Pushdown 2016](https://learn.microsoft.com/en-us/archive/blogs/sql_server_team/columnstore-index-performance-sql-server-2016-aggregate-pushdown) · [Grouped Aggregate Pushdown (Paul White)](https://sqlperformance.com/2019/04/sql-plan/grouped-aggregate-pushdown) · [Columnstore Query Performance](https://learn.microsoft.com/en-us/sql/relational-databases/indexes/columnstore-indexes-query-performance) · [ColumnStore Segment Elimination](https://www.sqlpassion.at/archive/2017/01/30/columnstore-segment-elimination/)
- ClickHouse: [Projections docs](https://clickhouse.com/docs/data-modeling/projections) · [AggregatingMergeTree (Altinity)](https://kb.altinity.com/engines/mergetree-table-engine-family/aggregatingmergetree/) · [SimpleState combinator](https://kb.altinity.com/altinity-kb-queries-and-syntax/simplestateif-or-ifstate-for-simple-aggregate-functions/)
- Druid: [Segments design](https://druid.apache.org/docs/latest/design/segments/) · [HLL old aggregator](https://druid.apache.org/docs/latest/querying/hll-old.html) · [Aggregations reference](https://druid.apache.org/docs/latest/querying/aggregations/) · [CALCITE-1670](https://issues.apache.org/jira/browse/CALCITE-1670)
- Pinot: [Star-Tree Index docs](https://docs.pinot.apache.org/basics/indexing/star-tree-index) · [Star-Tree Part 3](https://startree.ai/resources/star-tree-index-in-apache-pinot-part-3-understanding-the-impact-in-real-customer/)
- Snowflake: [Micro-partitions and clustering](https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions) · [Snowflake Optima](https://www.snowflake.com/en/engineering-blog/snowflake-optima-metadata-query-pruning/) · [Pruning paper (arXiv)](https://arxiv.org/html/2504.11540v1)
- Iceberg/Spark: [Iceberg PR #6622 (aggregate pushdown)](https://github.com/apache/iceberg/pull/6622) · [Iceberg statistics (Ryft)](https://www.ryft.io/blog/making-sense-of-apache-iceberg-statistics) · [Spark SupportsPushDownAggregates JavaDoc](https://spark.apache.org/docs/3.4.3/api/java/org/apache/spark/sql/connector/read/SupportsPushDownAggregates.html)
- DataFusion: [Query Optimizer](https://datafusion.apache.org/library-user-guide/query-optimizer.html) · [Issue #19938 (min/max in grouped aggs)](https://github.com/apache/datafusion/issues/19938) · [Limit Pruning blog (Mar 2026)](https://datafusion.apache.org/blog/2026/03/20/limit-pruning/) · [Optimizing SQL Part 2](https://datafusion.apache.org/blog/2025/06/15/optimizing-sql-dataframes-part-two/)
