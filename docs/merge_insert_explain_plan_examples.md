# MergeInsert explain_plan Examples

This document shows example outputs from the `explain_plan` method for merge_insert operations.

## Configuration

The example uses a simple dataset with schema:
- `id` (Int32, not null) - join key 
- `value` (Utf8, nullable) - data field

Merge insert configuration:
- `when_matched: UpdateAll`
- `when_not_matched: InsertAll` 
- `when_not_matched_by_source: Keep` (default)

## explain_plan(schema, verbose=false)

```
MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowaddr@0 as _rowaddr, id@1 as id, value@2 as value, CASE WHEN _rowaddr@0 IS NULL THEN 2 WHEN _rowaddr@0 IS NOT NULL THEN 1 ELSE 0 END as action]
      CoalesceBatchesExec: target_batch_size=8192
        HashJoinExec: mode=CollectLeft, join_type=Right, on=[(id@0, id@0)], projection=[_rowaddr@1, id@2, value@3]
          LanceRead: uri=test_explain_string/data, projection=[id], num_fragments=1, range_before=None, range_after=None, row_id=false, row_addr=true, full_filter=--, refine_filter=--
          RepartitionExec: partitioning=RoundRobinBatch(32), input_partitions=1
            StreamingTableExec: partition_sizes=1, projection=[id, value]
```

## explain_plan(schema, verbose=true)

```
MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowaddr@0 as _rowaddr, id@1 as id, value@2 as value, CASE WHEN _rowaddr@0 IS NULL THEN 2 WHEN _rowaddr@0 IS NOT NULL THEN 1 ELSE 0 END as action]
      CoalesceBatchesExec: target_batch_size=8192
        HashJoinExec: mode=CollectLeft, join_type=Right, on=[(id@0, id@0)], projection=[_rowaddr@1, id@2, value@3]
          LanceRead: uri=test_explain_string/data, projection=[id], num_fragments=1, range_before=None, range_after=None, row_id=false, row_addr=true, full_filter=--, refine_filter=--
          RepartitionExec: partitioning=RoundRobinBatch(32), input_partitions=1
            StreamingTableExec: partition_sizes=1, projection=[id, value]
```

## Execution Plan Breakdown

1. **StreamingTableExec**: Reads the source data stream
2. **RepartitionExec**: Distributes data across partitions 
3. **LanceRead**: Reads target table data for join keys
4. **HashJoinExec**: Performs outer join between source and target
5. **CoalesceBatchesExec**: Combines small batches for efficiency
6. **ProjectionExec**: Adds action column to determine merge operation type
7. **CoalescePartitionsExec**: Combines partitions before final merge
8. **MergeInsert**: Top-level executor that performs the merge logic

The plan shows this is using the "fast path" execution strategy that requires exact schema match and specific merge conditions.