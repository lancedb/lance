# MergeInsert analyze_plan Examples

This document shows example outputs from the `analyze_plan` method for merge_insert operations.

## Configuration

The example uses a simple dataset with schema:
- `id` (Int32, not null) - join key 
- `value` (Utf8, nullable) - data field

Test scenario:
- Target dataset has 3 rows: `[(1, "a"), (2, "b"), (3, "c")]`
- Source data has 2 rows: `[(1, "updated_a"), (4, "d")]`  
- Expected result: 1 update (id=1) and 1 insert (id=4)

Merge insert configuration:
- `when_matched: UpdateAll`
- `when_not_matched: InsertAll` 
- `when_not_matched_by_source: Keep` (default)

## analyze_plan(data_stream, verbose=false)

```
AnalyzeExec verbose=true, metrics=[]
  TracedExec, metrics=[]
    MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep, metrics=[output_rows=0, elapsed_compute=1ns, bytes_written=591, num_deleted_rows=0, num_files_written=1, num_inserted_rows=1, num_updated_rows=1]
      CoalescePartitionsExec, metrics=[output_rows=2, elapsed_compute=27.302µs]
        ProjectionExec: expr=[_rowaddr@0 as _rowaddr, id@1 as id, value@2 as value, CASE WHEN _rowaddr@0 IS NULL THEN 2 WHEN _rowaddr@0 IS NOT NULL THEN 1 ELSE 0 END as action], metrics=[output_rows=2, elapsed_compute=103.971µs]
          CoalesceBatchesExec: target_batch_size=8192, metrics=[output_rows=2, elapsed_compute=13.158µs]
            HashJoinExec: mode=CollectLeft, join_type=Right, on=[(id@0, id@0)], projection=[_rowaddr@1, id@2, value@3], metrics=[output_rows=2, build_input_batches=1, build_input_rows=3, input_batches=1, input_rows=2, output_batches=1, build_mem_used=160, build_time=427.932µs, join_time=139.043µs]
              LanceRead: uri=test_analyze_string/data, projection=[id], num_fragments=1, range_before=None, range_after=None, row_id=false, row_addr=true, full_filter=--, refine_filter=--, metrics=[output_rows=3, elapsed_compute=901.733µs, bytes_read=603, fragments_scanned=1, iops=2, ranges_scanned=1, requests=2, rows_scanned=3, task_wait_time=901.597µs]
              RepartitionExec: partitioning=RoundRobinBatch(32), input_partitions=1, metrics=[fetch_time=8.725µs, repartition_time=1ns, send_time=4.226µs]
                StreamingTableExec: partition_sizes=1, projection=[id, value], metrics=[]
```

## analyze_plan(data_stream, verbose=true)

```
AnalyzeExec verbose=true, metrics=[]
  TracedExec, metrics=[]
    MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep, metrics=[output_rows=0, elapsed_compute=1ns, bytes_written=591, num_deleted_rows=0, num_files_written=1, num_inserted_rows=1, num_updated_rows=1]
      CoalescePartitionsExec, metrics=[output_rows=2, elapsed_compute=30.879µs]
        ProjectionExec: expr=[_rowaddr@0 as _rowaddr, id@1 as id, value@2 as value, CASE WHEN _rowaddr@0 IS NULL THEN 2 WHEN _rowaddr@0 IS NOT NULL THEN 1 ELSE 0 END as action], metrics=[output_rows=2, elapsed_compute=74.069µs]
          CoalesceBatchesExec: target_batch_size=8192, metrics=[output_rows=2, elapsed_compute=9.078µs]
            HashJoinExec: mode=CollectLeft, join_type=Right, on=[(id@0, id@0)], projection=[_rowaddr@1, id@2, value@3], metrics=[output_rows=2, build_input_batches=1, build_input_rows=3, input_batches=1, input_rows=2, output_batches=1, build_mem_used=160, build_time=245.093µs, join_time=66.658µs]
              LanceRead: uri=test_analyze_string/data, projection=[id], num_fragments=1, range_before=None, range_after=None, row_id=false, row_addr=true, full_filter=--, refine_filter=--, metrics=[output_rows=3, elapsed_compute=462.065µs, bytes_read=12, fragments_scanned=1, iops=1, ranges_scanned=1, requests=1, rows_scanned=3, task_wait_time=461.872µs]
              RepartitionExec: partitioning=RoundRobinBatch(32), input_partitions=1, metrics=[fetch_time=7.551µs, repartition_time=1ns, send_time=3.539µs]
                StreamingTableExec: partition_sizes=1, projection=[id, value], metrics=[]
```

## Key Metrics Breakdown

### MergeInsert-Specific Metrics
- **bytes_written=591**: Total bytes written to storage (591 bytes)
- **num_deleted_rows=0**: No rows deleted (when_not_matched_by_source=Keep)
- **num_files_written=1**: Number of data files written (1 file)  
- **num_inserted_rows=1**: One new row inserted (id=4)
- **num_updated_rows=1**: One existing row updated (id=1)
- **output_rows=0**: MergeInsert executor outputs no rows (writes directly to storage)

### HashJoinExec Metrics  
- **build_input_rows=3**: Target table has 3 rows
- **input_rows=2**: Source has 2 rows
- **output_rows=2**: Join produces 2 result rows
- **build_mem_used=160**: Memory used for hash table (bytes)
- **build_time**: Time to build hash table
- **join_time**: Time to perform join operation

### LanceRead Metrics
- **bytes_read**: Raw bytes read from storage
- **fragments_scanned=1**: Number of Lance fragments accessed
- **iops**: Number of I/O operations 
- **requests**: Number of storage requests
- **rows_scanned=3**: Total rows read from target table
- **task_wait_time**: Time spent waiting for I/O

### Performance Insights
1. The execution correctly identified 1 update and 1 insert operation
2. Hash join efficiently processed the small dataset
3. LanceRead metrics show minimal I/O for this test dataset
4. Total execution time under 1ms for this small example

This analysis output provides comprehensive insights for debugging performance issues and understanding merge operation behavior.