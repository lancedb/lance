/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.lancedb.lance.spark.read;

import com.lancedb.lance.ipc.ColumnOrdering;
import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.utils.Optional;

import org.apache.arrow.util.Preconditions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.Statistics;
import org.apache.spark.sql.connector.read.SupportsReportStatistics;
import org.apache.spark.sql.internal.connector.SupportsMetadata;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import scala.collection.immutable.Map;
import scala.collection.mutable.HashMap;

import java.io.Serializable;
import java.util.List;
import java.util.stream.IntStream;

public class LanceScan
    implements Batch, Scan, SupportsMetadata, SupportsReportStatistics, Serializable {
  private static final long serialVersionUID = 947284762748623947L;

  private final StructType schema;
  private final LanceConfig config;
  private final Optional<String> whereConditions;
  private final Optional<Integer> limit;
  private final Optional<Integer> offset;
  private final Optional<List<ColumnOrdering>> topNSortOrders;

  public LanceScan(
      StructType schema,
      LanceConfig config,
      Optional<String> whereConditions,
      Optional<Integer> limit,
      Optional<Integer> offset,
      Optional<List<ColumnOrdering>> topNSortOrders) {
    this.schema = schema;
    this.config = config;
    this.whereConditions = whereConditions;
    this.limit = limit;
    this.offset = offset;
    this.topNSortOrders = topNSortOrders;
  }

  @Override
  public Batch toBatch() {
    return this;
  }

  @Override
  public InputPartition[] planInputPartitions() {
    List<LanceSplit> splits = LanceSplit.generateLanceSplits(config);
    return IntStream.range(0, splits.size())
        .mapToObj(
            i ->
                new LanceInputPartition(
                    schema,
                    i,
                    splits.get(i),
                    config,
                    whereConditions,
                    limit,
                    offset,
                    topNSortOrders))
        .toArray(InputPartition[]::new);
  }

  @Override
  public PartitionReaderFactory createReaderFactory() {
    return new LanceReaderFactory();
  }

  @Override
  public StructType readSchema() {
    return schema;
  }

  @Override
  public Map<String, String> getMetaData() {
    HashMap<String, String> hashMap = new HashMap<>();
    hashMap.put("whereConditions", whereConditions.toString());
    hashMap.put("limit", limit.toString());
    hashMap.put("offset", offset.toString());
    hashMap.put("topNSortOrders", topNSortOrders.toString());
    return hashMap.toMap(scala.Predef.conforms());
  }

  @Override
  public Statistics estimateStatistics() {
    return new LanceStatistics(config);
  }

  private class LanceReaderFactory implements PartitionReaderFactory {
    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
      Preconditions.checkArgument(
          partition instanceof LanceInputPartition,
          "Unknown InputPartition type. Expecting LanceInputPartition");
      return LanceRowPartitionReader.create((LanceInputPartition) partition);
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
      Preconditions.checkArgument(
          partition instanceof LanceInputPartition,
          "Unknown InputPartition type. Expecting LanceInputPartition");
      return new LanceColumnarPartitionReader((LanceInputPartition) partition);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
      return true;
    }
  }
}
