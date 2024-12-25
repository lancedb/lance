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

import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.types.StructType;

import java.util.List;

public class LanceInputPartition implements InputPartition {
  private static final long serialVersionUID = 4723894723984723984L;

  private final StructType schema;
  private final int partitionId;
  private final LanceSplit lanceSplit;
  private final LanceConfig config;
  private final Optional<String> whereCondition;
  private final Optional<Integer> limit;
  private final Optional<Integer> offset;
  private final Optional<List<ColumnOrdering>> topNSortOrders;

  public LanceInputPartition(
      StructType schema,
      int partitionId,
      LanceSplit lanceSplit,
      LanceConfig config,
      Optional<String> whereCondition) {
    this.schema = schema;
    this.partitionId = partitionId;
    this.lanceSplit = lanceSplit;
    this.config = config;
    this.whereCondition = whereCondition;
    this.limit = Optional.empty();
    this.offset = Optional.empty();
    this.topNSortOrders = Optional.empty();
  }

  public LanceInputPartition(
      StructType schema,
      int partitionId,
      LanceSplit lanceSplit,
      LanceConfig config,
      Optional<String> whereCondition,
      Optional<Integer> limit,
      Optional<Integer> offset) {
    this.schema = schema;
    this.partitionId = partitionId;
    this.lanceSplit = lanceSplit;
    this.config = config;
    this.whereCondition = whereCondition;
    this.limit = limit;
    this.offset = offset;
    this.topNSortOrders = Optional.empty();
  }

  public LanceInputPartition(
      StructType schema,
      int partitionId,
      LanceSplit lanceSplit,
      LanceConfig config,
      Optional<String> whereCondition,
      Optional<Integer> limit,
      Optional<Integer> offset,
      Optional<List<ColumnOrdering>> topNSortOrders) {
    this.schema = schema;
    this.partitionId = partitionId;
    this.lanceSplit = lanceSplit;
    this.config = config;
    this.whereCondition = whereCondition;
    this.limit = limit;
    this.offset = offset;
    this.topNSortOrders = topNSortOrders;
  }

  public StructType getSchema() {
    return schema;
  }

  public int getPartitionId() {
    return partitionId;
  }

  public LanceSplit getLanceSplit() {
    return lanceSplit;
  }

  public LanceConfig getConfig() {
    return config;
  }

  public Optional<String> getWhereCondition() {
    return whereCondition;
  }

  public Optional<Integer> getLimit() {
    return limit;
  }

  public Optional<Integer> getOffset() {
    return offset;
  }

  public Optional<List<ColumnOrdering>> getTopNSortOrders() {
    return topNSortOrders;
  }
}
