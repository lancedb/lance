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

import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.types.StructType;

public class LanceInputPartition implements InputPartition {
  private static final long serialVersionUID = 4723894723984723984L;

  private final StructType schema;
  private final int partitionId;
  private final LanceSplit lanceSplit;
  private final LanceConfig config;
  private final Optional<String> whereCondition;

  public LanceInputPartition(StructType schema, int partitionId,
      LanceSplit lanceSplit, LanceConfig config, Optional<String> whereCondition) {
    this.schema = schema;
    this.partitionId = partitionId;
    this.lanceSplit = lanceSplit;
    this.config = config;
    this.whereCondition = whereCondition;
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
}
