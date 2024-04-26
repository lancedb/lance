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

package com.lancedb.lance.spark.source;

import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.SupportsWrite;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
 
/**
 * Lance Spark Table.
*/
public class SparkTable implements SupportsRead, SupportsWrite {
  private static final Set<TableCapability> CAPABILITIES =
      ImmutableSet.of(
          TableCapability.BATCH_WRITE);

  // Lance parameters
  private final String datasetUri;
  // Spark parameters
  private final String tableName;
  private final StructType sparkSchema;

  /**
   * Creates a spark table.
   *
   * @param datasetUri the lance dataset uri
   * @param tableName table name
   * @param sparkSchema spark struct type
   */
  public SparkTable(String datasetUri,
      String tableName, StructType sparkSchema) {
    this.datasetUri = datasetUri;
    this.tableName = tableName;
    this.sparkSchema = sparkSchema;
  }

  @Override
  public ScanBuilder newScanBuilder(CaseInsensitiveStringMap caseInsensitiveStringMap) {
    throw new UnsupportedOperationException("Lance Spark scan");
  }

  @Override
  public WriteBuilder newWriteBuilder(LogicalWriteInfo info) {
    return new SparkWriteBuilder(datasetUri, sparkSchema, info);
  }

  @Override
  public String name() {
    return this.tableName;
  }

  @Override
  public StructType schema() {
    return this.sparkSchema;
  }

  @Override
  public Set<TableCapability> capabilities() {
    return CAPABILITIES;
  }
}
