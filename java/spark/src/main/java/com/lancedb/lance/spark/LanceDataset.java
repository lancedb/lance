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

package com.lancedb.lance.spark;

import com.google.common.collect.ImmutableSet;

import java.util.Set;

import com.lancedb.lance.spark.read.LanceScanBuilder;
import com.lancedb.lance.spark.write.SparkWrite;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.SupportsWrite;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Lance Spark Dataset.
 */
public class LanceDataset implements Table, SupportsRead, SupportsWrite {
  private static final Set<TableCapability> CAPABILITIES =
    ImmutableSet.of(
        TableCapability.BATCH_READ,
        TableCapability.BATCH_WRITE);

  LanceConfig options;
  private final StructType sparkSchema;

  /**
   * Creates a Lance dataset.
   *
   * @param config read config
   * @param sparkSchema spark struct type
   */
  public LanceDataset(LanceConfig config, StructType sparkSchema) {
    this.options = config;
    this.sparkSchema = sparkSchema;
  }

  @Override
  public ScanBuilder newScanBuilder(CaseInsensitiveStringMap caseInsensitiveStringMap) {
    return new LanceScanBuilder(sparkSchema, options);
  }

  @Override
  public String name() {
    return this.options.getDatasetName();
  }

  @Override
  public StructType schema() {
    return sparkSchema;
  }

  @Override
  public Set<TableCapability> capabilities() {
    return CAPABILITIES;
  }

  @Override
  public WriteBuilder newWriteBuilder(LogicalWriteInfo logicalWriteInfo) {
    return new SparkWrite.SparkWriteBuilder(sparkSchema, options);
  }
}
