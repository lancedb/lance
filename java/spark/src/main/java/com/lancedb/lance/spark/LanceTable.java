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
import com.lancedb.lance.spark.internal.LanceConfig;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Lance Spark Table.
 */
public class LanceTable implements Table, SupportsRead {
  private static final Set<TableCapability> CAPABILITIES =
    ImmutableSet.of(
        TableCapability.BATCH_READ);

  LanceConfig options;
  private final StructType sparkSchema;

  /**
   * Creates a spark table.
   *
   * @param config read config
   * @param sparkSchema spark struct type
   */
  public LanceTable(LanceConfig config, StructType sparkSchema) {
    this.options = config;
    this.sparkSchema = sparkSchema;
  }

  @Override
  public ScanBuilder newScanBuilder(CaseInsensitiveStringMap caseInsensitiveStringMap) {
    return new LanceScanBuilder(sparkSchema, options);
  }

  @Override
  public String name() {
    return this.options.getTableName();
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
